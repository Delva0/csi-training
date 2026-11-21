# `RuntimeError: cancelled`: the executor's notice that a worker process died

## When this page applies

The engine was already serving, then one EngineCore log shows the three-line dump
block and a *Python* traceback that never reaches a kernel:

```text
(EngineCore_DP2 pid=57922) ERROR [dump_input.py:72] Dumping input data for V1 LLM engine (v0.26.0) with config: …
(EngineCore_DP2 pid=57922) ERROR [dump_input.py:79] Dumping scheduler output for model execution: SchedulerOutput(…)
(EngineCore_DP2 pid=57922) ERROR [dump_input.py:81] Dumping scheduler stats: SchedulerStats(…)
(EngineCore_DP2 pid=57922) ERROR [core.py:1332] EngineCore encountered a fatal error.
… core.py:693 step_with_batch_queue → model_output = future.result()
… multiproc_executor.py:91/95 result → _wait_for_response
… multiproc_executor.py:390 status, result = mq.dequeue(timeout=dequeue_timeout)
… shm_broadcast.py:779 dequeue → acquire_read
… vllm_ascend/patch/platform/patch_shm_broadcast.py:70 raise RuntimeError("cancelled")
```

Recorded #16023: vllm-ascend v0.26.0rc1 (`f2f74a16`), vllm v0.26.0
(`568afb3a`), A3 1P1D, Kimi-K3 + DSpark drafter, TP16 × DP4,
`max_seq_len=133120`, `--enable-chunked-prefill`, `enforce_eager=True`
(`cudagraph_mode: NONE`); "短请求 curl 正常，百k输入 prefill 节点挂了". The same
shape arrives from any runtime kill — a bench client's `ConnectionRefused`, a
`EngineDeadError`, a decode peer that suddenly sees `Connection closed by peer`.

## Rule 1 — `cancelled` means the process died; do not walk the queue code

Verified chain at vllm v0.26.0 (`568afb3a`) / vllm-ascend v0.26.0rc1 (`f2f74a16`):

| Step | Site | Effect |
|---|---|---|
| a worker process exits | — | its sentinel fires |
| monitor | `vllm/v1/executor/multiproc_executor.py:275-298` (`monitor_workers`) | logs `Worker proc %s died unexpectedly (exit code: %s), shutting down executor.`, sets `is_failed`, calls `shutdown()` |
| teardown | `:459-495` (`shutdown`) | `worker_response_mq.shutdown()` / `mq.shutdown()` |
| flag | `vllm/distributed/device_communicators/shm_broadcast.py:541-546` | `shutting_down = True`, `_spin_condition.cancel()` |
| the raise | `shm_broadcast.py:701` **and** the downstream patched copy `vllm_ascend/patch/platform/patch_shm_broadcast.py:70` | a parked reader wakes and re-raises `RuntimeError("cancelled")` |

So the message text exists in **both** upstream and the downstream patch: the
patch preserves the semantics while adding a spin/recheck loop, and the paste's
frame is the patched copy because that is the one on the call path. Neither file
is a suspect, and no vllm-ascend flag can suppress the raise.

**The discriminator that makes this a death and not a bug**: a Python exception
in a worker never produces this. `WorkerProc.worker_busy_loop` catches it
(`multiproc_executor.py:1003-1008`), logs `WorkerProc hit an exception.`, and
`enqueue_output` converts the object to a `ResponseStatus.FAILURE` response
(`:947-952`) — so the EngineCore re-raises *that* exception and its own
traceback is the answer. Only a process that stopped without enqueueing anything
leaves the reader cancelled. The dying rank's own log is therefore the only
place the cause exists, and it is missing from this paste by construction.

## Rule 2 — the dump block is the step in flight, not the failing op

`dump_engine_exception` (`vllm/logging_utils/dump_input.py:56-83`) has exactly
one call site: `core.py:486-496` `log_error_detail`, wrapped around
`model_output = future.result()` at `core.py:693`. The dump is produced by the
exception that is already tearing the engine down, so it names **the batch that
was being executed**, never a kernel, an op or a branch.

Read it as provenance and stop:

- the config echo carries the launch shape — `tensor_parallel_size`,
  `data_parallel_size`, `pipeline_parallel_size`, `max_seq_len`,
  `speculative_config`, `quantization`, `enforce_eager` /
  `compilation_config.cudagraph_mode`, `enable_chunked_prefill`;
- `SchedulerOutput.num_scheduled_tokens` is the chunk size of the failing step
  (8064 in #16023 — i.e. one chunked-prefill chunk of a ~100 k prompt), and
  `num_computed_tokens` / `step_counter` say whether it was the first chunk;
- `SchedulerStats.kv_cache_usage` plus `prefix_cache_stats` bound the KV pool.

Do not spend calls on why 8064 rather than an upstream default: it changes no
candidate (rule 7 of SKILL.md applies only when the number *is* the symptom).

## Rule 3 — one grep names the death, then ask for the artefacts

1. `rg -n 'died unexpectedly' <engine log>` → `(exit code: N)`.
   Decode: `-6` SIGABRT (native abort — device/ACL/GE error path, the usual
   device-fault signature), `-11` SIGSEGV, `-9` SIGKILL (host OOM killer or
   cgroup limit), `None` (no code reported; the segfault lane of
   `worker-segfault-no-python-frames.md` also shows this), `1` (a Python
   exception that exited after logging — then the rank's own traceback is the
   answer and `WorkerProc hit an exception.` should be in that log).
2. Per-rank worker log for **that node** (`Worker_TP*` / `Worker_DP*` lines; an
   EngineCore-only excerpt is a data gap, say so). On a PD deployment ask for the
   whole node's worker output, not just the DP rank the EngineCore line names.
3. plog for that second: `fault kernel_name=`, `5070xx`, `EZ9999`, `EE9999`.
   A bare code is one call: `bash <skill_root>/scripts/cann-error-lookup.sh <code>`.
4. `dmesg -T | grep -iE 'oom|killed process'`, and `npu-smi info`.

## Rule 4 — rank the causes while that log is missing

| Candidate | Check | Standing |
|---|---|---|
| **driver/firmware × CANN mismatch** (the "CANN 与 HDK 冲突" family) | `npu-smi info` header (driver + firmware), `cat /usr/local/Ascend/driver/version.info`, `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`; ask whether the box was upgraded piecewise (driver only, toolkit only, HDK only) | **Reporter-confirmed on #16023**: 排查 core/日志打点/打流 后 "定位到底层算子兼容性适配，CANN 与 HDK 存在冲突，环境的 firmware 与 driver 不匹配。升级到 26.1.1 后解决" — an environment upgrade, no code change. Rank it first for a *runtime device death on a long-context step* with no Python cause. Read both firmware and driver values (the `npu-smi` header prints them separately) instead of comparing either with the CANN release alone — the reporter's conflict was between those two lines |
| A long-chunk attention/linear-attention kernel fault | `--max-num-batched-tokens` one step down (e.g. 24576 → 8192 → 4096) and see whether the death moves or disappears; plog `fault kernel_name=` | Hypothesis; on Kimi-K3 the prefill lane runs the KDA op (`chunk_kda_fwd`) plus MLA, and 8064 tokens is 126 chunks — pure arithmetic, not evidence |
| HBM / workspace at long context | the dump's `kv_cache_usage`, `npu-smi` free HBM at the death, `../runtime-memory/index.md` budget split | Hypothesis; an allocation failure normally prints `NPU out of memory` / `EL0004` / `207001` before any kill |
| Host OOM (container working set at 100 k prompt) | `dmesg`, per-rank RSS (`../runtime-memory/host-ram-and-per-rank-rss.md`) | Hypothesis; exit code `-9` favours it |

Lanes the paste itself closes, so do not spend a rung on them:
`enforce_eager=True` + `cudagraph_mode: NONE` in the config echo means there is
no capture/replay phase to blame and `--enforce-eager` is not an ablation here
(SKILL.md rule 15); short requests succeeding only bounds the *input length*, it
does not test a single flag; and `patch_shm_broadcast.py` is not a defect (Rule 1).

## Rule 5 — deliver on the second read

This attribution is two file reads and one grep. Recorded #16023 read both files
(correctly) and then kept going — never emitting a message, so the reporter
received nothing — while the follow-up shows the answer that mattered was a
version question with a one-line fix. First message = verdict (`cancelled` is the
teardown; a worker died) + the `died unexpectedly` grep + the artefact list
(per-rank worker log, plog that second, `npu-smi info`, `version.cfg`,
`driver/version.info`, `dmesg`) + the ranked table above. Continue into source
work only if the reporter supplies the worker log.

## Limits

- Line numbers are the two commits above; upstream moves them between releases,
  so re-grep `died unexpectedly` and `raise RuntimeError("cancelled")` rather than
  quoting a line at another ref.
- The **timeout** sibling of this text is a different lane: a bare `TimeoutError`
  / `RPC call to execute_model timed out` out of the same `get_response` /
  `acquire_read` frames means the worker is alive but stopped answering
  (`rpc-watchdog-timeouts.md`, one call:
  `bash <skill_root>/scripts/executor-timeout-check.sh <paste|->`).
- "CANN 与 HDK 冲突 … 升级到 26.1.1 后解决" is the reporter's account of one A3
  case. No export in this snapshot carries a driver/firmware/CANN compatibility
  matrix (`SOURCES.json` → `unavailable`), so present it as a reporter-confirmed
  resolution path, not as a version rule, and keep the kernel-level candidates
  open.
- The `cancelled` frame is in a fatally-dead engine: a paste that also carries
  `Destroying…` / `Parent process exited` lines has already lost the causal
  ordering (SKILL.md rule 13 — check timestamps before quoting them).
