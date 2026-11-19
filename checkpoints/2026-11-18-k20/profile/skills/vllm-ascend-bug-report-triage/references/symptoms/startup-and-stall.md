# Startup stage: silent workers, hangs, and where the memory went

## Scope

Reports that fail *during startup* — after the engine banner, before or around the first request —
where the deliverable is "which stage consumed what" rather than a wrong answer. Startup OOMs with a
specific frame (mask, weight load, KV cache) are routed to their own pages below; this page carries
the stall signature that tells you the stage, the "hang is not an error" rule, and the startup-stage
memory classes that do not own a page, plus two shapes that are not a worker at all: a multi-node DP
launch whose local ranks do not add up (§6) and a startup *time* that regressed between two tags (§7), and
§8, the wrapper message that tells you the worker was *killed* rather than raising.

## 1. Read the stall signature correctly

Signature: the last line from any worker/rank process is a normal startup message (weight load,
"Encoder cache will be initialized…"), then the engine core repeats
`shm_broadcast.py ... No available shared memory broadcast block found in 60 seconds. This typically
happens when some processes are hanging or doing some time-consuming work (e.g. compilation,
weight/kv cache quantization).`

Interpretation: the engine core is blocked waiting on workers stuck in startup work — weight load,
dummy-run memory profiling, attention-mask construction, a collective. It is a frontend symptom,
never the cause. **Look at the tail of the worker log, not the frontend lines.**

Observed in vllm-ascend #10872 (Atlas 300I DUO 310P3 ×4, `v0.21.0rc1-310p-openeuler`, Qwen3.6-35B-A3B
W8A8, TP4, `--enforce-eager`): after `Loading model weights took 10.5774 GB` no worker line follows,
while `shm_broadcast` repeats for five minutes; per-card usage from the attached `npu-smi` was
~17 GB of 21.5 GB. `[thread]` (attachment `default.txt`, itself fetched with `curl` from a comment's
`user-attachments/files/…` link). The memory went into the dense 310P mask — see
`attention-mask/310p-dense-mask.md`; size it (`max_seq_len² × 2 B × ~2`) before blaming the model.

## 2. Hangs are not errors

A worker stall surfaces to the user as a frontend timeout, not a traceback. Read the last worker log
line plus the frontend repetition pattern before hypothesising a code defect; and do not speculate
about hardware faults you cannot observe. For a pool/AICPU hang under PD, see `c8-and-kv-layout.md`;
for a *mid-run* freeze after successful serving, the CANN plog/HCCL heartbeat is the evidence
(`pd-collective-hang.md`); a device-kernel runtime error at `profile_run` is a different class again
(`device-kernel-faults.md`). A *serving* hang that begins at a prompt-length threshold and then blocks
every request behind it is a scheduler admission cliff, not a stall: `kv-admission-cliff.md`.

## 3. KV-cache allocation OOM right after weight load (hybrid models included)

Worked case: vllm-ascend #3308 — Qwen3-Next-80B-A3B-Instruct (hybrid: linear-attention layers plus
attention layers, which the v0.11.0rc0 KV-cache init handles separately), `v0.11.0rc0` (paired vLLM
`0.11.0rc3`, CANN `8.2.RC1`), TP8 on 8×910B4 32 GB cards, `--max-model-len 1024
--gpu-memory-utilization 0.95 --enforce-eager`.

Signature: the weights load (`Loading model weights took 18.6823 GB` on every rank), the engine logs
several `GPU KV cache size: <n> tokens` lines, then `RuntimeError: Worker with error ... NPU out of
memory` while the KV-cache tensors are created (allocator total ≈ 29.5 GiB, allocated ≈ 27.4 GiB,
~0.5 GiB free against a ~0.6 GiB request). This sits between the two neighbouring classes: nothing is
stalled, and the failing allocation is the KV cache, not the weights and not a request-time mask.

- **Lowest-risk workaround (attributed, maintainer `wxsIcey`):** `--gpu-memory-utilization 0.5` for
  TP8, plus `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32` and
  `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`. Those are one variable, so the two values are
  alternatives: try each, or combine them as a comma-separated value if the installed torch_npu
  accepts it. A user also reported success with `--gpu-memory-utilization 0.7
  --max-model-len 98304 --enforce-eager` on a patched `0.11.0rc0` wheel with a newer
  torch/torch-npu/bisheng/triton-ascend stack, i.e. a bundle, not an isolated flag.
- **Do not relay the closing summary as a fix.** The late automated comment says the OOM was fixed in
  `v0.11.0rc1` by PRs #3305 and #3557; both are closed and unmerged (a config fix and a doc update),
  so no commit is evidenced. Say "attributed, unverified" or omit it.
- **Mechanism is open.** The relevant code at `v0.11.0rc0` is
  `worker/model_runner_v1.py:2918 initialize_kv_cache_tensors`, which allocates one `torch.zeros(size,
  dtype=int8)` per `kv_cache_config.kv_cache_tensors` entry, splits ordinary attention groups into
  K/V halves, and shares a single tensor across the `linear_attn` layers of a tensor — with the
  in-tree marker `# TODO: REFACTOR ME to sharing hybrid cache`. The budget itself comes from the
  profile run's `determine_available_memory`. Confirm at the reporter's tag whether the per-group
  sizes sum above the reported available memory before claiming either accounting or sharing is at
  fault.
- **Why this is not #3357.** Both reporters call the two issues "the same error", but #3357 OOMs
  *during a request* inside `attention_mask.py` while #3308 dies at KV-cache init. Do not transplant
  the mask mechanism here (`attention-mask/prefill-dense-mask-a2-a3.md`).

## 3b. `max_model_len` larger than the KV budget: the check that fires *before* allocation

A tutorial can hand the reporter a `--max-model-len` this hardware cannot serve. Then vLLM rejects the
configuration before any block is allocated, and the correction is printed in the error:

```text
ValueError: To serve at least one request with the model's max seq len (1048576), (X GiB KV cache is
needed, which is larger than the available KV cache memory (Y GiB)). Based on the available memory, the
estimated maximum model length is N. Try increasing `gpu_memory_utilization` … or decreasing
`max_model_len` …
```

Verified source: `vllm/v1/core/kv_cache_utils.py` at vLLM `v0.30.0rc2` — `_check_enough_kv_cache_memory`
(`:872-905`, the `ValueError` at `:899`) is called from `get_kv_cache_configs` (`:1004`, and the group
variant `:2737`) with `vllm_config.model_config.max_model_len`, comparing `max_memory_usage_bytes(...)`
(`:912`) — the memory for one request at full length — against the per-worker KV budget, and
`estimate_max_model_len` (`:921`) binary-searches the length that fits. So a 1M `--max-model-len` on a
64 GB-per-NPU A2 fails deterministically, and the number in the message is the value to pin.

Worked case: vllm-ascend #18158 (`[Doc]`, DeepSeek-V4.1-Flash, Atlas 900 A2 PoDc 4 nodes × 8, HDK 25.5.2,
CANN 9.1.0, custom image). The guide the reporter copied is the defect: the page at `v0.30.0rc2` (the tag
its own `export IMAGE=…` names) pins `--max-model-len 1048576` in the A2 tab (`:691`,
`--gpu-memory-utilization 0.90`) for servers §3.1 describes as 64 GB per NPU, and its §11 says the A2
configuration is "retained unchanged and is not revalidated by this update". The tutorial exists only at
HEAD/`v0.30.0rc2` — `git show v0.18.0:docs/source/tutorials/models/…` has no `DeepSeek-V4.1-Flash.md` at
all — so pin the tag before comparing two versions of the page.

Deliverable for that class: the doc line (`file:line` at the tag the image names), the corrected block
(`--max-model-len` ≈ the estimate the engine printed, plus the `--gpu-memory-utilization` the tab uses),
and the disclaimers that contradict the block — the page's own §11, the sibling page and the release
notes; the full line set, the matrix row and the post-OOM half live in
`doc-guide-vs-capability.md`. Do **not** hunt a `vllm_ascend/` line: no device memory is visible to any
clone. The second half of such a report — "it still will not start after reverting" — is
residual-process/`/dev/shm`/HCCL-teardown territory (`npu-smi info`, stale `python` processes, the
image's `--shm-size`), owned by the operator; ship it as `hypothesis` with "`npu-smi info` shows no
process holding the card" as the falsifier.

Falsifier for the mechanism: if the log shows an NPU allocator OOM (`NPU out of memory`, a
`torch.zeros`/`empty` that failed) instead of this `ValueError`, the failure is before the KV-capacity
check and the lever is weight/HCCL/activation memory (§3), not `max_model_len`.

## 3c. `Available memory: 0`: the profile pass itself overshot the budget

`check_enough_kv_cache_memory` has **two** raises and the message says which one fired
(`vllm/v1/core/kv_cache_utils.py:678-701` at vLLM `v0.11.0`): `available_memory <= 0` raises the short
``ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization …`` at
`:698-701`, while `needed_memory > available_memory` raises the long message carrying `the estimated
maximum model length is {N}` (§3b). The short one means the **measurement** came back zero — or was
clamped at zero — so no flag that scales the KV pool helps: the number to fix is whatever sized the
measurement.

The measurement, at `v0.11.0` `vllm_ascend/worker/worker_v1.py:219-259`:

- `profile_run()` first (`:227`), then `peak_memory = torch_npu.npu.memory_stats()["allocated_bytes.all.peak"]`
  plus any `non_torch_allocations` (`:241-251`);
- `available = int(total_npu_memory * gpu_memory_utilization - peak_memory)`, clamped by `max(..., 0)`
  (`:252-255`) and logged as `Available memory: {N}, total memory: {M}` (`:256-258`).

`profile_run` is not a small warm-up: it dummy-runs **`self.max_num_tokens`**, which the runner takes
straight from the scheduler (`worker/model_runner_v1.py:256` `self.max_num_tokens =
self.scheduler_config.max_num_batched_tokens`; `:2567-2571` `_dummy_run(self.max_num_tokens,
with_prefill=True)`). Every mechanism that raises `max_num_batched_tokens`, including a platform rewrite
the reporter never asked for, therefore raises the *profile* size, and `Available memory: 0` is that
rewrite's footprint rather than a KV-capacity question.

Worked report: vllm-ascend #3843 — Qwen3-Next-80B-A3B-Instruct (hybrid, 48 layers), TP8 on 8 cards whose
log reports `total memory: 65452113920`, `--gpu-memory-utilization 0.5`, **no `--max-model-len`** (so
`Using max model len 262144` comes from the checkpoint), `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`:

```text
[scheduler.py:205] Chunked prefill is enabled with max_num_batched_tokens=2048.
[platform.py:141] Non-MLA LLMs forcibly disable the chunked prefill feature, …
… 23 minutes later, on all eight ranks:
[worker_v1.py:259]  Available memory: 0, total memory: 65452113920
[core.py:708]        ValueError: No available memory for the cache blocks. …
```

The rewrite is `vllm_ascend/platform.py:139-163` **at `v0.11.0rc0`** `[verified]`: when
`not model_config.use_mla and not scheduler_config.async_scheduling` it logs that line, and then — with
the structured-output backend `auto` (default, `vllm/config/structured_outputs.py:20`), policy `fcfs`
(`vllm/config/scheduler.py:107`), `send_delta_data` off (`:101`) and no scheduler delay factor — it
enables `ascend_scheduler_config`, whose `enable_chunked_prefill` defaults `False`
(`vllm_ascend/core/schedule_config.py:28`), and sets
`scheduler_config.max_num_batched_tokens = scheduler_config.max_model_len` (`:160-163`). Every condition
holds for this command, so the profile became one 262 144-token prefill: that is the 23-minute gap between
the platform line and the `Available memory:` line, and that is why `--gpu-memory-utilization 0.9`
(`reported`: 「没用，都试了」) changes nothing — the peak moved, not the fraction.
`--max-num-batched-tokens` by itself is overwritten (`../vllm-ascend/config-flag-lifetimes.md`, Trap D).

**Version scope, and how the log settles it** `[verified]`: the block is present at `v0.11.0rc0` (that
message on `platform.py:142`) and **absent from the `v0.11.0` tree** — removed on main by #3967
(`a1558b99c`, 2025-11-10, first tag `v0.12.0rc1`) and cherry-picked to the release branch as `ebd45b659`
(#4094, in `v0.11.0rc1` and `v0.11.0`, absent from `rc0`). The reporter's `platform.py:141` is that
pre-removal code, one line off rc0, so an image tagged `vllm-ascend:v0.11.0` can run scheduler semantics
its own tag no longer has: date the build from the log line, and a build of the same release line from
`rc1` on for this mechanism — never from the tag name (`../workflow/version-pinning.md`).

Levers on such a build, cheapest first:

1. `--max-model-len 32768` — the rewrite then sizes the profile at 32 768 tokens. Cost: the 256 K context
   the reporter wanted. The thread's own working shape is this plus `--gpu-memory-utilization 0.8
   --max-num-batched-tokens 4096 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
   from `zhangkx-777`, and a maintainer linked a dedicated Qwen3-Next-80B-A3B-Instruct image
   (`Ascend-SACT/…_vllm-ascend`).
2. `--additional-config '{"ascend_scheduler_config": {"enable_chunked_prefill": true}}'` — the rewrite is
   skipped, `max_num_batched_tokens` keeps vLLM's 2048 and chunked prefill bounds the prefill allocation.
3. `--async-scheduling` — the outer condition (`not scheduler_config.async_scheduling`) skips the block
   entirely; the plugin's own interaction with async scheduling is a gap to declare.

Falsifiers: a plausible `Available memory:` number followed by the long `To serve at least one request …`
message is §3b; `NPU out of memory` from an allocator instead of the `ValueError` is §3. If a start with
`--max-model-len 32768` still prints `Available memory: 0`, the peak is in weight load or compile, not in
the dummy batch, and the case is hybrid/mamba-state accounting (§3) — do **not** answer a hybrid model by
insisting on `--gpu-memory-utilization` alone.

## 4. Startup networks: port bind, address already in use

`zmq.error.ZMQError: Address already in use` in `shm_broadcast.py` during `_init_message_queues`
comes from vLLM core's port allocation, not from a KV connector or the Ascend stack, and the peer
node usually fails with `Connection closed by peer`: `startup-port-conflicts.md`. No `HCCL_*`/`GLOO_*`
setting and no vllm-ascend flag addresses it.

## 5. "Each card loads the whole model" with no error — per-card accounting, not a TP bug

A report with no traceback, no OOM and a healthy service saying that a TP job "loads the entire model on
every card" (or that every NPU sits at ~90 % of HBM) is the *expected* memory identity: the KV pool is
sized to fill `--gpu-memory-utilization` of the card, so every rank ends near the same total whatever its
weight shard, and the alternative reading is one process per device (full replication, no
`--tensor-parallel-size`). Worked case #1316; the arithmetic, the two shapes and the discriminating
numbers (the per-rank `Loading model weights took … GB`, the `# NPU blocks` lines, `npu-smi`'s
per-process table, the parallel-config banner) are in `memory-sizing-and-parallelism.md` — answer from
there, do not hunt a `vllm_ascend/` sharding line.

## 6. A multi-node DP launch that hangs with only the engine banner — count the launched ranks first

Signature: `vllm serve` on every node prints its banner (`Available plugins for group
vllm.platform_plugins`, `Registered model loader …`, the `non-default args:` dict, maybe
`Started DP Coordinator process`, then the plugin banners again as the local cores start) and then
**nothing at INFO level on any node**: no `Loading model weights`, no worker, no HCCL, no traceback. The
failing and the working command differ only in the parallel shape — "DP4 TP4 does not start, DP2 TP8
does". When engine **cores** never start, the stall is in vLLM's own startup handshake, not in the model,
the quantiser or the NPU stack.

**Do the arithmetic before any code read.** The number of DP ranks the deployment actually starts must
equal `--data-parallel-size`: across nodes that is the **sum of `--data-parallel-size-local`**, and each
headless node's `--data-parallel-start-rank` must be the running total of the earlier nodes. A two-node A2
deployment that passes `--data-parallel-size 4 --data-parallel-size-local 1` on *each* node starts ranks 0
and 1 only. `4 DP × 4 TP = 16` cards on `2 × 8` is two DP replicas per node, so `--data-parallel-size-local
2` there.

Verified at vLLM `v0.13.0` (no plugin line is involved):

- `vllm/config/parallel.py:84` `data_parallel_size: int = 1`, `:87` `data_parallel_size_local: int = 1`,
  and the only validation is `:286` `data_parallel_size_local > data_parallel_size` — nothing checks that
  the **sum across nodes** reaches `data_parallel_size`.
- `vllm/v1/engine/utils.py:841-848`: `elif dp_rank == 0:` builds `engines_to_handshake = [CoreEngine(index=i,
  local=(i < local_engine_count)) for i in range(dp_size)]` — rank 0's frontend handshakes with **every**
  DP engine, local and remote.
- `wait_for_engine_startup` (`:914`) seeds `conn_pending, start_pending = [local_count, remote_count],
  [0, 0]` (`:924-927`) and polls in `while any(conn_pending) or any(start_pending)` (`:941`); its only
  progress lines are `logger.debug("Waiting for %d local, %d remote core engine proc(s) to connect.")`
  (`:945-952`). At the default INFO level a short launch therefore waits **silently and forever** for
  ranks that were never started.

Worked case **#5942** (`vllm-ascend v0.13.0rc1` + vLLM `0.13.0`, 2× Atlas 800I A2, DeepSeek-V3.2-w8a8,
`--data-parallel-size 4 --data-parallel-size-local 1` on both nodes, `--tensor-parallel-size 4
--enable-expert-parallel`, `FULL_DECODE_ONLY`, MTP 2; DP4 TP4 hangs while the same script with DP2 TP8
starts): the reporter resolved it himself in the thread — `--data-parallel-size-local 2` on both nodes and
`--data-parallel-start-rank 2` on the headless node, plus `--gpu-memory-utilization > 0.95` for the
separate OOM. Cost: none, it is pure config (the reporter's own 「问题已解决」 names both flags); the cards are then
fully used.

- **Do not hunt a defect here.** At that tag `vllm_ascend/` reads `data_parallel_size_local` in exactly two
  places (`worker/worker.py:208`, `distributed/mooncake_connector.py:998`) and neither accounts ranks. The
  reply needs the sum-vs-total check and the debug-level retry line; a recorded run spent six messages
  inside the vLLM clone on the wait loop and shipped nothing.
- **A "duplicate of #5399, fixed by #5764" is not this stall**: #5764 (`pulls/5764`, merged, `base: main`)
  is the MLAPO/`bmm_transpose` input-constraint fix for a run-time `RuntimeError: bIndex is out of range`,
  not a startup handshake.
- **Falsifier**: `VLLM_LOGGING_LEVEL=DEBUG` is the one lever that makes the missing count visible (the
  loop's `Waiting for %d local, %d remote core engine proc(s) to connect.` names the shortfall), so it is
  also the reply's check: if the same job with `--data-parallel-size 2 --data-parallel-size-local 1` starts
  normally, the rank arithmetic is confirmed rather than a collective or HCCL fault. With the sums matching
  and the stall persisting it is a real handshake failure — `--headless` placement, the RPC port and
  `VLLM_RPC_TIMEOUT` are the suspects.

## 7. `init engine (profile, create kv cache, warmup model) took N seconds` — a cross-tag startup-time regression

That line is vLLM's `EngineCore` timing exactly one call,
`self.model_executor.initialize_from_config(kv_cache_configs)` (`vllm/v1/engine/core.py:167-172` at
`v0.9.1`, `:210-215` at `v0.10.1`), and the executor's `initialize_from_config` issues
`collective_rpc("compile_or_warm_up_model")` (`vllm/v1/executor/abstract.py:58-66`, same lines at both
tags). So the number is
profile run + KV allocation + **graph capture / warmup**, and the question is not "what is slow" but
"what newly participates in compile/capture on the newer tag".

Diff the two tags on the files that decide participation and read the one hunk:

```bash
git diff v0.9.1 <newer-ref> -- vllm_ascend/platform.py vllm_ascend/ascend_config.py
```

Worked report #3045 (`main` at 2025-09-19: `52.96` s; `v0.9.1`: `3.79` s; the whole environment
section was a `diffchecker.com` link, so no flag list exists to read). The decisive delta in that window
is `88ca8a051` (#2707 `[Feat][Graph] Support DeepSeek with ACL Graph`, 2025-09-16, first tag
`v0.11.0rc0`): it deletes the hard block at `v0.9.1` `ascend_config.py:231-240` —

```text
if "deepseek" in model_type:
    raise NotImplementedError("ACL Graph does not support deepseek. … Or set `enforce_eager=True` …")
```

— and adds `vllm.mla_forward` to `splitting_ops` (`platform.py:230-235`). So on `v0.9.1` a DeepSeek
serve *cannot* be on the ACL-graph path (eager or torchair only), while from that commit the same
command captures piecewise graphs and pays the capture inside the timer. Two hypotheses this kills:
the default graph mode did **not** flip (both tags default to `PIECEWISE` — `v0.9.1 platform.py:234-242`,
main `:224-235`), and `cudagraph_num_of_warmups` already exists at `v0.9.1`
(`worker/model_runner_v1.py`, `compilation/piecewise_backend.py`).

- **Lever** (proposed; also the cheapest discriminator): `--enforce-eager` on the newer tag reproduces
  the older path, cost = no ACL graph; a smaller `cudagraph_capture_sizes` / `max_cudagraph_capture_size`
  cuts the capture count instead. If the time does not move under `--enforce-eager`, the delta is
  weight load / HCCL / MoE init, not capture — say which one, do not guess.
- **Deliverable**: the two tags, the deleted/added gate with `path:line` and first tag, the lever, and
  the missing environment diff in `limits`. A report whose environment block is a third-party
  paste/diff URL has no fetchable flags (`../workflow/evidence-sources.md`).

## 8. `WorkerProc initialization failed due to an exception in a background process` with no child traceback: the child was *killed*, it did not raise

Signature: the engine core prints `EngineCore failed to start.` with a traceback that ends in
`Exception: WorkerProc initialization failed due to an exception in a background process. See stack trace
for root cause.` — and there is **no child stack trace anywhere**, in any rank's log and in plog. Often the
worker's last normal line is a load-time message (`Loading drafter model..`), the parent then logs
`[shutdown] Executor: workers still running after grace period; sending SIGTERM/SIGKILL count=<n>`, and the
dead worker's own subprocesses print `[ERROR] TBE Subprocess[task_distribute] raise error[], main process
disappeared!`. Those TBE lines are the **children of the dying worker noticing it is gone**, and the SIGTERM
is the parent's cleanup — neither is the cause. `[verified]` in vLLM `v0.23.0` (the tag
`v0.23.0rc1:.github/vllm-release-tag.commit` pins for a `v0.23.0rc1` image):

- `vllm/v1/executor/multiproc_executor.py:732-768` `WorkerProc.wait_for_ready` waits on each worker's
  ready pipe and does `pipe.recv()`; **only `EOFError` is caught** — `e.__suppress_context__ = True;
  raise e from None` (`:760-762`). `e` is the generic message above, so the parent deliberately chains
  nothing: the "root cause" it points at is text the *child* was supposed to print first.
- The child (`worker_main`, `:807-899`) sends `{"status": "READY", …}` only after `load_model()` returns
  (`:855-869`), and on a Python exception during init it runs `logger.exception("WorkerProc failed to
  start.")` (`:887-888`) — the full traceback, in that rank's own log stream — before closing the pipe.

The reporter's own frames are the tag fingerprint: `multiproc_executor.py:201`
`self.workers = WorkerProc.wait_for_ready(unready_workers)` and `:762` `raise e from None` are exactly those
lines at `v0.23.0`, so the image's vLLM half is that tag — quote it instead of guessing a pairing.

So the diagnosis is a grep, not a theory: **does `WorkerProc failed to start.` occur anywhere?**

```bash
grep -rn "WorkerProc failed to start" *.log ; journalctl -k | grep -iE "killed process|out of memory" 
```

- present → read *that* traceback; the parent's message is only the wrapper (the rank's log is often a
  separate per-rank file, so grep before concluding it is absent).
- absent and plog clean → the child died by a signal (host OOM-killer, `SIGSEGV`/`abort()` in native
  code, a container cgroup kill). Evidence lives outside vLLM: `dmesg`/`journalctl -k` around the
  timestamp, `ulimit -c` / core files, the cgroup's `memory.events`. Ask for exactly that, not for a
  re-run.
- Either way the deliverable is *how to capture* the first error plus the levers that remove the most
  load from the dying stage; a mechanism nobody can see is `hypothesis` with a falsifier.

Worked report **#12891** (`vllm-ascend`, image `0.23.0RC1`, GLM-5.2-w8a8, 2×Atlas 800T A2, one
`vllm serve` per node with `--data-parallel-size 2 --data-parallel-size-local 1` + `--data-parallel-start-rank 1`,
TP8, `--enable-expert-parallel`, `deepseek_mtp` 5, `--async-scheduling`, `FULL_DECODE_ONLY`,
`multistream_overlap_shared_expert: true`): the log is exactly this shape — `Loading draftter model..` on
one rank, then the grace-period SIGTERM, the TBE lines, and no rank traceback; `【Plog日志】无报错`. The
thread (5 comments, stale-closed) contains **no lever**: two maintainers and a bot ask only for the initial
error log ("The error log you posted does not show the initial error message"). Two facts that save a batch
on a recurrence `[verified at v0.23.0rc1]`:

- Its serve block is **byte-identical** to the tag's own A2 two-node block in
  `docs/source/tutorials/models/GLM5.2.md:292-402` (env list, `--max-num-seqs 16`, `--block-size 128`,
  `--safetensors-load-strategy prefetch`, the MTP JSON with `enforce_eager`), so "you copied the wrong
  block" is ruled out — do not spend a batch diffing it. That page states the w8a8 hardware floor as
  2×A2 (64G×8) at `:20`, matching the report.
- The one line they deviated from is the doc's `export VLLM_VERSION=0.21.0` (`:321`, `:378`), which they
  commented out; leaving it **in** would be worse, not better — see `../vllm-ascend/version-pairing.md`
  ("`VLLM_VERSION` is a switch, not a label"). Their commenting it out is the default install, so it is
  not the fault; mention it only as a `limits` line if the doc block is the subject.

What survives as a reply for this class: the wrapper-vs-child rule above, the grep, the two doc facts, the
memory/prefetch hypothesis with its falsifier (`--safetensors-load-strategy prefetch` on 8 ranks per node
plus the MTP drafter is the load-time page-cache/NAMED-memory pressure the report's own log shows at
`weight_utils.py:856-879`), and the request for `dmesg` + the rank's own log.

## Routes

- KV-cache init dies with `Available memory: 0` on every rank, then `No available memory for the cache
  blocks` → §3c: the profile pass, not the KV check; bound the dummy batch with `--max-model-len`.
- Multi-node DP (`--headless`, `--data-parallel-address`) stalls right after the banner or after
  `Started DP Coordinator process`, nothing else logged → §6: sum the launched
  `--data-parallel-size-local` values against `--data-parallel-size` before reading any code;
  `VLLM_LOGGING_LEVEL=DEBUG` prints the missing core count.
- Worker silent after weight load + `shm_broadcast` repetition → §1, then `attention-mask/` (310P
  mask) or the profile run.
- No error at all, but "the whole model is on every card" / every NPU near 90 % of HBM → §5 and
  `memory-sizing-and-parallelism.md` (pool accounting or one process per device), not a TP defect.
- OOM before any request while weights are being created → `quantization-load.md`.
- OOM while the KV cache is being allocated after a successful weight load → §3.
- Startup refused because `max_model_len` cannot fit the KV budget, often from a copied tutorial
  command on smaller-per-NPU hardware → §3b (the message prints the estimated length; the doc block is
  the defect), then `doc-guide-vs-capability.md` for the doc lines and the post-OOM half.
- OOM during a request that names `attention_mask.py` → `attention-mask/prefill-dense-mask-a2-a3.md`.
- `ZMQError: Address already in use` / `Connection closed by peer` at worker init →
  `startup-port-conflicts.md`.
- `init engine (profile, create kv cache, warmup model) took N seconds` far above an earlier tag's,
  nothing failing → §7: diff `platform.py` + `ascend_config.py` across the two tags for what newly
  enters compile/capture; `--enforce-eager` is both the lever and the discriminator.
- Crash inside a device kernel (DFX line, runtime error code) → `device-kernel-faults.md`.
- `Exception: WorkerProc initialization failed due to an exception in a background process` with no
  child traceback, TBE `main process disappeared!` lines, plog clean → §8: grep for
  `WorkerProc failed to start` before theorising; absent means the child was killed.
