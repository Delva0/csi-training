# A "communication timeout" with no rank and no code: vLLM's own RPC watchdog

Recorded #1865: vllm-ascend `v0.9.2rc1` (`b5b7e0e`), vllm `v0.9.2` (`a5dd03c1`),
`设备800TA2, box16` (16 cards), DeepSeek-R1-W8A8, `--data-parallel-size 4
--tensor-parallel-size 4 --enable-expert-parallel`, `torchair_graph_config` and
`ascend_scheduler_config` on. The report was a **screenshot** whose title reads
多进程卡间通讯超时 ("multi-process inter-card communication timeout").

Read this page whenever the only failure text is a bare `TimeoutError` whose
frames are `multiproc_executor.py` and `shm_broadcast.py`. It is not an HCCL
or NIC timeout, and the paste usually cannot name the owner — so the deliverable
is the classification plus the per-rank artefacts to request.

## What the frames are (verified at the pinned commits)

| Frame in the paste | Site (vllm `v0.9.2`, `a5dd03c1`) | What it means |
|---|---|---|
| `EngineCore ... run_busy_loop` → `_process_engine_step` → `execute_model` | `vllm/v1/engine/core.py` | the step whose reply never came |
| `collective_rpc` → `get_response` → `dequeue(timeout, cancel)` | `vllm/v1/executor/multiproc_executor.py:195-215` | caller waits on the worker's response mailbox |
| `with self.acquire_read(timeout, cancel) as buf` | `vllm/distributed/device_communicators/shm_broadcast.py:502-507` (`dequeue`) | the mailbox read |
| `raise TimeoutError` (no message) | same file `:469` (`acquire_read`); also `:403` (`acquire_write`) and `:526` (`recv`, socket path) | deadline expiry inside the ring buffer |
| `TimeoutError: RPC call to <method> timed out.` | `multiproc_executor.py:220-221` | re-raise that names the method only |

The deadline comes from the caller: `MultiprocExecutor.execute_model` passes
`timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
(`multiproc_executor.py:163`), which is `300` seconds default (`vllm/envs.py:134`,
env override `:933-934`). `collective_rpc` turns it into an absolute deadline
(`:176`) and re-raises as above. `check_health` uses 10 s (`:269`) — a
`check_health` timeout means the worker was already dead before the step.

So: the reporter's "card-to-card communication timeout" **is the framework's
watchdog for one worker that stopped answering**, not a message from HCCL. The
same mailbox also produces the DEBUG line
`No available shared memory broadcast block found in 60 second(s)`
(`VLLM_RINGBUFFER_WARNING_INTERVAL`, `vllm/envs.py:15`) — same mechanism, quieter
level.

## What it proves, and what it cannot

- Proves: no reply arrived to that method from at least one worker within the
  bound; the engine core then logged `EngineCore encountered a fatal error` and
  everything after it is teardown.
- Proves the *side* that stalled, via the frame: `acquire_read` = the caller is
  waiting for a reply; `acquire_write` = a reply could not be written because the
  readers are not consuming (the stalled side is the reader set, not the writer
  in the frame); `recv` = the socket/overflow path, i.e. a remote reader.
- Does not prove: which rank, which operator, OOM, or a link fault. The
  `TimeoutError` carries no rank, no op, no error code, and the worker's own
  traceback is *not* in this paste. Say that and ask for the per-rank log.
- An OCR'd screenshot is enough to classify by **file name plus function name**
  and never by line number: the recorded OCR rendered `507` as `50 不` and the
  raise line as `409`. Quote the function, footnote the line.

`bash <skill_root>/scripts/executor-timeout-check.sh <paste|->` prints the family,
the direction and the next artefacts in one call.

## Why DP>1 makes one stalled rank hang everyone (source-verified, v0.9.2rc1)

- Every MoE layer does a **cross-DP** collective when `dp_size > 1`:
  `vllm_ascend/ops/fused_moe.py:1343` `get_dp_group().all_gather(hidden_states, 0)`
  (router logits at `:1347`), i.e. each decode step waits for all DP groups.
- Token counts are equalised per step by a MAX all-reduce on the DP CPU group:
  `vllm_ascend/worker/model_runner_v1.py:546-556`, injected at `:1013-1018`.
- `--enable-expert-parallel` sets `expert_tensor_parallel_size = 1` and
  `expert_parallel_size = world_size_across_dp // etp`
  (`vllm_ascend/platform.py:137-149`), so the MoE collectives span all cards.

Consequence: with DP×TP cards and EP, **any** rank that stalls (device fault, OOM,
graph replay stuck, a collective entered by a different set of ranks) converts
into a whole-group stall and then into this 300 s `TimeoutError`. No exotic defect
is required. Do not present this as proof of which rank; it is why the watchdog
firing is expected in this configuration at all.

## Ranked causes, each with its one check (worked #1865)

1. **One worker stopped on the device** (OOM, `EZ9999`, HCCL hang, silent death):
   per-rank worker log for the last successful step, plog for the same minute
   (`plog-timeline.sh`), and `npu-smi info` taken *while the process is still
   hung*. Highest prior in almost every reported case.
2. **Graph-mode padded batch differs per rank.** With `torchair_graph_config`
   on, the MoE-side pad is deliberately skipped
   (`if not self.torchair_graph_enabled`, `fused_moe.py:1329-1342`) and the
   padded batch is chosen in the runner (`model_runner_v1.py:1019`). A rank whose
   padded batch disagrees enters the all-gather with a different shape. Check:
   re-run with `--enforce-eager`; if it passes, the graph path owns it.
3. **The command was copied from a different recipe.** In #1865 the serve command
   was the multi-node tutorial's node0 block with
   `--data-parallel-size-local/-address/-rpc-port` and the HCCL env exports
   (`HCCL_BUFFSIZE=1024`, `HCCL_IF_IP`, `*_SOCKET_IFNAME`, `OMP_*`) deleted. On one
   node the DP flags may be dropped, but a wrong/missing socket-interface or an
   undersized `HCCL_BUFFSIZE` can hang a collective instead of erroring. Check:
   diff the command and env against the recipe it came from.
4. **A real collective timeout** (rare in this shape): only if the plog carries its
   own timeout/EI code inside the incident window — then this is lane C and
   `collective-error-chain.md` applies, not this page.

Actions in order: env + flags from the recipe → `--enforce-eager` re-run →
`--enable-expert-parallel` off (EP=TP, no cross-DP MoE collective) → TP=8/DP=2 →
`ASCEND_LAUNCH_BLOCKING=1` re-run for attribution. Raising
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` buys observation time; it is never the fix.

Ask back for: the full per-rank log (every `WorkerProc` / rank-prefixed line), the
plog covering the failure minute, `npu-smi info` from during the hang, the CANN /
driver / torch_npu versions, and whether any timeout env var was raised.

## Scope

Bound to vllm `v0.9.2` line numbers: re-grep `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
and the `raise TimeoutError` sites per version before quoting lines, and re-read
the DP/EP anchors (`fused_moe.py`, `model_runner_v1.py`, `platform.py`) if the
deployment is graph-mode or EP. The classification — *watchdog, not HCCL* — holds
whenever the frames are `multiproc_executor.py` + `shm_broadcast.py`; if the paste
also carries `hcclRet`, `ERR00100` or `EI…` text, run the plog timeline first
(that is a different, lane-C paste).
