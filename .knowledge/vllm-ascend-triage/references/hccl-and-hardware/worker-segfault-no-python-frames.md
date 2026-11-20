# A rank dies by segfault inside a collective, and the dump has no application frames

## When this page applies

The report is a **benchmark-client** failure (`ConnectionRefused [Errno 111]`,
an evalscope/vllm-bench traceback, "推理正常、性能测试崩溃") and the server log
carries `!!!!!!! Segfault encountered !!!!!!!` followed by a backtrace, then
`Worker proc VllmWorker-<n> died unexpectedly (exit code: None), shutting down
executor.` (`multiproc_executor.py`). Recorded #16079: vllm-ascend v0.26.0rc1
(`f2f74a16`) + vllm v0.26.0 (`568afb3a`), 2 nodes × (tp=8, dp=4) with EP, a
MTP/DFlash draft, `expandable_segments:True`, 128 K prompts, and the API server
on the master dying mid-run.

Distinguish it first from the two neighbours: an **OOM** leaves
`NPU out of memory` / `EL0004` / `207001` (`../runtime-memory/index.md`), and a
**Python exception** leaves a `WorkerProc hit an exception` traceback
(`../startup-failures/environment-correspondence.md`). A signal death leaves
neither — only the dump and `exit code: None`.

## Read the paste in this order

1. **The client traceback is secondary.** `ConnectionRefused` at
   `127.0.0.1:<port>` means nothing was listening any more; the API server died
   first. Find the earliest in-window server-side fault and give its timestamp
   (rule 13: re-printed records are common) — everything else in the client log
   is retry noise.
2. **Separate the death from its messengers.** In the recorded paste the other
   DP ranks failed in a gloo `all_reduce` inside
   `_sync_metadata_across_dp` (`Connection closed by peer`) and the engine ended
   with `EngineDeadError`. Those frames are peers noticing a dead rank; the
   dying rank's own log is the only place the cause appears.
3. **Read the dump in the right direction.** The *top* frames are the last
   native calls — `rtEventCreateExWithFlag` → `aclrtCreateEventExWithFlagImpl` →
   `c10_npu::NPUEvent::createEvent` → `NPUEvent::record` →
   `c10d_npu::ProcessGroupHCCL::_allgather_base` in the recorded case — and the
   *bottom* frames are CPython's own entry (`PyEval_EvalCode`,
   `pymain_run_command` …). What matters is what is **missing**: count the
   application frames with `grep -c '\.py"' <block>`. Zero `vllm_ascend/*.py`
   frames means the paste does not say *which* `all_gather` this was, so the
   call site is a data gap, not a conclusion. Ask for the plog around that
   second, and for a rerun with `--enforce-eager` and
   `ASCEND_LAUNCH_BLOCKING=1` (the gate raises
   `ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1`, so eager is
   mandatory with it — same rule as `collective-error-chain.md`).
4. **Take the free context the log already gives.** In the recorded paste two
   `Replaying aclgraph` lines (`acl_graph.py:249`) precede the dump by ~2
   minutes of served traffic, so the process was past capture and dying inside
   a replayed step's collective — the paste also prints the timeout-free
   `max_num_scheduled_tokens is set to N based on the speculative decoding
   settings` warning at start-up;
   `enable_reduce_sample=False` (the default, `vllm_ascend/ascend_config.py:346`
   at this ref) decided which sampling path ran. Quote such lines as
   *provenance of the launch shape*, never as the cause.
5. **One history search names the family** (`bash <skill_root>/scripts/history-titles.sh
   'Segfault|rtEventCreate'`). At this snapshot the `!!!!!!! Segfault
   encountered` shape covers: a SIGSEGV right after a CANN/NNAL upgrade with a
   rebuilt vllm-ascend (#10136), GLM-5.1 PD P-node crash with a failing kernel
   attribute (#10522), a Mooncake KV-pool transfer timeout (#10532), a 310P
   W8A8 serve (#12015), `assert curr_block is not None` under load_async
   (#14283), and both `/start_profile` and `/stop_profile` (#14860, #14974).
   Several unrelated mechanisms share the marker, so pick the precedent by
   *mechanism*, not by the string.

## Ranked candidates, each with its one check (all unverified in #16079)

| Candidate | Check |
|---|---|
| Event/stream resources exhausted at collective entry (event create is the allocating call) | count the capture sizes and replayed steps; `npu-smi`, `/proc/<pid>/fd` and thread count on the rank; retry with fewer `cudagraph_capture_sizes` / `--enforce-eager` |
| A per-step tp-group `all_gather` inside sampling/rejection sampling | `vllm_ascend/sample/rejection_sampler.py` `greedy_sample` (`tp_group.all_gather` twice per call) and the `Using fallback (non-reduce-sample) path … enable_reduce_sample=False` line; A/B `enable_reduce_sample: true`, then MTP off |
| A DP/EP collective sized from the padded batch | `<skill_root>/scripts/batch-split-audit.sh <vllm-ascend-root>`; the `rankSize` in the failing line against DP×TP |
| Image/CANN/torch_npu pairing (after a recent upgrade) | the release's own support table for the card and the `pip show` triple; see `../image-dependency-faults/index.md` for the abort shapes, which are different |

Also state the support caveat when it is true: v0.26.0rc1's release note is a
model-restricted release and names the validated models (`Kimi K3`,
`GLM-5.2`, `DeepSeek V4 Flash 0731`, …), so a `kimi-k2.7-Code` report on it is
outside the validated set — say it, and do not turn it into the root cause.

## Delivery shape for this report

Classification (rank dies by signal, no Python cause in the paste) + timestamps +
the ranked candidates with one check each + the artefact list (plog for that
second, the dying rank's own log, `npu-smi info`, `pip show
vllm vllm-ascend torch torch_npu`, both launcher scripts). The Python call site
is *not* derivable from a C-only dump, so the first answer is already complete
without it — do not open `dsa`/attention sources hoping the stack reappears.
