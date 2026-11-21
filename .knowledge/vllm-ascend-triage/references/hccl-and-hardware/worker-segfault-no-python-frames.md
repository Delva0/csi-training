# `!!!!!!! Segfault encountered !!!!!!!`: reading a signal death from its dump

## When this page applies

The log carries `!!!!!!! Segfault encountered !!!!!!!` and there is no Python
exception. Two phases carry it, and they are answered differently:

- **Runtime (the recorded case).** A **benchmark-client** failure
  (`ConnectionRefused [Errno 111]`, an evalscope/vllm-bench traceback,
  "推理正常、性能测试崩溃"), the marker plus a backtrace in the server log, then
  `Worker proc VllmWorker-<n> died unexpectedly (exit code: None), shutting down
  executor.` (`multiproc_executor.py`). Recorded #16079: vllm-ascend v0.26.0rc1
  (`f2f74a16`) + vllm v0.26.0 (`568afb3a`), 2 nodes × (tp=8, dp=4) with EP, a
  MTP/DFlash draft, `expandable_segments:True`, 128 K prompts, and the API server
  on the master dying mid-run.
- **Engine start.** The marker lands **before** `Starting to load model`, the
  server never becomes ready, and the parent side says `RuntimeError: Engine core
  initialization failed. See root cause above. Failed core proc(s):
  {'EngineCore': -11}` — `-11` **is** SIGSEGV. Recorded #12015 (310P,
  `v0.22.1rc1-310p-openeuler`, Qwen3-VL-2B fp16, `--enforce-eager`) and #10136
  (SIGSEGV after a CANN 9.0.0 + NNAL 9.0.0 install plus a source rebuild). See
  `## The startup variant`.

Distinguish both first from the two neighbours: an **OOM** leaves
`NPU out of memory` / `EL0004` / `207001` (`../runtime-memory/index.md`), and a
**Python exception** leaves a `WorkerProc hit an exception` traceback
(`../startup-failures/environment-correspondence.md`). A signal death leaves
neither — only the dump and the executor's own wording (`exit code: None`, or a
negative core-proc value).

The same family also arrives **without** this marker: when the paste's deepest
frame is `RuntimeError: cancelled` out of `patch_shm_broadcast.py` /
`shm_broadcast.py` under the `dump_input.py` dump block, the engine has already
noticed a dead worker and cancelled its response queue — same conclusion, same
missing cause, but the text looks like an ordinary Python error, so read
`cancelled-executor-teardown.md` first instead of the queue code.

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
   fp16 VL serve dying at engine start (#12015), `assert curr_block is not None`
   under load_async (#14283), and both `/start_profile` and `/stop_profile`
   (#14860, #14974). Several unrelated mechanisms share the marker, so pick the
   precedent by *mechanism and phase*, not by the string: #10136 is the only
   self-contained startup case and it is unresolved (an install/version-matrix
   question with no maintainer answer in the corpus).

## The startup variant

Nothing is served, so there is no client traceback and no `Replaying aclgraph`
provenance. Read five things and nothing else:

1. **Which process died.** A dump whose bottom frames are `Py_BytesMain` →
   `pymain_run_python` → `pymain_run_command` → `PyRun_StringFlags` belongs to
   the **EngineCore child** (`multiprocessing` spawn runs the child as
   `python -c "… spawn_main …"`), not to the APIServer. Its last log-prefixed
   line (`(EngineCore pid=N) …`) dates the death; the `(APIServer …)` traceback
   under it is the parent reporting the child.
2. **`Failed core proc(s): {'EngineCore': -11}`** = the child was killed by
   SIGSEGV, so there is no Python exception to wait for; asking the reporter for
   one spends a round trip.
3. **The `[ERROR] TBE Subprocess[task_distribute] raise error[], main process
   disappeared!` block is a messenger.** The TBE op-compile helpers notice the
   parent gone and print it *after* the death; it names no op and no cause
   (`../startup-failures/silent-worker-exit.md`).
4. **The top frame is only the last native call — check whether it is trivial.**
   A small **CPU** `torch.zeros` (`at::native::zeros_symint` → `zero_` →
   `fill_out` → `fill_kernel` → `at::native::DEFAULT::VectorizedLoop2d<…fill_non_native_type<c10::Half>…>`)
   allocates a few MB and cannot fault on its own: the process was already
   damaged (the image's own native init, a corrupted heap, or a signal delivered
   from another thread). Do not compute mask/KV bytes from it, and do not file it
   under the `max_model_len²` host-mask lane
   (`../runtime-memory/attention-mask-host-allocation.md`) — that lane has a
   *large* allocation and no dump. `at::native` also proves the tensor was on
   **CPU**; an NPU tensor dispatches to the `torch_npu` kernel and never shows
   this frame.
5. **Take the launch shape from the config echo, not the dump.** In #12015 the
   paste itself carried `--enforce-eager` **plus** a `FULL_DECODE_ONLY`
   `--compilation-config` (so the graph flags are void: "Enforce eager set,
   disabling torch.compile and CUDAGraphs"), a checkpoint outside the pinned
   doc's validated table, and a title naming a different model from the command.
   Name all three as findings, then ask which command actually failed.

Ranked candidates, one check each: the **image / CANN / torch_npu / source-build
pairing** (the #10136 family — `pip show vllm vllm-ascend torch torch_npu`,
`cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`, and whether
vllm-ascend was built on the node or came from the image); a **model/feature
combination the pinned doc does not validate** (diff the command against the
release's own tutorial table for that SoC); **host-memory pressure during init**
(explicit `--max-model-len`, `dmesg -T | grep -iE 'oom|segfault'`); and only then
a real code defect. What settles it is the **plog**
(`$HOME/ascend/log/…`, that second) plus a rerun of the release's validated
command on a small model — not more of the dump.

## Ranked candidates for the runtime variant (all unverified in #16079)

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
