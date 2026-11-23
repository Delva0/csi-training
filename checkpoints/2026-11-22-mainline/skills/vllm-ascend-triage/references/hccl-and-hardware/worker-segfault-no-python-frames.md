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
   dying rank's own log is the only place the cause appears. **Expect that frame
   on a rank that had no request, and expect it on the master while the dump is
   on the slave**: upstream runs a full dummy step on any DP rank with nothing
   ready while its peers are serving (`vllm/v1/engine/core.py:2055` and
   `has_unfinished_requests_dp` → `should_execute_dummy_batch` at
   `llm_engine.py:196-203`, vllm v0.26.0 `568afb3a`), and vllm-ascend serves it
   with `execute_dummy_batch`
   (`vllm_ascend/worker/worker.py:1079-1087`, whose `:1082` calls
   `self.model_runner._dummy_run(num_tokens, uniform_decode=True,
   skip_gdn_state_update=True)`), which reaches the DP-group `all_reduce` above.
   The recorded paste's own master traceback is that chain end to end —
   `_dummy_run` (`worker/model_runner_v1.py:3305`) →
   `_determine_batch_execution_and_padding` (`:2837`) → `_sync_metadata_across_dp`
   (`:725`, `dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)`, all
   at `f2f74a16`); `cpu_group` is why the message is **gloo**'s
   `pair.cc:547 Connection closed by peer` and not an HCCL error. So the recorded
   benchmark's `--parallel 1` against `--data-parallel-size 4` keeps three of the
   four ranks in that loop, and its first timestamped error (13:57:56, `DP1_TP6`
   on the master) **is** a dummy-stepping rank reporting a peer death, ~45 s
   before the slave's own engine noticed it (13:58:41, `EngineCore_DP2`:
   `VllmWorker-6 died unexpectedly (exit code: None)`) — read it as the messenger,
   not the origin. The master then only shuts down (`SIGTERM` 13:58:07, force kill
   13:58:12, `ApiServer_1 … died with exit code None`), so its block is not a
   second crash. That is also the cheapest ablation on the reporter's own harness:
   rerun at `--parallel` ≥ `--data-parallel-size` (their script already sweeps
   `--parallel 1 4 8 16 32 48` and dies in the first stage) so no rank is left
   dummy-stepping.
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
   **The profiler pair is the one sub-family this page does not treat as a
   mechanism hunt**: when the marker lands on `POST /start_profile` /
   `/stop_profile` (or on `Invocation of profile method failed` out of
   `vllm/v1/engine/core.py`), the dump is CPython's own free path in both #14860 and
   #15983 — it names no op and no thread — and the decisive fact is the reporter's
   cross-`torch_npu`-patch control plus the release's own `torch-npu==` pin. Read
   `../observability-and-logs/online-profiler-stop-segfault.md` and answer the patch
   lane; do not rank `all_gather`/event-resource candidates from the table below.

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

## Ranked candidates for the runtime variant (the peer's call site is in the
paste; the segfaulting rank's is not)

**Send this section's ladder before working the table.** Every row's check is a
reporter-side rerun, an ablation or a plog, so no row can be closed from the
snapshot and the call site stays a gap — the table is the reporter's, not yours.
Two recorded attempts at #16079 were cut exactly here (16 turns, 43.5 K
characters of reasoning against 3.3 K of text; then 16 turns, 44.7 K against
1.1 K) while re-checking the release-note model list and the
`enable_reduce_sample` default (both already here) and the `torch-npu` pin —
none of which is a row in the table below. Name the launch-shape keys as
provenance if they are in the paste — `--async-scheduling`,
`enable_balance_scheduling`, `enable_flashcomm1`,
`multistream_overlap_shared_expert`, a dflash/MTP draft with
`num_speculative_tokens`, `cudagraph_mode: FULL_DECODE_ONLY` with an explicit
`cudagraph_capture_sizes` list, `expandable_segments:True`,
`HCCL_OP_EXPANSION_MODE=AIV` — because no tutorial command produces them and the
release does not validate that combination.

| Candidate | Check |
|---|---|
| Event/stream resources exhausted at collective entry (event create is the allocating call) | count the capture sizes and replayed steps; `npu-smi`, `/proc/<pid>/fd` and thread count on the rank; retry with fewer `cudagraph_capture_sizes` / `--enforce-eager` |
| A per-step tp-group `all_gather` inside sampling/rejection sampling | `vllm_ascend/sample/rejection_sampler.py` `greedy_sample` (`tp_group.all_gather` twice per call) and the `Using fallback (non-reduce-sample) path … enable_reduce_sample=False` line; A/B `enable_reduce_sample: true`, then MTP off |
| An SP/flashcomm hidden-state gather at the end of every forward | the `enable_flashcomm1: true` key → `forward_context.flash_comm_v1_enabled` → `_all_gather_hidden_states` (`vllm_ascend/worker/model_runner_v1.py:2642`, once per forward plus once per Eagle3 aux state) → `tensor_model_parallel_all_gather`; A/B that one key off |
| A DP/EP collective sized from the padded batch | `<skill_root>/scripts/batch-split-audit.sh <vllm-ascend-root>`; the `rankSize` in the failing line against DP×TP |
| Image/CANN/torch_npu pairing (after a recent upgrade) | the release's own support table for the card and the `pip show` triple; see `../image-dependency-faults/index.md` for the abort shapes, which are different |

Also state the support caveat when it is true: v0.26.0rc1's release note is a
model-restricted release (`docs/source/user_guide/release_notes.md:5`) and names
the validated models (`Kimi K3`, `GLM-5.2`, `DeepSeek V4 Flash 0731`, `DeepSeek
V4 Pro 0813`), so a `kimi-k2.7-Code` report on it is outside the validated set —
say it, and do not turn it into the root cause. The pin's own stack for that
comparison is `torch==2.10.0` / `torch-npu==2.10.0.post4` (`requirements.txt:17`,
also `pyproject.toml:21`); quote it against the reporter's `pip show` rather than
asking them for a version list.

## Delivery shape for this report

Classification (rank dies by signal, no Python cause in the paste) + timestamps +
the ranked candidates with one check each + the artefact list (plog for that
second, the dying rank's own log, `npu-smi info`, `pip show
vllm vllm-ascend torch torch_npu`, both launcher scripts). The *segfaulting* rank's Python
call site is not derivable from a C-only dump, so the first answer is already
complete without it — do not open `dsa`/attention sources hoping the stack
reappears. **The dying rank's log may already be attached**: in #16079's case view
both node logs are present as `*.log` (the segfault dump in the slave's, the
dummy-batch traceback in the master's), so list
`/workspace/attachments/<issue>/` with `ls -la` before asking the reporter for a
log (`../evidence-workspace/case-layout.md` §*Attached log dumps* — a `*.txt` glob
hides exactly those files). Ask only for what the paste and the case view do not
carry: the plog for that second, `npu-smi info -t health` at that minute,
`pip show vllm vllm-ascend torch torch_npu`, and the working-versions launcher.
