# Device and communication cases: HCCL, cluster and watchdog

The `../hccl-and-hardware/`-owned half of the device-and-communication family:
collective errors, communicator creation, transport and link establishment,
MC2/AIV dispatch and profile-run sync points, the RPC watchdog, cancelled-executor
teardown, multi-instance and stack-held resources, and the latched-error and
segfault dumps that topic owns. Routes: `../hccl-and-hardware/index.md`,
`../runtime-memory/index.md`, `../pd-disaggregation/index.md`.

Inherits the scope and rules of `recorded-cases/index.md`, whose page for this
family carries one line per case in this file and in
`recorded-cases-device-and-communication.md` (the device, op and capture half).

**#13688** image `vllm-ascend:v0.22.1rc1-a3` (pin v0.22.1rc1 `5f6faa0`), torch_npu
2.10.0, CANN 9.0.0, 910C ×4 double-die = 8 chips; GLM-4.6V TP8+EP8 with
`HCCL_OP_EXPANSION_MODE=AIV`: `rtEventRecord execution failed, reason=task not
supported` + `507009` inside a `graph_task_group_begin`/`all_reduce` minimal repro,
`507903` at `AclmdlRICaptureEnd`, and the serving first request hanging at
`Replaying aclgraph` with `AIVector` 8–16 % / `AICore` 0 %. Two findings, two
owners: `507009` = `ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT` (`rt_error_codes.h:91`) in
torch_npu/CANN, while the serving deadlock is the MC2 × AIV family (#3881's
mechanism, `../hccl-and-hardware/aiv-mc2-env-ablation.md`) — the reporter's matrix
isolates an API pair the stack never builds, since all ten `graph_task_group_begin`
regions at that pin wrap one compute op. The body also blames
`patch_minimax_m2_config.py`'s AIV auto-enable, whose gate is `minimax_m2` +
`not enforce_eager` (`:110-127`), so it did not fire for GLM-4.6V. Page:
`../hccl-and-hardware/aiv-ric-task-group-capture.md` (new); family row
`../hccl-and-hardware/aiv-mc2-env-ablation.md`.

**#3128** vllm-ascend v0.10.2rc1 (`048bfd5`) + vllm v0.10.2rc1 (`e017120`), one node,
Qwen3-235B-A22B at `-dp 8 -tp 2 --max-num-seqs 8 --no-enable-expert-parallel
--max-model-len 66560 --compilation-config '{"cudagraph_capture_sizes": [8,1]}'` with
`HCCL_OP_EXPANSION_MODE=AIV`: the body is two lines of narrative (「单条推理请求在途重放时
卡住，之后报RPC超时；多并发请求可以正常处理」) plus the env block — no log, no per-rank
artefact. Track: RPC watchdog with a traffic-shape trigger. The recorded run loaded
the skill, then spent every one of its turns deriving a DP collective-shape mismatch
from source (`_sync_metadata_across_dp`, `execute_dummy_batch`, the MoE
`get_dp_group().all_gather`, `_get_fused_moe_state`), was cut at the 16 K output cap
and shipped **no answer**. That derivation was largely wasted: the pad target is
closed by construction (the idle ranks join the same all_reduce), and the two facts
that decide the next step — the idle ranks run `_dummy_run(1)` **eager** while the
busy rank replays the bucket `pad_for_cudagraph` picks, and `[8,1]` maps a 1-token step
to bucket 1 — are three file reads. Route now:
`../hccl-and-hardware/dp-aclgraph-single-request-hang.md` +
`scripts/dp-single-request-bucket.sh`; the mechanism remains unconfirmed.

**#4148** vllm-ascend v0.11.0rc1 (`c5fe179c`) + vllm 0.11.0 (`b8b302cd`), torch_npu 2.7.1,
`npu-smi 25.2.0`, 4×310P3 = **two** 300I Duo cards (`0000:03:00.0` → devices 0,1;
`0000:83:00.0` → devices 2,3), Qwen3-32B `--tensor-parallel-size 4 --enforce-eager
--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.7`, with a 537 KB
`error.log` attached. Engine start died in `determine_available_memory` → `profile_run`,
wrapped as `… current working operator name is HcclAllreduce` over a `rotary_embedding`
frame; the real record is `Transport_Init_Error(EI0009)` / `[Create][DestLink]Create Dest
error!` for `rank0→rank3` and `rank1→rank2` with `halShmemOpenHandle … drvRetCode=17` and
`rtsIpcMemImportByKey … 507899`. The recorded run got the layer-1 reading right
(cross-card transport, driver IPC import, the Python frame is an async sync point) and
correctly separated the second layer (the generic image vs `is_310p()`, which no guard
enforces at this pin), but it (a) read the paste's single `[NETWORK_ADDRESS_1]` placeholder
on both sides as a duplicated `device_ip` — history `#486` shows that field is printed from
the local rank on both sides, so it is not evidence — and (b) kept investigating past its
mid-session verdict until the 16 384-token cap, ending with `stop_reason:
output_truncated` and **no final answer message**; the reporter only replied "好的，了解了，幸苦",
so the fix is unconfirmed. Lessons now in
`../hccl-and-hardware/transport-destlink-ipc-import.md` and
`scripts/hccl-link-pair-audit.sh` (edge set, cross-card vs card-local, the redaction trap).

**#18315** vllm-ascend `v0.26.0rc1` (`f2f74a1`) pinned, title says v0.27.1RC1, env
block v0.26.0rc1; GLM-5.3-w8a8c8 on 4×A2 混部, `HCCL_OP_EXPANSION_MODE=AIV` +
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` → `aclnnMoeDistributeDispatchV4`
`107020` after an `event wait timeout … task_timeout`, starts when both are removed.
Real finding (built in reasoning, never delivered): the AIV × MC2 dispatch/combine
defect family #14928/#13688/#15195, plus the CaMem pool's
`expandable_segments:True` assert. **Shipped no answer**: 18 turns, 16 K output
tokens (~42 K characters of reasoning against 259 characters of message text), 17 bare
tool-call turns, no `findings.md`, cut inside a reasoning block whose last step was
rebuilding the paste into a scratch file to feed
`plog-timeline.sh` — a gate that needs a long-form timestamped record
(`[PID: n] <YYYY-MM-DD-HH:MM:SS>`) the paste did not have. The ranked answer
existed from about turn 8 (`../hccl-and-hardware/aiv-mc2-env-ablation.md`).

The next pass **did** deliver it, in 7 turns, and the route is the reusable part: the
per-token `rg -l` loop of `SKILL.md` step 2b left
`hccl-and-hardware/aiv-mc2-env-ablation.md` as the only page on all four token lists,
so the owning page was read on turn 3, and the answer cost one selector read
(`ascend_forward_context.py:322-340` at `f2f74a1`) plus the CaMem assert
(`device_allocator/camem.py:153`, message `:154`). Two things that pass added to the
page's ladder, which the earlier attempt never reached:

- **The reporter's own ablation is a two-variable change** (both env vars removed), so
  rung 1 is the single-variable pair, not a repeat of the matrix: AIV only (expected
  still to fail, same `DispatchV4` + `107020`) and `expandable_segments` only (expected
  to pass). That A-row/B-row split is what names the owner; "两个一起拿掉能拉起" decides
  nothing on its own.
- **The MC2 lane itself is conditional, and the answer has to say so**: the A2 selector
  returns ALLGATHER before the MC2 test when `num_experts > 512` (`:332-333`), and the
  expert count lives in the reporter's `config.json`, which the paste does not carry —
  so state it as the check that decides whether this lane applies instead of assuming
  4×A2 EP lands in it.

The stack-gap sentence to send is the op name: the paste's
`aclnnMoeDistributeDispatchV4` against the pin's
`torch_npu.npu_moe_distribute_dispatch_v2`
(`ops/fused_moe/token_dispatcher.py:120,241,343`) means their CANN/torch_npu is newer
than the pin, so the V4 kernel is not verifiable locally — that is a scope statement,
not a kernel to hunt.

**#5509** vllm-ascend v0.13.0rc1 (`1b5d5ab`) + vllm v0.13.0 (`72506c9`),
DeepSeek-V3.1-Terminus-w4a8, 2 nodes × (DP4, `--data-parallel-size-local 2`),
TP4, `--enable-expert-parallel`, `--async-scheduling`, MTP 3, FULL_DECODE_ONLY,
`VLLM_ASCEND_ENABLE_MLAPO=1`, `VLLM_ASCEND_BALANCE_SCHEDULING=1`,
`HCCL_INTRA_PCIE_ENABLE=1`/`HCCL_INTRA_ROCE_ENABLE=0`. Weights and
`torch.compile` succeed, then ~10 min of `No available shared memory broadcast
block found in 60 seconds` and `507014` aicore timeout at
`torch.npu.synchronize()` in `profile_run` → `determine_available_memory`, fault
kernel `MoeDistributeDispatchV2_…_high_performance_…`, title blaming
`aclnnInplaceCopy`. The session read the selector, corrected its own first-turn
blame of the `HCCL_INTRA_*` pair (the dispatcher comment and the tutorial both set
it), and named the real reason the kernel appears during profiling: `profile_run`
runs an extra `_dummy_run(mc2_tokens_capacity, with_prefill=True)` at
`model_runner_v1.py:2202-2208`. Answer shipped in Chinese with the DP4×TP4 = 16
boundary arithmetic, the tutorial diff, a refuted-rows list and a four-rung ladder.
Track: lane C, MC2 dispatch during profiling
(`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md`); the hang's own cause
stayed open (no `npu-smi`, no headless-node log in the paste).

**#8294** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be9`), 2×A2
(`2*8*910B3`), GLM-4.7-W8A8, `--data-parallel-size 2 --data-parallel-size-local 1
--tensor-parallel-size 8 --enable-expert-parallel`, DP=2 across two `--headless`/master
nodes, CANN 8.5.1, npu-smi 24.1.0.3. Title: 无法成功 torch.compile and initial
profiling/warmup. The paste's own traceback is
`worker.py:343 determine_available_memory` → `model_runner_v1.py:2544
_dummy_run(mc2_tokens_capacity, …)` → `:2492` → `:1818`, all matching the pin line for
line, so this is the same extra profiling dummy run as #5509 — with `507014` / `fftsplus
aicore timeout` after ~11 min of `shm_broadcast` lines. What the paste additionally
carried: the two nodes' **own** launch commands, and they disagree —
`cudagraph_capture_sizes [1,2,4,8,16,32,64]` on node 0 vs
`[1,2,4,8,16,32,64,128,256,512]` on node 1, i.e. capacity 64 vs 512 and `global_bs` 128
vs 1024 inside one `ep_world_size = 16` group. The recorded run found that in 11 calls
and also refuted the A2 no-ops (`VLLM_ASCEND_ENABLE_FUSED_MC2` is A3-only there) and
corrected the reporter's "EP needs FLASHCOMM1" (the real dependency is
`enable_shared_expert_dp` → `enable_sp`). Post-hoc, the reporter reported (2026-04-17)
that all combinations of three envs, including all unset, still fail at the same
phase — the env lane is closed and the symmetry/transport rows are what remain. The
second (follow-up) turn restated the whole ranking under a 8 192-token cap and was cut
in its closing section: `../evidence-workspace/delivery-discipline.md`.
Tracker: `../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` (Step 1b, and
`scripts/mc2-capacity-asymmetry.sh`).

**#9887** vllm-ascend v0.18.0 reported, pinned at v0.19.1rc1 (`da421afad7`), GLM-5.1-w8a8,
P2D2, decode nodes crash with `MoeDistributeDispatchV2` in the title, an EP group of
32 ranks (`rankSize:[32]`; TP4 is the value that reproduces the reporter's capacity
104, so DP8 is `32/4`), `HCCL_OP_EXPANSION_MODE="AIV"`: the only attached plog (device 5)
carries `fault kernel_name=aiv_all_gather_cn_int32_t`, `[TaskExecStage][Timeout][AIV]`,
`cmdType:[6] size:[1024] blockDim:[32] rankSize:[32] aivRdmaStep:[32] isOpbase:[1]`,
`opData … index[9], count[256], dataType[int32]`, `retCode=0x30`, one shared
`pc start` on 28 cores, and its `opData timeStamp` 12:41:08 against a fault print at
12:46:09 — HCCL's own timeout, not the trigger minute. The reporter's screenshot
claims the device-0 plog holds MTE DDR out-of-range + SMMU, and that plog was never
uploaded (416), so the owner was unverified at the time. Track: a **per-rank MoE comm
branch split at serving time**, not a kernel or transport defect — `count` 256 is
GLM-5.1's `num_experts` and `rankSize` the EP world, i.e. the All2AllV dispatcher's
preprocess histogram all-gather, which MC2 ranks never issue; some DP ranks crossed
`mc2_tokens_capacity` 104 (`ceil(102/4)*4` at TP4) because
`--max-num-batched-tokens` was 272 during MTP. Reporter-confirmed 2026-07-21 with the
workaround `max_num_batched_tokens = max_num_seqs*3`, a multiple of 32. What it cost:
the recorded run had the correct reading (the plog's own kernel is not the title's op;
`0x30` timeout vs `0x31` exception) but spent the whole 16 384-token budget in
thought and shipped **no answer** — `../evidence-workspace/delivery-discipline.md` G1
was owed at turn 3. Page: `../hccl-and-hardware/aiv-collective-kernel-fault.md`
§ "The other retCode"; arithmetic in `scripts/mc2-capacity-asymmetry.sh`.

**#6275** vllm-ascend v0.14.0rc1 (`52d4acf`) + vllm v0.14.1 (`d7de043`), 2×A2
910B2 64G, `DeepSeek-V3.2-Exp-w8a8`, DP2 × TP8 + EP + MTP `num_speculative_tokens:
2`: `fftsplus aivector … The DDR address of the MTE instruction is out of range`
at `_dummy_sampler_run` (`model_runner_v1.py:2158`, `profile_run`), followed by
the UCE-repair block (`507899` / `rtGetMemUceInfo` / `DSARandomNormal` /
`aclnnInplaceNormal`) and a `WorkerProc hit an exception`. Track: lane C device
fault at a profile-run sync point
(`../hccl-and-hardware/error-report-timestamps.md`). The launch command is
verbatim the tree's A2 tutorial block, so the config diff is exhausted; the paste
does carry two environments (`npu-smi 24.1.rc3` / openEuler 5.10 versus
`25.3.rc1` / Ubuntu 5.15). Delivery: **no message text at all** — 10 assistant
turns, ~48 K characters of reasoning, 2.7 K characters of status lines, cut with
an empty final message; much of it spent deciding whether `aclnnInplaceNormal` was
a random-number defect while the paste's own async caveat and the
`checkUceErrAndRepair` frame already said it was not. A second session on the same
paste had this file's own `#6275` entry in its turn-3 `references/` hit list and
read two route pages instead: 16 assistant turns, 20 tool calls, cut inside ranked
cause 1 (`SKILL.md` procedure step 2b).

**#16079** vllm-ascend v0.26.0rc1 (`f2f74a16`) + vllm v0.26.0 (`568afb3a`), two
nodes × (tp=8, dp=4) + EP, MTP/DFlash draft, 128 K prompts, `kimi-k2.7-Code`:
an evalscope run dies with `ConnectionRefused [Errno 111]` on port 8016 because
the API server died first; the worker-node log then shows `!!!!!!! Segfault
encountered !!!!!!!` whose top frames are `rtEventCreateExWithFlag` →
`NPUEvent::record` → `c10d_npu::ProcessGroupHCCL::_allgather_base`, then
`VllmWorker-6 died unexpectedly (exit code: None)`, and the peers' gloo
`all_reduce` in `_sync_metadata_across_dp` fails with `Connection closed by
peer`. Track: HCCL/device (signal death). **Both node logs are attached to the
case view** as `*.log` (`…-m0904-log.log` = master, `…-s0904-log.log` = slave;
the launch and load-driver files are `.txt`), so a `*.txt` glob hides them
(`case-layout.md` §*Attached log dumps*). From them: the dump sits immediately
after the slave's two `Replaying aclgraph` lines at 13:55:51 and is
**untimestamped**; the master's first fault (13:57:56, `DP1_TP6`) is a peer whose
own traceback reads `execute_dummy_batch` (`worker.py:1082`) → `_dummy_run`
(`model_runner_v1.py:3305`) → `_determine_batch_execution_and_padding` (`:2837`) →
`_sync_metadata_across_dp` (`:725`, `dist.all_reduce(…, group=get_dp_group()
.cpu_group)` — gloo, hence `pair.cc:547`); the master then only shuts down
(`SIGTERM` 13:58:07 → force kill 13:58:12 → `ApiServer_1 … died with exit code
None`) while the slave's `EngineCore_DP2` reports `VllmWorker-6` dead at 13:58:41.
So the *segfaulting* rank carries **no** `vllm_ascend/*.py` frame — only native
symbols and CPython's own entry — and its call site is the data gap; the peers'
call site is in the paste. Candidates are event/stream resources at collective
entry, the per-step tp-group `all_gather` in `rejection_sampler.greedy_sample`
(`enable_reduce_sample` defaults `False`, `ascend_config.py:346`, and the sampler
prints `Using fallback (non-reduce-sample) path`), or the image/CANN pairing
after an upgrade. Delivery: the run emitted three partial message blocks in
Chinese with the right first-half classification (the client error is secondary,
this is a worker segfault, not OOM) and was then cut **inside a source read** —
16 turns, ~43.5 K characters of reasoning against 3.3 K of message text, no
ranked ladder and no artefact request. The shape now lives in
`../hccl-and-hardware/worker-segfault-no-python-frames.md`.

**#3162** vllm-ascend `v0.10.2rc1` (`048bfd5`), image `v0.10.2rc1`, trl 0.19 + ms-swift 3.7.3
GRPO with the rollout server on `ASCEND_RT_VISIBLE_DEVICES=7` and seven training ranks on
0-6: the rollout's weight sync dies first (`RuntimeError: HCCL error: not found specific
resource` from `pyhccl.py:166` in `broadcast`, reached from
`swift/llm/infer/rollout.py … patched_update_named_param`), and the training ranks then paste
`HcclAllgather, error code is 6` / `EI0009 Transport init error … [Create][DestLink] …
dst_rank[0]`. One failure: the singleton (trainer rank 0) never reached `gather_object`, so the
other six could not build links to it. Source-verified at the pin: `PyHcclCommunicator`
(`pyhccl.py:32`) is not aliased over upstream's in this tree (`communicator.py:33` TODO), the
error text is CANN's own `HcclGetErrorString` (`pyhccl_wrapper.py:115-116`, `:203-209`), and
init already runs a warmup collective (`pyhccl.py:110-122`), so NOT_FOUND at the first real
broadcast is not a "peer never joined" story. Reporter-side precedents: #459 (same question at
0.7.1), #942 (a two-process `StatelessProcessGroup` + `PyHcclCommunicator` broadcast returned
all zeros). The reporter's later public comment is only a ping ("@wangxiyuan Hi, could you
please take a look at this?"), so no resolution is recorded. Route:
`../hccl-and-hardware/weight-sync-hccl-cascade.md`. Delivery: cut at the 16 K cap inside a
reasoning block with no answer text at all.

**#541** vllm-ascend v0.7.3rc1 (`f025df05`) + vllm v0.7.3 (`ed6e907`), torch-npu
2.5.1.dev20250308, CANN 8.0.0, 2×910B2 on one host: title "[Bug]: dp attention failed
when use hccl backend in dist.all_reduce", body = the offline
`examples/offline_inference/data_parallel.py` launcher (`VLLM_DP_*` env vars plus
`LLM(..., device="npu:N")` in one `multiprocessing.Process` per rank). Rank 0 prints
its two outputs at `2/2` and exits after `sleep(1)`; rank 1 dies at the first step in
`set_forward_context` → `dist.all_reduce(num_tokens_tensor,
group=get_dp_group().cpu_group)` with gloo `pair.cc:534 Connection closed by peer
[127.0.0.1]:21428`. Track: DP multiprocess launch / engine mode. The recorded run got
both load-bearing facts and delivered neither: pinned vllm v0.7.3 has no `get_dp_group`
and a 101-line `forward_context.py` (`:92` is inside the batchsize-stats block) and
vllm-ascend v0.7.3rc1 has no DP symbol at all, so the paste comes from a newer or
patched checkout; and the lockstep story is in its own evidence (`init engine … took
14.98` vs `17.27` seconds, `max_tokens=[16, 20][global_dp_rank % 2]`, `0/2` on the
surviving rank). Cut at the 16 K cap inside a reasoning block, no answer text.
Reporter-confirmed resolution (2025-04-18): "经过调试，demo可以运行，但
multiprocess_mode推理只在V1版本支持"; the follow-up log's `parallel_state.py` prints
`test current_platform.device_name` and `config.parallel_config.data_parallel_size is
2`, i.e. an instrumented tree, so the working V0 run does not make V0 supported. Route:
`../hccl-and-hardware/dp-group-peer-close.md`; the classification row now lets the
verdict land on the first turn.

**#2754** vllm-ascend v0.10.1rc1 (`7e16b4a7`) + vllm v0.10.1.1 (`1da94e67`), `deepseek-v3.1-w8a8`, two
Atlas 800 A2 (64G×8) nodes, launched per `docs/source/tutorials/multi_node.md` (DP4, DP-local 2, TP4,
`--enable-expert-parallel`, `--max-num-seqs 16`, `--max-model-len 32768`, `HCCL_BUFFSIZE=1024`,
`torchair_graph_config` + `ascend_scheduler_config` on). Graph capture dies on every rank at
`compile_or_warm_up_model` → `capture_model` → `_compile_torchair_graph`: `EI0007 … Failed to allocate
resource[qp] … [aicpu rank:7, … deviceLogicId:3, send_cq_depth:32768]. Reason: Memory resources are
exhausted`, then `[Create][DestLink]Create Dest error!` for **all 15** peers, `GetHcomHiddenInputs …
== ge::SUCCESS` failed, `Failed to init task index 207`, `Load graph failed, error code:1343225857`,
`ERR03005 GRAPH internal error`. The pasted frame line numbers match this pin exactly
(`torchair_model_runner.py:159,185,224`, `worker_v1.py:256`, `model_runner_v1.py:2583,2133`).
Maintainer record: "the root cause is OOM", plus the question whether the card is 64 GB
(docs:60 assumes 64G×8); the reporter answered 64×8, so capacity was not the mismatch and the budget
config was the remaining axis. The recorded run got the phase, the all-peer DestLink refutation and
the KV-before-capture ordering right, then spent its whole output budget on an unexported-source
mechanism hunt (`cann/hccl`, `cann/hcomm`, `cann/ge` are `unavailable` here) and was **cut inside the
first sentence of the answer** — the delivery failure, not the diagnosis, is the lesson. Route:
`../hccl-and-hardware/qp-resource-exhausted-at-graph-load.md`.

**#4178** vllm-ascend v0.11.0rc1 (`c5fe179`) + vllm v0.11.0 (`b8b302c`), 8×910B4,
`Qwen3-VL-8B-Instruct` TP8, `--max-num-seqs 16 --max-model-len 16384`, chunked prefill
+ prefix caching, `compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"}`, one
image prompt; title extends the hang to Qwen2.5-VL-72B / Qwen2.5-72B. The paste is
the full 290-line log: mode banner (`platform.py:220`), all eight ranks
`Graph capturing finished in 6 secs`, then the first request never returns — four
`shm_broadcast` 60 s lines, the `dump_input.py` dump, `TimeoutError: RPC call to
execute_model timed out.`. Track: graph mode engaged + capture complete + request-path
hang, i.e. the mode's own path (the general watchdog classification is
`../hccl-and-hardware/rpc-watchdog-timeouts.md`). The recorded run never shipped an
answer: 22 assistant turns, ~48 K characters of reasoning, one status line of message
text, cut at the cap while still opening new code areas (`acl_graph.py` update
stream, `platform.py` splitting ops, upstream dispatcher) — the mechanism it was
hunting is not in the rc1 tree at all, because the reporter later reported the bug
fixed on **0.11.0rc2** (later public observation on the same issue, not a reply to
these investigators). What it did establish and should have delivered: `tests/e2e`
validates `FULL_DECODE_ONLY` only in `multicard/test_full_graph_mode.py` (Qwen3-30B-A3B
TP2, `max_model_len=1024`, `max_tokens=32`), no workflow invokes that file, the wired
ACL-graph runs are PIECEWISE at 1024 context on Qwen3-0.6B / DeepSeek-V2-Lite, and the
runner's only mode-specific provisioning at rc1 is the MLA `cos`/`sin` buffer
(`model_runner_v1.py:374-377`). Route now:
`../hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`, SKILL.md rule 16.

**#6018** vllm-ascend `0.12.0rc1` in the title / `0.12.0rc2` in the environment
block (`SOURCES.json`: `reported_ref_status: conflicting_versions`; snapshot pins
v0.13.0rc1 `1b5d5ab`), 910B4 ×8 single node, "Qwen2.5 30B" (dense 32B vs MoE
Qwen3-30B-A3B — never resolved), CANN 8.3.RC2 + torch_npu 2.8.0 (the documented
pair for the 0.13 line, `docs/source/installation.md:15`): `FULL_DECODE_ONLY`
serve crashes under concurrency with `507011` and plog `fftsplus aivector error`
core 8 error code 0x10, `errorStr: Illegal instruction, which is usually caused by
unaligned UUB addresses`, `mte error devId=5 streamId=1243`, plus
`sqe_type=7(notify wait) … sq sw status error` on stream 1206. The reporter's
later comment supplies the config: TP8 / DP1 / ray, `max_model_len 32768`,
`async_scheduling false`, `cudagraph_capture_sizes [1, 8, 16]`, `max_num_seqs 16`,
`max_num_batched_tokens 4096`, crash at concurrency 8. Track: FULL-mode replay
argument patch (`acl_graph.py:189-196` sync-before-replay, `:198` paged-attention
path, `:256` FIA path, `:299-304` dispatch); `pa_shape_list` defaults to `[]`
(`ascend_config.py:119`) so the legacy paged-attention workspace note is *not*
this run's path, and `507011` is `ACL_ERROR_RT_MODEL_EXECUTE`
(`rt_error_codes.h:93`) naming no op — the ask is `origin_op_name` /
`fault kernel_name`. The recorded run reached this family and the coverage verdict
but emitted **zero message text on its first 13 turns**, writing the whole ranked
answer on the 14th and last. Page:
`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`.

**#4326** vllm-ascend v0.11.0rc1 (`c5fe179`) + vllm v0.11.0 (`b8b302c`),
Qwen3-30B-A3B-W8A8 (TP1×DP8, `--enable-expert-parallel`, 32768 ctx,
`--async-scheduling`, `VLLM_ASCEND_ENABLE_FLASHCOMM`): "[Bug]: Qwen3-moe
FULL_DECODE_ONLY inference crash". Body = the OpenAI frontend re-raising
`EngineDeadError … See stack trace (above)` ~15 times (once per queued streaming
output, i.e. death at **concurrency**, not on the first request) plus a plog from
44 min earlier at interpreter teardown (`cann_kb_finalize`, `ConnectionRefused`) —
no root-cause stack, no in-window plog. Track: FULL-mode replay (ranked:
async-scheduling × per-replay argument patch, the GQA `_npu_paged_attention`
workspace note, EP/DP collectives inside the graph, flashcomm/SP). Two version
facts worth the two greps: at this pin `acl_graph.py` records update events
(`:249`, `:307`) with **no** `torch.npu.synchronize()` before `replay()` (`:190`)
and no `event.wait` consumer, so the ordering hazard is live on the 0.11 line and
`--async-scheduling False` is a real lever; and neither `using_paged_attention` nor
`pa_shape_list` exists there, so the paged-attention path is default, not opt-in.
Coverage: `test_full_graph_mode.py` (TP2, `max_model_len=1024`, bf16) is referenced
by no `.github/workflows` file; the wired graph runs are `singlecard/test_aclgraph.py`
and `test_aclgraph_mem.py` (`_e2e_test.yaml:77,91-92`). The run delivered the ranked
answer plus the per-replay mechanism on its 11th and last turn, with seven of its
eleven turns carrying no message text. Page:
`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`.

**#10370** vllm-ascend v0.20.2rc1 (`367b8e62`, the snapshot's own pin), two nodes ×
910B, GLM-5.1-w4a8, 两个局点 both reproducing: title only — 图模式算子通信问题，只有开启
单算子才能规避 — with the log attachment failing to upload and the single pasted line
`saved AOT compiled function to …/torch_compile_cache/…` from
`[decorators.py:668]`. Track: **execution-mode split**, answerable as an ablation ladder.
The recorded run reasoned correctly that single-op working on the same cluster exonerates
ranktable/HCCL/NICs, then spent the budget reading the MC2 selector and `npugraph_ex`
gates and was cut at the cap (14 043 output tokens) **inside its own answer** — it had
reached the verdict and the plog ask. Afterwards a maintainer proposed
`VLLM_DISABLE_COMPILE_CACHE=1` and the reporter answered **没有效果**, i.e. the cache lane
is refuted and the mode lane stands. Page:
`../hccl-and-hardware/graph-vs-single-op-comm-fault.md`.

**#11542** vllm-ascend `v0.22.1rc1` image, 2× 910B3 (A2), a fork branch
  (`qwertyjack/deepseek-v4-dspark-main`), `DeepSeek-V4-Flash-DSpark-bf16`, launching from the
  **DeepSeek-V4-Pro** tutorial's multi-node script: title `create_config /torch_npu/csrc/distributed/
  HCCLUtils.cpp:140 HCCL function error: hcclCommInitRootInfoConfig(…), error code is 1`, with the two
  screenshots OCR'd into `attachments/` showing the same `RuntimeError` at `deepseek_v4.py` →
  `FusedMoE` → `token_dispatcher.py` `get_hccl_comm_name(local_rank)` plus a peer's TCPStore
  `connection was closed` and `broadcastMasterID` frames. The recorded run localized the call site,
  the enum (`1` = `HCCL_E_PARA`) and the decisive mechanism — `HcclCommInitRootInfoConfigInner`
  (`cann/hcomm … op_base.cc:1378-1411`) folds `InitExternalInput()`, `InitEnvConfig()` and
  `commConfig.Load()` into `HCCL_E_PARA`, so the naming text is an `HCCL_ERROR` line, not the
  traceback — then spent 20 assistant turns and ~48 K characters of reasoning, emitted **30
  characters of message text** and was cut at the output cap: the reporter never got the ranked
  answer, and their later comments are only "here are my node0/node1 scripts" and "with dp=tp=4 it
  hangs instead". Page: `../hccl-and-hardware/comm-create-errors.md`; the recipe gap it never said
  — V4-Flash at that pin has a single-node (§5.1) and a PD-only multi-node recipe, while the Pro A2
  multi-node script needs 4 nodes at DP4 × TP8 with a unique `--data-parallel-start-rank` per node.

**#5947** vllm-ascend `v0.13.0rc1` (`1b5d5ab`), DeepSeek-R1-0528-W8A8 on 2×A2 64G, engine
log says TP4 / DP4 / DP-local 2 / EP / MTP 3 / `FULL_DECODE_ONLY` / `HCCL_BUFFSIZE=200`
(the body claimed TP8/DP8, which cannot fit the 16 dies): weights load at 47.2958 GB per
card, then every rank dies in the same 60 ms with
`call aclnnMoeDistributeDispatchV3 failed, error code is 561000` →
`Allocation_Failure(EI0007): Failed to allocate resource[qp] … send_cq_depth:32768 … Memory
resources are exhausted` + `HcclAllocComResourceByTiling … ret = 4`, `DestLink` only to the
8 remote ranks (intra-node is PCIe under `HCCL_INTRA_PCIE_ENABLE=1`). No Python frame and
no `LoadGraph failed` block, so the phase (MC2 `profile_run` dummy run vs torchair capture)
is inferred, not pasted. The run routed to `qp-resource-exhausted-at-graph-load.md` — whose
ordering paragraph is the v0.10.1.1 torchair-capture case — re-derived the capacity
arithmetic over five calls and shipped no answer (16 384 output tokens, 1.05 M cumulative
prompt tokens, 29 KB spent on a batched `ls`/`cat SOURCES.json`/`ls history` beside
`skill_view` that the skill's first rule forbids). Pages now carry the phase split:
`../hccl-and-hardware/qp-resource-exhausted-at-graph-load.md` "the phase depends on the ref"
and `../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` §Step 3b.

**#13527** vllm-ascend v0.18.0 (`e18643f8a4d5`), DeepSeek-V4-flash EP, 8×A2 P/D,
  `HCCL_OP_EXPANSION_MODE="AIV"` on both sides: after serving for a while six ranks'
  plogs carry `fault kernel_name=aiv_reduce_scatter_bfloat16_t` with MTE
  `DDR address … out of range` / `isSmmuFault is 1` / `retCode=0x31` and one
  `tag[ReduceScatter_group_name_3]`. The run found the fault kernel and refuted a
  per-rank count mismatch (six identical `opData` lines) by its 10th turn and then
  spent six more on the DSV4 registry absence and the hcomm AIV templates: 43.8 K
  characters of reasoning vs 2.2 K of text, no ranked answer at the cap, and a close
  call on ranking "partial participation" off the missing device 0/6 plogs. Page,
  with the name→owner rule, the `opData` comparison and the AIV-not-a-misconfig
  split: `../hccl-and-hardware/aiv-collective-kernel-fault.md`.

**#9872** vllm-ascend `v0.19.1rc1` (`da421af`), MiniMax-M2.5 on 2×910B2 (16 dies),
`--data-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel`, title
`aclnnMoeDistributeDispatchV4 failed, error code is 561000`. The DP2×TP8 = 16 arithmetic
does select MC2 (A2: `num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens
<= mc2_tokens_capacity`, `vllm_ascend/ascend_forward_context.py:272`), and the run spent
12 turns proving that plus the pinned A2 dp2tp8 recipe
(`docs/source/tutorials/models/MiniMax-M2.5.md:194-250`) — but the question it never asked
was whether the comm existed at all: the reporter's later full-card plogs open with
`hccl_socket_manager.cc:860 [Wait][LinksEstablishCompleted] is failed. ret[9]` and a
`LINK_ERROR_INFO` table (device[7], port 16666, one `time out` peer and seven
`connecting`), i.e. `ret = 4` from `HcclAllocComResourceByTiling` wraps a link timeout and
no budget lever applies. The maintainer's own reply says the aclnn error
"不一定是根因". Delivery: 12 assistant turns, 39 K characters of reasoning, **one**
partial message (2.9 K) that was cut mid-answer, after it had verified everything except
this. Pages: `../hccl-and-hardware/link-establish-timeout.md` (new),
`../startup-failures/environment-correspondence.md` (the `collect_env` ENV filter that
makes the missing `HCCL_*` block meaningless).

**#9447** vllm-ascend `v0.18.0rc1` (`99e1ea0`), GLM-5.1 on 2×Atlas 800I A3 (16 dies each)
behind a 400G switch: "双机灵衢直连环境下可以正常运行，但是在双机 roce 组网下服务启动
profile 阶段报错 `aclnnMoeDistributeDispatchV4` 算子通信失败 hccl_ret 9; 测了双机之间的
hccl_test，通信正常". `9` is `HCCL_E_TIMEOUT` (`cann/hcomm/include/hccl/hccl_types.h:33`), so
the op was launched and never completed — the family of
`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md`, with an HCCL code instead of a
device timeout. The run verified the pin's selector path and found the one in-tree RoCE
knob (`enable_mc2_hierarchy_comm`, "Enable dispatch/combine op inter-node communication by
ROCE", default `False`, `ascend_config.py:169` / `additional_config.md:48` /
`token_dispatcher.py:120-121`) and was about to answer with it — then spent the rest of the
budget re-reading the A3 two-node tutorial tabs: 16 turns, 24 tool calls, 46 K characters
of reasoning, **39** characters of message text (the opening status line) and the
16 384-token cap. Corrective: the reporter's
"works on the other fabric" premise is the thing to re-confirm, not the thing to answer
from — the public follow-up (2026-06-22) says 没有解决 with the direct-connect setup failing
too, so no RoCE-only lever is confirmed, and the *passing* `hccl_test` never covered the
dispatch's own `MC2MultiServer` comm plane. Pages:
`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` §Step 3c (new),
`../hccl-and-hardware/aiv-mc2-env-ablation.md` (row corrected: it is the `hccl_ret`-9
sibling, not AIV-family support).

**#7239** image `quay.io/ascend/vllm-ascend:v0.16.0rc1`, pinned at vllm-ascend
v0.16.0rc1 (`a78a00e`) ↔ vllm v0.16.0 (`89a77b1`), 910B4 (32 GB dies), TP4,
`--max-model-len 512 --max-num-seqs 10 --async-scheduling`, an explicit
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":
[1,2,4,6,8,10]}'`, and **no** `--enable-expert-parallel`; Qwen3-Omni-30B-A3B-Instruct.
Every rank loads weights (14.92 GB), then one rank reports inside
`worker.py:337 determine_available_memory` → `model_runner_v1.profile_run:2475` →
upstream `gpu_model_runner.profile_run:5156 embed_multimodal` →
`qwen3_omni_moe_thinker.py:1789 _process_video_input` (defined in
`qwen2_5_omni_thinker.py:942`) → `qwen2_5_vl.py:395 self.attn` →
`vllm_ascend/ops/mm_encoder_attention.py:108 seq_lens_cpu =
torch.diff(cu_seqlens).to("cpu")`, with `RuntimeError: The Inner error is reported as
above … the current working operator name is HcclAllreduce` and `ERR00100 PTA call acl
api failed`. Track: lane C, the **first** collective of the profile run.

What the recorded run got right, cheaply: the sync-point rule in one read (it found
the page's #7477 instance — same `:108` frame, blocking mode landing in
`vocab_parallel_embedding.py:204 maybe_pad_and_reduce`), the fact that the pinned tree
carries that chain (`vocab_parallel_embedding.py:185,204`), the blocking-mode guard at
this ref (`platform.py:332-343`, so `--enforce-eager` must accompany
`ASCEND_LAUNCH_BLOCKING=1` because the reporter set FULL_DECODE_ONLY), and the config
contrast with the pin's own `tests/e2e/models/configs/Qwen3-Omni-30B-A3B-Instruct.yaml`
(TP4 **with** `enable_expert_parallel: True`, `max_model_len 8192`, Atlas A2).

What it cost: the TP3 `image` vs TP0-2 `video` line in the paste became a
rank-divergent-dummy-profile theory that took several turns — including one call on the
invented path `vllm/multimodal/budget.py` — to refute with
`vllm/multimodal/encoder_budget.py:181-185` (`max()` over an insertion-ordered dict, so
identical on every rank) and a `PYTHONHASHSEED` tie-break hypothesis it had to withdraw.
It also never asked for the inner-error block that the paste itself points above
(“The Inner error is reported as above”), and its last message was cut mid-ladder by the
output cap. Page, with the four-line rank-parity ask and both artefacts:
`../hccl-and-hardware/collective-error-chain.md` “First collective inside
`profile_run`'s multimodal encoder”.

**#488** vllm-ascend `v0.7.1rc1` (`f17417f2d5b5`) + a self-built `vllm 0.7.2.dev`,
torch-npu `2.5.1.dev20250218`, transformers 4.49.0, CANN 8.0.0, 910B cards 4+5,
DeepSeek-R1-Distill-Qwen-32B TP2 `--gpu-memory-utilization 0.8 --max-model-len 20480`:
~100 short chat requests, and around request 40-50 the V0 background loop dies with
`RuntimeError: ACL stream synchronize failed, error code:507035` at
`torch_npu … _gather` → `broadcast_object_list` → `_object_to_tensor`, followed by
`AsyncEngineDeadError` on every later request. Track: the TP gather emulation's H2D
copy is the **sync point**; `507035` is `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`
(`rt_error_codes.h:94`) and the owner is whatever the decode step enqueued before
`compute_logits`, named only by the plog's `fault kernel_name=`/`chipId`. The
recorded run got that mechanism right and delivered it (29 turns, 4.9 K-character
answer), but excluded the version lane — "版本不匹配：排除" from the matching
`torch_npu` wheel — and ranked a device fault alone; the issue closed on the
maintainer's "you are using a old version … install 0.7.3rc2" and the reporter's
"i tried latest version (vLLM Ascend 0.7.3rc2) and the problem solved." The missing
rung was the install-and-re-run, one command, ahead of the plog ask. Pages:
`../hccl-and-hardware/collective-error-chain.md` §Step 1a,
`../startup-failures/environment-correspondence.md` §*A self-built vllm beside an rc
vllm-ascend*.

**#4725** — vllm-ascend v0.11.0rc3 (`b6d63bbd`) + vllm v0.11.0, A3 16 dies, Qwen3-235B-A22B
w8a8, P and D started on that **one** machine by two invocations of
`examples/external_online_dp/launch_online_dp.py` (DP2×TP4 / DP8×TP1, different
`--vllm-start-port`): the second instance dies at worker init with
`Bind_Failed(EJ0003)` + `hcclCommInitRootInfoConfig … error code is 7` out of
`w8a8_dynamic.py:137`, plog `[HcclSocket][Listen] … specific port[16666] fail` per rank.
The recorded run opened `control-plane-port-binding.md` first and spent its **entire**
16 384-token budget inside HCCL port arithmetic (`HETEROG_CCL_PORT`,
`SetRanksPort`, `GetLocalNicPort`, `GetHostPort`), never reached the launcher, and was
cut with no final message; the reporter's later comment says the
`ASCEND_DEFAULT_DEVICE_PORT` line "has nothing to do with" the case and that offsetting
the decode instance's cards (0-7 → 8-15) fixed it. What would have redirected it: the
report's own `npu-smi` table already showed two ~59 GB workers per chip on NPU 0-3 with
NPU 4-7 empty, and one grep of `visiable_devices` in the launcher shows the card list
starts at 0 in every invocation. Page:
`../pd-disaggregation/same-host-pd-device-partition.md` (new), with the family
separation in `../hccl-and-hardware/comm-create-errors.md` §code 7.

**#8975** — vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be96`), CANN 8.5.1,
8×910B / 16 dies with both engines' `VLLMWorker_DP` ranks resident, a P/D deployment:
偶发性卡死, the API server still listening, `curl` returning nothing, `AICore 0 %` on every
chip. The body was a Feishu link plus a QR image — no log at all. The recorded run ranked
the **host-side** DP barrier `dist.all_reduce(…, group=get_dp_group().cpu_group)`
(`worker/model_runner_v1.py:512-552`) as cause #1 on the strength of "AICore 0 ⇒ no kernel
ran ⇒ the host is stuck"; the follow-up plog showed a **device-side** MC2 AlltoAll deadlock:
`group_name_233` with 32 ranks over 2 servers, `newtag[…mc2RunAlltoAllDirectFullmesh_device]`,
both decode nodes frozen on operator 2706 at `03:12:39`, `bufferSize[2097152000]` (= 2000 MiB)
although the reporter believed `HCCL_BUFFSIZE` was 2560, and the prefill node printing
`status[STUCK]` while its own operator count still climbed 110822 → 4191386. Track: silent
multi-node collective deadlock. What would have redirected it: one plog read
(`scripts/hccl-stuck-timeline.sh`) instead of a utilisation inference — 0 % is also the state
of a rank waiting inside a collective. Delivery: the first answer shipped 3.4 K characters
with a real ladder (a `py-spy` dump, both sides' `/metrics`, the mooncake
`Force freed request: <id>` line, and the correct re-scoping that `scheduler_reserve_full_isl`
does not exist at v0.18.0); after the follow-up it wrote the reversal in one sentence and then
spent the whole 8 192-token second segment on the MC2 source walk, so its last turn carried no
message and the lane it had already named never got its ladder. Page:
`../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md` (new).

**#3172** vllm-ascend v0.9.1 (`0740d10`) + vllm v0.9.1 (`b6553be`), 8×910B3, torch_npu
2.5.1.post1, CANN 8.2.RC1, `VLLM_USE_V1=1`: "multi-card, sleep model A, load a new
instance running model B → HCCL resources occupied, B fails to start". The only evidence
was a screenshot (`attachments/screenshot-*.ocr.txt`): `hcclCommInitRootInfoConfig(...)
error code is 7` + `ERR02200 DIST call hccl api failed` + `EJ0001 ... Failed to
initialize the HCCP process. Reason: Maybe the last training process is running.`, in a
`VllmWorker` on the **new** instance's first `all_reduce(embedding)` of the TP group.
Track: communicator creation, not OOM, not the fabric. The run classified and ranked
correctly (sleep returns only the pluggable allocator's blocks; comms registered for
release at process exit; `7 = HCCL_E_UNAVAIL` from `third_party/hccl/.../hccl_types.h:38`)
but spent ~5 of its 10 tool rounds inside `ProcessGroupHCCL`/`HCCLUtils` C++ that bought
nothing, and was cut at the 16 384-token cap inside its evidence table — the ranked
causes, ladder and scope never shipped. What it never tried, and what decides the case in
one call: the CANN error table, which names the code `HCCP_Process_Initialization_Failure`
and gives the vendor's own fix, *"Wait for 10s after killing the last training process and
try again."* Page: `../runtime-memory/sleep-mode-multi-instance.md` (new),
`../hccl-and-hardware/comm-create-errors.md` §code 7.

**#7015** image `quay.io/ascend/vllm-ascend:v0.14.0rc1` (pin v0.14.0rc1 `52d4acf` ↔
vllm v0.14.1 `d7de043`), 2×A2 (8 dies each), DeepSeek-V3.2-W8A8, DP2 × TP8 + EP16,
`deepseek_mtp` n=2, `FULL_DECODE_ONLY` with `cudagraph_capture_sizes [3..48]`,
`HCCL_OP_EXPANSION_MODE=AIV`: "probability TimeoutError" whose whole paste is the
EngineCore's bare `TimeoutError` from `shm_broadcast.acquire_read` → `EngineDeadError`.
Track: the RPC watchdog reporting a worker that never returned, phase = the **first
prefill step** (`step_counter=0`, `num_computed_tokens=0`, 37 scheduled tokens,
`scheduled_spec_decode_tokens={}`). Two reusable derivations the run got right: the
watchdog deadline is `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 300 s, and the A2+EP branch
sends that 37-token step over cross-node MC2 because capacity = `max_cudagraph_capture_size`
= 48 (the reporter's own list) while `world_size_across_dp/pp = DP2 × TP8 = 16` sits
exactly on the `>= 16` gate — and it corrected its own DP-asymmetry hypothesis by reading
`_sync_metadata_across_dp` (pad to the DP max). The coverage fact it shipped: the pin's
only dual-node DSv3.2-W8A8 nightly is **A3** (`npn 16`); the A2 recipe does not exist,
and the A3 command differs from the reporter's in `--additional-config
'{"layer_sharding": ["q_b_proj","o_proj"]}'`, `VLLM_ASCEND_ENABLE_FLASHCOMM1`,
`HCCL_BUFFSIZE`, `--no-enable-prefix-caching` and `--max-num-seqs`. Pages:
`../hccl-and-hardware/rpc-watchdog-timeouts.md`,
`../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md`,
`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md`.

**#16023** vllm-ascend v0.26.0rc1 (`f2f74a16`) + vllm v0.26.0 (`568afb3a`), A3,
Kimi-K3 + DSpark drafter (7 spec tokens), TP16 × DP4, `max_seq_len=133120`,
chunked prefill + prefix caching, `enforce_eager=True` / `cudagraph_mode: NONE`,
1P1D: short curl requests are served, a ~100 k-token prompt kills the prefill
node. The paste is the EngineCore's `Dumping input data / scheduler output /
scheduler stats` block (`num_scheduled_tokens=8064`, `step_counter=0`) followed by
`EngineCore encountered a fatal error` and a pure-Python traceback ending
`patch_shm_broadcast.py:70 raise RuntimeError("cancelled")` under
`core.py:693 future.result()` → `multiproc_executor.py:390 mq.dequeue`. Track:
a **worker process death**, not a queue bug — `monitor_workers` saw it die,
logged `died unexpectedly (exit code: N)` and shut the response queue down, and a
worker *Python* exception cannot produce this text (it is caught and returned as a
`FAILURE` response). The recorded run derived exactly that, source-verified, and
then kept reading the queue and the tutorial — it emitted **no message text**, so
the reporter got nothing. The reporter's own follow-up (same issue, later) closes
it as an environment answer the ranking never reached: 排查后 "定位到底层算子兼容性
适配，CANN 与 HDK 存在冲突，环境的 firmware 与 driver 不匹配。升级到 26.1.1 后解决"
— one `npu-smi info` + `version.cfg` ask and an upgrade, no patch. What the run
should have sent on its first turn: the `cancelled` attribution, the
`died unexpectedly` grep with its exit-code decode, the artefact list, and the
driver/firmware × CANN lane first. Page:
`../hccl-and-hardware/cancelled-executor-teardown.md`.

**#397** vllm-ascend v0.7.3rc1 (`f025df05`) + vllm v0.7.3 (`ed6e907`), torch-npu
`2.5.1.dev20250308`, CANN 8.0.0 (`V100R001C20SPC001B251`), 2×910B2, open-r1 GRPO on the
TRL-npu fork: title `RuntimeError: create:…/torch_npu.dir/compiler_depend.ts:89 HCCL
function error: HcclCommInitRootInfo(numRanks, &rootInfo, rank, &(comm->hcclComm_)), error
code is 2` + `ERR02200 DIST call hccl api failed`, `Device:0, RankID:0`; the frames are
`trl … prepare_deepspeed` → `deepspeed/runtime/engine.py _broadcast_model` →
`dist.broadcast(…group=self.seq_data_parallel_group)`, so the group is the trainer's and no
`vllm_ascend/` frame appears anywhere. The recorded run got the enum (`2` = `HCCL_E_PTR`),
the `.hpp`-not-`.cpp` location of `HCCLComm::create` and the world-size split
(#116/#109 ran the same recipe at 7 processes and passed this broadcast) — then spent the
rest of 41 messages on candidate mechanisms (one-rank-comm theory, port/HCCP reuse,
`maybe_chunk_residual`, upstream `qwen3_5` files absent from the pin) and hit the 16 384
output cap with **no message**. The reporter's later comment is the launch line that
settles `numRanks`: `accelerate launch --num_processes=1 … src/open_r1/grpo.py` with
`ASCEND_VISIBLE_DEVICES=1,2` and the `HCCL_*`/`TASK_QUEUE_ENABLE`/`COMBINED_ENABLE`/
`CPU_AFFINITY_CONF`/`NPU_ASD_ENABLE=0`/`ACLNN_CACHE_LIMIT` list — world size 1. Ask for it
on turn one. Page: `../hccl-and-hardware/comm-create-errors.md` §*The no-config `create`
sibling*; route row in `../first-checks/signature-routes/device-ops-and-collectives.md`.

**#5684** image `quay.io/ascend/vllm-ascend:v0.13.0rc1` (pin v0.13.0rc1 `1b5d5ab` ↔
vllm v0.13.0 `72506c9`), one Atlas 800 A2 server — 16 `910B2C` dies at `65536 MB`
across 8 Bus-Ids, `npu-smi` 24.1.0, CANN 8.3.RC2 — DeepSeek-V3.2-W8A8
`--quantization ascend --tensor-parallel-size 16 --enable-expert-parallel
--async-scheduling --max-model-len 16384 --max-num-batched-tokens 4096
--gpu-memory-utilization 0.8`, `cudagraph_capture_sizes [4,16,32,48,64]` +
`FULL_DECODE_ONLY`, under the `DeepSeek-V3.1.md` **single-node (A3)** env block
(`HCCL_OP_EXPANSION_MODE=AIV`, `VLLM_ASCEND_ENABLE_MLAPO=1`,
`VLLM_ASCEND_BALANCE_SCHEDULING=1`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`),
started from a container that mounts `davinci0…15`. Every rank dies inside
`determine_available_memory` → `profile_run`'s MC2 dummy run: `call
aclnnMoeDistributeDispatchV3 failed, error code is 561000`,
`Nnopbase fails to invoke the HcclAllocComResourceByTiling … ret = 4`. Read the
box right: the tutorial's w8a8 row is *1 A3 (64G × 16)* **or** *2 A2 (64G × 8)*, and
the only single-node recipe at that pin is A3 `dp4tp4`, so a TP16 single-A2-box
launch is the A3 recipe with the DP factor folded into TP. Delivery: two full
answers, the second correcting the first's A2 recipe shape (that row is
`--data-parallel-size 4 --data-parallel-size-local 2 --tensor-parallel-size 4` across
two nodes, not DP2×TP8) and hardening the verdict to the topology invariant, which the
later public evidence leaves standing rather than refutes. The reporter's own follow-up
thread says the advice
they had collected was "cann8.2, hdk 24.1, vllm-ascend v0.13.0rc1 can be deployed"
(#5468) and asks how to check/downgrade CANN — their dump was already on the pin's
documented CANN (`installation.md:15` `== 8.3.RC2`, `Dockerfile:18`
`FROM cann:8.3.rc2`) — and then they re-ran on an 8.2-based `v0.11.0rc0` image with
pip `vllm==0.13.0` + `vllm-ascend==0.13.0rc1` on top: same phase, sibling code
(`aclnnMoeDistributeDispatchV2 … 107000`, `Parameter verification failed` / `EL0003`,
`ret = 2`, `Failed to allocate resource[DeviceMemory] with info
[size:140737488355328]` = 2^47 B). The launch line was the only constant between the
two attempts, so the topology/AIV invariants stay ranked first; the retry is **not** a
clean CANN-8.2 control (its pip layer skewed a 0.11.0rc0 `torch_npu` under
vllm-ascend 0.13.0rc1) and #5684 has no public resolution. Pages:
`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` §Step 3b2 (CANN-pairing
negative + the `107000`/`ret = 2`/128 TiB sibling), and
`../hccl-and-hardware/index.md` (one-box branch paragraph).
