# Device and communication cases

A device dump naming an operator, HCCL/transport link faults, RPC watchdog and
profile-run sync points, indexer-scatter corruption, segmentation deaths, and
mixed decode+prefill batch faults. Routes: `../custom-ops-runtime-faults/index.md`,
`../hccl-and-hardware/index.md`, `../dsa-sparse-attention/index.md`,
`../mixed-batch-splits/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

**#10522** vllm-ascend v0.20.2rc1 (`367b8e62`) + vllm 0.20.2 (`bc150f5`), GLM-5.1,
PD with two A3 nodes as the P instance at `PP4 TP8 EP32`, CANN 9.0.0, 16 dies `OK`:
the bad-case harness floods over-length prompts, the API server rejects a
202753-token request (`VLLMValidationError … maximum context length is 202752`), then
a request carrying `presence_penalty=0.5 frequency_penalty=0.5 repetition_penalty=1.1
max_tokens=1 min_tokens=1` kills the EngineCore — `failed (engine dead)`, teardown,
`!!!!!!! Segfault encountered !!!!!!!` with `llm::LlmDatadistTimer::ProcessTimerInContext()`
frames — and the plog names `fault kernel_name=token_bin_counts_and_mask_kernel_0`
with `aivec error`, `The DDR address of the MTE instruction is out of range`,
`CheckSmmuFault:Get isSmmuFault is 1` and `507035`. Track: a device fault inside a
**downstream Triton-Ascend** kernel, not a `csrc/` or CANN op. The recorded run found
the owning file at its 12th tool result (`vllm_ascend/ops/triton/bincount.py:32`,
reached only under non-default penalties) and then went upstream — reading
`prompt_token_ids`/`output_token_ids` construction and deriving the
`[AIC_INFO] args(0 to 9)` byte layout — for the rest of a 38-message, 16 384-token
session whose three message texts total 230 characters: **no answer shipped**. The
reporter closed the loop themselves the same day: "已验证该commit可修复"
(`f95a31ec266552a73cd8443810884085ae2e3880`, i.e. after the pin), so the mechanism
was never obtainable from this tree. What the run should have sent on that turn:
the ownership verdict, "the chain only runs because that request sets penalties —
retry with them at defaults", the teardown decoy retired in one line, and the ask.
Page: `../custom-ops-runtime-faults/triton-jit-kernel-device-fault.md`.

**#18345** vllm-ascend v0.23.0rc1 (`f4a08bddd0cc`) + vllm 0.23.0, triton-ascend
v3.2.1 (`2badfc89e70a`), 8×910B3, CANN 9.1.0, torch 2.10.0 / torch_npu 2.10.0.post2:
"`extract_slice` of triton-ascend gives wrong data when the start address is not
32-byte aligned", bf16 only, with the upstream PR referenced in the body. Track:
**ownership of a third-party extension op**, not a vllm-ascend defect. The recorded
run pinned the wheel (`requirements.txt:20`), mapped the extension op to an MLIR
`tensor::ExtractSliceOp` and its Ascend rewrites, contrasted the doc's "bf16
supported / 无特殊限制" with a UT that slices only 1-D fp32 at 2048-byte-aligned
starts, and answered the in-tree exposure from the 42 `extract_slice` call sites
(offsets `0` / `HALF_ROPE_DIM` / `COL_BLOCK_SIZE`, aligned at rope_dim 64/128). It
shipped the full answer with the mechanism correctly left open and the
print-path-versus-data-path check first. Page:
`../image-dependency-faults/triton-extension-op-silent-miscompute.md`.

**#8730** vllm-ascend `0.14.0.RC1` (newer than the pinned `v0.13.0rc3` `bbe9bc0` +
vllm `v0.13.0` `72506c9`): "强化学习场景用户设置top后报错，调用了 `ApplyTopPTopCustom`
算子，算子侧反馈非cann算子，需要框架侧定位为什么会下发这个" — no framework traceback, only
an op name. Track: **ownership**, not a device fault. The recorded run split it the right
way: `…Custom` ops are built from `csrc/build_aclnn.sh` and installed under
`vllm_ascend/_cann_ops_custom`, so "not a CANN op" is by design; it named the three
top-k/top-p dispatch sites (`sample/sampler.py:64-78`,
`sample/rejection_sampler.py` via `patch/worker/patch_rejection_sampler.py`,
`worker/v2/sample/sampler.py:19,48`), showed the name exists at no pinned ref, and kept
the registry lane separate. Two costs worth not repeating: its second call dumped
`SOURCES.json` plus a ~900-name history listing (~25 KB, against `SKILL.md` rule 7), and
it typed `rg -rn` (the documented `--replace` trap) and had to kill the job and redo it.
Page: `../custom-ops-runtime-faults/custom-op-ownership-and-dispatch.md`.

**#7443** vllm-ascend v0.17.0rc1 (`e20f0b1a`) + vllm v0.17.0 (`b31e9326`), DeepSeek-V3.1
on a W8A8 checkpoint, `--quantization ascend --enable-expert-parallel
--data-parallel-size 4 --tensor-parallel-size 4`: dies at `determine_available_memory()
→ profile_run()` on the first quantized dense linear with `call
aclnnQuantMatmulWeightNz failed, error code is 161002` and
`AclNN_Parameter_Error(EZ1001): Tensor scale not implemented for DT_FLOAT16, should be
in dtype support list [DT_UINT64, DT_BFLOAT16, DT_INT64, DT_FLOAT]`. Track: custom-op
**parameter** rejection — the op is registered, the launch was refused for one operand
dtype — whose owner is the quantization method, not the package and not DP/EP/TP. The
recorded run verified the reporter's reading at the pin (`w8a8_dynamic.py:86-89` passes
`layer.weight_scale` while `:103` builds a `weight_scale_fp32` nobody reads), noticed
that only `DT_FLOAT16` is missing from the printed support list, and *did* find the
asymmetry it should have led with — `:249` still passes the non-fp32 `w2_weight_scale` —
but dismissed it as safe because `moe_mlp.py:175` casts w1's scale, a rescue that covers
gmm1 only while `moe_mlp.py:144` derives the GMM2 **output** dtype from
`w2_scale[0].dtype`, the gate's second operand. The reporter's later public comment
confirms the miss: after the linear repair the identical error returned in the MoE path
(`aclnnGroupedMatmulWeightNz … 161002`, fix pushed as PR #7444). Answer with the whole
parameter family plus the output-dtype half, one call:
`scripts/quant-scale-dtype-audit.py`. Page:
`../custom-ops-runtime-faults/quant-scale-dtype-gate.md`.

**#6243** vllm-ascend v0.12.0rc1 (reported ref, no commit named; pinned views at
v0.13.0rc2 `a17059a2`), DeepSeek-V3.2 on two 800I A2 nodes: hundreds of requests served,
then `call aclnnKvRmsNormRopeCache failed, error code is 561002`, `ERR00100 PTA call acl
api failed`, `EZ9999 … cos or sin shape is invalid.[FUNC:GetShapeAttrsInfo][FILE:
kv_rms_norm_rope_cache_base_tiling.cpp][LINE:291]` and `<Op> do tiling failed, ret is -1`
with the `NnopbaseExecutor*` chain, at `sfa_v1.py:804 forward → :559 exec_kv`. The body's
"做出如下修改：" is followed by nothing, i.e. the modification the reporter applied is not
in the paste, and the later comments add the current `exec_kv` source and the operator
owner's answer "算子侧认为为调度问题". Track: custom-op **attribute** rejection (nothing
launched). The frame is the 1024-token-gated `exec_kv`, so MLAPO was off or the batch was
a prefill chunk; `cos` is `[tokens,1,1,64]` from `rotary_embedding.py:107-119` and the
rows can differ from the kv rows under graph buckets, the DSA-CP slice
(`sfa_v1.py:205-236`) or the SP gather (`:790-796`). Same message family as #2732
(DS-V3.1, torchair graph with `graph_batch_sizes:[32]`, DP2, `mla_v1.py` path) and, via
`interleave_rope`, #2702. The recorded run had the reading right by turn 12 and shipped
**no answer** — 15 turns, 50.7 K characters of reasoning against 67 characters of text,
the last turn a 14.4 K block holding the whole ranked answer cut at the cap. Page:
`../custom-ops-runtime-faults/tiling-attribute-rejection.md`.

**#6529** vllm-ascend v0.14.0rc1, DeepSeek-V3.2 on the pinned 4-machine 1P1D recipe
(2 P nodes `--enforce-eager`, 2 D nodes with `FULL_DECODE_ONLY` + `deepseek_mtp` +
`--async-scheduling`, decode DP8 × TP4): the first `curl` takes **both** D nodes down with
`AclrtSynchronizeDeviceWithTimeout … error code is 507015` plus
`rtMemcpy … the current capture mode does not support this operation`, at
`model_runner_v1.py:1905 _generate_dummy_run_hidden_states` → `acl_graph.__call__` →
`acl_graph.py:199 torch.npu.synchronize()`, three workers on chip 7/die1, 6/die0, 0/die0
at once. Track: the MTP + captured-FULL-decode-graph device-fault family (#10700 shapes,
#6002/#5661 precedents), not a PD connector or a bad card; the sync point is only where
the trap surfaces, and the DP groups that lost the request are the ones replaying the same
bucket. First rung: `--enforce-eager` on the D nodes with MTP and PD kept. The run's
environment check also cleared the stack (`torch==2.9.0` / `torch-npu==2.9.0` / CANN 8.5.0
is what the pin requires; the release note's "torch-npu 2.8.0.post1" line is stale). It
shipped a ranked answer from turn 6 — the only run of its batch that did.

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

**#5236** vllm-ascend v0.12.0rc1 (`42ceaf08`) + vllm v0.12.0 (`4fd9d6a8`),
Qwen3-235B-A22B on the `v0.12.0rc1` image: the body was the **empty template** (no env
dump, no command, no traceback, no error code) and the whole symptom was the title,
"an aclnnRmsNorm operator execution error occurs". Track: device-side operator execution
error. The recorded run shipped a complete ranked answer — source-verified that the pin has
**no** `csrc/rms_norm` (so the named op can only be CANN's built-in `aclnnRmsNorm` behind
`torch_npu.npu_rms_norm`, `vllm_ascend/ops/layernorm.py:53-79`), that upstream
`qwen3.py:135-136,147-150` reaches the non-residual branch through per-head
`q_norm`/`k_norm` (so `aclnnRmsNorm` names a sync point, not an owner), and warned that the
topic's `add_rms_norm_bias` map is verified at v0.19.1rc1 only. What it ranked wrong: cause
1 was an async error surfacing at the norm, with the environment/driver pairing at 2; the
reporter's later record is **"This issue was solved by updating the npu driver to 25.25.0"**.
Lesson now in `../custom-ops-runtime-faults/index.md`: for a built-in `aclnn<Op>`
*execution* error with no code, the host driver/CANN pairing is the first ranked cause and
the version triple is the cheapest ask. Delivery: the ranked answer landed on the 11th
assistant turn (G2 was the miss), and the driver pairing sat only inside the rank-2
environment row as a version check — no driver-update rung appeared in it.

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

**#8826** vllm-ascend v0.13.0rc3 (`bbe9bc0d`) + vllm v0.13.0, A2 single node,
DeepSeek-V4-Flash-w8a8-mtp at `--data-parallel-size 2 --tensor-parallel-size 4`:
`call aclnnMoeInitRoutingV3 failed, error code is 561103`, with the usability ask
"如果有 dptp 限定，是否在模型启动时通过配置校验拦截". Track: device-side operator
error on the MoE init-routing op, answered from the dispatcher arithmetic rather
than the kernel. The run got the shape right — the code is a tiling check, not a
card/HCCL fault, and DP2×TP4 = 8 cannot reach MC2 on A2
(`ascend_forward_context.py:246-255`), so `TokenDispatcherWithAllGather` is the only
caller (`token_dispatcher.py:324-350`) — and shipped a ranked answer with the doc
diff (`docs/source/tutorials/DeepSeek-V4.md:149-167` is DP1×TP8 + EP, MTP 1,
`FULL_DECODE_ONLY`; no DeepSeek-V4 file exists under `tests/`). What it left open:
the hop from `torch_npu.npu_moe_init_routing_v2` to an aclnn entry named `…V3`, and
the code's meaning. Both are settled by prior issue **#3855**, whose paste carries
the same error line *with* its EZ9999 body — `The first dim of
expertTokensCountOrCumsum should be 128. [FUNC:CheckTokenCount]
[FILE:moe_init_routing_v2_tiling.cpp:131]` — which is the argument the answer should
have asked for (`../custom-ops-runtime-faults/moe-init-routing-tiling.md`).
Delivery: 19 assistant turns, a working verdict from the second turn and the
complete answer in the last message, so nothing was lost — but the decisive prior
paste sat one `sed` away from that second turn, which is the whole cost of treating
`history-titles.sh` output as the end of a precedent search.

**#17991** vllm-ascend `v0.23.0rc1` (`f4a08bdd`) / vllm `0.23.0` (`0fc695fc`),
8×910B3, FA3 NPU `v0.4.2`, Qwen3-0.6B with `attention_config={"backend":
"FLASH_ATTENTION"}` + `ngram` spec decode and `max_num_batched_tokens=256`:
`RuntimeError: ACL stream synchronize failed, error code:507015` (`EE9999 … aicore
exception`) at `vllm_ascend/attention/fa3_v1.py:122`. Real downstream defect one
line away from correct: the prefill half slices the request-indexed `block_tables`
by `num_decode_tokens` and passes an absolute `query_start_loc` as `cu_seqlens`, so
with spec decode the offset triples and the kernel reads outside the page table;
line 122 is the decode half's blocking `.npu()`, not the faulting kernel
(`../mixed-batch-splits/fa3-mixed-batch-split.md`, one call:
`scripts/batch-split-audit.sh`). **Shipped no answer**: 17 turns, ~51 K characters
of reasoning against 255 characters of message text — the complete source-verified
diagnosis was already written in the reasoning at the turn whose message text was
"Confirmed the fault mechanism. Let me pin two last facts", and the truncation kept
the plan and dropped the answer.

**#11231** pinned vllm-ascend `v0.21.0rc1` (`80610e4438`); the issue *title* says
0.21.0rc1 while the body's service ran v0.22.1rc1 (not in the snapshot). Qwen3.5-27B,
one host, TP2 / PCP2 / DCP1, `--max-num-seqs 32`, `FULL_DECODE_ONLY` +
`--async-scheduling` + MTP `num_speculative_tokens: 2`: serves, `aisbench` runs ~348
iterations, then every worker dies with `ValueError: could not broadcast input array
from shape (29,) into shape (31,)` at `pcp_utils.py:667` from `:660` inside
`_prepare_inputs` — a **host-side** member of the mixed-batch unit family, no device
frame. The recorded run decoded the two lengths (`num_decode_reqs=1`,
`num_reqs=32` = `--max-num-seqs`), pinned :660/:759 against the correct sibling :685,
named the fix and asked for the four source lines instead of another launch, and
shipped that with its version limits stated; the later v0.23.0 pin (`5cb98caa`) shows
the repair in-tree at `:821/:845/:938`. The reporter's later comment on the same
issue ("同时开启之前，已测过单独开启MTP，无报错") is the ablation the mechanism predicts —
page `../mixed-batch-splits/pcp-schedule-index-units.md`.

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

**#9984** vllm-ascend v0.19.1rc1 (`da421afad7`) / vllm v0.19.1 (`b1388b1fbf5a`),
GLM-5.1-w4a8 started with the tutorial's FLASHCOMM1 + MTP flags:
`AttributeError: 'AscendRowParallelLinear' object has no attribute
'aclnn_input_scale'` at `sfa_v1.py:707` ← `:544`. Track: DSA-CP o_proj weight
switching (`../dsa-sparse-attention/o-proj-quant-attribute-crash.md`). Delivered a
complete ranked answer in 10 assistant turns: `enable_sp()` ← FLASHCOMM1 →
`enable_dsa_cp()` (needs `index_topk`) → the non-layer-shard branch clones three
tensors only a static-W8A8 method creates, so the fix is FLASHCOMM1 off
(reporter-confirmed) or the W8A8 checkpoint; the pinned GLM5 tutorial ships
FLASHCOMM1 only with `glm5-w8a8`, and no `tests/e2e/**` FLASHCOMM1 config is W4A8.
The decisive detour was one `history-titles.sh 'aclnn_input_scale|o_proj_tp|dsa_cp'`
call (4 hits, one exact precedent: #7937, v0.18.0-a3 nightly, same frame on a
decode instance).

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
`checkUceErrAndRepair` frame already said it was not.

**#16079** vllm-ascend v0.26.0rc1 (`f2f74a16`) + vllm v0.26.0 (`568afb3a`), two
nodes × (tp=8, dp=4) + EP, MTP/DFlash draft, 128 K prompts, `kimi-k2.7-Code`:
an evalscope run dies with `ConnectionRefused [Errno 111]` on port 8016 because
the API server died first; the worker-node log then shows `!!!!!!! Segfault
encountered !!!!!!!` whose top frames are `rtEventCreateExWithFlag` →
`NPUEvent::record` → `c10d_npu::ProcessGroupHCCL::_allgather_base`, then
`VllmWorker-6 died unexpectedly (exit code: None)`, and the peers' gloo
`all_reduce` in `_sync_metadata_across_dp` fails with `Connection closed by
peer`. Track: HCCL/device (signal death). The dump carries **no**
`vllm_ascend/*.py` frame — only native symbols and CPython's own entry — so the
call site is a data gap; candidates are event/stream resources at collective
entry, the per-step tp-group `all_gather` in `rejection_sampler.greedy_sample`
(`enable_reduce_sample` defaults `False`, `ascend_config.py:346`, and the sampler
prints `Using fallback (non-reduce-sample) path`), or the image/CANN pairing
after an upgrade. Delivery: the run emitted three partial message blocks in
Chinese with the right first-half classification (the client error is secondary,
this is a worker segfault, not OOM) and was then cut **inside a source read** —
16 turns, ~43.5 K characters of reasoning against 3.3 K of message text, no
ranked ladder and no artefact request. The shape now lives in
`../hccl-and-hardware/worker-segfault-no-python-frames.md`.

**#4679** vllm-ascend v0.11.0rc3 (`b6d63bb`) + vllm v0.11.0 (`b8b302c`), 910B,
Qwen3-Reranker-8B with `--task score` + `Qwen3ForSequenceClassification`: engine
start dies inside `determine_available_memory` → `profile_run` →
`_dummy_pooler_run` with `call aclnnMatmulWeightNz failed, error code is 161002`
and `AclNN_Parameter_Error(EZ1001): Float32 weight NZ is unsupported by the
current SOC version [Ascend910B], now self is DT_FLOAT, mat2 is DT_FLOAT`;
reporter states v0.10.2rc1 worked. Track: weight-layout fault — upstream builds
the score head fp32 (`adapters.py:305-313`, `params_dtype=torch.float32`;
`:341-343` feeds it `x.float()`), and the downstream gate
`is_enable_nz` (`utils.py:74-84`) excludes only fp16/bf16 while
`VLLM_ASCEND_ENABLE_NZ` defaults to 1 (`envs.py:167-169`), so
`AscendUnquantizedLinearMethod.process_weights_after_loading`
(`ops/linear.py:43-51`) casts the fp32 head to `FRACTAL_NZ`. Escape:
`VLLM_ASCEND_ENABLE_NZ=0`. Delivery: the recorded run shipped a correct answer in
its 5th turn (verdict + `VLLM_ASCEND_ENABLE_NZ=0` first, then the evidence table,
the patch option, and the version-scope caveat that the reporter's 0.10.2rc1 tree
is not in the snapshot). Its two detours are the reusable part: it first routed
`EZ1001` to the custom-op **build/install** page (`../custom-ops-runtime-faults/op-not-registered-for-soc.md`,
14 KB) because the classification table keys on the code, and it wasted a call on
`rg -rn` (a `--replace` rewrite, not recursion) while hunting the cast site.
Route now: `../pooling-embedding/fp32-head-nz-format-crash.md`; the two `EZ1001`
text families are split in `../custom-ops-runtime-faults/index.md`, and the
fp32 policy is stated per ref because v0.23.0 rewrote the gate
(`_should_trans_nz`, "FP32 cannot use NZ", `utils.py:260-298`).

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

**#4323** vllm-ascend v0.11.0rc1 (`c5fe179c`) + vllm v0.11.0 (`b8b302cd`), Qwen3-32B at
`-tp 4` with an `AngelSlim/Qwen3-32B_eagle3` draft (`num_speculative_tokens: 4`,
`draft_tensor_parallel_size: 1`, `--max-model-len 4096`): v0.11.0rc0 answered 乱码 and
0.11.0rc1 died on the first request — all four ranks in the same second with
`SelfAttentionOperation setup failed!` + `atb::OperationSetup` inside
`torch_npu/lib/libop_plugin_atb.so`, then `… the current working operator name is
SelfAttentionOperation` / `ERR00100 PTA call acl api failed` at
`spec_decode/eagle_proposer.py:476`. Track: ATB op-plugin **setup** failure surfacing at
a proposer join point. The recorded run read the two layers correctly (the frame is not
the owner — history `#447` carries the same wording over `_npu_reshape_and_cache` — and
the ATB dump is this process's, not a replay) and eliminated two candidates by grep
(`draft_tensor_parallel_size` has no consumer under `vllm/v1/**`, so the drafter runs at
target TP; the pin's eagle3 e2e is single-card Llama-3.1 drafts only) — then shipped
**no message text on any of its 26 turns**, the worst delivery arithmetic on record,
with the ranked answer already written in the final reasoning block. The reporter's
later comment supplies the counter-check the run had recommended and never got: with
ACL graph on, `ASCEND_LAUNCH_BLOCKING=1` is refused (`ValidationError: ACL graph is
incompatible with ASCEND_LAUNCH_BLOCKING=1`), and plog only repeats the same line — so
the attribution order is `--enforce-eager` **then** blocking. Lessons in
`../spec-decode-mtp/atb-op-setup-failure-at-proposer.md`; delivery in
`delivery-discipline.md`.

**#7635** vllm-ascend v0.17.0rc1 (`e20f0b1`) + vllm v0.17.0 (`b31e9326`), two A2
nodes / 16 cards, Qwen3.5-397B-A17B-w8a8, `--tensor-parallel-size 8
--data-parallel-size 2 --enable-expert-parallel`: engine start dies in `profile_run`
with `call aclnnMoeDistributeDispatchV4 failed, error code is 561002` and the
self-explaining line `moeExpertNum is 32, in case of unlayered, it must no more than
24` (`MoeDistributeDispatchA2CheckAttrAndSetTiling`); the same command without EP
starts. Track: MC2 MoE communication at startup. The recorded run read the number as
experts **per card** (not the model's expert count), walked
`select_moe_comm_method`'s A2 row and the `moe_expert_num = len(expert_map)`
construction, and shipped the arithmetic (`512 // 16 = 32`) with the "no flag
escapes it" proof — no `comm_alg` in the kwargs so A2 always dispatches unlayered,
`HCCL_INTRA_*` only sets `need_expert_scale`, and `profile_run` always runs *at*
`mc2_tokens_capacity` so capture sizes cannot dodge the MC2 point. The verdict landed
on the first answer; the two ablations it offered (EP off, EP group ≥ 22 ranks) were
right but it did not predict the project's own fallback. Caution from that session:
`rg -rn '<pattern>'` is `--replace n`, so it silently rewrote its own matches and cost
a turn — use `-n` alone. Reporter-confirmed afterwards (2026-04-02): **PR #7364**
switches this case to ALL GATHER — service up, load test normal, expected comm cost —
with per-card experts > 24 as the CANN memory-safety cap. Now
`../feature-enable-recipes/moe-expert-parallel-with-dp.md` § Step 2b instead of a
plog/HCCL hunt.

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

**#15985** vllm-ascend `v0.26.0rc1` (`f2f74a16`) + vllm `v0.26.0` (`568afb3a`), filed
2026-09-08 as a maintainer **known-issues bulletin** (not a user report):
`[Bug][v0.26.0rc1]: Known Issues for DeepSeek-V4-Flash/Pro service hang on A3`,
tracking #15892 (decode node dies with `AclrtSynchronizeStreamWithTimeout` `507034`,
fault kernel `MoeDistributeDispatchV2`) and #15930 (1M-context PD + DSpark,
`IndexCheck_int64_high_performance_0` OOB `507035` then the same dispatch timeout).
Both tracked ids are outside the history corpus (earlier-month issues only), so the
`MISSING` files are coverage, not a failed search. The delivered answer treated it
as a claim-check: the release's own `tests/vllm_ascend_v0.26.0rc1_test_conclusion.md:7`
validates both models (Flash A3 4 nodes large DP 1P1D, Pro A3 2 nodes large EP
1P1D) and its §3 table attaches `507034` to an unrelated #15268 Ascend-950
`dsa_cp` start hang, so the code alone localizes nothing; the Pro run's 1M context
is ~7.4× the tutorial's `--max-model-len 135000`
(`docs/source/tutorials/models/DeepSeek-V4-Pro.md:408`, DSpark
`{"num_speculative_tokens":5,"method":"dspark"}`). Output-truncated at the 16 K cap
after 50 turns (1.1 M cumulative prompt tokens — its first block `cat`-ed a 25 KB
`SOURCES.json` plus the history listing inside one 62 K-character tool result),
verdict and both ranked lists shipped, the action ladder cut mid-item. Lane:
runtime device fault (lane C) / PD decode MC2, plus a second int64-index write
family — `../release-known-issues/index.md` (new topic), and rule 7 there on
splitting a two-issue bulletin across messages.

**#2349** vllm-ascend `v0.9.2rc1` (`b5b7e0e`) + vllm `v0.9.2` (`a5dd03c`), reporter's
later comment confirms torch 2.5.1 / torch-npu 2.5.1.post.dev20250619 / CANN
8.2.RC1; Qwen3-32B at `-tp 2 --max-model-len 80000` with
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `--enable-prefix-caching`,
`--enable-chunked-prefill`, `VLLM_CONFIGURE_LOGGING=0`: after serving for a while,
`rtKernelLaunchWithHandleV2 failed: 507035` / `reason=[vector core exception]`,
`OpExecCache run fail`, `ERR01005 OPS internal error`, `[ASSERT] … ascendc/gather_v3/gather_v3_base.h:137: Assertion (0 <= val && val
< this->gxSize_) Index 41513 out of range[0 40960)!`, and on rank1 `call aclnnAddRmsNorm failed,
detail:EZ9903`. Track: host-side index-space overflow — the bound is the row count
of the RoPE `cos_sin_cache`, which upstream builds from the **checkpoint's**
`max_position_embeddings` (`qwen3.py:178` → `rotary_embedding.py:132`) and Ascend
indexes by `positions` (`ops/rotary_embedding.py:78-85`, installed at `:286`), so
`--max-model-len 80000` past a 40960-row cache is the overrun; `507035` and the
other rank's `AddRmsNorm` line are wrappers/victims, not owners. The recorded run
had this in hand from its fifth turn and shipped **386 characters** of text across
16 turns, the whole ranked answer inside a 6.3 K-character reasoning block cut at
the 16 384-token cap. Page:
`../custom-ops-runtime-faults/gather-index-out-of-range.md`.

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
**#11188** vllm-ascend v0.21.0rc1 (`80610e44`), image `v0.21.0rc1-310p`, Atlas 300I Duo
310P3 ×2 at TP2, Qwen3.6-35B-A3B `w8a8` (GDN/hybrid), `--enforce-eager --max-model-len
8192`: `/v1/models` 200, first chat request 500 with `507018` / `[Error]: The aicpu
execution is abnormal` and a fault block `kernelName=IndexPut, soName=libpt_kernels.so,
funcName=RunCpuKernel, errorCode=0x2a` under `gdn_linear_attn.py`, plus host dmesg
`[ascend][devmm][ERROR] Reserve addr is invalid, should call halMemMap … ret=-22` once
per device. Track: **device-op fault**, but the owner is a Python statement or an
ungated config, not a kernel. The run got the reading right — fault block = CPU fallback
op, devmm = downstream, hardware lane retired by the reporter's own ablations (board,
BIOS 3101, firmware 7.7.0.1.231 `Compatibility: OK`, clean HDK 25.2.0 reinstall), family
= documented **preview** at `310p.md:405` — and delivered a ranked answer with a
`PYTORCH_NPU_ALLOC_CONF` injection claim it then corrected in two later turns (it
asserted the tutorial value "would not stop" `expandable_segments` before reading
`platform.py:696-715`, where the append is skipped when any of `expandable_segments` /
`max_split_size_mb` / `garbage_collection_threshold` is already present and `:715` logs
the result). Cheap lesson: read the gate before stating a negation about it. Page:
`../soc-platform-paths/310p-cpu-fallback-op-fault.md`.

**#8284** vllm-ascend v0.18.0rc1 (`99e1ea0f`) + vllm v0.18.0, image
`v0.18.0rc1-310p-openeuler`, Qwen3-30B-A3B-Instruct-2507, TP4 + `--enable-expert-parallel`,
`--dtype float16 --enforce-eager --max-model-len 8192` **plus `--kv-cache-dtype fp8`**:
`Using fp8 data type to store kv cache` (`config/cache.py:212`), `Available KV cache
memory: 8.82 GiB`, `GPU KV cache size: 768,384 tokens`, engine up in 7 s — then the
first request kills all four ranks at `device/device_op.py:32`
`torch_npu._npu_reshape_and_cache` → `RuntimeError: ReshapeAndCacheOperation setup
failed!`. Track: **device-op fault**, cause = an unsupported dtype on a `-310p` image
(`310p.md:5` float16-only; `kv_cache_dtype` is only *stored* on the attention impl,
`attention_v1.py:371`, and consulted nowhere; `_310p/` has no dequant path; the cache is
FRACTAL_NZ by construction). The run found the
whole chain and **shipped no message**: 35 assistant turns, 1.85 M cumulative prompt
tokens, cut at the 16 384-token cap inside a reasoning block that already held the ranked
answer, with three of its calls empty greps for `setup failed` in trees the snapshot does
not carry and a mid-session 33 K-character platform-page read. Page:
`../soc-platform-paths/310p-kv-cache-dtype.md`.

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

**#17082** vllm-ascend `v0.23.0rc1` (`f4a08bd`) + vllm v0.23.0, Atlas 300I Duo
310P3 (one logical NPU), image `v0.23.0rc1-310p` with CANN `9.1.0-beta.1`,
`Eco-Tech/Qwen3.5-9B-W8A8-310P` (`W8A8_DYNAMIC`, `--quantization ascend`), TP1,
fp16, `--enforce-eager`: all weights load, then
`EngineCore.determine_available_memory -> profile_run` dies and the device DFX
block names `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21`
(kernel hash `5247287448945562503`), `runtime error: 507015`, `error: L0C
read/write conflict`, while the Python frame differs per attempt
(`aclnnDynamicQuantV2`, `aclnnRmsNorm`). The recorded run answered in 12 turns
and was right that no in-tree flag reaches it (`Dockerfile.310p:18`'s CANN base,
no `507015`/`L0C`/`QuantBatchMatmul` string in the tree), but it told the
reporter **not** to add `VLLM_USE_V2_MODEL_RUNNER` and left the retest as
"rebuild with a newer CANN base". The reporter's later public retest passed on
`quay.io/ascend/vllm-ascend:nightly-main-310p` (CANN 9.1.0, vLLM 0.29.0,
vllm-ascend `0.19.1rc2.dev2294+gc173a64a4`) with
`VLLM_USE_V2_MODEL_RUNNER=1` and the banner `Using NPUWorker310 and
NPUModelRunner310V2`, 8 workload cases / 100 measured requests, prefix caching
off — with CANN **and** runner both changed, so neither axis is isolated and the
V1-path question stays open. The 310P runner is a version-gated opt-in, so on the
reported pin the flag was indeed inert; the transferable corrective is to offer
both version axes with the banner as the activation proof
(`../soc-platform-paths/310p-int8-matmul-startup-fault-and-runner-v2.md`).

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

**#2807** — vllm-ascend `v0.10.1rc1` official image (`m.daocloud.io/quay.io/ascend/vllm-ascend:v0.10.1rc1`,
tree `7e16b4a7cdb1`), Qwen3-30B-A3B-W8A8 on 4 devices, `vllm serve --quantization ascend -tp 4
--enable-expert-parallel` (both also self-quantized with MindStudio 8.1RC2, same error): OCR-garbled paste of
`call aclnnGroupedMatmulV5 failed, error code is 161002`, `EZ1001 … Tensor … dtype is DT_BFLOAT16 but found
DT_INT32` and `GMM:y dtype does not match with required dtype[DT_BFLOAT16]`, deepest frame
`vllm_ascend/ops/common_fused_moe.py:99` `gate_up_output = torch_npu.npu_grouped_matmul(` — which matches that
line in the pinned tree exactly, so the failing call is the MoE's first grouped matmul and not the scale
operand. The recorded run correctly separated the two `161002` families, read the launch's
`output_dtype=torch.int32 if use_int8_w8a8` (`:106`), the `ep_size < 16 → allgathercommimpl` selection
(`:483-490`) whose `permute` ignores `apply_a8_quantization` and hardcodes `quant_mode=-1`
(`moe_comm_method.py:166/195/202`), and the e2e coverage gap (the only W8A8 case is TP2 + `enforce_eager=True`)
— then spent the remaining budget deciding what `quant_mode=-1` *means*, which no export carries
(`rg -ln 'moe_init_routing' Ascend/` is empty in that case view), and shipped **no message text at all**. Page:
`../custom-ops-runtime-faults/quant-scale-dtype-gate.md` rule 5 (the two owners, the effective anchors, the
`--enforce-eager` rung, and the "do not derive an op attribute's semantics locally" stop).

**#2387** image `v0.10.0rc1` (vllm-ascend `4604882a3e2c` + vllm `6d8d0a24c02b`),
4×910B4, `Qwen3-32B-w8a8` served with no `--dtype` (so the checkpoint's
`torch_dtype: float16`), `--max-model-len 10240 -tp 4`. The engine gets through
weight load and `torch.compile takes 12.21 s in total`, then all four ranks die in
the same second: `NPU function error: call aclnnAddRmsNormQuant failed, error code
is 561000` + `ERR00100 PTA call acl api failed` + `EZ9999 … Cannot find bin of op
AddRmsNormQuant, integral key 0/1/|float16/ND/float16/ND/float16/ND/float16/ND/int8/ND/int8/ND/float16/ND/.`
+ `Kernel Run failed. opType: 31`. Track: a pre-launch **kernel-binary** gap for one
dtype key — the op is registered (no `EZ1001`), so neither the registry lane nor
`npu-smi`/image-tag lane applies, and the `561000` token is what sent the early
turns through the `does not has any binary` (#6082) and MC2/HCCL precedents. What
settles it is the key's own dtype list against the engine's `dtype=torch.float16`
line, then `--dtype bfloat16`; the fused-op caller is `models/qwen3.py:47-57` →
`ops/layernorm.py:49-54` with the scale in `params_dtype` (`w8a8.py:60-64,125-135`),
and no env switch exists at that ref. Delivery: 15 assistant turns, 35 K
characters of reasoning against 6.2 K delivered, **twelve of the turns carrying no
message text at all** and the first fourteen no verdict line — the ranked answer
exists but arrived on the last turn with ~1.7 K tokens of the cap left. Page:
`../custom-ops-runtime-faults/cannot-find-bin-integral-key.md` (new).

**#393** vllm-ascend `v0.7.3rc1` (`f025df05f614`) + vllm `0.7.3`
(`ed6e9075d31e`), torch-npu `2.5.1.dev20250308`, ms-swift 3.2.1 GRPO whose rollout
engine is the **V0** engine, Qwen2.5-class model. After ~5 steps
(`Train: 0%|… 5/8724 [02:25<69:44:50]mki_log log dir:/root/atb/log exist`) every
rank dies with a bare `RuntimeError: setup failed!` whose deepest `vllm_ascend`
frame is `attention.py:869` `torch_npu._npu_flash_attention_qlens`. Track: host-side
ATB operation setup refusing the pack — nothing launched. The reading that paid:
the branch (`:866-876`) requires a prefix-cache hit with chunked prefill off, and
upstream defaults `enable_prefix_caching` to `bool(VLLM_USE_V1)` = False on V0
(`vllm/engine/arg_utils.py:217-220`), so the rollout engine set it explicitly; the
mask it passes is a literal `128` one-block `[128, 128]` (`:612-617`, `platform.py:120-123`,
`model_runner.py:614-645`) while a partial hit's `query_lens` and a full hit's
`context_lens` are unbounded. Two cheap discriminators (prefix caching off, or
chunked prefill on) plus the honest gap: ATB's validation text is in no export, so
the emission reason stays a hypothesis. Delivery: 9 turns, 7 of them carrying text, the
ranked answer on the 9th (5.9 K characters against 41.9 K of reasoning) — the
answer shipped because the session stopped there, not because the reasoning/text
ratio was healthy; the reason `#3128`/`#4178` above do not ship at 9 turns is that
they keep opening new areas once the cause is named. Page:
`../custom-ops-runtime-faults/setup-failed-call-site.md` (new).

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

**#3810** vllm-ascend v0.11.0rc0 (`00ba07102212`) + vllm v0.11.0 (`b8b302cde434`),
one 910B4 (CANN 8.2.RC1), Qwen3-VL-8B-Instruct with
`--max_model_len 128000 --gpu-memory-utilization 0.75` and one image request:
`call aclnnRmsNorm failed, error code is 507015` → `fftsplus aivector error` +
`errStr: The DDR address of the MTE instruction is out of range` +
`fault kernel_name=unpad_flashattention_bf16_1_mix_aic` + `retCode=0x26, [aicore
exception]`, `EngineCore encountered a fatal error`. The engine's own
`Dumping scheduler output … SchedulerOutput(...)` puts the step at
`scheduled_encoder_inputs={}`, `num_computed_tokens=[2048]`,
`num_scheduled_tokens={…: 2048}`: a **continuation** chunk of a split image prompt
(the encoder step is behind it). Which LM branch ran is `_build_attn_state`
(`model_runner_v1.py:1595-1618`) × `torch.version.cann` (`attention_v1.py:316/340`,
`:486` on CANN 8.3+, `_npu_paged_attention_splitfuse` `:505` on the reporter's
8.2.RC1), and the vision tower shares the same unpad-attention family
(`qwen2_5_vl_without_padding.py:122`) — so the launch needed
`ASCEND_LAUNCH_BLOCKING=1` or the plog's `origin_op_name`, not a source walk.
Reporter-confirmed afterwards: the same serve answered `The information in this
picture is` (a collapsed answer), `0.12.0` still had the problem, and the
**v0.13.0rc0 image answers the same request correctly** — a post-pin fix, so the release is part of the verdict, with
`--max-num-batched-tokens` / `--no-enable-prefix-caching` as the interim
ablations. The recorded run opened `attention_v1.py`, `platform.py`,
`qwen2_5_vl_without_padding.py` and the `max_num_batched_tokens` arithmetic for
28 tool calls and shipped **no message text** —
`../custom-ops-runtime-faults/aclnn-error-dump-reading.md` (built-in-kernel
section).

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
