# Device and communication cases

Device dumps naming an operator, op-store and quant-lane faults, graph-capture
crashes, indexer-scatter corruption, segmentation deaths, and mixed decode+prefill
batch faults. Routes: `../custom-ops-runtime-faults/index.md`,
`../soc-platform-paths/index.md`, `../dsa-sparse-attention/index.md`,
`../mixed-batch-splits/index.md`.

The family is split by owning page: HCCL/cluster cases — collective errors,
communicator creation, transport and link faults, MC2/AIV dispatch and profile-run
sync points, the RPC watchdog, executor teardown, multi-instance resources, and the
latched-error and segfault dumps that topic owns — are in
`recorded-cases-hccl-and-watchdog.md` (routes `../hccl-and-hardware/index.md`,
`../runtime-memory/index.md`, `../pd-disaggregation/index.md`); every other case
stays here.

Inherits the scope and rules of `recorded-cases/index.md`, whose page for this
family carries one line per case in these two files.

**#6562** image `v0.14.0rc1-openeuler` (pin vllm-ascend v0.14.0rc1 `52d4acf`
+ vllm v0.14.1 `d7de043`), A2 910B3 ×8, `vllm serve Qwen3-Coder-Next --tensor-parallel-size 4
--enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-batched-tokens 4096
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --gpu-memory-utilization 0.8`:
titled 「推理中出现告警」, and the body is only `(Worker_TPn pid=…) [WARNING] Please DO
NOT tune args ['num_warps']!` / `['num_warps', 'num_stages']!` lines under a
`POST /v1/chat/completions … 200 OK`. Delivered as a **no-fault verdict** (14 turns, 19
tool calls, 37.7 K characters of reasoning against 3.4 K of message, `completed`): the
bracketed lists are upstream vLLM's two autotune config shapes (`fla/ops/l2norm.py:23`
warp-only; `chunk_o.py:33`, `chunk_delta_h.py:34`, `solve_tril.py:31`, `wy_fast.py:22`
warp+stages), vllm-ascend v0.14.0rc1 has **zero** `autotune` in its tree, and the string
exists in no export at that pin — an installed-wheel string. What it still got wrong is
the search order: four calls went to which upstream/downstream kernel fired (no rung of
the answer) and the history search ran titles only, so the one body that carries the
block — **#3548**, where it sits directly above a real `EE1001 … coreDim=66976 can't be
greater than UINT16_MAX` in `layer_norm_fwd_kernel` — was never opened. Page:
`../observability-and-logs/triton-autotune-tune-args-warning.md`; the crash under the
same block is `../custom-ops-runtime-faults/triton-grid-uint16-limit.md`.

**#15906** vllm-ascend v0.26.0rc1 (`f2f74a1`) + vllm v0.26.0 (`568afb3`), GLM-5.3-w8a8c8,
TP8 × DP4 over 2×910B2C (32 dies), `--decode-context-parallel-size 8
--cp-kv-cache-interleave-size 128`, `--max-model-len 1048576`, both C8 flags,
`FULL_DECODE_ONLY` + MTP 3: the serve comes up, `evalscope perf` at `--parallel 1`
reports `Total / Success / Failed 5 / 1 / 4` and then `ConnectionResetError: [Errno 104]`,
and the master log has the incident at 14:21:22 on `Worker_DP0_TP5_DCP5_EP5` — `507035`
through `acl_graph.py:262`, `[Error]: The vector core execution is abnormal.`, aivec dump
`core id 47, error code = 0x800000`, `errorStr: MTE accesses an invalid GM address`,
`retCode=0x31, [vector core exception]`, with the `tiling入口` marker at
`kv_quant_sparse_flash_attention_tiling.cpp:1672` (a leftover `OP_LOGE("ccccc", …)`, the
op's name rather than the fault site). The peer node's `507034` `[vector core timeout]`
arrives 6 min later. Same class as #14320 (LIQ variant, reporter-confirmed); the recorded
answer ranked the DCP replicated-block-table reading first and put the A3-only 1M
statement below it, then was cut at the cap — `../dsa-sparse-attention/dcp-c8-kv-address-fault.md`.

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

**#4373** vllm-ascend v0.11.0rc1 (`c5fe179`) + vllm `0.11.0+empty`, BigKernel prompt FA:
`npu_fused_infer_attention_score_symint … call aclnnFusedInferAttentionScoreV3 failed,
error code is 561002` then `E89999 … When layout is TND, not support Page Attention[FUNC:
RunBigKernelTilingWithParams][FILE:prompt_flash_attention_tiling.cpp][LINE:3400]` +
`FusedInferAttentionScore do tiling failed, ret is -1`, traceback through
`patch_attention_layer.py:67 → attention_v1.py:593 → :691 → :668 → :539`. Track: built-in op
tiling **capability** refusal (nothing launched), owner = the installed toolkit; the
pin documents `CANN >= 8.3.RC1` (`docs/source/installation.md:15`) and the collect_env block
carried only the two vllm lines. The run verified the launch pair at `:540-554` (`block_table`
with `input_layout="TND"`, `sparse_mode=3`) and ranked it cause 1 — chunked-prefill avoidance
rung 1, the CANN pairing rung 3 marked unverifiable — then shipped a full answer on that
ranking (run: complete answer, wrong order). The reporter's public follow-up the same day:
*"updating cann version to 8.3 can solve this."* Page:
`../custom-ops-runtime-faults/tiling-capability-refusal-cann-release.md`; the `not support`
verb, not the launch arguments, is what separates it from `#6243`.

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

**#2822** vllm-ascend `v0.10.1rc1` (`7e16b4a`) + vllm `[NETWORK_ADDRESS_1]`,
Phi-4-mini-instruct TP1, `lm_eval --tasks gsm8k --batch_size auto` (the paste is a CI
action log, so its `attention_v1.py:379` is one line off the tag it names): serves
1064/1319 samples, then `RuntimeError: shape '[255, 1, 3072]' is invalid for input of
size 786432` at `_forward_decode_only`'s `self.sliding_window is not None` branch. The
request count (`attn_metadata.seq_lens.shape[0]` = 255) met the ACL-graph-padded token
count (`786432 / 3072` = 256, `pad_for_cudagraph` under `use_aclgraph`) — a host-side
member of the mixed-batch unit family with no device frame, one division to classify
(`../mixed-batch-splits/decode-pad-request-vs-token-count.md`). **Shipped no answer**:
14 assistant turns, 16 384 output tokens spent almost entirely on reasoning and 51
characters of message text (turn 1); the division `786432/3072 = 256` was in the
opening reasoning and the rest went to `graph_pad_size`, DP-metadata sync and
`sliding_window` plumbing. The reporter closed it the next day ("Fixed by
…/pull/2830"), and the guard the pin lacks is visible at v0.14.0rc1
(`seq_lens.shape[0] == query.size(0)`).

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

**#1603** vllm-ascend `v0.9.0rc2` (`8dd686d`) + vllm `v0.9.0` (`58738772410c`),
Qwen3-32B bf16 TP4 on 910B, `--max-model-len 32768`, `--enable-prefix-caching`,
chunked prefill at `max_num_batched_tokens=2048`, PIECEWISE ACL graph
(`cudagraph_capture_sizes [512 … 1]`): after an hour of single-request serving, six
requests inside 8 s, then `[ASSERT] … gather_v3_base.h:137 … Index 775 … 829 out of
range[0 391)` on every rank, `RuntimeError: ACL stream synchronize failed, error
code507035` and `EngineDeadError`. Track: host-side index-space overflow **whose
table only the dump names** — 391 is the new request's own
`prompt_token_ids_len=391` (`SchedulerStats(queries=391, hits=256)`), the step
scheduled `135` tokens for it with `block_ids=[[47,48,116,117]]`, so `775 … 829`
belongs to the co-scheduled decoding request; the same log's RoPE cache is 40960
rows and `GPU KV cache size: 631,424 tokens`, so neither config table matches. The
deepest frame is `eval_with_keys-1.1 line 49` → `torch/ops.py:1116`, i.e. the op is
not in the paste. The recorded run spent all 16 384 output tokens naming that op and
shipped **0** characters; the reporter later confirmed a code patch fixed it. Page:
`../custom-ops-runtime-faults/gather-index-out-of-range.md` (§*When M matches
no config value*).

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
ratio was healthy; the reason `#3128`/`#4178` in `recorded-cases-hccl-and-watchdog.md` do not ship at 9 turns is that
they keep opening new areas once the cause is named. Page:
`../custom-ops-runtime-faults/setup-failed-call-site.md` (new).

**#4771** vllm-ascend v0.11.0rc2 (`a2e4c3f`) + vllm v0.11.0 (`b8b302c`),
Qwen2.5-32B-Instruct TP2 with its own trained **eagle3** draft
(`draft_tensor_parallel_size: 1`, `n=5`, `--max-num-batched-tokens 4096`, no graph
flag): the first request answers (the metrics line even reads `Drafted: 15,
Accepted: 2`), the **second** dies with `RuntimeError: operator():…
LocalScalarDenseNpu.cpp:47 … AclrtSynchronizeStreamWithTimeout(copy_stream), error
code is 507018` at `vllm_ascend/sample/sampler.py:28`, and the same block carries
`queryT(1)` against `actualSequenceLengthQ(33)` **then** `(2)`, plus an AICPU task
record (`kernelName=ScatterElements`, `retCode=0x2a`, `task_id=7315`, `chipId=1`).
Track: not a card, CANN or eagle3-weights fault. `sampler.py:28` is the
`1 <= int(k.max())` fast-path *condition* in `_apply_top_k_top_p`, i.e. a
`LocalScalarDenseNpu` D2H copy and therefore a host sync point (SKILL.md rule 11);
and the pair's **direction** — `N` far below both `M` values — excludes the
padding-bucket reading the FIA page's arithmetic assumes (at that pin the
`DecodeOnly` padding branch's `arange` stops one element short and can only give
`N = M + 1`). The run delivered the sync-point verdict, three ranked candidates and
the ladder, and left the owner open — the correct label; what it spent was ~40 K
characters of reasoning on the decomposition and a second `M`. Durable correction:
`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md` §*Third reading* — read the device
record first (`plog` `[AIC_INFO] fault kernel_name=`), `--enforce-eager`/`PIECEWISE`
rung 1, `n: 5→1` rung 2.

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

**#960** vllm-ascend v0.8.5rc1 (`ec27af34`) against a **self-built vllm main**
(`0.8.5.post2.dev0+g3015d5634`, pinned upstream v0.8.5 `ba41cc90`), verl DAPO rollout
(`vllm_rollout_spmd.generate_sequences` → `LLM.generate`, `VLLM_USE_V1=1`, one host
with 8 910B cards / 16 dies, `ASCEND_RT_VISIBLE_DEVICES=0..15`): the **first** generation step dies as `RayTaskError(AttributeError:
'NoneType' object has no attribute 'sampled_token_ids')` at upstream
`v1/core/sched/scheduler.py::update_from_output`, every frame `vllm/**` with no
`vllm_ascend` frame, and the reporter's four screenshots point at
`vllm_ascend/worker/worker_v1.py:175 return output if self.rank == 0 else None`.
Track: **downstream/upstream drift in a returned value**, not a device fault. The
recorded run found the textual asymmetry against upstream's `is_driver_worker`, then
refuted it correctly *for the multiproc executor* (`multiproc_executor.py:288` derives
one from the other) and had no third branch: the divergence is
`ExecutorWithExternalLauncher` (`vllm/executor/uniproc_executor.py:68-120`), where
`rank = os.environ["RANK"]` and `is_driver_worker = True` for every process — the
backend torchrun/SPMD trainers select — so only `RANK=0` returns an output and every
other engine dies in `update_from_output`. The paste's `core_client.get_output →
engine_core.step()` (the `InprocClient`) is the corroborating tell. The run also burned
its whole 16 384-token output budget inside that two-candidate sweep, emitted **no**
message text, and never fingerprinted the reporter's vllm ref (pasted `core.py:294`
vs the pin's `:192`; `dump_engine_exception` absent from the pin). Page:
`../upstream-downstream-drift/execute-model-none-external-launcher.md` (run: no answer
shipped).

**#13449** vllm-ascend v0.20.2rc1 (`367b8e62`), image `v0.20.2rc1-310p-openeuler`,
**one** Atlas 300I Duo 310P1 die (44 GB), CANN 9.1.0 / driver 25.5.t7, Qwen3.5 0.8B and
9B `--dtype float16 --enable-chunked-prefill --enable-prefix-caching
--mamba-cache-mode align --mamba-ssm-cache-dtype float16`, `FULL_DECODE_ONLY`
`[1,4]`: weights load, capture finishes, KV sized, `/v1/models` 200, the first chat
request kills EngineCore with `507018` + `The aicpu execution is abnormal` +
`kernelName=IndexPut, soName=libpt_kernels.so, funcName=RunCpuKernel` and
`RuntimeError: operator(): LocalScalarDenseNpu.cpp:23
AclrtSynchronizeStreamWithTimeout(copy_stream)`. Track: the 310P CPU-fallback op family
(#11188/#11450), with three things this case adds — the reporting point is the 310P
conv path's per-request `.item()` (`_310p/ops/causal_conv1d.py:137,142`), both
`index_put_` writes are inside `if attn_metadata.num_prefills > 0:`
(`gdn_310.py:249-269`, decode uses a dedicated op) so graph capture and `--enforce-eager`
are irrelevant, and `utils.py:793-794` proves the GDN 310P class is reached for a
Qwen3.5 checkpoint. The run delivered twice, corrected its own "preview only" verdict
after finding `tests/e2e/310p/singlecard/test_dense_model_singlecard.py:75,91` running
`Qwen/Qwen3.5-4B` on 310P single card, and turned that row into the config control.
Page: `../soc-platform-paths/310p-cpu-fallback-op-fault.md`.

**#13904** vllm-ascend **v0.23.0rc1** (`f4a08bddd0cc`) + vllm v0.23.0, 4× Ascend950PR,
DeepSeek-V4-Flash w8a8 MXFP8 (`quantization_config` `{"quant_method": "fp8", "fmt":
"e4m3", "scale_fmt": "ue8m0", "weight_block_size": [128,128]}`), `--data-parallel-size 4
--tensor-parallel-size 4 --enable-expert-parallel`: architecture detection, 46/46 shards,
KV init and the API all pass, `GET /v1/models` 200, and the **first** chat request is a
500 at `vllm_ascend/attention/dsa_v1.py:1566` `_forward_o_proj` →
`npu_transpose_quant_batchmatmul` → `device error type 0, error code is 161002` +
`EZ1001: Tensor x1Scale not implemented for DT_FLOAT8_E8M0, should be in dtype support
list [DT_FLOAT,]`; the body carries a framework-free `torch_npu` repro of the same call,
so the serve flags are not the variable. This is the `EZ1001` dtype gate's **MX member**
and the side of it where the caller is right: a **one-entry** support list means the
installed kernel build has only the non-MX scale variant, so the owner is the kernel ×
the SoC die, and the caller (`group_sizes=(0,0,32)`, `x1_scale`/`x2_scale` as
`….view(torch.float8_e8m0fnu)`, `:1572-1573`; twin at
`attention/context_parallel/dsa_cp.py:1234`) is doing what an MX checkpoint requires.
`device/mxfp_compat.py`'s `hasattr`-only shims all pass on that stack and are never
called from the attention path, so "symbol present" is not "kernel supports it", and the
A5 branch is selected by device type alone with no probe and no fallback — the remaining
branches consume an unquantized `wo_a` (`dsa_v1.py:1580`, `:1650-1662`), so a capability
gate needs a dequant path and is not a drop-in. The report also asks which 950 die was
validated: `AscendDeviceType` has one `A5` member and `soc_version == 260` sends both
dies to it (`utils.py:824-828`, `:857-858`; any `ascend950*` build string maps to A5,
`setup.py:181-187`), `Dockerfile.a5:52` defaults `ascend950dt_9582`, and the pin's `docs/`
contain neither `950PR` nor `950DT` while the model row is `A2/A3`
(`support_matrix/supported_models.md:20`) — an enumeration gap, answered as such with the
build-with-your-own-`SOC_VERSION` and sibling-die checks. The aclnn kernel's dtype table
and `StructKernelNpuOpApi.cpp` are not in the export (`Ascend/op-plugin` 103 files, no
`opapi/`), so the die-versus-CANN split is a hardware check, not a source read. Page:
`../soc-platform-paths/a5-mxfp-refusal-and-die-coverage.md`; caller-side gate:
`../custom-ops-runtime-faults/quant-scale-dtype-gate.md` rule 0.

**#2700** vllm-ascend `v0.10.0rc1` (`4604882a3e2c`) + vllm `v0.10.0`
(`6d8d0a24c02b`), MedGemma-4B (Gemma-3 family) TP1, bf16, `max_model_len=131072`,
`enforce_eager=True`: the first chat request answers, the next one — a new completion
whose prompt shares 256 tokens with it — kills EngineCore on its first step with
`[rank0]:[E902 …] PagedAttentionOperation setup failed!` out of
`atb::OperationSetup` in `libop_plugin_atb.so`, then
`RuntimeError: The Inner error is reported as above … the current working operator
name is PagedAttentionOperation` + `ERR00100 PTA call acl api failed`. Nothing was
misconfigured: `enable_chunked_prefill` and `enable_prefix_caching` are **V1
defaults** (`vllm/engine/arg_utils.py:1530-1536`, "V1 always uses chunked prefills and
prefix caching"), and the engine's own `Dumping input data … with config` line prints
them. The branch is arithmetic in the attachment's dumped `SchedulerOutput`:
`num_computed_tokens=256` + `num_scheduled_tokens={req: 27}` + `prompt_token_ids_len=283`
+ `prefix_cache_stats(queries=283, hits=256)` gives `seq_lens=283 ≠ 27` and `27 ≠ 1`,
and with chunked prefill on the state is `ChunkedPrefill`
(`model_runner_v1.py:1119-1138`) ⇒ `_npu_paged_attention_splitfuse`
(`attention_v1.py:419`; a `head_size == 192` model diverts to
`vanilla_chunked_prefill`). Request 1's 283-token prompt fitted one step under the
2048/5120 token budget, so it took `PrefillNoCache` → `_npu_flash_attention` and then
`DecodeOnly` → `_npu_paged_attention`: the prefix hit, not a warm-up or a capture,
creates the third state, and `--enforce-eager` is already in the dump. The splitfuse
mask is built for this state only (`get_splitfuse_attn_mask`,
`attention/attention_mask.py:54-96`; builder seeded at
`min(max_model_len, PAGED_ATTENTION_MASK_LEN=10000)`) and its shape closes
(`[27, 283]`, rows = absolute positions), so the #393 "mask narrower than the
context" story does not apply, and ATB's own validation text is in no export
(`Ascend/op-plugin` here carries no `paged_attention` string). Ladder:
`--no-enable-prefix-caching`, then `--max-num-batched-tokens` below the prompt length
for a fresh-prompt repro of the same branch. The model row is ✅ at this pin
(`support_matrix/supported_models.md:32`, `:71`), so the ❌-row page — the only other
page naming this op, at ACL-graph replay — does not own the paste. Recorded run: 17
assistant turns / 31 tool calls, 47.5 K characters of reasoning, **zero** characters
of message text, `output_truncated`; the dumped `SchedulerOutput` had settled the
state by turn 9. Page:
`../custom-ops-runtime-faults/setup-failed-call-site.md` (§`PagedAttentionOperation`),
row `../first-checks/signature-routes/device-ops-and-collectives.md`.

**#4135** vllm-ascend **v0.11.0rc1** (`c5fe179`) + vllm v0.11.0 (`b8b302c`), image
`vllm-ascend:v0.11.0rc1`, Qwen3-Next-80B-A3B-Instruct TP4
(`--gpu-memory-utilization 0.7 --enforce-eager`, default `max_model_len 262144`):
the serve comes up and the first long chat request kills all four workers with
`npu_fused_infer_attention_score_symint … call aclnnFusedInferAttentionScoreV3 failed,
error code is 561002` + `E89999 … When layout is TND, queryD(256), keyD(256) and
valueD(256) must be same equal 192/128/64, or queryD and keyD equal 192 and valueD equal
128[FUNC:CheckInputShapeWhenLayoutIsTND][FILE:prompt_flash_attention_tiling.cpp][LINE:3701]`,
traceback `patch_attention_layer.py:67 → attention_v1.py:593 → :691 → :668 → :539`.
The sentence prints **dims**, so the refused property is the model's `head_dim`: Qwen3-Next
defaults `head_dim=256` (`vllm/transformers_utils/configs/qwen3_next.py:208`, taken
verbatim at `vllm/model_executor/models/qwen3_next.py:640-643`) and the paged-TND tiling
accepts 64/128/192 (plus the MLA pair 192/128). Reachability is settled by the paste's own
`Dumping scheduler output … SchedulerOutput(...)`: two fresh prompts (17 and 25 tokens)
beside `CachedRequestData(num_computed_tokens=[350])` scheduled **1** token ⇒
`seq_lens != num_scheduled_tokens` ⇒ `ChunkedPrefill` ⇒ `_forward_v1_style`
(`model_runner_v1.py:1576-1600`, dispatch at `attention_v1.py:651-670`), whose only
head-size escape is `head_size == 192` (`:503-518`); the chunk budget is
`DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` (`vllm/utils/__init__.py:88`). The CANN-release rung
of #4373 does **not** apply: the image builds from
`quay.io/ascend/cann:8.3.rc1-910b-ubuntu22.04-py3.11` (`Dockerfile:18`), the documented
floor, and the documented `docker run` mounts only the driver — a `version.cfg` ask would
restate the base tag. The reporter later confirmed the other half of the split on the same
image: short `curl` chats answer. Ladder: `--max-num-batched-tokens` ≥ the longest prompt
(`PrefillNoCache` → unpaged `_npu_flash_attention`, `attention_v1.py:651-652,:325,:350`;
its D=256 acceptance is a **hypothesis**, one run settles it), then keep batches from mixing
a continuation with a fresh prompt, else it is a toolkit/coverage gap with a framework-free
FIA repro. `--enforce-eager` is inert (already in the launch, and the state gate is not
graph-dependent). Recorded run: the chain above was verified by its 15th assistant turn (message 39) and the
session shipped **zero** message text — 20 assistant turns / 30 tool calls, 46.7 K
characters of reasoning against 86 characters (the opening line only), `output_truncated` at
the 16 384-token cap while checking the unpaged op's dim set and the pin's e2e tree (which
has no `qwen3_next` case). Page:
`../custom-ops-runtime-faults/fia-tnd-head-dim-refusal.md`, rows
`../first-checks/signature-routes/device-ops-and-collectives.md` and
`../custom-ops-runtime-faults/index.md`.
