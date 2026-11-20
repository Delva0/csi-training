# Device and communication cases

A device dump naming an operator, HCCL/transport link faults, RPC watchdog and
profile-run sync points, indexer-scatter corruption, segmentation deaths, and
mixed decode+prefill batch faults. Routes: `../custom-ops-runtime-faults/index.md`,
`../hccl-and-hardware/index.md`, `../dsa-sparse-attention/index.md`,
`../mixed-batch-splits/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

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
