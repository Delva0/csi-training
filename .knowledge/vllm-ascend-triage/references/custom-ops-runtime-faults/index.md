# Custom-op runtime faults (aclnn / AscendC)

## Scope

Device-side failures raised while running a vllm-ascend custom AscendC operator
(and the pre-launch counterpart, where aclnn cannot resolve the op for the SoC at
all — a build/install fault, see the first rule below):
`call aclnnX failed`, `AclNN_Runtime_Error`, Nnopbase executor asserts, `fftsplus aivector
error`, MTE/DDR "out of range", retCode `0x7150050` / runtime result `507011`, or a signal
from a `csrc/<op>/*_torch_adpt.h` frame. The silent counterpart belongs here too: an
attention op that runs, returns the right shape and the wrong numbers, with no dump and
often a framework-free `torch_npu` repro (`attention-op-numeric-errors.md`).
**Host-side tiling refusals are in scope even though nothing launched**: an
`OP_LOGE` + `GRAPH_FAILED` from `csrc/<op>/op_host/*_tiling_validation.h` or a
`TORCH_CHECK` in an op adapter is a *contract* refusal, and the numbers it prints are
the caller's per-request counts (`host-tiling-table-bound.md`,
`gdn-conv1d-accepted-tokens-varlen.md`). The rule for both: reconcile the printed pair
with the caller's own bookkeeping before reading any kernel.
A Triton-Ascend kernel that fails to *compile* — `MLIRCompilationError`,
`[ConvertLinalgRToBinary] encounters error`, `bishengir-compile … returned non-zero
exit status 1` with `vllm_ascend/ops/triton/**` frames — is **not** this lane even when
the reporter words it as "failed to compile custom operators": nothing here touches
the AscendC extension, and the owner is the installed wheel/compiler pair
(`../image-dependency-faults/triton-ascend-pair.md`). **One detail line under that same
banner is this topic, not the pair**: `ub overflow, requires N bits while M bits
available` is the vector-core UB budget refusing the kernel's own `tl.constexpr` tile, so
the owner is the kernel's tile constants × the model's shape and the fix is a smaller
tile or a path with no compile step — `triton-ub-overflow-tiling-budget.md`. It sits here
because it is the same tiling-budget family as `fa-tiling-argument-refusals.md` and
`tiling-attribute-rejection.md`, only refused at compile time instead of at launch. A Triton-Ascend kernel that
*compiles and runs but returns the wrong values* (a bf16 `extract_slice`/
`insert_slice` at an unaligned start, a hand-written extension-op kernel) is not this
lane either: no `csrc/` operator is involved, so the owner is the wheel and the audit
is the in-tree Triton call sites —
`../image-dependency-faults/triton-extension-op-silent-miscompute.md`.
The third Triton shape **is** this topic: a kernel that runs and **faults the device**,
where the plog's `fault kernel_name=` resolves to a `@triton.jit` function in
`vllm_ascend/` (the sampler/penalty kernels are the recorded family). The kernel is
downstream code and the owner is the launch site's arguments —
`triton-jit-kernel-device-fault.md`. Its **launch-refused** sibling belongs here too:
`EE1001 … The argument is invalid.Reason: coreDim=<N> can't be greater than UINT16_MAX`
with `rtKernelLaunch execute failed, reason=[invalid value]` is a host-side grid
overflow in a `@triton.jit` kernel (the backend launches one block per program, ceiling
65 535), so `N ÷ (per-rank head count)` is the token count of that forward and the
caller's row count is the lever — `triton-grid-uint16-limit.md` (recorded #3971).

## Rules

- **Split parameter/registry errors from device faults first.** `AclNN_Runtime_Error`
  / a dump / tiling text means the kernel ran or was launched: the operator and shape
  matter. `AclNN_Parameter_Error(EZ1001)` is a **family** and the text after the code
  decides which:
  (a) `Get regInfo failed`, `binary_info_config.json of socVersion [...] does not
  support opType [...]` — aclnn could not resolve the op for this SoC **before any
  launch**: the owner is the build/install chain, and no tiling fact will help —
  `op-not-registered-for-soc.md`.
  (b) `<Dtype> weight NZ is unsupported by the current SOC version [...]`, `now self
  is DT_<x>, mat2 is DT_<x>` (also `… is unsupported … now self is DT_…`) — the op is
  registered and the launch was rejected because an input arrived in a dtype × layout
  pair this SoC has no kernel for. Owner: the Python that produced the layout, not
  the package. A recorded instance is an fp32 pooling/score head cast to
  `FRACTAL_NZ` by the weight-format gate, which dies on the startup dummy pooler run
  (#4679, v0.11.0rc3) — `../pooling-embedding/fp32-head-nz-format-crash.md`. Answer
  it with that track's gate grep, not with `SOC_VERSION` or `ASCEND_CUSTOM_OPP_PATH`.
  (c) **quant-parameter dtype**: a rejection on the same code —
  `Tensor scale not implemented for DT_FLOAT16, should be in dtype support list
  [DT_UINT64, DT_BFLOAT16, DT_INT64, DT_FLOAT]` on `aclnnQuantMatmulWeightNz`
  (`161002`) or on the grouped twin. Owner: the quantization method's `apply()`,
  not the package — and **one repaired site is not the answer**, because the linear
  and MoE methods (and their `_list` variants) repeat the same contract violation;
  the sweep and the output-dtype half of the gate are in
  `quant-scale-dtype-gate.md`.
  (d) **shape**: `Not support mat2 n = 1 or k = 1 when format is FRACTAL_NZ, mat2: [2048, 1]` —
  the same `aclnnMatmulWeightNz` / `161002` wrapper but a **shape** refusal: the
  weight is ND-legal and NZ-illegal, so any linear with a 1-length dimension (a MoE
  shared-expert gate `ReplicatedLinear(hidden, 1)`, a bias-like projection) kills the
  load whenever the NZ gate converts it. Still not an operator or install fault — the
  owner is the gate's missing shape test, and the lever is the NZ mode
  (`../startup-failures/nz-weight-format-refusal.md`). Read the text after `EZ1001`
  before choosing between (a)-(f): all six arrive in the same wrapper.
  (e) **output dtype**: the same `161002` but the named tensor is **`y`** —
  `GMM:y dtype does not match with required dtype[DT_BFLOAT16]` beside
  `Tensor … dtype is DT_BFLOAT16 but found DT_INT32` on `aclnnGroupedMatmulV5`. The
  required dtype then comes from the *activations*, so the two owners are the launch's
  `output_dtype=torch.int32 if use_int8_w8a8` and the MoE permute that was supposed to
  quantize them — check both before any kernel/driver theory; rule 5 of
  `quant-scale-dtype-gate.md` (v0.10.1rc1 anchors, `ep_size < 16` forcing the
  non-quantizing permute, the `--enforce-eager` control and the e2e coverage gap).
  (f) **quant-parameter dtype, MX variant — and this one is not the caller's fault**:
  the same `161002` wrapper with `AclNN_Parameter_Error(EZ1001): Tensor x1Scale not
  implemented for DT_FLOAT8_E8M0, should be in dtype support list [DT_FLOAT,]` on
  `npu_transpose_quant_batchmatmul` (also a `DT_FLOAT4_E2M1FN` operand on an A5 fused
  op). The **length of the printed list decides the side of the gate**: a multi-entry
  list that omits only the model dtype is (c), while a **one-entry** list means this
  kernel build carries the non-MX scale variant only, so the caller (an A5 o_proj /
  a W8A8-MXFP8 method handing over `…view(torch.float8_e8m0fnu)`) is doing what an MX
  checkpoint requires and a cast to fp32 is an experiment, not the fix. Owner:
  installed kernel × SoC die — see rule 0 of `quant-scale-dtype-gate.md` and
  `../soc-platform-paths/a5-mxfp-refusal-and-die-coverage.md`.
  **A tiling line whose verb is `not support <mode>`, not `… is invalid` and not `must be
  equal`, is a toolkit-release question.** `When layout is TND, not support Page Attention`
  (`RunBigKernelTilingWithParams`, `prompt_flash_attention_tiling.cpp`) says the installed
  CANN's prompt tiling has no path for that layout × paged-KV pair, so the pin's documented
  CANN floor versus the reporter's `version.cfg` is the first check and the caller's arguments
  are not — `tiling-capability-refusal-cann-release.md`.
  A **built-in** op dying the same way before launch (`call aclnnArange failed …
  EZ9999 … Parse dynamic kernel config fail`, `AclOpKernelInit failed opType`,
  `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed`) is not a custom op at all: the
  image's CANN kernel package does not cover the running SoC —
  `../soc-platform-paths/kernel-package-vs-card.md`. The `EZ9999 [PID: …] Op <Op>
  does not has any binary` wording (beside `call aclnn<Op> failed, error code is
  561000` and `ERR00100 PTA call acl api failed`) is the same drawer and takes the
  same answer, on 310P most often for a torch_npu *built-in* (`aclnnIsNegInf` on a
  VL/vision path, recorded #6082) — where the Python caller is in no pinned tree, so
  one `rg` is the whole budget before the standalone op probe on that page.
  **Seven pre-launch wordings, seven owners — and the one with an `integral key`
  is the dtype axis.** `EZ1001 … does not support opType` = no registry entry for
  the SoC (`op-not-registered-for-soc.md`); `EZ9999 … does not has any binary` /
  `Parse dynamic kernel config fail` = no kernel for the op at all on this SoC
  (`../soc-platform-paths/kernel-package-vs-card.md`); **`EZ9999 … Cannot find bin
  of op <Op>, integral key <dtype list>` + `Kernel Run failed. opType: N`** = the
  op is registered and no binary matches that dtype/format key, so the **caller's
  dtypes** (the model dtype the engine init line prints, `dtype=torch.float16`)
  are the first lever and `--dtype bfloat16` is the one-command test —
  `cannot-find-bin-integral-key.md`, recorded #2387; and a bare **`setup failed!`**
  (no op name, no code) is the caller's own mask/size argument, read at the deepest
  `vllm_ascend` frame — `setup-failed-call-site.md` (recorded #393; the `<Op>Operation
  setup failed!` forms route to their own pages from there, and the `tensor dimNum 0
  is invalid` wording inside it is an **uninitialised operand**, not a wrong shape —
  `atb-input-dim0-none.md`); and **`aclnn<Op> or
  aclnn<Op>GetWorkspaceSize not in libopapi.so`** is the *dlsym* root: the symbol was
  not in the vendored `libcust_opapi.so` nor in the toolkit's `libopapi.so`, so the
  container's custom-op package is the first suspect and the CANN release the last —
  `opapi-symbol-absent.md`; and **`EZ1001` wrapped round `Get path and read
  binary_info_config.json failed, please check if the opp_kernel package is installed!`
  with `Check NnopbaseCollecterWork(binCollecter.get()) failed` → `Assert ((NnopbaseInit())
  == 0) failed` → `Check NnopbaseCreateExecutorSpace(&executorSpace) failed`** is the
  **kernel manifest never found**: opbase built no collector, so nothing could be looked
  up at all. Its checks are environmental and come first — `echo $ASCEND_OPP_PATH`, the
  `…/opp/built-in/op_impl/ai_core/tbe/kernel/config/<soc>/binary_info_config.json` path
  *inside that container*, whether `set_env.sh` reached the process, and the pin's own
  separate `Ascend-cann-kernels-<soc>` install — then the image's CANN against the driver
  generation `npu-smi` prints. One phrase separates it from the registry wording above: **`Get path and read …
  failed` = the file was never found (install/env); `Get regInfo failed … does not support
  opType` = the file was read and has no entry for this SoC (build/tag).** Recorded #573
  (v0.7.3rc2, `aclnnSwiGlu` at `profile_run`) opened the registry page first —
  `../soc-platform-paths/kernel-package-vs-card.md` §*Third wording*; and **`EZ1001`
  wrapped round
  `Check NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize) failed` /
  `Assert ((executor) != nullptr) failed`** is the *opbase launch* path, not a registry
  statement, so the registry page's remedy does not apply. Read the `Check …` lines
  **between** the `EZ1001` and the assert: `NnopbaseExecutorPrepareParamsExt` →
  `NnopbaseExecutorGetMc2Num` → `NnopbaseGetHcomResource` →
  `HcclAllocComResourceByTiling … ret = 4` is the MC2/HCCL-resource lane, recorded twice
  in the corpus (#5680 single-node and #5993 two-machine, both A2 DeepSeek
  `MoeDistributeDispatch*`) and owned by
  the link/comm pages (`../hccl-and-hardware/link-establish-timeout.md`), never by the
  op's SoC support. A paste that elides it as `EZ1001: … Assert` has dropped exactly that
  discriminator, so ask for the block before classifying — recorded #6488 (310P MoE, 21
  turns, no answer) read the `EZ1001` registry page first and then hunted the op's Python
  caller across vllm-ascend, torch_npu and op-plugin for the rest of the session. None of
  the seven got a kernel to run in the recorded cases, so `561000` / `ERR00100` / `EZ9999` /
  a bare `EZ1001` is a wrapper and a grep on it alone lands in the MC2/HCCL rows.
  The **create-time** wording is not in that list: `RuntimeError: <Op>Operation
  CreateOperation failed!` out of a `torch_npu._npu_*` call means ATB refused to *build* the
  op (no tensor described yet), so the owner is the installed ATB/NNAL build × SoC and the
  caller's shapes are irrelevant — `atb-create-operation-failed.md` (#5170, 310P `_warm_up_atb`).
- **A garbled MoE output or an AI-Core invalid-GM-address under
  `enable_fused_mc2=1` is a workspace/stride question inside one of three op variants,
  and the variant is the lever.** `csrc/mc2/dispatch_ffn_combine{,_bf16,_w4_a8}/` share a
  kernel body but differ in how they size the aliased GMM1 workspace: W8A8 and W4A8 alias
  `ptrC`/`ptrC2` and `ptrA`/`ptrPermutedToken` into two regions, so each must be the
  **wider** of its two users (`max(N, K)`, `max(K, N/2)` where `N = 2 ×
  moe_intermediate`), and the GMM1 row stride must be `N`, not the hidden `K`. Two
  discriminators isolate it without a device: flip only `enable_fused_mc2`, then keep the
  flag and dequantize only the experts to bf16 — a return to coherence names the W8A8
  kernel and clears the MC2 dispatch/Combine orchestration. Never accept "the nightly op
  test passes" as coverage until you have read its shapes and scales
  (`mc2-variant-workspace-and-strides.md`).
- **A report that already names the file, the line and the fix is a claim-check, not a
  hunt.** Verify each `path:line` at the `SOURCES.json` pin, say which are off, quote the
  sibling variant the reporter used as reference precisely, name the sites where the
  current expression is **correct** so the patch is not swept too wide, and keep
  reporter-measured numbers (relative error, tok/s, eval parity) labelled
  reporter-confirmed. The checklist is in
  `mc2-variant-workspace-and-strides.md` §"Reviewing a self-diagnosed defect report".

- **A built-in `aclnn<Op>` *execution* error has no vllm-ascend owner, and the host
  driver outranks every tiling theory.** `aclnnRmsNorm` / `aclnnAddRmsNorm` /
  `aclnnLayerNorm` and friends are CANN library ops reached through `torch_npu`
  (`vllm_ascend/ops/layernorm.py` calls `torch_npu.npu_rms_norm` /
  `npu_add_rms_norm`); `ls csrc` at the reporter's ref is the first cut, but it is too
  coarse to *disown* an op: the registrations are the `ops.def("<name>", …)` lines in
  `csrc/torch_binding.cpp` and each fused op's `csrc/<dir>/op_host/aclnn_<name>.h`, so
  match the printed op name against those before calling it CANN's built-in — at
  v0.12.0rc1 there is no `csrc/rms_norm` yet `csrc/matmul_allreduce_add_rmsnorm/` does
  exist (an MC2 fused op reached as `aclnnMatmulAllReduceAddRmsNorm`, recorded #5236's
  second run), while `RmsNorm*` text inside `csrc/mla_preprocess/op_kernel/*.hpp` is MLA
  kernel internals, not a registration (the `add_rms_norm_bias` map in this topic —
  verified at v0.19.1rc1 — stays unusable for that report). When the sentence is
  `aclnn<Op> operator execution error` with no
  error code, no dump and no `csrc/` op at the pin, rank the **host driver / CANN
  pairing** first: `npu-smi info` header version, `cat
  /usr/local/Ascend/driver/version.info`, the image's CANN base — reporter-confirmed
  on #5236 ("solved by updating the npu driver to 25.25.0", Qwen3-235B,
  v0.12.0rc1). Second is the async-attribution rerun (`--enforce-eager` +
  `ASCEND_LAUNCH_BLOCKING=1`, plog) because the named op is then a sync point
  (`../hccl-and-hardware/collective-error-chain.md`). Ask for the error text before
  mapping any caller: with only the op name in the title, the version triple and the
  first error line are the whole next step.
- **"This op is not a CANN op, why is the framework dispatching it?" is an
  ownership question, and the usual answer is that the op is vllm-ascend's own.**
  A CamelCase `…Custom` op built from `csrc/build_aclnn.sh`'s `CUSTOM_OPS_ARRAY`
  installs into `vllm_ascend/_cann_ops_custom` and is mounted through
  `ASCEND_CUSTOM_OPP_PATH`, so it is absent from CANN's inventory by design. Name the
  dispatching call site and stop; treat the `EZ1001 Get regInfo failed` text as the
  separate install lane. For a sampling op the dispatch site is one of three known
  call sites, chosen by whether MTP/spec-decode is on —
  `custom-op-ownership-and-dispatch.md`.
- **A numeric aclnn code is an argument, not an owner.** `call aclnnX failed, error
  code is 561103` states that this launch's tiling rejected one input; the same
  561103 appears on `aclnnFlashAttentionScore` (#115), `aclnnInplaceNormal` (#2892)
  and `aclnnIndexPutImpl` (#4430), so it must never be graded as hardware, HCCL or
  "this parallelism is unsupported". On #115 the rejected input was the **rank of a
  caller-built attention mask** (`atten mask dim should be 2 or 4, but got 3` out of the
  vision tower, `tiling-attribute-rejection.md` §*Mask rank*), which no card, TP degree or
  mode flag can change. Three shapes recur on MoE deployments: a
  **built-in routing op** fed by the token dispatcher (`aclnnMoeInitRoutingV3` /
  `MoeInitRoutingV2`, where the EZ9999 line names the offending argument —
  `moe-init-routing-tiling.md`), the **MC2 dispatch's own tiling** at
  `MoeDistributeDispatchA2 CheckAttr/CheckShapeAndSetTiling` (the EZ9999 line names
  the argument it refused — the per-card expert count or the token dim `x.shape[0]`;
  compute both from the launch line before any transport theory,
  `../feature-enable-recipes/moe-expert-parallel-with-dp.md` §Step 2b/2c), and the
  backport custom op the tree builds itself.
  The op name in the paste may not exist in the tree at all: the caller's
  `torch_npu` function, the aclnn entry point and the tiling opType can carry three
  different names in one block, so grep the *Python* caller before any `csrc/` walk.
  **And the printed code may not be this call's code at all.** PTA's
  `NPU_CHECK_ERROR` replaces the failing expression's return with
  `AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)` — peeked, never cleared — when the thread
  holds a latched runtime error, so `call aclnnQuantMatmulV5 failed, error code is 107030`
  on a serve that has been decoding for hours is a **capture-window** code with this call's
  static text beside it, and the op's real status is masked. Judge the number by whether
  this op could mint it before ranking anything on it, and answer with the plog at the
  failure second (`../hccl-and-hardware/latched-runtime-error-masking.md`).
- **`Failed to register custom ops, all custom ops will be disabled` is not a defect
  in every image.** At vllm-ascend v0.7.3 the official image ships without the
  extension by design: `setup.py:296-298` builds `vllm_ascend.vllm_ascend_C` only
  when `COMPILE_CUSTOM_KERNELS` is set (Dockerfile never sets it) and
  `vllm_ascend/platform.py:29-38` warns for any `ImportError` whose text is not the
  `PyInit_vllm_ascend_C` one. Read the build gate before calling it an incomplete
  install; by v0.23.0rc1 the string is gone and the import is lazy
  (`vllm_ascend/utils.py:435`).
- **An op that faults only inside the engine is a call-site question before it is an
  address question.** When the reporter's own control is "I dumped the inputs to a pkl
  and the op runs standalone", audit the control: one `<op>` family has several call
  sites in this tree with different keyword lists, and the fault kernel's own name can
  rule one out. `paged_attention_mask_1_mix_aic` cannot come from
  `torch_npu._npu_paged_attention`, which has **no mask operand** at v0.7.3 — only
  `_npu_paged_attention_splitfuse` (`attention.py:867`, `attention_v1.py:261`) passes
  `mask=` — so a replay of the decode call proves nothing, and the batch regime that
  reaches the mask branch (mixed prefill+decode, prefix-cache hit) is the discriminator.
  The mask is host-built (`attention.py:614-623`: `cat(attn_mask[context_len:seq_len])`
  over a `max(prefill, decode)` square), the block table is padded to the longest
  sequence (`:597-602`), and both runners' `npu_format_cast(<cache>, 2)` calls
  **discard their return** (`worker/worker.py:315-316`, `worker/model_runner_v1.py:850`)
  — do not build a format mechanism on them. The op body is ATB/op-plugin and in no
  export: one `rg` per tree, then the ladder
  (`engine-only-op-replay.md`, recorded #3263).
- The dump names the operator and the error class; it does **not** identify the shape. Never
  assert "fails for hidden size N" without a shape-bearing artefact (reporter log line,
  tiling log, or a computed tiling decision).
  The same holds for the pre-launch cousin: `EZ9999 … <tensor> shape is invalid` /
  `cos or sin batchSize must same as x batchSize` carrying
  `[FUNC:GetShapeAttrsInfo][FILE:<op>_tiling.cpp]` plus `<Op> do tiling failed, ret is -1`
  is an **attribute rejection of the caller's arguments** (nothing launched), intermittent
  only because a rarer batch regime changes the rows — `tiling-attribute-rejection.md`.
- **No error text at all is a third family.** A correctly shaped, silently wrong
  result (uniform-average output, softmax apparently dropped, error only against
  an fp32 reference) is an **operator numerics** report, not a device fault: the
  evidence is the caller's kwargs combination, the tree's other call sites, the
  caller's own mocked test, and whether the op source is even in the snapshot —
  `attention-op-numeric-errors.md`. Do not hunt a tiling crash for it, and do not
  accept "the op is broken" without splitting ownership between the op owner
  (`cann/ops-nn`, often `unavailable`) and the framework that chose the
  combination and ships no fallback.
- **Timestamp-gate the block before adopting its text.** A device/runtime error block
  inside a fresh traceback keeps its own original date, so an earlier date can mean
  a re-printed record rather than this incident (#9183: a `107030` capture-mode
  block from the previous day) — **or** it can mean this process's own init/capture
  window, reported tens of minutes late (#10700: record `10:01:23` inside engine
  init, report `10:28:19`). Distinguish by process and log position, not by the gap.
  Run `../hccl-and-hardware/error-report-timestamps.md` (script
  `plog-timeline.sh --since <date>`) before naming the operator or the mechanism.
- **A scatter/index write is judged by its index tensor, not by the op.** When the
  fault names `ScatterNdUpdate`-style kernels and `../hccl-and-hardware/error-report-timestamps.md`
  shows one `pc start` across every core, the owner is the Python caller that passed the
  index tensor (row count and value range), not the kernel's tiling: in #4296 the DSA
  indexer wrote its cache with the whole batch's `slot_mapping` while holding one
  phase's rows, and the crash tracked a mixed prefill+decode batch — the answer is an
  A/B on request concurrency, no plog needed
  (`../dsa-sparse-attention/indexer-cache-scatter-fault.md`).
  The gather twin does not write, it **asserts**: `gather_v3_base.h:137
  Assertion (0 <= val && val < this->gxSize_) Index N out of range[0 M)` names the
  gather's source table in `M` and the offending element in `N`, so match `M` to a
  table row count you can compute (RoPE `cos_sin_cache` = the checkpoint's
  `max_position_embeddings`, the KV block table = `num_gpu_blocks`, a token
  buffer, the vocab) before any hardware theory — `gather-index-out-of-range.md`.
- **A custom op named inside a GE op-kernel-info init record is not the failing
  op.** `E29999: [InitOpInfoLib][InitOpKernel] Inconsistent format size [1] and data
  type size [1]!` → `Model<Op>: failed to initialize unknown shape dtype and format.
  The input or output is input2.blockIDs` → `[GraphOptLib] … [FILE:sub_op_store.cc]`
  is GE building kernel info for the whole custom-op package at process start: its
  own stamp precedes the fault, the strings live in no export, and the op may have a
  single in-tree caller the launch never reaches (at v0.17.0rc1
  `transpose_kv_cache_by_block` is called only from
  `distributed/kv_transfer/kv_p2p/mooncake_connector.py:599`). One `rg -n '<op_name>'
  vllm_ascend/ -g '*.py'` ends it — `../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`
  §2.
- **`NonZero` is the mask, not the `nonzero` call.** A paste naming `NonZero`
  (`Kernel Run failed. opType: 59, NonZero`, `launch failed for NonZero, errno:507053`)
  with `The DDR address of the MTE instruction is out of range` / `subErrType:4` /
  `retCode=0x31` is a built-in kernel launched by a **boolean-mask advanced index**
  (`x[~mask] = 0`), so `rg 'nonzero'` returns host numpy and EPLB noise while
  `rg '\[~' vllm_ascend/` returns three sites at v0.17.0rc1. On a hybrid
  linear-attention model (Qwen3-Next, Qwen3.5-A3B) the owner is the GDN/KDA SSM-state
  clear, it runs on **prefill steps only** (so the fault arrives hours in and tracks
  request arrivals), `507053`/`507899` are the sync report and the UCE query rather
  than causes, and the version lane is the answer — reporter-confirmed on 0.19.1rc1
  (#7774) and again at a v0.11-era pin, where the same clear sits in
  `vllm_ascend/models/qwen3_next.py:331` and the reporter closed the generic
  `copy between host and device_opapi … 507053` at `profile_run` as fixed for
  `>= v0.13.0rc1` (#4483). The sites per pin, the start-phase variant, the rank
  check and the stop rules:
  `hybrid-ssm-mask-nonzero-fault.md`.
- **A built-in KV store that faults (`fault kernel_name=reshape_and_cache_…`, MTE GM
  address / `The DDR address of the MTE instruction is out of range`) on a hybrid
  GDN/Mamba model is not a KV-group contamination question.** On Qwen3-Next at
  v0.10.2rc1 the per-group metadata loop writes one shared `slot_mapping_cpu` buffer
  (`worker/model_runner_v1.py:1329-1360`, with a `# TODO: change this to the right block
  table for linear attn`), but each group's `build()` copies it to the device inside its
  own iteration (`attention/attention_v1.py:234-238`, `.to(self.device, non_blocking=True)`),
  so a later group's values cannot reach an earlier group's device copy — settle that
  with one `sed` instead of deriving it (recorded #2976 spent five turns on the
  derivation and never shipped an answer). Two facts worth carrying: the mamba group is
  rewritten to `block_size=max_model_len` with one block per sequence
  (`model_runner_v1.py:2838-2846`) and prefix caching is refused for mamba on that path,
  so `num_gpu_blocks` cannot be recovered by dividing the logged `GPU KV cache size`
  token counts (the four lines with one differing value are per-rank budgets) — do the
  arithmetic for a per-group block size and it will not close. Grade the release
  contract first: v0.10.2rc1 is the *first* release with Qwen3-Next support and states
  EP and MTP do not work (`docs/source/user_guide/release_notes.md:9`), and the
  reporter's serve line is the tutorial's own
  (`docs/source/tutorials/multi_npu_qwen3_next.md:89`), so the command is not the
  variable.
- **A pre-launch `TORCH_CHECK` from a host adapter is a table-slot bound keyed on a
  tensor dimension, and the number it prints is the op's own constant.** `RuntimeError:
  batchIdx is out of range: 1024` out of `torch.ops._C_ascend.batch_matmul_transpose`
  (#8392) is `csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.h` refusing
  `batchIdx = tensorAShape[0] - 1` against `constexpr MAX_CAPTURE_NUM = 1024` — the **token**
  dim, not `max_cudagraph_capture_size` and not any request-count knob. Reconcile the
  printed number with the reporter's named parameter first (their 128 vs the printed
  1024 refutes their mechanism in one line), then read the caller's guard
  **in the reporter's release, not only in the pin** — absent at v0.13.0rc1, where it is a
  `not has_prefill` proxy the tree's own TODO calls a bug — and stop: nothing launches, so
  `cudagraph_mode` / drafter-eager / capture-size ablations cannot move it —
  `host-tiling-table-bound.md`.
- Check the error family before the operator. When the same output also carries
  `SUSPECT REMOTE ERROR` / `error code is 507057` / an `EZ9999 … chipId:` dump /
  `error code = 0x8…`, it is a lane C device fault: the `fftsplus aivector error` there
  names the chip's failure class, not a tiling defect in the named op (recorded #10829:
  the same output carried an `EZ9999 … (chipId:4, dieId:0) … fftsplus aivector error`
  dump while one node printed only `SUSPECT REMOTE ERROR`). Route to
  `../hccl-and-hardware/index.md` and do not compute tiling for it.
- **A custom op that faults only in varlen mode is a null-optional read before it is a
  geometry bug.** When `507015` / `The MPU address access is invalid` names an FLA/GDN
  op and the kwargs carry `cu_seqlens`, check whether the optional chunk-index buffer
  the kernel dereferences was passed at all (Python `None` → binding optional →
  `nullptr`), and compare against the sibling op that ran on the same null argument —
  the one that survived does not read that buffer. Aligning the sequence lengths does
  not remove a null read: `fla-chunk-varlen-null-chunk-offsets.md`.
- Map the op name to its source tree first (`csrc/<op>/`), then to the tiling mode table in
  `op_host/<op>_tiling.cpp`: the same kernel entry runs several very different kernels
  (tiling key dispatch), so "the kernel" is ambiguous until you know the mode.
- Compute the tiling decision from the reported/prefill-decode shape with a short python
  snippet; it is far cheaper than reading every kernel and it tells you which kernel file
  is even relevant.
- Distinguish the three failure surfaces before blaming the kernel:
  (a) GM/DDR addressing (offset or element count outside the allocation),
  (b) UB sizing (buffer partitions exceeding the allocated `TBuf`),
  (c) tiling arithmetic (zero/underflowed factors, `CeilDiv` edge cases, core-count
  mismatch with `blockDim`). They produce different symptoms; do not swap them.
- "Ran for a while, then failed" points at a *regime switch* (new batch/token shape or
  dtype selecting a different tiling mode), not at slow corruption. Ask for the request
  shape and the tiling log line at fault time.
- Custom ops are often gated by a runtime switch (e.g. `enable_custom_op()`), with a
  `torch_npu` fallback. Verify the failing op was actually enabled in the reporter's run;
  otherwise the dump came from another path.
- Ask for the artefacts the dump itself requests (plog at fault time), plus the tiling I-line
  (`Tiling Key` / `Block Dim` / `num_row` / `num_col` / `ub_factor`), dtype, token/latency
  shape at failure, and whether `--enforce-eager` changes it. That is a complete, cheap ask.

## Pages

- `mc2-variant-workspace-and-strides.md` — the fused-MC2 workspace family: the three
  `dispatch_ffn_combine` variants, the aliasing rule and why it is `max()` not a sum, the
  verified pre-fix sites at v0.26.0rc1 `f2f74a16c` (`_tiling.cpp:297-304`,
  `_kernel.hpp:473,522,927-931,946-950,1129,1133`, the epilogue row stride at
  `utils/block_epilogue_pertoken_swiglu.hpp:189`, W4A8's already-`max()` host tiling and
  the maintainer comment that states the rule), the `N > K` / `N > 2K` thresholds, the
  sites where `k()` is correct (the remote peer A buffer), the discriminators, the
  degenerate op test (`n = k = 1024`, `randint(0, 1)` scales) and the single A3 config
  that enables the feature, plus the review checklist for a self-diagnosed report
  (recorded #16561).
- `custom-ops-disabled-by-device-type.md` — `AttributeError: '_OpNamespace'
  '_C_ascend' object has no attribute '<op>'` where **every** op is absent because
  the gate, not the build, says so: `enable_custom_op()` returns `False` for
  `AscendDeviceType.A5` and for `VLLM_BATCH_INVARIANT` (`utils.py:293-295`, FIXME →
  vllm-ascend#7157) **before** importing `vllm_ascend_C`, so no
  `Failed to register custom ops` warning is printed and the lazy import
  (`platform.py:575-596`, `sfa_v1.py`'s eight ungated `torch.ops._C_ascend` calls)
  reaches an empty namespace. Carries the three-gate capability verdict (device-type
  gate → `build_aclnn.sh` branch → `<op>_def.cpp` `AddConfig`) recorded on #10074's
  "is SFA ready on a5" question at v0.19.1rc1.
- `opapi-symbol-absent.md` — `aclnn<Op> or aclnn<Op>GetWorkspaceSize not in
  libopapi.so` from a `torch.ops._C_ascend.<op>` call (recorded #16379: DeepSeek-V4-Flash
  `hc_pre` on A3 at `profile_run`): the `EXEC_NPU_CMD` three-root dlsym search, the
  in-tree `hc_pre_def.cpp`/`build_aclnn.sh` evidence that A3 **is** a build target (so a
  vendor table listing the op A5-only is CANN's fallback copy), the two texts that split
  "extension absent" from "vendor package without the op", and the checks that replace a
  CANN `version.cfg` diff. Two rules added from #7095 (`aclnnAddRmsNormBias` at
  `profile_run` on A3, pip-installed): the three-root search is the **v0.2x**
  implementation — at v0.15.0rc1 the same macro `dlopen`s `libcust_opapi.so` /
  `libopapi.so` by bare soname through the RPATH baked in at `CMakeLists.txt:141`,
  so the vendor `op_api/lib` directory must physically exist in the installed
  package; and the release **wheel is built for one SoC only**
  (`Dockerfile.buildwheel`: `ARG SOC_VERSION="ascend910b1"`, `bdist_wheel` runs
  `build_aclnn`), so an A3/910_93 host with `pip install vllm-ascend` ranks the
  SoC mismatch first and rung 1 is the `…-a3` image or
  `SOC_VERSION=ascend910_9391` from source.
- `engine-only-op-replay.md` — the "works standalone, fails in the engine" control
  (recorded #3263, v0.7.3 `paged_attention_mask_1_mix_aic` + MTE DDR out of range,
  `runtime result 507053`): the five `_npu_paged_attention*` call sites and the argument
  that separates them, the host-built splitfuse mask arithmetic
  (`attention.py:595,614-623`), the padded block table (`:597-602`), the discarded
  `npu_format_cast` return at both runners, the closed ATB/op-plugin boundary, the
  ladder (single request → `--no-enable-prefix-caching` → `--enforce-eager` →
  `ASCEND_LAUNCH_BLOCKING=1`), the asks, and why that run's 46 messages produced no
  answer.
- `triton-jit-kernel-device-fault.md` — the plog names a downstream `@triton.jit`
  kernel (`fault kernel_name=token_bin_counts_and_mask_kernel_0` with `aivec error`,
  `The DDR address of the MTE instruction is out of range`, `isSmmuFault is 1`,
  `507035`): the name→file resolution at v0.20.2rc1 including which worker generation
  each sampler kernel name proves (`patch/worker/__init__.py:22` skips the v2 patches
  on exactly vllm 0.20.2), what the `[AIC_INFO] args(0 to 9)` blob does **not** pin,
  the two decoys (the `LlmDatadistTimer` teardown segfault, the upstream tensor
  construction chain), the four-rung ablation (penalties at defaults first), the
  recorded #10522 case with its reporter-confirmed post-pin fix, and the three
  skipped nightly tests that make "the op test passes" unavailable here.
- `triton-ub-overflow-tiling-budget.md` — `error: ub overflow, requires N bits while
  … available` under `Failed to run BiShengHIR pipeline` (recorded #18354 at v0.20.2rc1,
  custom model, 9.9× over the 1572864-bit budget; precedent #8037 at v0.17.0rc1, 1.17×):
  the two numbers are the finding, the `available` half is a per-SoC constant, the tiled
  `_triton_rope` versus un-tiled `_triton_rope_siso` split is read from the tile constants
  and the traceback's line number, the tree's own validated shape envelope is the Triton
  test's parameter list, and the levers are the tile constant or the `torch_npu` path —
  with no env switch that forces it at v0.20.2 (`HAS_TRITON` is driver detection only).
- `moe-init-routing-tiling.md` — `aclnnMoeInitRoutingV3 … 561103` (and the
  `MoeInitRoutingV2` tiling wording): the decoded EZ9999 `CheckTokenCount` check,
  which caller parameters the tiling compares, the dispatcher arithmetic that
  decides they are reached at all, the CANN 8.3-vs-8.5 operator-availability axis,
  and why the "please validate DP/TP at start-up" request resolves to three
  specific gates rather than a layout matrix.
- `fla-chunk-varlen-null-chunk-offsets.md` — `507015` (`MPU address access is
  invalid`) from `chunk_fwd_o` in the chunked GDN prefill at v0.20.2rc1: the two
  candidate mechanisms and the sibling-op decider, the null `chunk_offsets` chain
  (`chunk.py:95` → binding → `nullptr` → scheduler), the Python-gate gap, the fix
  ladder with the `prepare_chunk_indices` route, and why a shape-only nightly op test
  reports PASSED while the kernel faults (plus the `[N,H,V,K]` vs `[N,H,K,V]` oracle trap).
- `tiling-attribute-rejection.md` — `<tensor> shape is invalid` /
  `… batchSize must same as x batchSize` out of `<op>_tiling.cpp` `GetShapeAttrsInfo` with
  `do tiling failed, ret is -1` and `error code is 561002`: what the line proves (an argument
  check, not a split bug), the three recorded wordings on `npu_kv_rmsnorm_rope_cache` and
  `npu_interleave_rope`, the caller's tensor construction sites at v0.13.0rc2
  (`sfa_v1.py:511-553`, `mla_v1.py:1098/1126`, `rotary_embedding.py:107-119`), the four
  regimes that change the rows at the call (MLAPO's 1024-token branch, graph buckets,
  DSA-CP slicing, SP gather), the five-line shape-dump spy, the framework-free `torch_npu`
  sweep, and how to answer when the op owner returns the case as a "调度问题".
- `fa-tiling-argument-refusals.md` — the built-in **`FlashAttentionScore`** tiling family, where
  the `FUNC:`/`FILE:` pair in the paste (`flash_attention_score_tiling_general.cpp`) names which
  argument was refused: `AnalyzeOptionalInput` → `get unsupported atten_mask shape` for a mask
  broadcast on the query axis (the accepted list has `Sq` in slot 3, so `[B,1,1,Skv]` is never
  legal — the model-side `get_extended_attention_mask` of an audio/VL tower, #6573), and
  `Analyze3DimLayout` → `The end of actualSeqQLen & actualSeqKvLen should be 0 in EOD scenario`
  for the TND tail invariant, where the pooling/encoder FA call is the one site that never trims
  `query` to `actual_seq_lengths_q[-1]` while its FIA sibling does (#10757).
- `tiling-capability-refusal-cann-release.md` — the **capability** sibling, where the verb is
  `not support`: `call aclnnFusedInferAttentionScoreV3 failed, error code is 561002` +
  `When layout is TND, not support Page Attention[FUNC:RunBigKernelTilingWithParams][FILE:prompt_flash_attention_tiling.cpp]`.
  The refused object is a mode × cache-kind combination, so the owner is the installed toolkit's
  prompt-FA tiling: compare the reporter's `version.cfg` with the pin's own floor
  (`installation.md:15`, `CANN >= 8.3.RC1` at v0.11.0rc1) before any caller walk; #4373 is
  reporter-confirmed solved by CANN 8.3, and the launch pair (`attention_v1.py:540-554`, TND +
  `block_table` + `sparse_mode=3`) is what the plugin intends, so chunking avoidance is only a sidestep.
- `aclnn-error-dump-reading.md` — field-by-field reading of the Ascend device dump and what
  each line does/does not pin down; its step 0 is the check that an op-map page at the same
  pin already answers the tiling question, which the #8944 forward skipped before
  re-deriving that page from source; plus the **built-in CANN kernel** lane where there is no
  `csrc/<op>/` to open (`fault kernel_name=unpad_flashattention_…`, `FlashAttentionScore…`):
  the V1 `Dumping scheduler output … SchedulerOutput(...)` disaster dump read field by field
  (`scheduled_encoder_inputs` ⇒ vision tower vs LM, `num_computed_tokens > 0` ⇒ continuation
  chunk, `num_common_prefix_blocks` ⇒ *not* a cache hit), and recorded #3810 (v0.11.0rc0,
  Qwen3-VL-8B image request, fixed by the v0.13.0rc0 image). Its **sampler-mask**
  sub-case (`fault kernel_name=ApplyTopKTopPWithSorted_fp32_high_performance_0`, 偶现
  `507035` while the serve replays a captured graph) carries the two rungs that precede
  any op-defect theory — `--enforce-eager` (reporter-confirmed on #2763 at ~5 vs ~40
  tok/s) and `--compilation-config '{"cudagraph_mode": 0}'`.
- `cannot-find-bin-integral-key.md` — `EZ9999 … Cannot find bin of op <Op>,
  integral key <dtype/format list>` + `Kernel Run failed. opType: N` (#2387,
  v0.10.0rc1): its pre-launch wording table and their owners, how to read the key
  against the engine's `dtype=` line, the fused-W8A8 caller chain that puts the
  model dtype into `aclnnAddRmsNormQuant` (`qwen3.py:47-57` → `layernorm.py:49-54` →
  `w8a8.py:60-64,125-135`), the bf16 re-serve plus the two-dtype `torch_npu` probe,
  and why `--enforce-eager` cannot rule this lane out.
- `setup-failed-call-site.md` — a bare `RuntimeError: setup failed!` (#393,
  v0.7.3rc1 V0 rollouts): the eight-row `setup failed!` surface table (it now carries
  `PagedAttentionOperation` on an eager V1 engine, where the dumped `SchedulerOutput`
  picks the branch and `chunked_prefill`/`prefix_caching` are V1 defaults — #2700), the
  #393 branch derivation that proves the engine set `enable_prefix_caching=True` with
  chunked prefill off,
  the hard-coded `128` compress mask at `attention.py:612-617` against the
  `query_lens`/`context_lens` the same branch can carry, and the two-flag ladder.
- `atb-input-dim0-none.md` — `SelfAttentionOperation setup failed!` + the ATB log's
  `tensor dimNum 0 is invalid … inTensor [1] CheckTensorShape failed`: the slot was
  never filled (Python `None`), not a size or mask limit. Recorded #2779
  (v0.10.1rc1, Qwen2.5-72B TP4, torchair graph + Ascend scheduler +
  `--enable-prefix-caching`): the torchair impl declares `self.key_cache`/`self.value_cache`
  and never binds them (`torchair_attention.py:305-306`) while the eager sibling binds
  them at `attention_v1.py:534-536`, so the `PrefillCacheHit` branch passes `None` into
  `_npu_flash_attention_qlens` — first prefix-cache hit after minutes of clean serving.
  The one-command probe (`scripts/init-none-attr-probe.py`), the `_build_attn_state`
  reachability and the ladder.
- `gather-index-out-of-range.md` — the `ascendc/gather_v3/gather_v3_base.h:137`
  `Index N out of range[0 M)` assert (v0.9.2rc1 #2349): the bound is a table row
  count, the four candidate tables, the verified `config.max_position_embeddings`
  → `cos_sin_cache` rows → `_npu_rotary_embedding(positions, …, cos_sin_cache)`
  chain that makes a `--max-model-len` above the checkpoint value an index
  overflow (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` only warns), the two-command
  decider, the fix ladder with the YaRN mscale caveat, and the `dfx/ops/*.json`
  artefact that names the table.
- `host-tiling-table-bound.md` — `<name> is out of range: N` from a `csrc/<op>/op_host/`
  adapter (#8392 `batchIdx` 1024 in `sfa_v1._v_up_proj`): the printed number is the
  table's `constexpr`, the index is a token count, the caller guard
  (`BMM_TRANS_MAX_SUPPORTED_TOKENS`) exists at v0.13.0 and v0.18.0rc1 but **not** at
  v0.13.0rc1 (`1b5d5ab`, `not has_prefill` + a TODO calling it a bug) — so grep the
  reporter's exact build instead of inferring either way — plus the token-vs-heads
  polarity that keeps `m` equal to the token count, the layout mismatch that survives the
  guard, the host-side `TORCH_CHECK` implication for graph-mode ablations, and the
  FlashComm1/`FULL_DECODE_ONLY` assert that explains why `piecewise` appears (and why the
  op is unreachable with SP on).
- `add-rms-norm-bias-op-map.md` — worked map of `csrc/add_rms_norm_bias` at vllm-ascend
  v0.19.1rc1: files, tiling key = dtype*10+mode, mode-selection table, buffer sizes, arch
  macros, python call sites, plus the cross-checks already closed at this pin (host rejects an
  `x2 != x1` row mismatch, NORMAL last-core arithmetic, the `rstd` `#if`) and the artefact
  order for an `aclnnAddRmsNormBias` MTE dump. Read it **before** `op_host/*_tiling.cpp` —
  at this pin it *is* the tiling read — and only once the error family is a device fault.
- `attention-op-numeric-errors.md` — silent numeric wrongness in
  `npu_fused_infer_attention_score` / `npu_fusion_attention` /
  `_npu_flash_attention_unpad` with a framework-free `torch_npu` repro: which caller
  holds the uncovered combination, the `npu_fusion_attention` counter-example already in
  the tree, the mocked-zeros unit test that hid it, the upstream sdpa/native semantics,
  the three fix sites (eager / capture+workspace / graph-replay rebind), and the cheap
  discriminators (mean-of-V check, causal-vs-all-False-mask row, axis sweeps) — recorded
  #13699 at v0.23.0rc1 `f4a08bdd` with `cann/ops-nn` absent.
- `../hccl-and-hardware/error-report-timestamps.md` — the ordering rule for a report
  that mixes runtime records with device dumps: which block is the incident, and
  when several chips share one `pc start` that means software, not several bad cards.
- `custom-op-ownership-and-dispatch.md` — the op owner returns the case ("算子侧反馈
  非cann算子，需要框架侧定位为什么会下发"): the `CUSTOM_OPS_ARRAY` →
  `vllm_ascend/_cann_ops_custom` → `ASCEND_CUSTOM_OPP_PATH` chain at v0.13.0rc3 that makes
  "not a CANN op" the design, the three top-k/top-p dispatch sites and the flag that picks
  one (`sample/sampler.py:64-78`, `sample/rejection_sampler.py:16,74` via the patch at
  `patch/worker/patch_rejection_sampler.py:1-12`, `worker/v2/sample/sampler.py:19,48`), the
  greedy/`top_p` trigger, the name-order hypothesis it refuses to grade, and the in-image
  reads that decide a release newer than the snapshot (recorded #8730, `0.14.0.RC1`).
- `quant-scale-dtype-gate.md` — `161002` / `EZ1001 … Tensor scale not implemented for
  DT_FLOAT16` on `aclnnQuantMatmulWeightNz` / `aclnnGroupedMatmulWeightNz` at
  vllm-ascend v0.17.0rc1: the scale is allocated in `params_dtype` so the gate is
  fp16-only (bf16 passes, which is why CI is blind), the float32 companion exists and
  is read nowhere on the linear path (`w8a8_dynamic.py:86-89` vs `:103`), the MoE w2
  side has **no** fp32 companion (`:249`) — reporter-confirmed as the identical
  failure right after the linear repair (#7443 → PR #7444) — plus the second gate
  operand (`moe_mlp.py:144` derives the output dtype from the scale) and the one-call
  sweep (`scripts/quant-scale-dtype-audit.py`).
- `../startup-failures/nz-weight-format-refusal.md` — the `EZ1001`/`161002` text this
  topic does **not** own: `Not support mat2 n = 1 or k = 1 when format is FRACTAL_NZ`
  is a shape refusal by an op that is registered, so the answer is the NZ mode
  (`weight_nz_mode` 0/1/2), not a package or SoC question (#10014, v0.20.2rc1).
- `atb-create-operation-failed.md` — `RuntimeError: LinearOperation CreateOperation failed!`
  (or `GroupTopkOperation …`) out of a `torch_npu._npu_*` call at start-up (#5170, v0.11.0,
  310P3): the create-vs-setup split, why the wording lives in no export (one grep, then stop),
  the prior corpus's resolutions for the family (#19 FAQ #8 = CANN `8.1.RC1.alpha001` + the next
  rc; #206; the link-time #117/#750), the unconditional `_warm_up_atb`
  (`worker_v1.py:332,337-341`) that makes the box unable to start at that ref, the dead levers
  (`--enforce-eager` when the paste already disabled cudagraph, `ATB_*` vars), the three checks
  (one-op repro, `nnal/atb` identity, `_build_info.__soc_version__`/`is_310p()`) and the
  `Dockerfile.310p:18,47-53` + `single_node_300i.md:4-9,21` rungs.
- `op-not-registered-for-soc.md` — the opposite class for the same op: `EZ1001 Get
  regInfo failed … does not support opType [AddRmsNormBias]` for `ascend910_93`
  (#11519, v0.18.0). Where `SOC_VERSION`, the per-SoC `build_aclnn.sh` branches,
  `vllm_ascend/_cann_ops_custom` + `ASCEND_CUSTOM_OPP_PATH`, and the `-a3` image tags
  decide which binaries exist, with the checks to send the reporter.
- `hybrid-ssm-mask-nonzero-fault.md` — `NonZero` (`opType: 59`) + MTE DDR out of range
  hours into a healthy hybrid-model serve (#7774, v0.17.0rc1 Qwen3.5-35B-A3B TP8 with
  `FULL_DECODE_ONLY` + prefix caching + `--async-scheduling`): the recorded field
  breakdown, the masked-assign sites at that pin (`patch_qwen3_5.py:180`,
  `patch_qwen3_next.py:239`, upstream `qwen3_next.py:776`/`kda.py:406`, mask from
  `gdn_attn.py:295`), the three other producers to rule out from the serve line
  (EPLB, PCP, host numpy), the prefill-only placement that explains the latency to
  fault, the coverage statement the pin forces (no Qwen3.5 row in the support matrix;
  e2e covers tp4 / `max_model_len 4096`), and the two stop rules — no MTE address
  arithmetic inside an unexported CANN kernel, no `ASCEND_LAUNCH_BLOCKING` when the
  kernel is already named.
- `gdn-conv1d-accepted-tokens-varlen.md` — `numAcceptedTokens[0]=4 exceeds varlen
  segment length=3` from `csrc/moe/causal_conv1d/op_host/causal_conv1d_tiling_validation.h:358-364`
  on a hybrid GDN + MTP launch (v0.20.2rc1 `367b8e62`): the `inputMode == 0` varlen
  contract (`batch = qslSize-1`, `nat.size == batch`, `nat[i] <= seg_i`,
  `qsl[last] == x.shape[0]`), the two feeders that disagree — segments from the
  **scheduler's** `query_start_loc_cpu` (`patch_gdn_attn.py:454-480`) vs the accepted
  count from the runner's **CPU/optimistic** buffer (`model_runner_v1.py:926-952`,
  whose device-side correction touches `num_computed_tokens` only) — the scheduler
  caps that shorten a segment (`vllm/v1/core/sched/scheduler.py:413-421`, the
  `--max-model-len` one being the usual suspect; the `num_output_placeholders` term at
  `:408-412`/`:526-530` **cancels**, so `--async-scheduling` alone does not shrink a
  segment), the capture-padding half
  (`gdn.py:89-130`, `q_per_seq`), the ablations (`--async-scheduling` off, then
  `--enforce-eager`), why a clamp is not the fix, and the recorded #10372 row whose
  follow-up reports the fed tuple as `(3,6,9)` — ask for the plog and both host tuples.
