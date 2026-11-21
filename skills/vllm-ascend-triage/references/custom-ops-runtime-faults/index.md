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
A Triton-Ascend kernel that fails to *compile* — `MLIRCompilationError`,
`[ConvertLinalgRToBinary] encounters error`, `bishengir-compile … returned non-zero
exit status 1` with `vllm_ascend/ops/triton/**` frames — is **not** this lane even when
the reporter words it as "failed to compile custom operators": nothing here touches
the AscendC extension, and the owner is the installed wheel/compiler pair
(`../image-dependency-faults/triton-ascend-pair.md`).

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
  before choosing between (a)-(d): all four arrive in the same wrapper.
  A **built-in** op dying the same way before launch (`call aclnnArange failed …
  EZ9999 … Parse dynamic kernel config fail`, `AclOpKernelInit failed opType`,
  `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed`) is not a custom op at all: the
  image's CANN kernel package does not cover the running SoC —
  `../soc-platform-paths/kernel-package-vs-card.md`.
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
  `npu_add_rms_norm`); one `ls csrc` at the reporter's ref says whether *any*
  vllm-ascend kernel with that name exists there (it does not at v0.12.0rc1, so the
  `add_rms_norm_bias` map in this topic — verified at v0.19.1rc1 — is unusable for
  that report). When the sentence is `aclnn<Op> operator execution error` with no
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
  "this parallelism is unsupported". Two shapes recur on MoE deployments: a
  **built-in routing op** fed by the token dispatcher (`aclnnMoeInitRoutingV3` /
  `MoeInitRoutingV2`, where the EZ9999 line names the offending argument —
  `moe-init-routing-tiling.md`), and the backport custom op the tree builds itself.
  The op name in the paste may not exist in the tree at all: the caller's
  `torch_npu` function, the aclnn entry point and the tiling opType can carry three
  different names in one block, so grep the *Python* caller before any `csrc/` walk.
- **`Failed to register custom ops, all custom ops will be disabled` is not a defect
  in every image.** At vllm-ascend v0.7.3 the official image ships without the
  extension by design: `setup.py:296-298` builds `vllm_ascend.vllm_ascend_C` only
  when `COMPILE_CUSTOM_KERNELS` is set (Dockerfile never sets it) and
  `vllm_ascend/platform.py:29-38` warns for any `ImportError` whose text is not the
  `PyInit_vllm_ascend_C` one. Read the build gate before calling it an incomplete
  install; by v0.23.0rc1 the string is gone and the import is lazy
  (`vllm_ascend/utils.py:435`).
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
- Check the error family before the operator. When the same output also carries
  `SUSPECT REMOTE ERROR` / `error code is 507057` / an `EZ9999 … chipId:` dump /
  `error code = 0x8…`, it is a lane C device fault: the `fftsplus aivector error` there
  names the chip's failure class, not a tiling defect in the named op (recorded #10829:
  the reporting rank printed `Device:0`, the dump named `chipId:4`). Route to
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
- `aclnn-error-dump-reading.md` — field-by-field reading of the Ascend device dump and what
  each line does/does not pin down.
- `gather-index-out-of-range.md` — the `ascendc/gather_v3/gather_v3_base.h:137`
  `Index N out of range[0 M)` assert (v0.9.2rc1 #2349): the bound is a table row
  count, the four candidate tables, the verified `config.max_position_embeddings`
  → `cos_sin_cache` rows → `_npu_rotary_embedding(positions, …, cos_sin_cache)`
  chain that makes a `--max-model-len` above the checkpoint value an index
  overflow (`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` only warns), the two-command
  decider, the fix ladder with the YaRN mscale caveat, and the `dfx/ops/*.json`
  artefact that names the table.
- `add-rms-norm-bias-op-map.md` — worked map of `csrc/add_rms_norm_bias` at vllm-ascend
  v0.19.1rc1: files, tiling key = dtype*10+mode, mode-selection table, buffer sizes, arch
  macros, python call sites. Read it only once the error family is a device fault.
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
- `op-not-registered-for-soc.md` — the opposite class for the same op: `EZ1001 Get
  regInfo failed … does not support opType [AddRmsNormBias]` for `ascend910_93`
  (#11519, v0.18.0). Where `SOC_VERSION`, the per-SoC `build_aclnn.sh` branches,
  `vllm_ascend/_cann_ops_custom` + `ASCEND_CUSTOM_OPP_PATH`, and the `-a3` image tags
  decide which binaries exist, with the checks to send the reporter.
