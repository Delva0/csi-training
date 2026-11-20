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
  matter. `AclNN_Parameter_Error(EZ1001)`, `Get regInfo failed`,
  `binary_info_config.json of socVersion [...] does not support opType [...]` means
  aclnn could not resolve the op for this SoC **before any launch**: the owner is the
  build/install chain, and no tiling fact will help — `op-not-registered-for-soc.md`.
  A **built-in** op dying the same way before launch (`call aclnnArange failed …
  EZ9999 … Parse dynamic kernel config fail`, `AclOpKernelInit failed opType`,
  `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed`) is not a custom op at all: the
  image's CANN kernel package does not cover the running SoC —
  `../soc-platform-paths/kernel-package-vs-card.md`.
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
  inside a fresh traceback keeps its own original date; if that date precedes the
  reported failure it is a re-printed record, not this incident (#9183: a `107030`
  capture-mode block from the previous day). Run
  `../hccl-and-hardware/error-report-timestamps.md` (script
  `plog-timeline.sh --since <date>`) before naming the operator or the mechanism.
- **A scatter/index write is judged by its index tensor, not by the op.** When the
  fault names `ScatterNdUpdate`-style kernels and `../hccl-and-hardware/error-report-timestamps.md`
  shows one `pc start` across every core, the owner is the Python caller that passed the
  index tensor (row count and value range), not the kernel's tiling: in #4296 the DSA
  indexer wrote its cache with the whole batch's `slot_mapping` while holding one
  phase's rows, and the crash tracked a mixed prefill+decode batch — the answer is an
  A/B on request concurrency, no plog needed
  (`../dsa-sparse-attention/indexer-cache-scatter-fault.md`).
- Check the error family before the operator. When the same output also carries
  `SUSPECT REMOTE ERROR` / `error code is 507057` / an `EZ9999 … chipId:` dump /
  `error code = 0x8…`, it is a lane C device fault: the `fftsplus aivector error` there
  names the chip's failure class, not a tiling defect in the named op (recorded #10829:
  the reporting rank printed `Device:0`, the dump named `chipId:4`). Route to
  `../hccl-and-hardware/index.md` and do not compute tiling for it.
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

- `aclnn-error-dump-reading.md` — field-by-field reading of the Ascend device dump and what
  each line does/does not pin down.
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
- `op-not-registered-for-soc.md` — the opposite class for the same op: `EZ1001 Get
  regInfo failed … does not support opType [AddRmsNormBias]` for `ascend910_93`
  (#11519, v0.18.0). Where `SOC_VERSION`, the per-SoC `build_aclnn.sh` branches,
  `vllm_ascend/_cann_ops_custom` + `ASCEND_CUSTOM_OPP_PATH`, and the `-a3` image tags
  decide which binaries exist, with the checks to send the reporter.
