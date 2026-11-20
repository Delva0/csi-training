# Custom-op runtime faults (aclnn / AscendC)

## Scope

Device-side failures raised while running a vllm-ascend custom AscendC operator
(and the pre-launch counterpart, where aclnn cannot resolve the op for the SoC at
all — a build/install fault, see the first rule below):
`call aclnnX failed`, `AclNN_Runtime_Error`, Nnopbase executor asserts, `fftsplus aivector
error`, MTE/DDR "out of range", retCode `0x7150050` / runtime result `507011`, or a signal
from a `csrc/<op>/*_torch_adpt.h` frame.

## Rules

- **Split parameter/registry errors from device faults first.** `AclNN_Runtime_Error`
  / a dump / tiling text means the kernel ran or was launched: the operator and shape
  matter. `AclNN_Parameter_Error(EZ1001)`, `Get regInfo failed`,
  `binary_info_config.json of socVersion [...] does not support opType [...]` means
  aclnn could not resolve the op for this SoC **before any launch**: the owner is the
  build/install chain, and no tiling fact will help — `op-not-registered-for-soc.md`.
- The dump names the operator and the error class; it does **not** identify the shape. Never
  assert "fails for hidden size N" without a shape-bearing artefact (reporter log line,
  tiling log, or a computed tiling decision).
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
- `op-not-registered-for-soc.md` — the opposite class for the same op: `EZ1001 Get
  regInfo failed … does not support opType [AddRmsNormBias]` for `ascend910_93`
  (#11519, v0.18.0). Where `SOC_VERSION`, the per-SoC `build_aclnn.sh` branches,
  `vllm_ascend/_cann_ops_custom` + `ASCEND_CUSTOM_OPP_PATH`, and the `-a3` image tags
  decide which binaries exist, with the checks to send the reporter.
