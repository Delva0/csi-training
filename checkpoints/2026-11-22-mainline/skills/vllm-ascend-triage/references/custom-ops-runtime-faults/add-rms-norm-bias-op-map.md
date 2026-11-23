# Map: `csrc/add_rms_norm_bias` at vllm-ascend v0.19.1rc1 (commit `da421afa`)

Verified in the v0.19.1rc1 tree. Constants and thresholds are version-specific — recheck
against `SOURCES.json` before reusing on another release.

## Files

- host: `csrc/add_rms_norm_bias/op_host/{add_rms_norm_bias_tiling.cpp,tiling.h,def.cpp,infershape.cpp}`
- kernel: `op_kernel/add_rms_norm_bias.cpp` (tiling-key dispatch), `add_rms_norm_bias.h`
  (MODE_NORMAL), `_split_d.h`, `_merge_n.h`, `_single_n.h`, `_multi_n.h`, plus
  `rms_norm_base.h` (DataCopy/Reduce helpers) and `reduce_common.h`
- adapter: `csrc/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h` (allocates `y`, `x`,
  `rstd`; `rstd` shape = dims before gamma dims, all 1s)

## Tiling key and modes

`tiling_key = dtype_key * 10 + mode_key` (`SaveTilingData`), dispatched by
`TILING_KEY_IS(...)` in `op_kernel/add_rms_norm_bias.cpp`.

- dtype keys: FP16 `1`, FP32 `2`, BF16 `3` → 10/20/30 = normal, 11/21/31 = split-D,
  12.. merge-N, 13.. single-N, 14.. multi-N
- modes: `MODE_NORMAL 0`, `MODE_SPLIT_D 1`, `MODE_MERGE_N 2`, `MODE_SINGLE_N 3`,
  `MODE_MULTI_N 4`
- BF16 kernels are compiled out under `__NPU_ARCH__ == 3003`.

## Shape normalisation (`CalculateRowAndColParameters`)

`numCol` = element count of input 2 (gamma); `numRow` = product of x1 dims before the gamma
dims; `numColAlign = CeilDiv(numCol, dataPerBlock) * dataPerBlock` with `dataPerBlock` 16
(fp16/bf16) or 8 (fp32). `blockFactor = CeilDiv(numRow, coreNumAiv)`,
`useCoreNum = CeilDiv(numRow, blockFactor)`, `blockDim = useCoreNum`, last core gets the
remainder `latsBlockFactor`.

## Mode selection order (`DetermineModeParameters`)

1. `numCol > ubFactor` → `MODE_SPLIT_D`, then `ubFactor` recomputed aligned to
   `dataPerBlock` from the cut-D constants.
2. `blockFactor == 1` and SoC != 310P → `MODE_SINGLE_N` (i.e. `numRow <= coreNumAiv`).
3. small aligned column (`numColAlign <= 2000` without beta, `<= 1600` with beta) and
   SoC != 310P → `MODE_MERGE_N`, `rowFactor = ubSize / (numColAlign*weight + DIV_FACTOR)`,
   `ubFactor = rowFactor*numColAlign`; weights are `OTHER_WEIGHT 18` / `FP32_WEIGHT 24`
   without beta and `OTHER_WEIGHT_WITH_BETA 20` / `FP32_WEIGHT_WITH_BETA 28` with beta,
   plus `DIV_FACTOR 260`.
4. (`fp16` or `isPerformance==1`) and `numCol == numColAlign` → `MODE_MULTI_N`, rowFactor
   from `ubSize`; if it rounds to 0 → fall back to `MODE_NORMAL`, `rowFactor=64`,
   `ubFactor=UB_FACTOR_B16`.
5. otherwise `MODE_NORMAL` (ubFactor defaults below).

`isPerformance = 1` only on ASCEND910B with x rank 2 or 3 (leading dim <= 512, second <= 8),
`numCol <= 5120`, dtype fp16/bf16 (`getPerformanceFlag`).

## Budget constants (`add_rms_norm_bias_tiling.cpp`)

`UB_USED 1024` (subtracted from `ubSize`); `UB_FACTOR_B16 12288`, `B32 10240`,
`B16_CUTD 12096`, `B32_CUTD 9696`, with-beta `9216`(B32) / `11264`(B16) / `8096`(B32_CUTD) /
`10752`(B16_CUTD); `SMALL_REDUCE_NUM 2000`, `SMALL_REDUCE_NUM_WITH_BETA 1600`;
workspace = 256 B user + 16 MiB system. Tiling logs an I-line with
`Tiling Key` / `Block Dim` / `num_row,num_col,block_factor,row_factor,ub_factor` — the
cheapest artefact to request.

## Kernel-side facts worth knowing

- `SINGLE_N` forces `blockFactor = 1` and allocates one `TBuf` of `MAXBUFFER = 195584` bytes
  (`add_rms_norm_bias_single_n.h:24`), slicing it by `ubFactor`; for 16-bit input the layout
  needs ~`16*ubFactor` bytes (x1,x2 as T in the low `2*ubFactor` bytes, three fp32 buffers at
  `4/8/12*ubFactor`), so large `ubFactor` leaves little slack — check this arithmetic
  explicitly if a UB-sizing bug is suspected, and keep it labelled as a hypothesis.
- `MERGE_N` uses a compile-time `isNumColAlign` (true only for `__NPU_ARCH__ == 3003`) to
  choose strided multi-row copies vs flattened copies; arch changes alter its GM extents.
- `rms_norm_base.h` guards `DataCopyPad` on `__CCE_AICORE__ == 220` / `__NPU_ARCH__ == 3003`
  and uses plain `DataCopy` elsewhere; non-aligned counts take different code paths per arch.
- `rstd` GM window is `blockIdx_*blockFactor` wide `blockFactor`, while `rowWork` may be
  smaller on the last core — verify against `numRow` when any rstd overrun is suspected.

## Already checked at this pin — do not redo

Re-verified against `da421afa` after a recorded forward (#8944, the GLM-5.1 / A2
`aclnnAddRmsNormBias` MTE-DDR dump) spent most of its budget here. Each line below is a
*closed* question; cite it instead of re-reading the files:

- **The host refuses a row-count mismatch, so a TP-chunked residual cannot produce this
  dump.** `op_host/add_rms_norm_bias_tiling.cpp:157-190` (`CheckInputOutputShape`) compares
  `x2`, `y` and `x` with `x1` dim-by-dim and returns `GRAPH_FAILED` with
  `Input x2/x1 shape invalid, shape is not equal x1 shape.` So
  `vllm_ascend/ops/register_custom_ops.py:23-36` (`_maybe_chunk_residual_impl`: `F.pad` by
  `_EXTRA_CTX.pad_size`, then `torch.chunk(residual, tp_size, 0)[tp_rank]`, uneven whenever
  rows are not divisible by TP) can only yield a **tiling refusal in the log**, never a
  device MTE fault. No such line in the paste retires that candidate.
- **`MODE_NORMAL`'s last-core arithmetic is consistent.** `op_kernel/add_rms_norm_bias.h:43-47`
  sets `rowWork = numRow - (blockNum-1)*blockFactor` on the last core, the GM windows
  `:49-57` are sized `rowWork*numCol` (rstd `blockFactor`), and the copy loop bounds
  `:83-90` walk `CeilDiv(rowWork, rowFactor)` rows. A NORMAL-mode overrun claim needs a
  *specific* mode boundary, not "the remainder path".
- **`rstd` is written only on some builds.** `CopyOutRstd` is compiled in under
  `__CCE_AICORE__ == 220 || __NPU_ARCH__ == 3003` (`:327-333`); elsewhere the rstd window is
  allocated and never written (`DataCopyPad` guards in `rms_norm_base.h` follow the same
  macros). A "rstd stays uninitialised" report on another arch is that `#if`, not tiling.

**Deliverable for a #8944-shaped dump** (op named, one rank/chip, runs a while before
faulting). Rung 0 is not in this tree: at v0.19.1rc1 the release's own `Known Issues` row
declares that GLM-5/5.1 runs "in single-node (non-PD-disaggregated) scenarios" may produce
runtime errors (`release_notes.md:201`, tracked as #8843), and that issue's log repeats this
paste's whole device record on **other** chips while naming `rejection_greedy_sample_triton` — so the op name here is a victim, and the
scenario row plus an `--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1` replay of the same load
rank ahead of the three causes below (`../release-known-issues/glm5-single-node-runtime-fault-row.md`).

Then the deliverable: the failing rank's tiling I-line (`Tiling Key` / `Block Dim` /
`num_row,num_col,block_factor,row_factor,ub_factor`), whether `enable_custom_op()` and the
compile-time fusion passes are on for that run, and the version lane. Repeated with
`--enforce-eager` the passes do not run, so the op is then reached only from
`layernorm.py`'s eager branch — that is the ablation which separates the two call sites.
Ranked candidate causes, each with its one check — (1) that rank's `numRow` at a mode
boundary (I-line above); (2) the fusion-pass call site, whose inputs (`beta` plus a
`maybe_chunk_residual` residual) differ from the eager call; (3) the CANN/`torch-npu` triple
and any post-pin fix, checked by re-running the same load on the current image. None of
these is settled by reading the kernel again: the paste carries no shape, so the I-line is
the discriminator.

## Who reaches this op (vllm-ascend v0.19.1rc1)

- `vllm_ascend/ops/layernorm.py` `forward_oot`: uses
  `torch.ops._C_ascend.npu_add_rms_norm_bias` only when `residual is not None` **and**
  `enable_custom_op()` (`vllm_ascend/utils.py`), else `torch_npu.npu_add_rms_norm`; the Gemma
  variant passes `beta=None`.
- fusion passes: `vllm_ascend/compilation/passes/{sequence_parallelism,sequence_parallelism_moe,norm_quant_fusion_pass,allreduce_rmsnorm_fusion_pass}.py`
  (these pass real `beta`, i.e. `nullptr_beta == 0`).
So: if custom ops are disabled in the reporter's run, an `aclnnAddRmsNormBias` fault cannot
come from this path — check the switch before analysing kernels.
