# A Triton-Ascend extension op that computes the wrong values (`extract_slice`)

## Scope

The engine serves, nothing raises, and a reporter's own Triton kernel returns wrong
data whose wrongness moves with a *slicing argument*: `extract_slice(tile,
offsets=(0, 0, rope_dim // 2), sizes=(…))` yields values from elsewhere and zeros
where the source has data, for bf16, while the fp16/fp32 variant of the same kernel
looks right. The report usually arrives with an upstream tracker id attached.

Owner: the **triton-ascend wheel** (and its compiler pair), not vllm-ascend Python —
unless one of the tree's own kernels is exposed (rule 6). This page is the silent
sibling of `triton-ascend-pair.md`, which is the same wheel failing to *compile*.

Leave this page for: an AscendC/aclnn op returning wrong numbers
(`../custom-ops-runtime-faults/attention-op-numeric-errors.md`), output that degrades
over a long generation (`../accuracy-benchmarks/index.md`), or a model-level
disagreement with no hand-written kernel in the path
(`../accuracy-benchmarks/harness-vs-engine-attribution.md`).

## Rules

1. **Pin the wheel first; the tracker's applicability is a wheel fact.**
   `rg -n 'triton-ascend' requirements.txt pyproject.toml` gives
   `triton-ascend==3.2.1` at vllm-ascend v0.23.0rc1 (`requirements.txt:20`,
   `pyproject.toml:32`). `collect_env.py` never prints the installed triton-ascend, so
   ask for `pip show triton-ascend`: a dev build or a mirror "latest" laid over the
   image changes the answer completely (the maintainer-confirmed rule in
   `triton-ascend-pair.md` — do not add or replace triton-ascend in these images).

2. **Read the op's published contract against its own test coverage; the gap is
   usually the explanation.** `docs/zh/triton_api/Extension_Ops/extract_slice.md`
   lists bf16 as supported on A2/A3 (§2.2.1) and says 无特殊限制 (§2.3, `:49`), while
   `third_party/ascend/unittest/pytest_ut/test_extract_slice.py:34-55` slices only a
   1-D `torch.rand` (fp32) at `block_start = pid * 1024` — 2048-byte-aligned starts,
   one dtype, one rank. So "documented as supported, corrupts in practice" is a
   quotable coverage statement, and the artefact to ask upstream for is a UT with a
   non-32-byte-aligned start in bf16. Do not report the doc as the bug.

3. **Separate the print path from the data path before naming a lowering.**
   `tl.device_print` of a sub-tile copies it into a print buffer with its own
   base/alignment, so the first check is a `tl.store` of the same slice into a debug
   tensor (or `insert_slice` into a zero tile and print the container) read back on
   the host. One rebuild decides whether the corruption is real.

4. **A dtype control is only a control if everything else is equal.** "fp16 fine,
   bf16 wrong" isolates the dtype **only** with identical offsets, sizes and upstream
   casts; otherwise the offset axis is still live. Note that bf16 and fp16 have the
   same element width, so an alignment-granularity story has to explain the split and
   a dtype-path story has to explain the byte arithmetic. Hand the reporter the two
   runs that decide it: bf16 at a 32-byte-aligned start (16 bf16 elements) and fp16 at
   the same unaligned start.

5. **Map the lowering surface, then label the mechanism a hypothesis.** In the pinned
   tree the Python extension op
   (`third_party/ascend/language/cann/extension/vec_ops.py:97-136`) creates a plain
   MLIR `tensor::ExtractSliceOp` via `create_extract_slice`
   (`third_party/ascend/triton_ascend.cc:52,81`), which the Ascend passes then rewrite
   (`include/AutoBlockify/AutoBlockify.h:81-82`,
   `include/TritonToUnstructure/OffsetAnalysis.h:251`,
   `include/TritonToLinalg/BlockPtrAnalysis.h:235`). That chain is where a
   32-byte-granular copy can round a base or mis-scale a stride — but the C++ is
   largely outside the truncated export, so keep the mechanism open and let rule 4
   decide.

6. **Answer the tree's own exposure explicitly, with one grep.**
   `rg -n 'extract_slice|insert_slice' vllm_ascend/` → 42 hits in six kernel files at
   v0.23.0rc1: `ops/triton/linearnorm/split_qkv_rmsnorm_rope{,_simt}.py`,
   `split_qkv_tp_rmsnorm_rope.py`, `split_qkv_rmsnorm_mrope.py`,
   `ops/triton/activation/swiglu_quant.py:43,44,53`, `ops/triton/fla/solve_tril.py:106`,
   all resolved through `ops/triton/triton_utils.py:38`
   (`_resolve_triton_ascend_op`, which exists because the op moved namespace — #9045).
   Their offsets are `0`, `HALF_ROPE_DIM` or `COL_BLOCK_SIZE` steps, i.e. aligned for
   rope_dim 64/128 in bf16, so the honest answer is "not exposed at this pin, but any
   custom kernel and any odd head dim is" — and name the file to re-audit first if an
   offset ever becomes data-dependent.

## Workarounds to hand back (host-side, no wheel patch)

- Two `tl.load`s at the two offsets instead of slicing an already-loaded tile: removes
  the op from the kernel at the cost of one extra load.
- Keep the slice start a multiple of 32 bytes — 16 elements for bf16/fp16, 8 for
  fp32, 32 for int8 — padding the row to that boundary when the head dim is small.
- Replace the slice with masked arithmetic (`tl.where(cols < HALF, x, 0)` plus a
  masked store) so all indexing stays in element-indexed vector ops.
- Pin and test a newer wheel once the upstream fix lands, verified with the reporter's
  own repro rather than the repo's UT (rule 2).

## Worked case

**#18345** — vllm-ascend v0.23.0rc1 (`f4a08bddd0cc`), triton-ascend v3.2.1
(`2badfc89e70a`, `content_export` 1467/1489), 8×910B3, CANN 9.1.0, torch 2.10.0 /
torch_npu 2.10.0.post2: a 4×4 bf16 sub-tile of a 4×8 tile taken at
`rope_dim // 2 = 4` elements (8 bytes) came back with foreign values in rows 0 and 2
and zeros in rows 1 and 3. The recorded run produced the pin, the lowering map, the
doc-versus-UT gap, the in-tree call sites and the workaround ladder, put the
print-versus-data check first, and correctly left the mechanism open. Nothing here is
executable offline, so the answer's value is the ownership sentence plus the two runs
of rule 4; do not spend turns on the C++ that is not in the snapshot.

Route: parent `index.md`; the compile-failure sibling `triton-ascend-pair.md`; the
AscendC/aclnn lane this is not `../custom-ops-runtime-faults/index.md`.
