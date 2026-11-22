# A Triton-Ascend-compiled kernel that computes the wrong values

## Scope

The engine serves, nothing raises, and a reporter's own Triton kernel returns wrong
data. Two mechanisms live here; they share the wheel, the ownership sentence and the
workaround ladder, and nothing else, so the first job is to say which one the dumps
put you in.

- **(a) slicing.** The wrongness moves with a *slicing argument*:
  `extract_slice(tile, offsets=(0, 0, rope_dim // 2), sizes=(…))` yields values from
  elsewhere and zeros where the source has data, for bf16, while the fp16/fp32 variant
  of the same kernel looks right. Often arrives with an upstream tracker id attached.
- **(b) a zeroed reduction.** Everything up to `tl.sum(..., axis=1)` is right and the
  **first non-identity scalar scale** applied to the reduction result comes back `0`
  (`square_sums / head_size`, `* 2.0`, `* 0.5` → zeros; `* 1.0` fine), only while a
  later broadcast-multiply consumer of a **slice of that same tile** exists
  (`orig_qk = extract_slice(k_normalized, …)` then `orig_qk * cos_tensor`), and only at
  one head count (kv heads 4 wrong, 2 right). The slice's own values are correct —
  which is exactly the sentence that makes an (a)-shaped page look like the wrong one.

| what the reporter's own dumps show | mechanism | read |
|---|---|---|
| a `tl.store` of the *slice* carries foreign values / zeros; the pattern moves with offsets, sizes or dtype | (a) | rules 1-6 |
| `squares` and `tl.sum(squares, axis=1)` are correct, the *scaled* value is 0, `* 1.0` is correct, deleting the later cos-multiply fixes it, one shape keyed | (b) | rules 7-11, then the exposure check |
| the reduction, the scale **and** the slice are all right and the model still disagrees | neither | `../accuracy-benchmarks/index.md` |

Owner: the **triton-ascend wheel** (and its compiler pair), not vllm-ascend Python —
unless one of the tree's own kernels is exposed (rules 6 and 9). This page is the
silent sibling of `triton-ascend-pair.md`, which is the same wheel failing to *compile*.

Leave this page for: an AscendC/aclnn op returning wrong numbers
(`../custom-ops-runtime-faults/attention-op-numeric-errors.md`), output that degrades
over a long generation (`../accuracy-benchmarks/index.md`), or a model-level
disagreement with no hand-written kernel in the path
(`../accuracy-benchmarks/harness-vs-engine-attribution.md`).

## Rules for the slicing form (a)

1. **Pin the wheel first; the tracker's applicability is a wheel fact.**
   `rg -n 'triton-ascend' requirements.txt pyproject.toml` gives
   `triton-ascend==3.2.1` at vllm-ascend v0.23.0rc1 (`requirements.txt:20`,
   `pyproject.toml:32`) and `==3.2.2` at v0.23.0 (`5cb98caa`) — the wheel moves inside
   one release line, so a source-identical kernel is *not* the same exposure.
   `collect_env.py` never prints the installed triton-ascend, so
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
   rope_dim 64/128 in bf16, so the answer for **this** mechanism is "not exposed at
   this pin, but any custom kernel and any odd head dim is" — and name the file to
   re-audit first if an offset ever becomes data-dependent. That sentence covers (a)
   only; the same `split_qkv_rmsnorm_mrope.py` is exposed to (b) at the same pin
   (rule 9).

## Rules for the zeroed-reduction form (b)

7. **Read the reporter's ladder as a shape × consumer bisect, not as a numerics
   question.** Five rows settle it, each one kernel edit: store `squares` back
   (correct) → store `tl.sum(squares, axis=1)` back (correct) → store the **scaled**
   value back (0) → replace the non-identity scale with `* 1.0` (correct, i.e. the
   identity was folded away and no scaling op exists) → delete the downstream
   cos-multiply and keep the scale (correct). Then the shape axis: one head count
   wrong, its neighbour right. If the reporter has not stored the *scaled* value, that
   single store is the request that separates "the scale is miscompiled" from "the
   reduction is fine and the divide is a red herring". Ask for it rather than deriving
   it — a divide and a multiply behave identically here, so the operator symbol proves
   nothing.

8. **One call prints the exposure and the coverage:**
   `bash <skill_root>/scripts/qknorm-mrope-exposure.sh <vllm-ascend-root> --kv-heads N`.
   It lists the per-head reduction sites, which kernel hands a slice of the
   *normalized* tile to a cos/sin multiply, every caller of
   `triton_split_qkv_rmsnorm_mrope`, the FX pass's `head_size != 128` / bf16-only gates,
   the absence of an env switch, and the nightly test parameter tuples with the kv-head
   counts CI actually runs. Quote its lines; do not re-walk the tree by hand.

9. **Exposure and coverage are two sentences, and rule 6 is not the second one.**
   *Exposure*: at v0.23.0rc1 the op is reached by direct `torch.ops` calls from
   `patch/worker/patch_qwen3_5.py:51` (`model_type` contains `qwen3_5`) and
   `patch/worker/patch_qwen3vl.py:43`, which bypass the FX fusion pass's
   `layer.head_size != 128` gate (`compilation/passes/qknorm_rope_fusion_pass.py:208`)
   and its bf16-only gate (`:197`); `envs.py` carries no qknorm/rope-fusion switch, so
   no flag turns it off. *Coverage*: `tests/e2e/nightly/single_node/ops/singlecard_ops/
   triton/test_split_qkv_rmsnorm_mrope.py:7-10` runs `NUM_QKV_HEADS = [(2, 1), (16, 2)]`
   with `HEAD_SIZES = [128, 256]` and `MROPE_SECTION` including `[11, 11, 10]`
   (rope_dim 64) — kv heads = 4 is never run on the mrope kernel, while the non-mrope
   `test_split_qkv_rmsnorm_rope.py:11` does run `(64, 4)` at `head_size` 128. So "green
   CI" and "the reporter's working point is the CI-tested point" are the same fact, and
   kv=4 is exercised only on the kernel whose chain weights and casts before the slice.

10. **Hand back a structural fix before a temporary store.** The working repair is to
    break the data chain: in the `IS_PARTIAL_ROPE` branch take `orig_qk` from the
    input tile (a fresh `tl.load` of the same K region, or
    `extract_slice(in_k_tensor, …) * reciprocal_std * k_rmsnorm_weight`) instead of
    slicing `k_normalized`. That is mathematically identical, removes the
    reduction→scale→broadcast→slice→broadcast-multiply chain, and does not depend on a
    compiler decision. The reporter's own workaround (store `square_sums` into the
    unused first `num_kv_heads` lanes of the output, then overwrite them) is a **dead
    store**: a compiler is allowed to eliminate a store whose values are overwritten
    and never read, so label it a workaround, keep the note "delete it once the
    compiler is fixed", and re-test it with the reporter's minimal repro after every
    triton-ascend bump (rule 1).

11. **Keep the mechanism labelled.** Which pass folds the scale is in the wheel's C++
    (`AutoBlockify`, `TritonToLinalg`) and outside the export, so state "the scale op
    on the reduction result is miscompiled at this shape" as source-inferred from the
    reporter's ladder and leave the pass open. Do not present the shape count (4) as
    the cause; it is the switch that selects the lowering.

## Workarounds to hand back (host-side, no wheel patch)

Slicing form (a):

- Two `tl.load`s at the two offsets instead of slicing an already-loaded tile: removes
  the op from the kernel at the cost of one extra load.
- Keep the slice start a multiple of 32 bytes — 16 elements for bf16/fp16, 8 for
  fp32, 32 for int8 — padding the row to that boundary when the head dim is small.
- Replace the slice with masked arithmetic (`tl.where(cols < HALF, x, 0)` plus a
  masked store) so all indexing stays in element-indexed vector ops.
- Pin and test a newer wheel once the upstream fix lands, verified with the reporter's
  own repro rather than the repo's UT (rule 2).

Zeroed-reduction form (b):

- Break the chain at the slice (rule 10) — the first fix to offer.
- If a store-based workaround is kept, keep it out of the output buffer if a spare
  kernel argument is acceptable: a scratch tensor is not overwritten, so no dead-store
  elimination can undo it.
- Re-run the reporter's five-row ladder (rule 7) as the acceptance test after any
  wheel change; a green nightly is not that test (rule 9).

## Worked cases

**#18345** (form (a)) — vllm-ascend v0.23.0rc1 (`f4a08bddd0cc`), triton-ascend v3.2.1
(`2badfc89e70a`, `content_export` 1467/1489), 8×910B3, CANN 9.1.0, torch 2.10.0 /
torch_npu 2.10.0.post2: a 4×4 bf16 sub-tile of a 4×8 tile taken at
`rope_dim // 2 = 4` elements (8 bytes) came back with foreign values in rows 0 and 2
and zeros in rows 1 and 3. The recorded run produced the pin, the lowering map, the
doc-versus-UT gap, the in-tree call sites and the workaround ladder, put the
print-versus-data check first, and correctly left the mechanism open. Nothing here is
executable offline, so the answer's value is the ownership sentence plus the two runs
of rule 4; do not spend turns on the C++ that is not in the snapshot.

**#12642** (form (b)) — reported on Ascend 950PR (one card per `npu-smi`), Qwen3.5-4B
bf16, `num_q_heads 16 / num_kv_heads 4 / head_size 256 / rope_dim 64` (partial RoPE);
the reporter's own dump says vLLM 0.21.0 / vllm-ascend `0.21.0rc2.dev0+g80610e443`,
the dated reference pin is v0.23.0rc1 (`f4a08bddd0cc`) + triton-ascend 3.2.1 — the
same wheel the reporter ran. The pinned kernel carries the chain verbatim:
`:156-159` `squares → tl.sum(…, axis=1) / head_size → reciprocal_std.reshape(num_kv_heads, 1)`,
`:232-238` `orig_qk = extract_slice(k_normalized, (0, 0), (num_kv_heads, rope_dim))`,
`:240` `roped_k = cat_y * sin_tensor + orig_qk * cos_tensor`, `:250` the insert back,
and `:308` `IS_PARTIAL_ROPE = rope_dim != head_size` → true at 64 vs 256; `:139-140`
builds the cos tile as `broadcast_to((2, half_rope_dim)).reshape(1, rope_dim)`, i.e.
line-for-line the reporter's minimal repro, and no protective store exists in the file
at either v0.23.0rc1 or v0.23.0 (`5cb98caa`, wheel 3.2.2). Their ladder: `squares` and
`tl.sum` correct (`226.81 / 255.20 / 267.00 / 268.82`), every non-identity scale → all
zeros, `* 1.0` correct, kv heads 2 correct and 4 wrong, deleting the cos-multiply
fixes it, and the `cat_y` path (fresh `tl.zeros` + `insert_slice`) never triggers. What
the recorded run added beyond the reporter's own conclusion: the two call-site patches
that skip the FX gate, the absence of an env switch, and the CI coverage gap of rule 9
— deliverable as "live at this pin, reaching Qwen3.5/Qwen3-VL weights with 4 kv heads,
and green CI is the coverage statement, not a contradiction". Owner: triton-ascend
3.2.1 codegen; the tree's math is correct.

Route: parent `index.md`; the compile-failure sibling `triton-ascend-pair.md`; the
AscendC/aclnn lane this is not `../custom-ops-runtime-faults/index.md`;
`scripts/qknorm-mrope-exposure.sh` is the one call for rules 8-9.
