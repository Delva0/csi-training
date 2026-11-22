# The same op takes a different implementation because `HAS_TRITON` flipped

## Scope

The engine serves and the reporter compares two **implementations of one op inside this
tree** — "rotary_embedding.py 中 Triton 实现和 ascend 实现结果不一致 … 我理解不论是 triton 实现还是
ascend C 实现，结果应该是一样的，但是这里会有乱码现象，为何？" — usually on a box where **both**
`triton` and `triton-ascend` are installed, with a garbled or repeated completion as the
observable damage and no traceback anywhere.

Recorded at vllm-ascend **v0.17.0rc1** (`e20f0b1a`) + vllm **v0.17.0** (`b31e9326`), #8047:
硬件 G8600, 镜像 `vllm-ascend 0.17.0rc1`, CANN `9.0.0beta2`, the reporter's own `curl` body
answered `user user user …` at `finish_reason: length`. **That case recorded no
resolution**, so the dispatch reading below is `source-verified` while "this is why the
output garble" stays a **hypothesis** and must be labelled as one.

Leave for its siblings: a kernel that fails to **compile**
(`triton-ascend-pair.md`), an `import` residue (`triton-module-residue.md`), a logged
`Failed to import Triton kernels` (`triton-kernels-import-error.md`), an
`'_OpNamespace' … has no attribute <op>'` at `torch.compile`
(`../startup-failures/missing-triton-op-in-compile-pass.md`), and a UB-overflow tiling
error (`../custom-ops-runtime-faults/triton-ub-overflow-tiling-budget.md`).

## The branch is install state, not a capability or a parity check

- `vllm_ascend/ops/rotary_embedding.py:153` `rope_forward_oot` → `:166` `if HAS_TRITON:` →
  `rope_forward_triton` (`vllm_ascend/ops/triton/rope.py`), else `:189`/`:205`
  `torch_npu._npu_rotary_embedding` (the CANN aclnn op). The same pair gates
  `AscendMRotaryEmbedding.forward_oot` at `:518`. Nothing logs which branch ran.
- `HAS_TRITON` = `find_spec("triton")` **and exactly one active** triton backend driver
  (`vllm/triton_utils/importing.py:12-15`, `:42-48`, vllm v0.17.0): 0 or >1 active drivers
  → `False`, with `Triton is installed but N active driver(s) found (expected 1). Disabling
  Triton to prevent runtime errors.`; absent → `Triton not installed or not compatible …`
  (`:69-72`). Which `triton/` directory wins in `site-packages` is a file-level fact, so
  the boolean — not the package list — names the branch: the report's "这两个都会同时存在，这时
  会走到下面的分支" *is* `HAS_TRITON == False`, and after the reporter's uninstall/reinstall it
  flipped to the Triton kernel, i.e. the comparison is two install states, not two releases.

## Why both packages exist (recorded mechanism) and why they are not interchangeable

- The recipe removes it on purpose: `Dockerfile:54-57` — comment "In x86, triton will be
  installed by vllm. But in Ascend, triton doesn't work correctly. we need to uninstall it."
  — then `python3 -m pip uninstall -y triton`, and the tree pins `requirements.txt:39`
  `triton-ascend==3.2.0`. A box holding both is a **drifted** install; one
  `site-packages/triton/` directory then belongs to two distributions.
- The x86 route is recorded in #6737's own analysis (reporter-side, not
  maintainer-verified): "the dependency statement of the xgrammar package will install more
  triton 3.6 packages … causing it to conflict with the existing triton-ascend 3.2.0 …
  As a result, the correct triton-ascend device cannot be obtained when detecting the triton
  driver". vllm v0.17.0's `requirements/common.txt:27` is the x86-included `xgrammar`
  line — quote it as the *route*, not as the proof of the install.
- The two paths are **not** two spellings of one contract: `ops/triton/rope.py:23-25`
  states the kernel's own envelope — "I only have tested this kernel on Deepseek V3.2 and
  Qwen3-Next. For models with larger n_q_head and n_kv_head such as GLM 4.6, this is not
  supported yet" — while the aclnn op is vendor-maintained. #5352 ("accuracy issue for
  triton kernel split_qkv_rmsnorm_rope_kernel") is the tree's own precedent that a Triton
  rope-family kernel can disagree with the reference. "They should agree" is the
  reporter's premise, not the tree's claim.

## Two calls, then one A/B

1. Which branch: `python3 -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"`
   and `pip show triton triton-ascend` in the serving env (grep the log for the two
   `importing.py` lines first — they may already be in the paste).
2. Which was compared: ask for the exact head of both sides of the diff. A comparison
   against a **hand-edited** copy is a different question from on-box behaviour, and
   `forward_triton` vs `forward_oot` is a *class* split (mrope VL) that has nothing to do
   with the `rope_forward_oot` `if HAS_TRITON` split — the two appear in the same file
   (`:153` vs `:512`) and one screenshot can be of either.
3. The A/B that answers "should they be equal": run both paths on the same tensors in one
   process — `torch_npu._npu_rotary_embedding(pos, q, k, head_size, cos_sin_cache,
   is_neox_style)` (in place) against `rope_forward_triton(q, k, cos_sin_cache=…,
   positions=…, rope_dim=…, is_neox_style=…)` on cloned bf16/fp16 `q`/`k` of shape
   `[T, H, D]` and one `positions`/`cos_sin_cache` pair taken from the model's own config
   (head_size 128 / rotary_dim 128 is the kernel's tested envelope). A non-zero
   `(a-b).abs().max()` at that shape is the finding; keep the shapes in the answer.
4. The repair rung is the recipe, not a version: `pip uninstall -y triton` and no second
   triton-ascend (`triton-ascend-pair.md`'s maintainer rule — do not add or replace
   triton-ascend in these images), then re-check `HAS_TRITON` and re-run the same request.
   State that the garble may survive this: with `HAS_TRITON=False` the engine runs the same
   aclnn op the arm image runs, so "path changed" and "output fixed" are separate claims.
