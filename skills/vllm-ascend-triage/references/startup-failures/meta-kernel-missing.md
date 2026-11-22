# Custom op with no Meta/fake kernel during compile (`attempted to run this operator with Meta tensors`)

Symptom family: startup gets past model load and memory profiling, then a per-rank
worker traceback whose frames are `torch/_subclasses/fake_tensor.py` and
`torch/_ops.py` and which ends

```
NotImplementedError: _C::rotary_embedding: attempted to run this operator with Meta tensors.
You may have run into this message while using an operator with PT2 compilation APIs
(torch.compile/torch.export); in order to use this operator with those APIs ...
```

(or `_C_ascend::<op>`, or a `torch_npu` `npu_*` name). The wording is PyTorch's
dispatcher, not vllm-ascend's: PIECEWISE compilation traces the model with fake
tensors, and the op was registered for `PrivateUse1` only. Nothing on the card
ran, so no plog, device health or CANN question is live. Recorded at #1895
(vllm-ascend v0.9.2rc1 `b5b7e0e`, vllm v0.9.2 `a5dd03c1`, 4×910B3, Qwen2.5-32B).

## Step 1 — the namespace in the message names the era and the owner

- **`_C::<op>`** — trees before the rename to `_C_ascend` (≈v0.9.x). Here the `_C`
  namespace is registered by **vllm-ascend's own extension**, not by vLLM:
  `csrc/torch_binding.cpp:290 TORCH_LIBRARY_EXPAND(_C, ops)`, the
  `rotary_embedding` schema at `:299-301` and
  `ops.impl("rotary_embedding", torch::kPrivateUse1, …)` at `:302`;
  `grep -rn 'kMeta\|TORCH_LIBRARY_IMPL' csrc/` returns only the macro definition in
  `csrc/utils.h:17-20`, so **no** op in the extension has a Meta impl.
  The `Failed to import from vllm._C` warning in the same paste is normal on NPU
  (the wheel carries no CUDA extension) and is not the cause. The crash itself
  proves `vllm_ascend_C` did import: the branch is gated on `enable_custom_op()`
  (`vllm_ascend/utils.py:217-234`), which returns `False` and logs
  `Failed to register custom ops, all custom ops will be disabled` when the
  extension is absent, and then the branch is skipped instead of raising.
- **`_C_ascend::<op>`** — newer trees, where meta impls exist.
  v0.14.0rc1 (`52d4acfa`): `vllm_ascend/meta_registration.py:108-111` registers
  `rotary_embedding`, `get_masked_input_and_mask`, `bgmv_expand` and
  `sgmv_expand` through `register_meta_if_necessary` (`:45-52`, which skips an op
  that already has a Meta registration), imported from inside `enable_custom_op()`
  (`utils.py:255-258`), plus `csrc/torch_binding_meta.cpp` for the C++ side. So on
  those refs the same text means the registration was skipped or the failing op is
  a *new* one added without a meta function —
  `rg -n '<op>' vllm_ascend/meta_registration.py csrc/torch_binding_meta.cpp`
  before blaming the release.

## Step 2 — read the one branch that reaches the unregistered op

The gate is usually a dtype/shape predicate, so the same model in another dtype
dodges it — that is what the answer turns on. #1895, at `b5b7e0e`:
`vllm_ascend/ops/rotary_embedding.py:286` assigns
`RotaryEmbedding.forward_oot = rope_forward_oot`; the only exemption from the
custom kernel is `get_ascend_config().torchair_graph_config.enabled` (`:42-48` →
`forward_native`), which does **not** cover the ACL-graph path, and the branch at
`:60-70` needs `query.dtype == torch.float16 and neox_style and head_size % 32 == 0
and enable_custom_op()` (`:29-31`) **and `not is_310p()`** (`:60-61` — on a 300I/310P die
the branch is never taken, so the same text there is a SoC question, not this one:
`../soc-platform-paths/index.md`). The reporter's log carried
`Casting torch.bfloat16 to torch.float16` and `dtype=torch.float16`, so an fp16
launch selected the branch, and Qwen2's `head_dim` 128 with neox-style satisfied
the rest. The other shim is not a fallback: `register_dummy_fusion_op`
(`vllm_ascend/ops/__init__.py:35-49`, called from `worker/worker_v1.py:71`)
replaces `torch.ops._C.rms_norm` and the fp8 quant names with Python dummies and
does not touch `rotary_embedding` — check it once, then stop looking for one.

## Step 3 — the ladder

1. `--enforce-eager` first: `platform.py:160-162` sets
   `CompilationLevel.NO_COMPILATION`, so nothing is traced with fake tensors and
   the custom kernel runs eagerly (it is a real device kernel, not a stub). This
   is the rung that unblocks the reporter on #1895.
2. The dtype that does not take the branch (`--dtype bfloat16` there). Label it
   **hypothesis**: at #1895 the fallback is `torch_npu._npu_rotary_embedding` and
   its fake support lives in the op-plugin C++ the snapshot does not carry. The
   discriminator is free — if it still fails, the new message names
   `npu_rotary_embedding`, which proves the branch moved.
   There is no env knob to disable the op instead: at v0.9.2rc1
   `rg -n 'CUSTOM_OP|custom_op' vllm_ascend/envs.py` returns nothing (the only gate is
   the lazy `enable_custom_op()` import in `utils.py:217-234`), so "turn the custom
   kernel off and keep graph mode" is not a reporter-side rung there. Re-run that grep
   on the reported ref before repeating the negative: later releases add envs.
3. Maintainer-side only: register the Meta impl (C++
   `TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, m)`, as `meta_registration.py` later did in
   Python) or gate the custom kernel off while capturing. Do not tell the reporter
   to patch the caller to `forward_native`: that changes numerics for every user.

## Step 4 — what the paste lacks

The frames between `fake_tensor` and `_ops.py` (reporters' pastes drop them, so the
calling `file:line` has to come from the pinned tree) and the **effective dtype**
(`dtype=torch.float16` in the engine-config line, or the `Casting …` warning).
Ask for both; the checkpoint's declared bf16 does not tell you what the launch
requested, and the dtype is what selects the branch.

## Version scope

v0.9.2rc1 + vllm v0.9.2 for the `_C` chain; the `_C_ascend` /
`meta_registration.py` correspondence was read at v0.14.0rc1. The
`register_dummy_fusion_op` coverage and the dtype gate both move between releases,
so re-run the two greps of step 1 on the image in the report before quoting them.
