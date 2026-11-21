# The norm/quant fusion reads `.quant_method` off a module LoRA replaced

Verified at vllm-ascend `v0.11.0rc0` (`00ba07102212`) × vllm `v0.11.0`
(`b8b302cde434`, the snapshot's pair); recorded for #4308, where the reporter's
env was exactly that pair.

Use this page when a **`--enable-lora` serve dies during startup** — Lane B,
`determine_available_memory` → `profile_run` → `_dummy_run` — with an
`AttributeError` that names a vllm-ascend class's **own** attribute:

```text
vllm_ascend/worker/model_runner_v1.py:2509 profile_run -> :2475 _dummy_run -> :2320 _generate_dummy_run_hidden_states
vllm/model_executor/models/qwen3.py:323 forward -> qwen2.py:364 -> qwen3.py:235 post_attention_layernorm
vllm/model_executor/custom_op.py:44 forward -> vllm_ascend/ops/layernorm.py:70 forward_oot
torch/nn/modules/module.py:1940 __getattr__
AttributeError: 'AscendRMSNorm' object has no attribute 'next_need_quant_fusion_linear'
```

Two facts make this shape identifiable in one turn: the log carries
`Using PunicaWrapperNPU.` (`punica_selector.py:19`) just before the crash, and the
reporter's own matrix has **three** rows — W8A8+LoRA fails, FP16+LoRA works,
W8A8 without LoRA works.

## 1. The named attribute is a property, so the message names the wrong object

`next_need_quant_fusion_linear` *is* defined at `vllm_ascend/ops/layernorm.py:77`
as a `@property`. `object.__getattribute__` calls the descriptor; when the getter
body raises `AttributeError`, CPython falls through to `torch.nn.Module.__getattr__`
(`module.py:1940`), which raises the canned message naming the **outer** property.
So the missing name is inside the getter body — and, in this case, on a *different
object* the getter selected at runtime. Read the property, not the log; do not
report the outer name as a rename or a missing feature.

Same masking, different owner: `../upstream-downstream-drift/stale-module-binding.md`
(stale consumer binding) and `../upstream-downstream-drift/swallowed-init-exception.md` (wrapper-swallowed
constructor). Here the class is fine in isolation; the fusion path is what is
not LoRA-aware.

## 2. The getter walks `model_instance`, and LoRA has replaced those modules in place

| Step | Site |
|---|---|
| the fusion gate: quant config **and** dense `model_type` **and** a v1 `layer_idx` — no env term, so no launch-time off switch | `vllm_ascend/ascend_forward_context.py:149-156`; `layer_idx` from `model_instance.model.start_layer` `:126-128` |
| the getter indexes `model_instance.model.layers[i].self_attn.qkv_proj` / `.mlp.gate_up_proj` itself | `vllm_ascend/ops/layernorm.py:91-102` |
| then dereferences `next_linear.quant_method.quant_method` against `AscendW8A8LinearMethod` | `layernorm.py:103-107`; `quantization/w8a8.py:38`; the outer `AscendLinearMethod` that legitimately owns `.quant_method` is `quantization/quant_config.py:229,238` |
| with `--enable-lora`, upstream **replaces the linear modules in place**: `_create_lora_modules` → `replace_submodule(… from_layer(module, …))` | `vllm/lora/models.py:474-502` |
| `qkv_proj` becomes `QKVParallelLinearWithLoRA`, `gate_up_proj` `MergedColumnParallelLinearWithLoRA` — `BaseLayerWithLoRA(nn.Module)` with **no** `quant_method` and no `__getattr__` delegation; quant is reached as `self.base_layer.quant_method` | `vllm/lora/layers/base.py:16`, `base_linear.py:21,141`, `column_parallel_linear.py:70,168,309`, `row_parallel_linear.py:21,137` |
| so the dereference raises `AttributeError` **inside** the getter, and step 1 turns it into the reported sentence | — |

The reported matrix follows directly: FP16+LoRA stops at the gate's first term
(`quant_config` is not `AscendQuantConfig`), W8A8 without LoRA has no wrappers, and
only W8A8+LoRA reaches the unguarded line. A working/failing matrix whose single
moving axis is `--enable-lora` **is** the diagnosis — do not then rank the
checkpoint, the dtype, TP or the adapter.

One call confirms the plugin is not LoRA-aware anywhere, which is what makes this a
defect rather than a knob: `rg -n 'base_layer' vllm_ascend/` returns no hits outside
`tests/` at this ref.

## 3. Path detail that costs calls

At this ref the upstream LoRA layer classes live in the **package**
`vllm/lora/layers/` (`base.py`, `base_linear.py`, `column_parallel_linear.py`,
`row_parallel_linear.py`), not in `vllm/lora/layers.py`. `rg` the package; a
`vllm/lora/layers.py` grep returns `No such file or directory` and reads like a
coverage gap.

## 4. Answer shape

1. Verdict first: real defect at this ref, the pasted message is misleading (the
   name exists; it is a property whose getter raised), and the computing fact is
   that the fusion path dereferences `.quant_method` on a module upstream swapped
   for a LoRA wrapper. Two greps reach this — `rg -n '<attr>' vllm_ascend/`, then
   read the getter body for the module lookups.
2. Unblock now, without a patch: drop **one** axis (FP16 + LoRA, or W8A8 without
   LoRA) — the reporter has usually already run both, so quote their own rows.
   Check `vllm_ascend/envs.py` before promising a switch:
   `VLLM_ASCEND_ENABLE_PREFETCH_MLP` and `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` exist
   (`:139`, `:152`) but neither gates this fusion, so at this ref there is no
   launch-shell off switch — say so.
3. Minimal fix and its placement: decline the fusion when the module is a wrapper,
   **after** the `fusion_linear` / `layer_idx` bookkeeping (`layernorm.py:91-102`),
   because the `qkv_dense`↔`gate_up_dense` alternation and `layer_idx += 1` must
   keep advancing or the next layer's fusion is off by one. Working form:

   ```python
   base = getattr(next_linear, "base_layer", next_linear) if next_linear is not None else None
   if base is None or not isinstance(base.quant_method.quant_method, AscendW8A8LinearMethod):
       next_linear = None
   ```

4. Do not present "unwrap and keep the fusion" as equivalent: the fused op consumes
   `layer.aclnn_input_scale/offset` and emits the matmul input, while the LoRA
   wrapper computes its delta from the normed tensor and only then calls
   `self.base_layer.quant_method.apply(self.base_layer, x, bias)`
   (`base_linear.py:141`). Returning `None` deliberately falls back to plain
   `npu_add_rms_norm` plus the layer's own quantisation. **Acceptance = one real
   completion through the adapter compared against the FP16+LoRA baseline**; an
   engine that merely starts is not a validation.
5. Scope: coverage gap is quotable as such — at this ref the only tests touching
   the file are `tests/ut/ops/test_layernorm.py` and `tests/ut/test_utils.py`, with
   no quant × LoRA case. "Fixed later?" is unanswerable from a one-commit export;
   give the recheck — `rg -n 'base_layer|next_linear.quant_method'
   vllm_ascend/ops/layernorm.py` at a newer ref.

## Recheck conditions

The gate's model list and field names move (`ascend_forward_context.py`), as does
the fusion's own placement inside `forward_oot` (`layernorm.py:60-107` here). The
pairing to re-verify before reusing the mechanism: the fusion still indexes
`model_instance.model.layers[...]` rather than reading the layer being executed, and
upstream still substitutes LoRA wrappers in place rather than shadowing them.
