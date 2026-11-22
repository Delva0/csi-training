# A native FP8 checkpoint inherits the DeepSeek-V4 fp8 config (A5)

## Scope

Engine start dies **during model construction**, before any weight is read, on a
checkpoint the publisher released in FP8 — `config.json` carries
`quantization_config.quant_method: "fp8"` (`Qwen/Qwen3.8-27B-FP8` and its siblings; the
reporter's own directory in #14467) — with this chain:

```text
vllm_ascend/ops/linear.py AscendLinearBase.__init__
  → quant_config.get_quant_method(prefix=…)
  → vllm_ascend/quantization/fp8_config.py get_quant_method → create_scheme_for_layer(…, "ds_linear", …)
  → vllm_ascend/quantization/methods/fp8.py __init__
AttributeError: '<Model>Config' object has no attribute 'o_groups'
```

(or `o_lora_rank`; `Qwen3_5Config` in the recorded case). The log's `Using the vLLM
Ascend fp8 Quantization now!` line above it is `create_scheme_for_layer`'s own
`logger.info_once`, so its presence dates the failure to scheme construction, not to
model-attribute drift. Inherit `index.md`. Two neighbours own other shapes of the same
word "fp8": a **modelslim** export is `quantization-method-gate.md` §B/§C, and an
`fp8` checkpoint on A2/A3 (refused at the platform gate) is its §E. Attribute errors
raised from a `vllm_ascend` frame on a class that mirrors an upstream one are
`../upstream-downstream-drift/index.md` — this page is the case where the class is
reachable and *scope-wrong*.

Verified at vllm-ascend **v0.23.0** (`5cb98ca`, vllm v0.23.0 `0fc695f`), the pin of
#14467; re-grep every line at another ref before quoting it.

## The chain, all four facts source-verified at that pin

| # | Site | Fact |
|---|---|---|
| 1 | `quantization/fp8_config.py:30-37` | `remove_quantization_method()` deletes **both** `fp8` and `deepseek_v4_fp8` from `QUANTIZATION_METHODS`, then `:68` registers `AscendFp8Config` under the generic `FP8_METHOD` and `:137` registers the *same class* for `deepseek_v4_fp8` ("handled identically to fp8 on Ascend"). Upstream keeps the two apart: `"fp8": Fp8Config` (`vllm/model_executor/layers/quantization/__init__.py:143`) vs `"deepseek_v4_fp8": DeepseekV4FP8Config` (`:165`). |
| 2 | `quantization/utils.py:132-133` | `detect_quantization_method` returns `fp8` for *any* directory whose `quantization_config.quant_method == "fp8"` **and** `get_ascend_device_type() == AscendDeviceType.A5` (A5 is `soc_version == 260`, `utils.py:857-858`). So the auto-detection itself routes the model into the Ascend class; the model family never enters the test. |
| 3 | `quantization/methods/fp8.py:45,47` | The only linear scheme registered for `FP8` is `ds_linear` (`:30`), and its `__init__` reads `hf_config.o_groups` / `o_lora_rank` **unconditionally** — fields only `vllm_ascend/models/deepseek_v4.py` sets (`:733`). The only consumer is the MLA weight reshape behind `if layer.prefix.endswith("wo_a")` (`:71-77`), and the same tree already uses the guarded idiom `getattr(hf_config, "o_groups", 0)` at `ops/linear.py:455`. |
| 4 | `quantization/fp8_config.py:77` | `self.ignore` is assigned from the checkpoint's `ignore` list and **read nowhere** in the tree: `get_quant_method` (`:113-133`) sends every `LinearBase` to `ds_linear`. The two ignore-aware paths are upstream `Fp8Config` (`…/quantization/fp8.py:155-172,180-185` → `is_layer_skipped(...)` → `UnquantizedLinearMethod()`) and `quantization/compressed_tensors_config.py:292,443` (`should_ignore_layer` + `hf_to_vllm_mapper.apply_list`). |

`NPUPlatform.supported_quantization` (`platform.py:140-145`) advertises `fp8` and
`deepseek_v4_fp8`, which is why `--quantization fp8` is accepted at parse time and why
no flag permutation escapes.

## Rules

1. **Read the checkpoint's `quantization_config` before any source read.**
   `uv run --no-project python -c "import json,sys;print(json.load(open(sys.argv[1]))['quantization_config'])" <model dir>/config.json`
   prints the three keys that decide the whole report: `quant_method` (`fp8` → this
   page; `compressed-tensors` → its own path), `weight_block_size`, and
   `ignored_layers` / `modules_to_not_convert`. Without it, "does the DS block-128
   scale reorg fit this checkpoint?" cannot be answered, and the stopgap below is not
   assessable.
2. **The device line decides which of two reports this is.** Same checkpoint, A5 →
   this AttributeError; A2/A3 → detection returns `None`, the checkpoint's own `fp8`
   name survives to the platform gate and raises
   `ValueError: fp8 quantization is currently not supported in npu`
   (`quantization-method-gate.md` §E, recorded #119). Read the image/soc line and the
   raise site together; do not carry the A5 verdict onto a 910B report.
3. **`--quantization fp8` is not an escape.** The name resolves to the same
   `AscendFp8Config` (`fp8_config.py:68`), so flag permutations (`--dtype`,
   `--enforce-eager`, TP/DP size, graph mode) are spent turns. Say "no serve
   flag fixes this" early and give the two routes that do work at the pin: serve the
   BF16 checkpoint, or re-export with msmodelslim (`quant_model_description.json`
   present → detection case 1 returns `ascend`, and the fp8 config is never built).
4. **A `getattr(hf_config, "o_groups", 0)` guard removes the exception, not the
   defect.** `process_weights_after_loading` (`methods/fp8.py:56-70`) re-blocks *every*
   linear's `weight`/`weight_scale` for the DS block-128 layout regardless of family,
   so after the guard the engine can start and silently miscompute. **Acceptance is one
   real request whose output matches a BF16 serve of the same weights, never a healthy
   port** — state that in the answer and do not present the guard as the fix.
5. **The dropped ignore list is a second, independent defect.** The reporter's failing
   layer is `visual.merger.linear_fc1`, which their `config.json` keeps in BF16 and
   lists in `ignored_layers`/`modules_to_not_convert` (their claim; the checkpoint is
   not in the case view): with defect 4 patched, this layer is still treated as
   quantized. The repair mirrors the two references above
   (`should_ignore_layer(prefix, ignore=hf_to_vllm_mapper.apply_list(self.ignore),
   fused_mapping=self.packed_modules_mapping)` before scheme selection — the mapper
   matters because the checkpoint names layers in HF form).
6. **Ownership is code, and the durable repair is the registration scope.** Filing the
   fix against `methods/fp8.py` alone leaves the next native-FP8 model on the same
   route: scoping `AscendFp8Config` to `deepseek_v4_fp8` and leaving the generic `fp8`
   name to a family-agnostic FP8 linear scheme is what stops detection from handing any
   `quant_method: fp8` checkpoint to a DeepSeek-V4 MLA scheme. In #14467 the reporter's
   later resolution is exactly a code change — the issue closed on a linked PR
   (#14852), which is a link only: its contents are not in the case view, so do not
   claim which of the two defects it fixes.

## Limits

- Line numbers are pin-scoped; the reporter's paste drifts from the tag by a few to
  ~30 lines (`platform.py:90-95` vs `:140-146`, `fp8_config.py:125` vs `:130`) while the
  `methods/fp8.py:45` frame matches exactly — say "same code path, near-but-not-identical
  image", do not re-derive the tree from a paste's numbers.
- A claim about "current main" (e.g. that this path now raises `NotImplementedError` at
  `fp8_config.py:34-40`) is unverifiable in a one-commit-per-repo case view; at the pin
  those lines are inside `remove_quantization_method()`. Say so instead of adopting it.
- The MoE half is DS-specific too: `get_quant_method` sends `FusedMoE` layers to
  `w4a8_moe` (`fp8_config.py:126-132` → `AscendW4A8MXFPDSDynamicFusedMoEMethod`,
  `methods/fp8.py:81+`), i.e. mxfp4 experts from an FP8 checkpoint. Out of scope for a
  dense model, open for a MoE native-FP8 sibling, and unverified against any run.
