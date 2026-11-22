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
| 2 | `quantization/utils.py:132-133` | `detect_quantization_method` returns `fp8` for *any* directory whose `quantization_config.quant_method == "fp8"` **and** `get_ascend_device_type() == AscendDeviceType.A5` (A5 is `soc_version == 260`, `utils.py:857-858`). So the auto-detection itself routes the model into the Ascend class; the model family never enters the test. **The device half is a v0.23.0-final line: at v0.23.0rc1 (`f4a08bd`) the same branch (`utils.py:130-133`) has no device test at all**, so at an rc pin an A2/A3 report takes this crash instead of §E's refusal. |
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
   raise site together; do not carry the A5 verdict onto a 910B report. **Grep the gate
   for the reported ref before applying either half**: the A5 test is absent at
   v0.23.0rc1 (`f4a08bd`, `utils.py:130-133` — the fp8 branch there is unconditional),
   so on an rc an A2/A3 report reaches the `o_groups` crash, not §E.
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

## A model-specific rescue patch is re-armed by the platform's own auto-detect

Recorded **#12685** (v0.23.0rc1 `f4a08bd`, `MiniMax-M2.7` served from a directory whose
`config.json` declares `quantization_config.quant_method: "fp8"`, TP4 / DP4 / EP): the pin
**ships a rescue for exactly this model** and still dies in `ds_linear`, because the two
hooks run in opposite order.

```text
vllm  config/model.py:747          ModelConfig.__post_init__ → self._verify_quantization()
  → patch/platform/patch_minimax_m2_config.py:74,133   forces cfg.quantization = None when
      device_name == "npu" and model_type == "minimax_m2" and quantization == "fp8"   (:61)
vllm  config/vllm.py:1335          VllmConfig.__post_init__ → current_platform.check_and_update_config(self)
  → vllm_ascend/platform.py:465    maybe_auto_detect_quantization(vllm_config)
  → vllm_ascend/quantization/utils.py:120-133  re-reads <model dir>/config.json → "fp8"
  → vllm_ascend/quantization/utils.py:199      model_config.quantization = "fp8"; quant_config rebuilt
```

No `--quantization` value escapes that: `fp8` is cleared at verify and re-armed, `none` /
`None` is not a method (`vllm/config/model.py:1075-1080`, `Unknown quantization method`),
and any other name is **respected** by the auto-detect (`utils.py:186-195` warns and
returns), leaving the checkpoint with a quantizer it cannot load. `--hf-overrides
'{"quantization_config": null}'` does not reach it either — detection `open()`s the model
directory's file itself. The general shape outlives this model: **a downstream patch on a
`ModelConfig` verifier is undone by whatever `check_and_update_config` does later**, so when
a patch that should have rescued a case looks absent, compare its hook site with the
config-construction order before concluding its gate missed.

Two checks, in this order:

1. `grep -n "Disabling fp8 quantization" <full log>` — the line is the patch's own
   `logger.info` (`patch_minimax_m2_config.py:31-35`, *Detected fp8 MiniMax-M2 checkpoint on
   NPU. Disabling fp8 quantization and loading dequantized bf16 weights instead.*).
   **Present** + the `o_groups` crash ⇒ re-armed (this section). **Absent** ⇒ the gate
   missed: `_get_model_type` (`:38-58`, `model_arch_config` → `hf_text_config` →
   `hf_config` → `cfg.model_type`) did not return `minimax_m2` for that checkpoint.
2. The serve line plus `<model dir>/config.json`. A launch with **no** `--quantization`
   beside `Using the vLLM Ascend fp8 Quantization now!` already says the file armed it —
   which is how #12685's later public thread settled the fork the recorded run was still
   preparing to ask for (the reporter posted the full command: no `--quantization`,
   `--additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'`,
   TP4/DP4/EP, KV store on mooncake, then 已成功拉起服务 after a modification).

From the reporter's side the route that works is a **copy** of the model directory with
`quantization_config` removed (detection falls to its Case 3, `utils.py:137`), and it is
safe **only if the weights really are BF16** — census the shards for `*_scale_inv` /
`weight_scale_inv` first, because for a genuine fp8 checkpoint the intended state at this
pin is the opposite one (`quantization=None` **while** the file still declares fp8): the
worker's dequant iter (`patch/worker/patch_minimax_m2.py:98-101,172-181`) gates on
`self.config.quantization_config.quant_method == "fp8"`, so stripping the key turns the
dequant off and loads fp8 bytes as bf16. The public record of #12685 does not say which
side its eventual modification was, so keep the ranking (file declares fp8 → class armed)
and do not claim the fix.

Coverage at the same pin: `docs/source/tutorials/models/MiniMax-M2.md:23-28` lists only the
w8a8 / w8a8c8 quantized M2.7 and M2.5 weights (plus the Eagle3 heads) and every serve block
carries `--quantization ascend` (`:202`, `:270`, `:391`, `:454`), so a **BF16 M2.7 recipe is
a coverage gap there, not a refusal of the model** — say which, and do not build the
"unsupported model" verdict on it.

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
