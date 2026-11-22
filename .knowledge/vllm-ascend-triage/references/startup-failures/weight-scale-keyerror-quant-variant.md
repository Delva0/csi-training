# Load-phase `KeyError: '<param>.weight_scale'` / `.weight_offset`, or `ValueError: There is no module or parameter named '<module>.<aux>'`: the checkpoint's quant aux tensors vs the engine's quant config

Lane B, every rank, after `Starting to load model` (the shard bar is usually
visible). The traceback's deepest frames are the model loader:

```text
(Worker_PP0_TP7_EP7 pid=125) ERROR ... vllm_ascend/models/deepseek_v4.py", line 1535, in load_weights
    param = params_dict[name]
KeyError: 'model.layers.2.self_attn.indexer.wq_b.weight_scale'
```

and the other ranks name *different layers* from a **small set of module
families** — `self_attn.wkv`, `self_attn.indexer.wq_b`, occasionally
`self_attn.compressor.wkv` — with `Multi-thread loading shards: 2% Completed |
1/46` above the first traceback. Recorded #12675 (image v0.23.0rc1,
DeepSeek-V4-Flash, TP8 + `--enable-expert-parallel`, DP1/PP1).

This is the **mirror** of `quantization-method-gate.md` section B. There, the
checkpoint has int8 *weights* a dense model cannot hold; here the checkpoint
carries *scale*/*offset* tensors the engine built no parameter for. Classify on the
key: `.weight_scale` / `.scale` / `.weight_offset` (any quant **aux** tensor) → this
page; a plain `.weight` → the gate page. The `.weight_offset` family has its own
reading rules — section below.

The same family has a **shape** form with no missing key at all — a loader `copy_`
raising `RuntimeError: The expanded size of the tensor (P) must match the existing size
(C)` because the engine allocated the aux param with a parallel divisor the branch it
took never shards → `quant-aux-param-tp-shape-mismatch.md` (recorded #7088, W4A8 MoE
`w2_scale_bias`). Classify on the deepest frame (`param = params_dict[name]` vs
`expert_data.copy_(loaded_weight)`) before opening the loader.

## What the pin does (verified at v0.23.0rc1 `f4a08bd`, one-commit snapshot)

1. `vllm_ascend/models/deepseek_v4.py:1381-1382` renames every checkpoint tensor
   ending `.scale` to `.weight_scale` before the lookup, so a modelslim export
   (`…wkv.weight_scale`) and a native one (`…wkv.scale`) meet the same
   `params_dict` name.
2. There is **no** `if name not in params_dict: continue` guard: `:1535`
   (`param = params_dict[name]`) subscripts directly, and the only skips in that
   loop are `.bias` and PP-missing parameters, plus the FP8 kv-scale remap at
   `:1528`. Any extra scale tensor the model does not own is therefore a hard start
   failure, not a warning.
3. A `weight_scale` parameter exists only if that linear was built with a quant
   method that allocates it. At that ref `Attention.wkv` and `Indexer.wq_b` pass
   `quant_config` unconditionally (`:758`, `:554`), so a missing `weight_scale`
   there is a statement about **the engine's quantisation**, never about those two
   modules being special-cased in code.

## The `.weight_offset` variant: the engine built that MoE unquantised (recorded #986)

Same lane, same two calls, plus one reading rule. Signature — image
`quay.io/ascend/vllm-ascend:v0.8.5rc1-openeuler`, `DeepSeek-R1-bf16-hfd-w8a8` from
modelers.cn, TP8 × PP2 + `--distributed-executor-backend ray`, `--enforce-eager`,
**no** `--quantization` flag, and the checkpoint's `quant_model_description.json`
merged into `config.json` by hand (the advice in #856):

```text
(... RayWorkerWrapper pid=12580) ... vllm/model_executor/models/deepseek_v2.py", line 788, in load_weights
    param = params_dict[name]
KeyError: 'model.layers.40.mlp.experts.w2_weight_offset'    # PP1 rank; the PP0 ranks say layers.23
```

1. **That name is the engine's parameter, not the checkpoint's.** Upstream asks for it
   by substring rewrite: `FusedMoE.make_expert_params_mapping` builds
   `experts.{expert_id}.{down_proj}.` and the loader does
   `name.replace(weight_name, "experts.w2_")`
   (`vllm/model_executor/layers/fused_moe/layer.py:542-557` and
   `vllm/model_executor/models/deepseek_v2.py:564-616` in the pinned vllm tree), so a
   per-expert tensor `…experts.40.down_proj.weight_offset` reaches `params_dict` as
   `…experts.w2_weight_offset`. **The absent expert index does not mean the checkpoint
   is a fused/stacked export**, and which of `_offset`/`_scale` a shard reaches first
   needs no explaining. The recorded run inferred a fused msmodelslim layout from
   exactly this, then walked quantizer/create_weights, and shipped **no answer** at
   the 16 384-token cap — that inference is the whole trap.
2. **What a missing offset means, per layer family** (vllm-ascend v0.8.5rc1
   `ec27af34`, the recorded pin — re-grep at another ref, the whole map is two classes
   and a `SUPPORT_ASCEND_QUANTIZER_TYPE` dict):

   | layer family | allocation site | registers `.weight_offset`? |
   |---|---|---|
   | linear, `W8A8` | `quantization/w8a8.py:75` (`AscendW8A8LinearMethod`) | yes |
   | linear, `W8A8_DYNAMIC` | `quantization/w8a8_dynamic.py:337` | yes |
   | MoE experts, `W8A8_DYNAMIC` | `w8a8_dynamic.py:417`, `:426` (`AscendW8A8DynamicFusedMoEMethod`) | yes — `w13_weight_offset`, `w2_weight_offset` |
   | MoE experts, `W8A8` (static) | none: `W8A8Quantizer` defines only `build_linear_method` (`quantization/quantizer.py:266-270`), so the inherited `VLLMAscendQuantizer.build_moe_method` (`:210-212`) raises `NotImplementedError("MoE method is not implemented for the current quant type.")` while `AscendFusedMoEMethod.__init__` builds the layer | never reached — it dies **before** `Starting to load model` |
   | MoE experts declared `FLOAT` | `AscendUnquantizedFusedMoEMethod` (`quantization/quant_config.py:96-99` through `is_layer_skipped_ascend`, `:104-129`) | no |

   The lookup the branch performs is `quant_description['…mlp.experts.weight']`
   (`is_layer_skipped_ascend`; upstream builds the MoE with
   `prefix=f"{prefix}.experts"` — `models/deepseek_v2.py:115-125` in the pinned tree,
   which also defines no `packed_modules_mapping`, so it is that one fused key;
   the pin is v0.6.6.post1, so re-read the construction site for the reporter's vllm). Four values, four
   outcomes, and only one is a load-phase offset `KeyError`: the key absent →
   `KeyError: '…mlp.experts.weight'` during construction; `W8A8` → row 4's
   `NotImplementedError`; `W8A8_DYNAMIC` → row 3 allocates both offsets;
   **`FLOAT` → unquantised experts, and the first `_weight_offset` tensor the loader
   reaches has no parameter.** So the description that was in effect marks the experts
   `FLOAT` while the checkpoint ships expert offsets — it does not describe these
   weights (wrong variant, or a hand-merged/renamed file).
3. **The description in effect is the artefact — ask, do not walk.** At that ref
   `AscendQuantConfig.get_config_filenames()` is exactly
   `["quant_model_description.json"]` (`quant_config.py:69-70`), while the tutorial's
   own converted tree lists `quant_model_description_w8a8_dynamic.json`
   (`docs/source/tutorials/multi_npu_quantization.md:65`, produced by the
   `--is_dynamic True` conversion at `:54`; its serve line `:78` carries no
   `--quantization` flag). Collect: `ls <model dir>`, the `config.json`
   `quantization_config` if the description was merged there, and the value of
   `model.layers.<failing>.mlp.experts.weight` — the documented recipe needs
   `W8A8_DYNAMIC` on the experts, not `W8A8` and not `FLOAT`.
   **Which of the two inputs the run actually used is one line of code, and it
   decides what to ask for.** At this ref a `config.json` `quantization_config`
   block short-circuits the file entirely: `override_quantization_method` returns
   `"ascend"` whenever `torch.npu.is_available()`, without reading the block's
   `quant_method` (`quant_config.py:77-81`), and `from_config` hands that dict
   straight in as the description (`:72-74`) — so a hand-merge counts only when the
   description itself is nested under `quantization_config`, and *any* foreign quant
   block (compressed-tensors', a stale one) selects `ascend` too, after which
   `is_layer_skipped_ascend`'s `quant_description[prefix + '.weight']` (`:104-131`)
   raises a `KeyError` at **construction**, not the load-phase offset `KeyError` in
   the paste. With no quant block it falls back to the file, so the name must be
   exact (`quant_model_description_w8a8_dynamic.json` ends differently and is not
   discovered). Both readings keep the same ask; an expert-`_offset`-only paste leans
   to the built-config one (with no config at all the dense/attention `_weight_scale`
   names are missing too and `deepseek_v2.load_weights` has no guard that skips them),
   but shard order and threaded loading decide which key is printed first — the
   `mlp.experts.weight` value is what settles it.
4. **One log line says which quantizer decided.** `Using the vLLM Ascend Quantizer
   version now!` (`quantizer.py:88`) is printed only when `mindie_turbo` is *not*
   importable (`:50-53`). Without that line a `MindIETurboQuantizer` in the image owns
   every decision above and the table does not apply — say so and let `pip show
   mindie-turbo` (or the image's file list) settle it instead of deriving a layout.
5. Either way state the ask: at that ref nothing converts a `FLOAT`/static-`W8A8`
   expert block into the dynamic method, so "re-convert with `--is_dynamic True` (or
   serve the modelers.cn *w8a8_dynamic* directory)" and "send the description entry"
   are the deliverable, not a patch.

## The VL-MoE offset form, and the route when no traceback reaches the paste (recorded #7744)

Verified at vllm v0.14.1 `d7de043` ↔ vllm-ascend v0.14.0rc1 `52d4acf` (Atlas 800I A2
×8, `Qwen3-VL-235B-A22B-Instruct-W8A8`, TP8 + `--enable-expert-parallel`,
`FULL_DECODE_ONLY`, `--mm_processor_cache_type shm`). The reporter's later log resolves
**line for line** against that pin:

```text
(Worker_TP3 pid=2777) ERROR … [multiproc_executor.py:749] WorkerProc failed to start.
(Worker_TP3 pid=2777) ERROR … [multiproc_executor.py:749]   File "…/vllm/model_executor/models/qwen3_vl.py", line 2083, in load_weights
(Worker_TP3 pid=2777) ERROR … [multiproc_executor.py:749]     return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
   … models/utils.py", line 335, in load_weights   → set(self._load_module("", self.module, weights))
   … model_loader/online_quantization.py", line 173, in patched_model_load_weights → return original_load_weights(…)
   … models/qwen3_vl_moe.py", line 279, in load_weights
       param = params_dict[name_mapped]
KeyError: 'layers.37.mlp.experts.w2_weight_offset'
```

1. `qwen3_vl_moe.py:279` is the **non-fused** expert branch's subscript, and its guard
   one group above (`:272-278`) skips only `ignore_suffixes`, defined at `:173-184` as
   `.bias / _bias / .k_scale / _k_scale / .v_scale / _v_scale / .weight_scale /
   _weight_scale / .input_scale / _input_scale`. **No `*_offset` member** — so a
   scale-only export is skipped silently while an offset-carrying W8A8 export reaches
   the subscript. That is why the quoted key is an `_offset` and not a `_scale`, and it
   is the model-file half of the family table above.
2. **The `online_quantization.py` frame says nothing about online quantisation being
   on.** `support_quantized_model_reload_from_hp_weights` decorates
   `AutoWeightsLoader.load_weights` itself (`models/utils.py:321`), and on the first
   (or offline-quantised) run `patched_model_load_weights` returns
   `original_load_weights(...)` at `:173`. It is the standard wrapper at this ref, not
   evidence about the checkpoint.
3. **Resolution here is a tag.** The same weights and the same serve line worked on
   `v0.17.0rc1` (reporter-confirmed), and v0.14.0rc1's matrix leaves the Qwen3-VL-MOE
   W8A8 cell empty (`docs/source/user_guide/support_matrix/supported_models.md:81`) —
   upgrade is rung 1, the mechanism below it.

**When the traceback never reaches the paste** (a screenshot whose OCR is noise, a
title-only report), the lane is still decidable in two calls: read the serve path for a
quant suffix (`…-W8A8`, `…-w8a8_dynamic`) and the matrix cell for that quant, then ask
for the block **below** `WorkerProc failed to start.` in one rank's log (`logger.exception`
prints it under that same `[multiproc_executor.py:749]` prefix) plus
`scripts/checkpoint-key-prefix.py --scales <model dir>`. Do not fill the gap with the
nearest unusual flag — in #7744 that was `--mm_processor_cache_type shm`, whose page
requires its own `Data for address:id …` error.

## The `ValueError` form: the autoloader names a module that has no quant aux home (recorded #7200)

Same family, other raise site, and the message itself carries the classification:

```text
(Worker_DP0_TP3_EP3 pid=1596) ERROR ... vllm/model_executor/models/kimi_k25.py", line 478, in load_weights
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
(Worker_DP0_TP3_EP3 pid=1596) ERROR ... vllm/model_executor/models/utils.py", line 326, in _load_module
    raise ValueError(msg)
ValueError: There is no module or parameter named 'vision_tower.encoder.blocks.0.mlp.fc0.weight_offset' in
KimiK25ForConditionalGeneration. The available parameters belonging to
vision_tower.encoder.blocks.0.mlp.fc0 (AscendColumnParallelLinear) are:
{'vision_tower.encoder.blocks.0.mlp.fc0weight', 'vision_tower.encoder.blocks.0.mlp.fc0bias'}
```

1. **Every field of that sentence is computed — none of it is checkpoint text.** The
   raise is upstream's `AutoWeightsLoader._load_module` (vllm v0.16.0 `89a77b1`:
   `vllm/model_executor/models/utils.py:249`, `raise ValueError(msg)` at `:326`, reached
   from `load_weights` at `:329-342`), i.e. the *common* loader — the model file's own
   `load_weights` only forwarded to it, so its frame is not the owner. The parenthesised
   class is the **runtime** class: vllm's CustomOp swaps `ColumnParallelLinear` for
   `AscendColumnParallelLinear` at registration (`vllm_ascend/utils.py:634`, inside
   `REGISTERED_ASCEND_OPS`), so a model file that writes `ColumnParallelLinear` still
   reports the Ascend subclass — it says nothing about the checkpoint and nothing about
   a plugin having replaced a layer. The `{…}` set is `base_prefix + k` with no
   separator inserted, so `'…fc0weight'` is `prefix` + `weight`, not a name missing a dot.
2. **The aux suffix on the quoted name is the diagnosis.** `.weight_offset` /
   `.weight_scale` / `.deq_scale` / `.quant_bias` after a module path means the export
   quantised that module while the module was built with **no** quant method: the
   message's own parameter set is `weight`/`bias` only. Same two-sided question as the
   KeyError form above read from the other end — there a downstream `params_dict[name]`
   misses an engine name, here the upstream loader meets a checkpoint tensor for a module
   whose `named_parameters()` has no home for it.
3. **Which side over-reached is one artifact: the export's declaration.** At #7200's pin
   (vllm v0.16.0 `89a77b1` + vllm-ascend v0.16.0rc1 `a78a00e`) `kimi_k25.py:352` reads
   `vllm_config.quant_config`, but `:362-365` builds
   `MoonViT3dPretrainedModel(config.vision_config, prefix=…)` with no `quant_config`,
   `kimi_k25_vit.py:544-547` has no such parameter, and `:313-326` builds the MLP's
   `fc0`/`fc1` without it — the tower is unquantised **by construction**, so every
   `vision_tower.*` aux key is homeless. The checkpoint's
   `quantization_config.config_groups.group_0.targets` (or the description's per-module
   method) says what the converter covered; a blanket `["Linear"]` covers a ViT MLP whose
   model half never accepts a quant method. The repo documents the narrowing step for a
   Kimi sibling — `docs/source/tutorials/models/Kimi-K2-Thinking.md:74` edits that field
   from `["Linear"]` to `["MoE"]` — so ask for that block before offering a patch.
4. **Never offer "delete the aux tensors".** In w4a8 the packed `weight` already holds two
   int4 values per int8 (`vllm_ascend/quantization/methods/w4a8.py:60-68`,
   `_packed_factor=2`) and the aux set is allocated per group (`:78-88`: `weight_scale`,
   `weight_offset`, `weight_scale_second`, `weight_offset_second`, `scale_bias`).
   Dropping only the offsets makes the start succeed and feeds packed integers to a bf16
   MLP — the exception-only fix. Either re-export (or take the vendor's narrower variant)
   so the tower stays bf16, or plumb `quant_config` into the tower upstream; acceptance is
   one image request compared against a bf16 tower's answer, not "it starts".
5. **Ask for the census, not the layout**: `scripts/checkpoint-key-prefix.py --scales
   <dir>` grouped by family shows whether the whole tower is quantised (`patch_embed`,
   `attn.wo`, every `blocks.*.mlp`), which decides "exclude one MLP" vs "re-export".

Scope: the raise site, the CustomOp class map and the Kimi constructors are verified at
that pin; the ValueError form itself is generic (any model file that builds a submodule
without `quant_config` — ViT towers and multimodal projectors are the common ones), so
re-read that constructor at the reporter's ref before quoting lines. If upstream later
passes `quant_config` there, this error disappears and the question becomes which variant
the export should carry.

## Procedure — two calls before any source walk

1. **Get the contract first: the exact `vllm serve` line (or launch script) and
   the served model directory.** In the recorded case that alone decided it; the
   investigation instead spent its whole budget in `modelslim_config.py` and never
   asked.
2. **Census the checkpoint's quantisation tensors** — one call, header-only, no
   device, reuses `../../scripts/checkpoint-key-prefix.py`:
   `uv run --no-project python ../../scripts/checkpoint-key-prefix.py --scales <model dir>`
   It prints every `.scale` / `.weight_scale` / `.weight_scale_inv` /
   `.weight_offset` / `.quant_bias` key grouped by module family, plus whether
   `quant_model_description.json` and a `quantization_config` in `config.json`
   exist. Read the **family count**, not the tensor count:
   - scales across the main linear/MoE families (`self_attn.wq_a`/`wq_b`/`wo_a`/
     `wo_b`, `mlp.gate`, `mlp.experts.*`) and/or a declared quantisation → a
     quantised export; the question is which method the engine selected
     (`--quantization ascend` present? description file matching?) →
     `quantization-method-gate.md` section B.
   - scales confined to a **handful of sub-projections** (`self_attn.wkv`,
     `self_attn.indexer.wq_b`, `compressor.wkv`) with nothing declaring a
     quantisation → the directory is a **full-precision release** whose few
     FP8-stored projections ship a per-tensor scale. The engine builds those
     linears unquantised, the loader meets the scale first and raises. Nothing in
     the reporter's tree needs patching: the served weights are not the variant the
     release documents.
   - **no** scale tensors at all → the name in the traceback came from another
     tree: re-check the served path, the bind mount (`vllm serve` on a mount root
     instead of on the checkpoint directory) and any stale cached copy.
3. **Compare against the tutorial's weight list, not a sibling model's command.**
   `docs/source/tutorials/models/DeepSeek-V4-Flash.md:24-28` lists **one** weight
   variant — `DeepSeek-V4-Flash-w8a8-mtp` (quantised) — and every documented
   `vllm serve` line (`:161`, `:209`, `:435`, `:505`, `:691`, `:769`) carries
   `--quantization ascend`; the PR e2e test uses
   `gdydems/DeepSeek-V4-Flash-w4a8-mtp` with `quantization="ascend"`
   (`tests/e2e/pull_request/four_card/test_deepseek_v4.py`). Serving a plain
   `deepseek-ai/DeepSeek-V4-Flash` directory
   (`-v /data/models/DeepSeek-V4-Flash:/root/.cache` + `vllm serve /root/.cache`)
   with no `--quantization` flag sits outside that list — that was the recorded
   cause.

## Traps this page exists for

- **A different failing layer on every rank is not a per-layer code bug.** The
  loader is multithreaded/prefetched (`--safetensors-load-strategy prefetch`,
  `--model-loader-extra-config '{"enable_multithread_load": "true", …}'`), so each
  rank dies on whichever unmatched key its threads reach first. The stable fact is
  the set of **module families**, not the layer index — say so instead of reading
  the spread as layer-specific behaviour.
- **Do not infer "so exactly these two modules are skipped in code" from a short
  key set.** That inference is what pulls an investigation into
  `modelslim_config.is_layer_skipped_ascend` (`:681-708`),
  `create_scheme_for_layer` (`:417`) and `quant_prefix_mapper` (`:569-602`): real
  code, but load-bearing only *after* step 2 shows scales on the main families.
  Taken in the wrong order it explains a modelslim export to a reporter who is not
  serving one.
- The A5 branch `quant_config=None if get_ascend_device_type() in
  {AscendDeviceType.A5}` (`deepseek_v4.py:626`, `:634`) belongs to the
  **compressor's** `wkv`/`wgate`, so it produces a key containing
  `.compressor.wkv`. The key text tells you whether that branch is in play.
- The raise is load-phase: `--enforce-eager`, `--max-model-len`,
  `--compilation-config` and every graph knob are irrelevant, and a rank that never
  printed `Loading model weights took` never issued a device op.

## Scope and limits

Verified at vllm-ascend v0.23.0rc1 (`f4a08bd`): the remap site, the unguarded
`params_dict[name]`, and the two `quant_config` sites. The recorded case's launch
script arrived from the reporter after the investigation. The native checkpoint's
exact FP8 sub-projection list is **not** in the snapshot — step 2 prints it from the
reporter's own directory, so report "the census shows X", never a recalled list. Not
verified: whether this release accepts the full-precision variant at all (no
conversion path for it is documented).
