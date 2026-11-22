# Quantized-checkpoint conversion (msmodelslim / msit) vs the engine

## Scope

Reports whose symptom appears while a model is being *converted*: a per-layer
`MSE loss` table printed by an msmodelslim example script, one layer whose loss is
orders of magnitude above its neighbours, a bad `QuantModel` artifact, or "the
quantized checkpoint answers garbage" with no engine-side failing flag. Inherit
`index.md`; this page owns the phase *before* the engine starts, so lanes A/B/C and
the harness-vs-engine checklist only apply once the checkpoint's own quality is
settled. If the converted artifact *loads and serves* but prices and behaves like
the BF16 model (same throughput, a whole layer class with no quant kernel), that is
a coverage question owned by
`../performance-throughput/quantized-artifact-layer-coverage.md`, not a conversion
quality question.

## Rules

1. **A per-layer loss printout is converter output, not a vllm-ascend log line.**
   vllm-ascend has no conversion entry point — it consumes the checkpoint through
   `--quantization ascend` (`docs/source/user_guide/feature_guide/quantization.md:5`
   at v0.13.0rc1 `1b5d5abf`) and its quantization modules implement only the
   inference path (`vllm_ascend/quantization/w4a4_flatquant_dynamic.py`). The owner
   of that number is Ascend/msit on gitcode. Say so before analysing it, and name
   which component must receive the report at the end.
2. **A garbled artifact is a provenance question before it is an engine question.**
   When the symptom is output text — 乱码, fluent-but-wrong, one token repeated — on a
   serve whose weights the reporter built themselves, the first ask is *who converted
   them and with which algorithm*, not a walk of `vllm_ascend/quantization/*`: the
   engine reader knows one exporter's layout, and the exported numbers are whatever the
   converter's tuning decided. Recorded #2318 (Qwen3-32B W8A8 per-token, v0.9.1rc2,
   `--quantization ascend`, TP4, "works in Transformers, garbled in vLLM"): the
   reporter's own resolution was that the model had been quantized with **m4 (an
   improved-SmoothQuant variant)** — a conversion-recipe fault, no engine defect. Two
   things follow. "Works in Transformers" is not the control it looks like: the same
   tensors dequantized in PyTorch are not the numerics the NPU kernel executes, so it
   acquits the *weights*, not the export. And the recipe **selection** matters more than
   the tool version, so ask for the exact conversion command and its tuning /
   anti-outlier choice beside `pip show msmodelslim` and `git describe`. The cheap
   discriminator is the published-checkpoint split below — a *published* artifact of the
   same family, one command on the same image: garbage there too reopens the engine,
   sane output localises the defect to the reporter's conversion run. Cost of skipping
   this: a recorded run on this shape read `vllm_ascend/quantization/{quant_config,
   quantizer,w8a8_dynamic}.py`, the custom `models/qwen3.py`, the docs and the pin's
   test tree (~1.0 M cumulative prompt tokens) and was cut by the output cap with no
   answer at all — the engine cannot validate an artifact it did not produce.
3. **Two gates, both in the docs, both required.** The W4A4 tutorial states the
   engine gate and the quantizer gate in one sentence: "W4A4 is supported since
   `v0.11.0rc1`. For modelslim, W4A4 is supported since
   `tag_MindStudio_8.2.RC1.B120_002`" (`docs/source/tutorials/Qwen3-32B-W4A4.md:6`,
   clone at `:43`). A reporter who quotes only the engine version — the common case,
   since that is what `pip show vllm-ascend` prints — has satisfied half the tutorial.
4. **The doc set pins different msit branches on different pages.** The W4A4
   tutorial pins `tag_MindStudio_8.2.RC1.B120_002` (`Qwen3-32B-W4A4.md:6,43`), while
   the general guide pins `br_release_MindStudio_8.1.RC2_TR5_20260624`
   (`quantization.md:14-15`, repeated in the FAQ at `:102` and `:106`). Same
   documentation tree, two branches. Ask which page was followed and have the
   reporter run `cd msit && git describe --tags --always` plus
   `pip show msmodelslim`. Do not declare one branch "the correct one": which
   applies is per-model and version-specific, and the converter source is not in the
   case export (see Limits).
5. **Split ownership with the published checkpoint before any converter reasoning.**
   The tutorial links a pre-converted artifact
   (`https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4`,
   `Qwen3-32B-W4A4.md:39`). Serving that on the reporter's own image is one
   command. If it answers sensibly, the engine, the version gate and the serve flags
   are clean and the whole defect lives in their conversion run; if it also fails,
   the engine side is reopened. This is cheaper than any source read and needs no
   access to msit.
6. **Localize by shape invariance, not by log volume.** All decoder layers of one
   model share the same `input_size`/`output_size`, and the W4A4 reader derives its
   transform dims from `input_size` alone (`get_decompose_dim`,
   `w4a4_flatquant_dynamic.py:34`, used at `:107`). A layout or factorization bug
   would therefore corrupt *every* layer equally. Exactly one layer being bad points
   at that layer's calibration data or tuning numerics, not at the checkpoint
   layout. The same argument applies to epoch 0: if the first-epoch loss is already
   blown up, the learning rate cannot be the cause — the initial transform/scale is.
7. **The engine does not validate the tuned values.** `get_pertensor_param`
   allocates `left_trans`/`right_trans`/`clip_ratio` empty
   (`w4a4_flatquant_dynamic.py:105-117`) and the loader fills and registers them
   as-is, taking `clip_ratio.item()` as the operator's clip ratio (`:186-190`). The
   module's two `ValueError`s only check format and shape consistency —
   `input_size % 8` (`:95`) and `left_dim * right_dim == in_features` (`:152`) —
   never whether the tuned values are numerically sane. So "the conversion loss was
   high" is not cosmetic: a layer whose transforms were never tuned ships and is
   used, and a dimension-consistent but wrong artifact passes every check here.
8. **Loss is a proxy; ask for output.** No generation sample means no evidence about
   the artifact's usability. Request it, and treat the loss table as the localizer
   only.

## What to ask for

- Full per-layer log (all layers, not the failing slice) and the epoch-0 vs final
  loss for the first ~8 layers, so "one bad layer" vs "systematically bad" is
  decidable.
- `git describe --tags --always` in the msit clone, `pip show msmodelslim`, and the
  exact `w4a4.py` command line: `--calib_file` (the tutorial's is relative to
  `example/Qwen`, `Qwen3-32B-W4A4.md:63`), `--batch_size` (`:65`),
  `ASCEND_RT_VISIBLE_DEVICES` (tutorial: two idle cards, `:59`).
- The transformers pin the tutorial requires (`transformers==4.51.0`, `:51`) and the
  reporter's actual version.
- One generation from the converted checkpoint (`--quantization ascend`), plus the
  same prompt on the published ModelScope artifact.

## Recorded case (#5585, vllm-ascend v0.13.0rc1 pinned, reporter on 0.12.0rc1)

Qwen3-32B converted to W4A4 with msmodelslim: layer 2 epoch-0 loss `136268` against
`0.002` for layers 0/1/3, still `2.29` after 15 epochs; layer 6 also unstable. The
`N/64` counter matches Qwen3-32B's 64 decoder layers, so the index is not off by one.
No engine failure and no generation sample in the report. Following the rules above:
the counter format and the missing quantizer version make this a converter-side
report to Ascend/msit, with the published checkpoint as the one command that proves
the vllm-ascend side innocent.

## Recorded case (#2318, vllm-ascend v0.9.1rc2 = reporter's own version)

Qwen3-32B, a self-quantized `…-w8a8pertoken` directory, `vllm serve --quantization
ascend -tp 4` on 4 devices, garbled output while the same weights ran in Transformers.
The report carried no engine-side failure and no conversion log. The reporter's later
comment closes it on the conversion side: the model had been quantized with **m4, an
improved-SmoothQuant variant**, which is what produced the garbage — not the engine,
the flags, or the card. The lesson is rule 2's order of evidence: this artifact could
not be validated by any amount of `vllm_ascend/quantization/*` reading, because the
engine never chose those numbers.

## Limits

- `Ascend/msit` / msmodelslim is **not** in the `SOURCES.json` export for either
  recorded case (#5585, #2318), so the converter's loop, initialization, clip search
  and anti-outlier method menu cannot be verified here — including whether "m4" in
  #2318 denotes the reporter's SmoothQuant variant by that name or a tool option.
  Every mechanism claim about `w4a4.py` is a hypothesis by construction; say that in
  the answer rather than implying a source read.
- The pinned tree is the tutorial's version, not the reporter's image; the
  `conflicting_versions` marker in `SOURCES.json` means the doc line numbers must be
  re-checked in the reporter's own tree one release at a time.
- Whether the two msit branches differ in `w4a4.py` numerics was not verified
  anywhere; only the pin discrepancy is verified.
