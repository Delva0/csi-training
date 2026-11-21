# Worked case: `attention_type` missing on an Ascend-patched decoder layer

Issue vllm-ascend #11661, image `quay.io/ascend/vllm-ascend:v0.22.1rc1`,
model DeepSeek-OCR-2. Reporter's question was literally "please confirm whether
this problem exists" — a yes/no plus the owner, not a patch.

**Filed twice on the same image**: #11926 (2026-07-13) reaches the identical line
with the identical message at the same `v0.22.1rc1`, so this is reproducible on
that image rather than one host's accident — and that second run is also the
delivery lesson below: it had the decisive rule doc by its 29th turn and shipped
no answer at all.

## Symptom

```
File "/vllm-workspace/vllm-ascend/vllm_ascend/ops/qwen2_decoder.py", line 233, in forward
  attention_mask=causal_mask_mapping[decoder_layer.attention_type],
AttributeError: 'AscendQwen2DecoderLayer' object has no attribute 'attention_type'
```

## Evidence table (as recorded; vllm-ascend v0.22.1rc1 @ `5f6faa0cb883`,
vllm v0.22.1 @ `0decac0d96c4`)

| Claim | Evidence |
|---|---|
| The offending line indexes a dict with a per-layer attribute | `vllm_ascend/ops/qwen2_decoder.py:233` |
| The layer class is a thin subclass that calls the transformers `__init__` and adds no `attention_type` | `qwen2_decoder.py:249-255` (`class AscendQwen2DecoderLayer(Qwen2DecoderLayer)`, `super().__init__(config, layer_idx)`) |
| The model class replaces the layer list, then runs a re-implemented forward loop | `qwen2_decoder.py:169-175` (`AscendQwen2Model(Qwen2Model)`, `super().__init__(config)`, `self.layers = nn.ModuleList([...])`), loop at `:230-241` |
| Entry point: the whole thing is wired in as a replacement for the upstream `qwen2_decoder` plug-in layer | imported at `vllm_ascend/utils.py:668`; upstream class `vllm/model_executor/models/deepencoder2.py:22` (`@PluggableLayer.register("qwen2_decoder")`) |
| Upstream does **not** re-implement the loop: it hands the mask mapping to `super().forward(...)` | `deepencoder2.py:78-118` (`CustomQwen2ModelInner.forward` at `:79`, `super().forward(..., attention_mask=causal_mask_mapping, ...)` at `:104`) |
| transformers' own modeling rule says per-layer metadata comes from `self.config`, not from the layer object | `huggingface/transformers` export v5.12.1, `docs/source/en/modeling_rules.md:170`; the before/after snippet at `:178-179` shows `causal_mask_mapping[decoder_layer.attention_type]` → `causal_mask_mapping[self.config.layer_types[i]]` |
| Dependency pins differ between the two repos | vllm-ascend `requirements.txt:38` `transformers==5.5.4`; vllm `requirements/common.txt:10` `transformers >= 4.56.0, != 5.0.*, … != 5.5.0` |

The per-model transformers source is **not** in that export (`src/transformers/
models/` contained only `gpt2`, 1515 of 5776 files), so the doc rule above is the
evidence, not `modeling_qwen2.py`.

## Diagnosis to deliver

- Reporter-confirmed: the crash exists on image v0.22.1rc1; the layers built at
  `qwen2_decoder.py:172-174` do not expose `attention_type` in that image. Say
  this first — it is what was asked.
- Source-verified: the Ascend forward is a copy of the library body that reads a
  per-layer attribute removed by the library's own modeling rules, while upstream
  vLLM delegates to `super().forward()` and therefore tracks the library. The
  downstream copy is the owner.
- Cheapest fix for the reporter: use the image that works. Reporter subsequently
  reported (public comment, 2026-07-09) that `0.18.0` of the image deploys fine,
  and pinged twice more (2026-07-22, 2026-08-03) without a confirmed fix, so the
  defect was still open at the time of record — do not report it as fixed.
- Repair pattern for a patch (verify against the pinned pin before proposing):
  index `causal_mask_mapping` from `self.config.layer_types[i]` exactly as
  `modeling_rules.md:179` prescribes, or delete the copied loop and call
  `super().forward()` the way upstream does.

## Decisive check the reporter can run in their image

```bash
uv run --no-project python - <<'PY'
import transformers
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
cfg = Qwen2Config(hidden_size=896, num_hidden_layers=2)
layer = Qwen2DecoderLayer(cfg, 0)
print(transformers.__version__,
      "layer.attention_type:", hasattr(layer, "attention_type"),
      "config.layer_types:", getattr(cfg, "layer_types", None))
PY
```

A `False` for the attribute settles mechanism (b); `True` means the layer list in
the image is built differently from the pinned source (`runtime_commit_verified:
false`) and the question moves to which build the reporter actually runs.

## The reporter's follow-up: "按着文档 0.16.0 版本就可以支持 ocr2 模型启动了"

That is the pinned tree's own claim, so quote it and hand the disagreement back
rather than explaining it away: `docs/source/tutorials/models/DeepSeekOCR2.md:7`
— "first supported in `vllm-ascend:v0.16.0` and can stably run in v0.16.0 and
later version" — in the same tree whose `qwen2_decoder.py:233` raises on
v0.22.1rc1. Also diff their launch against that tutorial's serve block
(`DeepSeekOCR2.md:134-158`, `vllm serve` at `:144`): #11926's command is the block
**verbatim** except
`--async-scheduling` and a model path of `/root/.cache/` where the tutorial
writes `/root/.cache/DeepSeek-OCR-2`, so the flags are not the lever and telling
them to follow the tutorial wastes their time.

Coverage, quotable as coverage and never as cause: at the pin no test references
`qwen2_decoder`/`CustomQwen2Decoder` (`rg -ln 'CustomQwen2Decoder|qwen2_decoder'
tests/` returns nothing) and no e2e config names DeepSeek-OCR-2, so no CI path runs
this copy's mask loop — the two reports are the only exercise it gets.

## Delivery: the repair form is not worth a turn

Two forms exist and the export cannot rank them (`modeling_qwen2.py` is absent, so
whether `Qwen2Config` defaults `layer_types` is unverified): the TRF011 form
`causal_mask_mapping[self.config.layer_types[i]]`, and a local fallback
`causal_mask_mapping[getattr(decoder_layer, "attention_type", None) or "full_attention"]`
— exact only if the config has no sliding layers, because the wrapper that calls
this forward passes a one-key `{"full_attention": …}` mapping
(`qwen2_decoder.py:90-95`) while the forward itself can add a `sliding_attention`
key (`:222-223`). Ship both with the probe above as the decider. The recorded
#11926 run spent its last turn — and its whole remaining output budget — weighing
them, emitted no text and no tool call, and was cut: the choice is one line of
the ladder, not the verdict.

## Still open in the recorded case

- Whether transformers 5.5.4 itself ships the attribute — unresolved, because
  the per-model source is missing from the export. Do not assert a removal
  version.
- Why an older image works while a newer one fails (the copy could simply be
  absent or different in the older tree). Stated as a hypothesis, not a fact.
