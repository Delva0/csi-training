# Worked case: `attention_type` missing on an Ascend-patched decoder layer

Issue vllm-ascend #11661, image `quay.io/ascend/vllm-ascend:v0.22.1rc1`,
model DeepSeek-OCR-2. Reporter's question was literally "please confirm whether
this problem exists" — a yes/no plus the owner, not a patch.

**Filed three times on the same image**: #11926 (2026-07-13) and #11991 (2026-07-14)
reach the identical line with the identical message at the same `v0.22.1rc1`, so this
is reproducible on that image rather than one host's accident — and #11926 is also the
delivery lesson below: it had the decisive rule doc by its 29th turn and shipped no
answer at all. #11991 then supplies the fact the first two reports could not settle:
the reporter applied the **guard** form, and it **is not a fix** (see *The guard form
was tried and rejected* below).

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
- Repair pattern for a patch: delete the copied loop and let the installed body run
  (`super().forward()`), the way upstream does at `deepencoder2.py:104`. The other
  candidate, indexing `causal_mask_mapping` from `self.config.layer_types[i]` as
  `modeling_rules.md:179` prescribes, is the library's own rule but not safe *inside
  this wrapper* — the wrapper supplies only a `full_attention` key
  (`qwen2_decoder.py:90-95`). A `getattr` default is excluded by #11991 (below).

## Decisive check the reporter can run in their image

```bash
uv run --no-project python <skill_root>/scripts/qwen2-body-drift-probe.py \
    --config /data2/DeepSeek-OCR-2
```

One call in the reporter's interpreter, no NPU needed. It prints
`transformers.__version__` and its file, the checkpoint's `layer_types` /
`sliding_window`, whether a `Qwen2DecoderLayer` exposes `attention_type`, whether
`Qwen2Model.has_sliding_layers` exists, then the installed `Qwen2Model.forward` and
`Qwen2DecoderLayer.__init__` source. Read the two summary lines first:

- `layer.attention_type: False` + `INDEXES BY config.layer_types` → the copy is from
  another generation of the library, and the delegation is the repair (the guard only
  silences the raise).
- `layer.attention_type: True` → the layer list in the image is built differently from
  the pinned source (`runtime_commit_verified: false`), and the question moves to which
  build the reporter actually runs.

**The source dump is the check the truncated export can never be.** This snapshot's
`transformers` carries no per-model `qwen2/modeling_qwen2.py` (1515 of 5776 files), but
the reporter's own site-packages has it: ask for `inspect.getsource` on the installed
class instead of concluding "unverified". It also answers whether the body accepts a
mask-mapping dict at all, which is what makes a delegation safe.

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
this copy's mask loop — the three reports are the only exercise it gets.

## The guard form was tried and rejected (#11991)

#11991 is the third filing of this exact crash — same image, same line, same
DeepSeek-OCR-2 launch. The reporter first reported it solved with the local guard

```python
attention_mask=causal_mask_mapping[getattr(decoder_layer, "attention_type", "full_attention")],
```

and then retracted it in two public comments (2026-10-10): *"this modification only
allows the model service to start up normally, but it will cause problems with model
recognition"*, and, after an investigator proposed an attribute guard, *"It can't
solve the problem"*. Reporter-confirmed, so:

- **A guard is a rejected repair in this lane, not one of two equal options.** Do not
  ship `getattr(decoder_layer, '<attr>', <default>)` as the fix, and do not present
  "the service starts" as validation: the same reporter watched the service start and
  then get image recognition wrong. When a reporter says they applied a guard, the next
  thing to ask for is one image request's output, not the startup log.
- **The repair to name is the delegation**, which is what upstream vLLM does at the
  same call (`deepencoder2.py:104` hands the mask mapping to `super().forward()`):
  delete the copied loop in `AscendQwen2Model.forward` (`qwen2_decoder.py:180-245`) and
  let the installed library body consume the mapping. Keep the layer/norm swaps in
  `AscendQwen2Model.__init__` (`:169-175`) — the NPU mask itself is built by the
  wrapper, not by the copy.
- **Why the guard cannot be reasoned into correctness here**: the wrapper that calls
  this forward always passes a one-key `{"full_attention": …}` mapping
  (`qwen2_decoder.py:90-95`, passed at `:99`), so the copy's own dict-building branch
  (which can add `sliding_attention`, `:207-223`) never runs on this path. The TRF011
  form `causal_mask_mapping[self.config.layer_types[i]]` therefore raises
  `KeyError: 'sliding_attention'` as soon as the checkpoint declares a sliding layer,
  while the guard silently sends every layer to the OCR mask. Neither index form is
  verifiable from the tree, and the reported outcome says the copied body differs from
  the installed one in more than this one name — the same conclusion the upstream
  delegation states directly.
- If a reporter must patch in place to unblock a demo, label it **start-only
  diagnostic** and give the acceptance check with it: one image request whose
  recognition result is compared against a working image, plus the body diff below.

## Delivery: the form choice is not worth a turn

The recorded #11926 run spent its last turn — and its whole remaining output budget —
weighing the two repair forms, emitted no text and no tool call, and was cut. That was
a wasted turn even then, and #11991 has since decided the question: ship the delegation
as the repair, exclude the guard with the reporter's own words, and spend the saved
turn on the acceptance check.

## Still open in the recorded case

- Whether transformers 5.5.4 itself ships the attribute — unresolved, because
  the per-model source is missing from the export. Do not assert a removal
  version.
- Why an older image works while a newer one fails (the copy could simply be
  absent or different in the older tree). Stated as a hypothesis, not a fact.
- **What exactly a guard leaves wrong.** #11991 establishes the outcome (startup
  fine, image recognition broken), not the mechanism. Two readings stay open: the
  copied body builds or serves masks differently from the installed generation in ways
  beyond the one name, or the guard uncovered a second, still-unreported defect on the
  same path. The body diff above separates them; until it is run, say "the guard is
  rejected by the reporter's own output; the residual difference is unverified".
