# `AssertionError: Failed to apply prompt replacement for mm_items['<modality>'][0]` at engine init

Scope: a VL serve that dies **before any request and before KV sizing** with

```
File "vllm/multimodal/encoder_budget.py", line 32, in get_mm_max_toks_per_item
  mm_inputs = mm_registry.get_dummy_mm_inputs(
File "vllm/multimodal/registry.py", line 240, in get_dummy_mm_inputs
  mm_inputs = processor.apply(
File "vllm/multimodal/processing/processor.py", line 1562, in _apply_prompt_updates
  assert update_idx is not None, (
AssertionError: Failed to apply prompt replacement for mm_items['vision_chunk'][0]
```

(verified at vllm `b31e9326a7d9`/v0.17.0; the caller chain is
`api_server` → `AsyncLLM` → `InputProcessor.__init__` → `MultiModalBudget.__init__`
→ `encoder_budget.py:87`). The modality name in the brackets is the **model's own**
(`vision_chunk`, `image`, `video`, `audio` — never assume `image`), the index is the
*item* index in the dummy batch, and the model file may be a downstream one.

Read the phase first: nothing here is an NPU, collective, quant, graph or
`--max-model-len` symptom. The engine is computing the **encoder budget from a
dummy prompt**, so the only two inputs are the checkpoint's tokenizer/config and
the HF processor version — no device op has run and no `vllm_ascend` frame exists.

## The arithmetic the assertion encodes

At v0.17.0, for Kimi-K2.5 (`vllm/model_executor/models/kimi_k25.py`):

- `:195` `supported_mm_limits` = `{"vision_chunk": None}`.
- `:206-208` `get_dummy_text` = `"<|media_pad|>" * mm_counts.get("vision_chunk", 0)`.
- `:175`, `:288-297` the prompt update is `PromptReplacement(modality="vision_chunk",
  target=[hf_config.media_placeholder_token_id], replacement=[…])`.
- the budget asks for **one** item: `encoder_budget.py:90`
  `mm_counts=dict.fromkeys(active_modalities, 1)`, and an unlisted modality's
  limit is 999 (`vllm/config/multimodal.py:261-265`), so `active_modalities`
  contains `vision_chunk` unless `--limit-mm-per-prompt`/`language_model_only`
  zeroed it.

So the pass condition is exactly

```
tokenizer("<|media_pad|>").input_ids == [hf_config.media_placeholder_token_id]
```

and `processor.py:1528-1565` tries it twice before asserting: a token-id match,
then a **text** match on `_seq2text(tokenizer, token_ids)` re-encoded with
`_seq2tokens`. Both failing together means the decoded placeholder text and the
configured id do not describe each other — a **checkpoint-directory ↔ tokenizer**
correspondence, not a vLLM bug.

## Step 0 — one command on the reporter's box (before any source read)

```bash
python - <<'PY'
import json
from transformers import AutoTokenizer
d = "/root/.cache/.../<model dir>"          # the path the banner prints
t = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
cfg = json.load(open(d + "/config.json"))
mid = cfg.get("media_placeholder_token_id") or \
      cfg.get("text_config", {}).get("media_placeholder_token_id")
print("config id:", mid)
print("encode:", t("<|media_pad|>").input_ids)      # must be [mid]
print("decode:", t.convert_ids_to_tokens(mid))
PY
```

`encode != [mid]` (or a missing `media_placeholder_token_id`) is the answer: the
directory in the banner is not a self-consistent Kimi checkpoint. A **converted /
quantised** dir (`…-w4a8`, `…-W8A8`, a renamed `Kimi-K2___5-w4a8`) is the usual
owner — the conversion rewrote `config.json` and/or the tokenizer files. Rung 1 is
then: copy `tokenizer.model`/`tiktoken.model`, `tokenizer_config.json`,
`tokenizer.json`, `preprocessor_config.json`, `chat_template*` from the original
bf16 dir (or re-download), keeping `media_placeholder_token_id` consistent, and
re-run the same command.

## Step 1 — read the paste's own tokenizer lines

Two lines *above* such a traceback are the run's own evidence, not decoration:

- `The tokenizer you are loading from '<dir>' with an incorrect regex pattern: …
  You should set the fix_mistral_regex=True flag when loading this tokenizer to fix
  this issue.` — transformers stating that tokenization will be wrong.
- `Could not extract SentencePiece model from '<dir>/tiktoken.model … Falling back
  to TikToken extractor.` — the tiktoken-family tokenizer took the fallback path.

Either one, printed once per tokenizer load, points at the same object the
assertion names.

## Step 2 — the dependency line in the env dump, against the pin's own range

Do this before walking the model file: the failing layer is upstream's
multimodal processor plus `transformers`, and `transformers` majors changed
exactly this tokenizer/decode surface.

- vllm v0.17.0 `requirements/common.txt:10` — `transformers >= 4.56.0, < 5`.
- the pinned resolution — `requirements/test.txt:1230` `transformers==4.57.5`.
- vllm-ascend v0.17.0rc1 `requirements.txt:37` — `transformers>=4.57.4`, i.e.
  **floor only**, so a 5.x install satisfies the plugin while violating upstream.
- `tests/models/registry.py:40,128-135` is the tree's own per-model statement of
  the requirement (`min_transformers_version`): `:283` GLM-5 `5.0.1`, `:263`
  K-EXAONE `5.1.0`, `:760` `5.0.0` … while the `KimiK25ForConditionalGeneration`
  entry `:845-848` carries **none** — upstream believes `4.57.x` serves it, and
  `docs/source/user_guide/release_notes.md:52` reserves the transformers-5 case
  for GLM5 (`requires transformers==5.2.0`, pending vllm#30566). One call:
  `bash <skill_root>/scripts/dep-range-check.sh transformers 5.5.0.dev0 <tree>`.

Rung: reinstall the pin's transformers line in the image (`4.57.5`, or whatever
`requirements/test.txt` resolves) and re-run the identical command. Do not present
it as proven — the probe above is the proof; this is the ordering that makes the
first re-run informative.

## Do not offer these

- **`--limit-mm-per-prompt '{"vision_chunk":0}'` does not bypass it.** The base
  builder always builds the multimodal data and only the *text* depends on the
  counts (`vllm/multimodal/processing/dummy_inputs.py:83-84`), and the Kimi dummy
  builder ignores `mm_counts` outright (`kimi_k25.py:239-247` returns one
  `vision_chunk` item). Zeroing the limit gives an **empty prompt plus one item** —
  the same assertion, one call later.
- `--max-model-len`, `--enforce-eager`, TP/DP, `--quantization`, graph flags: the
  raise is before the device exists.
- "Kimi-K2.5 unsupported on Ascend": the paste's own `Resolved architecture:
  KimiK25ForConditionalGeneration` is upstream's registry, and no `vllm_ascend`
  frame appears; check `tests/models/registry.py:845` before saying it.
- "upstream bug, wait for a fix": only if the probe in Step 0 passes on a
  **pinned** transformers. Say which of the two you verified.

## Recorded cost

#7876 (v0.17.0 + v0.17.0rc1, `transformers==5.5.0.dev0`, Kimi-K2.5-w4a8,
`vision_chunk`): the session never opened this topic, spent 25 turns and
1.2 M prompt tokens walking `allowed_mm_limits`, `get_dummy_processor_inputs` and
`apply_text_matches`, and emitted **no answer text at all** before the budget ran
out — while the deciding facts (the two tokenizer warnings, the 5.x dev
transformers against a `< 5` pin) were in the first paste. Deliver the Step 0
command and the rung in the first message, then continue.

Recheck conditions: the line numbers are vllm v0.17.0's; at later refs re-grep
`get_dummy_text`, `media_placeholder_token_id` and `Failed to apply prompt
replacement`. A family added after `b31e932` may carry its own dummy builder.
