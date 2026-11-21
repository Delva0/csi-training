# Chat template resolution: `As of transformers v4.44 …` and `--tokenizer-mode`

## Trigger

Serving starts, the engine is healthy, `POST /v1/completions` answers, and **every**
`/v1/chat/completions` returns

`ValueError: As of transformers v4.44, default chat template is no longer allowed, so
you must provide a chat template if the tokenizer does not define one.`

Recorded #6345 (vllm-ascend v0.13.0rc2 `a17059a` ↔ vllm v0.13.0 `72506c9`, a
DeepSeek-V3.2-w8a8 checkpoint; the body names no SoC or card): the whole case is one
serve flag plus one doc gap, and no Ascend component is involved. A quantized checkpoint in the report is a red herring
until step 4 says otherwise.

## Step 1 — the sentence is vLLM's own raise, so grep vLLM first

Both transformers' `get_chat_template` and vLLM emit this text, and vLLM re-raises it
**verbatim** from `apply_hf_chat_template`
(`vllm/entrypoints/chat_utils.py:1800`, the raise at `:1816-1822` at v0.13.0), so
transformers frames in the paste prove nothing about ownership. One grep of the
shortest clause settles it:

```bash
grep -rn "default chat template is no longer" vllm/          # chat_utils.py:1818
grep -rn "apply_hf_chat_template\|resolve_hf_chat_template" vllm/entrypoints/
```

The string is split over two adjacent literals in the source: a full-sentence grep
returns **nothing** and looks like "not in this tree" (it cost #6345 three calls).

## Step 2 — the resolution ladder, and which rung was empty

`resolve_hf_chat_template` (`vllm/entrypoints/chat_utils.py:505-549` at v0.13.0) tries,
in order: (1) `--chat-template` / per-request `chat_template`; (2) the AutoProcessor
template when `tools is None`; (3) `tokenizer.get_chat_template(...)`, whose exception
is **swallowed** to a DEBUG record (`:520-526`); (4) the builtin fallback table
`_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK`
(`vllm/transformers_utils/chat_templates/registry.py:32-47`: `blip-2`, `chameleon`,
`clip`, `deepseek_ocr`, `deepseek_vl_v2`, `fuyu`, `minicpmv`, `paligemma`, `qwen`,
`siglip`, `siglip2`). The table has no entry for a DeepSeek **text** family, so rung 4
is empty too and the raise above fires.

Two consequences worth stating in the answer:

- Since transformers 4.44 there is **no default template at all**: a tokenizer dir
  without `tokenizer_config.json:"chat_template"` (or `chat_template.jinja`) fails
  every chat request while the engine stays up — an error response, not a crash.
- The swallowed rung-3 exception is the diagnostic: restart with
  `VLLM_LOGGING_LEVEL=DEBUG` and read `Failed to load AutoTokenizer chat template for
  <path>`; a jinja parse/render error there is a different case from "no template".

## Step 3 — when the tree has a native tokenizer, the answer is a mode flag

At vllm v0.13.0 a model family may ship its chat encoding as **Python**, not jinja, and
vLLM then selects a dedicated tokenizer class by `--tokenizer-mode`:

| Fact | Evidence (vllm v0.13.0 `72506c9`) |
|---|---|
| mode → class map | `vllm/tokenizers/registry.py:32-36` `"deepseek_v32": ("deepseek_v32", "DeepseekV32Tokenizer")` |
| allowed values | `TokenizerMode = Literal["auto","hf","slow","mistral","deepseek_v32"]`, `vllm/config/model.py:76`; CLI `--tokenizer-mode` `vllm/engine/arg_utils.py:626` |
| the chat path bypasses `apply_hf_chat_template` | `vllm/entrypoints/openai/serving_engine.py:1151` `elif isinstance(tokenizer, DeepseekV32Tokenizer)` |
| the encoding is a port of the model's own script | `vllm/tokenizers/deepseek_v32_encoding.py:5` ("copy from …/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py"), wrapped by `DeepseekV32Tokenizer(CachedHfTokenizer)` at `deepseek_v32.py:16-19` |
| thinking is **off** unless asked | `deepseek_v32.py:52-56` `thinking = kwargs.get("thinking", False)` → `chat` mode; per request `chat_template_kwargs: {"thinking": true}` |
| `auto` never picks it | `tokenizers/registry.py:144-158`: auto → mistral only when `mistral_common` is present, otherwise `hf`. So a default serve command always lands on rung 3/4 |

In-tree corroboration that this is the intended deployment, not a guess:

- the project's own W8A8 e2e config passes it —
  `tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml:40,64`
  (`--tokenizer-mode deepseek_v32`, with `--reasoning-parser deepseek_v3`);
- the published tutorial does **not** — `docs/source/tutorials/DeepSeek-V3.2.md:272,
  347,424,503` are four `vllm serve` blocks and none carries the flag, so a customer
  who copies the tutorial reproduces the report exactly (file the doc gap separately);
- the release already promised the repair — `docs/source/user_guide/release_notes.md`
  v0.11.0rc3 *Known Issues*: "DeepSeek 3.2 doesn't work with chat template … vLLM
  v0.12.0 doesn't support it. We'll support in the next v0.13.0rc1 version".

## Step 4 — the remaining causes, each with one check

1. **Mode left at default** (most likely when the tutorial was followed). Check
   `pip show vllm | head -2` in the container and the serve command for
   `--tokenizer-mode`. Fix is the flag, same command, restart, same curl.
2. **Image carries vLLM < 0.13**: the mode does not exist and start fails with
   `No tokenizer registered for tokenizer_mode='deepseek_v32'`
   (`vllm/tokenizers/registry.py` `load_tokenizer_cls`). This is the version check,
   not a second mechanism.
3. **Tokenizer files absent from a converted checkpoint** (msmodelslim-type output with
   only `config.json` + weights): even in the native mode the HF tokenizer files must
   exist (the class subclasses `CachedHfTokenizer`) — `ls <model_dir> | grep -E
   'tokenizer|vocab|merges'`, then `--tokenizer <official dir>`.
4. **A template exists but will not render** — the DEBUG line of step 2 shows the
   exception; that is a jinja/transformers-version question, not this page.

Do not propose a hand-written `--chat-template` for a family whose encoding is Python:
a V3.1-style template mis-encodes the model's tool-call delimiters and its thinking
switch. Mention it as a stopgap only if the reporter cannot move the flag.

## What to send back, and what stays open

`pip show vllm vllm-ascend`, `ls <model_dir>`, the key list of `tokenizer_config.json`
(`python -c "import json;print('chat_template' in json.load(open('<dir>/tokenizer_config.json')))"`),
and the `VLLM_LOGGING_LEVEL=DEBUG` window around one failing request.

Open here: the reporter's own checkpoint contents (not in the snapshot), and whether the
official model repo also ships a `chat_template.jinja` (the pinned `transformers` export
is `files=1478/4849 TRUNCATED`, so `tokenization_utils_base.py` is absent — that absence
is a gap, not a fact about the string). No precedent exists in the corpus: a title search
for `chat.?template` returns nothing and the bodies mention templates only incidentally,
so the answer rests on the pinned upstream code.
