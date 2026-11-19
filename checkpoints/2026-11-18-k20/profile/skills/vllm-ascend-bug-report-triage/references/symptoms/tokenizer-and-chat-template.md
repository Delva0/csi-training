# `ValueError: As of transformers v4.44 …`: chat templates and tokenizer artifacts

## Scope

Request-time failures whose owner is the **checkpoint's tokenizer directory**, not the engine, the
scheduler or the NPU. The service starts, the weights load, `/v1/completions` with a raw prompt answers
normally, and the first `/v1/chat/completions` call returns

```text
ValueError: As of transformers v4.44, default chat template is no longer allowed, so you must provide a
chat template if the tokenizer does not define one.
```

The worked case is vllm-ascend #6345 (DeepSeek-V3.2 W8A8, image `quay.io/ascend/vllm-ascend:v0.13.0rc2`,
duplicate of #5353 on `0.12.0rc1`).

Not this page: a malformed response *body* (`tool-call-streaming.md`), reasoning that arrives in another
field or a thinking-mode question (`reasoning-parser-output.md`), a tokenizer **class** the installed
transformers lacks (`Tokenizer class <X> does not exist or is not currently imported`, the tokenizer twin in
`model-type-support-floor.md`), a load-time quantisation failure (`quantization-load.md`), or an accuracy
question. Nothing here is an
Ascend defect — do not route it to the graph, device or quantisation pages, and do not ask for `npu-smi`
output.

## The message names transformers, but vLLM raises it (verified at vLLM `v0.13.0`)

- `vllm/entrypoints/chat_utils.py:1800 apply_hf_chat_template(...)` calls `resolve_hf_chat_template` and
  raises that `ValueError` at `:1816-1821` when the resolver returns `None`. The same literal repeats in
  `vllm/entrypoints/openai/protocol.py:632` / `:1874` and in the pooling protocols
  (`pooling/classify/protocol.py:103`, `pooling/embed/protocol.py:115`), where it is only the
  **description of the `chat_template` field** — a reporter quoting it from the API docs is quoting
  documentation, not a second raise.
- A `git grep` for the sentence in **transformers** returns nothing at `v4.44.0` … `v4.57.6` (checked),
  and the literal itself is split across two source lines, so a full-sentence search cannot match
  anyway. Grep 3–4 words (`default chat template`) in the vLLM clone, and read an empty result there as
  "wrong repo", never as absence.
- No `vllm_ascend/` file is involved: the tokenizer files in the served directory carry no chat
  template, or the model ID the request used resolved to a tokenizer that does not.

## Resolution order — read it before proposing a workaround

`resolve_hf_chat_template` at vLLM `v0.13.0` (`chat_utils.py:505-549`) tries, in order:

1. the template given in the request or by `--chat-template` (the request half only when
   `--trust-request-chat-template` is set, `cli_args.py:113`, default `False`);
2. the **AutoProcessor**'s `chat_template` — skipped entirely when the request carries `tools`;
3. `tokenizer.get_chat_template(...)`: `tokenizer_config.json`'s `chat_template`, or a sibling
   `chat_template.jinja` read by transformers (`tokenization_utils_base.py:2185-2188` at `v4.57.6`;
   the engine's pin is in `requirements/common.txt`);
4. a per-`model_type` fallback from `vllm/transformers_utils/chat_templates/registry.py`
   (`_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK`, `:32`, ~10 entries).

All four returning nothing yields the `ValueError`. Two consequences decide most reports:

- **The flag works at any version**: `--chat-template /path/template.jinja` is priority 1 and
  independent of the checkpoint, of `trust_remote_code` and of the parser settings. It accepts a
  filesystem path, a template string containing Jinja characters, or the *name* of a built-in template
  in `vllm/transformers_utils/chat_templates/` (e.g. `--chat-template template_chatml.jinja`);
  `validate_chat_template` (`chat_utils.py:1170-1194`) rejects a path-like string that exists nowhere.
- **A missing registry entry is not a bug**: at `v0.13.0` the fallback table covers `blip-2`,
  `chameleon`, `clip`, `deepseek_ocr`, `deepseek_vl_v2`, `fuyu`, `minicpmv`, `paligemma`, `qwen`,
  `siglip`, `siglip2` — **no `deepseek_v32`/`deepseek_v3`** — so a DeepSeek-V3.2 directory without a
  template in its tokenizer files has no automatic recovery at that tag.

## Check order for a new report

1. Which endpoint failed, and does `/v1/completions` with raw prompts still work? A chat-only failure
   means template resolution, not the model, the NPU or a parser.
2. `ls <model-dir>` plus
   `python -c "import json;print(json.load(open('<model-dir>/tokenizer_config.json')).get('chat_template'))"`,
   and look for `chat_template.jinja`. A **quantised** or re-exported directory (modelslim,
   llm-compressor, a vendor mirror) is the usual place for the template to be missing — the export
   copied `tokenizer.json` and left the template behind. `[hypothesis]` for a given report until the
   directory is listed; the falsifier is that one `ls`.
3. `git show <ref>:docs/source/user_guide/release_notes.md | grep -n -i "chat template"` — the
   `Known Issues` bullet and **which `## vX.Y.Z` heading it sits under**. At `v0.13.0rc2` the bullet
   「DeepSeek 3.2 doesn't work with chat template … We'll support in the next v0.13.0rc1 version」 sits
   under the **v0.12.0rc1** section, so for a 0.13.0rc2 reporter it is history: quote it as a
   known-issue line, not as a fix, and read its absence as a further sign that the template is missing
   from their files rather than unsupported by the release.
4. Was the issue closed as a duplicate? Take the duplicate's **title and link**, not its whole body: the
   body is usually a 16 KB `collect_env` paste and the decisive sentence is in the closing comment.

Diagnosis to hand over (their hardware, one command):

```bash
uv run --no-project python - <<'PY'
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("/path/to/served/model")
print("chat_template:", repr(t.chat_template)[:200])
PY
ls /path/to/served/model | grep -i chat_template
```

`None` for the template and no `chat_template.jinja` file names the directory as the defect; a template
that *is* present moves the question back to the launch line (`--chat-template`,
`--chat-template-content-format`, `--tokenizer` pointing at a different directory than the weights).

## Workaround ladder — CPU-side request preprocessing, nothing traded away on the device

1. `--chat-template /path/to/chat_template.jinja`, using the template from the model's own HF repo (or
   the `deepseek-ai` client code the maintainer cited). Cheapest and version-independent.
2. Drop `chat_template.jinja` next to the tokenizer files in the served directory: it is picked up at
   priority 3 and fixes every client at once, but only on a transformers new enough to read it (check
   the image's pin).
3. `--trust-request-chat-template` and send `chat_template` per request — useful for a mixed fleet, but
   any client can then supply template text.
4. Use the official DeepSeek-V3.2 client, or `/v1/completions` with prompts the client rendered itself.
   This is the maintainer's answer in #6345 (`reported`, verbatim: specify the startup parameters from
   the vLLM recipes docs, or use the official DeepSeek-V3.2 client code) and the closing advice of the
   duplicate #5353.

## Worked case: vllm-ascend#6345 — DeepSeek-V3.2-w8a8 on the `v0.13.0rc2` image

Reported: the vendor's container, DeepSeek-V3.2 W8A8, `curl` to the chat endpoint returns the message
above; no start command, no model path. `[reported]`

- Thread: `triomino` closed the loop — "Duplicated issue #5353. You can either specify the startup
  parameters as shown in the vLLM documentation (DeepSeek-V3.2 recipe) or use the official
  DeepSeek-V3.2 client code"; then the stale bot. `[reported]` A duplicate link is a workaround to relay,
  not a mechanism.
- Mechanism `[verified]`: the release's own tokenizer files are the only template source at that tag
  (registry priority 4 has no `deepseek_v32` entry) and vLLM raises before tokenising, so nothing in
  `vllm_ascend/` can be at fault. Which file the served directory lacks is `[hypothesis]` until listed —
  the falsifier is the `ls`/`chat_template` print above.
- Do not answer "upgrade the image": the reporter is already past the release the note was written for,
  and the note points the opposite way (support "in `v0.13.0rc1`" — which is what they run).

## Tokenizer artifacts more generally

Two `ValueError`s reach the same frontend path with different owners: this page's chat-template guard
(`apply_hf_chat_template`) and `Tokenizer class <X> does not exist or is not currently imported`, which
transformers raises from `models/auto/tokenization_auto.py` when the served `tokenizer_config.json` names a
class the installed transformers does not carry — a pin floor, answered in `model-type-support-floor.md`
("The tokenizer twin"), not by a template.

A re-quantised export (ModelSlim, llm-compressor) writes only the tokenizer files the tool decided to
keep. A dropped `chat_template`, a stale `tokenizer_config.json`, or a tokenizer directory that differs
from the weights directory all surface as **request-time** errors on a healthy server — the
discriminator against every load-time class in `quantization-load.md` is that weight load, graph capture
and the first forward all succeeded.

## Falsifier and recheck

Falsifier: `--chat-template` supplied, same request, same 400 → the failing path is not
`apply_hf_chat_template` (check the endpoint: embedding/pooling protocols carry their own copy), or the
template itself was rejected by `validate_chat_template`.

Recheck: the priority list, the fallback registry and the message move between vLLM releases — at `main`
the same text lives in `vllm/renderers/hf.py` plus
`entrypoints/openai/chat_completion/protocol.py`. Read them at the tag paired with the reporter's
vllm-ascend version, not at `main`, and re-read both before quoting a line number.
