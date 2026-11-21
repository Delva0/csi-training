# `RuntimeError: Already borrowed` from the fast tokenizer during chat preprocessing

Two recorded cases, same mechanism, different upstream call site and log line:

- #7197 — vllm-ascend v0.16.0rc1 (`a78a00e0b1b6`) + vllm v0.16.0 (`89a77b10846f`),
  Ascend 910B3, image `vllm-ascend:v0.16.0rc1`, Qwen3-Next on a customer dataset. The
  paste's line is `Error in preprocessing prompt inputs` and the constructor sits in
  `engine/serving.py:1044`.
- #7609 — vllm-ascend v0.17.0rc1 (`e20f0b1a0d2f`) + vllm v0.17.0 (`b31e9326a7d9`),
  910B3, image `vllm-ascend:v0.17.0rc1`, **Qwen3-Omni** with tools under client
  concurrency (「并发场景报错」). The paste's line is `Error in tool parser creation.`,
  emitted from the **streaming** generator. Anchors below are labelled `v0.16.0` /
  `v0.17.0`; recheck every `path:line` at another pin with the three greps in
  *Pin the sites in one call*.

## The answer first

Not the model, not the NPU, not the customer dataset and not vllm-ascend: the
**API-server process** used one HuggingFace fast tokenizer from two threads at once,
and the Rust tokenizer behind the PyO3 wrapper raised `already mutably borrowed`
(huggingface/tokenizers#537). Upstream vLLM knows this exception — it sleeps and
retries around it in exactly one place — and the chat preprocessor's tool-parser
construction is not that place. vllm-ascend contains **no** tokenizer code at either
version (`rg -ln -i tokeniz vllm_ascend/` = 0 hits at v0.17.0rc1), and no frame in the
pasted traceback belongs to it.

Cheapest next action: send the same request **alone** (client concurrency 1), or retry
the failed request 1-3 times with a 0.2-0.5 s pause. The borrow window is
microseconds-to-milliseconds, so a sporadic failure is the expected shape and the
retry usually succeeds.

## The two pasted shapes

Preprocessing (v0.16.0, non-streaming path):

```text
ERROR [serving.py:315] Error in preprocessing prompt inputs
  serving.py:299  render_chat_request  →  await self._preprocess_chat(...)
  engine/serving.py:1044  request = tool_parser(tokenizer).adjust_request(request)
  hermes_tool_parser.py:63  self.tool_call_start_token_ids = self.model_tokenizer.encode(
  site-packages/transformers/tokenization_utils_base.py:2872 encode → :3263 encode_plus
  tokenization_utils_fast.py:627 _encode_plus → :541 _batch_encode_plus
  tokenization_utils_fast.py:475 set_truncation_and_padding
      → self._tokenizer.no_truncation()
RuntimeError: Already borrowed
```

Tool-parser creation inside the stream (v0.17.0; #7609 pasted the `ERROR` line and the
`Hermes2ProToolParser` frames, not the preprocessing wrapper):

```text
ERROR [chat_completion/serving.py:702] Error in tool parser creation.
  create_streaming_error_response(...) → data: {...error...}
  data: [DONE]
```

Both are the same race. The log line tells you **which** construction site ran — read
it before assuming which upstream path the reporter is on, and note where each message
lives at v0.17.0: `Error in tool parser creation.` has exactly two emitters
(`chat_completion/serving.py:702` streaming, `engine/serving.py:1238` non-streaming),
while `Error in preprocessing prompt inputs` is **absent** from
`chat_completion/serving.py` and appears only in the completion, tokenize and
speech-to-text servers (`openai/completion/serving.py:120`,
`entrypoints/serve/tokenize/serving.py:96`, `openai/speech_to_text/speech_to_text.py:530`).
A v0.17.0 *chat-completions* paste carrying the preprocessing message is therefore on a
different path or a different version — ask, rather than reusing this page's v0.16.0
row.

## The chain at vllm v0.16.0 (`89a77b1`)

| Site | Fact |
|---|---|
| `vllm/entrypoints/openai/chat_completion/serving.py:299,313-316` | `render_chat_request` awaits `_preprocess_chat`; the `except (ValueError, TypeError, RuntimeError, jinja2.TemplateError)` branch logs exactly `Error in preprocessing prompt inputs` and returns an error response. The server survives; the request does not. |
| `vllm/entrypoints/openai/engine/serving.py:1031-1044` | Per request, **only** when a tool parser is configured and `tool_choice != "none"`: `tokenizer = renderer.get_tokenizer()` (the *synchronous* tokenizer) and `tool_parser(tokenizer).adjust_request(...)`, executed on the **event-loop thread**, after `render_chat_async` returned. |
| `vllm/tool_parsers/hermes_tool_parser.py:63,66,70-80` | The constructor encodes `<tool_call>` and `</tool_call>` and then decodes each id — 2+N tokenizer calls per chat request. `hermes`, `functiongemma` and `longcat` pre-encode in `__init__` at this pin. |
| `vllm/utils/async_utils.py:24-50,105-141` | The same tokenizer object is driven by `AsyncMicrobatchTokenizer` (`ThreadPoolExecutor(max_workers=1)`) through `run_in_executor` — the **second thread**. |
| `vllm/multimodal/processing/context.py:437-459` | The one place upstream catches this condition: `isinstance(exc, RuntimeError) and exc.args[0] == "Already borrowed"` → `time.sleep(0.5)`, up to 5 tries, comment `# See https://github.com/huggingface/tokenizers/issues/537`. |

## The chain at vllm v0.17.0 (`b31e9326a7d9`)

| Site | Fact |
|---|---|
| `vllm/entrypoints/openai/chat_completion/serving.py:627,634` | `async def chat_completion_stream_generator(request, request_id, tokenizer: TokenizerLike, …)` — the **synchronous** tokenizer is a parameter of the async generator. |
| `…/chat_completion/serving.py:695-698` | When `tool_choice_auto and self.tool_parser`: `tool_parsers = [self.tool_parser(tokenizer)] * num_choices` — a **fresh parser per streaming request**, on the event-loop thread. |
| `…/chat_completion/serving.py:701-706` | `except Exception` → `logger.exception("Error in tool parser creation.")` (`:702`) → `create_streaming_error_response(e)` → the SSE error frame then `data: [DONE]`; the connection stays up. |
| `vllm/entrypoints/openai/engine/serving.py:1235-1239` | Non-streaming counterpart: `try: tool_parser = tool_parser_cls(tokenizer)` → `except RuntimeError` → same log at `:1238` and `raise e`, so there the client sees a 500 instead of an in-stream error. |
| `vllm/tool_parsers/hermes_tool_parser.py:63-79` | Unchanged constructor; **four** parsers pre-encode in `__init__` at this pin — `hermes`, `functiongemma`, `longcat`, `llama` (`llama_tool_parser.py:59` encodes only `<|python_tag|>`). |
| `vllm/renderers/base.py:116-118`, `vllm/utils/async_utils.py:24-50` | `get_async_tokenizer()` wraps `self.get_tokenizer()` in `AsyncMicrobatchTokenizer`, i.e. the executor thread and the generator share one object. |
| `vllm/multimodal/processing/context.py:270-285` | The single `Already borrowed` retry, moved by ~170 lines; the comment and the `time.sleep(0.5)` loop are unchanged. |

Mechanism: `PreTrainedTokenizerFast.encode` → `_batch_encode_plus` →
`set_truncation_and_padding` → `self._tokenizer.no_truncation()` needs the Rust
tokenizer **mutably**, so it fails whenever another thread holds a shared borrow
mid-`encode`/`decode`. Same-process concurrency is therefore required: one request's
prompt being encoded in the executor thread while another request's tool parser is
built in the event-loop thread. `--api-server-count > 1` gives each process its own
tokenizer object, so it neither causes nor cures this.

## Pin the sites in one call (do this instead of trusting the line numbers)

```bash
cd <vllm-root>
rg -n 'Error in preprocessing prompt inputs|Error in tool parser creation' -g '*.py' .
rg -n 'Already borrowed' -g '*.py' .          # expect exactly one: the retry site
uv run --no-project python - <<'PY'
import ast, pathlib, re
for p in sorted(pathlib.Path('vllm/tool_parsers').glob('*.py')):
    t = ast.parse(p.read_text())
    for c in [n for n in ast.walk(t) if isinstance(n, ast.ClassDef)]:
        for f in c.body:
            if isinstance(f, ast.FunctionDef) and f.name == '__init__':
                body = ast.get_source_segment(p.read_text(), f) or ''
                if re.search(r'model_tokenizer\.encode\(|tokenizer\.encode\(', body):
                    print(f'{p.name}:{f.lineno} {c.name} pre-encodes in __init__')
PY
```

Read the first grep by **file**: the tool-parser messages must include the chat endpoint the reporter called, and `Already borrowed` must still be a single hit (a second one means the release added a retry path — then read it before repeating "unprotected"). The third call is not optional pedantry: `rg -n 'encode\(' vllm/tool_parsers/*.py`
**over-counts** — `str.encode("utf-8")` appears inside `extract_tool_calls` of
`deepseekv3`, `deepseekv31`, `kimi_k2` and `mistral`, which do **not** touch the
tokenizer in `__init__` (a recorded v0.17.0 run listed all eight and had to be
corrected). The parser's `__init__` is what shares the tokenizer with the encode
thread, so only the ast/grep-of-`__init__` list is citable.

## The trigger, and why the model is only a proxy

The request needs **tools** (`tool_choice != "none"`) plus a `--tool-call-parser` on
the serve command (`api_server.py:389` validates the pair, `cli_args.py:158,322` at
v0.16.0); without either, the constructor never runs and the crash disappears. Nothing
in the path inspects the model type, so "it happens with Qwen3-Omni / Qwen3-Next and my
customer dataset" means "that dataset sends tools concurrently". Say it that way, then
prove it with the next two rows. One supporting read for a Qwen3-Omni report:
`rg -n 'Qwen3-Omni' docs/source/user_guide/support_matrix/supported_models.md` on the
vllm-ascend tree shows the row is 🔵 (planned) at v0.17.0rc1 — do not let that become
an "unsupported model" explanation for a frontend race, but do report it as a coverage
gap the reporter will hit with any *other* error.

## Checks, cheapest first

1. **Concurrency 1** — serialise the client (or one `curl` of the failing body). Gone
   ⇒ the race is confirmed; keep it as the rerun baseline.
2. **Retry** — resend the same body 1-3 times with a short pause; success on a retry is
   the same evidence, and it is the workaround the reporter can ship.
3. **Tools off** — the same request without `tools` (or the server without
   `--tool-call-parser`) skips the construction site entirely.
4. **A parser that does not pre-encode in `__init__`**, taken from the ast list above
   (any parser *not* in it at that pin), same concurrency: if the failure disappears,
   the constructor is the sharing window.
5. **Same concurrency, other model** — a text-only model with the same tools and the
   same client is the row that removes the model from the answer.
6. **Every request or sporadic?** Sporadic is the signature of a race. If *every*
   request fails, you are reading a different branch — re-read the full log line (which
   of the two messages above, and whether `[DONE]` still arrived) rather than the issue
   title.

## Workaround shape

Wrap the tool-parser construction with the retry upstream already uses in
`context.py` (catch `RuntimeError` whose `args[0] == "Already borrowed"`, sleep,
bounded retries) **in the serving process**, or keep the frontend serial. Label it a
workaround: the durable fix belongs upstream (vLLM's tool-parser path or
`AsyncMicrobatchTokenizer`), not in `vllm_ascend/`.

## Version scope, gaps, recheck

- Verified at vllm v0.16.0 + vllm-ascend v0.16.0rc1 and at vllm v0.17.0 +
  vllm-ascend v0.17.0rc1 (source only; neither recorded run re-executed the repro).
  The call site **moved** between them (preprocessing wrapper → streaming generator)
  and so did the retry block's line range, which is why every row here names its pin.
- Upstream pins `transformers >= 4.56.0, < 5` (`requirements/common.txt:10`) and
  vllm-ascend `transformers>=4.57.4` (`requirements.txt:37`), so the installed stack is
  4.57.x — and the snapshot's transformers export is **v5.0.0rc0 and truncated** (no
  `tokenization_utils_fast.py`), so the reporter's `tokenization_utils_fast.py` line
  numbers cannot be checked here. State that gap and ask for
  `pip show transformers tokenizers`.
- Recheck if a release moves the tool-parser construction off the event loop, adds the
  retry to it, changes `AsyncMicrobatchTokenizer` away from the single-thread executor,
  or gives a *new* parser an `encode()` in `__init__` (the ast call above is the check).
- Unverified in both recorded runs: the reporter's client concurrency (no client detail
  in either paste). The mechanism is source-verified; the trigger is the hypothesis that
  the serial rerun settles.
