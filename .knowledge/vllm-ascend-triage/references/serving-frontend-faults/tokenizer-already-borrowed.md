# `RuntimeError: Already borrowed` from the fast tokenizer during chat preprocessing

Worked case #7197, vllm-ascend v0.16.0rc1 (`a78a00e0b1b6`) + vllm v0.16.0
(`89a77b10846f`), Ascend 910B3, image `vllm-ascend:v0.16.0rc1`, Qwen3-Next on a
customer dataset. Recheck every `path:line` at another pin.

## The answer first

Not the model, not the NPU, not the customer dataset: the **API-server process**
used one HuggingFace fast tokenizer from two threads at once, and the Rust
tokenizer behind the PyO3 wrapper raised `already mutably borrowed`
(huggingface/tokenizers#537). Upstream vLLM knows this exception — it sleeps and
retries around it in exactly one place — and the chat preprocessor's tool-parser
construction is not that place. vllm-ascend contains **no** tokenizer code at
this version (`grep -rn tokeniz vllm_ascend/` = 0 hits), and no frame in the
pasted traceback belongs to it.

Cheapest next action: send the same request **alone** (client concurrency 1), or
retry the failed request 1-3 times with a 0.2-0.5 s pause. The borrow window is
microseconds-to-milliseconds, so a sporadic 4xx is the expected shape and the
retry usually succeeds.

## The pasted shape

```
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

## The chain (vllm v0.16.0, `89a77b1`)

| Site | Fact |
|---|---|
| `vllm/entrypoints/openai/chat_completion/serving.py:299,313-316` | `render_chat_request` awaits `_preprocess_chat`; the `except (ValueError, TypeError, RuntimeError, jinja2.TemplateError)` branch logs exactly `Error in preprocessing prompt inputs` and returns an error response. The server survives; the request does not. |
| `vllm/entrypoints/openai/engine/serving.py:1031-1044` | Per request, **only** when a tool parser is configured and `tool_choice != "none"`: `tokenizer = renderer.get_tokenizer()` (the *synchronous* tokenizer) and `tool_parser(tokenizer).adjust_request(...)`, executed on the **event-loop thread**, after `render_chat_async` returned. |
| `vllm/tool_parsers/hermes_tool_parser.py:63,66,70-80` | The constructor encodes `<tool_call>` and `</tool_call>` and then decodes each id — 2+N tokenizer calls per chat request. `hermes`, `functiongemma` and `longcat` are the three parsers that pre-encode in `__init__`; the rest do not, which is why the parser *value* matters. |
| `vllm/utils/async_utils.py:24-50,105-141` | The same tokenizer object is driven by `AsyncMicrobatchTokenizer` (`ThreadPoolExecutor(max_workers=1)`) through `run_in_executor` — the **second thread**. Chat prompts reach it via `vllm/renderers/protocol.py:60,121,381`. |
| `vllm/multimodal/processing/context.py:437-459` | The one place upstream catches this condition: `isinstance(exc, RuntimeError) and exc.args[0] == "Already borrowed"` → `time.sleep(0.5)`, up to 5 tries, comment `# See https://github.com/huggingface/tokenizers/issues/537`. `grep -rn "Already borrowed" -g '*.py' vllm/` returns this single hit. |

Mechanism: `PreTrainedTokenizerFast.encode` → `_batch_encode_plus` →
`set_truncation_and_padding` → `self._tokenizer.no_truncation()` needs the Rust
tokenizer **mutably**, so it fails whenever another thread holds a shared borrow
mid-`encode`/`decode`. Same-process concurrency is therefore required: one
request's prompt being encoded in the executor thread while another request's
tool parser is built in the event-loop thread. `--api-server-count > 1` gives each
process its own tokenizer object, so it neither causes nor cures this.

## The trigger, and why the model is only a proxy

The request needs **tools** (`tool_choice != "none"`) plus a `--tool-call-parser`
on the serve command (`api_server.py:389` validates the pair,
`cli_args.py:158,322`); without either, the constructor never runs and the crash
disappears. Nothing in the path inspects the model type, so "it happens with
Qwen3-Next and my customer dataset" means "that dataset sends tools
concurrently". Say it that way, then prove it with the next two rows.

## Checks, cheapest first

1. **Concurrency 1** — serialise the client (or one `curl` of the failing body).
   Gone ⇒ the race is confirmed; keep it as the rerun baseline.
2. **Retry** — resend the same body 1-3 times with a short pause; success on a
   retry is the same evidence, and it is the workaround the reporter can ship.
3. **Tools off** — the same request without `tools` (or the server without
   `--tool-call-parser`) skips `engine/serving.py:1044` entirely.
4. **Same concurrency, other model** — a text-only model with the same tools and
   the same client is the row that removes the model from the answer.
5. **Every request or sporadic?** Sporadic is the signature of a race; if every
   request fails, the message is being read from the wrong branch — re-read the
   full log line rather than the title.

## Workaround shape

Wrap the tool-parser construction with the retry upstream already uses in
`context.py` (catch `RuntimeError` whose `args[0] == "Already borrowed"`, sleep,
bounded retries) **in the serving process**, or keep the frontend serial. Label it
a workaround: the durable fix belongs upstream (vLLM's tool-parser path or
`AsyncMicrobatchTokenizer`), not in `vllm_ascend/`.

## Version scope, gaps, recheck

- Verified at vllm v0.16.0 + vllm-ascend v0.16.0rc1 only. Upstream pins
  `transformers >= 4.56.0, < 5` (`requirements/common.txt:10`) and vllm-ascend
  `transformers>=4.57.4` (`requirements.txt:37`), so the installed stack is
  4.57.x — and the snapshot's transformers export is **v5.0.0rc0 and truncated**
  (no `tokenization_utils_fast.py`), so the reporter's `tokenization_utils_fast.py`
  line numbers cannot be checked here. State that gap and ask for
  `pip show transformers tokenizers`.
- Recheck if a release moves the tool-parser construction off the event loop,
  adds the retry to it, or changes `AsyncMicrobatchTokenizer` away from the
  single-thread executor; also if the report's parser is not one of the three
  that pre-encode.
- Unverified in the recorded run: the reporter's client concurrency (no client
  detail in the paste). The mechanism is source-verified; the trigger is the
  hypothesis that the serial rerun settles.
