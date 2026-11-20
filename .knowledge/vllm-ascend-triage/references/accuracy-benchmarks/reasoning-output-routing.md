# Thinking/reasoning output routing (verified in vllm v0.19.1)

Source: `vllm/reasoning/qwen3_reasoning_parser.py` and
`vllm/entrypoints/openai/chat_completion/serving.py` at vllm commit `b1388b1f`.

## The mechanism

`Qwen3ReasoningParser.extract_reasoning` (documented in its own docstring):

- strips a leading `<think>` if the model emitted one (Qwen3.5+ templates put it in the
  *prompt*, so only `</think>` appears in the output);
- if `</think>` is absent and `thinking_enabled` is False → `(None, model_output)`
  (everything is content);
- if `</think>` is absent and `thinking_enabled` is True (the default) → `(model_output, None)`
  — "output was truncated", everything is reasoning and **content is `None`**;
- otherwise splits at `</think>` and returns `content or None`.

In `serving.py`: with a reasoning parser, `content` is whatever the parser returned and
`reasoning` is only populated when the request asks for it (`include_reasoning`); without a
parser, `content` is the full text (thinking tags included).

## Why this changes benchmark scores

Served with `--reasoning-parser qwen3` and thinking enabled, any request that hits
`max_tokens` while still inside the thinking block returns `message.content = null`. A
harness that only reads `message.content` records "no answer" for exactly those samples —
it looks like a hidden accuracy loss of roughly (lost samples)/(total samples), concentrated
in the subsets with the longest CoT. A harness that strips think tags from the raw text
(ais-bench's `extract_non_reasoning_content`) or reads `reasoning` too does not lose them.
That asymmetry alone can produce a multi-point tool-vs-tool gap on the same server.

## The inverse shape: "thinking is inside `content` and `reasoning_tokens` is 0"

Re-verified at vllm v0.22.1 (`0decac0d96c4`) + vllm-ascend v0.22.1rc1 (`5f6faa0`),
for #11715. When *no* reasoning output appears at all — the CoT text sits in
`content`, thinking language is mixed (English thinking, Chinese answer),
`completion_tokens_details.reasoning_tokens` is `0` — the first gate is whether
the server was started with `--reasoning-parser` at all. It is **upstream
OpenAI-frontend configuration**, invariant to MTP, image tag and the Ascend
layer, so say that before any device theory.

| claim | evidence at vllm v0.22.1 |
|---|---|
| the flag defaults to empty | `vllm/config/structured_outputs.py:35` `reasoning_parser: str = ""` |
| no parser class ⇒ no split and no counting | `vllm/entrypoints/openai/chat_completion/serving.py:98,125-126,143,253-255` build the parser only when the name is set; the split branch is guarded by `elif reasoning_parser:` / `if reasoning_parser` (`:343-362,1124-1133`) |
| `qwen3` covers Qwen3 **and** 3.5/3.6 | `vllm/reasoning/__init__.py:107-110`; the parser strips a leading `<think>`, since from Qwen3.5 the template puts it in the prompt (`qwen3_reasoning_parser.py:21-24`) |
| `reasoning_tokens` is parser-derived, not a sampler counter | `vllm/reasoning/basic_parsers.py:183` `count_reasoning_tokens`, consumed at `vllm/entrypoints/openai/responses/serving.py:861-893` |
| a *wrong* name is not this failure | `vllm/entrypoints/openai/api_server.py:524-530` raises `KeyError: invalid reasoning parser` at boot, so a server that came up had either the right name or none |

Reporter-supplied scope (#11715, later comment): 910B + Ubuntu 22.04 + CANN 25.5.1,
two images (v0.20.2rc1, v0.18.0), MTP on and off, English and Chinese prompts, both
a 35B-A3B and a 27B w8a8 Eco-Tech checkpoint — every combination identical. That
invariance is itself the argument for a frontend config gate over any
model/device/kernel cause.

Field-name drift, checked before concluding "`reasoning_content` is missing": at
vllm v0.22.1 the chat response field is `message.reasoning`
(`chat_completion/protocol.py:64`), `include_reasoning` defaults to `True`
(`:229`), and `reasoning_content` survives only as a deprecated **input** alias
that a pre-validator renames (`:429-446`). Advice to read
`choices[0].message.reasoning_content` is right for older releases and wrong for
this one; name the field the pinned version emits.

## The kwarg-name trap

`thinking_enabled` is read from `chat_template_kwargs["enable_thinking"]` (default `True`).
`tools/aisbench.py` in vllm-ascend instead writes `chat_template_kwargs={"thinking": True}`,
and some Qwen3.5-era templates expose the switch as `thinking`. If the served template's
switch name and the parser's key differ, thinking control silently does nothing and the
parser mis-classifies output. Do not assume either spelling: check the model's
`chat_template.jinja` and prefer `enable_thinking` with the parser. (Corroborating issue
precedent: #7061 / #7094 in `history/`, which also track the upstream parser fix PR 34779.)

The spelling is per-parser, so one client body flips the branch by model family. At
vllm **v0.23.0** (`0fc695f`), `vllm/reasoning/kimi_k2_reasoning_parser.py:40-46`
does `thinking = bool(kwargs["chat_template_kwargs"].get("thinking", True))` and
falls back to `IdentityReasoningParser` only when that is falsy — the **opposite**
key from Qwen3's `enable_thinking`. Two consequences for a "`</think>` leaked into
`content`" report on a Kimi-K2.x model (#13002's shape 1) before any corruption
theory: (a) a request that sends `enable_thinking: false` does not by itself select
the identity fallback, and (b) while the thinking branch *is* active, text arriving
before an end token is emitted as `reasoning`, so a client that concatenates only
`delta.content` / reads only `message.content` cannot distinguish a routing
decision from a model-emitted token. Ask for one response that prints both
`content` and `reasoning` (and `return_token_ids: true` for the emitted ids) — that
is the whole gate; the probe is also right when the tag appears *after* a complete
answer, which is the shape a truncated `</think>` strip produces.

## Documentation gap that produces this signature by itself

At vllm-ascend v0.22.1rc1 only seven model pages pass `--reasoning-parser`
(`DeepSeek-V4-Flash.md`, `DeepSeek-V4-Pro.md`, `GLM4.x.md`, `GLM5.md`,
`GLM5.2.md`, `Hy3-preview.md`, `MiniMax-M2.md`); **no Qwen page does** — the
Qwen3.6 serve block has none (`docs/source/tutorials/models/Qwen3.6-35B-A3B.md:106-123`)
and `Qwen3.5-27B-Qwen3.6-27B.md` does not mention the flag either. A reporter who
followed the Qwen3.6 tutorial verbatim therefore gets exactly this signature, so
quote the doc block as the reason they hit it and treat the omission as a
separate doc bug (recheck the page's flags at whatever ref the report names; the
sibling pages' flag choices move between releases).

## Checks to request from the reporter

- Server flags: `--reasoning-parser`, `--default-chat-template-kwargs`, `--max-model-len`.
- Endpoint and body: `/v1/chat/completions` (a raw `/v1/completions` or a
  text-in/text-out bench bypasses both the chat template and the parser, so all
  text must land in `content`), plus whether `chat_template_kwargs.enable_thinking`
  is passed (the parser reads `enable_thinking`, default `True`,
  `qwen3_reasoning_parser.py:46`). With it `False` the template prefills
  `<think>\n\n</think>` and plain content is expected.
- The checkpoint's own `chat_template.jinja` (a re-quantised export can ship a
  template without `</think>` / `enable_thinking`, which defeats the parser too).
- Mixed thinking language is model behaviour, not a serving switch: with an empty
  system prompt Qwen3.6-family models often reason in English and answer in the
  prompt's language. The only levers are a system prompt or disabling thinking.
- One `/v1/chat/completions` call with small vs large `max_tokens`: compare `finish_reason`,
  `message.content`, `message.reasoning`.
- Harness behaviour: does it read `content` only, `reasoning` too, or strip `<think>` tags
  from raw text? Is thinking supposed to be on for an accuracy run at all?
- Whether the reported per-subset gaps are concentrated where CoT length is longest — that is
  the fingerprint of this failure mode.
