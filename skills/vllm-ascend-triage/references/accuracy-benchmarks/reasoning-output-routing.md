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

## The kwarg-name trap

`thinking_enabled` is read from `chat_template_kwargs["enable_thinking"]` (default `True`).
`tools/aisbench.py` in vllm-ascend instead writes `chat_template_kwargs={"thinking": True}`,
and some Qwen3.5-era templates expose the switch as `thinking`. If the served template's
switch name and the parser's key differ, thinking control silently does nothing and the
parser mis-classifies output. Do not assume either spelling: check the model's
`chat_template.jinja` and prefer `enable_thinking` with the parser. (Corroborating issue
precedent: #7061 / #7094 in `history/`, which also track the upstream parser fix PR 34779.)

## Checks to request from the reporter

- Server flags: `--reasoning-parser`, `--default-chat-template-kwargs`, `--max-model-len`.
- One `/v1/chat/completions` call with small vs large `max_tokens`: compare `finish_reason`,
  `message.content`, `message.reasoning`.
- Harness behaviour: does it read `content` only, `reasoning` too, or strip `<think>` tags
  from raw text? Is thinking supposed to be on for an accuracy run at all?
- Whether the reported per-subset gaps are concentrated where CoT length is longest — that is
  the fingerprint of this failure mode.
