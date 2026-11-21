# "Tool call empty": the model's own output carries no call at all

## When this page applies

The complaint is the *absence* of a tool call, not damage to one:

- `tool_calls` empty or missing, `finish_reason: "stop"`, `content` normal
  prose — "tool call empty", "工具调用为空", "没有调用工具";
- it is **conditional on mode** (non-think / thinking off /
  `chat_template_kwargs: {"enable_thinking": false}`) or **概率性** ("a certain
  probability", "sometimes it doesn't call"), on an otherwise healthy serve — no
  traceback, no device text, engine log clean;
- the reporter ships engine-side dumps (a `raw_completion.json`, a
  `/v1/completions` body, a rendered prompt) or asks you to explain such a dump.

Different owner for each neighbouring shape, so check which one you have before
reading code:

| shape | owner |
|---|---|
| model output has **no** tool-call marker for the configured `--tool-call-parser` | *this page* — model choice / prompt behaviour |
| marker present but the client sees no `tool_calls` | parser lane: `index.md` rule 1 and `scripts/sse-toolcall-audit.sh` |
| markup lands in `content` and `tool_calls` is empty | no parser is consuming it — the flags are unset or the reasoning parser defaults to Identity: `index.md`, `../accuracy-benchmarks/reasoning-output-routing.md` |
| `tool_calls[0].function.name` present, `arguments` `{}` or truncated | parameter extraction / delta arithmetic: `arg-type-gate-unquoted-value.md`, `streamed-arguments-audit.md` |
| `content` stops **mid-turn** at a lead-in line (ends in `:`), `tool_calls` empty, `finish_reason: "stop"` | the marker *was* emitted and withheld — `_extract_content` returns nothing after an unclosed `<tool_call>` (`glm4_moe_tool_parser.py:238-283`), the region is marked `(raw, False)` (`:284-303`) and streaming `break`s with no name (`:453`) at vllm v0.23.0, so it reaches neither `content` nor `tool_calls`: parser lane, decided by `return_token_ids` — `index.md` scope (#17391), not a rate |
| first response after start is `content: ""` with `completion_tokens: 1` | one model-issued blank/stop token: `../spec-decode-mtp/empty-first-response.md` |
| `stream: true` returns only CoT and no `tool_calls` | #5683, reporter-confirmed A/B on `--async-scheduling`: `index.md` scope |

## Step 0 — one call decides the lane

```bash
uv run --no-project python <skill_root>/scripts/toolcall-raw-output-audit.py /workspace/attachments/<issue>
```

It classifies every dump it finds (request body / rendered prompt / tokenized
request / completion) and prints the facts below plus a verdict, so the lane is
named before any parser, delta-assembly or sampler source is opened. Three
verdicts:

- **no marker in the completion** → the sampled sequence contains no call; no
  parser can invent one. Model-choice lane, and the remaining question is a
  *rate*, not a defect (next section).
- **marker in the completion, no `tool_calls` in the response** → parser lane;
  go to `index.md` rule 1 / rule 4 (`return_token_ids: true`).
- **no completion dump** → ask for it; that request is the first artefact, and
  it is one call for the reporter (`POST` the rendered prompt to
  `/v1/completions`, or add `return_token_ids: true` to the same chat body).

## The four dumps and what each one exonerates

| dump (names from #12658) | the field that decides something |
|---|---|
| request body `bad_case.json` | `tools` present and their names, `chat_template_kwargs` (`thinking` / `enable_thinking`), `stream`, `tool_choice` |
| rendered prompt `rendered_prompt.json`, or `POST /v1/chat/completions/render` | does *this* thinking mode's prompt actually carry the tool grammar and the schemas (`'## Tools'`, the model's `tool_calls` block spelling, each tool name)? which think token closes the prompt (`…<｜Assistant｜></think>` = chat mode) |
| tokenized request `rendered.json` | `prompt_token_ids` count (join key to the completion), and `sampling_params` — **`temperature`** |
| raw completion `raw_completion.json`, or `POST /v1/completions` | `finish_reason`, the sequence `token_ids`, the marker scan, `usage.prompt_tokens` (join key) |

Read them in that order: prompt-side facts first, because a prompt that never
taught the grammar is a rendering bug and not model behaviour — the script warns
on `tool grammar in prompt: NONE`.

**Join the dumps before believing them.** In #12658 the shipped body's last user
message is 香港 while the rendered prompt and the completion answer 上海; the
pair that does belong together is `rendered.json` (5944 prompt ids) and
`raw_completion.json` (`usage.prompt_tokens: 5944`). Compare by token count, say
which pair you used, and do not blend a body with a completion from another
request.

## Rule — with no call in the model's output, the deliverable is a rate, not a mechanism

Say it in the first message and stop there: *the engine returned exactly what the
model produced; nothing downstream dropped a call.* Then give these four rows,
cheapest first, each one command:

1. **N repeats of the identical body** (N ≈ 100-1000 for a "certain probability"
   claim; the #12658 reporter used a 1000-request script) and count requests with
   an empty `tool_calls`. This converts "sometimes" into a number.
2. **Same body, thinking on** (`chat_template_kwargs: {"enable_thinking": true}`
   for DeepSeek-style renderers, `{"thinking": true}` where that is the key). If
   the tools-side prompt is identical and only the mode changes the rate, the
   behaviour is the model's per-mode policy.
3. **The vendor's own API for the same model**, same messages / tools /
   temperature. Reproduces → close as model behaviour, not a vllm-ascend defect.
   Reproduces only on the quantised Ascend engine → the numeric lane (w8a8
   quantisation, or the sampler) is back in play.
4. **`temperature: 0` repeat.** A rate that survives greedy decoding is not
   sampling; it re-opens the sampler / spec-decode bookkeeping rows
   (`../spec-decode-mtp/index.md`), so report both the sampled and the greedy
   rate. A body with `temperature: 1.0` (the #12658 `sampling_params`) makes a
   non-zero miss rate expected from sampling alone — that is the explanation, not
   a fix, so do not propose a code change for it.

## Recorded case #12658 (facts read from the uploaded dumps, not recalled)

Report: `[v0.21.0rc1] DeepSeek-V4-Flash-w8a8-mtp, Non-think, tool call empty`,
Atlas 800I A2, TP8 + EP, `--tool-call-parser deepseek_v4 --enable-auto-tool-choice
--reasoning-parser deepseek_v4`, MTP n=1, `FULL_DECODE_ONLY`,
`--async-scheduling`; pins vllm-ascend v0.21.0rc1 (`80610e44`) + vllm v0.21.0
(`ad7125a4`).

| dump | fact |
|---|---|
| `bad_case.json` | `stream: false`, one tool `get_weather`, `chat_template_kwargs {"enable_thinking": false}`, user message 香港 |
| `rendered_prompt.json` | 10 043 chars, `## Tools` present, `<｜DSML｜tool_calls>` + `<｜DSML｜invoke` present, `get_weather` present, ends `<｜Assistant｜></think>` |
| `rendered.json` | 5944 prompt ids, `temperature 1.0`, `top_p 1.0`, `max_tokens 127176`, `stream: false` |
| `raw_completion.json` | `finish_reason: "stop"`, `stop_reason: null`, 132 sequence tokens, 248 chars of Chinese prose, **no `｜DSML｜` anywhere**, `usage.prompt_tokens: 5944` |

Verdict the dumps support: prompt rendering fine, tools rendered in chat mode,
no call in the model's own tokens ⇒ model-choice lane. The reporter's later
public comment confirms it: in a 1000-request run non-thinking mode missed the
tool call with "a certain probability" while thinking mode always called, and the
official `dsv4-flash` API reproduced the non-thinking miss — so quantisation and
the engine are excluded and the issue is the model in non-thinking mode.

The recorded run never opened these files (a token grep of the attachments
manifest returned nothing, which it read as "no attachments") and spent its whole
budget inside `vllm/parser/abstract_parser.py`, `patch_deepseek_v4_thinking.py`
and the DSV4 encoding module, shipping no answer — `../evidence-workspace/index.md`
(attachments) and `../evidence-workspace/recorded-cases-accuracy-and-output.md` (this case).

## Owners this page excludes, with the check that excludes them

- **DeepSeek-V4 tool-call parser patch.** `vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py`
  reshapes *how* an already-present `<｜DSML｜tool_calls>` block streams
  (`_ensure_parser_regexes` `:43`, `_drain_pending_tool_call_deltas` `:403`,
  assigned `:785`); §12 of `vllm_ascend/patch/__init__.py` states its scope as
  "does not guarantee incremental `arguments` emission". It cannot create a call
  the sequence lacks, and nothing else in the tree calls
  `drain_pending_tool_call_deltas` — a symptom in *this* page's shape is not it.
- **Thinking/reasoning-effort patch.** `patch_deepseek_v4_thinking.py` maps
  `reasoning_effort` → `chat_template_kwargs["enable_thinking"]` (`:59-63`) and
  the tokenizer's `thinking_mode` (`:91-106`) — prompt side. If the rendered
  prompt shows the chat-mode tail, it did its job; the output text is the model's.
- **`tool_calls: []` vs an absent key.** `patch_tool_choice_none_content.py:40-54`
  drops `"tool_calls": []` from the dumped response body, so a client may see no
  key at all. Cosmetic, downstream, and not a cause — quote it before a reporter
  chases "empty tool call" wording.
- **`finish_reason`.** It is `"tool_calls"` only when a parser reported a call
  (`vllm/entrypoints/openai/chat_completion/serving.py:1093-1097` at vllm
  v0.21.0), so `"stop"` here is consistent with "no call", not with a lost one.
- **Text marker vs parser input.** At this ref the non-streaming chat path calls
  the parser with `token_ids`, not text (`serving.py:1070-1078`) — so a marker
  scan of `content` alone is not a statement about the parser's input. That is
  why the raw `/v1/completions` dump (the model's own text for the same prompt)
  is the artefact this page works from.
