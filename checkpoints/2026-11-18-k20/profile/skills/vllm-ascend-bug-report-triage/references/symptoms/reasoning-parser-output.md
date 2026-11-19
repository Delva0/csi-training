# Reasoning output on the chat endpoint: empty `reasoning`, thinking that "disappears", slow first token

## Scope

A `--reasoning-parser <name>` deployment (Qwen3/Qwen3.5, GLM, DeepSeek, MiniMax, …) — or the same service
with no parser at all — where the report says the thinking text is:

- `reasoning_content` always `null` while `content` holds either the whole `<think>…</think>text` blob (no
  parser configured) or only the answer (parser configured);
- gone once `--reasoning-parser <name>` is added, with the client describing the first token as far slower;
- absent from the **second turn** of a tool-call loop: turn 1 returns reasoning, turn 2+ returns `content` +
  `tool_calls` with none (「偷懒」);
- `KeyError: "Reasoning parser '<name>' not found. Available parsers: …"` at startup,

while the service serves normally and **nothing raises** in the first three shapes. The owning code is the
upstream vLLM frontend (`chat_completion/serving.py`, `entrypoints/chat_utils.py`, `reasoning/`), not a
`vllm_ascend/` module: at `v0.18.0rc1`/`v0.18.0` no `vllm_ascend/` file touches the parser
(`git grep -l reasoning <ref> -- vllm_ascend` → `patch/__init__.py`,
`patch/platform/patch_glm_tool_call_parser.py`, `patch/platform/patch_minimax_usage_accounting.py`), and the
reasoning-specific downstream patches start later (`patch/platform/patch_deepseek_v4_thinking.py` on the
`v0.20` line, `patch/platform/patch_glm53_reasoning.py` at `main`).

One exception worth knowing: `patch/platform/patch_tool_choice_none_content.py` (first tag `v0.18.0`) exists
because a reasoning parser can consume the whole output and leave `content=None`, which forced named tool
choice then treated as an error.

Not this page: the *shape* of `tool_calls`/deltas while the body is otherwise sane
(`tool-call-streaming.md`); an accuracy score below the vendor number or a DeepSeek-V4 thinking **mode**
question (`deepseek-v4-thinking-and-effort.md`); a chat-template `ValueError` at render time
(`tokenizer-and-chat-template.md`); a `Tokenizer class … does not exist` pin floor
(`model-type-support-floor.md`). A parser registry `KeyError` at startup is the registry rule here, not a
load defect.

## Common rules

- **`reasoning_content` is not the response field from vLLM `v0.16.0` on.** `reasoning` replaced it:
  `ChatMessage` and `DeltaMessage` carry `reasoning` only from `v0.16.0` on (checked at `v0.16.0`, `v0.17.0`,
  `v0.18.0`, `v0.20.2`, `v0.23.0`: `entrypoints/openai/chat_completion/protocol.py:65`/`:64`/`:64`/`:64`/`:67`,
  `entrypoints/openai/engine/protocol.py:261`). At `v0.14.0`/`v0.15.0` both keys existed
  (`protocol.py:1499-1500`, then `engine/protocol.py:259-260`). The removal commit is `c5113f60f2`
  (#33402 "Remove deprecated `reasoning_content` message field"), first tag **`v0.15.2rc0`/`v0.16.0rc1`**;
  request-side normalisation (`reasoning_content` → `reasoning`) only returns with `346cf163a1` (#42664),
  first tag **`v0.22.0rc1`**, and it is a *request* normalisation, not an output alias. `reasoning_content`
  survives in the Responses API path (`entrypoints/openai/responses/utils.py:209`,
  `parser/responses_parser.py:64`) and as a deprecated **request-side** key. So a client or curl recipe
  printing `delta.reasoning_content` shows `null` on a perfectly working `0.16`–`0.21` server — the most
  common "the parser does not work" report.
- **The request-history field is `reasoning` too.** `_parse_chat_message_content` reads
  `reasoning = message.get("reasoning")` (`vllm/entrypoints/chat_utils.py:1502`/`:1511` at `v0.18.0`) and
  renders it for interleaved thinking; the same commit `c5113f60f2` dropped the earlier
  `or message.get("reasoning_content")` fallback that #28531 had shipped. `OpenAIBaseModel` is
  `extra="allow"` (`entrypoints/openai/engine/protocol.py:28`), so an agent framework that echoes
  `reasoning_content` back is **accepted and silently ignored** — the previous turn's thinking leaves the
  prompt with no warning at all.
- **A parser moves thinking out of `content`; it does not delete it.** `serving.py:1381-1387` at `v0.18.0`:
  with a parser `reasoning, content = reasoning_parser.extract_reasoning(output.text, request=request)` and
  `if not request.include_reasoning: reasoning = None` (`:1384`); without one,
  `reasoning = None; content = output.text`. So the reporter's two cases (no parser → thinking + `</think>`
  inside `content`; parser → answer-only `content`) are the *same* server behaviour under two flags. Read
  the function once; do not infer "the parser dropped my text" from `content` alone.
- **Two prompt states decide the whole behaviour.** `BaseThinkingReasoningParser.is_reasoning_end`
  (`vllm/reasoning/basic_parsers.py:64-73`) scans the prompt's token ids backwards and returns `True` as
  soon as it meets the end token before any start token. The serving loop computes it once per choice
  (`chat_completion/serving.py:686-691`) and, when it is true, routes **every** delta as `content`
  without calling the parser (`:998-1001`, "when only reasoning"). So:
  - prompt ends with the think-open marker (thinking on) → everything before `</think>` becomes
    `reasoning`, and `content` stays `None` until `</think>` (`qwen3_reasoning_parser.py`
    `extract_reasoning_streaming`; tests `WITHOUT_THINK`/`WITHOUT_THINK_STREAM` in
    `tests/reasoning/test_qwen3_reasoning_parser.py` assert "all output is reasoning" when no `</think>`
    appears);
  - prompt already ends with the think-**end** marker (the template wrote `<think>\n\n</think>\n\n` because
    thinking was disabled) → `reasoning` is `null` and `content` carries the answer. **No thinking was
    generated** — nothing was "dropped" by the parser.
- **`include_reasoning` and `reasoning_effort: "none"` can discard it on purpose.** Default
  `include_reasoning: bool = True` (`chat_completion/protocol.py:183`/`:184` by tag). At `v0.18.0` the
  validator `set_include_reasoning_for_none_effort` flips it to `False` when the request says
  `reasoning_effort: none` (`:784-786`), and only the non-streaming branches plus the harmony path consult
  the flag (`serving.py:1329`, `:1384`, `:766`) — the ordinary streaming path does not, so
  `include_reasoning: false` is not a reliable way to hide reasoning from a streamed request there. The
  validator is gone at `v0.20.2`, whose streaming path *does* gate on it (`serving.py:327`, beside `:1334`,
  `:1389`, `batch_serving.py:272`). Read the flag's readers at the reported tag instead of assuming either
  behaviour; a gateway that injects `reasoning_effort: "none"` produces exactly "thinking gone, reasoning
  empty".
- **`--reasoning-parser` and `--tool-call-parser` are different registries, and the error is upstream's.**
  `Reasoning parser '<name>' not found. Available parsers: …` is `vllm/reasoning/abs_reasoning_parsers.py:198`
  at `v0.18.0`; the names it prints are that tag's `_REASONING_PARSERS_TO_REGISTER` keys in
  `vllm/reasoning/__init__.py` (at `v0.18.0`: `deepseek_r1`, `deepseek_v3`, `ernie45`, `glm45`,
  `openai_gptoss`, `granite`, `holo2`, `hunyuan_a13b`, `kimi_k2`, `minimax_m2`, `minimax_m2_append_think`,
  `mistral`, `nemotron_v3`, `olmo3`, `qwen3`, `seed_oss`, `step3`, `step3p5`). The tool names come from
  `vllm/tool_parsers/__init__.py` (`qwen3_coder`, `qwen3_xml`, `hermes`, `llama3_json`, …), so a name from
  that list is never in the reasoning registry and fails at startup. The newer `vllm/parser/` package
  (`abstract_parser.py`, `parser_manager.py`, `minimax_m2_parser.py`) is a different, coexisting registry at
  the same tag — read `vllm/reasoning/` for reasoning names.
- **The qwen3 parser already handles the Qwen3.5 no-start-token template at `v0.18.0`.**
  `vllm/reasoning/qwen3_reasoning_parser.py` documents it ("Starting with Qwen3.5, the chat template
  places `<think>` in the prompt so only `</think>` appears in the generated output"), implements the
  `prompt_is_reasoning_end` shortcut, and its test parametrises `Qwen/Qwen3.5-397B-A17B`. "Ascend has no
  Qwen3.5 thinking support" is therefore a wrong answer for that tag.
- **A turn-2 loss is decided by the prompt.** The parser is per-response and stateless, so only the prompt
  can differ: `chat_utils.py:1511` reads `message.get("reasoning")` — the deprecated `reasoning_content`
  field is **not** read — and re-emits it to the template as both `reasoning` and `reasoning_content`
  (`:1537-1540`). The client must append the response's `reasoning` string to the assistant message it
  sends back; the standard OpenAI loop that appends `content` + `tool_calls` only drops every prior
  thinking block. vLLM's own interleaved-thinking doc does exactly that
  (`docs/features/interleaved_thinking.md`, example `"reasoning": response.choices[0].message.reasoning`)
  and its support table lists **only** `kimi_k2` and `minimax_m2` — unchanged at `v0.18.0`, `v0.20.2`,
  `v0.23.0`, `v0.26.0` and `HEAD`. Qwen3.5 is in no version of that table.
- **A client-side TTFT that measures `delta.content` is not a server slowdown.** With a parser and
  thinking on, every pre-`</think>` chunk carries `reasoning` and `content=None`, so such a client times
  the whole thinking phase as "first token". Ask which field and which chunk the measurement reads before
  treating it as a performance defect.
- **Do not reconstruct a chat template.** Qwen3.5's template lives in the model repo, not in either clone
  (`git grep -rn preserve_thinking` is empty at `v0.18.0` and `HEAD`), so what a template does with
  `reasoning`/`enable_thinking` is `hypothesis` plus a request-side probe (`chat_template_kwargs`, a
  rendered-prompt dump), never a derivation from memory.

## Procedure (one batch; `scripts/reasoning_output_probe.sh`)

```bash
bash scripts/reasoning_output_probe.sh <vllm-ascend-ref> [vllm-ref]   # exit 3 = no such clone/ref
```

It prints the paired tags with dates, `ChatMessage`/`DeltaMessage` field names, the `reasoning_content`
occurrence list, the history-field read, the `include_reasoning` sites, both parser registries, whether the
qwen3 parser knows Qwen3.5, and any vllm-ascend reasoning patch at the ref. Then two reads settle it:

1. **Pin the vLLM tag** (`.github/vllm-release-tag.commit` at the reporter's ref, else the same-version
   rule) and read the field name in one call:
   `git grep -n "reasoning: str" <vllm-ref> -- 'vllm/entrypoints/openai/*protocol.py'
   'vllm/entrypoints/openai/*/*protocol.py'`. Two fields (`reasoning`, `reasoning_content`) ⇒ pre-0.16
   semantics; one ⇒ any client printing the other key reports a false failure.
2. **Read the model family's parser at that tag**:
   `git show <vllm-ref>:vllm/reasoning/<name>_reasoning_parser.py` (plus `basic_parsers.py` for
   `BaseThinkingReasoningParser`), then the serving branch that applies to the reporter's flags
   (`serving.py` "when only reasoning" / "auto tool choice and reasoning parser" / "named tool choice") and
   decide which of the two prompt states they are in.

Classify the report into one of four, then answer: (a) the client reads the wrong field — the workaround is
the field name, no server change; (b) the thinking moved to `reasoning` and the client only renders
`content` — same; (c) `include_reasoning`/`reasoning_effort: none` discarded it; (d) later turns lose it —
the client must round-trip `reasoning`. Only after (a)–(d) are excluded is
a parser defect on the table, and only then look for a fix lineage
(`git log -S'include_reasoning' --oneline <vllm-ref>..`, `git log --oneline --grep='interleaved'`, the release
notes' `Known Issues` at the reporter's tag). That is enough for the reply; the chat template itself lives in
the model repo and cannot be read from these clones.

## Apparent TTFT slowdown (part of this class, not a performance bug)

With a parser configured, `extract_reasoning_streaming` emits `DeltaMessage(reasoning=…)` for every chunk
before `</think>` and `content=None` (`vllm/reasoning/qwen3_reasoning_parser.py`); the content that shares
the `</think>` chunk is even deferred one chunk (`serving.py:879-883`, and `:940-949` on the
`tool_choice="auto"` path: `current_text = delta_message.content; delta_message.content = None`). A client
that measures first token by the first **`delta.content`** therefore sees TTFT ≈ prompt + the whole thinking
block, with unchanged prefill latency. Reproduce/refute by timing first `delta.reasoning` versus first
`delta.content` on the same prompt, or with `stream: false` — one measurement, not an engine hunt. Keep the
real candidates (`--enforce-eager`, graph mode, quantisation) for a report that shows slow *decode* or slow
*prefill*.

## Worked case A: #8022 — `reasoning_content` null, thinking "lost", slow first token (`0.18.0rc1`)

Reporter: vllm-ascend `0.18.0rc1` (paired vLLM `0.18.0`), Qwen3.5-27B, A2-class hardware, streaming. Case 1
(no `--reasoning-parser`): `reasoning_content` null, thinking + `</think>` inside `content`, speed normal.
Case 2 (`--reasoning-parser qwen3` or `deepseek_r1`): `reasoning_content` still empty, `content` now holds
only the answer, first token much slower. Thread: `qwen3_coder` fails as a reasoning parser with the name
list above; `chenxi-hh` posts a raw body whose field is `"reasoning"` on a newer build; `LoganJane` closes it
「resolved in the latest version … use the latest release or rc version」 (stale bot, no further evidence).

- **Mechanism, `verified` for the code half:** the parser works and its output is in `message.reasoning` /
  `delta.reasoning`; the client reads `reasoning_content`, which does not exist at that tag. Case 1 is the
  no-parser branch (`reasoning = None; content = output.text`), Case 2 the parser branch, where `content` is
  withheld until `</think>`. The perceived TTFT follows from that withholding (§"Apparent TTFT slowdown"
  below). The thread's `KeyError: Reasoning parser 'qwen3_coder' not found` is a **tool**-parser name under
  `--reasoning-parser` (the registry rule).
- **Workaround:** read `reasoning` (and `delta.reasoning`) in the client, or upgrade to a build whose
  frontend normalises the request field (`>= v0.22.0rc1` for the request half; the response field stays
  `reasoning`). Cost: a client change, or a version move.
- **Second candidate to rule out, cheaply:** a gateway adding `reasoning_effort: "none"`
  (`include_reasoning=False`) — the falsifier is the request body as it reaches the server.
- Conclusion: field/tooling mismatch plus client-side TTFT accounting, not a vllm-ascend defect. Status: no
  maintainer diagnosis in the thread; the reporter never confirmed.

## Worked case B: #9659 — Qwen3.5-27B loses reasoning from the second tool-calling turn (`0.18.0`)

Reporter: vllm-ascend `0.18.0`, Qwen3.5-27B, A2, `--enable-auto-tool-choice --tool-call-parser qwen3_coder
--reasoning-parser qwen3`; turn 1 returns reasoning then a tool call, later turns return `content` +
`tool_calls` with no reasoning. Thread: two ping comments (`@LoganJane` 2026-05-29, then `@chenxi-hh`
2026-06-11), issue open, no diagnosis.

- **The parser is not the defect and cannot be:** it is stateless per request, so turn 2 differs only in
  the prompt. The reported shape (`content` + `tool_calls`, `reasoning` null) is the
  `prompt_is_reasoning_end` state — the rendered prompt already closed the think block. Candidates, in
  order: the client sends `chat_template_kwargs: {"enable_thinking": false}` on later turns (a common
  agent-framework token-saving default); the framework echoes the previous assistant message with
  `reasoning_content` instead of `reasoning`, so interleaved thinking silently loses the history; the
  model's own template drops previous-turn thinking (Qwen documents that behaviour for multi-turn). The
  "model is lazy" reading is not evidence.
- **Falsifier (one request pair):** log the rendered prompt for turn 1 and turn 2 and compare the tail —
  whether it ends with the think-open marker decides between "thinking disabled in the prompt" and
  "thinking generated but parsed away". Also log the incoming request body: the field name
  (`reasoning` vs `reasoning_content`) and `chat_template_kwargs` are visible there. If the turn-1
  `<think>…</think>` block *is* present and the output still has no thinking, the remaining owner is the
  model's template/weights, not vLLM-Ascend.
- **Sibling report with the same shape, one turn:** #9355 (`v0.18.0`, `--reasoning-parser qwen3`) sends
  `chat_template_kwargs.enable_thinking=false` explicitly — a report whose own request disabled thinking is
  configuration, not a defect (`issue-ledger/reports-later.md`).

## Answer shape

- `symptom → mechanism`: which field the client reads, and which of the two prompt states the run is in,
  scoped to the tag.
- `workaround`: the field rename in the client (`reasoning`), the version that normalises the request
  field (`>= v0.22.0rc1`), the explicit `enable_thinking` key when mode is the question, or — for a turn-2
  loss — appending `reasoning` (the response field) to the assistant message, as vLLM's own doc example
  does. Say whether the reporter confirmed it.
- `root cause`: for the field half, upstream vLLM `c5113f60f2` + the tag's `protocol.py:line`; for the
  history half, `chat_utils.py:1511` reading only `reasoning`; for the mode half, the rendered prompt, with
  the template outside these clones; for the `KeyError`, the tag's reasoning registry.
- `status and limits`: filed-as-bug reports here are usually client or config; name what you did not read
  (the checkpoint's chat template, the client's request body).

## Falsifiers and limits

- A monkeypatched or vendored frontend (`site-packages/vllm/…` frame, an image that pins a different vLLM)
  invalidates every line number here — read the field at *that* tag.
- If the response carries `reasoning` **and** the client reads it, and `content` still contains a
  `</think>`, the parser never ran: check that the flag reached the server (log line, `vllm serve` args)
  before blaming the parser.
- A genuinely empty `reasoning` with a parser configured and `include_reasoning` true, on a model whose
  output has no `</think>`, is the parser reporting a truncated/preempted generation
  (`extract_reasoning` returns `(model_output, None)`), so look at `finish_reason`/token budget first.
- #9659's template-level half is `hypothesis` with its falsifier (dump the rendered turn-2 prompt); the
  chat template lives outside these clones. Nothing here was reproduced on NPU.

## Recheck conditions

The field name, the parser registry and the serving branches moved several times: at `main` the chat
message is still `reasoning` (`vllm/entrypoints/openai/chat_completion/protocol.py:88`), the renderer moved
to `vllm/renderers/chat_utils/`, and parsers moved under `vllm/parser/` (registry
`vllm/parser/engine/registered_adapters.py`, `vllm/config/reasoning.py`), so a grep for
`vllm/reasoning/<name>_reasoning_parser.py` at a new tag can return nothing without the feature being gone.
Re-run the probe script at the reporter's pair; never quote these line numbers at another ref.
