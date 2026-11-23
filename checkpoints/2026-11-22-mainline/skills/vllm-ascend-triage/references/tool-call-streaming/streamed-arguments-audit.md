# Auditing a streamed tool call: who lost the arguments

## When this page applies

A client-side JSON error or a "truncated / malformed / repeated function call"
report on an OpenAI-compatible endpoint, with the deployment healthy. The engine
did not crash; the question is which layer produced the bytes the client could
not parse.

## Step 1 — get the bytes, not the error

The reporter's paste of the client's exception is not evidence about the stream.
Ask for one of, in order of value:

1. `curl -N -s http://<host>:<port>/v1/chat/completions -H 'Content-Type:
   application/json' -d @body.json` — the raw SSE body, `data:` lines intact;
2. the reporter's own echo of every `data:` line (the #7580 reporter ran exactly
   this and pasted the eleven chunks);
3. the request body (needed anyway: `stream`, `tools`, `tool_choice`,
   `return_token_ids`, sampling params) and the serve command.

An Open WebUI / opencode / LangChain transcript, a rendered tool panel or a
"JSON Parse error" line alone cannot be audited — say that and ask, it is one
question.

## Step 2 — run the audit

```bash
bash <skill_root>/scripts/sse-toolcall-audit.sh capture.txt      # or: ... - < capture.txt
```

It concatenates `choices[i].delta.tool_calls[j].function.arguments` in arrival
order, per tool-call index, and prints for each: fragment count, empty fragments,
the balance of `{ } [ ] "`, the first/last fragments, the assembled string, and
`json.loads` on the assembly, and — because a balanced assembly clears only the
*punctuation* — the parsed `key = value` list of every tool call, which is where a
value cut shows (`guest_phone = '138123'`). It also lists the `stop_reason` chunks, the
`finish_reason` chunks, the last non-empty argument chunk, and any chunk whose
`arguments` is the empty string — because "the tail arrived as `""`" is the shape
that hides a lost tail. Since #12261 it also prints a per-tool-call **verdict** —
`balanced`, `tail-lost` (an opening brace/bracket left open), `head-lost` (more
*closing* than opening braces: the assembly starts inside the object),
`quote-gap` (braces and brackets balance, the quote count is odd) or `malformed` —
and those map onto outcomes A/B/B2/C below: a `quote-gap` verdict is case C and a
`head-lost` verdict is case B2, **neither** is a lost tail, so do not overrule
either by hand. It also counts the chunks that carry neither `content` nor
`tool_calls` **and** no `finish_reason`/`stop_reason` — the terminal `delta: {}`
chunk is excluded, so the count is the buffering steps: a buffering tool parser
emits a run of them while it walks the model's XML, so the count alone is not a
fault. When `stop_reason` and `finish_reason` arrive in *different* chunks it
says so, because vllm sets both on one choice object (a re-serializer was in the
path). It tolerates HTML/Excel clipboard
wrappers, console prefixes (`decoded ok: data: {...}`) and `[REDACTED]`
placeholders (those inject
brackets, so read the braces and quotes, not the bracket pair).

Calibrations on the recorded pastes:

| Paste | Result |
|---|---|
| #7580 opencode reproduction, 11 chunks, `num_speculative_tokens: 3` | one tool call, 10 fragments, assembled `{"filePath":"/…/K8sYamlAgent.java` — `{` 1 / `}` 0, odd quote count, `json.loads` → `Unterminated string starting at: line 1 column 13 (char 12)`, which is the reporter's own error verbatim. Tail missing while `finish_reason: "tool_calls"` follows, and the two closing lines carry
`"stop_reason":154829` then `"finish_reason":"tool_calls"` — one logical finish chunk split
by the LiteLLM front end (`model: litellm-test`), and the int is a *declared* stop id
(case B below), i.e. the engine ended generation inside the string — confirm on a
proxy-free `curl -N`, since a re-serializer can add the field. |
| #7682 GLM-5-w4a8 paste, 29 chunks (`delta: {}` × 28) | one tool call, **one** fragment, 2 chars: assembled `"}` — `{` 0 / `}` 1, 1 quote, `json.loads` → `Unterminated string starting at: line 1 column 1 (char 0)`. Head-lost, not a lost tail: the non-streaming body returns the complete `{"todos": "…"}` and the assembly is its last two characters. |
| #8122 Qwen3-Next-80B-A3B-Instruct response dump, 97 chunks | one tool call, 95 fragments, assembled 268-char arguments, braces 1/1, brackets 1/1, 40 quotes (even), `json.loads` OK — **and a failing sample anyway**: the assembled values end `"guest_phone": "138123"}` where the request gave `13812345678`, so one *value* was cut while every piece of punctuation was present, and the closing `"}` arrived in the chunk that also carries `finish_reason: "tool_calls"`. Re-audited 2026-08 on the issue's own paste; the row previously read "the pasted stream is healthy — not one of the 大概率截断 requests", and a later session repeated that verdict to the reporter as `guest_phone` digits 全部到齐. |

## Step 3 — route on the outcome

**A. Assembled JSON parses — and the values are the next question, not the last.**
The punctuation lanes are clear; nothing yet shows the server emitted a complete
*value*. Compare the printed `key = value` list against the request's own constants
(a phone, date, id, item list or count whose length the prompt fixes): the #8122 row
above is a value cut from 11 digits to 6 under a perfect brace/quote balance, and the
"healthy, client owns it" reading of it cost a later session's answer. Values plausible
→ the failure is in the client (or in whatever re-serialized the arguments): a parser
that reads a single chunk instead of the concatenation, a strict schema, a duplicate
id, an `arguments` object re-encoded by a proxy. Ask for the client version and its
parser; do not modify the serving command. A value is short or implausible → treat it
as **B asked about a value instead of the tail**: `"return_token_ids": true` on the
same body splits an engine-side early stop from a parse/delta cut, and `stream: false`
shows whether the non-streamed arguments carry the whole value. Also check the
fragments are not duplicated or reordered — the script prints them in order, and a
repeated fragment set means an id/`index` mix-up rather than a parser bug.

**B. Tail missing** (unbalanced braces/quotes at the end, `finish_reason:
"tool_calls"` after it). The stream stopped short. Split engine from assembly with
one request: add `"return_token_ids": true` to the same body. The choice then
carries vLLM's per-chunk `token_ids` (`vllm/entrypoints/openai/
chat_completion/protocol.py:98`, filled only under `request.return_token_ids` at
`chat_completion/serving.py:1307-1310`, vllm v0.17.0).

- token ids also stop before the closing tokens → the engine emitted a short
  sequence: the model issued a stop token before closing the JSON. That is a
  decoding-correctness question, and with `num_speculative_tokens >= 3` it is the
  block-verify acceptance path — `mtp-block-verify-acceptance.md`.
- token ids complete but the text is short → the delta assembly in the serving
  path lost them; grep the tool parser's `extract_tool_calls_streaming` delta
  state and the `previous_text` slicing, not the sampler.

**Read the finish chunk's own `stop_reason` before buying that re-run.** The chat
choice carries a vLLM-only `stop_reason` (`protocol.py:120` on
`ChatCompletionResponseStreamChoice`, set from `output.stop_reason` at
`serving.py:1303`), and it is the *engine's* ending, while
`finish_reason: "tool_calls"` is a **frontend label**: once any tool call was
streamed the generator overwrites the engine's reason with `"tool_calls"`
(`serving.py:1291-1300`, `auto_tools_called or tools_streamed[i]`), so a
clean-looking `finish_reason` beside short arguments says nothing about why
generation stopped. At vllm v0.17.0 (`b31e932`) the value's meaning is
`vllm/v1/core/sched/utils.py:94-117`: last token `== sampling_params.eos_token_id`
finishes with `stop_reason` left **None**; a token in `stop_token_ids` finishes
with `stop_reason` = **that id**; a `max_model_len`/`max_tokens` cap leaves it
None. So an **integer** `stop_reason` in the paste is the engine naming a
*declared* stop id that ended the stream: the tail was never produced,
`return_token_ids: true` cannot change it, and the next question is where that id
came from (`--stop-token-ids`, or the checkpoint's `generation_config.json`
`eos_token_id` list through `SamplingParams.update_from_generation_config`) —
`../accuracy-benchmarks/early-eos-short-output.md` §*`stop_reason` names the
token*, then ablations 3-4 below for *why* the model emitted it. A **string**
`stop_reason` is a different owner: the recompute scheduler's `"recomputed"`
empty completion, with no model output. **Envelope check:** vllm sets both fields
on the same choice object and serializes with `model_dump_json(exclude_unset=True)`,
so a paste showing `stop_reason` in one chunk and `finish_reason` in the next
passed a re-serializer (a LiteLLM/proxy front end, a client SDK) — read the pair
as one finish chunk and do not build a chunk-ordering or "two endings" mechanism
on the split. One observation redirects the reading: if a direct `curl -N` to the
engine (no proxy in front) shows no `stop_reason` field at all, the proxy added it
and the id is not the model's.

**B2. Head missing entirely** — the assembly is only the object's closing
punctuation (`"}`; `{` 0 / `}` 1) while `stream: false` returns the full string,
and every other chunk is `delta: {}`. Nothing but a suffix of the arguments was
ever emitted, so the owner is the layer that keeps the streamed **length**, not
the model and not the client: the parser's `_compute_args_diff` /
`streamed_args_for_tool`, or the frontend's finish-step
`remaining_call = expected.replace(actual, "", 1)`. One arithmetic is the whole
diagnosis — `assembled == non_stream_args[-len(assembled):]` — and no parser read
is needed to state it. Note the run of `delta: {}` chunks is a *buffering* parser
(GLM walks `<tool_call>…</tool_call>` before emitting anything), so do not read
it as dropped tokens. Full mechanism, the two code sites and the
`--tool-call-parser` registry check that dates the export against the runtime:
`suffix-only-arguments.md` (#7682).

**C. Unbalanced in the middle, gap inside a value** — braces balanced, quotes
odd, no opening `"` after `": ` while the closing `"` and `}` are present
(`{"hvd": Pura900"}`). **Do not stop at "the text is wrong".** The braces and
quotes alone cannot separate the owners, and this shape is often the **parser's
incremental delta**: its own final `args_so_far` is valid JSON, the deltas are a
length prefix of it, and a character the parser inserted before the streamed
length never reaches the client (`_compute_args_diff`). The trigger is the
argument's declared schema type — check it with
`uv run --no-project python <skill_root>/scripts/tool-schema-type-audit.py
<request-body.json>`: if the failing argument has no top-level
`"type": "string"` (a nullable `anyOf`, for instance), that is the owner. Full
mechanism, working arithmetic and reprinter: `arg-type-gate-unquoted-value.md`
(#12261, GLM-5.1, vllm v0.20.2).

**D1. Missing head plus a repeated fragment** — the value loses its first
characters (`[`, an opening `"`) and a fragment reappears a few characters later,
`stream: false` clean, damage clustered where the script changes (ASCII ↔ 中文 ↔
arrows). The streamed **text** was damaged before the parser saw it: the
incremental detokenizer. Two sub-owners, and one log line or one comparison
decides which — `detokenizer-text-delta.md`:
- `content` carries the same damage at the same place → the detokenizer's text
  delta (`vllm/v1/engine/detokenizer.py`); grep the server log for
  `resetting decode stream` (the fast `DecodeStream` path's recovery) and ask for
  the image's `tokenizers` version.
- `content` clean, damage only inside `arguments` → the parser's own
  `_compute_args_diff` length prefix over a re-rendered `args_so_far`.

**D2. Any other mid-string damage** — reordered fragments, a brace pair inside a
string, two requests' deltas interleaved. The text itself is wrong: a
generation/decoding report, not a protocol question.

The cheap discriminator between C and D1: reassemble the fragments and compare the
*parser's* final string with what the deltas accumulated to. Where the assembled
string is the valid JSON **minus one character before the last streamed fragment**,
it is the delta arithmetic, not the model; where characters are missing at the head
*and* extra ones repeat later, one producing step emitted them and the text path
owns it.

## Step 4 — the ordered ablations

One axis per launch, cheapest first. Do not run them all before answering: name
the owner from steps 2-3 and give these as the confirmation ladder.

1. `return_token_ids: true` on the identical body — engine vs assembly (step 3B).
2. `stream: false` on the identical body — non-streamed `message.tool_calls`: if
   the arguments there are also cut, it is token-level and stream-independent.
3. `num_speculative_tokens: 2` (keep `method: deepseek_mtp`) — the `>= 3`
   threshold in `vllm_ascend/sample/rejection_sampler.py:114` is the only
   speculative-decoding branch keyed on the reported value. `2` must behave like
   `1`; if `2` also fails, the threshold is not the owner.
4. `temperature: 0` on the same body — greedy rows skip block verify
   (`non_greedy_mask`), so this isolates the acceptance path from everything else
   in the decode step.
5. `--async-scheduling` off (reported together with `stream: true` and a tool
   parser in #5683 and #8122) and `--tool-call-parser` as the model's tutorial
   names it. Note for the answer whether the model+flag pair exists in CI:
   `rg -n 'async_scheduling' tests/e2e/**` — at vllm-ascend v0.18.0rc1 no
   `async_scheduling` appears in `tests/e2e/multicard/4-cards/test_qwen3_next.py`,
   so "covered by CI" is not available for that axis.

## Step 5 — what the answer owes

- The audit result on the pasted bytes, quoted (the assembled string's tail, the
  brace/quote balance, the `json.loads` message). That sentence is the answer's
  evidence, and it is often the same string the client printed.
- Which layer owns it, with the label: source-verified at commit X (the branch /
  field / filter), reporter-confirmed (an A/B the reporter ran), or hypothesis
  (a mechanism the paste cannot decide).
- The `return_token_ids: true` request as the next step, plus the ablation row
  that confirms the named layer.
- A refusal to build a mechanism out of a healthy stream: if the audit parses,
  say the pasted sample is not the failing one and ask for a failing capture.
- For the mid-string gap (outcome C), the argument's schema as it appears in the
  reporter's own request body, because that is what the workaround changes: a
  plain `"type": "string"` (or `stream: false`) is the reporter's own fix and it
  needs no release.
- For a "repeated output" phrasing, one extra question: repeated *fragments*
  (protocol) versus a repeated *sentence* (generation). The script answers the
  first; the second belongs with `../accuracy-benchmarks/short-prompt-corruption.md`
  and the sampler rows there.

## Ownership cheat sheet

Paths and line numbers here were checked at vllm v0.17.0-v0.23.0 /
vllm-ascend v0.17.0rc1-v0.23.0; re-read the branch at the reporter's ref when the
report is about a version the topical page does not name.

| Symptom in the paste | Owner |
|---|---|
| assembled arguments valid, client errors | client accumulation / proxy |
| `stop_reason: <int>` on the finish chunk | the engine's own ending: a **declared** stop id (the `stop_token_ids` set, `vllm/v1/core/sched/utils.py:105-111` at v0.17.0 — a primary-eos stop leaves it None) ended generation there, so no tail exists to recover and `return_token_ids` is moot; compare the id with `--stop-token-ids` / the checkpoint's `generation_config.json`, then read `../accuracy-benchmarks/early-eos-short-output.md` and the `>= 3` block-verify ablation for why the model emitted it |
| `finish_reason: "tool_calls"` with short arguments | the stream ended early — but `tool_calls` is only a frontend label (`serving.py:1291-1300`), so split engine from delta with that chunk's `stop_reason` first, and with `return_token_ids` only when it is absent |
| arguments valid but the model answered only CoT / no `tool_calls` | parser+flag gate: `--tool-call-parser`, `--enable-auto-tool-choice`, `--async-scheduling` row |
| assembled string is only the closing `"}` / `}` (more closing than opening braces), all other chunks `delta: {}`, `stream: false` clean | the delta **bookkeeping** length, in its extreme form: only a suffix was ever emitted — `suffix-only-arguments.md`; test `assembled == non_stream_args[-len(assembled):]` before any parser read |
| balanced braces, odd quotes, no opening quote on a value, `stream: false` clean | the tool parser's delta arithmetic, keyed on the argument's declared schema type (`arg-type-gate-unquoted-value.md`) — upstream vLLM, not the model and not vllm-ascend |
| leading characters of a value missing **and** a fragment repeated later, `stream: false` clean, often at a script boundary | the streamed text: the incremental detokenizer (`detokenizer-text-delta.md`) when `content` is damaged identically, the parser's length diff when only `arguments` is — one grep for `resetting decode stream` separates them |
| token ids short with `num_speculative_tokens >= 3` | vllm-ascend sampler (`sample/rejection_sampler.py:114`) |
| token ids short with `num_speculative_tokens <= 2` | not block verify: look at the stop-token/`parse_output` trimming and the model's own early stop |
