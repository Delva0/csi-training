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
`json.loads` on the assembly. It also lists the `stop_reason` chunks, the
`finish_reason` chunks, the last non-empty argument chunk, and any chunk whose
`arguments` is the empty string — because "the tail arrived as `""`" is the shape
that hides a lost tail. It tolerates HTML/Excel clipboard wrappers, console
prefixes (`decoded ok: data: {...}`) and `[REDACTED]` placeholders (those inject
brackets, so read the braces and quotes, not the bracket pair).

Calibrations on the recorded pastes:

| Paste | Result |
|---|---|
| #7580 opencode reproduction, 11 chunks, `num_speculative_tokens: 3` | one tool call, 10 fragments, assembled `{"filePath":"/…/K8sYamlAgent.java` — `{` 1 / `}` 0, odd quote count, `json.loads` → `Unterminated string starting at: line 1 column 13 (char 12)`, which is the reporter's own error verbatim. Tail missing while `finish_reason: "tool_calls"` follows. |
| #8122 Qwen3-Next-80B-A3B-Instruct response dump, 97 chunks | one tool call, 95 fragments, assembled 268-char arguments, braces 1/1, 40 quotes (even), `json.loads` OK. The pasted stream is healthy — so that sample was not one of the "大概率截断" requests and no mechanism should be built on it. |

## Step 3 — route on the outcome

**A. Assembled JSON parses.** The server stream is self-consistent. The failure is
in the client (or in whatever re-serialized the arguments): a parser that reads a
single chunk instead of the concatenation, a strict schema, a duplicate id, an
`arguments` object re-encoded by a proxy. Ask for the client version and its
parser; do not modify the serving command. Also check the fragments are not
duplicated or reordered — the script prints them in order, and a repeated
fragment set means an id/`index` mix-up rather than a parser bug.

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

**D. Any other mid-string damage** — duplicated or reordered fragments, a brace
pair inside a string, two requests' deltas interleaved. The text itself is wrong:
a generation/decoding report, not a protocol question.

The cheap discriminator between C and D: reassemble the fragments and compare the
*parser's* final string with what the deltas accumulated to. Where the assembled
string is the valid JSON **minus one character before the last streamed fragment**,
it is the delta arithmetic, not the model.

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

## Ownership cheat sheet (vllm v0.17.0 / vllm-ascend v0.17.0rc1)

| Symptom in the paste | Owner |
|---|---|
| assembled arguments valid, client errors | client accumulation / proxy |
| `stop_reason: <int>` on a chunk | upstream vLLM field carrying the stop token id |
| `finish_reason: "tool_calls"` with short arguments | the stream ended early — engine or delta, decided by `return_token_ids` |
| arguments valid but the model answered only CoT / no `tool_calls` | parser+flag gate: `--tool-call-parser`, `--enable-auto-tool-choice`, `--async-scheduling` row |
| balanced braces, odd quotes, no opening quote on a value, `stream: false` clean | the tool parser's delta arithmetic, keyed on the argument's declared schema type (`arg-type-gate-unquoted-value.md`) — upstream vLLM, not the model and not vllm-ascend |
| token ids short with `num_speculative_tokens >= 3` | vllm-ascend sampler (`sample/rejection_sampler.py:114`) |
| token ids short with `num_speculative_tokens <= 2` | not block verify: look at the stop-token/`parse_output` trimming and the model's own early stop |
