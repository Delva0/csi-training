# Tool-call / function-call output integrity

## Scope

The deployment is up, the engine answers, and the complaint is about a **tool
call** rather than about prose. Entry shapes, in the order they were recorded:

- the client raises a JSON error while assembling the call: `JSON Parse error:
  Unterminated string`, `Expecting ',' delimiter`, `Unexpected end of JSON
  input`, `Unterminated string starting at: line 1 column N` — often after
  `--speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}'`
  where `num_speculative_tokens: 1` "正常" (#7580);
- the arguments are reported truncated / occasionally malformed / repeated:
  "function call 模型返回时大概率会截断输出，小概率会格式错误，重复输出" (#8122,
  Qwen3-Next-80B-A3B, `stream: true`, `--async-scheduling`);
- `stream: true` plus a tool parser returns only CoT and no `tool_calls`
  (#5683, reporter-confirmed A/B: async scheduling off → correct call);
- the assembled `arguments` has **balanced braces but an odd quote count**, the
  damage inside a value (`{"hvd": Pura900"}` — no opening quote, closing quote
  and `}` present), and `stream: false` does not reproduce it (#12261, GLM-5.1,
  `anyOf: [string, null]` argument schema). That shape is the parser's
  per-token delta arithmetic, not a lost tail and not a model fault —
  `arg-type-gate-unquoted-value.md`.
- the name parses but `arguments` is exactly `{}` (an empty JSON object, both
  braces present, nothing truncated), typically on a GLM-family serve with
  `--tool-call-parser glm47`: the GLM parser's literal output when a *complete*
  region carried no `<arg_key>/<arg_value>` pair, and the `stream: false` A/B
  splits a lost payload from an argument-format mismatch from a zero-argument
  call — `empty-arguments-object.md` (#14731);
- the assembled `arguments` are **only the closing punctuation** — `"}` with no
  `{`, no key — while the same body with `stream: false` returns the complete
  string, and every other chunk is `delta: {}` (#7682, GLM-5-w4a8,
  `--tool-call-parser glm47`, MTP n=3, `--async-scheduling`). That is the
  bookkeeping family's extreme form, not a lost tail and not the type gate:
  `suffix-only-arguments.md`.
- **the call is assembled and the client rejects it anyway** — Claude Code
  `invalid tool parameters`, a plugin's type-validation error, no JSON syntax
  error anywhere. Nothing is truncated and the audit can say `balanced`: the
  string is valid JSON that disagrees with the client's own tool schema (a value
  of the wrong type, a `null` where a string was declared, the model's own quotes
  kept, a required key missing, or the same arguments emitted twice under one
  `index`). The split is `stream: false` plus a field-by-field comparison against
  the client's schema, not another look at the delta arithmetic —
  `client-schema-invalid-arguments.md` (#9076, MiniMax-M2.7 + `minimax_m2`).
- **no call at all**: `tool_calls` empty or missing with `finish_reason: "stop"`
  and ordinary prose in `content`, often conditional on non-thinking mode or
  "概率性" ("tool call empty", #12658 DeepSeek-V4-Flash-w8a8-mtp,
  `enable_thinking: false`). The uploaded dumps decide it: the engine's own
  completion contained no tool-call marker, so the model chose not to call and
  the deliverable is a *rate* plus the mode/vendor-API ablations, not a parser —
  `no-tool-call-in-raw-output.md`, one call:
  `scripts/toolcall-raw-output-audit.py <attachments dir>`.
- **no call at all, but the text stops *mid-turn* right where the call should
  begin** — a lead-in line ending in `:` ("现在创建全局异常处理器:", "edit 工具不可用，我改用
  bash 的 Python 脚本来修改："), `finish_reason: "stop"`, `tool_calls` empty: that cut is
  itself the marker's fingerprint. `_extract_content` returns nothing after an
  **unclosed** `<tool_call>` (`glm4_moe_tool_parser.py:238-283`, the
  `pos = start; break` at `:271-274`), `_extract_tool_call_regions` marks the region
  `(raw, False)` (`:284-303`) and streaming `break`s once no name can be cut
  (`if not tool_name: break`, `:453`) — at vllm v0.23.0 `0fc695f`, so the region
  reaches neither `content` nor `tool_calls`. Rank the PARSER/HOLD-BACK lane, not the
  model-choice lane this bullet's prose shape belongs to, and decide it with
  `return_token_ids: true` (is the `<tool_call>` id in the stream?) plus a
  `stream: false` A/B; the same page's table row carries the split, and the
  gen-side owner (which stop token ended it: MTP n, `--async-scheduling`,
  `stop_reason`) follows (`no-tool-call-in-raw-output.md`, #17391).
- **`content` null and `tool_calls: []` in the same message**, `finish_reason:
  "tool_calls"`, on a serve that *does* pass `--tool-call-parser deepseek_v3`
  (`--enable-auto-tool-choice` on), often right after the reporter added exactly
  those flags to a report whose raw markers were showing in `content` — the
  DeepSeek-V3 extractor saw the start marker, matched no complete call, and
  returned `tools_called=True` with zero calls and `content=None`
  (`deepseek-parser-silent-drop.md`). Both halves are the answer: the first
  shape is a missing parser, the second is a parse failure, not a fix.
- **the stream itself dies server-side mid-request**, not at the client:
  `Error in chat completion stream generator` then a traceback whose deepest
  `vllm_ascend` frame is a `patch/**` tool-call parser returning `DeltaMessage(...)`,
  ending in `pydantic_core._pydantic_core.ValidationError: 1 validation error for
  DeltaMessage` / `tool_calls` / `Input should be a valid list [type=list_type,
  input_value=None, input_type=NoneType]`. The patch wrote an explicit `None` into a
  field the pinned protocol declares **non-Optional**
  (`tool_calls: list[DeltaToolCall] = Field(default_factory=list)`), so the frontend
  aborts the SSE generator and the client sees a truncated stream rather than a JSON
  error — `delta-message-none-field.md` (#10376).

Not this topic — leave with the linked route:

- the acceptance metric collapses (`Accepted: 0`, `Mean acceptance length: 1.00`),
  a stall with drafts still produced, or a first response with `content: ""` and
  `completion_tokens: 1` → `../spec-decode-mtp/index.md`;
- long-prompt 乱码, short-prompt corruption, a VL `!!!!` run, or an
  accuracy/score gap → `../accuracy-benchmarks/index.md`;
- the thinking text sits in `content` / `reasoning_tokens: 0` →
  `../accuracy-benchmarks/reasoning-output-routing.md`;
- nothing ever started → `../startup-failures/index.md`; a traceback **elsewhere**
  (a device dump, a collective, a model/op frame) → `../hccl-and-hardware/index.md`
  (lane C) or `../upstream-downstream-drift/index.md` — but a traceback raised while
  the SSE generator is building deltas, whose deepest frame is `patch/**`, is the
  server-side shape above, not a route away.

## Rules

**Before rule 1, split the shape: is there a call to damage at all?** Rule 1
audits an assembled call; a report whose `tool_calls` is *empty* has no
assembled string to audit. For that shape run
`uv run --no-project python <skill_root>/scripts/toolcall-raw-output-audit.py
<attachments dir>` on the request / rendered-prompt / raw-completion dumps: no
marker in the model's own output ends the investigation in the model-choice
lane (`no-tool-call-in-raw-output.md`), and only a marker that survives to the
parser puts you back on rule 1.

1. **The message the reporter pastes is the client's, so the first artefact is
   the assembled string, not the model.** Concatenate
   `choices[i].delta.tool_calls[j].function.arguments` in arrival order and ask
   what the concatenation is (`bash <skill_root>/scripts/sse-toolcall-audit.sh
   <raw-sse-or-paste>`; it labels the outcome itself — `balanced`, `head-lost`,
   `quote-gap` = balanced braces with an odd quote count, `tail-lost`,
   `malformed` — so its verdict no longer has to be overridden by hand). Its
   outcomes have different owners:
   parses as JSON → *not* a closed case: it clears the string lanes, and then a
   client that validates types/keys is still on the table
   (`client-schema-invalid-arguments.md`), otherwise the client's
   accumulation/framing; missing tail
   (unbalanced `{`/`"` at the end, `finish_reason` still `tool_calls`) → the
   stream stopped short; **`head-lost`: more closing than opening braces, so the
   assembly starts inside the object** (`"}` for a complete
   `{"todos": "…"}`) → only a suffix of the arguments was ever emitted, so the
   delta bookkeeping (`_compute_args_diff` / `streamed_args_for_tool`, or the
   frontend's `remaining_call`) was ahead of the bytes; one arithmetic decides it
   — `assembled == non_stream_args[-len(assembled):]` — and this outcome must not
   be sent to the schema-type gate (`suffix-only-arguments.md`); **braces balanced
   but quotes odd, with the gap inside a value** → the parser's incremental delta,
   decided by the argument's declared JSON-Schema type
   (`arg-type-gate-unquoted-value.md`; run
   `scripts/tool-schema-type-audit.py` on the request body); **characters missing
   at the head of a value plus a fragment repeated a few characters later** → the
   streamed *text* was already damaged, and that has two owners: the incremental
   detokenizer when `content` shows the same damage at the same place
   (`detokenizer-text-delta.md`; one log grep for `resetting decode stream`
   decides it), the decoding/spec-decode lane when `content` is clean; a brace
   pair inside a string or reordered fragments → the text itself is wrong. The
   `quote-gap` outcome looks like the others until you check, so check it: a parser that
   emits valid *deltas* never leaves a quote out, while a parser whose final
   `args_so_far` is valid but whose deltas are a length-prefix of it reproduces
   exactly this string (the recorded #12261 run recovered it from the two
   functions alone).
   Do this before reading any parser, sampler or kernel. Two recorded
   calibrations: #7580's paste reassembles to `{"filePath":"/…/K8sYamlAgent.java`
   and `json.loads` reproduces the reporter's own `Unterminated string starting
   at: line 1 column 13 (char 12)`; #8122's pasted 97-chunk stream reassembles to
   fully valid JSON, so the sample in the issue was *not* one of the failing
   requests — say so instead of inventing a mechanism for a healthy stream.
   **Both mid-string owners are version-specific, so read the branch at the
   reporter's ref before naming it**: the schema-type gate documented for vllm
   v0.20.2 is fixed at v0.23.0 (`_is_string_type` now uses
   `extract_types_from_schema`, so a nullable `anyOf` string takes the quoted
   branch), and at v0.23.0 the incremental detokenizer has a fast
   `DecodeStream` path whose reset warning is the fingerprint. A report at that ref
   showing a lost *leading* character therefore points at the text delta, not at the
   gate — `arg-type-gate-unquoted-value.md` §Version correspondence.
2. **A numeric `stop_reason` in a chunk is upstream vLLM's own field, not
   corruption.** `ChatCompletionResponseStreamChoice.stop_reason: int | str |
   None` is vLLM-specific (`vllm/entrypoints/openai/chat_completion/protocol.py:120`
   at vllm v0.17.0) and carries the stop *token id*; it is filled from
   `output.stop_reason` where the choice is built
   (`chat_completion/serving.py:1303`, also `:1492`, `:1701`). A pasted
   `"stop_reason":154829` beside an empty `arguments` fragment says "a stop token
   ended the request here" and nothing about the JSON error.
3. **`finish_reason: "tool_calls"` does not prove the arguments were complete —
   and it does not prove a call exists.** It is set from `auto_tools_called` when
   the parser streamed any tool call (`chat_completion/serving.py:1290-1297`), so a
   call whose tail never arrived still ends with that finish reason; and the
   non-streaming path copies the parser's `tools_called` through unchanged, so a
   parser that matched only the opening marker reports no calls and no content
   *under that same finish reason* (`deepseek-parser-silent-drop.md`). Read it as
   "a call was started, or the parser thought one was".
4. **One request decides engine-vs-assembly: add `return_token_ids: true` to the
   same body.** The stream choice then carries vLLM's per-chunk `token_ids`
   (`protocol.py:98`; filled at `serving.py:1307-1310` only under
   `request.return_token_ids`). If the token ids also stop before the closing
   tokens, the engine emitted a short sequence; if they are complete, the delta
   assembly lost them. This is cheaper and more decisive than any source walk and
   it is what the two recorded runs lacked.
5. **A constant in the code that equals the number in the report is the branch to
   name, and it is named in the first answer, not after its kernel is read.**
   `num_speculative_tokens: 3` fails and `1` works: `vllm_ascend/sample/
   rejection_sampler.py:114` is `using_block_verify = max_spec_len >= 3` with the
   comment "When num_speculative_tokens>=3, using block verify" — a threshold on
   the reported value. **Read the line at the reporter's ref first**: from
   v0.23.0rc1 it also ANDs `rejection_sampler_config.enable_block_verify`, which
   defaults `False` (`ascend_config.py:766` at v0.23.0rc1, `:797` at v0.23.0 —
   grep `enable_block_verify`; the line moves), so the threshold alone may select
   nothing there. Where it does hold, state it as the branch, label it
   source-verified, give the n=2 A/B that confirms it, and put the mechanism
   (`mtp-block-verify-acceptance.md`) behind that A/B. The recorded #7580 run
   reached that branch on its tenth turn and then spent its remaining nine turns
   on the acceptance kernel, the pytorch twin and the test files, and ended with
   the answer cut inside its evidence table and no action shipped — the branch
   was already the answer when it appeared.
6. **Split the three ownership layers before proposing a fix**: upstream vLLM
   OpenAI frontend (delta assembly, parser delta state, `stop_reason`/
   `token_ids` fields), vllm-ascend sampler (`sample/rejection_sampler.py`, the
   ops it patches), and the model/tokenizer (did the model emit a stop token
   before closing the JSON). The answer names which layer the evidence puts it in
   and which one instrument decides the next step. **Downstream does patch this
   path, so establish that it does not own the symptom before saying "upstream"**:
   grep `vllm_ascend/patch/` by *model family* (`glm`, `deepseek`, `minimax`), not
   by the symptom's word, and by the spelling **that release** uses: at v0.23.0
   the family is two files — `patch_glm_tool_call_streaming.py` wraps
   `OpenAIServingChat.chat_completion_stream_generator` and replaces
   `_create_remaining_args_delta`, `patch_glm47_tool_call_parser.py` rewrites the
   GLM47 region extractor — while at **v0.18.0 both jobs sit in one file,
   `platform/patch_glm_tool_call_parser.py`** (header: "backport GLM tool-call
   parser and finish-suffix fixes"; `:1083` assigns the patched generator and
   `:1084` the patched `Glm4MoeModelToolParser.extract_tool_calls_streaming`, both
   re-stamped as upstream names at `:1079-1082`, plus four `_*` helpers at
   `:1073-1076` and an import-time `RuntimeError` unless the MiniMax
   usage-accounting patch ran first; imported unconditionally at
   `platform/__init__.py:32`, and `glm47` subclasses `glm4_moe` so both
   `--tool-call-parser` names are covered). A grep that finds no match for one
   spelling at an older pin is **name drift**, not "downstream does not patch this
   path" — `ls vllm_ascend/patch/platform/` before blaming upstream. Recorded
   #12244 (v0.18.0, GLM-5.1 1P1D, 11-turn tool-call body): `content` deltas
   repeated prefixes and carried `</arg_value>`/`<arg_key>` fragments, and the
   reporter's own follow-up made **MTP on the prefill node**
   (`num_speculative_tokens: 3`) the lever — MTP off on P stopped the garbling —
   so this patch is the assembler, not the cause. The residue it then showed
   (`think` text inside `content`, `reasoning_content` empty) is the pre-filled
   `</think>` path: `is_reasoning_end` scans backwards and returns `True` when the
   last marker is `end` (`vllm/reasoning/basic_parsers.py:64-73` at vllm v0.18.0),
   which the patched generator consumes as `prompt_is_reasoning_end_arr`
   (`:354-355`, `:419`, `:484`, `:525`) to route every later token to `content`.
   `patch/__init__.py` carries one numbered comment block per patch with its
   `Why`/`How`/`Future Plan` and the upstream PR it backports; quote the branch
   condition that excludes it (see `arg-type-gate-unquoted-value.md`).
   **Reconcile the parser name with the pinned tree before reading any parser:**
   `--tool-call-parser <name>` is validated against that release's registry
   (`rg -n '"<name>"' <vllm tree>/vllm/tool_parsers/__init__.py`), so a name the
   pinned tree does not register means the export **predates the runtime** —
   vllm v0.13.0 registers `glm45` only, and a serve line carrying `glm47` cannot
   be that tree. Say that as the coverage gap in one line, argue from the nearest
   documented ref, and do not conclude "unreachable here, so not this layer"
   (`suffix-only-arguments.md`, §Before arguing the mechanism).

## Pages

- `empty-arguments-object.md` — the `{}` shape: where
  `glm4_moe_tool_parser.py` prints it (streaming `:401-402`, non-streaming
  `:214` at vllm v0.23.0), why a JSON-bodied region yields *no call* in
  non-streaming instead, the four-row `stream: false` discriminator, the
  downstream patch 7b that makes zero-argument calls surface as `{}`, and the
  #8130 / #8478 precedents, with #7682 as the neighbouring *suffix-only* row.
- `streamed-arguments-audit.md` — the audit procedure: how to obtain the raw SSE,
  how to read the script's output and the outcomes (A, B, C, D1, D2), the ordered
  ablations
  (`return_token_ids` → `stream: false` → `num_speculative_tokens: 2` →
  `temperature: 0` → `--async-scheduling` off), what to send back, and the worked
  #7580 / #8122 rows.
- `no-tool-call-in-raw-output.md` — the *missing* call: which of the four
  dumps (request body, rendered prompt, tokenized request, raw completion)
  exonerates which layer, how to join them (prompt-token count, not the user
  message), the four-row rate protocol (N repeats → thinking on → vendor API →
  `temperature: 0`), the #12658 facts read from its own uploads, and the owners
  the dumps exclude (`patch_deepseek_v4_tool_call_parser.py`,
  `patch_deepseek_v4_thinking.py`, `patch_tool_choice_none_content.py`'s
  dropped `tool_calls: []`, `finish_reason` set only when a call was seen).
- `deepseek-parser-silent-drop.md` — `content` null **and** `tool_calls: []` on a
  `--tool-call-parser deepseek_v3` serve: the one branch that produces it
  (`deepseekv3_tool_parser.py:84-107` at vllm v0.9.2 — start marker seen, strict
  fenced-call regex unmatched, `tools_called=True`), the regex's three required
  pieces as the comparison that names the missing element, the `stream: true`
  witness, and the reporter's parser-free system-prompt workaround with its scope.
- `detokenizer-text-delta.md` — the third owner, for damage **inside the text**:
  missing head plus a repeated fragment at a script boundary, `stream: false`
  clean. The two incremental-detokenizer paths at v0.23.0
  (`vllm/v1/engine/detokenizer.py:49-66`), the fast path's
  `resetting decode stream` warning as the one decisive log line, the slow path's
  `new_text[len(prefix_text):]` slice, the `content`-vs-`arguments` comparison, the
  `tokenizers >= 0.22.0` gate, and the client-side validate-and-repair workaround
  (#14656, GLM-5.2 PD).
- `arg-type-gate-unquoted-value.md` — the other, more common shape: a *string*
  argument streamed with no opening quote while the closing quote and `}` arrive.
  The `_is_string_type` gate (`glm4_moe_tool_parser.py:127-148`, only a top-level
  `type: "string"` counts, so a nullable `anyOf` argument loses), the raw branch
  at `:408`, the length-prefix delta at `:424`, the arithmetic that reproduces the
  paste, why `stream: false` never reproduces, the two Ascend GLM patches and the
  condition that excludes each, the reporter's schema workaround.
- `suffix-only-arguments.md` — the assembled string is *only* the object's
  closing punctuation: the suffix arithmetic against `stream: false`, the two
  bookkeeping sites (parser length diff, frontend finish-step `remaining_call`),
  why the silent `delta: {}` chunks are a buffering parser behaving normally, the
  `--tool-call-parser` registration witness that dates the export against the
  runtime, the four reporter-side checks, and the #7682 case (reporter-confirmed
  A/B; the recorded run shipped no answer).
- `client-schema-invalid-arguments.md` — the *valid but schema-invalid* shape
  (`invalid tool parameters`): why the audit's `balanced` does not close it, the
  three artefacts and the six-row field comparison table, the four
  `minimax_m2_tool_parser.py` conversion sites at vllm v0.18.0 (`:163` null
  coercion before the type check, `:170-183` type priority, `:190` string
  passthrough, `:209-219` `json.loads`), the untested multi-block streaming reset
  (`:414-421`), the two registered parser names and why a name mismatch is a
  *different* shape, and the one sentence that retires the deployment lane.
- `delta-message-none-field.md` — the server-side abort: a `patch/**` parser passing
  `tool_calls=None` into the pinned non-Optional field (`protocol.py:262` at vllm
  v0.20.2), why the upstream sibling passes a list (`minimax_m2_tool_parser.py:434-437`),
  the reachable branch (`if delta_tool_call or content_before:` with
  `_extract_delta_tool_call` returning `None`), the `[]` fix and why the frontend's
  truthiness tests make it safe, the existing UT's blind spot (prose in a separate
  chunk) and the one-`_feed` regression test. Worked for #10376 at v0.20.2rc1.
- `mtp-block-verify-acceptance.md` — what the `max_spec_len >= 3` branch does
  (cumulative-product acceptance over the whole draft block, recovered token,
  bonus token), which requests it cannot touch (greedy rows), why a wrongly
  accepted draft token can end a tool call early, the `parse_output` placeholder
  filter that silently removes slots, the missing equivalence test, and the
  confidence labels.
