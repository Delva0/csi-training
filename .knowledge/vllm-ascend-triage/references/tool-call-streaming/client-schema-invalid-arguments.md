# The arguments parse as JSON and the client still rejects them

## Scope

The client refuses a call it has already assembled: Claude Code prints
`invalid tool parameters`, a plugin raises `AI_TypeValidation_Error`, a
function-calling SDK rejects the object. There is **no** `JSON Parse error`, no
`Unterminated string`, and nothing in the paste looks truncated. Every sibling
page owns a *damaged string* (`arg-type-gate-unquoted-value.md`,
`suffix-only-arguments.md`, `streamed-arguments-audit.md`); this page owns a
well-formed string that disagrees with the client's own tool schema.

The turn-1 trap: `scripts/sse-toolcall-audit.sh` returns `balanced`, and a
balanced assembly reads as "the string is fine, so the client's framing is at
fault". That inference covers JSON *validity* only. A well-formed object still
lands here when the client validates types, key names or required fields **before**
it runs the tool — so run the audit (it costs one call and rules out every
string-damage lane), and when it answers `balanced`, compare fields instead of
closing the case.

## 1. Three artefacts, one field-by-field comparison

1. The client's own words — the screenshot or the text. It usually names the
   tool and the offending field; a bare `invalid tool parameters` does not, so
   ask for the tool name.
2. The tool's JSON schema **as that client declares** it. Illustrative shape
   only — `Read{file_path: string, offset?: number, limit?: number}` — a
   *recalled* client built-in, not verified here: take the reporter's own
   definitions from their client config.
3. The raw SSE for the failing request, plus the same body with `"stream":
   false`.

Then compare, in this order, and stop at the first mismatch:

| Field-level defect | What it looks like in `arguments` | Which owner produces it |
|---|---|---|
| type flipped | `{"limit": "200"}` where the schema says number, or `{"timeout": 30}` where it says string | the parser's conversion, or the model writing a bare token |
| value forced to `null` | `{"command": null}` although the schema says string and the model wrote a word such as `none`/`null`/`nil` | the parser's null coercion (§2) |
| structure laundered | a JSON body arrives as a *string*, or a string arrives already `json.loads`-ed into an object/array | the parser's object/array and fallback branches (§2) |
| quoting/quotes kept | `{"file_path": "\"/tmp/a.py\""}` — the value carries the model's own quotes | string branch returns the raw text (§2) |
| key missing | required key absent, `arguments` short, or a key renamed by the model | the parser's `<parameter name=…>` regex only matches one spelling (§2), or the model |
| duplicated arguments | the same key twice, or `{…}{…}` under one `index` | the streaming state reset (§3) |

`stream: false` is the split that decides the last row and most of the others: if
the non-stream object matches the schema and the streamed one does not, no
model-side explanation is needed.

## 2. The conversion sites, worked on `minimax_m2`

`--tool-call-parser minimax_m2` is upstream's `MinimaxM2ToolParser`
(`vllm/tool_parsers/minimax_m2_tool_parser.py`, 452 lines at vllm **v0.18.0**
`bcf2be9`, the ref recorded #9076 was filed against). It buffers each
`<invoke name=…>` and re-emits the whole `arguments` JSON per call, so the client
never sees a partial object — which is why this shape produces *invalid* rather
than *truncated* parameters. Four sites can make a schema-invalid object:

- **The null coercion runs before the type check** (`:163-164`,
  `if value.lower() in ("null", "none", "nil"): return None`). Any parameter whose
  literal text is one of those three words becomes JSON `null` even when the
  schema says `string` — a path, a branch name or a command written as `none`.
- **The type is taken from the schema and preferred in a fixed order**
  (`:170-183` `integer > int > number > float > boolean > object > array >
  string`). Where the schema admits several types (`anyOf`, `oneOf`, `enum`, or
  `type: [...]`, gathered by `_extract_types_from_schema`) the most specific one
  wins, so a schema of `["string","number"]` turns the id `007` into `7` and a
  `["string","boolean"]` schema turns `1` into `true`.
- **The string branch returns the text untouched** (`:189-190` `return value`), so
  the model's own quoting, leading/trailing spaces and escapes reach the client
  verbatim.
- **The object/array branch and the fallback `json.loads`** (`:209-212`,
  `:215-219`) parse anything that looks like JSON, so a value the schema declares
  `string` arrives as a list/dict whenever the model wrote `[...]`/`{...}`.

**Coverage, and why it is not yet a cause**: none of this is exercised against a
real client schema — `tests/tool_parsers/test_minimax_m2_tool_parser.py` asserts
`json.loads(arguments)` round-trips strings and never checks declared types. Say
that as coverage, not as the defect: with no pasted `arguments`, which of the four
sites produced *this* rejection is open, and the answer names the one-line check
for each (`stream: false` first).

## 3. The multi-block streaming reset (source-visible, reachability open)

`extract_tool_calls_streaming` resets its cursors whenever the delta contains the
start token (`:414-421`: `if not previous_text or tool_call_starting:
self.current_tool_index = 0; self.prev_tool_call_arr.clear();
self.streamed_args_for_tool.clear()`), while `_extract_delta_tool_calls` reads
`invoke_complete_regex.findall(current_text)` over the **whole accumulated text**
and emits the first `current_tool_index` invokes. So a reply carrying **two**
`<minimax:tool_call>` blocks re-emits block 1's `<invoke>`s from `index 0`; a
client that accumulates by `index` concatenates the same `arguments` twice and
the JSON becomes invalid or duplicated.

Verified at the same ref: every fed stream in the parser test file carries at
most one `<minimax:tool_call>` (`grep -c` per test; the two-invoke tests at
`:203-238` put two `<invoke>`s inside **one** block), so the reset is untested —
citable as coverage. Whether the model emits two blocks per turn on the
reporter's prompt is what decides it, and the paste decides that: count the
`<minimax:tool_call>` occurrences in `return_token_ids`-decorated output or in
the raw `stream: false` completion.

## 4. Two registered parser names, and the mismatch is not this shape

`vllm/tool_parsers/__init__.py:97` registers `minimax_m2` (the M2 generation,
`<minimax:tool_call>` / `<invoke name=…>` / `<parameter name=…>`, markers at
`minimax_m2_tool_parser.py:38-39`) and `:101` registers `minimax` (the older
family, `<tool_calls>` JSON, `minimax_tool_parser.py:44-45`). Serving an M2
checkpoint with `minimax` — or the reverse — produces **no call at all** and a
raw-markers-in-`content` symptom (`no-tool-call-in-raw-output.md`), never
`invalid tool parameters`. So a client-side parameter complaint already proves
the marker set matched; do not spend a turn on the name. What is worth one grep
is the pair that *does* gate this shape: `reasoning/__init__.py:59,63` register
`minimax_m2` and `minimax_m2_append_think`, and the tutorial notes
`append_think` leaves `<think>…</think>` inside `content`
(`docs/source/tutorials/models/MiniMax-M2.5.md`), which pollutes the client's
context on later turns but cannot make a parameter invalid.

## 5. What retires the deployment lane

**A report that says the same deployment serves benchmark / ordinary chat
correctly has already excluded the config lane for this symptom.** The failure is
then in call assembly, and the recipe deviations (context parallel, FLASHCOMM2, a
TP8+PCP serve where the shipped MiniMax-M2.5 recipe is TP16 with neither) are not
ranking candidates — a corrupt serving path would corrupt prose and benchmark
answers too. Say that in one line rather than ranking them. Recorded #9076
ranked "deployment deviates from the recipe" fourth, and the reporter's next
message was "we can test benchmark in this deployment" plus a screenshot of the
client error: one line in the first answer ("if plain benchmark answers are
correct, the deployment is not the suspect") would have removed that row and the
turn it cost.

Before blaming the model, prove the downstream does not own the tool path:
`vllm_ascend/patch/__init__.py` carries one numbered block per patch, and the
three MiniMax entries — `:111-123` (`patch_minimax_m2_config.py`, quantization
verification), `:194-210` (`patch_minimax_usage_accounting.py`, reasoning token
counts), `:477-515` (`patch_minimax_m2.py`, MoE forward, `k_norm` sharding, fp8
dequant) — touch no parser, no delta assembly and no frontend. Grep by family
(`rg -l 'minimax' vllm_ascend/patch/`), as `index.md` rule 6 says; a
`patch_*_tool_call*.py` for that family would reopen it.

## 6. Delivery for this shape

The answer is short: the layer that owns it (upstream parser vs client vs
deployment, as of §5), the ranked field-level candidates with the one `stream:
false` A/B that separates them, and the ask — the client's error text with the
tool name, the tool's schema, and the raw SSE. Do not walk the parser's 452 lines
before the three artefacts exist; two `sed` windows (`:155-220`, `:405-430`) hold
everything this page cites.
