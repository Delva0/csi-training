# A `patch/**` parser writes `None` into a non-Optional `DeltaMessage` field

Scope: `stream: true` request dies **server-side mid-stream** — `ERROR … Error in
chat completion stream generator`, then a traceback whose deepest `vllm_ascend` frame
is inside `patch/**` returning `DeltaMessage(...)`, ending in
`pydantic_core._pydantic_core.ValidationError: 1 validation error for DeltaMessage` /
`tool_calls` / `Input should be a valid list [type=list_type, input_value=None,
input_type=NoneType]`. The client sees a truncated stream (the SSE generator died), not
a 400 and not a JSON-parse error, so this is not the client-assembly lane. Inherit
`index.md`; recorded #10376 (v0.20.2rc1 + 910B, `minimax-m2.7`).

## Mechanism (verified at vllm-ascend v0.20.2rc1 `367b8e62da79`, vllm v0.20.2 `bc150f502991`)

- the pinned protocol model declares `tool_calls` **non-Optional**:
  `DeltaMessage.tool_calls: list[DeltaToolCall] = Field(default_factory=list)`
  (`vllm/entrypoints/openai/engine/protocol.py:262`), against `str | None` for
  `role`/`content`/`reasoning` (`:259-261`). Omitting the field gives `[]`; writing it
  explicitly as `None` is a validation error.
- the downstream copy admits a `None` and then writes it:
  `if delta_tool_call or content_before:` (`vllm_ascend/patch/platform/patch_minimax_m2_tool_call_parser.py:493`)
  → `tool_calls=[delta_tool_call] if delta_tool_call else None` (`:496`). The guard's
  `or content_before` arm is reachable with `delta_tool_call is None`, and
  `_extract_delta_tool_call` returns `None` whenever no complete `<invoke>` is in
  `current_text` (`:414-415`) or there is nothing new to send (`:457`).
- this is the downstream delta, not an upstream guard that went missing: the upstream
  sibling at the same pin keeps the value a **list**
  (`tool_calls=delta_tool_calls`, `vllm/tool_parsers/minimax_m2_tool_parser.py:434-437`,
  built as `delta_tool_calls: list[DeltaToolCall] = []` at `:305`).
- trigger: one decoded delta that carries lead-in prose **and** the tool-call start
  marker while no complete `<invoke>` has arrived (a chunk boundary decides it, so the
  symptom looks intermittent — some streaming calls fine, the ones with preamble text
  abort).

## Fix and acceptance

`tool_calls=[delta_tool_call] if delta_tool_call else []`, or split the two cases
(`if delta_tool_call: DeltaMessage(tool_calls=[delta_tool_call])` / `if content_before:
DeltaMessage(content=content_before)`). Keep `or content_before` in the guard: dropping
it discards the preamble text the model emitted. `[]` is safe because the frontend only
tests truthiness (`vllm/entrypoints/openai/chat_completion/serving.py:1011,1059,1808`),
never `is not None`. Do not "fix" it with `try/except` or by relaxing the protocol field
— that un-validates every parser on the path.

Check the siblings before generalising: at this pin `:496` is the only `None` handed to
`tool_calls` under `vllm_ascend/patch/` (`rg -n 'tool_calls=\[' vllm_ascend/patch/`;
`patch_deepseek_v4_tool_call_parser.py:351,458,474` build lists,
`patch_glm_tool_call_streaming.py:55` omits the field).

**Coverage is a UT gap, not an absent test file.** The tree carries
`tests/ut/patch/platform/test_patch_minimax_m2_tool_call_parser.py` with
`test_plain_content_before_tool_call_is_preserved` (`:86`), but it feeds the prose in a
*separate* chunk, so `content_before` truthy with `delta_tool_call is None` is never
exercised — that is why CI is green. The regression test is one `_feed(parser, ["Let me
check. <minimax:tool_call>"])` call via the file's own helper; quote the gap, not "no
test".

## Confirmation the reporter can run

- the same request body with `"stream": false` — this function is not called on the
  non-streaming path, so success confirms the streaming delta path (no restart needed).
- the paste's own line number dates the image: `:494` is `return DeltaMessage(` at this
  pin, `:496` the offending expression.
- `return_token_ids: true` on the failing request shows the chunk carrying prose plus
  the marker, which is also what the reporter should send back.

## Scope

Every frame is in the request-parsing path, so the NPU model, quantization and card are
irrelevant; say that in one line. The snapshot holds a single vllm-ascend ref, so "is it
fixed in a later rc" cannot be answered — recheck with
`rg -n "tool_calls=\[delta_tool_call\]" vllm_ascend/patch/platform/patch_minimax_m2_tool_call_parser.py`
(`[]` or no `None` arm means the image predates the fix). Same shape on another parser:
read that patch's construction sites and the pinned field declaration.
