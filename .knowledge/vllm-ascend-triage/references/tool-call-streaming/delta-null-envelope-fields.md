# `null` in the delta envelope: `content: null` beside `tool_calls`, `id`/`type`/`name`: `null`

## When this page applies

`stream: true` **works** — the stream reaches `finish_reason` and `[DONE]` — but every
tool-call chunk's `delta` carries `"content": null`, and the continuation chunks
carry `"id": null, "type": null, "function": {"name": null, "arguments": "…"}`:

```
{"delta":{"content":null,"tool_calls":[{"id":"call_…","type":"function","index":0,
 "function":{"name":"read_file","arguments":""}}]},"finish_reason":null}
{"delta":{"content":null,"tool_calls":[{"id":null,"type":null,"index":0,
 "function":{"name":null,"arguments":"{\"path\":\""}}]},"finish_reason":null}
```

The reporter's question is almost never "what owns this" but the triad: is there a
config/env switch, does an older tag or another image do it correctly, and is the
model's own XML tool syntax (`<｜DSML｜>…`) being mis-converted. Inherit `index.md`.

Not this page's shape, and the split is one line of the paste: `stream: true` **dies
server-side** with `ValidationError … DeltaMessage` / `tool_calls` / `Input should be a
valid list` and the client sees a truncated stream → `delta-message-none-field.md`
(a *non-Optional* field written as `None`). Here nothing validates wrongly: all four
fields are nullable **by declaration**, so the paste is a serialization question.

## Mechanism: the SSE writer drops only fields that were never passed

Verified at vllm `v0.22.1` (`0decac0`) + vllm-ascend `v0.22.1rc1` (`5f6faa0`):

| Fact | Location |
|---|---|
| every envelope field is nullable: `DeltaFunctionCall.name/arguments: str \| None`, `DeltaToolCall.id: str \| None`, `type: Literal["function"] \| None`, `DeltaMessage.content: str \| None` | `vllm/entrypoints/openai/engine/protocol.py:242-252,267-271` |
| the chunk is dumped with **`exclude_unset=True`**, which drops fields that were never *passed* and keeps ones passed as `None` | `vllm/entrypoints/openai/chat_completion/serving.py:553,589,940` (only the trailing usage chunk adds `exclude_none=True`, `:966-967`) |
| the first chunk of every stream is `DeltaMessage(role=role, content="")` — a **passed `""`**, which is why the paste's opening chunk shows `""` and not `null` | `chat_completion/serving.py:520-524` |
| upstream's finish-step top-up passes `id`/`type`/`name` explicitly (`id=original_tc.id if original_tc else None`) | `chat_completion/serving.py:1573-1600`, `:1589-1594` |
| upstream's own DeepSeek-V3.2 streaming delta passes `content` explicitly, so it is `None` on every pure tool-call chunk; `DeepSeekV4ToolParser` is a 31-line subclass (start/end token + structural tag) and inherits it | `vllm/tool_parsers/deepseekv32_tool_parser.py:300`; `vllm/tool_parsers/deepseekv4_tool_parser.py:14-31` |

So **`null` vs `""` vs absent is decided by the construction site, not by the model,
the card or a flag.** Read the parser's builders and say which one produced the paste:

- downstream `patch/platform/patch_deepseek_v4_tool_call_parser.py` owns the
  mid-stream continuation nulls: `_pop_pending_delta_message` computes
  `content = "".join(content_parts) or None` (`:447`) and returns
  `DeltaMessage(content=content, tool_calls=…)` (`:448`), so every drained tool-call
  chunk carries an explicit `content=None`; for a *new* `index` it builds
  `DeltaToolCall(index=…, id=tool_call.id, type=tool_call.type,
  function=DeltaFunctionCall(name=…), …)` (`:422-430`), and both source builders leave
  those fields unset (`_emit_tool_name_delta` `:456-467` omits `content`,
  `_emit_tool_args_delta` `:469-478` sets only `arguments`) — the merge re-adds the
  `None`s.
- the paste's null *pattern* therefore fingerprints the builder: real `id`/`type`/`name`
  plus `content: null` is the name delta; `id`/`type`/`name` all `null` plus an
  `arguments` fragment is the merged args delta; `role` plus `content: ""` is the
  frontend's first chunk. That is also the one-line proof that the reporter serves with
  `--tool-call-parser deepseek_v4` (the patch binds `DeepSeekV4ToolParser` only).

## No switch exists, and no tag rollback removes the upstream half

- The patch is imported unconditionally at `vllm_ascend/patch/platform/__init__.py:37`
  and the file has no `getenv`/`environ` guard (`rg -n 'getenv|environ' …` → empty at
  v0.22.1rc1). There is no `--additional-config` key, env var or
  `chat_template_kwargs` value that turns the nulls off; say that in the first line.
- "Does an older tag handle it correctly?" cannot be answered from a one-pin snapshot,
  and for the `content: null` half a rollback cannot help: the upstream sibling does
  the identical thing at the pin (`deepseekv32_tool_parser.py:300`). Only the
  continuation `id/type/name: null`s are downstream, and removing that patch trades
  them for **no incremental argument streaming**, which the patch's own header explains
  it was written to fix (`patch/__init__.py:332-346` block 12).
- Q3 ("is the `<｜DSML｜>` conversion at fault?"): the conversion is the parser, the
  chunk shape is that parser's merge builder, and nothing is parsed wrongly — the
  assembled `arguments` are valid JSON. Keep the answer in the serialization lane.

## What the answer owes

1. Verdict + rung 1 in the first paragraph: **not a bug and not a model fault**;
   tolerate absent-or-null on `delta.content` and `tool_calls[].{id,type,function.name}`
   in the client (all four are Optional in the engine's own schema), or send
   `stream: false` for tool-call turns (the non-streaming path emits no deltas).
2. The two owners, one line each: `content: null` is upstream behaviour at the pin;
   the mid-stream `id/type/name: null`s are the downstream builder. Do **not** call the
   whole paste a downstream defect, and do not offer a rollback as the fix.
3. If the client genuinely cannot change, name the minimal server-side change as a
   *patch proposal*: pass `content`/`id`/`type`/`name` only when non-`None` in the two
   patch builders. Note it is not a complete cure — the frontend's own
   `_create_remaining_args_delta` (`:1589-1594`) still passes explicit `None`s on the
   finish step. A `try/except` around the dump, or relaxing the protocol model, is the
   wrong repair (it un-validates every parser on the path).
4. What to send back: the failing **raw SSE** for one request, the client library and
   its version plus the validator/traceback text (the "parsing failure" is theirs), and
   the serve line. Without the SSE the null pattern cannot be fingerprinted.

## Version correspondence

Paths move between refs, so grep the **class**, never assume the module:

| ref | serving module | `DeltaMessage` lives in |
|---|---|---|
| vllm v0.13.0 | `vllm/entrypoints/openai/serving_chat.py` | `chat_completion/protocol.py` |
| vllm v0.17.0–v0.18.0 | `vllm/entrypoints/openai/chat_completion/serving.py` | `chat_completion/protocol.py` |
| vllm v0.20.2–v0.22.1 | `vllm/entrypoints/openai/chat_completion/serving.py` | `engine/protocol.py` |

`rg -ln 'class DeltaMessage' vllm/` in one call beats two failed path guesses (recorded
#14501 lost three calls to `chat_completion/protocol.py`, absent at v0.22.1).

## Recorded case

**#14501** (2026-08, image `quay.io/ascend/vllm-ascend:v0.22.1rc1`, DeepSeek-V4-Flash,
TP8+EP, `stream: true` + `tools` + `tool_choice: "auto"`, `enable_thinking: false`):
pasted SSE with the two chunk shapes above and three questions (workaround / older tag /
DSML conversion). The recorded run read the protocol fields, the `exclude_unset` dump
site, the downstream builder and the upstream sibling, and answered "not a model fault,
no switch, client-side tolerance, the upstream half survives any rollback" — with the
continuation-null ownership split and the one open item (which client library raises)
in the send-back list. It never had to open the model tree or a device path, and neither
should the next reader.
