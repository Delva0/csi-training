# The call arrives as *prose*: client shows `<parameter …>` / `<| DSML | invoke>`, `tool_calls` is empty, the agent retries and looks hung

## Signature

The serve answers prose fine, but a turn that should carry a tool call comes back with an empty
`tool_calls` and the call's markup **inside the text**: the agent UI prints
`<parameter name="command">…</parameter>` with a stray `</invoke>` (Anthropic's spelling), or
`<| DSML | invoke>` / `</| DSML | invoke>` with **ASCII** pipes where DeepSeek's token is
`<｜DSML｜…>` (U+FF5C), sometimes a bare `<parameter name="description" string="true">` line inside the
assistant message. Codex-class clients then re-issue the same turn until their retry budget is spent
(reported as 13 attempts); Claude-Code-class clients just wait, so the report reads 卡死 / hang /
"模型返回的 tools 经常是随机的". Recorded #10856 (`v0.20.2rc1-a3`, DeepSeek-V4-Pro, PD 3P1D on 64 A3
cards, codex behind a `/v1/responses`→`/v1/chat/completions` converter plus Claude Code against the
same serve).

Not this page: `tool_calls: []` with plain prose and no marker anywhere (`no-tool-call-in-raw-output.md`,
model-choice lane); `content` and `tool_calls` both empty on a DeepSeek-V3 serve
(`deepseek-parser-silent-drop.md`); every delta field `null` but the call assembled
(`delta-null-envelope-fields.md`).

## First call: byte-compare the marker in the raw body

One failing request, `curl` **straight at** `/v1/chat/completions` (not through the converter) with
`stream: false`, then the same body with `stream: true`. Compare the marker character by character with
the parser's own token at the pin — DeepSeek-V4's are `<｜DSML｜tool_calls>`, `</｜DSML｜tool_calls>`,
`<｜DSML｜invoke name="`, `</｜DSML｜invoke>`, `<｜DSML｜parameter name="…" string="true|false">`,
`</｜DSML｜parameter>` (constants at `tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py:13-19`
at v0.20.2rc1). Four outcomes, four owners:

| raw text | owner |
|---|---|
| the exact token, `tool_calls` still empty | parser / registration, §A |
| Anthropic `<parameter …>` + `</invoke>`, or the client's own XML | prompt ↔ converter dialect, §B |
| a *damaged* token (an ASCII `\|`, spaces, `string="…"` missing) | tokenizer / upload contents, §C |
| no marker at all | model-choice lane → `no-tool-call-in-raw-output.md` |

**A screenshot is not the engine's text.** Both pastes in #10856 are client-rendered, and a client may
re-wrap or normalise the block it prints, so neither spelling can be attributed to the engine from the
issue alone; ask for the raw body (or `return_token_ids: true` plus the token counts) first.

## §A The downstream V4 parser turns an unmatched marker into prose, silently

Source-verified at vllm-ascend `v0.20.2rc1` (`367b8e6`),
`vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py`:

- `:350-351` — `self.tool_call_start_token not in model_output` ⇒ `tools_called=False, content=model_output`
- `:368-370` — start token present but **zero** matched `<｜DSML｜invoke …>` ⇒ the same verbatim return
- `:379-380` — bare `except Exception` ⇒ the same verbatim return, **with no log line at any level**
- `:633-643` — streaming: no start token in the buffer ⇒ the whole buffer is emitted as
  `DeltaMessage(content=…)`, holding back only a partial-tag overlap (`_partial_tag_overlap`, `:60-66`)

So "the server sent the markup back" is no evidence that the parser ran, and a one-character difference
is indistinguishable from prose server-side. `patch/platform/__init__.py:36` imports this patch
unconditionally, i.e. it is on whenever upstream `vllm/tool_parsers/deepseekv4_tool_parser.py` imports —
so the **vllm** version decides whether `--tool-call-parser deepseek_v4` is even a legal value:
#8875 (v0.13.0rc3 rejected both `--tool-call-parser deepseek-v4` and `--reasoning-parser deepseek-v4`),
#8670 (that release's own DeepSeek-V4 sample command carries no parser flag), #9732 (the patch's own gap
list: registration through the ascend patch, `string="false"` streaming-vs-non-streaming divergence,
`reasoning_effort` values), #9266 (`_coerce_param_value` raises
`AttributeError: 'list' object has no attribute 'lower'` on a `type: ["string","null"]` schema and the
call is dropped). Check `--enable-auto-tool-choice` too: without it nothing is parsed and the markup
necessarily lands in `content`.

## §B Anthropic XML in the output is a prompt/converter question, not a parser choice

Claude-Code-class clients and Responses-API converters describe tools in *their* format inside the
system prompt, and a model served behind such a converter can imitate it. No vllm tool parser accepts
`<parameter name=…>` + `</invoke>` (the MiniMax/GLM/Qwen XML parsers have their own tag pairs), so
changing `--tool-call-parser` cannot fix it: the repair belongs in the converter (translate the block
into `tool_calls` / `tool_use`) or in the prompt (stop teaching a non-native dialect).
Check: the same body directly against `/v1/chat/completions` with `tools` — if the model then emits
DSML, the converter primed the wrong dialect.

## §C A damaged marker is a tokenizer / upload artefact

`<｜DSML｜…>` is a special token; a sequence that spells it as ordinary characters either had no such
token or was rewritten after the fact. Tokenizer files travel with the checkpoint, so on a
**re-uploaded/quantized** weight set (`…-w4a8-mtp` from a hub upload) they are that upload's: read
`<model dir>/tokenizer_config.json` `added_tokens_decoder` and `chat_template.jinja` and compare with the
reference release — `../image-dependency-faults/modelscope-hub-patch-processor-revision.md`, and the
shard/copy check in `../accuracy-benchmarks/constant-token-degeneracy.md` (Step 3b).

## Case #10856: what that pin does and does not carry

- Given: image `v0.20.2rc1-a3`; 3P(dp2tp16)1D(dp16tp2) on 64 A3 cards; weights
  **DeepSeek-V4-Pro-w4a8-mtp**, the 2026-06-03 ModelScope upload (reporter's later public comment);
  clients codex (via CodexRosetta) and Claude Code. Not given: any serve command, raw body/SSE, engine
  log, or model-directory listing.
- That release has **no DeepSeek-V4 row** in `docs/source/user_guide/support_matrix/supported_models.md`
  and **no `docs/source/tutorials/models/DeepSeek-V4*.md`** (V3/3.1, V3.2, R1, OCR2 only) while
  `vllm_ascend/models/__init__.py:5-7` registers both V4 classes and the DSV4 patches ship — so there
  is no validated serve line to diff the reporter's against. State that as coverage, never as
  "unsupported".
- `patch_deepseek_v4_thinking.py:83-120` replaces `apply_chat_template` with
  `tokenizer.encode_messages(…)` (`drop_thinking` defaults `True`, `:115`; `reasoning_effort` collapses
  to `max` / `high` / `chat`, `:101-109`), so a `--chat-template` jinja is bypassed on the DSV4
  tokenizer path, and the *thinking* half of such a report ("reasoning 内容里出现 xxx") is that gate plus
  `--reasoning-parser` (`../accuracy-benchmarks/reasoning-output-routing.md`) — not the tool parser.
- Snapshot gap: this case's vllm export was `v0.13.0` (`72506c9`), which has neither
  `vllm/tool_parsers/deepseekv4_tool_parser.py` nor `vllm/models/deepseek_v4/`, while the patch's §12
  text targets **vLLM v0.20.2** — cite downstream only and say the upstream half is unverified.

## What to send back

The serve line (`--tool-call-parser`, `--reasoning-parser`, `--enable-auto-tool-choice`,
`--chat-template`), one raw `stream: false` body **and** the raw SSE for the same request, the
converter/proxy config, the model-directory listing (`tokenizer_config.json`, `chat_template.jinja`), and
one run with the parser flags removed so the model's own text is visible.

## Limits

Line numbers are v0.20.2rc1's; later refs move them (`delta-null-envelope-fields.md` cites `:408-448` in
the same file at v0.22.1rc1) — re-grep `_patched_extract_tool_calls` and `_process_streaming_buffer`.
Which of §B/§C produced #10856's two spellings is **not** settled by the issue: both pastes are
client-rendered, so the byte compare is the deliverable, not a verdict.
