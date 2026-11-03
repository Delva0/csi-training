# Serving-side parsers: streaming tool calls and reasoning content

Scope: reports where the server *shapes* text rather than computes it — a `tool_calls[*]` entry
without `id`, empty/truncated/out-of-order `function.arguments`, leaked markup inside the
arguments, `finish_reason: "tool_calls"` for a call the client cannot execute, `reasoning_content`
that is always null or stops appearing, thinking absent from turn 2 of a tool-call conversation,
`thinking_token_budget` ignored, a `</think>`/`</parameter>`/`<tool_call>` marker inside `content` —
while the plain text output is otherwise fine. Verified against the read-only clones on 2026-11-01 at the
tags named per statement; re-anchor with `git grep -n <symbol> <tag> -- <paths>`. Nothing ran on an
NPU: the delta is assembled from token text by Python regexes plus `DeltaMessage`/`DeltaToolCall`
models, with no NPU kernel in that path, so do not start at MTP/spec-decode kernels, the sampler or
the quant scheme.

The first question is always *which parser is running and who owns it*, because vLLM Ascend
patches these classes from `vllm_ascend/patch/platform/` and the rest lives in upstream vLLM.

## Where the code lives

- Upstream vLLM holds the parsers: `vllm/tool_parsers/<name>_tool_parser.py` and
  `vllm/reasoning/<name>_reasoning_parser.py`, chosen by the serve line's
  `--tool-call-parser <name>` / `--reasoning-parser <name>`. The directories exist from `v0.13.0`
  on (at `v0.9.0` the tool parsers still sat under `vllm/entrypoints/openai/tool_parsers/`). Both
  `v0.17.0rc1` and `v0.23.0` carry `vllm/tool_parsers/glm4_moe_tool_parser.py`,
  `vllm/tool_parsers/glm47_moe_tool_parser.py` and
  `vllm/reasoning/kimi_k2_reasoning_parser.py`. Read them with `git show <tag>:<path>`: the clone's
  working tree is `main`, where the only `entrypoints/openai/tool_parsers` is the `tests/` copy, so
  an `ls` on the old source path returns nothing and costs a round.
- **From vLLM `v0.24.0rc1` the parsers are declarative engines** — but `vllm/parser/` itself is
  older, so do not infer from the directory that a pre-0.24 report is on the engine: at `v0.18.0`
  and `v0.20.2` it holds only `abstract_parser.py`, `minimax_m2_parser.py`, `parser_manager.py`
  (and at `v0.23.0` the minimax module is gone, still no `engine/`); `vllm/parser/engine/` and the
  per-model modules appear at `v0.24.0rc1`. The engine migration
  `f00e163f35` (#45701, "Add Streaming Parser Engine and new MinimaxM2 Parser") and
  `6c379b9e54` (#45915, GLM4.7/GLM5.1/GLM5.2) both land first in `v0.24.0rc1`, and from there a
  model's format is a transition table in `vllm/parser/<name>.py` driven by
  `vllm/parser/engine/` (`parser_engine.py`, `parser_engine_config.py` with `ParserState`
  (`CONTENT`/`REASONING`/`TOOL_*`) and `Transition(next_state, events)`, `events.py` with
  `EventType`, `streaming_parser_engine.py`, `incremental_lexer.py`,
  `registered_adapters.py`). The same CLI flags select it; reasoning parsers stay in
  `vllm/reasoning/` but delegate to the engine through `make_adapters(...)` (at `v0.26.0`
  `MiniMaxM2ReasoningParser` is a subclass of `MinimaxM2ParserReasoningAdapter`). At `v0.26.0`
  the tree also carries a separate `rust/src/parser/` crate — that is not the Python path a
  report runs. So for a `>= v0.24` report, read the engine file and the transition table, and
  remember an empty `events` tuple means the terminal text is swallowed rather than emitted.
- `glm47` is thin at every tag checked — at `v0.17.0rc1` a 23-line subclass, at vLLM `v0.20.2` a
  ~40-line subclass — of `Glm4MoeModelToolParser` that only redefines the `<tool_call>`/`<arg_key>`
  regexes (`func_detail_regex` for the tool name plus optional `<arg_key>` section, `func_arg_regex`)
  and, at 0.20.2, `supports_required_and_named = False`. All streaming logic is inherited from
  `vllm/tool_parsers/glm4_moe_tool_parser.py`, so a `glm47` streaming defect is read in the parent
  file. The parent's line numbers move: at `v0.17.0rc1` `extract_tool_calls_streaming` is `:217`
  (with `_emit_tool_name_delta`/`_emit_tool_args_delta`); at vLLM `v0.20.2` the file is 504 lines
  and the entry points are `extract_tool_calls` `:190`, `_extract_tool_call_regions` `:293`,
  `_extract_tool_name_from_region` `:316`, `_build_args_json_so_far` `:331`,
  `_compute_args_diff` `:418`, `_ensure_tool_state_for` `:429`, `extract_tool_calls_streaming`
  `:440`.
- The plugin's side is a patch inventory, and it changes shape per tag. Filter to the parser-related
  subset — `git ls-tree -r --name-only <tag> | grep 'vllm_ascend/patch' | grep -iE
  'parser|tool|reason'` — and confirm registration in
  `vllm_ascend/patch/platform/__init__.py`. The full `patch/platform/` directory holds 13 files at
  `v0.18.0` and 29 at `v0.23.0`, so list the subset rather than the directory.
  - `v0.17.0rc1`: no parser/tool/reasoning patch at all.
  - `v0.18.0rc1`–`v0.19.1rc1`: `patch_glm_tool_call_parser.py` and
    `patch_tool_choice_none_content.py` (registered at `patch/platform/__init__.py:32-33` at
    `v0.18.0`). At `v0.18.0` the first is not a small wrapper: 1084 lines that replace
    `OpenAIServingChat.chat_completion_stream_generator` wholesale
    (`:1083 = _patched_chat_completion_stream_generator`, body `:200-818`), reimplement
    `Glm4MoeModelToolParser.extract_tool_calls_streaming` (`:819`, assigned at `:1084`), and graft
    helpers onto `OpenAIServingChat` as static/class methods
    (`_create_remaining_args_delta`/`_record_streamed_tool_args`/`_compact_json_fragment`/
    `_compute_remaining_tool_args`, `:1073-1076`). So on 0.18.x an entire-stream defect — leaked
    markers anywhere in `content`, tool-name contamination, wrong terminal chunk — is read in this
    plugin file, not only in the upstream parser, and its `from
    vllm.entrypoints.openai.chat_completion.serving import ...` header pins the upstream layout
    (that package path exists at vLLM `v0.18.0`).
  - `v0.20.2rc1`, `v0.21.0rc1` and `v0.23.0`: split and grown to `patch_glm47_tool_call_parser.py`,
    `patch_glm_tool_call_streaming.py`, `patch_minimax_m2_tool_call_parser.py`,
    `patch_deepseek_v4_tool_call_parser.py` and `patch_tool_choice_none_content.py` (the first
    two sets are identical). The MiniMax parser's `</parameter>` leak is
    covered in [minimax-m2-family.md](minimax-m2-family.md).
  - `main`: only `patch_glm53_reasoning.py` remains — absence at a later tag means the patch was
    relocated or dropped, never that the parser is unpatched everywhere.
- Read the patch body, not its name: `patch_glm47_tool_call_parser.py` at `v0.20.2rc1` wraps
  `_extract_tool_call_regions` and, for a *complete* region containing neither `<arg_key>` nor a
  newline (a zero-argument call `<tool_call>func</tool_call>`), appends `"\n"` after the tool name.
  That exists because of the name rule below — evidence that zero-argument GLM-4.7 calls were a
  known streaming gap at that tag, not proof that any given report is that gap.

## The legacy `qwen3xml` parser: whitespace-only `content` before the first tool call

Report (#15911, image `quay.io/ascend/vllm-ascend:v0.23.0`, upstream vLLM v0.23.0, streaming Qwen3
XML tool calling): after `</think>` the client receives a `content` delta containing only `"\n\n"`
before the first `tool_calls` delta; the expectation is no content delta, while a real preamble
("I will query this first:") must still be emitted. The reporter names the file and a `.strip()`
fix.

Verified at vLLM `v0.23.0` in `vllm/tool_parsers/qwen3xml_tool_parser.py` (1300 lines):

- `parse_single_streaming_chunks` `:175 if self.text_content_buffer and self.tool_call_index == 0:`
  emits the buffer as `DeltaMessage(content=...)` (`:177`) and clears it — no whitespace test.
- `_process_complete_xml_elements` `:260-266` flushes the pre-call buffer when the first
  `<tool_call>` / `<function name=` starts, guarded only by `and self.text_content_buffer`.
- The *between-tool-call* flush at `:954` does use `if self.text_content_buffer.strip():`, so the
  file itself carries the intended guard; the reporter's fix matches that precedent, which is the
  reason to treat "suppress only an all-whitespace buffer" as the narrow repair.

Ownership and upgrade answer: `vllm-ascend` `v0.23.0:Dockerfile:42` is `ARG VLLM_TAG=v0.23.0` and
clones that tag into `/vllm-workspace/vllm`, so this parser is upstream's file verbatim; the tag's
parser patch subset is `patch_glm47_tool_call_parser.py`, `patch_glm_tool_call_streaming.py`,
`patch_minimax_m2_tool_call_parser.py`, `patch_deepseek_v4_tool_call_parser.py` and
`patch_tool_choice_none_content.py` — no qwen3/qwen3xml patch, so no plugin flag or plugin patch
changes this. Upstream retired the file instead of fixing it: `2043258dec` "[Frontend] Support
strict mode for tool calling (#45003)" deletes `vllm/tool_parsers/qwen3xml_tool_parser.py`, and the
first tag containing it is `v0.23.1rc0`, where the registry alias `qwen3_xml` was re-pointed to
`qwen3coder_tool_parser` (`vllm/tool_parsers/__init__.py:161-164`); at `v0.24.0` the same alias maps
to `qwen3_engine_tool_parser` (`vllm/parser/qwen3.py`, the declarative engine above). So for a
v0.23.0 image the answers are an image/bundled-vLLM bump, `--tool-call-parser qwen3_coder` (the
alias target upstream itself selected — candidate, not run here), or patching the two guards.

## Streaming tool-call payloads: two distinct shapes

### A missing `id` or an empty `arguments` can be the correct delta shape

Rules read from `glm4_moe_tool_parser.py` at vLLM `v0.20.2`; each is a check to run before filing:

- `id` is emitted only on the first delta of an index: `extract_tool_calls_streaming` appends
  `DeltaToolCall(index=i, id=self._tool_call_ids[i], ...)` only while `"name" not in
  prev_tool_call_arr[i]` (`:465-475`); every later chunk for that call carries `arguments` alone. A
  client or proxy that renders only the last chunk, or merges by name, therefore shows a call with
  no `id`.
- `arguments` is a **diff**: `_compute_args_diff` (`:418`) returns `None` unless `args_so_far` grew,
  so a zero-argument call has no argument delta and `arguments` stays `""`. An empty string is the
  correct payload there, and `finish_reason: "tool_calls"` is correct with it.
- A tool name is recognised only once a `\n` or `<arg_key>` follows it
  (`_extract_tool_name_from_region` `:316` returns `None` otherwise and the streaming loop `break`s
  at `:462`), so a chunk holding `<tool_call>func` alone legitimately emits nothing; a client-side
  strictness check that treats "no delta" as "malformed" mis-reports it.
- `_ensure_tool_state_for` `:429` mints ids per index with
  `make_tool_call_id(id_type="random", func_name=None, idx=None)`
  (`vllm/entrypoints/chat_utils.py`), so ids differ run to run and say nothing about the request.
- The **non-streaming** path has the other empty shape. `Glm4MoeModelToolParser.extract_tool_calls`
  seeds `arg_dct: dict[str, Any] = {}` and always returns
  `arguments=json.dumps(arg_dct, ensure_ascii=False)`, so a zero-argument call comes back as
  `arguments: "{}"` (not `""`) — verified at vLLM `v0.23.0`
  (`vllm/tool_parsers/glm4_moe_tool_parser.py`, `extract_tool_calls`). An `arguments: "{}"` whose
  schema requires an argument is therefore *a call the model emitted with no argument tags*, and the
  server did the correct thing; the question moves to the chat template/tool schema and to the raw
  completion, not to argument parsing.
- Zero-argument GLM-4.7/5.x calls are supported **only by the `glm47` subclass**, and that is a
  parser-choice check: `glm47` replaces `func_detail_regex` with
  `<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>`, whose optional arg group exists (per its
  module docstring at `v0.23.0`) precisely so `<tool_call>func</tool_call>` parses. The parent
  `glm4_moe` regex at the same tag is `<tool_call>([^\n]*)\n(.*)</tool_call>` — a **mandatory**
  newline after the name — so the same text either fails to match (the parent logs
  `"Failed to parse tool call details from: ..."` and `continue`s, dropping the call) or matches
  with the name swallowed. So: a GLM-5.x report with `{}`/dropped calls is first checked against the
  serve line's `--tool-call-parser`; the tag's own tutorial is the authority for which name that
  model uses (`v0.23.0rc1:docs/source/tutorials/models/GLM5.2.md:144-145` prescribes
  `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice` for
  GLM-5.2-W8A8 on A2, repeated per hardware tab).
- **Before the guard, that optional group *crashes* instead of parsing.**
  `TypeError: expected string or buffer` at the parent's
  `pairs = self.func_arg_regex.findall(tc_args)` is `tc_args is None` from the `glm47` regex: a
  zero-argument `<tool_call>name</tool_call>` leaves group 2 (`(<arg_key>.*?)?`) unmatched, while the
  raising statement stays in the *inherited* `Glm4MoeModelToolParser.extract_tool_calls`. So a
  GLM-4.7 report whose frames name `glm4_moe_tool_parser.py` can still be a `glm47` defect — read
  the class `--tool-call-parser` selects before blaming the parent file (at vLLM `v0.14.1` the
  `glm47` module is 23 lines and redefines only `func_detail_regex`/`func_arg_regex`).
- The guard is upstream `a52d1396a7` (#32321, "fix: avoid crash on zero-arg tool calls in glm4
  parser"): `if not tc_detail: continue` plus `findall(tc_args) if tc_args else []`, **first release
  tag `v0.15.0rc0`**. Switching to `--tool-call-parser glm45` is not a workaround on those tags —
  the parent's `func_detail_regex` is `<tool_call>([^\n]*)\n(.*)</tool_call>`, so the same text
  raises `AttributeError: 'NoneType' object has no attribute 'group'` one line earlier. `glm47`
  itself exists from upstream `v0.14.0` (`8a7a414374`/#30876; `v0.13.0`'s `tool_parsers/__init__.py`
  maps only `glm45`), so a `v0.14.x` image has the parser but not the guard: the workaround is an
  image whose `Dockerfile` `VLLM_TAG` is >= `v0.15.0rc0`.
- The exception wording identifies the regex implementation, not the release: both GLM parser files
  `import regex as re`, and `regex` 2.5.148 raises exactly `TypeError: expected string or buffer`
  for `findall(None)`, whereas stdlib `re` on Python 3.11 raises `expected string or bytes-like
  object, got 'NoneType'` and `regex` 2026.9.29 raises `a bytes-like object is required` (checked
  locally, not on the reporter's host). A maintainer's "fixed by #7710 / #8832 / #8833" is the
  plugin's GLM tool-call *streaming* set — #7710 finish backfill, #8832 streaming-fix backport,
  #8833 `patch_tool_choice_none_content.py` — not this zero-arg crash.

The checks that separate server from client: which endpoint (`/v1/chat/completions` vs
`/v1/completions` — framework adapters such as Claude Code's Anthropic shim post to completions),
streaming or not, and how the client merges deltas by `index`. The server is the suspect only when
the stream itself has no first chunk carrying `id`+`name` for an index. Ask for the raw SSE stream
(`--enable-log-requests` or a proxy dump); the assembled object in a bug report is the client's
product.

### Empty deltas, then a tail fragment, is finish-chunk handling

Symptom shape (#7682, GLM-5-w4a8, `v0.17.0rc1`, image `v0.17.0rc1-a3-glm5-v2`, TP=16,
`--tool-call-parser glm47 --reasoning-parser glm45`): the stream opens with
`delta: {"role":"assistant","content":""}` and then a run of bare `delta: {}` chunks, and the only
tool-call chunk carries the tail of the JSON (`arguments: "\"}"`), while the same request with
`stream: false` returns the complete `arguments: "{\"todos\": \"- [x] ...\"}"`. The reporter's own
table is the useful evidence; there is no maintainer mechanism in the thread (open, one ping to a
teammate).

This is finish-chunk/terminal-chunk handling, not the sampler, the parser flag or an NPU kernel.
The fix history in the plugin, each proven by `git tag --contains`:

- `f83cb0e6d` — #7710, "Fix GLM47 tool-call finish backfill", 2026-03-28, **first tag
  `v0.18.0rc1`** (absent from `v0.17.0rc1`). It drains terminal parser chunks that carry both the
  final argument text and the closing `</tool_call>` suffix, and computes the finish backfill from
  the argument bytes actually emitted to the client instead of trusting parser-internal buffered
  state. Adds `tests/ut/patch/platform/test_patch_glm_tool_call_parser.py`.
- `0d8a70ae2` — #8832, "Backport GLM tool call streaming fixes", 2026-04-30, **first tag
  `v0.19.1rc1`**. Drains terminal chunks so inline final argument values are emitted, stops
  re-emitting function metadata in the final remaining-argument chunk (#8178), and splits a terminal
  argument chunk that carries `finish_reason=tool_calls` into an argument chunk with
  `finish_reason=null` plus an empty finish chunk so usage chunks follow (#8327). It cites upstream
  vLLM PR 37845 — so for tags after this one, check the paired upstream vLLM parser before
  concluding the plugin owns the behaviour.
- `patch_glm_tool_call_streaming.py` at `v0.20.2rc1`/`v0.23.0` carries the same
  `_create_remaining_args_delta`/`_compute_remaining_tool_args` helpers; the sibling
  `patch_glm47_tool_call_parser.py` rewrites
  `Glm47MoeModelToolParser._extract_tool_call_regions`.

Triage: pin the reported build, then ask which of those commits it contains
(`git merge-base --is-ancestor <sha> <ref>`); "upgrade to a tag containing #7710" is a *candidate*
workaround, not a confirmed one, because #7682 was never re-tested and no maintainer named the
cause. Do not spend the pass on the NPU side: at `v0.17.0rc1` the plugin's patches are unrelated,
and the tag's own `GLM5.md` tutorial documents no `--tool-call-parser`/`--reasoning-parser`, so
those flags are the reporter's own choice.

## Release boundaries and fix history

The `glm47`-subclass layout above belongs to the 0.20.x line; the same parsers were replaced by the
engine in `v0.24.0rc1`, so neither `main` nor a `>= v0.24` tag answers a 0.20.x report:

| commit | date | title | first `v*` release |
| --- | --- | --- | --- |
| `6c379b9e54` (#45915) | 2026-06-19 | `[Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser` | `v0.24.0rc1` |
| `f00e163f35` (#45701) | 2026-06-17 | `[Frontend] Add Streaming Parser Engine and new MinimaxM2 Parser` | `v0.24.0rc1` |
| `255431b5ed` (#64535) | 2026-10-29 | `[Bugfix][Parser] Parse closed GLM-4.7 tool arguments once while streaming` | no `v*` tag |
| `dc420193b7` (#64615) | 2026-10-29 | `[Bugfix][Frontend] Leave GLM-4.7 array and object tool arguments unpinned` | no `v*` tag |

Read those from `git tag --contains <sha> --sort=creatordate | grep '^v' | head -1`, not from a
plain `sort -V` (which prints `proto-*` tags first and puts a final release before its own rc). The
engine pair therefore *is* an upgrade path for a GLM4.7/MiniMax parser report on `0.21`–`0.23`
(pre-engine layout) but not for a 0.20.x one; the last two commits are still unreleased, and a title
alone is not a mechanism — do not paraphrase those titles into a root cause. The plugin-side
boundaries are `v0.18.0rc1` (#7710) and `v0.19.1rc1` (#8832), above.

## Reasoning markers in `content`

- `enable_thinking: false` yet `</think>` (or its HTML-escaped form) inside `content` means the
  reasoning parser never split it. At the reported tag that parser is upstream
  (`vllm/reasoning/kimi_k2_reasoning_parser.py` exists at both `v0.17.0rc1` and `v0.23.0`). The
  plugin adds no Kimi or GLM reasoning patch in that range — its parser-related patches there are
  the tool-call ones above plus `patch_tool_choice_none_content.py`, and
  `patch_glm53_reasoning.py` appears only on `main` — so read the parser's `adjust_request` /
  `extract_reasoning_streaming` and the chat template's rendered thinking prefix at the tag.
- A repro script's `html.unescape("".join(parts))` (as in #13002) turns an escaped `&lt;/think&gt;`
  into a literal `</think>`; check the raw `delta.content` before calling it a server leak.
- A `</parameter>` leak from a MiniMax tool call is the separate MiniMax parser patch, and a
  `</think>` that vanishes only when a tool call is present is the MiniMax append-think /
  tool-engine composition — both in [minimax-m2-family.md](minimax-m2-family.md). The second is the
  engine-era trap to know: an empty `events` tuple on the `CONTENT`/`THINK_END` transition consumes
  the marker, and no released vLLM tag up to `v0.33.0rc2` fixes it.
- Malformed token soup, repetition collapse and English system-prompt fragments arriving in the
  *same* report as a reasoning-marker leak are a different class (sampling, PD KV transfer,
  speculative decoding): one parser fix does not explain them
  ([sampling-and-output-quality.md](sampling-and-output-quality.md),
  [pd-disaggregation.md](pd-disaggregation.md)). Kimi-K2.6 PD reports at `v0.23.0` are triaged that
  way — [model-tutorial-recipes.md](model-tutorial-recipes.md).

## Which field carries the thinking: `reasoning`, never `reasoning_content`

vLLM's split thinking travels in a vLLM-specific `reasoning` field; `reasoning_content` exists only as
an *input* alias from `v0.22.0rc1`. Checked with
`git grep -c reasoning_content <tag> -- vllm/entrypoints/openai/engine/protocol.py
vllm/entrypoints/openai/chat_completion/protocol.py`:

| tag range | streaming delta | non-streaming `message` | accepted on input |
| --- | --- | --- | --- |
| `v0.18.0`-`v0.21.x` | `DeltaMessage.reasoning` (`engine/protocol.py:258-263` at `v0.18.0`) | `ChatMessage.reasoning` (`chat_completion/protocol.py:64`, under the comment "vLLM-specific fields that are not in OpenAI spec") | `reasoning` only: `CustomChatCompletionMessageParam` (`chat_utils.py:273-300`) declares it, `_parse_chat_message_content` reads `message.get("reasoning")` and mirrors it into `result_msg["reasoning"]`/`["reasoning_content"]` (`:1511`, `:1537`); the `ConversationMessage.reasoning_content` entry at `:329` says "Deprecated" and is only ever written by that mirror (`:1538`) — nothing reads it out of an incoming request |
| `v0.22.0rc1`+ | unchanged | unchanged (`:72` at `v0.30.0`) | both names: `346cf163a1` (#42664) "Normalize reasoning_content to reasoning for client compatibility", a `mode="before"` request validator that pops `reasoning_content` into `reasoning` |

Two consequences decide most reports of this shape:

- A client (or a curl recipe) that reads `reasoning_content` prints **null while the parser is
  working perfectly**: at `v0.18.0` the field is simply not in the schema, so the symptom survives
  every upgrade that does not change the client. Ask for one raw chunk / the response JSON and look
  for `reasoning` before touching parser code; a maintainer reply that pastes a working response is
  evidence of this, not of a fixed defect.
- **An assistant turn echoed back with `reasoning_content` loses its thinking before `v0.22.0rc1`**,
  because the field is dropped at request validation; the chat template then sees an assistant turn
  with no think block. That is the first candidate for "turn 2 has no thinking" on 0.18-0.21, and the
  workaround is to send the previous turn's text as `reasoning` (or upgrade to a tag with #42664).
  Whether the model's own template needs the history to carry it at all is a **separate, unverified**
  step: the Qwen3.5 template lives in the checkpoint repository, not in any clone here, so ask the
  reporter for `tokenizer.apply_chat_template(...)` output or do not claim it.

## Qwen3.5 and Qwen3.6: the `<think>` is in the prompt, not in the output

At `v0.18.0` `vllm/reasoning/qwen3_reasoning_parser.py` (class `:16`) states the arrangement in its
docstring: the Qwen3.5 template places `<think>` in the prompt, so only `</think>` is generated, the
`enable_thinking` switch renders `<think>\n\n</think>\n\n` into the prompt, and "the serving layer
detects this via `prompt_is_reasoning_end` and routes deltas as content without calling the streaming
parser". Mechanics to check at the reported tag:

- `start_token`/`end_token` are `<think>`/`</think>` (`:45`, `:50`); `thinking_enabled` comes from
  `chat_template_kwargs` (`:42`), and the serving layer really does pass them
  (`chat_completion/serving.py:233-235` at `v0.18.0`), so a request-level
  `chat_template_kwargs={"enable_thinking": false}` changes non-streaming extraction
  (`extract_reasoning`: no `</think>` then means "all content" instead of "truncated thinking").
- `is_reasoning_end` scans the *prompt* token ids backwards — start token seen first returns `False`,
  end token `True` (`basic_parsers.py:64`), and `serving.py:686-690` computes it once per choice.
  `True` means "prompt already ended reasoning, emit deltas as content"; a Qwen3.5 generation prompt
  ending in `<|im_start|>assistant\n<think>\n` evaluates `False`, i.e. the parser runs.
- `extract_reasoning_streaming` therefore returns `DeltaMessage(reasoning=...)` for every delta before
  `</think>` and only starts `content` after it. "Thinking missing from `content`" plus "reasoning
  field has the thinking" is this code working, not a partially applied parser.
- `deepseek_r1` behaves the same way on a no-start-tag model: `deepseek_r1_reasoning_parser.py` adds
  the "no start token in this delta" branch and assigns everything before `</think>` to `reasoning`.
  So "both `qwen3` and `deepseek_r1` give no thinking in `content`" is one expectation, not two bugs.
- A perceived **TTFT regression** with the parser on is normally the client: the stream emits reasoning
  deltas from the first token, so a client that renders only `content` waits until `</think>`. Treat
  "first token much slower" as a candidate and separate it with a timestamped raw SSE dump: a chunk at
  t≈0 carrying `delta.reasoning` refutes a server-side buffering cause. At `v0.18.0` the streaming
  extractor returns a `DeltaMessage` per delta with no buffering, and `include_reasoning` gates only
  the harmony/gpt-oss branch plus `reasoning_effort: "none"`
  (`chat_completion/protocol.py:783-788`).

## `thinking_token_budget` is a sampler gate, not a parser feature

"Parser loaded, thinking split into `reasoning`, but `thinking_token_budget=50` truncates nothing and
`reasoning_tokens` stays 0/null" is a **budget-holder** report. The chain, verified at upstream
`v0.21.0rc1`-`v0.23.0` and plugin `v0.21.0rc1`-`v0.26.0rc2`:

1. `--reasoning-parser <name>` alone creates the config: `_set_default_reasoning_config_args`
   (`vllm/engine/arg_utils.py:2440`, called at `:1742` at `v0.23.0`; `:2370`/`:1676` at `v0.21.0rc1`)
   instantiates `ReasoningConfig()` and sets `.reasoning_parser`, and
   `ReasoningConfig.initialize_token_ids` (`vllm/config/reasoning.py:62`+, same fallback at both tags)
   takes `reasoning_start_str`/`reasoning_end_str` from the parser class when the strings are empty. So
   no `--reasoning-config` is needed for `<think>`/`</think>` models.
2. If `reasoning_config` were absent/disabled, the request would fail loudly:
   `vllm/v1/engine/input_processor.py:101-108` raises "thinking_token_budget is set but
   reasoning_config is not configured". A *silent* no-op therefore rules that gate out.
3. The enforcement point is sampling: `vllm/v1/sample/sampler.py:381` reads
   `sampling_metadata.thinking_budget_state_holder` and applies the holder only when it is not `None`
   and `has_tracked_requests()`; the holder is built in `vllm/v1/worker/gpu_input_batch.py:110` via
   `maybe_create_thinking_budget_state_holder(reasoning_config, max_num_reqs, num_spec_tokens,
   device, pin_memory)` and surfaced through `_make_sampling_metadata`.
4. The Ascend batch re-implements `__init__` instead of calling `super()`, and hardcodes
   `self.thinking_budget_state_holder = None` (`vllm_ascend/worker/npu_input_batch.py:57-59`; the
   comment cites upstream vllm#34668, introduced with #8983 "Upgrade vLLM to v0.20.1"), with no
   `reasoning_config` parameter at all. Every other method is inherited, so the sampler always sees
   `None`.

Release boundary: present at `v0.21.0rc1`, `v0.23.0rc1`, `v0.24.0rc1`, `v0.25.1rc1`, `v0.26.0rc2`;
fixed by `676e91b72` (#13146, merged 2026-08-10), which adds the `reasoning_config` parameter and the
`maybe_create_...` call — **first tag `v0.27.1rc1`**, and it is *not* an ancestor of
`origin/releases/v0.23.0`. Two boundaries on that fix: its PR text is about speculative decoding (its
own test drives `reasoning_config` explicitly and shows think counts capped at the budget), so
"budget works on NPU end-to-end" is an inference from an identical code path rather than an
NPU-verified result; and a second, independent gate exists — with
`VLLM_USE_V2_MODEL_RUNNER=1` the engine warns "Model Runner V2 does not yet support the
thinking_token_budget request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required"
(`vllm/config/vllm.py:2071-2076` at `v0.23.0`). Check for that warning before blaming the holder.

Triage bookkeeping this class teaches: the reporter's title ("parser cannot recognise `<think>`") is
wrong in a self-evident way — their own evidence is that `reasoning` holds the full thinking — so say
plainly that the parser ran and the budget was skipped. In the recorded thread the correct file and
line came from a commenter with no repo association, the fix PR they advertised (#13157, base
`releases/v0.23.0`) was **closed unmerged**, and the change that actually shipped is a different
number and author (#13146) whose title mentions spec decoding. Resolve every cited PR with
`gh`/`pulls/<n>` `state`+`merged` **and** `git log -S<symbol>` before repeating a fix number.

## Recorded instances

- #15911 — v0.23.0 image, streaming Qwen3 XML: whitespace-only `content` delta before the first
  tool call, class and lifecycle in the `qwen3xml` section above.
- #11788 — GLM-5.1 w4a8 on 910C, PD split (P: dp1/tp16; D: dp16/tp2 with
  `--speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp"}'`),
  `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --async-scheduling`:
  some responses carry a `tool_calls` entry with no `id` and `arguments: ""` while `finish_reason`
  is `tool_calls`. The thread carries one comment ("plz see this issue") and the labels
  `glm5/triaged/stale/advanced-features` — no mechanism, no fix. Keep the MTP/PD coincidence a
  candidate at most: MTP only changes which text is produced, so the separating evidence is the raw
  SSE stream (is there a first chunk with `id`+`name`, then a `{}` argument delta?) and the same
  request with speculative decoding off.
- #8022 (Qwen3.5-27B, `vllm-ascend 0.18.0rc1`, streaming, `--reasoning-parser qwen3` or
  `deepseek_r1`) was closed 2026-08-11 with labels `stale` + `wait-feedback` and no root cause:
  "`reasoning_content` always null, and with a parser `content` loses the thinking" is the
  `reasoning`-field table above, and the thread's own answer pastes a response whose thinking sits
  under `reasoning`. The tag's `docs/source/tutorials/models/Qwen3.5-27B.md` prescribes no
  `--reasoning-parser` at all (its verification is a `/v1/completions` call), so the flag is the
  reporter's choice — check the tutorial before treating the feature as documented-supported.
- #9659 (Qwen3.5-27B, `v0.18.0`, `--tool-call-parser qwen3_coder --reasoning-parser qwen3`, open,
  two ping comments, no mechanism): thinking disappears from turn 2 of a tool-call loop. Report the
  `reasoning`-input boundary (`v0.22.0rc1`) as the leading candidate, the template as the unverified
  half, and the raw prompt as the check that separates them.
- #13060 (Qwen3.6-27B-w8a8 on 910B, `vllm-ascend 0.23.0rc1` and `v0.21.0`, open, 5 comments) is the
  hardcoded-`None` budget holder above; the reporter's own control-variable post is the useful part.
- #7580 (GLM-5, `v0.17.0rc1` A3, large-EP PD, still open with 3 comments, `--speculative-config
  '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'`) — client reports
  `JSON Parse error: Unterminated string` while `num_speculative_tokens: 1` is clean, and the
  reporter's follow-up comment is a raw SSE dump whose `arguments` fragments concatenate correctly
  (reassembly rule below). Same family as #7682/#11788: no maintainer mechanism, no fix, so deliver the
  reassembly/control answer; the dump is a single ~75 KB body — read it once (or by comment id), not
  by character offset, and it will still not name a root cause.
- #7682 (GLM-5 w4a8, `v0.17.0rc1`) is open with no mechanism; #13002 (Kimi-K2.6 w4a8 PD,
  `v0.23.0`) is closed `not_planned` after maintainers asked for the matched upstream vLLM commit
  and a run with Eagle3 disabled — those two experiments are the next verification boundary, not a
  fix.
- #10892 (Qwen3.6-35B-A3B-w8a8, Atlas 300I DUO, `v0.21.0rc1`, closed 2026-07-23) reported
  "过度思考" over a range of `max_tokens`: the serve line carries `--tool-call-parser qwen3_coder` and
  **no `--reasoning-parser`**, the response's `reasoning` is `null`, and `content` opens with the
  model's English thinking plus a literal `</think>` before the JSON (2548 of 4096 completion
  tokens). Two independent findings, not one cause: add `--reasoning-parser qwen3` (or send
  `chat_template_kwargs={"enable_thinking": false}`) so the thinking stops filling `content`, and
  the separate 310P MoE routing-accuracy defect fixed at `v0.23.0rc1`
  ([310p-atlas-300i-duo.md](310p-atlas-300i-duo.md)).

## Verification recipe

- Server-side: send the same request with `stream: false` and `stream: true`, concatenate the
  streaming `delta.tool_calls[*].function.arguments`, and assert it parses as JSON equal to the
  non-streaming `arguments`. A mismatch is the finish-backfill class; a match moves the question to
  the parser name, the chat template or the client's reassembly.
- Cheap presence marker at a tag: `tests/ut/patch/platform/test_patch_glm_tool_call_parser.py`
  exists only where the backfill patch does.
- `completion_tokens` that dwarfs the delivered `content` (9665 tokens against 831 characters in
  #13002, `finish_reason: stop`) means the tokens went to a channel the client did not render —
  reasoning, tool-call arguments, or a repetition loop. Compare `usage` with the reassembled stream
  before calling it truncation or a transport fault.

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **A serve line with no `--reasoning-parser` returns the thinking in `content` — that is the whole
  answer for most "过度思考 / overthinking / thinking is not suppressed" reports.** Read the serve
  command and one response object before any parser or sampler theory: `reasoning: null`, a literal
  `</think>` inside `content`, and a `completion_tokens` dominated by prose means nothing was
  configured to split the stream. Only after the flag is on does a surviving marker point at the
  parser, the template or `prompt_is_reasoning_end`.
- **`reasoning_content` null / thinking gone / "parser not loaded" on a Qwen3.5-class model is the
  field name first.** Read one raw chunk: `reasoning` carrying the thinking means the parser ran and
  the client looks at the wrong key (or, on <= `v0.21.x`, the history was sent back under a key the
  request schema drops). Do not open `vllm_ascend/patch/` for this class before that check.
- **`thinking_token_budget` ignored is the batch's `thinking_budget_state_holder`, not the parser.**
  A silent no-op means the loud "reasoning_config is not configured" gate passed, so read
  `npu_input_batch.py` for a hardcoded `None` and quote the first tag that fixes it (`v0.27.1rc1`),
  plus the `VLLM_USE_V2_MODEL_RUNNER` warning as the sibling cause.
- **Grep `thinking` and `budget` separately in this tree.** `git grep -iE "thinking|budget"
  <tag> -- vllm_ascend` is swamped by scheduler `token_budget`/`encoder_compute_budget` hits and a
  `head -40` window can show none of the real ones; use `git grep -n -i thinking <tag> -- vllm_ascend`
  for the feature and keep `budget` out of the pattern.
- **A `tool_calls` entry with no `id` or `arguments: ""` is the parser's delta shape, not MTP or the
  model.** `id` is streamed only on an index's first chunk and `arguments` only as a diff, so a client
  that renders the last chunk shows exactly that object; get the raw SSE stream and the client's merge
  rule before reading kernels, and read a 0.20.x `glm47` report at its tag because the parser was
  replaced on `main` afterwards.
- **A client-side `JSON Parse error: Unterminated string` is the reassembly boundary first, not
  dropped tokens.** `arguments` deltas are emitted as token-boundary fragments and routinely end
  mid-word (`…/agent/de` then `ploy_agent/src/main`), which concatenate to valid JSON. Reassemble the
  stream in `index` order before attributing the break to speculative decoding or the sampler: in the
  recorded #7580 dump the fragments joined into the complete `{"filePath": ".../K8sYamlAgent.java"}`,
  so that dump does not show lost characters and cannot carry a token-loss claim on its own. The
  check that separates the two is the non-streaming call in the Verification recipe above (equal JSON =
  parser/merge, mismatch = finish-backfill class); say which one the evidence supports.
