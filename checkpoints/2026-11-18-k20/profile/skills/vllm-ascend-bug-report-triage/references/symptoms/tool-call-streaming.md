# OpenAI tool-call output shape: missing `id`, empty/truncated `arguments`, finish-chunk packing

## Scope

The service is healthy — HTTP 200, no traceback, no device fault — but the tool call is wrong. Four
shapes appear, and they have different owners:

| shape | where it is produced |
| --- | --- |
| a streaming `tool_calls[]` entry with no `id`, `function.name` missing, `arguments` an empty string | the vLLM serving finish chunk, or the vllm-ascend monkeypatch that rebuilds it (§2) |
| streamed `arguments` arrive only as a one- or two-character tail (`"`, `"}`) after empty deltas, while `"stream": false` is complete | parser delta construction + the serving finish backfill (§3) |
| a `finish_reason: "tool_calls"` chunk carrying argument bytes, with no `delta.tool_calls` at all | the finish-chunk split, and the zero-argument inline call the parser drops (§2, §4) |
| the same wrong shape in a *non-streaming* body | client/proxy artefact: `ToolCall` always sets an `id` and `arguments` is `json.dumps(...)`, `{}` for no args (§1) |
| an assistant `content` delta that is **whitespace only**, immediately before the first tool call, while the tool call itself is correct | the legacy XML parser's text buffer, flushed without a `.strip()` test — and it may be an **upstream-bundled** file, not a `vllm_ascend/patch/` one (§4b) |
| no tool call at all, HTTP 200, model text returned as `content`, and the log carries `Failed to extract tool call spec` / `TypeError: expected string or buffer` from `glm4_moe_tool_parser.py`, then `Failed to extract any tool calls.` | the **upstream** `glm4_moe` parser on a zero-argument call, before its guard existed — the call is caught and dropped, not streamed badly (§4a) |
| the model's own markup reaches the client: `</think>`, `<think>`, `<tool_call>`, `<arg_key>`/`<arg_value>` inside `content`, or that markup inside `function.name`/`arguments` | **parser selection**, not the engine: no/mismatched `--reasoning-parser` leaves the thinking block and its closing tag in `content` by design, and an unconsumed tool region means the tool parser does not match the model's format (§4c) |

Reports phrase it as 「id不存在、function.arguments 为空字符串；finish_reason 为 tool_calls」 or "the
client drops the last fragment". GLM deployments dominate (`--tool-call-parser glm47 --reasoning-parser
glm45 --enable-auto-tool-choice`), and the reporter often also runs MTP
(`--speculative-config '{"method":"deepseek_mtp"}'`) — a red herring here (§6).

Not this page: a parser crash inside a guided-decoding library (`grammar-bitmask-and-dep-skew.md`), a
worker that dies while serving (`pd-collective-hang.md`), content that is fluent nonsense
(`flashcomm1-and-moe-reduce.md`), or a connector/registration failure (`kv-connector-registration.md`).

## 1. A delta is not a whole call — settle this before reading any code

- `DeltaToolCall` (`vllm/entrypoints/openai/engine/protocol.py:239-243` at `v0.20.2`) makes every field
  optional but `index`; `ToolCall` (`:228`) declares `id: str = Field(default_factory=make_tool_call_id)`
  and `FunctionCall.arguments: str` (required). A **non-streaming** body therefore always carries an
  `id`, and `arguments` is `json.dumps(<parsed args>)` — never `""`.
- A **streaming** chunk is a delta: `id`/`type`/`function.name` belong to the first chunk for that
  `index`; later chunks carry only `function.arguments`. One chunk quoted out of the stream therefore
  *legitimately* has no id, and a `finish_reason` in the report means the reporter is looking at a chunk
  (or at a client object assembled from chunks), not at a non-streaming response.

So the first question is "which chunk, and did an earlier chunk for that `index` carry the id?". Get the
raw stream rather than the client's rendering:

```bash
curl -sN http://<host>:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"<the failing prompt>"}],
       "tools":[...],"tool_choice":"auto","stream":true}' > /workspace/scratch/stream.txt
uv run --no-project python scripts/merge_tool_deltas.py /workspace/scratch/stream.txt
```

`scripts/merge_tool_deltas.py` merges by `index` (standard library only; also accepts a JSON body or a
log holding `data: {...}` lines) and prints, per index, whether an `id` was ever streamed and which
chunk carried `arguments` together with `finish_reason`:

| probe result | what it means |
| --- | --- |
| an earlier chunk carried an `id` for the index | the stream is API-conformant; the defect is the client's delta merge (§5) |
| no chunk ever carried an `id` for the index, and a `finish_reason: "tool_calls"` chunk has no `delta.tool_calls` | the tool-call header never streamed — parser drop (§4) |
| `function.arguments == ""` with `finish_reason: "tool_calls"` | the terminal remaining-args delta (§2) |
| argument bytes in the same chunk as `finish_reason` | the finish chunk was not split (§2) |
| the report is `stream:false` | not this page: a missing id or `""` arguments in a non-streaming body is a client/proxy artifact |
| the client errors with `JSON Parse error: Unterminated string`, or the merged arguments are bare value text (no `{"key":` prefix, no closing `"}`), while `stream:false` returns the complete JSON | the same class as §3: the parser withheld the deltas and the finish backfill emitted only a tail, so the concatenation is not JSON |

## 2. Where the terminal chunk is built — three layers, check them in this order

1. **vllm-ascend's own monkeypatches**: `vllm_ascend/patch/platform/patch_*tool_call*.py`,
   `patch_*thinking*.py`, `patch_*usage_accounting*.py`, `patch_tool_choice_none_content.py`. They are
   imported unconditionally by `vllm_ascend/patch/platform/__init__.py`, so a patch named after one
   model runs for every deployment on that build. Read the `Why`/`How` block for the file in
   `vllm_ascend/patch/__init__.py` — it states the defect being papered over and the upstream PR.
2. **The upstream parser**, `vllm/tool_parsers/<model>_tool_parser.py` at the *paired* vLLM tag
   (`glm4_moe_tool_parser.py`, `glm47_moe_tool_parser.py`, `deepseekv4_tool_parser.py`,
   `minimax_m2_tool_parser.py`, …): region regexes, the incremental diff against
   `prev_tool_call_arr`/`streamed_args_for_tool`, and whether `id` is produced at all.
3. **The serving layer**, `vllm/entrypoints/openai/chat_completion/serving.py`: the streaming
   generator's final chunk and the non-streaming path (`extract_tool_calls` → `make_tool_call_id`,
   `~:1345` at `v0.20.2`).

```bash
# batch-2 probe: what does this build patch, and what does the parser look like?
git -C /workspace/repos/vllm-project/vllm-ascend ls-tree --name-only <ascend-ref>:vllm_ascend/patch/platform
git -C /workspace/repos/vllm-project/vllm-ascend show <ascend-ref>:vllm_ascend/patch/__init__.py | sed -n '140,220p'
git -C /workspace/repos/vllm-project/vllm grep -n "_create_remaining_args_delta\|_should_check_for_unstreamed" <vllm-ref> -- vllm/entrypoints/openai/chat_completion/serving.py
```

`ref_snapshot.sh` prints the `patch/platform` file list, so batch 1 already carries it.

At vLLM `v0.20.2` the terminal chunk is built like this:

- `:1112-1119` — only on the **final** step (`output.finish_reason is not None`) with a tool parser and
  `auto_tools_called`;
- `_should_check_for_unstreamed_tool_arg_tokens()` (`:1790-1812`) fires when the current delta has a tool
  call whose `function.arguments is not None` — an *empty* string satisfies that test;
- the remaining text is `expected_call.replace(actual_call, "", 1)` (`:1159`), expected from
  `tool_parser.prev_tool_call_arr[index]["arguments"]`, actual from `streamed_args_for_tool[index]`
  minus the current delta — when nothing is left, `remaining_call == ""`;
- `_create_remaining_args_delta(delta_message, remaining_call, index)` (`:1161`, staticmethod at
  `:1815-1841`) **replaces** the delta wholesale, and upstream copies `id`/`type`/`name` from the
  current delta, which at finish time is usually a later args-only delta, so they are often `None`
  anyway;
- the same choice object then gets `finish_reason_ = "tool_calls"` whenever `auto_tools_called`
  (`:1170-1179`), so a delta and the finish reason share one chunk.

vllm-ascend replaces step 3 for a window of releases, and the replacement decides the shape at the
reporter's tag:

| tag | file (registered in `vllm_ascend/patch/platform/__init__.py`) | shape |
| --- | --- | --- |
| `v0.17.0rc1` | none — `patch/platform/` holds only balance-schedule/distributed/fusion-matcher/mamba-config/minimax-config/multiproc/sched-yield patches | upstream behaviour only |
| `v0.18.0rc1` | `patch_glm_tool_call_parser.py` (#7710, `f83cb0e6d`): drain terminal parser chunks, compute the finish backfill from the argument bytes **actually emitted** | tail-only finals disappear; the patch is the first fix of §3 |
| `v0.19.1rc1` | the same patch rewritten (#8832, `0d8a70ae2`), `_create_remaining_args_delta` with `fallback_tool_call_id/_type/_name`; UT `test_create_remaining_args_delta_uses_fallback_metadata_for_args_only_delta` | args-only delta unless the caller passes fallback metadata |
| `v0.20.2rc1` | `patch_glm_tool_call_streaming.py` (registered `:33`; `_create_remaining_args_delta` `:34-55`, `_terminal_tool_arg_choice` `:58-66`, `_split_terminal_tool_arg_chunk` `:69`) + `patch_glm47_tool_call_parser.py` (registered `:35`; region rewrite `:39-44`) | `_create_remaining_args_delta` emits `arguments` only, no `id`/`type`/`name`, when no fallbacks are passed (upstream calls it with three positional args); `_split_terminal_tool_arg_chunk` splits an argument chunk away from the finish chunk **only when `function.arguments` is truthy**, so an empty fragment stays glued to `finish_reason: "tool_calls"` |
| later | #9410 (`e1db5050a`) removes the parser-level patch as redundant (upstream parser drains); #9901/#9902 (`86797d44a`/`1033e4110`) carry the GLM47 inline zero-arg region; #11453 (`b2dc73890`) deletes the obsolete parser patches entirely | re-read the registry at the ref |

Net at `v0.20.2rc1` + vLLM `v0.20.2`: the terminal choice can be exactly

```json
{"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]},"finish_reason":"tool_calls"}
```

with no `id`, `type` or `function.name` anywhere in that chunk — the reported symptom, produced by the
serving layer, not by the model or the NPU. The patch's own UTs cover a non-empty terminal fragment
(`test_terminal_argument_chunk_is_split_before_finish_chunk`) and a metadata-free remaining-args delta
(`test_remaining_args_delta_omits_metadata_by_default`); the empty-fragment case is untested, which is
why it survives.

## 3. Streaming-only payload loss: the parser withholds, the backfill emits the tail

At vLLM `v0.17.0rc1`, `vllm/tool_parsers/glm4_moe_tool_parser.py` (inherited by
`glm47_moe_tool_parser.Glm47MoeModelToolParser`) decides per argument how to stream:

- `_is_string_type(tool_name, arg_name, tools)` (`:122-144`) returns **False** when `tools` is `None`,
  when the matching tool has no `parameters`, or when the property's `type` is anything but `"string"`;
- string-typed argument: `:280-300` emits the `{"key":"` prefix and then escaped content fragments, so
  the client sees the value grow;
- non-string argument: `:316-320` **waits for the complete `</arg_value>`** before emitting anything, so
  a value that never closes streams nothing at all;
- when the `</tool_call>` closes with nothing started, `_close_args_if_needed()` (`:506-518`) emits
  `"{}"`; with a partially started object it emits `"}"`.

The **one- or two-character final payload comes from the serving layer's finish backfill, not from the
parser**: it computes "remaining arguments" as `expected_arguments[len(streamed):]`, where `streamed`
used to be the parser's *internal* buffer rather than the bytes actually written to the client. A
parser that withholds deltas therefore leaves the internal buffer complete while the wire saw nothing,
and the backfill emits only the last one or two characters (`"`, `"}`). That is #7710's own
description — "compute finish backfill from the tool argument bytes actually emitted to the client,
instead of trusting parser-internal buffered state" — and its regression tests assert exactly this
subtraction (`expected '{"a":1}'` − `streamed '{"a":1'` → `"}"`).

Non-streaming `extract_tool_calls` (`:165-215`) parses the whole text at once and is therefore always
complete — which is exactly the reported asymmetry. The discriminator is free: `"stream": false` on the
same request.

## 4. Zero-argument inline calls never stream

`glm47` parses `<tool_call>get_current_time</tool_call>` in non-streaming but drops it in streaming
(vllm#44326, closed): the streaming path uses the parent `_extract_tool_name_from_region()`, which
returns a name only when the region contains `\n` or `<arg_key>`, so the run ends with a
`finish_reason: "tool_calls"` chunk and no `delta.tool_calls`. vllm-ascend patches exactly that region
extractor (`patch_glm47_tool_call_parser.py`, upstream vllm #44326/#44327). Falsifier: `stream:false`
on the same prompt returns the call and the model output contains the inline zero-argument form.

## 4a. A zero-argument call on `glm4_moe` raises, and the call is dropped (#6765)

Report shape: image `quay.io/ascend/vllm-ascend:v0.14.0rc1`, GLM4.7. Nothing crashes and HTTP is 200, but the
call never arrives as a tool call; the log shows, twice per request,

```text
ERROR [glm4_moe_tool_parser.py:122] Failed to extract tool call spec
  File "/vllm-workspace/vllm/vllm/tool_parsers/glm4_moe_tool_parser.py", line 104, in extract_tool_calls
    pairs = self.func_arg_regex.findall(tc_args)
TypeError: expected string or buffer
WARNING [glm4_moe_tool_parser.py:171] Failed to extract any tool calls.
```

**Verified:** the failing statement is `pairs = self.func_arg_regex.findall(tc_args)`, and its **line number
is the version fingerprint** — `:104` at vLLM `v0.13.0` (exactly the quoted line) versus `:118` at
`v0.14.0rc1`/`v0.14.0`/`v0.14.1`, where the `_is_string_type` helper sits above it. The image tag
`v0.14.0rc1`'s own `Dockerfile:51` says `ARG VLLM_TAG=v0.14.1`, so the bundled file and the tag story
disagree; the guard is missing from **every** `v0.14.x`, which is what the answer rests on. The raise is
caught by `except Exception: logger.exception(...)` and the handler returns
`ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)` — hence a fluent
answer that contains the raw `<tool_call>…` text and no `tool_calls`.

The fix is upstream `a52d1396a7` (#32321, 2026-01-15, "fix: avoid crash on zero-arg tool calls in glm4
parser"): `if not tc_detail: continue` plus `pairs = self.func_arg_regex.findall(tc_args) if tc_args else []`.
`git tag --contains a52d1396a7 --list 'v0.*' | sort -V` → first release tag **`v0.15.0`** (`v0.15.0rc0`
already has it). The module does `import regex as re`, so a non-string argument reports the *library's*
old-style `TypeError: expected string or buffer`; stdlib `re` would say "expected string or bytes-like
object", which is the tell that this is not a Python-3 `re` message.

**Do not spend a batch deriving how `tc_args` becomes `None`.** The guard covers both a missing detail match
and a falsy args group, the commit subject already names the trigger (a zero-argument call), and the
deliverable is the version lever; deriving it is how a run spends three batches and ships nothing.

Fork side (`reported`, the thread's only comment — `yiz-liu`, 2026-05-28): "Should have been fixed by #7710,
#8832, #8833". All three are vllm-ascend PRs and all merged — #7710 `[Bugfix][Platform] Fix GLM47 tool-call
finish backfill` into `releases/v0.18.0` (2026-03-28), #8832 `[BugFix][Patch] Backport GLM tool call streaming
fixes` and #8833 `… forced tool choice none-content handling` into `main` (2026-04-30) — so they are the
fork's GLM tool-call backports, not a fix for this exact guard. Ship them attributed, with the upstream fact
beside them (`unguarded through v0.14.1`, guard from `v0.15.0rc0`), and the workaround: patch the two guards
into `/vllm-workspace/vllm/vllm/tool_parsers/glm4_moe_tool_parser.py` inside the image (lost on rebuild),
avoid zero-argument tools meanwhile, or move to an image carrying either fix.

Distinguish from §4: `glm47` *drops a zero-argument call while streaming* with no error; this one logs a
`TypeError` in the non-streaming `extract_tool_calls` and returns the text as `content`. Both start from a
zero-argument call on a GLM parser, which is why a report about "the tool call never fires, the model just
prints markup" is settled by the **log**, not by the wire format.

## 4b. A whitespace-only `content` delta before the first tool call (legacy `qwen3_xml`)

Here nothing is missing — the client receives an **extra** `content` delta of `"\n\n"` (the separator
between `</think>` and `<tool_call>`) in the same turn as the tool call, so its Markdown renderer prints
a blank block, and any client that treats a non-null `content` as "the answer started" mishandles the
turn. Report shape (#15911, `v0.23.0` image, `--tool-call-parser qwen3_xml --enable-auto-tool-choice` +
`--reasoning-parser`, agent client), nothing crashes. Owner: the parser that ships inside the release
image, not the serving layer.

Verified at upstream `vllm-project/vllm` tag `v0.23.0` in
`vllm/tool_parsers/qwen3xml_tool_parser.py` (this legacy parser is upstream's; vllm-ascend carries no copy):

- `:175` `if self.text_content_buffer and self.tool_call_index == 0:` — no `.strip()` test. When the
  `</think>`→`<tool_call>` separator arrives as its own chunk (no complete element yet), the buffer holds
  only whitespace, is wrapped in `DeltaMessage(content=…)`, emitted, cleared and **returned**, so the
  tool-call bytes are only parsed on the next chunk.
- `:258-266` — `_process_complete_xml_elements`'s first-`<tool_call>` flush is guarded by
  `) and self.text_content_buffer:` only, so the same separator escapes when `<tool_call>` lands in the
  same chunk.
- `:954` — the *between-tool-calls* flush already tests `if self.text_content_buffer.strip():`. That
  in-file inconsistency is the strongest argument that the whitespace emission is unintended, and it is
  why the fix is two `… and self.text_content_buffer.strip()` guards that still emit real prose such as
  `"I will query this first:\n\n"`.

Version map for the parser **name**, which is stable while the class behind it is not — check the mapping
at your own tag (`vllm/tool_parsers/__init__.py`, the `"qwen3_xml"` entry) instead of assuming the
implementation:

| tag | `--tool-call-parser qwen3_xml` resolves to | legacy file |
| --- | --- | --- |
| `v0.23.0`, `v0.23.0rc2` | `Qwen3XMLToolParser` (`__init__.py:161-163`) | present, has the whitespace flush |
| `v0.23.1rc0` | **`Qwen3CoderToolParser`** — a different format, not a fix | removed |
| `v0.24.0rc1`, `v0.24.0` | `Qwen3EngineToolParser` (`vllm/parser/qwen3.py` engine) | removed |

The removal commit is `2043258dec` "[Frontend] Support strict mode for tool calling (#45003)". So no
`0.23.x` patch release can carry the fix — the levers are:

1. patch the two lines inside the image (`/vllm-workspace/vllm/vllm/tool_parsers/qwen3xml_tool_parser.py`),
   cost: reapply after every image rebuild/pull; the cleanest is a `sed`/`patch` step in the deployment
   wrapper, and it needs the image to have writable source (it does — the Dockerfile clones vLLM there);
2. strip whitespace-only `content` deltas client-side (drop a delta whose `content.strip() == ""` while
   `tool_calls` is still empty), cost: a client change, and it would also drop a legitimately empty delta;
3. move to an image whose paired vLLM is ≥ `v0.24.0rc1` (engine parser), cost: a full stack move.

**Check ownership before writing a `vllm_ascend/` line.** `git grep -ln qwen3xml <ascend-ref> -- vllm_ascend`
returns nothing at `v0.23.0` (only a `.github/workflows/scripts/upstream_config.yaml` test entry), and the
file is absent from `v0.23.1rc0` and `v0.24.0` (replaced by the parser engine, `vllm/parser/qwen3.py` +
`rust/src/tool-parser/`). So the answer is "the image's `VLLM_TAG` file, patch it in the image or move to
a release whose parser engine ships" — not "the Ascend plugin emits it". Generalise: for any mis-shaped
`content`/`arguments` delta, run that grep **before** listing `vllm_ascend/patch/platform`; an empty
result routes the report to the paired vLLM tag. Keep "a wrong response body is a `patch/platform` report
first" as the *first* check, not the only one.

Routing: a report that quotes a path under `/vllm-workspace/vllm/…` is quoting the **upstream** file at the
tag the image's `Dockerfile` passes as `VLLM_TAG` (`ARG VLLM_TAG=` at `v0.23.0:Dockerfile:42`); read it in
the `vllm` clone, and expect `git grep <symbol> <vllm-ascend-ref> -- vllm_ascend` to be empty — that is
not a negative result. `scripts/open_answer.sh "<repo>#N — symptom" v0.23.0 vllm` prints that clone's ref
facts (`ref_snapshot.sh` skips the plugin sections when `vllm_ascend/` is absent at the ref).

Thread state: the maintainer (`QwertyJack`, 2026-09-11) agreed with the analysis and the proposed fix,
said the parser is bundled from upstream so the fix belongs in the vLLM source or an image-build backport,
and that otherwise it will be documented as a known limitation; no fix landed in any `0.23.x`. Limits: not
reproduced; the whitespace only becomes a content delta when the separator arrives as its own streaming
chunk (the report's own boundary), so a stream where `</think>\n\n<tool_call>` arrives in one chunk takes
the `:258` path instead — both are the same defect.

## 4c. The model's markup leaks into the client (`</think>`, `<tool_call>`, `<arg_key>`)

Report shape (#12398, `vllm-ascend 0.18.0` + vLLM `0.18.0`, GLM-5.1 W4A8, transformers 5.2.0, A2,
streaming chat completions **with** tools): the body is fine, HTTP 200, but the streamed text contains
`</think>` before a tool call, whole `<tool_call>Bash<arg_key>command</arg_key><arg_value>…</arg_value>`
blocks inside the text stream, and — worst — that markup inside the tool *name*; stray `<button>`/`<summary>`
tokens can appear in argument text as well. Nothing crashes.

`<tool_call>Name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` is **GLM's native tool-call
text**, the format `glm4_moe_tool_parser` parses and `_extract_tool_name_from_region` splits at `\n` or
`<arg_key>` (§2, §4). Seeing it in `content` means the parser was not configured, not matched to the
model, or not reached for that request; it is not evidence about the model's output. The documented GLM
pair is `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.

**Question 1 is where the markup is, and it decides the owner:**

| observed | owner | why |
| --- | --- | --- |
| `</think>`/`<thinking>` (and the thinking text itself) in `content` | **no `--reasoning-parser`, or one for another model family** | a reasoning parser *moves* the thinking out of `content`; without it the CoT and its closing boundary are ordinary output. Not a leak the server makes (`reasoning-parser-output.md`) |
| a raw `<tool_call>…</arg_value></tool_call>` block in `content` | the **tool parser did not consume the region** | `--tool-call-parser` absent, `--enable-auto-tool-choice` missing, or a parser whose region regex is for another dialect; the raw model text then streams as content |
| markup inside `function.name` (e.g. `Grep-n</arg_key><arg_value>true</arg_value>`) | the parser's **name extraction** region | the streaming path takes the name up to `\n`/`<arg_key>` (§4's parent `_extract_tool_name_from_region`), so a region without that boundary, or a parser applied to a template that uses a different separator, yields a name with markup in it |
| `[tool_call:NAME]{json}` in the reporter's excerpts | **their client** | that is a client's rendering of a structured `tool_calls` delta; do not model it as a server text format (one recorded run spent ~5.5 K characters of reasoning on that ambiguity). Ask for the raw body or `scripts/merge_tool_deltas.py` output instead |

The first two checks are free and cost no clone read: the server's own start command (which parsers are
set) and one `stream: false` request on the same prompt. Only then list the tag's
`vllm_ascend/patch/platform` files with their `Why`/`How` blocks.

**So this class is a config answer, and the body must carry the serve command.** #12398 pastes no
`vllm serve` line at all — no `--tool-call-parser`, no `--reasoning-parser`, no `--enable-auto-tool-choice` —
and its thread has only "Plz see see this issue" forwards. Nothing in the clone can be localised without
that line: the parsers live upstream at the image's `VLLM_TAG`, and the *selection* is the flag. Reply with
the two flags, `"stream": false` as the discriminator (a non-streaming body that is clean proves the
streaming parser path), and the one line you need from them.

**The vllm-ascend side to check before blaming upstream** (at `v0.18.0`): the patch registry
(`vllm_ascend/patch/platform/__init__.py:32-33`) imports `patch_glm_tool_call_parser.py` and
`patch_tool_choice_none_content.py`. `patch_glm_tool_call_parser.py` is not a small fix-up — it
**replaces `OpenAIServingChat`'s streaming loop**: `:1083`
`OpenAIServingChat.chat_completion_stream_generator = _patched_chat_completion_stream_generator`, with
four helper rebinds at `:1073-1076` and a patched `extract_tool_calls_streaming` at `:819`. Inside that
generator the branch at `:245` (`if tool_choice_auto or reasoning_parser:`) decides which stream handling
runs, and `:399-437` routes each delta to `reasoning` or `content` (`reasoning_parser.
extract_reasoning_streaming(...)`, `is_reasoning_end(...)`) — so on a vLLM `v0.18.0` image this file, not
upstream's class, owns whether `</think>` ends up in `content`. On a GLM deployment of that era it therefore outranks
upstream's parser for exactly this symptom (`git grep -ln <parser name> <ref> -- vllm_ascend` decides whether the plugin even carries it —
§4b's rule). A GLM name/tag leak at `v0.18.0` therefore starts with the patch
file's stream branch, not with `vllm/tool_parsers/glm4_moe_tool_parser.py`; a non-GLM model leaves the
upstream parser as the owner.

Status: #12398 is open with only routing pings. A leak that survives a correct parser pair is a buffer
question, not a configuration one — that is §3/§4b territory, and the owner table above is the
configuration/ownership answer the report is owed.

Falsifier for the config reading: a `stream: false` request that still comes back with markup inside
`content` or inside `function.name` — then the parser mangled a complete response and the owner is the
parser code at that tag, not the flag.

## 5. Workarounds and what they cost

| workaround | cost |
| --- | --- |
| merge deltas by `index` in the client, keep the first `id`/`name` seen, read `arguments` from the accumulated fragments | none server-side; needs a client change, and does not help a client that validates every chunk as a whole call |
| `stream:false` for the tool-call path (agent/tool clients) | loses token streaming and partial-render UX; the cheapest answer for an agent client, and it removes the whole class — #7682's reporter confirmed the non-streaming body is complete |
| carry a tag whose `patch/platform` contains the chunk split, or upgrade once the paired vLLM contains the serving split | check the paired release by ancestry, not by PR state (§7); a version bump also moves CANN/torch_npu |
| patch `_terminal_tool_arg_choice` to treat `arguments is not None` as terminal, so the empty fragment is split off | local monkey-patch only; not upstream, and it changes the chunk count clients see |

## 6. Do not route this to spec decode, quantisation, the tokenizer or the NPU

A shape mismatch on a *delta* has no mechanism through the graph, the tokenizer or the quant scheme: the
arguments are empty because the diff between expected and streamed text was empty, not because tokens
went missing. Recorded wrong turns on this class: "MTP returns garbage tokens", "incremental
detokenisation of spec tokens is out of order", "w4a8 corrupts the XML so the parser yields no args",
"`--async-scheduling` / cudagraph". MTP changes only *what the parser has streamed by the terminal step*
(in #11788 it is a frequency factor, `hypothesis`; a report that says "MTP 3 breaks tool calls, MTP 1 is
fine" is that frequency factor, not a corrupted draft token — §7's #7580). One `merge_tool_deltas.py` run
on the raw stream settles which mode applies before any of those is worth entertaining.

The one shape where the spec-token count *is* the differential: the reporter shows
`num_speculative_tokens: 3` failing and `1` working **and** the stream's missing bytes are the terminal
ones (`"}`), as in #7682/#7580 below (§7). There the discriminator is still free — `stream: false` on the
same prompt plus the delta merge: a complete non-streaming body means parser/backfill (§3), a truncated one
moves the report to the spec-decode skip path and the answer becomes the version lever, not another parser
read.

Also do not read the parser alone, and do not call a missing `id` an upstream vLLM regression before
listing `vllm_ascend/patch/platform` at the reported tag.

## 7. Worked cases, status and recheck

- **vllm-ascend#11788** (`vllm==0.20.2` + `vllm-ascend==0.20.2rc1`, GLM-5.1-w4a8, 910C, PD over
  `MooncakeHybridConnector`, `--tool-call-parser glm47`, MTP 3): probabilistic `tool_calls[]` entry with
  no `id`, `arguments == ""`, `finish_reason: "tool_calls"`. **Verified downstream regression, fixed one
  release line later**: the `patch_glm_tool_call_streaming.py` above emits the args-only terminal delta
  and the split refuses an empty fragment; fix `c4a26bbce` (#10361 "Preserve GLM streaming tool call
  name", 2026-06-12) early-returns the parser's own delta when `remaining_call == ""` and inherits
  id/type/name otherwise; ancestry `v0.20.2rc1` no, `v0.21.0rc1`/`v0.22.1rc1` yes; the patch is deleted
  by `b2dc73890` (#11453) and absent at `v0.26.0rc2`. Thread has no mechanism (a bot note cites a
  comment that is not there — do not spend calls looking for it).
- **vllm-ascend#7682** (image `v0.17.0rc1-a3-glm5-v2`, GLM-5-w4a8, A3 16 cards, TP16 + EP,
  `--tool-call-parser glm47`, MTP 3, `--async-scheduling`): ~28 empty deltas then one chunk with
  `arguments: "}"`; non-streaming returns the full `{"todos": "- [x] …"}`. **Verified tag window, open
  thread**: no GLM tool-call patch exists at `v0.17.0rc1`, so the fix is later (#7710 at `v0.18.0rc1`,
  #8832 at `v0.19.1rc1`, #9410/#9787/#9788 and #9901/#9902 from `v0.20.2rc1`/`v0.21.0rc1`); mapping
  those 2026-06 patches onto this report is `hypothesis` (their PR bodies name #8327/#8178, not #7682).
  Not verified: whether the reporter's tool schema declared `todos` as `"string"`.
- **vllm-ascend#7580** (`vllm-ascend 0.17.0rc1`, A3, GLM-5 大EP, `--speculative-config
  '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}'`; `num_speculative_tokens: 1` serves fine):
  the agent client aborts with `JSON Parse error: Unterminated string` — the **same tag window and class
  as #7682**, reached from the client side. At `v0.17.0rc1` `vllm_ascend/patch` carries **no** GLM tool-call
  patch at all, the parser streams nothing for an argument it does not classify as `"string"`, and the
  finish backfill emits only the remaining tail, so the client concatenates bare fragments — the thread's
  own dump shows `arguments: "agent/K8s"` then `"YamlAgent.java"`, a stray `stop_reason` in one delta,
  then `arguments: ""` with `finish_reason: "tool_calls"`, i.e. the closing `"}` never arrives. MTP 3 is
  a frequency factor here, not the owner (§6). The fix chain starts at `v0.18.0rc1` (#7710, `f83cb0e6d`;
  `patch/platform/patch_glm_tool_call_parser.py`, "compute the finish backfill from the argument bytes
  actually emitted"). Levers: `num_speculative_tokens: 1` (reporter-confirmed), `stream: false`, or a tag
  carrying #7710+. **Mechanism `hypothesis`** — the report never names `--tool-call-parser`, so that flag
  plus one non-streaming run is the whole ask.
- **vllm-ascend#15911** (image `v0.23.0`, `--tool-call-parser qwen3_xml`, agent client): a
  whitespace-only `content` delta between thinking and the first tool call — mechanism, version map,
  levers and thread state in §4b. **Verified upstream-bundled legacy parser, no 0.23.x fix possible.**
- Upstream tracking: vllm#44098 (closed) names both halves — final remaining-argument chunks re-emitting
  metadata, and a terminal argument chunk combined with `finish_reason`; PR vllm#44099 is **closed
  unmerged** (main's GLM parser rewrite #45915 supersedes the metadata half). "Closed" is not "shipped":
  run `git -C /workspace/repos/vllm-project/vllm tag --contains <sha>` before promising a version.
  On the ascend side #8327 (open; the protocol-packing half, with a code walk in its comments and a note
  that #8178 fixed only the duplicate-name part) and #7682 (open) are the trackers.
- Sibling defects in the same patch family (all present at `v0.20.2rc1`):

  | patch file | what it says upstream gets wrong | cites |
  | --- | --- | --- |
  | `patch_glm_tool_call_streaming.py` | final remaining-argument chunks repeat tool-call metadata; terminal argument bytes share a chunk with `finish_reason="tool_calls"` | vLLM #44098/#44099, ascend #8327/#8178 |
  | `patch_glm47_tool_call_parser.py` | complete inline zero-argument calls dropped in streaming only | vLLM #44326/#44327 |
  | `patch_deepseek_v4_tool_call_parser.py` | streaming does not guarantee incremental `arguments` for long payloads | upstream main behaviour |
  | `patch_minimax_m2_tool_call_parser.py` | arguments buffered until `</invoke>` | vLLM #40253/#40298 |
  | `patch_tool_choice_none_content.py` | forced tool choice with `content=None` (reasoning ate the text, e.g. `max_tokens`) asserts | vLLM #40148, ascend #8400 |
  | `patch_minimax_usage_accounting.py` | `completion_tokens_details.reasoning_tokens` missing | vLLM #37955 |

- Recheck conditions: the patch files move and are deleted (#11453), and the upstream serving line
  numbers drift. Re-run the three-layer listing at the reported ref; the `patch/__init__.py` `Why`/`How`
  blocks, the UTs under `tests/ut/patch/platform/`, and the fix commits above are the durable part.
