# A whitespace-only `content` delta just before the first tool call

Scope: `stream: true`, the reasoning text and the `tool_calls` are **correct**, and the
only complaint is one extra assistant-content chunk holding nothing but a separator —
`"\n\n"`, `"\t\n\n\t\n"` — emitted between the end of the thinking block and the first
`<tool_call>`. The client renders a blank Markdown block ("an empty card before the
tool call"). Nothing is truncated, nothing is missing, no JSON error.

```text
reasoning_content: "…"
content: "\n\n"          <-- the extra delta
tool_calls: [ {get_weather…} ]
```

Worked case **#15911**, vllm `v0.23.0` (`0fc695fc6d1d`) + vllm-ascend
`v0.23.0` (`5cb98caaadef`), official `quay.io/ascend/vllm-ascend:v0.23.0` image,
`--tool-call-parser qwen3_xml`, Qwen3 XML tool calling.

Not this page: damage **inside** `arguments` (unbalanced quotes, `{}`, suffix-only)
→ the rest of this topic; a missing call → `no-tool-call-in-raw-output.md`.

## Where the delta comes from (three flush sites, one guarded)

`vllm/tool_parsers/qwen3xml_tool_parser.py` at v0.23.0 has exactly three places that
emit `text_content_buffer` as `content`, and only the third is whitespace-aware:

| Site | Code | Guard |
|---|---|---|
| `:175-181` (`parse_single_streaming_chunks`, no complete element found) | `if self.text_content_buffer and self.tool_call_index == 0:` → emit, clear, `return text_delta` | none |
| `:260-266` (`_process_complete_xml_elements`, first `<tool_call>` / `<function name=`) | `) and self.text_content_buffer:` → emit, clear | none |
| `:954-956` (`_end_element`, text *between* tool calls) | `if self.text_content_buffer.strip():` | **`strip()`** |

That asymmetry is the whole answer: the same file already suppresses a
whitespace-only buffer at the between-calls site, so the reporter's one-line
`strip()` fix makes the three agree. Quote it as the consistency argument — it is
stronger than "the extra delta is ugly".

Reachability, source-verified rather than assumed: `_should_skip_element`
`:331-335` collects **any** non-empty text node into `text_content_buffer` while
`current_call_id is None`, and `_find_next_complete_element` `:345+` returns a bare
text run as its own "element" (`buffer[:tag_end]` when the next `<` precedes the next
`>`). So the `\n\n` after `<｜end▁of▁thinking｜>` reaches the buffer and is emitted at `:175` on
the first chunk that yields no complete element — before `<tool_call>` arrives.

Two properties of the proposed hunk worth stating in the review:

- **Not clearing on the suppressed path discards nothing.** The whitespace stays in
  front of the next text and ships merged with it, so a real lead-in
  (`"I will query this first:\n\n"`) is preserved intact — test that case, not only
  the empty one.
- **The one behaviour change to accept deliberately:** a response whose *tail* is
  whitespace-only now drops it, since `:175` is where it would have been emitted and
  `reset_streaming_state` (`:1140`) only clears. Desired on a tool turn, cosmetic on
  prose. Say so rather than letting a reviewer find it.

## Ownership and how a fix could ever be verified

- `qwen3_xml` is registered at `vllm/tool_parsers/__init__.py:161-163` at the pin, so
  "legacy" means superseded on main, not unreachable in the image.
- vllm-ascend ships no patch for this path: `rg --hidden -l 'qwen3xml|qwen3_xml'`
  over the downstream tree hits only `.github/workflows/scripts/upstream_config.yaml:594`,
  which lists upstream's `tests/tool_parsers/test_qwen3xml_tool_parser.py` as a test
  the plugin CI runs. (`--hidden` is not optional here: a recursive grep from the repo
  root skips `.github/`, so the plain form exits 1 on this very tree and reads as "no
  downstream mention" — `../evidence-workspace/case-layout.md` §*Search-command
  traps*.) So the file in `/vllm-workspace/vllm/` is upstream's byte-for-byte
  (`Dockerfile:42` `ARG VLLM_TAG=v0.23.0`, `:49 git clone --depth 1 -b $VLLM_TAG`),
  the fix belongs upstream (a `v0.23.x` patch release, not just main), and
  **downstream CI cannot gate it**.
- The tests cannot either: `tests/tool_parsers/test_qwen3xml_tool_parser.py:57-68`
  carries `xfail_streaming` entries — `"Qwen3XML streaming has systematic issues"`,
  including `test_surrounding_text` and the reconstruction case — so the very
  behaviour the fix must preserve has no passing coverage at the pin. A PR needs its
  own two cases (whitespace-only suppressed; non-whitespace lead-in kept) and should
  drop the `xfail` entry it makes pass.
- If a maintainer decides the image must carry it, the mechanism exists at that ref:
  `vllm_ascend/patch/platform/` already holds `patch_glm_tool_call_streaming.py`,
  `patch_glm47_tool_call_parser.py`, `patch_minimax_m2_tool_call_parser.py`,
  `patch_deepseek_v4_tool_call_parser.py`, `patch_tool_choice_none_content.py`.
- **Answer the ownership verdict before the ladder, because that is the literal
  question** ("fix/backport it into the release image, or document it as a known
  limitation?"): the file is upstream's byte-for-byte, so a downstream backport is not
  an action the project has — documented limitation + the rung-1 client suppression is
  the proportionate answer and a patch shim is the exception. Do not promise the
  backport, and do not read "downstream cannot gate it" as "nobody owns it": the
  release notes row (rung 3) is the downstream artefact that owns it.

## Ladder

1. **Client-side, no rebuild:** drop `content` deltas that are entirely whitespace
   before rendering. That is the whole user-visible defect.
2. Patch upstream's two sites with `strip()` (+ the tests above); do not "fix" the
   third — it is already right, and making it unconditional would change
   between-call text handling.
3. If the answer is "documented known limitation", it has a concrete home:
   `docs/source/user_guide/release_notes.md:134` is the `### Known Issues` list under
   `## v0.23.0` (fourteen rows at the pin, **no** tool-parser row yet), and each row
   names its issue; put the client-side workaround in the same row.
4. Only if the release image is a release blocker: a downstream patch file under
   `vllm_ascend/patch/platform/`, with the upstream PR linked from it.

## Recheck conditions

- Parser names and line numbers move: `rg -n 'text_content_buffer' <parser>` first;
  the *three-site asymmetry* is the claim, the numbers are only its location.
- A newer upstream parser that rewrites the buffer from a whole-response
  reconstruction (no per-chunk flush) retires the site list — check that the parser
  in the reporter's tree still has `parse_single_streaming_chunks`.
- Do not carry this page to a `stream: false` report: the non-streaming path has no
  deltas at all.
