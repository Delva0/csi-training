# `arguments` is exactly `{}`: the GLM parser had no `<arg_key>` pair to print

## Signature

`tool_calls[0].function.name` is a clean function name and
`tool_calls[0].function.arguments` is the two-character string `{}`
(`arguments: "{}"`, or the client shows an empty object). Nothing is truncated:
both braces arrived, `finish_reason` is often `tool_calls`, and the call is
otherwise well formed. Reported for GLM-5.2-w8a8 at vllm-ascend v0.23.0rc1 with
`run_in_terminal` (#14731), after the same model family's *truncated*-argument
reports (#12261, #14656) and the neighbouring *suffix-only* report (#7682).

Read this page when the shape is `{}` on a GLM-family serve. The other
tool-call shapes stay with `index.md` rule 1: a short/malformed/unbalanced
`arguments` is a delta or detokenizer question, and `{}` is none of them —
`{}` is *valid* JSON with zero keys.

## Mechanism (verified at vllm v0.23.0 `0fc695f`)

`--tool-call-parser glm47` selects `glm47_moe_tool_parser.py`, a 40-line
subclass that only replaces the two regexes; the argument building is
`glm4_moe_tool_parser.py`:

- Name and args come from `<arg_key>k</arg_key><arg_value>v</arg_value>` pairs
  found in the `<tool_call>` region (`func_arg_regex`). The name is the region
  text before the first newline or `<arg_key>`
  (`_extract_tool_name_from_region`, `:307-320`).
- **Non-streaming** (`extract_tool_calls`, `:180-235`): pairs are collected into
  `arg_dct`, then `arguments=json.dumps(arg_dct, ensure_ascii=False)` (`:214`).
  Zero pairs ⇒ `"{}"`.
- **Streaming** (`_build_args_json_so_far`, `:322-407`):
  `if not parts: return "{}" if is_complete else ""` (`:401-402`). A **complete**
  region with zero pairs ⇒ `arguments: "{}"`, sent as one delta by
  `_compute_args_diff` (`:409-426`).
- The glm47 region regex is `<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>`
  (DOTALL). Anything else between the name and `</tool_call>` — e.g. a JSON body
  `{"command": "ls"}` — makes the match fail: non-streaming then logs
  `Failed to parse tool call details from: …` and returns **no tool call at
  all**, while streaming still yields the first line as the name and `{}` for the
  body. That asymmetry is the discriminator below.

So `{}` means: *the region was complete, the name parsed, and not one
`<arg_key>/<arg_value>` pair matched.* It is not evidence of lost bytes, and it
says nothing yet about quantization, MTP or the sampler.

**Downstream patch 7b closes the zero-argument case.**
`vllm_ascend/patch/platform/patch_glm47_tool_call_parser.py` wraps
`_extract_tool_call_regions` and, for a *complete* region with no `<arg_key>` and
no newline, appends `\n` so the name parses. It is imported only under
`vllm_version_is("0.23.0")` (`patch/platform/__init__.py:34-36`). On a vllm
0.23.0 host, therefore, a genuinely zero-argument `<tool_call>f</tool_call>`
now surfaces as name + `{}` although upstream would have dropped it — the same
string as a format mismatch. Patch 7a
(`patch_glm_tool_call_streaming.py`, imported unconditionally, `:32`) is the
unrelated terminal-chunk split backported from upstream #44098/#44099
(downstream #8327/#8178) and only matters when the *tail* is wrong.

## One-request discriminator

Re-send the identical body with `stream: false` and read the same two fields:

| `stream: true` | `stream: false` | reading |
|---|---|---|
| name + `{}` | name + complete `arguments` | the payload was lost in the stream: parser delta arithmetic or the incremental detokenizer — `streamed-arguments-audit.md`, `detokenizer-text-delta.md` |
| name + `{}` | **no `tool_calls`** (or the warning `Failed to parse tool call details`) | the model's argument text is not `<arg_key>/<arg_value>` XML: an argument-syntax mismatch, owner = the model's emission or a missing parser for that format |
| name + `{}` | name + `{}` | the model emitted no argument pairs (zero-argument call): generation / model-choice lane — take a rate, do not read the parser further |
| name + `{}` | HTTP 400 / schema error | tool-schema gate: the declared `required` arguments were never produced |

The raw model text settles all four in one grep:
`VLLM_LOGGING_LEVEL=DEBUG` prints `model_output: %s` at
`glm4_moe_tool_parser.py:186`, and `return_token_ids: true` on the same request
carries the emitted ids (`index.md` rule 4). Ask for one of them rather than for
"the logs".

Do not send the reporter into `patch_glm_tool_call_streaming.py` first: that
patch fixes a *tail* that arrives in the same chunk as `finish_reason`
(the tail-lost row, #7580), and `{}` has no tail to lose.

## Precedents

- #7682 — GLM-5-w4a8, `--tool-call-parser glm47 --reasoning-parser glm45
  --enable-auto-tool-choice` + MTP3: non-streaming `arguments` complete, streaming
  `delta: {}` chunks plus an assembled `"}` beside `finish_reason: "tool_calls"`.
  Same flags as #14731, different row of the table above: the assembled string is
  a *suffix* of the non-streaming arguments (head lost, not zero pairs), so its
  owner is `suffix-only-arguments.md`.
- #8130 / #8478 — "GLM5 MTP tool_calls empty string" is a *known issue* in the
  v0.19.0rc1 conclusion: an MTP axis exists, so a rate at
  `num_speculative_tokens` 1 vs 3 belongs in the action ladder, after the table.
- #14656 — v0.23.0 GLM-5.2 PD, leading character lost and a fragment repeated:
  the streamed text delta, one grep for `resetting decode stream`.
- #14731 — the `{}` report this page was written from: the project's own GLM-5.2
  weekly e2e config serves with `--tool-call-parser glm47`
  (`tests/e2e/weekly/multi_node/internal_dp/config/GLM-5.2-W4A8C8-A3_3.5k_1.5k_0_50.yaml:36,68`
  at v0.23.0 `5cb98ca`), so the parser choice is the documented one and the
  parser is not "wrong for the model".

## Gaps that keep the answer honest

The recorded report carried no serve command, no `finish_reason`, no raw SSE and
no raw model output — so it could not pick a row of the table, only name them.
Say that, and ask for the `model_output` line: one paste decides. GLM-5.2's chat
template is not in the exported trees, so the `<arg_key>` format itself is
upstream's documented one, not something the snapshot can confirm for that
checkpoint.
