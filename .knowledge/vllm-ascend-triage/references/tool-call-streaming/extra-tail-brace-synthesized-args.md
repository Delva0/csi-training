# An extra closing brace at the tail of a *synthesized* arguments string

## When this page applies

An assembled tool call whose braces are **balanced up to one or more extra closing
`}` at the very end** — `"arguments": "{\"question\": \"…\", \"type\": 1, \"limit\": 8,
\"similarity\": 0.5}}"` (one `{`, two `}`) — with the head and every key present, one
clean call beside it in the same response, and the client failing when the history is
resubmitted ("以上下文信息再提交给模型时报错"). The `--tool-call-parser` in the command is
one that **synthesizes** the JSON: `qwen3_xml`, `qwen3_coder`, `glm47`, `minimax_m2`,
`hermes`. Recorded #16001: Qwen3.8-27B, 910B2, image `v0.23.0rc1`, `--data-parallel-size 2
--tensor-parallel-size 2`, `--tool-call-parser qwen3_xml`, `--speculative-config
'{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}'`.

Not this topic, leave with the link: only closing punctuation arrived (no `{`, no key)
→ `suffix-only-arguments.md`; quotes odd with balanced braces →
`arg-type-gate-unquoted-value.md`; `arguments` is `{}` → `empty-arguments-object.md`;
no `tool_calls` at all → `no-tool-call-in-raw-output.md`.

## 1. Resolve the parser name before reading any implementation

At vllm v0.23.0 `0fc695f` the same parsers exist **twice**: Python under
`vllm/tool_parsers/<name>_tool_parser.py` and a Rust crate under
`rust/src/tool-parser/src/` (`json/{qwen,hermes,llama,mistral,phi4mini,internlm2}.rs`,
`qwen_coder.rs`, `deepseek_*`, `glm_xml/`, `kimi_k2.rs`, …). The Rust copies are reached
only through the separate Rust frontend (`VLLM_USE_RUST_FRONTEND` /
`VLLM_RUST_FRONTEND_PATH`, `vllm/envs.py:143,550-559`), **not** through
`--tool-call-parser`; that map is `vllm/tool_parsers/__init__.py` (`"qwen3_xml":
("qwen3xml_tool_parser", "Qwen3XMLToolParser")`, `:161-164`). One grep there decides
which file you are auditing. The recorded #16001 session read
`rust/src/tool-parser/src/json/qwen.rs` and `rust/src/tool-parser/src/utils.rs` (the
generic JSON-scan state machine plus its chunked-scan tests) for four calls, then found
the 1300-line Python `qwen3xml_tool_parser.py` was the running implementation and spent
the rest of a 16 384-token output budget inside it.

## 2. Downstream carries no patch for this family

`rg -l 'tool_parser' <vllm-ascend>/vllm_ascend/patch/` at v0.23.0rc1 returns only
`patch_deepseek_v4_tool_call_parser.py`, `patch_minimax_m2_tool_call_parser.py`,
`patch_glm47_tool_call_parser.py`. A `qwen3_xml` / `qwen3_coder` / `hermes` arguments
string is produced by upstream vLLM: say so, and do not spend rungs on Ascend flags,
ops or device checks.

## 3. Two owners, both source-verified at the pin

Because the parser **writes** the JSON, the extra brace need not be in the model's
tokens at all. Both owners live in `vllm/tool_parsers/qwen3xml_tool_parser.py` at vllm
v0.23.0:

**(a) a close brace emitted twice (parser lane).** `}` is emitted in `_end_element`'s
function branch only when `self.parameters` is non-empty (`:902-915`; `{}` when empty),
and that branch sets `current_function_open = False` (`:929`) **without clearing
`self.current_function_name`**, while `self.parameters` is cleared only by
`_reset_xml_parser_after_tool_call` (`:1132-1133`) or the next `<tool_call>` start
(`:631`). Three guards can therefore reach the emission for a call that is already
closed: the `</function>`-in-chunk fallback, whose `has_function_close` test inspects
only the deltas of the *current* chunk, so a `}` emitted before this chunk reads as
"missing" (`:105-137`); and the new-`<tool_call>` guard
`if self.current_function_open or self.current_function_name:` (`:270-283`), true for a
name that is merely still set. (The `</tool_call>` guard at `:934-938` is the legitimate
close for a function that was genuinely never closed — `current_function_open` is
already `False` after both paths above, so it cannot add a third brace.) Both extra-`}`
paths index the delta as `self.tool_call_index - 1`, i.e. the call that is closing or
the one before the call that just started — which is exactly why one of two parallel
calls is damaged and the other is clean, and why the damage sits on the earlier index
when a third call begins. Reachability was **not** proven in the recorded session (no
raw SSE capture): report it as a candidate, not the cause.

**(b) a stray character inside the last numeric parameter rides through unquoted
(model lane).** `_convert_param_value` degenerates to the raw string when the declared
JSON-Schema type is a number and `float(text)` raises (`:1078-1093`, one
`logger.warning … degenerating to string` line), and `_convert_for_json_streaming`
returns a `str` **verbatim, unquoted** for a non-string type (`:1105-1113`). So a model
that emitted `<parameter=similarity>0.5}</parameter>` yields `"similarity": 0.5}}`.
The reporter's `tools` schema decides whether this path is open: without a declared
`type: number`, the value is quoted and the brace lands inside the string instead.

## 4. The checks, cheapest first

1. **`stream: false` on the same body** (plus `return_token_ids: true` on the streamed
   one). The non-streaming path parses the whole text once through `extract_tool_calls`;
   a single `}` there moves the owner to the model's own text / decoding, two `}` keeps
   it in the synthesized-close lane.
2. **Drop `--tool-call-parser`** (and `--enable-auto-tool-choice`): the raw
   `<tool_call><function=…><parameter=…>` text then appears in `content`, so whether the
   model itself wrote the extra `}` before `</parameter>` is visible in one request.
   `#14781` shows that this model family's own output is that XML
   (`<function=add_numbers><parameter=a>2</parameter>` lines).
3. **Parser A/B**: `--tool-call-parser qwen3_coder` reads the same
   `<function=`/`<parameter=` markers with a regex, not the expat state machine
   (`vllm/tool_parsers/qwen3coder_tool_parser.py:56-77`). The extra `}` disappearing
   localizes it to the qwen3_xml state machine with no code change.
4. **Sampling / spec-decode last, not first.** At v0.23.0rc1
   `using_block_verify = max_spec_len >= 3 and bool(get_ascend_config().
   rejection_sampler_config.enable_block_verify)` (`vllm_ascend/sample/rejection_sampler.py:400`)
   and that key defaults `False` (`vllm_ascend/ascend_config.py:766`), so
   `num_speculative_tokens: 3` alone selects the strict path
   (`mtp-block-verify-acceptance.md`); a duplicated token is also the least likely to
   land exactly on the closing `}`.

## 5. Precedents, each with a different owner

- `#14781` — same model (Qwen3.8-27B, A3): the **whole** `<tool_call><function=…>` XML
  inside `arguments`, i.e. no parser ran at all (missing/incorrect
  `--tool-call-parser`), not a stray brace.
- `#10758` (v0.20.2rc1, minimax-m2.5): a `</parameter>` appended at the end of the
  arguments — same family, "the parser appends a tail token the model never wrote",
  different parser.
- `#7989` (DeepSeek-V3.1-w4a8): `"arguments": "}!}!"`.
- `#8521` (Qwen3.6-35B, A2, 0.18.0rc1) — tool-call failure in the sibling Qwen3.x
  family; `#7759` — a non-empty `tool_calls` array in the last chunk, a protocol tail,
  not an arguments defect.

## 6. Version scope and what to send back

Every `path:line` above was verified at vllm v0.23.0 `0fc695f` / vllm-ascend
v0.23.0rc1 `f4a08bdd` — the reporter's own image. This file has been patched over time
(the fallback blocks exist because "some tags may not have been triggered"), so re-grep
the call sites on another tag and say plainly that a fix cannot be seen from a
one-commit snapshot.

Send back: the raw SSE capture (`curl -N`) of one failing response including the
`finish_reason`/`stop_reason` chunks, the same body with `stream: false`, one run with
the parser disabled, and the `tools` schema entry for the parameter that carries the
extra brace.
