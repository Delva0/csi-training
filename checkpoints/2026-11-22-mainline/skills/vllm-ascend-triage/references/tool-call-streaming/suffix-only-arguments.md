# Suffix-only arguments: the head of a streamed tool call never arrives

## Signature (recognise it from the paste, before any code read)

The client's assembled `arguments` is **only the closing punctuation of the
object** — no `{`, no key, no value — while `stream: false` on the same body
returns the complete string:

```
non-stream : {"todos": "- [x] 分析项目结构…\n- [-] 优化代码架构…"}
streaming  : "}                     <- 0 `{`, 1 `}`, 1 quote
json.loads -> Unterminated string starting at: line 1 column 1 (char 0)
```

Three more properties of the stream, each cheap to read off the paste:

- **every other chunk is `delta: {}`** — no `content`, no `tool_calls` (the audit
  script counts them: `chunks carrying NEITHER content nor tool_calls: N of M`);
- `finish_reason: "tool_calls"` and (vLLM-only) numeric `stop_reason` sit on the
  **same terminal chunk** that carries the fragment;
- `content` never arrives at all, so the usual "is the text damaged too?"
  comparison has nothing to compare.

This is the extreme form of the **delta-bookkeeping** family. The mild form is
`quote-gap` (`{"hvd": Pura900"}` — braces balance, one inserted character lost)
in `arg-type-gate-unquoted-value.md`; here the loss is the whole head, so the
giveaway is arithmetic, not a missing quote. `scripts/sse-toolcall-audit.sh`
labels this outcome `head-lost` (more closing than opening braces) and no longer
folds it into `quote-gap`.

State this in the answer as one line of arithmetic, because the reporter can run
it with no restart:

```
stream:false args = A   (message.tool_calls[0].function.arguments, same body)
streaming  args = S
S == A[-len(S):]        -> strict suffix: only the tail was ever emitted
```

## Owner: the layer that keeps the length, not the model

Both candidate sites are the same defect class — a **length prefix trusted
against a string that was re-rendered or never emitted**:

- **parser** — `_compute_args_diff` returns `args_so_far[len(streamed_args_for_tool[i]):]`
  and stores the new full string (vllm v0.20.2 `glm4_moe_tool_parser.py:418-427`;
  unchanged in kind at v0.23.0 `:409-417`). Any step that re-rendered characters
  it had already "streamed" loses them for ever; in the extreme, a step whose
  rendered `args_so_far` grew long while its delta reached no client leaves the
  stored length ahead of the bytes.
- **frontend finish-step reconciliation** — `remaining_call =
  expected_call.replace(actual_call, "", 1)` with
  `actual_call = tool_parser.streamed_args_for_tool[index][:-latest_delta_len]`
  (vllm v0.13.0 `entrypoints/openai/serving_chat.py:1147-1200`, guarded by
  `_should_check_for_unstreamed_tool_arg_tokens` at `:1759-1790`). When the
  stored string is already everything but the last characters, the terminal
  chunk carries exactly those characters — the observed `"}`.

The **silent chunks are not evidence of a fault**: a GLM parser buffers while it
walks the model's native XML and emits the arguments once. At vllm v0.13.0
`tool_parsers/glm4_moe_tool_parser.py:136-200`, every step inside `<tool_call>`
returns `DeltaMessage(content=cur_text[:start_idx])` (`:199-200`, empty), and the
`</tool_call>` step builds the whole `arguments` string, writes
`self.streamed_args_for_tool[self.current_tool_id] = tool_call.function.arguments`
(`:178-180`) and returns it in one delta. So "28 empty chunks, then one argument chunk" is
that parser working as designed; what is *not* by design is that the final chunk
carried a 2-character suffix of the string it had just written to bookkeeping.

Downstream ownership still has to be excluded, not assumed: grep
`vllm_ascend/patch/` by *model family* (`glm`, `deepseek`, `minimax`) as in
`index.md` rule 6, and quote the branch condition that excludes each hit.

## Before arguing the mechanism: reconcile the ref with the serve line

`--tool-call-parser <name>` is a **registration witness**. One grep:

```bash
rg -n '"glm47"' <case>/vllm-project/vllm/vllm/tool_parsers/__init__.py
```

At vllm v0.13.0 that registry holds `glm45` only — `glm4_moe_tool_parser.py` is
reachable as `glm45`, and neither the tool nor the reasoning registry has
`glm47` (`vllm/reasoning/__init__.py:35`). vLLM validates the parser name at
startup, so a deployment that runs `--tool-call-parser glm47` **cannot** be
serving that tree: the export predates the reported runtime.

What to do with that, and what not to:

- Say it as the coverage gap in the first answer (rule 5 / rule 10): "the pinned
  vllm v0.13.0 does not register `glm47`; the runtime is newer, so the parser
  code I cite is the nearest documented shape, not the reported one."
- Argue from the nearest documented ref (`arg-type-gate-unquoted-value.md`
  §Version correspondence: v0.20.2 incremental parser, v0.23.0 gate fix) and
  keep the mechanism labelled as source-verified *there*.
- **Do not** derive "unreachable in the exported tree, therefore this layer is
  not the owner". That inference is unfalsifiable when the tree is older than the
  runtime, and it is what the recorded #7682 run spent ~8 of its 16 turns on.

## Checks, cheapest first (all reporter-side)

1. **Suffix arithmetic** above, from one `stream: false` request with the same
   body. No restart; decides "tail only" vs "the client lost the earlier chunks".
2. **`return_token_ids: true`** on the streaming body — if the per-chunk
   `token_ids`/text are complete while the arguments are 2 characters, the loss
   is post-sampling in the serving path (`index.md` rule 4).
3. **Is `content` ever non-empty on the failing request?** Empty throughout (the
   recorded case) means the text path is unobservable from the chat stream; use
   the `/v1/completions` `stream: true` vs `stream: false` pair of
   `arg-type-gate-unquoted-value.md` §Procedure step 3 to exonerate the engine.
   Non-empty *and* damaged moves the owner to `detokenizer-text-delta.md` (one
   grep for `resetting decode stream`).
4. Only then the serve flags: `--speculative-config num_speculative_tokens`
   (`>= 3` block verify, `mtp-block-verify-acceptance.md`) and
   `--async-scheduling`. The paste cannot separate them, and the arithmetic of
   step 1 already answers the question the reporter asked.

The request body is a required artefact: without `tools`, the failing argument's
declared type is unknown and the `quote-gap` variant cannot be excluded. Ask for
the file the curl posted (`-d @test-todolist.json` in #7682) in the same breath as
the answer.

## Recorded case #7682 (GLM-5-w4a8, vllm-ascend v0.17.0rc1, A3 16 cards)

Reported: `--tool-call-parser glm47 --reasoning-parser glm45`, `deepseek_mtp`
`num_speculative_tokens: 3`, `--async-scheduling`, `FULL_DECODE_ONLY`, chunked
prefill, `--max-num-seqs 8`; the non-streaming call is complete and the streaming
call carries `"arguments":"\"}"`. The reporter restated the same A/B in a
follow-up minutes after filing (2026-03-26T08:29Z,
`#issuecomment-4132650891`: 非流式 `{"todos": "- [x] …"}` = 完整正常, 流式 =
`"\"}"` → 严重截断/错误), so this paste is the **head-lost** form and the
non-streaming string is the suffix-test target. Its contrast with #8122 is where the
cut sits, not failing-versus-healthy: #8122 keeps the head and the punctuation and
loses digits inside the last value, which no brace/quote balance can see.

How the recorded run went: the shape was called correctly on turn 3 ("this is
not the model — the streamed call emits only its last two characters"), then ~8
turns went into the missing `glm47` registry and the parser/sampler walk, and the
session ended `status: output_truncated` at 16 384 output tokens with **no answer
text ever sent** — the last block was reasoning about which file to open next.
The verdict was available before any source read; the delivery gates (`SKILL.md`
rule 1) were the failure, not the evidence.
