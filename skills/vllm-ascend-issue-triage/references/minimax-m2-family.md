# MiniMax-M2 family: the patch surface behind M2/M2.5/M2.7 reports

MiniMax-M2 reports (garbled output, wrong tool-call text, missing `reasoning`) are usually
decided inside `vllm_ascend/patch/**/patch_minimax_*`, not in the upstream model file the
report names. List the model's hooks at the reported ref before reading model code:

```sh
git ls-tree -r --name-only v0.20.2rc1 | grep -i minimax
```

At `v0.20.2rc1` (`367b8e62d`, 2026-06-03; upstream vLLM `v0.20.2`) that gives
`patch/worker/patch_minimax_m2.py` (MoE forward, attention init + k_norm sharding,
`load_weights`), `patch/worker/patch_minimax_m2_linear_attn.py`,
`patch/platform/patch_minimax_m2_tool_call_parser.py`,
`patch/platform/patch_minimax_m2_config.py`,
`patch/platform/patch_minimax_usage_accounting.py`, plus docs, e2e configs and unit tests.
`vllm_ascend/patch/__init__.py` documents each hook's why/how/future plan — read the entry
instead of inferring intent from the diff.

The hooks are release-specific. On `main` (2026-11-01) `_patched_moe_forward` no longer
exists: `d901dd421` (#10906) trimmed upstreamed hooks, `17f690569` (#11384) replaced the
parser backports, and `f61a694c0`/`2c870a615` removed the fp8-dequant and fp32-gate patches.
The file itself is still there — 54 lines whose only definition is `_patch_forward` (`:29`), with
no `_patched_moe_forward` and no `maybe_all_reduce` — so `git show main:<path>` succeeds while a
symbol grep prints nothing; that is a trimmed hook, not a missing file.
A `main` grep therefore neither confirms nor refutes v0.20.2rc1 behaviour.

## Garbled output under plain TP: double TP all-reduce (code-verified; cause candidate)

Chain at `v0.20.2rc1` / vLLM `v0.20.2`; all five sites are needed for the claim:

| site | statement |
| --- | --- |
| `vllm_ascend/patch/worker/patch_minimax_m2.py:65-67` | `final_hidden_states = self.experts(...)`; `if self.tp_size > 1:` → `self.experts.maybe_all_reduce_tensor_model_parallel(...)` |
| `vllm/model_executor/layers/fused_moe/layer.py:1545-1555` | `FusedMoE.forward` → `self.runner.forward(...)` |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:651` | `result = self._maybe_reduce_final_output(result, og_hidden_dim)`; it all-reduces when `not is_sequence_parallel and (tp_size > 1 or ep_size > 1) and not self._fused_output_is_reduced` (`:385-390`) |
| `vllm_ascend/ops/fused_moe/fused_moe.py:271-283` | overrides `_fused_output_is_reduced`: True for `ALLTOALL`/`MC2`/`FUSED_MC2`, or `ALLGATHER` with `flash_comm_v1_enabled` |
| `vllm_ascend/ops/register_custom_ops.py:148-156` | `maybe_all_reduce_tensor_model_parallel` returns the input unchanged for the same comm types/flashcomm, else calls `tensor_model_parallel_all_reduce` |

So in the default state — `ALLGATHER` without `VLLM_ASCEND_ENABLE_FLASHCOMM1` — both layers
reduce the same routed output. An all-reduce sums one partial per rank, so reducing an
already-reduced tensor multiplies it by the TP size (×8 in the TP=8 report) rather than
failing: the MoE output enters the next layer far out of scale, which is a plausible
garbage-output mechanism. Under expert parallel (`ALLTOALL`), fused MC2, or FLASHCOMM1
exactly one reduce runs, and each of those switches flips the same predicate.

The correct guard already exists in a sibling file at the same tag:
`vllm_ascend/models/deepseek_v4.py:377-383` reduces only
`elif self.tp_size > 1 and fused_moe_out_is_tuple` — a legacy tuple output that the upstream
`MoERunner` has not reduced. The MiniMax patch has no `isinstance(fused_moe_out, tuple)`
check, which is precisely what PR #10443 ("[BugFix][Model] Fix MiniMax-M2 double TP
all-reduce on v0.20+", base `main`) adds. That PR was still open/`merged: false` on
2026-11-01, and issue #10466 (M2.7-w8a8-QuaRot, TP=8, `--enforce-eager`, no EP/FLASHCOMM1)
is still open: the mechanism is code-verified, the causal link to the reporter's garbage is
inferred from workaround convergence, not measured.

Workaround to give: the v0.20.2rc1 tutorial recipe, which sets both
`export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` and `--enable-expert-parallel` in every
MiniMax-M2.5 serve line (`docs/source/tutorials/models/MiniMax-M2.5.md:144-156`), and whose
FAQ (`:449-451`) answers "output is garbled in EP mode" with "keep
`--enable-expert-parallel` and `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`". The thread's earlier
suggestion `VLLM_ASCEND_ENABLE_FUSED_MC2=1` reaches the same predicate. Confirm or refute
with one run: log which of the two reduce sites executes (comm type + `tp_size`), or compare
the same prompt with FLASHCOMM1 on/off.

## Streaming tool calls append `</parameter>` (code-verified at v0.20.2rc1)

`patch/platform/patch_minimax_m2_tool_call_parser.py` backports incremental argument
streaming for `--tool-call-parser minimax_m2`. The leak is in the partial-value helpers:

- `_serialize_partial_param_value(..., is_complete=False)` ends with
  `return json.dumps(value, ensure_ascii=False)[:-1]` (`:266`) — an unterminated JSON
  string for a still-open `<parameter>` value;
- `_build_partial_arguments` emits that value for the open parameter (`:300-315`); while
  the closing `</parameter>` has not fully arrived, the tag characters received so far are
  part of the streamed string;
- once the parameter closes, the rebuilt `args_json` is shorter than the previously
  streamed `sent_args`, so `args_json.startswith(sent_args)` (`:441`) fails and
  `extract_intermediate_diff` (`:444`) streams the leftover tag text to the client. The
  final non-streamed parse is correct, so a single-shot request looks fine.

Only `extract_tool_calls_streaming` is patched (`:511`); the non-streaming
`extract_tool_calls` is upstream's. "Do not stream tool-call requests" is therefore a
plausible workaround by code shape, not a runtime-verified one.

Fix status: PR #10817 ("[BugFix][v0.20.2rc] Avoid streaming partial MiniMax parameter tags",
base `releases/v0.20.2rc`) removes the partial-value emission, and the reporter reported the
symptom resolved after applying it on site. A second, different failure followed —
`ValidationError: DeltaMessage.tool_calls Input should be a valid list [input_value=None]` —
which the reporter patched with PR #10377 and a maintainer then called incorrect; keep the
two apart, #10377 is not the `</parameter>` fix.

Both PRs were open/`merged: false` on 2026-11-01, and no commit on `origin/releases/v0.20.2rc`
after `v0.20.2rc1` touches that file, so no rc image carries the change; on `main` the file
itself was replaced by #11384. A reporter saying the PRs were 合入 and the symptom went away
describes their on-site build — check `state`/`merged` and
`git log <tag>..origin/releases/<ver> -- <path>` before calling it an upstream or image-level
fix. Issue #10758 is closed.

## Tool calls rejected on 0.18 (`invalid tool parameters`): the 0.17 -> 0.18 streaming rewrite

The `minimax_m2` tool parser and both reasoning parsers predate the parser engine. At vLLM
`v0.17.1` and `v0.18.0`, `vllm/tool_parsers/minimax_m2_tool_parser.py` and
`vllm/reasoning/minimax_m2_reasoning_parser.py` both exist, and the latter already defines
`MiniMaxM2ReasoningParser` and `MiniMaxM2AppendThinkReasoningParser` (`:22`/`:81` at `v0.18.0`;
`tests/reasoning/test_minimax_m2_append_reasoning_parser.py` is in the same tree). The
`v0.24.0rc1` boundary in the `</think>` section below is only the move to the declarative engine —
do not read it as "the parser combination needs 0.24", and do not answer a 0.17/0.18 report with
"that parser did not exist yet".

What changed between `v0.17.1` and `v0.18.0` is how `MinimaxM2ToolParser` streams
(`git diff v0.17.1 v0.18.0 -- vllm/tool_parsers/minimax_m2_tool_parser.py`). 0.17.1 carried the
per-parameter state machine (`_reset_streaming_state`, `_convert_param_value`, `json_started`,
`streamed_args_for_tool`, incremental `extract_intermediate_diff`). 0.18.0 replaces it with
buffer-until-complete-invoke: `_extract_delta_tool_calls` (`:293-339`) emits **one** `DeltaToolCall`
carrying the function name and the complete `arguments` JSON as soon as a whole
`<invoke>...</invoke>` block exists, and `extract_tool_calls_streaming` (`:395-420`) resets
`current_tool_index`/`streamed_args_for_tool` whenever a `<minimax:tool_call>` start token arrives.
`_parse_single_invoke` and the non-streaming path are unchanged (the old
`if param_value.startswith("\n")` trim that disappeared was already dead code: `param_value =
param_match.group(2).strip()` runs first, `:275`).

For a "0.17 works, 0.18 sends invalid tool parameters" report:

- The thread's pointer to upstream PR #36138 ("Grammar was ignored when reasoning ended within
  speculated tokens", closed, `merged: false`) cannot be the whole mechanism on a serve line
  without `--speculative-config`: `StructuredOutputManager.should_advance`
  (`vllm/v1/structured_output/__init__.py:302-337` at `v0.18.0`, called from
  `v1/core/sched/scheduler.py:1406/1658/1687`) returns False for the step in which reasoning ends
  and True from the following step, so without speculation only the reasoning-end marker itself
  escapes the grammar. It also only bites when a grammar was attached at all, and at `v0.18.0`
  chat completions get one solely from `response_format` (`json_object`/`json_schema`/
  `structural_tag`, `vllm/entrypoints/openai/chat_completion/protocol.py:457-486`; the Responses
  path only rewrites a `structural_tag` that already exists,
  `vllm/entrypoints/openai/responses/serving.py:468-481`). No `tool_choice` value — `auto`,
  `required` or named — synthesizes a tool grammar there, so an ordinary tool-call request is
  ungrammared and a malformed `arguments` is the parser's or the model's output, not xgrammar's.
- The verified 0.17 -> 0.18 delta to inspect is the streaming rewrite above — what the client
  receives per delta changes shape (nothing until an invoke closes, then the full `arguments`
  string), and an OpenAI-compatible client or Anthropic->OpenAI router may validate the accumulated
  partial JSON.
- Unfinished at the recording: no captured response body from that deployment was replayed through
  both tags' parsers, so neither the rewrite nor the reasoning-end/grammar hole is established as
  the cause (the latter also needs a `response_format` in the request). Next check: replay one
  streamed tool-call response through `v0.17.1` and `v0.18.0`
  `MinimaxM2ToolParser` on CPU (no NPU needed) and diff name/arguments/id per delta.

## `reasoning` empty with `minimax_m2_append_think`

Documented as expected in the same tutorial FAQ (`:453-455`): the parser keeps
`<think>...</think>` inside `content`; for `/v1/responses` reasoning semantics use
`--reasoning-parser minimax_m2`. Do not diagnose this as a parser loading failure.

## Closing `</think>` lost only when a tool call is parsed (vLLM `v0.24.0rc1` and later)

Symptom: `--reasoning-parser minimax_m2_append_think --tool-call-parser minimax_m2
--enable-auto-tool-choice`, `tool_choice: "auto"`. With `tools` in the request the `content` keeps
`<think>…` but loses the closing `</think>`, streaming and non-streaming alike, while the extracted
tool call and its numeric arguments stay correct; the same prompt without `tools` keeps both
delimiters, and `reasoning` is `null` in both (that part is the append contract above).

- Read it as an upstream parser-composition defect, not a plugin or NPU one. `vllm-ascend`
  `v0.26.0rc1` pairs upstream `VLLM_TAG=v0.26.0` (`Dockerfile.a3:57`) and no longer carries a
  MiniMax tool-parser patch — at `v0.26.0rc1` the only parser-adjacent hit in
  `git ls-tree -r --name-only v0.26.0rc1 | grep -i minimax` is
  `vllm_ascend/patch/platform/patch_minimax_m2_config.py` (model-type/config), and
  `patch_minimax_m2_tool_call_parser.py` is gone, so the parse is upstream's.
- Mechanism, two sites at vLLM `v0.26.0`: `MiniMaxM2AppendThinkReasoningParser`
  (`vllm/reasoning/minimax_m2_reasoning_parser.py:29-66`) puts `<think>` in front and returns
  everything as `content` (`extract_reasoning` `:63-66`, `extract_reasoning_streaming` `:50-61`,
  `extract_content_ids` returns all ids), so the tool engine receives the markup. Its table has
  `(ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT, ())`
  (`vllm/parser/minimax_m2.py:128-131`) — an empty event tuple consumes the marker and emits no
  text — and there is no `CONTENT`/`THINK_START` transition, which is exactly why only the closing
  tag disappears.
- The engine is the right layer to read, and the table is the wrong thing to patch blindly.
  `vllm/parser/abstract_parser.py` only sets `tool_parser.skip_reasoning_parsing = True` when
  `self._reasoning_parser is None` (`:156-166` at `v0.33.0rc2`); a reasoning parser that keeps
  markup in content gets no such flag, so the tool engine still treats `</think>` as a reasoning
  boundary. That attribute does not exist at all before `v0.30.0rc1` (`git grep -c
  skip_reasoning_parsing v0.26.0 -- vllm` is empty), so the composition is worse on `0.24`–`0.29`.
  Emitting `EventType.TEXT_CHUNK` at that transition also restores the tag (the reporter's patch),
  but it changes marker handling for every `minimax_m2` request — check the plain-reasoning path
  before recommending it.
- Applicability and fix status (checked 2026-11-01): the engine landed in `v0.24.0rc1` (#45701
  `f00e163f35`, #45915 `6c379b9e54`) — the parser classes themselves are older than that tag (see
  the 0.17/0.18 section above), so 0.24 is not when the combination became possible; whether the
  legacy path loses the same tag is not verified here, and the table below is the engine's — the
  transition is unchanged at `v0.24.0`, `v0.25.0`,
  `v0.26.0`, `v0.27.0`, `v0.30.0` and `v0.33.0rc2`, with no `keeps_reasoning_markup_in_content` in
  any of those trees — **no released vLLM tag fixes it**, so "upgrade vLLM" (to any tag, including
  the newest) is not the advice; only a local parser patch or the PRs below are. Upstream issue
  vllm-project/vllm#58486 is open; PR #58505 ("Keep closing
  `</think>` delimiter in MiniMax M2 append-think mode") is closed with `merged: false`; PR #58501
  is open with `merged: false` and its fix is the composition above plus a
  `tests/reasoning/` regression. Re-read `state`/`merged` before quoting either.
- Workarounds, labelled: `--reasoning-parser minimax_m2` avoids the combination and keeps thinking
  in the separate `reasoning` field (an API-contract change — append-think exists to keep it
  inline), and the transition patch above. Both were confirmed by CPU parser replays only; no NPU
  run is needed to reproduce or fix this, and the recorded thread says so explicitly.

## Boundary

Nothing here is a runtime observation of the reporter's machine. Numerics, ranks and the
comm type actually selected come from worker logs; the source shows which branch each
switch selects, not which one the report ran. Report state (open/merged) moves: re-read it
before quoting.
