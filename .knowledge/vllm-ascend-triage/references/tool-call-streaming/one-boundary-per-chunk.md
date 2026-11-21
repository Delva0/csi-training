# Two tool-call markers in one SSE chunk (`</tool_call><tool_call>`) — MTP / spec decode

## When this page applies

A title like "[Bug]: GLM-4.7 Tool Call failed with MTP" and a body one line long:
after enabling MTP "the engine returns two tokens in a single response, causing the
simultaneous appearance of `</tool_call><tool_call>` within one chunk, which leads to a
failure in parsing the tool_call". Or the same observation phrased as several markers /
two complete `<invoke>` regions / `</arg_value><arg_key>` arriving in one chunk while
the client's tool-call parser dies. Inherit `index.md`.

Two things this page keeps apart, because they have different owners:

1. **"Two tokens in one chunk" is the transport contract, not the defect.** One SSE
   chunk is one *engine step* — `delta_text = output.text`
   (`vllm/entrypoints/openai/serving_chat.py:746` at vllm v0.13.0 `72506c9`) — and under
   spec decode a step accepts 1…`num_speculative_tokens+1` tokens, so a chunk carrying
   both the closing and the opening marker of adjacent calls is **normal**. The suspect
   is always the consumer that assumed one marker per chunk.
2. **Which field carried the markers decides which consumer.** Ask for the raw SSE and
   read the chunk: `content` and `arguments` have different owners below.

Do **not** route this to `mtp-block-verify-acceptance.md`: that page owns exact-form
*damage* with `num_speculative_tokens >= 3` (truncated JSON, a duplicated span, an early
stop token), not two clean boundaries, and no sampler or kernel read is owed here.

## The discriminator: `content` or `arguments`

`vllm/tool_parsers/glm4_moe_tool_parser.py` at vllm v0.13.0 (`Glm4MoeModelToolParser`,
registered as `glm45` — `vllm/tool_parsers/__init__.py:41`):

- markers in **`content`** → the parser never saw the start token. The only branch that
  returns the buffer verbatim is `start_idx == -1` (`:149-153`), and it suppresses the
  text once a call has already been emitted (`if self.current_tool_id > 0: cur_text =
  ""`, `:151`). Two candidates, both cheap: the parser's `tool_call_start_token`
  (`<tool_call>`) does not match the model's own template (a GLM-4.7 report on a
  `glm45`-era stack), or the text the reporter pasted was never routed through a tool
  parser at all (`--tool-call-parser` absent, or a client that logs the raw delta).
- markers in **`arguments`** → this parser cannot do that: it re-serializes parsed
  values, `arguments=json.dumps(arg_dct)` (`extract_tool_calls`, `:59-135`). Raw markers
  inside `arguments` mean the client concatenated raw text, or a different generator on
  the path supplied it — look for the downstream GLM patches at newer refs (below)
  before blaming upstream.

That single question is what the recorded case's body never answered, so it is rung 1
for the next reader, ahead of any parser mechanism.

## What the parser's per-chunk loop actually costs

Same file, streaming extractor `:136-200`:

| Step | Line | Consequence |
|---|---|---|
| `self._buffer += delta_text`, then `end_idx = cur_text.find(self.tool_call_end_token)` — the **first** end marker only | `:146,155` | one boundary consumed per call |
| `extract_tool_calls(cur_text[:end_idx+len(end)])`, then `tool_call = extracted_tool_calls.tool_calls[0]` | `:167-173` | only the first region of the slice is used |
| `self.current_tool_id += 1`; `self._buffer = cur_text[end_idx+len(end):]` | `:195-196` | the next call is **parked**, emitted only if a later step arrives |
| `if len(extracted_tool_calls.tool_calls) == 0: return None` | `:171` | `self._buffer` is left as it was, so nothing is emitted and the same region is retried on later chunks. The detail regex needs a **name line** before the body (`func_detail_regex = r"<tool_call>([^\n]*)\n(.*)</tool_call>"`, `:43-45`), so if the checkpoint's own template puts JSON straight after `<tool_call>` the whole call is dropped with one `Failed to extract any tool calls.` warning — a hypothesis to check against that model's chat template (`return_token_ids: true` shows the raw text), not a fact about GLM-4.7 |

Nothing flushes `_buffer` when generation ends: the finish-step helper only inspects a
delta that **already carries** a call
(`_should_check_for_unstreamed_tool_arg_tokens`, `serving_chat.py:1759-1782`), so a second
call parked on the last step is silently dropped — that is the shape to name when the
report says one call is *missing* while the first is complete, not when it says both
markers arrived together.

## Version correspondence (do this before any mechanism)

- `glm47` is registered in **neither** pinned tree — vllm v0.13.0's registry carries
  `glm45` → `glm4_moe_tool_parser` only (`tool_parsers/__init__.py:41`). A GLM-4.7
  report on vllm-ascend `0.11.rc2` is therefore at least one parser generation behind
  the model, so the upgrade rung is legitimate (keep the mechanism below it).
- At vllm-ascend **v0.13.0rc1** (`1b5d5abf`) the streaming path is pure upstream:
  `vllm_ascend/patch/platform/` holds eight files and none is GLM, and
  `rg -ln -i 'glm' vllm_ascend/` hits only `quantization/quant_config.py` and
  `ops/triton/rope.py`. At v0.18.0 the family is one file
  (`platform/patch_glm_tool_call_parser.py`) and at v0.23.0 two
  (`patch_glm_tool_call_streaming.py`, `patch_glm47_tool_call_parser.py`) — name drift,
  so re-grep `ls vllm_ascend/patch/platform/` at the reporter's ref (index rule 6)
  before writing "upstream owns it".

## Ladder

1. One raw SSE plus `"return_token_ids": true` on the same body: both markers inside the
   same chunk's `token_ids` ⇒ the engine emitted them in one step (normal) and the
   consumer owns it; markers split across chunks ⇒ the parser's parked-buffer loop.
2. `num_speculative_tokens: 1`: fewer accepted tokens per step, so the symptom rate must
   fall — a cheap confirmation that the chunk size is the trigger, not a cure.
3. `stream: false`: the non-streaming extractor reads whole regions, so a correct answer
   there retires the model and weights lanes.
4. The serve line's `--tool-call-parser` name against the pin's registry (above) — a
   name the pin lacks is a coverage statement, in one line, not a cause.

## Recorded case

**#5596** (vllm-ascend `0.11.rc2`, GLM-4.7 + MTP, one-line body, no serve command, no
log, no SSE). Pin: vllm-ascend `v0.13.0rc1` / vllm `v0.13.0`. The recorded run
source-verified the step-per-chunk premise (`serving_chat.py:746`), the one-boundary loop
and the missing finish-time flush, established that no downstream GLM patch exists at the
pin and that `glm47` is unregistered, and ranked the parser first with the upgrade second.
What no local evidence could settle — and what the answer had to ask for — is which field
carried the markers and what the serve line passed; keep that ask (with
`return_token_ids: true`) in rung 1 instead of deriving a mechanism from a one-line body.
