# First request after start answers with empty content

Scope: `POST /v1/chat/completions` returns HTTP 200, `choices[0].message.content`
is `""`, `usage.completion_tokens` is 1, `finish_reason` is `"stop"`, and the next
identical request answers normally. Typically MTP plus a graph-mode decode on a
hybrid GDN model (Qwen3.5/Qwen3.6 family), 2 cards, `--speculative_config` with
`num_speculative_tokens ≥ 1`.

## Step 1 — place the response in the right lane before any mechanism

| Response fields | Lane |
|---|---|
| `finish_reason: "length"`, `completion_tokens == max_tokens` | length budget; `../accuracy-benchmarks/harness-vs-engine-attribution.md` |
| `finish_reason: "stop"`, `completion_tokens` in the tens, text sane | early EOS / degeneration; `../accuracy-benchmarks/early-eos-short-output.md` |
| `content: null`, text inside `reasoning` | output routing; `../accuracy-benchmarks/reasoning-output-routing.md` |
| `finish_reason: "stop"`, `completion_tokens: 1`, `content: ""` | **this page**: the engine emitted one token that detokenizes to nothing (a stop token stripped by `skip_special_tokens`, or a token whose text is empty). `enable_thinking`, the chat template and the weights are not implicated by this shape |
| `content: ""` with `completion_tokens` 0 and no `finish_reason` | the request never ran: check the scheduler/queue and the frontend, not the model |

`usage.prompt_tokens` also tells you whether the template produced a plausible
prompt (a Qwen chat turn with an empty system block is ~14-20 tokens; a much
smaller number means the request, not the decode, is the story).

## Step 2 — reproduce before naming a one-shot mechanism

Recorded #9355 is the calibration row: the investigation built a "server-lifetime
first decode is corrupted, subsequent requests get zeroed state" mechanism from a
single report, and the reporter's later public comment shows a **different request
body** (another prompt, `temperature` 0.7 → 0.9, same `enable_thinking: false`,
same v0.18.0 image) answers normally. So request content and sampling were never
ruled out, and the one-shot-state story stayed a hypothesis.

Three launches settle the trigger class:

1. restart the server, send the **identical** body twice → does request 1 fail and
   request 2 succeed? (order-triggered) or do both fail? (content-triggered)
2. restart again, send the reporter's *modified* body first, then the original →
   is the failure attached to the request or to its position?
3. restart, send one throwaway request (`max_tokens: 1`) at start, then the real
   request → if the artifact moves into the throwaway request, a warm-up request is
   both the diagnosis and the mitigation to hand back.

## Step 3 — read the token that was actually emitted

The response above carried `"token_ids": null`, which is a choice, not a limit:
ask for `"return_token_ids": true` (or `logprobs`), or start the server with
`--enable-log-outputs` (`../observability-and-logs/request-output-logging-flags.md`),
and look at the single id. An EOS/`<|im_end|>` id means the model chose to stop and
the question is why its first logits were degenerate; an id that decodes to
whitespace or a junk id means the sampled token itself was wrong. Those two point
at different code, and no amount of source reading substitutes for the id.

## Step 4 — check that the first decode really is the MTP graph step

DEBUG serve logs make this one line each:

```
Running batch with cudagraph_mode: NONE, BatchDescriptor(num_tokens=14, …)   # prefill, eager
Running batch with cudagraph_mode: FULL, BatchDescriptor(num_tokens=3, num_reqs=1, …)  # 1 + 2 drafts
Replaying aclgraph
```

`num_tokens == 1 + num_speculative_tokens` on the first decode status tells you the
first generated token is produced on the speculative + full-graph path, which is
where rows 1-3 below point. It does **not** prove the graph is at fault — it only
means the eager-prefill/eager-decode ruling-out has to be done by ablation.

## Step 5 — do not chase the padded-rows theory

`vllm_ascend/worker/model_runner_v1.py` fills padding with zeros in two places and
this looks like "padding rows write into the first request's cache":

- `:2098-2099` `slot_mapping[num_tokens:num_tokens_padded].fill_(-1)` and
  `blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)`;
- `:2171-2174` (v0.18.0) under full cudagraph + `GDNAttentionMetadataBuilder` with
  `num_prefills == 0`, `spec_state_indices_tensor[num_spec_decodes:].fill_(0)`,
  with the comment that the Triton op cannot take `-1` in FullGraph mode.

Refuted, cheaply: upstream `vllm/v1/core/block_pool.py:173-175` reserves the first
free block as `null_block` ("To represent a placeholder block with block_id=0",
`free_block_queue.popleft()`), so index 0 is never a live request's block and a
fill with 0 writes into the null block. Padding rows therefore cannot corrupt a
real request's KV or GDN state. This costs one grep; it saved a whole theory in the
recorded #9355 run.

## Step 6 — ablation rows

| Row | Change | What it proves |
|---|---|---|
| 1 | warm-up request at start, then the real one | order/warm-up class; also the mitigation to hand back |
| 2 | delete `--speculative_config` (MTP off) | whether the MTP draft/verify path owns the first decode (`acceptance-collapse.md`) |
| 3 | `--enforce-eager` | graph replay vs eager on the first decode |
| 4 | `--no-enable-prefix-caching`, then `--max-num-seqs 1` | hybrid `mamba_block_size` geometry, and single-request isolation |
| 5 | same body at `temperature: 0` and with `ignore_eos: true` | sampling-sensitive degeneration versus a deterministic wrong token |

## Recorded run: #9355 (vllm-ascend v0.18.0, Qwen3.6-27B bf16, 2×910B3)

Reporter facts: `curl` with `enable_thinking: false`, `temperature: 0.7`,
`max_tokens: 1024`, prompt `你是谁？`; first request →
`{"content": "", "finish_reason": "stop", "usage": {"prompt_tokens": 14,
"completion_tokens": 1}}`; the next request, same script, answers normally. Run in
the debug log: prefill `cudagraph_mode: NONE num_tokens=14`, then
`cudagraph_mode: FULL num_tokens=3 num_reqs=1` + `Replaying aclgraph`. Version from
the follow-up: `quay.io/ascend/vllm-ascend:v0.18.0`, i.e. the pinned tree.

The recorded run (13 assistant turns, 17 tool calls, its answer cut mid-way)
produced a correct lane classification and a good ablation list, correctly refuted the
null-block theory, and got the MTP/graph first-decode suspicion right — but its
lead framing was the unestablished one-shot-state mechanism, and the answer text
was cut before the actions block. Written in the shape of `index.md` step 2, the
same conclusion costs about half the turns: classify from the two response fields,
ask for the restart reproduction and the token id, hand over rows 1-5.

Open items for this case: what the single emitted token id was; whether a restart
reproduces it with the identical body; whether the modified body also fails when
sent first; the worker-side log for that first step.

Gaps: nothing in the snapshot links Qwen3.6-27B bf16 + MTP + `FULL_DECODE_ONLY` to
a fixed defect at v0.18.0; the recorded release notes carry neighbouring fixes only
(Qwen3-Next-MTP batching accuracy #4932, "MTP prefill misclassified as decode"
#6835, `max_cudagraph_capture_size = max_concurrency × (1 + num_speculative_tokens)`
for the large-EP capture overflow, `docs/source/user_guide/release_notes.md:456,
:252, :267`). Cite them as family precedent, not as this bug's fix.
