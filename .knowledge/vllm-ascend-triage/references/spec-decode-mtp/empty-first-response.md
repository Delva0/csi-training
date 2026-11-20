# First request after start answers with empty content

Scope: `POST /v1/chat/completions` returns HTTP 200, `choices[0].message.content`
is `""`, `usage.completion_tokens` is 1, `finish_reason` is `"stop"`, and the next
identical request answers normally. Typically MTP plus a graph-mode decode on a
hybrid GDN model (Qwen3.5/Qwen3.6 family), 2 cards, `--speculative-config` with
`num_speculative_tokens ≥ 1`.

## Step 1 — place the response in the right lane before any mechanism

| Response fields | Lane |
|---|---|
| `finish_reason: "length"`, `completion_tokens == max_tokens` | length budget; `../accuracy-benchmarks/harness-vs-engine-attribution.md` |
| `finish_reason: "stop"`, `completion_tokens` in the tens, text sane | early EOS / degeneration; `../accuracy-benchmarks/early-eos-short-output.md` |
| `content: null`, text inside `reasoning` | output routing; `../accuracy-benchmarks/reasoning-output-routing.md` |
| `finish_reason: "stop"`, `completion_tokens: 1`, `content: ""` | **this page**: the engine emitted one token that detokenizes to nothing (a stop token stripped by `skip_special_tokens`, or a token whose text is empty). `enable_thinking`, the chat template and the weights are not implicated by this shape |
| `content: ""` with `completion_tokens` 0 and no `finish_reason` | the request never ran: check the scheduler/queue and the frontend, not the model |
| **offline `LLM.generate()`**, call 1 only, **no MTP**: a constant token id repeated to `finish_reason: "length"`, an immediate stop after `<think>`, or `AttributeError: 'NoneType' object has no attribute 'size'` through `compilation/acl_graph.py`, on a VL-capable architecture at TP≥2 with `enable_sleep_mode=True` | not this page (no draft model, and the answer is degenerate rather than blank): `../accuracy-benchmarks/first-generate-degenerate.md` — text-only `is_multimodal_model` drift, then the first-request sync/allocator rows |

`usage.prompt_tokens` also tells you whether the template produced a plausible
prompt (a Qwen chat turn with an empty system block is ~14-20 tokens; a much
smaller number means the request, not the decode, is the story).

Keep "no defect here" on the table as a deliverable, not a failure to diagnose. If
the one emitted id is the model's own EOS/`<|im_end|>` and a restart with the
identical body does not reproduce the blank answer, the honest verdict is a
sampling tail event at `temperature > 0` with `ignore_eos` unset, and the thing the
reporter can act on is `min_tokens: 1` — a per-request field
(`vllm/sampling_params.py:217` at v0.18.0) that bans the stop ids until one token
has been produced (`vllm/v1/sample/logits_processor/builtin.py:195-198`, where the
docstring at `:246` ranks `min_tokens` above `stop_token_ids` / EOS). Offer it next to the
mitigations; do not spend the answer hunting a kernel for a single sampled token.

## Step 2 — reproduce before naming a one-shot mechanism

Recorded #9355 is the calibration row: the first investigation built a
"server-lifetime first decode is corrupted, subsequent requests get zeroed state"
mechanism from a single report, and the reporter's later public comment
(2026-05-20) shows a **different request body** — prompt `你是谁？` →
`可以给我介绍一下杭州吗？`, `temperature` 0.7 → 0.9, same `enable_thinking: false`,
same v0.18.0 image — answering normally. So request content and sampling were
never ruled out, and the one-shot-state story stayed a hypothesis. Two cautions on
reading that comment: it moved two axes at once, and its URL is
`localhost:6006` while the failing script posted to `:6710`, so "the same process
with a different body" is not established either. Ask which of the two it was
before writing "content-triggered".

Three launches settle the trigger class:

1. restart the server, send the **identical** body twice → does request 1 fail and
   request 2 succeed? (order-triggered) or do both fail? (content-triggered)
2. restart again, send the reporter's *modified* body first, then the original →
   is the failure attached to the request or to its position?
3. restart, send one throwaway request (`max_tokens: 1`) at start, then the real
   request → if the artifact moves into the throwaway request, a warm-up request is
   both the diagnosis and the mitigation to hand back.

## Step 3 — diff the launcher against the family sibling tutorial (one `ls`)

Do this before any source walk. At v0.18.0 there is **no** Qwen3.6 page
(`rg -l "Qwen3\.6" docs/` returns only `docs/source/tutorials/hardwares/310p.md`),
so the reference point is the same-family hybrid-GDN recipe,
`docs/source/tutorials/models/Qwen3.5-27B.md:105-120`:

```
--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}'
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'     # and no cudagraph_capture_sizes
```

with its notes at `:139-141`: `FULL_DECODE_ONLY` recommended, and
`cudagraph_capture_sizes` left at the default ("Only in some scenarios is it
necessary to set this separately"). #9355's command deviates in four rows:

| Row | #9355 | Sibling tutorial | Why it matters |
|---|---|---|---|
| 1 | no `enforce_eager` in `--speculative-config` | `"enforce_eager": true` | keeps the **draft-side** graph path on; the single most useful ablation row (Step 7) |
| 2 | hand-written `cudagraph_capture_sizes: [3,6,…,96]` | none (default) | caps the captured range at 96 tokens; see the arithmetic below |
| 3 | `num_speculative_tokens: 2` | 3 | n=2 never enters block verify (Step 7), so the `max_spec_len >= 3` kernel is out of scope here |
| 4 | bf16 checkpoint, no `--quantization` | `w8a8-mtp` + `--quantization ascend` | the sibling is a **precedent, not a validated bf16 config**: never quote it as "the official command for your model" |

Capture-size arithmetic (vllm v0.18.0): the default is
`min(max_num_seqs × (1 + num_speculative_tokens) × 2, 512)` and then
`min(max_num_batched_tokens, …)` (`vllm/config/vllm.py:1337-1347`); a batch larger
than the largest listed size silently runs without a graph (`:1323`). #9355 lists 96
against `--max-num-seqs 64`, so decode batches above ~32 concurrent requests drop
out of graph mode. That is a throughput item, not this single-request bug — say so
in one line instead of leaving the list unexamined.

## Step 4 — read the token that was actually emitted

The response above carried `"token_ids": null`, which is a choice, not a limit:
ask for `"return_token_ids": true` (or `logprobs`), or start the server with
`--enable-log-outputs` (`../observability-and-logs/request-output-logging-flags.md`),
and look at the single id. An EOS/`<|im_end|>` id means the model chose to stop and
the question is why its first logits were degenerate; an id that decodes to
whitespace or a junk id means the sampled token itself was wrong. Those two point
at different code, and no amount of source reading substitutes for the id.

## Step 5 — check that the first decode really is the MTP graph step

DEBUG serve logs make this one line each:

```
Running batch with cudagraph_mode: NONE, BatchDescriptor(num_tokens=14, …)   # prefill, eager
Running batch with cudagraph_mode: FULL, BatchDescriptor(num_tokens=3, num_reqs=1, …)  # 1 + 2 drafts
Replaying aclgraph
```

`num_tokens == 1 + num_speculative_tokens` on the first decode status tells you the
first generated token is produced on the speculative + full-graph path, which is
where rows 3-5 below point. It does **not** prove the graph is at fault — it only
means the eager-prefill/eager-decode ruling-out has to be done by ablation.

Two version facts that cost turns when they are missing:

- `"method": "qwen3_next_mtp"` is rewritten to `"mtp"` with a deprecation warning
  (`vllm/config/speculative.py:354-358`, v0.18.0; a `qwen3_next` draft model maps to
  `qwen3_next_mtp` → `Qwen3NextMTP` at `:295-300`). Downstream `if self.method ==
  "mtp"` gates therefore do apply to such a command — do not spend a turn on "the
  method name is unusual".
- Line numbers drift between the pinned commit and the reporter's image: #9355's
  log prints `model_runner_v1.py:1213` where the pinned v0.18.0 tree prints that
  debug line at `:1214`. Quote a line number with that caveat rather than claiming
  the image and the pin are identical.

## Step 6 — two padding stories; only one of them is refuted

**(a) Scheduler-side padded rows — refuted.** `vllm_ascend/worker/model_runner_v1.py`
fills padding with zeros in two places and this looks like "padding rows write into
the first request's cache":

- `:2098-2099` `slot_mapping[num_tokens:num_tokens_padded].fill_(-1)` and
  `blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)`;
- `:2171-2174` (v0.18.0) under full cudagraph + `GDNAttentionMetadataBuilder` with
  `num_prefills == 0`, `spec_state_indices_tensor[num_spec_decodes:].fill_(0)`,
  with the comment that the Triton op cannot take `-1` in FullGraph mode.

Upstream `vllm/v1/core/block_pool.py:173-175` reserves the first free block as
`null_block` ("To represent a placeholder block with block_id=0",
`free_block_queue.popleft()`), so index 0 is never a live request's block and a fill
with 0 writes into the null block. Padding rows therefore cannot corrupt a real
request's KV or GDN state. This costs one grep; it saved a whole theory in the
recorded #9355 run.

**(b) The MTP proposer's padded drafter batch — live, and one key switches it.**
`vllm_ascend/spec_decode/eagle_proposer.py:136-142` (v0.18.0):

```python
self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
if self.method == "mtp":
    self.use_cuda_graph = (self.use_cuda_graph
                           and not self.use_async_scheduling
                           and not self.speculative_config.disable_padded_drafter_batch)
```

With `method: mtp` — what `qwen3_next_mtp` becomes — `disable_padded_drafter_batch:
false`, which is both the default (`vllm/config/speculative.py:109`, whose own
docstring talks only about EAGLE even though MTP is in `EagleModelTypes`,
`:49`) and #9355's value, is what **keeps the MTP proposer replaying aclgraph with
padded drafter batches**. So do not answer "your `disable_padded_drafter_batch` is
just the default, it is not an anomaly": the value selects the padded + graph draft
path, and the same line shows `"enforce_eager": true` in the spec config turns that
path off while leaving the target model's `FULL_DECODE_ONLY` graphs on.

## Step 7 — ablation rows

| Row | Change | What it proves |
|---|---|---|
| 1 | warm-up request at start, then the real one | order/warm-up class; also the mitigation to hand back |
| 2 | delete `--speculative-config` (MTP off) | whether the MTP draft/verify path owns the first decode (`acceptance-collapse.md`) |
| 3 | `"enforce_eager": true` **inside** `--speculative-config` | draft/MTP proposer eager while the target keeps its graphs (`eagle_proposer.py:136`) — the row that separates the two graph paths, and the sibling tutorial's own setting |
| 4 | `"disable_padded_drafter_batch": true` (with `method: mtp`) | same effect reached through the padded-drafter gate (`:137-142`), so padded drafter batches are split from "graph replay" |
| 5 | `--enforce-eager` (target and draft) | both graph paths off; run it only after rows 3-4 |
| 6 | `--no-enable-prefix-caching`, then `--max-num-seqs 1` | hybrid `mamba_block_size` geometry, and single-request isolation |
| 7 | same body at `temperature: 0`, and with `min_tokens: 1` | sampling-sensitive degeneration versus a deterministic wrong token; `min_tokens` is the client-side fix if the id was EOS |

A `num_speculative_tokens` of 1 or 2 never reaches the block-verify kernel —
`using_block_verify = max_spec_len >= 3 and draft_probs is not None`
(`vllm_ascend/sample/rejection_sampler.py:116` at v0.18.0; newer refs also AND
`rejection_sampler_config.enable_block_verify`, default `False` at v0.23.0rc1) — so
name that branch (`../tool-call-streaming/mtp-block-verify-acceptance.md`) as out of
scope instead of walking it.

## Recorded runs: #9355 (vllm-ascend v0.18.0, Qwen3.6-27B bf16, 2×910B3)

Reporter facts: `curl` with `enable_thinking: false`, `temperature: 0.7`,
`max_tokens: 1024`, prompt `你是谁？`; first request →
`{"content": "", "finish_reason": "stop", "usage": {"prompt_tokens": 14,
"completion_tokens": 1}}`; the next request, same script, answers normally. Run in
the debug log: prefill `cudagraph_mode: NONE num_tokens=14`, then
`cudagraph_mode: FULL num_tokens=3 num_reqs=1` + `Replaying aclgraph`. Serve flags
include `--tensor-parallel-size 2`, `--max-num-seqs 64`, `--no-enable-prefix-caching`,
`--reasoning-parser qwen3`, `--speculative-config
'{"method":"qwen3_next_mtp","num_speculative_tokens":2,"disable_padded_drafter_batch":false}'`
and an explicit `cudagraph_capture_sizes` list ending at 96. Version from the later
public comment: `quay.io/ascend/vllm-ascend:v0.18.0` (2026-06-02), i.e. the pinned
tree.

The first recorded run (13 assistant turns, 17 tool calls, its answer cut mid-way)
produced a correct lane classification and a good ablation list, correctly refuted
the null-block theory, and got the MTP/graph first-decode suspicion right — but its
lead framing was the unestablished one-shot-state mechanism, and the answer text
was cut before the actions block. Written in the shape of Steps 1-2, the same
conclusion costs about half the turns.

Two later runs on the same knowledge (18 and 31 messages, both delivered in full)
landed the classification, the refutation and the warm-up/token-id ask. What both
still missed, and what Steps 1-7 above now carry: neither answered the follow-up's
port/axes question, neither turned `disable_padded_drafter_batch: false` or
`enforce_eager` in the spec config into a discriminating row (the 31-message run
offered `enforce_eager` only as a copied workaround, and the 18-message run never
opened the sibling tutorial), and neither offered `min_tokens`.

Open items for this case: what the single emitted token id was; whether a restart
reproduces it with the identical body; whether the modified body also fails when
sent first **in the same process**; the worker-side log for that first step.

Gaps: nothing in the snapshot links Qwen3.6-27B bf16 + MTP + `FULL_DECODE_ONLY` to
a fixed defect at v0.18.0, and the sibling recipe above is w8a8-mtp rather than
bf16. The recorded release notes carry neighbouring fixes only (Qwen3-Next-MTP
batching accuracy #4932, "MTP prefill misclassified as decode" #6835,
`max_cudagraph_capture_size = max_concurrency × (1 + num_speculative_tokens)` for
the large-EP capture overflow, `docs/source/user_guide/release_notes.md:456, :252,
:267`). Cite them as family precedent, not as this bug's fix.
