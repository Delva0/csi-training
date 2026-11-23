# `response_format` set, and the answer is JSON-shaped but corrupt

## Signature, and why it is not simply "the schema was not enforced"

A chat body carries `response_format: {"type": "json_schema", ...}` (or the older
`guided_json` spelling, or `structured_outputs`) and the answer is **shaped like the
schema but invalid**: braces, quotes and key names doubled or interleaved at character
level — `"content": "{\"{\"eexxxtttrrraaacctceded__eentntnitiiteesss\":\":[[{\"{\"nnaamameee\"\"...”`
— with `finish_reason: "stop"` (not `length`), non-streaming, and a token count far
above the text's own size (`completion_tokens: 207` against `prompt_tokens: 12` for a
~250-character body). Recorded **#5835**: vllm-ascend **v0.11.0**, Qwen3-235B, env dump
literally `none`, title "使用 Json Schema 特性返回精度有问题" with no serve command, no
log and no plain-text control body. The title contrasts the feature with normal serving;
that is a hint to confirm by rerun, not the control run itself.

The project claims the feature: `docs/source/user_guide/feature_guide/structured_output.md:15`
at v0.13.0rc1 says its usage "is totally the same as that in vllm", and the support
matrix row is `Guided Decoding | 🟢 Functional`
(`docs/source/user_guide/support_matrix/supported_features.md:22`). So the report is a
defect claim, and the answer is which layer stopped constraining, not "supported/not
supported".

Two facts decide the page:

- **A mask applied to the row that was sampled cannot emit invalid structure.** The
  bitmask is compiled from the schema and applied in place on the logits row *before*
  `_sample`, so invalid JSON out of a masked row is a contradiction: either the mask
  landed on a different row or a stale FSM state (**lane A**), or the corruption sits
  below the grammar layer and the schema only made it visible (**lane B**).
- **Character-level doubling is a token-emission signature, not a JSON-validation
  one.** It says the model really emitted several times the tokens that text needs; no amount of
  schema reading changes that, so do not open with the schema.

`finish_reason: "length"` sends the report elsewhere first: that is the thinking-budget
/ output-routing lane (`reasoning-output-routing.md`) — a CoT cut before the answer is
not this page.

## Step 0 — one body splits the lanes, and it is not the serve flag

Resend the **same** body with `response_format` removed, asking for JSON in words, and
a plain-text prompt alongside it:

| Result | Lane | First check |
|---|---|---|
| still corrupt without the field | B — decode-level | the four ablation rows below |
| clean without the field, corrupt with it | A — the mask path | serial (batch-1) rerun, then the backend restart |

Then, inside lane A, the **serial** rerun is the cheapest discriminator and costs one
launch: row-mapping faults need two or more rows in the batch to bite, so
`--max-num-seqs 1` (or one request at a time) clean ⇒ misalignment; still corrupt at
batch 1 ⇒ back to lane B.

## Lane A — the mask never reached the sampled row

Verified anchors, both trees in this library's snapshots:

| Fact | Anchor |
|---|---|
| The downstream runner does not call upstream's compiled path: it copies the whole logits tensor to host, casts to float32, applies the mask, copies back | `vllm_ascend/worker/model_runner_v1.py:1849-1855` at v0.19.1rc1, same block at `:1558-1566` at v0.13.0rc1, with the in-tree comment "here we are different from gpu_model_runner … ascend does not support it now" |
| Upstream maps bitmask rows by `input_batch.req_id_to_index` **plus** a cumulative spec-decode offset, then skips `indices` entirely when the counts happen to match | `vllm/v1/structured_output/utils.py:48-126` at vllm v0.13.0 (`cumulative_offset` `:76-84`, `out_indices` `:86-104`, `skip_out_indices` `:114`, `apply_token_bitmask_inplace` `:126`) |

So lane A is a **row-order contract**, and the contract is exactly what a downstream
fork scheduler or a spec-decode bookkeeping change can break silently: the mask is
applied to another request's row (that request samples unconstrained, and any legal
token id can then be emitted), or to none. The two recorded assert/raise forms of the
same misalignment are already pages — read them before re-deriving:

- `../pd-disaggregation/structured-output-draft-placeholder-assert.md` — seeded `-1`
  draft placeholders (`The token id -1 is out of range … Rejecting the token` →
  `AssertionError: (-1, …)`), the *loud* form.
- `../upstream-downstream-drift/scheduler-fork-missing-status.md` — the fork never sets
  `WAITING_FOR_FSM` and hardwires the bitmask empty, the *silent* form, with the state
  the request carries (`guided_*` / `response_format`, or a forced `tool_choice`).

## The backend A/B is a restart, not a request field (vllm ≥ 0.13)

Do not design the ablation as "same body, `json_schema` vs `json_object`, one serve".
At vllm v0.13.0:

- `--guided-decoding-backend` no longer exists — `rg -ln guided_decoding_backend vllm/`
  has no hit; the launch flag is `--structured-outputs-config '{"backend": …}'`,
  default `auto` (`vllm/config/structured_outputs.py:23`).
- `_backend` is resolved per request in `vllm/v1/engine/input_processor.py:258-284`, and
  a request that sets it is **rejected**: "Request-level structured output backend
  selection is not supported".
- With `auto`, the request is validated against xgrammar first and falls back to
  guidance on a schema feature xgrammar refuses (`:300-352`, `_backend_was_auto`), and
  `StructuredOutputManager` then caches **one** backend for the engine's life
  (`vllm/v1/structured_output/__init__.py:109-149`, "We do NOT support different
  backends on a per-request basis") — so the *first* structured request of a process
  decides the backend for every later one.

Consequence for the answer: every backend rung is a **restart with the backend pinned**,
and a result harvested from two body shapes inside one process says which schema
validated, not which backend a later request used.

Older pins answered this question differently: at vllm v0.7.3 the frontend stamped
`guided_decoding_backend="xgrammar"` for `response_format.type == "json_schema"` only
(`../serving-frontend-faults/guided-decoding-backend-aarch64.md`, where
`entrypoints/openai/protocol.py:455-465` is the anchor and `openai/processor.py` is the
file that later carries the logic). Those paths do not exist at v0.13.0 — re-grep the
pin's own tree before treating a miss as "the field is gone from the report's version".

## Rung order

0. the lane split above (one curl per shape);
1. **serial** rerun of the failing body (`--max-num-seqs 1`);
2. restart with `--structured-outputs-config '{"backend": "xgrammar"}'`; a second
   restart with `"guidance"` only if the schema needs it;
3. lane B rows, in this order — `VLLM_ASCEND_ENABLE_NZ=0`, `--dtype bfloat16`,
   speculative decoding off, `--enforce-eager`.

Lane B is real on this symptom family: at v0.13.0rc1 `VLLM_ASCEND_ENABLE_NZ` defaults to
**1** (`vllm_ascend/envs.py:120-121`) and the worker itself warns that fractal NZ "may
cause model parameter precision issues" (`vllm_ascend/worker/worker.py:163-166`). Two
precedents keep it ranked but not first: **#4323** (vllm-ascend 0.11.x, Qwen3-32B +
`eagle3`, `num_speculative_tokens: 4` → `tctctctctcctctcctc1llllllllll");nll`, i.e. the
same repetition family with a **spec-decode** config and no structured output at all),
and **#5144** (v0.11.0, fp16, `VLLM_ASCEND_ENABLE_NZ=0` already exported, the *first*
request of a concurrent pair answering `!!!!…` to `finish_reason: length` — so the env
var is not a universal fix, and the trigger is batch-position dependent). One more
pinned fact to save a wasted reply: spec decoding **plus** async scheduling **plus**
structured outputs is a hard `ValueError` at this pin
(`vllm/v1/engine/input_processor.py:147-162`), so a garble cannot come from that
combination.

## Scope and open ends

- Verified in the pinned reference trees: vllm-ascend **v0.13.0rc1**
  (`1b5d5abf865f`) and vllm **v0.13.0** (`72506c98349d`), plus the same bitmask block at
  vllm-ascend v0.19.1rc1 (`da421afa`, `:1849-1855`) — the host roundtrip is a durable
  downstream divergence, not a one-release accident.
- The report's own tag is **v0.11.0**, which no snapshot carries, and it pasted no serve
  command and no env dump. Cite the mechanism as the reference tree's, ask for the
  launch line (`--structured-outputs-config` / old `--guided-decoding-backend`,
  `--max-model-len`, `--speculative-config`, `--dtype`), and keep the owner open.
- Not evidence: the number of doubled characters, the reporter's JSON body itself, and
  a screenshot of the answer. Do not report the FSM/bitmask arithmetic as proven for
  their tag; report the lane split, the rung order and the two artifacts to send back.
