# Speculative decoding on a PP > 1 deployment (#5227)

## Signature

`--pipeline-parallel-size` is 2 or more **and** the launch carries a
`--speculative-config`; the engine comes up and the **first request** dies with a
plain Python error (`IndexError`, `AttributeError`, `TypeError`) and **no** device
record (`EZ9999`, `507*`, `ERR0*`, no HCCL line, no `fftsplus`). The same page
answers "does PP work with spec decoding at all on this model". Not the lane for a
PP-only failure without a spec config (`index.md`), and not the proposer's own
numeric or acceptance symptoms (`../spec-decode-mtp/index.md`).

Recorded **#5227** (vllm-ascend `v0.12.0rc1-openeuler` image, Qwen3-32B, 910B,
`--pipeline-parallel-size 2 --dtype bfloat16 --max-model-len 32768
--max-num-batched-tokens 5120 --no-enable-prefix-caching`, driver 25.2.2,
`HCCL_OP_EXPANSION_MODE=AIV`, `ASCEND_USE_FIA=0`, `VLLM_ASCEND_ENABLE_NZ=0`):
`IndexError` on the first request. The reporter's own control, two days later:
**the same error with `method: "ngram"`** (`n=10`, `prompt_lookup_max: 4`), which
retires the suffix tree parameters (`suffix_decoding_max_tree_depth: 24`,
`suffix_decoding_max_cached_requests: 1000`) as the trigger.

Three field facts of that report stay unresolved and change the ablations, so ask:
the title says PP>2, the serve line says `--pipeline-parallel-size 2`, and the OCR'd
stage labels in the screenshot run `PP0…PP3` (garbled, so not proof of a 4-stage
run); the queue depth below equals `pp_size`, so the ablations differ by answer.

**What the paste must carry.** That report's only attachment was a screenshot OCR'd
offline: ~40 duplicated `Worker … PP*` / `pid: 558` / `Replaying aclgraph` lines and
a single `torch/csrc/utils/tensor_new.cpp:1578` tail — **no `vllm*/` frame**. That
file name is torch's tensor-construction frame (*recalled*, not verifiable here:
this snapshot's pytorch export has no `torch/csrc/`), so the raising Python frame sat
above it and did not survive the OCR. Ask for the traceback as text before ranking
anything; an `IndexError` cannot be localized from that OCR.

## 1. PP > 1 builds a batch queue of depth `pp_size`, and no flag removes it

Source-verified at vllm `b8b302c` (v0.11.0):

- `MultiprocExecutor.max_concurrent_batches` → `2 if scheduler_config.async_scheduling
  else parallel_config.pipeline_parallel_size`
  (`vllm/v1/executor/multiproc_executor.py:329-333`);
- `EngineCore.__init__` stores it as `batch_queue_size` (`vllm/v1/engine/core.py:145`),
  logs `Batch queue is enabled with size N` when it is > 1 (`:148-151`) and selects
  `self.step_fn = step_with_batch_queue` (`:167`, body at `:300`);
- downstream keeps the same branch: `vllm_ascend/patch/platform/patch_multiproc_executor.py:97`
  (`if self.max_concurrent_batches > 1:`).

So a PP=2 launch already has two `SchedulerOutput`s in flight and answers are consumed
by `step_with_batch_queue`, **without** `--async-scheduling`. This is the one place in
this library where the standard advice ("turn `--async-scheduling` off") is inert: the
only ablation that removes the queue on a PP deployment is
`--pipeline-parallel-size 1`. Say that instead of proposing an `async-scheduling` row,
and read the depth off the log rather than the flags when the report names two PP sizes.

## 2. "The proposer only exists on the last rank, so the draft ids are lost" is closed

Do not spend turns on it — one read each, at the same pins:

| Fact | Anchor |
|---|---|
| output is collected from the **last** PP stage's first TP worker | `_get_output_rank` returns `world_size - tensor_parallel_size` (`multiproc_executor.py:335-343`) |
| the drafter and the rejection sampler are created **only** there | `vllm_ascend/worker/model_runner_v1.py:341-345` under `if get_pp_group().is_last_rank:` (upstream `vllm/v1/worker/gpu_model_runner.py:277-280`, whose own comment says "currently we put the entire draft model on the last PP rank") |
| a non-last rank never calls a proposer | `propose_draft_token_ids` early-returns `None` when `self.drafter` is falsy (`model_runner_v1.py:1797`) |

Proposer and consumer are therefore on the same process. What *is* unguarded is the
bookkeeping that ships the draft ids back: `self._draft_token_ids = self.propose_draft_token_ids(...)`
(`:2178`) and `take_draft_token_ids()` (`:2226-2233`), which builds
`DraftTokenIds(self.input_batch.req_ids, …)` — a per-request **list alignment**
between the draft results and the *current* `input_batch.req_ids`. With ≥2 batches in
flight that batch can move between the step that produced the drafts and the step that
ships them, and a bare `IndexError` is what a shorter draft list against `req_ids`
looks like. That is an unverified hypothesis at the pin (label it so); the checks are
in section 6.

## 3. Method swap is the proposer-innocence probe; a token-count step is not

The non-model proposers share one worker-side path, and #5227's own ngram run is the
cheap probe: `--speculative-config '{"method":"ngram","num_speculative_tokens":2}'`
on the same PP=2 launch. Same error ⇒ the method, its tree depth, its cache size and
`num_speculative_tokens` are all retired and PP owns the case; different error ⇒ the
proposer's own parameters are back in play. Do not propose a `num_speculative_tokens`
rung as the first ablation here: for `suffix`/`ngram` the block-verify kernel is never
reached at all (the sampler is called with `draft_probs=None`; the anchor and its pin —
v0.19.1rc1, `rejection_sampler.py:114` — are in
`../spec-decode-mtp/draft-graph-capture-device-fault.md`'s non-MTP section, not at the
v0.11.0 pin used here), so `n` moves no kernel.

## 4. Pin gap: the method may not exist in the snapshot

At vllm-ascend `2f1aed98ccdb` / vllm `b8b302c` (v0.11.0) `method: "suffix"` **does not
exist**: `vllm/v1/spec_decode/` holds `ngram_proposer.py`, `eagle.py`, `medusa.py`,
`metadata.py`, `metrics.py`, `utils.py` and nothing suffix-shaped
(`rg -li suffix vllm/v1/spec_decode/` → `ngram_proposer.py` only), the accepted-method
`Literal` is `"ngram" | "eagle" | "eagle3" | "medusa" | … | "mtp"`
(`vllm/config/speculative.py:32-35`), and downstream
`get_spec_decode_method` (`vllm_ascend/spec_decode/__init__.py:24-35`) knows
`ngram | eagle | eagle3 | deepseek_mtp` and raises
`ValueError: Unknown speculative decoding method: <method>` for anything else — so a
`0.11.0` build given `method: "suffix"` would die at model-runner init, not at the
first request. A 0.12.0rc1 suffix report therefore cannot be source-verified at such a
pin: say so in one line, cite the `ValueError` as the proof of the gap, and ask for
`sed -n '<frame-5>,<frame+5>p'` around the reporter's own line numbers instead of
guessing what the newer file says.

## 5. Coverage: the combination has no e2e at any pin

- At `048bfd5` (v0.10.2rc1) no workflow invoked the PP e2e test; at **v0.11.0**
  (`2f1aed98ccdb`) `.github/workflows/_e2e_test.yaml:196` does
  (`pytest -sv tests/e2e/multicard/test_pipeline_parallel.py`). Re-grep the workflow
  before repeating "PP is not CI-backed".
- The test itself still carries no speculative config: `Qwen/Qwen3-0.6B` +
  `DeepSeek-V2-Lite-Chat`, tp 1, pp 2, `mp`+`ray`
  (`tests/e2e/multicard/test_pipeline_parallel.py:18-24,38-46`), while every
  `speculative_config` e2e in that tree is single-card
  (`tests/e2e/singlecard/spec_decode_v1/`).
- The project's own PP × graph statement is still the v0.9.2rc1 known issue
  (`docs/source/user_guide/release_notes.md:383` "Pipeline parallel does not work with
  ray and graph mode") — and #5227's paste shows `Replaying aclgraph` immediately
  before the failure, so graph mode was on.

## 6. Ladder, cheapest first

1. `--pipeline-parallel-size 1` with the **identical** `--speculative-config`, TP
   absorbing the cards: this is the only rung that removes the batch queue (§1). Works
   ⇒ PP owns the case and the proposer is exonerated.
2. Same PP, `"method": "ngram"` (or `mtp` if the checkpoint carries the head): the
   proposer-innocence probe (§3). #5227 has already run this one — reuse its result
   instead of asking again.
3. Graph off: `--compilation-config '{"cudagraph_mode": "NONE"}'`, or
   `--enforce-eager` **without** a `torchair_graph_config` key
   (`check_ascend_config` refuses the pair). The accurate-mirror rung: the OCR's
   `Replaying aclgraph` proves graph mode was on, so this confronts the known issue
   above. `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` is *not* a
   graph-off rung.
4. The text traceback, resolved PP size, and the `SpecDecoding metrics: acceptance
   length` line from a request that did run (its `Drafted: 0` vs `> 0` says whether
   the proposer produced anything on the step that died).
5. Re-read `take_draft_token_ids` / `_draft_token_ids` at the reporter's own ref
   (`vllm_ascend/worker/model_runner_v1.py`) only after 1-3 named the PP step; a
   patch proposal before that is a guess about a file the snapshot may not carry.

## 7. Precedents, and what stays open

- `#1751`: V1 pipeline parallel failed with the ray backend; `#1754`: "The ACLGraph
  graph model does not support pipeline parallelism" — the pair behind the
  release-note known issue.
- `#660`: "[Spec Decode] MLP and ngram spec decode mode run failed" — the non-model
  proposer family has failed in ways unrelated to the method's own knobs before.
- `#5227` itself: `suffix` and `ngram` both fail on the same PP launch; the raising
  frame was never published, so no cause is established here.

Open at the pin: whether the `IndexError` is the draft-id list alignment of §2, a
scheduler-side `scheduled_spec_decode_tokens` mismatch, or a graph-replay contract.
The paste that closes it is the text traceback plus rungs 1-2; do not present a
mechanism as established without them.
