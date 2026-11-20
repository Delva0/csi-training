# Speculative decoding / MTP runtime symptoms

## Scope

The deployment is up and the symptom is produced by the speculative-decoding path
(`--speculative-config` with `method: mtp`, `qwen3_5_mtp`, `ngram`, `eagle`,
`draft_model`, or an `AscendEagleProposer` / `AscendDraftModelProposer` frame in the
stack). Usually there is no traceback; one shape is a plain `AssertionError` raised
at request time. The shapes:

- the spec-decode metrics collapse and request throughput follows: `Accepted: 0`
  with `Drafted: N > 0`, `Per-position acceptance rate: 0.000` for every position,
  `Mean acceptance length` pinned at 1.00, "推理卡住" while requests stay
  `Running`;
- a response comes back empty (`content: ""`) with `usage.completion_tokens: 1`,
  usually on the first request(s) after the server starts, and later requests are
  fine;
- the **process** dies instead: a worker `KeyError` in
  `vllm_ascend/patch/worker/patch_model_runner.py` `_update_states`, a
  `'NoneType' object has no attribute 'sampled_token_ids'` in the EngineCore and a
  negative `Counters can only be incremented by non-negative amounts`, together,
  with `--async-scheduling` and `--speculative-config` in the same launch and a
  fractional `temperature` (`async-spec-bookkeeping-desync.md`).
- the **process** dies with a *device dump* instead of a Python error: the deepest
  frame is `counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)` in
  `_copy_valid_sampled_token_count` (`model_runner_v1.py:1393`) and the paste also
  carries `rtMemcpyAsyncEx … reason=the model stream execute failed`, an `aivec
  error … timeout or trap error` and an HCCL task line with `rankSize` /
  `numBlocks` (`drafter-count-copy-device-fault.md`);
- or it dies with a device dump whose frames are
  `execute_dummy_batch` → `_dummy_run` → `acl_graph.__call__` →
  `stream.synchronize()`, on a launch carrying MTP + `FULL_DECODE_ONLY`
  (+ `npugraph_ex`), with `fftsplus aicore/aivector … MTE DDR out of range` and a
  capture-mode `rtMemcpy … 107030` record stamped inside engine init
  (`draft-graph-capture-device-fault.md`).
- or it dies with an **operator-level message and no Python frame**: `ERR00100 PTA
  call acl api failed` + `EZ9999 … When layout is TND, queryT(16) must be equal to
  the last element of actualSequenceLengthQ(4)` (`CheckFASeqlenDataInTND`,
  `fused_infer_attention_score.cpp`), on MTP + `FULL_DECODE_ONLY`, often with
  `--data-parallel-size > 1` and EP (`fia-tnd-seqlen-mismatch.md`).
- or it dies at **request time** on every rank in the same second with a bare
  `AssertionError` whose last frame is `attention_v1.py … assert
  attn_metadata.attn_mask is not None`, on a dense model with `method: ngram` (any
  non-`eagle` method): `AscendAttentionState.SpecDecoding` is declared in the enum
  but the dense backend has no branch for it, and `_make_attention_mask` returns
  `None` for that state (`dense-attn-state-gap.md`). The traceback may pass through
  `<eval_with_key>.N` / `unified_ascend_attention_with_output` — that is the graph
  wrapper around the same eager body, not a capture fault.

Not this topic: a launch/startup failure (`../startup-failures/index.md`), a
collective or device fault whose sync point is **not** the drafter's count copy
(`../hccl-and-hardware/index.md`; the copy-shaped variant stays here, because the
frame proves the launch shape and the ablations are spec-decode rows), wrong or alternating
answers on a **PD** pair where the decode node received the state
(`../pd-disaggregation/hybrid-state-transfer.md`), and a "does this plan fit on N
cards" question (`../deploy-sizing/index.md`). A **tool call** whose JSON the
client cannot parse (`Unterminated string`, 格式错误, 参数被截断) while the text
itself is fine is also not here: the open question is whether the stream or the
assembly lost the tail, so start at `../tool-call-streaming/index.md`. When its
failing `num_speculative_tokens` is >= 3 and 1 or 2 works, that topic's
`mtp-block-verify-acceptance.md` carries the `max_spec_len >= 3` branch (at newer
refs it also ANDs `rejection_sampler_config.enable_block_verify`, default `False`)
— name it with the `n=2` and `temperature: 0` ablations rather than re-deriving it
from the acceptance numbers.

## Rules

- **The MTP-off ablation is row 1, not row 5.** Deleting `--speculative-config` is
  one launch and splits "the draft/verify path is wrong" from "everything else".
  Recorded #10495: the run ranked KV capacity and queue thrash first on the
  strength of 98 % cache usage, and the reporter then removed MTP and reported
  normal inference — the metric collapse tracked MTP, not the queue. Rank the
  capacity arithmetic *after* the MTP row unless MTP-off has already been run.
- **Read the metric pair, not the throughput number.** Drafts still arriving
  (`Drafted: N`, non-zero) while nothing is accepted means both models run; the
  question is whether they see the same context (positions, `num_computed_tokens`,
  accepted-token bookkeeping). A batch that is merely too large to schedule looks
  different: fewer drafts per step, `Waiting` growth, preemption lines, and
  acceptance that is *low*, not exactly 0.000 on every position.
- **Two launcher axes are outside the project's own validated matrix for hybrid
  GDN models** (Qwen3.5/Qwen3.6, DeepSeek-V4, GLM): `--async-scheduling` together
  with spec decode, and `--enable-prefix-caching` together with MTP. Both are real
  code paths with their own fix-ups (page below), so "it is supported, therefore
  not the cause" is not available; they are ablation rows.
- **A first-response-only symptom needs a restart reproduction before it gets a
  one-shot mechanism.** "The first decode after capture is wrong, the state is
  never zeroed for the first request" is a plausible story that a single report
  cannot carry: the recorded #9355 follow-up got a normal answer by changing the
  prompt and `temperature` (and posting to another port), so neither request
  content nor process identity was ever pinned. Restart, repeat the identical body
  twice, then the modified body. In the same first pass, diff the launcher against
  the family sibling tutorial in `docs/source/tutorials/models/` when the exact
  model has no page — for #9355 that sibling (`Qwen3.5-27B.md`) shipped
  `"enforce_eager": true` inside `--speculative-config` and default capture sizes,
  and those two deviations are the cheapest rows.

- **When the operator names two numbers that must be equal, the arithmetic is the
  classification — not the third source file.** `queryT(N)` vs
  `last(actualSequenceLengthQ)(M)` (or any op message of the form "A must equal
  B") is a host-side bookkeeping disagreement: `N` is the graph bucket after
  `pad_for_cudagraph` *and* the DP-wide maximum, `M` is the real per-request
  cumsum. Run `scripts/fia-tnd-seqlen-check.sh` on the pair and the launch, state
  both readings with the one log line that separates them (`Running: <n>`), and
  put `PIECEWISE`/`--enforce-eager` in the first paragraph. Recorded #6598 spent
  its whole 16 K output budget re-deriving which of two padding branches produced
  the pair and shipped no answer (`fia-tnd-seqlen-mismatch.md`).

- **Three tracebacks from a spec-decode deployment are usually one provenance.**
  Read them bottom-up: the worker exception is causal, the EngineCore `None`
  output is the worker having died, and the sampler/metrics complaint is a
  *second* symptom of the same bad state, not a third bug. The recorded #8724 set
  (`KeyError` in `_update_states` + `NoneType … sampled_token_ids` + negative
  counter) all traces to one request whose sampled-token row was discarded, i.e.
  to `num_accepted = valid_sampled_token_count[...] - 1 == -1`. Say "one
  provenance, three symptoms" in the first sentence; a reporter reading three
  tracebacks otherwise reads three causes.

- **The attention backend, not the proposer, decides whether a spec method is
  supported.** One grep — `rg -n '<StateName>|<mask field>' vllm_ascend/attention/`
  — returns which backends consume the state and which merely declare it. A state
  that exists in the enum and is consumed by zero branches of the backend that is
  running is a coverage gap, not a config value and not a device fault. The recorded
  #2071 run reached that verdict only after re-reading three overlapping regions of
  `model_runner_v1.py`; the grep answers it in one call (`dense-attn-state-gap.md`).

## Pages

- `acceptance-collapse.md` — the acceptance metric reading, the ordered ablation,
  the async-spec-decode `num_computed_tokens` drift correction, the
  `mamba_block_size` prefix-caching gate, where capacity fits, and the worked
  #10495 row (v0.20.2rc1).
- `empty-first-response.md` — `content: ""` + `completion_tokens: 1` +
  `finish_reason: "stop"` on the first request: the lane split against budget and
  reasoning routing, the restart reproduction, the sibling-tutorial diff
  (`Qwen3.5-27B.md`) and the capture-size arithmetic, reading the emitted token id,
  the two padding stories (scheduler rows refuted; the MTP proposer's padded
  drafter batch live and switchable at `eagle_proposer.py:136-142`), the
  draft-side `enforce_eager` / `disable_padded_drafter_batch` ablation rows, the
  `min_tokens` mitigation, and the worked #9355 row (v0.18.0).
- `drafter-count-copy-device-fault.md` — a device dump surfacing at the drafter's
  valid-count D2H copy: why the frame is a sync point, how the frame itself
  proves `--async-scheduling` + spec decode + padded drafter batch (event created
  only under `num_spec_tokens and use_async_scheduling`), resolving the collective
  by rank size against the DP×PCP×TP group arithmetic (`fc3_quant_x`/`mc2` are not
  DP-sized), the ordered ablations, the DeepSeek-V4 `VLLM_ASCEND_APPLY_DSV4_PATCH`
  and `deepseek_mtp` remap gates with the only in-tree validated config, and the
  worked #10546 row (v0.20.2rc1, which shipped no answer).
- `draft-graph-capture-device-fault.md` — the draft forward captured into the
  `FULL` graph under `npugraph_ex`: what the `execute_dummy_batch` frame proves
  (a DP dummy pass, not a request) and does not prove (it runs the target **and**
  the drafter), the two-timestamp reading that makes "ran fine for a while"
  compatible with a fault recorded at init, the ranked candidates with the
  recorded `aclnnAddRmsNormBias` sibling stack, and the ablation ladder with its
  measured cost (spec-scoped `enforce_eager` = stable but several times slower;
  still broken at v0.22.1rc1). Worked #10700 row.
- `fia-tnd-seqlen-mismatch.md` — the FIA TND refusal (`queryT` vs
  `last(actualSequenceLengthQ)`): reading the two numbers as bucket-vs-real, the
  five padding sites that disagree at v0.14.0rc1 (`:588-591` local
  `pad_for_cudagraph`, `:606` DP-wide max, `:856-881` FULL+`uniform_decode` gate,
  `mla_v1.py:565-605` `graph_pad_size`, `mtp_proposer.py:434-448` drafter arange),
  the `FIA_SEQ_LEN_LIMIT = 16` / RFC #3003-#4300 constraint text, the GLM-4.x
  tutorial diff that is the first half of the answer, the six-rung ladder and the
  worked #6598 row (v0.14.0rc1, DP2×TP8 A2 two-node, `num_speculative_tokens: 3`).
- `dense-attn-state-gap.md` — `assert attn_metadata.attn_mask is not None`
  (`attention_v1.py:413`) under `method: ngram`: the three state branches plus one
  catch-all `else:` that serves `ChunkedPrefill` *and* the unknown states, the
  `use_eagle`-only split at `model_runner_v1.py:977-981` that hands every non-eagle
  method `SpecDecoding`, `_make_attention_mask` returning `None` for it, the
  `spec_attn_mask` consumer table (MLA only) as the one-grep verdict, why the crash
  arrives after a healthy period (first draft-producing decode step), the
  `pytest.skip("Not current support for the test.")` at
  `tests/e2e/singlecard/spec_decode_v1/test_v1_spec_decode.py:72` as the tree's own
  statement, the ladder that excludes `--enforce-eager`, and the worked #2071 row
  (v0.9.2rc1).
- `async-spec-bookkeeping-desync.md` — `--async-scheduling` + MTP and the
  three-traceback signature: the discarded-request chain
  (`model_runner_v1.py:838-842` mask → `prev_req_id_to_index` built without those
  indices → `-1`-filled row → `num_accepted = -1` → unguarded lookup), why the
  statement is upstream v0.13.0's and the patch only carries it, the
  `0 < temperature < 1` branch anchor in the rejection sampler with what remains
  unverified, the ladder (async off → MTP off → token step), and the upstream fix
  shape (#8724).
