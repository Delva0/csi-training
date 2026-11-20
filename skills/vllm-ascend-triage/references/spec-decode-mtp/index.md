# Speculative decoding / MTP runtime symptoms

## Scope

The deployment is up, there is no traceback, and the symptom is produced by the
speculative-decoding path (`--speculative-config` with `method: mtp`,
`qwen3_5_mtp`, `eagle`, `draft_model`, or an `AscendEagleProposer` /
`AscendDraftModelProposer` frame in the stack). Two shapes:

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
  `numBlocks` (`drafter-count-copy-device-fault.md`).

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

- **Three tracebacks from a spec-decode deployment are usually one provenance.**
  Read them bottom-up: the worker exception is causal, the EngineCore `None`
  output is the worker having died, and the sampler/metrics complaint is a
  *second* symptom of the same bad state, not a third bug. The recorded #8724 set
  (`KeyError` in `_update_states` + `NoneType … sampled_token_ids` + negative
  counter) all traces to one request whose sampled-token row was discarded, i.e.
  to `num_accepted = valid_sampled_token_count[...] - 1 == -1`. Say "one
  provenance, three symptoms" in the first sentence; a reporter reading three
  tracebacks otherwise reads three causes.

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
- `async-spec-bookkeeping-desync.md` — `--async-scheduling` + MTP and the
  three-traceback signature: the discarded-request chain
  (`model_runner_v1.py:838-842` mask → `prev_req_id_to_index` built without those
  indices → `-1`-filled row → `num_accepted = -1` → unguarded lookup), why the
  statement is upstream v0.13.0's and the patch only carries it, the
  `0 < temperature < 1` branch anchor in the rejection sampler with what remains
  unverified, the ladder (async off → MTP off → token step), and the upstream fix
  shape (#8724).
