# Speculative decoding / MTP runtime symptoms

## Scope

The deployment is up, there is no traceback, and the symptom is produced by the
speculative-decoding path (`--speculative_config` with `method: mtp`,
`qwen3_5_mtp`, `eagle`, `draft_model`, or an `AscendEagleProposer` /
`AscendDraftModelProposer` frame in the stack). Two shapes:

- the spec-decode metrics collapse and request throughput follows: `Accepted: 0`
  with `Drafted: N > 0`, `Per-position acceptance rate: 0.000` for every position,
  `Mean acceptance length` pinned at 1.00, "推理卡住" while requests stay
  `Running`;
- a response comes back empty (`content: ""`) with `usage.completion_tokens: 1`,
  usually on the first request(s) after the server starts, and later requests are
  fine.

Not this topic: a launch/startup failure (`../startup-failures/index.md`), a
collective or device fault (`../hccl-and-hardware/index.md`), wrong or alternating
answers on a **PD** pair where the decode node received the state
(`../pd-disaggregation/hybrid-state-transfer.md`), and a "does this plan fit on N
cards" question (`../deploy-sizing/index.md`). A **tool call** whose JSON the
client cannot parse (`Unterminated string`, 格式错误, 参数被截断) while the text
itself is fine is also not here: the open question is whether the stream or the
assembly lost the tail, so start at `../tool-call-streaming/index.md`. When its
failing `num_speculative_tokens` is >= 3 and 1 or 2 works, that topic's
`mtp-block-verify-acceptance.md` carries the `max_spec_len >= 3` branch — name it
with the `n=2` and `temperature: 0` ablations rather than re-deriving it from the
acceptance numbers.

## Rules

- **The MTP-off ablation is row 1, not row 5.** Deleting `--speculative_config` is
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
  cannot carry: the recorded #9355 follow-up got a normal answer by changing only
  the prompt and `temperature`, so request content is still a live variable.
  Restart, repeat the identical body twice, then the modified body.

## Pages

- `acceptance-collapse.md` — the acceptance metric reading, the ordered ablation,
  the async-spec-decode `num_computed_tokens` drift correction, the
  `mamba_block_size` prefix-caching gate, where capacity fits, and the worked
  #10495 row (v0.20.2rc1).
- `empty-first-response.md` — `content: ""` + `completion_tokens: 1` +
  `finish_reason: "stop"` on the first request: the lane split against budget and
  reasoning routing, the restart reproduction, reading the emitted token id, the
  refuted "padding rows corrupt block 0" theory, the warm-up mitigation, and the
  worked #9355 row (v0.18.0).
