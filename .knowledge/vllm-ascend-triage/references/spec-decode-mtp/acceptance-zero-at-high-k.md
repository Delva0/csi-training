# Acceptance 0 that tracks `num_speculative_tokens`: the K-threshold shape

## When this page applies

The metrics are `acceptance-collapse.md`'s (`Accepted: 0`, every per-position rate
`0.000`, `Mean acceptance length: 1.00`, `Drafted: N > 0`) **and** the report also
carries a speculative-length axis: a lower `num_speculative_tokens` is stated to work,
the launch's K is above the project's own validated value, or the same K served
correctly on an older release. Then K is the classification — sweep it before reading
the proposer, and if the reporter's control is an older release, the deliverable is a
regression window rather than a mechanism (Step 4c).

Not this page: acceptance 0 with no K axis (`acceptance-collapse.md`, the
positions/`num_computed_tokens` walk), acceptance 0 while the target's own text is
correct on a model family the dispatch never names (`family-proposer-parity.md`), and a
**truncated tool-call JSON** whose failing K is ≥ 3 (`../tool-call-streaming/
mtp-block-verify-acceptance.md` — the block-verify branch is a format symptom and does
not zero the acceptance number).

## Step 1 — read K off the launch, and the two metric numbers off the paste

The K is in `--speculative-config` (`num_speculative_tokens`, or `n`). Read it together
with `cudagraph_capture_sizes`: on a uniform-decode launch the bucket is
`max_num_seqs × (1 + K)` (`vllm_ascend/compilation/compiler_interface.py`, static-kernel
`uniform_decode_query_len = num_spec_tokens + 1`), so the reporter's
`[14, 28, … 112]` **is** `14 × {1..8}` for K=13 and `--max-num-seqs 8` — a covered list,
not a stray one. A bucket list that does not decompose that way is a padding bug; one
that decomposes is not the leak.

Then the metric pair, exactly as `acceptance-collapse.md` Step 1 reads it, plus the
capacity line: `Running`/`Waiting`/`GPU KV cache usage`. On this shape those must show
**one or few requests and low cache usage** (~2 % in #9290's screenshot, `Running: 1`,
`Waiting: 0`); if they do, the capacity lane is refuted by the reporter's own numbers and
must not be ranked at all. When the body carries only a screenshot, its
`*.ocr.txt` under `/workspace/attachments/<issue>/` is that paste — read it whole in the
first block, before any source grep (its `9`/`2` glyphs are OCR of `0`, so
`Accepted: 9 tokens` / `Per—POS1t10n … 9.øøø` is `Accepted: 0` / `0.000`).

## Step 2 — what the pin bounds and what it does not

Verified at vllm-ascend v0.19.1rc1 (`da421afad719`):

| Anchor | What it says |
|---|---|
| `vllm_ascend/attention/attention_v1.py:243-251`, and the same block at `sfa_v1.py:194-200` and `mla_v1.py:261-267` | `self.decode_threshold = 1 + K` then `assert self.decode_threshold <= 16` — *"decode_threshold exceeded npu_fused_infer_attention_score TND layout's limit of 16, got N"*. **Three backends, one bound, and it is 16.** So K ≤ 15 passes silently and a reported threshold of 8 has no owner to find in the tree: an `assert` that never fires cannot be the check, and K ≥ 16 is a different report (an *init-time* `AssertionError` naming the limit) |
| `vllm_ascend/spec_decode/eagle_proposer.py:105` `self.decode_threshold = 1 + self.num_speculative_tokens`, `:160-168` `seq_lens_group` / `query_start_loc_group` = lists of `num_speculative_tokens` zeros, `:445,459,621` `num_reqs * decode_threshold` | open-ended in K by construction — these carry K into buffer shapes but impose no cap, so reading them cannot produce a threshold |
| `vllm_ascend/sample/rejection_sampler.py:116` `using_block_verify = max_spec_len >= 3 and draft_probs is not None` | the only K-gated *branch* below 16, and it is the format lane (`../tool-call-streaming/mtp-block-verify-acceptance.md`). Its kernels (`ops/triton/reject_sample.py:233` `sample_recovered_tokens_kernel`, `:362` `rejection_random_sample_block_verify_kernel`) loop over `num_draft_tokens` at runtime; `BLOCK_SIZE`/`SUB_BLOCK` are vocabulary tiles, not K. `worker/v2/sample/penalties.py` `MAX_SPEC_LEN` is set *from* `num_speculative_tokens` |

Consequence: at the failing pin, source reading can confirm that no in-tree code owns a
K ≥ 8 threshold. That is a finding to state (it closes the "find the clamp" lane), not a
reason to keep walking the proposer.

## Step 3 — the coverage number is K = 3, and every row above it is another method

`rg -o 'num_speculative_tokens"?\s*:\s*[0-9]+' tests/ docs/ examples/` at `da421af`
returns 1 (×35), 2 (×18), 3 (×75), 5 (×1), 8 (×2), 10 (×1), 15 (×1) — and **every row
above 3 belongs to another proposer**: `method: ngram` at K=5
(`docs/source/user_guide/feature_guide/speculative_decoding.md:24`) and `method: suffix`
at K=8/10/15 (`:143`, `tests/e2e/singlecard/spec_decode/test_v1_spec_decode.py:218,256,364`).
For the Qwen3.5/MTP family the validated envelope is K = 3, in three places:

- `docs/source/tutorials/models/Qwen3.5-27B.md:119` —
  `{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}` with
  `:120` `FULL_DECODE_ONLY`;
- `tests/e2e/nightly/single_node/models/configs/Qwen3.5-27B-w8a8-A3.yaml:40,42` and
  `.../Qwen3.5-397B-A17B-W8A8-mtp-A3.yaml:43,45` — same K and same in-config
  `enforce_eager`;
- `tests/e2e/multicard/4-cards/test_qwen3_5.py:78-80,109-111` — K=3 under
  `FULL_DECODE_ONLY` (no `enforce_eager` in the spec config).

So "K=13 on this family is outside every measured row" is quotable coverage, and the
tutorial's own K is the value rung 1 asks for. Two things it does **not** mean, both of
which the recorded run had to rule out explicitly: the reporter's `method: mtp` is not a
deviation (upstream rewrites every `MTPModelTypes` alias to `mtp`, and
`patch/platform/patch_speculative_config.py` turns `qwen3_5` into the `qwen3_5_mtp` draft
— `acceptance-collapse.md` Step 3), and an out-of-envelope value is not by itself the
cause.

## Step 4 — the ordered ladder

a. **Sweep K across the reported boundary on the reporter's own command**, one axis per
   launch: for a "≥ 8 breaks" report, K = 7 and K = 8; then the validated K = 3. Two
   launches bracket the threshold and one lands on the documented value. Ask for the
   acceptance line of each, not a benchmark.
b. **The tutorial deltas, in the same message**: `"enforce_eager": true` *inside*
   `--speculative-config` (the drafter eager while the target keeps its graphs), and the
   default capture sizes. Prefix caching and `--async-scheduling` are the two other
   unvalidated launcher axes for hybrid GDN families (`index.md` Rules).
c. **Dating a regression when the reporter's control is an older release.** "K=13 fine
   on v0.17.0, zero on v0.19.1" makes the older tag the control, so the failing tree
   cannot contain the answer — no amount of `eagle_proposer.py` reading finds "what
   changed". The deliverables are the window (`<works>..<fails>`, both named by the
   reporter), the one-command bisect (the same K on each intermediate rc), and the
   workaround (K below the threshold, or the validated 3). If a clone with history is
   available, `git log <works>..<fails> --oneline -- vllm_ascend/spec_decode
   vllm_ascend/sample vllm_ascend/attention` plus the release notes' spec-decode bullets
   are the whole source move; the snapshot's `one_commit_git` trees have depth 1, so say
   that instead of implying a diff was checked.
d. **Leave the mechanism open with the window**, labelled as hypothesis. Do not let a
   named-but-unproven mechanism displace the sweep: at #9290 the two published answers
   ahead of the reporter's own follow-up were "MTP broken in this release" and silence.

## Step 5 — dead ends, so the next run skips them (all verified at `da421af`)

- The `max_spec_len >= 3` block-verify branch: a *format* symptom page; it improves
  acceptance, and its kernels have no constexpr K cap (Step 2).
- `_run_merged_draft` / `attn_update_stack_num_spec_norm` / `seq_lens_group`: K-sized
  buffers with no bound; reading them yields a story, not a check.
- Capture-size accounting: the reporter's list already covers `max_num_seqs × (1 + K)`.
- `decode_threshold <= 16`: fires only at K ≥ 16, i.e. it is not this report's gate. Its
  value here is to say out loud that the 8…15 band is silently legal.
- The metric pair itself: `Accepted: 0` / `Drafted: 2132` / per-position `0.000` with
  `Running: 1` and ~2 % KV cache says nothing about queue pressure — do not spend a turn
  re-deriving capacity.

## Recorded run: #9290 (v0.19.1rc1 image, Qwen3.5-122B-A10B W8A8, TP=8 + EP, 8 dies)

Reporter facts: `--speculative-config '{"num_speculative_tokens": 13, "method": "mtp"}'`,
`FULL_DECODE_ONLY` with `cudagraph_capture_sizes [14,28,…,112]`, `--no-async-scheduling`,
`--max-num-seqs 8`, `enable_npugraph_ex`, no prefix-caching flag; the body has **no log**,
only a screenshot whose OCR reads `acceptance length: 1.00`, `Accepted: 0 tokens`,
`Drafted: 285 tokens`, per-position `0.000…`, `Running: 1`, `Waiting: 0`, KV cache ~2 %.
Complaint: acceptance 0 and very high latency.

Delivery outcome: **1.22 M cumulative prompt tokens, 16 384 output tokens (the cap),
zero message text** — 24 tool blocks, every one of them reasoning-only, on the last turn
an unfinished sentence about one merged draft graph. The material for a complete answer
was in hand by block 8 (metric pair + the `<= 16` bound + the tutorial's K=3 recipe); the
run instead read `rejection_sampler.py`, `reject_sample.py`, `penalties.py`,
`block_table.py`, `eagle_proposer.py` and the compilation config looking for an in-tree
clamp that does not exist, and opened the screenshot OCR only at block 14.

Reporter-confirmed later (2026-06-08): with this model, **speculative length ≥ 8 gives
acceptance 0 and a corrupted final output; below 8 it is normal; v0.19.1 and v0.20.2 both
fail while v0.17.0 served K=13 with normal accuracy**. Threshold and window are
reporter-confirmed; the mechanism and the fixing commit are not identified, and this
snapshot cannot name them.

## Evidence limits

- The threshold (8) and the version window come from the reporter on one model, one box,
  one weight set. Do not generalise the number to other families: the validated value in
  this tree is 3 and the only hard bound is 16.
- "Corrupted output" and "acceptance 0" arrive together in the follow-up. They may be one
  defect (drafts computed from a wrong state are both rejected and absorbed) or two; say
  which one the reporter's next launch shows, and keep the ranking valid either way.
- Recheck conditions: a later release note naming a spec-decode acceptance fix in the
  v0.18–v0.20 window, or any tree that gains a `num_speculative_tokens` validator below
  16, changes this page's Step 2 table.
