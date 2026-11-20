# Spec-decode acceptance collapses to zero and throughput follows

Scope: a server that serves traffic, then the spec-decode metrics read
`Accepted: 0` with `Drafted: N > 0`, `Per-position acceptance rate: 0.000,
0.000, …` for every position, `Mean acceptance length: 1.00`, and generation
throughput falls to ~0 while requests stay `Running`. The reporter's words are
usually "卡住 / stuck" or "MTP 有问题". This is a bookkeeping question about what the
draft and the target model each see, not a numerics or device question — there is
no dump, no `EZ…` code, and both models keep executing.

## Step 1 — read two numbers, not the throughput line

| Metric line | What it says |
|---|---|
| `Drafted: N` non-zero | the draft model runs every step; the proposer is alive |
| `Accepted: 0` **and** every per-position rate `0.000` **and** `Mean acceptance length: 1.00` | not one draft token matched the target for this whole window: draft context ≠ target context (positions, `num_computed_tokens`, accepted-token counts, or a stale draft KV for the request) |
| acceptance *low* (e.g. 0.3-0.5, per-position rates differ) plus `Waiting` growth, `Running` at `max-num-seqs`, `GPU KV cache usage` ≥ 98 %, preemption lines | scheduling/capacity pressure: the batch is thrashing, the draft path is not necessarily broken |
| acceptance 0 **and** throughput 0 **and** no drafts | the worker is not stepping at all — go to `../hccl-and-hardware/index.md` / `rpc-watchdog-timeouts.md` |

The exactly-0.000-everywhere pattern across many steps is the discriminator: a
capacity-limited engine produces *worse* acceptance, not a perfectly uniform zero,
and it does not keep drafting at a steady rate while emitting nothing.

## Step 2 — the ordered ablation (one axis per launch)

| Row | Change | What it proves |
|---|---|---|
| 1 | delete `--speculative_config` entirely | owner = draft/verify path versus target path. **Recorded #10495: normal ⇒ MTP owned.** This is the row to run before any code walk |
| 2 | keep MTP, drop `--async-scheduling` | the CPU/GPU token-count bookkeeping below, which only exists in async mode |
| 3 | keep MTP, add `--no-enable-prefix-caching` | the hybrid `mamba_block_size` / aligned-state path (and it is what the project's own Qwen3.5-MTP perf config runs) |
| 4 | `"num_speculative_tokens": 1` instead of 3-5 | whether the padding/lookahead budget, not the mode, is the trigger |
| 5 | `--max-num-seqs 8`, `--max-model-len` down to input+output, AISBench concurrency ≤ 8 | capacity only; run this *after* 1-3, it changes slowness, not the 0.000 signature |
| 6 | `--enforce-eager` (target eager, or keep the draft eager as the weekly config does) | whether the graph-mode decode path participates |

Report rows 1-3 back even when the reporter has not run them: they are the
difference between "MTP is broken in this release" and "this flag combination is
unvalidated".

## Step 3 — the code areas, at the pinned downstream commit

| Anchor (vllm-ascend v0.20.2rc1, `367b8e62da79`) | What it tells you |
|---|---|
| `vllm_ascend/spec_decode/utils.py:6-30` `update_num_computed_tokens_for_batch_change` — docstring "Correct num_computed_tokens for async spec decode drift" | async spec decode keeps `num_computed_tokens` **optimistic on CPU** ("all drafts accepted") and repairs it on GPU from the previous step's `valid_sampled_token_count`, `prev_positions`, `prev_num_draft_tokens`; `participating = (prev_positions >= 0) & (prev_drafts > 0)`, everything else falls back to the CPU value. A wrong mapping after a batch-composition change (a finish → `condense()`, a preemption, a request re-entering) puts positions/seq_lens off and drops every draft — permanently, because the next step starts from the drifted value |
| callers `vllm_ascend/worker/model_runner_v1.py:968` and `_310p/model_runner_310p.py:406` | where the repair is applied; `:1013` onward is the matching `seq_lens`/`optimistic_seq_lens_cpu` sync whose comment names the stale-CPU hazard |
| same runner, `discard_requests_mask` from `optimistic_seq_lens_cpu < num_tokens` | explains "looks stuck": sampled tokens for requests whose optimistic length overshot are discarded |
| `vllm_ascend/patch/platform/patch_mamba_config.py:103-106` | `mamba_block_size = block_size` **only** when `enable_prefix_caching` *and* `mamba_cache_mode == "align"`; otherwise `mamba_block_size = model_config.max_model_len`. For a hybrid GDN model, prefix caching therefore changes the state-block geometry the draft path sits on |
| `vllm_ascend/spec_decode/__init__.py` `get_spec_decode_method` | `mtp`/`eagle`/`eagle3` → `AscendEagleProposer` (one proposer for all three since the #6349/#7033 refactor); `patch/platform/patch_speculative_config.py` rewrites the draft `hf_config` — `qwen3_5`/`qwen3_5_moe` → `model_type` `qwen3_5_mtp` with `Qwen3_5MTP`/`Qwen3_5MoeMTP`, `qwen3_next` → `qwen3_next_mtp` with `Qwen3NextMTP` — which is how a "Qwen3.6" config lands on the Qwen3-Next MTP class. The e2e configs also pass `method: qwen3_5_mtp`; how that string becomes a proposer was not traced at either pinned commit, so do not assert it |

## Step 4 — what the project validates (coverage, not opinion)

Verified by grep at v0.20.2rc1:

- `tests/e2e/multicard/4-cards/test_qwen3_5.py` runs `qwen3_5_mtp` with
  `FULL_DECODE_ONLY` at `max_model_len=4096`, four short prompts, no prefix
  caching, no async scheduling.
- `--async-scheduling` appears in **four** lines of `tests/`:
  `tests/e2e/nightly/multi_node/external_dp/config/GLM5_1-W8A8-EP-external.yaml`
  (×2), `.../external_dp/scripts/utils.py`, and
  `tests/e2e/nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml:58`
  (which does combine it with `mtp` + `--enable-prefix-caching`). No
  Qwen3.5/Qwen3.6/GDN config does.
- The Qwen3.5-MTP long-run perf config
  `tests/e2e/weekly/single_node/configs/Qwen3.5-122B-A10B-W8A8-A2.yaml` uses
  `--no-enable-prefix-caching` (`:21`) with
  `{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}`
  (`:49`), 3.5K-token inputs and bs 64 — i.e. prefix caching + 64K prompts + 64
  concurrent requests is outside the measured envelope on this model family.

## Step 5 — where capacity genuinely lands

Capacity arithmetic is real, but it is a *second-order* answer here: it explains
the size of the stall, not the 0.000 acceptance. One startup-log read settles it
(`grep -E "KV cache size|Maximum concurrency" <serve.log>`): the
`Maximum concurrency for <max_model_len> tokens per request: Nx` line is the
number of requests the pool can hold, and if the harness concurrency exceeds it,
preemption/recompute dominates and every throughput number becomes meaningless —
including the acceptance metric. Fix the concurrency and `--max-model-len` first,
then re-read the acceptance metric; do not spend the answer on the KV arithmetic
while the MTP row is unrun.

## Recorded run: #10495 (vllm-ascend v0.20.2rc1, Qwen3.6 MoE + `qwen3_5_mtp`, 2×A2)

Reporter facts: 128 AISBench requests with ~64K-token inputs, `--async-scheduling`
+ `--enable-prefix-caching` + MTP, TP=2; metrics went from ~46 % acceptance to
`Accepted: 0` / per-position `0.000` / mean acceptance length 1.00, `Running: 33,
Waiting: 31`, `GPU KV cache usage 98.6→99.1 %`, generation throughput
4.3 → 1.8 → 0.0 → 0.7 tokens/s. Later public follow-up from the same reporter:
**"If we remove MTP, inference was normal. We suspect MTP has certain issue on
0.20.2rc1."**

What the recorded investigation got right: it found the async-spec-decode drift
correction, the `mamba_block_size` gate, and the coverage facts above, and it
stated the upstream gap honestly. What it got wrong is ordering: it led with
capacity/queue thrash and only then the MTP bookkeeping, so the delivered answer's
first two hundred lines argue a cause the reporter's own ablation later removed.
The durable rule is in `index.md`: MTP-off first, capacity later.

Gaps: the pinned upstream `vllm` for this case is **v0.13.0** (`72506c98349d`)
while vllm-ascend v0.20.2rc1 references `use_async_spec_decode`,
`cache_config.mamba_cache_mode` and `HybridAttentionMambaModelConfig`, none of
which exist at that upstream commit — so the upstream half of both mechanisms is
unverifiable in this snapshot; say so instead of quoting upstream semantics.
