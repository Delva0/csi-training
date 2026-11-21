# Two runs, one number each: turn the ratio into rates before ranking causes

## Scope

The report compares **two runs** — two platforms (Ascend vs H800/GPU), two images or
releases on the same command, or the two sides of a PD deployment — and pastes one
number per run: "12s vs 4s", "吞吐高了一倍", an EvalScope/AISBench summary table, or a
screenshot of one. Nothing crashed and the reviewer is being asked to accept "this
platform/version is slower" as the premise.

This is a *measurement* page, not a fault lane. Its job is the one step that decides
whether there is a defect at all: convert each side into a rate, on a stated axis,
before any mechanism is ranked. Both recorded cases in this batch needed it and both
took the wrong turn without it — one spent its budget on per-step arithmetic from an
OCR'd table whose average columns were transposed, the other ranked a PD mechanism
before comparing output token counts.

## Step 1 — reconcile the table; do not read its small columns

EvalScope / `vllm bench serve` / AISBench print the same block: `Time taken for tests
(s)`, concurrency, request count, `Output token throughput (tok/s)`, `Total token
throughput (tok/s)`, then the averages and per-token latencies. On a screenshot only
the **two large throughput fields and the wall time** survive OCR intact; the small
ones get transposed (`4094` → `4994`, `100` → `199`, `0.1046` → `0.1946` in recorded
#2833).

One call:

```bash
uv run --no-project python <skill_root>/scripts/benchmark-table-normalize.py \
  --run 'Ascend 57.6811 27.7387 16 8' --total-tok-s Ascend=2419.29 --ttft Ascend=3.2645 \
  --run 'H800   116.712 140.621  16 8' --total-tok-s H800=348.944  --ttft H800=2.0687 \
  --compare Ascend H800
```

It derives the per-request token counts, the decode step (`concurrency / output tok/s`),
the prefill cost (`TTFT / per-request input tokens`), a wall-clock closure check
(`wall / (waves × tokens × step)`, `>>1` = queueing, warmup or a fixed per-request
cost), and names the axis that moved more. #2833's output, which is the shape to
quote: Ascend `avg out 100, avg in 4094`, step **138.7 ms**; H800 `avg out 1026,
avg in 2041`, step **68.5 ms**; prefill 0.80 vs 1.01 ms per input token. Same
concurrency, so the output-tok/s ratio (2.02×) *is* the per-step ratio even though the
two runs used different input/output lengths — the *totals* are what mix-dependence
spoils, never the step ratio.

## Step 2 — split the ratio into its two axes

- **Decode**: `step = concurrency / output_tok_s`; compare per-step or per-sequence
  (ms, tok/s). Only this axis sees the MoE comm path, the graph/eager choice, the
  quantization width and the spec-token budget.
- **Prefill**: `TTFT / per-request input tokens`. Only this axis sees
  `--max-num-batched-tokens`, chunked prefill and sparse/prefill op coverage.

Rank causes on the axis that moved more and say the other one is *not* the report.
In #2833 decode was 2.0× while prefill per token *favoured* Ascend — so every
`max-num-batched-tokens` / chunked-prefill suggestion is off-target for that report,
and the all-gather/MC2 question is on it (`moe-comm-path-when-ep-off.md`).

## Step 3 — a same-command, two-release comparison is not a rate either

Wall clock per request moves with the **number of tokens produced**, so a
release-to-release "12s vs 4s" has two innocent explanations before any engine
change:

1. **Output length / thinking.** With a reasoning parser on, a minor bump can change
   the CoT length. Resend the same body 5× with `"stream": false` and compare
   `usage.completion_tokens` and `finish_reason` on both builds: equal tokens/s with a
   3× token count is not a regression. Recorded #14291 (v0.23.0rc1 → 0.24.0rc1,
   A3 PD, GLM-5.2-w4a8c8, `1+1=?` → 4s → 12s) never had these numbers.
2. **The spec-decode budget on the decode side.** A step emits up to
   `1 + num_speculative_tokens` tokens, so a collapsed acceptance is a
   `num_speculative_tokens+1` multiplier on decode wall time with no other change.
   #14291's D command carries `{"num_speculative_tokens":3, …}` — a 4-token/step
   budget against a 3× observed slowdown. The number is logged, not inferred:
   `SpecDecoding metrics: … Mean acceptance length: %.2f, Accepted: %d tokens,
   Drafted: %d tokens, Per-position acceptance rate: %s, Avg Draft acceptance rate:
   %.1f%%` (`vllm/v1/spec_decode/metrics.py:100-118` at vllm v0.23.0 `0fc695f`), and
   as counters `rate(vllm:spec_decode_num_accepted_tokens_total) /
   rate(vllm:spec_decode_num_draft_tokens_total)` (`:124-136`). `Mean acceptance
   length` *is* the tokens-per-step multiplier: read it on both builds before any
   source walk (`../spec-decode-mtp/acceptance-collapse.md`). When the reporter has no
   log and pastes the tables instead, `tokens per step ≈ ITL_p50 / TPOT_p50` gives the
   same multiplier, and the step-cost ratio against it is the break-even
   (`spec-decode-gain-accounting.md`).

A fixed per-request offset instead (TTFT up ~8 s while tokens/s is unchanged) moves
the question to the PD transport / first-token path
(`../pd-disaggregation/llmdatadist-done-signal-and-ttft.md`,
`../pd-disaggregation/recipe-contract-diff.md`).

## Step 4 — name the missing controls in the first message

- A variant measured on **one side only** is not a control. #14291 ran
  w8a8c8 solely on the new build, so "w4a8 and w8a8 are both 12s" retires the
  quant-width hypothesis and nothing about the version.
- First request after start is graph capture/warmup, not steady state — resend and
  report request 1 vs request 5.
- The reporting snapshot may not contain the newer build at all (vllm-ascend
  v0.24.0rc1 was absent for #14291). Say so, and hand back the diff command to run
  on the reporter's own tree (`git log --oneline v0.23.0rc1..v0.24.0rc1 --
  vllm_ascend/spec_decode vllm_ascend/attention vllm_ascend/distributed/kv_transfer
  vllm_ascend/compilation vllm_ascend/platform.py`) instead of hunting a mechanism
  no local file can settle.

## Step 5 — the delivery

The normalized table *is* the first answer: one line per run (step time, prefill per
token, per-request tokens), then the ranked causes with one check each. The arithmetic
of a mechanism is the last thing owed, not the first. Recorded #2833 spent 22 turns
and its whole 16 384-token budget on per-step byte-traffic arithmetic after the
decisive numbers were in hand, and shipped 656 characters; the normalized table plus
the flag finding was a two-turn answer
(`../evidence-workspace/delivery-discipline.md`).
