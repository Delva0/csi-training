# "Turning MTP on made it slower": the gain arithmetic before any spec-decode code walk

## When this page applies

A **performance** report on a healthy engine where the two runs differ by one
`--speculative-config` (MTP / `qwen3_x_mtp` / eagle), the direction is *negative*,
and the evidence is two benchmark tables (AISBench / EvalScope / `vllm bench serve`)
or two wall clocks. The reporter usually adds the control that makes it a puzzle:
the same release and flags on a sibling model *do* improve ("27b、122b 性能都有提升").
Recorded **#8967** — vllm 0.18.rc1, Qwen3.5-35B-A3B-w8a8-mtp on **one** 910B3,
`{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}`,
`FULL_DECODE_ONLY`, concurrency 1, 28 requests, E2EL 21.3 s → 26.0 s.

This is not the acceptance-collapse lane: acceptance is *fine* in this shape, which
is why `../spec-decode-mtp/acceptance-collapse.md` (Accepted: 0) does not answer it.
The deliverable is arithmetic over the reporter's own table plus the one experiment
that measures the slope.

## Step 1 — tokens per decode step comes out of the table, no log needed

With streaming, one spec-decode step emits several tokens in **one chunk**: the
intra-chunk gaps are ~0.01-0.03 ms (`Min ITL`) and the step wall is the **modal /
median** gap. So on a single-request run

```text
tokens per step  ≈ ITL_p50 / TPOT_p50        (both per-token, same unit)
step wall        = ITL_p50
```

#8967: no-MTP `ITL p50 14.84 ms` / `TPOT 14.81` ⇒ 1.0 token/step; MTP
`ITL p50 53.80` / `TPOT 16.20` ⇒ **≈3.3 tokens/step** against a 4-token budget
(`1 + num_speculative_tokens`) — i.e. acceptance is healthy and needs no log line.
When a log *is* available, take `SpecDecoding metrics: … Mean acceptance length:
%.2f` instead (`../spec-decode-mtp/acceptance-collapse.md`); the table derivation is
the fallback when the reporter pasted no log at all, which is the usual case here.

Do **not** use the mean ITL or the total token throughput for this: the mean is
dragged by the near-zero intra-chunk gaps, and `Total token throughput` mixes the
input side.

## Step 2 — the break-even is a ratio of ratios

```text
step cost ratio = ITL_p50(with spec) / ITL_p50(plain)        # here 3.62×
token ratio     = tokens per step                             # here 3.32×
per-token cost  = TPOT(with spec) / TPOT(plain)               # here 1.09×  (+9 %)
```

- `step ratio > token ratio` ⇒ spec decode is already losing; per-token cost is the
  number to quote, not the reporter's wall clock.
- The headline percentage is usually **E2EL**, which also carries TTFT
  (635 → 801 ms, +26 % here, ≈5 s of the 4.7 s regression on its own) — separate the
  two before ranking anything.
- Normalize per output token: the two runs did **not** emit the same text
  (#8967: 39 146 vs 43 274 generated tokens, min completion 548 vs 364). A
  length/termination change is not the MTP mechanism; say which number survives
  normalization and reject the cross-run comparison as stated.

## Step 3 — decide whether a gain is even possible on this model class

Spec decode pays only when the step is **weight-bandwidth bound**, because the
weights are read once for `1 + K` tokens. Price that term from the paste:

```text
weight term per step ≈ (active params × bytes/param) / HBM bandwidth
```

#8967's own numbers: 35B-A3B w8a8 ⇒ ≈3 GB active weights per step, ≈2 ms at the
card's ~1.6 TB/s class, while the measured plain step is **14.8 ms**. The step is
therefore dispatch/latency bound (MoE permute + grouped matmul, attention, sampler,
launch overhead), not weight bound — and everything in that remainder scales with
the number of verified tokens. That is why 4 tokens cost 3.62× instead of 1.2-1.5×,
and why the sibling models behave differently: Qwen3.5-27B is dense (the weight term
dominates its step, so MTP amortizes) and the 122B-A10B carries a much larger active
weight read. The same arithmetic is what `qwen3-next-decode-throughput.md` uses to
call a 190 ms step host-dispatch bound.

So the first verdict sentence is usually: **"MTP is working (acceptance ≈3.3/4) and
the loss is the model class at concurrency 1, not a config error"** — then the
measurement that proves it (step 4). Keep a secondary mechanism explicitly labelled
as a hypothesis: on hybrid GDN families the linear-attention state update is
token-serial, so the verify tokens cannot be batched through those layers at all.
#8967 could not confirm the layer mix locally (the snapshot's `transformers` export
carried no `qwen3_5` config), so it stayed a hypothesis there.

## Step 4 — the slope experiment, not a code walk

1. **`num_speculative_tokens` 3 → 2 → 1 → off**, same 28 requests. If the step wall
   falls roughly linearly with the token count while per-step tokens fall with it
   too, the cost is token-additive and the conclusion is settled with no source read.
   This is the one experiment the report is owed.
2. **Concurrency > 1** (`--max-num-seqs`, a multi-prompt benchmark). Single-stream
   decode is the worst case for an A3B-class MoE; the aggregate case is where spec
   decode is normally claimed to win, and the reporter's table is `Concurrency 1.0`.
3. Then the launcher rows: drop the spec-scoped `"enforce_eager": true` (it only
   affects the **drafter** graph — step 5), and re-run with `PIECEWISE` if
   `FULL_DECODE_ONLY` is the only mode tried.
4. Recipe provenance: #8967's spec block is line-for-line
   `docs/source/tutorials/models/Qwen3.5-27B.md:119` copied onto the 35B-A3B, and the
   pin has **no** `Qwen3.5-35B` page (only 27B and 397B-A17B). A recipe validated for
   a dense sibling is an unverified variable on an A3B, not a config error — check the
   tutorial list at the pin before saying either way.
5. On TP=1/DP=1 the launch's `--enable-expert-parallel` and
   `VLLM_ASCEND_ENABLE_FUSED_MC2=1` cannot matter: `ep_size` is 1, which routes to
   `AllGather`/`NaiveMulticast` and never MC2 — one row of the selector table in
   `moe-comm-path-when-ep-off.md`, not an ablation worth a launch when the budget is
   short.

## Step 5 — the two code stories that are refutable in one grep each

Both were chased and closed in #8967; take them as results, not as a walk.

| Story | Result at vllm-ascend v0.19.1rc1 / vllm v0.19.1 | Grep |
|---|---|---|
| "`qwen3_5_mtp` is not a method the downstream dispatch knows, so the `self.method == "mtp"` branches are skipped" | **refuted**: upstream normalizes every `MTPModelTypes` member to `"mtp"` with `method … is deprecated and replaced with mtp` (`vllm/config/speculative.py:367-371`; the literal at `:34-47` includes `qwen3_5_mtp`, `qwen3_next_mtp`, `deepseek_mtp`), so `get_spec_decode_method` takes its `("eagle","eagle3","mtp")` branch (`vllm_ascend/spec_decode/__init__.py:34-47`) and the downstream `mtp` code does run | `rg -n 'MTPModelTypes\|replaced with mtp' vllm/config/speculative.py` |
| "`enforce_eager` inside `--speculative-config` turned off the target's decode graph, so every step is eager" | **refuted for the target**: the drafter's graph is `self.runner._use_aclgraph() and not self.speculative_config.enforce_eager` (`vllm_ascend/spec_decode/eagle_proposer.py:144`); the runner's gate reads only `model_config.enforce_eager` (`worker/model_runner_v1.py:513-519`). It can make the **drafter** slower (eager MTP layer + full-vocab lm_head per step) — keep it as a rung, not as the mechanism | read both sites; re-grep per version |

## Step 6 — delivery

Verdict sentence with the three ratios (step ratio, token ratio, per-token ratio),
then one line saying what the headline percentage actually is (E2EL vs per-token),
then the slope experiment as rung 1, then the refutations as *closed* lanes so the
reporter does not repeat them. Ask for the `SpecDecoding metrics:` line and a
concurrency>1 run; those two turn the arithmetic into a measurement.

Scope: pins above (vllm-ascend `da421afad719`, vllm `b1388b1fbf5a`); the reporter was
on vllm 0.18.rc1, so every source claim needs the one-command recheck on their tree.
The gain model is a bandwidth/dispatch argument, not a measured profile — say so.
