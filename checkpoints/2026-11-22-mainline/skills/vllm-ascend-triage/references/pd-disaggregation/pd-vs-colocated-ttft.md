# PD TTFT versus the same model's colocated/multi-node TTFT: which term doubles

Worked case: #7182 — "**[Bug]: DSv3.1, the TTFT of Prefill-Decode Disaggregation is 2
times longer than TTFT of Multi-node Deployment**", vllm-ascend v0.13.0 (`6281c12`) /
vllm v0.13.0 (`72506c9`), both A2 and A3, "you can check the performance report of any
project … but the typical model of qwen no this issue". No log, no metric block, no
serve commands, empty `collect_env` block.

## Scope

A **measurement/comparison** report, not a fault lane: two deployments of one model
(PD vs colocated multi-node) and one number each, nothing crashed. The answerable
question is never "is PD slower" — that is expected by construction — but **which term of
the chain doubles**, and whether the two runs are even the same configuration.

Not this page: a KV-pool cold/hit latency split (`../kv-pool-store/pool-attached-cold-request-cost.md`);
the `LLMDataDistCMgrConnector` "fail to receive KV Cache transfer done signal in 120s"
lane (`llmdatadist-done-signal-and-ttft.md`); a version-to-version regression
(`../performance-throughput/upgrade-vs-older-baseline.md`); general rate normalization
of a pasted benchmark table (`../performance-throughput/cross-run-ratio-normalization.md`).

## Step 1 — the chain, typed into the first message

Client TTFT in a PD deployment = P queueing + P prefill + KV transfer + D's first decode
step + the proxy hop. The project recorded the structural part itself: **#3099**
(`[Bug]: PD分离下首token由D节点返回，TTFT偏高`, DS-R1 W8A8 MTP float, 2P1D, 0.9.1-dev) —
the PD proxy returns the first token only after **D** has produced one, so a PD TTFT
above the colocated number is by construction. Quote that, then say the report as filed
is a comparison without numbers, and that the deliverable is the two metric blocks plus
the per-term split. That is a second-turn answer; no source walk is owed before it.

## Step 2 — the pinned tree's own PD number is the baseline

At v0.13.0 the DeepSeek-V3.1 tutorial carries the project's own measurement
(`docs/source/tutorials/DeepSeek-V3.1.md:692-706`): A3-752T, 4 nodes, **2P1D**, P
`DP2+TP8`, D `DP32+TP1`, input/output **3.5k/1.5k**, **TTFT = 6.16 s**, TPOT = 48.82 ms,
478 TPS per card. Use it as the expected-value anchor on two axes:

- `TTFT / input_tokens` (prefill-side rate) — ≈ 568 prompt tok/s end-to-end for the
  reference point;
- `TTFT_PD − TPOT_D ≈ P prefill + transfer` (D's first step costs about one TPOT).

If the residual is ≈ the colocated TTFT, the gap is on the **prefill side**; if it is a
constant offset that shrinks in relative terms as the prompt grows, it is the **hop**.
This is the same two-axis split as
`../performance-throughput/cross-run-ratio-normalization.md`, applied to PD terms.

## Step 3 — "same model" is not "same configuration" (rank 1)

In that same file the PD recipe and the colocated blocks are *not* the same flags, so a
report that compares them is comparing two configurations:

| | PD prefill node | PD decode node | colocated / multi-node blocks |
|---|---|---|---|
| graph / async scheduling | `--enforce-eager` (`:319`, `:397`) | `--async-scheduling` (`:478`) + `FULL_DECODE_ONLY` (`:480`) | `--async-scheduling` (`:121`, `:189`, `:244`) + `FULL_DECODE_ONLY` (`:129`, `:197`, `:252`) |
| admission | `--max-num-seqs 8` (`:318`) | `--max-num-seqs 28` (`:473`) | `--max-num-seqs 16` (`:122`, …) |
| chunk budget | `--max-num-batched-tokens 16384` (`:317`) | `256` (`:472`) | `4096` (`:124`, …) |
| spec tokens | `num_speculative_tokens 1` (`:324`) | `1` (`:479`) | `3` (`:128`, …) |

TTFT is spent on the prefiller's window, and the documented PD prefiller is the
graph-less, async-scheduling-less side. So before any transport theory: **A/B one flag
on P** — add `--async-scheduling`, then replace `--enforce-eager` with the
`--compilation-config` block — and report `Mean TTFT` on the same dataset. Also note the
D-side `--additional-config '{"recompute_scheduler_enable":true}'` (`:325`, `:403`):
when D's KV is short the request goes back to P and recomputes (`:585`), which puts a
**second full prefill** inside TTFT.

## Step 4 — rule out transport bytes with one line of arithmetic

MLA/KV-volume theories are the expensive wrong turn here. Per-token KV for
DSv3.1-class MLA is `(kv_lora_rank 512 + qk_rope 64) B × ~61 layers ≈ 35 KB/token`
(fp8); a Qwen3-class GQA model carries 4-8 KV heads × 128 × 2 (K and V) × 2 B × 64-94
layers ≈ 130-380 KB/token. The Qwen run would move **several times more** bytes, so a
bandwidth theory predicts Qwen worse — the opposite of "qwen no this issue". What *is* model-specific is
that DSv3.1's per-token prefill is cheap (MLA + W8A8 + EP), so any constant handoff term
is a bigger fraction of it. Label this arithmetic (configs, not a pinned source), and
check it the cheap way: **TTFT versus prompt length** (constant offset ⇒ hop;
proportional ⇒ prefill), plus the concurrency sweep of Step 5.

## Step 5 — queueing is inside the metric

P's admission is `--max-num-seqs × DP` = 8 × 2 = **16 in-flight prompts cluster-wide**,
and the tree says the time spent waiting is counted in TTFT and that
`--max-num-seqs × --data-parallel-size` should be ≥ the total concurrency
(`docs/source/tutorials/long_sequence_context_parallel_single_node.md:114`). A benchmark
at 32/64 concurrency on 16 prompt slots can produce a 2× on its own. Sweep 1 / 8 / 16 /
32 on the same body: flat ⇒ per-request cost (Step 3 or the hop); rising ⇒ queueing.

## Step 6 — switches documented as TTFT-negative under PD

`VLLM_ASCEND_BALANCE_SCHEDULING=1` "may help increase output throughput and reduce TPOT …
However, TTFT may degrade in some scenarios. Furthermore, enabling this feature is not
recommended in scenarios where PD is separated"
(`docs/source/tutorials/DeepSeek-V3.1.md:135`). It is exported in the colocated launch blocks
(`:109`, `:173`, `:226`), so a reporter who carried the env line across is running a known TTFT-negative
switch on the PD side. Check `env` on both nodes before ranking anything else.

Prefix caching is *not* a difference in the documented recipes — `--no-enable-prefix-caching`
is on the colocated blocks too (`:126`, `:194`) — so it becomes a cause only if the
reporter's own colocated baseline dropped that flag. Ask, do not assume.

## Step 7 — the delivery

Verdict first: "as filed this is not yet a defect; PD TTFT above a colocated multi-node
TTFT is expected by construction (#3099) — the question is which term doubles", then the
chain, the pinned anchor, the ranked causes with one check each, the ladder (post both
metric blocks → normalize → concurrency sweep → one flag on P → prompt-length sweep →
only then a profiler trace), the scope (v0.13.0 `6281c12` / `72506c9`; flags are that
release's recipe), and the gaps (no numbers, no serve commands, no token counts, empty
`collect_env`).

Recorded run: the content above was reached but only at the last turn — seven
reasoning-only assistant turns first, then one 5.7 K-character answer whose `## Scope and
gaps` section was cut by the cap. The whole page was a two-turn answer from the pinned
tutorial plus #3099, with no `kv_p2p/` read at all.
