# Two models on one box: the expected decode ratio before any flag

## Scope

Same machine, same image, same release, **two different models**, and a table of
rates — "910B 部署 <large model> 为什么推理很慢, 20k 上下文不到 25 t/s, 比 <smaller>
的还慢, 是部署参数问题吗" — usually a screenshot with context length and concurrency
as its axes.

Not `cross-run-ratio-normalization.md`: there the two sides are the *same* model
across a platform, image or release, and the job is to recover a rate from mixed
totals. Here the second model is a **valid control** (same cards, same engine, same
flags), so the question is what ratio the bytes predict — and whether the slow side
is below its own roofline at all.

## Step 1 — TTFT and t/s are two axes; keep the per-row ratio

Recorded #11260 (image `v0.18.0` = the pinned `e18643f8a4d5`, 2×910B 64G, TP2,
`--max-model-len 262144`, `--gpu-memory-utilization 0.90`, both tables at mtp=2,
values off an OCR'd screenshot):

| rows | 35B-class model | 27B-class model (control) |
|---|---|---|
| t/s, 20k / 80k / 200k, 1 concurrency | 22 / 18.9 / 14 | 48 / 40 / 27 |
| TTFT, same rows | 7 / 12 / 96 s | 7 / 11 / 95 s |
| step = `1000 / t_s` | 45.5 / 52.9 / 71.4 ms | 20.8 / 25.0 / 37.0 ms |
| t/s, 2 concurrency | 41.5 / 35.8 / 27 | 96 / 75 / 52 |

TTFT matches within a few percent while t/s differs 2.1-1.9× across every row:
**prefill is not the complaint**, so `--max-num-batched-tokens`, chunked prefill and
prefill op coverage are off this report (they belong to the TTFT axis, steps 1-2 of
`cross-run-ratio-normalization.md`). Quote the ratio per row, never the raw t/s.

## Step 2 — the paste's own concurrency row closes the scheduler lane

Per-stream step at concurrency 2 = `2 / aggregate_t_s` × 1000: 41.5 → **48.2 ms**
against 45.5 ms at concurrency 1, i.e. double the batch for ~6 % more step wall
(control: 20.8 → 20.8 ms). A step that barely moves when the batch doubles is
**weight-byte bound**, and it retires the queue/KV/`--max-num-seqs`/preemption lanes
for this report. `Available KV cache memory: %.2f GiB`
(`vllm_ascend/worker/worker.py:358` at v0.18.0) enters only when the per-stream step
*did* rise under concurrency — the capacity shape in
`kv-capacity-step-and-preemption.md`, not this one.

## Step 3 — price the step in bytes, with the control as calibration

```text
bytes read per decode step ≈ params × bytes_per_param / TP
step floor                ≈ bytes / card HBM bandwidth class
```

910B 64G is the ~1.6 TB/s class (same figure `qwen3-next-decode-throughput.md` and
`spec-decode-gain-accounting.md` use). #11260: 35 B bf16 / TP2 = 35 GB → ~22 ms floor
against 45.5 ms observed (~48 % of class); control 27 B bf16 / TP2 = 27 GB → ~17 ms
against 20.8 ms (~80 %). Say the assumptions out loud — bf16 = 2 bytes/param, the
bandwidth is the *class* figure, and a 32 G / 910B3/B4 or a bf16-vs-w8a8 mix moves
every number.

Then the arithmetic that decides the ranking: the nominal size ratio is 35/27 = 1.30×
but the observed ratio is 2.1×, so "it is just a bigger model" does **not** close it.
Two bytes-level readings remain, and they need different answers:

- **dtype asymmetry** — the control's checkpoint quantized (w8a8/fp8 halves bytes:
  13.5 GB vs 35 GB = 2.6×) and the slow one not.
- **effective roofline** — same nominal bytes, half the achieved bandwidth: an
  MoE whose per-expert GEMMs are tiny at batch 1-2, an unfused decode path for that
  architecture, or a per-step host/launch term.

Both are settled by the artifacts in step 4, not by a source walk.

## Step 4 — the one artifact that separates them: both startup logs

- `Model loading took %s GiB memory and %.6f seconds`
  (`vllm/v1/worker/gpu_model_runner.py:4567` at vllm v0.18.0 `bcf2be96`, logged once
  per rank) — the *loaded* weight footprint. Divide by the step from step 1: GB/s per
  model. Equal GB/s on both rows ⇒ nothing is misconfigured, the only lever is a
  quantized (or smaller) checkpoint; half the control's ⇒ the decode path is the story.
- `config.json` of both: `quantization_config`, `torch_dtype`, `model_type`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
  `num_experts` / `num_experts_per_tok`, and the MTP/drafter head key if the
  checkpoint carries one.
- `Available KV cache memory` from the same startup (and later the `GPU KV cache
  usage` metric), only for the capacity branch of step 2.

Ask for exactly these in the first answer. They are one paste each and they close
dtype, MoE-vs-dense and "is it my flag" simultaneously; no local tree contains them.

## Step 5 — suspect the table before the engine

- **The command and the table disagree.** #11260's table is labelled `mtp=2` while
  the pasted `vllm serve` line carries no `--speculative-config` and no
  quantization flags: the rows came from a different launch than the one shown. Ask
  for the command behind each row before ranking `--speculative-config` levers; once
  it is real, price K=2 with `spec-decode-gain-accounting.md` (a verify pass per step
  plus acceptance below the budget shows up as t/s that do not scale with
  `1 + K` tokens/step).
- **`--served-model-name` naming another model** is harmless as a label but means a
  row can be benchmarked against the wrong endpoint — the same trap recorded at
  #4357 (`qwen3-next-decode-throughput.md`). One `/v1/models` plus the startup banner
  per row pins which weights answered.
- **`--max-model-len` is a capacity number, not a rate.** 262144 on a 200k-context
  table only matters if the per-stream step rose under concurrency (step 2); at
  2-conc × 200k the pool must hold ~400 K tokens. Otherwise say so in one line and
  leave it out of the ranking.

## Step 6 — an unknown model name is a gap, not a lane

In #11260 neither model name hit the pinned tree or the prior-month corpus. That is
not evidence of "unsupported", and it does not license a registry hunt: the ranking
stays per config field from step 4, and the reporter owns the identity. If the
architecture still matters, the check is theirs to run — the `architectures` value
from `config.json` grepped against `vllm_ascend/models/` and against the pinned
`docs/source/tutorials/models/` list — and a miss is a one-sentence *unverified path*
caveat, not a mechanism.

## Delivery — this lane's recorded failure is the write-up, not the arithmetic

#11260 spent its entire 16 384-token output budget inside four reasoning blocks that
re-derived the same 45.5-vs-20.8 ms step arithmetic, said "now write the answer", and
shipped **zero** message text (`output_truncated`); the ranking above was complete by
its second block. One arithmetic pass plus the ranking *is* the deliverable here,
because every remaining discriminator lives on the reporter's machine. Concretely:
carry the verdict line in the same message as each tool call, and when a block starts
re-pricing a step it has already priced, emit the answer instead
(`../evidence-workspace/delivery-discipline.md` — G1/G2/G3 and the reasoning-budget
arithmetic).
