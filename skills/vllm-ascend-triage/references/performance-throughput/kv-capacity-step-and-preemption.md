# A step in inter-token latency at a concurrency number: the KV pool, priced by recompute

## When this page applies

TPOT / inter-token time jumps 2-5× at one *offered concurrency*, nothing errors and
nothing OOMs — recorded #5859 (vllm-ascend 0.11.0rc2 image, single 8-card A2 64G,
GLM-4.7-W8A8, 32K-in / 1K-out):

| concurrency | TPOT | Running | Waiting | GPU KV cache usage |
|---|---|---|---|---|
| 6 | 107 ms | 6 | 0 | 79.4 % |
| 7 | 114 ms | 7 | 0 | 92.4 % |
| 8 | 440 ms | 8 | 0 | 98.9 % |
| 12 | 423 ms | **8** | 4 | 98.7 % |

This is the same *shape* as `sequence-count-cliff-capture-size.md` (a fixed-size
structure crossed at one concurrency) with a different owner, and **the KV
utilization at the step is the discriminator**: 81-84 % at the collapse (#14440)
leaves the graph ceiling as the candidate, ≈100 % at the step makes it the KV pool.
Read both, decide from that one column, and do not walk the capture list for a
capacity report (nor the eviction path for a graph report).

## Do this first: three numbers from the reporter's own rows

Per-request footprint = input + output tokens (one request holds its whole prompt
plus what it generates). Then

    pool tokens ≈ footprint × running / usage

computed from **two** rows; their agreement is the finding. #5859: 33 792 × 6 /
0.794 ≈ 255 K and 33 792 × 7 / 0.924 ≈ 256 K, so the pool holds ≈256 K tokens, one
32K request is ≈13 % of it, and the capacity ceiling is 256 K / 33.8 K ≈ **7.5**
requests. The 8-request row cannot fit, and the 12-request row states the ceiling
directly: `Running` pinned at 8, `Waiting` 4, 98.7 %. So the step is a capacity
wall — going 7 → 8 adds one request's compute, not 4× of it.

Ask for the two startup lines that already carry the same numbers:
`GPU KV cache size: N tokens` and
`Maximum concurrency for <max_model_len> tokens per request: X.XXx`
(vllm v0.13.0 `72506c9`, `vllm/v1/core/kv_cache_utils.py:1291,1297`; the ratio is
block arithmetic over `max_model_len`, `:778-796` — so it is a divisor for the
*reported* length, not the admission count).

## The mechanism, pinned

| Site | Fact |
|---|---|
| `vllm/v1/core/sched/scheduler.py:316-362` (vllm v0.13.0) | `kv_cache_manager.allocate_slots(...)` returning `None` has exactly one handler: remove the lowest-priority (or last) running request and preempt it |
| `scheduler.py:759-781`, the assignment at `:775` | `_preempt_request` frees the request's blocks and sets **`request.num_computed_tokens = 0`** — V1 has no swap, so the whole prompt is re-prefilled |
| `vllm/v1/metrics/loggers.py:221-223` | `Preemptions: N` is appended to the metric line **only when non-zero** — a pasted block without it is not evidence there were none |
| `vllm_ascend/core/scheduler_dynamic_batch.py:249-259` (vllm-ascend v0.13.0rc1 `1b5d5ab`) | the downstream SLO fork repeats the same `num_computed_tokens = 0`, so the reading survives a scheduler swap; it is active only with `additional_config.SLO_limits_for_dynamic_batch != -1` (`vllm_ascend/platform.py:313-319`, default `-1`) |
| `vllm_ascend/utils.py:420-437` | the default capture list is `[1,2,4] + range(8, min(max+1,256), 8) + …` — **8 is captured**, so an 8-sequence cliff with `num_speculative_tokens = 0` is not a capture ceiling |

Why a capacity wall becomes a 4-5× TPOT rather than a slow admission: the
preemption discards the prefill of a full-length prompt, that re-prefill is
scheduled in `max_num_batched_tokens`-sized chunks, and each such step leaves the
resident decodes almost no token budget — the survivors' per-token time multiplies
for the duration of the re-prefill, then returns. State it as the mechanism and
name the log line that proves it.

## Checks, cheapest first

1. `rg -n 'Preemptions|GPU KV cache size|Maximum concurrency' <serve.log>`. A
   `Preemptions` count that appears at the stepping concurrency and keeps growing
   closes it; `Maximum concurrency … X.XXx` should land between N-1 and N.
2. Ablate **capacity**, not the graph: cap the client (or `--max-num-seqs N-1` as an
   admission guard) at ceiling − 1 and re-run that one row. Prediction for #5859:
   TPOT back to ≈114 ms. `--enforce-eager` is not this ablation — it deletes the
   graph and moves the baseline.
3. Then the capacity levers: `--gpu-memory-utilization` up in small steps with a
   same-host check that no other process holds HBM; `--max-model-len` down to what
   the workload needs (it is the divisor in the pool arithmetic above); prefix
   caching when the long prompts share a prefix (`zero-prefix-cache-hit-hybrid.md`
   first — on a hybrid model the switch may not exist).
4. If the ask is an SLO guarantee rather than a diagnosis, the downstream
   `SchedulerDynamicBatch` (`--additional-config '{"SLO_limits_for_dynamic_batch": N}'`,
   `docs/source/user_guide/feature_guide/dynamic_batch.md` at v0.13.0rc1) is a
   decode-first chunking scheduler with a cost table — but that page scopes it to
   dense Qwen/Llama 8B-32B on A2 with `BLK128`, so on a MoE model it is an
   experiment, not a recommendation.

Do not offer `--max-num-seqs` as the *fix*: on this shape the ceiling is KV blocks,
not the flag — the 12-row's `Running: 8` is that ceiling showing itself.

## Traps

- "Success of the waiting mechanism" is not a contradiction. Admission limits entry;
  preemption is what the **resident** requests pay. A report that says "排队有效，但
  保不住服务性能" is this page.
- `Waiting: 0` at the step does not exonerate capacity: 8 offered and 8 running with
  a full pool queues nothing and still has nowhere to allocate.
- The `×(1 + num_speculative_tokens)` multiplier in the sibling page's budget
  collapses to ×1 when MTP is unavailable on that image. GLM-4.7 is such a model:
  the original reporter states (2026-01-15, on #5859) that enabling MTP needs the
  latest main branch and that releases 0.11-0.13 run it **without** MTP. A
  reporter-side claim, not source-verified — but it retires the graph candidate on a
  released image for that model.

## Recorded status

- The arithmetic and the five pinned sites: source-verified at vllm v0.13.0
  (`72506c9`) / vllm-ascend v0.13.0rc1 (`1b5d5ab`).
- **Not reporter-confirmed.** #5859's later reporter message answered only the
  checkpoint-provenance and MTP-version question; the model's own GLM-4.7 row is
  absent from that pin's support matrix (only GLM-4.5 is listed), so nothing about
  its MLA/MTP configuration was pinned there either. Check 1 above is what settles
  the case; keep the capture and mixed-prefill candidates open until it runs.
