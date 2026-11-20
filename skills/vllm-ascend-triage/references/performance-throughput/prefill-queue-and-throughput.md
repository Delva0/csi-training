# "Requests queue for so long" and "generation throughput is single digits"

Worked case #6833 (vllm-ascend v0.14.0rc1 + vllm v0.14.1, Qwen3-Next-80B-A3B on
4×910B4, 40 concurrent chat requests): the reporter pasted seven 10 s metric
samples and asked why requests queue so long and why the prefix-cache hit rate is
0. Two of the four numbers in that block answer the queueing half; the cache half
is a capability question (`zero-prefix-cache-hit-hybrid.md`).

## The block, as pasted

| t | prompt tok/s | gen tok/s | Running | Waiting | KV % |
|---|---|---|---|---|---|
| 09:21:20 | 2457.5 | 355.5 | 5 | 35 | 5.8 |
| 09:21:30 | 8192.2 | 8.5 | 15 | 25 | 17.3 |
| 09:21:40 | 6553.4 | 14.0 | 23 | 17 | 26.4 |
| 09:21:50 | 5734.5 | 17.5 | 30 | 10 | 34.5 |
| 09:22:00 | 4915.2 | 18.9 | 36 | 4 | 41.4 |
| 09:22:10 | 4913.7 | 38.5 | 40 | 0 | 46.1 |
| 09:22:20 | 0.0 | 1039.6 | 40 | 0 | 47.9 |

## What the ratios say (do this before reading any scheduler source)

- `Waiting` 35 → 0 in ~50 s and drains *monotonically*; `GPU KV cache usage` peaks
  at 47.9 % and never approaches 100 %; no `Preemptions` line appears anywhere
  (`vllm/v1/metrics/loggers.py:230-233` prints it only when non-zero). Capacity,
  eviction and OOM are all out — do not offer `--gpu-memory-utilization` or
  `--max-num-seqs` as fixes: `Running` never reached the 256 default for the API
  server at v0.14.1 (`vllm/engine/arg_utils.py:1856-1866`).
- Integrating prompt throughput over the six 10 s samples gives ≈ 328 K tokens in
  ~60 s, i.e. ≈ 5.5 K tok/s of prefill and ≈ 8.2 K prompt tokens per request, and
  the observed `Waiting` drain (35 → 0) took 50 s. TTFT here is total prefill work
  ÷ prefill throughput — the queue is prefill-bound, and that is the answer to
  "为什么排队那么久". Sanity-check the per-request prompt length against the load
  driver before quoting the number.
- Per-request generation rate is the second finding:
  `gen tok/s ÷ Running` = 0.57, 0.61, 0.58, 0.53, 0.96 during the prefill window,
  then 26.0 after it. The engine did not get faster at 09:22:20; the prefill work
  competing for the same step left. The first sample (355.5 / 5 = 71 tok/s) is the
  healthy decode rate when no prefill is in flight, so 26 tok/s at 40 concurrent
  is normal batching, and 0.5-1 tok/s is the interference.
- One scheduler step explains it: running requests are scheduled first (one token
  each), then the remaining token budget is filled with prefill chunks
  (`vllm/v1/core/sched/scheduler.py:271,281-283` at vllm v0.14.1). While prefills
  are in the batch, step latency ≈ prefill chunk ÷ prefill rate ≈ 1 s, so every
  running request advances ~1 token per second and `gen tok/s ≈ Running ÷ step
  latency` (40 ÷ 1 s = 40 tok/s, matching the 38.5 sample). Prompt tok/s ≈
  `max_num_batched_tokens ÷ step latency` — the same two numbers, one step.
  Quote the startup line
  `Chunked prefill is enabled with max_num_batched_tokens=%d`
  (`vllm/config/scheduler.py:229-232`) instead of assuming the default: it is the
  ceiling that makes those two numbers consistent.

## Levers, and what each one does *not* fix

| Reporter's complaint | Lever | Effect | Note |
|---|---|---|---|
| TTFT / 排队 | total prompt tokens per wave (client concurrency, prompt length, prefix reuse) | linear in prompt tokens | prefix reuse is unavailable on this model at this version — see the sibling page |
| TTFT / 排队 | `--max-num-batched-tokens` up (e.g. 8192) | slightly higher prefill throughput (bigger matmuls), fewer, longer steps | inter-token latency for already-running requests gets *worse* |
| inter-token latency during prefill | `--max-num-batched-tokens` down (the project's Qwen3-Next tutorial uses `4096`, `docs/source/tutorials/Qwen3-Next.md:64` at v0.14.0rc1) | shorter steps, decodes advance more often | total prefill time grows a little |
| both, at scale | P/D disaggregation (prefill and decode on separate pools) | removes the interference by construction | topology cost; `../pd-disaggregation/index.md` |
| throughput per card | TP/EP layout, `FULL_DECODE_ONLY` graph vs `--enforce-eager`, quantization | the real capacity answer | diff against the tutorial command above, not against the vendor blog |

## Artefacts to request, and the one to ask for first

1. The serve command plus the `Chunked prefill is enabled ...` /
   `Setting attention block size ...` startup lines — the budget and block size
   in force.
2. The load driver's prompt lengths (`usage.prompt_tokens` of a sample request, or
   the benchmark's `--random-input-len`), because the whole TTFT arithmetic is
   per-token.
3. Whether the client sends one wave of N requests or a steady arrival rate: the
   recorded block shows a single wave (Waiting 35 at the first sample).

Do not ask for a `plog`, and do not open `vllm_ascend/core/*scheduler*` or the
`patch_balance_schedule.py` diff for this shape — the recorded run spent 19
assistant turns inside the hybrid-KV-cache and prefix-caching sources and shipped
no answer, although the ratios above were already computed from the paste at turn
four.
