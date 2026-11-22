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

## No server metric block: the client's own bench-serve table

Second shape, same complaint ("TTFT 极高") but the only evidence is the
summary `vllm bench serve` itself printed. Worked case **#3112**: vllm-ascend
`v0.10.2rc1` (`048bfd5`), vllm 0.10.2 (that upstream tag is **not** in the
snapshot — the pinned vllm is the v0.9.1 export, `dated_reference_only`, so quote
the tool's semantics from that sibling and the arithmetic from the paste),
Qwen3-8B bf16 on **one 910B4 (32 GB)**, `--enable-prefix-caching`,
`vllm bench serve --dataset-name random --random-input-len 1024
--random-output-len 2048 --request-rate 10 --num-prompts 200`. Pasted block:
duration 8206.37 s, 204065 input / 369494 output tokens, output 45.03 tok/s,
mean/median/P99 TTFT 3754758.86 / 3593202.68 / 7717673.05 ms, mean/median TPOT
596.96 / 457.58 ms, mean/median ITL 580.53 / 42.91 ms.

Two properties of the driver decide the answer before any source:

- **The arrival rate is open loop and the prompt count is fixed.** `--request-rate`
  sleeps a gamma interval between sends, `burstiness=1.0` → exponential
  (`vllm/benchmarks/serve.py:84-121` at the v0.9.1 export), so 200 requests are
  offered over ~20 s whether or not the engine can serve them.
- **TTFT is measured from before the send**, `ttft = time.perf_counter() - st`
  (`vllm/benchmarks/endpoint_request_func.py:124-125`, same at `:226-228`,
  `:337-339`), so it contains the whole queue wait; ITL/TPOT are pooled across
  successful requests (`serve.py:151-179,231-234`).

Then three derived numbers, one call:

```bash
uv run --no-project python <skill_root>/scripts/benchmark-table-normalize.py \
  --run '3112 45.03 8206.37 200' --total-tok-s 3112=69.89 --ttft 3112=3754.75886 \
  --mean-itl-ms 3112=580.53 --mean-tpot-ms 3112=596.96 --median-itl-ms 3112=42.91 \
  --arrival-rate 3112=10
```

- `served = requests / wall` = **0.024 req/s against 10 offered → 410×
  oversubscribed**, and `mean TTFT / wall = 0.458` — about half the run, which is
  exactly one FIFO drain: each request waits for its predecessors' 1848-token
  decodes. On that evidence the TTFT is **queue time**, full stop; it is not a
  prefill latency, and no prefill op / `--max-num-batched-tokens` question is on
  the report.
- **In-flight sequences = output tokens × mean ITL / wall** = 26.1 (26.9 by
  `output tok/s × mean TPOT`) → per stream 1.7 tok/s. Use the **mean** columns:
  the median ITL 42.9 ms gives C ≈ 2, *13× low*, and a run that trusts it starts
  proposing KV-capacity and `--gpu-memory-utilization` fixes for a queue whose
  length has nothing to do with them. (The recorded run computed C ≈ 2 from the
  median on its 4th assistant turn and retracted it in its 13th and last, after a
  full scheduler walk.)
- `mean / median ITL = 13.5×` says a minority of gaps carry the wall time, i.e.
  most steps emit **no decode tokens**. That is the second finding, not the TTFT
  answer.

Why decode tokens arrive in bursts at this pin (check before generalising): a
default **non-MLA** dense serve logs `Non-MLA LLMs forcibly disable the chunked
prefill feature …`, rewrites `max_num_batched_tokens = max_model_len` (40960
here) and swaps in `AscendSchedulerConfig` — `vllm_ascend/platform.py:136-159`
(the rewrite at `:156-159`) and `:264-271`; `vllm_ascend/core/schedule_config.py:26,44`
(`enable_chunked_prefill = False`). `AscendScheduler.schedule()` is
"prefill-first" (`core/scheduler.py:37-38`) and reaches the decode loop **only
when no prefill was scheduled that step** (`:297-301`). Consequences: the
`Chunked prefill is enabled with max_num_batched_tokens=%d` line the #6833 section
tells you to quote **does not exist on such a serve — do not ask for it**, and the
mixed-step model above does not apply. `vllm_ascend/core/` does not exist at
v0.9.x; re-grep both files per ref.

Order of the reply: (1) queue verdict, offered-vs-served, implied in-flight count
from the means; (2) the two capacity numbers to request — the `Running` /
`GPU KV cache usage` lines (26 sequences × 3072 tokens ≈ 11 GB of a 32 GB card is
KV-limited admission at `--max-model-len` default) and whether the non-MLA line is
in the startup log; (3) then the levers below. The cheapest change is the offered
work: 200 × 2048 output tokens is ~2.5 h at 45 tok/s on any card of that class.

## Levers, and what each one does *not* fix

| Reporter's complaint | Lever | Effect | Note |
|---|---|---|---|
| TTFT / 排队 | total prompt tokens per wave (client concurrency, prompt length, prefix reuse) | linear in prompt tokens | prefix reuse is unavailable on this model at this version — see the sibling page |
| TTFT on an arrival-rate table, offered rate ≫ capacity | total **output** tokens requested (`--random-output-len × --num-prompts`) and `--request-rate` | the wall time is `output tokens ÷ aggregate rate`; TTFT ≈ half of it | the queue is decode-bound here, not prefill-bound; the lever is the load, then capacity |
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

Do not ask for a `plog`. For the **metric-block** shape do not open
`vllm_ascend/core/*scheduler*` or the `patch_balance_schedule.py` diff — the
recorded run spent 19 assistant turns inside the hybrid-KV-cache and
prefix-caching sources and shipped no answer, although the ratios above were
already computed from the paste at turn four. For the **bench-serve** shape the
two identities above are the answer and the only source read allowed is the
decode gate of the scheduler actually in force (`AscendScheduler.schedule()`'s
`if len(self.scheduled_req_ids) == 0` at the reported pin); the recorded #3112 run
spent its last five turns in `platform.py`, the KV pool and the operator wrappers
and ended at the 16 384-token cap with **no verdict**: 13 assistant turns, 763
characters of text, every one of them a plan to answer next turn.
