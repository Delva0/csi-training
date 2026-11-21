# Scheduler admission: requests that never get scheduled

## Scope

The engine is up and its frontend answers, but requests stop finishing: one
request waits forever and everything behind it waits too, or a class of requests
(long prompts, a second concurrent request, one bigger batch) is never admitted
while short ones are served. There is no traceback and no device fault — the
scheduler simply decided not to schedule, and nothing in the log says so.

This topic owns the **admission decision**: which request gets blocks from the KV
pool, what happens to the queue when one request cannot have them, and the
counters that show it. Throughput/queue arithmetic on an engine that keeps
draining is `../performance-throughput/index.md`; a worker that stopped
answering mid-collective is `../hccl-and-hardware/index.md`.

## Common rules

- Inherit every rule in `SKILL.md`.
- **Decide wedged vs slow from the engine's own counters before any source read.**
  `vllm:num_requests_running`, `vllm:num_requests_waiting_by_reason{reason="capacity"|"deferred"}`
  and `vllm:kv_cache_usage_perc` (`vllm/v1/metrics/loggers.py:494-521, 1109-1125`
  at vllm v0.26.0). `Running` = 0 with `Waiting` ≥ 1 and KV usage near 0 is an
  admission state, not a compute one; `Running` > 0 with a queue that drains is
  the throughput topic. This one call replaces reading the scheduler.
- **Read the pool's capacity from the startup lines, not from the flags.** The
  engine prints `GPU KV cache size: N tokens` and
  `Maximum concurrency for M tokens per request: kx`
  (`vllm/v1/core/kv_cache_utils.py:2177-2183` at the same pin); a `k<1` line
  already says one full-length request cannot be admitted, whatever
  `--gpu-memory-utilization` and `--max-model-len` claim.
- **Label the claim.** A mechanism verified at the pinned ref explains the
  *shape* of the report; which trigger is live in the reporter's build stays a
  hypothesis until the pool number or the metrics block says so. Two builds of the
  same vLLM version cannot differ in the gate itself — the differing variable is
  pool size or per-request demand.
- Never propose `--max-num-seqs` or a memory bump before the pool number is known:
  admission failure is a per-request demand question first.

## Pages

- `waiting-queue-wedge.md` — the `full_sequence_must_fit` gate, the waiting-loop
  `break` that head-of-line blocks the queue, the compressed-MLA / recycling
  admission cap downstream re-applies, and the checks that separate "the pool can
  never hold this request" from "the abort never arrived".
