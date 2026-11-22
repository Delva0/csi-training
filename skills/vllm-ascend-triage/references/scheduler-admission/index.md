# Scheduler admission: requests that never get scheduled

## Scope

The engine is up and its frontend answers, but requests stop finishing: one
request waits forever and everything behind it waits too, or a class of requests
(long prompts, a second concurrent request, one bigger batch) is never admitted
while short ones are served. There is no traceback and no device fault — the
scheduler simply decided not to schedule, and nothing in the log says so.

The topic's second shape keeps the requests **running**: `num_requests_running`
stays > 0 while the generation and completion counters do not move at all, health
answers 200 and nothing is logged. That is not the throughput topic either —
nothing drains — so it is a scheduler/KV-*state* failure and its page is
`kv-block-lifetime-async-scheduling.md`.

This topic owns the **admission decision**: which request gets blocks from the KV
pool, what happens to the queue when one request cannot have them, and the
counters that show it. It also owns the crashes the scheduler raises while taking
that decision: `vllm_ascend/core/scheduler.py` reaching into upstream's
`KVCacheBlocks` (`scheduler-fork-block-index.md`), which is fork drift in origin and
a scheduling-decision failure in the report, and the engine dying inside the
**prefix-cache lookup** that same decision performs, whose deepest pasted frame is
upstream `kv_cache_manager.get_computed_blocks` → `coordinator.find_longest_cache_hit`
(`prefix-cache-lookup-crash.md` — a KV-coordinator-patch lane, not a fork;
`../upstream-downstream-drift/index.md` keeps the general fork method). Throughput/queue arithmetic on an engine that keeps
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
  **`Running` > 0 with `generation_tokens_total` and
  `request_generation_tokens_count` flat in every scrape is neither of those** —
  read it as a per-request KV-state question
  (`kv-block-lifetime-async-scheduling.md`) and take the two series' *deltas*, not
  their levels: a stuck rank's `Running` level looks healthy.
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
  admission failure is a per-request demand question first — **unless a downstream
  scheduling patch is enabled**, because there the flag itself can be the decision. Read
  the launch line for `VLLM_ASCEND_BALANCE_SCHEDULING` (and any other env that swaps
  `vllm.v1.core.sched.scheduler.Scheduler`) before reading the pool:
  `balance-scheduler-admission-stall.md` wedges *every* request at
  `--max-num-seqs 1`, and there `--max-num-seqs 2` is the answer.
- **A guard that reads shared state needs its writer checked first.** For any fork's
  admission condition over a cross-rank or process-level structure, prove the writer is
  reached in the configured topology (`data_parallel_size`/`dp_rank` gates a whole
  engine-core class in the balance patch) and that the initial value does not already
  satisfy the break. A zeros-initialised field plus a `>=` threshold is a
  capacity-one deadlock and an inert feature everywhere else, and the reporter's
  `balance_queue: [0]`-style capture is the confirmation, not a sampling artefact.

## Pages

- `waiting-queue-wedge.md` — the `full_sequence_must_fit` gate, the waiting-loop
  `break` that head-of-line blocks the queue, the compressed-MLA / recycling
  admission cap downstream re-applies, and the checks that separate "the pool can
  never hold this request" from "the abort never arrived".
- `scheduler-fork-block-index.md` — `IndexError: tuple index out of range` at
  `blocks = new_computed_blocks.blocks[0]` in `AscendScheduler.schedule()` (recorded
  #4207, v0.11.0rc0 image): the fork's own body runs whenever chunked prefill is off,
  which `AscendSchedulerConfig` forces (`schedule_config.py:48`), and the guard is one
  expression because the consumer already normalises empty (`scheduler.py:512`, `:178`).
  The pinned upstream cannot produce an empty tuple, so the producer is the reporter's
  vllm build — print it, do not derive it.
- `kv-block-lifetime-async-scheduling.md` — requests stay `Running` while generation
  stops, with no error and per-rank (recorded #13964, fixed by vllm-ascend #13518):
  the skip-block prune runs on `num_computed_tokens`, which async scheduling
  advances at *schedule* time, so SWA/chunked-local blocks can be freed and reused
  under an in-flight step. The pinned anchors, the `--async-scheduling` /
  `--speculative-config` ablations, why a self-recovering hours-long stall is a
  lifetime story, and the two collective/graph-sync theories the counters refute.
- `prefix-cache-lookup-crash.md` — the EngineCore dies under
  `EngineCore encountered a fatal error` with the deepest visible frame at
  upstream `kv_cache_manager.py:261 get_computed_blocks` →
  `coordinator.find_longest_cache_hit`, on a serve with more than one KV-cache
  group and `--enable-prefix-caching` (recorded #16760: DeepSeek-V4-Flash-Vision,
  8×910B3, `--block-size 32`, one request after `POST /start_profile`). Ownership
  (the Ascend coordinator + per-spec manager are what the cut frames hide), the
  four contract facts that exclude the arity / kwarg / pop-loop / cold-pool
  theories at the pin, the proof that `core.py step_with_batch_queue` implies async
  scheduling the printed command does not show, why `/start_profile` is not the
  trigger, and the two ablations that split the remaining lanes.
- `balance-scheduler-admission-stall.md` — `VLLM_ASCEND_BALANCE_SCHEDULING=1`
  admits nothing at `--max-num-seqs 1` (recorded #17953, v0.18.0rc1 `99e1ea0fe685`):
  the patch's waiting-loop guard `max(balance_queue) >= max_num_running_reqs - 1` over a
  zeros-initialised queue, the `data_parallel_size > 1 or dp_rank > 0` gate that keeps it
  from ever being refreshed on a DP=1 serve, the inertness above capacity 1 that explains
  the control arm, the per-rank widening at any DP size, the reported `== ` repair and its
  unverified cross-DP trade, and the nightly-config-only coverage.
