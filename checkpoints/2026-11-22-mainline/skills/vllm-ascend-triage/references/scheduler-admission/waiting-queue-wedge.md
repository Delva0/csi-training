# One oversized request wedges the engine: the KV admission gate stops the queue

Symptom family: after one prompt at or above some size, the service answers
**nothing** — not even a 7-token request — for minutes and until a restart. TCP
and `/v1/models` still answer, a sibling model behind the same gateway is fine, the
gateway returns 504 or an empty body, and the server log carries no traceback.
Reporters describe it as "the engine is poisoned" or "requests sit in waiting
forever", often with a cliff at a round token count and a build that used to serve
100K prompts.

Pins for every `path:line` below: **vllm v0.26.0 (`568afb3a1380`)** and
**vllm-ascend v0.26.0rc1 (`f2f74a16c3c5`)**. The gate is a *version* property —
grep the reporter's own tree before claiming it exists there (a 0.23.x vLLM may
predate `scheduler_reserve_full_isl`; if it does, the waiting request's first chunk
is admitted and this page's failure shape is not available).

## Step 1 — one metrics read decides wedged vs slow

`vllm:num_requests_running` = 0, `vllm:num_requests_waiting_by_reason{reason="capacity"}`
≥ 1, `vllm:kv_cache_usage_perc` ≈ 0: nothing is running and nothing is allocated,
so the request is being refused at admission. With no `/metrics` exposed, the
startup pair `GPU KV cache size: N tokens` / `Maximum concurrency for M tokens per
request: kx` (`vllm/v1/core/kv_cache_utils.py:2177-2183`) answers the same
question: compare `N` with the failing request's `usage.prompt_tokens`. `N` ≤
prompt tokens means that request can never be admitted, which is the finding.

## Step 2 — the mechanism, source-verified at the pins

- `scheduler_reserve_full_isl: bool = True` (`vllm/config/scheduler.py:140`,
  docstring "checks whether the full input sequence length fits in the KV cache
  before admitting a new request, rather than only checking the first chunk"), cached
  at `vllm/v1/core/sched/scheduler.py:298-299` and passed as
  `full_sequence_must_fit=` on the **waiting**-request admission call at `:944`.
- `vllm/v1/core/kv_cache_manager.py:411-425` then requires the blocks for
  `min(request.num_tokens, max_model_len)` **plus** `watermark_blocks` (non-zero
  only when requests are already running, `:402-409`) to be free, and returns
  `None` otherwise.
- `vllm/v1/core/sched/scheduler.py:949-956` — `if new_blocks is None: … break`.
  The waiting loop stops for the step; the request stays at the head of the FCFS
  queue, the next step reaches the same `None` and breaks again, so every request
  behind it is never even attempted. That is the "poisoning": no crash, no
  recovery, restart only.
- The `break` is deliberate FCFS policy, not an oversight — the same loop *skips*
  every other non-schedulable case: a blocked/paused waiting status `:685-693`, the
  `max_loras` budget `:695-708`, an async KV load `:979-983`, each with
  `step_skipped_waiting.prepend_request(request); continue`, and the `skipped_waiting`
  queue exists for that purpose (`:185`, `:667`). So "a stuck oversized request must
  not block unrelated small requests" is a real scheduling question with a concrete
  site, not a misconfiguration the reporter made.
- An abort **does** clear it: `finish_requests` removes waiters from both queues
  (`:2144-2191`). Ten-plus minutes without recovery therefore means the engine never
  received the abort — check the frontend before blaming the scheduler.

## Step 3 — which per-request demand can exceed the pool *forever*

Each candidate with its check; a wedge needs a mismatch that no amount of waiting
fixes.

- **The pool is smaller than the prompt's full-block demand.** Check the startup
  `GPU KV cache size` / `Maximum concurrency … : kx` line against
  `usage.prompt_tokens`; `k < 1` is the proof. Levers: `--max-model-len` down to
  what the pool backs, `--gpu-memory-utilization` up, fewer/cheaper extra cache
  groups.
- **Compressed-MLA (DSA/DSv4) accounting.** Downstream's
  `CompressAttentionManager` divides token counts by `compress_ratio`
  (`vllm_ascend/core/single_type_kv_cache_manager.py:30, 148-176, 204-214`), so the
  pool, the block size and the admission cap must use the same divisor. Without a
  cap an admitted request demands more blocks than the pool was sized to back —
  the downstream comment states exactly this and cites vLLM issue #40863
  (DSv4 + MTP, cc≥1, prompt ≥32K).
- **Recycling specs with no cap** (`SlidingWindowSpec` / `ChunkedLocalAttentionSpec`):
  `full_sequence_must_fit` then reserves the full `max_model_len` worth of blocks
  per request, which exhausts the pool at concurrency ≥ 2.
- **MTP / speculative lookahead**: `num_lookahead_tokens` is added to the
  per-request demand at `scheduler.py:940` (same call), so MTP on/off moves the
  cliff without touching the pool.
- **The downstream fix already present at the pin**:
  `get_manager_for_kv_cache_spec` (`vllm_ascend/core/single_type_kv_cache_manager.py:273-325`)
  sets `max_admission_blocks_per_request = cdiv(max_model_len // compress_ratio, block_size) + 1`
  for `AscendMLAAttentionSpec(compress_ratio>1)` and re-applies upstream PR #40946's
  cap for the recycling specs, because the downstream factory override leaves
  upstream's own block in dead code. A build without these caps has the wedge
  available; a build with them is the upgrade argument.

## Step 4 — the one observation that redirects

The full-sequence check uses `min(num_tokens, max_model_len)` and is independent of
the per-step chunk budget, so a **smaller `--max-num-batched-tokens` cannot fix an
admission failure**. If that knob demonstrably restores service, this page is not
the cause: go to the per-step peak/chunking route
(`../performance-throughput/prefill-queue-and-throughput.md`, hybrid block-aligned
chunks) and say so. Likewise, if `kv_cache_usage_perc` is high and `Running` > 0, or
the queue drains slowly, it is queueing, not admission.

Other redirects: a *single* request hanging while others are served is the DP
pad/ACL-graph shape (`../hccl-and-hardware/dp-aclgraph-single-request-hang.md`); an
`RPC call to execute_model timed out` frame means the worker stopped answering
(`../hccl-and-hardware/rpc-watchdog-timeouts.md`).

## What the report usually lacks, and what to ask for

The startup block (`GPU KV cache size`, `Maximum concurrency`, `Chunked prefill is
enabled with max_num_batched_tokens`, `Setting attention block size`, the
`--speculative-config`, any `compress`/state-cache `--additional-config`), the
metrics series during the hang, the diff between the two builds being compared, and
whether the engine logged the abort when the client gave up. "Requests that cannot
be admitted should fail fast with a 400" is **not** implemented at this ref: only
`prompt > max_model_len` is rejected in the frontend, while pool capacity is a
scheduler-side number — say that, cite `kv_cache_manager.py:411-425`, and put the
fix (sizing, the admission cap, or a startup cross-check of `max_model_len` against
the pool) in the action ladder.
