# `Running` > 0 with a flat generation counter: KV-block lifetime under async scheduling

## Scope

A deployment that has been serving goes **silent per rank**, with no error anywhere:

- `vllm:num_requests_running` stays > 0 on the affected engines while
  `vllm:generation_tokens_total` and `vllm:request_generation_tokens_count` show a
  **zero delta in every scrape** — request IDs stay assigned and never get a
  finish time;
- health and `/metrics` keep returning 200, the router stays alive and gradually
  sends everything to the remaining healthy rank (recorded #13964: 15 of 16
  decode ranks held `Running` > 0 with zero deltas — one of them for 2 h 34 min —
  while DP1 finished all 116 requests in the window);
- no `EngineDeadError`, traceback, HCCL/vector-core timeout and no plog block in
  the window; two affected ranks began progressing again after 2½-3 h;
- the serve carries `--async-scheduling` (or MTP/spec decode) **and** a hybrid KV
  cache — `--no-disable-hybrid-kv-cache-manager` on a model with
  sliding-window/chunked-local (or linear-attention) groups.

Classify it as a **state** failure (KV bookkeeping) that surfaces as a liveness
failure. It is neither the throughput topic (nothing drains at all) nor the
admission wedge (`Running` = 0 there, `index.md` rule 1).

## The recorded mechanism: reporter-confirmed, #13964

The original reporter closed #13964 as resolved by vllm-ascend **#13518**
(2026-08-20), verified by their own follow-up: the decode-rank stall was caused by
**premature reuse of SWA / chunked-local KV blocks under async scheduling**.
`num_computed_tokens` could include tokens from batches that were *scheduled* but
whose outputs had not been processed yet, so blocks still referenced by an
in-flight attention step were pruned, freed and handed to another request. The fix
tracks scheduled-but-unprocessed tokens per request, prunes on processed tokens
instead of the optimistic count, forbids reuse of SWA/chunked-local blocks while an
in-flight step can still reach them, and settles the in-flight counter on the
scheduler's exit paths. A separate **capacity** regression from the more
conservative admission sizing is tracked in #14678 — it does not ask to revert
#13518, so do not read #14678 as "the fix was wrong".

Label strength honestly: the *mechanism* is reporter-confirmed; the *patch text* of
#13518 is not in any snapshot (PRs are unavailable), so at the reporter's own pin
you verify its **absence**, not its diff.

## Where the accounting is, verified at vllm v0.23.0 `0fc695f` (vllm-ascend `f4a08bd`)

| Fact | Anchor |
|---|---|
| The scheduler advances the count **optimistically, at schedule time**: `request.num_computed_tokens += num_scheduled_token`, with the comment "If some tokens (e.g. spec tokens) are rejected later, the number of computed tokens will be adjusted in update_from_output" | `vllm/v1/core/sched/scheduler.py:997-1032` |
| The prune runs off that count, *before* the step's allocation: `total_computed_tokens = min(request.num_computed_tokens + num_new_computed_tokens + num_external_computed_tokens, max_model_len)` then `coordinator.remove_skipped_blocks(request_id, total_computed_tokens)`, "We can do this even if we cannot schedule this request … before allocating new blocks to reduce the number of evicted blocks" | `vllm/v1/core/kv_cache_manager.py:344-382` |
| `remove_skipped_blocks` frees them **to the front of the free queue**: `self.block_pool.free_blocks(removed_uncached_blocks, prepend=True)` — "the next allocation candidates" — and `block_pool.free_blocks(…, prepend)` is documented as "put newly-free blocks at the front of the free queue to be prioritized for reuse", so a freed skipped block is exactly what the same step then hands to another request | `vllm/v1/core/single_type_kv_cache_manager.py:448-501`; `vllm/v1/core/block_pool.py:419-429` |
| **Only the Mamba manager discounts the optimistic count**, and its comment names this exact hazard: "the `num_computed_tokens` can contain draft tokens from the previous step that may or may not be rejected later … assume that all tokens are rejected so we don't free blocks that we might actually need" → `max(0, num_computed_tokens - self.num_speculative_blocks)` | `vllm/v1/core/single_type_kv_cache_manager.py:1018-1028` |
| `SlidingWindowManager.get_num_skipped_tokens` uses the raw count: `max(0, num_computed_tokens - self.sliding_window + 1)` | same file `:767-793` |
| `ChunkedLocalAttentionManager.get_num_skipped_tokens` floors the raw count to the chunk boundary: `(num_computed_tokens // attention_chunk_size) * attention_chunk_size` | same file `:902-946` |
| The plugin sends its DeepSeek-V4 sliding-window groups to upstream `SlidingWindowManager` (`AscendSlidingWindowMLASpec`, `KVCacheSpecRegistry.register(…, manager_class=SlidingWindowManager, …)`) and does **not** override `remove_skipped_blocks` — the coordinator it builds (`AscendHybridKVCacheCoordinator`) defines only `__init__`, `_get_effective_block_size`, `verify_and_split_kv_cache_groups`, `find_longest_cache_hit*` | `vllm_ascend/core/kv_cache_interface.py:253-258, 305-314`; `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:60,69-357` |
| Nothing downstream tracks scheduled-vs-processed tokens at this pin: `rg -n 'inflight\|in_flight' vllm_ascend/` → no hits (the worker side handles `use_async_scheduling` placeholders only) | `vllm_ascend/worker/model_runner_v1.py:410,733,883` |

So the *invariant* is upstream's and the *group population* (which SWA/compressed
groups exist, which `--block-size` they get) is downstream's — the same ownership
split `../upstream-downstream-drift/kv-block-free-list-assert.md` records for the
crash form of this family.

## Cheapest checks, in order

1. **Ablate the async batch queue, then spec tokens.** The hazard needs a queue of
   in-flight steps: re-run the same load with `--async-scheduling` **off**; if it
   survives hours, re-run with `--speculative-config` removed (MTP/spec
   placeholders widen the gap between scheduled and processed). The reporter's own
   report lists exactly these as the community workarounds quoted from the
   neighbouring issue — they are the first ladder rungs, not background reading.
2. **Name the KV groups.** Which specs the checkpoint builds
   (`get_kv_cache_spec` in `vllm_ascend/models/deepseek_v4.py`; grep
   `AscendSlidingWindowMLASpec`/`AscendMLAAttentionSpec`), and whether the run's
   `--block-size` matches the recipe for that model — an SWA group whose effective
   block size does not divide the window makes the prune window move in
   block-sized jumps.
3. **Instrument the prune, do not re-derive it.** In a repro, print
   `request.num_computed_tokens`, `request.num_tokens`,
   `request.num_output_placeholders` and the returned `num_skipped_blocks` at
   `remove_skipped_blocks` for the stalled request, and the block ids it frees. A
   block id that reappears in another request's `req_to_blocks` **while the same
   step is still in flight** is the mechanism confirmed; a request whose
   `num_computed_tokens` never stops growing while it generates nothing is the
   other half.
4. **Scope the version.** At the reporter's pin the fix is absent by construction
   (the pin predates the merged PR), so rung 1 for a reporter on an rc is *the
   next rc/final* with the same load, and the downgrade control is
   `--async-scheduling` off. Do not offer a hand-patch of the scheduler arithmetic
   as a fix: the lifetime change spans the manager set, the coordinator and the
   in-flight counter, and a partial patch leaves blocks reusable.

## Two theories that read the same counters and are refuted by the paste

- **"The rank is blocked in a cross-DP collective / MC2 all-to-all."** Then its
  peers' collectives wait too, so the *other* ranks cannot keep finishing
  requests, and the paste carries the RPC-watchdog or plog `STUCK`/frozen
  `count of currently executed operators` block
  (`../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md`). In #13964
  the router recorded 116 *finishes* on DP1 inside the stall window — that
  sentence is the refutation.
- **"The DP cudagraph-mode sync (`synced_cudagraph_mode`) diverged."** A real
  mechanism for hangs, but it needs the per-step metadata all-reduce to be
  *live*; on a MoE KV-consumer rank whose decode path selects MC2 the plugin
  deliberately skips it (`vllm_ascend/utils.py` `should_skip_allreduce_across_dp_group`),
  so a mode disagreement is not caught and not causal here. The recorded #13964
  run spent its whole output budget building this argument from the ACL-graph and
  MC2 sources while the report's own `--async-scheduling` + hybrid-KV flags and
  its quoted "disabling async scheduling" workaround went unused.

A per-rank silent stall that **recovers by itself after hours** is a lifetime
story: a block gets reused, the request degenerates or wedges, and the request
that eventually frees/evicts it lets the rank move again.

## Not this page

- The same family **asserting** instead of stalling — `prev_free_block of
  fake_free_list_tail should always exist`, `free_blocks`/`append_n` in the stack:
  `../upstream-downstream-drift/kv-block-free-list-assert.md`.
- `Running` = 0, requests waiting, nothing scheduled: `index.md` /
  `waiting-queue-wedge.md`.
- A worker that died, or a collective/device error, or the plog's own fault block:
  `../hccl-and-hardware/index.md`.
- A request that is stuck with `WAITING_FOR_REMOTE_KVS` on the decode side of a PD
  pair (no KV arrival): `../pd-disaggregation/index.md`.
