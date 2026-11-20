# KV pool write path healthy, read path returns 0 hits (hybrid coordinator)

## Scope

`AscendStoreConnector` (memcache or mooncake backend) as a `MultiConnector` member on a
hybrid model (DeepSeek-V4 / DSA, GLM, GDN): writes succeed (`memcache_stored_keys` > 0,
`memcache_segment_allocated_bytes` > 0 per rank), the store-side and lookup-side keys are
byte-identical, `batch_is_exist` answers `1` for every key the lookup queried — and
`find_longest_cache_hit` / `get_num_new_matched_tokens` still returns `0`, so
`vllm:external_prefix_cache_hits_total` stays 0 while `…_queries_total` grows.
`use_layerwise=true` shows the same via `_get_layerwise_hit_tokens`. Checked at
vllm-ascend `v0.27.1rc1` @`3b318862`, vllm `v0.27.1` @`6e448d0e` (case #18051).

This is the **glue** layer, not the store: the reported "the data plane is fine" evidence
(segment allocated, keys visible cross-engine) is *expected* and does not narrow anything.
Do not re-verify the store's put/get before reading the four facts below — they are what
the numbers in the report mean.

## The four facts that decide it

1. **Every size is derived from the group block sizes, not from `block_size`.**
   Upstream `resolve_kv_cache_block_sizes` (`vllm/v1/core/kv_cache_utils.py:626-688`):
   `scheduler_block_size` = lcm of group sizes (`* dcp` for attention specs),
   `hash_block_size` = `cache_config.prefix_match_unit` else **gcd** of group sizes, and it
   falls back to the scheduler size when there is no prefix caching / connector or a mamba
   group's block size differs from `cache_config.block_size`. The pool worker derives its own
   (`pool_worker.py:173-181`): `prefix_match_unit` else **`min(original_block_size)`**, then
   `* cp_scale` (`pcp_size * dcp_size`), and asserts every group block size is divisible by
   it. `min` and `gcd` agree only when the smallest size divides all the others; upstream
   scales by `dcp` only, the worker by `pcp*dcp`. A logged `hash_block_size: 2,
   lcm_block_size: 4096` therefore says "the group block sizes have min 2 and lcm 4096 at
   cp_scale 1" — it does **not** say the prompt hashes are 2-token blocks.
2. **Chunk keys are group-granularity.** `get_block_hashes`
   (`ascend_store/metadata.py:657-665`) returns `_LazyGroupedBlockHashList`, whose
   `__getitem__(i)` is `block_hashes[(i+1)*scale-1]` (`:698-708`): one key per group block,
   identified by the *last* hash-block hash of the chunk. So the per-group `ones=n` in the
   report is a chunk count = `aligned_len / group_block_size` (`aligned_len` =
   `token_len` rounded up to `lcm_block_size`). Use it to *measure* the block-size set; a
   count that no divisor of the lcm can produce means the set was mask-filtered (fact 4).
3. **The hit is a dense chain from chunk 0, per attention group, then rounded down to
   `cache_transfer_granularity`.** `FullAttentionManager.find_longest_cache_hit`
   (`vllm/v1/core/single_type_kv_cache_manager.py:682-780`) walks full blocks from index 0
   and `break`s on the first `get_cached_block` miss (chained hashes), then does
   `hit_length -= hit_length % alignment_tokens`. `AscendStoreCoordinator._find_hit_blocks`
   takes `hit_length = max_length` and `min`s it per group (`coordinator.py:212-300`), passing
   `alignment_tokens=self.lcm_block_size`, which the worker set to
   `cache_transfer_granularity` (`pool_worker.py:613-624`, `metadata.py:278-287`). Two
   consequences: one group at 0 zeros the whole lookup, and a matched prefix shorter than one
   transfer granularity (here 4096 tokens) is reported as **0**, not as the matched length.
   The single-attention-group branch (`coordinator.py:222-234`) uses `spec.block_size`
   instead, so this rounding bite exists only on the hybrid/multi-group path.
4. **Store and lookup masks are not the same function of the input.** `store_mask` builds
   masks with `retention_interval` + `num_prompt_tokens`; `lookup_mask` passes `None` for
   both (`coordinator.py:162-206`). `_uses_reachable_mask` is `{None, "default", "c1"}`
   (`:389-390`), so compressed families (e.g. `c4`) are dense and never filter. The mask is
   compiled into `chunk_filter` in `_lookup_with_coordinator`
   (`pool_worker.py:2271-2297`), so filtered chunks are **never queried**, are absent from
   `exists`, and break fact 3's chain at index 0. `ExternalCachedBlockPool.get_cached_block`
   also requires *every* group id of the attention group to hold the hash
   (`coordinator.py:42-52`), so one missing group id nulls the block for its siblings.

## Checks, in this order

1. Print per-group `new_hit_length` and the post-rounding `hit_length` inside
   `coordinator.py:_find_hit_blocks`. All groups > 0 with a 0 total → fact 3's rounding
   (answer: it is the alignment, and the actionable ask is a prefix of at least one
   `cache_transfer_granularity` *per group*, densely stored). Exactly one group 0 → that
   group's chain, go to check 2.
2. For that group, compare the queried chunk indices with `range(0, aligned_len //
   group_block_size)`. A gap at the head = fact 4 (mask filter or a store that used a
   different mask); `exists` being all-1s for the queried keys is then *consistent* with a 0
   hit, not evidence against it.
3. Compare `pool_worker.self.hash_block_size` / `grouped_block_size` with upstream's
   `resolve_kv_cache_block_sizes(...)` for the same `kv_cache_config` (fact 1). Unequal
   means every chunk index is shifted, while store and lookup still agree with each other —
   exactly the report's "byte-identical keys, all exists, zero hit".
4. Only then consider the refactor story. The coordinator is built only when
   `use_hybrid` and a `kv_cache_config` exist (`pool_worker.py:612-624`), and
   `_lookup_with_coordinator` returns `None` — falling back to the old per-group path —
   when it is absent or `use_layerwise` (`:2251-2263`). That fallback takes `min` over
   per-group `hit_end` already aligned to `cache_transfer_granularity`
   (`pool_worker.py:2181-2209`), which is what the reporter's older refs ran
   (#12761 is their attribution). Quote the behavioural difference; it is not proof of
   which line is wrong.

## Rules

- Ask which of the two numbers moved: hit 0 with group-level hits > 0 is a *different*
  answer (alignment/rounding) from all-group 0 (chain), and one reply from the reporter
  separates them.
- Do not reverse-engineer the reporter's per-group counts as mask evidence until fact 2 is
  applied to `aligned_len`; `n` values that contradict the lcm is the useful signal.
- Name the owner precisely: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`
  (`coordinator.py` hit/mask glue, `pool_worker.py` hash/scale derivation,
  `metadata.py` chunk keys) versus the upstream manager's hit semantics. A "granularity
  mismatch" claim that does not say which of facts 1/3/4 it is has not been checked.
- Version scope: the coordinator only exists on refs carrying the #12761 glue; on older
  refs the same report is `lookup_scheduler` alone. State the ref you read.
