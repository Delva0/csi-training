# The EngineCore dies inside the prefix-cache lookup (`get_computed_blocks`)

Scope: the engine has started, `--enable-prefix-caching` is on, and the death is
a **scheduler-side** Python exception under
`EngineCore encountered a fatal error` (`vllm/v1/engine/core.py:1351`) whose
deepest visible frame is upstream `vllm/v1/core/kv_cache_manager.py:261`
`get_computed_blocks` → `self.coordinator.find_longest_cache_hit(`, with
`vllm_ascend/patch/platform/patch_balance_schedule.py … return super().schedule(…)`
one or two frames above it. Nothing ran on the device: no `ERR`/`EZ` code, no
plog, no collective. Recorded shape: a DeepSeek-V4-Flash-**Vision** serve on
8×910B3, `--block-size 32`, `--enable-prefix-caching`, `FULL_DECODE_ONLY`,
crashed one request after `POST /start_profile` (#16760) — but the lane is any
hybrid/multi-group serve with caching on, not one model.

Verified at vllm-ascend **v0.26.0rc1** (`f2f74a1`) + vllm **v0.27.1**
(`6e448d0e`), the #16760 pair. Line numbers move between releases; the four
contracts below do not.

The same two classes meet with the drift running the **other** way: the upstream
*consumer* starts aggregating a per-group list while the downstream producer still
returns the scalar (`TypeError: '<type>' object is not iterable` at
`max(per_group_hits)`). Decide the direction from the paste's failing line first —
§6 — because the exclusions in §2 hold at the pins named there and not at every
release.

Sibling lanes: a device fault in a DSA kernel is
`../dsa-sparse-attention/indexer-cache-scatter-fault.md`; a "服务崩溃 / 崩溃" with
no log at all is `../dsa-sparse-attention/concurrency-crash-with-no-log.md`; an
admission decision that never raises (requests never scheduled) is `index.md`;
a free-list assert under a healthy serve is
`../upstream-downstream-drift/kv-block-free-list-assert.md`.

## Answer rule

The deepest pasted frame is **upstream**, the owner is **downstream**. With more
than one KV-cache group and `enable_caching` on, the Ascend patch replaces the
whole coordinator object and dispatches each group to its own manager class, so
the frames between `schedule()` and that line are `vllm_ascend` code the paste
cut away — "upstream vLLM bug" is the wrong verdict, and no local read produces
the exception type. Deliver: the ownership, the four exclusions below (one line
each, so the next reader does not re-derive them), the two ablations, and the
ask for the missing tail.

## 1. Which coordinator and which manager are live

- `patch/platform/patch_kv_cache_coordinator.py:421-497` overrides
  `vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator` (and re-binds the
  copy `kv_cache_manager` imported). It returns `AscendHybridKVCacheCoordinator`
  when `_is_deepseek_v4_kv_cache_config(...)` **or** `len(kv_cache_groups) > 1 and
  enable_caching`; a single-group serve, or caching off, is handed to the original
  upstream factory — so the same paste on a single-group dense model is upstream
  code and not this page.
- The per-group class comes from `vllm_ascend/core/single_type_kv_cache_manager.py:273-325`
  `get_manager_for_kv_cache_spec`: an `AscendMLAAttentionSpec` with
  `compress_ratio > 1` → `CompressAttentionManager` (`:304`, the DSv4 compressed
  groups, `find_longest_cache_hit` at `:220-270`); sliding-window /
  chunked-local groups keep upstream managers plus an admission cap.
- `AscendHybridKVCacheCoordinator.find_longest_cache_hit`
  (`patch_kv_cache_coordinator.py:267-378`) is a fixed-point loop over the
  *attention groups*: each group returns a hit length, the loop re-runs until no
  group reduces it (`:363-370`), then full-attention groups are truncated to the
  reconciled length (`:371-377`).

## 2. Four contract facts that exclude the shape theories

Each is one read, and each kills a candidate that otherwise costs a session:

| Fact | Site | What it excludes |
|---|---|---|
| The coordinator always returns a **3-tuple** `(cache_hit_blocks, hit_length, longest_hit_length - hit_length)`; upstream unpacks three (`kv_cache_manager.py:257-262`, `num_uncached`) and its own `HybridKVCacheCoordinator` returns the same (`vllm/v1/core/kv_cache_coordinator.py:685,817`) | `:378` | "not enough values to unpack (expected 3, got 2)" is not this lane — an older override with the two-value signature is not installed here |
| The manager call is keyword-only and both sides accept every name: `block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec, drop_eagle_block, alignment_tokens, dcp_world_size, pcp_world_size` | `:337-353`; upstream `single_type_kv_cache_manager.py:547-558`; Ascend `core/single_type_kv_cache_manager.py:220-234` (same names, different order — harmless because the call is by keyword) | `TypeError: … unexpected keyword argument` is not this lane at this pair |
| The alignment pop loop cannot `IndexError`: `while logical_block_size != alignment_tokens and len(computed_blocks[0]) * logical_block_size % alignment_tokens != 0: computed.pop()` — at length 0 the modulo is `0`, so the loop exits before popping an empty list | `core/single_type_kv_cache_manager.py:263-268` | "pop from empty list" is a red herring (recorded #16760 ranked it and spent turns on it). What the loop really does is *discard* a sub-alignment hit, i.e. a hit-rate effect, not a crash |
| A cold pool cannot raise here: nothing cached → the first `block_pool.get_cached_block` misses, the loop `break`s, `hit_length` is 0, and the coordinator's `curr_hit_length >= hit_length` test (`:363`) exits | `:249-257`, `:363` | "the very first request on an empty pool" does not explain an exception in this frame; the state that raised it is in the caller, i.e. in the part the paste cut |

`alignment_tokens` is also never 0 on a serve that got this far: it is
`hash_block_size` only for Mamba `align` groups (`:162-167`, `:182-185` — false on
a DSv4/hybrid-attention serve), otherwise `scheduler_block_size or lcm_block_size`,
and the init-time `assert all(effective_block_size % hash_block_size == 0)` would
have divided by zero first (`:158-161`). One consequence worth stating rather than
debugging: the hit length must be a multiple of the LCM of the groups' *effective*
block sizes (`:187-197`, `block_size × compress_ratio × dcp`), so a short prefix
never hits at all — that is `../performance-throughput/zero-prefix-cache-hit-hybrid.md`'s
question, not this page's.

## 3. What the paste's own frames already prove

- **`core.py:653 step_with_batch_queue` ⇒ a batch queue ⇒ `async_scheduling` (or
  PP > 1).** `vllm/config/vllm.py:540-550` returns `pp_size` normally and `2` when
  `scheduler_config.async_scheduling` (V1 runner, PP ≤ 1); `core.py:206-232`
  builds the queue only when that is > 1 and then picks `step_with_batch_queue`.
  So a launch whose command shows neither `--async-scheduling` nor `-pp` is *not*
  the launch that crashed: ask for the running command. It matters twice — an
  `AsyncScheduler` uses a different `_update_request_with_output`, which
  vllm-ascend patches (`patch/platform/patch_async_scheduler.py`), and async
  scheduling moves when `num_computed_tokens` and `cache_blocks` advance.
- **`/start_profile` is not the trigger.** The only profiling→scheduler coupling
  downstream is the profiling-*chunk* scheduler
  (`core/scheduler_profiling_chunk.py:109 run_profiling_chunk_init`, installed by
  `patch/platform/patch_profiling_chunk.py:200-202` and `platform.py:704-706`),
  gated on `scheduler_extension_config.profiling_chunk_config.enabled` **and
  `pp > 1`** (`ascend_config.py:87-106`) and run once at EngineCore init. The
  torch profiler route starts a worker-side profiler and changes no scheduling
  state; `ERROR: External init callback must run in same thread as registerClient`
  is kineto's callback-thread complaint, printed by the API server process. The
  discriminator is one launch: restart, send the same request, no `/start_profile`.

## 4. The ranked answer this frame supports

1. **Request state in the caller** (the cut tail decides): `request.block_hashes`
   length vs `max_cache_hit_length = request.num_tokens - 1`
   (`kv_cache_manager.py:259`), a group whose manager returns a length another
   group cannot consume, or a `block_pool` bookkeeping error. Ask for the lines
   after the reporter's `@@@` cut plus `VLLM_LOGGING_LEVEL=DEBUG` (it prints the
   per-request hash/block lines this frame consumes). No local read substitutes
   for the exception class.
2. **The multi-group alignment/compress arithmetic on this checkpoint** — a wrong
   *length contract between groups* (the caller reslices by the returned
   `hit_length`), not the pop loop §2 rules out: `--block-size 32` with
   `compress_ratio > 1` groups makes `logical_block_size = block_size ×
   compress_ratio` while the alignment is the LCM over the groups. Ablation 1 is
   `--no-enable-prefix-caching` (the whole lookup disappears), ablation 2 is
   `--block-size 64` or `128` — the other two values the DSv4 page-size tables carry
   (32/64/128). Neither is a fix; each removes one term.
3. **Build skew.** A `0.1.devN+g<sha>` vllm-ascend against upstream **v0.27.1**
   is not the pinned pair: v0.26.0rc1's own release notes say "aligned with
   upstream vLLM v0.26.0" (`docs/source/user_guide/release_notes.md:5`), and the
   KV-lookup API is exactly what moved between those upstream versions
   (`resolve_block_hashes`, `hash_block_size`, `supports_fine_grained_hash_lookup`,
   the 3-tuple). Two one-liners on the reporter's box decide it:
   `python -c "import vllm_ascend; print(vllm_ascend.__version__)"` and the first
   forty lines of `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`
   there. Do not quote pin line numbers as theirs until that matches.

## 5. Delivery on this shape

A `@@@` (or any marker) in the middle of the traceback is the reporter's own
truncation: the exception line and every `vllm_ascend` frame between `schedule()`
and the coordinator are gone, and no local read reconstructs them. So the ranked
list plus the artefact asks **is** the message, owed on the turn the shape is
classified. Recorded #16760 spent 21 assistant turns and shipped no message text
at all, after reaching the coordinator, its manager dispatch and the alignment
derivation — the frame, the exclusions and the two ablations were in hand well
before halfway.

## 6. The mirror case: the consumer widened to per-group hits (#13998, reporter-confirmed)

Recorded #13998 (vllm-ascend **0.24.0rc1** + vllm **0.24.0**, 910B2C, 1P1D
Qwen3.5-35B-A3B-w8a8-mtp + Mooncake). The paste's own traceback ends:

```text
vllm_ascend/patch/platform/patch_balance_schedule.py:152 in schedule
    return super().schedule(throttle_prefills)
vllm/v1/core/sched/scheduler.py:701 in schedule
    num_new_local_computed_tokens = max(per_group_hits)
TypeError: 'int' object is not iterable
```

and the reporter's control is in the same body: the **P** command carries
`--enable-prefix-caching` with `--no-disable-hybrid-kv-cache-manager
--mamba-cache-mode align` (so more than one KV-cache group), the **D** command
carries `--no-enable-prefix-caching`, and only P dies. Later comments confirm
it: `p 节点关闭prefix-cache后可以正常推理`, then the next release fixed it on the
unchanged launch.

Read it as the §1 gate run backwards. Upstream's *consumer* now takes an
iterable of per-group hit lengths (`max(per_group_hits)`); the downstream copied
producer still returns the pre-change scalar. So here the shape **is** the
cause, and which side is stale is decided by the printed failing line, not by
this page's pins: a `max(...)`/`sum(...)`/tuple-unpack over a name the pin does
not contain means upstream moved. Grep the *producer of that value*, not
`get_computed_blocks` — at v0.23.0rc1 the chain is
`kv_cache_manager.get_computed_blocks → coordinator.find_longest_cache_hit`
returning `(tuple(blocks per group), hit_length: int)`
(`patch/platform/patch_kv_cache_coordinator.py:350,453`; upstream declares the
same 2-tuple at `vllm/v1/core/kv_cache_coordinator.py:320,586` and
`kv_cache_manager.py:196`) — note the tuple **width** itself moved (2 here, 3 at
v0.26.0rc1 in §2), which is why §2's exclusions must be re-grepped per release.

- **The `patch_balance_schedule.py` frame is a shell, not a cause.** At
  v0.23.0rc1 the patch assigns `vllm.v1.core.sched.scheduler.Scheduler =
  BalanceScheduler` unconditionally at module import (`:718`), so every Ascend
  engine shows that frame; do not rank `enable_balance_scheduling`,
  `recompute_scheduler_enable` or the PD layout, and do not read the balance
  scheduler's body (in the 0.24 paste it is the delegating wrapper).
- **One flag is both the discriminator and the workaround:
  `--no-enable-prefix-caching`.** It routes the serve to the original upstream
  factory (§1 gate), removing the downstream producer without touching anything
  else — reporter-confirmed on #13998. The durable fix is the next release
  (`版本更新至…正常推理，bug 已修复`), so rank *ablation, then upgrade*, never a
  local patch.
- **The snapshot may pin the working side of the comparison** (v0.23.0rc1 +
  vllm v0.23.0 for #13998): quote the pin for the producer's pre-change shape,
  label the newer side a hypothesis taken from the paste, and hand the reporter
  the one grep that settles it on their tree —
  `rg -n 'per_group_hits' <site-packages>/vllm/v1/core/sched/scheduler.py`
  beside `rg -n 'return tuple' -A2
  <site-packages>/vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`.
- **Prefix caching off is also the fastest rung for a reporter who needs the
  service up now**, and the D node of that very report is the working arm — say
  so rather than only naming the mechanism.
