# `prev_free_block of fake_free_list_tail should always exist` — an upstream KV-block assert, with a downstream-assembled pool

## Scope

The engine has been **serving for hours** and EngineCore dies on a bare `AssertionError`
inside upstream vLLM's KV-block bookkeeping:

```text
File "vllm/v1/engine/core.py", line 1810, in run_busy_loop
  executed = self._process_engine_step()
File "vllm/v1/engine/core.py", line 475, in step_with_batch_queue
  scheduler_output = self.scheduler.schedule()
File "vllm-ascend/vllm_ascend/patch/platform/patch_balance_schedule.py", line 73, in schedule
  return super().schedule()
File "vllm/v1/core/sched/scheduler.py", line 425, in schedule
  new_blocks = self.kv_cache_manager.allocate_slots(
File "vllm/v1/core/kv_cache_manager.py", line 362, in allocate_slots
  self.coordinator.remove_skipped_blocks(
File "vllm/v1/core/kv_cache_coordinator.py", line 251, in remove_skipped_blocks
  manager.remove_skipped_blocks(request_id, total_computed_tokens)
File "vllm/v1/core/single_type_kv_cache_manager.py", line 426, in remove_skipped_blocks
  self.block_pool.free_blocks(removed_blocks)
File "vllm/v1/core/block_pool.py", line 420, in free_blocks
  self.free_block_queue.append_n(
File "vllm/v1/core/kv_cache_utils.py", line 337, in append_n
  assert last_block is not None, ("prev_free_block of fake_free_list_tail should always exist")
```

with the APIServer side repeating `EngineDeadError: EngineCore encountered an issue. See
stack trace (above)`. Recorded #10724 (v0.21.0rc1 + vLLM 0.21.0, DeepSeek-V4-Flash-w8a8
MTP, PD-mix, 2×A2 910B4 **32 GB**, `--block-size 16`, `--enable-prefix-caching`,
`--no-disable-hybrid-kv-cache-manager`, `--async-scheduling`, MTP 3, Mooncake
`AscendStoreConnector` `kv_both`, 8 cards at ~30.8/32.8 GB): ~3 h uptime, 1 of ~10
instances, closed **non-reproducible** after the reporter changed the deployment, so no
fix is established. What is established is the ownership and the discriminators below.

## Ownership first — two calls, then stop deriving

1. **Confirm the tree is the pin by line number, not by diffing.** `block_pool.py:420`
   (`free_block_queue.append_n`), `scheduler.py` `allocate_slots`, and
   `kv_cache_utils.py` `append_n` are within a line or two of vLLM v0.21.0, so the
   reporter's stack is the pinned release and no file comparison is owed.
2. **The path is only reachable through a manager whose `get_num_skipped_tokens` > 0.**
   `FullAttentionManager` returns 0 (nothing is ever skipped), so a plain dense
   full-attention deployment cannot free here. `SlidingWindowManager`,
   `ChunkedLocalAttentionManager` and `MambaManager` can. On DeepSeek-V4 the group set is
   **downstream-assembled**: `AscendHybridKVCacheCoordinator`
   (`vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`) creates *one* `BlockPool`
   plus one manager per KV group, sending compressed groups (`compress_ratio` 4/128,
   `model_version="deepseek_v4"`) to `CompressAttentionManager(FullAttentionManager)` and
   SWA groups (`SlidingWindowMLASpec`) to upstream `SlidingWindowManager`
   (`vllm/v1/core/single_type_kv_cache_manager.py:1147`). So the *invariant* is upstream's
   and the *population* (which groups, which block sizes, which pool) is downstream's.
3. **A `vllm_ascend/patch/...` frame in the stack is not downstream ownership.**
   `patch_balance_schedule.py:66/72-73` is `if not self._balance_enabled: return
   super().schedule()` (the reporter's frame lands on the `return` at `:73`), and it is
   enabled only by
   `--additional-config '{"enable_balance_scheduling": true}'`
   (`ascend_config.py:77`; refused in PD-disaggregated mode, `platform.py:620-625`).
   Check the reporter's `--additional-config` — absent ⇒ the wrapper is a pass-through and
   the failing `schedule()` is upstream's. That frame does prove the *mode*: it is reached
   through `step_with_batch_queue`, i.e. `--async-scheduling` or PP>1.
4. **Stop rule.** `append_n`'s `last_block is None` is the *late* symptom of a free queue
   whose links were broken earlier — `FreeKVCacheBlockQueue.remove()` raises its own
   `RuntimeError: remove() called on an invalid block` when a block's pointers are already
   null, so the corruption predates the crash (consistent with hours of uptime and one
   instance of ten). The doubly-linked-list arithmetic **cannot be settled from the
   paste**. After the two reads above, deliver: upstream owns the invariant, the ranked
   triggers below each carry their own ablation, and the artifact ask is the EngineCore
   log before the crash.

## Ranked triggers, each with its one check

1. **One `BlockPool`, several groups, different effective block sizes.** `remove_skipped_blocks`
   computes `num_skipped_blocks = num_skipped_tokens // self.block_size` per manager while
   the queue is global; downstream's `_get_effective_block_size` multiplies by
   `compress_ratio`/CP factors and `SlidingWindowMLASpec.storage_block_size` is
   `block_size // compress_ratio`. Check: list the specs the model actually builds
   (`get_kv_cache_spec` per cache class in `vllm_ascend/models/deepseek_v4.py`) and whether
   the run's `--block-size` matches the release's recipe for that model; a mismatch in group
   count/block size is the cheapest thing to change.
2. **External (KV-pool) computed tokens widen the skip window.**
   `total_computed_tokens = min(num_local_computed_tokens + num_external_computed_tokens,
   max_model_len)` is what `remove_skipped_blocks` receives, so only a pool hit can push the
   window past locally allocated blocks (the `min(num_skipped_blocks, len(blocks))` cap is
   upstream's guard against exactly that, and it is present at this pin). Check: does the
   crash coincide with a pool hit / preemption in the EngineCore log; ablation = drop
   `--kv-transfer-config` and let it run the same hours.
3. **Prefix caching + a kept-on hybrid manager.** `--enable-prefix-caching` shares blocks
   (`ref_cnt > 1`) and the reporter passed `--no-disable-hybrid-kv-cache-manager`
   deliberately. Ablate one at a time: prefix caching off, then the hybrid manager off if
   the model's recipe allows it.
4. **Async scheduling + MTP placeholders.** `step_with_batch_queue` means requests are
   scheduled with `num_output_placeholders` and spec tokens; a rolled-back
   `num_computed_tokens` is the other way the skip window moves. Ablation:
   `--async-scheduling` off, then MTP off.

The first ablation that survives a few hours names the family. Report the assert as
upstream's invariant with the named trigger, not as a device or driver fault — and do not
ask for `npu-smi` first: nothing in this stack is a hardware report.

## The plog is teardown noise for this shape

If the reporter later supplies a plog, expect the one-sided-service block from the
*destruction* of the comm after EngineCore died: `one_sided_service_adapt.cc … The
parameter memHandle[…] is invalid`, `HcclDeregisterGlobalMem … hcclRet -> 1`,
`[HcclOneSidedService][DeInit] There are memories still bound to this comm; please unbind
them before destroying the comm`, `hccl_comm_host.cc … DeinitOneSidedService`, then driver
`rtFreeHost … 507899` / `devmem … Set free error (ref_count=2)`. Those lines describe a
comm torn down while pool segments are still bound; they follow the crash and are not its
cause (recorded #10724 sent exactly this block; the maintainer's PD-Mix/A2 question stayed
open and the thread closed non-reproducible).

Ask instead for: the last few hundred lines of the **EngineCore** log before the assert
(which request was scheduled/finished/preempted, whether a pool hit landed, any
`Preemption`/recompute line), `--additional-config` and the full serve command per node,
and the model's KV-group list. Check the opening body's folded `<details>` blocks before
requesting the launch script — in #10724 both node scripts were already in the report and
the reporter's answer was "start script is in my first bug description, folded".

## Not this page

- A request that waits forever with no traceback → `../scheduler-admission/index.md`.
- A `RuntimeError`/`AttributeError` raised in a `vllm_ascend/` frame → the sibling pages
  here (`patched-forward-api-drift.md`, `scheduler-fork-missing-status.md`).
- A device dump, `507011`, or `ERR02005` → `../hccl-and-hardware/index.md`.
