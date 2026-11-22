# AscendStore held blocks: `kv_cache_usage_perc` climbs to 1.0 and never recovers

Recorded vllm-ascend #10098 (title: "AscendStoreConnector: KV cache blocks never freed when all
blocks already exist in KV Pool (delay_free deadlock)"), verified at vllm-ascend `v0.20.2rc1`
`367b8e6` / vllm `v0.20.2` `bc150f5`. The reporter ran `releases/v0.18.0` and `origin/main`
(neither is in the snapshot — its repos carry one commit each) and had already written two
hunks. Deliverable for this shape is the **patch**, not only the verdict: the recorded thread's
next comment was "If you have any drafted patch, I am happy to test with it".

## Signature — and the neighbours it is not

- An `AscendStoreConnector` deployment (alone, or as a `MultiConnector` member with
  `MooncakeConnector`), PD or single node, serving normally; then **one** prefill instance's
  `vllm:kv_cache_usage_perc` climbs to ≈0.999 and never comes back, new requests queue behind
  it, and only a restart recovers it. No traceback, no device code, no collective error, and
  the pool itself answers lookups.
- The connector did take the **delay-free** path: upstream `Scheduler._free_request(...,
  delay_free_blocks=True)` (`vllm/v1/core/sched/scheduler.py:1813-1822`) skips `_free_blocks`,
  and only `_update_from_kv_xfer_finished` (`:2108-2135`, `for req_id in
  kv_connector_output.finished_sending: self._free_blocks(...)`) releases them. So the whole
  case reduces to **who reports `finished_sending` for that id** — at this pin, a counter, not
  a set (below).
- Not `no-put-under-spec-decode.md`: there the pool stays empty and the hit rate is 0 %. Here
  lookups succeed and the instance may put for other requests; the leak is on the requests the
  connector chose to hold.
- Not `pool-lookup-overlap-zero-load.md` (zero load is the upstream contract) and not
  `pool-attached-cold-request-cost.md` (slow, not wedged).
- Not `../pd-disaggregation/prefill-scheduler-kv-xfer-assert.md`: that is the **mirror**
  direction — a `finished_sending` id arriving after the request is gone hits
  `assert req_id in self.requests` (`scheduler.py:2130`). Same two bookkeeping sides, opposite
  symptom; check which one the paste carries before reading either page.
- **Two reporter-side strings that do not exist at the pin**: `delayed_free_reqs_in_requests`
  (the only `delayed_free*` name in the tree is the Mooncake p2p connector's
  `delayed_free_requests` OrderedDict, `kv_p2p/mooncake_connector.py:126`, `:166-177` — i.e. the
  MultiConnector's *other* member), and `AscendStore send:` / `AscendStore recv:` (the pin logs
  `Storing KV cache for N out of M blocks (missing_count=K) for request <id> in group <g>`,
  `kv_transfer.py:297-302` and `:520-524`, and `Retrieved N out of M out of total K tokens`,
  `pool_worker.py:667`). Their absence from a paste is not evidence of anything; ask which
  build prints them.

## The contract at the pin (all anchors verified in `kv_pool/ascend_store/`)

| fact | anchor |
|---|---|
| the completion set `finished_requests` and its accessors | `kv_transfer.py:46`, `:63-76` (`KVTransferThread`) |
| the completion **counter** `stored_requests` and its helpers | `kv_transfer.py:230-241` (`add_stored_request` / `dec_stored_request` / `delete_finished_stored_request`) |
| `KVCacheStoreSendingThread._handle_request`: per-group `if not missing_indices: continue`, then `dec_stored_request` at the end, **no `set_finished_request` anywhere in the class** | `kv_transfer.py:209`, `:244-368`, guard `:248-250`, `:289`, `:366` |
| `KVCacheStoreLayerSendingThread` *does* call `set_finished_request` — on the last chunk only (`is_last_chunk and layer_id == final_layer_id`) | `kv_transfer.py:442`, `:484-486`, `:494-497`, `:517` |
| the send path's reader: `done_sending` comes from `stored_requests[ req_id ] == 0` (plus a `finished_store_req` pending set); `req_remain_jobs is None` → **nothing at all** (neither finished nor pending) | `pool_worker.py:731-732`, `:755-788` |
| preemption deletes the counter record | `pool_worker.py:757-760` (`meta.preempted_req_ids` → `delete_finished_stored_request`) |
| the *only* `add_stored_request` caller: `wait_for_save`, gated on `ReqMeta.can_save` | `pool_worker.py:569-599` (`:573-575`, `:587`), then `request_queue.join()` `:599` |
| the layerwise save registers **nothing**: `save_kv_layer` → `store_layer` only `add_request` | `pool_worker.py:544-566`, `:675-727` |
| layerwise skips `wait_for_save` at the connector | `ascend_store_connector.py:84` (flag), `:212-221` (`save_kv_layer`), `:223-231` (`wait_for_save` returns early) |
| the scheduler's delay decision is `num_saved_tokens`, not `can_save`; `tracker is None` falls through to `len(block_ids) > 0` | `pool_scheduler.py:506-524` (`:517-520`), `:526-546` (`:534-535`) |
| `RequestTracker` has **no** `can_save` field; `ReqMeta.can_save` is the worker's gate, set from the same `skip_save` test that sets `num_saved_tokens` | `config_data.py:485-518`, `:557`, `:567`, `:628-674` (`:650-651`, `:655`, `:674`) |
| preemption also drops the tracker | `pool_scheduler.py:343-346` |

Reading: the send side reports completion only through the counter, and the counter exists only
where a save was actually **enqueued**. So a request that delays free while its id has no
counter record can never be released — `get_and_clear_finished_requests` iterates the counter
(absent id → no iteration) and then, for an id in `finished_req_ids`, takes the `is None`
branch and does nothing. Three ways to get there:

1. **Layerwise save** (`"use_layerwise": true` in the `kv_connector_extra_config` of
   `--kv-transfer-config`): the connector returns from `wait_for_save` before calling the
   worker (`ascend_store_connector.py:228-229`), so `add_stored_request` never runs and every
   request that computed ≥1 block delays free with no reporter. Deterministic, from the first
   requests: usage rises monotonically from startup. `KVCacheStoreLayerSendingThread`'s
   `set_finished_request` calls are **dead code on this path** — nothing on the send side reads
   `finished_requests` (the only `get_and_clear_finished_requests()` no-arg call is the *recv*
   thread's, `pool_worker.py:741`).
2. **Preemption**: `pool_scheduler` pops the tracker (`:345`) and the worker deletes the counter
   (`:757-760`); the request then finishes with `tracker is None`, so `request_finished` returns
   `delay_free = len(block_ids) > 0` and nothing can ever report it. Self-feeding: a busy
   prefill preempts → leaks blocks → higher occupancy → more preemption. This is the shape that
   matches a **gradual** climb to 0.999 under prefix-reuse load.
3. **An exception inside `_handle_request`** is not a leak but a step-level hang: `run()`
   logs only `Error in KVCacheTransferThread: <e>` and never calls `task_done()`
   (`kv_transfer.py:79-92`), so the `request_queue.join()` at `pool_worker.py:599` never
   returns inside `wait_for_save` and EngineCore stops stepping for everyone. One line in the
   log, no traceback, prefill serves nothing afterwards — grep the paste for it before
   ranking 1/2.

## What a proposed `set_finished_request` hunk does at this pin: nothing

The recorded report's Fix 1 adds `self.set_finished_request(req_id)` to the
`if not missing_indices:` branch of `KVCacheStoreSendingThread._handle_request`. That method
has **no reader** on the send path, so the hunk changes no behaviour; the same file's
`KVCacheStoreLayerSendingThread` already contains the identical pattern (`:494-497`), which is
where the reporter's snippet comes from. The finding is **connector-scoped, not tree-wide**: the
Mooncake p2p connector's worker *does* read its send thread's set
(`kv_p2p/mooncake_connector.py:1345` → `:217-223`), which is why the same hunk would be correct
in the sibling connector and is inert in this one — check the reader before generalising either
way. Fix 2 (`can_save` on `RequestTracker`, gated in
`request_finished`) is the right direction but not sufficient by itself: the scheduler-side
gate cannot release a request whose save was never enqueued, and in the layerwise path the
enqueue call is a different one (`store_layer`), not `wait_for_save`.

Repair that covers all three, in one place each:

- `pool_worker.get_and_clear_finished_requests`: report any id in `finished_req_ids` whose
  counter record is **absent** as finished (treat `req_remain_jobs is None` as done, the
  opposite of the `:780-785` reading), and merge in
  `self.kv_send_thread.get_and_clear_finished_requests()` so the set the layer thread already
  writes becomes authoritative too.
- Register the counter for the layerwise path (`add_stored_request` in `save_kv_layer`, or at
  `store_layer` submission) so the two enqueue points agree.
- `run()`: `try/finally: self.request_queue.task_done()` so one failed put cannot wedge the
  step.

Acceptance is not "it runs": replay a full-pool-hit burst (an identical prompt twice across
two DP instances) and watch the held-block count return to 0 and `kv_cache_usage_perc` recover;
unit-level, take a `ReqMeta` whose keys all exist and assert `get_finished` returns the id. A
restart that clears the gauge is not a pass.

## The two checks that decide the reporter's build (no run needed)

```bash
# 1. which authority does their send path read? (the whole case turns on this line)
rg -n -A30 'def get_and_clear_finished_requests' \
   <vllm-ascend>/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py
# 2. is the recording save path even taken?
rg -n 'use_layerwise|kv_connector_extra_config' <their --kv-transfer-config JSON / launcher>
```

A reader that iterates `stored_requests` (rather than `finished_requests`) makes the
`set_finished_request` hunk inert; `use_layerwise: true` selects hole 1, and then the
discriminator against hole 2 is whether the usage climbs from the first requests or only once
the instance is loaded. `Error in KVCacheTransferThread` in the paste selects 3 instead.

## What not to re-derive

- Whether a **full pool hit** leaks on the non-layerwise path: it does not at this pin — the
  all-blocks-exist branch `continue`s per group and `dec_stored_request` runs unconditionally at
  the end of `_handle_request` (`kv_transfer.py:289`, `:366`). The report's "all blocks already
  exist → no finished signal" premise is refuted here, and it is refuted *only* for this pin.
- Store capacity, `global_segment_size`, RDMA/RoCE, `--block-size` and the memcache versions:
  none of them can release a request the connector already delayed free.
- Upstream `Scheduler` behaviour: `_free_request(delay_free_blocks=True)` is the contract on
  every v1 release; the defect surface is entirely downstream.

## Scope and recheck

Bound to vllm-ascend `v0.20.2rc1` (`367b8e6`) with vllm `v0.20.2` (`bc150f5`). The reporter's
own tree (v0.18.0 / main) is not in the snapshot, so both answers above are the *pin's*; run the
two greps on their tree before quoting line numbers at them, and re-run them after any bump —
the send-side completion contract has changed shape at least once (the counter and the pending
set are both newer than the v0.18 lineage). PRs #7717 and #9385, which the report cites, are
outside the snapshot and are not evidence here.
