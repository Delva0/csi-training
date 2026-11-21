# `Error in KVCacheTransferThread. error=unhashable type: 'list'` is a mask, not a cause

Verified at vllm-ascend **v0.21.0rc1** (`80610e4438db`) + vllm **v0.21.0**
(`ad7125a`); recorded for #10962 (A3 four-node 2P1D, kimi-k2.6, `terminal-bench2`
at concurrency 8, the **D node** prints the line while the service keeps
answering). The reported version was v0.22.1rc1, which the snapshot does not
carry — the trigger line is ref-specific and the recheck command is below.

## The signature and what it cannot contain

```
Error in KVCacheTransferThread. error=unhashable type: 'list'.
```

This is the *catch-all* of the receive thread:
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:563`
`logger.error("Error in KVCacheTransferThread. error=%s. ", e)`, inside
`KVCacheRecvingThread.run()` (`:551-563`). The same sentence is also printed by
`mooncake_hybrid_connector.py:499`, so the text does **not** name the connector —
the payload does: only `mooncake_connector.py` can raise `unhashable type: 'list'`
(see below), the hybrid file has no such bookkeeping.

Read it as: *a real failure happened inside `_handle_request`, and the logging
that would have named it was replaced by a bookkeeping `TypeError`.*

## The mechanism (one file, five lines)

| Site | Fact |
|---|---|
| `mooncake_connector.py:473` | `self.invalid_block_ids: set[int] = set()` — flat by type |
| `:542-545` | `_mark_failed_recv_request(request_id, local_block_ids: list[int])` → `self.invalid_block_ids.update(local_block_ids)` |
| `:99`, `:494`, `:509` | the value handed to it is `req_meta["local_block_ids"]`, declared `BlockIds` — **per-group** |
| `:1463` | where it is built: `local_block_ids = blocks.get_unhashed_block_ids_all_groups()`; upstream `vllm/v1/core/kv_cache_manager.py:87` is `-> list[list[int]]`, and `:626`/`:666` index it `local_block_ids[group_idx]` |
| `:576`, `:585` | the two call sites; both run *after* something already failed (`:576` = request previously marked failed; `:585` = inside `except Exception` around `_transfer_kv_cache_all_groups`) |
| `:585-586` | `self._mark_failed_recv_request(...)` sits **before** `logger.exception("Failed to transfer KV cache for request %s: %s", …)` — so the mark's `TypeError` replaces the real traceback; the real one is never printed |

`set.update(<list of lists>)` raises `TypeError: unhashable type: 'list'` even for a
single KV group (`[[…]]` is still a list of lists), so **any** failed recv through
this connector ends at this log line. The exception then leaves `_handle_request`
(its `finally` still runs) and is caught by `run()`.

Two consequences belong in the answer:

- The real cause is whatever raised in `_transfer_kv_cache_all_groups` — a pull
  failure, a group block-count mismatch, a spec-decode block arithmetic — and the
  paste carries **no** information about it.
- `_handle_request`'s `finally` (`:587-599`) still calls `update_done_task_count`,
  `_clear_failed_recv_request` and sends the done/free signals, so the D side can
  go on and decode a request whose KV never arrived. "The service still serves" is
  therefore not evidence that the transfer worked; garbled or empty answers are
  the expected companion symptom, and nothing retries.

Ownership is downstream, not a bad port of upstream: upstream vllm v0.21.0's
`kv_connector/v1/mooncake/mooncake_connector.py` has no `invalid_block_ids` /
`_mark_failed_recv_request` at all (it records failures via
`xfer_stats.record_failed_recv()`, `:1579`/`:1588`), and the contract it satisfies
is flat — `vllm/distributed/kv_transfer/kv_connector/v1/base.py:374`
`get_block_ids_with_load_errors() -> set[int]`, implemented downstream at
`mooncake_connector.py:1924-1926`. The same commit's sibling does it correctly:
`mooncake_layerwise_connector.py:1350`
`self._invalid_block_ids.update(block_id for group in meta.local_block_ids for block_id in group)`.
That is the one-line fix, and it also proves the flatten is intended.

## Procedure

1. **Unmask first — it is cheaper than any transport theory.** In the reporter's
   tree either flatten `:545`
   (`self.invalid_block_ids.update(b for g in local_block_ids for b in g)`) or move
   `:585` below `:586`. Re-run the same load; the
   `Failed to transfer KV cache for request …: <real error>` line is the missing
   stack. One `sed -n '473p;542,546p;583,588p;1460,1465p' <file>` on *their* tree
   says whether the shape still holds at their ref.
2. **Ablate the connector instead of patching, when patching is not allowed:**
   the same load on `MooncakeLayerwiseConnector` (already flattened) serves or
   prints the real error without masking. If it serves cleanly, the bookkeeping
   bug is confirmed *as the mask* — not as the cause.
3. Only then classify the masked failure, in this order:
   - a **group/block-count mismatch** — the MambaSpec branch
     (`:699-702`) raises `RuntimeError("For MambaSpec num block should equal on P
     node and D node.")` and then indexes
     `len(remote_group_block_ids) - self.num_speculative_tokens - 1`: asymmetric
     `--max-model-len`, differing spec-decode settings or P-side prefix caching
     land here first. Ablations: concurrency 1, `--no-enable-prefix-caching`,
     drop `--speculative-config`, one axis per run;
   - a **transfer-engine pull failure** (HCCL/ADXL, the concurrency-shaped one) —
     D-side plog, `Failed to transfer KV cache` / `Mooncake transfer failed` after
     the unmask patch.
   - **not** the asymmetric-TP pull-count assert: `chosen_rank_list(…) does not
     match num_group_pulls(…)` is raised in the worker, not in this thread, so
     its absence from the log is informative rather than a mystery
     (`asymmetric-tp-pull-count.md`).
4. Ask for: both serve commands, the `kv_connector` name from
   `--kv-transfer-config`, the **unfiltered** D-node log, and the D-side plog.
   Guidance the reporter can act on: that line is not a transport error code and
   not an HCCL fault, so re-running with more cards or a longer timeout changes
   nothing.

## Traps

- **Do not build a HCCL/rank-table/driver answer from this text.** By
  construction the message carries no transport information; the safest sentence
  in the reply is "the line you pasted is the mask".
- **Do not call it "Mooncake is broken".** The connector is vllm-ascend's own copy
  (`kv_p2p/`), and the same-commit sibling in the same directory is correct.
- **`unhashable type: 'list'` elsewhere is a different owner** — a `KeyError`/`None`
  hole in the pooling batch bookkeeping is `../pooling-embedding/batch-index-desync.md`;
  only the `KVCacheTransferThread` sentence is this page.
- **`kd`/hybrid models**: if the deployment uses `MooncakeHybridConnector` the
  same sentence can appear without the set, so `error=unhashable type: 'list'`
  itself is the evidence that the v1 connector printed it.

## Recheck conditions

- Later refs may have flattened `_mark_failed_recv_request` (the layerwise file
  already is) — re-grep `invalid_block_ids.update` at the reporter's release
  before presenting the bug as present.
- If `get_unhashed_block_ids_all_groups()` is ever replaced by a flat accessor at
  the call site, the grouped type disappears and the bug becomes unreachable.
- The `finally` cleanup behaviour (`:587-599`) is the part that turns the mask
  into silent partial-KV decode; verify it before promising that a fixed log line
  also fixes the output.

## Recorded run (#10962) and how it ended

Twelve assistant turns of pure reasoning (one status sentence on turn 2), then
the whole ranked answer written on the last turn and cut mid-sentence at the
16 384-token cap — the mechanism above was source-verified by the end of the third tool result. The answer to ship from this page is four lines plus the unmask patch;
the transport ladder is the *follow-up*, not the deliverable
(`../evidence-workspace/delivery-discipline.md`).
