# `MultiConnector` composition: a layerwise P/D connector plus a KV-pool connector

## Scope

A `--kv-transfer-config` whose `kv_connector` is `MultiConnector` with two sub-connectors — typically a
**P/D transport** connector (`MooncakeLayerwiseConnector`) beside a **KV-pool/cache** connector
(`AscendStoreConnector`, `MooncakeConnectorStoreV1`, memcache) — where the failure is *block accounting
inside the pair*, not connectivity and not a stall: an assert on the length of local versus remote
transfer block ids, a connector that suddenly has no local blocks, or a request whose layerwise send and
pool lookup disagree about which blocks the request owns.

Not this page: a single connector that cannot connect or times out (`mooncake-transfer-errors.md`), a pool
that never receives a PUT, or a load that kills the worker (`kv-pool-store-hang.md`), a connector dying at
`register_kv_caches` with a layer-name `KeyError` (`kv-connector-registration.md`), and hybrid
recurrent-state garble (`hybrid-gdn-pd.md`). Those are one-connector defects; this page is the
*composition contract*.

## Common rules

- **Only the connector that won the external-token match gets the request's real blocks.** Upstream
  `MultiConnector.update_state_after_alloc` forwards `blocks`/`num_external_tokens` to the chosen
  connector and `blocks.new_empty()`/`0` to **every other** one:

  ```python
  # vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:292-303 at vLLM v0.13.0
  # (same shape at :322-333 on v0.16.0)
  chosen_connector = self._requests_to_connector.get(request.request_id, -1)
  empty_blocks = blocks.new_empty()
  for i, c in enumerate(self._connectors):
      if i == chosen_connector:
          c.update_state_after_alloc(request, blocks, num_external_tokens)
      else:
          c.update_state_after_alloc(request, empty_blocks, 0)
  ```

  That contract is right for two *cache* connectors and wrong for a transport connector: a layerwise P/D
  connector needs the local block ids even when the pool connector won the match, because its job is to
  ship those blocks to the decoder, not to serve a cache hit. Composed naively, it is handed an empty
  block list and its own bookkeeping diverges — the symptom the reporters see.
- **So the composition works only where the fork replaces `MultiConnector`, and that is a specific
  release.** Verified: at `v0.13.0` `vllm_ascend/distributed/__init__.py`'s `register_connector()` binds
  `MooncakeConnectorV1`, `MooncakeConnectorStoreV1`, `AscendStoreConnector`, `MooncakeLayerwiseConnector`
  and `UCMConnector` — **no** `MultiConnector` entry, so the upstream class runs. `675387f1f` (#7032,
  "[P/D][KVPool]Mooncake Layerwise Connector supports kv_pool") adds
  `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py` (22 lines) and pops the upstream
  registration to bind the *same* name to it (`kv_transfer/__init__.py:22-27` at `v0.16.0rc1`:
  `KVConnectorFactory._registry.pop("MultiConnector")` then register → `AscendMultiConnector`).
  `git tag --contains 675387f1f` ⇒ first tag **`v0.16.0rc1`**.
- **The fix is one override — read it, it *is* the mechanism.** `AscendMultiConnector` overrides only
  `update_state_after_alloc` and forwards the real `blocks`/`num_external_tokens` to the chosen connector
  **and to every `MooncakeLayerwiseConnector`**, keeping `blocks.new_empty()` for the rest. The same
  commit moves `ensure_kv_transfer_initialized(...)` from `NPUWorker` setup into
  `initialize_from_config(kv_cache_config)` (`worker/worker.py`, `v0.16.0rc1`), i.e. the connector is
  initialised with the allocated KV config rather than before it.
- **Nothing user-facing changes on upgrade.** The config still says `kv_connector: "MultiConnector"`; the
  registration swap makes it an `AscendMultiConnector`. Say that explicitly, or the reporter will go
  looking for an `AscendMultiConnector` flag that does not exist.
- **Do not build a "both connectors overwrite the same `kv_transfer_params` key" story.** The layerwise
  connector writes `request.kv_transfer_params["remote_block_ids"]` itself, from its own `SendReqInfo`
  (`mooncake_layerwise_connector.py:756-757` at `v0.13.0`); the verified defect is the *blocks it was
  given*, one call earlier. A hypothesis about shared-dict overwrites adds a second unproven mechanism to
  an answer that already has a verified one.
- **Quote the assert's *statement*, not the reporter's line number.** #7085 reports `:758` while the
  statement sits at `:759-761` on `v0.13.0` (the f-string is `:760`); the frame above it is
  `patch_core.py:70`, i.e. the scheduler step, not the connector's own entry point. The connector module
  also moves: `vllm_ascend/distributed/mooncake_layerwise_connector.py` up to `v0.13.0`,
  `vllm_ascend/distributed/kv_transfer/kv_p2p/` from `v0.16.0rc1` — a missing path at one tag is a move,
  not a removal.
- **The maintainer comment usually names the fix, so the reply is one tag lookup away.** #7085's whole
  thread is a ping plus `liziyu179`'s "We fixed this issue with this PR.
  https://github.com/vllm-project/vllm-ascend/pull/7032". `open_answer.sh` + `/issues/N/comments` is batch
  1; `git tag --contains <sha>` on the PR's commit is the only thing batch 2 owes.

## The check that settles it

```bash
R=/workspace/repos/vllm-project/vllm-ascend; V=/workspace/repos/vllm-project/vllm
# 1. does this ref bind MultiConnector itself? (path moved: distributed/__init__.py at v0.13, distributed/kv_transfer/__init__.py later)
git -C $R show <ref>:vllm_ascend/distributed/kv_transfer/__init__.py 2>/dev/null | grep -n -i "multi" \
  || git -C $R show <ref>:vllm_ascend/distributed/__init__.py | grep -n -i "multi"
# 2. first tag of the fix, if the thread names it
git -C $R log --format=%H -1 -i --grep='layerwise connector supports kv_pool'; git -C $R tag --contains <sha> | sort -V | head -3
# 3. the assert site at the reported ref (function name is misspelled "add_tranfer_task")
git -C $R show <ref>:vllm_ascend/distributed/mooncake_layerwise_connector.py | sed -n '745,760p'
```

## Worked case: vllm-ascend#7085 (`v0.13.0`, 910B ×8, Qwen3-4B, P 0-3 / D 4-7, TP=1 DP=1)

Reported: `MultiConnector` = `MooncakeLayerwiseConnector` + `AscendStoreConnector` on both sides,
`/v1/completions` on the D node produces

```text
AssertionError: len of local trans block ids : 0 not equal to the len of remote trans block ids : 2
  File ".../vllm_ascend/distributed/mooncake_layerwise_connector.py", line 758, in add_tranfer_task
```

Verified at `v0.13.0`: the assert is at `:758`, directly under the two slices
`local_block_ids[local_transferred_tokens//block_size : local_computed_tokens//block_size]` and the
remote twin offset by `remote_cache_tokens` (`:747-755`), so the empty side is the *local* block list —
exactly what a starved connector is given. No `MultiConnector` replacement exists at that tag; the fix
`675387f1f` is first in `v0.16.0rc1`.

Answer shape (this one fits in ~1 200 characters, and the thread already carries the workaround):

- **Not supported at `v0.13.0`**; the maintainer confirmed and named #7032.
- **Workaround:** upgrade to `>= v0.16.0rc1`, config unchanged (`kv_connector: "MultiConnector"`); keep
  both sides symmetric with `kv_role` `kv_producer`/`kv_consumer`. On `v0.13.0` the fallback that runs is
  one connector only — drop `AscendStoreConnector` (lose pool reuse) or drop the layerwise connector (lose
  layer-wise overlap); say which capability is traded.
- **Root cause:** `vllm/distributed/…/multi_connector.py:292-303` (vLLM `v0.13.0`) gives non-chosen
  connectors empty blocks, so at `v0.13.0` the layerwise connector receives none while the pool side
  matched 2 blocks → `:758`.
- **Status:** fixed by #7032, first release tag `v0.16.0rc1`; the issue was still **open** at the
  2026-11-17 fetch — the thread carries the fix link (`liziyu179`) and no close.

Falsifier: with only `MooncakeLayerwiseConnector` configured (no pool child) the assert disappears on the same
tag — that isolates the combination rather than the connector.
