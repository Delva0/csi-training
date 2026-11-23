# Legacy wrapper vs a moved upstream core API: audit four contracts, not the first `TypeError`

## Scope

A report whose deepest downstream frame is a **wrapper** class that delegates to an
upstream vLLM core object (`BlockPool`, `SingleTypeKVCacheManager`) or to a small
ZeroMQ metadata service of its own. The recorded case is the legacy CPU-offload
connector, `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/`
(`CPUKVCacheManager` + `CPUOffloadingConnectorWorker/Scheduler`), later removed from
main by #11904; the shape recurs for any downstream copy of an upstream manager API.

The paste names **one** call. The file lags the upstream object in four different
ways, so (a) "pass the missing kwarg" is not the fix, and (b) a call-site count taken
from *argument* diffs is wrong — the statement that fails first after the constructor
is shimmed has a **matching signature**.

Verified at vllm-ascend v0.18.0rc1 `99e1ea0fe685` / vllm v0.18.0 `bcf2be961200`
(the reporter's own pins, case #17965). Statements about the v0.23.0 image
(`sha256:660ce23a…`, ascend `5cb98caa`, vllm `0fc695fc`) are *reporter-observed* and
are not claims about main.

## The four contracts — one read each, all in the same two trees

| Contract | Pinned evidence (v0.18.0rc1 / vllm v0.18.0) | What you miss by stopping earlier |
|---|---|---|
| 1. **required kwarg added** | wrapper `cpu_kv_cache_manager.py:69-73` calls `get_manager_for_kv_cache_spec(kv_cache_spec=…, block_pool=…, kv_cache_group_id=0)`; upstream `__init__` (`single_type_kv_cache_manager.py:34-40`) also requires `enable_caching`, and the factory forwards `**kwargs` verbatim (`:1120-1125`) | the reported `TypeError` itself. Correct value is `True`: with `False`, `allocate_new_computed_blocks` asserts on any hit (`:190-195`), i.e. the CPU prefix pool stops being a pool |
| 2. **methods renamed / re-signatured** | wrapper `:133` `get_num_blocks_to_allocate(request_id, num_tokens, new_computed_blocks)` lacks `total_computed_tokens` + `num_tokens_main_model` (upstream `:78-85`); wrapper `:144` calls `save_new_computed_blocks`, a name that does not exist anywhere in vllm v0.18.0 (the equivalent is `allocate_new_computed_blocks`, upstream `:142`); wrapper `:146` `allocate_new_blocks(request_id, num_tokens)` lacks `num_tokens_main_model` (upstream `:215-217`) | `TypeError`/`AttributeError` three calls later |
| 3. **return / container shape** | `find_longest_cache_hit` is declared `-> tuple[list[KVCacheBlock], ...]` (`:311-322`, both classmethods, one list per KV-cache group). The wrapper stores `computed_blocks[0]` (`:106-107`) and then feeds the **whole tuple** to `BlockPool.touch` (`:109`), whose loop reads `block.ref_cnt` (`block_pool.py:392-405`) | `AttributeError: 'list' object has no attribute 'ref_cnt'` on the *first* lookup — the call a signature diff reports as "the only one that still matches". It fails on a cold pool too, because the tuple holds an (empty) list either way. Fix: `touch(computed_blocks[0])` |
| 4. **attribute written on a field the upstream class no longer has** | `cpu_offload_connector.py:159` and `:221` do `request.get_hash_new_full_blocks = None` before the RPC. `Request` keeps the hasher in `self._block_hasher` (`vllm/v1/request.py:169`, used at `:218-219`); `get_hash_new_full_blocks` has no hit in `vllm/` | the statement is a **no-op that creates a new attribute**, so the closure stays on the object and `pickle.dumps(request)` (`metadata.py:63`) raises `AttributeError: Can't get local object 'get_request_block_hasher.<locals>.request_block_hasher'`. `get_request_block_hasher` (`kv_cache_utils.py:562-576`) returns a nested function, and EngineCore installs one whenever prefix caching is on **or any KV connector is attached** (`v1/engine/core.py:198-206`) — so the blocker is unavoidable in a real engine and only a hand-built fixture `Request` (no `block_hasher=`) gets past it. That `AttributeError` is a picklability fact about a closure, not a drift in the callee |

Contract 4 is the one that hides best: the write looks like an existing "strip the
hasher" preparation, and a fixture-built `Request` without a hasher passes. Two greps
settle it — `rg -n '<field>' <upstream tree>/vllm` (absent → the write is dead) and a
read of the upstream `__init__` for the live field name.

## The same wrapper also drops the release notification (source-verified)

`get_finished` (`cpu_offload_connector.py:332`) removes the finished worker requests,
then returns early for a single rank: `if self.tp_world_size == 1: return done_sending`
(`:342-343`). Only the rank-0 `TP>1` branch reaches `_sending_finished`
(`:359-362`, `:368-372`), which is the `cache_and_free_slots` RPC
(`cpu_kv_cache_manager.py:157-167`) that stores then frees the pool blocks; the
scheduler-side record is `record_request_cache_and_free_slots` (`:153-155`,
`req_to_free`). At TP=1 the CPU metadata server therefore keeps the request's blocks
owned, with no path left that frees them.

Cheap probe, CPU only: snapshot the pool before and after the worker reports
completion — free-block count, `manager.single_type_manager.req_to_blocks`,
`manager.req_to_free`. Reporter-observed at v0.23.0: `free` 7 → 5 after a byte-exact
save with `owners {'cold': [1, 2]}` and `pending_free ['cold']`; invoking
`_sending_finished({'cold'})` by hand restores 7 and empties both maps. The same branch
shape exists at v0.18.0rc1, so the gap is not a v0.23.0 regression.

## Decide reachability and version before the repair count

- **Registered?** `vllm_ascend/distributed/kv_transfer/__init__.py` registers
  `MultiConnector`, `MooncakeConnectorV1`, `MooncakeStoreV1`/`AscendStoreConnector`,
  `MooncakeLayerwiseConnector`, `UCMConnector`, `LMCacheAscendConnector` — no
  CPU-offload name. `--kv-connector CPUOffloadingConnector` cannot reach this class, so
  a stack through it came from importing it directly: say so.
- **Deprecated?** `docs/source/user_guide/release_notes.md:314` (v0.13.0) and `:449`
  (v0.13.0rc2): "`CPUOffloadingConnector` is deprecated. We'll remove it in the next
  release. It'll be replaced by CPUOffload feature from vLLM". The pinned tree's
  validated path is already the replacement: `tests/e2e/singlecard/test_cpu_offloading.py:137`
  (`kv_connector="OffloadingConnector"`, `spec_name=NPUOffloadingSpec`), wired at
  `.github/workflows/scripts/config.yaml:24`. At v0.18.0rc1 no guide page mentions CPU
  offload at all (`rg` over `docs/source`: release notes only), so the v0.23.0 guide
  the reporter cites postdates the pin.
- **Which version breaks?** The constructor `TypeError` is **v0.18.0rc1 and earlier**:
  reporter-observed at v0.23.0, the legacy `CPUKVCacheManager` passes
  `enable_caching=True` and the original exception no longer reproduces — while
  contracts 2-4 above still fail. Never answer "still broken in the newer image" from
  the paste alone; re-read that file at the reporter's ref.

## Before recommending the replacement, import it

"Switch to the supported connector" is advice about a class the reporter will import in
*their* image, so check the module path at their pin. At v0.23.0 the documented
`vllm_ascend.kv_offload.npu` / `NPUOffloadingSpec` dies at import:

```text
vllm_ascend/kv_offload/npu.py:7
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
ModuleNotFoundError: No module named 'vllm.v1.kv_offload.abstract'
```

that image ships `vllm/v1/kv_offload/base.py` instead. At v0.18.0rc1 the same line
resolves — `vllm/v1/kv_offload/abstract.py` exists — so this is an upstream module
rename out from under a pinned downstream import, contract kind 2 in module form, and it
blocks the replacement path *before* any KV code runs. One call decides it on the
reporter's box: `python -c "import vllm_ascend.kv_offload.npu"` (or
`ls <site-packages>/vllm/v1/kv_offload/`). When the replacement does import, its own
failure mode is the MLA layer filter in `kv-offloading-layer-coverage.md`.

## Answer shape

1. Yes, the reported crash is real on their pin, and it is downstream/upstream drift in
   a deprecated, unregistered path — not a device, model or allocator fault.
2. Do **not** stop at the quoted call. List the four contracts with `path:line` and say
   which one fails next; that is what turns "pass `enable_caching`" into an honest plan
   and what keeps the reporter from reporting the next three as separate bugs.
3. Count call sites over all four kinds. A grouped-vs-flat block list and a write to a
   field upstream no longer reads are invisible to an argument diff.
4. Verdict: unwired and deprecated, so the action is the replacement connector — with
   its **import check** — and, only if the legacy path must run, both the
   hasher/pickle blocker and the TP=1 missing release.
5. Scope block: which claims are source-verified at the pin, which are the reporter's
   later image observation, and what needs a re-grep (the constructor, the
   `abstract`→`base` name, `touch`'s shape).

## Recheck conditions

- `enable_caching` is present in this wrapper at some ref after v0.18.0rc1; re-read
  `cpu_kv_cache_manager.py` at the reporter's ref before repeating the `TypeError`.
- `vllm.v1.kv_offload.abstract` vs `base`: list the image's `vllm/v1/kv_offload/`
  instead of trusting either version named here.
- If upstream ever flattens the grouped return of `find_longest_cache_hit`, contract 3
  disappears and `touch(computed_blocks)` becomes correct.
