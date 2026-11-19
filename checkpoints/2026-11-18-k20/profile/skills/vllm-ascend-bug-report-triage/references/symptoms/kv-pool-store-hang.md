# KV pool / connector: nothing is ever PUT, the lookup line repeats, the load is skipped, or the worker dies on a sync load

## Scope

A request stops making progress — or the worker process dies — while a `--kv-transfer-config` connector
is active. Five halves fail independently, and the log tells them apart before any code read:

- **PUT half** — the producer never saves: the store master shows no data (`Keys: 0`,
  `PutStart=0/0, PutEnd=0/0`) while the P node's KV cache grows until new requests sit in `waiting`. A
  PD deployment with a store/P2P connector and a `--speculative-config` on the node is the tell
  (#8808, section A below).
- **LOAD half** — the scheduler re-enters the waiting queue: the pool lookup line repeats for the same
  request id with an unchanged `need to load` (#6512, section B).
- **ZERO-LOAD half** — nothing stalls and nothing errors, but the pool earns nothing: `kvpool hit
  tokens` is positive while every request logs `need to load: 0`, so each request still pays the lookup
  RPC and loads nothing, and throughput lands *below* the same workload without the connector (#13356,
  section E1). This one is upstream semantics, not a defect.
- **SYNC-RELOAD (DEATH) half** — the pool is nearly full, one request triggers a top-up with
  `load_async=False`, and the **worker process dies** seconds later: the load-spec line appears once, then
  `Worker proc VllmWorker-N died unexpectedly` with no Python traceback above it and no core dump (#16342,
  section C). Not a stall: nothing repeats.
- **CONFIG half** — nothing starts because the store backend refuses the config: a `protocol` other
  than `ascend` in `mooncake.json` (section D), which is a support question ("must I build Mooncake with
  `-DUSE_ASCEND_DIRECT=ON`?") as often as it is a crash.
- **INIT half** — the worker dies at connector construction with a bare `assert res == 0` inside the
  backend's `_setup_store` (section E2): the store library's own status code was non-zero and nothing
  printed it, so the cause is in *its* log, not in a vLLM frame.

Not PD P2P/Mooncake layerwise hangs with an HCCL `STUCK` (`pd-collective-hang.md`), not a Mooncake
transfer that **fails** with an explicit transport error or times out between P and D
(`mooncake-transfer-errors.md`), not sparse C8 (`c8-and-kv-layout.md`), not `store_kv_block` (same
page), and not a hybrid model's missing recurrent state (`hybrid-gdn-pd.md`).

## Path drift that costs a run

At `v0.13.0rc1` the scheduler side of the pool connector is
`vllm_ascend/distributed/kvpool/pool_scheduler.py` (`class KVPoolScheduler`, peer packages
`ascend_store_connector.py`, `pool_worker.py`, `backend/{mooncake,memcache}_backend.py`). From
`v0.14.0rc1` the package moved to
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py` (distributed refactor
`295018ec0`, #5719/#5910) and keeps that path through `HEAD`.

**`vllm_ascend/core/pool_scheduler.py` has never existed** (checked `v0.13.0rc1`, `v0.14.0rc1`,
`v0.18.0rc1`, `v0.20.2rc1`, `v0.23.0rc1`, `HEAD`). A log line quoted as `core/pool_scheduler.py` is a
reporting or transcription error: grep the two real paths above, or `git ls-tree -r --name-only <ref>
| grep pool_scheduler` (`../vllm-ascend/code-layout.md`).

## Read the line before reading the code

At `v0.13.0rc1` (`distributed/kvpool/pool_scheduler.py`), `get_num_new_matched_tokens` starts at `:51`
and computes

```python
num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes)
if num_external_hit_tokens == request.num_tokens: num_external_hit_tokens -= 1
need_to_allocate = num_external_hit_tokens - num_computed_tokens        # :84
logger.info("Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d", ...)   # :86
if need_to_allocate <= 0: return 0, False
self.load_specs[request.request_id] = LoadSpec(..., can_load=False)
```

- `need to load` is a **gap** (`hit − computed`), not a transfer in flight. A pure-positive value
  repeated for one req id means the scheduler re-entered the waiting-queue path for that request
  without progress: the load never completed, or the request never got the blocks it asked for.
- The lookup is skipped altogether when `kv_role == "kv_consumer"` and `consumer_is_to_load` is unset
  (`:68`). So a repeating hit line also tells you the deployment is `kv_both`, a producer, or has
  `consumer_is_to_load: true` — a `kv_consumer`-only config cannot be this symptom.
- Per-step repetition is expected in itself (the connector is consulted for waiting requests); what
  localises the bug is the *same* request id with an unchanged `need to load` across many steps.
- **The same two lines at `v0.23.0rc1`** (`kv_pool/ascend_store/pool_scheduler.py`): the hit/need line is
  `:534-540` and the load-spec line is `:554-562`
  (`KV pool load spec created req=%s vllm_cached=%d kvpool_cached=%d need_to_allocate=%d load_async=%s
  use_layerwise=%s`). Both are `logger.info`, so a report that shows them is quoting the scheduler, not the
  worker. `:565` `return need_to_allocate, self.load_async and not self.use_layerwise` is how the connector
  tells vLLM whether the load will complete asynchronously — the flag the DEATH half turns on.

## A. Nothing is PUT: the runner skipped `finalize_kv_connector` under spec decode (#8808)

`v0.18.0rc1`, GLM-5.1-w4a8, A2, two-node PD with `AscendStoreConnector` over a Mooncake master — as a
`MultiConnector` with `MooncakeConnectorV1` (P2P), both `kv_role: kv_producer` on P — DP2 + TP8 + EP,
`--enable-chunked-prefill`, `--no-enable-prefix-caching`, `--enforce-eager`,
`VLLM_ASCEND_BALANCE_SCHEDULING=1`, and
`--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'` on both nodes.
Symptom: the P node's KV cache keeps rising, then every further request stays `waiting`; the engine
never errors. On `v0.18.0` and later the same config serves.

- **Read the store master first.** The reporter's mooncake master log repeats
  `Mem Storage: 0 B / 32.00 GB | Keys: 0 | Requests: PutStart=0/0, PutEnd=0/0 | ExistKey:(Req=5/0/5,
  Item=665/665)` — lookups happen, **zero puts** ever start. That single line separates this half from
  a load-side stall, which shows hits in the master and a repeating hit/need line in the scheduler.
  Maintainer `Pz1116` asked exactly this: "can you check whether KV Cache is correctly put into KV
  Cache Pool in your case?"
- **Mechanism, verified at `v0.18.0rc1`** (`vllm_ascend/worker/model_runner_v1.py`): `:1367`
  `clear_kv_metadata = self.speculative_config is None`; `:1382-1387` passes
  `maybe_get_kv_connector_output(scheduler_output, **{"defer_finalize": not clear_kv_metadata})`, so
  with spec decoding the target-model forward defers finalization and does not `wait_for_save`; the
  post-draft block at `:1581` then calls only `get_kv_transfer_group().clear_connector_metadata()`.
  Upstream vLLM had already moved that step: `v0.18.0` `vllm/v1/worker/gpu_model_runner.py:3784` passes
  `defer_finalize=defer_kv_connector_finalize` and `:4046-4047` runs `if spec_config is not None:
  self.finalize_kv_connector()`, where `finalize_kv_connector()` is `wait_for_save()` +
  `clear_connector_metadata()` (`kv_connector_model_runner_mixin.py:81-89`; `_get_kv_connector_output`
  skips both when `defer_finalize` is set, `:71-95`). The ascend runner forked `execute_model` and kept
  the old `clear_connector_metadata()` line, so a queue whose save is awaited later is cleared instead
  of awaited → the PUT (P2P save and KV-pool put alike) never completes, the blocks stay referenced, KV
  usage climbs, and the scheduler has nowhere to put new requests.
- **Fix, verified**: `1225c613f` — "[BugFix][0.18.0][KV Pool] Fix KV Pool not putting kv cache for vllm
  v0.18.0" (#7874, merged 2026-04-02 into `releases/v0.18.0`, 5+/2− in `model_runner_v1.py`); it
  replaces the `clear_connector_metadata()` line with `if self.speculative_config is not None:
  self.finalize_kv_connector()` (`v0.18.0:vllm_ascend/worker/model_runner_v1.py:1554`). Present at
  `v0.18.0` and at `v0.20.2rc1:2233`; **absent at `v0.18.0rc1`** — `finalize_kv_connector` does not
  exist there at all (`git grep -c` → nothing), which is itself the discriminator for an rc1 report.
  So this is a release-rc gap, not an open bug.
- **Workaround**: a build that contains #7874 — `v0.18.0` GA or any later tag — or the cherry-pick.
  Reporter-confirmed (2026-05-14): "This #7874 pr is usable", later 「没有出现waiting的问题」 (no more
  waiting), after the zero-put master log; maintainer `Pz1116` pointed at #7874 twice (`reported`). The
  same reporter then measured a TTFT regression on the pooled 0.18 line (127k in/1k out, 4-way: 40 s on
  `0.17.0rc1` vs 150 s on `0.18.0rc1`/`0.18.0`) — a separate issue, do not fold it into this cause.
- **Not the lever**: removing `VLLM_ASCEND_BALANCE_SCHEDULING=1` (tried, 无效) and raising
  `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` to 600 both did nothing (`reported`). Both belong to the
  #8975-style scheduler hang (`pd-collective-hang.md`); #8063 was closed unmerged and is not the fix.
  Offering them for a zero-put stall costs the reporter a restart.
- **Falsifying checks:** a build without speculation (no `--speculative-config`) that still shows
  `PutStart=0/0` is a different producer bug; a run at `v0.18.0rc1` whose master log moves `PutEnd`
  while P still waits is the load half, not this one; and a non-zero `PutEnd` with garbage output is
  the decode-side cache path (`c8-and-kv-layout.md`).

Recheck: `git grep -n 'defer_finalize\|finalize_kv_connector' <ref> -- vllm_ascend/worker`; the symbol
only appears on builds that contain the fix, and its absence is a positive finding, not a failed read.

## B. The lookup line repeats: #6512 (`v0.13.0rc1`, Mooncake + Qwen3-Embedding-8B, `vllm bench`, 9000 requests, `--runner pooling`, `kv_role: kv_both`, `--async-scheduling`)

Stalls around request 382 of 9000, repeating `kvpool hit tokens: 1024, need to load: 1024` (~45 times
inside one second); the next request logs `need to load: 0`, i.e. that one found the prefix in the local
cache.

- **Workaround** (reporter-confirmed, two independent statements): removing `--kv-transfer-config`
  lets all 9000 requests finish; upgrading to `v0.14.0rc1` fixes it on the same cluster.
- **Root cause: range only, not bisected.** No NPU here, and the pool path was rewritten twice in the
  range. Commits touching this package in `v0.13.0rc1..v0.14.0rc1` (one `git log --oneline
  v0.13.0rc1..v0.14.0rc1 -- vllm_ascend/distributed/kvpool vllm_ascend/distributed/kv_transfer/kv_pool`):
  #5751 (KV-pool memory retention / missing PUT), #6045 (double-free of stacking buffers on pooling
  layers under recompute), #6049 (rewrites preempted/resumed-request tracking in `pool_scheduler.py`,
  +88/−33), #6113 (rank 0 HBM), plus two directory refactors. #6045 and #6049 are the closest fits;
  neither was shown to fix this report.
- **Do not cite #6153** — "fix kv pcp+pooling+pd separation bug" is a plausible-looking subject in
  that range, but the commit (4173255c0) only touches
  `distributed/kv_transfer/kv_p2p/mooncake_connector.py`, the P2P connector (Mooncake multi-node
  `get_remote_port_send_num` mapping), not the store/pool scheduler. Posting it as the fix is wrong.
- Honest form: "reporters confirm `v0.14.0rc1` fixes it; the change is somewhere in the pool-scheduler
  rework between the tags (#6045/#6049 candidates), and the request-stall signature is the one above."

## C. The sync-reload death: `load_async=False`, a near-full pool, and a worker that vanishes (#16342)

`v0.23.0rc1` (image `v0.23.0rc1-a3-openeuler`), GLM-5.2 **W4A8C8**, 4 × Atlas 800 A3, PD disaggregation with
`MultiConnector` (`MooncakeConnectorV1` with `use_ascend_direct` + `AscendStoreConnector` over a memcache
backend), MTP speculation, `enable_sparse_c8`, `--max-model-len 990000`, prefill DP2×TP8. The report holds
**two independent crashes**; this section is the second one. (The first — `EE9999 rtEventQueryStatus … aicore
execution times out` plus an HCCL watchdog abort under ~192 concurrent clients — is a device/watchdog class:
`device-kernel-faults.md`, `pd-collective-hang.md`.) The second arrives after 17 h of steady low-concurrency
traffic, with the P-device KV pool at 95.7 % (`994095` tokens; startup line `Maximum concurrency for
990,000 tokens per request: 1.00x`).

### Signature, in order

```text
Reqid: chatcmpl-<id>, Total tokens 455394, kvpool hit tokens: 453632, need to load: 1024
KV pool load spec created ... vllm_cached=452608 kvpool_cached=453632 need_to_allocate=1024 load_async=False
ERROR ... Worker proc VllmWorker-5 died unexpectedly, shutting down executor.
EngineCore encountered a fatal error. RuntimeError: Executor failed.
(DP2) TimeoutError: RPC call to sample_tokens timed out.
```

No Python exception and no core dump on the dead rank; the host is healthy (no kernel OOM, no container
restart), and the driver prints `trs_drv "Logic cq stat show"` queue dumps on several cards at that moment.
**`load_async=False` in the load-spec line is the discriminator**: it selects the inline transfer path
below, and the later `Executor failed` / `sample_tokens timed out` lines are the cascade, not the fault.

Both log lines are the **scheduler**'s, not the worker's — their positions and format strings at this tag
are in "Read the line before reading the code" above — so the death is downstream of a scheduling decision
the connector had already announced to vLLM (`:565` returns `need_to_allocate, self.load_async and not
self.use_layerwise`).

### What the ref does (verified at `v0.23.0rc1`, `kv_pool/ascend_store/pool_worker.py`)

- `:133` `self.load_async = extra_config.get("load_async", False)` — the connector's extra config decides
  the mode, default async **off**; documented in `docs/source/user_guide/feature_guide/kv_pool.md` at that
  tag.
- `_start_kv_transfer_threads` (`:350`) creates `self.kv_recv_thread` only on the `if self.load_async:`
  branch (`:455`); with the default there is **no receive thread at all**.
- `start_load_kv` (`:~745`) then does the transfer on the caller's thread: after computing
  `load_spec.kvpool_cached_tokens` / `token_len`, `if self.load_async: self.kv_recv_thread.add_request(...)
  ; continue`, otherwise it builds `addr_list` / `size_list` / `key_list` / `block_id_list` and calls
  `self.m_store.get(key_list_c, addr_list_c, size_list_c)` **inline** (`:840`). The backend's `get` runs the
  host/device copies on that worker thread; nothing is isolated from the engine loop, so a native fault
  there ends the process with no Python frame.
- A **returned** failure status is already handled: `record_failed_blocks` → `self._invalid_block_ids`
  (`:842`, `:857`, with a hybrid-group branch that logs `KV load failed for hybrid request …` and skips the
  invalid-block fallback). So "the store returned an error" cannot explain this death on its own; what is
  left is the native call itself or its device queue.
- Consequence for triage: the failing rank left no Python frame, so whatever killed it happened below
  Python (CANN op, HCCL/driver, or the transfer engine inside `memcache`). No clone settles that —
  `hypothesis` territory, not a `path:line` for the death.

### Leads after the ref (subjects that name the mechanism; none confirmed)

One call, 25 subjects, all **after** `v0.23.0rc1` so none can be in the reporter's build:
`git log --oneline v0.23.0rc1..HEAD -- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`

| commit | subject | why it is a lead |
| --- | --- | --- |
| `d5923bcd7` | `[BugFix][KVPool] Fix segmented bulk KV transfers (#19273)` | segmented bulk transfer arithmetic matches the shape of a 1024-token top-up over a 453 632-token cached prefix; its bulk `prepare_value` scaled every physical segment's length by the requested fraction and ignored the token offset. Only the subject was read in the recorded run — open `git show --format= d5923bcd7 -- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/` before quoting a mechanism, and do not restate an unread diff from memory |
| `2581a69bc` | `[BugFix][KVPool] Isolate Memcache transfer failures (#19101)` | "memcache returns failed operation statuses, but Python transfer paths can raise on these results and leave layer fences or queue tasks incomplete"; it also notes the scheduler cannot yet terminate a **synchronous** failed request |
| `41158806d` | `[BugFix][KV Pool] Invalidate stale load plans and bound worker transfers (#19785)` | bounds worker transfers and invalidates stale plans — the sync path's block/addr list |
| `a21925774` | `[BugFix][KVPool] Copy Mamba state after synchronous KV loads (#18901)` | only for hybrid models: a sync load left the recurrent state stale |
| `7e081f466` | `[BugFix][KVPool] Preserve physical blocks in bulk and layerwise cache transfers (#19070)` | block-id handling in bulk transfers |

`git tag --contains` is empty for all five, so none is in a release the reporter could move to.

### Workaround to propose (proposed, not reporter-confirmed)

`load_async: true` in the connector's `extra_config` so the receive thread owns the transfer instead of
the engine thread — documented, and the cheapest A/B; cost: it changes P-node scheduling behaviour
(ordering/overlap of the load) and does not remove the underlying transfer defect. Secondary: keep the pool
below ~90 % (shorter `--max-model-len`, one more pool node, or a lower `--max-num-seqs`) so a top-up no
longer runs at 96 % occupancy — cost: a different workload — and drop the features on the transfer path
(`enable_sparse_c8`, MTP) to see whether the death follows the cache layout. Say that none of these was
tested on the reporter's cluster.

### Falsifiers

- Does the failing request log `load_async=True`? Then this branch was not on the path.
- Does the death move or disappear with `load_async: true`? Then the inline load is the carrier.
- Is `need to load` unchanged for that req id while the worker is still alive? Then it is the §B stall, not
  a death.
- A plog/HCCL record naming a device error code on the failing card turns it into a device fault
  (`device-kernel-faults.md`), and the transfer path becomes a trigger, not the mechanism.

This is the third recorded crash whose owner is below Python in the pool — with #8808's missing PUT and
the repeating lookup of #6512 — and the rule is the same: do not look for a `vllm_ascend/` line that raises;
quote the inline call, the backend, and the fact that the returned-status path was already guarded.

## D. The store backend supports one protocol: `ascend` (`-DUSE_ASCEND_DIRECT=ON` is mandatory)

`AscendStoreConnector` (`kv_role: kv_both`, `backend: mooncake`) is a different reader from the P2P connector:
it constructs `MooncakeBackend`, whose `__init__` **hard-requires `protocol: "ascend"`** in the
`MOONCAKE_CONFIG_PATH` json. Verified, and the shape of the refusal changes by release:

| ref | the gate |
| --- | --- |
| `v0.12.0rc1`, `v0.13.0rc1`, `v0.14.0rc1`, `v0.18.0rc1` (each checked) | `if self.config.protocol == "ascend":` is the only branch that calls `self.store.setup(...)` and the only one that assigns `ret` (`distributed/kvpool/backend/mooncake_backend.py:32-44` at `v0.12.0rc1`; `…/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:35-49` from `v0.14.0rc1` — the path moved there), and `if ret != 0:` (`:45`/`:64`) is **unconditional** → any other protocol raises `UnboundLocalError: cannot access local variable 'ret' where it is not associated with a value`, i.e. a crash nobody diagnosed, not a fallback |
| `v0.19.1rc1`+ | an explicit `if self.config.protocol != "ascend": raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")` — #8514 (`f7ce74858`, 2026-04-22, ref issue #8482) moved the `ret` check into the branch and added the `else` raise; `v0.20.2rc1` carries it at `:28` with the lazy `_setup_store()` (#9774 `145e994a1`, backport `6a7d0ce1b`/#9771 first tagged `v0.21.0rc1`), `:388-389` at HEAD |

`MooncakeStoreConfig.from_file` defaults `protocol` to `"ascend"` (`:99`) and reads the whole struct from the
file `MOONCAKE_CONFIG_PATH` names (`:104-109`); `pool_worker.py:26-29` maps only `{"mooncake", "memcache"}`
as backends. The engine behind that branch is fixed to ADXL, not chosen by config:
`mooncake_transfer_engine.py` (`distributed/` at `v0.12.0rc1`, `distributed/kv_transfer/utils/` later)
initialises `TransferEngine().initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)` and its IPv6 refusal
names "mooncake's Ascend Direct Xfer Library". ADXL exists only in a Mooncake built with
`-DUSE_ASCEND_DIRECT=ON` — `docs/source/user_guide/feature_guide/kv_pool.md:24-28` at `v0.12.0rc1` ("Make sure
to build with -DUSE_ASCEND_DIRECT to enable ADXL engine"), the same flag in
`pd_disaggregation_mooncake_{single,multi}_node.md`, and Mooncake's own
`option(USE_ASCEND_DIRECT … OFF)` (`mooncake-common/common.cmake:108`), which gates the ADXL transport and
platform sources (`mooncake-transfer-engine/src/transport/CMakeLists.txt:47`,
`tent/src/platform/ascend/CMakeLists.txt:2`, `mooncake-store/src/CMakeLists.txt:647/739/807`).

**Answer shape — the question, not a crash.** (1) *Is it supported in release R*: locate the backend module at
that tag (`git ls-tree -r --name-only <ref> | grep -i "kvpool\|kv_pool"` — it moved at `v0.14.0rc1`), name the
registry import, then the docs page at the same ref; the maintainers' "supported since 0.11.0" (#5299,
`Yikun`) is `reported`. (2) *Can the flag be avoided*: no — `protocol` is `ascend` by default and by gate. The
thread's "without the flag it falls back to Host RDMA/TCP" (`MingYang119`, `LCAIZJ`) is true of **Mooncake's
transfer engine transports**, which a PD `mooncake_connector` deployment exercises; it does not describe a code
path in the pool backend, whose non-`ascend` branch does not exist. Adding one is a feature request, so ship it
as one rather than as a config hint. Levers: build Mooncake with `-DUSE_ASCEND_DIRECT=ON` and keep
`"protocol": "ascend"` (the documented path; cost is the HIXL/ADXL toolchain and the pinned CANN — that doc
says `CANN == 8.3.rc2`, torch/torch-npu `2.8.0`), or move to a ref that raises the clear `NotImplementedError`
(`>= v0.19.1rc1`) — that only improves the error, it adds no transports. Read the thread's last comment first:
#5299 was answered, labelled `resolved`, then stale-closed — a lever with no mechanism, which is enough here.

Falsifier / recheck: `git show <ref>:<mooncake_backend path> | grep -n 'protocol'` — if `__init__` calls
`setup` outside the branch at some ref, the `UnboundLocalError` does not apply there; and grep that ref's own
`docs/.../kv_pool.md` for `USE_ASCEND_DIRECT`, not the `latest` docs, which track `main`.

## E. Two store-backend halves that look like a connector bug: a lookup that loads nothing (#13356), an init that never returns 0 (#13418)

### E1. `kvpool hit tokens` is positive but every request loads nothing: the `need_to_allocate` clamp is upstream semantics (#13356)

Reported on `vLLM 0.23.0` + vllm-ascend `v0.23.0rc1` (CANN `9.0.1`, transformers `5.5.4`, Ascend910
A2-class, a `MultiConnector` with an `AscendStoreConnector` child, `backend: memcache`), as a *throughput
regression*, not a failure: memcache `lookup()` reports hits per request, the connector then loads **0
tokens**, and the run is slower than both the local-prefix-cache-only and the Mooncake-only baselines
because the lookup RPC is paid for nothing.

- The clamp, verified at `v0.23.0rc1` in
  `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`:
  `:517 num_external_hit_tokens = self.client.lookup(...)`, `:529-532 if num_external_hit_tokens <
  num_computed_tokens: need_to_allocate = 0 else: … - num_computed_tokens`, `:545-546
  force_layerwise_load = self.use_layerwise and num_external_hit_tokens > 0` / `if need_to_allocate <= 0
  and not force_layerwise_load: return 0, False`, `:565 return need_to_allocate, self.load_async and not
  self.use_layerwise`. So when the external hit is **already covered by the local prefix cache**
  (`num_external_hit_tokens <= num_computed_tokens`), nothing is loaded and no block is allocated — by
  design. Prefix-reuse workloads hit exactly that state.
- **It is upstream's own logic, which is why the maintainers call it an optimisation.** The same block,
  same log string, exists at upstream vLLM `v0.23.0` in
  `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:85-98` and `:108-109`, and in
  `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:1186-1204`. The
  fork's `pool_scheduler.py` is a copy; there is no Ascend-specific defect to fix, and "raise the hit
  ceiling" is the wrong ask.
- What the thread's lever actually buys: cheaper lookups, not fewer skipped loads. `Pz1116` closed it as
  「This is an optimization, not a bug, as currently upstream vLLM follows the same behavior. We have
  already optimized this a bit in #12783 and some follow-up prs, test with the latest commit of 23.0
  branch」 [reported]; the stale bot then closed it. Verified ancestry: `9e45da191`
  (`[v0.23.0][Performance] Optimize AscendStore key construction and miss path`, #12783) is in
  `releases/v0.23.0` and in tag **`v0.23.0`**, and **not** in `v0.23.0rc1` (`git merge-base --is-ancestor
  9e45da191 v0.23.0`); the follow-ups land on the same branch (`385491799` #12819, `88b10a44a` #13169,
  `2bcf78b37` #17667). Its body claims: shorter serialized keys, cached immutable prefixes, lazy grouped
  hashes, "propagates HBM hit tokens through lookup and avoids redundant external lookups and stores",
  async save by default — and it warns that a lookup immediately after a save can race save visibility.
  The clamp itself **survives** at `v0.23.0` (`:555-557`, `:571`), so do not promise the loads will start.
- Workaround ladder (`proposed` unless labelled): (1) move to `v0.23.0`/the `releases/v0.23.0` tip — the
  maintainer's own advice, cost = a release move plus the async-save race above; (2) drop the
  `AscendStoreConnector` child for a prefix-reuse workload, which removes the RPC and the zero-value work
  while keeping the P2P child (composition rules: `multi-connector-composition.md`); (3) `use_layerwise:
  true` forces `force_layerwise_load` and *does* load, but transfers per-layer data the pool may not hold —
  a trade, not a fix. Do not offer `retention_interval`-style knobs you cannot read at the reporter's tag.
- Falsifier: `num_external_hit_tokens > num_computed_tokens` in the connector's log and `need to load: 0`
  anyway — then the clamp is not the branch that ran, and the hit accounting (block granularity,
  `_block_size`, `discard_partial_chunks`) is the suspect family (`prefix-cache-granularity.md`).

### E2. The store backend will not initialise: bare `assert res == 0` in `_setup_store` (#13418)

Signature (`v0.23.0`; the same bare assert is at `:41`/`:46` at `v0.16.0rc1`, `:66` at `v0.20.2rc1` and
`:246` at `HEAD`, so the line number is only as good as the ref): the worker dies while the connector is
constructed — `worker.py initialize_from_config` → `ensure_kv_transfer_initialized` →
`KVConnectorFactory.create_connector` → `pool_worker.py _init_backend` → backend `__init__` →
`memcache_backend.py:58 self.store = self._setup_store()` → **`:95 assert res == 0`** with no message after
it (`v0.23.0`), after `:87 res = store.init(self.local_rank, init_bm=self._init_bm)` returned nonzero from
the vendor `memcache_hybrid` package (`:84 store = DistributedObjectStore()`). `:50 self._lazy_init =
lazy_init and _is_device_sdma()` decides only whether that call happens now or later; a report that hits
`:58` is on the eager branch. The two `except` branches above the assert (`:88-93`) are the informative
ones: a `ValueError` there is configuration loading, any other exception is dependency/setup, and a
*nonzero return* is neither, which is why this shape reaches the bare assert.

Why the assert is uninformative: `res` is the return of `MmcacheStore::Init`
(`Ascend/memcache` `src/memcache/csrc/mmcache_store.cpp:169`) → `mmc_init`
(`src/memcache/csrc/mmc.cpp:295`), whose non-zero returns are (read at `9434148b`, `describe`
`memcache_release_0.1.0-346`; an image's wheel can be older, so re-grep `mmc_init` in the version that
image ships before quoting a line):

| status | site (`mmc.cpp`) | cause |
| --- | --- | --- |
| `MMC_INVALID_PARAM` | `:300-301` | `deviceId > MAX_DEVICE_ID` (`383`, `src/memcache/csrc/config/mmc_config_const.h:155`) |
| `MMC_INVALID_PARAM` | `:310-311` | `MMC_LOCAL_CONFIG_PATH` unset (env read at `common/mmc_env.cpp:21`) |
| `MMC_INVALID_PARAM` | `LoadAndValidateConfig` `:238-263` | the config file fails validation (`Wrong configuration because of following mistakes:`) |
| `MMC_ERROR` | `:325-326` | `initBm` set and `mmcs_local_service_start` returned null (no HBM/bm service) |
| return of `mmcc_init` | `:332-338` | client/meta-service init failed (`mmcc init failed, ret:<n>`) |

So the readable error is in **memcache's** log, not vLLM's: `Failed to setup config`,
`Invalid local service config`, `failed to create or start local service`, `mmcc init failed, ret:`. The
vllm-ascend-side `ValueError("The environment variable 'MMC_LOCAL_CONFIG_PATH' is not set.")` (`:19-21`)
appears only when the lazy branch is taken, so its absence proves nothing.

Worked case: `vllm-ascend 0.23.0` (tag `5cb98caa`), HDK 26.0.0, CANN 8.5.1, GLM5.1 on A3-560T 大EP
`2×2P + 1×2D` (P: DP4 TP8, D: DP8 TP4) with **PP** and a MemCache pool; the engine otherwise serves, with
`Delaying free of … blocks` and `External prefix cache hit rate: 0.0%`.

- **The thread's lever is stale — check ancestry before relaying it.** The single comment
  (`maomaodedipan`, 2026-08-03, closed) says "Solve the problem by merging these PR
  [13349][12762][13028][12643]", but `#12762` (`df54ea61f`), `#13028` (`a1b574089`) and `#12643`
  (`4c30e4f7d`) are all **ancestors of `v0.23.0`** (`git merge-base --is-ancestor`), and `#13349`
  ("Add AscendStore PP save diagnostics") is a **closed, unmerged** PR whose base is `releases/v0.23.0`
  (`github /pulls/13349`, `merged: false`), with no commit in the clone. What those commits actually
  carry: finalize KV connectors on **non-last PP ranks** after the target forward under MTP/spec decode,
  ignore generic PP handshake metadata in AscendStore, and replace backend silence with typed errors.
- **What the PP framing does not establish:** at `v0.23.0` nothing in the PP path reaches this call except
  `local_rank` (`:48`), so "PP + memcache" is the report's configuration, not a verified cause — keep the
  mechanism `hypothesis` and ask for the memcache log line.
- **Probes, in order:** `git diff v0.23.0 main -- …/backend/memcache_backend.py` (the DP-init barrier was
  later removed — #19370, #20321 — and the docs then require `memcache-hybrid >= 1.3.0` and
  `memfabric-hybrid >= 1.3.0`); the same deployment with `pp=1` as the A/B that separates "PP + pool" from
  "pool"; and the vendor call's own log line, since `res` is `memcache_hybrid`'s return code and no clone
  shows its table.
- Levers to propose: supply/repair the config the env names and re-read the memcache log for the status
  line; `initBm: false` for the scheduler-side client is already what `create_scheduler_client` does
  (`:100-105`), so it is not a knob; or drop the memcache child from `connectors` and keep the PD connector
  (`multi-connector-composition.md`) so only the pool offload is lost. Falsifier: the memcache log shows
  `mmc_init` succeeding and the assert still firing — then the return code is coming from a different build
  of `memcache_hybrid` than the one the wheel pins.

## Cheapest localisation path for the next report

1. **Pick the half from one artifact.** Store-master metrics with `Keys: 0` and `PutStart=0/0` while the
   P/D log is busy ⇒ PUT half (section A): go straight to the runner's connector lifecycle at the ref.
   Hits in the master plus a repeating scheduler `hit tokens / need to load` line for one req id ⇒ LOAD
   half (section B). A load-spec line with `load_async=False` followed by `Worker proc … died` and no
   Python traceback ⇒ SYNC-RELOAD half (section C).
2. SYNC-RELOAD half: the load-spec line carries the mode — `load_async=False` means the transfer runs
   inline on the worker thread with no receive thread (`pool_worker.py` `_start_kv_transfer_threads` /
   `start_load_kv`, `:840` at `v0.23.0rc1`), so a rank that dies without a Python frame died below Python;
   `True` means the receive thread exists and the question is its completion bookkeeping, not this half. One
   `git log --oneline <ref>..HEAD -- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/` and read
   the subjects; do not read the diffs speculatively.
3. PUT half: `git grep -n 'finalize_kv_connector\|defer_finalize\|clear_connector_metadata' <ref> --
   vllm_ascend/worker`, and compare with the paired vLLM tag's `gpu_model_runner` /
   `kv_connector_model_runner_mixin.py`. A downstream fork of `execute_model` can lose any lifecycle
   step; `--speculative-config` and chunked prefill are the two branches that move them.
4. LOAD half: `git log --oneline <tagA>..<tagB> -- <pool package>` (one call), then `git show <c> --
   <pool_scheduler>` for the one commit whose hunk touches `get_num_new_matched_tokens`,
   `LoadSpec.can_load` or `_request_trackers`/`_preempted_req_ids`. Do not rank commits in prose. Read
   the loading half in `pool_worker.py` (save/load completion, `can_load`) at the same ref. If the
   complaint is *throughput* and `need to load` is `0` beside a positive hit count, stop: that is §E1, and
   one `git show <vllm-ref>:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`
   settles that it is upstream behaviour.
5. Ask for two numbers from the reporter: does `need to load` change for that req id, and does the run
   finish with `--kv-transfer-config` removed (that separates connector faults from the model/runner).
   For the PUT half the equivalent questions are whether the store master ever shows a non-zero `PutEnd`
   and whether the node runs `--speculative-config`.
6. INIT half (section E): read the deepest frame's **line**, not the exception type — a bare
   `assert res == 0` means a status code was thrown away, so the next artifact is the store library's log
   (`mmc_init`'s `Failed to setup config` / `failed to create or start local service` /
   `mmcc init failed, ret:`), not another vLLM read. One `git grep -n 'assert res == 0' <ref> -- vllm_ascend`
   locates it at the ref.

## Recheck conditions

Paths, log wording and the `LoadSpec` fields move with each release; re-grep the log format string
(`git grep -n -F 'kvpool hit tokens' <ref>`) before quoting a line number, and re-read the range's
commit list rather than reusing the candidate set above. The connector lifecycle also moves: later refs
put `finalize_kv_connector` in `vllm_ascend/worker/v1/` or `worker/v2/`, and `MultiConnector`
sub-connectors each have their own `wait_for_save`. Nothing here was reproduced on hardware.
