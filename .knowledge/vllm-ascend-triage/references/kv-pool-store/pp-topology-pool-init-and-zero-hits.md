# PP > 1 + KV pool: the init `assert res == 0`, and the 0 % external hit rate beside it

## Scope

A **PD deployment whose parallel layout includes PP** — the paste's worker prefixes
carry `PP<n>_TP<n>_EP<n>`, or the title is "PP + memcache 不兼容" / "PP 和 KV pool
一起开报错" — where the engine dies **before any request**:

- worker traceback ends
  `.../kv_pool/ascend_store/backend/memcache_backend.py", line <N>, in _setup_store`
  → `assert res == 0` → `AssertionError`, reached through
  `initialize_from_config` → `ensure_kv_transfer_initialized` →
  `KVConnectorFactory.create_connector` → `AscendMultiConnector.__init__` →
  `AscendStoreConnector` → `KVPoolWorker` → `MemcacheBackend.__init__`;
- and the same body often carries a second, **separate** symptom: `External prefix
  cache hit rate: 0.0%` beside a healthy `Prefix cache hit rate: ~9-10%`, a stream
  of `Delaying free of N blocks`, and occasional `Force freed expired request …
  exceeded timeout threshold (480 seconds)`.

Not this page: a pool fault on the **second** request (`HCCL watchdog`,
`ERR02005`, `507001`) or a `host_shm` device-id anomaly → `memcache-a3-dsv4-init.md`;
zero hits with a healthy write path on a hybrid model → `cross-engine-lookup-zero-hits.md`;
a PD pair that cannot connect at all → `../pd-disaggregation/index.md`.

## Pins

Verified here at vllm-ascend **v0.23.0rc1 `f4a08bdd`** (upstream vllm v0.23.0
`0fc695fc`), against the `Ascend/memcache` **v1.1.2** export. Re-grep every
`path:line` before quoting it at another pin: the lazy-init gate moved between
0.23 and 0.26 (Step 2), and the inference it licenses moves with it.

## Step 0 — the paste's own line numbers are the pin check (one call)

Five Python frames of that traceback land on exactly these statements at
`f4a08bdd`:

| Paste frame | In the pinned tree |
|---|---|
| `worker/worker.py:896`, `initialize_from_config` | `vllm_ascend/worker/worker.py:896` `ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)` |
| `ascend_store/ascend_store_connector.py:115` | `:115` `self.connector_worker = KVPoolWorker(` |
| `ascend_store/pool_worker.py:105` | `:105` `self._init_backend(parallel_config, extra_config)` |
| `backend/memcache_backend.py:58`, `__init__` | `:58` `self.store = self._setup_store()` |
| `backend/memcache_backend.py:95`, `_setup_store` | `:95` `assert res == 0` |

Five of five ⇒ the reporter's tree is that release's lineage (a "vllm-ascend
0.23.0" string is the release, not proof of a self-build), so the pin's code can be
quoted as *their* code. Say that in the answer; it is what makes the rest citable.

## Step 1 — the assert carries no cause; memcache already logged it

`res` is `mmc_init`'s raw `int32_t` (`Ascend/memcache` v1.1.2 `csrc/mmc.cpp:211-256`)
and `_setup_store` discards it, so the pasted `AssertionError` is
information-free. Four return classes, each printing its own line **above** the
assert:

| `mmc_init` return | Where | The line to ask for |
|---|---|---|
| `MMC_INVALID_PARAM` | `deviceId > MAX_DEVICE_ID` (`csrc/config/mmc_config_const.h:86`, = 383) | `Invalid param deviceId: <n>` |
| `MMC_INVALID_PARAM` | `LoadAndValidateConfig` (`csrc/mmc.cpp:160-180`) → `ValidateConf` / `ValidateLocalServiceConfig` (`mmc_configuration.h:365-430`: alignment, `dram.size` > `MAX_DRAM_SIZE`, `max.dram.size` < `dram.size`) | `Failed to setup config` (`:163`), then `Wrong configuration because of following mistakes:` (`:169`) with one line per mistake, or `Invalid local service config` (`:179`) |
| `MMC_ERROR` (`-1`) | `mmcs_local_service_start` returned null (`csrc/mmc.cpp:242`) | `failed to create or start local service` |
| the meta client's own code | `mmcc_init` failed (`:248-253`), then the local service is stopped again | `mmcc init failed, ret:<n>` |

So the one ask is **the worker's stdout block around the crash** (memcache logs at
ERROR by default; raise its level in the conf only if that block is empty), plus
`pip show memcache_hybrid memfabric_hybrid` and the **resolved** file behind
`MMC_LOCAL_CONFIG_PATH`. Do not walk the C++ first: `mmc_init` also returns
`MMC_OK` early for a second call in the same process (`csrc/mmc.cpp:219-222`
`if (mmcInit) { "mmc is already init"; return MMC_OK; }`), so per-rank differences
are real and only the log line says which class the failure is.

## Step 2 — is `:58` eager or lazy? A version question, and the answer moved

`MemcacheBackend.__init__` takes `lazy_init` and keeps it only for one protocol:
`memcache_backend.py:50` `self._lazy_init = lazy_init and _is_device_sdma()`
(`:18-31` reads `ock.mmc.local_service.protocol` out of the `MMC_LOCAL_CONFIG_PATH`
file, and raises `ValueError` when that env var is unset). The **caller's** gate is
what changed between releases:

| Pin | `KVPoolWorker._init_backend` | Consequence |
|---|---|---|
| v0.23.0rc1 `f4a08bdd` | `pool_worker.py:248-252` passes `lazy_init=True` for **memcache unconditionally**; the `use_compress` gate at `:256` belongs to the **mooncake** branch | a memcache backend is lazy **only** on `device_sdma`; on any other protocol the store is created eagerly at `:58`, inside engine init |
| v0.26.0rc1 `f2f74a16` | `pool_worker.py:316-319` `backend_kwargs["lazy_init"] = self.use_compress` (the gate covers memcache too) — recorded on `memcache-a3-dsv4-init.md`, not re-verified here | a non-`compress_ratios` model is eager whatever the protocol |

Two rules follow. At **0.23-lineage** a `:58` frame *does* mean the resolved conf's
protocol is not `device_sdma` (the `except ValueError` at `:88-90` did not fire, so
the env var is set and the file parsed) — one `cat` of that file confirms it before
anything else, and the doc's own guidance is `device_sdma` for A3 when HCCS is
available, `device_rdma` "supported for A2 and A3 … recommended for A2"
(`docs/source/user_guide/feature_guide/kv_pool.md:497`, sample conf `:485-487`). At
**0.26-lineage** the same frame says nothing about the protocol. Do not carry the
inference across the pin (`memcache-a3-dsv4-init.md` holds the 0.26 reading).

## Step 3 — where PP actually enters the init, and what it does not touch

Do this algebraically before any code walk; it is what makes "PP + pool
incompatible" a checkable claim instead of a label.

- **PP is read from the config, not from the process group**:
  `pool_worker.py:120-121` `pp_size = parallel_config.pipeline_parallel_size`,
  `pp_rank = (parallel_config.rank // tp_size) % pp_size` (same pair at
  `pool_scheduler.py:161-162`). So a PP>1 topology changes **which keys exist**
  (each stage writes only its own layers) — it does not change the connector class
  or the init sequence above.
- **The memcache device id is the node-local rank and PP does not renumber it**:
  `store.init(self.local_rank, init_bm=…)` uses `envs.LOCAL_RANK`
  (`pool_worker.py:111`) or, when the backend is constructed without one,
  `get_world_group().local_rank` (`memcache_backend.py:48`). Both are node-local;
  PP multiplies the *number of processes per node*, not their indices — so
  `Invalid param deviceId` is not a PP story unless the node also over-subscribes
  its devices (`scripts/card-budget.sh` settles that arithmetic).
- **The participant count is the PP-sensitive quantity**: every rank that reaches
  `_setup_store` calls `mmc_init(initBm=True)`, and the pool is created from
  `ock.mmc.local_service.dram.size` **per card** with
  `ock.mmc.local_service.world_size` as the participant envelope
  (`kv_pool.md:496-498`). Adding PP adds processes (and their claimed DRAM) to the
  same shared-memory group on the same node — the one arithmetic a PP-crossing
  config must be re-derived for, and the first thing to check with
  `scripts/card-budget.sh` when the `failed to create or start local service`
  class is the one in the log.
- **`prefill_pp_size` / `prefill_pp_layer_partition` are not this crash**: they are
  read only on the *consumer-puts* path (`pool_worker.py:196-214`, under
  `kv_role == "kv_consumer" and self.consumer_is_to_put`), documented for the MLA
  decode-stores-for-prefill case (`kv_pool.md:41-42,282-293`), and a mismatch raises
  a `ValueError` naming both counts — a different exception from the paste's.

## Step 4 — the 0 % external hit rate is the PP *key* axis, not the store

Read `pool_scheduler.py:212-232` once. The block key carries every axis the worker
stored under, including `@pp_rank:<n>` (`config_data.py:35,69`), and the lookup
enumerates them asymmetrically:

- non-layerwise (`include_layers=False`): `pp_ranks = range(self.pp_size)` — all
  stages must have stored the block for the whole key set to resolve;
- layerwise (`include_layers=True`, `use_layerwise`): `pp_ranks = [self.pp_rank]` —
  only this stage's slice is asked for.

So a PP topology whose stages do not all store (one stage's put fails, or the
decode/prefill PP layouts and layer partitions differ) yields **0 hits with a
healthy store**, and `keys_per_block_hash` omits `pp_size`
(`pool_scheduler.py:179-180`: `pcp*dcp*(tp//put_step)*num_layer_keys`) while
`_expected_worker_count` is the full `world_size` including PP (`:142`) — quote both
as the internal asymmetry, not as the cause. A hit rate of exactly 0.0 with
non-zero `Prefix cache hit rate` on the same line is this shape's fingerprint.

## Step 5 — coverage at the pin is part of the verdict

- No PP>1 case exists in the pool's own tests: `rg -n 'pipeline_parallel_size'
  tests/ut/distributed/ascend_store/` is empty, and the only e2e that builds an
  `AscendStoreConnector` (`tests/e2e/weekly/single_node/models/test_qwen3_30b_acc.py:95-124`)
  is TP-only (`--tensor-parallel-size`, no `--pipeline-parallel-size`).
- The project does refuse one PP combination at this ref:
  `platform.py:436-447` raises `ValueError: PP+MTP is only supported on
  PD-disaggregated P nodes (kv_role='kv_producer'). D nodes must use
  pipeline_parallel_size=1 …` — so on a PD pair PP belongs on the **prefill** side,
  and a decode node carrying PP without MTP is outside what the tree validates.
- Quote "PP with the pool is unvalidated here" as **coverage**, never as the cause.

## Answer ladder

1. **The release carrying the fixes, first.** A defect on this path that closed
   after the pin is invisible in the snapshot by construction. #13418's own public
   follow-up (2026-08-03) says the report was solved by the changes merged as
   #13349 / #12762 / #13028 / #12643 (written as `/issues/` links, which GitHub
   redirects for PR numbers) — so the rung-1 sentence is "re-run the same
   P/D commands on a build that carries them; `<tag>` predates the PP+pool fixes",
   with the ranked mechanism below it and the pin's coverage stated.
2. **The `mmc_init` log line** (Step 1) and the resolved `MMC_LOCAL_CONFIG_PATH`
   file: this splits config from code in one paste.
3. **`ock.mmc.local_service.protocol`** = `device_sdma` on A3 with HCCS — and at a
   0.23-lineage pin that also switches the store to lazy init, moving the failure
   out of engine bring-up (Step 2). One re-run, same command.
4. **The participant arithmetic** (Step 3): PP ranks per node × per-card
   `dram.size` against host DRAM and `world_size`.
5. **PP out of the picture for one run** — `--pipeline-parallel-size 1` with TP/DP
   absorbing the cards, on the *prefill* node only, pool unchanged; this is the
   ablation that decides whether the axis is PP at all, and it is cheaper than any
   source walk. Then one layerwise/non-layerwise comparison for the hit-rate half.
6. `pip show memcache_hybrid memfabric_hybrid` — but note that at this pin
   `requirements.txt:33-35` lists the pair **unpinned** (`memfabric_hybrid`,
   `memcache_hybrid` with no `==`), unlike the 0.26-lineage `==1.2.0` rows, so there
   is no target in the tree to compare against: the verdict is the installed pair
   plus the memcache master's own log, and the doc only says
   `pip install memcache-hybrid` (`kv_pool.md:466,504-513`).

## Not established / recheck

- Which of the four `mmc_init` classes the recorded report hit: the paste carries
  `AssertionError` only. The later resolution is a code merge, not a stated
  mechanism, so keep the cause open and label the config-side reading a hypothesis.
- Whether the fix set touched `pool_worker.py` / `memcache_backend.py` or the
  memcache/memfabric pin: inaccessible from this snapshot; say so.
- The PP participant arithmetic is documented per card and per `world_size`
  (`kv_pool.md:496-498`) but never verified against a run here.
