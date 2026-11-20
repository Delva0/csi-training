# MemCache on A3 with DeepSeek-V4: lazy init, config keys, and the crash family

Pins: vllm-ascend `v0.26.0rc1` (`f2f74a16`) with upstream vllm `v0.26.0`;
memcache/memfabric exports in that case's `SOURCES.json` were `v1.1.2` while the
tree pins `1.2.0` — say which one a claim comes from. Re-grep the `path:line`
below at other refs (the lazy-init block is recent, #9771/#9975 lineage).

## The report shape this answers

800I/T A3 single node, `DeepSeek-V4-Flash-*` (w8a8) with `--data-parallel-size 4
--tensor-parallel-size 4`, `AscendStoreConnector` backend `memcache`, memcache
`ock.mmc.local_service.protocol = device_sdma`; the service starts, the **first**
curl succeeds, the **second** crashes the worker with
`HCCL watchdog thread terminated with exception` / `ERR02005 DIST internal error`
(the report may also say the reporter suspects "vllm-ascend 懒加载"). The same
report often carries a second case with `host_shm` that fails **at memcache init**
with a "device id 转换异常".

## What vllm-ascend actually does (source-verified)

- `lazy_init` is enabled only for compress models, i.e. models whose HF config
  carries `compress_ratios` (DeepSeek-V4): `pool_worker.py:114`
  `self.use_compress = self.compress_ratios is not None`, then
  `pool_worker.py:316-319` `backend_kwargs["lazy_init"] = self.use_compress` with
  the comment naming Mooncake `ASCEND_ENABLE_FABRIC_MEM=1` and Memcache
  `device_sdma`.
- The backend re-gates on the protocol read out of the file named by
  `MMC_LOCAL_CONFIG_PATH`: `memcache_backend.py:16-29` `_is_device_sdma()` (raises
  `ValueError` if the env var is unset), `:51`
  `self._lazy_init = lazy_init and _is_device_sdma()`. So "lazy init happened" is
  not a defect signal by itself; it is the documented behaviour for exactly this
  model+hardware+protocol combination.
- With lazy init the store is **not** created in `__init__` (`:58-60`); the first
  real forward step triggers it from `ascend_store_connector.py:269-277`
  (`get_finished` → `ensure_store_initialized()` under
  `if self._current_step_has_real_forward`), and the save path can trigger it
  earlier from the KV-send thread (`memcache_backend.py:133,143,175,211`).
- Documented as intentional: #9975 item 2 — "if you are running on 800I/T A3
  series and using Mooncake FabricMem Mode or Memcache with `device_sdma`, DSV4
  now requires a lazy init behavior … This will be reverted once the known issue
  is solved" (#9771). Quote that when a reporter proposes "fix the lazy loading".
- **Two initialisations in one log are expected** and are not proof of a double
  init in one process: the scheduler process creates a metadata-only client
  (`memcache_backend.py:101-108`, `local_rank=0, init_bm=False`, comment says it
  must not initialise storage), the worker initialises with its own rank. In
  `Ascend/memcache` `mmc_init` is process-global and first-wins
  (`csrc/mmc.cpp:211-250`, `if (mmcInit) { "mmc is already init"; return MMC_OK; }`,
  and `g_localServiceConfig.deviceId` is only assigned after that guard) — so a
  second init *inside one PID* would bind device 0 and never start the local
  service. Decide that by PID, not by log order; the scheduler comment at
  `:102-104` is the reason the pattern exists.

## Config gates to diff before any mechanism

Docs at the same pin: `docs/source/user_guide/feature_guide/kv_pool.md`.

| Gate | What the doc/rule says | Why it matters here |
|---|---|---|
| `ock.mmc.local_service.dram.size` | per **die**, not per node ("on A3, to allocate 640GB as KV pool … 640/16 = 40GB"); `kv_pool.md:511` also says **set 0GB for A3 when HCCS is available** | a per-rank `20GB` on a 16-rank A3 node claims ~320GB of host DRAM; the value is also what the pool advertises, so a mismatch is a real config fault, not a parser quirk |
| `ock.mmc.local_service.max.dram.size` | must be ≥ every process's `dram.size` (`kv_pool.md:512`) | `max < size` is rejected at `ValidateLocalServiceConfig` |
| size alignment | `device_rdma`/`device_sdma` align to **1GB**, host protocols to 2MB (`mmc_configuration.h:371-380`); the memcache header (`include/mmc.h`) notes every protocol must be 1GB-aligned on A3 or `mmc init` fails | a "20GB" style value is fine, "20.5GB" is not |
| separated (standalone) MemCache | `kv_pool.md:761-790`: **two** conf files, both using `ock.mmc.local_service.dram.size` — the vLLM-side file sets `0GB`, the standalone process contributes the DRAM, `max.dram.size` covers both | `ock.mmc.local_service.standalone.dram.size` **is not a key anywhere** (absent from the memcache v1.1.2 export, memfabric, and the vllm-ascend docs) — a launcher that emits it silently contributes nothing |
| `world_size` | `ock.mmc.local_service.world_size` ≥ the number of local services | too small is an init-time rejection |

Read the *resolved* file, not the launcher JSON, and state that you did.

## Versions

- `requirements.txt:33-35` at v0.26.0rc1: `memfabric_hybrid==1.2.0`,
  `memcache_hybrid==1.2.0`.
- Layerwise pool transfer requires `>=1.2.0` for both (#12390 item 1), and SSD
  cache requires `memcache_hybrid >= 1.2.0` (`kv_pool.md`).
- The snapshot export of `Ascend/memcache` may be `v1.1.2` (as in the recorded
  case) — that is the *source available to you*, not necessarily the reporter's
  install. Ask `pip show memcache_hybrid memfabric_hybrid` before attributing
  behaviour to a release.

## The crash family: a swallowed save failure becomes a collective fault

Precedent #14106 (`Non-layerwise MemCache save failure on one TP rank can
desynchronize collectives and kill all Prefill engines`, vllm-ascend
0.19.1rc2.dev, MemCache/MemFabric 1.1.3, `device_sdma`): the first low-level
trigger was a MemFabric `device_sdma` CQE status 9
(`sqeType:11 cqeErrorCode:9`, `BatchCopyData failed, ret:-1`, `MemCache put
failed, error_code=-1`), then ~3 min later `HCCL Notify Wait timeout`, `EI0002` /
`507001`, worker `SIGABRT`, `EngineDeadError`, all Prefill APIs down. A one-shot
220 s delay injected before `batch_put_from_layers()` on **one** TP rank
reproduced the same collapse with no CQE error at all, and an immediate `-1`
return was logged while the request still answered `200` — the save result is
swallowed (`request_queue.join()` waits for `task_done()`, not for success; the
issue quotes `memcache_backend.py` logging `failed_count` and returning nothing at
its pin).

Consequence for a "second curl crashes the service" report: the HCCL watchdog line
is **downstream**. Ask for the first MemCache/MemFabric error in the same window
(CQE status, `BatchCopyData`, `error_code`, `FreeBlobs`) and the PLOG block; the
identity of `ERR02005` with #14106's `507001` is *not* established — treat it as
the same family (rank-local save failure → collective desync) and label it
hypothesis.

## Answer ladder for the recorded #16111 shape

1. Version row: `pip show memcache_hybrid memfabric_hybrid` vs
   `requirements.txt:34-35`; an install below `1.2.0` is a finding on its own.
2. Config diff of the resolved `mmc-local.conf` against the `kv_pool.md` tables
   above (per-die `dram.size`, `max.dram.size`, 1GB alignment, unknown keys). The
   recorded report's JSON carried `standalone.dram.size` and `hugepage_mem_gb`,
   neither of which exists in memcache — cheap, decisive, reporter-runnable.
3. Say that lazy init is documented/intentional (#9975 item 2) and is *not* the
   fix target; the revert is upstream's, gated on the known issue.
4. Save-failure check: repeat one request and grep the worker + MemCache logs for
   the first backend error before the HCCL line; if there is none, run the #14106
   discriminator (fail/slow one rank's save) only on an isolated pair.
5. For the `host_shm` init case, the failure is at `mmc_init` before any KV moves:
   ask for the local-service startup lines naming the device id **and the PID**
   (scheduler client passes `0` by design), the resolved protocol line, and the
   `mmc_init` return/log line; the "device id conversion" text itself lives in
   MemCache/MemFabric, so name that owner and ask for their log rather than
   theorising from vllm-ascend.
6. Note the reporter's own later comment if one exists (the recorded #16111
   follow-up added a log link for the device-id case): the newest evidence decides
   which of the two sub-cases is live, and comments are only as good as the file
   they point at — if the artefact is not readable, say so and ask for the lines.

## Not established / recheck

- Which process the "device id 转换异常" belongs to (scheduler client vs worker) —
  the recorded evidence was a log link, not a log.
- Whether `ERR02005` and `507001` share one trigger; only the family is supported.
- The `host_shm` variant's requirements (DRAM-only, no hcom) are in the memcache
  docs, not verified against a run here.
