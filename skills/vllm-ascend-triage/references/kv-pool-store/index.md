# KV pool store: AscendStore + MemCache/Mooncake attached to serving

## Scope

A deployment with a **KV pool / three-tier cache attached** — `AscendStoreConnector`
(alone or in a `MultiConnector`), backend `memcache` or `mooncake`, the
`ock.mmc.*` config family, `MMC_LOCAL_CONFIG_PATH` — where the fault is in storing
or fetching KV, not in the model. Both shapes occur: **single node, no PD**
(`--data-parallel-size N --tensor-parallel-size M` on one host, one pool) and PD /
EPD. The pool's *own* config and versions decide before any vLLM code does.

Different from `../pd-disaggregation/index.md`: that topic starts from a P/D pair
that cannot connect; this one starts from "the pool is configured and the request
path breaks". Ask which one you have first — a pool fault on a single node has no
transport topology to inspect.

## Routes

- MemCache backend, DeepSeek-V4 (`compress_ratios`) on **800I/T A3** with
  `ock.mmc.local_service.protocol = device_sdma`, and the crash arrives on the
  **second** request (`HCCL watchdog thread terminated with exception`,
  `ERR02005 DIST internal error`, `EI0002`, `507001`) or at store init with a
  "device id" anomaly → `memcache-a3-dsv4-init.md`. Do not walk `MemcacheBackend`
  or the memcache C++ first: the lazy-init behaviour, the config keys and the
  failure family are already recorded there.
- `503900` / `E19999` / `HcclCommPrepare ret: 0x13` on the **first** transfer with
  a pool configured → the store segment size:
  `../pd-disaggregation/kv-pool-store-segments.md` and
  `scripts/kvpool-segment-check.sh`.
- The request **succeeds** but nothing is stored: prefill logs
  `Failed to invoke HcclBatchPut, hccl_result = 4` (or `HcclBatchGet`) and
  `Failed to put N keys out of N. error_codes=[-1]`, while the master shows
  `PutStart`/`PutRevoke` > 0 and `PutEnd` 0 with memory unused → the store's
  **put data plane**, not the segment size and not capacity:
  `hccl-onesided-store-put.md` (one call: `scripts/store-put-code-check.sh
  <paste>`). `4` = `HCCL_E_INTERNAL` and `-1` = the store's unclassified
  `INTERNAL_ERROR`; an out-of-space put returns `-200`.
- The pool **writes** fine and cross-engine keys are visible, but the read path returns
  nothing: `vllm:external_prefix_cache_hits_total` stays 0 while `…_queries_total` grows,
  `batch_is_exist` answers 1 for every queried key, and `find_longest_cache_hit` /
  `get_num_new_matched_tokens` returns 0 on a hybrid model (DSA/DeepSeek-V4, GLM, GDN,
  2+ KV cache groups) → `cross-engine-lookup-zero-hits.md`. Do not re-check the store's
  put/get first: hash granularity (`gcd`/`min`), group-granularity chunk keys and the
  dense-chain + `cache_transfer_granularity` rounding are what produce the 0.
- Nothing fails: the pool is attached, requests are served, and the **cold
  (cache-miss) requests are several times slower** than the same deployment without
  the pool while hits are faster ("初始 TTFT 慢 6-7 倍，命中后很快"), or a benchmark's
  first row is 100 % cold against warm rows → `pool-attached-cold-request-cost.md`.
  The miss path does no load work at all; what only a miss pays is the full-KV put
  into the pool (plus, on A2, the store init/`all_gather`) — and the pinned sample
  `mmc-local.conf` sets an **A3** protocol (`device_sdma`) while the table recommends
  `device_rdma` for A2. Split one-time init from steady state by timing the same
  prompt twice before reading the connector.
- PD on a hybrid linear-attention model and the decode node emits corrupt output →
  `../pd-disaggregation/hybrid-state-transfer.md`.
- The connector module itself will not import (`ascend_transport.so`, `No module
  named mooncake`) → `../image-dependency-faults/launch-env-and-library-path.md`.

## Rules

Inherit the rules in SKILL.md; these add the pool-specific ones.

1. **The store's config file is evidence; the launcher's JSON is not.** vllm-ascend
   reads the memcache protocol straight out of the file named by
   `MMC_LOCAL_CONFIG_PATH` (`memcache_backend.py:16-29` at v0.26.0rc1) — a JSON
   block in the reporter's launcher script is only the *intended* content. Ask for
   the resolved `mmc-local.conf` (or `pip show memcache_hybrid` →
   `{Location}/memcache_hybrid/config/`) before reading the key values as fact.
2. **A key that exists nowhere is a finding.** Config keys in these reports are
   frequently invented by a launcher wrapper. Grep the pinned memcache/memfabric
   export and the vllm-ascend docs for the literal key before explaining what it
   does; an unknown key is either ignored (silent no-op) or a validation error,
   and that difference is the answer.
3. **Name the owning component: MemCache/MemFabric or Mooncake's transport, not
   vllm-ascend.** A CQE error, an SDMA/`BatchCopyData` failure, an `mmc_init`
   return code and a device-id conversion message come from `Ascend/memcache` +
   `Ascend/memfabric_hybrid`; a `Transfer failed for key …`, a `HcclBatchPut` /
   `HcclBatchGet` / register-mem line and the master's `PutRevoke` come from the
   Mooncake store client plus its Ascend transport build. The vllm-ascend side owns
   *when* the store is initialised, which keys/addresses it puts, which device id
   it passes, and whether a failure is swallowed. Say which side you are naming.
4. **A pool fault surfaces as a collective fault.** The save path runs on a
   per-rank thread and its result is not propagated, so the first visible error is
   often HCCL (`ERR02005`, `EI0002`, `507001`, `STUCK`) several seconds later. The
   in-window first error in the MemCache/MemFabric log is the one to classify.
5. **Versions first.** The tree pins `memcache_hybrid` / `memfabric_hybrid`
   (`requirements.txt:33-35`, values differ per release), a snapshot export of
   `Ascend/memcache` may be a *different* release than the pin, and layerwise
   transfer has a hard floor (`>=1.2.0`, #12390). Ask `pip show memcache_hybrid
   memfabric_hybrid` in the reporting environment; a version-only verdict is
   already actionable.

## Pages

- `memcache-a3-dsv4-init.md` — A3 + DeepSeek-V4 + `device_sdma`: the deliberate
  lazy init (and why it is not the bug), the two inits a reporter sees, the
  `dram.size` / separated-deployment key gates, the version pins, and the
  put-failure → collective-desync family with its ablation ladder.
- `cross-engine-lookup-zero-hits.md` — write path healthy, read path 0: the four derived
  facts (hash_block_size = gcd vs the worker's `min*cp_scale`, chunk keys at group
  granularity, the dense chain from chunk 0 with the `min` over groups and rounding to
  `cache_transfer_granularity`, and store/lookup masks that differ), then the three checks
  that separate them.
- `pool-attached-cold-request-cost.md` — the deployment serves and only the *cold*
  requests are slow: the miss/hit cost asymmetry at v0.18.0rc1 (`load_async` default
  `False` makes the *hit* path blocking; the save runs off-step but syncs an NPU
  event per layer), the A2 `all_gather` + `register_buffer` branch, the
  `device_sdma`-in-the-sample vs `device_rdma`-for-A2 doc trap with
  `HCCL_INTRA_ROCE_ENABLE=1`, and the five checks in order (same-prompt-twice first).
- `hccl-onesided-store-put.md` — the request succeeds but the pool stays empty:
  `HcclBatchPut` `hccl_result = 4` (generic `HCCL_E_INTERNAL`) with
  `error_codes=[-1]` (the store's generic code, not `-200` out-of-space), the
  master `PutStart`/`PutRevoke`/`PutEnd` reading that proves the control plane
  worked, which build flags pick the `ascend` transport, why the emitter is not in
  any pinned export, and the two artefacts that decide the rest.
