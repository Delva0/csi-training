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

- **A PD deployment whose layout includes PP** ("PP + memcache 不兼容",
  `PP<n>_TP<n>_EP<n>` worker prefixes) dies **before any request** with
  `AssertionError` at `memcache_backend.py` `_setup_store` / `assert res == 0`, and
  the same body shows `External prefix cache hit rate: 0.0%` beside a healthy
  local `Prefix cache hit rate` → `pp-topology-pool-init-and-zero-hits.md`. The
  assert carries no code (ask for the `mmc_init` log line above it), the
  eager-vs-lazy reading of the `:58` frame is version-specific, and the hit-rate
  half is the `@pp_rank` key axis. A fix merged after the pin is invisible in the
  snapshot, so the release rung comes first.
- `503900` / `E19999` / `HcclCommPrepare ret: 0x13` on the **first** transfer with
  a pool configured → the store segment size:
  `../pd-disaggregation/kv-pool-store-segments.md` and
  `scripts/kvpool-segment-check.sh`.
- The request **succeeds**, nothing is stored, and there is **no error text at all** —
  the master shows `PutStart` **0** and `PutEnd` 0 — on a launch that carries
  `--speculative-config`: the pool was never handed a save. That is not the store and not
  the segment size: `no-put-under-spec-decode.md` (spec decoding makes the runner defer
  connector finalization, and the defer half of that port is what runs). The row below is
  the *other* zero-store shape — it carries `HcclBatchPut … hccl_result = 4`.
- The pool serves, then **one** instance's `vllm:kv_cache_usage_perc` climbs to ~0.999 and
  never recovers until restart, with no traceback, no device code and lookups still answering:
  the connector took the `delay_free` path and its `finished_sending` never arrives. At the pin
  the send side reports through the **`stored_requests` counter**, so `set_finished_request`
  (the hunk such reports usually propose, copied from the layer thread) is inert; the reachable
  holes are a layerwise save and a preempted request, both of which leave the id with no counter
  record at all → `ascendstore-held-blocks-never-finished.md`. The mirror symptom (a
  `finished_sending` id that arrives late and trips `assert req_id in self.requests`) is
  `../pd-disaggregation/prefill-scheduler-kv-xfer-assert.md`.
- The request **succeeds** but nothing is stored: prefill logs
  `Failed to invoke HcclBatchPut, hccl_result = 4` (or `HcclBatchGet`) and
  `Failed to put N keys out of N. error_codes=[-1]`, while the master shows
  `PutStart`/`PutRevoke` > 0 and `PutEnd` 0 with memory unused → the store's
  **put data plane**, not the segment size and not capacity:
  `hccl-onesided-store-put.md` (one call: `scripts/store-put-code-check.sh
  <paste>`). `4` = `HCCL_E_INTERNAL` and `-1` = the store's unclassified
  `INTERNAL_ERROR`; an out-of-space put returns `-200`.
- The pool **works for minutes and then one peer times out**: `Transfer timeout to:
  <ip>:2009x, you can increase the timeout duration … ASCEND_TRANSFER_TIMEOUT` →
  `transfer failed and disconnect to:` → `TRANSFER_FAIL (-800)` on the keys →
  `Failed to complete transfers after 60 seconds` → `revoke … OBJECT_NOT_FOUND` →
  `Target adxl engine: … is not connected.` for peers that never failed → optional
  `!!! Segfault encountered !!!` in `AscendDirectTransport::processSliceList` →
  `std::_Hashtable<unsigned long, …>::erase`, on an **aggregated** `kv_role: kv_both`
  pool with no P/D split → `store-put-transfer-timeout-and-adxl-segfault.md`. The
  `503900` that appears *after* the first timeout is a downstream slice status, not the
  startup channel-prepare code, and the `ASCEND_TRANSFER_TIMEOUT=10000` the project's
  own recipe exports equals the code default, so it changes nothing.
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
- The serve answers, then the **APIServer** raises `ValueError: Counters can only be
  incremented by non-negative amounts` from `metrics/loggers.py` `.inc(…)` while the pool
  is attached and the prompts are long → not the store and not the request path: upstream's
  `PromptTokenStats` subtracts the external KV count from a `num_cached_tokens` latched at
  the request's **first** schedule pass, so a request preempted mid-prefill and resumed with
  a bigger pool hit goes negative in `local_cache_hit`. One metric read
  (`vllm:num_preemptions`) and one `--disable-log-stats` launch confirm it →
  `prompt-token-counter-negative.md`. Do not re-check Mooncake's put/get.
- The complaint is a **throughput / TTFT regression** while the log shows pool hits
  and `need to load: 0` (or `external_prefix_cache_hits_total` pinned at 0 while
  queries grow): the zero load is the upstream contract and the cost is the
  per-request blocking lookup RPC → `pool-lookup-overlap-zero-load.md`. Read it
  **before** explaining the subtraction as a bug; the genuinely-zero-hit case is
  `cross-engine-lookup-zero-hits.md`.
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
- `pp-topology-pool-init-and-zero-hits.md` — PP>1 + AscendStore/memcache: the
  five-frame pin match that proves the reporter's tree, the four `mmc_init` return
  classes behind a bare `assert res == 0`, the `lazy_init` gate difference between
  the 0.23 and 0.26 lineages (and the inference each licenses), what PP does and
  does not change (participant arithmetic vs device id vs `prefill_pp_size`), the
  `@pp_rank` key asymmetry behind a 0.0 % external hit rate, the pin's own
  PP-coverage gap and the `PP+MTP` refusal, and the release-first ladder.
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
- `pool-lookup-overlap-zero-load.md` — hits ⊆ local prefix cache and every request
  loads 0 tokens: the docstring + upstream `scheduler.py:617-636` contract, the
  metric pair that follows from it, the blocking `LookupKeyClient` REQ/REP over all
  `block_hashes` (and the `query_start_block` skip only the store-scheduler path
  has), and the downstream-only `force_layerwise_load` branch to A/B.
- `hccl-onesided-store-put.md` — the request succeeds but the pool stays empty:
  `HcclBatchPut` `hccl_result = 4` (generic `HCCL_E_INTERNAL`) with
  `error_codes=[-1]` (the store's generic code, not `-200` out-of-space), the
  master `PutStart`/`PutRevoke`/`PutEnd` reading that proves the control plane
  worked, which build flags pick the `ascend` transport, why the emitter is not in
  any pinned export, and the two artefacts that decide the rest. It also decodes the
  master's `SSD Storage: X / Y` line: a master started with `--root_fs_dir` prints a
  non-zero denominator there while every client has offload off, so that line is not
  the offload flag and does not route the report to the offload/size families.
- `no-put-under-spec-decode.md` — the pool stays empty with **no error anywhere** and spec
  decoding on (recorded #8002, v0.18.0rc1 `99e1ea0` / vllm v0.18.0 `bcf2be9`): upstream splits
  the connector lifecycle into `defer_*` + `finalize_kv_connector()` (`gpu_model_runner.py:3768`,
  `:4047`, mixin `:81-89`), the downstream copy keeps only the defer (`model_runner_v1.py:1367`,
  `:1580-1581`) and calls no `finalize_kv_connector` anywhere, so `wait_for_save()` — the only
  enqueue point for `pool_worker.py:329-352` — never runs. Carries the two-grep check, the
  one-line repair, the put/hit acceptance test, the layerwise and `kv_role` exceptions, and the
  general reading of a half-ported defer/finalize contract (absence is the evidence).
- `ascendstore-held-blocks-never-finished.md` — the pool serves, then one instance's KV usage
  climbs to 0.999 and never recovers (recorded #10098 at v0.20.2rc1 `367b8e6` / vllm v0.20.2
  `bc150f5`): the full anchor table for the send-side completion contract (`finished_requests`
  set vs `stored_requests` counter, and which one the worker reads), the two holes that leave a
  delayed-free id with no counter record (layerwise `save_kv_layer`, preemption), the
  `request_queue.join()` hang an exception in the thread turns into, why the report's
  `set_finished_request` hunk is inert here, the repair that covers all three, the two greps
  that decide the reporter's own build, and the "full pool hit does not leak on the
  non-layerwise path" refutation.
- `store-put-transfer-timeout-and-adxl-segfault.md` — puts land, then one ADXL peer
  times out every 10 s, `-800` on the keys, 60 s batch timeout, `OBJECT_NOT_FOUND` on
  the revoke and a crash in `processSliceList`: which lines are downstream (all of
  them), the recorded `mooncake.json` and why the K8s Service port remap and the
  `20000 + phy_id*100` endpoint band are not findings, the doc's 3000 ms default vs the
  header's 10000 ms (so the recipe's own value is a no-op), the crash frame's build
  divergence, and the six checks in order — starting with making the failure repeatable,
  because the recorded case closed with no confirmed cause.
- `prompt-token-counter-negative.md` — the APIServer `Counters can only be incremented …`
  `ValueError` on a pool-attached serve: the `cached + recomputed - external` identity at
  vllm v0.18.0 (`stats.py:280-286`), why the reporter's `cached + external > prompt_len`
  cannot be it, the `num_cached_tokens` latch (`request.py:150`, `scheduler.py:805-807`) that
  `_preempt_request` (`:941`) never re-arms while the connector is re-queried at
  `num_computed_tokens == 0`, the connector's own `load_async=False` /
  `discard_partial_chunks=True` defaults that prune the async branches, and the two checks
  (`vllm:num_preemptions`, `--disable-log-stats`).
