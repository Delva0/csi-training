# Pool attached, serving fine, but the cold requests are several× slower

## When this page applies

A KV pool (`AscendStoreConnector`, memcache or mooncake backend) is attached to a
deployment that **works**, and the claim is about speed: "初始 TTFT 比不带
memcache 的版本慢 6-7 倍，但后续命中缓存的请求快很多" — recorded #11923
(vllm-ascend 0.18.0rc1 + the reporter's own #7874/#10098 patches, 910B/A2, GLM-5.1
DSA/MLA, 2P1D, 4+4 servers, `memcache_hybrid`/`memfabric_hybrid` 1.1.1, numbers in an
attached metrics image). The same reading applies in reverse: a benchmark whose
first row is 100 % cold and later rows are cache hits is not an A/B of the store.

Not this page: the transfer **fails** (`hccl-onesided-store-put.md`,
`../pd-disaggregation/kv-pool-store-segments.md`), the read path returns 0 hits
(`cross-engine-lookup-zero-hits.md`), or the engine dies at init
(`memcache-a3-dsv4-init.md`).

## What a cold request pays that a hit does not (v0.18.0rc1 `99e1ea0`)

| Site | Fact |
|---|---|
| `pool_scheduler.py:56-111` | `get_num_new_matched_tokens` returns `(0, False)` on a miss **before** any store call, so a miss adds no *load* work to the step |
| `pool_scheduler.py:355-363` + `backend/memcache_backend.py:67` | the scheduler-side lookup is a blocking ZMQ `REQ`/`REP` round trip **per new request**, answered by the worker's `m_store.exists` → `batch_is_exist`; both paths pay it, and it scales with request count |
| `pool_scheduler.py:33,111` + `pool_worker.py:294` | `load_async` defaults **False**, so a *hit*'s `m_store.get` runs synchronously inside the worker step unless layerwise transfer is on — the hit path is the blocking one, the miss path is the computing one |
| `pool_worker.py:329-345` + `kv_transfer.py:238,358-359` | `wait_for_save` only enqueues, but each chunk/layer records an NPU event and calls `current_event.synchronize()` before `m_store.put` — a cold request copies **its whole prefilled KV** into the pool, which is the one large piece of work only a miss does |
| `backend/memcache_backend.py:71,80` | `get` = `batch_get_into_layers(COPY_G2L)`, `put` = `batch_put_from_layers(COPY_L2G)`: the copy's cost belongs to the store's data plane, not to vLLM |
| `backend/memcache_backend.py:30-45,58-63` | on **A2** a device `all_gather` runs before `store.init` and `register_buffer` registers every KV buffer (A3 does neither) — a 910B-specific startup/first-use cost |

So the sentence to give the reporter is a decomposition, not a cause: cold TTFT ≈
prefill + per-request lookup RPC + the cold request's share of the full-KV put
competing with compute, while a hit pays load + decode start and skips the prefill
entirely. Which term dominates is decided by check 1 below, from the report alone.

## The A2 config trap (source-verified in the doc, not in the reporter's file)

| `docs/source/user_guide/feature_guide/kv_pool.md:530` | the pinned sample `mmc-local.conf` sets `ock.mmc.local_service.protocol = device_sdma` |
| `kv_pool.md:566` | the parameter table: `host_rdma` is the **default**, `device_rdma` is "recommended for **A2**", `device_sdma` is "supported for **A3** when HCCS available, recommended for A3" |
| `kv_pool.md:100` | 800I/T A2 needs `export HCCL_INTRA_ROCE_ENABLE=1` for the direct-transmission scheme |
| `kv_pool.md:534-535,567` | memcache sizes are `ock.mmc.local_service.dram.size` / `hbm.size`, per card, aligned downwards to 2MB (`host_*`) or 1GB (`device_*`); Mooncake instead takes `global_segment_size`, 1GB-aligned (`:119-127`) |
| `kv_pool.md:620-623` (A2 example) | the A2 example only exports `MMC_LOCAL_CONFIG_PATH=…/mmc-local.conf` (and `nr_hugepages`), so **the protocol in force is whatever that file says** — ask for the resolved file (index rule 1) |

On 910B a reporter who copied the sample config (or an A3 recipe) is therefore
either on an A3-recommended protocol or on the host-staged default, and the A2
recommendation exists only in a table row. That is a one-line item for the first
answer, not a source walk.

## Checks, cheapest first

1. **Send the same prompt twice and time request #1 vs #2.** Only #1 slow →
   one-time init (store init, the A2 `all_gather`, buffer registration, QP setup).
   Every cold miss slow → steady-state store cost. This single check splits the two
   families the report cannot distinguish.
2. **TTFT vs concurrency, all rows cold.** Flat → the full-KV put. Growing with N →
   the blocking per-request lookup RPC and prefill queueing.
3. **Ask for the resolved `mmc-local.conf`, `pip show memcache_hybrid
   memfabric_hybrid`, and both serve commands.** Protocol per the table above;
   wheels against the release's pin (index rule 5 — the snapshot exports 1.1.2 while
   #11923 runs 1.1.1, and layerwise transfer has a `>=1.2.0` floor per #12390, so do
   not pair `use_layerwise` with 1.1.x); the two serve commands should differ only
   by the connector block.
4. **Compare cold row to cold row** at the same topology, block size,
   `--max-num-batched-tokens` and graph config. A PD-versus-non-PD baseline, or a
   cold row against a warm row, inflates the ratio structurally (`../pd-disaggregation/index.md`
   precedents #3062, #7182 are the same trap).
5. Only then the store-side levers: `protocol = device_rdma` on A2 +
   `HCCL_INTRA_ROCE_ENABLE=1`, aligned segment/`dram.size` values, and an
   `load_async: true` / `use_layerwise` toggle to see which side the cost sits on.

## Recorded status

- Code and doc anchors: source-verified at vllm-ascend v0.18.0rc1 (`99e1ea0`).
- **Not a verdict.** #11923 pasted no log, the metrics image was not readable in
  that session, and the reporter's own #7874/#10098 patches are not in the tree — so
  this page is a check ordering with a named anchor per rung, and the answer owes the
  reporter the decomposition plus check 1, not a selected cause.
