# KV pool (`AscendStoreConnector`) store segments: `global_segment_size` ceilings and `503900` channel failures

Worked case: #13439 — 2 nodes × 8×910B2, `npu-smi` 25.2.3, CANN 9.0.1
(V100R001C10SPC002B220), torch_npu 2.10.0.post2, vLLM 0.23.0 /
vllm-ascend 0.23.0rc1, 1P1D across two nodes over `launch_online_dp.py`
(dp8/tp1) + `load_balance_proxy_server_example.py`; `HCCL_INTRA_ROCE_ENABLE=1`
exported on both sides and `Roce mode is enabled` in the transport log; P runs
`--enforce-eager`, D adds `enable_npugraph_ex` + `recompute_scheduler_enable`.
The kv-transfer config is a `MultiConnector` of `MooncakeHybridConnector`
(`kv_port` 20001/20002, `kv_buffer_size` 16 GiB, `kv_block_size` 64 MiB) plus
`AscendStoreConnector` (`backend: mooncake`), with
`MOONCAKE_CONFIG_PATH=/mnt/fzx/mc_config/mooncake.json` on both sides. Symptom,
on every request:

```text
E ... transfer_executor_base.cpp:263] Failed to connect to target: <P>:20064, status: 503900, errmsg: E19999
E19999 ... Call llm::HcclAdapter::GetInstance().HcclCommPrepare(...) fail, ret: 0x13 [FILE:channel.cc][LINE:81]
   Call channel->Initialize(enable_use_fabric_mem) fail ... [channel_manager.cc][LINE:416]
   Call DoConnect(remote_engine, timeout_in_millis) fail ... local engine:<D>:20006, remote engine:<P>:20064, timeout:120000 ms
E ... [mooncake_hybrid_connector.py:704] Mooncake transfer failed for request ... ret=-1
RuntimeError: Mooncake transfer failed, ret: -1
```

and the D node then answers with garbled output. Reporter-confirmed resolution
(2026-08-10): **`global_segment_size` was 40 GB in the Mooncake store config;
their HDK 25.2.3 requires < 20 GB, and setting it to 2 GB fixed everything.**

## 1. Two different faults share the `503900` code — decide which one this is

`503900` / `E19999` / `HcclCommPrepare ret: 0x13` only says "the ADXL channel to
that engine could not be prepared". It is produced both by a *topology* fault
(two addresses = two HCCL servers, `transport-topology.md`) and by a *store
segment* fault (this page). One question separates them before any source work:

> **Is a KV pool in the deployment?** `AscendStoreConnector` (alone or inside a
> `MultiConnector`), `kv_pool/…`, a `pool_worker.py` / `pool_scheduler.py` line,
> `MOONCAKE_CONFIG_PATH`, or a `mooncake_master` in the recipe.

- **Yes →** the store's own memory registration is in play. Check the segment
  size first (§3); the rank table is the second question, not the first.
- **No** (pure p2p `MooncakeConnectorV1` / `MooncakeHybridConnector` /
  `MooncakeLayerwiseConnector` / LMCache-Ascend `hixl`) → go to
  `transport-topology.md` as written.

A **third** shape is neither: the channel prepared, the request succeeds, and the
**put** then fails (`Failed to invoke HcclBatchPut, hccl_result = 4`, master
`PutEnd` 0 with `PutRevoke` > 0). Segment size is then excluded (`-1`, not the
out-of-space `-200`) and the lane is `../kv-pool-store/hccl-onesided-store-put.md`.

Reporter-confirmed discriminator for #13439, worth asking for verbatim in any
lookalike: **both `MooncakeConnectorV1` and `MooncakeHybridConnector` fail
identically once the pool is attached, while PD disaggregation without the KV
pool runs fine.** A p2p-connector-only change therefore cannot fix it, and a
report that only varies the p2p connector has not tested anything yet.

## 2. Mechanism (source-verified at the pins below)

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:23-24
    DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824   # 1.0 GiB per rank
    DEFAULT_LOCAL_BUFFER_SIZE  = 1073741824    # 1.0 GiB per rank
:300-336   MooncakeStoreConfig.from_file: MOONCAKE_CONFIG_PATH json; MOONCAKE_MASTER
           overrides master_server_address, MOONCAKE_GLOBAL_SEGMENT_SIZE overrides
           global_segment_size (env wins over the file)
:338-395   _parse_global_segment_size accepts int bytes or "2GB"/"512MB"/"4096B"
           (units gb|mb|kb|b only — "40g" raises ValueError)
:126-152   store.setup(global_segment_size=… if self._contribute_memory else 0,
                       local_buffer_size=0 under fabric mem)
vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py:5-46
    global_te: ONE process-wide TransferEngine, created on first use and reused
mooncake_hybrid_connector.py:1560 (engine) / :1686 (register_buffer) use the same
    global_te, and mooncake_backend.py:126 passes transfer_engine.get_engine() in
Mooncake @v0.3.11.post1:
  mooncake-store/src/real_client.cpp:742-800  mount segment; split only above
      max_mr_size (include/config.h:41 = 1 TiB default) -> a 40 GiB request stays
      ONE memory region
  mooncake-transfer-engine/src/transport/ascend_transport/ascend_allocator.cpp:168-180
      aclrtMallocHost for the store segment, range recorded at :181-196 (the log's
      "aclrtMallocHost suc, addr: 0x…" / "Failed to allocate memory" pair)
  .../ascend_direct_transport/transfer_executor_base.cpp:404-452
      RegisterMem per engine: engine_indices = all engines when
      (roce_mode && dummy_real_mode && ascend_is_store_memory(addr, length)) or the
      process has exactly one engine, else the current device's engine
      (dummy_real_mode = globalConfig().ascend_agent_mode,
      ascend_direct_transport.cpp:151; both default false, config.h:79-80)
```

Because the store and the p2p connector share one engine (`global_te`), a store
segment the platform refuses poisons every channel that engine later dials — so the error
surfaces on whichever connector transfers first, with the p2p connector's Python
traceback. That is why "remove the pool" is diagnostic and "swap the p2p
connector" is not.

The value is in the pasted log even when `mooncake.json` is not:

```text
I0805 01:44:59.569026 158386 ascend_allocator.cpp:173] aclrtMallocHost suc, addr: 0x12e4c0000000
I0805 01:44:59.569270 158386 transfer_executor_base.cpp:452] TransferExecutor register mem
      addr:0x12e4c0000000, length:42949672960, mem type:host, engine index:0
```

`42949672960` = 40 GiB exactly = the configured `global_segment_size`. A
`mem type:host` registration in the tens of GiB, immediately followed by
`HcclCommPrepare` failures, is this family. Ignore `Local buffer size is 0, skip
registering local memory` / `Global segment size is 0, skip mounting segment` in
the same log: that is the scheduler-only store client
(`contribute_memory=False`, `mooncake_backend.py:73, 171-173`), not the worker.

## 3. Procedure (2 checks, before the rank table)

1. **Get the number, then run the checker.** Ask for the file behind
   `MOONCAKE_CONFIG_PATH` (or `echo $MOONCAKE_GLOBAL_SEGMENT_SIZE`), plus the
   `npu-smi` version line for the HDK. Then
   `bash <skill_root>/scripts/kvpool-segment-check.sh <mooncake.json|serve-log|->`
   — it prints the per-rank segment in GiB, recovers the size from a
   `register mem … mem type:host` line when the json is absent, and flags the
   bands. **< 20 GB per rank on HDK 25.2.3** is the reporter-confirmed working
   region; 40 GB failed.
2. **Two ablations, cheapest first, and say which one is the fix:**
   - lower `global_segment_size` to `"2GB"` (keep everything else, including the
     env block, unchanged) → the reporter's fix; or
     `MOONCAKE_GLOBAL_SEGMENT_SIZE=2147483648` if the file cannot be edited.
   - drop the pool: `AscendStoreConnector` removed from the `MultiConnector` (and
     `MOONCAKE_CONFIG_PATH` unset), p2p connector kept → reporter-confirmed to
     run fine, and it proves the pool is the carrier without touching HCCL.

Only if both ablations leave the failure, or if the deployment has no pool at
all, continue to the rank-table / plog route of `transport-topology.md`.

If you open the pool guide
(`docs/source/user_guide/feature_guide/kv_pool.md`) for any other reason, grep it
in the same call for `global_segment_size|MOONCAKE_GLOBAL_SEGMENT_SIZE|1GB`: the
json example (`:97-116`) and the fabric-mem budget (`:1010-1042`) both name this
knob, and reading that page only around `lookup_rpc_port` (a plausible wrong
lead on this symptom) misses both.

## 4. Documented numbers to quote (pinned docs, not recalled defaults)

`docs/source/user_guide/feature_guide/kv_pool.md` @ v0.21.0rc1…v0.23.0rc1:

- `:97-113` example json, `"global_segment_size": "1GB"`, described as
  *"Registered memory size per card to the KV Pool. Needs to be aligned to
  1GB"*, with `MOONCAKE_GLOBAL_SEGMENT_SIZE` taking precedence over the file.
- `:1010-1042` (A3 + `ASCEND_ENABLE_USE_FABRIC_MEM=1`): every fabric allocation
  must be a 1 GB multiple; the budget is
  `fabric_memory.max_capacity ≥ global_segment_size + MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`,
  and the doc's own example sizes a **20 GB** segment + 1 GB buffer →
  `ASCEND_GLOBAL_RESOURCE_CONFIG='{"fabric_memory.max_capacity":32}'`. So 20 GB
  per rank is the largest figure the pinned docs themselves use, and it is
  conditional on raising that quota.
- `:982-988` splits pool failures into Mooncake-reported `put`/`get` errors versus
  "HIXL (ascend_direct) transfer-layer issue" (plog + HIXL wiki). A failure at
  *channel creation*, before any key is put or got, fits neither — do not answer
  "failed to put/get key" for it.

## 5. What not to chase (already excluded by the reporter's own fix)

The reporter changed **only** `global_segment_size`; `HCCL_INTRA_ROCE_ENABLE=1`,
`HCCL_RDMA_TIMEOUT=17`, `ASCEND_CONNECT_TIMEOUT=120000`,
`ASCEND_TRANSFER_TIMEOUT=120000`, `LD_PRELOAD=libjemalloc`, the 8-card
`dp8/tp1` layout and both serve commands stayed as posted in the working run.
So none of those is the differentiator for this case, and the two recorded
answers that ranked them were wrong in the same way — see §6.

Mechanical notes that kill two plausible-looking leads:

- **The `timeout:120000 ms` in the message is the configured
  `ASCEND_CONNECT_TIMEOUT`, echoed.** It is not elapsed time: the two attempts
  are 0.45-0.49 s apart, so this was an immediate rejection, not a slow dial.
- **The two engine ports are not a mismatch.** `<D>:20006` and `<P>:20064` (later
  run: `:20039` / `:20071`) both sit in ADXL's `20000 + phy_id*100` band
  (`ascend_direct_transport/utils.cpp`, `base_port_ = 20000`, override
  `ASCEND_BASE_PORT`), which is exactly where per-device engine names live.
- **`kv_buffer_size` is not this knob.** `kv_buffer_size` (16 GiB here) /
  `kv_block_size` are in the *connector* `kv_connector_extra_config` and are
  registered by `register_buffer` (`mooncake_hybrid_connector.py:1686`);
  `global_segment_size` lives in the *store* json. They are different files,
  different allocators, and only the latter was the fault.
- **`rope_parameters` / `block size should be 32` / `shm_broadcast ... 60
  seconds` warnings in the same log are noise** — normal at startup and present
  in the working run.

## 6. Recorded failure mode of this page's absence

Both recorded answers on #13439 classified the report as a transport/HCCL fault
and answered from the issue body alone (the D-node log and the "both p2p
connectors fail, PD without the pool is fine" remark arrived afterwards). One
ranked `HCCL_INTRA_ROCE_ENABLE=1` second and the `HCCL_RDMA_TIMEOUT` /
`ASCEND_CONNECT_TIMEOUT` interaction third, and proposed plog greps for
`server_count|device_ip|HccsNum`; the other named the ADXL transport as owner and
stopped. Neither asked two questions that were available from the config in the
report:

1. **which store config** — the recipe exports
   `MOONCAKE_CONFIG_PATH=/mnt/fzx/mc_config/mooncake.json`, so `global_segment_size`
   is a value the reporter can read out in one line;
2. **does the failure survive without the pool** — the single ablation that
   separates a store fault from a link fault.

The missing knowledge was therefore the *route* (a pool deployment is not the
rank-table family) and the artefact to request, not a line in a log. Once the
reporter's own log arrives, `scripts/kvpool-segment-check.sh` recovers the number
from `length:42949672960` in one call.

## 7. Evidence limits to state

- **The 20 GB ceiling is reporter-confirmed, not documented.** It is their
  HDK 25.2.3 + CANN 9.0.1 + 910B2 + RoCE (no fabric mem) stack. Other HDKs, or
  A3 fabric mem with a raised `fabric_memory.max_capacity`, may allow more; quote
  the number with its scope and ask for the HDK version before generalising.
- **The mechanism link is circumstantial at one step.** The reporter never
  posted `mooncake.json` nor a log from the working run, so "the 40 GiB host MR
  is the store segment" rests on the exact byte count (42949672960 = 40 GiB) and
  `ascend_is_store_memory`/`aclrtMallocHost` in the pinned sources, not on an
  instrumented reproduction.
- **The failing library is not readable here.** `channel.cc`,
  `channel_msg_handler.cc`, `adxl_engine_impl.cc` and `HcclCommPrepare` are
  absent from the snapshot: `cann/hixl` is listed `unavailable` in
  `SOURCES.json`, and `Ascend/memfabric_hybrid v1.1.2` is a different product
  (`hybm`/`hcom` transport, `src/hybm/…`), not the ADXL engine Mooncake links
  against. Do not spend calls opening `channel.cc`; the plog's own error line and
  the segment-size ablation are the evidence.
- Recheck conditions: if `mooncake_backend.py` gains a size validator, or
  `DEFAULT_GLOBAL_SEGMENT_SIZE` moves off 1 GiB, the "default is small, the user
  enlarged it" framing changes; if a later HDK documents a per-MR cap, cite that
  instead of this page's empirical figure.
