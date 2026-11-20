# PD/EPD disaggregation: transport plane and KV-state transfer

Scope: a P/D (or EPD) deployment that starts but cannot move KV, or moves it and
produces wrong tokens — *not* the generic startup lane. Three failure families,
decided by two questions: **is a KV pool in the deployment?** then **did the
peers fail to connect, or did they connect and the decode node then produce
wrong output?**

- A KV pool (MemCache/Mooncake `AscendStoreConnector`) on a **single node, no PD**,
  or a store that initialises and then fails while saving: this topic's transport
  rules do not apply — `../kv-pool-store/index.md` (A3 + DeepSeek-V4 +
  `device_sdma`: `../kv-pool-store/memcache-a3-dsv4-init.md`).
- Connect/handshake failures **with a KV pool in the config**
  (`AscendStoreConnector`, alone or inside a `MultiConnector`, `MOONCAKE_CONFIG_PATH`,
  `pool_worker.py` / `pool_scheduler.py`): `Failed to connect to target … 503900`
  / `E19999` / `HcclCommPrepare … ret: 0x13` on the first transfer, garbled output
  after it, `kv_load_failure_policy: recompute` recomputing every request → the
  **store segment size** (`global_segment_size`) before anything else, then
  `kv-pool-store-segments.md`.
- Connect/handshake failures **without** a pool: `HIXL Connect failed | HIXL
  Error Code: 103900` (or 503900 and friends), `LMCache ERROR: wait_for_save
  failed ... skipping save`, HCCL `RanktableCheck` / `'device_ip' is not set
  correctly`, `get dev list error:serverId[...]`, P/D pairs that only work with
  `hostNetwork: true`, `--net=host`, or one shared IP →
  `transport-topology.md`.
- Connected, but the answer is corrupt/empty/alternating, or the decode node's
  first token is wrong, on a **hybrid linear-attention (GDN/Mamba)** model —
  Qwen3.5/Qwen3.6, DeepSeek-V4, GLM — under PD →
  `hybrid-state-transfer.md`.
- Connected, the answer degenerates only at long context **and** concurrency on a
  **dense-attention MoE** model (Kimi-K2.x MLA, Qwen3-MoE) with spec decode and/or
  prefix caching, and both serve commands are in the report: do not start in
  `kv_p2p/`. Diff the two commands against
  `docs/source/tutorials/models/<model>.md` first and hand back that diff plus the
  ablation rows (ranked, none an established cause; #13002 never reached a
  diagnosis) → `../accuracy-benchmarks/garbled-long-context-output.md`.
- Nothing connects at all because the connector module itself fails to import —
  `ImportError: ascend_transport.so: cannot open shared object file` at
  `kv_p2p/mooncake_connector.py:24`, `No module named mooncake` — or the decode
  side dies in `register_kv_caches` with an `AttributeError` on a config scalar →
  `../image-dependency-faults/launch-env-and-library-path.md` and
  `../startup-failures/connector-config-attributes.md` respectively; no transport
  or rank-table theory applies before the module imports.

Inherit the rules in SKILL.md. Rules specific to this topic:

1. **The transport identity is a topology input, not a tunable.** Any Ascend
   transport that hands HCCL one `ip:port` string per process (HIXL/ADXL through
   LMCache-Ascend or Mooncake's asc-end-direct transport) uses that same string
   as the HCCL rank-table `server_id`. Two processes on one physical node with
   two different addresses are therefore *two servers* to HCCL, which then
   demands `device_ip` and a NIC data path. Before reading connector code,
   establish what the rank table says and run
   `bash <skill_root>/scripts/ranktable-check.sh <table-or-log>`; it prints the
   effective `server_count`, the per-server ids and the decision that follows.
2. **A transport error code is not a diagnosis.** HIXL/HCCL numeric codes
   (`103900`, `503900`, `507057`) carry almost no information and two different
   codes are usually different failure classes. The reason lives in
   `~/ascend/log/debug/plog/plog-<pid>_*.log`; ask for that file, not for a
   better Python traceback. The exception is a pool deployment, where the
   deciding number is already in the worker log (`register mem … mem type:host`) —
   see rule 4 and `kv-pool-store-segments.md`.
3. **"Not in the documented recipe" is not "unsupported".** The LMCache-Ascend
   examples state RoCE as a prerequisite and mark `hixl` experimental, while
   reporter evidence shows the same channel completing over HCCS once the rank
   table converges to one server. Keep the two claims apart: no validated path
   vs. impossible on that hardware.
4. **`503900` is two faults wearing one code.** Decide with one question: is a
   KV pool configured? The pool's store segment is allocated with
   `aclrtMallocHost` and registered into the ADXL engine of that process — the
   same engine the p2p connector dials with (`transfer_executor_base.cpp:404-452`,
   single `global_te`), so an oversized `global_segment_size` breaks the p2p
   connector that dials first and the reported owner is the wrong one.
   Reporter-confirmed on #13439: both
   `MooncakeConnectorV1` and `MooncakeHybridConnector` fail identically *with*
   the pool, and PD without the pool is fine, which no p2p-connector change can
   explain. Ask for the file behind `MOONCAKE_CONFIG_PATH`, or run
   `bash <skill_root>/scripts/kvpool-segment-check.sh <json|log>` on what you
   have; then use `kv-pool-store-segments.md`.
5. **Name the owning repository.** vllm-ascend's `LMCacheAscendConnector` is a
   shim that re-exports upstream `LMCacheConnectorV1`
   (`vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py:1-5`
   at v0.23.0); an HIXL engine-id / rank-table change belongs to
   `LMCache/LMCache-Ascend` or `cann/hixl`, not to vllm-ascend.

## Pages

- `transport-topology.md` — P/D has two Pod IPs on one node and the transport
  will not connect: how the engine id becomes HCCL's `server_id`, the plog grep
  set and rank-table decision, `HCCL_INTRA_ROCE_ENABLE` / `HCCL_INTRA_PCIE_ENABLE`
  semantics (source-verified in the `cann/hcomm` export), port-band collisions,
  and what is verifiable in the snapshot when `cann/hixl` is absent.
- `kv-pool-store-segments.md` — a KV pool is in the deployment and ADXL channel
  creation fails (`503900` / `E19999` / `HcclCommPrepare ret: 0x13`) with garbled
  decode output: `global_segment_size` per rank versus the HDK ceiling
  (< 20 GB on 25.2.3, reporter-confirmed), the shared-`global_te` mechanism that
  makes a store fault surface on the p2p connector, the log's
  `register mem … mem type:host, length:N` fingerprint, the two ablations, and
  the documented 1 GB alignment / fabric-mem budget. Also says which knobs are
  *not* this one (`kv_buffer_size`, `HCCL_INTRA_ROCE_ENABLE`, the timeouts).
- `hybrid-state-transfer.md` — PD on a hybrid linear-attention model: the
  connector matrix (`MooncakeConnectorV1` / `MooncakeHybridConnector` /
  `MooncakeLayerwiseConnector` / `AscendStoreConnector`), the
  `mamba_cache_mode` gate that only the KV-store connectors satisfy, which
  runner paths that skips, the MTP-under-PD accounting sites (`num_draft_layers`,
  the MambaSpec `num_speculative_tokens` block index, the `"mtp" in layer_name`
  predicates) together with the refuted "our method name is not `mtp`" theory,
  and the ablation matrix to hand the reporter when the symptom is alternating
  good/bad output.
