# P/D transport identity, HCCL rank tables and `HIXL Connect failed`

Scope: the transport-identity family — no KV pool in the deployment. When an
`AscendStoreConnector` / `MOONCAKE_CONFIG_PATH` pool *is* configured, the same
`503900` block is usually the store segment size, not the rank table: take step 0
below and `kv-pool-store-segments.md` first.

Sibling for the **Mooncake** flavour: a `MooncakeConnectorV1` paste whose lines are
`ascend_direct_transport.cpp:820 Failed to connect to target … 103900` +
`Mooncake transfer failed, ret: -1` (not `HIXL Connect failed`) →
`mooncake-adxl-connect-103900.md`. Same rank-table family, but the deliverable there
is the D-side plog first, and `hostNetwork: true` — the recommendation in §3 below —
only works for two Pods of **one** machine; cross-node P/D keeps `server_count = 2`
and needs a valid `device_ip` instead.

Worked case: #14506 — single node, 8×910B3, **no RoCE** (all `hccn_tool -i N
-link -g` → DOWN, `-optical -g` → not present), k8s 1.36, 1P1D in two Pods with
`hostNetwork: false`, `LMCacheAscendConnector` with `transfer_channel: "hixl"`,
vllm-ascend v0.18.0rc1 + lmcache-ascend 0.4.4 + CANN/HIXL 8.5.1. Symptom:

```text
HIXL Connect failed | HIXL Error Code: 103900
(lmcache_ascend.v1.transfer_channel.hixl_channel._connect_to_peer)
HCCL topoinfo_ranktableConcise.cc:458 [InitGroupStage][RanktableCheck]
     'device_ip' is not set correctly, must be set when multi Server
     or HCCL_INTRA_ROCE_ENABLE enabled
HCCL topoinfo_ranktableConcise.cc:308 [Get][SingleServer] get dev list error:serverId[[NETWORK_ADDRESS_2]]
```

## 1. The mechanism (reporter-confirmed by their own rank tables)

The engine id is one string doing three jobs: the local TCP listen address, the
`Connect(remote_engine_id)` dial target, and the HCCL rank table's `server_id`.
LMCache-Ascend derives it and offers no override:

```text
lmcache_ascend/v1/transfer_channel/hixl_engine.py:37-42   (v0.4.3 @3474a0b)
    ip = get_ip(); port = _find_free_port()
    self.engine_id = f"{ip}:{port}"
    self.engine.initialize(self.engine_id, options)
lmcache_ascend/v1/transfer_channel/hixl_channel.py:68-73, 118-124
    engine.connect(remote_engine_id)          # same string, dial side
lmcache_ascend/v1/transfer_channel/hixl_protocol.py:24-28
    HixlInitRequest.local_id / .engine_id     # exchanged in the handshake
lmcache/v1/rpc_utils.py:89-104                (LMCache @operator-v0.5.0)
    get_ip() = UDP-connect 8.8.8.8 -> local outbound address
lmcache_ascend/v1/rpc_utils.py:43-46
    _find_free_port() = bind(("",0)) then close   # random, TOCTOU
```

Under a CNI that outbound address is the **Pod IP**, so two Pods on one physical
node advertise two addresses; HIXL writes them into two `server_list` entries,
the table HCCL receives therefore declares `server_count=2`, HCCL concludes
cross-server traffic, requires `device_ip`, and selects RoCE — which is DOWN. Same pattern on Mooncake's
asc-end-direct transport, where the engine name is the segment name's IP plus a
per-device port (`GenAdxlEngineName(host_ip, listen_port)`,
`FindAdxlListenPort` = random port in `[base + phy_id*100, base + (phy_id+1)*100)`,
`base_port_ = 20000`, override `ASCEND_BASE_PORT`;
`mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.cpp:131-200`,
`.../utils.cpp:26,68-81`).

With `docker --net=host` (or k8s `hostNetwork: true`) both sides report the same
host IP, `server_count` converges to 1, and HCCL picks HCCS: the reporter's
plog showed `MemcpyAsync` between the two cards with
`mac_tx/rx_total_pkt_num` and `roce_tx/rx_all_pkt_num` all zero for the whole
test while every NIC stayed DOWN. That is the observation that distinguishes
"data went over HCCS" from "data was supposed to go over RoCE".

## 2. Procedure (cheapest first, ~4 checks)

0. **First: is a KV pool in this deployment?** `AscendStoreConnector` (alone or
   inside a `MultiConnector`), `MOONCAKE_CONFIG_PATH`, a `pool_worker.py` /
   `pool_scheduler.py` line, or a `mooncake_master`. If yes, the `503900` /
   `HcclCommPrepare ret: 0x13` block is more likely the store's
   `global_segment_size` than the rank table: it is allocated with
   `aclrtMallocHost` and registered into the process's ADXL engine, so every
   channel that engine later dials fails — including the p2p connector that
   reports it.
   Run `bash <skill_root>/scripts/kvpool-segment-check.sh <mooncake.json|log>` and
   read `kv-pool-store-segments.md` before the three checks below. Reporter-
   confirmed on #13439 (both p2p connectors fail with the pool, PD without the
   pool is fine); the recorded answers that skipped this gate ranked
   `HCCL_INTRA_ROCE_ENABLE` and the timeouts instead and missed the real fix.
1. **Get the rank table and read it with the script.** It is printed by HIXL and
   visible in the plog or in the reporter's paste:
   `bash <skill_root>/scripts/ranktable-check.sh <table.json|log.txt>`.
   `server_count == 1` with one `server_id` → HCCS path is reachable and
   `device_ip` is not required; `server_count > 1` → the rest of the diagnosis is
   decided. **Ask for both sides' tables** (and both ranks when TP > 1): the
   older reports show each rank logging its own table, and one rank carrying a
   repeated `device_id` while another looked fine (#1680:
   `llmdatadist_c_mgr_connector.py:633` printed `server_count: 2` with
   `device_id 6` on both entries). One table is not the cluster.
2. **Ask for the plog, not the traceback.** On both P and D:
   ```bash
   export ASCEND_GLOBAL_LOG_LEVEL=1            # INFO
   export ASCEND_SLOG_PRINT_TO_STDOUT=0
   grep -nE "server_count|server_id|ranktable|RanktableCheck|device_ip|isUsedRdmaLevel0|HccsNum" \
        ~/ascend/log/debug/plog/plog-*.log
   ```
   Three lines decide it: the rank table's `server_count`; whether
   `'device_ip' is not set correctly` appears; and `HccsNum` / `isUsedRdmaLevel0`
   (non-zero `HccsNum` = HCCS links exist for this process). Also ask for each
   side's `Connecting to remote HIXL engine: <ip:port>` line —
   `hixl_channel.py:69` already logs the string that became `server_id`.
3. **Intra-node link type.** `cann/hcomm` (exported in the snapshot,
   `@818068c8`, `src/platform/common/externalinput.cc:634-715`) parses
   `HCCL_INTRA_PCIE_ENABLE` / `HCCL_INTRA_ROCE_ENABLE`:
   both unset → PCIe/HCCS intra-node; `HCCL_INTRA_ROCE_ENABLE=1` alone →
   intra-node RoCE (`intraPcie=0`), which on DOWN links always fails and also
   turns on the `device_ip` requirement; **`HCCL_INTRA_ROCE_ENABLE=0` alone is a
   hard `HCCL_E_PARA` error** ("pls set HCCL_INTRA_PCIE_ENABLE",
   `externalinput.cc:110-111, 665-676`), so the safe non-RoCE form is
   `HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0` or leaving both unset.
   Mooncake's ADXL reads the same variable independently and flips its own
   `roce_mode_` when it is 1 (`ascend_direct_transport.cpp:152-163`), so one
   leftover env var in an entrypoint defeats both layers.
4. **Only then the connector/source.** Cross-check device identity: with
   `server_count=1` ranks are distinguished by `device_id`, so confirm P and D
   really hold different cards (`ASCEND_VISIBLE_DEVICES` physical list vs
   `ASCEND_RT_VISIBLE_DEVICES` logical remap) — a duplicate `device_id` is a
   `PROBLEM:` line in the script's output. Watch port bands too: two Pods on the
   same physical card draw from the same 100-port band, and a collision makes the
   two engine ids identical.

## 3. What to propose, in order

- **Same server_id, supported way**: `hostNetwork: true` for both Pods (or
  `docker --net=host`) keeps every other isolation property. That is the
  configuration in which the reporter does not see the failure: both sides then
  advertise one address and `server_count` converges to 1. This is the
  **single-machine** answer; across two machines the table stays `server_count = 2`
  and the requirement is a `device_ip` per device (`hccn_tool -i N -ip -g`, A3
  `-vnic -g`) — `mooncake-adxl-connect-103900.md` §3 and
  `plane-selection-a3-parameter-plane.md` §1.
- **Sibling channel as an experiment**: LMCache-Ascend's `hcomm_onesided`
  channel builds its own rank table and takes `server_id` from
  `socket.gethostname()`, with `device_ip` optional
  (`hcomm_onesided_runtime.py:195-243, 246, 274-296`). It is the only place in
  that repo where `server_id` is a free string, so equal Pod `spec.hostname` is a
  cheap experiment — but `_get_device_ip()` falls back to
  `hccn_tool -i N -ip -g`, which returns an address even when the links are DOWN,
  so the table can still carry `device_ip`. Label this a hypothesis until a
  `server_count=1` table is shown.
- **Feature request upstream, not here**: identity/dial separation
  (`HixlInitRequest` gains a dial address; `HIXL_ENGINE_IP`/`PORT` or an
  `LMCACHE_<KEY>`-style config key, pattern at `lmcache/v1/config.py:990-991`)
  belongs in `LMCache/LMCache-Ascend`; `CreateTransferChannel(..., **kwargs)`
  already forwards kwargs (`transfer_channel/__init__.py:100-152`) while
  `storage_backend/pd/backend.py:151-162` passes none. The ZMQ handshake address
  is already configurable (`LMCACHE_PD_PEER_HOST` / `..._PEER_INIT_PORT`,
  `backend.py:114-126`) — only the HIXL identity is not.

## 4. Precedent (same family, older connector)

The mechanism predates LMCache-Ascend's HIXL channel: the `.9x`/`v0.1x` connectors
(`llmdatadist_c_mgr_connector.py`) used the same HCCL rank table, hand-built or
generated, and the same two outcomes.

- `#2399: link failed using llmdatadist for disaggregated prefill` — the reference
  v1.2 table declares `server_count: "1"` with prefill and decode devices under
  one `server_id`; the failing hand-written table declares `server_count: "2"`,
  with two entries that *share* one `server_id` and still fail
  (`LLM_LINK_FAILED`). The declaration wins: equal ids do not rescue a
  `server_count: 2` table. Its reproduction also exports
  `HCCL_INTRA_ROCE_ENABLE=1` on both sides.
- `#1680: 0.9.1-dev ... llm_datadist link error` — per-rank tables in one log with
  `server_count: 2`, `device_id` repeated inside a server, and different
  `device_ip` sets per rank.
- `#1528`, `#2498` — same `server_id`/`server_list` JSON in the pasted logs.

Use these as "the same symptom was reported with a different connector", not as
proof about the HIXL case: the string formatting and the API differ
(`link()` vs `connect()`), and all of these are redacted (IPs replaced by
`[NETWORK_ADDRESS_n]`), so only the shape is comparable.

## 5. Evidence limits to state in the answer

- `cann/hixl` is **not** in the snapshot (`SOURCES.json: unavailable:
  no_verified_public_reference_by_opening`), so "engine-id IP becomes
  `server_id`" is proven by the reporter's rank table, not by HIXL source.
- `cann/hcomm` **is** exported: `topoinfo_struct.h:145,156,35,62` gives the field
  constants (`device_ip`, `server_count`, `serverId` = 集群内服务器唯一标识) and
  `topoinfo_ranktableConcise.h:40-44` the `GetSingleServer` entry points, but the
  `.cc` implementations that emit `'device_ip' is not set correctly` are not in
  the export. Header + plog together are decisive; there is no further source to
  walk.
- LMCache-Ascend's own examples say RoCE is required and `hixl` is Experimental
  (`examples/disagg_prefill/1p1d/README.md:11,26-27`, same in
  `examples/kv_cache_reuse/share_across_instances/p2p_sharing/README.md:9,25-26`);
  their channel tests force `HCCL_INTRA_ROCE_ENABLE=1`
  (`tests/v1/transfer_channel/test_hixl_channel_write.py:89-91`). So single-node
  HCCS over `hixl` is untested-by-vendor, not documented, and evidently possible.
- Version gap to recheck: reporter on lmcache-ascend 0.4.4 / vllm-ascend
  v0.18.0rc1, snapshot pin is LMCache-Ascend 0.4.3 / vllm-ascend v0.23.0. Line
  numbers move; `hixl_engine.py` gaining an env override would invalidate the
  "no config entry" claim.
