# Mooncake `ascend_direct_transport` 103900: the decode node cannot open a KV segment

Scope: a PD deployment on **`MooncakeConnectorV1`** (or the layerwise/hybrid Mooncake
siblings) with **no KV pool**, both engines serving their own requests, and the first
proxied request failing on the decode worker. The paste is one connector line plus
Ascend-transport stderr:

```text
E1210 03:24:54.778028 2548307 ascend_direct_transport.cpp:820] Failed to connect to target: <P ip>:25389, status: 103900
E1210 03:24:54.778062 2548307 ascend_direct_transport.cpp:581] Failed to connect to segment: <P ip>:16381
(Worker_DP0_TP5 pid=2543213) ERROR mooncake_connector.py:410] Mooncake transfer failed for request cmpl-…
(Worker_DP0_TP5 pid=2543213) ERROR mooncake_connector.py:329] Failed to transfer KV cache for request …: Mooncake transfer failed, ret: -1
```

Not this page: **P and D on one machine, two cards**, `status: 103905` (or any
`1039xx`) printed **without** an `, errmsg:` field and citing
`ascend_direct_transport.cpp` → `adxl-connect-status-same-host.md` (the port band
names the physical card, the mask line dates the tree, and the cross-node
`device_ip` ladder does not apply). A KV pool in the recipe
(`AscendStoreConnector`, `MOONCAKE_CONFIG_PATH`)
→ `kv-pool-store-segments.md` first; `HIXL Connect failed` under **LMCache-Ascend**
→ `transport-topology.md` (engine-id/`server_id` identity); a module/import failure
→ `../image-dependency-faults/launch-env-and-library-path.md`.

Worked case: **#4866**, vllm-ascend `v0.11.0rc2` (`a2e4c3f`), vllm `v0.11.0`,
CANN 8.3.RC2, 8×910B2 per node, k8s, images `v0.11.0rc2`; reported as 2P2D with
`--tensor-parallel-size 8`, then narrowed by the reporter to **1P1D, one pod each,
Qwen3-4B — same error**. Cause unresolved in the thread (see §5); the deciding
artifact is the reporter's later `grep -rn ERROR /root/ascend/log` from the decode node.

## 1. What the paste already fixes, before any read

- `ret: -1` is `batch_transfer_sync_read` returning negative, not a Python fault:
  `mooncake_connector.py:407-412` raises the `RuntimeError`, `:329` logs it.
- The **side-channel handshake completed**: the decode side dialled the prefiller's
  ZMQ port (`kv_port + data_parallel_rank*tp_size + tp_rank`, `:714-717` / `:877-881`)
  and got `MooncakeAgentMetadata` back (`:51`, `:527-528`, `:1046-1048` — the peer's
  `te_rpc_port`). The discriminator is the text: a failed `_get_remote_metadata` is
  caught by the same `except Exception` (`:328-330`) and would be reported *as that
  exception*, whereas `Mooncake transfer failed, ret: -1` is the `RuntimeError` raised
  only after the metadata was in hand (`:407-412`). So "P and D cannot talk" is false;
  only the **data segment** dial failed.
- Every TP rank fails within ~2 ms of the same millisecond → nothing timed out.
  `HCCL_EXEC_TIMEOUT`, `HCCL_CONNECT_TIMEOUT`, `ASCEND_TRANSFER_TIMEOUT` and
  `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` are not the cause and cost a run each.
- `status: 103900` is not a socket errno: it is the Ascend compute-stack error id the
  transport surfaces when an internal HCCL call fails (the GE lines in the plog read
  `Call hccl api failed, ret: 0x1 … ErrorNo: 103900`). Do not translate it; read the
  plog.
  The `1039xx` values are distinct `adxl::Status` members at the `Connect()` site
  (`transfer_executor_base.cpp:261`), so 103905 is **not** 103900 relabelled — do not
  carry this page's 103900 reading onto it, and keep the enum out of the answer
  while `cann/hixl` is unexported.

## 2. The one artifact to request: the D-side plog

```bash
grep -rnE "SingleDeviceIp|GetRanktableInfo|CommClusterInfo|HcclCommInitClusterInfo|device_ip|server_id" \
     /root/ascend/log/debug/plog/*.log | head -40
```

On #4866 that returned, per decode rank and co-timed to the failed transfer (`08:04:09.80x`):

```text
[topoinfo_ranktableConcise.cc:446] [Get][SingleDeviceIp]'device_ip' is not set correctly,
    must be set when multi Server or HCCL_INTRA_ROCE_ENABLE enabled
[topoinfo_ranktableConcise.cc:295] [Get][SingleServer] get dev list error:serverId[[NETWORK_ADDRESS_4]]
[op_base.cc:436] [Init][CommClusterInfo] HcclCommInitClusterInfo failed, rankNum[0], rank[0],
    server[], device[-1], return[0x0000000005000001]
[hccl_comm.cc:1337] [DeinitZeroCopyMemoryAgent] ptr [communicator_] is nullptr
[op_base.cc:538] [Init][HcclCommInitClusterInfoMemConfig] comm Name exist.     # retry, same process
```

Reading it: the transport's own HCCL communicator was **never created**, so the
ADXL engine had no comm to bind its segment port on (inference — the transport source
is absent, §6) — which is why every later dial says `Failed to connect to target`. The `device[-1]` / `server[]` / `rankNum[0]` triple says
the rank table the transport handed HCCL carried **no usable `device_ip`**, and the
message names its own trigger (`multi Server or HCCL_INTRA_ROCE_ENABLE`).

Consequence for the action ladder: `ss -lntp` on P, `nc -vz` between the pods, and a
`hostNetwork: true` ablation do **not** discriminate here — a listening RPC port with
unbound segments is the expected shape once that init fails. Ask for the plog (or a
`grep -rn ERROR /root/ascend/log`) in the same message as the ranking, not after it.

## 3. Why `device_ip`, and why `hostNetwork` is not this fix

- "Multi server" is intrinsic to cross-node PD: P and D live on different machines, so
  `server_count` is 2 and HCCL's multi-server path requires a `device_ip` per device.
  `hostNetwork: true` is the **single-node** fix in `transport-topology.md` — there it
  collapses two pod IPs of one machine into one `server_id`; it cannot make two
  machines one server.
- The value comes from the **device plane**, not from the pod NIC:
  `examples/disaggregated_prefill_v1/gen_ranktable.py:78-101` fills `device_ip` from
  `hccn_tool -i <id> -ip -g` (non-A3) or `hccn_tool -i <id> -vnic -g` (A3), and
  `super_pod_id`/`super_device_id` from `npu-smi info -t spod-info`. Same source family
  `plane-selection-a3-parameter-plane.md` §1 reads for the A3 parameter plane.
- Check block to hand the reporter (both nodes, every device the role uses):

```bash
for i in $(seq 0 7); do hccn_tool -i $i -ip -g | grep ipaddr; hccn_tool -i $i -link -g; done
hccn_tool -i 0 -ping -g address <peer device ip>      # the address that answers is the one for device_ip
```

  Empty, `0.0.0.0`, or an address the peer cannot ping ⇒ that is the failure; the fix is
  device/NIC configuration (or `HCCL_INTRA_ROCE_ENABLE` removal, below), not a
  connector change.
- `HCCL_INTRA_ROCE_ENABLE=1` inherited from an entrypoint turns on the same requirement
  **and** intra-node RoCE, and fails on DOWN links (`transport-topology.md` §2 step 3,
  source-verified in the `cann/hcomm` export). One `env | grep HCCL` from the reporter
  removes or confirms it.

## 4. `use_ascend_direct`: check its default before treating it as the fix

At `v0.11.0rc2` the flag is read **from `kv_connector_extra_config` with default `True`**:

```text
vllm_ascend/distributed/mooncake_connector.py:905-911   (layerwise :667-671)
    if vllm_config.kv_transfer_config.get_from_extra_config('use_ascend_direct', True):
        hostname = self.side_channel_host                       # bare IP
    else:
        hostname = f"{self.side_channel_host}:0:npu_{self.device_id}"
    self.engine = get_global_te(hostname, device_name=None)     # initialize(hostname, "P2PHANDSHAKE", "ascend", "")
```

It only chooses the **engine-name form**; it cannot supply a `device_ip`. And a
top-level `"use_ascend_direct": true` in `--kv-transfer-config` is not a field of
upstream `KVTransferConfig` (`vllm/config/kv_transfer.py:20-86` — only
`get_from_extra_config` reads arbitrary keys), so moving the key into
`kv_connector_extra_config` changes **nothing** at this pin: the default already selects
the same branch. A commenter's advice to "add `use_ascend_direct`" for a 103900 connect
failure is therefore a no-op, and #4866's reporter reported "**same error still
occurs**" after doing exactly that — which is what the source predicts. Verify the
default in the *running* tree first
(`python -c "import inspect,vllm_ascend.distributed.mooncake_connector as m; print(inspect.getsourcefile(m))"`,
then grep the gate); a branch whose default is `False` is the only setting where the
advice bites, and there it changes the engine name, i.e. the segment identity.

## 5. Ranked causes after the plog read

1. **No usable `device_ip` on the device plane** (recorded #4866 follow-up, all 8 decode
   ranks): fix at `hccn_tool`/NIC/driver level with §3's block. This is the rank-table
   family of `transport-topology.md`, with `server_count = 2` by construction.
2. **`HCCL_INTRA_ROCE_ENABLE=1` with DOWN links** set by an entrypoint (§3), which also
   produces the identical `device_ip` message on a single machine.
3. **A genuine pod-network gap** for the transport ports (CNI/policy): only after 1-2 are
   excluded, and then `nc -vz -w2 <P ip> <port>` from D is cheap — but the ports are the
   peer's advertised transport ports (`:384` `session_id = f"{remote_host}:{remote_transfer_port}"`),
   not `kv_port`, so a policy written for the handshake range says nothing about them.
4. **Connector-side identity collision** (both pods of a role reusing `engine_id`/
   `kv_port`, handshake-port arithmetic against `data_parallel_rank`) — the 1P1D
   follow-up in #4866 refutes it as the primary cause; keep it only for a layout that
   still has two pods per role.

## 6. Evidence limits to state in the answer

- The Mooncake pin used for #4866 (`v0.3.4.post2`, 322/340 files) had **no**
  `mooncake-transfer-engine/src/transport/ascend_transport/`, so there
  "103900 = the transport's internal HCCL init failed" rested on the plog lines
  co-timed in the same process, and the per-port roles (`:820` target vs `:581`
  segment) on the message text alone. **That limit is lifted for later releases**:
  at `v0.3.11.post1` (`e9c610757200`) the directory is exported and both sites are
  readable — `transfer_executor_base.cpp:256-265` (`Connect timeout to:` vs
  `Failed to connect to target: …, status: …, errmsg: <aclGetRecentErrMsg()>`) and
  `ascend_direct_transport.cpp:131-176` (band allocation, `ASCEND_BASE_PORT`,
  `roce_mode_`). Two uses: the message text now dates the reporter's libmooncake
  (no `, errmsg:` ⇒ older than that pin), and the listen-port arithmetic is a
  device-identity check (`adxl-connect-status-same-host.md` §4). `cann/hixl` is
  still in the unavailable list, so the **numeric** status values stay unreadable.
- #4866's cause is **open**: the reporter's last message is the plog above, after the
  config move. Do not report `use_ascend_direct` or the DP=2 layout as the fix.
- Version scope: the `:905-911` gate, the default and the metadata field names move
  between releases; re-grep on the reporter's tree before quoting line numbers.
