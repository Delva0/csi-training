# Pool puts time out after minutes of service: `Transfer timeout to:` → disconnect → `-800` → ADXL `erase` segfault

## Scope, and the case this page is written from

An **aggregated** pool deployment (`AscendStoreConnector`, `kv_role: kv_both`, no P/D
split) on K8s, several pods across two nodes, Mooncake store over ADXL/RoCE. It starts,
serves, and **stores KV for a while** — then one peer starts timing out, the engine
disconnects it, every later put fails `-800`, the requests keep answering `200`, and
some minutes later the engine dies with `!!! Segfault encountered !!!`.

Recorded **#10532** (vLLM 0.18.0 / vllm-ascend `v0.18.0` image repackaged from
`quay.io/ascend/vllm-ascend:v0.18.0`, Qwen3-8B, 4 aggregated pods over 2 nodes,
`--tensor-parallel-size 1 --gpu-memory-utilization 0.8 --block-size 128
--max-num-batched-tokens 40960`, `AscendStoreConnector` `backend: mooncake`,
`kv_connector_extra_config {prefill/decode: dp1 tp1, lookup_rpc_port: "1"}`,
`kv_port: 20000`, `HCCL_INTRA_ROCE_ENABLE=1`, `MOONCAKE_CONFIG_PATH=/app/mooncake.json`).
**The case ends unresolved**: the reporter closed it on 2026-07-09 saying the failure
"occurred randomly", that they could not reproduce it, and that they did not apply the
proposed fixes. Nothing below is a confirmed cause — §1-§3 are established from the
paste and the pins, §4 is the ordered way to find out, §5 is what was never tested.

The paste, in order (times preserved because the order *is* the argument):

```text
I ... ascend_direct_transport.cpp:1148] Connected to segment: <ip>:20097, cost:586867 us.
I ... ascend_direct_transport.cpp:1148] Connected to segment: <ip>:20076, cost:74559 us.
I ... ascend_direct_transport.cpp:1148] Connected to segment: <ip>:20091, cost:41915 us.
   … ~2.5 min of 200 OK and "External prefix cache hit rate: 6.3%" (puts landing) …
E 12:35:24 ascend_direct_transport.cpp:859] Transfer timeout to: <ip>:20091, you can
             increase the timeout duration to reduce the failure rate by configuring
             the ASCEND_TRANSFER_TIMEOUT environment variable.
I 12:35:24 ascend_direct_transport.cpp:871] transfer failed and disconnect to:<ip>:20091
E 12:35:24 client_service.cpp:1181] Transfer failed for key Qwen3-8B@…: TRANSFER_FAIL (Transfer 0 failed)
   … the same pair every 10 s (…:34 …:44 …:54 …), then :20097 and :20076 join …
E 12:37:24 transfer_task.cpp:353] Failed to complete transfers after 60 seconds for batch 2814454…
E 12:37:24 client_service.cpp:1272] Failed to revoke put for key Qwen3-8B@…: OBJECT_NOT_FOUND
E ...      mooncake_backend.py:86] Failed to put key [<32 keys>],res:[0, -800, 0, …, -800]
E 12:38:24 ascend_direct_transport.cpp:864] Transfer slice failed with status: 503900
I 12:38:24 ascend_direct_transport.cpp:1169] Target adxl engine: <ip>:20091 is not connected.
E 12:39:04 …:20097 and :20076 time out; 12:39:04.626 :20076 "Connected" again
!!! Segfault encountered !!!
File "<unknown>", line 0, in std::_Hashtable<unsigned long, unsigned long, …,
     std::__detail::_Identity, …>::erase(std::__detail::_Node_const_iterator<unsigned long, true, false>)
File "<unknown>", line 0, in mooncake::AscendDirectTransport::processSliceList(
     std::vector<mooncake::Transport::Slice*, std::allocator<mooncake::Transport::Slice*> > const&)
```

Not this page: `Failed to connect to target … status: 503900` / `HcclCommPrepare ret:
0x13` **at startup** → `../pd-disaggregation/kv-pool-store-segments.md` (§3 of that page
also owns `global_segment_size`, which this page's case also has — see §3 below).
`Failed to invoke HcclBatchPut, hccl_result = 4` with the request succeeding →
`hccl-onesided-store-put.md`. `103900` on a `MooncakeConnectorV1` P/D pair →
`../pd-disaggregation/mooncake-adxl-connect-103900.md`. `103905` on one machine →
`../pd-disaggregation/adxl-connect-status-same-host.md`.

## 1. Cause vs consequence: read the timestamps, not the severity

Both of the codes that look alarming are **downstream** here:

- `Transfer timeout to:` at `12:35:24` and `Transfer slice failed with status: 503900`
  at `12:38:24` are three minutes apart. `503900` in this paste is the status of a
  slice whose engine was already disconnected, **not** the `Failed to connect to
  target … status: 503900` of the segment-size family (different log line, different
  verb, startup vs minute 4). Quote both times whenever you use either code.
- `-800` is `TRANSFER_FAIL` (`mooncake-store/include/types.h:307` at Mooncake
  `v0.3.11.post1`). `-200` (`NO_AVAILABLE_HANDLE`, `:268`) is the out-of-space code and
  `-400` (`INVALID_KEY`, `:275`) the key code, so capacity and key format are excluded
  by the codes themselves — do not spend a call on pool capacity or on the
  `Qwen3-8B@pcp0@dcp0@…` key shape.
- `OBJECT_NOT_FOUND` (`-704`) on the revoke is the **lease** story, not a second bug:
  the master's default lease is `DEFAULT_DEFAULT_KV_LEASE_TTL = 5000` ms
  (`types.h:85`, `static_assert … == 5000` at `mooncake-store/src/master.cpp:26`), and
  the recorded `mooncake_master` command carried **no** `--default_kv_lease_ttl`. A put
  that waits 60 s (`Failed to complete transfers after … milliseconds for batch`,
  `transfer_task.cpp:387-397`) therefore outlives its own lease. The pool guide already
  says the master needs `--default_kv_lease_ttl` larger than the transfer budget
  (`docs/source/user_guide/feature_guide/kv_pool.md:130` uses 11000 beside
  10000/10000; the rule is `:134`).
- Every later `Target adxl engine: … is not connected.` is the erased connection map,
  not new evidence: at the pin a disconnect erases both `engine_connections`
  (`transfer_executor_base.cpp:297`) and, on `disconnectAllForEngine`, the segment set
  (`:331`), so a single failed peer turns into "not connected" lines for peers that
  never timed out. The reconnect at `:20076` (`Connected to segment … cost:48724 us`)
  one second before the crash is the same story.

## 2. The crash frame, and why it dates the library

`AscendDirectTransport::processSliceList` → `_Hashtable<unsigned long, …, _Identity …>::erase`
is **not** the pinned layout. At `v0.3.11.post1` (`e9c6107`):
`ascend_direct_transport.cpp` is 431 lines, `processSliceList` is
`TransferExecutorBase::processSliceList` (`transfer_executor_base.cpp:516`), the failure
path is `markSlicesFailed` (`:42`, `:600`) with no container erase, `Connected to segment`
is `:269`, `is not connected` `:292`, and the timeout line is `:593-601` and reads
`Transfer timeout to: <engine>, errmsg: <aclGetRecentErrMsg()>` — **without** the
"you can increase the timeout duration … ASCEND_TRANSFER_TIMEOUT" advice suffix the
reporter's build prints. So the image ships a different (older, monolithic) libmooncake,
quoting `ascend_direct_transport.cpp:859/864/871/1148/1169` and a
`Transfer slice failed with status:` string that exists nowhere in the pin.

Consequence: "the crashing frame is gone" is a **build difference**, never proof the
defect is fixed. Identify the installed wheel (`pip show mooncake-transfer-engine-npu`,
`strings …/libmooncake_transfer_engine.so | grep -m1 'Transfer timeout'`), bump it toward
the build paired with the reporter's vllm-ascend release, and ask for a coredump
(`ulimit -c unlimited`, `/proc/sys/kernel/core_pattern`) before filing the crash with
kvcache-ai/Mooncake — vllm-ascend owns *when/how much* it puts
(`mooncake_backend.py:121-136` at `v0.20.2rc1`), never the transfer. Note the same
divergence on the Python side: `Failed to put key %s,res:%s` is
`mooncake_backend.py:132` at the pin, `:86` in the reporter's 0.18.0.

## 3. Three traps in the config surface (the reporter's actual `mooncake.json`)

Later comment on #10532, verbatim keys (client side, one file, `MOONCAKE_CONFIG_PATH`):

```yaml
local_hostname: "$POD_IP"
metadata_server: "redis://example.invalid:6379"
master_server_address: "mooncake-master-service:30089"
device_name: ""
protocol: "ascend"
global_segment_size: 42949672960
use_ascend_direct: true
```

with `mooncake_master --port 50051 --metrics_port 9003` and a Service whose `rpc` port
30089 targets `50051` (endpoint `:50051`), `http` 9003 → 9003.

1. **The Service port remap is correct, not a finding.** `mooncake_master` defaults to
   50051 and the client dials the *Service* port 30089; `targetPort` bridges them. A
   `master_server_address` port that differs from `--port` is therefore **not** evidence
   of a misconfiguration — verify via the Service's endpoints, never by comparing the two
   numbers. Same for the Redis metadata address: the control plane demonstrably worked
   (three `Connected to segment` lines, keys allocated replicas, `res:[0, …]`).
2. **The transfer endpoints are not the master.** `:20091/:20097/:20076` are ADXL
   listen ports: `ascend_direct_transport.h:79` defaults `base_port_ = 20000` and
   `utils.cpp:26,75-83` picks `base_port + physical_dev_id*100 + random(0..100)`, so a
   port in `[20000,20100)` is the **physical device 0 engine of that peer** — the three
   numbers are three peers, not three services. (The design doc's table still lists
   `ASCEND_BASE_PORT` default 11000; the header is what runs.) A
   `Transfer timeout to:` line is therefore per-device engine traffic and cannot be
   produced by the master, the metadata server, or their ports.
3. **`global_segment_size: 42949672960` = 40 GiB is the same byte count that failed in
   `../pd-disaggregation/kv-pool-store-segments.md` (#13439, HDK 25.2.3, 2 GB worked)** —
   but there the failure was a *startup* `HcclCommPrepare`/`503900`; here the segments
   mounted and puts landed for minutes, so that page's diagnosis does not transfer
   unmodified. Keep it as an ablation **after** the timeout discrimination (§4), and note
   that with `kv_both` every pod contributes its own host MR.

Also from that config: `protocol: "ascend"` names no transport by itself (build flags do,
see `hccl-onesided-store-put.md` §1); the `ascend_direct_transport.cpp` +
`processSliceList` frames are what prove ADXL. `device_name: ""` is the documented
container form (`rdma_devices` is passed straight through).

**Which keys that file actually reaches is a grep, and three of them are not the
story.** `MOONCAKE_CONFIG_PATH` is parsed here by vllm-ascend's own
`MooncakeStoreConfig` (`ascend_store/backend/mooncake_backend.py:166-197` at
v0.20.2rc1), **not** by the Mooncake wheel's `MooncakeConfig`
(`mooncake-wheel/mooncake/mooncake_config.py:120-146`), and it reads exactly
`metadata_server`, `global_segment_size` (env `MOONCAKE_GLOBAL_SEGMENT_SIZE` wins,
`:182-188`), `local_buffer_size`, `protocol` (default `ascend`), `device_name`,
`master_server_address` (env `MOONCAKE_MASTER` wins) and the two segment flags;
**every other key in the file is dropped silently**.

- `local_hostname: "$POD_IP"` is **inert on this path** — it is not one of the keys
  read. `MooncakeBackend._setup_store` names the segment from `get_ip()`:
  `local_seg = <pod ip>:<transfer_engine rpc port>` (`:65-71`; the fabric branch uses
  the bare ip, `:83`). So do not send the reporter to the init container over an
  unexpanded `$POD_IP`; ask for the mounted file only when a *read* key's value is in
  question (`kubectl exec … cat $MOONCAKE_CONFIG_PATH`).
- `use_ascend_direct: true` reaches no code here. It was read until the retirement
  (`get_from_extra_config('use_ascend_direct', True)` at v0.11.0rc2,
  `mooncake_connector.py:905-911`, default already `True` and only picking the
  engine-name form — `../pd-disaggregation/mooncake-adxl-connect-103900.md` §4), it
  exists nowhere in `vllm_ascend/` at v0.20.2rc1 (`rg -ni 'use_ascend_direct'
  vllm_ascend/` empty while the tutorials and nightly yamls still carry it), and it is
  in any case a `MooncakeConnectorV1`-family `kv_connector_extra_config` key, not a
  field of `mooncake.json`, whose reader drops it. So it is never the lever: the
  transport is `protocol` plus build/env (`ASCEND_ENABLE_USE_FABRIC_MEM`), and
  `protocol == "ascend"` is *mandatory* rather than optional — anything else raises
  `NotImplementedError: MooncakeBackend does not support protocol …` at backend
  construction (`:28-29`).
- **The default segment size differs between the two readers**: vllm-ascend's
  `DEFAULT_GLOBAL_SEGMENT_SIZE` is **1 GiB** (`:20`) while the wheel's is 3.125 GiB,
  so a file that omits the key sizes the store by whichever loader ran — read that
  line before judging "the segment is too small/large".

## 4. Checks, cheapest first, and what each one decides

1. **Make it repeatable before changing any value.** The reported failure was random and
   became rarer as TP grew (`--tp 1` here). That is a volume/timing shape, so first
   measure: the same prompt load, N minutes, count `Failed to put key` per request. A
   per-request-random rate or a rate that scales with block size is not a static
   misconfiguration. Arithmetic you may quote with its assumptions: Qwen3-8B,
   `--block-size 128`, bf16 — `2 × 8 kv_heads × 128 head_dim × 36 layers × 2 B ≈ 147 KB`
   per token, ≈ **18.9 MB per 128-token block key**, ≈ 600 MB for the 32-key put in the
   paste.
2. **One diagnostic run with a timeout *above* the effective default.** At the pin
   `InitParams` defaults are `connect_timeout = transfer_timeout = 10000` ms
   (`include/transport/ascend_transport/ascend_direct_transport/transfer_executor_base.h:54-55`)
   and `ASCEND_TRANSFER_TIMEOUT` is parsed at `transfer_executor_base.cpp:66-85`. The
   Mooncake design doc still says **3000** for both, so the doc is stale; and the pool
   guide's recipe exports `ASCEND_TRANSFER_TIMEOUT=10000`
   (`kv_pool.md:165,170,173` with `HCCL_RDMA_TIMEOUT=17`) — **which equals the code
   default and therefore changes nothing at this pin.** Exporting the doc block is a
   no-op unless a value above 10000 is used; the discriminating run is
   `ASCEND_TRANSFER_TIMEOUT=60000` (+ `HCCL_RDMA_TIMEOUT=17`, `ASCEND_CONNECT_TIMEOUT=10000`).
   Puts then succeed → slow/oversubscribed path with an undersized budget; still failing
   at 60 s → the path to that peer is down. Verify the value took with the
   `Set transfer timeout to:` line, which prints only when the variable is set.
3. **Fabric counters on the peer that timed out first** (here `:20091`):
   `hccn_tool -i <dev> -link -g`, `-stat -g` (drop/retrans), `-optical -g`,
   `-lldp -g`, plus `npu-smi info` on that node. `GLOO_SOCKET_IFNAME=eth0` says nothing
   about the RoCE plane. The connect costs in the paste (586 ms / 74 ms / 42 ms) are
   already an unhealthy-looking spread, but they are one sample.
4. **`global_segment_size` ablation** (§3.3): `MOONCAKE_GLOBAL_SEGMENT_SIZE=2147483648`
   (`mooncake_backend.py:182-190` — env wins over the file) for one run. Also drop the
   pool entirely if the reporter can: with `kv_both` aggregated serving, removing
   `AscendStoreConnector` costs only the external cache and separates "store segment"
   from "ADXL link" in one launch.
5. **Lease TTL** on the master: `--default_kv_lease_ttl` above the observed transfer
   wait (>= 60000 here), otherwise the `revoke → OBJECT_NOT_FOUND` chain survives any
   transfer fix.
6. **Per-transfer granularity**: `--block-size 16` (default) for one run. Same total
   bytes, 8× smaller slices; if the failure rate drops, the budget/latency reading in
   step 2 is confirmed rather than the fabric.

## 5. What this case never established (say so in the answer)

- **No confirmed cause.** The reporter did not run the ablations, could not reproduce the
  failure, and closed the issue. Do not present the timeout block, the 40 GiB segment or
  the crash frame as resolved; present them as ordered checks with what each decides.
- **No master-side numbers**: `PutStart`/`PutRevoke`/`PutEnd` and the memory line for the
  failing minute were never posted, nor a `mooncake_master` log, nor `hccn_tool` output,
  nor a coredump. Those four are the artefact request for any lookalike.
- **Do not re-request the client config or the master command line.** The reporter's
  own later comment supplied both and §3 quotes them verbatim (`mooncake.json` keys,
  `mooncake_master --port 50051 --metrics_port 9003`, the Service port map); a replay
  that paraphrased §3 and then asked for "the `mooncake.json` keys and the
  `mooncake_master` command line" burned the ask on facts already in hand. Cite §3
  instead, and add "if you now run something else, say so".
- **A 10 s cadence proves the effective timeout, not the cause.** It matches the 10 000 ms
  default (and therefore the absent env var), nothing more.
- Scope of the pins: vllm-ascend `v0.20.2rc1` (`367b8e6`), vLLM `v0.18.0`
  (`bcf2be9`), Mooncake `v0.3.11.post1` (`e9c6107`, content export 933 files). The
  reporter runs vllm-ascend 0.18.0, so every line number above is a *reference* pin;
  re-grep the reporter's tree before quoting a line to them.
