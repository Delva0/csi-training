# LLMDataDist P/D that links but never registers: `Checking query_register_mem_status again` forever

## Scope

`LLMDataDistCMgrConnector` (LLM-DataDist) PD that **starts, receives a request, and then
loops** instead of transferring: the decode worker prints the metadata handshake and

```text
INFO llmdatadist_c_mgr_connector.py:708 … try link with remote, comm id: pd_comm_<P ip>_<D ip>
INFO llmdatadist_c_mgr_connector.py:711 rank table  {'version': '1.2', 'server_count': '2', …}
INFO llmdatadist_c_mgr_connector.py:712 comm name: pd_comm_…
INFO llmdatadist_c_mgr_connector.py:713 cluster rank info: {1: 0, 2: 1}
INFO llmdatadist_c_mgr_connector.py:728 Checking query_register_mem_status again   × forever
```

while the upstream scheduler repeats `cmpl-… is still in WAITING_FOR_REMOTE_KVS state`.
Verified at vllm-ascend **v0.10.0rc1 `4604882`** (#2498, CANN 8.2.RC1, driver
`npu-smi 24.1.rc2`, one 910B3 per container on **one** host, container networking).
Not this page: the connector that never gets this far (`llmdatadist-ranktable-startup.md`),
the deployment that serves and only the 120 s done-signal / TTFT is wrong
(`llmdatadist-done-signal-and-ttft.md`), and the HIXL/Mooncake transport-identity family
(`transport-topology.md`, `mooncake-adxl-connect-103900.md`).

## What the pin's own code fixes (source-verified, `4604882`)

- `add_remote_agent` (`:610-744`) **builds the rank table itself**: `is_same_server =
  remote_metadata.server_id == self.local_agent_metadata.server_id` (`:619`),
  `rank_table["server_count"] = "1" if is_same_server else "2"` (`:634`),
  `comm_name = f"pd_comm_{prefill_metadata.device_ip}_{decode_metadata.device_ip}"` (`:627`),
  `cluster_rank_info = {prefill.cluster_id: 0, decode.cluster_id: 1}` (`:628-631`).
  The reporter's `ranktable.json` supplies only `server_id` / `device_id` / `device_ip` /
  `cluster_id`; every structural field above is derived, so **read the printed table, not
  the reporter's file**.
- `server_id` is `get_ip()` (`:177`, `:308`) — the process's outbound address: a **Pod IP**
  under a bridge/CNI netns, the **host IP** under `--net=host` / `hostNetwork: true`.
  Two engines in two containers on one machine therefore look like **two HCCL servers**.
- After `link()` returns a `comm_id` the worker polls `query_register_mem_status` in a
  `while True` with `time.sleep(1)` (`:716-728`); the only exits are `RegisterMemStatus.OK`
  (→ `Linking success`) and `FAILED` (→ `RuntimeError`). **There is no timeout**, so any
  third status is an infinite loop with no attribution at all — the absence of an exception
  is not evidence that the link is healthy.
- `llm_datadist` is **not** in the snapshot (the `Ascend/mind-cluster` export ships only
  `build/` and `component/`, `rg` for `query_register_mem_status|RegisterMemStatus` is
  empty), so what the enum's non-terminal values mean cannot be read locally. Do not spend
  calls on it.

## Ranked causes, one check each

1. **Two engines on ONE machine, each in its own container on the container network**
   (reporter-confirmed cause of #2498). Their `get_ip()`s differ → `is_same_server` false →
   `server_count: "2"` → the table declares two HCCL servers, so the datapath the link needs
   is **cross-server RoCE over the per-device `device_ip`** (`hccn.conf` / `hccn_tool`), from
   inside a container net namespace — which that CANN build does not complete. Fix and
   expected observable: **relaunch both containers with `--net=host`** (k8s
   `hostNetwork: true`); both sides then print the same `server_id`, the table's
   `server_count` becomes `"1"`, and the loop turns into `Linking success`. Reporter's own
   words (#2498, 2025-08-23): *"I got it working successfully in the container using the
   host network. I believe this is a limitation of CANN"*, citing the CANN 8.2.RC1 env
   reference `…/canncommercial/82RC1/maintenref/envvar/envref_07_0073.html`. The recorded
   deployment was **one host**; do not require two machines before proposing it.
2. **The two sides' three printed values must be identical.** That is the maintainer's own
   first reply on #2498: compare `ranktable`, `cluster_rank_info` and `comm name` as printed
   by the prefill node and the decode node — they "should print exactly the same". All three
   are `logger.info` (`:711-713`), so no DEBUG re-run is needed. A difference (a hand-written
   `device_id` / `device_ip` / `cluster_id` on one side, a TP/DP or
   `VLLM_LLMDD_RPC_PORT` asymmetry, two different builds) is a configuration cause and is
   fixed there, not here.
3. **Only if 1 and 2 are clean, and the peers are on different machines**, the owner is the
   NPU NIC path itself: the same maintainer reply escalates to *"contact the after-sell
   team"* when the configuration matches. Ask for `hccn_tool -i N -link -g` and `-i N -ip -g`
   **inside each container** (the address the container can use is the host-configured one)
   and the transport lines of the plog.

## What not to spend the session on (the recorded detour)

- **The `device_id` namespace of a hand-written `ranktable.json`** (`llmdatadist-ranktable-startup.md`
  step 5) is a real trap for a table the reporter typed, but it does not decide this shape:
  the table in the log is generated by the connector, and rung 1 above removes the cross-server
  path entirely.
- **`llm_datadist` sources / the `RegisterMemStatus` enum** — absent from every export (above).
- **The history corpus.** A body search for `query_register_mem_status` /
  `WAITING_FOR_REMOTE_KVS` over `/workspace/history` returns 0 hits, and the titles-only
  search cannot carry log tokens.
- **Which process the paste is.** `get_num_new_matched_tokens … do_remote_prefill: True`,
  `update states num_externel_tokens`, `Start to transmit` and `connect_to_remote_agent`
  place it on the **decode** side; a pasted `kv_role: kv_producer` command is the other node.
  Ask for both `vllm serve` lines in one line of text instead of deriving the role for turns.
- **Re-deriving the port formulas.** `remote_port: 5579` with the reporter's exported
  `VLLM_LLMDD_RPC_PORT=5579` is consistent (default `5557`, `envs.py:147-148`), and the
  metadata reply arrived over that socket, so the control plane is already proven working.

## Delivery for this shape

Message 1: the loop is this connector's own **no-timeout wait** and it proves the handshake
plus the *local* memory registration worked, so the request state line
(`WAITING_FOR_REMOTE_KVS`) is a consequence, not the fault. Message 1 also carries the
question that splits rungs 1-3: are both engines on one machine, and is either container
running on the container network (`--net=host` absent)? Then the fix (`--net=host`, with
`server_count "1"` as the observable) and, for the cross-machine case, the two
`hccn_tool` reads. Say which of the three the paste already excludes.

## Recorded case

#2498, v0.10.0rc1, two bridge-network containers each with one 910B3 on one host, 1P1D on a
hand-written `ranktable.json` (`cluster_id` 1/2, `device_ip` from `hccn.conf`), the reporter's
own fix `--net=host`. The first recorded session on this issue (this minibatch) shipped **no
answer at all**: 11 assistant turns, ~54 K characters of reasoning against 249 characters of
message text, `output_truncated` — the "launching two containers … using container networking"
sentence was in the issue body from the first read, and the session ended inside a reasoning
block weighing the `device_id` hypothesis after dismissing the netns one
(`../evidence-workspace/delivery-discipline.md`).
