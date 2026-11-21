# LLMDataDist P/D that never comes up: rank table, visible devices, RPC port

## Scope

`LLMDataDistCMgrConnector` (LLM-DataDist) P/D whose report is *"according to the P/D
example I wrote the parameters and ran the vllm instances, but in the end it failed to
run"* / 起不来 — typically a **hand-written `ranktable.json`**, one host with 2+ cards
(or one k8s pod), and **no log at all**. Verified at vllm-ascend **v0.10.0rc1
`4604882`** (SOURCES `2379`; upstream vllm v0.10.0 `6d8d0a24`); re-grep the anchors on
another pin — every line number here moved between v0.9.1 and v0.10.0rc1.

Not this page: the deployment serves and only the 120 s done-signal / TTFT is wrong
(`llmdatadist-done-signal-and-ttft.md`), a *serving* deployment whose connect step
fails (`transport-topology.md`), the pre-deployment A3 plane question
(`plane-selection-a3-parameter-plane.md`), and the second engine's HCCL port collision
`Bind_Failed(EJ0003) … error code is 7` (`same-host-pd-device-partition.md`). Nothing
here needs `kv_p2p/`, the Mooncake export or the transport plane: on this shape the
table arithmetic decides before any byte moves.

## Procedure

1. **State what the connector consumes.** `read_offline_rank_table`
   (`vllm_ascend/distributed/llmdatadist_c_mgr_connector.py:428-446`) reads **exactly
   two keys**, `decode_device_list` and `prefill_device_list`, and only each entry's
   `server_id` / `device_id`; `read_agent_metadata` adds `device_ip`, `cluster_id`
   (`int()`, `:483`) and the optional A3 `super_pod_id` / `super_device_id`. The
   generator's own top-level `version` / `server_count` / `status` are **never
   validated**. So a table modelled on the HCCL rank table (`server_list` / `device`)
   fails with `KeyError: 'decode_device_list'` — when the reporter says hand-written,
   that is the first thing to ask for, not a connector read.
2. **`server_id` must be this process's own `get_ip()`.** `self.local_ip = get_ip()`
   (`:177`, `:308`) and the filter keeps `d["server_id"] == self.local_ip and
   device_filter(d["device_id"])` (`:462-476`); a miss is fatal at `:495`:
   `Can't read the target server_id {ip} and device_rank {rank} from rank table`.
   `get_ip()` is the UDP-connect outbound address (see `transport-topology.md`), i.e.
   the **Pod IP** under k8s (cilium/vxlan included) — not the node IP, not the
   `hccn_tool` device IP. One command in the same container decides it:
   `python -c "from vllm.utils import get_ip; print(get_ip())"`, diffed against every
   `server_id`.
3. **Two engines in one pod land on one chip unless each process gets its own
   `ASCEND_RT_VISIBLE_DEVICES`.** The table supplies metadata only; the device the
   engine opens is `llm_config.device_id = self.local_rank` (`:416`), while the table
   filter reads **only** `ASCEND_RT_VISIBLE_DEVICES` (`:450-460`) and is a no-op when
   that is unset. `ASCEND_VISIBLE_DEVICES` — what an Ascend device plugin exports, e.g.
   `=5,2` in a collected env dump — is not that variable, so both engines take logical
   device 0 while the peer is told the other card's `device_ip`. Check: `npu-smi info`
   while both run; two EngineCore PIDs on one chip is the smoking gun.
4. **One `VLLM_LLMDD_RPC_PORT` per engine per host.** Default `5557`
   (`envs.py:147-148`); the advertised port is `dp_rank_local*tp_size + RPC_PORT`
   (`:184`) and the worker listener is `RPC_PORT + local_dp_rank*tp_size + tp_rank`
   (`:347`), bound as `tcp://0.0.0.0:<port>`. Two processes with the default collide.
   The pin's own same-host 1P1D recipe is the control to diff against:
   `ASCEND_RT_VISIBLE_DEVICES=0 VLLM_LLMDD_RPC_PORT=5559` (P) /
   `ASCEND_RT_VISIBLE_DEVICES=1 VLLM_LLMDD_RPC_PORT=6000` (D) in
   `tests/e2e/pd_disaggreate/run_edge_case_test.sh:73,:93`, generated with
   `--prefill-device-cnt 1 --decode-device-cnt 1` (`:27`).
5. **Regenerate rather than hand-write.** `examples/disaggregated_prefill_v1/gen_ranktable.sh
   --ips <pod ip> --npus-per-node 2 --network-card-name <iface> --prefill-device-cnt 1
   --decode-device-cnt 1`. `gen_ranktable.py:50-88` enumerates cards from `npu-smi info
   -l`, takes `device_ip` from `hccn_tool -i <device_id> -ip -g` (A2) or `-vnic -g`
   (A3), numbers `cluster_id` 1..N (`:98-101`) and writes the two lists (`:104-116`).
   Consequence for a hand-made table: its `device_id` values must be ids `hccn_tool`
   accepts **inside that container**, not the physical card numbers the node's
   `npu-smi` prints.
6. **Confirm the link initialized, then stop looking at the table**: `Done initialize
   llm_datadist in rank …, local rank …, cluster id …` (`:424-426`) with a *different*
   `cluster_id` per engine (each `cluster_id` is also the `BlocksCacheKey` cluster, so
   it must be globally unique across P and D).
7. Only then transport: pod-to-pod reachability on the advertised port, and the two
   `device_ip`s mutually pingable. Cilium/vxlan carries the Pod network, not RoCE, and
   `hostNetwork` is not the first instrument when the table itself was wrong.

## Pin's own recipe defects (report them separately)

At `4604882` the example cannot be followed verbatim: `README.md:212` and
`run_edge_case_test.sh:117` invoke `examples/disaggregate_prefill_v1/toy_proxy_server.py`,
which is **not in the tree** (the directory ships `load_balance_proxy_server_example.py`);
and `run_server.sh:19` carries `---max-num-batched-tokens` (three dashes), which argparse
rejects. Both are documentation/build defects, so "I followed the example and it did not
run" can be true before any connector question.

## What to ask for

Both `vllm serve` commands, the `ranktable.json` itself, which side died plus the first
~40 lines of each **worker** (EngineCore) log, `npu-smi info` taken while both run,
`hccn_tool -i <id> -ip -g` per card used, and the proxy invocation.

## Recorded case

#2379 (v0.10.0rc1, 2×910B3 inside a k8s pod, `ASCEND_VISIBLE_DEVICES=5,2`, table
generated by hand, "failed to run", no log). The ranked answer was built from the anchors
above; the report carries no log and the snapshot carries no resolution, so **the cause is
open** — steps 2-4 as three checks, with the ladder above, are the deliverable.
