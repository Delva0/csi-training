# HCCL init and link failures: keyword → EI code → cause → what to ask

Every row is a verbatim keyword from a CANN plog, the code the vendor's fault-diagnosis tree assigns
it, and the fix/ask. Page names are under `cann/hccl` `docs/zh/user_guide/fault_diagnosis/`
(English: `docs/en/user_guide/fault_diagnosis/`). Pin the clone (`HEAD f49377a5` = tag `v9.2.0` when
these were read) and re-read the page you quote: the keyword is stable, the guidance beside it grows.

| plog keyword (verbatim) | code / page | cause | lever or ask |
| --- | --- | --- | --- |
| `wait socket establish timeout, role[1] rank[N] timeout[120]` then `[LinksEstablishCompleted] is failed. ret[9]` and a `LINK_ERROR_INFO` table | EI0006 `link_timeout_EI0006.md` | parameter-plane TCP link never established: the peer never issued the connect (did not reach the op, hung, exited), unreachable network, or asymmetric TLS | find the failing pair from `LINK_ERROR_INFO` `dest_ip(user_rank)`/`src_ip(user_rank)`, then read *its* debug plog; both ends report → network; `dstRank` has another error → that error is the cause; neither → business-side op-dispatch skew. `HCCL_CONNECT_TIMEOUT` defaults to 120 s (range 120–7200, +20 s notification overhead) |
| `socket type[0|1], listen on ip[..] and specific port[16666] fail. Please check the port status and whether the port is been using by other process.` | EI0019 `param_port_bind_fail_EI0019.md` (`type[2]` = host NIC, `server_node_port_bind_fail_EI0019.md`) | the device NIC's parameter-plane port is held by another process — typically several processes on one NPU | if multiple processes per device are intended: `export HCCL_NPU_SOCKET_PORT_RANGE="auto"` (default NPU port is 16666; `auto` = OS-assigned). If not intended: fix the launch (one process per device) |
| `get host ip fail by socket Ifname. name[..] ip[..]` | EI0001 `hccl_socket_ifname_config_error_EI0001.md` | `HCCL_SOCKET_IFNAME` names a NIC that does not exist in the process's namespace (container!) | set it to a NIC listed in the same error line |
| `ra hdc message process failed ret(-67)`, `ra rdev init failed`, `[Init][RDMA]ra rdma init failed, devid[N] ip[..]`, `ra nic init rdma failed` | HCCP, no EI page in this tree | the device NIC cannot initialise RDMA at all (driver/hardware/optical/link-down), so *no* cross-server group can be created on it | `hccn_tool -i N -optical -g` (`present: not present` = no optical module), `-ip -g`, `-link_stat -g`, `npu-smi info`. This is a fabric/installation answer: nothing in vllm-ascend or in the model changes it |
| `EJ0002 … The network port is down.Suggest: Run the following hccn_tool commands …` (`-optical`, `-ip`, `-lldp`) + `Getting socket times out. Reason: 1 … some NPUs in the cluster are abnormal 2 … the collective communication operator is started too late … 3 … the communication link is disconnected` | framework message (not in the `cann/hccl` or `cann/ge` clones) | the three reasons are the EI0006 branches in one line: abnormal peer, late/absent op dispatch, or a broken/TLS-asymmetric link | run the three `hccn_tool` commands it prints; the `-optical` output alone decides "hardware" vs "dispatch skew" |
| `Task run failed, base information is streamID:[..], taskID[..], tag[<comm>]`, plus host-side `event wait timeout occurred during task execution … timeout=<N>s`, `Notify record failed, notify_id=…, ret_code=0x150059`, `AclrtRecordNotify … failed` | EI0002 `notify_wait_timeout_EI0002.md` | collective execution timeout: the two sides' post/wait sync does not match (a rank never dispatched the op, ranks dispatched different ops, or params/counts differ) | `grep -rn "Entry-HcclCommInit" run/plog \| grep "<tag>"` gives the group's ranks and their device plogs; decide full-cluster vs partial timeout; compare op/count/dtype/tag and each rank's dispatch `timeStamp`; `HCCL_EXEC_TIMEOUT` default 1836 s (AI_CPU/AICPU_CacheDisable) or 1091 s (AIV). `HCCL_ENTRY_LOG_ENABLE=1` dumps every dispatch with `streamId`/`comm` — the tool for "same comm, two streams" concurrency bugs, which the page names as a real cause of timeouts and wrong results |
| `fault kernel_name=aiv_all_reduce_***` (A2) | `aiv_comm_op_exec_fail.md` | an HCCL operator executed *as a kernel* under `HCCL_OP_EXPANSION_MODE=AIV` failed | follow EI0002's task-exception path; and check the AIV limits in `env-support-matrix.md` before blaming the model |
| `InitGroupStage][RanktableDetect` , `topo detect timeout`, `partial rank not connect to server` | EI0015 family (`topo_detect_timeout_EI0015.md`, `typical_multi_node_comm_init_fail_EI0015.md`, `partial_rank_not_connect_to_server_EI0015.md`) | cluster negotiation: the root node lists the ranks that did connect; the missing ones are the question | root node's log names the unconnected rank range; the common cause is a wrong/absent `HCCL_SOCKET_IFNAME` on those nodes |

## Worked case: vllm-ascend #1955 (`Getting socket times out`, torchair graph)

Report (`reported`): 910B2C ×16 on one node, image `v0.9.2rc1`, DS-R1-W8A8, `DP4 × TP4`,
`--enable-expert-parallel`, `--additional-config '{"torchair_graph_config":{"enabled":true}}'`, launch
script exporting `HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=4096 … (then again
1024) HCCL_IF_IP=127.0.0.1 ASCEND_LAUNCH_BLOCKING=1`. Failure at graph load:
`EJ0002 … The network port is down`, `Load graph failed, error code:1343225857`, and the plog has,
in this order:

```text
[ERROR] HCCP … [ra_hdc_rdma.c:1199] ra hdc message process failed ret(-67) phy_id(12)
[ERROR] HCCL … [network_manager.cc:639] [Init][RDMA] … ra rdma init failed, devid[12] ip[192.168.20.160]
[ERROR] HCCL … [hccl_socket.cc:100] socket type[1], listen on ip[192.168.20.160] and specific port[16666] fail …
[ERROR] HCCL … [hccl_socket_manager.cc:778] [Wait][LinkEstablish]wait socket establish timeout, role[1] rank[12] timeout[120]
[ERROR] HCCL … [hccl_socket_manager.cc:553] |  192.168.20.161(13)   |  16666  |   192.168.20.160(12)   |  0  |  client  | time out |
[ERROR] HCCL … [comm_factory.cc:221] tag[CreatecomResource_group_name_53_MC2MultiServer], commPlane[9], commType[14]
[ERROR] GE   … LoadGraph: … error code:1343225857
```

and the reporter's own `hccn_tool -i 0..7 -optical -g` prints `present: not present` on all eight.
Read that block as one chain — each line only establishes its own step:

| line (fragment) | what it establishes |
| --- | --- |
| `ra_hdc_rdma.c:1199 … ret(-67) phy_id(12)` | the RDMA (RoCE) device for phy id 12 never initialised |
| `network_manager.cc:639 ra rdma init failed, devid[12] ip[192.168.20.160], return[19]` | same failure, with the **device** IP: that NPU's transport is absent (`19` is the layer's own code) |
| `hccl_socket.cc:100 listen on ip[192.168.20.160] … port[16666] fail` | consequence: no listener on the device plane |
| `hccl_socket_manager.cc:778 wait socket establish timeout, role[1] rank[12] timeout[120]` + the `.161/.162/.163:16666` table | consequence: the peers' connects time out — what the Python log calls `Getting socket times out` |
| `comm_factory.cc:221 CreateCommPlane failed, tag[…_MC2MultiServer], commPlane[9], commType[14]` | the object that needed the transport: the MoE/MC2 op's multi-server graph comm resource |
| `RuntimeError: EJ0002 … The network port is down` then `Load graph failed, error code:1343225857` | the Python-side surface of the same chain (torchair backend, `_backend.py:138`) |

- **Settled (`verified` as doc facts, `reported` as log facts).** The failing communicator is the
  EP-derived MC2 group spanning *servers* (`MC2MultiServer`), i.e. HCCL chose the cross-server plane
  for it. `HCCL_INTRA_PCIE_ENABLE`/`HCCL_INTRA_ROCE_ENABLE` only select the **intra-server** link
  (`env-support-matrix.md`), so the reporter's assumption "PCIe enabled ⇒ no RoCE needed" is wrong
  for a cross-server group. With no optical modules the device NIC cannot init RDMA, which is why the
  listen on the device IP and the links to `.161–.163` never complete.
- **Two candidate owners for the listen line, and the discriminator.** EI0019 (port 16666 already
  bound — e.g. two processes on one device; the paste has two pids under `rank=1`) versus the HCCP
  RDMA init failure above it. Ask for `npu-smi info`'s process table per chip and `HCCL_NPU_SOCKET_PORT_RANGE=auto`
  as the cheap EI0019 test; if the NIC has no optics, no port setting helps.
- **Workarounds to relay.** (1) Fix the fabric: populated opticals/device IPs and a real host IP in
  `HCCL_IF_IP` (`127.0.0.1` is not a usable NCCL/HCCL host interface in a multi-process launch);
  then re-test. (2) Keep the EP/MC2 group inside one module: drop `--enable-expert-parallel` (the
  lever that removes the `MC2MultiServer` group entirely — `../symptoms/moe-ep-dispatch.md`), or
  reduce DP/TP so the group stays intra-module. Cost: fewer cards or no MC2 A2A.
- **Hypothesis, with its falsifier.** If the 16 devices are a Box16 in PCIe intra-server mode, the
  vendor constraint requires the two modules be used in matched pairs (device 0 with 8, 1 with 9 …);
  a DP4×TP4 EP group may not map onto matched pairs, which would explain HCCL falling back to the
  device NIC. Falsifier: the same script restricted to one module (8 devices) — if it starts, the
  layout is the gate, not the model.
- **Never spend the reply on** how many chips the "B2C*16p" box has, or on `ASCEND_LAUNCH_BLOCKING`
  arithmetic. `ASCEND_LAUNCH_BLOCKING=1` is a plausible cause of the "operator started too late"
  branch of `Getting socket times out` (it serialises every step), so it is worth one sentence as a
  thing to remove while retesting — labelled `hypothesis`, not derived.
- The thread itself says only "It looks like a network card issue." and was closed because TorchAir
  was deprecated: workaround `reported`, mechanism from the plog.

## No forward fix: TorchAir left the tree at `v0.12.0rc1` (`verified`)

A graph-mode report on `0.9.x`/`0.10.x` with `torchair_graph_config` enabled has **no release to be fixed
in**, so the durable answer is the ACL-graph migration and "the maintainers closed it as deprecated" is
`reported` status, not a defect finding.

- `v0.9.2rc1` reads the key in `vllm_ascend/ascend_config.py:39` (`additional_config.get(
  "torchair_graph_config", …)`, class at `:57-66`) and carries `attention/attention_v1_torchair.py`;
  `git ls-tree -r --name-only v0.11.0rc0 -- vllm_ascend | grep -c torchair` → 24, `v0.12.0rc1` → 0
  files.
- The release notes say it twice (`docs/source/user_guide/release_notes.md`, read at `v0.29.0.rc1`):
  `:1503` "Torchair graph has been deprecated, it'll be removed in v0.12.0rc1" and `:1547` "Torchair graph
  mode is removed. `--additional-config {"torchair_graph_config": {"enabled": true}}` doesn't work anymore.
  Please use aclgraph instead."
- Lever: upgrade off TorchAir — on `>= v0.12.0rc1` the key is not read at all, so the deployment must run
  aclgraph. A `torchair_graph_config` graph-load failure disappears with the feature; there is nothing to
  patch on `<= v0.11`.

## Check before reuse

```bash
H=/workspace/repos/cann/hccl
git -C $H log -1 --format='%h %s'
git -C $H show HEAD:docs/zh/user_guide/fault_diagnosis/README.md | sed -n '1,60p'   # code -> stage page
git -C $H grep -n "wait socket establish timeout" -- docs/                        # EI0006 keyword
git -C $H grep -n "Please check the port status" -- docs/                          # EI0019 keyword
git -C $H grep -n "Task run failed" -- docs/                                       # EI0002 keyword
R=/workspace/repos/vllm-project/vllm-ascend
git -C $R ls-tree -r --name-only <tag> -- vllm_ascend | grep -ci torchair      # 0 from v0.12.0rc1
git -C $R show <tag>:vllm_ascend/platform.py | nl -ba | sed -n '133,150p'      # ep_size = world_across_dp
git -C $R show <tag>:docs/source/user_guide/release_notes.md | grep -n -i 'torchair.*remov\|remov.*torchair'
# the report's own numbers, before any code reading:
#   grep -n "LINK_ERROR_INFO\|socket establish timeout\|listen on ip\|Task run failed\|task_exception" plog*.log
```
