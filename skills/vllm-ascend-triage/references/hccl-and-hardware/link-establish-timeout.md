# `[Wait][LinksEstablishCompleted] is failed. ret[9]` — the socket link never came up

## Scope

A **multi-node** start (two Atlas 800 nodes, a DP/EP or PD pair) whose visible error
names a collective or a device op, while the plog's first in-window block is HCCL's
*socket link establishment*:

```text
[ERROR] HCCL(55694,):[hccl_socket_manager.cc:860] [60446][Wait][LinksEstablishCompleted] is failed. ret[9].
[ERROR] HCCL(55694,):[hccl_socket_manager.cc:399] [60446][Create][Sockets]Wait links establish completed failed, local role is client. ret[9]
[ERROR] HCCL(55694,):[hccl_socket_manager.cc:622] [60446]  _________________LINK_ERROR_INFO____________________
[ERROR] HCCL(55694,):[hccl_socket_manager.cc:623] [60446]  |  comm error, device[7]
[ERROR] HCCL(55694,):[hccl_socket_manager.cc:625] [60446]  | dest_ip(user_rank) | dest_port | src_ip(user_rank) | src_port | MyRole |    Status    | TlsStatus |
[ERROR] HCCL(55694,):[hccl_socket_manager.cc:582] [60446]  | <ipA>(0)           |  16666    | <ipB>(15)         |    0     | client |   time out   | UNKNOWN   |
[ERROR] HCCL(55694,):[hccl_socket_manager.cc:582] [60446]  | <ipC>(1)           |  16666    | <ipB>(15)         |    0     | client |  connecting  | UNKNOWN   |
...                                  (one row per peer in the transport plan)
[ERROR] HCCL(55694,):[comm_base.cc:454] [60446][Create][InterLinks] socket manager create connections failed, ret[9]
[ERROR] HCCL(55694,):[comm_base.cc:174] [60446][CreateLinks]call trace: hcclRet -> 9
```

Line numbers move between releases (#1955's plog carries the same sentences at
`hccl_socket_manager.cc:830` / `:393` / `:591` and the row printer at `:553`), so grep
the message text, not the line.

`9` is `HCCL_E_TIMEOUT` (enum at `cann/hccl/docs/api/context/HcclResult.md`, quoted in
`index.md`): the local device asked each peer to establish its control socket and
none completed inside the window. Nothing was launched, no collective ran, and no
budget was involved. Recorded #9872 (v0.19.1rc1, 2×910B2, MiniMax-M2.5,
`--data-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel`) opened as
`[Bug] … aclnnMoeDistributeDispatchV4 failed, error code is 561000`; this block was
what the plog carried for `device[7]`, and the maintainer's reply was that the aclnn
error "**不一定是根因** — check the two nodes' connectivity and send the full-card
plog from both machines". The thread's later public comments, in order: the reporter
uploaded `log0.zip` / `log1.zip` the same day — per node a network-interconnect test
log, the inference startup log and the plog — so the block above came out of *those
zips*, not out of the issue body; the maintainer then quoted the block back and asked
for a connectivity check; and the reporter answered with a **card-to-card `ping`
sweep** (every die of one node reaching every die of the other, with a screenshot and
"上述日志中有记录"). That is the reply this table provokes and it does not answer it
(next section). The fix is still not established, so what *is* established is which
artifact and which line decide it — and that a `ping` answer needs one explicit re-ask
naming the port, not another round of "check connectivity".

## Why the surface error is not the owner

The same terminal text arrives from both lanes, so the *chain* is the evidence:

```text
... [Create][ExchangerNetwork] sync create connections Failed, ret[9]   (comm_base.cc)
... [Create][CommPlane] failed, tag[CreatecomResource_group_name_53_MC2MultiServer]
... [CreateComm] hcclRet -> 4                                          (hccl_impl.cc:607)
... [HcclAllocComResource] hcclRet -> 4                                (op_base.cc:3102)
... Failed to create ComResource by tiling                             (hcom_graph_mc2.cc:105)
GE: LoadGraph failed, error code:1343225857 / InitTaskInfoV2 failed
```

That full chain is recorded (#1955, single 16×910B2C node, vllm-ascend 0.9.2rc1 +
torchair, DP4×TP4+EP; its launcher set `HCCL_IF_IP=127.0.0.1` next to
`nic_name="eth0"`), and it is why `ret = 4` must be read as a wrapper:

- `hcclRet -> 4` is `HCCL_E_INTERNAL`, and above it sits the real code `ret[9]`
  (`HCCL_E_TIMEOUT`). **A `HcclAllocComResourceByTiling … ret = 4` / `561000` paste is
  not evidence for the QP/budget lane** — the resource request cannot succeed while
  the comm it is being created for has no link.
- The server half of the same failure prints as
  `[hccl_socket.cc:100] [HcclSocket][Listen] socket type[1], listen on ip[<ip>] and
  specific port[16666] fail. Please check the port status and whether the port is been
  using by other process.` (#1955) — a *local* bind failure on the port the peer is
  dialling. One node's plog therefore shows `Listen … fail`, the other's shows
  `LinksEstablishCompleted … ret[9]`: **ask for both nodes'
  plogs, and for every card** (`~/ascend/log/debug/plog/plog-<pid>_*.log`), because the
  failure is per local device (`comm error, device[N]`) and the printing rank need not
  be the one that could not listen. This is the *two-node* reading only. A `Listen …
  port[16666] fail` on a **single** host, with no `ret[9]` and no `LINK_ERROR_INFO` and
  the second of two engines dying (`Bind_Failed(EJ0003)`, `hcclCommInitRootInfoConfig …
  error code is 7`), is not a peer problem: both engines were given the same cards —
  `../pd-disaggregation/same-host-pd-device-partition.md` (#4725; its plog line is
  `hccl_socket.cc:106`, `socket type[0]`).

## Reading the LINK_ERROR_INFO table

Columns are `dest_ip(user_rank) | dest_port | src_ip(user_rank) | src_port | MyRole |
Status | TlsStatus`; rows are the peers this device had to reach.

- `src_port 0` and `Status: connecting` mean the connect never progressed (nothing was
  assigned on the local side); `time out` marks the row that consumed the window. In
  #9872 the header said `comm error, device[7]` and its `src_ip` columns carried
  user_rank 15 (the node's last rank), which listed **all eight** remote ranks on port
  16666 with one `time out` (user_rank 0 of the peer) and the rest `connecting` — the
  whole local link set, not one bad pair.
- `device[N]` in the header is the socket manager's **logic** device id
  (`deviceLogicId_`, constructor `hccl_socket_manager.cc:20`, printed at `:623` beside
  the separate `devicePhyId_`), i.e. the index inside that process's own visible set —
  map it through `ASCEND_RT_VISIBLE_DEVICES` before matching it to an `npu-smi` row or
  a `hccn_tool -i N`, or the plog you pull is the wrong card's.
- All peers in flight at once is the *wholesale* shape: the peer's socket service is
  not reachable on that port (bind, NIC choice, firewall, or the peer process never
  got that far). A **subset** of peers failing while the rest are up is the other
  page (`transport-destlink-ipc-import.md`) and suggests a card/cable or driver
  pairing.
- **The table and the reason list are one printer, and the list is canned text.** Both
  come from `HcclSocketManager::PrintErrorConnection` /
  `PrintSocketsInfo` in `cann/hcomm/src/algorithm/impl/resource_manager/hccl_socket_manager.cc`
  (in the export — grep there instead of theorising; the `" connecting"` / `"time out"`
  strings and `sRole = " client "` are literals). The seven-item explanation under it is
  a static template, so it lists the reachable causes without saying which one you have:
  the peer did not connect (some NPU abnormal, or that op started late), the link is down
  (IPs on different subnets, TLS mismatch), `the time difference between the execution of
  hcom on this device and the target device exceeds the timeout threshold … [Entry-]`,
  inconsistent graph execution between the nodes, inconsistent or expired TLS
  certificate, **"If src and dst IP address can be pinged, check whether the device IP
  address conflicts"**, and an invalid chosen port. Treat it as a checklist, never as the
  finding.
- **The window is `HCCL_CONNECT_TIMEOUT`.** `WaitLinkEstablish` waits
  `timeout > 0 ? timeout : GetExternalInputHcclLinkTimeOut()`, and that getter returns
  `g_externalInput.linkTimeOut`, parsed from `HCCL_CONNECT_TIMEOUT` with the default
  `HCCL_LINK_TIME_OUT_S` (`cann/hcomm/src/platform/common/externalinput.cc:560-585`). A
  launcher that exports `HCCL_CONNECT_TIMEOUT=7200` (such scripts exist) turns the 120 s
  report into a two-hour one, so read the variable before reading a long wall-clock gap
  as a hang. `ASCEND_CONNECT_TIMEOUT` / `ASCEND_TRANSFER_TIMEOUT` appear in the same
  launchers but do not feed this getter.

## A `ping` between the cards is not the check

The reporter's strongest form — all 8 dies of one node pinging all 8 of the other — is
still ICMP, while the failure above is a **TCP** control socket that never completed.
Two mechanisms make the difference usable, both read from the `cann/hcomm` export
(newer than the plog's own release: the sentences match, the line numbers do not, so
use it as an ABI-level fact the way `scripts/cann-error-lookup.sh` uses the code
tables):

- **The port is one constant, not a per-rank offset.** `GetNicPort`
  (`cann/hcomm/src/platform/common/sal.cc:646-661`) returns the rank table's own
  `deviceInfo.port` when it carries one; otherwise, on the **device-NIC** path and a
  non-310P SoC, `HETEROG_CCL_PORT` = **16666** (`pub_inc/hccl_common.h:166`,
  "通信默认端口"), while the host-NIC paths use `HOST_PARA_BASE_PORT + devicePhyId`
  (60008 + phy id, `:171`) or `HCCL_IF_BASE_PORT + HCCL_AISERVER_DEVICE_NUM (8) +
  phy id` (same function). So the identical 16666 on every peer row is expected — and
  it is also a **plane** statement: that number means the link went over the *device*
  NIC, which is what `hccn_tool -i N -ip/-optical/-lldp -g` and device-IP conflicts
  cover. On the 16666 form the peer address in the table is therefore a *device* IP, and a
  host-NIC variable mismatch does not explain this table by itself (the variables still
  decide other comm domains — item 3 below — so do not skip them). A launcher that
  exports `HCCL_IF_BASE_PORT` moves the number to base+8+phy id, so read `dest_port`
  from the table instead of assuming 16666.
- **The dialling side names the port it used, with no arithmetic.**
  `ss -tnp state syn-sent` on the failing node (iproute2; `netstat -anp | grep
  SYN_SENT` is the fallback) lists the connects still waiting for a SYN-ACK with their
  real destination ip:port: rows to `…:16666` = the peer is not answering on that
  port; rows to a *different* port = the two nodes disagree on the base;
  **no** rows while the table says `connecting` and the launch is still inside its
  connect window = the local side never reached `connect()` (port or NIC choice). Pair
  it with `ss -lntp | grep ':16666'` on the **peer**, during the window — the listener
  exists only while that rank's process is up, and its absence is the `Listen … fail`
  half of this fault.

## Checks, cheapest first

1. **Both nodes, all cards, the failing minute.** The plog block above plus the peer's
   `Listen … port[N] fail`. No plog ⇒ no answer: the same top-level `561000` text also
   comes from the budget lanes, so say which block you saw and ask for this one. **If
   the thread already carries them, do not ask again**: #9872's reporter delivered both
   nodes' zips on the day of the first request, and that thread's second week went to a
   `ping` reply instead of a port reply. In that state the next ask is item 2 plus the
   per-card env, nothing else.
2. **The port on both nodes.** Take the `dest_port` from the table (16666 in every
   recorded case, and 16666 is the device-NIC default — see the section above) and run
   `ss -lntp | grep ':16666'` on each node while it is failing, plus
   `ss -tnp state syn-sent` on the failing one; compare with `HCCL_IF_BASE_PORT` /
   `HCCL_HOST_PORT` if the launcher sets them. A port held by a stale process, or a
   listener that never came up, reproduces the two halves of this fault.
3. **The address variables, node by node.** `HCCL_IF_IP`, `HCCL_SOCKET_IFNAME`,
   `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` must name the same reachable NIC on both
   nodes. `HCCL_IF_IP=127.0.0.1` or `lo` (recorded launcher), a NIC name in an
   `_IF_IP` variable, or two different subnets breaks exactly this step. The
   `collect_env.py` ENV block **cannot** show them (its filter keeps only
   `TORCH*/PYTORCH*/ASCEND_*/ATB_*` and registered `vllm.envs` keys —
   `../startup-failures/environment-correspondence.md`), so ask for
   `env | grep -E 'HCCL|SOCKET_IFNAME'` per node instead of reading their absence as
   a cause.
4. **The device plane, not `ping`.** `for i in $(seq 0 7); do hccn_tool -i $i -ip -g;
   hccn_tool -i $i -optical -g; hccn_tool -i $i -lldp -g; done` — the same three
   commands the `EJ0002 The network port is down` text emits. HCCL's socket link is
   bound to the device IP on the chosen NIC, so host `ping` success between the nodes'
   data NICs is necessary and not sufficient (reason 6 above says so explicitly).
5. **Symmetry of the launch.** Compare both serve commands: rank counts, the DP
   address/rpc port, `HCCL_INTRA_*`, TLS settings, and *start order* (a node that
   takes minutes to load weights while the other waits out its connect timeout shows
   the same table with `connecting` everywhere). `rank[12] timeout[120]`-style lines
   date the wait.

## Not this page

- `EI0007 … Failed to allocate resource[qp] … Memory resources are exhausted` with no
  socket-manager block before it → the local comm-resource refusal of
  `qp-resource-exhausted-at-graph-load.md` (capture phase) or
  `mc2-dispatch-profile-run-hang.md` §Step 3b (MC2 dispatch in `profile_run`).
- `[Create][DestLink]Create Dest error!` + `halShmemOpenHandle … 507899` /
  `rtsIpcMemImportByKey` on a *subset* of edges → `transport-destlink-ipc-import.md`.
- `hcclCommInitRootInfoConfig … error code is N` at group creation →
  `comm-create-errors.md`.
- `EI0019 … port <n> have already been bound` with a port above 65535 /
  control-port arithmetic → `control-plane-port-binding.md`. `Bind_Failed(EJ0003)` on
  the normal port `16666` is the other family → `comm-create-errors.md` §code 7.
