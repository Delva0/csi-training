# The comm layer: HCCL/HCCP failures and what its env vars actually do

## Scope

Reports whose decisive evidence is a CANN **communication-layer** line rather than a Python frame:
`Getting socket times out`, `EJ0002 … The network port is down`, `wait socket establish timeout`,
`socket type[N], listen on ip[..] and specific port[16666] fail`, `ra rdma init failed` / `ra nic init
rdma failed`, `Task run failed … TaskExecStage`, `Notify record failed`, `fault kernel_name=aiv_*`.
The vllm-ascend half is only *which* communicator the plugin asked for and *why* it spans servers;
the failing allocator, socket and NIC are CANN's.

Not this page: a collective that completes but returns wrong numbers (`../symptoms/flashcomm1-and-moe-reduce.md`),
a mid-run silent freeze with `status[STUCK]` (`../symptoms/pd-collective-hang.md`), the MoE dispatch
op's own resource error (`../symptoms/moe-ep-dispatch.md` §A), or a startup stall with no HCCL line at
all (`../symptoms/startup-and-stall.md`).

## Common rules

- **Route by stage before reading anything else.** HCCL reports one code family per stage:
  communicator init (EI0001 env, EI0004 rank table, EI0014, EI0015, EI0016 TLS, EI0019 server port) →
  parameter-plane link (EI0019 param port, EI0011 QP memory, EI0006 link timeout, EI0005 param
  consistency) → task execution (EI0002 notify wait, EI0012 SDMA, EI0013 error CQE). The vendor's own
  index is a file in a clone: `cann/hccl` `docs/zh/user_guide/fault_diagnosis/README.md` (English
  stage pages beside it: `docs/en/user_guide/fault_diagnosis/param_link_stage.md`). Read that index
  once — the per-code pages carry the exact log keyword, the cause and the fix keyword table that
  `init-failures.md` condenses. Pinned for these notes: `cann/hccl` HEAD `f49377a5`
  (`git describe` = `v8.5.0-beta.1-1411-gf49377a5`, tag `v9.2.0` present); paths and line numbers move
  between CANN releases, so cite the page name with the code, not a line number.
- **The reporting rank is usually not the first scene.** Link and notify timeouts cascade: rank N is
  blocked on rank N−1 which is blocked on the pair that actually failed. `LINK_ERROR_INFO`'s
  `dest_ip`/`src_ip` (and EI0006's `CONNECT TIMEOUT DETECT RESULT` on A2/A3/950) name the failing
  pair; the answer is about that pair, and the reporter is asked for *its* plog.
- **No `vllm_ascend/` line owns these.** Verified: at `v0.9.2rc1` the package touches only
  `HCCL_SO_PATH` (`envs.py:71`, `utils.py:239-248`, `distributed/device_communicators/pyhccl_wrapper.py`);
  it never sets `HCCL_INTRA_*`, `HCCL_IF_IP`, `HCCL_CONNECT_TIMEOUT` or the port ranges. So the
  deliverable is the launch-script/hardware delta plus the EI code and the reporter's next command —
  never "vllm-ascend sets this". The one env var the plugin *can* set for you is
  `HCCL_OP_EXPANSION_MODE` (one model patch, and it is only read elsewhere): check
  `env-support-matrix.md` before saying the user chose the value.
- **Graph mode re-labels the same failure.** Under torchair/ACL-graph the communicator is built at
  graph load, so an HCCL init failure surfaces as `Load graph failed, error code:1343225857` /
  `EJ0002` inside `ModelLoad`, with the real cause only in the plog. One failure, two symptoms; do not
  answer with the graph.
- **The reporter's board layout is a product-doc question.** "单机 16p / B2C" is not derivable from
  device IPs in the plog: read the constraint (`env-support-matrix.md`, Box16 = two 8-device modules)
  or leave it as a `limits` line. Reconstructing chip-per-box arithmetic in the reply is the single
  most expensive dead end in this class (recorded: vllm-ascend #1955).

## Environment or plugin? Two questions decide it

1. **Does the first error mention a device-plane address or a device id?** If yes, the transport is the
   suspect and no plugin change removes it (`ra rdma init failed, devid[12] ip[192.168.20.160]`). If
   every line is a host-plane socket against an `HCCL_IF_IP`-style address, look at the host NIC/env
   instead (a wrong interface name, a `127.0.0.1` advertised where peers are remote, a port collision).
2. **Does the same command start with the world shrunk to one board?** Dropping
   `--enable-expert-parallel` (or `--data-parallel-size 1`) removes the multi-server communicator, so
   success there confirms the fabric shape is the trigger; failure says the transport is broken for the
   intra-node path too. That experiment is cheaper than any code read and is the reply's falsifier.

## Routes

- `init-failures.md` — keyword → EI code → cause → fix/ask, with a worked case (#1955) and the
  port-16666 / RDMA-init discrimination.
- `init-failures.md` also carries the TorchAir timeline: a `torchair_graph_config`/`LoadGraph` report on
  `<= v0.11` has no release to be fixed in (the feature is gone from the tree at `v0.12.0rc1`).
- `env-support-matrix.md` — `HCCL_INTRA_PCIE_ENABLE`/`HCCL_INTRA_ROCE_ENABLE`
  (intra-server only), `HCCL_OP_EXPANSION_MODE` per product (AIV's limits), the timeout/port/log env
  vars, and the rule for a reporter who already bisected the env block (#18315).
- `../symptoms/pd-collective-hang.md` — the collective that stalls with no error at all.
- `../symptoms/moe-ep-dispatch.md` §A — the MoE dispatch op's own resource/ACL error codes.
