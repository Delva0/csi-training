# P and D on one machine: the launcher's card numbering, not HCCL's port band

## Scope

**One host, two engines** — a prefill and a decode instance on a single multi-die box,
each started by its own invocation of
`examples/external_online_dp/launch_online_dp.py` (or of a hand-written copy of its
`run_dp_template.sh`). The **second** instance to start dies at worker init while the
first keeps serving: "start P first, D crashes; start D first, P crashes" (#4725).

Paste shape (recorded #4725, vllm-ascend v0.11.0rc3 `b6d63bbd` + vllm v0.11.0, A3
16 dies, Qwen3-235B-A22B w8a8, `--quantization ascend`, `--enable-expert-parallel`,
DP2×TP4 for P and DP8×TP1 for D):

```text
(Worker_DP0_TP0_EP0 pid=60571) RuntimeError: create_config:…compiler_depend.ts:149
  HCCL function error: hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config,
  &(comm->hcclComm_)), error code is 7
(pid=60571) [ERROR] ERR02200 DIST call hccl api failed.
(pid=60571) Bind_Failed(EJ0003): Failed to bind the IP port. Reason: The IP address
  and port have been bound already.
```

raised from the MoE all-to-all group's creation,
`vllm_ascend/quantization/w8a8_dynamic.py:136-137`
(`backend.get_hccl_comm_name(local_rank)` in
`AscendW8A8DynamicFusedMoEMethod.__init__`) — so the deepest `vllm_ascend` frame names
*a* communicator being built, never the owner. The plog repeats once per rank:

```text
[ERROR] HCCL(60570,):[hccl_socket.cc:106] [HcclSocket][Listen] socket type[0], listen
  on ip[<ip>] and specific port[16666] fail. Please check the port status …
[ERROR] HCCL(60570,):[hccl_socket_manager.cc:54] [ServerInit]call trace: hcclRet -> 7
… [InitRaResource] -> 7 … [InitNetResource] -> 7 … errNo[0x0000000005000007]
  hccl initialize failed … [HcclComm] comm is not exist
```

## `16666` is a real collision, so the port band is not the owner

Unlike `EI0019`'s `65536`, this port is a plausible socket: "the port has been bound
already" here is literal. `HETEROG_CCL_PORT = 16666` is HCCL's single default comm port
(`cann/hcomm/src/pub_inc/hccl_common.h:115` "通信默认端口", same export where
`HOST_CONTROL_BASE_PORT = 60000` is `:119` — line numbers move between hcomm exports,
so grep the constant). `HcclCommunicator::SetRanksPort`
(`src/framework/communicator/impl/hccl_communicator.cc:159-191`) stores that value for
every rank whose rank-table entry carries no usable `device_port`, and `GetLocalNicPort`
(`src/framework/communicator/impl/hccl_communicator_host.cc:1884-1909`) returns it; the
in-code comment at `:1899-1901` says it plainly — *"其余场景 isUseRankPort_ 应当为
false，使用默认 port"*. So all ranks of an instance listen on **the same port** and
differ only by the local device/NIC IP. Two processes whose card sets **overlap** want
the same `<device IP>:16666`. **So this error is a duplicate-device symptom** — with
disjoint cards the endpoints differ, unless both instances were also pinned to one host
NIC IP (`HCCL_IF_BASE_PORT` / `HCCL_NPU_SOCKET_PORT_RANGE` / `ip_local_reserved_ports`
move the number but not the collision, and the recorded fix was a card offset, not a
port change).

Reporter-confirmed on #4725: giving the decode instance cards 8-15 instead of 0-7
fixed it — "It has nothing to do with the `ASCEND_DEFAULT_DEVICE_PORT` macro in this
case."

## The owner: the launcher numbers cards from 0 in every invocation

`examples/external_online_dp/launch_online_dp.py` at that pin (`b6d63bbd`) ends with:

```python
for i in range(dp_size_local):
    …
    visiable_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
```

Inside one invocation the per-DP-instance slices are disjoint (i = 0 … dp_size_local-1).
**Across invocations nothing offsets them**: argparse exposes `--dp-size`, `--tp-size`,
`--dp-size-local`, `--dp-rank-start`, `--dp-address`, `--dp-rpc-port`,
`--vllm-start-port` and **no device-start**; the computed `num_cards =
dp_size_local * tp_size` is never used (the unused variable *is* the missing feature —
the reporter asked for exactly that argument). The template the script calls only
forwards the list: `run_dp_template.sh:16` `export ASCEND_RT_VISIBLE_DEVICES=$1`.

So this recorded pair, on a 16-die box,

```bash
python launch_online_dp.py --dp-size 2 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 \
    --dp-address <ip> --dp-rpc-port 12345 --vllm-start-port 7100   # cards 0-7
python launch_online_dp.py --dp-size 8 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 \
    --dp-address <ip> --dp-rpc-port 12345 --vllm-start-port 7200   # cards 0-7 again
```

separated the HTTP/vLLM ports (`7100`/`7200`, and `--port $2` in the template) and left
the **device sets identical**. Fix (reporter-confirmed):
`range(8 + i * tp_size, 8 + (i + 1) * tp_size)` for the second instance, i.e. add a
device-start argument to the launcher.

The pin's own same-host recipe is
`examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md`: P
`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` (`:34`) and D `4,5,6,7` (`:103`), `:74` stating the
count must equal `dp_size*tp_size`. Note what that guide does **not** separate: both
blocks use `--data-parallel-address localhost` and `--data-parallel-rpc-port 9100`;
they differ in card set, `--port` (8100 / 8200), `kv_port` (20001 / 20002) and
`engine_id`/`kv_rank`. So the card set is the variable an external-DP launch must get
right, and the shared DP rendezvous endpoint is not by itself a defect.

## Checks, cheapest first

1. **Read the launcher once and intersect the two card sets** —
   `rg -n 'visiable_devices|visible_devices|ASCEND_RT_VISIBLE_DEVICES|device[-_]start' examples/external_online_dp/`
   — then compute both sides from the offset the source shows (0 for the pinned file):

   ```bash
   uv run --no-project python -c "
   side={'P':(0,2,4),'D':(0,8,1)}   # (offset the launcher uses, dp_size_local, tp_size)
   cards={k:set(range(o,o+n*tp)) for k,(o,n,tp) in side.items()}
   print(cards); print('overlap', sorted(cards['P'] & cards['D']))"
   ```

   A non-empty overlap is the finding; that one call replaces the whole port hunt.
2. **The `npu-smi info` process table is usually already in the report.**
   #4725's env dump prints two ~59 GB workers per chip on NPU 0-3 with NPU 4-7 empty on
   a 16-die box — the overlap, visible before any log was read. Ask for
   `npu-smi info` if the paste has none and count processes per chip against the
   intended split.
3. **Only if the card sets are provably disjoint** does this become a leftover-process
   question: `ss -lntp | grep ':16666'` and `ps -ef | grep -E 'vllm|python'` on that
   host, kill the survivor. That is the `EJ0001` / sleep-mode lane
   (`../hccl-and-hardware/comm-create-errors.md` §code 7,
   `../runtime-memory/sleep-mode-multi-instance.md`), not this page.

## Do not

- Do not walk the port internals past the one read above — `GetHostPort`,
  `GetNicPort`, `HCCL_IF_BASE_PORT` parsing, `HCCL_NPU_SOCKET_PORT_RANGE` — for this
  shape. The recorded run spent its entire 16 384-token budget in exactly that walk and
  shipped no answer, and the reporter's closing note is that the
  `ASCEND_DEFAULT_DEVICE_PORT` macro had nothing to do with it.
- Not `../hccl-and-hardware/control-plane-port-binding.md`: that page's port exceeds
  65535 (control-plane base 60000 + offset) and no process holds it.
- Not `../hccl-and-hardware/link-establish-timeout.md`: there is no
  `[Wait][LinksEstablishCompleted] is failed. ret[9]` and no `LINK_ERROR_INFO` table
  here, and only one host is involved — the `Listen … fail` is not a peer's node being
  unreachable, it is the local second process.

## Limits

- Source-verified at v0.11.0rc3 (`b6d63bbd`); a later release may add the device-start
  argument the reporter asked for. Re-grep the launcher before asserting the pinned
  formula, and read `--help` output rather than this page if the two differ.
- The mechanism "same card ⇒ same listen endpoint" is derived from the pinned
  `SetRanksPort` / `GetLocalNicPort` and confirmed by the reporter's offset fix; the IP
  in the failed bind is redacted in the paste, and `GetNicPort`'s body is not in the
  hcomm export.
- Whether the decode side's first failure would instead surface as a DP-store/TCP
  collision on a differently staggered start was not observed.
