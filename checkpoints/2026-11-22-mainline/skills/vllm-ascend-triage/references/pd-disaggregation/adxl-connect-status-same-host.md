# `Failed to connect to target … status: 1039xx` on a **single-machine** P/D: read the port band and the errmsg suffix

Scope: PD on **one host, two cards**, `MooncakeConnectorV1` (no KV pool), both
engines up, every own-node request fine, and the **first proxied** request fails
on the decode node with

```text
E0730 14:48:07.674937  1133 ascend_direct_transport.cpp:1140] Failed to connect to target: <P ip>:20022, status: 103905
(EngineCore pid=268) ERROR mooncake_connector.py:781] Mooncake transfer failed for request. remote_request_id=…, ret=-1.
(EngineCore pid=268) ERROR mooncake_connector.py:565] Error in KVCacheTransferThread. error=unhashable type: 'list'.
… "POST /v1/chat/completions HTTP/1.1" 200 OK        # content: "\t\n\n\t\n█\n\n1\n1\n1…"
```

Worked case **#13175**, vllm-ascend `v0.23.0rc1` (`f4a08bddd0cc`) + vllm
`v0.23.0` (`0fc695fc6d1d`) + Mooncake `v0.3.11.post1` (`e9c610757200`, export
`files=933/1093`, so the paths below exist and the directories are partial).
Recorded run shipped **no message text** — the page ends with the block that
session owed.

Not this page: a **cross-node** 103900 that never gets past the plog's
`'device_ip' is not set correctly` → `mooncake-adxl-connect-103900.md` (multi-server
by construction, `hccn_tool`/`device_ip` the fix); a `HIXL Connect failed` under
LMCache-Ascend → `transport-topology.md`; a KV pool in the recipe →
`kv-pool-store-segments.md`.

## 1. What the paste already fixes

- **The segment dial failed; the handshake did not.** `ret=-1` is
  `Mooncake transfer failed`, raised only after `MooncakeAgentMetadata` came back
  over the side-channel (`mooncake-adxl-connect-103900.md` §1), and the P node's
  own `Delaying free of N blocks for request …` proves P served the prefill and
  held the blocks. So do not spend a rung on `nc`/`ss`/CNI: on **one** host the
  ZMQ path already works.
- **It is not a timeout.** At the pin the timeout branch carries its own message —
  `transfer_executor_base.cpp:256-259` `Connect timeout to: <name>, errmsg: …` —
  so a `status:` line means `Connect()` returned a non-`SUCCESS` enum, not a
  deadline. Longer `ASCEND_CONNECT_TIMEOUT` / `HCCL_CONNECT_TIMEOUT` /
  `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` change nothing.
- **103905 is not decodable here.** It is an `adxl::Status` (HIXL) value;
  `cann/hixl` is in `SOURCES.json`'s unavailable list and no export carries the
  enum, so a numeric reading would be invention. The *branch* is informative: not
  `TIMEOUT`, not `SUCCESS`.
- **The garbled answer is a consequence, not a second bug.** `200 OK`,
  `100` generated tokens of `1 1 1 1`, `finish_reason: length`, plus
  `External prefix cache hit rate: 100.0%` with `GPU KV cache usage 0.0%`: the D
  scheduler counted the whole prompt as externally cached and decoded with no KV.
  The `unhashable type: 'list'` line belongs to that failed recv
  (`kv-transfer-thread-masked-error.md`) and the connector's `finally` lets the
  request run on. Do **not** open a model/quant/`w4a4` lane for the `1 1 1` — and
  do not read the reporter's `单卡pd混合部署没有乱码问题` as a PD-vs-colocated
  control; it is an assertion about a run you cannot see.

## 2. Two version tests the paste settles for free — run them first

Both are `rg` against the pinned trees, so they cost one call and they change the
answer's owner:

1. **The `errmsg:` suffix.** At the pinned Mooncake the connect failure prints
   **three** fields (`transfer_executor_base.cpp:263-265`):

   ```text
   LOG(ERROR) << "Failed to connect to target: " << target_adxl_engine_name
              << ", status: " << status << ", errmsg: " << aclGetRecentErrMsg();
   ```

   A paste that ends at `status: 103905`, cites `ascend_direct_transport.cpp:NNNN`
   and carries no `errmsg:` is an **older libmooncake** than the pin — and that
   older build is the one that throws the inner error away, while the pinned build
   prints it. So "upgrade the `mooncake` package to the build paired with this
   release" is a *diagnostic* rung here, not just hygiene: the next run's own
   message carries the HIXL inner error the plog would be searched for.
2. **The `mooncake_connector.py` line numbers.** At `v0.23.0rc1` the three pasted
   messages sit at `:626` (`Error in KVCacheTransferThread`), `:940`
   (`Mooncake transfer failed for request`), `:1851` (`Delaying free of …`); the
   paste's `565 / 781 / 1529` are all *before* them, in that order. A monotone
   offset in one file is the installed-tree-is-not-the-tag signature
   (`../image-dependency-faults/installed-version-vs-checkout.md`). It is also
   load-bearing for the mask: see §3.

## 3. The mask is **fixed** at `v0.23.0rc1` — so its presence dates the tree

`kv-transfer-thread-masked-error.md` records the bug at `v0.21.0rc1`
(`set[int].update(<list[list[int]]>)`). At `v0.23.0rc1`
`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:605-608` is

```python
def _mark_failed_recv_request(self, request_id: str, local_block_ids: BlockIds) -> None:
    self.invalid_block_ids.update(local_block_ids[0])
```

i.e. group 0 only, flat — no model **can** raise `unhashable type: 'list'` through
this site at that tag (`BlockIds` is `list[list[int]]`, so `[0]` is a flat list;
multi-group models merely lose groups ≥1 from the mark). Two consequences:
the mask in the paste is not the tag's code (§2.2's offsets agree), and the real
`Failed to transfer KV cache for request …: <error>` line *is* reachable in their
build — ask for the unfiltered D-side log before theorising. Re-grep
`invalid_block_ids.update` at whatever ref the report names; a reloaded/older
wheel is the common cause on a host where `pip show` agrees with itself.

## 4. The one local computation: the target port is the P engine's **physical card**

`ascend_direct_transport.cpp:131-141` builds the segment port with
`FindAdxlListenPort(base_port_, device_id)`; `utils.cpp:26-27, 75-86`:

```text
kPortRange = 100;  kMaxGenPortAttempts = 500;
aclrtGetPhyDevIdByLogicDevId(device_id, &physical_dev_id);
min_port = base_port + physical_dev_id * kPortRange;
max_port = base_port + (physical_dev_id + 1) * kPortRange;   // LOG: "Find available between X and Y"
port     = min_port + rand() % (max_port - min_port + 1);
```

and `base_port_` defaults to **20000** (`ascend_direct_transport.h:79`), overridable
by `ASCEND_BASE_PORT` (`:163-176`, `LOG(INFO) "Set base port to:"`).

So `<P ip>:20022` with the default base is band `[20000, 20100)` ⇒ **physical
device 0** on the P side, chosen randomly inside the band. Read it as a card
identity, not a fixed port:

- ask for `ASCEND_BASE_PORT` (unset on this paste) and the P-side
  `Set base port to:` / `Find available between …` INFO lines — they name base and
  band outright, so no arithmetic is needed;
- compare the implied physical id with the card P was *assigned*
  (`ASCEND_RT_VISIBLE_DEVICES` per process, `npu-smi info`). Containers that expose
  only their own die remap logical 0 → another physical id, and the band moves
  (`203xx` ⇒ physical 3). A D side dialling a segment bound on a die it has no
  rank-table entry for fails exactly here;
- on **one** host the band is also a collision surface: if P and D both map to the
  same physical id (both launched with `ASCEND_RT_VISIBLE_DEVICES=0`), both pick a
  port in the same 100-wide band. (`#4725`, "P and D on the same machine, port
  conflict", is the neighbouring but different family: HCCL `16666` `Bind_Failed`
  *at startup*, `same-host-pd-device-partition.md`.)

## 5. Ranked causes

1. **Installed trees are older than the claimed tag** (§2, two independent tells).
   Check: `python -c "import vllm_ascend, vllm; print(vllm_ascend.__file__, vllm.__version__)"`,
   then `rg -n 'invalid_block_ids.update|Failed to connect to target' <that file>` /
   the `mooncake` package's `.so` provenance, and `sed -n '600,612p'` of their
   connector. Source-verified, cheapest, and it explains both oddities at once.
2. **Device identity / port band mismatch** (§4): P's segment on a physical card the
   D side's rank table does not carry, or both roles on one card. Check: the two
   `Find available between` lines + `npu-smi info` + the two `ASCEND_RT_VISIBLE_DEVICES`.
3. **`HCCL_INTRA_ROCE_ENABLE=1` inherited from an entrypoint** — at the pin that
   sets `roce_mode_ = true` and logs `Roce mode is enabled.`
   (`ascend_direct_transport.cpp:152-162`), which puts the transport on the RoCE
   path and makes the rank table demand `device_ip` even on one machine (HCCL's own
   `'device_ip' is not set correctly, must be set when multi Server or
   HCCL_INTRA_ROCE_ENABLE enabled`). Check: `env | grep HCCL` and one
   `hccn_tool -i <id> -ip -g`. Same-host recipes copy `export HCCL_INTRA_ROCE_ENABLE=1`
   from cross-node scripts routinely.
4. **A stale engine still holding P's band** from a previous run (§4): the D side
   connects to a dead segment and the *live* P never sees it. Check on the one host:
   `ss -lntp | grep ':200'` and one `npu-smi info` process list before relaunching.
5. **The garbled output itself** — never a separate owner while §1 holds; if it
   persists after a clean transfer, then the accuracy topics apply.

## 6. The first message (#13175 owed this and shipped none)

```text
The paste does not need the transport to be decoded: your D node got the handshake
and failed the segment connect, then decoded without KV — that is the "1 1 1" answer.

1. Your trees are not the 0.23.0rc1 tag. At that tag the connect line prints
   ", errmsg: <aclGetRecentErrMsg()>" and lives in transfer_executor_base.cpp:263;
   yours stops at "status: 103905" from ascend_direct_transport.cpp, and the mask
   line you pasted is already fixed there (:608 uses local_block_ids[0]).
   Send: python -c "import vllm_ascend, vllm; print(vllm_ascend.__file__, vllm.__version__)",
   pip show vllm-ascend, and the mooncake package version — then re-run on the
   paired build, whose connect line carries the inner HIXL error.
2. Your P segment bound on physical card 0 (default base port 20000 → 20022).
   Send ASCEND_BASE_PORT, both serve commands with ASCEND_RT_VISIBLE_DEVICES,
   the P-side "Find available between …" lines and npu-smi info.
3. Send env | grep HCCL: HCCL_INTRA_ROCE_ENABLE=1 switches the transport to RoCE
   and then needs device_ip even on one machine.
Do not widen timeouts or raise card counts: nothing timed out.
```

## Recheck conditions

- `kPortRange`, the 20000 default, the `errmsg:` suffix and the file that owns
  `Failed to connect to target` all move between Mooncake releases; re-grep
  `base_port_`, `FindAdxlListenPort` and the message text in the `mooncake`
  package's matching export before quoting a band.
- The 1039xx values themselves stay unreadable until `cann/hixl` is exported; if it
  ever is, replace §1's "not decodable" with the enum table.
- `invalid_block_ids.update` may be flattened again (per-group loop) at a later ref;
  re-read `_mark_failed_recv_request` before calling the mask absent.
