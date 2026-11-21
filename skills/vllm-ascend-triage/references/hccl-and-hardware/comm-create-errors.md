# Communicator creation fails: `HCCL function error: hcclCommInitRootInfoConfig … error code is N`

Reported as `RuntimeError: create_config build/CMakeFiles/torch_npu.dir/compiler_depend.ts:89 HCCL
function error: hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)),
error code is N` — often in the title, OCR-mangled (`hcclCommInintRootConfig`, `&rootInfo.ranl`)
and pointing at `…/csrc/distributed/HCCLUtils.cpp:140`. Verified at the v0.22.1rc1 snapshot
(`Ascend/pytorch` v26.0.1-pytorch2.10.0, `cann/hcomm` 818068c8):

- `HCCLUtils.cpp:133` is `HCCLComm::create_config`, `:140` the failing call. It is reached from
  `ProcessGroupHCCL::createHCCLCommOrigin` (`ProcessGroupHCCL.cpp:2575`) after rank 0's
  `hcclGetRootInfo` and `broadcastMasterID` (`:2586-2589`), with `HcclCommConfig` filled by
  `getHcclCommConfig` (`:302-308`).
- So the failing object is a **communicator**, not a launched collective: no kernel ran. Whichever
  group is being materialised is named by the deepest `vllm_ascend/` frame — on a MoE/EP serve the
  first cross-node one is the all-to-all group at
  `vllm_ascend/ops/fused_moe/token_dispatcher.py:465` (`TokenDispatcherWithAll2AllV`, `:108` for the
  sibling dispatcher) via `backend.get_hccl_comm_name(local_rank)`.
- `N` is the `HcclResult` enum (`cann/hcomm/include/hccl/hccl_types.h:24-40`): `1` PARA, `9`
  TIMEOUT, `11` TCP_CONNECT, `12` ROCE_CONNECT, `13/14` TCP/ROCE_TRANSFER, `15` RUNTIME, `16` DRV.

## Code 1 does not name its parameter — and it is not the fabric

`HcclCommInitRootInfoConfigInner` (`cann/hcomm/src/framework/op_base/src/op_base.cc:1362`) folds six
different rejections into `HCCL_E_PARA` and prints an `HCCL_ERROR` naming the sub-check:

| line | sub-check, as printed by HCCL |
|---|---|
| `:1377` | null `config` (an `RPT_INPUT_ERR` with `EI0003`) |
| `:1385` / `:1391` | `nRanks[%u] should be greater than 0` / `rank[%u] should be less than nRanks[%u]` |
| `:1396` | `init external input error` — `InitExternalInput()` failed |
| `:1399` | `init environment config error` — `InitEnvConfig()` failed |
| `:1405` | `load comm config failed` — `commConfig.Load(config)` |

Everything after that (`InitCommRootInfo`, just below the `Entry-…` log at `:1411`) returns its **own** code, so a transport or runtime
fault surfaces as `9/11/12/15` — it is never reported as `1`. Read `1` as "a parameter or an
environment value in the launch script was rejected", and the naming line is the HCCL one, not the
Python traceback. The launch-script values that reach this point through torch_npu's config:
`HCCL_BUFFSIZE` (`ProcessGroupHCCL.cpp:306` → `hcclBufferSize`;
`externalinput.cc:1423-1427` rejects non-digits or anything below `HCCL_CCL_COMM_BUFFER_MIN = 1` MB,
default `200` MB, `externalinput_pub.h:36-37`), and the config's magic word/version
(`comm_config.cc` `CheckMagicWord` → `EI0003 … Please make sure the config has been initialized by
HcclCommConfigInit()`, the shape a torch_npu/CANN mismatch takes).

## Order of work (3-4 calls, then deliver)

1. **Say what the code means on the first turn that has it**: code `1` at this entry is a rejected
   parameter/env value, not a NIC, rank table or card fault; the line that names it is an
   `HCCL_ERROR` in the process log / plog (`init external input error`, `init environment config
   error`, `load comm config failed`, `nRanks…`). Ask for exactly that line.
2. **Get both nodes' serve scripts** (reporters on a multi-node start usually have them) and diff
   them: `HCCL_BUFFSIZE` present and numeric, `HCCL_INTRA_PCIE_ENABLE`/`HCCL_INTRA_ROCE_ENABLE`,
   `HCCL_OP_EXPANSION_MODE`, `HCCL_DETERMINISTIC`, `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME`, and the DP
   flags — `--data-parallel-size`, `--data-parallel-size-local`, and a **unique**
   `--data-parallel-start-rank` per node.
3. **Check the recipe exists at the pin for that model, card and node count.** At v0.22.1rc1
   `docs/source/tutorials/models/DeepSeek-V4-Pro.md:141` needs ≥2 A3 or **4** A2 nodes and runs
   DP4 × TP8 with `--data-parallel-start-rank` per node (`:188-191`, `:260-268`, note `:429`), while
   `DeepSeek-V4-Flash.md` documents a **single** node (§5.1, `:139-141`) or multi-node **PD
   separation** only (§5.2), on the listed `w8a8-mtp` weights (`:26`). A 2-node non-PD DP serve of a
   bf16 variant of that model is outside every documented recipe, which is part of the answer, not a
   footnote.
4. **If a parallel-config change turns `1` into a hang**, that is rank asymmetry: one rank rejected
   its config while its peers block in the collective. Collect the per-rank HCCL `ERROR` lines
   (`ASCEND_LAUNCH_BLOCKING=1` does not help here) instead of re-reading the traceback.

In the same paste, `broadcastMasterID` frames and TCPStore
`Failed to recv, got 0 bytes, connection was closed. Did the remote server shutdown or crash?` are
**secondary** — they mean a peer or the store server went away, and the earliest pid with the
`RuntimeError` owns the fault. `ERR02200` (OCR'd from the log) is not in the pinned export; do not
translate it.

## Precedents

`#672`: same call site with `error code is 15` on a DP+EP offline DeepSeek-V2-Lite run.
`#2712`: `create_config … :148` on a `--pipeline-parallel-size` serve. `#3196`: "how to
troubleshoot" for code `11`. `#397`: the `create`/`HcclCommInitRootInfo` (no-config) sibling.

Recorded #11542 (v0.22.1rc1 image, 2× 910B3, fork branch, `DeepSeek-V4-Flash-DSpark-bf16`, MoE EP
group at `token_dispatcher.py`) walked this stack correctly down to the `:1396/:1399` folding and
then emitted **30 characters of message text in 20 assistant turns** before the output cap — the
mechanism above was in hand by turn 10. Deliver from facts, not from the bottom of the stack.
