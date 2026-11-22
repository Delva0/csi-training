# Communicator creation fails: `HCCL function error: hcclCommInitRootInfo(Config) … error code is N`

Two entry points, one decision: `create_config` (`hcclCommInitRootInfoConfig`, the newer generation,
first section below) and the older no-config `create` (`HcclCommInitRootInfo`, §*The no-config
`create` sibling*, which also owns code `2` and groups owned by a third-party trainer).

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

## Code 15 `HCCL_E_RUNTIME` at this entry is HCCL's *device-pair link-type* lookup

`15` is neither a launch-script rejection (that is `1`) nor a live previous process (`7`): during
communicator setup HCCL asks the runtime for the **link type between two physical devices** and the
driver declined, so no collective was launched. Verified at `cann/hcomm` `818068c8` — the same
commit this page's v0.22.1rc1 pin and the v0.18.0rc1 report cited below carry:

- The query is per **same-server rank pair**, from the rank list's `devicePhyId`s:
  `hccl_communicator_attrs_host.cc:275-299` calls `hrtGetPairDeviceLinkType(rankList[i].devicePhyId,
  rankList[j].devicePhyId, linkType)` to set `isConnectedWithHCCS` (which for `DEV_TYPE_910B` then
  picks `deviceNumPerAggregation_ = deviceNumPerServer_/2`, `:301-304`), and
  `topo_info_extractor.cc:199-208` does the same behind `CHK_RET`, aborting comm creation on the
  first pair it cannot resolve. A rank's own `devicePhyId_` comes from the attribute collector
  (`hccl_communicator.cc:462`), i.e. from its own runtime's logic→phy mapping.
- `hrtGetPairDeviceLinkType` (`…/adapter/adapter_rts.cc:907-948`) skips the runtime only on 310P;
  otherwise it goes to `hrtGetPairDeviceLinkTypeRaw` (`:831-839`), which prefers the driver's
  `rtGetPairPhyDevicesInfo` and falls back to `hrtGetPairDevicesInfo` (`:841-880`, phy→logic via
  `hrtGetDeviceIndexByPhyId` then `aclrtGetDevicesTopo`). `hrtGetPairPhyDevicesInfo` (`:889-897`)
  folds **any** non-`RT_ERROR_NONE` return into `HCCL_E_RUNTIME` with
  `[Get][PairPhyDevicesInfo] … rt get pair devices info failed, return[%d], para: phyDevId[%u],
  otherPhyDevId[%u]` — that is torch_npu's `error code is 15`.

Read the line printed *under* the code; it names which runtime call refused:

| sub-text | case |
| --- | --- |
| `get error: phyid:N realDeviceId:N is err:0x7010003[FUNC:GetDeviceIndexByPhyId]` + `rtGetDeviceIndexByPhyId execute failed, reason=[device id error]` | #454, TP4+DP2 on one node |
| `rtGetPairPhyDevicesInfo execution failed, reason=driver error:internal error` + `Invalid_Argument(EL0003)` | #8519, containerised TP4 on an Atlas 800I A2 (the reporter's "910B4 four cards") |

Both are **driver return codes**: the snapshot carries only declarations
(`cann/runtime/src/inc/runtime/dev.h:563`, the weak `halGetPairPhyDevicesInfo` at
`…/feature/src/drv/npu_driver_base.hpp:38`, `cann/driver/pkg_inc/ascend_hal_base.h:1390`) and no
implementation, and `cann/hcomm` only wraps them — so the precondition is **not in any exported
tree**, and one more source read cannot name it. Deliver on the turn the code arrives, with the
enum name, the fact that comm setup wanted the pair's link type, and three artefacts:

1. `ls /dev/davinci*` and `ASCEND_RT_VISIBLE_DEVICES` **inside the container**, plus
   `npu-smi info -t topo`. The pair query is by *physical* id and covers every pair in the rank
   list, so the visible set and the phy↔logical mapping must agree on every rank
   (`../deploy-sizing/card-budget-and-device-visibility.md` owns the ascending-list rule and the
   "ids index the driver's device list, not the `npu-smi` labels" trap).
2. A ~10-line `torch.distributed` HCCL `init_process_group` + `all_gather` on the **same image,
   container and device set**. The same `15` there exonerates vllm-ascend, `torch.compile` and the
   serve flags in one run; a clean run turns the lane back to the engine.
3. A **high or non-contiguous `--device /dev/davinciN` list is not a typo.** The #8519 reporter
   later confirmed (2026-04-22) that the machine is an Atlas 800I A2 and that its **own NPU-ID
   labelling is `10/12/14/16`** — the same box the title calls "910B4 four cards", not a version
   conflict — so "that device does not exist" is refuted before it is ranked: the finding is the
   mapping. Order the two halves, because the runtime list indexes the device list the container
   *exposes*, not the chassis labels (`../deploy-sizing/card-budget-and-device-visibility.md`):
   mount the **full** node set first, then pin `ASCEND_RT_VISIBLE_DEVICES` to those ids ascending (or
   drop to a contiguous set). A partial mount carrying the high ids can give **zero** devices, not
   four: `_device_count_ascend_hal` returns only the count of *leading* entries below
   `drvGetDevNum()` (`Ascend/pytorch` `v7.2.0.1-pytorch2.8.0` `torch_npu/npu/__init__.py:386-409`,
   trim at `:403-404`; the parse is `:341-366`), so
   `python -c "import torch_npu; print(torch_npu.npu.device_count())"` inside the container is what
   says whether the pin took effect — and a bare `0` is not "four cards, one missing".

**A `vllm_ascend/` frame in that traceback is a sync point, not the owner.** On #8519 the comm was
created from `torch.compile`'s pattern registration because the RMSNorm+quant fusion pass
real-traces a pattern that contains a TP all-gather
(`compilation/passes/norm_quant_fusion_pass.py:494` → `…/passes/base_pattern.py:49` →
`torch/_inductor/pattern_matcher.py` `make_fx(…, tracing_mode="real")`), so the process's first
collective runs there — after the weights load and before the model's own forward. That explains the
"error after loading the model" position; it does not move the cause, and `--enforce-eager` is
expected to only move the first collective to the model's forward, where the same comm-creation
refusal returns (hypothesis, not a verified fix).

**Which key removes that first collective, and why one key is not enough.** At v0.18.0rc1 the
collective lives in the *pattern bodies* of the fusion passes the pass manager appends from
`additional_config.ascend_compilation_config` (`compilation/graph_fusion_pass_manager.py:49-70`, gates
at `:52/57/62/67`): `norm_quant_fusion_pass.py:185,208,249,273,399,414-415,450`
(`torch.ops.vllm.maybe_all_gather_and_maybe_unpad`) and, only when that key is on,
`allreduce_rmsnorm_fusion_pass.py:69,121` (`tensor_model_parallel_all_reduce`). So
`--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'` removes the first
family and leaves the second, while `--enforce-eager` skips the pass manager altogether and is the
only single-axis ablation. Two traps around it: (a) `graph_fusion_pass_manager.py:62` reads
`.get("fuse_allreduce_rms", True)`, yet a serve that does not set the key still gets `False`, because
`AscendCompilationConfig` defaults it to `False` (`ascend_config.py:268`) and
`platform.py:259-264` writes the parsed dataclass back into `additional_config` before the manager
runs — grepping the manager alone reports the wrong default; (b) `ascend_fusion_config` is a
*different* key, read at `ascend_config.py:41` into `AscendFusionConfig` (`:305-320`,
`fusion_ops_gmmswigluquant` only), so a fusion toggle written there reaches none of these passes —
which key is read where is `../startup-failures/additional-config-schema.md`'s audit. Read the ablation both ways: the same `15` at the
model's own first collective says the pair query refuses for that device set; a clean run says the
trigger is the tracing context itself, and then the plog from *that* run is the evidence.

Recorded #8519 had this lane by its sixth assistant
turn, then spent five more turns in `cann/hcomm`/`cann/runtime` for a precondition that lives in the
closed driver: 52 K characters of reasoning against 1.2 K of message text, and no answer at the
16 384-token cap.

## Code 7 `HCCL_E_UNAVAIL` beside `EJ0001` is a *live previous process*, not a parameter

`error code is 7` plus `[ERROR] … ERR02200 DIST call hccl api failed` and
`EJ0001: … Failed to initialize the HCCP process. Reason: Maybe the last training
process is running.` is a different lane from code `1` above: nothing in the launch
script was rejected. `EJ0001` is an **HCCP** (host RDMA/HDC agent) error, and the CANN
table quotes its own fix — *"Wait for 10s after killing the last training process and
try again."* (`cann/hcomm/src/common/error_code/error_code.cc:186-197`, mirrored in
`cann/metadef/error_manager/error_code.json`; `bash <skill_root>/scripts/cann-error-lookup.sh EJ0001`).
The "last training process" is usually the reporter's **own earlier engine** — one that
was `sleep()`-ed rather than killed, an RL rollout that never exited, or a crashed
leader whose RA/HCCP side is still up. Note the paste/OCR renders it `HCCP`; grep
`Failed to initialize the HCC`, not `HCCL`. Read `../runtime-memory/sleep-mode-multi-instance.md`
for the sleep-mode form (what `sleep()` does not release, the disjoint-card
discriminator, the ladder) before walking any torch_npu C++.

### Code 7 with `Bind_Failed(EJ0003)` is the same lane, one device deeper

`error code is 7` can also arrive as `Bind_Failed(EJ0003): Failed to bind the IP port.
Reason: The IP address and port have been bound already.` with the plog repeating
`[hccl_socket.cc:106] [HcclSocket][Listen] socket type[0], listen on ip[<ip>] and
specific port[16666] fail …` → `[ServerInit] hcclRet -> 7`. Here the port is real
(`16666` = `HETEROG_CCL_PORT`, the single default comm port; `HCCL_IF_BASE_PORT` does
not move it) and the collision is **two live processes on the same device**: every
rank uses that one port and differs only by the local device IP, so a shared chip means
a shared endpoint. `EJ0001` (above) is the previous process not having exited;
`EJ0003` is the previous process still *holding* the device — usually two engines
started side by side on one host. Count cards per engine before anything else:
`../pd-disaggregation/same-host-pd-device-partition.md` (#4725, reporter-confirmed —
the external-DP launcher gives every invocation cards from 0). Only after the card sets
are provably disjoint does this reduce to the stale-process ladder above.

## The no-config `create` sibling

Code `2`, on a group owned by a third-party trainer rather than by `vllm_ascend/`.

Recorded #397: vllm-ascend v0.7.3rc1 (`f025df05`), vllm v0.7.3 (`ed6e907`), torch-npu
`2.5.1.dev20250308`, CANN 8.0.0, 2x910B2, open-r1 GRPO on the TRL-npu fork. Title
`RuntimeError: create:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:89 HCCL function error:
HcclCommInitRootInfo(numRanks, &rootInfo, rank, &(comm->hcclComm_)), error code is 2`, beside
`[ERROR] … (Device:0, RankID:0) ERR02200 DIST call hccl api failed`. Verified against the #397
snapshot (`Ascend/pytorch` `v6.0.0.alpha001-pytorch2.5.1`, vllm v0.7.3, `pytorch/pytorch` v2.5.1):

- **The class sits in the `.hpp` on this generation.** `HCCLComm::create` is
  `torch_npu/csrc/distributed/HCCLUtils.hpp:85-95` with the failing `HcclCommInitRootInfo` at `:91`
  (matching the reporter's `compiler_depend.ts:89` within drift); `HCCLUtils.cpp` at that export is
  117 lines of datatype maps only. A `grep HcclCommInitRootInfo HCCLUtils.cpp` that returns nothing
  is the wrong file, not a wrong row. Reached from `ProcessGroupHCCL::createHCCLComm`
  (`ProcessGroupHCCL.cpp:1133-1147`).
- **The traceback names the group, and it may not be a `vllm_ascend/` group at all.** #397's frames
  are `trl/trainer/grpo_trainer.py:568 prepare_deepspeed` → `deepspeed/runtime/engine.py:1201
  _configure_distributed_model` → `:1120 _broadcast_model` →
  `dist.broadcast(p.data, groups._get_broadcast_src_rank(), group=self.seq_data_parallel_group)` →
  `torch/distributed/distributed_c10d.py:2421`. No `vllm_ascend/` frame: the owner is torch_npu +
  HCCL invoked by a **third-party trainer colocated in the engine's process**, and the vllm-ascend
  half of the answer is only that coexistence (the engine had already initialised the device and its
  own process group). Say that on turn one instead of searching the tree for the caller.
- **Code `2` is `HCCL_E_PTR` ("empty pointer") and names no precondition**
  (`Ascend/pytorch` `third_party/hccl/inc/hccl/hccl_types.h:29`; `scripts/cann-error-lookup.sh 2`
  prints it even when the snapshot carries no `cann/hcomm`, as the #397 pin does). With no
  `op_base.cc` in the export there is nothing to fold into sub-checks — state that and stop; a
  twenty-turn enumeration of HCCL internals cannot be verified from the evidence the case carries.
- **The store/root-info plumbing is not the suspect, and on a one-rank group it cannot be.**
  `broadcastMasterID` (`ProcessGroupHCCL.cpp:976-1049`) has rank 0 `store_->set(std::to_string(
  hcclCommCounter_++), rootInfo)` and only non-zero ranks `store_->get`; a missing key throws the
  distinct `… retrieving hcclUniqueId from [0] via c10d key-value store by key '<k>' …` error, and a
  PTA skew only logs `PTA version mismatch`. Rank 0 therefore hands `HcclCommInitRootInfo` a rootInfo
  it just produced, so a code `2` on a single-rank group is HCCL's own precondition on
  `numRanks`/resources, not a store or rendezvous fault.
- **Ask for the trainer's launch line in the first reply — it is `numRanks`.** #397's reporter later
  posted it: `accelerate launch --config_file recipes/accelerate_configs/zero2.yaml
  --num_processes=1 src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml`
  with `ASCEND_VISIBLE_DEVICES=1,2`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`,
  `HCCL_RDMA_TIMEOUT=20`, `TASK_QUEUE_ENABLE=2`, `COMBINED_ENABLE=1`, `CPU_AFFINITY_CONF=1`,
  `HCCL_CONNECT/EVENT/EXEC_TIMEOUT=1800`, `NPU_ASD_ENABLE=0`, `ASCEND_LAUNCH_BLOCKING=1`,
  `ACLNN_CACHE_LIMIT=100000`, `ASCEND_GLOBAL_LOG_LEVEL=0`, `pkill -9 python3; sleep 10`. World size
  **1** — which DeepSpeed's own `Config mesh_device None world_size = 1` in the paste already said.
  The same open-r1 recipe at **7** processes (`#116`, on `#109`'s install and launch list) got past
  this broadcast and died later at `copy_d2d … 507001`, so quote that comparison rather than ablating
  the env list; none of those values names a `numRanks` precondition.
- **Before writing "single-rank HCCL is unsupported", check who already made a one-rank comm.** At
  vllm v0.7.3 `init_worker_distributed_environment` (`vllm/worker/worker.py:494-506`) calls
  `init_distributed_environment` with no `world_size > 1` gate, and that function
  (`vllm/distributed/parallel_state.py:803-823`) has **no** `world_size == 1` early return — so a
  TP=1 engine may already hold a one-rank default-group comm. Creation is lazy (it happens on the
  first collective), so ask whether any collective ran during warmup before ranking this.
- **Deliverable for this shape**: enum name + the trainer frame + the world size in the first
  paragraph; the actionable half is process topology (one process per card, or do not materialise a
  trainer HCCL group in the engine's process) plus "vendor-side code `2` on a one-rank comm is not
  something the launch env names". Recorded #397 spent all 16 384 output tokens on candidate
  mechanisms (one-rank comm, port/HCCP reuse, `maybe_chunk_residual`, absent upstream `qwen3_5`
  files) and shipped no message at all.

## Order of work (3-4 calls, then deliver)

1. **Say what the code means on the first turn that has it**: code `1` at this entry is a rejected
   parameter/env value, not a NIC, rank table or card fault; the line that names it is an
   `HCCL_ERROR` in the process log / plog (`init external input error`, `init environment config
   error`, `load comm config failed`, `nRanks…`). Ask for exactly that line. If the call is
   `HcclCommInitRootInfo` with **no** `config` argument and code `2`, it is not this entry at
   all — the group belongs to a trainer; read §*The no-config `create` sibling* below.
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

`#672`: same call site with `error code is 15` on a DP+EP offline DeepSeek-V2-Lite run — the
device-pair lane of §*Code 15*, not a launch-script value. `#454`: the sibling sub-text
(`rtGetDeviceIndexByPhyId … [device id error]`, `0x7010003`) on TP4+DP2 at one node, i.e. the same
phy↔logical mapping question.
`#2712`: `create_config … :148` on a `--pipeline-parallel-size` serve. `#3196`: "how to
troubleshoot" for code `11`. `#3172`: code `7` + `EJ0001` on the second engine after a
`sleep()` — the previous process was still alive
(`../runtime-memory/sleep-mode-multi-instance.md`). `#4725`: code `7` + `EJ0003`
`Bind_Failed` on the second of **two engines started side by side on one host** — the
external-DP launcher gave both invocations the same cards
(`../pd-disaggregation/same-host-pd-device-partition.md`). `#397`: the
`create`/`HcclCommInitRootInfo` (no-config) sibling with code `2` — read §*The no-config
`create` sibling* above before opening a torch_npu file, the group there belongs to a
trainer, not to `vllm_ascend/`. `#116`: the same open-r1 + vLLM + DeepSpeed recipe (on
`#109`'s install) at **7** processes, which got past this broadcast and died later
(`copy_d2d … 507001`) — the world-size discriminator for #397.

Recorded #11542 (v0.22.1rc1 image, 2× 910B3, fork branch, `DeepSeek-V4-Flash-DSpark-bf16`, MoE EP
group at `token_dispatcher.py`) walked this stack correctly down to the `:1396/:1399` folding and
then emitted **30 characters of message text in 20 assistant turns** before the output cap — the
mechanism above was in hand by turn 10. Deliver from facts, not from the bottom of the stack.
