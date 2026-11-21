# HCCL link creation dying on an IPC import (`Transport_Init_Error(EI0009)`, `DestLink`, 507899)

## Scope

The engine loads the weights and then the *first* collective dies with:

```text
Transport_Init_Error(EI0009): Transport init error. Reason: [Create][DestLink]Create Dest error!
createLink para:rank[i]-localUserrank[i]-localIpAddr[<addr>], dst_rank[j]-remoteUserrank[j]-remote_ip_addr[<addr>]
[drv api] halShmemOpenHandle or halShmemOpenHandleByDevId failed: name=<hex>, drvRetCode=17, device_id=N
rtsIpcMemImportByKey execute failed, reason=[driver error:internal error]
call rtsIpcMemImportByKey failed, runtime result = 507899
```

usually wrapped as `RuntimeError: Worker failed with error 'The Inner error is reported as
above … the current working operator name is HcclAllreduce'` with a Python frame inside a
model op (`vllm_ascend/ops/rotary_embedding.py`, a norm, a matmul).

Not this lane — same code family, different owner:

- control-plane bind/port (`Communication_Error_Bind_IP_Port(EI0019)`,
  `port <n> have already been bound`) → `control-plane-port-binding.md`;
- P/D connect (`HIXL Connect failed | 103900`, `RanktableCheck`, `'device_ip' is not set
  correctly`) → `../pd-disaggregation/transport-topology.md`;
- allocator unmap at `empty_cache` (`unmapHandles … getHcclComm() … 507899`) →
  `../runtime-memory/allocator-expandable-segments-unmap-crash.md`;
- UCE query/repair (`checkUceErrAndRepair … 507899`) → `error-report-timestamps.md`.
- a *cascade*: one rank pasted a different, non-collective error (a client RPC, a
  weight update) and the rest fail to build links **to that rank** →
  `weight-sync-hccl-cascade.md`; count the ranks before reading the edge set.
- a **refused local allocation** reported through the same sentence:
  `EI0007 … Failed to allocate resource[qp] … Reason: Memory resources are
  exhausted`, with a `DestLink` line for **every** peer and no
  `halShmemOpenHandle` / `507899` line, at graph capture → the OOM-class budget
  lane, `qp-resource-exhausted-at-graph-load.md`; the edge set is not the finding
  there.

## Read the fields, then the edge set

- `rank[i]` is the global rank, `localUserrank[i]` its user rank; `dst_rank[j]` /
  `remoteUserrank[j]` is the peer it was building to. `device_id=N` on the IPC line is the
  **local** device whose shared-memory import failed.
- **`507899` is `[driver error:internal error]`** — generic, and it does not name a
  direction. The verb above it names the operation (`rtsIpcMemImportByKey`,
  `AclrtUnmapMem`, `rtGetDeviceCount`), so quote the operation. When the same code appears
  before any model op (`rtGetDeviceCount:ErrCode=507899`), the device stack is already
  unusable — that is the driver, not HCCL; when the paste also shows `npu-smi` itself
  failing (no engine at all), that reading is
  `../startup-failures/no-npu-in-container.md`.
- **The two addresses being identical is not a duplicate-`device_ip` finding.** A recorded
  precedent prints the local rank's own address on *both* sides —
  `localIpAddr[11.254.19.30] … remote_ip_addr[11.254.19.30]` for a rank1 fan-out (#486) —
  and a redacted paste can collapse every address to one placeholder token. A duplicate
  `device_ip` verdict needs the rank table itself
  (`bash <skill_root>/scripts/ranktable-check.sh <rank_table|log>`) or HCCL's own
  `server`/`device_ip` lines, never this sentence. Building a "two ranks share an IP" theory
  on the placeholder is the recorded trap (#4148 spent two turns on it).
- **The failing edge set is the discriminator**, and it needs the rank→device map from the
  reporter's `npu-smi` block (`ASCEND_VISIBLE_DEVICES` order). A 300I Duo is two chips per
  physical card, so two device rows share one Bus-Id. One call:
  `bash <skill_root>/scripts/hccl-link-pair-audit.sh <log|->` prints the distinct
  `(src, dst)` pairs, the `drvRetCode` / runtime results, and classifies each failing pair
  same-card vs cross-card.
  Recorded #4148 (vllm-ascend v0.11.0rc1 + vllm 0.11.0, 4×310P3, `npu-smi 25.2.0`,
  torch_npu 2.7.1, Qwen3-32B, `--tensor-parallel-size 4 --enforce-eager --dtype float16`):
  devices 0,1 on `0000:03:00.0`, devices 2,3 on `0000:83:00.0`; only `rank0→rank3` and
  `rank1→rank2` failed — every failing edge is **cross-card**, and no card-local edge was
  recorded failing (that absence is the evidence; the log prints the failed edges only). That is a host transport/driver finding: the model, the flags and the rank-table
  ids are not the owner, and no vllm-ascend file is in the failing call.
- **The Python frame is a sync point.** The launch is asynchronous, so the frame that
  raised (`rotary_embedding`, `custom_op.forward`, an `HcclAllreduce`) is where the
  collectives were first joined, not the op that failed — same attribution rule as
  `collective-error-chain.md`, and `ASCEND_LAUNCH_BLOCKING=1` has nothing to add here
  because the HCCL record already carries the link.

## What to check, cheapest first

1. **Host driver vs image CANN pairing.** `npu-smi info` header version,
   `cat /usr/local/Ascend/driver/version.info`, and the `-310P`-class image's CANN base
   (`Dockerfile*.openEuler` at the pinned ref). A host driver older than the image's CANN
   line is the recorded first lever: #5236 is reporter-confirmed — a device-side
   `aclnnRmsNorm` execution error on Qwen3-235B was "solved by updating the npu driver to
   25.25.0". Keep the two apart when citing: that report never reached a link creation,
   while #4148 ran on `25.2.0` and its outcome was never confirmed.
2. **Reproduce the edge set without vLLM** — 2 ranks on the two chips of one card (expect a
   pass) and 4 ranks spanning both cards (expect the same `DestLink`), then swap the
   pairing. One `torch.distributed` allreduce or `hccl_test` run decides
   "host transport" vs "engine", and it is the cheapest artefact to ask for.
3. **Container device passthrough.** The 300I tutorial's `docker run` passes every
   `/dev/davinciN` *and* `/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc` plus the
   driver mounts (`docs/source/tutorials/single_node_300i.md`), whereas
   `--runtime=ascend` + `ASCEND_VISIBLE_DEVICES=0-3` relies on the runtime. A
   `/usr/local/Ascend/driver` mounted from a host of another driver version is the same
   pairing question as rung 1, asked of the container.
4. **Per-device health and the plog** for the fault second (`asys health -d=$i`, then
   `plog-timeline.sh`): a link-level code (`HCCS_PORT`, `Link diagnosis exception`) on either
   card makes it hardware — `device-health-checks.md`; do not keep describing it as a
   driver version.
5. **Then the intra-node transport knobs**, one axis per launch: `HCCL_INTRA_PCIE_ENABLE` /
   `HCCL_INTRA_ROCE_ENABLE`, `HCCL_IF_*`, `HCCL_BUFFSIZE`. They rank below the driver
   pairing because they cannot create the cross-card shmem handle the driver refused.

## Precedents and limits

- `#486`: the identical `[Create][DestLink]Create Dest error!` text on a training +
  co-located-vLLM multi-NPU host — precedent for the wording, and for the address field
  being printed from the local rank on both sides.
- `#4148`: the shape above (cross-card only, both card-local edges up); the reporter only
  acknowledged the answer, so which of rungs 1-3 fixed it is **not** established.
- `#5236`: device-side operator execution error resolved by a driver update (see rung 1).
- `cann/hccl` is mostly headers and the transport body is not exported; one grep in
  `cann/hcomm` for a distinctive fragment (e.g. `DestLink`, `IpcMemImportByKey`), then work
  from the driver pairing, the edge set and the plog instead of more source.
