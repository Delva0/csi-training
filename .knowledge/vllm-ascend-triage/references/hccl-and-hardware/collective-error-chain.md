# Collective error chain: from a vLLM traceback to the real failing op

## Step 1 — assume the Python frame is wrong

Ascend operators run asynchronously. A collective failure is reported by the
next operation that forces a synchronization (often a device-to-host copy), so
the traceback shows the copy, not the collective. Torch_npu says so explicitly in
the message:

- `RuntimeError: The Inner error is reported as above. The process exits for
  this inner error, and the current working operator name is HcclAllreduce.`
- `Since the operator is called asynchronously, the stacktrace may be inaccurate.
  If you want to get the accurate stacktrace, please set the environment variable
  ASCEND_LAUNCH_BLOCKING=1.`

Recorded instance (vllm-ascend issue #7477, v0.17.0rc1, TP=4, Qwen3.5-27B): the
traceback ended at
`vllm_ascend/ops/mm_encoder_attention.py:108  seq_lens_cpu = torch.diff(cu_seqlens).to("cpu")`
inside the vision encoder, while under blocking mode the same failure landed in
`vllm_ascend/ops/vocab_parallel_embedding.py:192 _forward_origin` →
`maybe_pad_and_reduce` → `tensor_model_parallel_all_reduce`. The vision frame was
an artifact of the sync point. (Both file/line references come from the
reporter's logs against the installed vllm-ascend v0.17.0rc1.)

## Step 1b — the sync point can be a compile-phase sample kernel

The sync point does not have to touch model data. While `torch.compile` runs,
vllm-ascend registers inductor fusion patterns whose *example inputs are real
device tensors*, so registration itself launches kernels on the device. The first
of those launches is a legitimate sync point, and it reports a device/collective
fault that happened earlier. Verified at vllm-ascend `155f689` (v0.26.0rc2 head)
with vllm v0.26.0 `568afb3a`:

- vllm `vllm/compilation/backends.py:934` `configure_post_pass`, called at `:1171`.
- `vllm_ascend/compilation/graph_fusion_pass_manager.py:54` appends
  `AddRMSNormQuantFusionPass` unless `ascend_compilation_config.fuse_norm_quant` is
  false (and not 310P); `fuse_qknorm_rope` / `fuse_muls_add` are appended the same way.
- `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:716` registers
  `AddRMSNormDynamicQuantPattern`; `get_inputs()` allocates
  `torch.randn(2, 4, device="npu", ...)` (`:291`) and the pattern body calls
  `torch.ops.npu.npu_add_rms_norm` (`:301`).
- `vllm_ascend/compilation/passes/base_pattern.py:49-60` hands those example inputs
  to `pm.register_replacement` (`:53`) and `nge.register_replacement` (`:55`).

So a `determine_available_memory` → `profile_run` → compile traceback whose deepest
frame is `torch.ops.npu.npu_*` on 2×4 or 2×64 tensors is **not** an operator bug:
those sizes exist nowhere in the model. That frame is where a pre-existing device or
HCCL fault surfaced. Do not walk tiling/shape code for it.

Recorded instance #17357 (image v0.26.0rc1-a3, DeepSeek-V4-Pro w4a8, A3 two-node
TP16/DP2): only `Worker_DP0_TP7_EP7` reported, at `worker.py:546
determine_available_memory` → `model_runner_v1.py:3681 profile_run` → … →
`norm_quant_fusion_pass.py:716` → `:301`, with `RuntimeError: SUSPECT REMOTE ERROR`
and `ERR00100 PTA call acl api failed`. Same stage, same error family in the history
corpus: #9503 (`npu_config.py:102 set_compile_mode` → `torch_npu.npu.synchronize()`
inside `BackendCompilerFailed`, error 507057) and #12461 (A3 two-node TP16/DP2,
`determine_available_memory` → `profile_run`, single rank `Device:6`).

## What `SUSPECT REMOTE ERROR` (507057) is

- It is a **device-error category**, not an operator-argument or tiling error:
  torch_npu's `get_device_error()` matches a fixed list in the `Ascend/pytorch` export
  (`torch_npu/csrc/distributed/ProcessGroupHCCL.cpp:526-544`) — `UCE ERROR`,
  `HBM MULTI BIT ECC ERROR`, `SUSPECT MEM ERROR`, `HCCS LINK ERROR`, `UB LINK ERROR`,
  `HCCL OP RETRY FAILED`, `SUSPECT REMOTE ERROR` — and the message carries
  `error code is 507057`.
- vllm-ascend uses the same pairing for an HCCL abort caused by rank-dependent buffer
  sizing: "ranks allocate mismatched buffers / at different times and HCCL aborts
  (SUSPECT REMOTE ERROR 507057)", `vllm_ascend/ops/fused_moe/moe_comm_method.py:320`.
  So a rank-divergent collective is a real software candidate, alongside hardware.
- The reporting rank is not necessarily the faulting chip: in #10829 node0 printed
  `(Device:0) ERR00100 PTA call acl api failed` and, in the same output, `EZ9999 …
  The error from device(chipId:4, dieId:0) … fftsplus aivector error, error code =
  0x800000`; node1 printed only the bare `SUSPECT REMOTE ERROR`. Search every rank
  for the `EZ9999` / `chipId:` line and the earliest plog ERROR before naming a card
  (see `device-health-checks.md`).

## Step 2 — re-run to get attribution

```
vllm serve ... --enforce-eager            # removes graph capture as a confounder
ASCEND_LAUNCH_BLOCKING=1 vllm serve ...   # synchronous ops: real failing op/frame
```
Unset `ASCEND_LAUNCH_BLOCKING` after debugging (documented performance penalty).

Blocking mode is *refused* while ACL graph is on: `AscendPlatform.check_and_update_config`
(`vllm_ascend/platform.py`; `:457-469` at v0.20.2rc1, elsewhere `:601-613` — grep the
message) raises `ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1`
whenever `compilation_config.cudagraph_mode != NONE`. Add `--enforce-eager` first, then re-run
with `ASCEND_LAUNCH_BLOCKING=1`; otherwise the flag does not take effect at all.

## Step 3 — read the plog

`~/ascend/log/debug/plog/plog-<pid>_<timestamp>.log` carries the HCCL-side story.
Shapes seen in the recorded case:

```
[ERROR] HCCL(...): [all_reduce_opbase_pipeline.cc:327] [AllReduceOpbasePipeline][Prepare]rank[0]
        streamNotifyMain_ size [2] error, is smaller than,intraRankSize_[4]
[ERROR] HCCL(...): [coll_all_reduce_mesh_opbase_pipeline_executor.cc:116] [KernelRun]call trace: hcclRet -> 4
[ERROR] HCCL(...): [coll_all_reduce_executor.cc:302] ... errNo[0x0000000005000004]kernel run error,
        tag[AllReduce_group_name_3], count[75497472], dataType[11], reduce op type[0]
[ERROR] APP(...): ExecFuncOpApi:452: "[PTA]:Custom hand fail! name=HcclAllreduce, ret=4"
[ERROR] APP(...): ReadQueue:437: "---Thread---...: device = 0, write_idx = 627, read_idx = 602, status = 1, ret = 4"
```

How to read that:

- `tag[AllReduce_group_name_N]` names the process group; check its rank set
  against the devices you exposed (`ASCEND_RT_VISIBLE_DEVICES`) and the
  `device = N` field in the PTA line. A group whose ranks include a faulty card
  fails as a group.
- `streamNotifyMain_ size [2] ... intraRankSize_[4]` is an HCCL-side resource
  inconsistency inside its opbase pipeline executor (notify objects per
  intra-node rank vs the node's rank count). It is not evidence about model
  shapes or which layer issued the collective.
- `errNo 0x...05000004` / `hcclRet -> 4` is CANN's internal error number; the
  human-readable cause string is what you quote, because the numeric meanings are
  version-specific and the pipeline sources are not in the public exports.
- `ERR00100 PTA call acl api failed` (blocking mode) and `ERR01100 OPS call acl
  api failed` (with `HcclAllreduce, error code is 4`) are PTA/runtime-level
  wrappers of the same HCCL failure.

## Step 4 — decide where the fault lives

Collective failures at the very first collective of `profile_run` mean the group
cannot communicate, which is environment-level. Take the reporting `device = N` id
*and* the `chipId` from any `EZ9999` dump (they can differ), then go to
`device-health-checks.md` before proposing any model or operator patch. Also confirm
that single-card work on each device still succeeds — HCCS link faults commonly
break only intra-node multi-card collectives. Two cheap non-hardware checks follow
below before you tell the reporter to replace a card.

## Software A/B for a compile-phase sync point (diagnostic, not a fix)

`--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'` skips
construction of `AddRMSNormQuantFusionPass` (`graph_fusion_pass_manager.py:54`) and so
skips that sample kernel. If startup proceeds, it confirms the frame was only the sync
point; a real device/link fault then reappears at the first collective. Offer it as an
A/B experiment, never as the repair.

## Multi-node DP start: check the addresses before the hardware

On a DP (or PD) multi-node start, a collective failure at the first group operation
can be a config error, not a fault. The env template the reporters copy says
"node0_ip must be the local_ip of Node0 on every node", yet #17357's pasted block has
`node0_ip="enp189***"` — the NIC name, next to `local_ip=194.x.x.11` — while
`--data-parallel-address` is the head node's *address*
(`vllm/engine/arg_utils.py:1063-1066`). Ask for both nodes' `local_ip`, `nic_name`,
`HCCL_IF_IP`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`, node1's
`--headless --data-parallel-start-rank`, and matched driver/CANN versions. None of it
is verifiable locally, so present it as an ordered check that costs one paste.
