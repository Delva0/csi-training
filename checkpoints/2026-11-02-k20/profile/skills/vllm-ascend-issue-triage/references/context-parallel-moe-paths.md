# Context parallel, EP and MoE paths in vllm-ascend

Scope of the verification: tag `v0.18.0rc1` of `/repos/vllm-project/vllm-ascend`, checked for the
DeepSeek-V3.2 W8A8 report with TP=8, PCP=2, DCP=1 and `--no-enable-expert-parallel`. Line numbers
move and later releases reorganize these modules; re-anchor each path with
`git grep -n <symbol> <ref> -- '*.py'` before quoting it. The C8/DSA-CP flag cluster that
later releases expose on top of this gating is covered in
[c8-dsa-deployment-flags.md](c8-dsa-deployment-flags.md); proposer metadata that a release never
populates under PCP/DCP + MTP is mapped in
[spec-decode-metadata-failures.md](spec-decode-metadata-failures.md). A PCP report that dies on
start-up/first request inside `attention/context_parallel/attention_cp.py` is usually a graph-mode
boundary instead: with `cudagraph_mode: FULL` (not `FULL_DECODE_ONLY`) the params update dereferences
`decode_meta`, which is `None` on a prefill-only step — chain and workaround in
[full-decode-only-graph-mode.md](full-decode-only-graph-mode.md). For a MoE *accuracy* collapse
(not a hang) that appears only with `data_parallel_size` / expert parallel, start with
[moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md); this file maps the dispatch paths a hang or
collective mismatch runs through. A CP/FlashComm run that is *slow* rather than wrong, at a specific
number of running sequences and with no error, is the throughput-cliff class instead:
[throughput-and-concurrency-cliffs.md](throughput-and-concurrency-cliffs.md).

## Which context-parallel path runs

- `vllm_ascend/attention/utils.py`: `enable_cp()` is true when
  `prefill_context_parallel_size > 1 or decode_context_parallel_size > 1`.
- `vllm_ascend/attention/sfa_v1.py`: the SFA metadata builder imports
  `attention/context_parallel/sfa_cp.py` (`AscendSFACPMetadataBuilder`, `AscendSFACPImpl`) when
  `enable_cp()`; without CP the non-CP SFA implementation stays in `sfa_v1.py`.
- At this tag `attention/context_parallel/` holds only `__init__.py`, `attention_cp.py`,
  `common_cp.py`, `mla_cp.py`, `sfa_cp.py`. Files visible at `main` (`dsa_cp.py`,
  `dsa_v41_cp.py`, `worker/v2/pcp_manager.py`) do not exist there — check the ref's tree first.
- DeepSeek-V3.2 is recognized as a sparse-attention model by `hf_text_config` exposing
  `index_topk`; that test appears both in `enable_dsa_cp()` and in the platform checks.

## How EP changes the numerics path around sparse attention

- `enable_dsa_cp() = sparse model AND enable_sp()` (`utils.py`).
- `enable_sp()` = `VLLM_ASCEND_ENABLE_FLASHCOMM1`, or legacy `VLLM_ASCEND_ENABLE_FLASHCOMM=1`, or
  `shared_expert_dp`.
- `platform.py` (near line 477 at this tag): when `enable_sp()` is true it asserts `tp_size > 1`
  and, for a MoE model, `enable_expert_parallel` — message
  "Flash Comm v1 requires enable_expert_parallel=True for MoE models." This is a start-up platform
  check, **not** part of `enable_sp()`'s body: `enable_sp()` itself only reads the env /
  `shared_expert_dp` (see the PCP-without-EP bullet below), so "EP is off" does not by itself mean
  `enable_sp()` is false.
- Therefore a MoE model running with EP disabled cannot enable FlashComm/SP, and
  `enable_dsa_cp()` is false. Everything conditionally built on it is off:
  - `enable_dsa_cp_with_layer_shard()` — true only on a PD prefill instance
    (`kv_transfer_config.is_kv_producer`).
  - `enable_dsa_cp_with_o_proj_tp()` — true only when `kv_transfer_config is None` (non-PD).
  - In `attention/sfa_v1.py` these gates select the o_proj full-gather / layer-shard branches and
    their async ops.
  - In `ops/linear_op.py` they select `ShardedCPColumnParallelOp` for `q_b_proj`/`kv_b_proj`,
    `ShardedCPRowParallelOp` for `o_proj`, and a pad-skip special case for `o_proj` under flash
    comm; `distributed/parallel_state.py` has a matching layer-shard branch.
- The CP attention files themselves contain no `enable_expert_parallel` or EP-group references at
  this tag (verified by `git grep`), so the EP dependence sits in the linear/SP/MoE plumbing
  around attention, not in CP metadata construction.

## MoE dispatch

`vllm_ascend/ops/fused_moe/moe_comm_method.py` registers ALLTOALL, ALLGATHER, MC2 and FUSED_MC2
implementations. Its docstrings state that MC2 and AlltoAll require
`enable_expert_parallel=True` (MC2: "`enable_expert_parallel=False` is not supported") and that
AllGather "should be compatible with all scenarios" and is the default. EP off therefore forces
the AllGather dispatch path.

Era note for the 0.10 line, because the module layout differs: at `v0.10.0rc1` the all2all MoE path is
`quantization/w8a8_dynamic.py::fused_experts_with_all2all`, which is host-synchronous by construction
(CPU round-trips for expert token counts and the packed output shape). It therefore cannot run inside
ACL-graph capture — that is the `107027` / `reason=[stream is captured]` start-up crash of the
Qwen3-MoE + EP reports, not a collective or device fault
([log-signatures.md](log-signatures.md)). From `v0.10.1rc1` the implementations live in
`distributed/moe_comm_method.py`, and `ops/common_fused_moe.py::forward_impl` overrides the selected
method to `allgathercommimpl` whenever `ep_size < 16`, so a small-EP run at that tag does not use
`AlltoAllCommImpl` at all.

Which of those implementations a given forward uses is decided per rank by
`vllm_ascend/ascend_forward_context.py: select_moe_comm_method(num_tokens, ...)` (A3: MC2/FUSED_MC2
up to a 512-token capacity per rank, ALLTOALL above it; A2: MC2 only for small EP-per-device and
`ep_world_size >= 16` from `v0.18.0rc1` — the `v0.13`/`v0.17` gate is
`world_size_across_dp / pipeline_parallel_size >= 16`; no EP: ALLGATHER). A DP group whose
ranks compute different `num_tokens` can therefore issue different collectives in one HCCL
group — the decode-hang candidate chain and its scheduler triggers are in
[pd-disaggregation.md](pd-disaggregation.md).

### Where the selector lives, and why its frame does not name the method

- `v0.12.0rc1`: the per-forward choice is `NPUModelRunner._select_moe_comm_method(num_tokens)`
  (`vllm_ascend/worker/model_runner_v1.py:1373`), also called by the eagle/MTP proposers and from
  the graph-size computation (`mc2_tokens_capacity`). Its docstring states the rule: no EP ->
  ALLGATHER; A2 with EP -> MC2 when `num_tokens <= get_mc2_tokens_capacity()` and
  `world_size_across_dp / pipeline_parallel_size >= 16`, else ALLGATHER; A3 -> MC2/all-to-all.
- `v0.13.0rc1`: the same decision moved to `vllm_ascend/ascend_forward_context.py:196
  select_moe_comm_method(...)`. Grep both spellings before concluding a tag has no selector.
- The implementations choose their prepare/finalize partner in
  `ops/fused_moe/moe_comm_method.py::_get_prepare_finalize`: `AllGatherCommImpl ->
  PrepareAndFinalizeWithAllGather`, `AlltoAllCommImpl` and `FusedAlltoAllCommImpl ->
  PrepareAndFinalizeWithAll2All`, `MC2CommImpl -> PrepareAndFinalizeWithMC2`, and
  `PrepareAndFinalizeWithMC2` **subclasses `PrepareAndFinalizeWithAll2All`** and inherits its
  `finalize()`. So a traceback frame `PrepareAndFinalizeWithAll2All.finalize` does not prove
  ALLTOALL — an MC2 run reports the same frame.
- At `v0.12.0rc1` that inherited `finalize()` pads the token count up to `tp_size`, splits the
  hidden states with `torch.tensor_split(..., self.tp_size)`, calls
  `dist.all_gather(list(split_hidden_states), hidden_states, tp_group.device_group)`
  (`ops/fused_moe/prepare_finalize.py:191`, the line recorded reports quote) and truncates back to
  `self.num_tokens`. The collective is inside the TP group, so cross-DP token divergence is not the
  mechanism there.

On a 16-card EP+DP run on 910B the `world_size_across_dp / pipeline_parallel_size >= 16` half of the
gate holds by construction, so whether a given forward is MC2 or ALLGATHER reduces to
`num_tokens <= get_mc2_tokens_capacity()` — a per-step decision that flips between stages. The
reporter's two "workarounds" (drop `--enable-expert-parallel`, or DP=1) instead force ALLGATHER for
every step, which is why they look like a fix without identifying one. When such a report dies at
**start-up** — `No available shared memory` then
`RuntimeError: Engine core initialization failed` with the `prepare_finalize.finalize` frames in the
same log — the raise happens during the initial profile/graph capture, so read the MC2 dispatch and
its HCCL comm-resource allocation (`npu_moe_distribute_dispatch*`, see the fault section below)
instead of decode-time scheduling. No root cause was established for that report (thread open, only
an automated follow-up), so keep it labelled open.

## FC1 (Flash Comm v1) enablement: per-forward gate, and the documented support boundary

FC1 is not a switch that is on for a whole run; `ascend_forward_context.py` decides per forward
(at v0.19.1rc1, lines 110-129):

- MoE model (or a MoE drafter): `flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is
  not None` — no token threshold, and `mmrs_fusion` is forced off.
- Draft model: forced `False` (`"for dense drafter, sp is redundant"`).
- Dense model: `enable_sp(vllm_config) and num_tokens > 1000` — an empirical concurrency threshold,
  so the same flag is active in a large prefill and idle in a small decode step.
- When FC1 or FlashComm2 is enabled the runner also pads to a TP multiple
  (`pad_size = (tp_world_size - num_tokens % tp_world_size) % tp_world_size`), so token counts at
  the model differ from tokens scheduled. Quote the gate that applies to the reported model class
  before claiming "the flag was on".

The documented matrix bounds it and did **not** change between the reported working and failing
releases: `docs/source/user_guide/feature_guide/sequence_parallelism.md` has the same Support
Matrix at `v0.18.0rc1` and `v0.19.1rc1` — FC1 `eager/graph` for non-VL + Dense and non-VL + MoE,
`x` for both VL classes, and the With-Quantization table is identical. There is no row for a
hybrid GDN / linear-attention model. So "worked on the older tag, fails on the newer" is not a
documented-support regression by itself; check the model class against the matrix first.

Recorded evidence for that boundary (`vllm-ascend#8990`, Qwen3.5-27B w8a8 + MTP, 910B A2, TP=2,
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`): the maintainer reply is "vllm-ascend does not support Flash
Comm V1 for 27B model" (pointing at the same doc), and a second reporter adds Qwen3.5-9B failing
the same way and disabling the variable as the workaround. The pasted failure is a decode
cudagraph-capture abort whose last Python frame is the compiled-graph call and whose message is

```
expected src to have a size equal to the slice of self. src size = [64, 5120], slice size = [32, 5120]
```

- That message is not raised by `copy_` or `index_copy_`: the string lives in the `*_scatter`
  family in `aten/src/ATen/native/TensorShape.cpp` (`slice_scatter`/`select_scatter`/
  `diagonal_scatter`/`as_strided_scatter`), mirrored in `torch/_prims/__init__.py`, and
  `FunctionalInverses.cpp` rewrites an in-place slice write into `at::slice_copy_symint`. Read it
  as "an in-place write into a slice with the wrong row count", dimension 0 first.
- With `64 = num_tokens` and `32 = num_tokens/TP` for TP=2, the shape pair matches FC1's sharded
  token layout, but no raising site was confirmed here: the PP+SP
  `sync_and_slice_intermediate_tensors` slice write needs pipeline parallelism, which the report
  does not use. Treat the exact statement as open and look for a token-dim slice write in the
  GDN/MTP path.
- The gate above does not settle whether FC1 was active in that step: the dense-model `num_tokens >
  1000` test is evaluated with the runner's token count, which at graph capture need not equal the
  64 scheduled decode tokens the shape implies. So neither "FC1 was on" nor "FC1 was off" follows
  from the message alone — ask for the worker log line that reports the flag, or for a run with the
  variable unset.
- Workaround to report: unset `VLLM_ASCEND_ENABLE_FLASHCOMM1` (supported by the maintainer
  statement and the second reporter); note that the >1000-token dense gate above means the flag
  only bites on large steps, which is why small prompts can look fine.

## SP/FC1 MoE finalize on the MTP drafter: `4096 vs 8192` at `final_hidden_states[0] + final_hidden_states[1]`

Report shape: MoE model + `--speculative-config '{"method": "qwen3_5_mtp", ...}'` +
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, server dies during `profile_run` →
`drafter.dummy_run` → `_run_merged_draft` → `vllm_ascend/patch/worker/patch_qwen3_5.py` `mlp` →
`vllm/model_executor/models/qwen3_next.py:358` (at vLLM `v0.18.0`)

```text
final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
RuntimeError: The size of tensor a (4096) must match the size of tensor b (8192) at non-singleton dimension 0
```

- Both addends come from the fused-MoE layer itself (`self.experts(...)` returns the two-element
  tuple holding the shared-expert and routed-expert outputs; the
  `tensor_model_parallel_all_gather` that would reunite the shard runs
  *after* the add, `vllm/model_executor/models/qwen3_next.py:352-361` at `v0.18.0`). Under EP/SP the
  MoE finalize is contracted to return the **local** EP/SP token shard: `_finalize_with_ep_group`
  (`vllm_ascend/ops/fused_moe/prepare_finalize.py:441`) calls
  `torch.ops.vllm.maybe_pad_and_reduce(hidden_states, True)` (:451, i.e. `is_ep_comm=True`) and its
  docstring says the reduce-scatter result is then added to the shared-expert result. So the
  mismatch is one branch returning the **global** layout: read the ratio, not the numbers — a
  `num_tokens` vs `num_tokens / (TP·EP·SP)` pair explains any values (8192/4096 at TP=2 here; the
  fixing PR records 4096/1024 in another topology), and the larger operand is normally the branch that
  skipped the scatter.
- Verified at `v0.18.0rc1`: `vllm_ascend/ops/register_custom_ops.py:83`
  `if not flash_comm_v1_enabled or (forward_context.is_draft_model and is_vl_model()):
  return tensor_model_parallel_all_reduce(x)` — a legacy draft-VL bypass that returns the global
  layout with no `not is_ep_comm` restriction, so it fires exactly on the EP/SP finalize that needs
  the shard. `spec_decode/eagle_proposer.py` (`_run_merged_draft` chain) had a matching multimodal
  early return that skipped pad-and-reduce entirely.
- Applicability check: the bypass needs **both** `is_draft_model` and `is_vl_model()`. Confirm the
  reported checkpoint is multimodal in the plugin's sense (`is_vl_model()` / its config) before
  assigning #7683; on a text-only model with the same ratio, read `prepare_finalize.finalize` and
  its caller instead of assuming this bug.
- Fix: `880e20fdd` (#7683, merged 2026-04-13, body `fix #7996`) adds `and not is_ep_comm` to the
  bypass and replaces the early return with
  `if _EXTRA_CTX.flash_comm_v1_enabled and not self.is_multimodal_model:`. First tag in the clone
  containing it: `v0.19.1rc1`. Backport `d94b1dc2d` (#8004) is in the `v0.18.0` release tag but
  **not** in `v0.18.0rc1`, so on an rc of that line the fix is absent.
- The guard regressed afterwards: `v0.28.0.rc1` calls `_maybe_pad_and_reduce_impl(x)` without
  `is_ep_comm` and keeps `if _EXTRA_CTX.is_draft_model and is_vl_model():`; `083d49858` (#16081,
  "Fix MTP MoE finalize layout with upstream SP") deletes the bypass at `v0.29.0.rc1`. A same-ratio
  mismatch reported between 0.19 and 0.28 is therefore this bug with a different guard, and on 0.28
  the bypass is unconditioned again — check `git show <tag>:vllm_ascend/ops/register_custom_ops.py
  | grep -n is_draft_model` before answering which side is wrong.
- Workaround to report: unset `VLLM_ASCEND_ENABLE_FLASHCOMM1` (the issue's own reproducer then
  starts with MTP), or move to a tag carrying the fix. The FC1 support matrix above has no VL row,
  so a VL model in this configuration is outside documented support as well.

## MC2 MoE dispatch fault: `aclnnMoeDistributeDispatchV3`/`V4` / `HcclAllocComResource* ret = 4`

The dispatch operator a forward reaches is chosen in
`ascend_forward_context.py: select_moe_comm_method`, and the A2 rule changed shape between the 0.13
and 0.18+ lines — read it at the reported tag rather than from memory:

- no `--enable-expert-parallel` (or EP group world size 1) -> `ALLGATHER` (all tags);
- `v0.13.0rc1`: A2 + EP + `num_tokens <= get_mc2_tokens_capacity()` +
  `world_size_across_dp / pipeline_parallel_size >= 16` -> `MC2`; otherwise `ALLGATHER`;
- `v0.18.0` through `v0.21.0rc1` (`ascend_forward_context.py:262-288` at `v0.21.0rc1`): A2 + EP +
  `num_experts_per_device <= 24` (`num_experts // ep_world_size`) + `ep_world_size >= 16` +
  `num_tokens <= mc2_tokens_capacity` -> `MC2`; otherwise `ALLGATHER`. That per-device cap comes from
  fix `5645ca839` (#7364, "For A2, the number of experts each single card cannot be greater than 16
  when using MC2"), contained from `v0.18.0rc1`/`v0.18.0`; its purpose is the opposite direction —
  a model with more experts per card (>256 experts at EP 16) now falls back to `ALLGATHER` instead of
  taking MC2;
- A3 adds `FUSED_MC2` under `VLLM_ASCEND_ENABLE_FUSED_MC2`. The code guard is
  `dispatch_ffn_combine_enable = get_ep_group().world_size <= 32 and (not is_draft_model)`
  (`ascend_forward_context.py:287` at `v0.20.2rc1`/`v0.22.1rc1`, `:254` at `v0.18.0rc1`; at
  `v0.23.0` the line is `:264` and the draft-model condition is gone) — not `EP <= 16`, which the
  tag's docs do not state either; the value-`1` comment block adds W8A8 MoE layers, `EP <= 32`,
  non-MTP and non-dynamic-EPLB (`envs.py:95-102` at `v0.22.1rc1`, with the prerequisites of both
  values in [env-vars-and-additional-config.md](env-vars-and-additional-config.md)).

`MC2`/`FUSED_MC2` are the paths that call `torch_npu.npu_moe_distribute_dispatch_v2` (when
`hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")`) or the non-`_v2` variant
(`ops/fused_moe/token_dispatcher.py:202-206`), i.e. the fused communication kernels that allocate
an HCCL MC2 comm resource. So a TP=16/EP A2 report that dies at start-up maps to MC2 by
configuration alone.

Signature mapping, all verified by reading the clones (not by running on NPU):

- `aclnnMoeDistributeDispatchV3 failed, error code is 561000` names the aclnn kernel behind
  `npu_moe_distribute_dispatch_v2`; the traceback's
  `third_party/op-plugin/op_plugin/ops/opapi/MoeDistributeDispatchV2KernelOpApi.cpp:161` is
  `op_plugin/ops/opapi/MoeDistributeDispatchV2KernelOpApi.cpp` in the `Ascend/op-plugin` clone
  (`third_party/op-plugin/` is torch_npu's vendoring prefix). The torch API suffix (`_v2`) and the
  aclnn suffix (`V3`) differ — grep both spellings before concluding a symbol is absent.
- The same op-api file also dispatches `aclnnMoeDistributeDispatchV4` when the CANN version offers
  it (`op-plugin` `MoeDistributeDispatchV2KernelOpApi.cpp:279
  if (check_aclnn_kernel_available("aclnnMoeDistributeDispatchV4"))`, with the
  "not supported in this CANN version" message at `:316` and `:115`), so `...V4 failed, error code is
  561000` is the *same* operator class on a newer CANN, not a different bug: at `v0.21.0rc1` on a
  16x910B2C single node the pasted frames are `MoeDistributeDispatchV2KernelOpApi.cpp:184` (V4) plus
  `Nnopbase ... HcclAllocComResourceByTiling ... ret = 4`. Do not gate the route on the `V3` suffix.
- `HcclAllocComResourceByTiling ... ret = 4` is `HCCL_E_INTERNAL` (`cann/hcomm/include/hccl/hccl_types.h`:
  0 success, 1 PARA, 2 PTR, 3 MEMORY, 4 INTERNAL, 5 NOT_SUPPORT, 6 NOT_FOUND), not a parameter or
  memory error. The symbol is `cann/hcomm/src/legacy/ascend910/framework/op_base/src/op_base_mc2.cc:219`
  (declared in `pkg_inc/legacy/hccl/hccl_ex.h:201`, class method in
  `framework/communicator/hccl_comm.cc:1290`). `/repos/cann/hccl` at the clone snapshot contains no
  such symbol — it is a partial mirror (top commit "feat: superkernel迁移") — so a grep miss there
  is not evidence that the API does not exist. These hosts are also ahead of the reporter's CANN
  8.3.RC2, so the enum is the documented value, not their binary's.
- The Python frames above it (`patch_compile_backend.py __call__ ->
  compiled_graph_for_general_shape`, and `determine_available_memory`) are where an asynchronous
  device fault surfaces; at `v0.13.0rc1` that statement really is
  `self.compiled_graph_for_general_shape(*args)`, so the pin is confirmed but the frame is not the
  bug. Set `ASCEND_LAUNCH_BLOCKING=1` to move the report to the real op.
- A device-side `Resource_Error(EI0007): Failed to allocate resource qp with info aicpu rank:N,
  localUserrank:N, ..., sendCqDepth:32768. Reason: Resources are exhausted.` next to the `561000`
  error means the MC2 communicator never got its queue pairs — a comm-resource exhaustion, not a
  tensor-shape or tiling fault. Candidate (needs the plog): 16 ranks on one node each allocating an
  MC2 resource with that CQ depth. On a single-node A2 box the documented route already avoids this
  class: the GLM5 tutorial says "For single-node deployment, we recommend using `dp1tp16` and turn
  off expert parallel in low-latency scenarios" (`docs/source/tutorials/models/GLM5.md`, present at
  both `v0.18.0` and `v0.21.0rc1`), and dropping `--enable-expert-parallel` forces `ALLGATHER`
  (rule above) so no MC2 op is launched at all.
- Workaround candidates, unverified on hardware: drop `--enable-expert-parallel` (forces
  `ALLGATHER`, no MC2 op), or change the topology so the A2 EP gate above is not met.
  `vllm-ascend#5680` was closed with zero comments and no linked commit, and #10944 (same signature
  at `v0.21.0rc1`, single node 16x910B, `--tensor-parallel-size 16 --enable-expert-parallel`,
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, MTP x3) is closed too, so no maintainer root cause exists for
  this signature; present these as the source-supported route plus the device-side evidence still
  needed (per-rank plog, `ASCEND_LAUNCH_BLOCKING=1` stack, CANN fix status for the `561000` op).

### Adjacent routes that also end in a dispatch-communication failure

- **`HcclAllocComResource ... ret = 4` reached through a dead host NIC, not through the op's
  arguments** (`Getting socket times out` / `RuntimeError: EJ0002 ... The network port is down`
  while a torchair graph loads). Read the plog bottom-up and stop at the first line:
  `ra_hdc_rdma ... ret(-67)` and `ra rdma init failed, devid[N] ip[<ip>]` ->
  `ra nic init rdma failed, devid[N]` -> `HcclSocket Listen ... listen on ip[<ip>] and specific
  port[16666] fail` -> `wait socket establish timeout ... role[1] rank[N]`, whose
  LINK_ERROR_INFO table shows that rank with `src_port 0` and status `time out` ->
  `CreateComm failed. result: Level0[0] Level1[0] Level2[0] Level0rdma[0] Level1rdma[1]` (the only
  entry that is not `0` is the `Level1rdma` plane, matching the RDMA init failure above) ->
  `CreatecomResource_group_name_53_MC2MultiServer` ->
  `HcclAllocComResource ... ret 4` -> GE `LoadGraph failed, error code:1343225857`.
  The rank that cannot listen is the faulty one; its RoCE/optical state is the finding, checked
  with the `hccn_tool -i <i> -optical -g`, `-ip -g`, `-lldp -g` commands HCCL's own message
  prints. Torchair is what creates and records these process groups while lowering MC2 ops into
  the GE graph (`Ascend/torchair`, `python/torchair/_ge_concrete_graph/hcom_utils.py:7-30`,
  "special for op like MC2, who is not record process group in convert"), which is why a
  `torchair_graph_config` run allocates an MC2 multi-server comm at load.
  Recorded #1955 (`v0.9.2rc1`, single node 16x910B2C, DP4/TP4, `--enable-expert-parallel`,
  `HCCL_INTRA_PCIE_ENABLE=1`/`HCCL_INTRA_ROCE_ENABLE=0`, `torchair_graph_config` on): those intra
  flags did not spare the RDMA plane, and the failing device was `devid[12] ip[192.168.20.160]`,
  i.e. one of the upper NPUs of the 16-device box (the device index, not a claim about physical
  partitioning). The launcher also sets `HCCL_IF_IP=127.0.0.1` while the
  plog shows per-device IPs — have that changed to the node's real NIC IP as hygiene, not as the
  proven cause. The issue was closed 2026-05 as "TorchAir is now deprecated" with one comment,
  "It looks like a network card issue", so no maintainer root cause exists: answer with this log
  ordering plus the NIC checks. Avoiding the MC2 comm (`--enable-expert-parallel` off, or the
  documented A2 `dp1tp16` recipe) is the configuration route and stays a candidate until the NIC
  is verified.
- **`aclnnMoeDistributeDispatchV4 ... error code is 107020` beside a task timeout is the same
  operator class as `561000`, one layer down.** GLM-5.3-w8a8c8 on 4x A2 (title says `v0.27.1rc1`,
  env block `v0.26.0rc1` — resolve which one ran): the paste carries
  `event wait timeout ... timeout=230s`, `rtNotifyRecord ... task_timeout` and, from another
  engine, `No available shared memory broadcast block found in 60 seconds`, i.e. a rank hung
  inside the dispatch rather than a kernel rejecting an argument
  ([log-signatures.md](log-signatures.md), `aicore execution times out`). Two cheap checks come
  before any kernel work: (i) the tag's own `docs/source/tutorials/models/GLM5.3.md` A2 block
  prescribes both variables the reporter removed (`HCCL_OP_EXPANSION_MODE="AIV"`,
  `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`) together with `--enable-expert-parallel`,
  `cudagraph_mode: FULL_DECODE_ONLY`, `HCCL_EXEC_TIMEOUT=200`,
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000` and `VLLM_RPC_TIMEOUT=360000`, so "removing the two"
  is a two-variable deviation from the documented recipe and the alloc variable is re-applied by
  the plugin anyway ([env-vars-and-additional-config.md](env-vars-and-additional-config.md));
  (ii) the same tag's release notes already carry the model-class workaround — the `v0.21.0rc1`
  Known Issues (so every later file repeats it) say GLM-5.1 with Expert Parallel plus FULL graph
  hits `MoeDistributeDispatchV2`/NPU graph failures with the reported workaround "disable Expert
  Parallel for FULL graph mode, or use PIECEWISE/eager" (#9503), and name the
  `MoeDistributeDispatchV2`/`aclnnMoeDistributeDispatchV4` AICore-timeout hang in 200K 1P1D
  workloads (#9958). State the model/node/CANN delta; the report's thread holds one ping, so
  nothing is confirmed for GLM-5.3.

## Candidate mechanism for "PCP accuracy drops only without EP" — unverified

Direction: with EP off, the run pairs PCP attention with AllGather MoE dispatch, so a defect
specific to PCP+no-EP most plausibly lives in that combination rather than in CP metadata. This is
a hypothesis from the gating structure only: no numerics comparison and no fix commit was found,
and the upstream report stayed open. Confirming it needs the same prompt and identical TP/CP
sizes run with EP on and off, comparing attention output and `o_proj` input around the CP gather.

Two corrections to the gating story, verified at `v0.18.0rc1`:

- **`enable_sp()` is not EP-gated, so the sequence-parallel specializations are not the EP-on /
  EP-off delta.** `vllm_ascend/utils.py:772-792` sets `_ENABLE_SP` from
  `VLLM_ASCEND_ENABLE_FLASHCOMM1` (or the legacy `VLLM_ASCEND_ENABLE_FLASHCOMM`, or
  `enable_shared_expert_dp`, which forces it on with a log line) and never reads
  `enable_expert_parallel`. The recorded #8579 launcher sets no `VLLM_ASCEND_ENABLE_FLASHCOMM1`, so
  the SP/DSA-CP-conditioned `o_proj`/`q_b_proj`/`kv_b_proj` paths are off in **both** arms. The
  structural difference is `select_moe_comm_method`
  (`vllm_ascend/ascend_forward_context.py:237-238`: no EP or EP world 1 → `ALLGATHER`, i.e. the
  DP-group prepare/finalize in
  [moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md)).
- **Do not transplant "keep EP and set `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`" into this class.** The
  `v0.18.0rc1` `docs/source/user_guide/release_notes.md` Known Issue says: "Currently, for
  DeepSeek v3.2, PCP & DCP do not yet work with FlashComm1 feature, which may cause serve errors or
  other unknown errors." The maintainer's stated workaround for #8579 is enabling EP alone.

## Configuration checks to run before deeper analysis

- `cp_kv_cache_interleave_size` vs `block_size`: with a sparse model and `cp > 1`, `platform.py`
  warns and overrides `cp_kv_cache_interleave_size` to `block_size`; in P/D disaggregation with a
  KV pool it asserts equality instead. A requested value that is silently replaced changes what
  the reporter's config actually ran.
- PD disaggregation (`kv_transfer_config` present) flips the layer-shard and o_proj gates above,
  so PD and non-PD runs of the "same" CP config take different code paths.

## PCP padding can exceed `max_num_batched_tokens`: prefill `IndexCheck` on a packed step

Report shape: PD split + `prefill_context_parallel_size=2` + MTP, prefill node dies under
concurrency with an operator-side bound error (`Index out of range in dimension 0: index value 8194
exceeds bounds 8192`, `index_check.h:130`), then vector-core exception -> `synchronize` failure ->
`EngineCore` 500. The scheduler-level cause is arithmetic and is verifiable without the reporter's
hardware. This section is verified outside the `v0.18.0rc1` scope above — at `v0.24.0rc1`
(statements) and `v0.26.0rc1` (file existence).

**Which file to read depends on the tag; check before believing the reporter's path.**
`vllm_ascend/worker/pcp_utils.py` exists at `v0.13.0` … `v0.24.0rc1` and is **absent** from
`v0.25.1rc1` on: `69f72c642` "[Refactor][PCP][1/N] remove PCP from MRV1" (#12592) deletes it and
`git tag --contains 69f72c642` starts at `v0.25.1rc1`; that tag carries neither file (`git grep -l
update_tokens_for_pcp v0.25.1rc1 -- vllm_ascend` is empty). From `v0.26.0rc1` the worker-side PCP
code is `vllm_ascend/worker/v2/pcp_manager.py` (+ `worker/v2/model_runner.py`), where
`num_actual_tokens_pcp_padded`, `update_tokens_for_pcp` and `generate_pcp_metadata` no longer exist
(`git grep -l num_actual_tokens_pcp_padded v0.26.0rc1 -- vllm_ascend` is empty, and the
`2 * pcp_world_size` alignment formula occurs nowhere at that tag). A report that labels itself
`v0.26.0rc1` while quoting `pcp_utils.py:811` is therefore a patched or backported build: say so,
and verify the statements at `v0.24.0rc1` or earlier.

**The chain at `v0.24.0rc1`** (`git show v0.24.0rc1:<path> | grep -n '<statement>'`):
- `pcp_utils.py:720` pads *each prefill request* up to a multiple of `2 * pcp_world_size` after the
  scheduler has already packed the step — `ceil(num_scheduled / (2*pcp)) * (2*pcp)` — so one step can
  grow by up to `(2*pcp-1) * num_reqs` tokens. `:746` then takes the per-rank chunk
  `pcp_tokens = num_padded_scheduled_tokens // pcp_world_size`.
- `model_runner_v1.py:973` writes those chunks back into `num_scheduled_tokens`, `:977`
  recomputes `total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])` (the per-rank sum),
  and `:3153` passes that into `generate_pcp_metadata`.
- `pcp_utils.py:1985-1991`: the hybrid branch (`qwen3_next`, `qwen3_5`, `qwen3_5_moe`) overrides the
  argument with `num_scheduled_tokens_padded.sum()`, then
  `num_actual_tokens_pcp_padded = total_num_scheduled_tokens * pcp_world_size` — the *padded* total,
  not the scheduled total. Both branches land on the same value, so the hazard is not hybrid-only.

So `max_num_batched_tokens` does not bound the token count the model sees: a step scheduled to the
budget (`6661 + 1531 = 8192`) pads to `6664 + 1532 = 8196` and the count that reaches the operators
is 8196. The PCP helper's own index tensors carry slack (`model_runner_v1.py:425-427`
`max_buffer_num_tokens = max_num_tokens + max_num_reqs * 2 * pcp_size`; `pcp_utils.py:182-186`), but
`input_ids_pcp_full` / `positions_pcp_full` (`pcp_utils.py:169-177`) are sized exactly
`max_num_tokens`. Which of those buffers the failing Ascend op indexes stays the remaining check —
name the arithmetic as established and the exact tensor as unverified.

Workaround, reporter-reported and not reproduced here: lower `--max-num-batched-tokens` so the
budget minus the worst-case padding fits (7936 for 8192 / PCP=2 / `max_num_seqs`=64) — 0/64 failures
at 16 concurrency in that run, at a claimed ~2.4% per-step token loss. The principled fix direction
is reserving the padding in the *scheduler* budget
(`vllm/config/vllm.py::_set_max_num_scheduled_tokens`, `:1689` at vLLM `v0.26.0`), not in the PCP
helper.

## PCP + MTP mixed batch: `could not broadcast input array from shape (29,) into shape (31,)`

Report shape: a hybrid GDN model (`qwen3_next` / `qwen3_5` / `qwen3_5_moe`) served with
`prefill_context_parallel_size > 1` **and** MTP starts clean, then a concurrent benchmark
(aisbench) kills a worker with `ValueError: could not broadcast input array from shape (29,) into
shape (31,)` raised inside `update_tokens_for_pcp`. The three statements below are present, with the
same logic, from `v0.18.0` through `v0.22.1rc1` (the raise itself is reported on the 0.21/0.22
builds), at `:335/:342/:434` (`v0.18.0`), `:642/:649/:741` (`v0.19.1rc1`),
`:647/:654/:746` (`v0.20.2rc1`) and `:660/:667/:759` (`v0.21.0rc1` and `v0.22.1rc1`, identical), so a
pasted `pcp_utils.py:667` pins 0.21/0.22 while `:654` pins `v0.20.2rc1`. The first fixed tag is
`v0.23.0rc1`. This is a different mechanism from the padding/`IndexCheck` chain above.

**The defect is a token count used as a request offset** (`vllm_ascend/worker/pcp_utils.py` at
`v0.22.1rc1`):
- `:194 self.num_decode_tokens = num_scheduled_tokens[: self.num_decode_reqs].sum()` — the *token*
  count of the decode requests.
- `:660 prefill_tokens_tensor = torch.Tensor(num_scheduled_tokens[self.num_decode_tokens :])` —
  slices the per-request length array at that token count, producing a per-prefill-request
  tensor of length `num_reqs - num_decode_tokens`.
- `:667 num_padded_scheduled_tokens[self.num_decode_reqs :] = num_prefill_scheduled_tokens_linear`
  — writes into the array sliced by *request* count, length `num_reqs - num_decode_reqs`.

The copy therefore raises exactly when the two counts differ: a decode request that schedules more
than one token (MTP, `1 + num_spec` tokens per decode request) in a step that also holds prefill
requests. Check it against the report's own log rather than re-deriving: 32 requests, 1 generation
request with 3 scheduled tokens -> dest `32-1 = 31`, source `32-3 = 29`, which are the pasted
shapes. The branch sits under `if self.pcp_use_hybrid_attn:` (`:650`), gated on
`model_type in ("qwen3_next", "qwen3_5", "qwen3_5_moe")` (`:131-135`), and
`update_tokens_for_pcp` is only called under `if self.pcp_size > 1:` (`model_runner_v1.py:835-836`),
so a dense model never enters the block and an MTP run without PCP never calls the function. The same offset recurs at `:759`
(`ori_tokens_start_loc = np.roll(np.cumsum(num_scheduled_tokens[self.num_decode_tokens :]), 1)`), so
treat the conflation as the class, not one statement: a build patched only at `:660-667` would still
mis-place the prefill restore positions.

**Fix and boundaries.** `c6343b13b` "[Feature] qwen3.5 support pcp+mtp (#10487)" slices by
`num_decode_reqs` at both places (`:821` and `:938` at `v0.23.0rc1`, unchanged at `v0.24.0rc1`);
`git tag --sort=creatordate --contains c6343b13b` starts at `v0.23.0rc1`. Do **not** credit
`#10172`: `2221a9e71` "fix qwen3.5+pcp+mtp single batch error" touches
`attention/context_parallel/attention_cp.py`, `ops/triton/mamba/causal_conv1d.py` and
`spec_decode/llm_base_proposer.py` (mrope `target_positions` slicing) and is contained from
`v0.21.0rc1`, which is why a `v0.22.1rc1` report saying "10172 is merged and it still fails" is
consistent — the mixed-batch path was untouched. `v0.18.0`'s own release notes already name the
adjacent classification hazard once PCP meets spec decode: a prefill request whose scheduled tokens
are below `1 + num_speculative_tokens` is treated as a decode request (#8402,
`release_notes.md:116`).

Workarounds to offer: drop MTP or drop PCP (the thread's maintainer asked for MTP-only and the
reporter confirmed that path is clean), or move to `v0.23.0rc1`+. Candidate, not reproduced: adding
`--additional-config '{"recompute_scheduler_enable":true}'`, which is what the repo's own PD MTP e2e
config does after `7ea0447f6` (#10357, "add recompute scheduler to fix mpt+pcp shape error") — that
changes decode classification per step, so verify the step's request mix instead of assuming the
arithmetic changed.

## Recorded report threads and what they establish

Locators are `vllm-project/vllm-ascend` issues; wording is quoted because the exact claim and its
scope matter more than a paraphrase. All three were still `state: open` when checked, so none is a
fix.

- `issues/8579` — PCP enabled with EP disabled on DeepSeek-V3.2 (TP=8, PCP=2, DCP=1, W8A8; score
  about 94 with EP vs 73 without, plausible-looking but wrong arithmetic). Maintainer reply:
  "Yes, known issue. Temporarily enable EP parallel processing to work." This establishes the
  stated workaround (enable expert parallelism) and that the combination was known broken; it does
  not identify the defect. Its launcher also passes `--no-enable-prefix-caching`, so the hybrid-KV
  per-group prefix-hit contract in [pd-disaggregation.md](pd-disaggregation.md) cannot apply.
  Searching PCP-titled commits between `v0.18.0rc1` and `main` found refactors and features but no
  accuracy fix for this combination — an absence within that search range, not proof that no fix
  exists.
- `issues/7225` — DeepSeek-V3.2 W8A8 on A3, TP=16, run-to-run LongBenchV2 spread (39.0-46.5) with
  `temperature=0.0`, so non-determinism rather than a steady offset. Labelled `wontfix`; the only
  reply asks the reporter to retry the tagged stable release. Nothing here names a mechanism.
  `vllm_ascend/batch_invariant.py` is absent at the reported `v0.13.0rc1` tag and present at
  `v0.13.0` (verified with `git cat-file -e <ref>:vllm_ascend/batch_invariant.py`), so batch
  invariance is a structural lead for this build, not an available workaround.
- `issues/15939` — DeepSeek-V4-Flash W4A8, 1P1D disaggregation on A2, DSpark speculative decode:
  garbled reasoning with `finish_reason: length`, intermittent, and different when prefix caching
  is turned off. No maintainer resolution in the thread; it supplies repro configs and a response
  whose `system_fingerprint` reports the serving stack (`vllm-0.26.0-dp4-ep-...`). The PD + spec
  decode combination, not CP, is the scope, and any root cause has to be derived from source at
  `v0.26.0rc1` / `origin/releases/v0.26.0rc`.
