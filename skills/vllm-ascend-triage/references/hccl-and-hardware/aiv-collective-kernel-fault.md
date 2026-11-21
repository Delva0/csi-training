# `fault kernel_name=aiv_*`: an HCCL AIV collective kernel, not a vllm-ascend op

## When this page applies

The plog's fault block names a kernel whose name starts `aiv_`, on a serve where
something set `HCCL_OP_EXPANSION_MODE=AIV` (launch script, tutorial, or a
`vllm_ascend/patch/` on a model that needs it). Recorded **#13527** —
vllm-ascend v0.18.0 (`e18643f8a4d5`), DeepSeek-V4-flash EP, 8×A2, P/D separation,
`HCCL_OP_EXPANSION_MODE="AIV"` exported on **both** sides, failing after the
engine had served for a while; every attached rank's plog carries:

```text
CheckSmmuFault:Get isSmmuFault is 1.
PrintCoreInfo: … there is an exception of aivec error, core id is 35, error code = 0x800000,
  dump info: pc start: 0x12c09e46fe08, … , para base: 0x12c1400c7000.
PrintCoreInfo: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range.
SetStarsResultForDavinciTask:AIV Kernel happen error, retCode=0x31.
PreCheckTaskErr:Kernel task happen error, retCode=0x31, [vector core exception].
[DFX_INFO]Aicore kernel execute failed, device_id=4, … fault kernel_name=aiv_reduce_scatter_bfloat16_t, …
[TaskExecStage][RunFailed][AIV]Task run failed, … taskType[Task AIV],
  tag[ReduceScatter_group_name_3], AlgType(level 0-1-2):[ring-ring-ring].
  para information is cmdType:[7], tag:[511], size:[3932160], numBlocks:[16], rankSize:[8],
  aivRdmaStep:[-1], flagMem:[0x12c09d600000], isOpbase:[1].
  opData information is timeStamp:[…], deviceId[4], index[34730511], count[1966080],
  reduceType[sum], src[0x…], dst[0x…], dataType[bfloat16].
```

Same family for the other collectives: `aiv_all_reduce_*`, `aiv_all_gather_*`,
`aiv_all_gather_v_*`, `aiv_all_to_all_*`, `aiv_reduce_scatter_v_*`,
`aiv_broadcast_*`.

**Not this page** — the same `aivec`/MTE text with a *different* `fault
kernel_name`: an operator name (`ScatterNdUpdate_…`, `InplacePartialRotaryMul`,
a `MoeDistribute*`) is `../custom-ops-runtime-faults/index.md` (or
`../dsa-sparse-attention/indexer-cache-scatter-fault.md`,
`mc2-dispatch-profile-run-hang.md`); an AIV × MC2 dispatch/combine hang without a
fault kernel (`107020`, `Dispatch/CombineV4`) is `aiv-mc2-env-ablation.md`; the
`507011` / aivector-under-concurrency shape is
`aclgraph-full-decode-runtime-crash.md`. **The same page still owns the
`retCode=0x30` variant** (no MTE/SMMU text, `errorStr: timeout or trap error`) and
the small-`int32` `aiv_all_gather_*` — § "The other retCode" below — and the paste
that stops **before** any `fault kernel_name=` (a `device_error_core_proc.cc` dump
with only `aivec error` / MTE DDR text) — §0, which ranks by deployment shape.

## The other retCode: `0x30` timeout, and the small-`int32` `aiv_all_gather_*`

`retCode=0x31` above is a vector-core **exception** (MTE DDR out of range, SMMU
fault) — the rank that printed it touched a bad address. `retCode=0x30` with
`errorStr: timeout or trap error` / `[TaskExecStage][Timeout][AIV]` and one identical
`pc start` on every core is a **spin/wait that no peer ever released**: the printing
rank entered a collective that some other rank of the same group never issued. Do
not read the timeout rank as the owner; the owner is the rank missing from the
collective, so count the ranks that issued the op (`device_id=`) before ranking
anything.

Recorded **#9887** (vllm-ascend v0.18.0 reported, GLM-5.1-w8a8, P2D2 decode crash
titled `MoeDistributeDispatchV2`; pinned here at v0.19.1rc1 `da421af`). The attached
device-5 plog carries:

```text
[DFX_INFO]Aicore kernel execute failed, device_id=5, stream_id=9, task_id=33, …
  fault kernel_name=aiv_all_gather_cn_int32_t, … hash=4712968614259692016
[TaskExecStage][Timeout][AIV]Task run failed, … taskType[Task AIV], tag[AllGather_group_name_219],
  AlgType(level 0-1-2):[ring-ring-ring]. para information is cmdType:[6], tag:[9], size:[1024],
  blockDim:[32], rankSize:[32], aivRdmaStep:[32], flagMem:[0x12c0fd200000], isOpbase:[1].
  opData information is timeStamp:[2026-06-01-12:41:08.576.208], deviceId[1], index[9],
  count[256], src[…], dst[…], dataType[int32].
```

Read it in this order:

1. **Neither attached log contains the string `MoeDistributeDispatch`.** The title's
   op name is the reporter's label, not evidence; the plog's own `fault kernel_name`
   is authoritative. (The device-0 plog that would carry the MTE/SMMU owner line was
   never attached — 416 on upload — so say "owner unverified", not "owner is device 0".)
2. **`0x30`, one shared `pc`, and a ~5 minute gap between the `opData` `timeStamp`
   (12:41:08) and the fault print (12:46:09)** = HCCL's own timeout fired, then the
   watchdog. The op was *issued* five minutes before it failed; the failure minute is
   not the trigger minute (`error-report-timestamps.md`).
3. **A small `int32` all-gather is not a generic collective.**
   `count == num_experts` (256 for GLM-5.1) with `size = count × 4` and
   `rankSize == ep_world_size` (32 = TP4 × DP8 here) is the MoE dispatcher's own
   preprocess histogram gather:
   `gather_from_sequence_parallel_region(torch.histc(topk_ids, bins=self.num_experts, …),
   group=self.ep_group)` in `TokenDispatcherWithAll2AllV._preprocess`
   (`vllm_ascend/ops/fused_moe/token_dispatcher.py:551-560` at `da421af`; class at
   `:407`, `token_dispatch` `:440`, `_dispatch_preprocess` `:522`). The dtype is the
   version-dependent part — at this pin `histc` is float, the plog says `int32`, so
   quote `count == num_experts` as the match and leave the dtype open unless the
   reporter's ref shows it. Map the *Python* traceback frame
   (`token_dispatcher.py:<line> token_dispatch`) to its class before quoting a line
   number: the class whose `_preprocess` runs is the branch, and only that branch
   issues this gather.
4. **The reporter-confirmed mechanism is a per-rank branch split, not a kernel bug**
   (2026-07-21): "有某个DP走了alltoallv分支，其余的DP走了MC2，并且是在MTP阶段" — the
   selection is per rank and per batch (`select_moe_comm_method(num_tokens, …)`,
   `ascend_forward_context.py:230-280` at `da421af`; A2 row = MC2 iff
   `num_experts_per_device <= 24` **and** `ep_world_size >= 16` **and**
   `num_tokens <= mc2_tokens_capacity`). The rank whose MTP/decode batch exceeded
   capacity left MC2, entered the fallback dispatcher's histogram gather, and the MC2
   ranks never join it → this timeout, while one rank reports MTE/SMMU. That gather is
   the `All2AllV` dispatcher's, so the fallback row matters: at v0.12 the A2 fallback
   was `ALLTOALL`, while at v0.19.1rc1 it is `ALLGATHER`, whose `token_dispatch`
   (`:329-405` here) uses `npu_moe_init_routing` instead — re-grep the selector before
   quoting this plog as an All2AllV branch on another ref. Their numbers:
   `max_num_batched_tokens=272`, `max_cudagraph_size=102`, capacity **104**
   (`ceil(102/4)*4`, TP4 — the formula and the reporter's number agree), "某个DP的
   tokens大于104了".
5. **The ask and the workaround.** Ask `--max-num-batched-tokens`, `--max-num-seqs`,
   `num_speculative_tokens`, the `--compilation-config` capture sizes and the DP size,
   then derive capacity (`scripts/mc2-capacity-asymmetry.sh`) and compare it with the
   largest per-rank batch (`max_num_seqs × (1 + num_spec)` for decode,
   `max_num_batched_tokens` for an extend/MTP step). Reporter's workaround: set
   `--max-num-batched-tokens` to `max_num_seqs × 3` (their layout) and keep it a
   multiple of 32 so the fused-MoE tiling does not reject it — generalised: keep the
   per-rank batch at or below the capacity derived from `max_cudagraph_capture_size`.
   That is a configuration fix, not a mitigation of a kernel defect.
6. **Route.** Capacity/`global_bs` arithmetic and the cross-node symmetry variant:
   `mc2-dispatch-profile-run-hang.md`; the selector table and the per-batch reading:
   `../feature-enable-recipes/moe-expert-parallel-with-dp.md` Step 2.

## 0. When the paste stops before the name (the usual screenshot)

A device dump that ends at `PrintCoreInfo` / `[FUNC:ProcessStarsCoreErrorInfo][FILE:
device_error_core_proc.cc]` — `aivec error`, `errcode:(0x800000, 0, 0) errorStr: The DDR
address of the MTE instruction is out of range`, `subErrType`, `aic error mask`, one
`chipId` — with **no** `[DFX_INFO] … fault kernel_name=` line, no `[AIC_INFO]` block and
no Python frame (the client answers `code: 599` / `EngineCore encountered an issue`, which
is only the replica having died). Recorded **#3911**: Qwen3-VL-235B-A22B-Instruct, DP2×TP8
+ `--enable-expert-parallel` across 2×A2, vllm-ascend v0.11.0rc0 (`00ba071`), dies at
request ~146 of an MMMU_Pro_V run.

**Do not read the owner off the text.** `aivec error` + MTE DDR out-of-range is printed by
every AI-vector-core kernel that faults, CANN built-ins included, so it names a *class*
(software address computation, not a dead card) and nothing more. Rank by the deployment
shape, which is computable at the pin:

1. **MoE comm path.** `_get_fused_moe_state(ep_size, with_prefill, …)` with
   `ep_size = get_ep_group().world_size if enable_expert_parallel else 1`
   (`vllm_ascend/ascend_forward_context.py:34-50,90-91` at v0.11.0rc0): `ep_size == 1` →
   `NaiveMulticast` prefill / `AllGather` decode; `< 16 or with_prefill` → `All2All`;
   else **`MC2`** on decode. #3911's 16 ranks put every decode step on MC2
   dispatch/combine and every prefill batch on All2All, so MC2 is the first device-op
   candidate. That ref has **no** MC2 env knob (`rg MC2 vllm_ascend/envs.py` empty), so
   the one-axis ablation is *topology*: the same model on one node (TP8 + EP, no DP) is
   `ep_size = 8` and moves decode to All2All. (`moe-comm-path-when-ep-off.md` in
   `../performance-throughput/` carries the same table for the perf reading; later refs
   move the selector and add `enable_fused_mc2`, so re-grep before quoting.)
2. **An HCCL AIV collective — only if the process actually set it.** The reporter's
   `collect_env` block cannot settle this: `get_env_vars()` prints a hardcoded whitelist
   plus `TORCH/PYTORCH/ASCEND_/ATB_` prefixes, and their own exported `HCCL_BUFFSIZE` is
   missing from the paste (`../startup-failures/environment-correspondence.md`). The check
   is `env | grep -i hccl` on **every** node of the group.
3. **A long-VL partial-prefill index fault.** When `--max-num-batched-tokens` ≪
   `--max-model-len` (here 4096 vs 32768) every long multimodal prompt is chunked, so the
   failing request may be a partially-computed one: precedents #1682 / #1731 (Qwen2.5-VL,
   V1) failed in `gather_v3_base.h` with the index equal to the prompt length on a request
   whose `num_computed_tokens` was non-zero (`../custom-ops-runtime-faults/gather-index-out-of-range.md`).
   Discriminator: the DEBUG `Dumping input data` / `SchedulerOutput(...)` block right after
   the fault (`num_computed_tokens`, `prompt_token_ids_len`, `mm_positions`), then one
   rerun with `--max-num-batched-tokens` raised to remove chunking.
4. **Card / driver last**, and only if the `pc start` values differ across chips (step 5.4).

Also check the launch line against the project's own recipe before blaming it: #3911's
command is byte-identical to `docs/source/tutorials/multi_node_qwen3vl.md` at that pin
(DP2, TP8, `--enable-expert-parallel`, 32768, 4096) except `--max-num-seqs 2`,
`--gpu-memory-utilization 0.75` and an **added** `--enforce-eager` — the validated
topology sitting on a path-specific defect, not a wrong command. What to send back: the
`fault kernel_name=` block from the failing rank's plog, `env | grep -i hccl`, and the
DEBUG scheduler dump of the failing request.

## 1. Read the owner off the name before any source walk

`aiv_<op>_<dtype>` is generated by HCCL's own macro, not by vllm-ascend's
`csrc/`:

```c
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF(type) \
extern "C" __global__ __aicore__ void aiv_reduce_scatter_##type(KERNEL_ARGS_DEF) { … } \
EXPORT_AIV_META_INFO(aiv_reduce_scatter_##type)
```

`cann/hcomm/src/algorithm/base/alg_aiv_template/aiv_communication.h:227-258`
(verified in the #13527 export; `aiv_all_reduce` `:80-120`, `aiv_all_gather`
`:192-211`, `aiv_all_to_all` `:169-188`, `aiv_reduce_scatter_v` `:262-271`,
`aiv_broadcast` `:275-283`). The suffix is the `##type` the report happens to
instantiate (`_bfloat16_t` there), so grep the family prefix, and do **not** run
the custom-op lane's `ls csrc` test — it returns nothing by construction, and
"the kernel is not in vllm-ascend" is not a finding. If the case's export has no
`cann/hcomm` (check `SOURCES.json`), the lane is a data gap: say so and stop at
the plog facts below.

Within that macro, `isOpbase:[1]` selects the `if (isOpBase)` branch and
`aivRdmaStep:[-1]` the non-RDMA one (`:238-247`), and `deterministic == 1` a
separate `aiv_reduce_scatter_deter_910b_*` family (`:229-237`) — quote which
branch the report's own fields select, as configuration, not as cause.

## 2. Compare the `opData` line across every attached plog first

That one line is the cheapest discriminator in the whole lane: `index`, `count`,
`src`, `dst`, `reduceType`, `dataType`. **Identical on every rank refutes "one
rank passed a different size"** (the mismatched-count/last-rank theory) and
leaves the kernel's own address computation or a peer mapping. Divergent
`index`/`count` puts it back on the callers and on arrival order.

#13527: six plogs, six identical `opData` lines — the refutation was available in
one `rg -o 'opData information is .*' plog-*` call and should precede any
`reduce_scatter` source walk. (`../first-checks/cheap-checks-by-track.md`.)

## 3. `size`, `count` and the one identity that holds

`size` is `count × sizeof(dataType)` (1966080 bf16 elements = 3932160 B in the
line above) — a free check that the two fields belong to the same op. What you
**cannot** read off the plog is which template ran: the branch tests
`len * sizeof(type)` against `AIV_REDUCE_SCATTER_MID_SIZE` (2 MiB) and
`UB_MAX_DATA_SIZE` (190 KiB) (`aiv_communication_base.h:150,176`), and `len` is a
separate kernel argument (`KERNEL_ARGS_DEF … uint64_t len`, `:87`) that the plog
never prints. State it as "not printed" rather than deriving the template from
`count`, and do not turn the arithmetic into a root cause (this topic's common
rule 3).

`numBlocks` vs `rankSize` (16 vs 8 above) is workload, not evidence: the
templates size their per-rank slices from `rankSize`, so compare the two only if
they disagree *between* ranks of the same collective.

## 4. AIV is the configured path, not a misconfiguration

At v0.18.0 `vllm_ascend/utils.py:506` branches on
`os.getenv("HCCL_OP_EXPANSION_MODE") == "AIV"` for ACL-graph capture-size
budgeting, and `:554` **recommends** the variable in a warning, so a deployment
setting it is following the tree (multi-node GLM tutorials export it too). An AIV
fault therefore cannot be answered with "unset the env" as a fix: unsetting it is
the **discriminator** (it moves the collective to host/AI-CPU expansion), and a
pass is a workaround plus a narrowed owner — the same split as
`aiv-mc2-env-ablation.md` §3. On A2 the documented default is host-side CPU
expansion, AI-CPU on A3 (`cann/hcomm/docs/api_c/context/HcclCommConfig.md:95`), so
a `Task AIV` line at all means someone set it: ask which script line, and whether
P and D disagree.

## 5. The steps after the plog

1. **Count the ranks, by field not by filename.** `rankSize:[N]` is in the line;
   the faulted ranks are the `device_id=` values. #13527 attached devices
   1,2,3,4,5,7 of an 8-rank group — devices 0 and 6 are a *collection* gap, and
   "partial participation" (a 2-rank shortfall) reads like a mechanism while being
   an artefact of which files were zipped. Ask for the missing ranks.
2. **The next op's error is a victim.** `aclnnInplacePartialRotaryMul failed` /
   `Execution_Error_SDMA(EI0012) … Remote rank: [0]` beside the same
   `taskID`/`tag[ReduceScatter_group_name_3]` is the following launch reporting a
   dead stream (rule 11), not a second defect — and `Remote rank: [0]` is not
   evidence that rank 0 faulted.
3. **`group_name_3` is an opaque per-process counter.** Do not map it to
   TP/DP/EP/CP from the name; take the group sizes from the two serve commands
   and the vLLM log's own group lines.
4. **Hardware stays on the table only until the shared-PC test.** One `pc start`
   on every core of every chip is software address computation; independent PCs
   and per-card codes send you to `device-health-checks.md` (which recorded a
   physical HCCS link fault on 6 of 8 cards — do not refuse that lane outright).
5. **`stackcore_tracer_*_python3_*.txt` beside a plog is not a Python
   traceback.** It is the native dump taken when the process was signalled
   (`crash reason: signal 15`), its `[stack]` section names libc/CANN frames
   (plog flush, watchdog, atrace threads) and its bulk is the memory map; read it
   for `kernel version:` (the node name) and nothing else. The Python owner of
   the collective is still owed: ask for the worker log or a
   `ASCEND_LAUNCH_BLOCKING=1` rerun.
6. **Rerun with `HCCL_OP_EXPANSION_MODE` unset on every node of the group**, then
   — if the fault survives — with `HCCL_DETERMINISTIC=1` (which selects the
   `deter` template family). Both are one axis each; neither is a patch.

## What to send back

The plogs of **all** ranks in the collective's group (`rankSize`), the worker/vLLM log
around the fault time so the issuing collective has a Python owner, both serve
commands (including every `HCCL_*` export and the parallel sizes), and `asys
health` per device if step 6 changes nothing.

Version scope: the macro, the thresholds and the branch names above are from the
#13527 export (vllm-ascend v0.18.0 `e18643f8a4d5`, `cann/hcomm` `818068c8ef5f`)
and are per-SoC (`_910b_`, `_91093_`) — re-grep `aiv_communication.h` at the
reporter's ref before quoting a line. The AIV-selection reasoning
(`utils.py:506,554`) is v0.18.0-specific; later releases move both the line and
the recommendation.
