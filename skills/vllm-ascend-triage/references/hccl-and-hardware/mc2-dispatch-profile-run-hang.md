# An MC2 MoE dispatch kernel in a `profile_run` timeout (`507014`)

## Signature

Engine start reaches `Loading model weights took …`, completes `torch.compile`,
then goes silent for ~10 minutes while `EngineCore` repeats
`No available shared memory broadcast block found in 60 seconds`, and dies:

```text
[rank0]:[E… compiler_depend.ts] NPU function error: call failed, error code is 507014
[ERROR] … ERR00100 PTA call acl api failed
[Error]: The aicore execution times out.
  … there is an exception of fftsplus aivector error … [FUNC:ProcessStarsCoreErrorInfo]
  Kernel task happen error, retCode=0x25, [aicore timeout]
... (Worker_DP0_TP0_EP0 pid=94) File ".../vllm_ascend/worker/worker.py", line 254, in determine_available_memory
...   File ".../vllm_ascend/worker/model_runner_v1.py", line 2215, in profile_run
...   File ".../vllm_ascend/worker/model_runner_v1.py", line 364, in _sync_device
RuntimeError: The Inner error is reported as above. … the current working operator name is aclnnInplaceCopy.
… fault kernel_name=MoeDistributeDispatchV2_ND_ND_BF16_INT8_high_performance_…
[E… ] [Rank 0] HCCL watchdog thread terminated with exception: ERR02005 DIST internal error
```

Read it as: **the first collective of the process (an EP MoE all-to-all in the
profiling dummy batch) never returned**, and the AI-core watchdog surfaced it at
the next sync. `aclnnInplaceCopy` is that sync point, `ERR02005` / `terminate
called` / `shm_broadcast` lines are consequences. The `[AIC_INFO] fault
kernel_name=` line is the authoritative name. Not this page when the reporter
themselves removed `HCCL_OP_EXPANSION_MODE`/`PYTORCH_NPU_ALLOC_CONF` and the op is
`…Dispatch/CombineV4` with `107020` (`aiv-mc2-env-ablation.md`), or when the
failing frames are Python-only (`collective-error-chain.md`).

## Step 1 — the batch that runs is `mc2_tokens_capacity`, not `max_num_batched_tokens`

`vllm_ascend/worker/model_runner_v1.py:2202-2208`, verified at v0.13.0rc1
`1b5d5ab`: when `self.max_num_tokens > mc2_tokens_capacity` **and**
`select_moe_comm_method(mc2_tokens_capacity, cfg)` is `MC2`/`FUSED_MC2`,
`profile_run` first runs an extra
`self._dummy_run(mc2_tokens_capacity, with_prefill=True, is_profile=True)` and
only then `super().profile_run()`. So the process issues an MC2 dispatch at
exactly `capacity` tokens during profiling, while the full profile batch
(`--max-num-batched-tokens`) would have stayed on `ALLGATHER`.

Three facts follow, and they are what makes the answer cheap:

- a startup traceback naming an MC2 dispatch kernel **dates the configuration**:
  EP on, `world_size_across_dp / pp >= 16`, and `capacity < max_num_tokens`;
- the capacity number is derived, so reproduce it rather than quoting the flag:
  `set_mc2_tokens_capacity` (`vllm_ascend/ascend_forward_context.py:160-173`)
  takes `max_cudagraph_capture_size` and returns
  `ceil(max_num_tokens / tp) * tp` (`[4,16,32,48,64]` + TP4 → **64**);
- per-rank payload is `capacity / tp` and the cross-rank shape is
  `global_bs = ceil(max_cudagraph_capture_size/tp) * ep_world_size`
  (`ops/fused_moe/token_dispatcher.py:127`), padded by `mc2_mask`.

Selector location is version-sensitive: at v0.13.0rc1 it is the module function
`select_moe_comm_method` (`ascend_forward_context.py:196-265`); at v0.12.0rc1 it
was `model_runner_v1._select_moe_comm_method`. Re-grep the name before citing a
line, and take the SoC row from the function, not from memory — at v0.13.0rc1 the
A2 row is `MC2` iff `num_tokens <= capacity` **and** `world_size_across_dp/pp >=
16`, else `ALLGATHER` (there is no A2 `ALLTOALL` row at this ref).

## Step 1b — the derivation is per rank, so diff the two nodes' own commands

Both numbers come from **the local node's** `--compilation-config`, so on a
multi-node EP start they are a *per-rank declaration* handed to one collective:
`set_mc2_tokens_capacity` (`vllm_ascend/ascend_forward_context.py:170-182` at
v0.18.0rc1 `99e1ea0`) takes `compilation_config.max_cudagraph_capture_size` (or
`min(max_num_reqs * uniform_decode_query_len, 512)` when the flag is absent) and
returns `ceil(max_num_tokens / tp) * tp`; `token_dispatcher.py:100-118` derives the
same per-rank value and then `global_bs = ceil(max_num_tokens / tp) *
ep_world_size`, which `get_dispatch_mc2_kwargs` passes to the dispatch op as an
attribute (`:156`).

So **diff the two nodes against each other, not only against the tutorial**. A
capture-size list that differs between nodes makes the two halves of the EP group
declare different sizes for the same all-to-all. Recorded #8294 (v0.18.0rc1, 2×A2,
GLM-4.7-W8A8, EP on, `--data-parallel-size 2 --tensor-parallel-size 8`): node 0
pasted `[1,2,4,8,16,32,64]`, node 1 `[1,2,4,8,16,32,64,128,256,512]` → capacity 64 vs
512, `global_bs` 128 vs 1024, 8× apart inside the `ep_world_size = 16` group. One
call reproduces the table:
`bash <skill_root>/scripts/mc2-capacity-asymmetry.sh --capture-a 1,2,4,8,16,32,64
--capture-b 1,2,4,8,16,32,64,128,256,512 --tp 8 --dp 2 --experts 128
--max-num-batched-tokens 8192` (pass `default` for a node that omits the flag); it
also prints whether `profile_run` still takes the extra dummy run and whether the A2
expert gate passes.

Keep the status honest: the derivation and the kwargs are source-verified; *that the
mismatch hangs instead of erroring* is a hypothesis — different per-rank token counts
are supported by design (that is what `global_bs` is for, `token_dispatcher.py:103-104`),
what differs here is the declared buffer size. Close it with the identical-config
rerun, and keep the transport rows and the missing headless log live until then.

## Step 2 — the env pair that changes the dispatch variant is the *recommended* one

`is_hierarchical_communication_enabled()` is exactly
`HCCL_INTRA_ROCE_ENABLE == "0" and HCCL_INTRA_PCIE_ENABLE == "1"`
(`vllm_ascend/utils.py:964-966`). It sets `need_expert_scale`
(`token_dispatcher.py:107`) and the dispatch then carries `expert_scales` as fp32
topk weights (`:168-171`). The dispatcher's own comment (`:102-106`) says that on
A2 those two variables "can reduce cross-machine communication traffic and
significantly improve communication performance", and the model tutorial sets
them — so **do not rank them as the misconfiguration**. The one-variable ablation
is to comment both out (it flips this predicate to `False`), labelled as removing
a documented performance setting, not as fixing a mistake. The kernel-name suffix
`_high_performance` is consistent with the extra scale argument but is not
source-tied in this tree: label it hypothesis.

## Step 3 — the config-diff track closes the flag lane

Both nodes' blocks in `docs/source/tutorials/DeepSeek-V3.1.md:165-197` (this ref)
are the validated two-node recipe, including `HCCL_BUFFSIZE=200`,
`VLLM_ASCEND_ENABLE_MLAPO=1`, `VLLM_ASCEND_BALANCE_SCHEDULING=1`, the
`HCCL_INTRA_*` pair, `--data-parallel-size 4 --data-parallel-size-local 2
--tensor-parallel-size 4 --enable-expert-parallel`, MTP `num_speculative_tokens
3`, and `cudagraph_capture_sizes [4,16,32,48,64]`. If the paste matches it, say
so and move the ranking to transport/driver; the answer is not a reshuffled
launch command.

For a two-node start the tutorial check is also a *symmetry* check, and it is
source-level at v0.18.0rc1 `99e1ea0`: both node blocks in
`docs/source/tutorials/models/GLM4.x.md` use the SAME `cudagraph_capture_sizes
[1,2,4,8,16,32,64,128,256,512]` (`:226`, `:279`). Quote the page's own multi-node
sentence verbatim for "is 2×A2 EP supported for this model?":
`Although the former tutorial said "Not recommended to deploy multi-node on Atlas 800
A2 (64G × 8)", but if you insist to deploy GLM-4.x model on multi-node like 2 × Atlas
800 A2 (64G × 8), run the following scripts on two nodes respectively.` (`:177`). A
shipped recipe exists and carries the project's not-recommended position; no
`tests/e2e/**` config runs it — so the verdict is neither "unsupported" nor
"validated", and the reporter's "EP does not work on 2×A2" is not yet established.

## Step 3b — when the text is `EI0007/qp` + `561000`, not `507014`

The same extra dummy run can fail as a **refused allocation** rather than a timeout:
`call aclnnMoeDistributeDispatchV3 failed, error code is 561000`, then
`Allocation_Failure(EI0007): Failed to allocate resource[qp] … send_cq_depth:32768 …
Memory resources are exhausted`, `DestLink` lines and `HcclAllocComResourceByTiling …
ret = 4` — no `fault kernel_name` line, no `ERR02005`. Everything above still holds for
the arithmetic (`mc2_tokens_capacity`, `global_bs`, per-rank derivation) and for the
`HCCL_INTRA_*` reading; what changes is the lever. Recorded #5947 (v0.13.0rc1, 2×A2,
DeepSeek-R1-0528-W8A8, DP4×TP4, EP, MTP 3, `FULL_DECODE_ONLY`, `HCCL_BUFFSIZE=200`):

- **The phase is not in that paste.** There is no Python frame and no `LoadGraph failed`
  block, so do not present `profile_run` as established: name the block you inferred it
  from (an `aclnn…Dispatch` + `HcclAllocComResourceByTiling` pair is an MC2 collective,
  `LoadGraph failed … ERR03005` is torchair capture) and put the missing line in the ask
  list. `qp-resource-exhausted-at-graph-load.md` owns the capture variant.
- **Do not derive the CQ depth from the batch.** `32768 ≠ max_num_batched_tokens × top-k`
  is a coincidence hunt: the batch that asks is `capacity` tokens (64 in that run, from
  `cudagraph_capture_sizes [4,16,32,48,64]` and TP4), and `send_cq_depth` is a request
  field of the transport, not a reporter knob (topic index rule above).
- **Read the run from the engine's own config line, not from the body.** The
  `non-default args` dict plus the `Initializing a V1 LLM engine … with config:` line give
  TP/DP/DP-local, `enable_prefix_caching`, `enable_chunked_prefill`, `async_scheduling`,
  the capture sizes and `max_cudagraph_capture_size`; #5947's body claimed TP8/DP8 while
  the log ran TP4/DP4/DP-local 2 (which is what makes 47.2958 GB of weights per card
  possible on 16 dies), and the log's engine line carried `enable_prefix_caching=True` /
  `enable_chunked_prefill=True` that the CI recipe below explicitly avoids.
- **Recipe diff, one file:** `tests/e2e/nightly/features/test_mtpx_deepseek_r1_0528_w8a8.py`
  at v0.13.0rc1 `1b5d5ab` is the project's own R1-0528-W8A8 launch: `HCCL_BUFFSIZE=1024`
  (`:72`), DP2×TP8 with EP (`:87-90`), `--no-enable-prefix-caching` (`:86`),
  `--max-model-len 40960 --max-num-seqs 14`, and per MTP depth — mtp2 `:101-105`
  (`--max-num-batched-tokens 4096`, `gpu_memory_utilization 0.92`), mtp3 `:108-114`
  (`HCCL_OP_EXPANSION_MODE=AIV`, `--max-num-batched-tokens 2048`, `0.9`,
  `FULL_DECODE_ONLY` with `cudagraph_capture_sizes [56]`). Two axes in #5947's run differ
  from it and both change the per-card budget or the dispatch path: TP4 instead of TP8
  (non-expert weights sharded 4 ways), and prefix caching/chunked prefill left on. Quote
  the file; do not assemble a launcher from the constraints.

## Step 4 — ranking, ladder, artefacts

**Before the ranking: if the reporter already ran the env matrix, do not re-issue
it.** #8294's reporter came back on 2026-04-17 with "all combinations of these 3 envs
… fails at the same phase (torch.compile). Even I unset the 3 envs, it fails too" —
that closes `HCCL_OP_EXPANSION_MODE` / `PYTORCH_NPU_ALLOC_CONF` / the FLASHCOMM1 pair
as causes for that report, and the phase in the title is the reporter's label, not the
phase (their traceback line is what dates it). Ask which three variables they mean
rather than guessing, then spend the turn on the symmetry check above and the missing
artefacts.

Settled by source at v0.18.0rc1 (`ascend_forward_context.py:236-247`): EP off or ep
world 1 → `ALLGATHER`; A2 → `MC2` iff `num_experts_per_device <= 24` **and**
`ep_world_size = world_size_across_dp // pp >= 16` **and** `num_tokens <= capacity`;
`VLLM_ASCEND_ENABLE_FUSED_MC2` is read only in the A3 branch, so on A2 setting it is a
no-op (never rank "unset `VLLM_ASCEND_ENABLE_FUSED_MC2`" as a fix), and there is no
env that forces `ALLGATHER` while EP stays on — `--enable-expert-parallel` off is that
route. A reporter's "single node works" is therefore expected at world 8 and does not
clear the MC2 path.

Ranked, one check each: (1) the cross-node leg of the all-to-all not completing
(rank table / `device_ip` / NIC / the hierarchical mode) — hypothesis, decided by
a 16-rank `hccl_test` plus `hccn_tool -i <id> -link -g` and a device-IP ping
before restarting vLLM; (2) image CANN vs the host driver the container mounts
(`cat /usr/local/Ascend/driver/version.info`, `npu-smi info` on both nodes) —
hierarchical MC2 is a newer kernel variant and a mismatch tends to hang rather
than error; (3) one rank not entering the same dispatch — decided by the
**headless node's** log and plog, which the report usually omits; (4) hardware
last, and only with a per-card code or the same `pc start` on several chips
(`device-health-checks.md`). Refuted rows to keep in the answer: the
`HCCL_INTRA_*` pair, `HCCL_BUFFSIZE` (too small prints its own
`HCCL_BUFFSIZE is too SMALL`, see #4051), `aclnnInplaceCopy`, and
`VLLM_ASCEND_ENABLE_MLAPO` (MLA/SFA only, non-prefill branch, auto-disabled on a
non-W8A8 checkpoint — `attention/mla_v1.py:845-856`).

Ladder, one axis per launch: (0) make the two nodes' commands identical (the
symmetry check above) — cheapest, and when asymmetric it also restores the
documented recipe; (1) comment out both `HCCL_INTRA_*`; (2) drop
`--enable-expert-parallel` (→ `ALLGATHER`, the kernel disappears: proves the
transport); (3) single node (world < 16, MC2 never selected) to separate
"cross-node" from "MC2"; (4) `--enforce-eager` / `cudagraph_mode: NONE`, then
`ASCEND_LAUNCH_BLOCKING=1` — in that order, because the platform refuses blocking
mode while an ACL graph is configured. Ask for: the headless node's full log,
both nodes' `version.info` + `npu-smi`, the `hccl_test` result, and the plog for
the failure minute on both nodes.

## Precedents and scope

Same op, same 2×A2 shape, title/body only (no comments in the snapshot): #4256
(DeepSeek-v3.2-EXP-W8A8, `MoeDistributeDispatchV2`, `507011` at a copy-stream sync
on device 0) and #2858 (0.10.1rc1 two-node Qwen3-235B-A22B,
`aclnnMoeDistributeDispatchV2 failed 561002`). Both are family support, not
resolutions. #8294 (v0.18.0rc1, 2×A2 GLM-4.7-W8A8 W8A8, EP with DP2×TP8, CANN
8.5.1) is the recorded cross-node-asymmetric-capture-size case: the
traceback is `worker.py:343 determine_available_memory` → `model_runner_v1.py:2544`
`_dummy_run(mc2_tokens_capacity, …)` → `:2492 _model_forward` → `:1818`, i.e. exactly
this page's extra dummy run, and the post-hoc reporter observation (2026-04-17) says
every env combination including all-unset failed at the same phase.

A `561002` is **not** always this page's hang. If the paste carries
`moeExpertNum is N, in case of unlayered, it must no more than 24`
(`MoeDistributeDispatchA2CheckAttrAndSetTiling`) the op refused its arguments and
returned immediately: that is the A2 MC2 per-card **expert cap** — `N` =
`global_experts // ep_size` plus EPLB redundant experts — decided at
`../feature-enable-recipes/moe-expert-parallel-with-dp.md` § Step 2b
(reporter-confirmed on #7635, fallback merged as PR #7364), not a transport
question and not worth a plog.

Pins: vllm-ascend v0.13.0rc1 `1b5d5ab`, vllm v0.13.0 `72506c9`; re-pinned for
the symmetry step and the A2 selector rows at vllm-ascend v0.18.0rc1 `99e1ea0`,
vllm v0.18.0 `bcf2be9`. The
`profile_run` dummy run, the selector location, the capacity formula and the
hierarchical predicate are the parts to re-grep at another ref (`profile_run`,
`select_moe_comm_method`, `set_mc2_tokens_capacity`,
`is_hierarchical_communication_enabled`).

**Not established** in #5509: the report carries no `npu-smi`, so the SoC is
inferred from the selector row + the `HCCL_INTRA_*` comment + the tutorial, not
confirmed — an A3 host takes a different selector row (`FUSED_MC2`/`MC2`/
`ALLTOALL`, gated by `VLLM_ASCEND_ENABLE_FUSED_MC2` and
`ep_group.world_size <= 16`), so the arithmetic would be re-derived. The cause of
the hang (transport vs driver vs a rank that never entered) was not settled by the
paste, and the op name `…DispatchV2` also says the installed torch_npu exports
`npu_moe_distribute_dispatch_v2` (`token_dispatcher.py:99-100,202-204`) — the
`…V4` names belong to a newer stack than this tree.
