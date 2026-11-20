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

## Step 4 — ranking, ladder, artefacts

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

Ladder, one axis per launch: (1) comment out both `HCCL_INTRA_*`; (2) drop
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
resolutions.

Pins: vllm-ascend v0.13.0rc1 `1b5d5ab`, vllm v0.13.0 `72506c9`. The
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
