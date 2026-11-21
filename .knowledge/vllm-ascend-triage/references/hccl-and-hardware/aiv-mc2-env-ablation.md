# "It starts when I remove the env vars": AIV × MC2 dispatch/combine hangs

Signature family — a **startup/profile_run** hang that ends in a device timeout on
an MoE all-to-all op, pasted with the reporter's own ablation "去掉
`PYTORCH_NPU_ALLOC_CONF`、`HCCL_OP_EXPANSION_MODE` 就能拉起":

```text
[rank0]:[E… OpParallelGuard.cpp:352] operator(): .../core/MoeDistributedDispatch/XNemoOpApi.cpp:194
NPU Function error: call aclnnMoeDistributeDispatchV4 failed, error code is 107020
[PID: 734] 1VF Invalid Argument (RUNTIME): event wait timeout occurred during task execution,
        stream_id:11 ... timeout=230s, [FUNC:ProcessStartWaitTimeoutErrorInfo]
rtNotifyRecord execution failed, reason=task_timeout
[EngineCore_DP0_TP6] INFO [shm_broadcast.py:802] No available shared memory broadcast block found in 60 seconds
<other ranks: AIV AllGather SDMA task exceptions / "dispatch success" warnings>
```

Tells: the first in-window error names `aclnnMoeDistribute{Dispatch,Combine}V4`;
`107020` arrives together with the runtime's own `task_timeout`, so read it as a
timeout, **not** a parameter error; the `shm_broadcast` 60 s lines and the
per-rank AIV AllGather exceptions are consequences (some rank never returned), so
do not answer with the RPC watchdog page or a NIC theory; the reporter usually
still reaches `EngineCore` and dies during `profile_run`, i.e. no traffic yet.

## When the whole matrix has already been run

Recorded #8294 (v0.18.0rc1, 2×A2, GLM-4.7-W8A8, EP + DP2×TP8) came back on
2026-04-17 with "I have tried all combinations of these 3 envs and it fails at the
same phase (torch.compile). Even I unset the 3 envs, it fails too." Treat that as a
**completed** ablation, not an invitation to re-run the matrix: the AIV/allocator
lane is closed for that report, the phase word in the title is the reporter's own
label (their traceback shows `profile_run`'s extra MC2 dummy run, not the compile
step), and the next lanes are the per-node symmetry of the derived
`global_bs`/`mc2_tokens_capacity` (`mc2-dispatch-profile-run-hang.md` Step 1b, one
call: `scripts/mc2-capacity-asymmetry.sh`), the cross-node transport check, and the
headless node's own log. Ask which three variables they mean instead of guessing, and
do not offer "unset them all" as a further experiment — that is the run they already
did.

**And the compile-cache lead is closed too.** On #10370 the paste's only log line was
`saved AOT compiled function … torch_compile_cache …`; a maintainer proposed
`VLLM_DISABLE_COMPILE_CACHE=1` and the reporter answered **没有效果** (2026-06-12) after
the fact. That line is `decorators.py` *saving* an artefact — a graph-mode tell, not the
fault owner — so treat the ablation as done, and take the mode split instead
(`graph-vs-single-op-comm-fault.md`).

## What the two knobs actually are (do not rank them equally)

- `HCCL_OP_EXPANSION_MODE=AIV` switches the collectives to AI-vector-core
  scheduling. On A3/A2 MoE models the vllm-ascend selector then runs the
  **MC2** dispatch/combine, and that combination is the precedent family below.
  Removing it does not "fix" anything: it changes the selected communication path.
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is an allocator setting; it
  is a documented part of the tutorial env blocks and has no mechanism for a
  dispatch-op event timeout. It *does* have one hard conflict worth naming: the
  CaMem memory pool asserts it away —
  `assert "expandable_segments:True" not in conf, "Expandable segments are not
  compatible with memory pool."` (`vllm_ascend/device_allocator/camem.py:153`,
  v0.26.0rc1 `f2f74a1`) — so a pooled/KV-pool deployment that trips the pool would
  raise that assert, not a hang. Do not carry "`expandable_segments` is innocent"
  beyond this shape: on a host where the driver rejects the segment unmap, the same
  variable is itself the crashing site (`unmapHandles … 507899` in
  `torch.npu.empty_cache()`, reporter-confirmed #12534) —
  `../runtime-memory/allocator-expandable-segments-unmap-crash.md`.
- A documented multi-node A2/A3 command sets both (and `HCCL_BUFFSIZE`,
  `HCCL_EXEC_TIMEOUT`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`); a missing
  `HCCL_EXEC_TIMEOUT` only changes how long the hang takes to surface.

## Procedure

1. Name the selected MoE method from the rank arithmetic before any source walk
   (`vllm_ascend/ascend_forward_context.py`, v0.26.0rc1 `f2f74a1`):
   A2 `_select_a2_moe_comm_method` (`:322-340`) returns ALLGATHER when
   `num_experts > 512`, else MC2 only when `num_experts_per_device <= 24` **and**
   `ep_world_size = world_size_across_dp // pp >= 16` **and** tokens fit
   `mc2_tokens_capacity`; A3 (`:412-428`) returns FUSED_MC2/MC2 by capacity and
   `enable_fused_mc2`, else ALLTOALL. A `num_tokens = None` call (the shape a
   profiling/warm-up batch can take) is eligible **by construction** — the test is
   `num_tokens is None or num_tokens <= capacity`. If the numbers say ALLGATHER,
   this page is not your lane, and the hang belongs to whatever op the log names.

   **The rule changed between pins, so quote the pin's own form.** At v0.20.2rc1
   (`367b8e62`, recorded #10370) the same decision is inline in
   `select_moe_comm_method` (`vllm_ascend/ascend_forward_context.py:233-319`): there is
   no `num_experts > 512` guard yet (A2 branch `:273-282`, A3 `:284-310`), so applying
   the newer rule at that pin would wrongly turn a >512-expert A2 case into
   "ALLGATHER, not my lane".
2. Give the ablation matrix, one variable per launch, cheapest first — the
   reporter has only tested "both removed":
   | run | env | expected if AIV × MC2 owns it |
   |---|---|---|
   | 1 | AIV + `expandable_segments` (reported) | FAIL, `Dispatch/CombineV4` + 107020 |
   | 2 | AIV only | FAIL — this is the discriminator |
   | 3 | `expandable_segments` only | PASS |
   | 4 | neither (their working run) | PASS |
   Also ask for the failing rank count: AIV × MC2 fails on the MC2 op, while a
   RoCE/rank-table fault fails on the same op with an HCCL return code instead
   (`hccl_ret 9`, #9447) and can be reproduced by `hccl_test`.
3. Answer with "workaround vs defect" clearly separated: keeping AIV and forcing
   the non-MC2 method (diagnostic selector override, as in #14928's passing row)
   is the diagnostic; dropping AIV is the workaround; the combo itself is a
   torch_npu/CANN-level defect the reporter should file with the ablation table.
   Do not present an `mc2` env switch that does not exist in the pinned tree —
   grep `ascend_config.py` for the knob before naming one.
4. If the escalation is wanted, hand over the four rows plus the first in-window
   error and stop. This lane is answerable in 3-4 calls.

## Precedent table

| Issue | Configuration | Result |
|---|---|---|
| #14928 | A3 single node, GLM-5.2 W4A8C8, TP8/EP8, **eager**, `max_num_batched_tokens=128`, AIV on | `profile_run` hangs in `aclnnMoeDistributeCombineV4`, `107020` after ~320 s; AIV + AlltoAll override PASSES, AIV unset + MC2 PASSES, DCP 1 and 8 both fail → selector-level, not graph or DCP |
| #13688 | 910C ×8, GLM-4.6V TP8+EP8, AIV on, NPUGraph | capture failure `rtEventRecord ... 507009` and replay deadlock (AIVector 8-16 %, AICore 0 %), plus "MC2 MoE dispatch/combine still deadlock with AIV" |
| #9447 | v0.18.0rc1, GLM-5.1, 2×A3 over **RoCE** vs 灵衢 direct connect | `aclnnMoeDistributeDispatchV4` comm failure, `hccl_ret 9`, works on direct connect → transport variant, not AIV |
| #15195 | A3, DeepSeek-V4-Flash w8a8, 16 decode ranks, MC2/MoE stage | `EVENT_WAIT` timeout at 234 s, `rtEventSynchronize` → `107020`; restart of the decode processes restored service (no hardware fault) |
| #12411 | 910C-560T, 4 nodes, GLM-5.1-w4a8, 1P1D | `aclrtSynchronizeEvent(event_)` / `107020` — same code, multi-node, MC2 stage |

Reading of the family: the op names come from the installed torch_npu/CANN
(vllm-ascend v0.26.0rc1's MC2 path calls
`torch_npu.npu_moe_distribute_dispatch_v2` / `..._combine_v2`,
`vllm_ascend/ops/fused_moe/token_dispatcher.py:120,241,343`, gated on
`hasattr(torch_npu, ...)`), so a paste naming `...V4` says the reporter's stack is
newer than the pinned tree — state that gap instead of "verifying" the V4 kernel
in a tree that does not contain it.

## Traps

- Do not answer "it is the doc's env block" or "it is your RoCE cabling" as the
  mechanism: the ablation isolates one variable, and the doc's env block is what
  the project validated on that hardware.
- `HCCL_BUFFSIZE` is not the cause here: too small gives its own explicit
  "HCCL_BUFFSIZE is too SMALL" line, not an event-wait timeout.
- The `dispatch success ... This warning log can be ignored` lines are PTA's
  per-work notices; they say the collectives completed, not that the MoE op did.
