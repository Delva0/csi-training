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

**The older kernel name and the timeout code are the same family.** `…DispatchV2` +
`507014 (aicore timeout)` + `fftsplus aivector error` on a paste that sets
`HCCL_OP_EXPANSION_MODE=AIV` is this lane too, not an exclusion: recorded #8294
(v0.18.0rc1, 2×A2, GLM-4.7-W8A8, EP16) is that variant, and the ablation its reporter
reported does not name this variable (§below). Read it with
`mc2-dispatch-profile-run-hang.md` §Step 1b for the per-rank
capacity/symmetry arithmetic and §Step 4 for the ranking, and run the matrix below one
variable at a time.

## When the whole matrix has already been run

Recorded #8294 (v0.18.0rc1, 2×A2, GLM-4.7-W8A8, EP + DP2×TP8) came back on
2026-04-17 with "I have tried all combinations of these 3 envs and it fails at the
same phase (torch.compile). Even I unset the 3 envs, it fails too." Treat that as a
**completed** ablation and do not offer "unset them all" as a further experiment —
that is the run they already did. Two things it does **not** settle:

- **Which three.** The public text does not name them, so resolve the set from the
  transcript instead of asking: it is normally the knobs the previous answer's
  ablation rows listed. On #8294 those were
  `VLLM_ASCEND_ENABLE_FLASHCOMM1` / `VLLM_ASCEND_ENABLE_FUSED_MC2` /
  `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` (see `mc2-dispatch-profile-run-hang.md` §Step 4
  for why each is a no-op at that pin). Ask for the three names in one line only if
  the thread does not show them.
- **Whether this page's AIV row is closed.** Under that reading it is **open**, not
  closed: both #8294 node scripts set `HCCL_OP_EXPANSION_MODE=AIV`, the device block
  is `fftsplus aivector error … aicore timeout` on the MC2 dispatch inside
  `profile_run`, and that is this page's shape (#14928's reporter showed *AIV unset +
  MC2 passes*). So the informative next run is row 2 or 3 of the matrix below, one
  variable at a time — **keep AIV and drop `expandable_segments`** (FAIL ⇒ AIV × MC2
  owns it) and **drop AIV, keep `expandable_segments`** (PASS ⇒ dropping AIV is the
  workaround). Only if the reporter says `HCCL_OP_EXPANSION_MODE` was one of the three
  is the matrix done.

Either way the phase word in their title is the reporter's own label (their traceback
shows `profile_run`'s extra MC2 dummy run, not the compile step), and the other lanes
are the per-node symmetry of the derived `global_bs`/`mc2_tokens_capacity`
(`mc2-dispatch-profile-run-hang.md` Step 1b, one call:
`scripts/mc2-capacity-asymmetry.sh`), the cross-node transport check, and the headless
node's own log — a single A2 node cannot stand in for the last two, since `tp8 dp1`
gives `ep_world_size = 8` and MC2 is never selected there.

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
   transport fault fails on the same op with an HCCL return code instead
   (`hccl_ret 9`, #9447). A **passing** `hccl_test` does not clear that lane —
   #9447's inter-node `hccl_test` was normal while the dispatch failed at profile,
   and the reporter's 2026-06-22 follow-up still had it open on two fabrics (row below).
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
| #3881 | DeepSeek-R1 + torchair graph, DP4 × TP4, MTP 1, `HCCL_OP_EXPANSION_MODE=AIV` | the body states the mechanism, not just a symptom: under torchair the **MC2 op runs in parallel with `all_reduce`** because comm ops take their own stream when no result dependency exists, and AIV moves comm ops from aicpu to the vector core, so they contend for cores with other parallel AIV comm ops → 编图卡死 / comm timeout, closing with 当前不支持 … AIV. Use it as the mechanism owner when a paste shows a hung AIV comm op, labelled as the issue body's statement, not a maintainer ruling |
| #13688 | 910C ×8, GLM-4.6V TP8+EP8, AIV on, NPUGraph | capture failure `rtEventRecord ... 507009` and replay deadlock (AIVector 8-16 %, AICore 0 %), plus "MC2 MoE dispatch/combine still deadlock with AIV"; the capture-phase codes, the compute-only task-group inventory and the two-owner split: `aiv-ric-task-group-capture.md` |
| #9447 | v0.18.0rc1, GLM-5.1, 2×A3: RoCE 组网 fails, 灵衢 direct connect works (reporter's premise) | `aclnnMoeDistributeDispatchV4` comm failure, `hccl_ret 9` at profile. **Unresolved** — the reporter's own follow-up (2026-06-22) says 没有解决 and the direct-connect setup was failing too, so the "works on the other fabric" premise did not hold and no RoCE-only knob is confirmed. The `hccl_ret`-9 **sibling** of this family, not AIV support: `mc2-dispatch-profile-run-hang.md` §Step 3c |
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
