# "图模式算子通信报错，只有开启单算子才能规避": the execution-mode split, no log

Signature family — a two-node (or DP>1) MoE serve whose body is one sentence:
图模式 has a 算子通信 problem and only 单算子 avoids it. Often the whole "log" is a
failed attachment (`<!-- Failed to upload "910B-1.md" -->`) plus a single INFO line
that merely proves compilation is on:

```text
(Worker_DP0_TP0_EP0 pid=167) INFO 06-11 06:32:34 [decorators.py:668] saved AOT compiled
function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash>/rank_0_0/model
```

Recorded #10370 (vllm-ascend v0.20.2rc1 `367b8e62`, two nodes × 910B, GLM-5.1-w4a8,
两个局点 both reproduce). Deliverable is the ablation ladder plus the missing
artefacts — **there is no mechanism to find without a paste**, and the mode split
itself is the classification (rule 15, working-control form: the same cluster
completes under single-op).

Not this page: a `profile_run` hang that names `aclnnMoeDistribute{Dispatch,Combine}V4`
+ `107020` (`aiv-mc2-env-ablation.md`), or a captured-graph first-request hang whose
paste carries the mode banner (`aclgraph-full-decode-runtime-hang.md`).

## `单算子` is the GE execution mode, not a config key

`单算子` / `图模式` is CANN's single-op (eager) vs graph execution pair. At
v0.20.2rc1 `rg -n "单算子" docs/ vllm_ascend/` returns exactly one hit, an
unrelated table row (`docs/source/_templates/Model-Deployment-Tutorial-Template.zh.md:240`),
so do not hunt for a "single-op switch": the reporter's workaround is one of
`--enforce-eager`, `--compilation-config '{"cudagraph_mode": "NONE"}'`, or a different
`cudagraph_mode`, and its launch line has to be asked for.

## First: the recipe the failure is being compared with (one call)

For the recorded model/hardware the pinned tutorial has the exact multi-node block —
`docs/source/tutorials/models/GLM5.md:360-420` (v0.20.2rc1) is A2 双机 and runs, on
**each** node, `HCCL_OP_EXPANSION_MODE="AIV"` + `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` /
`TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`, then
`--data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-address <node0_ip>
--data-parallel-rpc-port 13389 --tensor-parallel-size 8 --enable-expert-parallel
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--additional-config '{… "ascend_compilation_config": {"enable_npugraph_ex": true}}'
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'`.
A launch that reaches the same cards as **TP16 without DP**, drops the
`--data-parallel-*` triple, or drops MTP is a different graph from the validated one:
say which line differs before ranking kernels. (The single-node A2 block is
`:213-247`; both carry `HCCL_OP_EXPANSION_MODE=AIV`, i.e. the reporter keeping that
env is *following* the recipe, not misconfiguring it.)

## Second: which comm op can even be inside the graph (rank arithmetic)

`select_moe_comm_method` (`vllm_ascend/ascend_forward_context.py:233`, v0.20.2rc1)
picks the MoE communication method per batch, so a graph-replay fault has to belong to
the method the shape selects:

| SoC | rule at this pin | lines |
|---|---|---|
| any | no EP, or `ep_world_size == 1` → ALLGATHER | `:271-272` |
| A2 (910B) | **MC2** iff `num_experts // ep_world_size <= 24` **and** `ep_world_size = world_size_across_dp // pp >= 16` **and** `num_tokens <= mc2_tokens_capacity`; else ALLGATHER | `:273-282` |
| A3 | `FUSED_MC2`/`MC2`/`ALLTOALL` by `num_tokens` vs capacity and `enable_fused_mc2` (1 = `dispatch_ffn_combine`, 2 = `dispatch_gmm_combine_decode`) | `:284-310` |
| A5 | `MC2` iff tokens fit and `world_size_across_dp > 1` | `:311-318` |

DP2×TP8 on A2 is `world_size_across_dp = 16`, `pp = 1`, so `ep_world_size = 16` and the
MC2 dispatch/combine **is** in the captured graph whenever the checkpoint's routed-expert
count divided by 16 is ≤ 24 — ask for that number (`config.json`) instead of assuming it.
On A2 there is no MC2 off-switch to name: the only MC2 knobs in the tree are
`VLLM_ASCEND_ENABLE_FUSED_MC2` (`envs.py:97-104`, default 0, which the selector reads
only in the **A3** branch) and `enable_mc2_hierarchy_comm` (`ascend_config.py:269`), which
`platform.py:584-589` refuses together with fused MC2. Offer an *override of the
selector* only as a diagnostic, and only if you can name it in the tree.

## The ladder, one axis per launch, cheapest first

1. `--compilation-config '{"cudagraph_mode": "NONE"}'` — reproduces their working
   single-op run and fixes the phase. The proof line is the mode's own log
   (`cudagraph_mode` NONE also forces `compilation_config.mode = NONE` and
   `enable_npugraph_ex = False`, `platform.py:404-406`).
2. `PIECEWISE` vs `FULL_DECODE_ONLY` — the run that separates **ACL-graph replay** from
   the **npugraph_ex GE graph**: the PIECEWISE branch force-disables
   `enable_npugraph_ex` (`platform.py:407-426`) while the FULL_DECODE_ONLY branch keeps
   the configured value (`:427-451`). Same cluster, same recipe, one axis.
3. `HCCL_OP_EXPANSION_MODE` AIV → unset — the AIV × MC2 discriminator, matrix in
   `aiv-mc2-env-ablation.md`; on this shape it changes the selected path, so treat a pass
   as a workaround, not a cure.
4. Only then per-kernel evidence: the plog `[AIC_INFO] fault kernel_name=` line and the
   6-digit code. Two nodes failing at 两个局点 with the same serve line points at the
   op, not at a card.

Trap: **`ASCEND_LAUNCH_BLOCKING=1` cannot test this lane at this pin** — with any graph
mode on, `platform.py:457-467` raises `ACL graph is incompatible with
ASCEND_LAUNCH_BLOCKING=1`. Pair it with step 1 or leave it out; a reporter's "tried it,
same" is meaningless unless their run was eager.

## The route that is already closed: the compile cache

The `saved AOT compiled function … torch_compile_cache` line is `decorators.py`
*saving* an artefact — a graph-mode tell, never the fault owner. On this issue a
maintainer proposed `VLLM_DISABLE_COMPILE_CACHE=1` and the reporter answered
**没有效果** (2026-06-12). Do not rank the cache, do not ask for that ablation again on
this shape, and do not read the absence of a cache error as an instalment: the same
`torch_compile_cache` path shows up in healthy starts.

## What to send back

The failed attachment as plain text, the plog `[AIC_INFO] fault kernel_name=` block with
the 6-digit code, **both** nodes' full serve lines (so the recipe diff above is decided),
and which knob produced their working 单算子 run. State that the variable blamed in the
title (`图模式`) is a mode, not a component, and that the plan above is an ablation
matrix rather than a fix.
