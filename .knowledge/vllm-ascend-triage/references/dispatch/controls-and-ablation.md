# Controls and ablations: what a working path, and your own run, can prove

## Scope

Rule 15 of `SKILL.md` in full. The rule is short — a working sibling path in the
reporter's own evidence constrains the cause, and a mechanism that also predicts the
working path failing is not yet the answer. This page carries the five forms that
recurs, each with the recorded cost of getting it wrong, and the reasoning each one
licenses:

| Form | What it decides | Section |
|---|---|---|
| a sibling entry point, flag, model or card works | which mechanism is *excluded* before ranking | first |
| your own ablation (`--enforce-eager`, a spec/async toggle) | which *phase* was removed, and whether the paste proves it ran | *Your own ablations* |
| a workaround named in the report's prose | which rung is first | *A named workaround* |
| a tested-configuration matrix in the body | one verdict line per row, and which rows are the controls | *A matrix* |
| a control the reporter asserts, or built from a lower-level API | a premise, not a finding | last two |

Carry this page's distinction into the short entry: a report's own control is
evidence about *its* run; your ablation is evidence about the phase you removed;
neither is a finding until the log line that proves it is in the paste.

## A working sibling path in the reporter's own evidence

When the same box fails on one entry point and works on another — offline `LLM()` vs
`vllm serve`, one flag, one model, one card — a mechanism that also predicts the
working path fails is not yet the answer: name the difference first and rank the
mechanism below it, with the log line or ablation that separates them. #1534 ranked
"mindie_turbo replaces RMSNorm" above the working `vllm serve` path and left the
follow-up ("是因为在线推理默认没有开启 acl graph 吗?") unanswered. Two log lines
settle which path compiled and which process loaded the plugin — `PIECEWISE
compilation enabled on NPU` / `Compilation disabled, using eager mode by default`,
plus the wheel's INFO line
(`../image-dependency-faults/mindie-turbo-graph-break.md`); `enforce_eager=True` on
the failing path is the one-line ablation.

## Your own ablations: an ablation only rules out a phase it removes

**An ablation only rules out a phase it actually removes, and its own log line has to
be in the paste.** `--enforce-eager` is neither universal nor free: at vllm-ascend
v0.22.1rc1 it removes torch.compile, the ACL graph and even the `Using OOT custom
backend for compilation` banner (`vllm/config/vllm.py:1032-1038` →
`decorators.py:390` → `platform.py:590-592`), while the weight load runs identically
in both lanes — so it cannot test a load-phase stall, and a reporter's "tried it,
still stuck" eliminates nothing unless the paste carries that flag's own lines
(`Compilation disabled, using eager mode by default`, `Cudagraph is disabled under
eager mode`). Pick the ablation that removes the suspected phase, name the line that
proves it ran (`../startup-failures/compile-backend-banner-then-silence.md`).

**Reachability never cancels the run: a flag can move a phase it does not contain.**
`--enforce-eager` cleared a *sporadic* `507035` inside CANN's
`ApplyTopKTopPWithSorted` sampling kernel although the sampler runs after the forward
and outside the captured region (reporter-confirmed #2763, and the same comment
prices it at ~5 tok/s against ~40 tok/s — so it is the isolation run first, never the
recommendation; at v0.10.1rc1 the flag's own line is `Compilation disabled, using
eager mode by default`, `vllm_ascend/platform.py:145-147`). Do not reason "that op is
not in the graph, so eager cannot help": pool, workspace, stream overlap and task
concurrency all change with the mode.

**The same trap sits inside `--speculative-config`:** with `--async-scheduling` on,
`method: mtp` forces the drafter eager regardless of `enforce_eager`
(`spec_decode/eagle_proposer.py:136-142` at v0.18.0rc1), so "toggled `enforce_eager`,
both crash" is one path run twice — check the serve command for `--async-scheduling`
before crediting that ablation (`../spec-decode-mtp/empty-first-response.md` Step 6b).

On one family that knob is not weakened but **erased**: for `deepseek_v32` with
`method: mtp` upstream vLLM sets `speculative_config.enforce_eager = True`
(`vllm/config/speculative.py:254-257`, commented "cudgraph with v32 MTP is not
supported") and `vllm_ascend/platform.py:359-369` force-clears it (`:504-517` at
v0.18.0rc1), so `"enforce_eager": true` in `--speculative-config` is a dead lever
there — global `--enforce-eager` still works, and the plugin, not upstream, owns the
support claim (`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`). Scope
that erasure to the family *and* the pin before reusing it: the upstream auto-set is
gated on `deepseek_v32` alone (`vllm/config/speculative.py:559-563` at vllm v0.23.0),
and the downstream force-clear exists at v0.18.0rc1 but is **gone** by v0.23.0, where
`platform.py:513-527` only touches the xlite graph config and
`model_config.enforce_eager`. So on a GLM-5.x (`glm_moe_dsa`) v0.23 launch the
tutorial's `"enforce_eager": true` inside `--speculative-config` is a **live** recipe
key — grep the reporter's launch before calling it dead
(`../spec-decode-mtp/index.md`, the GLM-5.x rule).

## A workaround named in the report's own prose is rung 1

When the narrative quotes a neighbouring issue's comments ("comments mention
disabling async scheduling or MTP, or adding sampling-stream synchronization as
workarounds"), the reporter has already said which phase to remove: take it as rung 1
and rank the mechanism it points at, ahead of any theory that needs a code path the
paste cannot show. #13964 is the recorded cost — the body carries exactly that line
for a progressive per-rank decode stall (`Running` > 0, flat generation counters, no
error, one rank still serving), and the session spent its whole output budget on a DP
cudagraph-mode-sync / skipped-all-reduce argument that the report's own "DP1 finished
116 requests" sentence refutes, shipping nothing. The resolution was KV-block reuse
under `--async-scheduling`
(`../scheduler-admission/kv-block-lifetime-async-scheduling.md`).

## A tested-configuration matrix in the body is the strongest form

**The strongest form of this evidence is a tested-configuration matrix in the body**
(`DP=1,PP=1,TP=32 → works` / `PP=2,TP=8 → works` / `PP=4,TP=8 → fails` / `DP=2 →
fails`): each failing row then has a named owner and one check, so the matrix answer —
one verdict line per row — *is* the G1 message, and the rows that work are the
controls that close the transport, capacity and labelling lanes. #10910 walked 45
turns and 966 K prompt tokens past that matrix and shipped **no** answer text at all.
Check each row against the code's own gates before crediting it as a control: on a
sparse-attention model a row that toggles `--decode-context-parallel-size` also moves
`tp` (`platform.py:723-731` asserts `dcp == tp` at v0.26.0rc1), so "dcp=8 fails,
dcp=16 works" compares two recipe rows, not one variable
(`../dsa-sparse-attention/dcp-c8-accuracy-garble.md`).

## A control platform

The same test applies when the reporter's *control* is another platform: "nv 卡多次
一致，npu 多次不一致" makes sampling randomness (`没有 seed`, `temperature > 0`) a
refuted cause, because it would randomise NV too — so it can be a one-line
confirmation but not the leading rank. #7180 led with it and the reporter's next
comment pasted `temperature: 0, top_p: 1, top_k: 1` — reproducibility between repeated
runs, not accuracy (`../accuracy-benchmarks/repeated-run-nondeterminism.md`).

## A control the reporter only asserts is a premise

**A control the reporter only *asserts* is a premise, not a finding.** "works on
灵衢直连, fails on RoCE 组网" / "works on the sibling node" / "works in eager" is the
reporter's claim about a run you cannot see, and it can be stale, mis-scoped or
already false: build the ranking so the answer still holds if it is. Recorded #9447
(v0.18.0rc1, GLM-5.1, 2×A3, `aclnnMoeDistributeDispatchV4` + `hccl_ret 9` at profile)
derived its whole answer from the fabric premise — the RoCE-only
`enable_mc2_hierarchy_comm` key — shipped **no message text**, and the reporter's own
public follow-up (2026-06-22) said 没有解决 with the direct-connect setup failing too.
So state one line that re-confirms which setup fails now, and prefer a control you can
make true locally — the **one-node** run of the same EP group, the same flags with the
feature off — over the fabric or node the reporter believes is healthy
(`../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` §Step 3c).

**Its most common form is "direct `curl` at one node works".** That is a *body*
control, not an endpoint control — a shorter prompt or a smaller `max_tokens` is
exactly the variable under test — and a `400` from a P/D node is its frontend
rejecting a body before the engine runs, with the reason in a response body the
example proxy discards (`send_request_to_service` / `stream_service_response_with_retry`
log only httpx's text). Recorded #15527 (v0.23.0, D node 400 through the proxy,
direct curl fine, `--max-model-len` 40960 and 128000 fail while 1M works) is one
arithmetic, not a connector question: the proxy sends the prefill leg
`max_tokens = 1`, so a 400 there bounds the **prompt**
(`../pd-disaggregation/proxy-upstream-400-body.md`).

## A repro built from a lower-level API

**A repro the reporter built from a lower-level API is a premise about the *serving*
path, and it is the one to check before adopting its conclusion.** A tidy 4-row
matrix over `torch.npu` primitives ("baseline PASS / no-collective PASS / wrapped
collective FAIL / separate PASS — so the *wrapping* combination is the bug") does
prove that pair conflicts, and it usually belongs to a component you cannot fix from
here. What it cannot show is that the served failure travels through it. Check the
tree's own call sites for the primitive: one `rg -n 'graph_task_group_begin'
vllm_ascend/` at v0.22.1rc1 returns ten regions that each wrap a single compute op and
never a collective, so the reporter's `graph_task_group_begin` + `all_reduce` is a
combination the serving stack does not build. Recorded #13688 then has two findings
with two owners — `507009` (`ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT`) belongs to
torch_npu/CANN for the API pair, and the replay deadlock (`AIVector` 8-16 % /
`AICore` 0 %, no error) to the MC2 × AIV lane — and saying so is also the correction
the report's own "revert the patch" ask needs
(`../hccl-and-hardware/aiv-ric-task-group-capture.md`).
