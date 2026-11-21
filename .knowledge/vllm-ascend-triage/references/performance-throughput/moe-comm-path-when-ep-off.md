# A MoE perf report whose own command dropped `--enable-expert-parallel`

## Scope

A **throughput / per-token latency** complaint on a DP×TP (multi-node or
multi-card) **MoE** serve where the pasted command does **not** enable expert
parallelism — the flag is absent everywhere, present on one node only, or the body
says it was *removed on purpose* ("开 tp+dp+ep 会胡言乱语，所以改成 tp+dp"). No
error, no dump, the engine is healthy, and the model is a large MoE (GLM-4.5/4.6/5.x,
Kimi-K2, Qwen3-MoE, DeepSeek-V3.x).

The two complaints in that kind of body are **one configuration decision**, and the
performance half is a consequence of the accuracy workaround: turning EP off is what
moved every MoE layer onto the TP all-gather path. Answer both in one ranking —
"your garbled EP run is the thing to fix; the perf number you are asking about is
the price of having avoided it".

Not this topic: EP+DP that fails to *start* or a selector/tiling crash
(`../feature-enable-recipes/moe-expert-parallel-with-dp.md` owns the selector, the
MC2 capacity arithmetic and those fault lanes), a capacity/OOM plan
(`../deploy-sizing/index.md`), or a low rate with EP already on (start at `index.md`).
The same `ep_size < 16` selector has a **fault** reading — when that all-gather
`permute` skips `apply_a8_quantization`, a W8A8 grouped matmul can be rejected on the
activation's dtype (`161002`, `GMM:y dtype does not match with required
dtype[DT_BFLOAT16]`) — which is `../custom-ops-runtime-faults/quant-scale-dtype-gate.md`
rule 5, not this page.

## Step 1 — read the selector at the reported ref, one grep

At vllm-ascend **v0.10.1rc1** (`7e16b4a7`) the choice is made per batch from one
number, `ep_size`:

```python
# vllm_ascend/ascend_forward_context.py:87-88
ep_size = (get_ep_group().world_size if
           vllm_config.parallel_config.enable_expert_parallel else 1)
```

`_get_fused_moe_state` (`:26-40`) and `get_dispatcher_name` (`:43-52`) then give:

| `ep_size` | prefill | decode | dispatcher |
|---|---|---|---|
| `1` (**EP off**) | `NaiveMulticast` | `AllGather` | `TokenDispatcherWithAllGather` |
| `2 … 15` | `All2All` | `All2All` | `TokenDispatcherWithAll2AllV` |
| `≥ 16` | `All2All` | **`MC2`** | `All2AllV` prefill / `MC2` decode |

`ep_size` is the **EP group's world size**, i.e. the flattened `TP × DP` of that
layout, so a DP2×TP8 deployment has 16 ranks *either way* — with the flag it lands on
MC2 dispatch/combine fused into the expert GEMM; without it every MoE layer becomes a
TP all-gather + reduce-scatter around the expert compute. The cards did not change,
the flag did. (`VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` only produces
`AllGatherEP` when the checkpoint has `n_routed_experts == 256`, so a 128-expert GLM
is not rescued by it — `:28-31`.)

**Nothing logs the choice at this ref.** `setup_token_dispatchers`
(`vllm_ascend/ops/moe_dispatcher/token_dispatcher.py:48-60`) has no logger, and there
is no `moe_comm_type` / `[EP Rank r/N]` line — those arrive in later releases. Do not
ask the reporter for a line that does not exist here: the evidence is the serve flags
plus a same-request A/B. Re-grep before quoting this table at another version: at
v0.12/v0.13/v0.17 the same decision moved to `_select_moe_comm_method` /
`select_moe_comm_method` and gained the A3 `FUSED_ALLTOALL` row
(`../feature-enable-recipes/moe-expert-parallel-with-dp.md`).

## Step 2 — measure the axis before ranking mechanisms

The reporter's "慢一倍" is a *decode-step* number here, not a prefill one. Derive it
from the benchmark table with
`bash <skill_root>/scripts/benchmark-table-normalize.py` rather than trusting the
pasted per-token columns (`cross-run-ratio-normalization.md`). Recorded #2833:
Ascend c=8 → step 138.7 ms; the H800 reference, same concurrency → 68.5 ms; prefill
per input token 0.80 ms vs 1.01 ms *in Ascend's favour*. So the whole gap sits in the
decode loop where the MoE comm path lives, and `--max-num-batched-tokens` /
chunked-prefill tuning cannot move it.

## Step 3 — anchor the recipe, and say whether it can be followed

At v0.10.1rc1 the release note points GLM-4.5 at the Kimi multi-node tutorial
(`docs/source/user_guide/release_notes.md:140` → `docs/source/tutorials/multi_node_kimi.md`),
whose validated command is `--data-parallel-size 4 --data-parallel-size-local 2
--api-server-count 2 --tensor-parallel-size 8 --enable-expert-parallel
--quantization ascend --max-num-batched-tokens 4096`, `HCCL_BUFFSIZE=1024`, plus
`{"torchair_graph_config":{"enabled":true}}` (`:73-96`).

That last key **cannot be used on GLM-4.5 at this ref**: `TORCHAIR_MODEL_LIST =
["deepseek", "pangu", "kimi_k2", "qwen"]` and `_check_torchair_supported` is a
lower-cased substring test (`vllm_ascend/ascend_config.py:20-28`), so `glm4_moe` hits
`NotImplementedError: Torchair graph mode only works with following model types …`
(`:189-196`), while the residual aclgraph path logs
`ACL Graph is currently experimental` for any model_type without `qwen`
(`:211-215`). Also absent: MTP — `worker/mtp_proposer_v1.py:20-28` imports only
`CustomDeepSeekMTP`, although upstream vllm v0.10.1 registers `glm4_moe_mtp`
(`vllm/model_executor/models/registry.py:267`). So "no validated GLM-4.5 recipe at
this release" is part of the answer: graph mode and spec decode are not levers for
this model here, and the tutorial must be run with the torchair block removed.

## Step 4 — the deliverable

Verdict sentence, then the ladder (each rung one launch):

1. **Make both nodes identical and put EP back on both.** Treat a paste with
   `--enable-expert-parallel` on node 1 and not on node 2 as *unverified*, not as the
   EP-off measurement: the flag decides group membership and the per-batch MoE path on
   both nodes, so ask which command actually ran (the plausible readings are a dropped
   line in the paste, or two engines with different collective membership — the latter
   normally fails rather than serves). Then re-measure with the flag on both; that is
   also the row that decides whether MC2 is reachable at all.
2. **If EP reproduces the garbling, that is the bug to report**, not a reason to
   stay on the all-gather path: send its output plus the flags. Rank the accuracy
   owners from `../accuracy-benchmarks/moe-gating-topk-precision.md` and
   `../accuracy-benchmarks/index.md`; keep the mechanism open until the garbled text
   is in hand (for #2833 the reporter's later comment is only a screenshot of the
   garbled tp+dp+ep output — reporter-confirmed symptom, mechanism unestablished).
3. Then the recipe deltas that are independent of EP: the W8A8 artifact with
   `--quantization ascend` (a bf16 GLM-4.5 doubles per-step weight traffic against an
   H800 running the native FP8 checkpoint), `--max-num-batched-tokens` 4096 vs 2048,
   `HCCL_BUFFSIZE=1024`, `--data-parallel-size-local`/`--api-server-count` per the
   tutorial.
4. Drop the tutorial's `torchair_graph_config` block on GLM-4.5 (step 3) — otherwise
   the launch fails at config time and the reporter concludes the recipe is wrong.

Recorded #2833 (v0.10.1rc1, 2×8×910B2, GLM-4.5, DP2 TP8, `--enable-expert-parallel`
on one node, `--max-num-batched-tokens 2048`, no `--quantization`) had this whole
chain by its 12th turn and shipped **no answer**: 22 assistant turns, the full
16 384-token budget, 656 characters of message text. The first message was owed at
step 1+2 (the flag and the two step times), not after the MoE arithmetic.

## Scope

Pins: vllm-ascend v0.10.1rc1 `7e16b4a7cdb1`, vllm v0.10.1 `aab5498`. Re-grep
`_get_fused_moe_state`, `get_dispatcher_name`, `TORCHAIR_MODEL_LIST` and
`multi_node_kimi.md` at another ref; the selector's location and the ≥16 MC2
threshold are the parts that drift. The GLM-4.5 garbling under EP is
reporter-confirmed as a symptom and unverified as a mechanism.
