# MoE accuracy wrong when `data_parallel_size` / expert parallel > 1

Report shape: a MoE model answers plausibly (or fine under plain TP) but scores near-chance once
`data_parallel_size`, `enable_expert_parallel` or `expert_tensor_parallel_size` is added, with no
traceback. Verified by reading the clones at the tags named per statement; nothing was executed on
an NPU, so every runtime attribution below is a candidate backed by code shape plus thread
statements. GLM-specific DP+EP and stale-`slot_mapping` defects stay in
[glm-moe-accuracy.md](glm-moe-accuracy.md); PCP/DCP and the All2All/MC2 dispatch faults of the
0.18 line are in [context-parallel-moe-paths.md](context-parallel-moe-paths.md). Read this file
first when the *only* changed variable is the DP/EP dimension.

## Step 1: find out which MoE forward the model actually runs

Two MoE implementations coexist in the 0.9 line, and only one of them carries the group
communication:

- `vllm_ascend/ops/fused_moe.py::AscendFusedMoE` — plugin-owned layer. At `v0.9.2rc1` it is
  instantiated only by the plugin's own model files: `git grep -n 'AscendFusedMoE' <tag> --
  vllm_ascend` returns `models/deepseek_v2.py:330`, `models/deepseek_dbo.py:136` and
  `quantization/quant_config.py:108`. Its `forward` implements the DP/TP/EP story (below).
- upstream `vllm/model_executor/layers/fused_moe` whose **unquantized method is monkeypatched** by
  `vllm_ascend/ops/common_fused_moe.py` (`UnquantizedFusedMoEMethod.__init__` and `.forward_oot`,
  assigned at the file's end). The module is imported by `vllm_ascend/ops/__init__.py:21`, and
  upstream dispatches to it automatically: `CustomOp.__init__` sets
  `self._forward_method = self.dispatch_forward()` and returns `self.forward_oot` on the
  out-of-tree platform (`vllm v0.9.2:vllm/model_executor/custom_op.py:86,118`).

Every MoE model the plugin does **not** re-implement therefore takes the patched upstream path. At
`v0.9.2rc1` that includes Qwen3-MoE: `vllm_ascend/models/qwen3_moe.py` is one class,
`CustomQwen3MoeForCausalLM(Qwen3MoeForCausalLM)`, so its expert blocks are upstream's. Check the
model's plugin file (`git show <tag>:vllm_ascend/models/<model>.py | grep -n 'class ')` — a single
`Custom...ForCausalLM` subclass means "upstream MoE layer, patched method".

## Step 2: the 0.9.x unquantized path performs no group communication

At `v0.9.2rc1` `vllm_ascend/ops/common_fused_moe.py:42-108` `forward_oot` is:
`select_gating_top_k_softmax_experts`/`select_experts` → `fused_experts(expert_map=...)` (or
`fused_experts_moge` for `is_310p()`). It contains **no** `get_dp_group()`, no
`get_fused_moe_state()` dispatch, no TP all-reduce: the one check that settles it is
`git grep -c 'get_dp_group\|all_reduce\|all_gather\|reduce_scatter' <tag> --
vllm_ascend/ops/common_fused_moe.py` — 0 at `v0.9.2rc1` and `v0.10.0rc1`, 2 at `v0.10.1rc1`.

`AscendFusedMoE.forward` in the same tree does all of it:

- state selection `get_fused_moe_state(ep_size, is_prefill, is_deepseek_v3_r1)`
  (`vllm_ascend/utils.py:473-487`): `AllGatherEP` only when
  `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` and `ep_size > 1` and the model has 256 experts
  (`is_deepseek_v3_r1`); `ep_size == 1` → `NaiveMulticast` at prefill / `AllGather` at decode;
  `ep_size < 16 or is_prefill` → `All2All`; otherwise `MC2`. A 128-expert model with EP=2 is
  therefore in `All2All`, which the patched path never dispatches.
- `if self.dp_size > 1` → `get_dp_group().all_gather(hidden_states, 0)` and the same for
  `router_logits` (`fused_moe.py:1326-1347`), then `data_parallel_reduce_scatter` (AllGather) or
  `get_dp_group().all_reduce` (NaiveMulticast) at `:1397-1407`, and
  `tensor_model_parallel_all_reduce` at `:1414-1418`.

So on a `dp_size > 1` or `ep_size > 1` run, a model on the patched path skips the DP all-gather,
the reduce-scatter and the EP rank shuffle together — a near-chance score is the expected shape,
not a small numeric drift.

Evidence status: the code asymmetry is verified at `v0.9.2rc1`; the attribution to a reported score
is a candidate. The maintainer statement that matches it is the duplicate of bug report #1791,
issue **#1289** comment of 2025-07-14: the DP handling of the Qwen3 MoE "all_gather solution" must
follow `fused_moe.py`'s DP logic but "follows the `common_fused_moe.py` logic. There is no dp logic
here, so the lack of dp communication leads to incorrect accuracy." Read the duplicate target's
comments for the mechanism — the closed report's own comments only carried the wrong explanation.

## Recorded reproduction and its signature

The same thread carries a runnable reproduction on `v0.9.2rc1` — `python
examples/offline_data_parallel.py --model=Qwen3-30B-A3B --dp-size=2 --tp-size=2 --enforce-eager`
(the script is at `examples/offline_data_parallel.py` in every 0.9/0.10 tag). Its output is the
signature to ask a reporter for: **one DP rank degenerates and the other answers normally** — DP
rank 0 gave `' 777777777777777'`, repeated `the`, `1999 1999 1999`, while DP rank 1 produced
plausible sentences on the same prompts; the same split appeared with ACL graph enabled. Rank-wise
divergence on identical prompts is worth more than any score table: it rules out the checkpoint,
the prompt format and the harness in one shot.

Two merged fixes in the thread are *not* this defect: PR #1441 (tried, reporter: not solved) and
PR #1678 "Fix accuracy problem caused by mask pollution" (`643e6f548`, merged 2025-07-10) — the
latter is already inside `v0.9.2rc1`, and the reproduction above still failed there. Check
`git log --oneline <tag> --grep='<pr title>'` before presenting a thread-named PR as the fix.

Cheapest discriminator, before any source quoting: rerun the same prompt set with the DP dimension
removed (`data_parallel_size=1`, TP grown to the same world size) and with `enable_expert_parallel`
off. A score that returns to normal only when the DP/EP dimension disappears is this class; a score
that stays low is prompt/checkpoint/harness
([accuracy-eval-harness-gaps.md](accuracy-eval-harness-gaps.md)).

## Garbled output that appears only when `--enable-expert-parallel` is added

Report shape (recorded on `v0.17.0rc1`: MiMo-V2-Flash BF16, 16 cards, `--data-parallel-size 2
--tensor-parallel-size 8 --enable-expert-parallel --enforce-eager`, one `temperature: 0` request):
garbage with EP on, plausible output with EP off, no traceback and no numbers to compare. The
reporter's own follow-up fixed it by setting `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` while **keeping**
EP, following the plugin's MiniMax-M2.5 FAQ ("Q: What should I do if the output is garbled in EP
mode? A: It is recommended to keep `--enable-expert-parallel` and
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`"). That FAQ is `main`-era: `v0.17.0rc1`'s
`docs/source/tutorials/models/` has no MiniMax file, so label it *documented on `main`*, not at the
reported tag.

Why the two flags are one supported shape, verified at `v0.17.0rc1`:

- `enable_sp()` (`vllm_ascend/utils.py:763-786`) is exactly `VLLM_ASCEND_ENABLE_FLASHCOMM1`, or the
  legacy `VLLM_ASCEND_ENABLE_FLASHCOMM=1`, or `enable_shared_expert_dp` (which forces it on and
  logs `shared_expert_dp requires enable_sp = True`).
- `vllm_ascend/platform.py:408-420` asserts the mirror constraint: with `enable_sp()` it requires
  `tp_size > 1` and, for a MoE model, `enable_expert_parallel` ("Flash Comm v1 requires
  enable_expert_parallel=True for MoE models"). EP *without* FLASHCOMM1 is the combination nothing
  validates.
- On the ALLGATHER MoE strategy, `PrepareAndFinalizeWithAllGather.prepare`/`.finalize`
  (`vllm_ascend/ops/fused_moe/prepare_finalize.py:299-317`, `:401-414`) dispatch on `enable_sp()`:
  true → `_prepare_with_ep_group`/`_finalize_with_ep_group` (`:319`, `:416`); false →
  `_prepare_with_dp_group`/`_finalize_with_dp_group` (`:350`, `:430`), which gather across the
  **DP** group and reduce-scatter back. The class docstring (`:277-295`) states both pipelines —
  `Attn → TP AR → DP AG → MoE → DP RS → TP AR` versus `TP RS → EP AG → MoE → EP RS` — so on this
  strategy FLASHCOMM1 is what moves the MoE token exchange onto the EP group.
- "EP is on" does not decide the strategy. `select_moe_comm_method`
  (`vllm_ascend/ascend_forward_context.py:203-275`) picks per forward: no EP or EP world 1 →
  ALLGATHER; A2 + EP → MC2 when `num_tokens <= mc2_tokens_capacity` and
  `world_size_across_dp / pp >= 16`, else ALLGATHER; A3 + EP → FUSED_MC2/MC2/ALLTOALL by quant,
  EP size and MTP; A5 → MC2/ALLTOALL. Read the selector for the reported topology before claiming
  the dispatch above is the path that ran.

Boundary: the workaround is reporter-verified, the dispatch asymmetry is a candidate explanation
and nothing was executed here. Which strategy ran is decided **per forward** by
`select_moe_comm_method`, so compute it for the reported shape instead of assuming ALLGATHER:
on A2 at `v0.17.0rc1` (`:235-240`) MC2 needs `num_tokens <= mc2_tokens_capacity` and
`world_size_across_dp / pp >= 16`; from `v0.18.0rc1` (`:245`) the A2 gate is restated
with `ep_world_size >= 16` and `num_experts // ep_world_size <= 24` (the per-tag table is
in [context-parallel-moe-paths.md](context-parallel-moe-paths.md)). `mc2_tokens_capacity` is not a fixed 512 —
`set_mc2_tokens_capacity` (`vllm_ascend/ascend_forward_context.py:166-178`, called from
`worker/model_runner_v1.py:334` with `max_num_reqs` and `uniform_decode_query_len`) takes
`max_cudagraph_capture_size` when capture sizes exist, else
`min(max_num_reqs * uniform_decode_query_len, 512)`. On the recorded shape (`--enforce-eager`,
`--max-num-seqs 1`, no MTP) that is ~1 token, so prefill takes ALLGATHER — where the garbage is
generated — and MC2 covers only single-token decode steps; a report raising `--max-num-seqs` or
adding MTP moves the boundary. Keep the mechanism a candidate either way.

Keep it apart from the opposite-polarity defect in
[minimax-m2-family.md](minimax-m2-family.md): there an *extra* TP all-reduce runs when
FLASHCOMM1/EP is off on the MiniMax patch — same flag, different mechanism. For a DeepSeek-V3.2
run with PCP/DCP the pairing is also unavailable by documentation: `v0.18.0rc1`'s own Known Issue
says PCP & DCP do not yet work with FlashComm1
([context-parallel-moe-paths.md](context-parallel-moe-paths.md), recorded #8579 whose stated
workaround is enabling EP alone). For a VL checkpoint
`platform.py:409` refuses Flash Comm V1 outright and redirects to
`--compilation-config '{"pass_config": {"enable_sp": true}}'`, so an EP-mode VL report cannot use
this workaround ([vision-tower-accuracy.md](vision-tower-accuracy.md)).

## Fix boundary

- **v0.10.1rc1** is the first tag where the patched unquantized path regains communication:
  `992271b02` (`#2125`) adds `vllm_ascend/distributed/moe_comm_method.py`, and
  `common_fused_moe.fused_experts` then takes `moe_comm_method = get_forward_context()
  .moe_comm_method`; `get_dp_group` first appears in that file with `a6bb502e7` (`#2469`), whose
  first tag is also `v0.10.1rc1`.
- **v0.11.0rc1**: the two files are merged (`94dd83281`, `#3176`); `ops/fused_moe.py` no longer
  exists and the DP comm moved to `ops/moe/fused_moe_prepare_and_finalize.py`
  (`_prepare_with_dp_group` all-gathers at `:357-372`, `_finalize_with_dp_group` reduce-scatters /
  all-reduces at `:407-419`).
- Do not present the first communication-bearing tag as the first *correct* tag for every model:
  the `v0.10.0rc1 → v0.10.1rc1` window is also where GLM-4.5's DP+EP shared-expert reduce defect
  starts ([glm-moe-accuracy.md](glm-moe-accuracy.md), Defect B). Name the tag and the defect you
  are fixing, or recommend removing the DP/EP dimension instead.

## Don't inherit a closure reason that the report itself contradicts

#1791 was closed on "`--apply_chat_template` + `--fewshot_as_multiturn` cause it", but the same
reporter's TP=4 run used both flags and scored 0.884, and the maintainer's retest was TP=4/DP=2/
`expert_tensor_parallel_size=4` (a different expert-parallel shape) with only the unflagged number
printed. Before repeating a closure reason, hold it against every row of the reporter's own table;
the contradiction usually survives the maintainer's edit and is the finding.
