# The legacy `VLLM_ENABLE_MC2=1` knob on a layout whose expert-parallel group is 1

## Signature

A worker-only `TypeError` raised by plain Python, not by a kernel, on a DeepSeek-V2/R1
**W8A8** checkpoint whose script sets the env var before the engine is built:

```text
(VllmWorkerProcess pid=…) ERROR … [multiproc_worker_utils.py:238] Exception in worker
VllmWorkerProcess while processing method start_worker_execution_loop.
  … vllm_ascend/worker/model_runner.py:1441 execute_model
  … vllm_ascend/models/deepseek_v2.py:721 forward  →  :668  →  :583  →  :272  self.experts(
  … vllm_ascend/ops/fused_moe.py:1135  self.quant_method.apply(
  … vllm_ascend/quantization/quant_config.py:330  → vllm_ascend/quantization/w8a8_dynamic.py:697
  … vllm_ascend/quantization/w8a8_dynamic.py:156, in fused_experts_with_mc2
        moe_expert_num = len(expert_map) + global_redundant_expert_num
TypeError: object of type 'NoneType' has no len()
```

Tells, in the order they pay: every frame is `vllm_ascend/` or `torch/` (no aclnn, no
`EZ`/`507`/`ERR` code, nothing launched, so no plog/health/tiling check applies);
the deepest `vllm_ascend` frame is a **quant-MoE helper**, not the model; and the
script carries `os.environ["VLLM_ENABLE_MC2"] = "1"` (or an exported
`VLLM_ENABLE_MC2=1`) plus `quantization="ascend"`. Without that env line this page is
not the lane — after the selector era MC2 is chosen from rank arithmetic
(`../performance-throughput/moe-comm-path-when-ep-off.md`,
`../hccl-and-hardware/aiv-mc2-env-ablation.md`), and an EP-off layout cannot land
on it at all.

**The phase is part of the answer.** Weight load, `profile_run` and every **prefill**
step pass (the profile run forces `is_prefill = True`, `deepseek_v2.py:236-239`); the
process dies on the **first decode step**. The gate
is `if VLLM_ENABLE_MC2 and not is_prefill` and `is_prefill` comes from
`attn_metadata.num_prefills > 0` (`vllm_ascend/models/deepseek_v2.py:236-239`), so a
report that reads "it crashes at startup" is mis-set: nothing in the model or the
quantization is broken, one branch was entered.

## Mechanism (source-verified, vllm-ascend v0.9.0rc2 `8dd686d` / vllm v0.9.0 `5873877`; recorded #1243)

1. **The knob** — `vllm_ascend/envs.py:56-57` (`bool(int(os.getenv("VLLM_ENABLE_MC2", '0')))`),
   read once into `ops/fused_moe.py:40` and `quantization/w8a8_dynamic.py:32`.
2. **The dispatch** — `w8a8_dynamic.py:695` `if VLLM_ENABLE_MC2 and not is_prefill:` →
   `return fused_experts_with_mc2(…, expert_map=expert_map, …)`. The sibling branch
   two lines down, `:710` `elif self.torchair_graph_enabled or self.ep_group.world_size == 1:`
   → `fused_experts(…)`, is the path this same launch takes **without** the env var.
3. **The unguarded dereference** — `fused_experts_with_mc2` (`:140`) declares
   `expert_map: torch.Tensor = None` (`:148`) and its third statement is
   `len(expert_map)` (`:156`). The other three helpers in the file all test it:
   `fused_experts_with_all2all` (`:267`, guard `:291`) and `fused_experts` (`:394`,
   guards `:361`, `:412`, `:477`). MC2 is the only branch that assumes it is a tensor.
4. **Where `None` comes from** — `ops/fused_moe.py:1048`
   `self.local_num_experts, self.expert_map = determine_expert_map(self.ep_size,
   get_ep_group().rank_in_group, self.global_num_experts)`, and upstream returns
   `(global_num_experts, None)` whenever `ep_size == 1` (vllm v0.9.0
   `vllm/model_executor/layers/fused_moe/layer.py:661-662`). The only other source of
   a non-`None` map is static EPLB (`ascend_config.expert_map_path`,
   `ops/fused_moe.py:1033-1047`, which also sets `global_redundant_expert_num` and
   `log2phy`) — so an EPLB-served deployment is the exception, not the rule.
5. **Why `ep_size` is 1 in the reported launch** — `vllm_ascend/platform.py:144-158`:
   `expert_tensor_parallel_size` defaults to `tensor_parallel_size` (`:144`); only
   `parallel_config.enable_expert_parallel` forces it to `1` (`:148-149`);
   `expert_parallel_size = world_size_across_dp // expert_tensor_parallel_size`
   (`:156-158`). `tensor_parallel_size=2`, one node, no `--enable-expert-parallel`
   → `2 // 2 = 1`.

So the verdict sentence is: **`VLLM_ENABLE_MC2=1` with `ep_size == 1` is a
combination this MC2 branch cannot execute** — `expert_map` is `None` by construction,
and the branch dereferences it. The env var or the EP group has to change; nothing in
the checkpoint, the SoC, or the CANN/torch_npu pair is implicated.

## Version scope: the env knob is the dating tell

- Present at v0.9.0rc2 (`envs.py:56`); **zero** hits in `vllm_ascend/` at v0.20.2rc1
  (`rg -n VLLM_ENABLE_MC2 vllm_ascend/` is empty). Between those two pins the knob was
  retired in favour of the per-batch selector, which reaches MC2 only from rank
  arithmetic — EP on, and ≥16 ranks on the A2 branch — so no EP-off layout can land
  there. The selector pages must not be quoted at a 0.9-era report (nor this page at
  a 0.20-era one).
- **No recipe, no CI.** At v0.9.0rc2, `rg -n -i mc2 docs/` returns **0** hits and
  `rg -n VLLM_ENABLE_MC2 tests/ .github/` returns **0** hits. Quote that as coverage —
  it explains why "add EP and it will work" has no local evidence — never as the cause.
- **The EP flag is era-dependent.** `rg -n enable_expert_parallel vllm_ascend/` is
  empty at v0.8.5rc1 (`../performance-throughput/multicard-scaling-and-vendor-baseline.md`,
  where "there is no EP flag" is that pin's fact) but returns `platform.py:148` at
  v0.9.0rc2. A v0.8.x report therefore has **no** EP rung; a 0.9/0.10 report does.

## Ladder

1. **Drop `VLLM_ENABLE_MC2` and re-run the same script unchanged.** The dispatch falls
   to `:710` through `expert_map is None`-safe `fused_experts` — the path the failing
   run's own prefill already completed. Acceptance: the same prompts generate.
2. **If MC2 is the goal, the EP group must be ≥ 2**: `LLM(..., enable_expert_parallel=True)`
   (`--enable-expert-parallel`), which sets `expert_tensor_parallel_size = 1` and
   `expert_parallel_size = world_size_across_dp` (`platform.py:148-158`), making
   `determine_expert_map` return a real tensor. Say in the same breath that MC2 is
   undocumented and untested at this ref, so a **different** failure next is expected
   and is not evidence that the flag is wrong; ask for its first error block rather
   than predicting the next rung. Do not claim the EP run works — no local artefact
   shows one.
3. **Do not offer "guard the `None`" as the fix.** `moe_expert_num` means the *global*
   expert count, which `apply()` already holds and asserts against the router width
   (`global_num_experts`, `w8a8_dynamic.py:642`, assert `:653-654`), so the repair is to
   pass it into `fused_experts_with_mc2` and use it when `expert_map is None`, mirroring
   the siblings' guard, plus a test. A bare guard would let an all-to-all run with
   `all_to_all_group_size == 1`; the acceptance is **token-identical output against the
   non-MC2 run**, not "no exception".

## What the answer owes

- The phase (first decode step, not startup), the two numbers (`expert_map is None`
  because the launch computes `ep_size = 2 // 2 = 1`), and the one-line remedy
  (unset the env var) in the first paragraph.
- The ask list: the full `LLM(...)` kwargs (any `additional_config`, in particular
  `expert_map_path`/EPLB, and `tensor_parallel_size`/`data_parallel_size`), the
  checkpoint's `config.json` `n_routed_experts`/`n_shared_experts` when the model name
  says a modified ("lite") expert set — the redundancy arithmetic changes with it —
  and, after rung 1 or 2, the first error block only.
- Scope: verified only at the two pins above, for the DeepSeek-V2-family W8A8 dynamic
  quant method. Whether the MC2 operator is supported on the reporter's SoC and
  CANN/torch_npu pair is **not** knowable locally; say so instead of implying it.
