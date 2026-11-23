# 310P MoE + `--enable-expert-parallel`: the routed output is `ep_size`× too wide

Scope: a `-310p` image (Atlas 300I Duo / `Ascend310P3`) serving a **MoE** checkpoint
(Qwen3-30B-A3B class) with `--enable-expert-parallel`, and engine start dying in the
memory-profiling dummy run with `RuntimeError: shape '[<T>, <H>]' is invalid for input
of size <M>` whose deepest frame is upstream
`vllm/model_executor/models/qwen3_moe.py` `final_hidden_states.view(orig_shape)`. The
deliverable is "drop the EP flag, on that image EP cannot work", plus the coverage
fact that makes it a flag question rather than an engine defect.

Not this page: the same 310P MoE family **serving** and answering wrong but fluent
(`../accuracy-benchmarks/moe-gating-topk-precision.md`); a 310P MoE dying on an
`aclnn<Op>` / `EZ1001` op gate with no Python frame (`atlas-300i-310p.md`, its
v0.14.0rc1 bullets); the EP dispatch that *does* exist on A2/A3
(`../custom-ops-runtime-faults/moe-init-routing-tiling.md`); and the two other
`invalid for input of size` readings — a graph-pad request/token mismatch or a rope
cos/sin global (`../mixed-batch-splits/decode-pad-request-vs-token-count.md`,
`../spec-decode-mtp/rope-cos-sin-slice-token-mismatch.md`).

Recorded case: **#1813**, image `quay.io/ascend/vllm-ascend:v0.9.2rc1-310p`
(tree v0.9.2rc1 `b5b7e0e`) + vllm v0.9.2 (`a5dd03c`), 4×300I Duo in a container
(`--device /dev/davinci2 … /dev/davinci5`), `vllm serve /models/Qwen3-30B-A3B
--tensor-parallel-size 4 --enable-expert-parallel --enforce-eager --dtype float16
--max-model-len 32768 --max-num-seqs 32`. Reporter's question is the capability
one — "I want to know how can i run qwen3-30b-a3b on the machine?". Weights load
cleanly on all four ranks (`Loading model weights took 14.2915 GB`), so no memory,
`max_model_len` or TP-scaling lane is reachable from this paste.

## Step 1 — divide first: the quotient is a parallel size, not a token count

`M / (T × H)` = `16777216 / (2048 × 2048)` = **4**. Both widths come from the paste:
`T = 2048` is the profiling batch (`Chunked prefill is enabled with
max_num_batched_tokens=2048` above it, and `profile_run` calls
`_dummy_run(self.max_num_tokens)`), `H = 2048` the checkpoint's hidden size. Nothing
in that pair is a capture pad, a request count or an async-scheduling gap, so neither
of the two recorded division readings applies. A quotient that equals a **parallel
size** is the third reading: the MoE layer returned `ep_size` copies of the token
rows. The paste already carries the number — every rank line reads
`rank k in world size 4 is assigned as DP rank 0, PP rank 0, TP rank k, EP rank k`,
i.e. `ep_size = tp_size = 4` because `--enable-expert-parallel` with TP=4 and no DP
makes the EP group the TP group.

Cheapest confirmation is the ablation itself: the same command **minus**
`--enable-expert-parallel`. A shape error that is deterministic at engine start (this
is the profiling run, so every launch dies) and whose factor equals the EP width has
no other lever.

## Step 2 — why 310P multiplies the rows at v0.9.2rc1 (`b5b7e0e`)

- `is_310p()` diverts the **unquantized** MoE forward to a moge body:
  `ops/common_fused_moe.py:82` `if topk_ids.shape[1] < top_k or is_310p():` →
  `fused_experts_moge(...)`. `is_310p()` reads the build stamp
  (`utils.py:71-77`, `_build_info.__soc_version__.lower().startswith("ascend310p")`),
  so the image, not the command, selects it.
- That body has **no dispatch step**: `fused_experts_moge`
  (`ops/fused_moe.py:559`) computes `ep_size = get_ep_group().world_size`,
  `local_num_experts = global_num_experts // ep_size`,
  `local_num_group = top_k // ep_size` and its only combine is
  `unsorted_hidden_states.reshape(bsz, top_k // ep_size, -1).sum(1)`
  (`:643-644`). Every `top_k` slot of every token is computed **on every rank** —
  the helper that would have scattered them, `process_topk_ids` (`:55`), is called
  only from `fused_experts_with_all2all_buffer` (`:468`). With `ep_size = 4`,
  `top_k = 8` the reshape folds 8 slots into groups of `8 // 4 = 2`, so the returned
  tensor is `[2048, 8192]` — exactly the element count upstream rejected. With
  `ep_size = 1` the same line is `reshape(2048, 8, 2048).sum(1)` → `[2048, 2048]`,
  i.e. **correct by construction**, which is why dropping the flag is the fix and
  not a workaround.

## Step 3 — the drift: the EP-capable machinery is in the same file and this path never reaches it

The pin does ship a real EP state machine — `AscendUnquantizedFusedMoEMethod`
(`fused_moe.py:977`), `AscendFusedMoE` (`:1123`), `get_fused_moe_state(ep_size,
with_prefill, is_deepseek_v3_r1)` (`utils.py:473-489`: `All2All` for `ep_size < 16`
or prefill, `MC2` above 16, `AllGatherEP` env-gated to DeepSeek V3/R1),
`fused_experts_with_all2all` / `_with_mc2`. The Qwen3 path does not use any of it:
the log's own `registry.py` line rewrites `Qwen3MoeForCausalLM` to
`CustomQwen3MoeForCausalLM`, a 35-line subclass that keeps upstream's
`Qwen3MoeSparseMoeBlock` (`vllm_ascend/models/qwen3_moe.py:19-22`, registered at
`models/__init__.py:53-54`), whose `self.experts` is upstream `FusedMoE` with the
**method monkey-patched** to `forward_oot` (`common_fused_moe.py:111-112`). So on
310P the EP flag reaches a function that only *subtracts* by `ep_size` and never
dispatches: no config key, env var or later flag turns EP on for this path at this
pin. Say that, rather than proposing an `additional_config`.

## Step 4 — what the pin's own artifacts say about the flag

- The documented Qwen3-30B-A3B recipe at this pin is
  `docs/source/tutorials/multi_npu_qwen3_moe.md:44`
  `vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 4 --enable_expert_parallel`,
  and the line above it scopes the card budget to **"an Atlas A2 with 64GB of NPU
  card memory … for 32GB of memory, tensor-parallel-size should be at least 4"**.
  The reporter's command is that A2 recipe plus the 300I environment
  (`single_node_300i.md:65-66,135` — `--enforce-eager`, `--dtype float16`; `:194`
  "For 300I series, only eager mode is supported"). So the flag came from a
  non-310P page, and "remove it" contradicts no doc.
- The sibling recipe `docs/source/tutorials/multi_npu_moge.md` (Pangu Pro MoE, TP4)
  passes **no** `--enable-expert-parallel` either.
- Coverage: at this pin there is no `tests/e2e/310p/` directory at all; the only EP
  MoE case is `tests/e2e/multicard/test_fused_moe_allgather_ep.py`, i.e. the
  A2/A3 allgather variant that `get_fused_moe_state` env-gates to DeepSeek V3/R1.
  Quote that as the coverage gap, not as the cause.
- Do **not** conclude "there is no 310P doc at this pin" from
  `rg --files docs/source | rg -i 310` → empty: at v0.9.x the 300I tutorial is named
  `docs/source/tutorials/single_node_300i.md`. Recorded #1813's run made exactly
  that mistake and wrote "no 310P doc at all" into its answer.

## Ladder

1. Re-run the identical command **without** `--enable-expert-parallel` (keep
   `--tensor-parallel-size 4 --dtype float16 --enforce-eager`). This is the answer,
   not a probe: it is the doc's own 300I form plus TP sharding, and the arithmetic
   above says the shape becomes correct.
2. If the next launch raises something new, that is a second, unrelated case — send
   the traceback; do not re-add the flag.
3. Expect no EP on 310P at this pin and do not sell an upgrade as a fix: the lane is
   experimental here (no 310P e2e, no 310P-scoped recipe for MoE), and the first
   release that scopes 310P at all still says "only basic dense and vl models"
   (`release_notes.md:9`, v0.14.0rc1, #5776 — a later pin than this case, so
   label it as such).
4. `--max-num-seqs 32` / `--max-model-len 32768` are untouched by this fault; if the
   re-run instead reports `No available memory for the cache blocks`, that is
   `../startup-failures/max-model-len-budget.md`, a different rung.

## Verification state, and what would invalidate it

- Source-verified at v0.9.2rc1 `b5b7e0e` + vllm `a5dd03c` only (single-pin case
  view). Unverified: whether a later release wires dispatch into the 310P path or
  makes `AscendFusedMoE` the class Qwen3 MoE builds — re-grep
  `common_fused_moe.py` and `is_310p` per pin before quoting this page.
- Unverified numerically: that the `ep_size = 1` run computes *correct* MoE output.
  The reshape is structurally right and upstream's
  `maybe_all_reduce_tensor_model_parallel` covers the TP shard, but the acceptance
  check is one request compared against a 910B run of the same checkpoint, not a
  passing engine start.
- Open on the reporter's side: the checkpoint's `config.json`
  (`num_experts`, `num_experts_per_tok`) — the division uses the published
  `top_k = 8` for Qwen3-30B-A3B, so a different `top_k` moves the factor to
  `top_k / (top_k // ep_size)` rather than to `ep_size`.
