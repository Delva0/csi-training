# "Why isn't linear attention one big op on Ascend?" — GDN fusion lanes and the roadmap answer

## When this page applies

A **design-intent** report on a hybrid linear-attention (GDN) family — Qwen3.5/3.6
dense, Qwen3-Next — asking whether the multi-op linear-attention path is an Ascend
*platform limit* or a current *engineering tradeoff*, and whether the hot path will
be fused (projection + conv + gating + prefill/decode recurrent). Recorded #8940
(vllm-ascend v0.15.0rc1 `3d43ed9` + vllm v0.15.0 `f176443`, Qwen3.5-0.8B, 910B4):
no traceback, no metric block, the "linear_attention 2~3× slower than
full_attention" number is a profiler claim.

Siblings that are different questions: a numeric/degeneration report on the same
family → `../accuracy-benchmarks/qwen3-next-gdn-numeric-path.md`; a low tok/s
metric block on a hybrid GDN MoE → `qwen3-next-decode-throughput.md`. This page is
only for "is the *shape* of the implementation the limit, and what is planned".

## 1. Answer per lane, never for "linear attention"

The path is two mutually exclusive branches with different owners, so a single
verdict ("platform limit" / "engineering choice") is wrong by construction. Lane
map, source-verified at the pin in
`vllm_ascend/patch/worker/patch_qwen3_next.py`:

| Stage | prefill / mixed batch | pure decode |
|---|---|---|
| qkvzba split + reshape | `fused_qkvzba_split_reshape_cat` only when `num_v_heads//num_k_heads in [1,2,4]` **and** `is_cuda_graph` **and** `divide_grid < 65536` (`:58-60`) | same gate; otherwise the plain split/reshape chain, silently |
| conv state | `causal_conv1d_fn` (`:180`) | `causal_conv1d_update` (`:189`) |
| gating | `fused_gdn_gating_patch` (`:203`, taken when `num_prefills > 0 or spec_sequence_masks is not None`) | folded into the decode kernel below |
| recurrent core | `chunk_gated_delta_rule` (`:248`) | `fused_sigmoid_gating_delta_rule_update` (`:279`) — gating + recurrent + `ssm_state` writeback in **one** launch |
| decode inside a mixed batch | `fused_recurrent_gated_delta_rule` (`:263`) | — |

Two consequences to put in the answer: (a) the launcher's phase count matches
upstream vLLM v0.15.0 (`vllm/model_executor/models/qwen3_next.py`, whose pure-decode
lane even runs gating and recurrent as *two* kernels), so "Ascend splits it more,
therefore slower" is refutable at the pin; (b) what actually differs is per-kernel
efficiency — Triton-Ascend tiling is bounded by the UB budget and vector-core count
(`ops/triton/fused_gdn_gating.py` `UNIFIED_BUFFER_SIZE`/`get_vectorcore_num()`,
`ops/triton/fla/chunk_delta_h.py` `assert K <= 256`, the grid cap comment at
`patch_qwen3_next.py:56`). Re-grep those constants per pin; they are the *mechanism*
of the gap, not a stable fact.

## 2. Intent is not in the tree — label it and give the roadmap per lane

The current state is source-verifiable; the plan is a maintainer statement. On
#8940 the maintainers' later public reply was: *"We may replace chunk gdn op with
mega gdn fusion op once we test its performance and stability completely. Mega GDN
op PR is #8872."* The reporter's own follow-up then asked the question that
matters: *"这个算子好像仅对prefill有效果，PR里面提到decode 阶段会 fallback to
Triton。decode有相关计划吗?"* — the fusion op targets the **chunk (prefill)**
pipeline, and **decode keeps falling back to the Triton kernel**.

So an answer that says "decode is already fused at this release, done" is
technically right about v0.15.0rc1's single decode kernel and still misses what the
reporter asks next. State both lanes, mark the prefill fusion as an announced item
whose PR postdates the pin (nothing in the snapshot verifies it), and name the
decode lane's plan as the open question with the PR/issue to watch. Do not present
your own proposal (an AscendC op, a wider capture-safe ablation) as the project's
roadmap.

## 3. Deliverable: three pieces in one message

1. **Verdict** — engineering tradeoff at this pin, with the one place where the
   platform genuinely binds: fusing across a GEMM (Cube cores) and elementwise
   vector work needs an AscendC/`aclnn` custom op; the Triton-Ascend UB model is
   not the vehicle for that, which is exactly why the announced direction is a
   custom "mega" op rather than a better tile.
2. **The lane table** so the reporter can place their own profiler row, plus the
   prefill-fusion / decode-Triton roadmap split above.
3. **Two lane-specific checks, cheapest first** — which branch the profile lands
   in (`num_prefills` split at `:171`/`:202`), and whether a graph was captured at
   all (`PIECEWISE compilation enabled on NPU` / `FULL_DECODE_ONLY compilation
   enabled on NPU`; without `is_cuda_graph` the qkvzba fusion never runs). Then
   normalize: divide both layers' time by the profiler's own call counts before
   accepting "2~3× slower per layer".

## 4. Traps

- **Version the reporter's links first.** In #8940 the Ascend links pointed at
  `main` (`vllm_ascend/ops/gdn.py`, `patch/worker/patch_qwen3_5.py`) and the CUDA
  links at `releases/v0.19.1`, while the env was 0.15.0rc1: neither path exists at
  the pin (`rg --files -g '*qwen3*'` shows `patch_qwen3_next.py` +
  `ops/triton/fla/*`, and `vllm/model_executor/models/` has no `qwen3_5.py`). Say
  which tree you verified and give them the one-line check on theirs.
- **`--enforce-eager` is not a neutral ablation here** — it removes the qkvzba
  fusion as well, so it moves the baseline (`SKILL.md` rule 15).
- **Layer types are not comparable per forward.** A GDN layer runs two projection
  GEMMs + conv + the chunk pipeline; a short-context full-attention layer is qkv +
  one attention + o_proj. Only a per-op normalization separates structure from
  defect.
- **"On CUDA it's one kernel" must be checked at the reporter's ref**, not
  repeated: at v0.15.0 CUDA's decode lane is two kernels.
