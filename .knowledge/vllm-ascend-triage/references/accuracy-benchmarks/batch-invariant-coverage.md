# `VLLM_BATCH_INVARIANT=1` and the output still depends on batch composition

Verified at vllm-ascend `v0.18.0` (`e18643f8a4d5`) + vllm `v0.18.0`
(`bcf2be961200`). Recorded for #14884: Qwen3-VL-30B-A3B on a `v0.18.0rc1`
image, TP4 + `--enable-expert-parallel`, `--enforce-eager`,
`--no-enable-prefix-caching`, `--max-num-seqs 32`,
`--max-num-batched-tokens 16384`, `temperature: 0`, `HCCL_DETERMINISTIC=true`
plus the rest of the determinism block exported. Title: "enabling
VLLM_BATCH_INVARIANT=1 results in inconsistent output for the same request in
multi-concurrency scenarios".

## Scope — which report this answers

The reporter exported `VLLM_BATCH_INVARIANT=1` and the **same body** still
returns different text depending on **how many requests are in flight**
(concurrency, `max-num-seqs`, what shares the batch, where a chunk boundary
falls). The serve runs normally: no crash, no traceback, no error text.

This is not the run-to-run lane. `repeated-run-nondeterminism.md` owns
"repeat the same request N times and it differs", whose levers are sampling,
prefix-cache warmth and kernel nondeterminism; here the run is stable and the
*lever* is batch composition, so only the batch-invariance program can address
it. It is also not "the flag never applied": the flag applied, and its per-op
envelope is narrower than the reporter's config. That envelope is the answer.

## Rule 1 — split the two installations first, and do not use the log for it

`vllm_ascend/batch_invariant.py` has two mutually exclusive coverage branches,
selected by the AscendC library import at `:42-47` (`import batch_invariant_ops`
→ `HAS_ASCENDC_BATCH_INVARIANT`):

| Branch | Replaced | Left vanilla |
|---|---|---|
| `HAS_ASCENDC_BATCH_INVARIANT` (`:103-115`) | `aten::mm`, `aten::matmul`, `aten::sum`; `torch_npu.npu_fused_infer_attention_score` (attribute swap, `:109-111`); `torch_npu.npu_add_rms_norm` → add + `npu_rms_norm` (`:50-62,113`); `torch.sum` (`:115`); plus the Triton set below (`:96-100`) | `npu_fused_infer_attention_score_v2` and every `.out` call site (rule 2); MoE, EP, VL vision (rule 2) |
| **Triton only** (`:118-124`), library absent | `aten::addmm`, `aten::bmm`, `aten::softmax`, `aten::_softmax` (`:96-100`), `aten::mm`, `aten::matmul`, `aten::linear` (`:119-124`) | **attention** (FIA untouched), `aten::sum`, `npu_add_rms_norm`, `torch.sum` |

The start-up log does **not** discriminate: `init_batch_invariance()` prints
`Enabling batch-invariant mode for vLLM on Ascend NPU.` under
`if HAS_TRITON or HAS_ASCENDC_BATCH_INVARIANT:` (`:139-145`), so on any image
where triton-ascend imports that INFO appears **with attention unpatched**. The
warning form (`Batch-invariant mode requested but Triton or AscendC … not
available`, `:146-150`) fires only when *both* are missing. Re-grep the
condition at the reporter's ref before quoting it. Decisive probe inside the
image: `python -c "import batch_invariant_ops; print(batch_invariant_ops.__file__)"`.

A serve that keeps answering is itself a hint about which branch is live: with
the AscendC library present the FIA swap replaces a function with an
`OpOverloadPacket` while `attention_v1.py:508,603` call
`…npu_fused_infer_attention_score.out(…)`, so that path would raise rather than
degrade quietly. "Answers, only differently" therefore points at the Triton
branch (hypothesis, label it as such; the AttributeError variant is untested).

## Rule 2 — the ops that no branch replaces (ranked causes for a concurrency report)

1. **EP / MoE.** `torch_npu.npu_grouped_matmul`
   (`ops/fused_moe/moe_mlp.py:170,213,228,335,354`), `npu_moe_init_routing`
   (`token_dispatcher.py:354`), `npu_moe_distribute_dispatch_v2` /
   `dispatch` and `…_combine_v2` / `…_combine`
   (`token_dispatcher.py:214-216,307-309`) are `torch_npu` functions, not
   `aten::*`, so no `Library.impl` override reaches them. Tokens per expert
   change with batch composition, which changes the grouped-GEMM shapes and the
   all-to-all tiling. `--enable-expert-parallel` on a MoE model is an
   independent divergence source **even with the AscendC library installed**.
2. **Attention call sites the monkeypatch misses.**
   `torch_npu.npu_fused_infer_attention_score_v2` (`attention_v1.py:803`,
   `mla_v1.py:1396,1402`) is never swapped — only the v1 attribute is. A model
   that selects v2 stays vanilla whatever the branch.
3. **VL vision tower.** The ViT's own attention and patch-embed path is in
   neither list, and the encoder runs in the encoder cache over however many
   images are in flight. A VL checkpoint inherits this on top of 1–2, and the
   divergence then surfaces as a *content* difference (one greedy flip early in
   the prefill amplified over a long `max_tokens`), which reads like an
   accuracy bug.
4. **Chunked-prefill boundary.** Under `--max-num-batched-tokens` N a long
   prompt splits differently depending on its neighbours. Invariant attention
   is designed to neutralise exactly this — but only where attention *is*
   replaced (1–2), so it stays live in the Triton branch.

## Rule 3 — the project's own envelope is narrow; state it as coverage, not as a defect

- `docs/source/user_guide/feature_guide/batch_invariance.md:105-112` (Tested
  Models): `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B` — no VL
  checkpoint, and no expert parallelism anywhere in the list. Same page
  `:4-5,21-27`: beta, tracking issue #5487, Atlas A2 only, "requires a custom
  operator library".
- Both BI e2e files are `DEFAULT_MODEL = "Qwen/Qwen3-0.6B"` at
  `VLLM_TP_SIZE` default 1 (`tests/e2e/singlecard/test_batch_invariant.py:28,155`;
  `test_aclgraph_batch_invariant.py:28,168`), and the MoE row is commented out —
  `# enable_expert_parallel=True` at `test_batch_invariant.py:160`.
- So for a VL + MoE + EP report the verdict is: outside the feature's validated
  envelope at this ref. Cite the project's own list and test, and do not grade
  it as a regression; the escalation target is #5487.

## Ruled out — say so, because they are the next reviewer's first three suggestions

- **The env var / override landed too late for HCCL.** `worker.py:575-580`
  calls `init_batch_invariance()` *before* `init_distributed_environment`, so
  `HCCL_DETERMINISTIC=strict` / `LCCL_DETERMINISTIC=1`
  (`batch_invariant.py:76-85`) are set before HCCL init and the op swaps land
  before model load.
- **Sampler randomness.** With BI on, the Ascend sampler delegates to upstream's
  implementation (`vllm_ascend/sample/sampler.py:104-105`), and a greedy body
  never enters it. `--no-enable-prefix-caching` and `--enforce-eager` in the
  recorded config also remove the cache-warm and graph-bucket rows.
- **vllm-ascend custom kernels.** `enable_custom_op()` returns False under BI
  (`vllm_ascend/utils.py:274-276`: no env override; off for BI, A5, or a failed
  `vllm_ascend_C` import), so no fused custom op is in play. That is design,
  not the cause — and it is worth naming, because a reviewer will otherwise read
  the missing fusions as the defect.

## Procedure

1. `python -c "import batch_invariant_ops"` in the reporter's image → which
   branch (rule 1). One call, decides the whole ranking order.
2. Reproduce in the project's own shape: the same body **alone** (BS=1), then
   mixed into N concurrent requests (BS=N). `max_tokens` small, and
   `"logprobs": true, "top_logprobs": 5, "return_token_ids": true`; ask for the
   **first differing token index**. Step 0 differing ⇒ prefill/vision numerics;
   later ⇒ decode-side (MoE/comm). The tree's own needle test is the template
   (`tests/e2e/singlecard/test_batch_invariant.py:93-209`).
3. One axis per launch, in this order: drop `--enable-expert-parallel` → TP1 →
   text-only body (vision off) → prompt ≪ one chunk. Whichever axis removes
   the divergence names the layer; a row that does not remove it retires that
   cause.
4. Only then the options: stay inside the tested envelope, accept the gap with
   #5487, or wait for the AscendC library. Do not propose patching FIA v2
   call sites as a fix for the reporter.

## What to send back

The `import batch_invariant_ops` output, the serve command, the batch-invariance
lines and image tag, `pip show vllm-ascend torch-npu` plus CANN version, and
3–5 runs per concurrency with the first differing token index. The recorded
reporter had already exported the FAQ #16-style determinism block
(`HCCL_DETERMINISTIC`, `LCCL_DETERMINISTIC`, `PYTORCH_DETERMINISTIC`,
`ASCEND_LAUNCH_BLOCKING`, `VLLM_ASCEND_ENABLE_NZ=0`): those help *run-to-run*
reproducibility and cannot help a batch-dependent difference, which is why the
ladder starts at the library probe rather than at more env vars.
