# Per-token logprobs differ from a GPU golden (cross-platform numeric floor)

## Scope

The reporter scores a **golden from another platform** and compares: an H100/H200
run generates the reference tokens once, the Ascend serve re-scores that *fixed*
sequence through `prompt_logprobs` (teacher forcing), and the per-token deltas are
bucketed by the golden token's own probability. The report is deterministic (same
input → same output), shows **no signed bias**, and survives `--enforce-eager`/graph
mode and the torch_npu precision switches — so the question is not "is the engine
broken" but "how close can this platform get, is there a switch, and which kernels
own the residual".

Neighbours this page does not own: NPU-vs-NPU repeat differences
(`repeated-run-nondeterminism.md`), the same body differing only with concurrency
under `VLLM_BATCH_INVARIANT=1` (`batch-invariant-coverage.md`), two *images* of the
same command (`harness-vs-engine-attribution.md`, "Two images"), and a score table
from an eval harness (`harness-vs-engine-attribution.md`). And an entry printed as
`-inf` is **not** this page: a finite logit cannot become `-inf` through rounding, so that
is a mask/extreme-value question whose lane the ids decide —
`topk-logprob-negative-inf.md`.

Recorded: #18755 (vllm-ascend `v0.20.2rc1` `367b8e62` + vllm `v0.20.2` `bc150f5`,
910B2C DP4×TP4, 31B multimodal checkpoint served with images, H100×2 golden,
`mean abs per-token logprob delta` 0.0297-0.0301 against the reference's own
repeat noise 0.0135).

## Step 1 — the switch inventory, read at the reporter's pin

Take the project's own recipe first and *check that each variable is still live at
that ref*; at v0.20.2rc1 it is `docs/source/faqs.md:162-198` (FAQ #15), greedy plus

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
```

- The two collective variables are live: they pin the all-reduce order.
- The two `ATB_*` variables bind only to ATB calls, and at this pin those are
  `torch_npu.atb.npu_paged_cache_load` (`vllm_ascend/device/device_op.py:196`, KV
  paged load) and a one-shot warm-up `torch_npu._npu_matmul_add_fp32`
  (`vllm_ascend/worker/worker.py:573-578`, `_warm_up_atb`). The served matmuls and
  attention are aclnn/AscendC, so these two cannot move served numerics — say
  **that**, scoped to the release.
- **The grep key decides whether the claim is true.** `rg -n 'import atb|torch_atb'
  vllm_ascend/` returns nothing at this pin because ATB is reached as
  `torch_npu.atb.*`; the empty result is not evidence that ATB is gone, and
  "no `vllm_ascend` file imports atb, so the FAQ vars are inert everywhere" is a
  checkable error. Grep `rg -n 'torch_npu\.atb|_npu_matmul_add_fp32|_warm_up_atb'
  vllm_ascend/` and quote the sites.
- torch_npu's accuracy-class options are a different family, and saying so is
  half the answer: `ALLOW_CONV_HF32` reaches convolutions (on a VL model, the
  vision tower), `ACL_OP_SELECT_IMPL_MODE=high_precision` is an op-*selection*
  hint, neither is an accumulation-order control. That is why the reporter's two
  ablations changed nothing.

## Step 2 — retire the logprob-computation lane before ranking kernels

Verified at vllm `v0.20.2` + vllm-ascend `v0.20.2rc1`: both sides compute the
logprob in fp32 over the same full-vocab logits —

| side | site | fact |
|---|---|---|
| upstream | `vllm/v1/sample/sampler.py:291-292` | `compute_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)` |
| upstream GPU path | `vllm/v1/worker/gpu/sample/logprob.py:11-45` | the top-k/LSE Triton kernel casts each loaded block `.to(tl.float32)` before `exp`/sum |
| downstream | `vllm_ascend/sample/sampler.py:144,157` | same `log_softmax(dim=-1, dtype=torch.float32)` in the `processed_logits`/`processed_logprobs` branches |

So "the NPU softmaxes in bf16" is a refutable hypothesis: the *logits* are the
variable, not the softmax arithmetic. Two non-default knobs make the comparison
mean something else, so ask in one line: `logprobs_mode` is `raw_logprobs` by
default (`vllm/config/model.py:222`) and the downstream sampler still carries
`# TODO: support logprobs_mode in vllm-ascend` (`vllm_ascend/sample/sampler.py:68`),
while `enable_reduce_sample` (`vllm_ascend/ascend_config.py:272`, default `False`)
moves top-k/top-p onto per-rank vocab shards.

## Step 3 — what actually differs, ranked, one check each

1. **Topology, not silicon.** A DP4×TP4 NPU run against a smaller-TP golden splits
   every bf16 matmul along K/N and recombines with all-reduce; the fused
   matmul+all-reduce (MC2) family puts that reduce inside the op epilogue, where no
   precision env var reaches. `--tensor-parallel-size 1` (or the golden's own TP)
   on the same fixed sequences is the cheapest single launch, and it separates
   "platform" from "topology" in the reporter's own table.
2. **Prefill chunking.** `prompt_logprobs` are produced by whichever chunk carries
   the token, so `--max-num-batched-tokens` is a numeric knob: score the same
   sequences at two values. `--enforce-eager` does **not** remove this.
3. **Fused custom ops are default-on with no accuracy switch.** The kill switch is
   `vllm_ascend/utils.py:360-363` (`_CUSTOM_OP_ENABLED = False` when
   `envs.VLLM_BATCH_INVARIANT` or the SoC is A5), so `VLLM_BATCH_INVARIANT=1` is
   simultaneously the fusion ablation and the Ascend-internal noise floor. Its
   coverage is narrow and quotable: only
   `torch_npu.npu_fused_infer_attention_score` (v1) is swapped
   (`vllm_ascend/batch_invariant.py:106-109`), and the doc's tested list is Qwen3
   1.7B/8B and Qwen3-30B-A3B on Atlas A2, beta
   (`docs/source/user_guide/feature_guide/batch_invariance.md:4,21,26,105-110`).
4. **Vision preprocessing on a VL checkpoint.** Image resize/normalise is host-side
   and `transformers`-version dependent; a text-only rerun of the same fixed token
   sequences exonerates it in one launch. Ask for the golden's `transformers`
   version — the env dump's own version is evidence only for the NPU side.
5. **Retired by the reporter's own data:** execution mode (their two-mode
   correlation is the control, `SKILL.md` rule 15 applied to an execution-mode
   axis), sampling (`temperature=0`, teacher-forced ids), and the logprob math
   (Step 2).

## Step 4 — make the bucket table say whether it is one mechanism

Run `uv run --no-project python <skill_root>/scripts/logprob-bucket-fit.py` on the pasted buckets (it
takes `--row 'P_LOW,P_HIGH,N,MEAN_ABS'`). The reading, which is the sentence the
reporter needs:

- The statistic is `mean|Δ log p| / (1 - p)`: it is flat across the buckets below
  `p ≈ 0.99` when one logit-space perturbation explains them all. #18755's rows give
  0.180 / 0.185 / 0.213 → one scale, `mean|Δ log p| ≈ 0.20 × (1-p)`, and the table
  is a floor, **not** evidence of a per-bucket defect. The `p > 0.99` row comes out
  *lower* (0.040) and that is expected: a large margin damps the same perturbation.
  Do not report the quiet top bucket as a second mechanism.
- A bucket **above** the flat line is the opposite: it is not roundoff, and it owns
  one artefact request — `top_logprobs: 5` at that scored token on both sides.
- Same test in the `P(yes)` domain: #18755's aggregate was consistent while one
  "direct" case read golden `0.9997` against NPU `0.3486`, far outside its own
  bucket p95 (0.513). One case outside the family is a request, not a cause: dump
  that token's top-5 on both sides rather than ranking it.

## The deliverable

1. **Verdict first, in the reporter's language:** no supported switch closes a
   per-token logprob gap to GPU parity; the switches that exist buy
   *repeat-determinism*, not cross-platform agreement; the residual is a platform
   numeric floor, and a threshold calibrated against the GPU is measuring that
   floor.
2. The switch inventory with its pin-scoped liveness (Step 1), and why their two
   tested variables were the wrong family.
3. The ranked causes with one check each (Step 3), the bucket verdict (Step 4), and
   the kernel families to hand the model vendor — `aclnn` MatMulV3 in bf16 with the
   NZ/fractal weight layout, `npu_fused_infer_attention_score` v1/v2 against the
   GPU attention kernel, `npu_add_rms_norm`, and the fused matmul+all-reduce
   (`csrc/mc2/**`) whose reduce is invisible to every precision env var.
4. Scope: the conclusion holds for the pin; the model's own registration and
   per-model kernels are unverifiable when the checkpoint is withheld (state the
   gap, do not guess), and `Ascend/torch-mlir`, `cann/ops-nn`,
   `cann/ops-transformer` are listed `unavailable` in `SOURCES.json` — kernel
   internal accumulation order is not derivable locally.

## What to send back

Both serve commands (including the golden's TP/DP and `transformers` version),
`--max-num-batched-tokens`, both models' `dtype`, the TP-matched rerun's per-bucket
table, the BI-off/BI-on delta, and the one out-of-family case's `top_logprobs: 5`
on both sides.
