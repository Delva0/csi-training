# Same prompt, repeated runs, different output (repeat-run nondeterminism)

## Scope

The report compares the engine **with itself**: 同一条 prompt 推理若干次, and the
outputs differ — "nv 卡 vllm 输出 tokens 长度一致，npu A2 上多次长度差异大" (#7180),
"`temperature=0` 时推理结果不确定" (#9373), "同一个请求多次调用结果不一致". No
crash, no log, usually no request body pasted. The question is *reproducibility*, not
accuracy: nothing is being scored against a reference, so the deliverable is the
pattern of divergence plus the knob ladder — not a score attribution.

Two neighbours take the same phrasing and own their own answers: repeated requests
that **复读 / truncate reasoning / flip a tool call** on v0.18.0 DeepSeek-V4
(`long-output-repetition.md`, `../spec-decode-mtp/index.md`),
and a first-vs-second-call difference in **embedding vectors**
(`../pooling-embedding/index.md`; #5725 is quoted below because its *pattern* is the
cheapest discriminator here as well).

## Rule 1 — the reporter's control refutes platform-independent mechanisms

"On NV the same prompt gives the same length every time" is itself evidence: that
path is deterministic, so the reporter's own script is pinning sampling (greedy or a
fixed seed). A cause that would randomise NV identically — *no `seed`,
`temperature > 0`, checkpoint `generation_config.json` defaults* — predicts the
control fails too, so it cannot head the ranking. Rank the platform-dependent
branch first and ask for the body as a one-line confirmation, not as the answer.

This is `SKILL.md` rule 15 applied to a **control platform** instead of a sibling
entry point. Recorded #7180 ranked "采样随机性没有被固定" as cause 1; the reporter's
next public comment pasted the body — `temperature: 0`, `top_p: 1`, `top_k: 1`,
`repetition_penalty: 1.1`, `max_tokens: 65536`, prompt ≈8782 tokens — which refutes
it outright. The refutation was also available from source in the same session:
`all_greedy` takes `greedy_sample` / argmax before any RNG
(`vllm/v1/sample/sampler.py:144-172` at vllm v0.16.0), so the Ascend sampler's
Gumbel `q.exponential_()` path (`vllm_ascend/sample/sampler.py:12-31`) is not even
entered. `top_p` / `top_k` / `repetition_penalty` are deterministic functions of the
logits.

## Procedure

1. **Get the body, then read three fields.** `temperature`, `top_k`/`top_p`, `seed`
   decide whether sampling is a candidate at all:
   - `temperature` ≥ 1e-5 **and** `top_k > 1` **and** no `seed` → sampling is live;
     say so, and give the pinned body (or `seed: 42`) as the one-line fix.
   - `temperature < 1e-5`, or `top_k: 1` → the sampler is out; go to step 2. Do not
     spend a turn on `seed`.
2. **Read the *pattern*, not the size of the length gap.** Ask for 3–5 runs of the
   identical body with `"logprobs": true, "top_logprobs": 5` and
   `"return_token_ids": true` (both exist on chat completions at vllm v0.16.0:
   `vllm/entrypoints/openai/chat_completion/protocol.py:158-159,317`), and have them
   send the **first** differing token index together with `usage.completion_tokens`
   and `finish_reason`:

   | Pattern | Owner | First ablation |
   |---|---|---|
   | run 1 differs, runs 2..N identical | cold vs cache-warm prefill path | `--no-enable-prefix-caching` |
   | every run differs, first-token `logprobs` already differ | numerics (kernel / layout / comm) | step 3 ladder |
   | every run differs, `logprobs` identical but chosen ids differ | sampler path — re-read the body, this should be impossible at greedy | — |
   | every length pinned at `max_tokens` / `max_model_len`, `finish_reason: "length"` | budget, not divergence | budget question → `../startup-failures/max-model-len-budget.md` |

   #5725 is the worked pattern-1 case on this stack: the first call matched GPU vLLM
   at cosine 0.9997, calls 2+ were stable but 0.58 away — cache-warm output was
   *stably wrong*, not noisy. Prefix caching is default-on at the pin
   (`vllm/config/cache.py:76`, `enable_prefix_caching: bool = True`), and a prompt of
   this size is where it bites: `max_num_batched_tokens` defaults to **2048**
   (`vllm/config/scheduler.py:42`), so an ~8782-token prefill is ~5 chunks, and a
   cache hit recomputes a different remainder than the cold run that populated it.
3. **Length difference is an amplifier, not a magnitude.** On a long-CoT checkpoint
   (QwQ / R1-distill / a GRPO fine-tune) with `max_tokens: 65536`, one flipped greedy
   token forks the whole trajectory, so "长度差异很大" is the *expected* appearance
   of a sub-ULP logit difference. Never read it as "the quantization is broken"; the
   sharp probe is `max_tokens: 1` (and `2`) repeated 20×: identical first tokens move
   the divergence into decode steps, while a differing first token puts it in prefill.
4. **Take the project's documented determinism recipe before the beta program.**
   `docs/source/faqs.md:167-203` at the pin is FAQ #16, *"How to generate
   deterministic results when using vllm-ascend?"* — greedy plus four env vars:

   ```bash
   export LCCL_DETERMINISTIC=1
   export HCCL_DETERMINISTIC=true
   export ATB_MATMUL_SHUFFLE_K_ENABLE=0
   export ATB_LLM_LCOC_ENABLE=0
   ```

   Quote it as the project's own statement; it is SoC-agnostic and versioned with the
   tree — but check each variable is **live at that ref** before promising it: the two
   `ATB_*` vars bind only to the ATB calls the release still makes
   (`torch_npu.atb.npu_paged_cache_load`, the one-shot `_warm_up_atb` matmul at
   v0.20.2rc1), so a release whose matmuls are all aclnn/AscendC gets nothing from them
   (`cross-platform-logprob-parity.md` Step 1 carries the sites and the grep key). Only then offer batch invariance (`VLLM_BATCH_INVARIANT=1`), which is the
   broader/later answer and carries gates the FAQ does not:
   - **Status/hardware:** beta, tracking issue #5487; *"currently requires Ascend
     NPUs for 910B"*; requires a custom operator library
     (`docs/source/user_guide/feature_guide/batch_invariance.md:4-5,21-28,32-43`).
   - **What it forces** (`vllm_ascend/batch_invariant.py:77-86`):
     `VLLM_ASCEND_ENABLE_NZ=0`, `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0`,
     `HCCL_DETERMINISTIC=strict`, `LCCL_DETERMINISTIC=1`. With the AscendC op library
     importable (`:44-48`) it also swaps `npu_fused_infer_attention_score` and
     `npu_add_rms_norm` (`:104-116`); without it the Triton fallback covers only
     `mm`/`matmul`/`linear`/`addmm`/`bmm`/`softmax` and **not attention**
     (`:118-125`).
   - **Gate:** the env var is read once into a module constant upstream
     (`vllm/model_executor/layers/batch_invariant.py:989-1000`), so it must be in the
     process environment at start (`VLLM_BATCH_INVARIANT=1 vllm serve …`); setting it
     after `import vllm` does nothing.
   - **Confirm which branch is live, and do not use the log for it.** The INFO line
     `Enabling batch-invariant mode for vLLM on Ascend NPU.` is gated on
     `HAS_TRITON or HAS_ASCENDC_BATCH_INVARIANT` (`vllm_ascend/batch_invariant.py:139-145`
     at v0.18.0), so it also prints when the AscendC library is missing and **attention
     is not replaced**; the warning form (`… Triton or AscendC … not available`, `:146-150`)
     appears only when both are absent. Re-grep the condition per ref. The decisive
     probe is `python -c "import batch_invariant_ops"` inside the reporter's image; entry
     is the worker's `_init_worker_distributed_environment`
     (`vllm_ascend/worker/worker.py:575-580`). What each branch does and does not
     replace, and the sibling report whose lever is batch **composition** rather than
     repetition, are `batch-invariant-coverage.md`.
   - **Coverage is narrow:** the tested list is `Qwen3-1.7B/8B` and
     `Qwen3-30B-A3B`, no quantized checkpoint and no QwQ/Qwen2 architecture, and the
     registered op set is dense `mm`/`matmul`-shaped — W8A8 quantized linear is
     outside it. For an INT8 checkpoint state batch invariance as a *mitigation with
     unknown coverage*, not as the fix.
   - Two flags it silently changes, worth naming so a later report is not read as a
     regression: `additional_config.enable_async_exponential` is forced off
     (`vllm_ascend/ascend_config.py:127-133`) and upstream disables cascade attention
     (`vllm/config/vllm.py:1009`).
5. **Then the ablation ladder, one axis per launch:** identical body both platforms →
   FAQ #16 env block → `--no-enable-prefix-caching` (row 1 of the table) →
   `VLLM_BATCH_INVARIANT=1` → `VLLM_ASCEND_ENABLE_NZ=0` /
   `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0` alone → `--enforce-eager` (graph-bucket
   padding) → TP1 vs TP>1 → a non-quantized sibling checkpoint. TP1 fp16 with plain
   matmuls still differing run to run (#9373, Qwen3-4B, 910B, TP1) points at the
   kernel/ATB path rather than collectives; TP>1 adds HCCL reduction order.

## Version gate

Reported #7180 runs the **0.13.0** image; the snapshot pins vllm-ascend
`v0.16.0rc1` (`a78a00e`) + vllm `v0.16.0` (`89a77b1`), where `batch_invariant.py`,
its doc page, `tests/e2e/singlecard/test_batch_invariant.py` and FAQ #16 all exist.
Nothing establishes that the reporter's image carries them: have them check inside
the image — `python -c "import vllm_ascend;print(vllm_ascend.__file__)"`, then `ls`
for `batch_invariant.py` next to it, and
`VLLM_BATCH_INVARIANT=1 python -c "from vllm.model_executor.layers.batch_invariant
import vllm_is_batch_invariant as f; print(f())"` (a missing module raises) — and say
the version gate out loud. That check belongs in the first answer because it is the
one thing that can make the whole ladder unavailable.

## What to send back

Both platforms' request bodies; 3–5 runs with `usage.completion_tokens`,
`finish_reason` and the first differing token index (or the `logprobs` of it); the
serve command and the batch-invariant / NZ / aclgraph lines from the start-up log;
`python -c "import batch_invariant_ops"`; the four FAQ env vars' values; on TP>1 the
card count and `npu-smi info`.

## Precedents

- **#7180** (this page's origin): QwQ-32B-GRPO-INT8, 910B4, image 0.13.0, CANN 8.5.0;
  body pasted *after* the first answer (greedy + `max_tokens: 65536` on an
  ~8782-token prompt, three screenshots of the runs).
- **#9373**: v0.18.0, Qwen3-4B FP16, 910B, TP1, `temperature=0`, same prompt →
  different output; the issue text quotes the same four FAQ env vars and asks for a
  `--deterministic` switch. Evidence class: precedent, not a maintainer verdict.
- **#5725**: Qwen3-Embedding-0.6B, first vs second call differ, then stable — the
  cache-warm pattern.
- **#221** (history, v0.7.1 vs v0.7.3, "输出不相同，怀疑0.7.3有精度问题"; its own body
  exports `LCCL_DETERMINISTIC=1`): the cross-version form of the same question — a
  version-to-version output difference is not a precision regression until each build
  is shown to be *stable* on that body, so the repeat-run check comes first there too.
