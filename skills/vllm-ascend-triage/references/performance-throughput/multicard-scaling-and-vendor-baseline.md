# "太慢" against MindIE, or slower on more cards: price the scaling axis, not the era

Scope: a perf report whose only evidence is **one number or one screenshot table** and whose
reference point is another **engine** ("比 mindie 慢了一半多") or another **card count**
("多卡跑更慢"). Recorded #770 (2025-05, body `[Usage]: v0.8.4rc2 推理速度太慢` + a benchmark
screenshot; pin vllm-ascend v0.8.5rc1 `ec27af3`, vllm v0.8.5.post1). The recorded run answered
the *era* question (eager decode, no graph mode) and never reached the scaling question the
reporter was actually asking; the reporter's own later comments on that thread give the setting:
8×910B3 Qwen3-30B-A3B ≈ **6 tok/s**, single-card Qwen3-4B **17 tok/s** "比 mindie 慢了一半多",
"多卡跑更慢 应该是多卡通信机制上没有优化", and a sibling report of DeepSeek-R1-W8A8 multi-card at
**1 tok/s with wrong results**.

Read `pre-graph-mode-eager-era.md` first for the v0.8.x/v0.9.x lever list. That page answers
"why is one card slow"; this page answers "why do more cards not help", which is a different
owner and a different check.

## Step 1 — decide whether the paste can be priced at all before doing arithmetic

A screenshot table survives OCR only through its **big unique fields** (`Total token
throughput`, `Output token throughput`, `Time taken for tests`, concurrency, request count);
the small columns are transposed and the Chinese headers are the first thing an offline OCR
loses. Gate on that: if the paste does not carry concurrency **and** request count **and**
input/output token counts, the table cannot be normalized — say so and ask for the summary
text (EvalScope/AISBench `summary` block or the harness config), do not derive a rate.

- Recorded #770's OCR kept `RPS 14.791 / 24.522`, `均延迟 15.752 / 24.555`, `平均TPS 0.402`,
  and dropped the `并发数`/`成功率` columns. The recorded run published a Little's-law
  "in-flight ≈ 233 → 602" from the two latency cells — arithmetic over two OCR candidates,
  which no cell in the paste supports. **A row label that is a concurrency sweep is not a
  concurrency value**: if the derived count is not a round number the client would have set,
  the multiplication never had valid inputs.
- The same gate applies to the reporter's prose: "6 token/s" without ISL/OSL/card count is a
  per-request decode rate for an unknown batch, so it cannot be compared with a vendor number
  until both sides are converted (`cross-run-ratio-normalization.md`, one script call).

## Step 2 — the scaling control is already inside the report

Two card counts on the same box (or the same model at TP=1 vs TP=N) is the ablation matrix of
SKILL.md rule 15: the **single-card row works**, so no mechanism that also slows single card can
be the answer. Take the reporter's own framing as rung 1 ("应该是多卡通信机制上没有优化") and
turn it into one measurement — identical request body and client, `--tensor-parallel-size`
1 / 2 / 4 / 8, same ISL/OSL:

- per-request decode rate = `Avg generation throughput ÷ Running` (V1) or the client's
  `output tok/s ÷ concurrency` (any era);
- step time = `concurrency ÷ output tok/s`;
- per-card efficiency = per-request rate at TP=N ÷ the TP=1 rate.

A per-request rate that **falls** as cards grow while the weight bytes per card shrink is
per-step communication (or replicated work), not compute — and that is the number to put in
the answer, never AICore % from an `npu-smi` screenshot (index rule).

## Step 3 — the pin decides *which* multi-card path the model is on

At v0.8.5rc1 (`ec27af3`) the MoE communication path is not a flag the reporter can choose:

- **A non-DeepSeek MoE rides the plain TP path.** `vllm_ascend/ops/__init__.py:21` imports
  `common_fused_moe`, which installs `forward_oot` onto upstream
  `UnquantizedFusedMoEMethod` (`ops/common_fused_moe.py:27-67`; upstream
  `vllm/model_executor/layers/fused_moe/layer.py:76` at vllm v0.8.5.post1) → `fused_experts`
  (`ops/fused_moe.py:145`), whose tail is
  `if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1): … tensor_model_parallel_all_reduce(…)`
  (`ops/fused_moe.py:668-670`). One all-reduce per MoE layer per decode step.
- **The comm-fused decode path is DeepSeek-only and env-gated off.** `fused_experts_with_mc2`
  (`ops/fused_moe.py:35`) is reached only under `VLLM_ENABLE_MC2=1` and only when
  `not is_prefill` (`ops/fused_moe.py:496`), and the model-level wiring exists only in
  `vllm_ascend/models/deepseek_v2.py:208,228,244`. A Qwen3-MoE serve has no MC2 lever here.
- **There is no expert-parallel flag at this pin**: `rg -n 'enable_expert_parallel' vllm_ascend/`
  returns nothing (the EP/MC2 selector arrives in later releases —
  `../feature-enable-recipes/moe-expert-parallel-with-dp.md` — never propose it for a v0.8.x serve).
  That absence is **this pin's**, not the era's: at v0.9.0rc2 the flag is wired
  (`platform.py:148`), and it is then the only way to satisfy the legacy
  `VLLM_ENABLE_MC2` decode branch, which dereferences a `None` `expert_map` when
  `ep_size == 1` — `../feature-enable-recipes/mc2-env-knob-without-expert-parallel.md`.
- **Every layer is eager**, so the collective has no graph to hide in:
  `platform.py:124 enforce_eager = True`, `:133` logs `Compilation disabled, using eager mode by default`.
  On v0.8.x the only graph lever is torchair on V0 and it is documented for DeepSeek
  (`platform.py:127-160`), so for a Qwen3-MoE report the *era* answer has no action — which is
  exactly why the scaling answer has to be the delivered one.

Answer shape this implies: state that at that pin "多卡通信没有优化" is **source-verified for
every non-DeepSeek MoE** (per-layer TP all-reduce + eager decode, MC2 DeepSeek-only, no EP) and
that the levers are a newer release and/or the DeepSeek family, not a serve flag.

## Step 4 — a vendor-engine baseline is a data point, not "same conditions"

MindIE is a separate stack with its own fused kernels, scheduler and graph capture; a
vllm-ascend image at v0.8.x has eager decode and covers only one Qwen3 model in CI
(`Qwen/Qwen3-0.6B-Base`, single card, `tests/singlecard/test_offline_inference.py:35`) while
`docs/source/user_guide/supported_models.md:8-9` already prints `Qwen3 ✅` / `Qwen3-Moe ✅`.
The pin's own multi-card recipes are `docs/source/tutorials/multi_npu.md:45`
(`Qwen/QwQ-32B … -tp 4 --max-model-len 4096`, dense) and
`docs/source/tutorials/multi_npu_quantization.md:78` (DeepSeek-V2-Lite-w8a8, TP4) — there is
**no validated 8-card MoE recipe and no in-tree throughput number** at that ref. So:

- say which side of the comparison is measured and which is a vendor claim;
- when the reporter ran the vendor engine with a different harness (the thread's own follow-up
  points at a third-party benchmark tool), the clients and ISL/OSL differ too — the only
  comparable axes are per-request decode rate and step time on one client;
- the third-party harness result is quoted as its own row (model, cards, ISL/OSL, client), never
  merged with the vendor number.

## Step 5 — ladder and the artefacts that close it

1. Serve command + `npu-smi info` + model/precision + client config (concurrency, ISL, OSL,
   request count) — without them even the era answer is a guess.
2. `Avg generation throughput ÷ Running`, and the TP sweep of Step 2 on one client.
3. `env | rg VLLM_ENABLE_MC2` and the model family: MC2 only exists for DeepSeek at this pin, so
   on another family report the mechanism and stop.
4. Release ladder, cheapest first: the next rc (v0.8.5rc1 release notes carry the DeepSeek V3
   eager/graph perf PRs #598/#719) and, for MoE comm specifically, a release that has EP/MC2 for
   the reporter's family — label it as "not in this snapshot", never as "will fix".
5. If the model's numerics are also wrong (the DeepSeek-R1-W8A8 1 tok/s case), the perf claim is
   downstream of an accuracy/stdout defect: answer the accuracy track first
   (`../accuracy-benchmarks/index.md`), because a corrupted run also reads as slow.

Traps: don't quote AICore%/power from an idle `npu-smi` as the gap; don't answer a two-card-count
report with a single-card lever list; and don't spend the session deriving the era's graph-mode
gates when the reporter has already told you the direction (more cards = worse).
