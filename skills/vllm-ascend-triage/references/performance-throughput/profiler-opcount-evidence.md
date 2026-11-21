# Profiler-table evidence: op counts, NPU idle %, NPU-vs-GPU operator compare

## Scope

The report's *evidence* is a profiler table rather than a log: "单请求推理时
`QuantBatchMatmulV3_ND` 算子被调用了 98 万次", "采集出来的数据有问题（NPU 空闲时间
达到 95%）", a MindStudio / `msprof` operator-statistics screenshot, or an
`msprof analyze compare` line "NPU 上 matmul 算子调用次数是 GPU 上的 50 倍". The
engine is serving, nothing raised, and no output is wrong.

The deliverable is *whether the number may be used at all*, the normalization that
turns it into a per-step figure, and the artefact that settles it — not a kernel
walk. This is not the rate arithmetic of the parent index (there a metric block is
trusted and the question is why a rate is low); here the table itself is the
suspect. It is also not `../observability-and-logs/online-profiling-endpoints.md`,
which is about vLLM's own `/start_profile` route: an external `msprof`/MindStudio
collection has no route, and its owner is the CANN/profiler stack.

Verified pins for the case below: vllm-ascend **v0.13.0rc1** (`1b5d5ab`,
QwQ-32B-W8A8, TP=2, `FULL_DECODE_ONLY`, `--async-scheduling`), vllm **v0.13.0**
(`72506c9`), CANN runtime export `cann/runtime` (`0b16a5e`). Re-grep per release.

## Rules

1. **Collection health comes before any arithmetic, and one class of failure is
   reporter-confirmed.** In #7351 the reporter later reported that on vllm-ascend
   **v0.13.0-release** the collection *errored* because of a bug in the shipped
   `libruntime_v100.so`; after replacing that one library the collection completed,
   but the data was still wrong — **NPU idle time 95 %** — and the same requests
   ran with "性能几乎无差异" between 0.13.0rc1 and 0.13.0. Two consequences:
   - `libruntime_v100.so` is a **CANN runtime** artefact, not vllm-ascend and not
     torch_npu: `cann/runtime/cmake/package.cmake:271` installs the `runtime_v100`
     target next to `acl_rt`, `runtime`, `runtime_common` into `runtime/lib`
     (export `cann/runtime` @ `0b16a5e`). So a collection that dies in that library
     is a **CANN / profiler-package pairing** question — use the runtime library of
     the CANN release the image was built for, or the CANN+`msprof` pair the
     vendor ships together. It is never a reason to read `vllm_ascend/`.
   - A derived metric that contradicts a working service (95 % idle while a decode
     loop runs) is a statement about the **collection window**, not about the NPU.
     Do not compute on a table whose collection printed an error, and do not turn
     "95 % idle" into an engine-idleness theory: rule 2 tests the window first.
   Ask in the same message as the verdict: did the collection command exit 0 (paste
   any error text), which profiler package + CANN version, and did the collection
   window overlap the requests (profile a fixed `--num-prompts` run and keep
   collection on only for that window).

2. **An op count is cumulative over the window: recover the step count, then
   reconcile three ways.** One command replaces the derivation —
   `scripts/opcount-reconcile.py` (flags in `../first-checks/scripts-catalogue.md`):
   `--count NAME=N` per row (or `--table <paste>`), and it prints the step count
   that *every* row's own quotient agrees on, with each row's per-step factor.
   Then:
   - read the per-step factors against the model structure: for a dense
     transformer they should be the layer count times a small integer
     (one per matmul / norm / attention per layer);
   - reconcile with the table's own clock: `总耗时 = 调用次数 × 平均耗时` summed over
     the rows must sit under the window's elapsed time (`--elapsed-s`, `--avg-us`);
     over 100 % means the rows were not sampled in one window or the counts are
     inflated, under a few percent means those rows cannot be the window's cost;
   - reconcile with the engine's own metrics for the same window (`Avg generation
     throughput` × window ≈ output tokens); the implied step count from the table
     and from the metric block must match.
   If no step count reconciles, the table is two windows — a finding about the
   collection, and the answer is the missing columns, not a mechanism.

3. **Equal end-to-end performance retires the amplification claim.** Reporter-
   confirmed in #7351: same requests, 0.13.0rc1 vs 0.13.0-release, "性能几乎无差异".
   A per-step op-count difference that produces no TTFT/TPOT/throughput difference
   on the same request is a counting difference. The cheapest discriminator is one
   `vllm bench serve` on both sides (same dataset, same `--max-num-seqs`, same
   output length) and a TPOT comparison; rank it **above** every kernel-level
   mechanism. A 50× operator-count ratio never overrules an unchanged TPOT.

4. **NPU-vs-GPU tables differ by name bucket and by window.** NPU rows are aclnn
   names (`QuantBatchMatmulV3_ND`, `MatmulReduceScatterV2`, `AddRmsNormQuant`,
   `AscendQuantV2`), GPU rows are aten/cuBLAS (`aten::mm`, `aten::addmm`, a cutlass
   kernel); a W8A8 dynamic path emits NPU-only quant/fusion ops with no GPU twin, so
   unmatched rows inflate whichever side lacks the name. And the two collections
   rarely cover the same number of steps. Ask for the table's columns — 端别 /
   算子名 / 调用次数 / 总耗时 / 采集时长 / 步数(or `Iterations`) — and compare
   per-step, not totals. Do not attribute semantics to the compare tool from
   memory: in this case's exports the op-level `msprof analyze compare`
   implementation is **absent**, and `Ascend/msserviceprofiler`'s
   `msserviceprofiler compare` is a *service-level* KPI comparison
   (`docs/zh/ms_service_profiler_compare_tool_instruct.md`: 服务总体维度 / 请求维度 /
   批处理维度, CANN 8.1.RC1+ and MindIE 2.0.RC1+), not operator counts.

5. **Per-step parity is the null hypothesis for a dense model.** Anchor: a 64-layer
   dense W8A8 model at TP=2 has 4 matmuls per layer per token (qkv, o, gate_up,
   down — upstream fuses qkv and gate_up: `vllm/model_executor/models/qwen2.py:85`
   `gate_up_proj`, `:151` `qkv_proj` at vllm v0.13.0 / `72506c9`), so ≈256 matmul
   launches per decode step on either stack. #7351's own table ratios give
   3 `QuantBatchMatmulV3` + 1 FP16 `MatMulV2` per layer = 256/step: identical order,
   so a 50× *total* ratio is a window ratio (≈5147 steps vs ≈77), not structure.
   Label: derived from an OCR screenshot — the ratios are exact, absolute values may
   be ±1 digit.

6. **Counts are not time.** Single-stream decode is weight-bandwidth bound: compare
   ms/step against bytes-of-weights-per-card ÷ HBM bandwidth before proposing
   kernels. A correct count with the expected ms/step means nothing is wrong with
   the engine, and the lever is W4A8 / more TP / more concurrency, never "fewer
   kernels".

7. **A launch-command flag with no reader is a silent no-op, and the release note
   says so.** #7351's command exported `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1` and
   `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1`. At v0.13.0rc1 the first is **removed**
   (`docs/source/user_guide/release_notes.md:35`, PR #5272: use
   `VLLM_ASCEND_ENABLE_PREFETCH_MLP`, which the same command also sets; the release
   note's `### Deprecation & Breaking Changes` list is the one grep for these) and
   the second has no `vllm_ascend/` reader at all — its only hit is the env block of
   the project's own nightly feature-stack test
   (`tests/e2e/nightly/features/test_qwen3_32b_int8_a3_feature_stack3.py:73`), i.e.
   the reporter copied a stale export from a recipe, so the recipe is not evidence
   the flag exists —
   while `VLLM_ASCEND_ENABLE_FLASHCOMM` is still honoured as a backward-compatibility
   alias (`vllm_ascend/utils.py:785-786`, `platform.py:330`) for
   `VLLM_ASCEND_ENABLE_FLASHCOMM1` (`envs.py:87`). Decide it with one
   `rg -n '<name>' vllm_ascend/` plus one grep of the release note, and report it as
   a **correction**, never as the cause of a count (the fusion rows the table shows,
   `AddRmsNormQuant` / `SwiGlu`, are gated by `enable_npugraph_ex`,
   `ascend_config.py:113-114` default `False`). The warn-only
   `Unknown vLLM environment variable` line is a different question:
   `../observability-and-logs/env-var-validation-warning.md`.

## Procedure (one command, then answer)

1. `uv run --no-project python <skill_root>/scripts/opcount-reconcile.py --count … --steps … --elapsed-s …`
   (or `--table <paste>`). Quote its output as the check; keep rule 5's label.
2. Ask the three collection questions of rule 1 and the two table columns of rule 4
   in the same message.
3. Ranked answer: (1) window/step normalization (same per-step shape, ≈256
   matmuls/step); (2) name bucketing on the GPU side; (3) the collection itself is
   invalid — reporter-confirmed family, `libruntime_v100.so`/CANN pairing and a
   window that mostly covered idle time; (4) equal end-to-end performance retires
   the amplification claim. Then the artefacts: the compare table's columns, the
   profiler + CANN versions, the output-token count, whether the collection exited 0.
4. **Do not** walk `vllm_ascend/ops/linear_op.py` or
   `quantization/w8a8_dynamic.py` for an op-count question: the count is fixed by
   the layer structure and the number of steps, both of which the table already
   gives. #7351's recorded run spent one block there and its first attempt hit the
   output cap mid-derivation, shipping no message text until the retry; do the
   arithmetic with the script and put the verdict in the first paragraph
   (`../evidence-workspace/delivery-discipline.md`).

## Worked case

**#7351** vllm-ascend v0.13.0rc1 (`1b5d5ab`) + vllm v0.13.0 (`72506c9`), 2 cards
(`ASCEND_RT_VISIBLE_DEVICES=0,1`, card model not stated), QwQ-32B-W8A8,
`--quantization ascend`, TP=2, `FULL_DECODE_ONLY` with capture sizes
`[1,8,24,48,60]`, `--async-scheduling`, `--max-model-len 131072`: the body is one
sentence plus a MindStudio op-statistics screenshot (OCR is the paste). The recorded
run derived the window exactly — `988224/192 = 329408/64 = 653669/127 = 334555/65 =
5147` steps, ≈51.1 µs/call × 988224 ≈ 50.5 s, matching the screenshot's total — and
answered "these are per-request cumulative counts, ≈256 matmuls/step, not a bug".
The follow-up made it a cross-stack claim (50× vs GPU) and the run answered with the
window ratio plus "normalize by steps, send the compare columns". The reporter's
later comment confirms the two load-bearing points: end-to-end performance is the
same on 0.13.0rc1 and 0.13.0-release, and the collection itself was broken by a
`libruntime_v100.so` bug and then produced 95 %-idle data. What the recorded run
missed is rule 1: it never asked whether the collection had succeeded, and it
attributed the 50× only to the window.

Route: parent index `index.md`; the arithmetic helper
`../first-checks/scripts-catalogue.md`; when the same shape arrives as a
"profile says the NPU is idle" claim, rule 1 and rule 6 are the answer.
