# A quantized checkpoint that is no faster than BF16 — and scores lower

## Scope

One report, two complaints, usually one sentence and no evidence:
`[Usage]: Qwen2.5-7B-w8a8量化配置，没有性能收益（甚至有降低），精度同样降低`,
「用了 w8a8 没加速，精度还掉了」, or a `[Bug]` that pastes versions only and leaves
the environment block as the template's `The output of above commands`. The model is
**dense** (a plain Qwen2/Llama-shaped decoder), the artifact came from msmodelslim
(`--quantization ascend`), and there is no dump, no benchmark table and no serve line.
Recorded #10604 (vllm-ascend **v0.11.0** `2f1aed98ccdb` + vllm **v0.11.0**
`b8b302cd`; the same issue also asks 「是否有模型对应版本表」).

The complaint is not new and its direction is recorded: #1015 (QwQ-32B-w8a8,
910B3, TP4, v0.8.4rc2) benchmarked **10.49 tok/s output / TPOT 95.2 ms** against
**12.54 tok/s / 79.7 ms** for fp16 — the quantized dense serve was ~19 % *slower*.
So "even slower" is a measurement this stack has produced before, not by itself a bug
tell, and nothing here is a mechanism until the reporter's own A/B exists.

Not this page: a **hybrid** model where one architectural layer class has no quant
kernel (Qwen3.5/Qwen3-Next `self_attn` rows, #9443) — `quantized-artifact-layer-coverage.md`;
a number printed by the **converter** (per-layer `MSE loss`, a bad artifact, "who
converted it") — `../accuracy-benchmarks/quantized-checkpoint-conversion.md`; a
**load/start** failure on a quantized checkpoint — `../startup-failures/quantization-method-gate.md`;
two *quant formats* scored against each other (`w8a8` vs `w4a8`) —
`../accuracy-benchmarks/quantized-sibling-score-gap.md`; a 310P int8-matmul startup
fault — `../soc-platform-paths/310p-int8-matmul-startup-fault-and-runner-v2.md`.

## Rules

1. **Split the two claims in the first sentence and say which artifact decides each.**
   Accuracy is decided by the *scheme recorded in the artifact's own
   `quant_model_description.json`* and by the calibration the converter used; speed is
   decided by the parameter-byte ceiling, the graph lane and the weight layout. No
   single cause produces both, and "w8a8 doesn't work" is not an answer. When the body
   has no numbers, that split plus the census below **is** the deliverable — do not go
   looking for an engine defect first.

2. **One command censuses the artifact and decides the accuracy half.** At v0.11.0 the
   scheme string in the json is the lookup key into
   `ASCEND_QUANTIZATION_METHOD_MAP` (`vllm_ascend/quantization/utils.py:14-33`):
   `W8A8` → `AscendW8A8LinearMethod` (**static**, per-tensor activation scale),
   `W8A8_DYNAMIC` → `AscendW8A8DynamicLinearMethod` (**per-token**), `FLOAT` → the
   layer is skipped back to an unquantized linear (`quant_config.py:110-112`,
   `is_layer_skipped_ascend` `:136-173`). Run
   `uv run --no-project python <skill_root>/scripts/quant-description-census.py <ckpt-dir>` —
   value histogram, the `FLOAT` module families, and the ceiling of rule 5.
   - **Static** quantizes the activation on every forward with the converter's
     calibration-fixed scale: `npu_quantize(x, layer.aclnn_input_scale_reciprocal,
     layer.aclnn_input_offset)` inside `apply` (`w8a8.py:31-36`, call site
     `:119-123`), with `aclnn_input_scale` replicated `input_size` times at load
     (`:154-162`). One scale per layer ⇒ outlier-sensitive, and its score is a
     property of the calibration set.
   - **Dynamic** computes the activation scale at runtime:
     `quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(x)` then
     `npu_quant_matmul(..., pertoken_scale=dynamic_scale)` (`w8a8_dynamic.py:73-100`)
     — no calibration input at all.
   - Both call the **same** `npu_quant_matmul` (static `w8a8.py:131,135-150`, dynamic
     `w8a8_dynamic.py:92-99`, `npu_dynamic_quant` at `:82`), so static-vs-dynamic does
     **not** explain the speed half. Quote it for accuracy only.

3. **The doc set's own dense w8a8 recipe is static, calibration-pinned and specific.**
   `docs/source/tutorials/multi_npu_quantization.md:1,34,50-55,65` (「Multi-NPU
   (QwQ 32B W8A8)」) converts with
   `quant_qwen.py … --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --anti_method m1`
   and its file listing is `quant_model_weight_w8a8.safetensors` — the **static**
   scheme — while the generic guide's artifact is named
   `quant_model_weight_w8a8_dynamic…` (`user_guide/feature_guide/quantization.md:27,47-58`,
   a DeepSeek/Kimi export). The two pages also pin different msit branches
   (`modelslim-VLLM-8.1.RC1.b020_001` vs `br_release_MindStudio_8.1.RC2_TR5_20260624`).
   So "which recipe did you run" is a real question with a documentable answer, and a
   Qwen-family user following *the* w8a8 tutorial inherits `--anti_method m1` plus a
   general-domain calibration file. Ask for the exact python line before proposing a
   re-export; `--anti_method` / `--calib_file` are the accuracy levers, and neither is
   a serve flag.

4. **Price the ceiling before ranking any engine cause.** Decode is weight-byte bound,
   so with `u` = the *unquantized* share of parameters (from the census: `FLOAT`
   families plus embedding/lm_head/norms if they are not in the json at all) the best
   possible decode gain is `R = 2/(1 + u)` against BF16, with the attention/KV term
   and every per-step host cost ignored. Illustration only — a 7B dense model whose
   json leaves embedding **and** lm_head `FLOAT` (two `vocab × hidden` matrices: for
   the Qwen2.5-7B geometry ≈ 1.09 B of ≈ 7.6 B, `u ≈ 0.14`) gets `R ≈ 1.75×`, and
   the script prints that number from the reporter's own `config.json`; at TP1 with a
   short-context, single-stream benchmark the per-step host/launch term eats a large
   share of that, and the recorded #1015 measurement came out **negative**. If the
   census gives a small `R` (say < 1.15×), the report is arithmetic, not a defect —
   say so and name the only real lever (a re-export with fewer FLOAT layers, or
   staying BF16). A prefill tok/s number helps: prefill is compute-bound, so int8
   should show there first; if prefill also does not move, the ceiling story is wrong.

5. **Three asymmetric levers explain "the quantized side is slower" — check them in
   this order, on both sides.**
   - **Graph lane.** Quantization × ACLGraph arrived in v0.10.2rc1
     (`release_notes.md:34`, #2841) and W8A8 perf improved in the same release
     (`:45`, #2275), so at v0.11.0 a quantized serve should have a graph — but
     `--enforce-eager` on *either* side moves the baseline, and the quantized static
     path adds one `npu_quantize` launch per linear per step (7 projections × 28
     layers on a 7B) that only a graph amortises. Compare like with like and read the
     startup line. The pin's own feature matrix calls Graph Mode 🔵 Experimental
     (`support_matrix/supported_features.md:27`), so treat it as a check, not a promise.
   - **Weight layout.** `VLLM_ASCEND_ENABLE_NZ` defaults to 1 (`envs.py:168-169`) and
     both quant methods cast int8 weights to `ACL_FORMAT_FRACTAL_NZ` only when
     `is_enable_nz()` is true (`w8a8.py:166-168`, `w8a8_dynamic.py:107-109`). That gate
     is **dtype-asymmetric**: `is_enable_nz(dtype)` returns False for
     float16/bfloat16/float32 outside eagle mode (`utils.py:75-90`), so a launcher that
     exports `VLLM_ASCEND_ENABLE_NZ=0` for some other reason costs the int8 side only.
     One `env | grep VLLM_ASCEND_ENABLE_NZ` on the reporter's host.
   - **The comparison itself.** Same prompts, same `--max-num-seqs`, same
     `--max-model-len`, both warm, `temperature=0`, and report TTFT / TPOT /
     prefill tok/s rather than one tok/s number. A one-sided run is not evidence.

6. **There is no validated Qwen2.5-7B-w8a8 row anywhere in the tree — say it, then use
   the published same-family artifact as the control.** At v0.11.0 the only Qwen2.5
   quant reference is a 5-token greedy smoke test on the *published* artifact
   `vllm-ascend/Qwen2.5-0.5B-Instruct-W8A8`
   (`tests/e2e/singlecard/test_quantization.py:28-37`, `enforce_eager=False`,
   `generate_greedy(..., 5)` — no score, no timing), the nightly quant cases are
   Qwen3-235B-MoE and DeepSeek-R1 only, and
   `docs/source/developer_guide/evaluation/accuracy_report/` holds four model pages
   (plus an index), none for a dense Qwen2.5. Census that 0.5B artifact's json with the same script: it
   shows which scheme the project's own converter emits and which layers it leaves
   `FLOAT`, without reading a line of source. Its existence is not a validation claim
   — `supported_features.md:26` says Quantization 🟢 Functional, "W8A8 available",
   with no performance statement.

7. **「是否有模型对应版本表」 is two tables plus a coverage question, and the answer is
   that no model × version table exists.**
   - *Release* correspondence: `docs/source/community/versioning_policy.md`,
     section **Release compatibility matrix** (`:19`; table from `:23`) —
     `vLLM Ascend | vLLM | Python | Stable CANN | PyTorch/torch_npu`; at the pin the
     `v0.11.0rc3 | v0.11.0 | >= 3.9, < 3.12 | 8.3.RC2 | 2.7.1 / 2.7.1.post1` row is
     `:26`. Quote that row rather than `installation.md`: its requirements table
     (`:14-17`) agrees on torch-npu/torch but carries no vLLM pairing, and its pip
     block is `|vllm_ascend_version|` substitutions.
     The reporter's vllm 0.11.0 + vllm-ascend 0.11.0 pairing is the released one, so
     "version mismatch" is off the table — and the pinned tag's own
     `release_notes.md` tops out at **v0.11.0rc0**, so the notes are not the matrix.
   - *Model* support: `docs/source/user_guide/support_matrix/supported_models.md` —
     family-level ✅/❌ only (`Qwen2.5` ✅ at `:20`), `:3` "Get the latest info here:
     …/issues/1608", no version column and no quantization column.
   - *Validation* of a combination: `release_notes.md` highlights/known issues,
     `developer_guide/evaluation/accuracy_report/`, and `tests/e2e/**` coverage — the
     four reads that make "running is not validated" a quotable verdict
     (`../feature-enable-recipes/index.md` part 1).

## Delivery order

1. Verdict: two claims with two owners; nothing about "even slower" is attributable
   without the A/B, and the accuracy half is the artifact's scheme + calibration, not
   the engine. Name the recorded precedent (#1015) so the reporter knows the direction
   is known.
2. The census command and how to read it (`W8A8` vs `W8A8_DYNAMIC` vs `FLOAT`
   families), then the ceiling `R = 2/(1+u)` computed from their own config + census.
3. The A/B protocol (rule 5) and the three checks: graph line, `VLLM_ASCEND_ENABLE_NZ`,
   identical flags.
4. If the artifact is static: the re-export route with the conversion owner
   (dynamic scheme, or a better calibration file / `--anti_method`), plus
   `pip show msmodelslim` and `git describe --tags --always` in the msit clone.
5. The table answer (rule 7), with the Qwen2.5-7B w8a8 coverage gap stated.
6. What is missing, as a list: both serve commands, the benchmark numbers with the
   protocol, the census output, the conversion command line, the accuracy eval method
   with before/after scores, `env | grep VLLM_ASCEND`.

## Traps

- **No serve flag re-quantizes.** Do not answer with a flag for the accuracy half; the
  score moves only with a new export (or not at all).
- **Do not quote a speedup from the docs.** `supported_features.md:26` is a
  *functional* claim; the tutorials carry no performance number for a dense w8a8 serve.
- **Do not call it a version mismatch.** v0.11.0/v0.11.0 is the matrix row (rule 7);
  the same applies to rc/patch labels inside one release line.
- **`FLOAT` in the json is not a bug** — it is the artifact's per-layer decision, and
  it is the input to the ceiling arithmetic. Only an *architectural* class left FLOAT
  becomes a converter defect (`quantized-artifact-layer-coverage.md`, rule 3 there).

## Limits

The artifact is not in the case snapshot, so the census and the ceiling are a **check
for the reporter**, never a finding; the model's own `config.json` is not in the
snapshot either (a 7B's real `vocab_size`/`hidden_size` decide `u`); hardware bandwidth
and the SoC are unknown, so `R` is a ceiling and not a prediction; and the recorded
#1015 numbers are fp16-vs-w8a8 on 910B3 TP4 at v0.8.4rc2 — same direction, not the same
release or model.
