# Quantized-checkpoint accuracy drops and the QuaRot ↔ EAGLE3-draft rotation

Scope: "the w4a8/w8a8 checkpoint is less accurate than the full-precision one", "after adding
QuaRot outlier suppression the MTP/EAGLE acceptance collapses", and "quantizing gave me no
speedup (or made it slower)" reports (the `--quantization ascend` section below). The QuaRot
material is verified against the read-only clones at `v0.18.0rc1` / `v0.18.0` / upstream `v0.23.0`;
the no-speedup section against upstream `v0.8.4`–`v0.11.0` and plugin `v0.11.0` (2026-11-01).
Re-anchor with `git grep -n <symbol> <tag> -- <paths>`.

## Establish the scheme before reading numerics

- Method implementations at `v0.18.0rc1` live in `vllm_ascend/quantization/methods/`:
  `w4a16.py`, `w4a8.py`, `w4a4_flatquant.py`, `w4a4_laos_dynamic.py`, `w8a8_static.py`,
  `w8a8_dynamic.py`, `w8a8_pdmix.py`, `w8a8_mxfp8.py`, `w8a16.py`, `kv_c8.py` plus
  `registry.py`/`base.py`, with `vllm_ascend/quantization/modelslim_config.py` (packed-module map
  such as `in_proj_ba` → `["in_proj_b", "in_proj_a"]`) and `method_adapters.py` beside them; 310P
  variants are under `vllm_ascend/_310p/quantization/`. Older tags use
  `quantization/quant_config.py` and newer ones `quantization/configs/modelslim_config.py`, so check
  the tag before quoting a path.
- Scheme availability is a separate question from accuracy: the dense/linear `W4A8_DYNAMIC`
  method exists only up to `v0.26.0rc2` and is removed at `v0.27.1rc1`, so a "W4A8 dense worked
  before" report belongs to
  [quantization-startup-keyerrors.md](quantization-startup-keyerrors.md) (retired-scheme
  `NotImplementedError`), and an int4/FRACTAL_NZ `antiquantGroupSize` startup error belongs there
  too, not to the numerics below.
- The tag's `docs/source/developer_guide/feature_guide/quantization.md` carries the scheme table:
  `W4A16` (INT4 weights per-group, FP16/BF16 per-tensor activations, static, MoE expert layers,
  int32-packed weights), `W4A8_DYNAMIC` (INT4 per-group weights, INT8 per-token dynamic
  activations, optionally two-step 8-bit→4-bit), `W4A4_FLATQUANT_DYNAMIC` (INT4 per-channel with
  FlatQuant smoothing). Name the row the checkpoint actually is; a per-group vs per-channel mix-up
  is the usual source of "which layer lost precision".
- Flag-vs-doc contradiction to state rather than resolve silently: the `v0.17.0rc1` release-note
  section claims "Auto-detect quantization format from model files … `--quantization ascend` is not
  required now" (#7111), while the `v0.18.0` `Qwen3.5-27B.md` tutorial still says the quantized
  model "needs to start with parameter `--quantization ascend`". Report both; passing the flag
  matches the tutorial in play.

## QuaRot + EAGLE3 draft: the draft has to be un-rotated by hand

`vllm_ascend/patch/worker/patch_draft_quarot.py` exists at `v0.18.0rc1` and is the component to read
when a QuaRot target ships with an EAGLE3/MTP draft:

- `get_rotation_path()` reads the rotation entry from the **target** model's quant description,
  `quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]`, and `return None`
  on `KeyError`; `patch_load_weights()` then returns before patching anything **with no warning**.
  So a target whose description lacks that block, or a checkpoint packed without it, silently loads
  the draft un-rotated.
- `compute_rotataion_matrix3(Q)` returns `torch.block_diag(Q, Q, Q)` and is applied as
  `loaded_weight @ Q3` to every draft weight whose name contains `fc.` — i.e. it assumes exactly
  three concatenated auxiliary hidden states. A draft whose `fc` consumes a different number of
  aux states gets a shape/rotation mismatch.
- A draft that ships no embedding gets one rotated with `Q.T` from the target
  (`get_embedding_tensor(target_model_path)`), so draft-with-own-embedding and
  draft-sharing-target-embedding are two different paths — the second is the one the e2e test's
  "30B" model pair covers.

Consequence for triage: "low MTP acceptance after QuaRot requantization" points at the **draft-side
rotation/embedding path** before the target's 4-bit numerics, and the failure mode to look for is a
silently skipped patch or a `fc` shape mismatch, not an engine error. Candidate mechanism — the
patch code paths are verified, the causal link to an acceptance drop is not.

## Prior QuaRot/EAGLE fixes already shipped — do not call the area unfixed

Known-issue/others lists in the `v0.17.0rc1` release-note section (the section present in the
`v0.18.0rc1` file) name: "Fix Eagle3 acceptance rate for QuaRot quantized models" (#6914),
"Eagle3 now supports QuaRot quantization without embedding" (#7038), "Fix GLM5-W8A8 precision issues
caused by rotary quant MTP weights" (#7139) and "Fix a precision issue caused by wrong KV cache
reshape on Qwen3.5" (#7209). Cite whichever is relevant and check ancestry at the reported tag
before proposing a new fix; also check whether the reported checkpoint's rotation map matches what
the plugin expects.

Acceptance tolerance reference: `tests/e2e/multicard/2-cards/spec_decode/test_quarot_eagle.py` at
`v0.18.0rc1` compares float vs `…-W8A8-QuaRot` targets with `K = 4` speculative tokens and
`TOL = 0.06` (`assert all(abs(ref - quarot) <= TOL ...)`) on per-position acceptance. That is the
project's own definition of "acceptance did not move", so quote it instead of inventing a threshold
(it is W8A8, not W4A8 — say so).

## `--quantization ascend` did not make it faster (or the accuracy dropped with it)

A "quantized, no latency gain, and accuracy dropped too" report is an expectation question before it
is a defect. Answer in this order:

- **Small dense models are the wrong audience.** Maintainer reply on `vllm-ascend#10604`
  (Qwen2.5-7B W8A8, vllm/vllm-ascend 0.11.0, no config ever supplied): "Please use the latest
  version or rc version to see if the issue has been resolved" and "For models of this relatively
  small size, quantization may indeed not yield any latency benefits." The thread then closed stale,
  so this is a maintainer statement about expectations, not a root cause of that run.
- **The project warns in the log already.** Upstream `ModelConfig._verify_quantization` prints
  `"%s quantization is not fully optimized yet. The speed can be slower than non-quantized models."`
  for any method outside its CUDA-oriented `optimized_quantization_methods` list, and `ascend` is
  never in that list (`vllm v0.8.4:vllm/config.py:726-730`, warning `:767-771`; `v0.9.1:855-859`
  and `:930-934`; `v0.10.0:vllm/config.py:1086`). So the line is expected on every Ascend-quantized
  start, and it doubles as a **version pin**: it is absent at `v0.11.0` (grep `optimized yet`;
  the string is split across two source lines, so grep `optimized yet`, not
  `not fully optimized`). A report quoting it is a vLLM ≤0.10-era install — or a newer plugin on an
  old vLLM, which the pairing check should catch first.
- **The tag's own guide sets maturity**: `docs/source/user_guide/feature_guide/quantization.md` at
  `v0.11.0:5` — "Since version 0.9.0rc2, the quantization feature is experimentally supported by
  vLLM Ascend ... Currently, only Qwen, DeepSeek series models are well tested."
- **Check the weights actually went quantized** before hunting kernels, at `v0.11.0`:
  `VLLM_ASCEND_ENABLE_NZ` defaults to `1` (`vllm_ascend/envs.py:167-169`);
  `is_enable_nz()` (`vllm_ascend/utils.py:75-90`) returns it for int8 but **False for
  fp16/bf16/fp32 unless an EAGLE/EAGLE3 draft is configured**;
  `AscendW8A8DynamicLinearMethod` (`vllm_ascend/quantization/w8a8_dynamic.py`) transposes then
  NZ-casts `layer.weight` in `process_weights_after_loading` and applies `npu_dynamic_quant` +
  `npu_quant_matmul` with a per-channel `weight_scale` of shape `[output_size, 1]`. Layers the
  description marks `"FLOAT"` stay bf16 by design — including `mlp.down_proj` in the published
  W8A8 checkpoints; see the `"FLOAT"` mirror case in
  [quantization-startup-keyerrors.md](quantization-startup-keyerrors.md) before calling a layer
  un-quantized.
- **A report naming one layer type that shows no INT8 kernel in the profile is a description
  statement before it is a kernel gap.** At `v0.19.1rc1` (`da421afad`) `AscendModelSlimConfig.get_quant_method`
  (`vllm_ascend/quantization/modelslim_config.py:465`) sends every `LinearBase` through
  `is_layer_skipped_ascend(prefix, packed_modules_mapping)` and returns
  `AscendUnquantizedLinearMethod()` when it matches (`:499-503`). That helper (`:530`) treats a fused
  projection (e.g. `qkv_proj`) as skipped only when **every** shard `<shard>.weight` is `"FLOAT"` — a
  partial mixture raises `ValueError: Detected some but not all shards of <prefix> are quantized` —
  and any other layer when some key `startswith(prefix)` and `endswith('.weight')` is `"FLOAT"`. So
  one grep of the checkpoint's `quant_model_description.json` (`MODELSLIM_CONFIG_FILENAME`, `:49`) for
  the layer prefix settles it: a bf16 `qkv_proj`/`o_proj` in the profile is the checkpoint's own
  precision choice, not a missing plugin kernel, and only a re-quantized checkpoint changes it. A
  namespace mismatch does not degrade silently the other way: with no `FLOAT` key the code proceeds to
  `create_scheme_for_layer`, which needs that entry.
- **Keep the KV-cache FA-INT8 path apart from the projection kernels.** In that same description
  `fa_quant_type` sets `enable_fa_quant`, the per-layer list is built from `fa_k.scale` keys
  (`_add_kvcache_quant_metadata`, `:709`), and `enabling_fa_quant` requires a **KV-consumer (decode)
  instance**; at `v0.19.1rc1` the only consumer is `vllm_ascend/attention/mla_v1.py:761`, i.e. MLA
  models. A non-MLA model (Qwen3.5's gated full attention) therefore has no INT8 FA *KV* kernel at
  that tag on either PD role by design, and no P node shows one even for MLA — neither absence is the
  projection kernel missing. For a "Full Attention layer has no quantization kernel, so W8A8 buys
  nothing" report (`vllm-ascend#9443`, Qwen3.5-122B-A3B W8A8 `0.19.1rc1` / vLLM 0.19.1, 1P1D,
  thread closed with only pings): name the two paths, then ask for the model directory's description
  entries for the attention projections and for a per-layer `npu_quant_matmul`/INT8 timer in the
  profile. The claim that *the whole* W8A8 and bf16 deployments run at the same speed is not
  explained by the attention layers alone — leave that as unverified until the description is seen.
- **Where a "model <-> version table" actually lives**: `docs/source/user_guide/support_matrix/supported_models.md`
  is per-model (✅ / 🟡 need-test / ❌ with the tracking issue) and its header points at
  `vllm-ascend#1608` for the latest list; the vLLM <-> vllm-ascend pairing table is
  `docs/source/community/versioning_policy.md` ("Release Compatibility Matrix", see
  [revision-pinning.md](revision-pinning.md)). There is no per-model-per-version
  accuracy/throughput matrix — say so instead of implying one.

## Documented accuracy caveats in the same guide (v0.9.1 - v0.11.0)

`docs/source/user_guide/feature_guide/quantization.md` FAQ #3 at `v0.11.0`: when the **MLA part is
quantized with `W8A8_DYNAMIC` and torchair graph mode is enabled**, the CANN fusion passes corrupt
results — set `"AddRmsNormDynamicQuantFusionPass":"off"` and
`"MultiAddRmsNormDynamicQuantFusionPass":"off"` under `Switch.GraphFusion` in the CANN
`fusion_config.json` (`find /usr/local/Ascend/ -name fusion_config.json`). This is a
DeepSeek-MLA + torchair condition, not a general W8A8 claim: a Qwen dense-7B report does not inherit
it. FAQ #2 in the same file (`Could not locate the configuration_deepseek.py`) is likewise a
conversion-toolchain condition — both belong to the checkpoint's recipe, not to the numerics below.

## Leads that look stronger than they are

- "There is a precision issue with Qwen3-Next due to the changed tp weight split method. Will fix it
  in next release" sits in the **v0.17.0rc1** Known-Issue section, about a different model line and
  a different release. Useful as a hypothesis for a TP-sharded packed-INT4 accuracy drop only after
  checking that the fix is in the reported ref and that the reported model takes that path.
- A grounding-accuracy delta without per-item diffs (e.g. 93.34 % vs 94.86 %) has no maintainer root
  cause to inherit: re-run the offline harness with fixed sampling and seed, then diff per sample.
  Do not attribute a 1.5-point drop to a quant scheme from the aggregate alone.
- Before blaming quantization for acceptance, collect `SpecDecoding metrics: ... Per-position
  acceptance rate` and the per-position pattern; a zeros-across-positions collapse has its own
  triage path in [gdn-mtp-spec-decode.md](gdn-mtp-spec-decode.md).
