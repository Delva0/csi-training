# Batch invariance: greedy determinism under dynamic batching

Applies when a report says the same prompt, image, seed and sampling parameters produce
different decode token sequences at different batch sizes or concurrency ("批次漂移",
run-to-run drift, RL reproducibility), or asks whether determinism is guaranteed by default.
It is a feature-gate question before it is a kernel question.

Code and doc facts below were read at `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` =
`v0.23.0rc1` of `/repos/vllm-project/vllm-ascend` (issue #12610, Qwen3-VL-8B + 4 LoRA,
910B2, collected 2026-11). Nothing was executed against an NPU; the coverage gaps at the end
are candidates, not findings.

## The gate: build-time, runtime and graph mode

`docs/source/user_guide/feature_guide/batch_invariance.md` at that ref states, in order:

- Beta feature; progress tracked in issue #5487. Hardware: Atlas A2 and A3 only ("Ascend 950
  and other NPUs in the future").
- The custom operator library is installed **at build time**: "set `VLLM_BATCH_INVARIANT=1`
  before building vllm-ascend". `docs/source/installation.md:242` repeats it.
- Runtime: `VLLM_BATCH_INVARIANT=1 vllm serve <model> --compilation-config '{"cudagraph_mode":
  "PIECEWISE"}'`.
- Documented note: "The batch invariance attention operators currently do not support
  `FULL` / `FULL_DECODE_ONLY` cudagraph mode", and enabling it costs performance.

Build wiring at the same ref: `Dockerfile:66` runs `export VLLM_BATCH_INVARIANT=1 && ...`, and
`csrc/build_batch_invariant_ops.sh` builds the library. So a released image *may* already be
built with the ops — verify the image, do not infer it from the Dockerfile.

Consequence for triage: a reporter who never set `VLLM_BATCH_INVARIANT` (the recorded report
did not) has the feature entirely off, on Ascend and on the GPU control alike. "Greedy decode
must not depend on batch size" is a documented requirement of the feature, not the default
behaviour, so the ticket's premise — "it is occasionally inconsistent" — is answered first by
the gate, not by a kernel diff.

The same env var is also a **backend selector**: with
`--attention-config '{"backend": "FLASH_ATTN"}'` *and* `VLLM_BATCH_INVARIANT=1` (upstream
`selector.py:102` passes `use_batch_invariant=envs.VLLM_BATCH_INVARIANT`) the plugin returns
`vllm_ascend.attention.fa3_v1.AscendFABackend` (`platform.py:806-809`, dense models only); with
either of the two missing it logs "FA3 will not be enabled when not in training-inference
consistency scenario" and serves on the default plugin backend. So `VLLM_BATCH_INVARIANT` decides
which attention code a report ran — a crash in `fa3_v1.py` belongs to
[fa3-attention-backend.md](fa3-attention-backend.md), and removing the env var is what removes
that code path.

## What the feature actually patches

`vllm_ascend/batch_invariant.py`:

- `init_batch_invariance()` is called from exactly one place, `vllm_ascend/worker/worker.py:977`
  (`git grep -n init_batch_invariance <ref> -- vllm_ascend`).
- With the env set it first overrides its own optimisations: `weight_nz_mode = 0`,
  `enable_matmul_allreduce = False`, `HCCL_DETERMINISTIC=strict`, `LCCL_DETERMINISTIC=1`,
  `torch.use_deterministic_algorithms(True, warn_only=True)`.
- `enable_batch_invariant_mode()` registers Aten implementations for the NPU dispatch key only:
  `aten::addmm`, `aten::bmm`, `aten::softmax`, `aten::_softmax` from Triton kernels, plus —
  when the AscendC library is importable (`HAS_ASCENDC_BATCH_INVARIANT`) — `aten::mm`,
  `aten::matmul`, `aten::sum` via `torch.ops.batch_invariant_ops.*` and direct monkeypatches of
  `torch_npu.npu_fused_infer_attention_score`, `torch_npu.npu_add_rms_norm` and `torch.sum`.
  Without the AscendC library it falls back to the Triton registrations, which do **not** cover
  attention, RMSNorm or `sum` — a runtime-only `VLLM_BATCH_INVARIANT=1` on an image built
  without it is therefore a silent partial mode.
- `vllm_ascend/ascend_config.py:257-262` disables `enable_async_exponential` when
  `VLLM_BATCH_INVARIANT` is set ("which make batch_invariant mode not working"), the same
  pattern to look for in other `--additional-config` defaults.

## Coverage gaps (candidates, checked in source, not run)

1. **LoRA is outside the dispatch table.** Adapter deltas go through
   `torch.ops._C_ascend.{bgmv_shrink,bgmv_expand,bgmv_expand_slice,sgmv_shrink,sgmv_expand,
   sgmv_expand_slice}` (`vllm_ascend/lora/lora_ops.py:19-120`), i.e. custom aclnn kernels that
   no `aten::` registration can intercept. Their tiling depends on the token count, which is the
   usual mechanism for batch-dependent results, but batch dependence of these kernels was not
   measured — report it as "not covered by the feature", not as "the cause".
2. **The attention-layer hook looks removed.** `vllm_ascend/models/layer/attention/layer.py:114`
   carries `# NOTE(zxr): vllm_is_batch_invariant is delete during updating to v0.20.1` with the
   import commented out at `:15`. Whether the attention path still selects a batch-invariant
   implementation at that ref is the open question; the doc's own "verified models" list covers
   Qwen3 dense and MoE only, so a VLM + LoRA + dynamic-batching report is outside the claimed
   support surface and the honest answer says so (the reporter's third question — mark
   `Qwen3-VL` as "not verified" instead of absent — is answered by that same list).

## Check the control before attributing anything to Ascend

The recorded report's GPU comparison differed in more than the vendor: vLLM 0.19.0 against
0.23.0, transformers 4.57.6 against 5.5.4, `--max-model-len` 32768/65535,
`--max-num-batched-tokens` 32768/65535, `--max-num-seqs` 10/256, `--gpu-memory-utilization`
0.95/0.8, and `VLLM_BATCH_INVARIANT` unset on both. A cross-vendor "same payload, no drift"
statement from that pair does not isolate a backend defect; ask for the same vLLM/vllm-ascend
pair on both sides, or treat the difference as unexplained.

## Answer shape for this class

State (1) the feature is off unless build-time `VLLM_BATCH_INVARIANT=1` *and* runtime env
*and* `cudagraph_mode: PIECEWISE` are all present, (2) which ops the feature patches at the
reported ref and which of the report's inputs (LoRA, VL attention) are outside them, (3) that
the GPU control is confounded, and (4) the one check that resolves it: the same request at
batch 1 vs N with the flag on, comparing token SHA-256 per decode step. Do not present a
kernel-level cause without that experiment.
