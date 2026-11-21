# 310P W8A8 start-up dies inside an int8 matmul kernel: image op-binary vs runner V1/V2

## Signature

`[Bug][310P]` report, `-310p` image, a dense W8A8 checkpoint loaded with
`--quantization ascend` (ModelSlim `W8A8_DYNAMIC`), TP=1, fp16. Weights load, then
`EngineCore.determine_available_memory -> profile_run` dies and the **device DFX block**, not
the Python traceback, names the fault:

```
QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21
kernel hash: 5247287448945562503
runtime error: 507015
error: L0C read/write conflict
```

The Python frame moves between attempts (`aclnnDynamicQuantV2`, `aclnnRmsNorm`): that is
async attribution, and the kernel name + hash is the part that is stable. Report the DFX
name, never the frame.

The code alone does not route the case — four neighbouring shapes carry different owners: "MPU address access is
invalid" in the chunked-FLA/GDN `chunk_fwd_o` path is
`../custom-ops-runtime-faults/index.md`; a `161002` / `Tensor scale not implemented for
DT_FLOAT16` on the *same* 310P quant line is the dtype gate
(`../custom-ops-runtime-faults/quant-scale-dtype-gate.md`); a `Bias dtype should be INT32`
complaint is the artifact/scheme lane (`310p-quant-checkpoint-soc-match.md`); and an
`EZ9999 … Cannot find bin of op …, integral key <dtype list>` before any launch (no
`errCode` device block, `Kernel Run failed. opType: N`) is the dtype-key lane, not this
CANN kernel fault (`../custom-ops-runtime-faults/cannot-find-bin-integral-key.md`).

## Why no flag in the tree can move it (v0.23.0rc1 `f4a08bd`)

- Launch site for the 310P dynamic scheme:
  `vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py:202`
  `torch_npu.npu_quant_matmul(quantized_x, layer.weight.data, layer.weight_scale, …)`,
  with the weight made NZ in `process_weights_after_loading` at `:216`
  (`maybe_trans_nz(layer.weight.data).transpose(0, 1)`) — which is exactly the `NZ_NZ` in
  the kernel name — and `bias=bias` at `:207`, so the model's fp16 bias reaches the op.
- `rg -n '507015|L0C|QuantBatchMatmul' vllm_ascend/ docs/` at that pin: **no hits**. No
  env var, `--enforce-eager`, `--max-num-batched-tokens` or `--language-model-only`
  permutation is a lever; state that instead of proposing a ladder of them.
- What remains is two variables outside the Python: the image's CANN/op-binary package and
  the 310P model runner generation. Both are version questions, decided by reading files.

## Check 1 — the image's CANN base, in one line

`Dockerfile.310p:18` is `FROM quay.io/ascend/cann:9.1.0-beta.1-310p-ubuntu22.04-py3.12` at
v0.23.0rc1, and `-310p` tags come from that file
(`.github/workflows/schedule_image_build_and_push.yaml:86,127`). So "does the tag I need
exist" is `grep '^FROM' Dockerfile.310p` **on the branch that produced the tag**, plus on
the reporter's side `docker run --rm <tag> sh -c 'head -3
/usr/local/Ascend/ascend-toolkit/latest/version.cfg; pip show vllm | grep Version'`. If no
tag carries the newer base, the honest answer is "build it from a later branch, or rebase
only that FROM line" — the registry is not visible from the evidence workspace.

## Check 2 — the 310P runner is a version-gated opt-in; the banner is the proof

- `VLLM_USE_V2_MODEL_RUNNER=1` reaches the engine through
  `vllm_ascend/patch/platform/patch_use_v2_model_runner.py` (env-only property; upstream's
  architecture whitelists are bypassed), but `vllm_ascend/worker/worker.py:160-163` at
  v0.23.0rc1 forces V1 with
  `logger.warning("VLLM_USE_V2_MODEL_RUNNER is not supported on vllm 0.23.0; falling back to v1 model runner.")`
  — so on a report pinned to that upstream version the flag is a **no-op**, and a retest
  that sets it proves nothing either way.
- Even past that gate, no 310P V2 runner exists at this pin:
  `_310p/worker_310p.py:39-40` builds `NPUModelRunner310`
  (`_310p/model_runner_310p.py:66`) and logs `Using NPUWorker310 and NPUModelRunner310.`;
  there is no `NPUModelRunner310V2` class to select. Re-grep per pin:
  `rg -n 'NPUModelRunner310' vllm_ascend/`.
- When a newer image does carry it, the activation evidence is the worker banner
  `Using NPUWorker310 and NPUModelRunner310V2` — never the env var and never silence. A
  newer image that prints `…NPUModelRunner310.` is running V1 whatever the env says.

## Recorded outcome (#17082, reporter-confirmed retest)

Stack: `quay.io/ascend/vllm-ascend:nightly-main-310p` (digest `sha256:589fc7c9…`), CANN
9.1.0 (`V100R001C11SPC001B243`), vLLM 0.29.0 (`98dff2a8`), vllm-ascend
`0.19.1rc2.dev2294+gc173a64a4`, `VLLM_USE_V2_MODEL_RUNNER=1`, same checkpoint revision,
same ModelSlim `W8A8_DYNAMIC` / `--quantization ascend`, no FP16 fallback.

- worker reported `Using NPUWorker310 and NPUModelRunner310V2`;
- load, startup profiling, health check and greedy generation all succeeded, no `507015`;
- a `baseline-v1-core` run finished 8 workload cases (128/4096 input targets, 128/512
  output targets, concurrency 1/4; 24 warm-up + 100 measured requests), every request
  succeeded with streamed output reconstructed and SHA-256 verified, prefix caching off;
- container exit 0, NPU idle afterwards.

**Both** CANN (9.1.0-beta.1 → 9.1.0) and the runner (V1 → V2) changed, so this unblocks the
stack without isolating either: the V1 path on a CANN-fixed image, and MRv2 on the old CANN,
are both still untested. Say that — a report of "MRv2 fixed it" is not supported by this run.

## Answer shape

1. First sentence: a CANN int8 `QuantBatchMatmulV3` kernel fault at `profile_run`, not
   checkpoint integrity, memory or a vllm-ascend Python defect; the image's op-binary
   package is the first variable and no in-tree flag reaches it.
2. Ranked, each with one check: (a) CANN/op binary — `Dockerfile.310p` FROM line and the
   tag's `version.cfg`; (b) runner V1→V2 — the `NPUModelRunner310V2` banner and the
   version gate above; (c) artifact scheme — a `-310P`/W8A8SC conversion changes the op mix
   (`310p-quant-checkpoint-soc-match.md`, three checks, no engine walk).
3. Ask for: the whole plog `[AIC_INFO] fault kernel_name=` block with the tiling I-lines,
   a standalone int8 `npu_quant_matmul` probe on an NZ weight at the reported M/N/K with
   fp16 out (if it faults there, CANN owns it), and the quantization scheme of any working
   control model (W8A8_DYNAMIC vs W8A8SC/`-310P`).
4. Say what isolates the cause and what only unblocks: if a newer nightly is the only image
   with a CANN-fixed base, taking that image **with** the V2 runner is a legitimate
   unblocking retest — report it as the stack (image digest, CANN, vLLM, vllm-ascend,
   runner banner) and say plainly that V1-on-new-CANN and V2-on-old-CANN are both still
   untested. Two runs that do isolate it exist only when an image pair exists: same
   `-310p` tag family, one with the old base, one with the new, and a V2-capable image run
   with and without the flag (banner quoted each time). Never present the combined run as
   "MRv2 fixed it".

## Coverage gap worth stating

Mocked W8A8 unit coverage cannot catch a real `QuantBatchMatmulV3` device fault. A
single-card 310P hardware E2E for one dense Qwen3.5 W8A8 checkpoint under the V2 runner
(start-up + profile + one greedy generation) is the test that would protect this path —
name it as a gap, never as a cause.

## Limits

The emitter of the DFX block and of the `_high_performance_*` kernel variant is the CANN op
implementation, which no snapshot exports, so *which* M/N/K tile conflicts stays a
hypothesis until the plog tiling lines arrive. The `VLLM_USE_V2_MODEL_RUNNER` gate and the
absence of `NPUModelRunner310V2` are verified only at v0.23.0rc1; both moved by the
reporter's nightly, so re-grep before quoting them at another ref.
