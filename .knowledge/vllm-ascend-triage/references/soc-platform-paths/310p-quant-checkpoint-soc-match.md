# 310P + quantized model: the checkpoint artifact is the first variable

## Signature

`[Bug][310P] <model>-w8a8  Bias dtype should be INT32, actual dtype is DT_FLOAT16`
(#14792, vllm-ascend/vllm 0.23.0, `npu-smi` `310P3`, CANN 9.1.0, torch_npu
2.10.0.post4, `--dtype float16` implied): the body is a title and a `collect_env`
dump, no traceback and no serve command. The string is **operator tiling output**,
not a Python raise: `rg -n "Bias dtype" /workspace/repos/<issue>` finds nothing in
vllm-ascend, vllm, cann or the `_310p/` subpackage — the emitting site lives in the
CANN op implementation, which the snapshot does not export. **`-n`, never `-rn`:** `-r`
is `--replace`, so `rg -rn "Bias dtype" .` rewrites every match with `n` and its empty
output looks like the verified absence this paragraph describes. The recorded #14792
replay copied `-rn` from this page, got nothing, and spent a turn re-running it.

When a 310P report carries an op-level dtype/layout complaint on a quantized
model, do not open the engine source first. The artifact decides.

## Rule 1 — 310P has its own converted checkpoints; a generic `-w8a8` is an A2/A3 conversion

Verified in the pinned tree and in the thread:

- The 310P-validated rows of the model docs name **310-suffixed** artifacts:
  `Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310-vllm`
  (`docs/source/tutorials/models/Qwen3-Dense.md:46-48`), with the prose note
  "Atlas 300I DUO uses the W8A8SC-310 quantized weights from the Eco-Tech official
  ModelScope repository" (`:310`) and a serve line that loads a per-TP subdirectory
  (`:318`).
- **The resolution is an artifact swap, stated by a maintainer** (xxwei2026 on
  #14792, 2026-08-24T07:26Z: "310p的权重要换成
  https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-W8A8-310P"). Lead the answer
  with that swap id, and keep the scheme table in Rule 2 as the *explanation* of why a
  generic A2/A3 conversion cannot work — not as a rival cause to test first.
- **That swap changes model version, and the reporter said so.** Their follow-up in
  the same thread (2026-08-24T11:19Z) is the question "没3.8的嘛" — "isn't there a 3.8
  one?". A question is not evidence of absence: what it establishes is that the only
  310P 27B conversion named in this record is the **3.6** model. What the pin settles
  is narrower: `rg -n 'Qwen3\.8'` over the whole v0.23.0 tree (`5cb98caa`) is empty —
  no 3.8 tutorial, no model class — and the only 310-suffixed artifact rows are
  8B/14B/32B (`Qwen3-Dense.md:46-48`). A publish is not a doc row either: the
  maintainer's `...-W8A8-310P` naming appears nowhere in the pinned tree, so do not
  conclude from an empty repo grep that the repo does not exist. So say three things:
  the failing directory is the generic conversion; the 310P 27B publish is a
  *different model version*, so swapping is only an option if the reporter may change
  model; and a 310P conversion of **their** version is unconfirmed — check the
  Eco-Tech ModelScope listing, otherwise it is a 310P-targeted W8A8SC export of their
  own checkpoint, or a request to the maintainers. Never present the 3.6 repo as a
  drop-in for a 3.8 checkpoint.
- **Doc trap at v0.23.0 — now maintainer-contradicted.** The 27B tutorial says Atlas 300I
  DUO support starts at `v0.23.0rc1` (`Qwen3.5-27B-Qwen3.6-27B.md:9`), lists
  `Qwen3.6-27B-w8a8` as "… or Atlas 300I DUO" with the generic
  `Eco-Tech/Qwen3.6-27B-w8a8` download (`:29`), and the Atlas 300I DUO tab exports
  `MODEL_PATH=Eco-Tech/Qwen3.6-27B-w8a8` (`:348` is the tab, `:394` the export,
  `:396` `vllm serve $MODEL_PATH`) — while the maintainer's answer is a separate
  `-310P` publish. Whoever followed that tab lands on this failure: say it is a
  doc/artifact gap, not user error, give the `-310P` id, and note the tab's other 310P
  obligations (`--dtype float16` `:423`, `num_speculative_tokens: 1` `:424`,
  `enable_npugraph_ex: false` `:428`) that a paste without a serve command cannot be
  checked against.

## Rule 2 — why the op complains: the 310P scheme decides where `bias` comes from

Scheme dispatch is by `(quant_type, layer_type)` in the 310P-local registry
(`vllm_ascend/_310p/quantization/methods/registry.py:26-41`). At v0.23.0
(`5cb98caa`) the four W8A8 schemes differ in exactly the tensor the op's tiling is
checking:

| `quant_type` in the checkpoint | 310P scheme | bias handed to the op |
|---|---|---|
| `W8A8` (static) | `w8a8_static.py:45-75` | checkpoint `layer.quant_bias`, `int32` (`:39`, `:60`, `:72`) |
| `W8A8SC` (compressed) | `w8a8sc.py:72-104` | checkpoint `layer.quant_bias`, `int32` (`:91`; zeros on non-0 row-rank) |
| `W8A8S` (sparse) | `w8a8s.py` | same base class, `int32` (`w8a8_base.py:42-43`) |
| `W8A8_DYNAMIC` | `w8a8_dynamic.py:184-211` | **the model's own fp16 `bias`**: `bias=bias` at `:207` into `npu_quant_matmul` |

`npu_quant_matmul` with int8 × int8 takes an **int32** bias (it is added to the
int32 accumulator), so the dynamic scheme's fp16 passthrough is the one path in the
310P code that can produce "expected INT32, got FLOAT16". Two consequences for the
answer: (a) a W8A8_DYNAMIC artifact on 310P reproduces this on the first quantized
linear that has a bias; (b) a model with no bias-carrying quantized linear cannot
produce it at all, so `qkv_bias` / `attention_bias` in the checkpoint's
`config.json` is a cheap falsifier. The asymmetry is source-verified; that this
reporter's artifact is the dynamic variant is a **hypothesis** (the paste carries
no quant description) and the check is one file read.

## Checks, cheapest first

1. `head -c 2000 <ckpt>/quant_model_description.json` (msmodelslim artifacts) or
   `<ckpt>/config.json`'s `quantization_config` — which `quant_type`, and is the
   repo name one of the 310-suffixed publishes (`-310`, `-310P`,
   `w8a8sc-310-vllm`)?
2. `uv run --no-project python -c "import json;safetensors…"`-style dtype listing
   of one `quant_bias`/`deq_scale`/`bias` tensor, if the reporter can run it: the
   310P schemes allocate `quant_bias` **int32** and `deq_scale` **int64**
   (`w8a8_base.py:42-43`), so an A2-converted set is recognisable from dtypes alone.
3. Only then the engine: the three lines above and the registry key for the
   checkpoint's `quant_type`.

## Neighbouring shape, different owner

A 310P W8A8 model that dies **inside** the int8 matmul kernel instead of raising this
text — device DFX `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_*` +
`507015` + `L0C read/write conflict` at `profile_run` — is a CANN kernel/op-binary fault,
not an artifact-dtype one: the artifact check below is still worth one line (a `-310P`
conversion changes the op mix), but the ranking and the two version levers are in
`310p-int8-matmul-startup-fault-and-runner-v2.md`.

## Do not offer

`--enforce-eager`, `--dtype`, `PYTORCH_NPU_ALLOC_CONF`, `VLLM_ASCEND_ENABLE_NZ`
and `--compilation-config` cannot change a checkpoint's tensor dtypes: this is a
model-artifact question and belongs on the ModelScope side (a `-310P` conversion)
or with the maintainers when no such artifact exists. Also do not re-raise
`--quantization ascend` (`../startup-failures/quantization-method-gate.md`): on a
310P msmodelslim set it is required and already assumed once weights load.

## Answer shape and the ask

1. First line: an artifact question, not a device, flag or CANN fault — then the swap
   id for the 310P conversion (and, for the reporter's own version, whether one exists
   at all).
2. Then the ranking: the directory's `quant_type` (§Rule 2) and whether the repo id
   carries `-310` / `-310P` / `w8a8sc-310`; a version mismatch is an answer, not a
   footnote.
3. Ask for exactly two things, because the recorded paste carries neither: the
   `vllm serve` / `LLM(...)` line — *which ModelScope id or path was loaded* is the
   decisive question — and `head -c 2000 <ckpt>/quant_model_description.json`. Do not
   ask for `collect_env` output; the paste already is one.
4. A source-level repair (pass `bias=None` into `npu_quant_matmul` and add the bias
   after it) is untested at every pin: if it is worth offering, label it as unverified
   and keep it below the swap, never as an equal rung.

## Limits

- The op tiling that prints "Bias dtype should be INT32" is not in the export, so
  the exact shape/dtype rule (which of bias vs `quant_bias` vs `deq_scale` is
  examined first) stays a hypothesis; the code asymmetry and the artifact naming
  are verified.
- The behaviour is release-specific: the `_310p/` quantization subpackage did not
  exist at v0.11–v0.13 (`atlas-300i-310p.md`), so this page applies to the
  v0.22+/v0.23 refs where those files exist. Re-grep `is_310p()`/`_310p/` for an
  older image.
