# Quantization method rejected before any worker starts

## Signature family

```
ValueError: <method> quantization is currently not supported in npu.
```
raised inside a pydantic `ValidationError` while `ModelConfig` is built
(`vllm/engine/arg_utils.py` → `ModelConfig.__post_init__` → `_verify_quantization`),
so the log holds only `APIServer pid=…` lines and the traceback — no `WorkerProc`,
no rank prefix, no `Starting to load model`, no device or HCCL text. Worked case:
#6800, `--quantization awq` on a MiniMax-M2.5-AWQ checkpoint, image
`quay.io/ascend/vllm-ascend:v0.13.0`.

Two different messages live here, both pre-worker:

| Text | Owner | Meaning |
|---|---|---|
| `<method> quantization is currently not supported in <device_name>.` | the **platform** list — for Ascend, vllm-ascend's `NPUPlatform.supported_quantization` | the method exists upstream but this platform does not implement it |
| `Unknown quantization method: <method>. Must be one of [...]` | upstream `me_quant.QUANTIZATION_METHODS` (`vllm/config/model.py:943-946` at v0.13.0) | the string is not a vLLM quantization method at all (typo, a merged/vendor name) |

## Procedure (three calls)

1. Read the platform list in the pinned vllm-ascend tree:
   `sed -n '85,100p' vllm_ascend/platform.py` — at v0.13.0 `6281c120` it is
   `supported_quantization = [ASCEND_QUANTIZATION_METHOD,
   COMPRESSED_TENSORS_METHOD]` (`:91-93`) = `["ascend", "compressed-tensors"]`.
   That list *is* the answer for the reported method.
2. Confirm the raiser is upstream, not a vllm-ascend message: grep the sentence in
   the vllm tree. At v0.13.0 `72506c98`:
   `Platform.verify_quantization` (`vllm/platforms/interface.py:428-436`) raises
   it when `quant not in cls.supported_quantization`; it is called from
   `ModelConfig._verify_quantization` (`vllm/config/model.py:951`, defined `:855`).
3. Show the reporter that dropping the flag does not help — the checkpoint declares
   it: `_verify_quantization` parses `quant_method` from `hf_config`
   (`model.py:869-880` in the same function) and only then
   (`:936-951`) sets `self.quantization` from it, so a `config.json` carrying
   `quant_method: "awq"` reaches the same platform gate with no CLI flag at all.
   One command: `python -c "import json;c=json.load(open('<ckpt>/config.json'));
   print(c.get('quantization_config') or c.get('compression_config'))"`.

The CLI itself accepts the flag: `pre_register_and_update` only *appends* `ascend`
to the `--quantization` choices and leaves e.g. `awq`/`gptq` in place
(`vllm_ascend/platform.py:130-141`; the unit test that pins this is
`tests/ut/test_platform.py:62-74` at v0.13.0). So an accepted launch command and a
rejected run are consistent — do not read the surviving choices as support.

## What to answer

- The direct answer: not supported on the NPU platform in that version; nothing in
  the image, driver, weight files or HCCL is involved. The gate fires before any
  weight is read.
- The two supported routes, quoted from the pinned tree:
  ModelSlim/msmodelslim → `--quantization ascend`; LLM-Compressor → no flag, and
  only `W8A8` / `W8A8_DYNAMIC` are documented as supported
  (`docs/source/user_guide/feature_guide/quantization.md:87` and `:91` at v0.13.0).
- Per-layer coverage still applies after conversion: `ASCEND_QUANTIZATION_METHOD_MAP`
  (`vllm_ascend/quantization/utils.py:20-45` at v0.13.0) has `W4A16` for `moe`
  only, `W8A8`/`W8A16`/`W8A8_MXFP8`/`W4A4_FLATQUANT_DYNAMIC` for `linear` only,
  `W4A8_DYNAMIC`/`W8A8_DYNAMIC`/`W8A8_MIX` for both. A type missing for the layer
  being loaded raises `Currently, vLLM Ascend doesn't support <type> for <layer>`
  and an unknown type raises `Currently, vLLM Ascend only supports following quant
  types: [...]` (`utils.py:107-121`) — a *later* failure, so name it before the
  reporter converts and comes back with it.
- Model support is a separate question: a converted checkpoint still has to be a
  registered architecture in that release
  (`model-type-not-recognized.md`, `environment-correspondence.md`).
- Cheap verification for the reporter: run the same serve command without
  `--quantization`; if the gate still fires, the checkpoint's own `quant_method`
  is the cause and the file list (`config.json`) is the artefact to send.

## Cost note

The recorded run (#6800) reached the right answer in 10 turns / 17 tool calls, all
of it repo greps; the classification above turns it into the three calls of the
procedure. Nothing about the deployment (privileged container, device mounts,
TP=8 + `--enable-expert-parallel`) mattered.
