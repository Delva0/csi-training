# Quantization method: rejected at config time, or never selected at all

Two failures live on this page and they point in opposite directions. **A** is a
method that is named and refused before any worker starts; **B** is a method that
was never named, on a checkpoint whose own layout deliberately carries no
`quantization_config` — so nothing refuses it, the engine starts, and it dies in a
worker while loading int8 tensors into a dense model.

## A. Signature family (named and rejected)

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

## B. The other direction: the flag omitted on a layout that has no `quantization_config`

Signature: `RuntimeError: Engine core initialization failed. See root cause above.`
(upstream's wrapper) with **no worker traceback** in the paste, on an
msmodelslim-converted W8A8 checkpoint (recorded #2015, vllm-ascend v0.9.2rc1
`b5b7e0e` + vllm v0.9.2 `a5dd03c`, 910B, DeepSeek-R1-bf16-hfd-w8a8, offline
`LLM(model=..., enforce_eager=True)` with no `quantization=`). Nothing in the log
names a quantization problem, which is why this direction is missed.

The chain, source-verified at that pin:

1. **The documented export has no `quantization_config` on purpose.**
   `docs/source/user_guide/feature_guide/quantization.md:37` says of the converted
   `config.json`: *"Please make sure that there is no `quantization_config` field
   in it"*; `:5` says the feature is enabled *"by specifying `--quantization
   ascend`"*; `:94`'s validated serve line carries `--quantization ascend`.
2. **So upstream's auto-detection cannot fire.** `ModelConfig._verify_quantization`
   (`vllm/config.py:881`) parses a method only from
   `hf_config.quantization_config` / `compression_config`
   (`_parse_quant_hf_config`, `:874-879`) and gates the whole override loop behind
   `if quant_cfg is not None:` (`:895`); with both keys absent, `self.quantization`
   stays `None` unless the caller passes it.
   `AscendQuantConfig.override_quantization_method`
   (`vllm_ascend/quantization/quant_config.py:82-86`) returning `"ascend"`
   whenever `torch.npu.is_available()` therefore does **not** help: it is only
   consulted inside that block, i.e. it *renames* an existing method, it cannot
   invent one. (It also means "no quantization_config ⇒ must pass the flag" is a
   layout contract, not a platform limitation.)
3. **What follows.** No `AscendQuantConfig`, hence no
   `quant_model_description.json` reader (`quant_config.py:74-75`), hence
   `UnquantizedLinearMethod` for every `LinearBase` layer: the int8 weights and
   their `.weight_scale`/`.quant_bias` keys are matched against a dense bf16 model
   inside `load_model` on each rank. The pinned tree's own FAQ maps the resulting
   shape — `KeyError: 'xxx.layers.0.self_attn.q_proj.weight'` — to exactly this
   cause: *"First, make sure you specify `ascend` quantization method"*
   (`quantization.md:99-101`). Precedent #813 at a nearby ref reports the same
   `KeyError …q_a_proj.weight` from `load_model`.

**The mirror signature is a different page.** When the missing name is a quant aux
tensor — a *scale* (`KeyError: '<prefix>.weight_scale'`, e.g.
`…self_attn.wkv.weight_scale`) or an *offset* (`…mlp.experts.w2_weight_offset`, where
the index-free name is upstream's remap of a per-expert key and only the
`W8A8_DYNAMIC` MoE method allocates offsets at v0.8.5rc1) — and the
rank set clusters on a few sub-projections rather than spreading over every linear,
the direction is inverted: the checkpoint carries more scale tensors than the engine
has parameters, which at this family of checkpoints usually means a **full-precision
release** was served where the tutorial lists only the quantised variant —
`weight-scale-keyerror-quant-variant.md` (census first, `--quantization ascend`
second). The **offset** form reads differently and that page's `.weight_offset`
section owns it: at v0.8.5rc1 only `W8A8_DYNAMIC` allocates MoE offsets (the static
`W8A8` MoE raises `NotImplementedError` before load), so a load-phase
`…mlp.experts.w2_weight_offset` `KeyError` means the description in effect marked the
experts `FLOAT` — a wrong/hand-merged description, not a full-precision release
(recorded #986).

What the answer says, in order: (1) pass `quantization="ascend"` (serve:
`--quantization ascend`) — that is the documented enabling switch and the pinned
FAQ's first remedy; (2) the paste has no root cause, so ask for the **per-rank
worker traceback** (lanes: `silent-worker-exit.md`) and the checkpoint's
`config.json` + file list; (3) name the second candidate with arithmetic rather
than prose — a 671B int8 checkpoint is 671 GB / TP cards against
`hbm × gpu_memory_utilization`, `bash ../../scripts/card-budget.sh --params 671
--dtype w8a8 --tp 16 --gmu 0.8` leaves ~9.3 GB per card for activations **and**
the KV pool, and MLA KV is not head-sharded: per token per layer it is
`(kv_lora_rank + qk_rope_head_dim) × 2 bytes` (bf16 cache) and every TP rank holds
all layers' worth, so a 163840-token context is several GB per rank before any
activation. Read those two constants from the checkpoint's `config.json`; do not
recall them.

**State the exclusions too**, because a ranked list without them reads as a
fishing list: `enforce_eager=True` already satisfies the ACL-graph/DeepSeek guard
(`vllm_ascend/ascend_config.py:143-167`; the `NotImplementedError: ACL Graph does
not support deepseek` is raised only in the `else` branch of `if enforce_eager`),
so #1458's cause is out; and a method that is *rejected* produces section A's
`ValueError`, which this paste does not contain.

## Cost note

The recorded run (#6800) reached the right answer in 10 turns / 17 tool calls, all
of it repo greps; the classification above turns it into the three calls of the
procedure. Nothing about the deployment (privileged container, device mounts,
TP=8 + `--enable-expert-parallel`) mattered.
