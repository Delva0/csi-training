# Model type not recognised / "can version V deploy model M?"

Signature family:

- `value error, the checkpoint you are trying to load has model type 'qwen3_5_moe'
  but Transformers does not recognize this architecture. … your verison of
  Transformers is out of date` — the pydantic wrapper (`value error, `) and the
  typos are the reporter's retyping of a `ValueError` raised while the engine
  config is validated; the checkpoint is usually fine.
- `Model architectures ['X'] are not supported for now` (later, at model build).
- Plain question form: "does vllm-ascend <version> support <model>?" with no
  traceback at all.

This is a version gate, not a defect. Answer it before any other archaeology.

## Look-alike to rule out first: `Error in inspecting model architecture '<Arch>'`

```
(APIServer pid=286) ERROR 10-09 03:05:30 [registry.py:547] Error in inspecting model architecture 'Qwen3NextForCausalLM'
(APIServer pid=286) ERROR 10-09 03:05:30 [registry.py:547] Traceback (most recent call last):
```
Pasted exactly like that — the header and nothing else (recorded #3325, image
`v0.11.0rc0`, Qwen3-Next-80B-A3B at TP4) — this is **not** the gate above: the
architecture is registered in both trees and a tutorial exists for it. It is a
crash while the model class is imported, and the missing tail is the finding.

- The line is `logger.exception` in `_try_inspect_model_cls`
  (`vllm/model_executor/models/registry.py:540-549` at v0.11.0rc3), so the inner
  traceback **was** printed by the same logger, under the same `[registry.py:547]`
  prefix — ask for the lines after the header before theorising. Inspection runs in
  a fresh interpreter (`_SUBPROCESS_COMMAND = [sys.executable, "-m",
  "vllm.model_executor.models.registry"]`, `registry.py:329-331`, called at
  `:507-509` and `:959-961`; wrapper `RuntimeError: Error raised in subprocess:
  <stderr>` at `:964-969`), so the inner error is an **import-time** failure of
  `vllm_ascend.models.<model>` or of a module it imports.
- Two recorded inner frames for this exact signature: `#2889`
  `AttributeError: module 'triton.language' has no attribute 'exp'` at
  `vllm/model_executor/layers/fla/ops/op.py:26`, reached from the module-level
  `from vllm.model_executor.layers.fla.ops import …` in
  `vllm_ascend/models/qwen3_next.py:20-23`; and `#2929`
  `ModuleNotFoundError: No module named 'vllm_ascend.models.interfaces'`, a
  main-branch checkout whose `qwen3_next.py` imported it (gone at rc0, which imports
  upstream's `qwen3_next` instead).
- A **missing** triton is not what #2889 shows: vllm substitutes placeholders when
  `find_spec("triton")` fails (`vllm/triton_utils/__init__.py:7-14`,
  `triton_utils/importing.py:14-17`) and the release image removes the wheel on
  purpose (`Dockerfile:42-44`, "triton doesn't work correctly" on Ascend). An
  `AttributeError` on `triton.language` therefore means a triton **is** installed and
  it is the wrong one (the triton-ascend pairing).
- One command reproduces the real error, no server needed:
  `python -c "import vllm_ascend.models.<model_file>"` in the image (plus
  `pip show vllm vllm-ascend triton triton-ascend`, `uname -m`).
- Every upstream name `vllm_ascend/models/qwen3_next.py` imports exists at
  vllm v0.11.0rc3 (verified one by one), so an `ImportError`/`NameError` from that
  file means the **installed** vllm is not the tag — check `pip show vllm` rather
  than the export.
- If the engine does start and the *first request* dies inside
  `vllm_ascend/ops/triton/fla/**` with `MLIRCompilationError`, that is the runtime
  half of the same pair: `../image-dependency-faults/triton-ascend-pair.md`.

## Answer the literal question first

Deliver, in one paragraph: **no / yes for that version**, the earliest release
with support (from precedent, quoted as `#<id>`), and the image to use. Then the
three gates below as evidence. A "can I deploy" issue does not need a source
walk beyond the registries.

## The three gates — all must pass, one grep each at the pinned commits

1. **transformers knows the `model_type`.** Check the pins, not the model name —
   both layers can cap it. Recorded values: vllm-ascend `v0.13.0`
   `requirements.txt:34` = `transformers>=4.57.4, < 5` and vllm `v0.13.0`
   `requirements/common.txt:10` = `transformers >= 4.56.0, < 5`, so no 5.x-only
   architecture can load there; vllm-ascend `v0.22.1rc1` `requirements.txt:38` =
   `transformers==5.5.4` with vllm `v0.22.1` `requirements/common.txt:10` =
   `>= 4.56.0` excluding `5.0.*`–`5.5.0`. A `< 5` cap alone settles the answer;
   quote both lines.
2. **vllm has a config fallback for that `model_type`.**
   `vllm/transformers_utils/config.py:67` `_CONFIG_REGISTRY` (lazy names such as
   `qwen3_next="Qwen3NextConfig"`, implementation in
   `vllm/transformers_utils/configs/`); when the key is absent, `get_config`
   falls through to `AutoConfig.from_pretrained` (`config.py:148`) and that is
   where the "does not recognize this architecture" text comes from.
3. **vllm has the model class.** `vllm/model_executor/models/registry.py`
   architecture table (recorded v0.13.0 has `Qwen3NextForCausalLM`,
   `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3VL*` — no `Qwen3_5*`).
   A model file under `vllm/model_executor/models/` plus a registry row is the
   pair that matters; even with the config recognised, a missing row fails at
   build time with the second error text above.

Bypassing gate 3 by installing a newer transformers alone is the common wrong
answer: it moves the failure from config load to model build.

## vllm-ascend's own claim surface (reporter-checkable)

- `docs/source/user_guide/support_matrix/supported_models.md` — the per-release
  claim table (recorded v0.13.0 rows: `Qwen3`, `Qwen3-based`, `Qwen3-Coder`,
  `Qwen3-Moe`, `Qwen3-Next`, `Qwen3-Embedding`, `Qwen3-Reranker`, `Qwen2*`,
  `Qwen2*-VL` — no `Qwen3.5` row).
- `docs/source/tutorials/index.md` + `docs/source/tutorials/*.md` — a deployment
  tutorial exists only for claimed models; `release_notes.md` for the change log.
- `vllm_ascend/models/__init__.py` `register_model()` — Ascend overrides. In
  `v0.13.0` there is **no** `vllm_ascend/models/` directory at all (overrides
  live in `vllm_ascend/patch/`), so an "unsupported architecture" there is owned
  by upstream vllm, not by an Ascend override.
- Grep the whole vllm-ascend tree for the model family before claiming support
  either way (`grep -rli 'qwen3\.5\|qwen3_5' docs/` returned nothing at
  v0.13.0).

## Gate 4 for an MTP launch: the draft config is remapped downstream

When the report is a *spec-decode* launch (`--speculative-config`), the model type
that has to resolve is the **draft** model's, and vllm-ascend supplies it through a
patch, not through upstream vllm. Verified at vllm-ascend `367b8e6` (v0.20.2rc1):

- `vllm_ascend/envs.py:116` — `VLLM_ASCEND_APPLY_DSV4_PATCH`, `bool(int(os.getenv(...)))`,
  **default `0`**;
- `vllm_ascend/patch/platform/__init__.py:46-48` — only when that flag is truthy
  are `patch_kv_cache_coordinator` and `patch_speculative_config` imported;
- `vllm_ascend/patch/platform/patch_speculative_config.py:15-24` —
  `hf_config_override` rewrites `model_type` for `deepseek_v3`, `deepseek_v32`,
  `deepseek_v4`, `glm_moe_dsa` → `"deepseek_mtp"`, and for `deepseek_v4` also sets
  `architectures=["DeepSeekV4MTPModel"]`; the same hook does the equivalent for
  the other MTP families (`qwen3_next`, `qwen3_5`/`qwen3_5_moe`, `glm4_moe(_lite)`,
  `ernie4_5_moe`, `longcat_flash`, `step3p5`, MiMo, pangu).

For a DeepSeek-V4 MTP launch, gate 4 is therefore: is the flag in the launched
process's environment? The tree's own DSV4 tests set it
(`tests/e2e/multicard/4-cards/test_deepseek_v4.py:31,72`), as does the nightly
`tests/e2e/nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml`.
Without it the draft keeps the base `model_type`, so "the draft is not the
validated MTP model" is the finding even when the engine starts. At this ref DeepSeek-V4 has
**no** row in `docs/source/user_guide/support_matrix/supported_models.md` and no
`docs/source/tutorials/models/DeepSeek-V4*.md`, so the nightly yaml is the only
in-tree recipe — quote it as such, and treat the reporter's hardware deviation
(the yaml is A3) as a deviation to name, not a verdict. The MTP/runtime side of
the same family is `../spec-decode-mtp/drafter-count-copy-device-fault.md`.

## Precedent, cheaply

```bash
bash <skill_root>/scripts/history-titles.sh 'qwen3\.5|qwen3_5'   # skill_root: from skill_view
```

Look for a `[New Model]` summary issue — it states the first supporting release
in one line (`#7462: "Qwen3.5 is already supported in vllm-ascend v0.17.0rc1"`,
naming the exact model repos) — plus 2-3 bug issues whose environment block names
the release that actually ran.

Distinguish a maintainer claim from a reporter-confirmed outcome. Recorded
#8409: the summary issue claims v0.17.0rc1 support, while the reporter later
reported (public comment, 2026-04-24) that `0.17.0rc1` failed and `0.18` worked
on the same model — so recommend **0.18**, quote both, and label which is which.
An earlier maintainer-authored reference issue for the same error class (#3819)
resolves it as "upgrade transformers / wait for the vllm pin bump"; that is
correct only when gates 2 and 3 already pass.

## Also check the surrounding stack once

A version gate answer is more useful with the release's documented baseline: the
`docs/source/installation.md` requirements table of the *target* release
(recorded v0.13.0: CANN == 8.5.0, torch_npu 2.8.0, HDK 25.5 recommended). Do not
convert an HDK/driver number into a CANN version by memory — ask the reporter to
read `/usr/local/Ascend/ascend-toolkit/latest/version.cfg` and to send
`npu-smi info` if the stack matters to the answer.
