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
