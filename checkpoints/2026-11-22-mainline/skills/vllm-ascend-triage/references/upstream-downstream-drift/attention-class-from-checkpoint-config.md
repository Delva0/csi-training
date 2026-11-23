# `TypeError: <Model>Attention.forward() got an unexpected keyword argument 'kv_cache'`

Read the class name in this TypeError as a fact about **construction**: it names
which attention implementation the decoder layer instantiated, and on DeepSeek
models that choice is driven by the **checkpoint's config**, not by the
vllm ↔ vllm-ascend pairing. Recorded for #881 (vllm-ascend **v0.8.5rc1**,
`/share/model/deepseek-ai/DeepSeek-V3-0324-w8a8`, TP8 PP2, ray,
`--enforce-eager --max-model-len 32768`).

## The signature

```
File ".../vllm_ascend/models/deepseek_v2.py", line 496, in forward
  hidden_states = self.self_attn(
TypeError: DeepseekV2Attention.forward() got an unexpected keyword argument 'kv_cache'
```

Deepest downstream frame, first forward of `profile_run`
(`determine_num_available_blocks → profile_run → execute_model`), after ~34 min
of successful weight loading. Lane B; nothing ran on the device, so parallel
sizes, `gpu_memory_utilization`, the ray backend and `--enforce-eager` cannot be
the cause — say that in the first paragraph either way.

## The selection rule (verified at the local pin, re-grep per reported tag)

The case's snapshot was vllm-ascend v0.7.3 (`779eebb8021e`) / vllm v0.7.3
(`ed6e9075d31e`), **older** than the reported v0.8.5rc1, so `:496` itself is
unreadable. What is citable at v0.7.3:

| Claim | Evidence |
|---|---|
| The Ascend decoder layer picks its attention class from a config property, not from a fixed import | `vllm_ascend/models/deepseek_v2.py:131-132` — `attn_cls = DeepseekV2MLAAttention if model_config.use_mla else DeepseekV2Attention` |
| `use_mla` is computed from the **checkpoint's text config** | vllm `vllm/config.py:763-767` — `is_deepseek_mla` = `model_type in ('deepseek_v2','deepseek_v3','deepseek_mtp')` **and** `kv_lora_rank is not None`; `:998-999` — `use_mla = is_deepseek_mla and not envs.VLLM_MLA_DISABLE` |
| The non-MLA upstream class is the one the error names, and at this pin its `forward` does accept `kv_cache` positionally | vllm `vllm/model_executor/models/deepseek_v2.py:282-289` |
| The Ascend tree does use the keyword form elsewhere | `vllm_ascend/models/deepseek_mtp.py:94` (same tag) |

Consequence: for a DeepSeek-V3 checkpoint, `DeepseekV2Attention` in the message
means the MLA test failed on **their** `config.json` — `model_type` outside that
tuple, `kv_lora_rank` null/relocated, or `VLLM_MLA_DISABLE` set — because a
correct MLA config would have built the MLA class. That inference costs one
print and outranks any version-pairing story:

```bash
CKPT=/share/model/deepseek-ai/DeepSeek-V3-0324-w8a8
uv run --no-project python - <<'PY'
import inspect, os, vllm, vllm.envs as e
from transformers import AutoConfig
c = AutoConfig.from_pretrained(os.environ["CKPT"], trust_remote_code=True)
print("vllm", vllm.__version__)
print("model_type", c.model_type, "kv_lora_rank", repr(getattr(c, "kv_lora_rank", None)),
      "VLLM_MLA_DISABLE", e.VLLM_MLA_DISABLE)
from vllm.model_executor.models import deepseek_v2 as d
print([n for n in dir(d) if n.endswith("Attention")])
print(inspect.signature(d.DeepseekV2Attention.forward))
PY
```

`use_mla=True` plus an MLA class in `dir()` spends the class-name clue and
reopens the pairing/`patch/**` lane; `use_mla=False` on a DeepSeek-V3 config
points at the artifact.

## What the recorded case actually was: the checkpoint, not the pairing

The original reporter later published the maintainer's answer on the same issue
(public comment, 2025-05-20): *"this is a known issue by modelslim tool. It'll be
fix later. And you can try this weight
https://modelscope.cn/models/vllm-ascend/DeepSeek-R1-W8A8 again"* — the reporter
then asked for `deepseek-v3-0324-w8a8`. Label: maintainer statement, relayed by
the reporter; **not** a mechanism, and `will be fixed later` is not "fixed".

So for a **self-converted** `-w8a8` directory (a user path, not a ModelScope
id), rank checkpoint provenance first and make the swap the first action: the
project's own published `vllm-ascend/<model>-W8A8` weight is the same-day test,
and it is the only action that does not need the reporter to read code.

Two qualifiers the reporter's own follow-up exposes. The published set is **per
model**, and the maintainer named `vllm-ascend/DeepSeek-R1-W8A8` on a
`DeepSeek-V3-0324-w8a8` report; the reporter's next reply was the question that
leaves open — does a `deepseek-v3-0324-w8a8` exist at all. So (a) name the artifact
for *their* model and do not assert it exists (this snapshot cannot list the hub, and
the quoted comment is one model's), and (b) a swap to a sibling model's conversion
exercises the modelslim toolchain, not their checkpoint: treated alone it leaves the
reporter's own export unvalidated, so pair it with the `config.json` print above, or
report the swap as toolchain evidence and say the model itself is still open. The
maintainer's "will fix later" is a statement of intent, not a release or a date.

The recorded answer ranked "vllm and vllm-ascend are not the paired halves"
first, with a correct but secondary observation ("the error names
`DeepseekV2Attention`, **not** `DeepseekV2MLAAttention`") buried in its
reasoning. That observation *is* the discriminator; use it, do not footnote it.

## Why pairing is the weaker rank here, and when it is right

- A pairing defect also fails on the project's published weight, and it fails for
  the whole model family rather than one directory converted on one machine.
- A missing/renamed `kv_cache` parameter in the paired vllm is a real shape
  (compare the packaged commit with the tag's own table in
  `docs/source/user_guide/release_notes.md`), but it is *proved* by
  `inspect.signature` in their container, not by the class name.
- The same keyword form appears at `deepseek_mtp.py:94`; if MTP/spec-decode is on,
  check that call site too before proposing a one-line patch.
- Repair, if a patch is wanted: make the copied `forward` delegate to
  `super().forward()` (or match the installed signature exactly) instead of
  guessing which keywords this vllm accepts — the same rule as
  `patched-forward-api-drift.md`.

## What stays open

- How a modelslim conversion lands on the **non-MLA** class was not established.
  Two candidates, separated by the print above: the export's `config.json` lost /
  relocated `kv_lora_rank` (or changed `model_type`), or the quant config
  overrides the layer's attention selection. State the check, not the mechanism.
- Whether later v0.8.5+ releases carry a repair: the snapshot stops at v0.7.3, so
  neither the defect nor a fix is in any pinned tree. Do not answer "fixed
  upstream" from a newer doc.
