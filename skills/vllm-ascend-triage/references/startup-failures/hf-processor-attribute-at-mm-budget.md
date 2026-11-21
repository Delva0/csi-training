# A VL serve dies building `MultiModalBudget` with an `AttributeError` from a HF processor

## When this page applies

Serve start on a VL/omni checkpoint fails **before** EngineCore, before weight load and
before any NPU init, with a stack that walks the multimodal budget into the model's own
dummy-input code and ends in a `transformers` processor:

```text
File ".../vllm/v1/engine/input_processor.py", line 62, in __init__
    mm_budget = MultiModalBudget(vllm_config, mm_registry)
File ".../vllm/multimodal/encoder_budget.py", line 87, in __init__
File ".../vllm/multimodal/registry.py", line 251, in get_dummy_mm_inputs
File ".../vllm/multimodal/processing/dummy_inputs.py", line 83, in get_dummy_processor_inputs
File ".../vllm/model_executor/models/<model>.py", ... in get_dummy_mm_data
File ".../vllm/transformers_utils/processors/<model>.py", line NN, in __init__
AttributeError: '<HfProcessorClass>' object has no attribute '<x>'
```

Recorded **#13491**: MiniCPM-V-4.6, `npu-smi` `310P3`, image `v0.23.0rc1-310`,
`transformers==5.7.0`, `AttributeError: 'MiniCPMV4_6ImageProcessor' object has no
attribute 'version'`.

Not this topic: the *same* `Encoder cache will be initialized with a budget of N
tokens` line with no traceback (`mm-encoder-profile-stall.md`); a request-time
placeholder/feature count mismatch (`../multimodal-inputs/index.md`); a deepest frame
that is a device dump or kernel name (`../custom-ops-runtime-faults/`).

## 1. State the phase first — it deletes every device lane

The failing objects are the API server's `InputProcessor` and a **dummy** mm input; no
`EngineCore`, worker, HCCL communicator or `vllm_ascend` frame exists yet. So the 310P /
ACL / `--enforce-eager` / graph-capture / card-budget lanes are excluded by
construction, not ranked low, and no Ascend patch can reach the line either
(`rg -l -i minicpm` over vllm-ascend v0.23.0rc1 returns docs, `tests/e2e/…` and
`coverage.md` only). What is left is the **processor contract** and the **support row**.

## 2. The missing attribute is a version contract — read the pin, not a story

- vllm-ascend v0.23.0rc1 `requirements.txt:38` pins `transformers==5.5.4`, while vllm
  v0.23.0 `requirements/common.txt:10` sets only a floor with exclusions
  (`transformers >= 4.56.0, != 5.0.*, … != 5.4.*, != 5.5.0`). An install can satisfy
  vllm and still be off the release the image validated — the same shape as the
  `torch_npu .post` rule in SKILL.md, so read the env dump's `transformers==` against the
  plugin's pin and make "reinstall the pinned transformers, re-run the identical command"
  rung 1.
- The shim reads the attribute unconditionally: `vllm/transformers_utils/processors/
  minicpmv.py:61` `self.version = image_processor.version`, and the value is used only at
  `:159` (`if self.version == 2.5 or add_bos`). That is the pin's own code, so with the
  release's transformers the attribute is expected on the class the checkpoint resolves
  to.
- Which transformers release carries `.version` on the **native**
  `MiniCPMV4_6ImageProcessor` cannot be read from this snapshot: the transformers export
  holds `docs/source/en/model_doc/minicpmv4_6.md` (documenting `MiniCPMV4_6Processor`,
  `MiniCPMV4_6ImageProcessor`, `MiniCPMV4_6ImageProcessorPil`,
  `MiniCPMV4_6VideoProcessor`, contributed 2026-04-28) and **no**
  `src/transformers/models/minicpmv4_6/` — `rg --files -g '*minicpmv4_6*'` returns the
  docs page alone. Say "off the pin", never "transformers removed it in 5.x"
  (`../upstream-downstream-drift/index.md` §pins).

## 3. Which class the reporter's process actually built — one command decides the owner

`AutoImageProcessor` returns either the checkpoint's remote-code class or the
transformers-native one, and the module path separates the two owners:

```bash
uv run --no-project python -c "
import transformers as t
from transformers import AutoImageProcessor as A
p = A.from_pretrained('<model dir>', trust_remote_code=True)
print(t.__version__, type(p).__module__, type(p).__name__, hasattr(p, 'version'))"
```

module under `transformers.models.…` → native class, off-pin/attribute contract (rung 1
above); module under the model directory (`image_processing_minicpmv`, …) → the
checkpoint revision, so compare the hub revision and `config.json`. Ask for the
checkpoint's own `"version"` too: the 4.6 dispatch reads it
(`get_version_by_config`, `vllm/model_executor/models/minicpmv.py:441-451`) and silently
guesses `(2, 0)`/`(2, 5)` when the field is absent.

## 4. Two flags in the paste are not levers

- `VLLM_SKIP_MM_DUMMY_BUDGET_CHECK=1`: `rg -n 'SKIP_MM_DUMMY'` over both trees at these
  pins returns nothing, so it skipped nothing (the crash is *inside* the dummy-budget
  path it was meant to bypass). Say that in one line; it is usually why the reporter
  pasted the export at all.
- `--limit-mm-per-prompt '{"image":4,"video":1}'` on a 4.6 config: `get_supported_mm_limits`
  returns `{"image": None}` for `(4, 6)` (`minicpmv.py:578-583`), and
  `allowed_mm_limits` iterates the *supported* modalities only
  (`vllm/multimodal/processing/context.py:392-404`), so the `video` key is inert rather
  than rejected — a video request is what fails, after this crash is gone.

## 5. The support claim, if the reporter asks "should it work on 310P"

`docs/source/user_guide/support_matrix/supported_models.md:106` is
`MiniCPM-V | 🔵 | A2/A3` — experimental, no 310P column. The only MiniCPM e2e row is
text-only (`tests/e2e/pull_request/one_card/test_minicpm.py`: `OpenBMB/MiniCPM4-0.5B`,
`openbmb/MiniCPM-2B-sft-bf16`; `tests/e2e/coverage.md:74`), so VL MiniCPM has no CI row
at all. Precedents: `#3092` (300I Pro / 310P MiniCPM-V, answered experimental) and
`#5349` (MiniCPM-V-4.5, failing later, at `determine_available_memory`).

## 6. What to send back

The one-command image-processor dump above (inside the failing image), `config.json`'s
`version`, whether `transformers==5.5.4` changes the traceback, and — once serve starts —
the first real image request, because the shim's `version` value is only ever compared
against 2.5, so a patched or guarded `getattr` leaves the 4.6 placeholder arithmetic
unvalidated.
