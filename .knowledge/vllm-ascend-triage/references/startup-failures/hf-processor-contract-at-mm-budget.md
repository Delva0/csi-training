# A VL serve dies building `MultiModalBudget` inside a HF processor

## When this page applies

Serve start on a VL/omni checkpoint fails **before** EngineCore is ready, before weight
load and before any NPU init, with a stack that walks the multimodal budget into the
model's own dummy-input code and ends in a `transformers` processor. Two recorded
exception forms, one phase and one owner:

**Form A — a missing attribute** (the processor class exists; the checkpoint/config
contract does not):

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

**Form B — a duplicated constructor argument** (the class exists, the *call* contract is
off):

```text
File ".../vllm/v1/core/encoder_cache_manager.py", line 229, in _compute_encoder_budget_multimodal
File ".../vllm/multimodal/profiling.py", line 170, in _get_dummy_mm_inputs
File ".../vllm/model_executor/models/glm4_1v.py", line 1005, in get_dummy_text
File ".../vllm/transformers_utils/processor.py", line 110, in cached_processor_from_config
                                                        line 72, in get_processor
File ".../transformers/models/auto/processing_auto.py", line 379, in from_pretrained
File ".../transformers/processing_utils.py", line 1105, in from_args_and_dict
    processor = cls(*args, **valid_kwargs)
TypeError: Glm4vProcessor.__init__() got multiple values for argument 'tokenizer'
```

Recorded **#1768**: GLM-4.1V-9B-Thinking, one 910B4 (32 GB), image `v0.9.2rc1`, after the
reporter ran `pip install --upgrade transformers` to get that brand-new model's
processor. Its `vllm_ascend` frames (`worker/model_runner_v1.py:166`) are call sites; the
raise is upstream + transformers.

Not this topic: the *same* `Encoder cache will be initialized with a budget of N
tokens` line with no traceback (`mm-encoder-profile-stall.md`); a request-time
placeholder/feature count mismatch (`../multimodal-inputs/index.md`); a deepest frame
that is a device dump or kernel name (`../custom-ops-runtime-faults/`).

## 0. Which way the version moves — one grep on the pin's *contents*

Both forms are transformers-version contracts, and the recorded resolutions point in
**opposite** directions. The discriminator is not the pin's version number but whether the
pin's transformers ships the model family at all:

- **The pin has the family and the reporter is off-pin** → rung 1 is the pin (#13491:
  the release pins `transformers==5.5.4`, the reporter is on 5.7.0).
- **The family postdates the pin** → the pin cannot work, so "reinstall the image's
  transformers" is **not** a rung and the lever points forward (#1768: vllm-ascend
  `v0.9.2rc1` `requirements.txt:30` pins `transformers==4.52.4`, while vllm `v0.9.2`
  `vllm/model_executor/models/glm4_1v.py:40-45` imports
  `transformers.models.glm4v.{configuration,image_processing,video_processing}_glm4v`,
  which the pin's own pin line cannot supply — the family is documented at the
  transformers **v4.53.0** export (`docs/source/en/model_doc/glm4v.md`) and 4.52.4 is not
  in that case export, so "first release carrying it" is a reconstruction, while
  "the pin predates the family" is what the reporter's own upgrade step shows. The
  reporter upgraded, hit Form B, upgraded again and then **ran the model on the same
  910B4 with the same image**, reporter-confirmed 2025-07-16).

Grep the pin's **docs** page, not its source tree: these exports drop
`src/transformers/models/*` first. `rg --files <transformers-root> | rg -i
'model_doc/<family>'` — at the v4.53.0 export `docs/source/en/model_doc/glm4v.md` exists
(with its own `## Glm4vProcessor` section) while `src/transformers/models/` holds only
`gpt2` and there is no `src/transformers/processing_utils.py` at all. So an absent source
directory is a snapshot gap, never "transformers has no such class", and the release that
first carried a *fixed* processor stays unverified unless the record names it — say "a
newer transformers release" and ask for `pip show transformers`, do not mint a version.

## 1. State the phase first — it deletes every device lane

The failing objects are the API server's `InputProcessor` and a **dummy** mm input; no
worker, HCCL communicator or device op exists yet. So the 310P / ACL / `--enforce-eager` /
graph-capture / card-budget lanes are excluded by construction, not ranked low, and no
Ascend patch can reach the line either: `rg -l -i minicpm` over vllm-ascend v0.23.0rc1
returns docs, `tests/e2e/…` and `coverage.md` only, and `rg -n -i 'glm4|glm_4'` over
vllm-ascend `v0.9.2rc1` returns nothing at all — the model runs the generic upstream path.
What is left is the **processor contract** and the **support row**.

## 2. The exception is a version contract — read the pin, not a story

- vllm-ascend v0.23.0rc1 `requirements.txt:38` pins `transformers==5.5.4`, while vllm
  v0.23.0 `requirements/common.txt:10` sets only a floor with exclusions
  (`transformers >= 4.56.0, != 5.0.*, … != 5.4.*, != 5.5.0`). An install can satisfy
  vllm and still be off the release the image validated — the same shape as the
  `torch_npu .post` rule in SKILL.md, so read the env dump's `transformers==` against the
  plugin's pin. That comparison makes "reinstall the pinned transformers, re-run the
  identical command" rung 1 **only when §0 says the pin carries the family**; on the
  v0.9.2rc1/GLM-4.1V pair the same comparison proves the opposite (pin `4.52.4`, model in
  4.53.0+), and pinning back reproduces the earlier failure.
- The vllm-side floor is often looser than the plugin's pin and names no upper bound
  (vllm v0.9.2 `requirements/common.txt:10` is a bare `transformers >= 4.51.1`), so a
  `pip install --upgrade transformers` the reporter ran is *never* refused by vllm — the
  paste's own engine line and `pip show` are the only record of what is installed.
- The shim reads the attribute unconditionally: `vllm/transformers_utils/processors/
  minicpmv.py:61` `self.version = image_processor.version`, used only at `:159`
  (`if self.version == 2.5 or add_bos`). That is the pin's own code, so with the
  release's transformers the attribute is expected on the class the checkpoint resolves
  to.

## 3. Form B is one duplicated name — do not look for a flag

`__init__() got multiple values for argument '<name>'` means exactly one thing:
`ProcessorMixin.from_args_and_dict` called `cls(*args, **valid_kwargs)` and `<name>`
arrived twice — positionally (the slot the class's `attributes` list fills when
`_get_arguments_from_pretrained` loads the sub-processors) and again as a keyword from the
processor dict or the build's own kwargs. Two owners can do that, and the first check is
the cheap one:

- **vllm passes no `tokenizer=`** — `vllm/transformers_utils/processor.py:55-90`
  (`get_processor`) forwards only `processor_name`, `*args`, `revision`,
  `trust_remote_code` and merged mm kwargs (verified at vllm v0.9.2), and the caller
  passes none either (`multimodal/profiling.py:170` → `glm4_1v.py:1005`
  `self.info.get_hf_processor()` with no kwargs). So no serve flag can create it.
- The remaining owners are the **transformers build's processor contract** (first: the
  reporter has already changed it) and the **checkpoint directory's** saved processor
  config. Only if the version move does not clear it, list the keys of both files inside
  the model directory and treat a top-level `tokenizer`/`image_processor` entry as the
  duplicate:

  ```bash
  cd <model dir> && python -c "import json;[print(f, list(json.load(open(f)))) for f in ['processor_config.json','preprocessor_config.json']]"
  ```

- `VLLM_USE_MODELSCOPE` is **not** the owner of this shape. Its patcher frame
  (`modelscope/utils/hf_util/patcher.py:177`) only rewrites the path argument; the
  modelscope family is `NotExistError: The model: … has no revision: main !`
  (`../image-dependency-faults/modelscope-hub-patch-processor-revision.md`). A `_`-for-`.`
  directory name (`GLM-4___1V-9B-Thinking`) dates the download, it does not explain the
  duplicate.

## 4. Which class the reporter's process actually built — one command decides the owner

`AutoImageProcessor`/`AutoProcessor` returns either the checkpoint's remote-code class or
the transformers-native one, and the module path separates the two owners:

```bash
uv run --no-project python -c "
import transformers as t
from transformers import AutoProcessor as A
p = A.from_pretrained('<model dir>', trust_remote_code=True)
print(t.__version__, type(p).__module__, type(p).__name__)"
```

module under `transformers.models.…` → native class, so the question is the release
contract; module under the model directory (`image_processing_<model>`, …) → the
checkpoint revision, so compare the hub revision and `config.json`. For MiniCPM-V 4.6
also ask for the checkpoint's own `"version"`: the 4.6 dispatch reads it
(`get_version_by_config`, `vllm/model_executor/models/minicpmv.py:441-451`) and silently
guesses `(2, 0)`/`(2, 5)` when the field is absent.

## 5. Two flags in the paste are not levers

- `VLLM_SKIP_MM_DUMMY_BUDGET_CHECK=1`: `rg -n 'SKIP_MM_DUMMY'` over both trees at these
  pins returns nothing, so it skipped nothing (the crash is *inside* the dummy-budget
  path it was meant to bypass). Say that in one line; it is usually why the reporter
  pasted the export at all.
- `--limit-mm-per-prompt '{"image":4,"video":1}'` on a 4.6 config: `get_supported_mm_limits`
  returns `{"image": None}` for `(4, 6)` (`minicpmv.py:578-583`), and
  `allowed_mm_limits` iterates the *supported* modalities only
  (`vllm/multimodal/processing/context.py:392-404`), so the `video` key is inert rather
  than rejected — a video request is what fails, after this crash is gone. The same holds
  for GLM-4.1V's `--limit-mm-per-prompt '{"image":32}'`: the raise precedes it.

## 6. The support claim, if the reporter asks "should it work on 310P"

`docs/source/user_guide/support_matrix/supported_models.md:106` is
`MiniCPM-V | 🔵 | A2/A3` — experimental, no 310P column. The only MiniCPM e2e row is
text-only (`tests/e2e/pull_request/one_card/test_minicpm.py`: `OpenBMB/MiniCPM4-0.5B`,
`openbmb/MiniCPM-2B-sft-bf16`; `tests/e2e/coverage.md:74`), so VL MiniCPM has no CI row
at all. Precedents: `#3092` (300I Pro / 310P MiniCPM-V, answered experimental) and
`#5349` (MiniCPM-V-4.5, failing later, at `determine_available_memory`).
For GLM-4.1V at vllm-ascend v0.9.2rc1 the row is
`docs/source/user_guide/support_matrix/supported_models.md:51` `| GLM-4v | | Need test |`
— a coverage gap, not a refusal, and *not* the reason the processor raised.

## 7. What to send back

`pip show transformers vllm vllm-ascend modelscope` from inside the failing image, the
one-command processor dump of §4, `config.json`'s `version` for MiniCPM-V, the key lists of
§3 for Form B, and — once serve starts — the first real image request: the shims read
their values only on that path, so a start-up-only success leaves the placeholder
arithmetic unvalidated.
