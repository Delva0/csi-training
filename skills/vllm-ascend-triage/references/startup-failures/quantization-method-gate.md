# Quantization method: rejected at config time, unnameable on the CLI, never selected, or renamed onto a foreign checkpoint

Five failures live on this page and they point in different directions. **A** is a
method that is named and refused before any worker starts; **B** is a method that
was never named, on a checkpoint whose own layout deliberately carries no
`quantization_config` — so nothing refuses it, the engine starts, and it dies in a
worker while loading int8 tensors — or their static-W8A8 `deq_scale` aux tensor, §B-i —
into a dense model; **C** is a *foreign* format
(GPTQ/AWQ/…) that **is** named in `config.json` and is silently renamed to `ascend`,
so the refusal never happens and the format question surfaces as a construction-time
`KeyError` instead; **D** is a method the release supports but never named on its own
command line, so `argparse` rejects the flag before `ModelConfig` is built at all;
**E** is a checkpoint whose *weights* are FP8/E4M3 — no method name, config edit or
graph flag makes it run, and it surfaces as A's refusal or as C's `KeyError` depending
on whether that install registered `ascend` at all, so read the two error messages
together before ranking anything.
A and D are one grep apart and are told apart by the message form alone — `… not
supported in npu` versus `invalid choice`.

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

Whether the CLI even accepts the flag is a property of the **installed plugin**, not of
quantization support: on a ref whose `pre_register_and_update` rewrites the parser's
choices it accepts `ascend` and leaves e.g. `awq`/`gptq` in place (`vllm_ascend/platform.py:130-141`
at v0.13.0, `:134-147` at v0.17.0rc1; pinned by `tests/ut/test_platform.py:54-100`), so an
accepted launch command and a rejected run are consistent — do not read the surviving
choices as support. On an **older** ref the same hook ignores the parser and argparse
rejects the flag outright; that rejection is its own gate with its own fix, so read §D
before answering "not supported".

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
second). A plain `.weight` name raised from `is_layer_skipped_ascend` rather than from
`params_dict[name]` is neither of those: it is section C, a renamed foreign format —
**unless the quoted key starts with a dot** (`KeyError: '.weight'`, i.e. an empty
prefix), which is not a format question at all: `empty-prefix-quant-keyerror.md` — an upstream model file that
builds its `embed_tokens` without `prefix=`, met by the downstream hard dict index,
and fixed by that one missing kwarg. The **offset** form reads differently and that page's `.weight_offset`
section owns it: at v0.8.5rc1 only `W8A8_DYNAMIC` allocates MoE offsets (the static
`W8A8` MoE raises `NotImplementedError` before load), so a load-phase
`…mlp.experts.w2_weight_offset` `KeyError` means the description in effect marked the
experts `FLOAT` — a wrong/hand-merged description, not a full-precision release
(recorded #986). The **static-W8A8** aux spelling (`KeyError: '…gate_up_proj.deq_scale'`)
is the mirror neither: the directory is the quantised variant the release validates and
the engine was never quantised at all — §B-i, and §B-ii for the pasted launch block that
cannot be the command that ran.

### B-i. The aux-tensor spelling: `KeyError: '<prefix>.deq_scale'` — static-W8A8 checkpoint, dense engine

Same direction as the section above, entered from the other side of the same paste: the
unmatched key is a **quant aux tensor**, not a `.weight`, so it reads like the mirror
page and is not. Recorded #5812 (vllm-ascend v0.13.0rc1 `1b5d5ab` + vllm v0.13.0
`72506c98`, 4×910B3 Euler container, the tutorial's own `vllm-ascend/Qwen3-8B-W8A8`
directory, same on `0.11.0`):

```text
File ".../vllm/model_executor/models/qwen2.py", line 496, in load_weights
    param = params_dict[name]
KeyError: 'layers.0.mlp.gate_up_proj.deq_scale'
```

`deq_scale` is allocated by exactly one method at that pin —
`AscendW8A8LinearMethod.get_perchannel_param` (`vllm_ascend/quantization/w8a8.py:66-84`,
`deq_scale` at `:72-75`, float32 for bf16 params and int64 for fp16), the `W8A8` row of
`ASCEND_QUANTIZATION_METHOD_MAP` (`quantization/utils.py:19-47`); `W8A8_DYNAMIC`
allocates `weight_scale`/`weight_offset` and no `deq_scale`. So the key itself says the
served directory is a **static** modelslim export, and one grep decides the direction:
had `--quantization ascend` been in effect with a description saying `W8A8` for that
layer, that parameter would exist and the loader would never ask for it. The `KeyError`
therefore means **the engine built the linear unquantized** — this page's direction, not
`weight-scale-keyerror-quant-variant.md`'s ("checkpoint carries more scales than the
engine expects"). Precedent with the identical key: #1329, on `0.7.3.post1`, where
`--quantization ascend` was still an `invalid choice` and the same dense-model load
followed. Exonerate the directory in one read: the pin's own
`tests/e2e/models/configs/Qwen3-8B-W8A8.yaml` validates that exact model with
`quantization: ascend` (gsm8k 0.80), so "this W8A8 model is unsupported" is refutable
before any loader walk.

### B-ii. Ask for the effective command: a pasted launch block can be falsified in one grep

Since the whole direction is "the flag never took effect", the cheapest decisive
artefact is the **command that actually ran**, and a block can be shown not to be it
without leaving the snapshot: pick any flag in it and check the pinned tree defines it.
In #5812 the block ended `… --tensor-parallel-size 1 # 使用x个NPU --device ascend
--dtype auto … --quantization ascend`, and `--device` is **not an argument** of either
tree — `rg -n -- '--device' vllm/` at vllm v0.13.0 matches a single comment
(`vllm/engine/arg_utils.py:1997`, *"a new device to `--device` argument"*, i.e. a plugin
*could* add one), and `rg -n -- '--device' vllm_ascend/` at v0.13.0rc1 matches nothing
(`pre_register_and_update`, `platform.py:94-110`, appends only `ascend` to
`--quantization`). A launch carrying `--device ascend` therefore dies in `argparse`
before `APIConfig`; a log that reached `Starting to load model` cannot have come from it.

The mechanical reason worth naming, because it silently drops *every* later flag: a
docker-compose `command: >` **folds** its lines into one, and the inline `#` comments in
that folded line (here `1 # 使用x个NPU`) start a bash comment inside the `bash -c`
string — so `--device`, `--dtype`, `--max-model-len`, `--max-num-batched-tokens` and
`--quantization ascend` never arrive, and no "unknown quantization method" `ValueError`
is printed to contradict the block. This is the same truncation family as the
backslash-spaces-`#` `docker run` in
`device-id-invalid-at-worker-init.md` §*A `docker run` can silently map one card*, one
layer up. Ask for `docker inspect <ctr> --format '{{json .Config.Cmd}}'` (or
`docker exec <ctr> cat /proc/1/cmdline | tr '\0' ' '`) and the model directory's
`config.json` — a `quantization_config` there makes the flag unnecessary, its absence
makes the flag mandatory — then answer with the corrected command, not a mechanism.

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

## C. A foreign format that is named: renamed to `ascend`, then a `KeyError` at construction

Signature — engine start, after `Starting to load model` but before any shard
progress, one rank or every rank:

```text
File ".../vllm_ascend/quantization/quant_config.py", line 123, in is_layer_skipped_ascend
    is_shard_skipped = self.quant_description[shard_prefix + '.weight'] == "FLOAT"
KeyError: 'model.layers.0.self_attn.q_proj.weight'
```

and the `args:`/`Initializing a V0 LLM engine` line already shows
`quantization=ascend` even though nobody passed `--quantization ascend` (recorded
**#2960**: `Qwen2.5-32B-Instruct-GPTQ-Int8`, vllm-ascend v0.9.0rc2 `8dd686d` + vllm
v0.9.0 `5873877`). Not section A — nothing was refused — and not the `weight_scale`
page: the missing key is a plain `.weight`, and the frame is the **quant-method
lookup**, not `params_dict[name]`. The key still carries a **module path** here; a key
that is only `.weight` (`KeyError: '.weight'`, empty prefix) is
`empty-prefix-quant-keyerror.md`, where the checkpoint's format is innocent.

The chain, source-verified at that pin:

1. `NPUPlatform.supported_quantization = [ASCEND_QUATIZATION_METHOD]` — the platform
   list is `["ascend"]` alone (`vllm_ascend/platform.py:67`; later refs add
   `compressed-tensors`, so re-grep per pin).
2. Upstream's override loop asks each quant class to rename the checkpoint's declared
   method, and `AscendQuantConfig.override_quantization_method` returns `ascend`
   whenever `torch.npu.is_available()` (`quantization/quant_config.py:82-86`). So a
   `config.json` carrying `quant_method: "gptq"` becomes `ascend` and the gate in
   section A never fires — the plugin claims a format it never implements.
3. `get_quant_config` prefers the **HF dict** over the model directory's own files:
   `hf_quant_config = getattr(hf_config, "quantization_config", None)` … `return
   quant_cls.from_config(hf_quant_config)` (`vllm/model_executor/model_loader/weight_utils.py:154-165`
   at vllm v0.9.0), and `AscendQuantConfig` stores whatever dict it is handed as
   `quant_description` (`quant_config.py:55`). Its own reader for
   `quant_model_description.json` (`:74-79`) is never reached, and a GPTQ checkpoint
   has no reason to ship that file.
4. `get_quant_method` → `is_layer_skipped_ascend` indexes that dict with
   `<prefix>.weight` (`:109-124`), so the first linear dies on the GPTQ dict's keys
   (`quant_method`, `bits`, `group_size`, …). The `KeyError` is therefore a
   **format** answer, not a per-layer bug.

What to answer, in two calls: `rg -n 'supported_quantization' vllm_ascend/platform.py`
at the pin, and
`python -c "import json;c=json.load(open('<ckpt>/config.json'));print(c.get('quantization_config'))"`.
If the method is not `ascend` and `quant_model_description.json` is absent from the
directory, the checkpoint format is not implemented at that ref — say so, name the
supported routes (msmodelslim → `--quantization ascend`, W8A8/W8A8_DYNAMIC per
section A), and do not send them looking for a layer-name bug. A newer pin in the
snapshot may already declare more methods; that is the version scope. **One named
format overrides that answer**: if the census in section E finds `F8_E4M3` weight
tensors, the limit is the dtype the NPU executes, not the format the plugin skipped —
same `KeyError`, different verdict, so run the census before quoting either.
## D. Named, but never an argparse choice (`invalid choice: 'ascend'`)

```
vllm serve: error: argument --quantization/-q: invalid choice: 'ascend'
(choose from 'aqlm', 'awq', …, 'torchao', None)
```

This exits inside `argparse`, i.e. **before** `ModelConfig`, before the platform list of
§A, before any `APIServer pid=` line, worker, device or weight read — and the printed
choice list is upstream's static `QuantizationMethods` `Literal`, not a statement about
Ascend. Recorded #902 (vllm-ascend v0.8.5rc1 `ec27af3` + vllm v0.8.5.post1 `3015d56`,
image `v0.8.5rc1`, QwQ-32B-w8a8, `--quantization ascend`).

The chain, source-verified at those two pins:

- `--quantization` is built with `choices=[*QUANTIZATION_METHODS, None]`
  (`vllm/engine/arg_utils.py:600-604`) — a **copy taken when the argument is added**, from
  the module list seeded by the `QuantizationMethods` `Literal`
  (`vllm/model_executor/layers/quantization/__init__.py:8-37`).
- A plugin's `@register_quantization_config("ascend")` only *appends* to
  `QUANTIZATION_METHODS` at import time (`…/quantization/__init__.py:43-73`, append at `:73`), which cannot
  change a list already copied into an `argparse` action.
- The intended hook is `pre_register_and_update(parser)`, and upstream calls it **after**
  `EngineArgs.add_cli_args(parser)` (`vllm/engine/arg_utils.py:1640-1646`, hook call `:1645`), with a comment
  naming exactly this use ("adding a new kind of quantization method to --quantization
  argument"). So the plugin must mutate `action.choices`.

| ref | `NPUPlatform.pre_register_and_update` | `vllm serve --quantization ascend` |
|---|---|---|
| vllm-ascend v0.8.5rc1 (`ec27af3`) | `platform.py:66-75` takes `parser` and never touches it (only `adapt_patch` + `import AscendQuantConfig`) | rejected by argparse, while `supported_quantization = ["ascend"]` (`:61`) would have accepted it |
| vllm-ascend **v0.7.3.post1** (`c69ceaca`, 2025-05-29) + vllm **v0.7.3** (`ed6e9075`) | `platform.py:63-72` takes `parser` and only imports `patch_model_config` / `patch_outputs` / `patch_vllm_config` / `AscendQuantConfig`; `supported_quantization: list[str] = ["ascend"]` at `:60` | rejected, and this vllm has the same order: `--quantization` copies `[*QUANTIZATION_METHODS, None]` when the argument is added (`arg_utils.py:561-568`) while the hook runs after `add_cli_args` (`:1416`, args added `:1408`) |
| v0.13.0 (`6281c120`) / v0.17.0rc1 (`e20f0b1a`) | appends `ASCEND_QUANTIZATION_METHOD` via `parser._option_string_actions.get("--quantization")` (`platform.py:130-141` / `:134-147`) | accepted |

Procedure:

1. Classify from the message form: `argument --quantization/-q: invalid choice` is the
   parse-time lane; §A's `… not supported in npu` is the platform lane. They are different
   gates with different answers, and at an old ref the platform will accept a method the
   CLI refuses.
2. One command on the reporter's own tree, not the snapshot:
   `python -c "import inspect, vllm_ascend.platform as p; print(inspect.getsource(p.NPUPlatform.pre_register_and_update))"`
   — no `--quantization` in the body means the flag cannot be passed.
3. Three routes, cheapest first:
   - **Drop the flag** if the checkpoint declares the method itself:
     `python -c "import json;c=json.load(open('<ckpt>/config.json'));print(c.get('quantization_config'), c.get('quantize'))"`
     — a `quant_method: ascend` there makes the CLI flag unnecessary (§A step 3). On the
     documented msmodelslim export it is the opposite: `config.json` carries no
     `quantization_config` by design, so dropping the flag converts this parse error into
     §B's silent dense-model load — check the file before recommending this route.
   - **Import before parsing** (no image change): `register_quantization_config("ascend")`
     runs at import of the config module, so importing it *before* `make_arg_parser`
     builds the choices copy already containing `ascend`:

     ```python
     import vllm_ascend.quantization.quant_config          # registers "ascend"
     from vllm.entrypoints.openai.cli_args import make_arg_parser
     from vllm.entrypoints.openai.api_server import run_server
     from vllm.utils import FlexibleArgumentParser
     import uvloop; uvloop.run(run_server(make_arg_parser(FlexibleArgumentParser()).parse_args()))
     ```

     Then reuse the reporter's original flags. Source-inferred, not reporter-confirmed —
     label it so and let them run it.
   - **Upgrade** to a release whose hook rewrites the choices; the snapshot carries only
     v0.8.5rc1 in the failing state, so name the two verified states and the grep, never a
     "first fixed in" claim.
4. The **offline API is not the same door** and is a useful discriminator: `LLM(...,
   quantization="ascend")` never parses argv, and `ModelConfig._verify_quantization` reads
   `supported_quantization = QUANTIZATION_METHODS` (vllm `config.py:752-753` at that pin) —
   which the import-triggered decorator has already extended, because
   `EngineArgs.create_engine_config` calls `current_platform.pre_register_and_update()`
   (`arg_utils.py:1096`) before `create_model_config()` (`:1099`). So at v0.8.5rc1
   quantization *works* in-process and is merely unnameable on the CLI; say both, or the
   reporter will conclude the build cannot quantize at all.

### D-i. The wall behind the flag on the 0.7.3-era pair

There the flag is only the first of two stops, and the second is inside the library.
That tree has no native ascend quantizer: `AscendQuantizer.get_quantizer` returns early
only for a name in `CUSTOMIZED_QUANTIZER_TYPE`, which is empty (`:21`), then does
`importlib.import_module("mindie_turbo")` and raises
`NotImplementedError("There is no available ascend quantizer.")` when that import fails
(`vllm_ascend/quantization/quantizer.py:28-44`). The same pin's FAQ says it in words —
w8a8 is native "on v0.8.4rc2 or heigher", and under vllm 0.7.3 it works only "with the
integration of vllm-ascend and mindie-turbo, please use `pip install
vllm-ascend[mindie-turbo]`" (`docs/source/faqs.md:82`); the extra is
`extras_require={"mindie_turbo": ["mindie-turbo==2.0rc1"]}` (`setup.py:371`). Its
FAQ item 8 also states the version lock, so the honest ladder for this era is the
*paired* upgrade; hand over the wrapper/patch routes only with the `mindie_turbo`
install line beside them, or the reporter clears `argparse` and returns with a
worker-side `NotImplementedError`.

### D-ii. If you write the site-packages patch yourself, guard the lookup

`parser._option_string_actions.get("--quantization")` can be `None` for any parser that
has no such option, and that is on record: the whole body of #1042 is
`TypeError: argument of type 'NoneType' is not iterable` at `platform.py:79`, raised on
the line `if ASCEND_QUATIZATION_METHOD not in quant_action.choices:` by a CI run of
main. The same hook is also called with **no argument** from
`EngineArgs.create_engine_config` (`arg_utils.py:1097`), i.e. on the offline `LLM()` path — a body that subscripts the
lookup unconditionally breaks that path too. Two readings, both needed: take the `None`
check rather than a direct subscript, and note that main carried this mutation in May
2025 while the v0.7.3.post1 release tree (`c69ceaca`, 2025-05-29) had none of it — main and a release branch disagree here, so
neither an issue body nor a branch snapshot dates the fix. Quote the two verified states
and the one-line grep.

**The custom-op warning in the same paste is not this lane.** A 0.7.3-era log usually
also carries `WARNING:root:Warning: Failed to register custom ops, all custom ops will be
disabled`: the extension is built only when `COMPILE_CUSTOM_KERNELS` is set
(`setup.py:297-298`) and the warning fires for any other `ImportError`
(`vllm_ascend/platform.py:28-37`), so on a pip install of that era it is expected. At
v0.7.3.post1 the extension guards only the camem allocator and one `rotary_embedding` op
(`csrc/torch_binding.cpp:95-108`), and the Python rotary path calls
`torch_npu._npu_rotary_embedding` directly (`ops/rotary_embedding.py:45,77`), so it
cannot produce a parse-time error. A report that makes *that* warning the question is
`../custom-ops-runtime-faults/op-not-registered-for-soc.md`.

## E. The weights themselves are FP8/E4M3: a dtype limit, not a method or format question

The upstream DeepSeek-V3/R1 release is neither a GPTQ-like foreign format nor a
quantisation someone switched on: its `config.json` carries
`quantization_config: {quant_method: fp8, fmt: e4m3, weight_block_size: [128, 128],
activation_scheme: dynamic}` and its shards store **F8_E4M3** weights, which the 910
does not execute at the refs recorded here. It surfaces in two different places, and
which one is decided by **whether that install registered `ascend`**, not by the
version — the unconditional override exists at both pins:

| state of the running install | surface |
|---|---|
| the quant config module was imported, so `ascend` is a registered method: `override_quantization_method` returns `ascend` for *any* declared method, fp8 included (`quant_config.py:83-88` at v0.11.0rc0 `00ba071`, `:69-74` already at v0.7.1rc1 `f17417f`) | nothing is refused; construction dies on the lookup — `KeyError: 'model.embed_tokens.weight'` at `quant_config.py:152`, reached from `ops/vocab_parallel_embedding.py:81` (recorded #3534, DeepSeek-V3-0324 on vllm v0.11.0rc3 `8ce5d31`, torchair graph mode) |
| `ascend` never registered, so the checkpoint's own `fp8` survives to the platform gate | refused while `ModelConfig` is built: `ValueError: fp8 quantization is currently not supported in npu` (`vllm/config.py:599`; platform list `vllm_ascend/platform.py:60`, raiser `vllm/platforms/interface.py:243-251`). Recorded #119 — the reporter's later paste, on the *original* HF DeepSeek-V3 |

So a reporter who first hit the config-time refusal and later meets §C's `KeyError` is
not seeing a regression introduced by the upgrade: the two messages date which method
the engine resolved, and the verdict (that checkpoint cannot run) is the same. Say that
before they file it as a new bug.

Mechanism at the v0.11.0rc0 pin (source-verified at those two commits):
`AscendQuantConfig.override_quantization_method` returns `ascend` whenever
`torch.npu.is_available()` and never reads `hf_quant_cfg` (`quant_config.py:83-88`);
upstream takes that as the final method (`vllm/config/model.py:931,945`);
`get_quant_config` returns the HF dict *before* it ever looks for
`quant_model_description.json` (`vllm/model_executor/model_loader/weight_utils.py:239-250`
— the `Cannot find the config file for ascend` raise at `:289` is unreachable on this
branch); `AscendQuantConfig.__init__` stores that dict as `quant_description`
(`quant_config.py:57`). An fp8 dict holds `fmt`/`weight_block_size`/`activation_scheme`
and no weight names, so the first `is_layer_skipped_ascend` lookup dies. `embed_tokens`
is merely the first module built — `torchair/models/torchair_deepseek_v2.py:1078-1082`
does pass `prefix=` — which is why this is not `empty-prefix-quant-keyerror.md`.

**The decisive check is the weight dtype, not `config.json`.** Reporter-confirmed on
#3534 (2025-10-22): "check the weight dtype … If E4M3 weights are included, it can be
considered that this is not supported on the NPU". The declaration and the tensors are
independent facts: the pin's own guide tells converters to **delete** the quantization
fields from `config.json` before a W8A8 export
(`docs/source/user_guide/feature_guide/quantization.md:44-48` at v0.11.0rc0), so a
directory can hold E4M3 tensors with no `quantization_config` at all, and a bf16
directory can still carry a leftover one. One call censuses what is there:

```bash
uv run --no-project python <skill_root>/scripts/checkpoint-key-prefix.py --dtypes <model dir>
```

It prints the dtype histogram, the module families carrying FP8 *weight* tensors
(scale-like suffixes counted apart) and both declarations. Reading:

- **FP8 weight tensors** → the upstream FP8 release; no method name, config edit or
  graph flag makes it run. Routes: dequantize to bf16, or convert with msmodelslim to
  W8A8/W8A8_DYNAMIC and serve that (`--quantization ascend`, §B). Scope the verdict to
  the recorded refs — it is the *pinned release's* quant doc/support matrix that
  decides, so re-check it before applying the rule to a later release that adds an fp8
  row, and label the mechanism "reporter-confirmed on #3534", not "NPU-wide".
- **int8 weights + a declaration** → a converted export: §A/§B, and §C when the load
  dies on a lookup key.
- **bf16 weights + a stale `quantization_config`** → one config edit; that is exactly
  the guide's delete-the-fields step, and the weights are innocent.

### E-i. When the release carries the *name* and no implementation

The same wall as §D-i, one pin earlier and without the CLI stop in front of it. At
vllm-ascend v0.7.1rc1 the pinned tree says it twice: the release note
(`docs/source/user_guide/release_notes.md:21` — "Added the Ascend quantization config
option, the implementation will comming soon") and the only quantizer
(`AscendQuantizer.get_quantizer` imports `mindie_turbo` dynamically and otherwise raises
`NotImplementedError("There is no available ascend quantizer.")`,
`vllm_ascend/quantization/quantizer.py:30-38`). A W8A8/W8A8_DYNAMIC report there is a
**version answer in the first sentence** — wait for the release that ships the
implementation (maintainer-confirmed on #119: "The detail implementation will coming
next release"), or serve a bf16 checkpoint — not a flag hunt and not a patch. The paste
identifies the checkpoint class by itself: a modelzoo/MindIE export carries
`"quantize": "w8a8_dynamic"`, `"quantization_config": {"quant_method": "ascend"}` and
`"mla_quantize": "w8a8"` beside `torch_dtype: bfloat16` (#119).

**The same reporter's first error is the evidence for the second, and one detail stays
a gap.** #119's first paste printed `Unknown quantization method: ascend. Must be one of
[... 22 names ...]` — vllm v0.7.1's static `QUANTIZATION_METHODS` — although that tag's
`register()` imports the config module (`vllm_ascend/__init__.py:18-24`) and
`quant_config.py:39` decorates the class. That printed list is why the *later* paste is
the platform's `fp8 … not supported in npu` rather than §E's `KeyError`: the two errors
are one install state, so read them together instead of as two bugs. Which wheel
produced it stays open (the pin is a tag, `runtime_commit_verified: false`), and it
changes nothing in the deliverable — the implementation is absent either way. One
command on their box separates the readings: `python -c "import
vllm_ascend.quantization.quant_config as q; from
vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as m; print(m)"`;
no `ascend` after an explicit import means the installed package lacks the module, not
an import-order problem. Report it as a gap and stop: recorded #119 spent its whole
16 384-token budget on that question and shipped no answer.

## Cost note

The recorded run (#6800) reached the right answer in 10 turns / 17 tool calls, all
of it repo greps; the classification above turns it into the three calls of the
procedure. Nothing about the deployment (privileged container, device mounts,
TP=8 + `--enable-expert-parallel`) mattered.

The v0.7.3-era run (#2819, the same pair as the row above) then spent 25 tool calls in
26 turns and was cut by the output cap **mid-answer** — after the evidence was complete,
before §D-i, which is the paragraph that reporter needed. The turns that did not pay:
three on the custom-op warning (the paragraph above says it in one line), three on the
`vllm serve` wiring needed to write the wrapper (it is the snippet in step 3), two on
whether a msmodelslim `quant_model_description.json` can select the method by itself
(§B answers it). Six calls establish the whole lane: `supported_quantization` + the hook
body, the two `arg_utils` sites, `quantizer.py`, `setup.py:371`, the FAQ line.

The §E pair is instructive in both directions. #3534 shipped the right verdict (an FP8
checkpoint cannot run) after 12 reads, but its cheapest check went to `config.json`
first; the reporter's own rule — the weight dtype — is now the `--dtypes` call above.
#119 had the answer two calls in (release note + `quantizer.py`) and then spent its
entire 16 384-token budget on the `QUANTIZATION_METHODS` question, shipping no text at
all.
