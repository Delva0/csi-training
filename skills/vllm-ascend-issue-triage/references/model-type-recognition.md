# Checkpoint metadata vs the installed Transformers: unknown `model_type` and tokenizer class

Scope: start-up failures raised by transformers while reading the checkpoint's own metadata —
either

> The checkpoint you are trying to load has model type `<X>` but Transformers does not
> recognize this architecture. This could be because of an issue with the checkpoint, or
> because your version of Transformers is out of date.

or `ValueError: Tokenizer class <X> does not exist or is not currently imported.`, or a
processor-side `TypeError: <Model>Processor.__init__() got multiple values for argument
'tokenizer'`. All three are answered by the same tag pair plus version boundary; the processor
variant sits below the pin table.

This is a model-support and version boundary question ("can this vllm-ascend deploy this
model at all?"), not a kernel, accuracy or checkpoint-corruption report. The answer is a
tag pair — the vLLM/vllm-ascend revision and the transformers version that knows the
`model_type` or exports the tokenizer class — plus the flag or metadata the reported run
lacked. The two messages resolve differently: a `model_type` gap is fixed by a newer
containing library, which for `qwen3_5` is vLLM, while a tokenizer-class gap at 0.17/0.18 is
only fixed by installing transformers 5.x outside the image's declared pin.

## Who raises it

It is transformers' own `AutoConfig.from_pretrained`, not vLLM or the plugin. Read the
reporter's traceback path, not just the message:

- transformers `v4.57.6`: `src/transformers/models/auto/configuration_auto.py:1360`
  `config_class = CONFIG_MAPPING[config_dict["model_type"]]` → `KeyError: 'qwen3_5'`,
  then `:1362` raises the ValueError; the message body at `:1364` is the line to quote.
- transformers `main` (clone HEAD, 2026-11-01): the same statements at `:436` / `:440`.

Neither the vLLM nor the vllm-ascend tree contains that sentence:
`git grep -n "does not recognize this architecture" <vllm-tag> -- vllm/` prints nothing at
`v0.13.0` or `v0.18.0`, and `git grep -c qwen3_5 v0.13.0 -- vllm/` is also empty. A run that
greps the two plugin/upstream clones for the wording learns only that the wording is
absent; the transformers clone at `/repos/huggingface/transformers` is the owner and answers
"does the installed version know this `model_type`".

## Three checks, in this order

1. **Does the installed transformers know the `model_type`?** Compare the reporter's
   transformers version with the clone's tags before reading code:
   `git show v4.57.6:src/transformers/models/auto/configuration_auto.py | grep -c qwen3_5`
   returns `0`; the same file at `v5.2.0` returns `10`. Bound a family with
   `git log --diff-filter=A --format='%h %ad %s' --date=short -- src/transformers/models/<family>/configuration_<family>.py`
   then `git describe --contains <sha>` for the earliest tag (do not order `git tag --contains`
   with `sort -V`: it prints a final release before its own rc tag).
2. **Does the paired vLLM tag carry its own config class?** The plugin delegates model
   support to upstream, so check upstream at the same number:
   `git ls-tree -r --name-only <vllm-tag> | grep -i <model>` for
   `vllm/transformers_utils/configs/<model>.py`, and the `_CONFIG_REGISTRY` entries in
   `vllm/transformers_utils/config.py`. When the entry exists, the *model config* loads and
   the log says `Resolved architecture: ...`, so a failure after that line is a later load
   stage, not "unsupported model".
3. **Is the failing call the tokenizer/renderer path?** vLLM builds the registered class
   directly in the `if model_type in _CONFIG_REGISTRY:` branch
   (`v0.18.0:vllm/transformers_utils/config.py:194`) and only reaches
   `AutoConfig.from_pretrained` in the `else` (`:206`); it never calls
   `AutoConfig.register`, so transformers' `CONFIG_MAPPING` stays unaware of the type.
   `CachedHfTokenizer.from_pretrained` then calls `AutoTokenizer.from_pretrained` with no
   config argument (`v0.18.0:vllm/tokenizers/hf.py:85`). In transformers `v4.57.6`
   `AutoTokenizer` reads `tokenizer_config.json`; when `tokenizer_class` is absent
   (`config_tokenizer_class is None`, `models/auto/tokenization_auto.py:1088-1108`) it calls
   `AutoConfig.from_pretrained(..., trust_remote_code=...)` (`:1109`), which is where the
   `CONFIG_MAPPING` failure lands — `--trust-remote-code` does not help without an `auto_map`.
   A run that printed `Resolved architecture` and then died in
   `renderers/registry.py -> tokenizers/registry.py -> tokenizers/hf.py -> tokenization_auto.py
   -> AutoConfig.from_pretrained` is on this path.

## Sibling failure: `Tokenizer class <X> does not exist or is not currently imported.`

Same version boundary, different owner. transformers' `AutoTokenizer.from_pretrained` reads the
checkpoint's `tokenizer_config.json`, tries `f"{config_tokenizer_class}Fast"`, then the bare name,
and raises at `models/auto/tokenization_auto.py:1153-1154` (`v4.57.6`); `--trust-remote-code` does
not help because no `auto_map` is involved. The wording is gone at `v5.5.3`: 5.x maps
nearly every `model_type` to `TokenizersBackend`
(`models/auto/tokenization_auto.py:64-383`, with a
`TOKENIZER_MAPPING_NAMES[model_type] = "TokenizersBackend"` default at `:382-383`), so a checkpoint
written by a 5.x transformers saves that class name and only a 5.x install can resolve it. The class
is `src/transformers/tokenization_utils_tokenizers.py:86`, present from `v5.0.0rc0` and absent at
`v4.57.6` and `v4.58.0`.

The image pin comes from upstream vLLM's `requirements/common.txt`, not from vllm-ascend, whose
`Dockerfile`/`Dockerfile.a3` only carries `ARG VLLM_TAG` (`v0.17.0rc1` and `v0.18.0rc1` both point
at `v0.17.0`/`v0.18.0`):

| transformers range allowed | vLLM tags |
| --- | --- |
| `>= 4.56.0, < 5` (no 5.x) | `v0.17.0`, `v0.18.0` |
| `>= 4.56.0`, excluding `5.0.*`-`5.4.*` and `5.5.0` (so 5.5.3+ allowed) | `v0.19.1`, `v0.20.2`, `v0.23.0` |
| `>= 5.5.3` | `v0.24.0`, `v0.26.0` |
| `>= 5.16.1, < 5.23.0` | `main`, 2026-11-01 |

So a 0.17/0.18 image can never resolve a 5.x transformers: there is no in-pin upgrade, and the fix
is either a manual install or a newer image. Recorded instance (issue #8213,
`v0.18.0rc1-a3-openeuler` and the same reporter's `v0.17.0rc1-a3`, MiniMax-M2.7 bf16, start-up dies
at the message above, transformers 4.57.6): the reporter-confirmed fix is
`pip install transformers==5.5.3` inside the container — "可以正常启动了" then "问题已解决". From
`v0.19.1` on the pin permits 5.5.3+, so that image's own install no longer blocks 5.x. Label the
workaround **thread-confirmed, outside the declared pin**: numerics under 5.x on those images were
never checked, so do not present it as validated serving.

## Sibling failure: `TypeError: <Model>Processor.__init__() got multiple values for argument 'tokenizer'`

A multimodal start-up dies while *building* the HF processor, before any weight is read
(recorded on `v0.9.2rc1` + GLM-4.1V-9B-Thinking, issue #1768). Verified in
`/repos/huggingface/transformers` and `/repos/vllm-project/vllm`:

- **The traceback pins transformers `v4.53.0`, not the version string.** The pair is
  `models/auto/processing_auto.py:379` `return processor_class.from_pretrained(` ->
  `processing_utils.py:1304` `return cls.from_args_and_dict(args, processor_dict, **kwargs)` ->
  `:1105` `processor = cls(*args, **valid_kwargs)`. `v4.52.4` has `:376`/`:982`, `v4.53.1` and
  `v4.53.3` `:379`/`:1109`, `v4.54.0` `:385`/`:1113` — only 4.53.0 has the `:1105` statement, so
  reproduce the two statements before believing any reported version.
- **Mechanism:** 4.53.0's `from_args_and_dict` deletes positional `args` by *index*
  (`args_to_remove = [i for i, arg in enumerate(accepted_args_and_kwargs) if arg in processor_dict]`,
  then filters `args` by those indices) while leaving the same names in `valid_kwargs`; deleting an
  earlier attribute (e.g. `image_processor`) shifts `tokenizer` into the slot `__init__` names
  `image_processor`, and the surviving `tokenizer` keyword then collides with it. That is the
  reported message, and vLLM's explicit `tokenizer` (`vllm/inputs/registry.py`:
  `kwargs["tokenizer"] = self.tokenizer`) is what supplies the keyword. The fix `a10c2bf840`
  "Fix multimodal processor get duplicate arguments when receive kwargs for initialization"
  (#39125) keeps positions and pops each matched name out of `valid_kwargs` instead, **first tag
  `v4.53.1`** (`git describe --contains a10c2bf840` = `v4.53.1~3`): 4.53.0 is the only affected
  release, 4.52.x and 4.53.1+ are fine. vLLM's guard comment cites vllm#20224, and the affected
  processor there was `Glm4vProcessor` (`attributes = ["image_processor", "tokenizer",
  "video_processor"]`).
- **vLLM's own workaround is what a mixed local install defeats.** `v0.9.2`
  `vllm/inputs/registry.py:133-137` skips the explicit tokenizer only when
  `Version(TRANSFORMERS_VERSION) != Version("4.53.0")` — exact string equality. A source build of
  the same tree (traceback path `/workspace/transformers/src/transformers/...`, version string
  `4.53.0.dev0` or any local/post suffix) carries 4.53.0's code and is *not* skipped, so the guard
  passes and the crash survives. `01cae37713` (#20541, first tag `v0.10.0rc1`) removed the guard
  and always passes `tokenizer=`, so from `v0.10.0rc1` on only transformers >= 4.53.1 is safe.
- **Workaround:** `pip install transformers==4.53.1` (or 4.53.3, or hold 4.52.x) inside the
  container. Thread-confirmed direction only: the #1768 reporter's "升级transformer版本后已可以成功
  运行" and a second tester's `pip install transformers<4.54.0` both land on "not 4.53.0", which the
  code boundary above supports; nobody in the thread printed the resulting version.
- **The image does not pin transformers at this age.** vllm-ascend `v0.9.2rc1`'s `Dockerfile`
  installs vLLM `v0.9.2` editable and names no transformers line, so vLLM
  `requirements/common.txt` (`transformers >= 4.51.1`, no upper bound) decides by build date. Do
  not derive the installed transformers from the vllm-ascend tag; use the traceback's file:line.

## Qwen3.5 / Qwen3.5-MoE boundaries (verified 2026-11-01)

| stack | coverage of `qwen3_5` / `qwen3_5_moe` |
| --- | --- |
| vLLM `v0.13.0` | none — no reference anywhere under `vllm/` |
| vLLM `v0.17.0rc1`+ | `vllm/model_executor/models/qwen3_5.py`, `qwen3_5_mtp.py`, `vllm/transformers_utils/configs/qwen3_5.py`, `qwen3_5_moe.py`, registry entries |
| transformers `v4.57.6` | 0 hits in `models/auto/configuration_auto.py` |
| transformers `v5.2.0`+ | first tag containing `fc91372258` "Adding Support for Qwen3.5 (#43830)" |

- **Upgrading transformers is not the fix inside these vLLM installs.** The pin is
  `transformers >= 4.56.0, < 5` at `v0.13.0`, `v0.17.0rc1` and `v0.18.0`, while Qwen3.5
  first exists in transformers `v5.2.0`. `v4.57.6` is the last 4.x tag and still has 0 hits,
  so no in-pin upgrade makes `qwen3_5` resolve; the post-0.17 answer is a newer
  vLLM/vllm-ascend, not a `pip install --upgrade transformers`.
- **A title that says `Qwen3.6` maps onto this family.** There is no `Qwen3.6` tutorial at
  `v0.18.0` or `v0.20.2rc1` (only `Qwen3.5-27B.md` and `Qwen3.5-397B-A17B.md`), and the reports
  labelled `qwen3.5` name `Qwen3.6-27B` (dense) and `Qwen3.6-35B-A3B` (MoE, served with
  `--enable-expert-parallel` and `--speculative_config '{"method": "qwen3_5_mtp", ...}'`). Diff the
  serve line against the Qwen3.5 tutorial of the matching shape, and treat the reported model
  directory name as a local rename, not a version signal.
- **Thread #8409** (`vllm-ascend 0.13.0`, `Qwen3.5-397B-A17B-w8a8`, transformers 4.57.6,
  `qwen3_5_moe`): maintainer answered "qwen3.5从0.17.0rc1才开始支持"; the reporter then
  confirmed "0.17.0rc1失败，用0.18版本成功了". Use it as the documented upgrade path for a
  pre-0.17 dig, and say which side of the boundary each version sits on.
- **Thread #8315** (`v0.18.0rc1` image, `Qwen3.5-27B-w8a8-mtp`, transformers 4.57.6) is the
  tokenizer path of check 3: the log resolves `Qwen3_5ForConditionalGeneration` and
  `Qwen3_5MTP`, then fails inside `AutoTokenizer`/`AutoConfig` with `KeyError: 'qwen3_5'`.
  Its thread holds no verified root cause — "It seems you're using a outdated
  `transformers`, have you changed the version ...?" is a maintainer hypothesis, the
  reporter says the image was unchanged, another tester reports the *same image* starting
  fine with `Qwen3.5-27B-w8a8-org` weights, and the issue was closed stale after a
  "resolved in the latest version" note. So the open candidate is checkpoint-side tokenizer
  metadata for the `w8a8-mtp` repack (missing tokenizer class / `auto_map`), not the
  model-support boundary; label it a candidate until the reporter's
  `tokenizer_config.json` is inspected.

## Diagnostic order that answers the report

1. Pin the plugin and upstream vLLM tags from `collect_env.py`, plus the reporter's
   transformers version.
2. Run check 2 on the upstream tag and check 1 on the transformers version. If either side
   lacks the family, the answer is the first tag that has it (plugin/upstream) or the
   transformers release that has it, with the pin range as the caveat.
3. Only if both sides know the type, read the failing stage from the traceback (model
   config vs tokenizer/renderer vs processor construction) and ask for the checkpoint's
   `config.json` / `tokenizer_config.json` / `preprocessor_config.json` before proposing a code
   fix. A `TypeError` naming a processor `__init__` argument is the processor-construction stage,
   so the transformers pin (not the checkpoint) is the suspect.

## What does not belong here

- `KeyError: '<module>.weight'` from the quant config, or `params_dict[name]` KeyError
  during weight loading — those are checkpoint/quant classes, see
  [quantization-startup-keyerrors.md](quantization-startup-keyerrors.md).
- A `TypeError`/`AttributeError` raised inside a model class at a supported tag — that is a
  code defect, not a support boundary.
- A `model_type` that only the reporter's converted checkpoint invented: compare
  `config.json`'s `model_type` with the upstream repository's before quoting tag
  boundaries, because a repack typo mimics this message at any version.
