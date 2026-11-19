# A class the installed Transformers does not know (`model_type`, tokenizer class): a pin floor, not a code path

## Scope

Serve or `LLM(...)` startup dies while the tokenizer/renderer is built, before any weight loads:

```text
ValueError: The checkpoint you are trying to load has model type `qwen3_5` but Transformers does not
recognize this architecture. This could be because of an issue with the checkpoint, or because your
version of Transformers is out of date.
```

often preceded in the same log by `KeyError: 'qwen3_5'` inside `CONFIG_MAPPING[...]`. The container is
healthy (`npu-smi` fine), no NPU op ran, and nothing in `vllm_ascend/` appears in the traceback. This is
a **dependency-floor** report: the checkpoint's `config.json` names a `model_type` that the installed
`transformers` release does not register. Reported for the newest family an image ships (`Qwen3.5`,
`GLM-5`, a `-w8a8`/`-mtp` variant), usually on a container cut before that model's supporting
transformers release.

The same floor has a **tokenizer** member, reached one step later on the same frontend path and answered
the same way: `ValueError: Tokenizer class <X> does not exist or is not currently imported.`, where `<X>`
is the `tokenizer_class` the checkpoint's `tokenizer_config.json` declares (e.g. `TokenizersBackend`, a
transformers-v5-only class). The config loads fine; tokenizer construction dies. See "The tokenizer twin"
below.

The third member is a **processor** class the installed transformers *does* have, but whose constructor
cannot be called the way vLLM calls it: `TypeError: <X>Processor.__init__() got multiple values for argument
'tokenizer'` out of `transformers/processing_utils.py` `from_args_and_dict`. The usual cause is a source or
`dev0` install: the code is a release whose own bug the calling vLLM already guards against by comparing the
version string exactly, so the guard misses. See "The processor twin" below.

Not this page: the chat-template guard `As of transformers v4.44, default chat template is no longer
allowed` (`tokenizer-and-chat-template.md`, which names transformers but is raised by vLLM), reasoning
output that is present but in another field (`reasoning-parser-output.md`), an installed-wheel API
skew *inside* a library (`grammar-bitmask-and-dep-skew.md`), a `KeyError` on a weight name
(`quantization-load.md`), or a model patch (`patch/worker/patch_<model>.py` is imported later and is never
reached).

## Common rules

- **"vLLM supports the model" and "Transformers knows its config" are independent.** vLLM ships its own
  modelling code, so the engine can carry an architecture while the frontend still dies. The recorded
  traceback runs `vllm/entrypoints/openai/api_server.py` → `AsyncLLM.from_vllm_config`
  (`async_llm.py:135`) → `renderer_from_config` (`renderers/registry.py:91`) → `vllm/renderers/hf.py:625`
  → `vllm/tokenizers/hf.py:85/110` → `AutoTokenizer.from_pretrained` → transformers
  `AutoConfig.from_pretrained` → `models/auto/configuration_auto.py`
  `CONFIG_MAPPING[config_dict["model_type"]]` → `KeyError` → re-raised as the `ValueError` with the
  "pip install --upgrade transformers" hint. Those frames only prove the failure precedes
  `vllm_ascend/`; the owning artefact is the *transformers registry plus the image pin*. Do not hunt a
  plugin line, and do not read the vLLM-side model registry as support for the tokenizer path.
- **A repo pin is a floor, not a promise.** `vllm-ascend` v0.18.0/rc1 pins `transformers>=4.57.4`
  (`pyproject.toml:20`, `requirements.txt:38`); 4.57.6 satisfies it and predates the model entirely. The
  pin says what the release was built against, never what a new model needs.
- **The docs of a later release state the floor verbatim.** Grep them before the code: at `v0.19.1rc1`
  `docs/source/tutorials/models/GLM5.md:9` says "The version of transformers need to be upgraded to
  5.2.0" (FAQ `:1389`), `docs/source/user_guide/release_notes.md:295` says the same and that it "will not
  be included in v0.17.0", and `:92` records the flip — "Upgraded to transformers 5.5.3 (from 4.57.4)"
  (#8448). One `git grep -n -i transformers <ref> -- docs` is the cheapest confirmation that this class
  is the intended answer.
- **Check three versions, not one**: the image's installed version (the reporter's `collect_env`, or a
  maintainer's reply), the repo pin at the reported ref, and **upstream vLLM's cap at its paired tag**
  (`git show <vllm-ref>:requirements/common.txt | grep -i '^transformers'`). A version that satisfies
  the model but violates vLLM's cap is not a workaround — say why instead of recommending the pip bump.
- **A tag that was never cut prints exactly like a removed pin.** In a loop
  `for t in …; do git show $t:requirements.txt | grep -i transformers || echo MISSING; done`, a
  nonexistent tag (`v0.19.0rc1` in vllm-ascend) gives `MISSING` for every file, the same shape as a
  deleted requirement. `git rev-parse -q --verify <tag>` first; this false signal cost a recorded run a
  whole batch.

## Procedure — three reads, one batch

1. **Locate the registration in the transformers clone** (`/workspace/repos/huggingface/transformers`):

   ```bash
   git -C /workspace/repos/huggingface/transformers grep -n -F '"<type>", "' <ref> \
       -- src/transformers/models/auto/
   ```

   Search the **directory**, never one filename: the name table lives in
   `models/auto/configuration_auto.py` (`CONFIG_MAPPING_NAMES`) through v5.5 and in
   `models/auto/auto_mappings.py` (`("<type>", "<X>Config")` tuples, `:590+` at HEAD) from **v5.6.0**
   (`868e3c7391`, "Dynamic auto mapping" #45018, 2026-04-16). Search the *quoted* model type; a grep
   pinned to one filename at a newer ref returns *nothing* and reads exactly like "the model is
   unsupported" — two recorded runs tripped on precisely that empty result. `git ls-tree -r
   --name-only <tag> -- src/transformers/models | grep -i <type>` settles package existence without
   guessing a path.
2. **Date the support and find the first release**:

   ```bash
   git log -S'"<type>"' --format='%h %ad %s' --date=short -- src/transformers/models/auto/
   git tag --contains <sha> | grep -E '^v[0-9]' | sort -V | head -3
   ```

   Filter the tag list: non-release tags appear in `tag --contains` output on this clone
   (`backup-before-rebase` was first for the Qwen3.5 commit).
3. **Read the pin at the reporter's vllm-ascend tag**: `git show <ref>:requirements.txt | grep -i
   '^transformers'`, and on later tags `git grep -n -i transformers <ref> -- requirements*.txt
   pyproject.toml` — the line moves out of `requirements.txt` (`v0.18.1rc1`/`v0.19.0rc1` have none).

## Worked case #8315 (`v0.18.0rc1` image, Qwen3.5-27B-w8a8-mtp)

Reported: `quay.io/ascend/vllm-ascend:v0.18.0rc1`, ModelScope `Eco-Tech/Qwen3.5-27B-w8a8-mtp`, TP2 +
`qwen3_5_mtp` 3 spec tokens, Kylin V10, CANN 8.5.1; `KeyError: 'qwen3_5'` → `ValueError … model type
'qwen3_5' …` at renderer/tokenizer construction, before weight load.

| fact | evidence |
| --- | --- |
| `qwen3_5` registered by `fc91372258` (2026-02-09, "Adding Support for Qwen3.5", #43830); first release tag `v5.2.0` (entry at `configuration_auto.py:365` there) | transformers clone, `log -S` + `tag --contains` |
| `requirements.txt` at **both** `v0.18.0rc1` and `v0.18.0` says `transformers>=4.57.4` | vllm-ascend clone |
| transformers `v4.57.6` (the reporter's version) is 2026-01-16 — satisfies that floor and predates the model commit | `git -C $R log -1 --date=short v4.57.6` |
| upstream vLLM `v0.18.1`, `v0.19.0`, `v0.19.0rc1` all pin `transformers >= 4.56.0, < 5`, so no 5.x install is legal there | `git show <tag>:requirements/common.txt` |
| first vllm-ascend tag pinning 5.x is `v0.19.1rc1` (`transformers==5.5.3`, `cb18e84e6` #8448); its vLLM `v0.19.1` relaxes the cap to `>=4.56.0, != 5.0.* … != 5.5.0` | both clones |
| thread: `MengqingCao` "It seems you're using a outdated `transformers`"; reporter 「没有更改transformers的版本，容器内 4.57.6」; `AJF-cmd` "same image with the local Qwen3.5-27B-w8a8-**org** weights … did not encounter any errors"; closed "resolved in the latest version. Please use the latest release or rc version" | issue comments |

## Answer shape

- **workaround**: move to the release/image whose pin carries the model's floor (`v0.19.1rc1`+,
  transformers 5.5.3); the in-container experiment is `pip install -U transformers==<first-tag>`, but say
  why it is not the durable fix — vLLM's own cap at the paired version (`transformers >= 4.56.0, < 5` at
  vLLM v0.18.1/v0.19.0) makes a 5.x pin violate the vLLM requirement there. Serving a checkpoint whose
  `model_type` the installed release knows is the third option. Cost: a version move, and the accuracy
  baseline changes with it.
- **root cause**: the *pin*, not a bug — `vllm-ascend requirements.txt:<line> at <tag>` says `>=X` while
  the checkpoint needs `>=<first tag with the registration>`, and the image resolved to a release older
  than the model commit. Name the transformers fix commit and first tag, and the first vllm-ascend
  release that pins it.
- **status**: resolvable by image/release, no code change in vllm-ascend.
- **falsifier**: `pip show transformers` plus the served `config.json`'s `model_type` inside the
  reporter's container — a 5.x transformer with the same message means the checkpoint, not the floor.
  The #8315 thread's third comment is the useful control: same image, a different export of the model
  started fine ⇒ also compare the two `config.json` files (`model_type`, `architectures`), `hypothesis`
  with that comparison as its falsifier.

## The tokenizer twin: `Tokenizer class <X> does not exist or is not currently imported` (#8213)

Same report shape, same three-pin answer, different value: the frontend builds the renderer, calls
`AutoTokenizer.from_pretrained`, and transformers `models/auto/tokenization_auto.py` resolves the
`tokenizer_class` **string from the checkpoint's `tokenizer_config.json`** with
`tokenizer_class_from_name(...)`; `None` raises at `:1154` with
`Tokenizer class <candidate> does not exist or is not currently imported.` (line at `v4.57.6`). The frame
chain is the `model_type` one above (`api_server.py` → `AsyncLLM.from_vllm_config` →
`renderer_from_config` → `vllm/renderers/hf.py` → `vllm/tokenizers/hf.py:85` →
`AutoTokenizer.from_pretrained`), so nothing in `vllm_ascend/` owns it and no NPU op ran.

- **The value is the checkpoint's, not vLLM's.** A checkpoint exported against transformers 5 names a v5
  tokenizer class; nothing on the serving side can rewrite it, and `--tokenizer <other dir>` is the only
  server-side lever (point it at a directory whose `tokenizer_config.json` names a 4.x class).
- **The raise is transformers' own candidate resolution**, in the `elif config_tokenizer_class is not
  None:` branch of `tokenization_auto.py`: it tries `f"{cls}Fast"` first, then the plain name through
  `tokenizer_class_from_name`, then raises. The candidate comes from the checkpoint's `tokenizer_class`,
  not from a v4 mapping, so an absent class is the whole failure.
- **Where the class lives.** `TokenizersBackend` exists only in the v5 line: introduced by `05c0e1d390`
  ("rm slow tokenizers", #40936, 2025-11-27, first tag `v5.0.0rc0`); the auto-tokenizer map gains its
  model entries in `73a13f86f6` ("Refactor-tokenization-more", #42563, `v5.0.0rc1`), which is why a v5
  export records it (`("glm", "TokenizersBackend" …)` at `models/auto/tokenization_auto.py:133+` at
  `v5.0.0`, `"minimax_m2"` at `:365` at `v5.5.3`). At `v5.5.3` the class itself is
  `src/transformers/tokenization_utils_tokenizers.py:84` (`class TokenizersBackend(PreTrainedTokenizerBase)`).
  `git grep -c TokenizersBackend v4.57.6 -- src/transformers` returns **no** hit, which is the whole
  mechanism. Grep the symbol, not the sentence.
- **Run the same three-pin check.** Repo floor: `vllm-ascend` `requirements.txt` says
  `transformers>=4.57.4` at `v0.17.0rc1` (`:37`), `v0.18.0rc1` (`:38`) and `v0.18.0` (`:42`) — a floor a
  4.x release satisfies. Upstream cap at the paired tag: vLLM `v0.18.0`/`v0.19.0` pin
  `transformers >= 4.56.0, < 5`, so **no 5.x install is legal there**; `v0.20.0`/`v0.22.0` change it to
  `>= 4.56.0, != 5.0.* … != 5.5.0`, `v0.26.0` to `>= 5.5.3`, and `main` to `>= 5.16.1, < 5.23.0`. vllm-ascend
  itself pins `transformers==5.19.0` on `main`.
- **Third option, untested** — point `tokenizer_class` in the checkpoint's `tokenizer_config.json` at a
  class the installed release has (`PreTrainedTokenizerFast` keeps the same `tokenizer.json` backend).
  `hypothesis` until the reporter runs it.

### Worked case #8213 — MiniMax-M2.7 on the A3 `v0.17.0rc1`/`v0.18.0rc1` images

Reported: `quay.io/ascend/vllm-ascend:v0.17.0rc1-a3-openeuler` and `v0.18.0rc1-a3-openeuler`, A3 single
node, bf16 `MiniMax-M2.7` with `--trust-remote-code`, TP8 × DP2, `FULL_DECODE_ONLY`,
`enable_cpu_binding`, `transformers==4.57.6`; every `ApiServer` process dies during renderer construction
with `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.` (the
engine banner is `V1 LLM engine (v0.17.0)`, so the image, not the tag, is the ref).

| fact | evidence |
| --- | --- |
| the failing value is the checkpoint's `tokenizer_class` (`TokenizersBackend`), resolved by transformers, not by vLLM | reporter's traceback: `vllm/tokenizers/hf.py:85` → `AutoTokenizer.from_pretrained` → `tokenization_auto.py:1153-1154` (their build prints `:1153`; `:1154` is the raise at `v4.57.6`) |
| no 4.x transformers carries the class; v5.0.0+ does (`tokenization_utils_tokenizers.py:84` at `v5.5.3`) | transformers clone, `grep -c` / `git tag --contains` |
| vLLM at the reported pair caps `transformers < 5`, so the image's 4.57.6 is "correct" for the engine | `git show v0.18.0:requirements/common.txt` |
| reporter-confirmed fix: `pip install transformers==5.5.3` inside the `v0.18.0rc1-a3` container, then the service starts (`v0.18.0rc1` only; not tried on the 0.17 image) | thread, `yang0754` 2026-04-22, then 「问题已解决」 |
| 5.5.3 is the version the project itself moved to (`v0.19.1rc1` pins `transformers==5.5.3`, #8448), and vLLM's floor reaches `>= 5.5.3` at `v0.26.0` | both clones |
| the same thread's other lever is about the checkpoint, not the class: `ultra3g` says fp8 MiniMax-M2.7 is unsupported and points at `Eco-Tech/MiniMax-M2.7-w8a8-QuaRot` plus the `main` image | thread, `[reported]` |

**Answer shape** for this member: workaround = a release/image whose transformers is v5 (or the in-container
`pip install transformers==5.5.3` for a test, saying that it violates the paired vLLM's `< 5` cap), or a
checkpoint whose tokenizer files declare a 4.x-compatible class; root cause = the pin floor, with the class
symbol, its first transformers tag and the tag's `requirements.txt` line (the image resolved 4.57.6 because
the pin is a `>=` floor); status = answered by image/release/checkpoint choice, no vllm-ascend code change;
falsifier = `pip show transformers` plus the served `tokenizer_config.json`'s `tokenizer_class` — a 5.x
transformers with the same message means the checkpoint or the `--tokenizer` path, not the floor:

```bash
python -c "import json;print(json.load(open('<model>/tokenizer_config.json'))['tokenizer_class'])"
```

## The processor twin: `TypeError: Glm4vProcessor.__init__() got multiple values for argument 'tokenizer'` (#1768)

Same dependency floor, one layer later: `config.json` loads, the tokenizer builds, and the **processor**
constructor dies while `vllm serve` starts — `TypeError: Glm4vProcessor.__init__() got multiple values for
argument 'tokenizer'`, frames through `transformers/models/auto/processing_auto.py` →
`ProcessorMixin.from_args_and_dict`. Nothing in `vllm_ascend/` is in the traceback.

Worked report (#1768, `v0.9.2rc1` image, 1×910B4): `GLM-4.1V-9B-Thinking` from a local path, transformers
installed from a **source tree** (`/workspace/transformers/src/transformers/...`), and the model's dummy-input
profiling step dies — `glm4_1v.py:1005` `get_dummy_text` → `multimodal/processing.py:1075` →
`inputs/registry.py:138`:

```text
File ".../transformers/processing_utils.py", line 1304, in from_pretrained
    return cls.from_args_and_dict(args, processor_dict, **kwargs)
File ".../transformers/processing_utils.py", line 1105, in from_args_and_dict
    processor = cls(*args, **valid_kwargs)
TypeError: Glm4vProcessor.__init__() got multiple values for argument 'tokenizer'
```

**Two release facts, the second causing the symptom.** `glm4v` entered transformers with `af9870265e`
("GLM-4.1V Model support", #38431, 2025-06-25), whose first tag is **`v4.53.0`** — `git cat-file -e
v4.52.0:src/transformers/models/glm4v/processing_glm4v.py` fails, so on anything older the class does not
exist at all — and `v4.53.0`'s own `from_args_and_dict` double-binds `tokenizer`. Verified in the two clones:

1. **Identify the installed transformers by its line numbers, not by `pip show`.** At transformers `v4.53.0`
   those two anchors are exactly `processing_utils.py:1105` (`processor = cls(*args, **valid_kwargs)`) and
   `:1304` (`return cls.from_args_and_dict(...)`); `models/auto/processing_auto.py:379` is
   `return processor_class.from_pretrained(` in the `elif processor_class is not None:` branch. A one-command
   check across tags (`git show <tag>:src/transformers/processing_utils.py | sed -n '1105p'`) pins the
   release; the reporter's own `pip show` is the other half.
2. **The v4.53.0 defect — why the name arrives twice.** `from_args_and_dict` builds positional `args` from
   `cls.attributes` and then *deletes* the ones whose names also appear in `processor_dict` —
   `args_to_remove = [i for i, arg in enumerate(accepted_args_and_kwargs) if arg in processor_dict]`,
   `args = [arg for i, arg in enumerate(args) if i not in args_to_remove]` (`:1100-1102`) — so removing
   `tokenizer` (index 1) shifts `video_processor` into that slot **and** `tokenizer` is still passed as a
   keyword: duplicate `tokenizer`. At `v4.53.0`/`v4.54.0` the class is
   `class Glm4vProcessor(ProcessorMixin)` with `attributes = ["image_processor", "tokenizer",
   "video_processor"]` and `def __init__(self, image_processor=None, tokenizer=None, video_processor=None,
   chat_template=None, **kwargs)` (`models/glm4v/processing_glm4v.py:66-75` / `:68-75`) — note the **third
   positional slot** a `video_processor` occupies. The keyword comes from vLLM's own caller (step 3), which
   is what makes this repairable by a pin.
3. **vLLM already carries the workaround, keyed on an exact version string.** At vLLM `v0.9.2`,
   `InputProcessingContext.get_hf_processor` (`vllm/inputs/registry.py:119-141`) does
   `if Version(TRANSFORMERS_VERSION) != Version("4.53.0"): kwargs["tokenizer"] = self.tokenizer` with the
   comment "Transformers 4.53.0 has issue with passing tokenizer to initialize processor … See
   vllm-project/vllm#20224" (fix `e936e401de`, #20244). A **`4.53.0.dev0`/`rc`/source build compares
   unequal**, so the guard does not fire and the kwarg that triggers the bug is passed. That is why a
   reporter can match 4.53.0's line numbers and still fail.
4. **The upstream fix:** `a10c2bf840` "Fix multimodal processor get duplicate arguments when receive kwargs
   for initialization" (#39125) replaces the removal with an index-preserving substitution
   (`args_to_update = {i: valid_kwargs.pop(arg) …}`, `args = [arg if i not in args_to_update else …]`),
   first tag **`v4.53.1`**.

Answer: it is a transformers-version skew on the host, not an Ascend or model defect. Workaround:
`pip install -U 'transformers>=4.53.1'` (reporter-confirmed: 「升级transfromer版本后已可以成功运行模型」 with
an identical `vllm_api_request.py` call) or, if the image must keep a 4.53.0 tree, make the version string
compare equal to `4.53.0` so vLLM's guard fires. Check which transformers is actually importable — a
source/editable tree under `/workspace/...` shadows a pip upgrade, which is exactly how this report
survived its own `pip install --upgrade transformers` line.

- **The 310P twin in the same thread is a different, unhappier answer.** 300I Duo, `v0.9.2rc1-310p`,
  `pip install 'transformers<4.54.0'`: it dies later, in ACL-graph capture
  (`status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE INTERNAL ASSERT FAILED` inside
  `compile_or_warm_up_model`), and the reporter's own conclusion is 「应该是不支持」; a later commenter asks
  which version to downgrade to. So: 910B = upgrade transformers; 310P at that tag = no known-working pin,
  say so (`_310p` at that tag has no equivalent path; `aclgraph-mode-and-capture.md`).
- **Do not derive the processor internals.** A recorded run took this thread's lever in batch 1 and then
  spent **12.8 K characters of reasoning in one message** — the whole output budget — reconstructing
  `from_args_and_dict`, `attributes` ordering and the `4.51/4.52/4.53` signatures, was cut, and shipped
  nothing. The two files the traceback names are both in the clone: two `git show <tag>:<path> | sed -n
  'A,Bp'` reads settle it, and the rest is a `limits` line.
- Check order: the model's add commit and `git tag --contains <sha>` (`git log --diff-filter=A --
  src/transformers/models/<m>/processing_<m>.py`), the installed release's own `from_args_and_dict`, the
  caller's version guard at the paired vLLM tag, then the fork tag's `requirements.txt`/`requirements/*.txt`
  floor (a `>=` floor encodes no model floor) and the image's installed version.

## Twins — do not merge

- The same `ValueError` where the **vLLM model registry**, not transformers, lacks the architecture is a
  model-support question for `vllm`/`vllm-ascend`: check `models/` at the ref before blaming
  transformers.
- `WARNING: method qwen3_5_mtp is deprecated and replaced with mtp` in the same log is a warning, not
  the error.
- A `Tokenizer class … does not exist` message is *not* the chat-template guard
  (`tokenizer-and-chat-template.md`): no request was served, the failure is at renderer construction, and
  the missing thing is a class in the installed transformers, not a template in the checkpoint.
- `KeyError: '<module>.weight'` / `weight_scale` is `quantization-load.md`, not this page.
- A processor `TypeError` whose message names an argument (`got multiple values for argument 'tokenizer'`)
  is the same dependency-floor *class* but a different owner: the fix is in transformers (or in the version
  string vLLM compares), so do not search `vllm_ascend/` for it.

## Recheck conditions

The auto-mapping layout changed once inside this window (v5.6.0) and the pin file moves between
`requirements.txt`, `requirements/*.txt` and `pyproject.toml`. Re-run step 1 on the directory and step 3
at the reporter's own tag; do not reuse the numbers above without them. If the three versions do not
intersect, the deliverable is the release that makes them intersect (find the pin change with
`git log -S'transformers==' -- requirements.txt`), plus the labelled statement of which version the
reporter's container holds.
