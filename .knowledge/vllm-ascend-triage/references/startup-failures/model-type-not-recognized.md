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

The same crash also arrives one wrapper higher, with **no** `[registry.py]` line at
all (recorded #4206, image `v0.11.0rc0`, Qwen3-Next-80B-A3B, body in Chinese):

```
(APIServer pid=1)   File "/vllm-workspace/vllm/vllm/engine/arg_utils.py", line 994, in create_model_config
(APIServer pid=1) pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
(APIServer pid=1)   Value error, Model architectures ['Qwen3NextForCausalLM'] failed to be inspected. Please check the logs for more details.
```

The message text keeps it out of the gate above: `failed to be inspected` is raised
only for an architecture that **is** registered (`_raise_for_unsupported`, the
unregistered text is `are not supported for now`), while the `Value error, ` prefix
and `ValidationError for ModelConfig` are config validation wrapping it. So this is
not a transformers/vllm version answer either — the reporter cut the `[registry.py]`
lines, and the one-command repro below is the answer to send.

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
  main-branch checkout whose `qwen3_next.py` imported it. At the pinned
  `v0.11.0rc1` (`c5fe179`) that import is gone and no `vllm_ascend/models/
  interfaces.py` exists at all (`ls vllm_ascend/models/`), while `qwen3_next.py`
  imports **only** upstream `vllm.*` names — but the export is one commit
  (`git rev-list --count HEAD` = 1), so what the reporter's `v0.11.0rc0` image holds
  is not checkable here (stop rule below).
- A **missing** triton is not what #2889 shows: vllm substitutes placeholders when
  `find_spec("triton")` fails (`vllm/triton_utils/__init__.py:7-14`,
  `triton_utils/importing.py:14-17`) and the release image removes the wheel on
  purpose (`Dockerfile:42-44`, "triton doesn't work correctly" on Ascend). An
  `AttributeError` on `triton.language` therefore means a triton **is** installed and
  it is the wrong one (the triton-ascend pairing). The Qwen3-Next tutorial pulls the
  other way and is a distractor here: its mandatory "The Triton Ascend is required
  when you run Qwen3 Next" step plus the `Ascend-BiSheng-toolkit` install
  (`docs/source/tutorials/multi_npu_qwen3_next.md:41-72` at v0.11.0rc1) is a
  **kernel-compile** prerequisite for a serving engine, not an import-time one — an
  *absent* triton renders placeholders and cannot raise, so it cannot produce this
  signature. #4206 ranked a missing triton first on the strength of that page, then
  refuted it four turns later against `find_spec`, with the reproducing one-liner
  already in hand.
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

### Stop rule — this signature is one ask, not a mechanism

Three facts settle everything the local snapshot can settle: the arch is registered,
inspection is a subprocess import, and the inner traceback — which only the
reporter's log has — names the failing import. So deliver in one message: the
classification (not the gate, not a defect in the architecture), the
`python -c "import vllm_ascend.models.<file>"` line, the two side facts
(`pip show vllm` to test "the installed vllm is not the tag", `uname -m` /
`pip show triton triton-ascend` for the pair), and the request for the
`[registry.py]` lines — then list the recorded inners as candidate classes (wrong
triton pair → #2889's `tl.exp`; a downstream module the image lacks → #2929) without
choosing between them. The mistake to avoid is reconstructing the *reporter's
image* from the pinned tag: #4206 did exactly that after correctly
refuting the triton lane, spent 37 messages and its whole output budget on it, and
shipped **no answer at all**. The cut tail is obtainable; the predecessor image's
file content is not.

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
   `>= 4.56.0` excluding `5.0.*`–`5.5.0`. A `< 5` cap settles "is this architecture
   loadable with the image's own stack" (`v0.18.0rc1`: `requirements.txt:38`
   `transformers>=4.57.4`; vllm `requirements/common.txt:10` `< 5`) — quote both lines.

   **It does not settle "is the reporter's off-pin transformers the bug".** Ask that
   second question before ranking, because a family the release documents can *require*
   the version the cap excludes, and a report that volunteers its `transformers` number
   is often quoting an install the project itself told it to make. The pinned tree's own
   `docs/source/user_guide/release_notes.md` `### Known Issue` block is the one read
   (`rg -n -i 'requires|known issue' <pin>/docs/source/user_guide/release_notes.md`):
   at vllm-ascend `v0.18.0rc1` (`99e1ea0f`) `release_notes.md:52`, under `## v0.17.0rc1`,
   says "GLM5 requires transformers==5.2.0, and this will resolved by
   vllm-project/vllm#30566, will not included in v0.17.0" — so on that release a GLM-5
   launch on `transformers` 5.2.0 is the **documented** pairing and "restore the image's
   4.57.x" would break the model, not fix it. Corroborating code read: vllm v0.18.0
   `vllm/transformers_utils/config.py:380` branches on
   `Version(version("transformers")) < Version("5.0.0")` ("Transformers v4 installed,
   legacy config fields …", the comment above says the patching is needed "for both
   Transformers v4 and v5"), so the `requirements` cap is a support statement, not a
   runtime refusal, and 5.x has a code path at that pin.

   Two things to say in the answer, never assume: whether the bulletin row still applies
   to the reporter's *own* release (fixes land by PR id, and `#30566` is unreadable here),
   and that the release-notes section for the reporter's release may not exist yet — at
   `99e1ea0f` the newest section is `## v0.17.0rc1` (`:3`), so absence of a row for
   `v0.18.0rc1` is docs lag, not a clean bill. The check that discriminates instead is
   the traceback: with 5.2.0 installed the failing frame names the code that broke, and
   the same launch on the image's own `transformers` is the one-variable ablation.
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

**The startup `already registered … will be overwritten by the new model class`
warning is not evidence of support.** `vllm_ascend/models/__init__.py` registers its
overrides unconditionally inside `register_model()` (`Qwen3VLForConditionalGeneration
→ …qwen2_5_vl_without_padding:AscendQwen3VLForConditionalGeneration`, `:11-19` at
v0.11.0rc0 `00ba071`), while the module it names guards the upstream import and
substitutes placeholders — `try: from transformers.models.qwen3_vl.configuration_qwen3_vl
import Qwen3VLConfig … / except ImportError: pass` (`:30-36`) and, for the model
classes, `try: from vllm.model_executor.models.qwen3_vl import … / except ImportError:`
followed by `Qwen3VLForConditionalGeneration = object` and eight siblings (`:58-66`).
So the warning prints on an image whose transformers predates the architecture, the
lane is still gate 1, and a reader must not quote that line as "the image supports
Qwen3-VL". It also means the hollow-`object` base is what survives if an off-pin
config path ever gets past gate 1 — the next failure is then an inspection
`ImportError`, not a numerics bug.

**Scope of that sentence: an *architecture* gate.** When the vllm tree registers the
model and only the **processor** postdates the pin, the upgrade is the answer, not the
mistake — record it as such. #1768: vllm v0.9.2 has `Glm4vForConditionalGeneration`
(`registry.py:196`) and the model file, while the image's `transformers==4.52.4`
(`requirements.txt:30`) predates the family (glm4v is documented at transformers v4.53.0; 4.52.4 is not in the export); the
reporter's `pip install --upgrade transformers` was necessary, one further upgrade took it
past a `TypeError: Glm4vProcessor.__init__() got multiple values for argument 'tokenizer'`,
and the model then served on the same 910B4. So decide which of the three gates the paste
actually names before saying "off-pin": a missing **processor** is not a missing
architecture (`hf-processor-contract-at-mm-budget.md` §0 carries the direction rule).

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

## The removed-architecture family: `was supported in vLLM until vX`

Recorded #14855 (title: "Baichuan-7B拉起服务失败报错：Model architecture BaiChuanForCausalLM was
supported in vLLM until v0.23.0, and is not supported anymore"), pins vllm-ascend `1f95052cb43a`
and vllm `v0.26.0` `568afb3a1380` — the reporter's own pair. Nothing was removed by
vllm-ascend: **upstream vLLM deleted the model**, and the sentence itself names the last release
that had it.

Read the sentence before opening any tree — `_raise_for_unsupported`
(`vllm/model_executor/models/registry.py:1077` at v0.26.0) emits exactly four texts, and they
have different owners:

| text | meaning |
|---|---|
| `Model architectures [...] failed to be inspected. Please check the logs` | registered here, import crashed — the look-alike above, not this family |
| `Model architecture X was supported in vLLM until vY, and is not supported anymore` | in `_PREVIOUSLY_SUPPORTED_MODELS` (`:724-762`; v0.26.0 rows include `BaiChuanForCausalLM`/`BaichuanForCausalLM` → `0.23.0`, `AquilaModel` → `0.24.0`, `TeleChatForCausalLM` → `0.25.0`) |
| `Model architecture X is not supported in-tree anymore. Please install the plugin at <url>` | in `_OOT_SUPPORTED_MODELS` (`:764-769` — only `BartModel`, `BartForConditionalGeneration`, `Florence2ForConditionalGeneration`, `MBartForConditionalGeneration` at this pin) |
| `Model architectures [...] are not supported for now. Supported architectures: [...]` | unknown architecture, the original gate above |

Reporter-side one-liner (their install, not the pin):

```bash
python -c "from vllm.model_executor.models.registry import _PREVIOUSLY_SUPPORTED_MODELS as t; print(t.get('<Arch>'))"
python -c "from vllm import ModelRegistry; print([a for a in ModelRegistry.get_supported_archs() if '<substr>' in a])"
```

A non-`None` first answer makes the removal upstream's, and the second tells the reporter whether
any sibling in the family survived (`Baichuan-M1`/`M2` are different architectures and are not in
the table — do not hand them over as a workaround without that check).

**The message's own advice is half an answer on Ascend.** "Please use an older version of vLLM"
is vLLM's sentence; vllm-ascend pins vllm per release, so a vllm-only downgrade breaks the
plugin. Answer with the **pair** taken from the pinned `docs/source/community/versioning_policy.md`
(*Release compatibility matrix*): find the row whose vLLM column is the last supporting version
and read the vllm-ascend tag beside it (at `1f95052cb` the top row is `| v0.23.0rc1 | v0.23.0 | …`).
Two honesty notes: that matrix describes **releases**, so a paste whose version is a dev build
(`0.19.1rc2.dev1369+g1f95052cb`) is not a row — the `+g<sha>` suffix is the commit correspondence
(here the pin itself) and the `.devNNNN` distance is not a release; and a release that predates
the fix in question is invisible, so do not promise behaviour for a tag the snapshot cannot show.

**Why the message says "removed" instead of naming the real reason.** `inspect_model_cls`
(`registry.py:1218`) also tries the Transformers backend under `model_impl` `auto`/`transformers`
via `_try_resolve_transformers` (`:1122-1192`), which reads the checkpoint's `auto_map` and
returns `_get_transformers_backend_cls()` (`vllm/config/model.py:811-826`, i.e.
`TransformersForCausalLM` in `_TRANSFORMERS_BACKEND_MODELS`, `registry.py:670-682`). Its two
refusal paths return `None` **silently** under `auto` and `raise` only under
`model_impl == "transformers"`:
- no usable model class (transformers lacks the name **and** no `AutoModel*` entry in the
  checkpoint's `auto_map` resolved, `:1169-1178`) → `ValueError: Cannot find model module. … is
  not a registered model in the Transformers library … and 'AutoModel' is not present in the
  model config's 'auto_map'`;
- a class was found but `not model_module.is_backend_compatible()` (`:1181-1186`) →
  `ValueError: The Transformers implementation of '<Arch>' is not compatible with vLLM.`

So the removed-architecture sentence is already evidence that the fallback declined. Re-running
the reporter's line with `--model-impl transformers --trust-remote-code` replaces that silence
with one of the two texts above — the first means the checkpoint ships no usable `auto_map` (or
its remote code failed to import, which `warn_on_fail=True` logs as a warning), the second means
it loaded and vLLM refused the class. One launch, and it is the cheapest discriminating check.
Read `config.json`'s `architectures` + `auto_map` beside it, not instead of it. Coverage caveat for this rung: at
vllm-ascend `1f95052cb` a grep for `model_impl`/`TransformersForCausalLM` over the tree (code and
`tests/`) is **empty**, so the backend is unvalidated on NPU — promise the diagnostic, not the
outcome.

**Keeping the newer vllm: register the architecture back.** `ModelRegistry.register_model("<Arch>",
"<module>:<Class>")` from a `vllm.general_plugins` entry point (`vllm/plugins/__init__.py:18,77`)
is the same mechanism vllm-ascend uses for its own models (`vllm_ascend/models/__init__.py:1-31`
at this pin — Kimi-K3, DeepSeek-V4, MiniMax-M3, the drafter classes). `_raise_for_unsupported`
is only reached when no class resolves, so a registration short-circuits it. But a registration
is not a port: the removed file has to be adapted to the installed attention/linear API, and the
acceptance is one real request compared against a known-good output — never
`Application startup complete`.

**The claim surface to check and report.** The support matrix can still claim the family while
the paired upstream has no implementation at all: at `1f95052cb`,
`docs/source/user_guide/support_matrix/supported_models.md:69-70` carries
`| Baichuan | 🔵 | … | A2/A3 |` and `| Baichuan2 | 🔵 | …`, while `ls
<vllm root>/vllm/model_executor/models/ | grep -i baichuan` is empty at `568afb3a`. A 🔵
("experimental supported") row over a deleted architecture sends the next reporter down the same
dead end — quote the row, name the tracking issue from the file header, and ask for it to be
flipped with the last-supported version in the note.

Pitfalls specific to this family:

- Do **not** answer "upgrade transformers", and do not date transformers from the export: the
  pinned `huggingface/transformers` is a `content_export` (`1515/5776` files) and contains no
  `src/transformers/models/` at all (rule 3 of `../upstream-downstream-drift/index.md`).
- Do not treat a `🔵`/`Need test` matrix cell as a capability verdict for a model whose
  implementation is gone; the registry table at the *paired* upstream ref is the fact.
- `--enforce-eager`, `--block-size`, `--max-model-len` and every device-side check are
  irrelevant here: the raise is in `ModelConfig` validation, before a worker or an NPU exists,
  so no plog is owed.

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
