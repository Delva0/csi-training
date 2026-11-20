# Startup, install and environment cases

Engine-start failures (silent stall, worker exit, metadata/capture phase), image and
package faults (`libatb.so`, ABI split, pip index visibility, triton-ascend pair,
published artefacts), SoC and version gates, connector config attributes, tutorial
document cases and `[Community Test]` reports. Routes: `../startup-failures/index.md`,
`../image-dependency-faults/index.md`, `../soc-platform-paths/index.md`,
`../tutorial-verification/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

**#1784** vllm-ascend v0.9.2rc1 (`b5b7e0e`) + vllm v0.9.2 (`a5dd03c`), "installing
vllm-ascend 0.9.1dev and `pip show vllm-ascend` shows 0.9.0rc2": packaging metadata,
no lane — `setup.py:305-311` derives the version with `setuptools_scm` from the build
tree's tags (`LookupError → "0.0.0"`), `vllm_ascend/_version.py` is gitignored, no
`vllm_ascend/` code reads its own version (`utils.py:283-295` `vllm_version_is()`
reads `vllm.__version__` / `VLLM_VERSION`), and `-dev` is a branch name, not a pip
version. Answered on the last turn over 13 tool calls, ranking stale dist-info →
wrong checkout → tag/branch expectation; the recorded gap is the snapshot's own
one-commit export with no tags, so the tag-ancestry half is the reporter's
`git describe` to run, not a local check. Route now
`../image-dependency-faults/installed-version-vs-checkout.md`.

**#7352** v0.14.0rc1 (still broken at v0.17.0rc1) DeepSeek-R1-Distill-Qwen-1.5B
1P1D Mooncake PD, 910B4: prefill came up while the decode pod died in
`register_kv_caches` → `KVCacheRecvingThread.__init__` with
`AttributeError: 'Qwen2Config' object has no attribute 'head_dim'` (a flat config
whose `head_dim` is derived, not a nested-VL config — `hf_text_config` was already
in use), and the same reporter's next round hit
`ImportError: ascend_transport.so` that turned out to be the launcher's effective
`LD_LIBRARY_PATH` rather than the image; both lanes are now in
`references/startup-failures/connector-config-attributes.md` and
`references/image-dependency-faults/launch-env-and-library-path.md`;

**#1144** vllm-ascend v0.7.3.post1 (`c69ceac`) / vllm v0.7.3, torch+torch_npu 2.5.1,
aarch64 EulerOS: `import mindie_turbo` →
`ImportError: …/mindie_turbo/_ops.cpython-310-aarch64-linux-gnu.so: undefined
symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs`, while the env dump selects
`ATB_HOME_PATH=…/nnal/atb/latest/atb/cxx_abi_1` — a `std::string` ABI mismatch
between a prebuilt wheel and the selected NNAL/ATB root. Answered on its 9th
assistant turn with the demangled symbol, the three `nm -D` checks and the
`cxx_abi_0` export, and the
**reporter confirmed on 2025-06-10: "Thanks, the env setting works for me."** The
run also pinned the impact split (worker `try_register_lib` warning vs
`NotImplementedError: There is no available ascend quantizer` on the quantised
path); the weak spot was guessing which ABI torch_npu's own `libop_plugin_atb.so`
needs instead of asking for the one `nm -D` count, which its answer then correctly
listed as the deciding check (`../image-dependency-faults/abi-undefined-symbol.md`).

**#1696** vllm-ascend `v0.7.3` (`779eebb8`) + vllm v0.7.3, image
`quay.io/ascend/vllm-ascend:v0.7.3`, host CANN 8.1.RC1, an **Ascend 910C1** card
(`npu-smi` name), `LLM("Qwen/Qwen2.5-0.5B-Instruct")`: model construction dies in
upstream `rotary_embedding.py:109 _compute_inv_freq` → `torch.arange` →
`RuntimeError: call aclnnArange failed … EZ9999 … Parse dynamic kernel config fail /
AclOpKernelInit failed opType / ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed`, no
`vllm_ascend` frame; the reporter's question was literally "does this image support
910C?". Answerable from docs in four calls (`Dockerfile:16` base
`cann:8.1.rc1-910b`, `docs/source/conf.py:78`, `faqs.md:14-17` "ONLY Atlas A2",
`installation.md:87-89` the `-910b` kernels run) plus one standalone op probe, and
the run also disposed of the `Failed to register custom ops` /
`No module named vllm_ascend.vllm_ascend_C` red herring (`setup.py:296-298`,
`platform.py:29-38`). Answered in full on its 13th assistant turn, with the verdict already in the
seventh turn's message text; the
sibling of #11519's `EZ1001` on the same axis — built-in op here, custom op there
(`../soc-platform-paths/kernel-package-vs-card.md`).

**#2178** vllm-ascend `v0.9.2rc1` (`b5b7e0ec`), a 310P host on `linux/arm64/v8`:
"[Usage]: 310p的镜像找不到linux on arm64的" — the quay tag page
(`…?tab=tags&tag=latest`) looked like it had arm64, the second page was the
third-party mirror `docker.aityp.com/image/quay.io/ascend/vllm-ascend:v0.9.2rc1-310p-openeuler`,
and the pull printed the client's platform-mismatch warning. Track: published-artefact
availability (image/wheel pipeline), not a vllm-ascend defect. The recorded run took
8 assistant turns with **zero message text until the final one** — it had no route
for this shape and re-derived the workflow reading by hand — then answered with the
right frame for the old layout: 310P tags carry `-310p` / `-310p-openeuler` and the
bare tags (incl. `latest`) are the generic A2 line
(`image_310p_ubuntu.yml:69-71`, `image_ubuntu.yml:69-71`); `platforms:` is
`amd64,arm64` only on `push` by the upstream owner and otherwise the two 310P
workflows fall back to *different* single arches (`:96-100` vs
`image_310p_openeuler.yml:100-104`, whose runner also flips to `ubuntu-24.04-arm`,
`:43-47`); the arm64 leg needs an arm64 CANN base (`Dockerfile.310p:18`); the
mirror is not evidence. Two refinements the page carries that the answer stated too
loosely: `push:` and `platforms:` are the *same* condition, so a build allowed to
publish is always requested multi-arch and the single-arch fallbacks cannot explain
a published tag (the arm64 leg's base image or the client/mirror can); and a
non-prerelease release does get a suffixed `latest-310p`, only `latest` itself is
never a 310P tag. What it left open correctly: the actual manifest lists,
i.e. `docker buildx imagetools inspect`, plus the `-310p` base's own arch. One
addition the run found and the library now carries: `release_whl.yml:46-47` builds
on `ubuntu-24.04-arm` too (an aarch64 wheel exists), while
`.github/Dockerfile.buildwheel:18` is a **910b** base and `SOC_VERSION` is baked in,
so that wheel is an A2 build — and #1742's v0.9.2rc1 checklist still had
"Upload 310p wheel to Github release page" unchecked. Route:
`../image-dependency-faults/image-arch-and-tag-availability.md` (one call:
`scripts/image-arch-matrix.sh`).

**#11759** vllm-ascend `v0.18.0` image, community-test row for
yxr2_qwen3-32B (Qwen3-32B-W4A4 on 1×Atlas 800I A2, HDK 25.5.0.B078), i.e. the same
task shape as #11739 a batch later and the same outcome: **no report file and no
message text**, cut at the 16,384-token cap on its 16th turn while writing the
report heredoc. Every section of the answer existed before that: the absent
"Run basic benchmark" section by turn 6, the curl's ignored
`max_completion_tokens` (`/v1/completions` has only `max_tokens`, default 16, at
vllm v0.18.0) by turn 8, the sizing pass by turn 11 (16.4 GB of W4A4 weights
against a 57.6 GB per-card budget, so one card is right and bf16 would not fit),
and the `pip install transformers==4.51.0` line against the image's own
`>=4.56`/`>=4.57.4` pins by turn 14. See
`../tutorial-verification/community-test-report.md` for the page's verified
defect set and the write-granularity rule this run cost.

**#18130** vllm-ascend `v0.27.1rc1` (`3b31886`) / vllm `v0.27.1` (`6e448d0`),
GLM-5.3-w8a8c8 on 4×Atlas 800I A2 (910B2) multi-node co-located, command copied
from the v0.27.1rc1 GLM5.3 tutorial: `fuse_muls_add / Unexpected keyword argument
[type=unexpected_keyword_argument, input_value=True, input_type=bool]` at startup.
Answered in 17 turns with the whole mechanism: the key is nested-only
(`AscendCompilationConfig`, default `True`), the tutorial's serve command still
prints it flat although its own header says the scripts are based on v0.23.0, and
`AscendConfig` is `extra="forbid"` since the pydantic migration (#15418). The
reporter's later comment quotes exactly that tutorial page with that JSON, so the
document as the source is confirmed (whether deleting the key then started the
service is not reported). Lane worth copying: classify from the pydantic error *type* plus the
missing `WorkerProc`/`Starting to load model` lines, then one grep for the key's
owning class (`../startup-failures/additional-config-schema.md`).

**#8788** vllm-ascend `v0.13.0rc3` (`bbe9bc0`) + vllm `v0.13.0`; A3 16 cards,
"Cannot install from source in an offline pod: custom-ops step pulls dependencies
from the public network". Real finding (source-read, never delivered): the chain
`setup.py` `build_aclnn` → `csrc/build_aclnn.sh` → `csrc/build.sh --pkg` →
`csrc/CMakeLists.txt` fetches a `catlass` git submodule (always, since
`csrc/third_party` is gitignored) plus json/abseil-cpp/ascend_protobuf/makeself
from `gitcode.com`, each behind an existence gate under `CANN_3RD_LIB_PATH`;
`CANN_PKG_SERVER` reaches only two orphan cmake files, and `setup.py` never
forwards `CANN_3RD_LIB_PATH` to CMake. **Shipped no answer**: 30 turns, every
message a bare tool call, cut inside a block that had already written
"**Answer**: Yes, real…". Lane: name the fetch legs and the pre-seed directory
(`../image-dependency-faults/offline-source-build.md`).

**#1796** vllm-ascend `v0.9.2rc1` (`b5b7e0e`) + vllm `v0.9.2` (`a5dd03c`), install
report: "Unmatched torch version between vllm=0.9.2 and vllm-ascend=0.9.2rc1" —
`pip install vllm==0.9.2` brings torch 2.7.x, then `pip install vllm-ascend==0.9.2rc1`
ends in `could not find a version that satisfies the requirement
torch-npu==2.5.1.post1.dev20250619`; the reporter's later public comments add that a
manual `pip install torch-npu==2.5.1` uninstalls torch 2.7.1 and leaves
`vllm 0.9.2 requires torch==2.7.0 … but you have torch 2.5.1` warnings. Real finding
(source-read at the pin): `requirements.txt:25-27` holds `--pre` +
`--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi` +
`torch-npu==2.5.1.post1.dev20250619`, but `setup.py:331-351` skips every `--` line
(`:341`) when building `install_requires`, so the wheel names a mirror-only dev build
and pip can never see it; and the set is on the 2.5.x torch line
(`torchvision<0.21.0`, `:14`) against vllm's `torch==2.7.0` /
`torchvision==0.22.0` (`requirements/cuda.txt:9,12`), with torch_npu wheels hard-pinning
the matching torch (`Ascend/pytorch` `setup.py:625,650`). **Shipped no answer**: 39 tool
calls, every message a bare tool call, ~16 K completion tokens of reasoning cut at the
cap while speculating which torch that dev wheel was built against. Lane: pip index
visibility, not compatibility (`../image-dependency-faults/pip-index-visibility.md`,
`scripts/pip-pairing-check.sh`). Contrast pin: at `v0.11.0rc2` the same file is
`torch==2.7.1` + `torch-npu==2.7.1` with the mirror lines commented out — the version
prefixes agree.

**#4737** vllm-ascend image `v0.11.0rc2`/`0.11.0rc3` (tree `a2e4c3f`) + vllm `v0.11.0`
(`b8b302c`), Qwen3-32B-W8A8: `vllm bench serve --dataset-name custom` dies with
`ImportError: Please install vllm[bench] for bench support`, so the image "cannot run
vllm bench offline". Cause (source-read at the pin): the images install vllm with the
**audio** extra only (`Dockerfile:43`, same line in the a3/310p/openEuler variants) while
`vllm/setup.py:658` defines `bench = [pandas, datasets]`; the message is upstream's
`PlaceholderModule` rewrite (`utils/__init__.py:2582-2587`) of the soft imports at
`benchmarks/datasets.py:44-52`, and the project's own recipe
(`benchmarks/requirements-bench.txt`, `benchmarks/README.md:38`,
`docs/.../performance_benchmark.md:37`) already has the user install them.
**Answered** in 11 tool calls with that table plus `--dataset-name random/sharegpt` as
the pandas-free path. The gap a later public comment exposes: the ladder opened with a
wheelhouse download instead of naming the **index** hypothesis, and the reporter's next
message is `pip install pandas` → `No matching distribution found for pandas` — the
image carries `global.index-url` pointing at a public mirror (`Dockerfile:20,36`), which
an air-gapped pod cannot reach (`../image-dependency-faults/pip-index-visibility.md`
section B).

**#1895** vllm-ascend `v0.9.2rc1` (`b5b7e0e`) + vllm `v0.9.2` (`a5dd03c1`), 4×910B3,
Qwen2.5-32B, TP=4: v1 graph mode dies at startup with
`NotImplementedError: _C::rotary_embedding: attempted to run this operator with Meta
tensors`, after weights load and before capture. **Answered completely** in 29 turns
with a `path:line` table and correct confidence labels: the Ascend rope monkeypatch
(`ops/rotary_embedding.py:286`) takes the fp16-gated `torch.ops._C.rotary_embedding`
branch (`:29-31, 60-70`) whose only exemption is torchair (`:42-48`), and
vllm-ascend's own binding registers it `PrivateUse1`-only
(`csrc/torch_binding.cpp:290,299,302`, no `kMeta` anywhere in `csrc/`); the
`Failed to import from vllm._C` warning is normal on NPU and the crash itself proves
`vllm_ascend_C` loaded. Ladder given: `--enforce-eager`
(`platform.py:160-162`), then `--dtype bfloat16` as a labelled hypothesis (discriminator:
the new error names `npu_rotary_embedding`), then a maintainer-side Meta impl. Cost
note: the verdict text arrived only at the 7th of 11 substantive turns, after ~12 K
characters of reasoning; the pasted traceback's missing middle frames were correctly
listed as the open gap. Later-ref correspondence added from #8590's tree: v0.14.0rc1
moved the op to `_C_ascend` and registers Python Meta impls
(`vllm_ascend/meta_registration.py:108-111`, imported at `utils.py:255-258`) plus
`csrc/torch_binding_meta.cpp` — `../startup-failures/meta-kernel-missing.md`.

**#4309** vllm-ascend v0.11.0rc1 (`c5fe179cef7f`) + vllm v0.11.0
(`b8b302cde434`), 8×910 64 GB, `Qwen/Qwen2.5-7B-Instruct-1M` served with
`--max-model-len 200000 --dtype float16`: `Starting to load model` then
`AssertionError: Torch not compiled with CUDA enabled` at upstream
`rotary_embedding/dual_chunk_rope.py:37`, reached from `get_rope`'s
`dual_chunk_attention_config` branch; the reporter had already found that deleting
the key from `config.json` works. Track: checkpoint-config → upstream CUDA-only
path (`../startup-failures/checkpoint-config-cuda-only-path.md`, page written from
this case). The session shipped a complete answer in 12 assistant turns with the
verdict on turn 7, and got the useful extra facts: the siblings in the same package
use `current_platform.device_type`, `vllm_ascend/ops/rotary_embedding.py` patches
only four rope classes, and nothing under `vllm_ascend/attention/` mentions the
feature's fields. Two wasted calls worth avoiding: `rg -rn "dual_chunk"
/workspace/history` (the `-r` replacement trap of SKILL.md rule 7 — it printed
`self.n_attention_config` and nearly became a lead) and a `history-titles.sh`
alternation that ORed a common model name into the regex (53 hits, none relevant).

**#5750** image `vllm-ascend-0.13.0-rc1`, 4×910B, Qwen3-Next-80B-A3B-Instruct:
"Failed to compile custom operators with Triton Ascend" on the first request
(`MLIRCompilationError` / `[ConvertLinalgRToBinary] … operand is unlinked` /
`bishengir-compile … exit status 1` inside
`vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py`), in a container that had
`pip install`-ed `triton-ascend-3.2.0rc4` from a mirror over the image's own dev
package. Track: image/dependency pair
(`../image-dependency-faults/triton-ascend-pair.md`). The reporter's later
follow-up carries the maintainer answer — do not install triton-ascend in these
images, and v0.14.0rc1 needs none — which is exactly the action the session was
circling. Delivery: **no ranked answer at all**; 10 assistant turns, ~40 K
characters of reasoning against 3.8 K of message text, cut on the turn where it
had `findings.md` and the verdict in hand. Its first bash call paid 104 KB for
`cat SOURCES.json` + `ls /workspace/history` (SKILL.md rule 7), and the pinned
tree is v0.11.0, which has no `vllm_ascend/ops/triton/` directory — the version gap
it flagged correctly.

**#8345** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be9`), Qwen3-32B
(**dense**) under PD with `MooncakeConnectorV1`, `--data-parallel-size 2
--data-parallel-rank 0 --tensor-parallel-size 8` and
`kv_connector_extra_config.prefill/decode.dp_size = 2`: `ValueError: KV transfer
'prefill' config has a conflicting data parallel size. Expected 1, but got 2.`,
identical for DP=2 and DP=4, evidence only a screenshot (OCR = the paste). Track:
startup config validation, downstream raise site
(`../startup-failures/arg-validation-errors.md`, new second-raise-site section).
The mechanism the tree supports: upstream rewrites a non-MoE engine's DP to 1 in
`vllm/v1/engine/core.py:1061-1073`, the check compares against
`data_parallel_size`, and the extra-config numbers have no other reader — so
`dp_size: 1` is the one-line fix and the keys cannot be deleted (connector
asserts). Delivery: answered on turn **15** (G1/G2 missed), after ~41 K characters
of reasoning against 3.4 K of message text, and cut inside its final section; four
turns went into `ParallelConfig`/`arg_utils` before the twelve-line
`run_engine_core` branch resolved it — the cheap route is
`rg -n 'is_moe\|data_parallel_size = 1' vllm/v1/engine/core.py`.

**#2015** vllm-ascend v0.9.2rc1 (`b5b7e0e`) + vllm v0.9.2 (`a5dd03c`), 910B
(Ascend910B2C) 16 cards, `DeepSeek-R1-bf16-hfd-w8a8`, offline `LLM(...)` with
`enforce_eager=True` and **no** `quantization=`: `RuntimeError: Engine core
initialization failed. See root cause above.` with no worker traceback. Track:
startup failure, quantization gate in the *absent* direction
(`../startup-failures/quantization-method-gate.md`, section B). Mechanism the
pinned tree documents: the msmodelslim W8A8 layout deliberately has **no**
`quantization_config` in `config.json`
(`docs/source/user_guide/feature_guide/quantization.md:37`), the feature is enabled
"by specifying `--quantization ascend`" (`:5`, `:94`), and the page's own FAQ maps
`KeyError: 'xxx.layers.0.self_attn.q_proj.weight'` to "first, make sure you specify
`ascend` quantization method" (`:99-101`) — so upstream's auto-detect
(`vllm/config.py:881-895`, gated on `quantization_config`/`compression_config`)
has nothing to read and the int8 tensors meet a dense model inside `load_model`
(precedent #813, same `KeyError` shape). Second cause, with arithmetic: 671 GB of
int8 weights / 16 cards = 41.9 GB against a 51.2 GB budget (`0.8 × 64`), leaving
9.3 GB for activations *and* an unsharded MLA KV of several GB per rank at the
default 163840-token context. Excluded: #1458's ACL-graph refusal, because
`enforce_eager=True` satisfies `ascend_config.py:158-167`. Delivery: **no answer
shipped** — 17 turns / 25 tool calls, ~45 K characters of reasoning against 4.0 K
of message text, cut on a tool call, after the documented quantization gate had
already been read at turn 11. Cost detail worth keeping: `scripts/card-budget.sh`
was called with `--params 671e9` (a raw count, not billions) and printed a
confident inverted FAIL which the run then had to disown and redo by hand — the
script now refuses that input.

**#14817** vllm-ascend v0.23.0 (`5cb98ca`) + vllm v0.23.0 (`0fc695f`), Atlas
800T A2 / 910B2 (driver 26.1.1), Qwen3-Reranker-0.6B served with
`--runner pooling --hf_overrides '{"architectures":
["Qwen3VLForSequenceClassification"], "classifier_from_token": ["no","yes"],
"is_original_qwen3_reranker": true}'` exactly as the v0.23.0 tutorial page prints
it: startup dies in the **APIServer** with `TypeError: Invalid type of
HuggingFace config. Expected type: <class '…Qwen3VLConfig'>, but found type:
<class '…Qwen3Config'>` from `qwen3_vl.py get_data_parser` → `get_hf_config`,
then `ERR99999`. Track: architecture-name routing (document defect), not device,
driver or checkpoint. The VL prefix makes `registry._normalize_arch`
(`registry.py:1134-1158`) resolve `Qwen3VLForSequenceClassification` →
`Qwen3VLForConditionalGeneration`, i.e. the multimodal model, whose processing
info asserts `Qwen3VLConfig` (`qwen3_vl.py:865-885`); the text name
`Qwen3ForSequenceClassification` resolves to `Qwen3ForCausalLM` and works. The
tutorial's three wrong sites (`Qwen3-Reranker.md:141,156,284`) contradict the
repo's own e2e job (`tests/e2e/pull_request/one_card/lora/test_qwen3_reranker_lora.py:29`)
and upstream `docs/models/pooling_models/scoring.md:52,88` vs `:100,109`. Delivery:
the recorded run answered in Chinese with the corrected command first, the
process-location argument and the contradiction table — a correct, complete
answer. What cost it turns: it had to *rediscover* `_normalize_arch` by walking
`config.py`/`adapters.py`/`registry.py` over ~6 turns, and its first call bundled
`skill_view` with `cat SOURCES.json` + `ls /workspace/history` (24-28 KB, plus a
104 KB skill body) — the rule-7 bundle again. Route now:
`../pooling-embedding/seq-cls-arch-routing.md`, and check 10 of
`../tutorial-verification/tutorial-command-audit.md` for the doc-defect framing.

**#3098** vllm-ascend v0.10.2rc1 (`048bfd5`), ray over 2 nodes x 8 910B4,
Qwen3-Next-80B-A3B-Instruct: "该ray集群部署Qwen3-30B-A3B-Instruct正常运行，而部署
Qwen3-Next-80B-A3B-Instruct报错", with no error text at all — the body pastes only the
ray cluster resource view. Track: the model-specific support surface of the first
release that claims the model. Facts at that ref: the release note
(`user_guide/release_notes.md:9`, PR #2917) says EP and MTP "doesn't work with this
release"; the family tutorial makes triton-ascend + the BiSheng toolkit a manual
per-node, aarch64-only install (`tutorials/multi_npu_qwen3_next.md:41-68`, `:75-78`)
and validates exactly one shape (`:89`: `--tensor-parallel-size 4 --max-model-len
4096 --gpu-memory-utilization 0.7 --enforce-eager`); the model asserts no prefix
caching (`vllm_ascend/models/qwen3_next.py:1186`). The recorded run answered in 7
turns, ranking "node 2 lacks triton-ascend / has a different vllm-vllm_ascend" first
— and the reporter's later clarification (the working sibling is
`Qwen/Qwen3-30B-A3B-Instruct-2507`, not the original 30B) took away its third
candidate: the 2507 refresh carries the same 262,144 context as Qwen3-Next, so the
eager `max_model_len²` mask arithmetic could not separate the pair
(`../runtime-memory/attention-mask-host-allocation.md`, section on works/fails
pairs). Route now: `../startup-failures/environment-correspondence.md` (per-worker
environment on ray) plus the Qwen3-Next pages. Delivery: the answer shipped, but the
cap landed inside its own ladder and dropped the closing artefact list.

**#12015** vllm-ascend v0.22.1rc1 (`5f6faa0`, `v0.22.1rc1-310p-openeuler` image) +
vllm v0.22.1 (`0decac0d`), Atlas 300I DUO (310P), TP=1: the title says
Qwen3-14B-w8a8sc, the command serves `/model/Qwen3-VL-2B-Instruct/` in fp16 with
`--enforce-eager`, a `FULL_DECODE_ONLY` `--compilation-config`, prefix caching left
on, and the 310P doc's `--additional-config … fuse_norm_quant: false`. After
`rank 0 in world size 1 is assigned …` the EngineCore child died:
`!!!!!!! Segfault encountered !!!!!!!` with a dump whose top frame is a **CPU**
`torch.zeros` fp16 fill (`at::native::fill_kernel` → `VectorizedLoop2d` →
`fill_non_native_type<c10::Half>`), then eight
`[ERROR] TBE Subprocess[task_distribute] … main process disappeared!` lines and the
parent's `Failed core proc(s): {'EngineCore': -11}`. Track: signal death at engine
start on a combination the release's own 310P table does not validate. Delivery:
**no answer at all** — 20 turns and the whole 16 K output budget went into
investigation (fusion-config plumbing between `additional_config` and upstream
`pass_config`, the 310P mask builder, the doc's validated rows) and the cap landed
on a status note; it also broke G1/G2 by emitting no verdict text until turn 19.
What the next run should have: the startup section of
`../hccl-and-hardware/worker-segfault-no-python-frames.md` (child = EngineCore
spawn, `-11` = SIGSEGV, TBE block = messenger, trivial CPU fill = surfacing site)
and the v0.22.1rc1 block of `../soc-platform-paths/atlas-300i-310p.md` (the
`_310p/` subpackage, the doc's validated table, the inert fusion flag, the
`max_model_len`-sized mask). Root cause stays unresolved — no plog in the paste and
no maintainer answer in the corpus; the ranked candidates are the image/CANN/
torch_npu pairing (#10136 family), the unvalidated model/feature combination, and
host-memory pressure during init.

**#698** image `vllm-ascend:v0.8.4rc1` (pin `086423d`), vllm v0.8.4 (`dc1b4a6f`),
910B4, DeepSeek-R1-Distill-Qwen-1.5B, `--max-model-len 10000 --dtype float16
--tensor-parallel-size 1 --enable-reasoning --reasoning-parse deepseek_r1`: the log
ends at `[model_runner.py:904] Loading model weights took 3.3461GB`, nothing else,
HTTP port refused. Track: startup silent stall. The paste's line number reproduces
in the pin's **V0** runner (`worker/model_runner.py:904`; V1 is
`model_runner_v1.py:781`, and vllm's own `model_runner.py` carries no such string),
and the next code is `worker.py:240-256` `determine_num_available_blocks` →
`profile_run()` ending in `torch_npu.npu.synchronize()` (`model_runner.py:1078`) —
nothing captures graphs at that ref (`worker.py:325-326`). Both flags the reporter
used are valid at vllm v0.8.4 (`arg_utils.py:216-217,979-994`), so the command is
not the fault. The recorded session shipped **no answer** (14 turns, 11 bare tool
calls, ~47.6 K characters of reasoning against 322 characters of text); the public
thread moved only when the maintainer asked for the startup command and the
reporter supplied the `vllm serve` line — the ask, not a mechanism, was the
productive step. Page `../startup-failures/silent-stall-after-weight-load.md`.

**#11742** `[Community Test]` document task (InternVL3.5-38B-W8A8 on 2×800I A2,
`推理众测任务书-vllm.xlsx` row 7): the run never delivered. Delivery lesson only —
its report heredoc opened on the 9th of 10 turns, the session then returned to
research, and the 10th turn carried no message text at all while the harness
reported 608 K prompt tokens and the full 16 K output-token cap consumed. The rule
it broke is the document bullet of `SKILL.md` rule 1 (skeleton early, 2-3 K per
write, final message = condensed report); the 98 K-character `SKILL.md` read on
turn 1 that already carried the `[Community Test]` routing row, plus three
whole-page reference reads (tutorial-verification index, community-test-report,
tutorial-command-audit) on turns 2-4, is what made the prompt budget that tight.
