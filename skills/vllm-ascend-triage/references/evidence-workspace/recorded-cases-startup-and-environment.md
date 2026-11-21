# Startup, install and environment cases

Engine-start failures (silent stall, worker exit, metadata/capture phase), image and
package faults (`libatb.so`, ABI split, pip index visibility, triton-ascend pair,
published artefacts), SoC and version gates, connector config attributes, tutorial
document cases and `[Community Test]` reports. Routes: `../startup-failures/index.md`,
`../image-dependency-faults/index.md`, `../soc-platform-paths/index.md`,
`../tutorial-verification/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

**#2473** vllm-ascend v0.10.0rc1 (`4604882a3e2c`) + vllm v0.10.0
(`6d8d0a24c02b`), Qwen3-30B-A3B-W8A8 (msmodelslim) on 910B4 ×8, TP8 +
`--quantization ascend` + `--enable-expert-parallel`, `--max-model-len 4096
--max-num-batched-tokens 8192 --max-num-seqs 32`, chunked prefill + prefix
caching: engine start dies at graph capture with
`AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027`, `ERR00100
PTA call acl api failed`, `EE9999 … Not allow to synchronize captured-stream,
stream_id=9`, deepest frame `compilation/piecewise_backend.py:194` →
`entry.runnable`. Track: capture-phase host sync inside the W8A8-dynamic MoE
all2all dispatch (`w8a8_dynamic.py:1001` → `:384` `torch.bincount` in the
`init_routing_quant` fallback, `:446-448` still syncing on the newer branch);
`ERR00100` is the generic wrapper, not a fabric fault. The report carried **no
serve command** — `--enable-expert-parallel` was inferred from the code path,
which the reporter's later comment confirmed along with the cheapest
discriminator nobody had asked for: the unquantized Qwen3-30B-A3B-Instruct-2507
with `--quantization None` and identical flags serves clean, so the quantization
axis owns the extra sync (and the bf16 EP path's own `bincount` at
`ops/fused_moe.py:369` is the follow-up question that keeps it honest). The run
delivered its ranked answer on the 12th turn. Page:
`../startup-failures/captured-stream-host-sync.md`.

**#8286** vllm-ascend v0.18.0rc1 (`99e1ea0`) + vllm v0.18.0 (`bcf2be9`),
"[Installation]: vllm & vllm-ascend torch version conflict" — two wheel commands
(`pip install vllm==0.18.0`, then `uv pip install … vllm-ascend==0.18.0rc1` from the
huaweicloud variant index) followed by pip's post-install block: `vllm 0.18.0
requires torch==2.10.0, but you have torch 2.9.0`, the same for torchaudio/torchvision,
`opencv-python-headless>=4.13.0 … but you have 4.11.0.86`, plus `te 0.4.0 requires
ml-dtypes` and `outlines 0.1.11 requires outlines_core`. Install/pip-resolution track,
nothing crashed. The run delivered a complete ranked answer in 10 turns / 18 tool calls and got the
`torch_npu` hard-pin mechanism right, but framed the pair as intended from
`conf.py:80 ci_vllm_version: v0.18.0` and had to leave "does vllm 0.18.0 run on torch
2.9.0?" open; the two greps that close it are `Dockerfile:52-56` (`VLLM_TAG=v0.18.0`
installed with `VLLM_TARGET_DEVICE=empty`, so the project's own image pairs exactly
this way) and upstream `setup.py:843-864` (`_no_device()` → `common.txt` with no torch
pin, `_is_cuda()` → `cuda.txt` with `torch==2.10.0`), which is *why* the PyPI wheel is
the only side carrying 2.10. `conf.py:75-76` shows the pair the tag documents is
0.17.0 + 0.17.0rc1, and `tests/e2e/doctests/002-pip-binary-installation-test.sh:40-56`
installs it and then hand-pins `torchvision==0.24.0 torchaudio==2.9.0`. Route now
`../image-dependency-faults/pip-resolver-warning-block.md`.

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
`../startup-failures/connector-config-attributes.md` and
`../image-dependency-faults/launch-env-and-library-path.md`;

**#1496** vllm-ascend v0.8.5rc1 (`ec27af3`) + vllm v0.8.5.post1 (`3015d563`),
Qwen2.5-Math-1.5B, `tensor_parallel_size=4` with
`distributed_executor_backend="external_launcher"`, torch 2.5.1 / torch_npu 2.5.1 /
CANN 8.1.rc1: "TP=1 works, TP>1 deadlocks during driver initialization", the paste
ending after the worker's device line with no traceback, and both an `LLM()` example
and an `AsyncLLM` script reproducing it. This ref carries no `external_launcher`
string at all, so the running mechanism is upstream's; `init_process_group` is passed
`world_size=parallel_config.world_size` explicitly, which is why fewer processes than
TP hang instead of raising the `reshape` error. The recorded run shipped **no message
text** in 12 turns (~49 K characters of reasoning spread over four mechanisms), so the
lane now has its own ordering and its own page:
`../startup-failures/multiprocess-launch-hang.md`.

**#1383** vllm-ascend v0.9.1rc1 (`c30ddb8`) + vllm v0.9.1 (`b6553be`), 310P3 /
Atlas 300I Duo, `vllm serve Qwen/Qwen2.5-0.5B-Instruct` with no `--dtype`: V0
`profile_run` dies with `EZ3002 … Optype [TransData] of Ops kernel [AIcoreEngine] is
unsupported. Reason: [tbe-custom]:op type TransData is not found in this op store.`
plus `format and dtype is not precisely equivalent to … op information library`. bf16
on a SoC whose ATB rope path has no bf16 op; the same tag's tutorial says
`dtype="float16" … cause some ATB ops cannot support bf16 on 300I series`
(`docs/source/tutorials/single_node_300i.md:175`) and `rotary_embedding.py:52` keeps
310P off the custom op, so `+rotary_embedding` cannot move it. Delivered:
`../soc-platform-paths/atlas-300i-310p.md`, its `TransData`/`EZ3002` section.

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

**#11751** the third community-test row of that family (v0.18.0 image,
hjh2-Qwen3-VL-8B-Instruct on 1×Atlas 800I A2, HDK 25.5.0.B078, page
`Qwen-VL-Dense.md`): the report file *was* written this time — a 4.3 K skeleton
from the 10th assistant turn plus two appends, 15.4 K characters — and the run
still hit the 16 384-token cap, because those three appends were 16.1 K
characters of tool-call arguments beside 24.0 K of reasoning. The final message
was a second copy of the report and was cut inside row 5 of its defect table, so
the last defect, the three 完整性/易用性/正确性 scores and the FAQ never shipped.
The audit itself was at the quality of the runs that delivered — the loss is
write-and-message sizing only (`../tutorial-verification/community-test-report.md`).

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

**#15389** vllm-ascend v0.23.0 (`5cb98caaadef`) / vllm v0.23.0 (`0fc695fc`), the
Qwen3.5-397B-A17B tutorial's §5.6 请求转发 example: the reporter runs
`load_balance_proxy_server_example.py`, the proxy comes up and the P instance goes
down; swapping in `load_balance_proxy_layerwise_server_example.py` works. Document
defect, not an environment fault: the A3 tab's prose says "Use the layerwise proxy
script" (`:817`) and its block runs the p2p one (`:823`) although §5.4 configures
`MooncakeLayerwiseConnector` (`:521`, `:597`), the 950DT tab's block is the layerwise
script while §5.5 configures p2p `MooncakeConnector` (`:703`, `:786`), and the
section's `bash proxy.sh` (`:836-838`) names a file absent from
`examples/disaggregated_prefill_v1/`. The mechanism is source-visible — the p2p
proxy's `kv_transfer_params` (`load_balance_proxy_server_example.py:790-800`) omits
`remote_cached_tokens`, which `MooncakeLayerwiseConnector` reads unconditionally
(`mooncake_layerwise_connector.py:990`) — while the reporter's actual P crash text
stayed an open item. Checklist item added: check 11, tab block vs its own prose and
its own connector — `../tutorial-verification/tutorial-command-audit.md`.

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

**#6261** reported vllm-ascend 0.13.0.rc2 (pins v0.14.0rc1 `52d4acfa` / vllm v0.14.1
`d7de043`), Atlas 800T A3, Qwen3-Omni-30B-A3B-Instruct with
`export VLLM_ASCEND_ENABLE_FLASHCOMM=1`, TP2 + `--enable-expert-parallel` +
`FULL_DECODE_ONLY`: engine start dies in `profile_run`/`_dummy_run` with
`Dynamo failed to run FX node … add(*(FakeTensor(size=((s59//2), 2048)),
FakeTensor(size=(s68, 2048))))`. Track: startup Lane B, feature×model combination. The
run answered it (ranked Chinese answer with a `path:line` table): FlashComm1 is
sequence parallelism through `enable_sp()` (`utils.py:812-841`, the legacy
`VLLM_ASCEND_ENABLE_FLASHCOMM` still counts), SP keeps the row-parallel output
reduce-scattered (`ops/linear_op.py:386-396` skips the all-gather) while the Omni
thinker adds a **full-length** `deepstack_input_embeds_{i}` per layer
(`qwen3_omni_moe_thinker.py:644-650`), and `vllm_ascend/` has no `deepstack`/`omni`
file at all — so no downstream patch shards it. It also found the project's own guard
(`platform.py:394-403` `ValueError` for VL+FLASHCOMM, with `is_vl_model` counting
`thinker_config` at `utils.py:905-915`) and correctly warned that this would make the
same command fail fast on the newer ref, leaving "why 0.13.0.rc2 crashed instead" open
(a one-commit snapshot cannot diff it). It flagged the report's own contradiction
(单卡 vs `--tensor-parallel-size 2`). Page
`../feature-enable-recipes/sequence-parallel-flashcomm.md`.

**#6972** vllm-ascend v0.14.0rc1 + vllm v0.14.1, 8×910, Qwen3-Omni via
`vllm serve --omni`: `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",
"cudagraph_capture_sizes":[1,2,4,8,16,24,48]}'` and a stage yaml
`engine_args.compilation_config: {level: 1, …}` both "未生效". Track: config gate
audit, not a crash. The run shipped **no answer at all** — 54 messages, 1.27 M prompt
tokens, the whole 16 K output budget inside reasoning and tool calls, G1/G3 both
broken — although it had the decisive finding at hand: `level: 1` maps to
`CompilationMode.STOCK_TORCH_COMPILE` (`vllm/config/compilation.py:748-758`,
`:355-369`), which vllm-ascend refuses by setting `cudagraph_mode = NONE`
(`platform.py:241-245`, warning only). It also found that vllm-omni is absent from the
snapshot (`--omni` has no hit in vllm v0.14.1, so the CLI→stage merge is unreadable)
and that the Qwen2.5-Omni tutorial's `--compilation-config '{"full_cuda_graph": 1}'`
names a key this ref's `CompilationConfig` (`extra="forbid"`) does not define. Page
`../startup-failures/graph-mode-not-applied.md`.

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

**#11095** vllm-ascend v0.21.0rc1 (`80610e44`) + vllm v0.21.0 (`ad7125a4`), image
`quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler`, 2×910B3 (64 GB), TP=2,
Qwen3.5-27B, `--max-model-len 133000`,
`--speculative-config '{"method": "qwen3_5_mtp", …}'`,
`--additional-config '{"enable_cpu_binding":true}'`, container
`-v /mnt/xxx/model:/model`: the serve hangs after
`Loading safetensors checkpoint shards:   0% Completed | 0/11`, no error, both
`VLLMWorker_TP` processes alive at 26 618 MB each, port never opens. Track:
startup, silent stall **inside** the shard loop (`model_runner_v1.py:3635`
`get_model`, before `Loading model weights took` at `:3666`). The session shipped
**no answer** (11 turns, 606 K prompt tokens, the full 16 K output budget in
reasoning): it derived "HBM ≈ checkpoint/TP ⇒ the weights are already loaded, the
`0/11` frame is stale", and every later turn was post-load code (`qwen3_5`
registration, `gdn.py` `_warmup_prefill_kernels`, NZ cast, cpu binding) — an
inference the pin refutes, since `vllm/model_executor/model_loader/base_loader.py:53-64`
builds the model under `with target_device:` before `load_weights`, so the
parameters are on the device before the first shard is read. It also used
`weight_utils.py:938` `Filesystem type for checkpoints: EXT4` to dismiss the
storage lane; that line is `_get_fs_type`'s `/proc/mounts` lookup (`:784-812`),
used only to decide prefetching. The reporter's own later comment
(issuecomment-4831844713, 2026-06-29) resolved it by copying the checkpoint into a
**new host directory** and re-running the container against it — `--name` changed,
*"其他容器启动命令不变"*, *"真的是奇奇怪怪！！！"* — i.e. a reporter-confirmed outcome
on the checkpoint **source** with the mechanism unsettled. Page
`../startup-failures/weight-load-stall-in-shard-loop.md`, one call
`../../scripts/shard-source-probe.sh`.

**#6309** vllm-ascend v0.14.0rc1 (`52d4acfa`) + vllm v0.14.1 (`d7de043d`), image
`quay.io/ascend/vllm-ascend:v0.14.0rc1-310p`, Atlas 300I (310P), Qwen2.5-VL-7B
SFT checkpoint under `/mnt/nfs310p/…`, `--enforce-eager --dtype float16 --max-model-len 2048`: the
`[Doc]` question was "which image tag supports 310P", but the paste is a Lane B
import failure — `EngineCore failed to start` at `_310p/worker_310p.py:21` →
`patch/worker/__init__.py:20-21` (`HAS_TRITON`) → `patch_triton.py:4` →
`ops/triton/fla/chunk.py:15` → upstream `fla/ops/utils.py:141` module-level
`device = … get_available_device()` → `triton/backends/ascend/driver.py:188,91`,
`SystemError: <built-in function get_arch> returned NULL without setting an
exception`. The run answered it well (verdict first, one-line probe,
`pip uninstall -y triton` ladder, pins labelled, registry content marked
unverified); keep the shape. Page `../soc-platform-paths/atlas-300i-310p.md`
(triton-import section).

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

**#2973** `[Doc]` vllm-ascend v0.10.2rc1 (`048bfd5`; upstream export is vllm
v0.9.1 `b6553be`, `dated_reference_only`), one node, running the repo's own
`examples/offline_external_launcher.py --tp-size=4 --proc-per-node=2
--enable-expert-parallel` on Qwen3-30B-A3B: every rank dies in
`initialize_model_parallel` with `RuntimeError: shape '[-1, 1, 1, 4]' is invalid for
input of size 2`. Not a defect and not lane C — the script sets the world size from
`--node-size × --proc-per-node` (`:251`) while `--tp-size` is an independent value,
and upstream skips its own `init_process_group` when the group already exists, so
nothing compares the two. Delivered: the two numbers decoded (`[-1, dp, pp, tp]` vs
real world size), the documented `--tp-size=2 --proc-per-node=2` ladder, and three
doc defects found in the same file (the invariant is never stated; the multi-node
Node 1 block omits `--proc-per-node=2`, so it computes world 2 against Node 0's 4;
a world size that *is* a multiple of TP silently becomes external-DP groups).
16 tool calls / 9 turns / 9.3 K completion tokens, and the walk was longer than the
message needed: one grep of `world_size|tp_size|node_size|proc_per_node` in the
example gave the whole mechanism at call 2, and the `docs/source/` grep for the
example's name returned nothing (it is not documented outside the docstring). Page
`../startup-failures/arg-validation-errors.md` (third raise site). Open: how many
NPUs the reporter's node has, and whether the 2-process gloo group formed cleanly.

**#527** vllm-ascend v0.7.3rc1 (`f025df0`) + vllm v0.7.3+empty, one host, the
reporter's own `sh vllm-infer.sh` running `vllm serve … --max-model-len 4996
--port 8999 -tp 4`: title `[Bug]: cannot import name 'PoolingParams' from 'vllm'
(unknown location)`, two screenshots and no pasted log. The OCR carries the whole
traceback (`vllm/entrypoints/cli/main.py` → `serve.py` → `openai/api_server.py:34`
→ `vllm/engine/multiprocessing/__init__.py` `from vllm import PoolingParams`) plus
`ERR99999`, and its frames resolve `vllm` from `$HOME/hzy/vllm/vllm/…` — not
`site-packages`. Route `../image-dependency-faults/wrong-vllm-module-shadowed.md`;
the reporter cleared it himself the next day by deleting the `vllm` folder in the
current path (issuecomment-2808451427), which is the shadowing finding stated as a
fix. **Delivery failure**: 15 assistant turns, ~70 characters of message text in
total (turn 1 only; every later turn a bare tool call), the full 16 384-token cap
spent, status `output_truncated`, **no answer at all** — although turn 2's OCR
already held the decisive line. What took the budget instead: the `+empty` marker
walk (`setup.py:382,506-507`, README, `installation.md`, five calls), `patch_module.py`'s
`sys.modules` splice, and two history precedents (#291, #384) — the version marker
is the docs' own build route, and nothing in vllm-ascend imports the name.

**#152** vllm-ascend v0.7.1rc1 + vllm v0.7.1 (`0.7.1+empty`), 8×910B1, CANN
`8.0.RC1`, torch-npu 2.5.1.dev20250218, python3.9 env: title `ImportError:
libatb.so` from `import torch_npu` in the reporter's `vllm_test.py`, with
`No module named 'vllm._version'` and `No platform detected … UnspecifiedPlatform`
beside it. The run delivered in one message at turn 23 (6.3 K characters): NNAL
verdict first, the two probe commands, the pinned doc anchors
(`installation.md:88-93`), the `platform.py:23-26` swallow as the second symptom's
explanation, `vllm/version.py:1-8` as noise. Keep the shape. The reporter's later
comment (issuecomment-2677722481) quotes the doc's **source-build** block verbatim
and says installation had no errors — that block contains no NNAL line, so it
corroborates the verdict rather than refuting it
(`../image-dependency-faults/nnal-atb-missing-library.md`, "安装过程中没有报错").

**#540** vllm-ascend v0.7.3rc2 (`00459aea`) + vllm v0.7.3, `vllm serve <model>
--max-model-len 4096 --port 8000 -tp 4` from a conda env
(`/root/miniconda3/envs/hzy_dev_vllm`), env dump `vllm 0.7.3+empty` /
`vllm_ascend 0.7.3rc2` / torch-npu 2.5.1.dev20250308: the title
"RuntimeError: Failed to infer device type" is the last line of a chain —
`ERROR Failed to load plugin ascend` + `ModuleNotFoundError: No module named
'vllm_ascend'`, `No platform detected, vLLM is running on UnspecifiedPlatform`, then
`DeviceConfig` raising at `vllm/config.py:1660` inside the spawned `run_mp_engine`
process. Track: environment / plugin import, before any worker. The recorded run
classified the chain correctly and shipped one interim Chinese answer naming the
editable-install residue (the rc-era docs' only route is `pip install -e .`; the wheel
is marked "Unreleased yet"), then spent the rest of its budget on a synthetic
dist-info probe under `/tmp` that vanished between calls, and was cut at the 16 K cap
with no final answer. No reporter follow-up in the corpus: which residue case applied
stays open. Route: `../image-dependency-faults/plugin-load-module-missing.md` (three
candidates, one `python -c` check each).

**#15836** vllm-ascend v0.23.0 image (`5cb98caa`), 1×Atlas 800 A2 (64 GB × 8) 910B3,
DeepSeek-V4-Flash-w8a8-mtp from ModelScope, launcher "完全拷贝" the tutorial's single-node A2 block. The
body carried two screenshots and no log; the OCR shows
`ValueError: DSA CP requires SP to be enabled. Please enable SP(set VLLM_ASCEND_ENABLE_FLASHCOMM1=1) to
use DSA CP.` wrapped as `EngineCore failed to start`. Cause: `--additional-config` kept
`"enable_dsa_cp": true` while the copied env block dropped `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`
(doc `:196`), and `enable_dsa_cp()` raises during model construction
(`utils.py:1515-1536` from `models/deepseek_v4.py:739`) before any weight load. The recorded run
delivered verdict + two fixes + the doc/line table on its 15th turn, and correctly flagged that the
screenshot's `utils.py:1388` differs from the pin's `1533` (image built from a nearby commit; message
identical). Route: `../dsa-sparse-attention/index.md`.

**#955** vllm-ascend image `v0.8.4rc2` (env dump says `vllm-ascend 0.8.5rc1`,
vllm 0.8.5.post1, CANN 8.1.RC1, torch-npu 2.5.1), Qwen2.5-VL-7B-Instruct on 910B2C
×10, `tensor_parallel_size=4`: "[Bug] … Failed with HeaderTooLarge Error". All four
ranks die in the same second as `Loading safetensors checkpoint shards: 0% … 0/5`
with `safetensors_rust.SafetensorError: Error while deserializing header:
HeaderTooLarge`, reached from `vllm_ascend/worker/worker.py load_model` →
`model_runner.py get_model` → upstream `qwen2_5_vl.load_weights` →
`model_loader/loader.py` → `weight_utils.safetensors_weights_iterator`
`safe_open`, plus `_pickle.PicklingError: Can't pickle <class
'safetensors_rust.SafetensorError'>` masking it on ranks 1-3. Track: the checkpoint
*file* is not a safetensors container (Git-LFS pointer / partial copy / wrong
file) — not TP, not model support (`supported_models.md:11` lists Qwen2.5-VL), not
the image/CANN pair, not the title-vs-env version disagreement. The corpus has no
`HeaderTooLarge` hit at all (title grep and issue search), so there was no product
defect to find. The recorded run computed the LFS arithmetic locally (first 8
bytes `b"version "` → u64 2336927755350992246 ≫ the 100 MB cap), could not verify
the cap itself (no network for `uv --with safetensors`), and delivered the verdict,
`ls -l`/`head -c 80`, a per-shard header script and the `--max-model-len` aside in
one answer at turn 8. Page `../startup-failures/checkpoint-file-not-safetensors.md`,
one call `../../scripts/checkpoint-header-scan.py <model dir>`. Open: how the
weights were obtained and the shard sizes.

**#12985** vllm-ascend `v0.23.0rc1-310p-openeuler` (tree v0.23.0rc1 `f4a08bdd`) +
CANN 9.1.0-beta.1, 4× Atlas 300I Duo = 8 dies of ~43 GiB, Qwen3.5-35B-A3B-w8a8-MTP:
`vllm serve --tensor-parallel-size 8 --max-model-len 100000
--gpu_memory_utilization 0.6` with `FULL_DECODE_ONLY` + `enable_npugraph_ex` stalls
behind repeated `No available shared memory broadcast block found in 60 seconds`,
while a TP=4 / 60 K / 0.4 run with `--enforce-eager` serves fine — and that working
command exists only inside the report's screenshots. Track: 310P platform limits, not
HCCL. The recorded run read the OCR first (upstream's watchdog text, the reporter's
own working script, and a co-tenant `mindie_llm_backend --npu_device_id 2`/`3` beside
`Worker_TP0..3`), then answered with the `O(max_model_len²)` mask cost at the
configured context, the documented TP>1 graph-capture budget, and a three-axis ladder;
the reporter later reported the same stall on Qwen3-VL-32B and asked "是硬件的问题么".
That second checkpoint retires every model lane, so the answer stays the documented
platform limits plus co-tenancy — a broken card is the last row. Route:
`../soc-platform-paths/atlas-300i-310p.md` (ranked actions, items 2-3).

**#5571** vllm-ascend v0.13.0rc1 (`1b5d5ab`) + vllm v0.13.0 (`72506c9`, matched by the
paste's `gpu_model_runner.py:4434`), a multimodal serve: `AttributeError:
'NPUModelRunner' object has no attribute 'mm_budget'` raised in upstream `profile_run`.
Classified as downstream-patch drift in the run's first sentence and answered with
nothing at all: 12 assistant turns, 21 tool calls, **226 characters** of message text,
`output_truncated` at the 16 384-token completion cap. Both decisive facts were in hand
by turn 6 — upstream binds the attribute on every path
(`gpu_model_runner.py:565-573`, `MultiModalBudget(...) if self.supports_mm_inputs else
None`) and the Ascend subclass does call `super().__init__()`
(`model_runner_v1.py:191-193`) — and the run then spent six turns re-testing a
"newer upstream moved the initialiser" hypothesis against the two facts instead of
reading the third line of the call site. What was never grepped: the wrapper.
`_torch_cuda_wrapper` (`model_runner_v1.py:3453-3488`) puts `except Exception:` under
its `yield`, so whatever upstream's constructor raised is suppressed by contextlib and
`mm_budget` is never bound — the attribute error is the *second* symptom, and the
reporter's real error (inside `MultiModalBudget.__init__`, i.e. the HF processor /
max-token table) was destroyed. Delivery lesson and mechanism both: when two verified
facts contradict, the missing third fact is a code path, not a re-measurement. Route:
`../upstream-downstream-drift/swallowed-init-exception.md`.

**#3325** vllm-ascend v0.11.0rc0 (`00ba071`) + vllm v0.11.0rc3 (`8ce5d31`), image
`quay.io/ascend/vllm-ascend:v0.11.0rc0`, Qwen3-Next-80B-A3B-Instruct at TP4:
`ERROR [registry.py:547] Error in inspecting model architecture
'Qwen3NextForCausalLM'` followed only by `Traceback (most recent call last):`. Answered
in 11 assistant turns / 19 tool calls with the verdict first and in the report's
language: not a version gate (the arch is registered at
`vllm_ascend/models/__init__.py:59-60` and a tutorial exists), inspection runs in a
subprocess so the cut tail is the finding, then triton-ascend pairing ranked first with
`#2889`'s `tl.exp` inner frame, `#2929`'s `ModuleNotFoundError` second, and the
tutorial's `--enforce-eager` / shorter context as the follow-up. What it added beyond
the paste: an absent triton cannot produce `AttributeError: module 'triton.language'`
(vllm substitutes placeholders), so a triton is installed and mispaired. Cost: one
`cat SOURCES.json` (~21 KB, rule 7) and a full read of the tutorial including its
x86_64 "Coming soon" tab. Route:
`../startup-failures/model-type-not-recognized.md` (inspection-crash section).

**#4948** vllm-ascend v0.11.0rc3 image (`quay.nju.edu.cn/…`, tree `b6d63bb`) + vllm v0.11.0,
"部署满血版 DeepSeek R1 0528 BF16 失败" on Atlas 800I A2 / 910B4, `--tensor-parallel-size 8
--pipeline-parallel-size 4 --enable-expert-parallel --distributed-executor-backend ray
--additional-config '{"ascend_scheduler_config":{"enabled":true},"torchair_graph_config":
{"enabled":true}}'`, `--max-model-len/--max-num-batched-tokens 65536`: every rank printed
`Capturing torchair graph …` and then `Error executing method 'compile_or_warm_up_model'` with
`torch._dynamo.exc.Unsupported: call_function DelayGraphBreakVariable()` from
`dispose_tensor` → `x.set_` (`utils.py:504-505`, called at
`torchair/models/torchair_deepseek_v2.py:985-986`) inside the `fullgraph=True` compile
(`torchair_model_runner.py:423,446`); precedent #1615 carries the same signature. Co-equal
cause on this checkpoint: the paste's `Available memory: 5080964608, total memory:
65452113920` (`worker_v1.py:257`) leaves 7.8 % headroom at a 65 K-token profile, and
reporter-confirmed later on the same issue: full-weight BF16 R1-0528 needs ≥4 A2 servers
(32 × 910B4) and the DeepSeek-R1 tutorial suits the W8A8 weights. Known issue in the pinned
tree: `release_notes.md:383` still lists "Pipeline parallel does not work with ray and graph
mode" (#1751/#1754), and both recorded reports ran PP=4. The recorded run reached the verdict
by turn 10 but shipped **no answer** (one 270-character paragraph, the rest reasoning) before
the 16 K cap. Page `../startup-failures/torchair-dynamo-compile-unsupported.md`. Open: the
image's `torch`/`torch_npu` versions against the pinned 2.7.1 pair.

**#12059** vllm-ascend v0.19.1rc2.dev849 (`05e095a20`, from the reporter's `collect_env`)
+ vllm 0.24.0, aarch64 Ubuntu 22.04 host, Mooncake-adjacent report: "env vars set in
`~/.bashrc` are missing on non-interactive container startup (`LD_LIBRARY_PATH` /
`LD_PRELOAD`)". Track: build-recipe / launched-process environment
(`../image-dependency-faults/launch-env-and-library-path.md`). Delivered in 19 turns with
the verdict first and the source-verified table: the preload is **the image's own recipe**
(one `RUN echo … >> ~/.bashrc` pair per variant, all variants), `LD_PRELOAD` expands empty
at build time (the trailing colon the reporter saw), no `ENV LD_*` carries it, and
`CMD ["/bin/bash"]` is non-interactive without a TTY — so the image's advertised default
start is broken independently of `--entrypoint`. The fix is `ENV`, the workaround on
released tags is the export in the launch command, and verification is
`/proc/1/environ`/`docker exec … bash -c`, never an interactive shell — #7352 is the
reporter-confirmed precedent. Re-verified at the snapshot pin (`a78a00e`): the same pair
exists in six `Dockerfile*` (`Dockerfile:81-82`, `.a3:80-81`, `.openEuler:78-79`,
`.310p:63-64`, `.a3.openEuler:78-79`, `.310p.openEuler:59-60`), `CMD ["/bin/bash"]` at
`Dockerfile:84`, and the base is `quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11`
with no `ENV LD_*` in the recipe. Left open, correctly: whether the reporter's
`v0.22.1rc1-a3` tag carries the same lines (tag not in this snapshot) and any concrete
Mooncake failure (nothing in the paste shows one).

**#2725** reporter on v0.10.0rc1 (pinned tree v0.9.1rc3 `763ed69`), openEuler23.03
aarch64, `vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --enforce-eager
--dtype float16` (verbatim from the 300I tutorial): weights load (14.2488 GB) and then
`the socversion Ascend910B1 of bin package does not match the current device socverison
Ascend310P3. Please modify default socversion in run.sh or execute run.sh with
socversion parameter.` The recorded run reached the right verdict (910B image/package on
a 310P3 card, pull the `-310p` variant) but spent four calls grepping the pinned trees
and the history bodies for the message text, which is in neither — it comes from the
CANN custom-op build's `run.sh`. Also landed on the self-build variant
(`vllm_ascend/envs.py:58-61` `SOC_VERSION` default `"ASCEND910B1"`, `setup.py:168`
`-DSOC_VERSION=`). Page: `../soc-platform-paths/atlas-300i-310p.md`, its `bin package`
section.
**#2315** vllm-ascend image `v0.9.2rc1-310p` (tree `b5b7e0e`) + vllm v0.9.2 (`a5dd03c`),
2×`310P3` (host `NPU-SMI 24.1.0.1`), Qwen3-32B TP2: engine start dies inside torch_npu
device open — `RuntimeError: SetPrecisionMode: … AclSetCompileopt(aclCompileOpt::
ACL_PRECISION_MODE, precision_mode), error code is 500001`,
`[Error]: The internal ACL of the system is incorrect.`,
`EH9999 … [PLATFORM]PlatformInfoInit runtime platform info failed. SocVersion Ascend@910B`,
`GEInitialize failed`, ending `ERR99999` — with no `vllm_ascend` frame, and the paste is
OCR'd from screenshots. Same two lines in #302/#564/#692 on 910B2 hosts. Track: lane A
startup, environment-owned (`../startup-failures/acl-init-500001-and-soc-version.md`).
The recorded run confirmed the "not vllm-ascend code" verdict in a turn-17 status note but
shipped **no answer**: 18 turns / 26 tool calls / 232 characters of message text, the
budget spent in `torch_npu` `LazyInitAclops.cpp`, the `SocVersion` enum and `is_310p()`.
The three facts that answer it are the printed SocVersion against `npu-smi`, the image's
CANN base (`Dockerfile.310p:18`, `requirements.txt:27`) against the host driver, and the
framework-free `torch.npu.set_device(0)` probe. Semantics established here, not by a
reported fix: `SetPrecisionMode` only *supplies* the precision string for the SoC it
resolved, so the numeric code is not a flag problem.

**#6578** vllm-ascend v0.14.0rc1 (`52d4acfa`) + vllm v0.14.1 (`d7de043d`), Qwen3-32B on
910B, title `no attribute qkv_rmsnorm_rope`, body's first line an
`AttributeError: '_OpNamespace' '_C_cache_ops' object has no attribute 'reshape_and_cache'`:
the `torch.compile` pass `qknorm_rope_fusion_pass.py:214` traces its replacement and the
triton op is imported only `if HAS_TRITON` (`ops/__init__.py:25-26`) while the pass is
appended by default (`fuse_qknorm_rope` True) and gated only on dtype + `head_size == 128`.
`_C_cache_ops` has zero hits in the vllm-ascend tree, so it is a second launch's line, not
this traceback's. Track: lane A/B graph-compilation
(`../startup-failures/missing-triton-op-in-compile-pass.md`). The run reached the mechanism
by turn 5 and shipped a ranked answer on turn 11, but the cap landed inside its closing
scope paragraph — the fix (`--additional-config '{"ascend_compilation_config":
{"fuse_qknorm_rope": false}}'`, the project's own line at
`docs/source/user_guide/configuration/additional_config.md:79`) did ship.
**#16581** vllm-ascend v0.23.0rc1 (`f4a08bd`, not byte-verified) + vllm v0.23.0
(`0fc695f`), one 910 (64 GB), ModelSlim W8A8 Qwen3-32B (32.03 GiB) on
`--max-model-len 16384 --gpu-memory-utilization 0.9`: `ACL graph capture fails with
error code 507015` at `torch.npu.graph.__enter__` → `torch.npu.synchronize()`
(`torch_npu/npu/graphs.py:519`), wrapped as `NPUModelRunner failed, error is …`, with
`fftsplus aivector error … error code = 0x800000` / `The DDR address of the MTE
instruction is out of range` / `retCode=0x26, [aicore exception]`; the reporter's own
workaround changed util 0.9→0.85, capture sizes 51→`[1,2,4,8]` and `--max-model-len`
16 K→40 K at once. Both frame numbers match the pin (`acl_graph.py:192`,
`model_runner_v1.py:5177`). The recorded run found the two decisive facts — the advice
gate only knows `207008` (`acl_graph.py:28-51`) and nothing reserves the graph pool
(upstream `profile_cudagraph_memory()` is `is_cuda()`-guarded, downstream's
`determine_available_memory` at `worker.py:527-597` drops it) — and correctly refused to
name one cause because the workaround moved three variables, but it spent a turn on a
442-hit history listing for `graph capture` and its answer was cut **mid-sentence inside
its scope block** (3.9 K characters delivered against a 16.4 K completion-token cap), so
the "what to send back" half never shipped. Route now
`../startup-failures/acl-graph-capture.md` (v0.21–v0.23 section) and the new
signature row for `NPUModelRunner failed … AclrtSynchronizeDeviceWithTimeout`.

**#5075** vllm-ascend v0.10.0rc1 as reported (pin v0.12.0rc1 `42ceaf0` + vllm
v0.12.0 `4fd9d6a8`), Atlas 300I Pro (310P), Qwen2.5-VL-3B, `--enforce-eager`,
"310系列什么时候支持图模式和量化" plus "身份证识别 8 s vs T4 3 s": capability/roadmap
question, not a crash. Answered in ~12 turns: quantization already has a 310P path at
the pin (`--quantization ascend`, `quantization/w8a8.py:51,163,320`), graph mode does
not (`docs/source/tutorials/310p.md:5` "only supports eager mode and the float16 data
type"), the reporter's command is that file's Qwen2.5-VL-3B tab verbatim (`:120-131`),
and the staging evidence is RFC #1309's empty `Stage2: Performance improvement`. Two
corrections it needed: `is_310p()` does not exist at v0.12.0rc1 (the predicate is
`get_ascend_device_type() == AscendDeviceType._310P`), and "no 310P e2e coverage at
this ref" was wrong — `tests/e2e/310p/test_offline_inference_310p.py` exists, is wired
by a label-gated workflow, and is eager-only, which is the sharper statement. Route now
`../soc-platform-paths/atlas-300i-310p.md` (v0.12.0rc1 section) and
`../feature-enable-recipes/index.md` rule 9.

**#3524** vllm-ascend `v0.11.0rc0` (`00ba0710`) + vllm `v0.11.0rc3` (`8ce5d319`),
DeepSeek-V3.1-W8A8 with `--quantization ascend`, 2 nodes × 8 910B2, ray,
`--tensor-parallel-size 8 --pipeline-parallel-size 2 --enforce-eager`: startup is
clean (both stages print `Loading model weights took 38.9 GB`, KV cache sized, API
server up) and the first `/v1/completions` kills the engine with
`AttributeError: 'PPMissingLayer' object has no attribute 'self_attn'` at
`vllm_ascend/attention/mla_v1.py:311`, wrapped in `RayTaskError` from the remote
node. The recorded run read it right — PP-blind `layers[0]` RoPE-cache lookup, three
backend twins, the tree's own `start_layer` pattern, the PP e2e test commented out in
CI — but spent most of its turns on a source walk to get there. Page now carries the
mechanism, the stage-aware patch and the `--pipeline-parallel-size 1` ablation:
`../pipeline-parallel/pp-blind-first-layer-lookup.md` (new).
