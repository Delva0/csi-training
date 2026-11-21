# Install, image, version-gate and document cases

Image and package faults (`libatb.so`, the ABI split, pip index visibility, the
triton-ascend pair, published artefacts and wheel variants), SoC and version gates,
connector/config attribute mismatches, capability rows, cross-version drift,
tutorial document cases and `[Community Test]` reports. Routes:
`../image-dependency-faults/index.md`, `../soc-platform-paths/index.md`,
`../feature-enable-recipes/index.md`, `../upstream-downstream-drift/index.md`,
`../tutorial-verification/index.md`, `../release-known-issues/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file. Engine-start failures live in `recorded-cases-startup.md`.

**#13199** image v0.23.0rc1 `-310p` on x86 310P3, Qwen3-VL-8B: `import torch`
dies with `AttributeError: module 'triton' has no attribute 'language'` inside
`torch/_dynamo/utils.py` (`common_constant_types.add(triton.language.dtype)`),
no `vllm_ascend` frame anywhere; the `-310p` Dockerfile removes
`triton`/`triton-ascend` on purpose (`:45-63`, plus
`docs/source/tutorials/hardwares/310p.md:7`), so a resolving `triton` is
uninstall residue or a shadow directory. The run held both facts by
mid-session, spent its whole budget deriving *why* a module can lack `language`
(a property of the reporter's site-packages) and shipped no answer —
`../image-dependency-faults/triton-module-residue.md`.

**#11681** vllm-ascend v0.18.0rc1 (`99e1ea0f`) + vllm v0.18.0 (`bcf2be9`), 910C 64G,
CANN 9.0.0, torch/torch-npu 2.9.0, triton-ascend 3.2.0, reporter's env matches the
pin's own `requirements.txt:16,35,40`: the paste's title is
`Failed to import Triton kernels … Error: No module named 'triton.language.target_info'`,
and its body lists an already-tried ladder (`triton 3.5.0`/`3.6.0` →
`module 'triton' has no attribute 'next_power_of_2'`, hand patches → `triton.csdiv`,
then a `kernel[grid]` launch raising `TypeError: 'function' object is not
subscriptable`), with the failures "集中出现在 `ops/layernorm_gated.py`" and a dense VL
model serving fine on the same box. The title sentence is **upstream vllm's logged,
non-fatal `logger.error`** (`fused_moe/config.py:25-34` at vllm v0.18.0), so the
report's own advice is out of scope; the missing name is `triton_kernels`' import of a
submodule only newer triton ships (`triton_kernels/target_info.py:5`), resolved against
triton-ascend's tree. Track: image/environment (hand-installed triton beside
triton-ascend) — `../image-dependency-faults/triton-kernels-import-error.md`.

**#11308** vllm-ascend v0.21.0rc1 (`80610e4438db`) + vllm v0.21.0 (`ad7125a431e1`),
Atlas 300I Duo, Qwen3.5-35B-A3B-w8a8 + MTP, image CANN hand-upgraded to 9.1.0.beta3: engine
start reaches `Capturing CUDA graphs (decode, FULL)` and a rank dies with `aclrtMemcpy, error
code is 107030` + `ERR00100 PTA call acl api failed` + the canned `EE1006 … Upgrade the CANN
software version` block, deepest frames `_310p/attention/attention_v1.py:293` →
`_310p/attention/attention_mask.py:90 get_splitfuse_mask` → `query_start_loc.to("cpu")`. Track:
capture-phase host readback in the *fallback* splitfuse mask; `:273`'s capture-safe compressed
branch is gated on `hasattr(torch_npu,'_npu_flash_attention_v3') and
hasattr(torch_npu,'_npu_paged_attention_splitfuse_v2')`, so the traceback itself proves the gate
was False and no toolkit/CANN install flips it; MTP is what reaches that branch
(`_310p/model_runner_310p.py:599-607` → `:191-193` forces `SpecDecoding`). The first run was cut
on output tokens with only verdict + ladder head typed (825 characters of message text against
~40 K of reasoning across 17 steps); the reporter's follow-up ("单独 eager mode 可以启动", PR
#10309 promised graph+MTP) was answered in the second run with the coverage-gap table. Routes:
`../soc-platform-paths/310p-mtp-graph-splitfuse-mask.md`,
`../startup-failures/captured-stream-host-sync.md`.

**#15536** image `0.23.0RC1-310P`, 8x 300I Duo, Qwen3.6-27B, serve line copying the
300I DUO tab (`qwen3_5_mtp` + `FULL_DECODE_ONLY` + `enable_npugraph_ex:false`): a block
carrying **both** `copy_between_host_and_device_opapi … aclrtMemcpy … 107030` and
`EE9999 … Not allow to synchronize captured-stream, stream_id=35` /
`synchronize stream failed, runtime result = 107027`, deepest frames
`_310p/attention/attention_v1.py:293` → `attention_mask.py:90` — the #11308 lane, same
refusal reported twice (copy, then the stream sync). Pin read vllm-ascend v0.23.0
(`5cb98ca`): the *qlens* readback is now precomputed (`attention_v1.py:253` +
`:257-262` capture guard + `metadata_builder.py:128-131`) but the fallback **mask** still
reads back (`attention_mask.py:90,93`), so rc1→final is not a fix rung; the capture gate
is `_spec_dummy_capture` with no `method == "mtp"` term (`model_runner_310p.py:631-637`);
and the tutorial's own 300I DUO serve line (`docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md:377-379,407-409`)
is the failing combination, with the `n*(num_speculative_tokens+1)` capture-size rule at
`:427` (a `[2,4]` paste is valid) and `enable_npugraph_ex` force-off at
`ascend_config.py:643-649`. Routes:
`../soc-platform-paths/310p-mtp-graph-splitfuse-mask.md` (v0.23 section).

**#6271** image `quay.io/ascend/vllm-ascend:v0.13.0rc2-openeuler` (tag named by the
reporter afterwards), Qwen3-VL-8B on one card: `vllm serve` reaches
`determine_available_memory → profile_run → _dummy_run` and dies on
`RuntimeError: Failed to compile /tmp/tmpXXXX/launcher_cxx11abi1.cxx` with
`AclOpCompileInterface.h:146: use of undeclared identifier 'aclOpExecutor'` — a host
C++ compile of triton-ascend's ABI-probe TU, i.e. the #6429 header-guard collision, not
the model or vllm-ascend Python. The decisive pairing fact is the image's own recipe:
`Dockerfile.openEuler:18` `cann:8.5.0-910b-openeuler24.03-py3.11` + `:75`
`'torch-npu==2.8.0.post1'`, against `requirements.txt:31` `torch-npu==2.8.0` — so the
reported post1 is the *shipped* pairing and "use the release image" is not a rung. The
run got there (log grep → skill page → header facts → recipe), then finished at the
16 384-token cap with the tail of its Chinese answer cut. Route:
`../image-dependency-faults/host-cxx-compile-header-conflict.md`.

**#6082** vllm-ascend v0.10.0rc1 (`4604882a3e2c`) + vllm v0.10.0 (`6d8d0a24c02b`),
`quay.io/ascend/vllm-ascend:v0.10.0rc1-310p`, 300I Duo ×8 (driver 25.2.0, firmware
7.7.0.6.236), OpenGVLab/InternVL3-8B, `--dtype float16 --enforce-eager --trust-remote-code`:
`NPU function error: call aclnnIsNegInf failed, error code is 561000`, `ERR00100 PTA call acl
api failed`, `EZ9999 … Op IsNegInf does not has any binary`, attributed by torch_npu to
`_execute_mm_encoder` → `scatter_mm_placeholders` (async, so the frame is a sync point). Text
requests serve; only image requests fail. Track: built-in aclnn op with no kernel for the
SoC — the image's CANN is `8.2.rc1-310p` (`Dockerfile.310p:18`), the 300I page states
experimental/eager/fp16 only (`docs/source/tutorials/single_node_300i.md:4-5`), and the
multimodal-only trigger is explained by the vision tower being the only added code
(`vllm/model_executor/models/intern_vit.py:275`, SDPA with no mask). No precedent exists for
the signature (0 hits in the 794-issue corpus and in the issue search). The recorded run
spent its whole output budget hunting the `isneginf` caller across five trees (none of which
contains it) and shipped **no answer**; the deliverable was one standalone probe plus
`ASCEND_LAUNCH_BLOCKING=1`. Page: `../soc-platform-paths/kernel-package-vs-card.md`.

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

**#12856** image `vllm-ascend:v0.18.0-310p-openeuler` (tree `e18643f8`), Atlas 300I Duo ×8,
Qwen3.6-35B-A3B (`Qwen3_5MoeForConditionalGeneration`) TP4+EP, BF16 with `--dtype half`,
`--enforce-eager`: weights load, then `profile_run` raises
`AttributeError: 'AscendSharedFusedMoE310' object has no attribute
'_shared_experts_part1'`. Real downstream bug — the 310P class copied A2's caller
(`_310p/fused_moe/fused_moe.py:301-302`) without its methods
(`ops/fused_moe/fused_moe.py:612,616`), and its `forward` also unpacks a two-tuple from a
parent that returns a bare tensor (`:290` vs `:259`), so the reporter's two earlier
workarounds (bf16 rejection, then the dynamo unpack) were the same class, not three
faults. The run answered all three of their questions in one pass and left the
upstream-`SharedFusedMoE` side open (the pinned vllm v0.24.0 no longer has that module,
so the parent cannot be read at the reporter's pairing). Page:
`../soc-platform-paths/310p-sibling-class-drift.md` (new).

**#10303** 310P + a `vllm-omni` image tag — "310p 使用 quay.io/ascend/vllm-omni:v0.18.0
这个镜像来创建容器部署 Qwen3-TTS 么?" (pin vllm-ascend v0.20.2rc1; vllm-omni absent from
`SOURCES.json`). The run answered "no such 310P tag, use a `-310p` variant" from another
issue's `collect_env`, never opening the post's own screenshot — whose OCR is the
vllm-omni install page listing **Atlas A2 / Atlas A3 only**, the same verdict in one
call. The reporter's later public comments (2026-07-28) supply the route that works:
source build on a `-310p` CANN base with `SOC_VERSION=ascend310p1` (vllm `[audio]` →
vllm-ascend → vllm-omni, triton and triton-ascend removed) plus a two-stage fp16
`--deploy-config`. Delivery lesson: keep the tag verdict to one sentence and put the
build+config route in the same message, because that is the next question. Pages:
`../soc-platform-paths/310p-omni-model-deploy.md` (new),
`templates/omni-310p-deploy-config.yaml`.

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

**#10910** vllm-ascend v0.19.1rc1 (`da421afa`) + vllm v0.19.1 (`b1388b1f`), 2 nodes × 16
910C, Qwen3-235B-A22B: two independent symptoms with a full tested-configuration matrix
(`TP=32` works, single-node `PP=2,TP=8` works, `PP=4,TP=8` fails with
`KeyError: 'model.layers.47.self_attn.attn'`, `DP=2,TP=16` fails with
`ValueError: Current node has no NPU available` and `NPU_group_9_<pgid>` labels). Track:
**two owners**, both upstream ray plumbing, both one check each: the PP key is the
per-worker kv-config index (`adjust_rank` rewrites only `rpc_rank`, `worker_base.py:303`
indexes by the frozen `global_rank`; the module moved to `v1/executor/ray_utils.py:54-61`
at this pin) and the DP message is the *free*-resource read with a live placement group
holding the node. The recorded run confirmed the PP fingerprint on turn ~12 and kept
walking (`launch_core_engines`, `DPMoEEngineCoreActor`, `arg_utils`) for 20+ more turns;
it ended at the 16 384-token cap with 966 K cumulative prompt tokens and **no answer
text**. Pages: `../pipeline-parallel/kv-config-rank-mismatch.md`,
`../startup-failures/ray-dp-placement-group-npu-exhausted.md`.

**#1369** vllm-ascend `v0.9.1rc1` (`c30ddb8`) + vllm `v0.9.1`, DeepSeek-R1-W8A8, 910B3 ×2,
depth pruned to 5 layers, `torchair_graph_config.enabled` with chunked prefill left on:
capture succeeds, then the first `llm.generate` kills both ranks at
`vllm_ascend/attention/mla_v1.py:1099` with
`UnboundLocalError: local variable 'decode_hs_or_q_c' referenced before assignment`. The
recorded run had the decisive artefacts by turn 10 — `graph_mode.md:50-51` and
`test_torchair_graph_mode.py:46-60`, both `only work without chunked-prefill`, both pairing
`torchair_graph_config` with `ascend_scheduler_config` — and the reporter's own commented
key misspelt `asacend_scheduler_config`, then spent the rest of the budget deciding *which*
scheduler step mixed decode with prefill (a 13 K- and a 19 K-character block, 46 messages,
zero delivered answer, output-truncated at 8 192 tokens). Public resolution is that
contract: the maintainer's later comment says torchair does not work with chunked prefill
and points at `AscendScheduler` (which disables it); after being pointed at
`mla_v1.py#L1075` the reporter answered "I was careless". Page now carries the binder
structure, the silent-key/cached-config check and the "patching one line moves the crash"
follow-up: `../feature-enable-recipes/torchair-mla-graph-and-chunked-prefill.md` (new).

**#1456** `[Doc]` vllm-ascend v0.9.1rc1 (`c30ddb8`) + vllm v0.9.1 (`b6553be`),
DeepSeek-R1-w8a8 on one host, `vllm serve … -tp 16 --max-model-len 4096
--quantization ascend` with `export ENFORCE_EAGER=1`: config-time
`NotImplementedError: ACL Graph does not support deepseek … Or set enforce_eager=True`
from `vllm_ascend/ascend_config.py:148-157`, before any weight is read. Track: the
`export` is not a knob (0 `ENFORCE_EAGER` bindings in either tree; the gate reads
`model_config.enforce_eager`, `platform.py:151`), so `--enforce-eager` is the whole
fix, and the same answer owes the page defect that invited the guess (`graph_mode.md:10`
gives only the Python form; `:58`'s TorchAirGraph online example serves Qwen). The
recorded run verified the env-var no-op in two greps and delivered the answer plus
three doc defects, and its own deduction that TP=16 would then hit the MLA
heads-per-rank gate (128/16 = 8) if the reporter moved to torchair. Page:
`../feature-enable-recipes/torchair-mla-head-count-tp-gate.md`, and
`../startup-failures/graph-mode-not-applied.md` for the no-op-env-var check.

**#11926** vllm-ascend/image `v0.22.1rc1`, DeepSeek-OCR-2 single card, the
  tutorial's own serve command plus `--async-scheduling`: `profile_run`'s dummy
  multimodal forward dies at `vllm_ascend/ops/qwen2_decoder.py:233` with
  `AttributeError: 'AscendQwen2DecoderLayer' object has no attribute
  'attention_type'` — the second report of #11661's defect on the same image, so
  the same page answers both (`../upstream-downstream-drift/patched-forward-api-drift.md`).
  The reporter's own datum is the pinned tree's claim that the model "is first
  supported in `vllm-ascend:v0.16.0`" (`docs/source/tutorials/models/DeepSeekOCR2.md:7`)
  while that tree raises, and the pin has no OCR-2 e2e config, so no CI path runs
  the decoder. Track: downstream drift. The recorded run found the dependency's
  TRF011 rule by turn 29 and shipped **no answer** (31 turns, ~48 K characters of
  reasoning, final turn all reasoning with no text); it had spent the earlier
  turns probing a truncated `transformers` export for a `qwen2/modeling_qwen2.py`
  that is not there (`src/transformers/models/` holds only `gpt2`) — the body the
  reporter's own site-packages has, so asking for `inspect.getsource` (now
  `scripts/qwen2-body-drift-probe.py`) was the cheap out.
  **Third filing #11991** (2026-07-14, same image and line) closes the repair question:
  the reporter patched `causal_mask_mapping[getattr(decoder_layer, 'attention_type',
  'full_attention')]`, first reported it working, then retracted it publicly on
  2026-10-10 — *"this modification only allows the model service to start up normally,
  but it will cause problems with model recognition"* — and rejected the suggested
  attribute guard outright (*"It can't solve the problem"*). So the lane's answer is the
  delegation (upstream `deepencoder2.py:104` hands the mapping to `super().forward()`),
  and any start-only patch carries a real-request acceptance check.
  `../upstream-downstream-drift/patched-forward-api-drift.md`.

**#3029** image `v0.10.2rc1` (vllm-ascend `048bfd55`), one Atlas 800T A2 node, Qwen3-Next
  via `vllm serve --trust-remote-code`: start dies with `ModuleNotFoundError: No module
  named 'triton'` raised at `vllm_ascend/ops/casual_conv1d.py:14` (`ops/fla.py:11-12` is
  the sibling) while the log above it carries `Triton not installed or not compatible`.
  The image is triton-free **by design** — `Dockerfile:42-44` runs `pip uninstall -y
  triton` under "in Ascend, triton doesn't work correctly" — and this model imports Triton
  at module import, so the reporter's own tutorial (`docs/source/tutorials/
  multi_npu_qwen3_next.md:46-61` at that ref — BiSheng toolkit `:48-54`, then the
  `triton_ascend-3.2.0.dev20250914` aarch64 wheel at `:58-61`) makes triton-ascend plus
  that toolkit a manual install. Track: image content / documented manual step, not a defect and
  not the model's config; `pip install triton` from PyPI is the wrong repair. The recorded
  run reached the verdict through the Dockerfile, the tutorial and the two import sites,
  after first opening `../image-dependency-faults/triton-ascend-pair.md`, whose scope is
  the *compile* failure (the missing-module shape is the signature row instead) —
  `../startup-failures/environment-correspondence.md` carries the same per-worker install
  for the ray case (#3098).

**#16379** vllm-ascend v0.26.0rc1 (`f2f74a16`) + vllm v0.26.0 (`568afb3a`), one
Atlas 800 A3 node, `DeepSeek-V4-Flash-0731-w8a8` DP4/TP4/EP with the doc's
"A3 series with DSpark" recipe: every worker dies in the first `profile_run`
forward with `RuntimeError: aclnnHcPre or aclnnHcPreGetWorkspaceSize not in
libopapi.so, or libopapi.sonot found.`, deepest frame
`vllm_ascend/models/deepseek_v4.py:974` → `torch.ops._C_ascend.npu_hc_pre_v2`.
Track: the `EXEC_NPU_CMD` dlsym search (`csrc/aclnn_torch_adapter/op_api_common.h:687-700`),
whose first two roots are the vendored custom-op package and `ASCEND_CUSTOM_OPP_PATH`,
with the toolkit's `libopapi.so` last. The recorded run ranked "your CANN 9.1.0
lacks the 910_93 op" first and "A3 is genuinely unsupported" third and never named
`_cann_ops_custom`/`libcust_opapi.so`, although the pinned tree declares and builds
`hc_pre` for `ascend910_93` (`csrc/moe/hc_pre/op_host/hc_pre_def.cpp:65-73`,
`csrc/build_aclnn.sh:173`) — so the vendor table the reporter screenshotted is
about CANN's fallback copy. Delivery: 14 assistant turns carried no message text
and the verdict arrived on the last turn, cut inside the action ladder before the
scope block. Page (new): `../custom-ops-runtime-faults/opapi-symbol-absent.md`.

**#881** reported on vllm-ascend v0.8.5rc1, snapshot pin v0.7.3 (`779eebb8`) +
vllm v0.7.3 (`ed6e9075`), `DeepSeek-V3-0324-w8a8` TP8 PP2 ray `--enforce-eager`:
after 34 min of weight loading, `profile_run` raises `TypeError:
DeepseekV2Attention.forward() got an unexpected keyword argument 'kv_cache'` at
`vllm_ascend/models/deepseek_v2.py:496`. The recorded run answered "the vllm and
vllm-ascend halves are not paired" — a plausible rank, but it had noticed in
reasoning that the class named is the **non-MLA** `DeepseekV2Attention` and did
not use it: `attn_cls` is chosen from `model_config.use_mla`
(`vllm_ascend/models/deepseek_v2.py:131-132`), which comes from the checkpoint's
`model_type`/`kv_lora_rank` (`vllm/config.py:763-767, 998-999`). The maintainer's
public answer (quoted by the reporter 2025-05-20) was the **modelslim** tool plus
trying the project's own `vllm-ascend/DeepSeek-R1-W8A8` weight — a checkpoint
verdict, not a pairing one. Page (new):
`../upstream-downstream-drift/attention-class-from-checkpoint-config.md`.

**#8990** image `vllm-ascend:v0.19.1rc1-openeuler` (pin v0.19.1rc1 `da421afa` + vllm
`v0.13.0` in the snapshot, while the paired upstream is v0.19.1), 910B3 ×8,
`ASCEND_RT_VISIBLE_DEVICES=0,1`, Qwen3.5-27B-w8a8-mtp, TP2, `qwen3_5_mtp` num_spec 3,
`FULL_DECODE_ONLY` capture `[1,2,4,8,16,32,64]`, `--async-scheduling --quantization ascend
--max-model-len 250000`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`: engine dies in
`compile_or_warm_up_model` while capturing decode graphs with `expected src to have a size
equal to the slice of self. src size = [64, 5120], slice size = [32, 5120]`, and the
reporter's own claim is "works on v0.18.0rc1". The recorded run narrowed it to the family
correctly — layer 0 of `patch/worker/patch_qwen3_5.py` allocating `ceil(tokens/tp)` rows
and `ops/gdn.py:136` `output[:num_tokens], _ = self.out_proj(...)` as the linear-attention
writer, with `_get_row_parallel_op`'s `sp_row_prefixes` deciding who reduce-scatters — then
reversed the message direction, could not find `vllm/model_executor/models/qwen3_5.py`
(the pin is older than the paired vllm) and hit the 16 384 output cap with **no message**.
Page: `../feature-enable-recipes/sequence-parallel-flashcomm.md` §*v0.19.x hybrid layers*.

**#18041** vllm-ascend v0.27.1rc1 (`3b31886`) / vllm v0.27.1 (`6e448d0`),
Atlas 800 A3, DeepSeek-V4-Flash(-w8a8) two-node 1P1D from the
`/zh-cn/latest/tutorials/models/DeepSeek-V4.1-Flash.html` page: the branded
`deepseek-v4.1-flash-a3` image will not start and `v0.27.1rc1-a3` does. The
substitution at the pin equals the working tag, so the build was not stale; the
defect is a literal model-branded tag in prose beside the templated command
block, no release-scope line for a section outside the release's claim table, and
the `zh_CN` `.po` msgstr that will keep printing the old tag after an
English-only edit. Delivered with the gap stated in the first message (the
`V4.1` page does not exist at the pin) —
`../tutorial-verification/docs-version-space-and-image-tag.md` §6.

**#1774** vllm-ascend `v0.7.3.post1` (`c69ceaca1ecc`) + vllm `v0.7.3`, 910B,
`CANN 8.1.RC1 / torch-npu 2.5.1 / torch 2.5.1`, install by
`COMPILE_CUSTOM_KERNELS=1 python setup.py install`: the AscendC kernel build reaches
93 %, then `ascendc_pack_kernel`'s `recompile:` line drives `ld` into
`multiple definition of 'rope_custom_true_half'` / `'rotary_embedding_impl'` and
`setup.py:250` `cmake --build` dies. The deciding artefact is inside that printed
command: the **same** `csrc/kernels/pos_encoding_kernels.cpp` appears as two `.o`
files under two absolute prefixes of one tree (`/efs/<user>/projects/...` beside
`<HOME>/work/<user>/projects/...`), i.e. one tree compiled under two path spellings —
not a kernel, C++ or version defect (`docs/source/installation.md:15-17` at that pin
lists exactly the reporter's CANN/torch-npu/torch). Rung 1 that ended it: unset
`COMPILE_CUSTOM_KERNELS`, `rm -rf build`, `pip install --no-build-isolation -e .`
(`setup.py:217-218`, `:297-298`, then `platform.py:28-38` warns that all custom ops
are disabled); the reporter's public reply is 「Thanks a lot. Solved!」 with no
statement of which rung, so the path-alias mechanism stays open — CANN's
`ascendc.cmake` / `recompile_binary.py` are not in any snapshot. Delivery: 16
assistant turns, 39.7 K characters of reasoning against 4.1 K of message text and
15.6 K of the 16.4 K completion-token cap, so the answer arrived on the last turn and
only just; the version envelope and rung 1 were available from the paste plus one
docs grep. Page (new): `../image-dependency-faults/custom-op-build-stage.md`.

**#2838** `[Installation][v0.9.1]`, reporter quotes his own two links: README
`torch-npu>=2.5.1.post1` vs `pyproject.toml` `torch-npu==2.5.1.post1`, and asks
whether the pin is mandatory. No error, no log, no engine — a doc-versus-metadata
question, and the answer is that the `==` is the enforced contract and the README
prose is the defect: at the snapshot pin (v0.10.1rc1 `7e16b4a7cdb1`) the two exact
files name each other as the mirror pair (`pyproject.toml:2`, `requirements.txt:1`)
while the README is not in it, and `setup.py:331-341` feeds `requirements.txt` into
`install_requires` (`:388`) beside the same pins in `[build-system] requires`. The
recorded run answered in 13 tool calls and flagged that the report's own v0.9.1 tree
is absent from the snapshot; its overreach was deriving *why* the pin is exact
against upstream vLLM's torch pin, which no local file settles. Page:
`../image-dependency-faults/pip-index-visibility.md` (rule 4).

**#6257** vllm-ascend v0.9.2rc1 (`b5b7e0e`) / vllm v0.9.2 (`a5dd03c1`), "离线推理
gemma-3-1b-it 报错" with three screenshots and no log text: the OCR carried an
ACL-graph replay fault (`super().replay()`, `current working operator name is
PagedAttentionOperation`, `ERR33133 PTA call acl api failed`, `RankID: 3`) plus the
launch env (`NPROC_PER_NODE=8`, `VLLM_USE_V1=1`) and code using
`tensor_parallel_size = torch_npu.npu.device_count()`. The answer was right and
arrived late: `Gemma-3 | ❌ | #496` at
`docs/source/user_guide/support_matrix/supported_models.md:26` (Qwen3 `✅` at `:12`)
is the verdict, but the run first grepped `docs/source/user_guide/support_matrix.md`
— a path that does not exist — read the empty result as "no gemma handling", and
spent four turns on head-count arithmetic (`vllm/model_executor/models/gemma3.py:112`
`assert total_num_heads % tp_size == 0`, `:115-123` KV replication) and on the
`attention_v1.py:379` decode branch before finding the row. The public follow-up was
the reporter's "其他错误：" with **four more** failure screenshots — the shape of an
unvalidated model path, and what the answer should have pre-empted. Page (new):
`../feature-enable-recipes/unsupported-model-row.md`.

**#8201** image `vllm-omni:v0.18.0` (`FROM quay.io/ascend/vllm-ascend:v0.18.0rc1`),
vllm-ascend v0.18.0rc1 (`99e1ea0f`) + vllm v0.18.0 (`bcf2be96`), `qwen-image-edit`:
the vllm-omni request path raises a pydantic `ValidationError` on `UsageInfo`
(`type=model_type`) and the reporter had already localised it to
`patch_minimax_usage_accounting.py`, whose `engine_protocol.UsageInfo = UsageInfo`
assignment plus `_rebuild_model_field`/`model_rebuild(force=True)` replaces a public
protocol **class** rather than wrapping a function. The recorded run read the patch,
the upstream `engine/protocol.py:104-109` shape (no `CompletionTokenUsageInfo` in
vllm v0.18.0), the unconditional import (`patch/platform/__init__.py:32`) and the
apply site (`platform.py:138`), and ranked the stale-binding and cached-schema forms of
the `model_type` failure, with the reporter's dict-passing workaround as the
discriminator — then shipped the ownership split (root downstream, workaround in the
consumer) and a no-NPU probe. **Public outcome (reporter, 2026-04-14, on the same
issue):** a maintainer answered "It may be a bug in vllm-ascend … Could you please
submit a PR to vllm-omni as a workaround?", and the reporter replied "done.
https://github.com/vllm-project/vllm-omni/issues/2777" — i.e. the recorded ranking's
ownership and repair split were right, and which of the two failure forms bit him is
still open (his `loc`/`input_type` was never posted). Page (new):
`../upstream-downstream-drift/protocol-model-class-swap.md`.
