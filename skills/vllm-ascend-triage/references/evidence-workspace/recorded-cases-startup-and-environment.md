# Startup, install and environment cases

Engine-start failures (silent stall, worker exit, metadata/capture phase), image and
package faults (`libatb.so`, ABI split, pip index visibility, triton-ascend pair,
published artefacts), SoC and version gates, connector config attributes, tutorial
document cases and `[Community Test]` reports. Routes: `../startup-failures/index.md`,
`../image-dependency-faults/index.md`, `../soc-platform-paths/index.md`,
`../tutorial-verification/index.md`.

Inherits the scope and rules of `recorded-cases.md`, whose index carries one line
per case in this file.

**#8224** image v0.18.0rc1 + vllm v0.18.0 (`bcf2be96`), Atlas 800I A2,
Qwen3-VL-8B-Instruct in graph mode: engine dies in `profile_run` → `_dummy_run`
with `torch._dynamo.exc.Unsupported: Attempted to call function marked as
skipped` (`posix.putenv`) raised from `ops/rotary_embedding.py:500`
`os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"` behind `query_shape[0] > 65535`,
with `--max-num-batched-tokens 81920`. The model is fine and the reporter's
Qwen2.5-VL control never enters `forward_triton` (no `mrope_interleaved`). This
run found the owning line, the `fullgraph=True` wrapper and the coverage gap, and
shipped a complete ranked answer; the dispatch condition, the launch-side rung
and the check that the code is unchanged at v0.23.0rc1 are on
`../startup-failures/env-write-in-compiled-forward.md`.

**#10481** image v0.18.0rc1 (`99e1ea0f`), Atlas A2 with one visible card,
`ASCEND_RT_VISIBLE_DEVICES=7`, Qwen3.5-0.8B: every rank's `WorkerProc` dies at the first
statement of `Worker.__init__` (`worker.py:109 check_ascend_device_type()` →
`utils.py:723 get_soc_version()` → torch_npu `_lazy_init` → `_npu_init`) with
`ERR99999 PTA call acl api failed` + `set default device id failed, ret: 107001`
(`[FUNC:SetDefaultDeviceId]`, `reason=device id error`) + `invalid device ID` and the
`107002` `ctx is NULL` cascade. `107001 = ACL_ERROR_RT_INVALID_DEVICEID`; the guard never
compared a SoC, so no serve flag and no model question is reachable from that log. The run
also showed the reporter's entire flag set is verbatim
`tests/e2e/multicard/2-cards/test_qwen3_performance.py:61-82`, which retires the flag
hypothesis in one grep — and it over-reached once, calling the `--device /dev/davinci0 \  #注释`
line a docs-template defect although both pinned trees list `davinci0` … `davinci7` on clean
continuations (`installation.md:434-441`). The shell behaviour itself is real (the `#` ends
the `docker run`, so only `davinci0` is mapped and the `-v` lines fail as separate
commands), which is why `ls /dev/davinci*` inside the container is the first check —
`../startup-failures/device-id-invalid-at-worker-init.md`.

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

**#2064** vllm-ascend reported as `v0.9.2rc1(e7d32ed3f18d)` with vllm v0.9.2, image
path `/__w/vllm-benchmarks/.../vllm-empty/vllm/` (a source build), OpenGVLab/InternVL3-9B,
`--trust_remote_code`: eager dies at worker import with `ImportError: cannot import name
'bind_kv_cache' from 'vllm.v1.worker.utils'`; graph mode dies earlier, in
`check_and_update_config` → `update_aclgraph_sizes` → `get_max_hidden_layers` →
`hf_config.to_dict()`, inside the checkpoint's own `configuration_internvl_chat.py:87`
(`AttributeError: 'InternVLChatConfig' object has no attribute 'llm_config'. Did you mean:
'sub_configs'?`). Track: version **pair** plus a composite-config read. The reporter's
`utils.py:292/321` and `model_runner_v1.py:71` match the **v0.10.0rc1** tree (`utils.py:291-292/321`, `model_runner_v1.py:70-71`) to within one line
(`4604882a3e2c`), whose vllm counterpart is 0.10.0 — where `bind_kv_cache` moved to
`vllm/v1/worker/utils.py:120` (vllm 0.9.2 has it at `vllm/v1/utils.py:278`, and v0.9.2rc1's
`model_runner_v1.py:66` imports it from there). So the printed tag is only the nearest tag.
The recorded run answered "your build is not the release tag" and called the graph half a
residual hypothesis; the sibling snapshot settles it — `get_max_hidden_layers` exists at
v0.10.0rc1 and the flat read is recorded raising for InternVL2_5-38B at v0.9.0rc2 (#1276,
`utils.py:134`), so the class is real at a pinned release. Pages:
`../startup-failures/environment-correspondence.md`, `../startup-failures/vl-layer-count-attribute.md`.

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

**#986** image `quay.io/ascend/vllm-ascend:v0.8.5rc1-openeuler` (tree v0.8.5rc1
`ec27af34`; the snapshot's vllm pin is v0.6.6.post1, **not** the 0.8.x in the image),
`DeepSeek-R1-bf16-hfd-w8a8` from modelers.cn, 2 nodes, TP8 × PP2 + ray,
`--enforce-eager`, **no** `--quantization` flag, the checkpoint's
`quant_model_description.json` hand-merged into `config.json` per #856: all 16 ranks
die while loading (shards `2/157`) at upstream
`vllm/model_executor/models/deepseek_v2.py:788 param = params_dict[name]` with
`KeyError: 'model.layers.23.mlp.experts.w2_weight_offset'` on the PP0 ranks and
`layers.40…` on PP1. Track: the index-free name is upstream's own remap of a
per-expert key (`make_expert_params_mapping` + `name.replace(…, "experts.w2_")`), and
at this ref only `W8A8_DYNAMIC` allocates MoE offsets (`w8a8_dynamic.py:417,426`),
the static `W8A8` MoE path raising `NotImplementedError` before load
(`quantizer.py:266-270` + `:210-212`) — so the experts resolved to the unquantised
method, i.e. the description in effect did not say `W8A8_DYNAMIC`. The recorded run
read the fused name as proof of a stacked msmodelslim export, then walked
quantizer/create_weights, and ended **output-truncated at 16 384 tokens with zero
answer text** (913 K cumulative prompt tokens); it never asked for `ls` of the model
dir or the experts' description entry. Page:
`../startup-failures/weight-scale-keyerror-quant-variant.md` (`.weight_offset`
section).

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

**#4206** image `quay.io/ascend/vllm-ascend:v0.11.0rc0` (snapshot pin: vllm-ascend
`v0.11.0rc1` `c5fe179` + vllm `v0.11.0` `b8b302c`), Qwen3-Next-80B-A3B, body in
Chinese: start dies in `create_model_config` with
`pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig` /
`Value error, Model architectures ['Qwen3NextForCausalLM'] failed to be inspected.
Please check the logs for more details.` — the pydantic wrapper around #3325's crash,
with no `[registry.py]` line pasted at all. Track: the model-inspection look-alike,
**not** the version gate (`../startup-failures/model-type-not-recognized.md`). What the
pin settles: `failed to be inspected` is raised only for an architecture that is
registered (`_raise_for_unsupported`); the replacing class is downstream
(`vllm_ascend/models/__init__.py:46-48` → `CustomQwen3NextForCausalLM`), so the
subprocess import goes through `vllm_ascend/models/qwen3_next.py`, which at rc1 imports
**only** upstream `vllm.*` names and no `vllm_ascend` internal module; and Qwen3-Next
was supported from v0.10.2 and "stable" at `v0.11.0rc0` (`release_notes.md:22`), so the
architecture is not the question. Delivery: **no answer at all** — 37 messages, every
turn after the opener a bare tool call, the whole output budget spent on two dead
lanes: a missing `triton` ranked first off the tutorial's mandatory `triton_ascend` +
BiSheng install step, then refuted against `find_spec`/`HAS_TRITON` placeholders four
turns later; and reconstructing the `v0.11.0rc0` image's `qwen3_next.py` from the
`v0.11.0rc1` export, which a one-commit snapshot cannot contain. Owed at turn 5: the
classification, one `python -c "import vllm_ascend.models.qwen3_next"`, and the ask for
the `[registry.py]` lines.

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

**#5484** vllm-ascend v0.11.0 image (`quay.io/ascend/vllm-ascend:v0.11.0-my`, a
`pip install vllm-ascend==0.11.0` build over the CANN 8.2.rc2 base, tree `2f1aed98`),
DeepSeek-V3.2 `w8a8-QuaRot` on 2 × A2 (910B) DP2/TP8/EP with
`torchair_graph_config` `{enabled, graph_batch_sizes:[16]}`: the torchair compile dies in
GE **shape inference** — `RuntimeError: E89999: Inner Error!` + `op[Transpose_5],
attr[perm], has wrong value[1], it should be less than input shape size[0]` +
`Call InferShapeAndType for node:Transpose_5(Transpose) failed` + `[Compile][Graph]Compile
graph failed … [FUNC:ge_api.cc][LINE:1365]` — from `_compile_torchair_graph` →
`_dummy_run(is_torchair_compile=True)` (`torchair_model_runner.py:286,245,223` at v0.11.0),
i.e. host-side before any kernel. Verified locally: `ascend_config.py:285` refuses
`torchair_graph_config` + `--enforce-eager`, so the graph-off ablation is deleting the
key; the pin has **no** rotation handling (`rg -il 'quarot|qua_rot|hadamard' vllm_ascend/`
→ 0) while `docs/source/tutorials/multi-node_dsv3.2.md:298-340` validates
`vllm-ascend/DeepSeek-V3.2-Exp-W8A8` on exactly this DP2/TP8 shape; the
`graph_batch_sizes` torchair fix (#2760) predates the pin, and no `tests/e2e/**` case
compiles DS3.2 in torchair mode. `cann/ge` (the error's own files) and the
`torch_npu/dynamo/torchair` package are both outside the case view's exports, so the
producing `.permute` cannot be sourced locally. Recorded run: verdict text at turn 35,
then ~10 calls hunting transpose sites, and a final 10.9 K-character reasoning block —
1.2 K characters of message text, cut at the cap with the ranked answer unsent.
Reporter's public follow-up refutes the upgrade rung: `0.11.0`, then an
`0.11.0-dev` source build, same error. Page
`../startup-failures/torchair-ge-infershape-rejection.md`. Open: which node produced the
rank-1 operand (decided by the compile's GE graph dump) and whether the rotated
checkpoint's `quant_model_description.json` is the owner.

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
framework-free `torch.npu.set_device(0)` probe, plus — on a run whose `docker run`
already matches the docs — the same command + `--privileged=true`. Semantics established
here, not by a reported fix: `SetPrecisionMode` only *supplies* the precision string for
the SoC it resolved, so the numeric code is not a flag problem. The 910B2 sibling **#302**
(below) is the one case in this family with a reporter-confirmed ending, and it is a
container flag: keep the driver-pairing theory under the privilege re-run.

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

**#11498** vllm-ascend release/v0.23.0 (`79928c2`) + vllm v0.23.0, DeepSeek-V4-Flash
w4a8 recipe on 4 cards, TP4 + `--enable-expert-parallel`,
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` + `enable_flashcomm1: true`, DP1, PP1: engine
start dies in `profile_run` with `RuntimeError: Promotion for Float8 Types is not
supported, attempted to promote Float8_e4m3fn and Bool` at
`ops/fused_moe/token_dispatcher.py:393 topk_weights = topk_weights * mask`. Track:
W4A8-MXFP activations (quantised only on the `_prepare_with_ep_group` path, i.e.
flashcomm1/SP) cast the routing weights to fp8 at `w4a8_mxfp4.py:202`, and the
AllGather dispatcher multiplies them by a bool EP mask. The run derived all three
conditions correctly and named the right owning line, but exhausted its response
room inside its own reasoning and shipped no answer — every mechanism it needed was
three `sed` reads plus one ablation. Page now holds them:
`../startup-failures/fp8-topk-weights-mask-promotion.md` (new).

**#12675** image v0.23.0rc1, DeepSeek-V4-Flash, TP8 + EP on one 8-card node,
`--dtype bfloat16`, `--block-size 128`, `--additional-config` with
`enable_dsa_cp`/`multistream_overlap_shared_expert`: every rank dies during weight
load at `vllm_ascend/models/deepseek_v4.py:1535 param = params_dict[name]` with
`KeyError: 'model.layers.<n>.self_attn.wkv.weight_scale'` /
`…self_attn.indexer.wq_b.weight_scale` — a different layer per rank, only two
module families across eight ranks. Track: the reporter is serving a plain
`deepseek-ai/DeepSeek-V4-Flash` directory (`-v /data/models/DeepSeek-V4-Flash:/root/.cache`,
`vllm serve /root/.cache`) with **no** `--quantization ascend`, while the tutorial
lists only the quantised `DeepSeek-V4-Flash-w8a8-mtp` (and the PR test the w4a8
sibling) and every documented serve line carries the flag. The recorded run
inferred "the model must be quantised, so exactly these two modules are skipped in
code" from the short key set and spent its budget in `modelslim_config.py`
(`is_layer_skipped_ascend` → `create_scheme_for_layer` → `quant_prefix_mapper`)
without ever asking for the serve command; the answer arrived in the reporter's own
later paste. Page: `../startup-failures/weight-scale-keyerror-quant-variant.md`
(new), whose first step is the scale-key census
(`scripts/checkpoint-key-prefix.py --scales <dir>`).
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

**#1579** vllm-ascend v0.9.1rc1 (`c30ddb8`), vllm v0.9.1
(`b6553be`), torch_npu `v7.1.RC1.alpha002` (`d33df55`), image
`quay.io/ascend/vllm-ascend:v0.9.1rc1-310p`, one 310P node × 8, Qwen3-32B dense,
`--tensor-parallel-size 8 --enforce-eager --dtype float16`, no `--max-model-len`
(40960 default). The paste opens with a passing `hccl_test all_gather_test`; engine
start then dies in the memory-profiling window: ranks 1-7 raise
`EOFError: Ran out of input` at the same second from
`worker/worker.py:288 determine_num_available_blocks` → `model_runner.py:1205
profile_run` → `:1472 compute_logits` → `logits_processor.py:99 _gather_logits` →
torch_npu `distributed_c10d.py:126` `dist.broadcast_object_list` →
`_tensor_to_object`, and the driver follows 119 s later with `HcclAllgather` `EI0006`
socket timeout / `DestLink` (the collective it was left waiting in). Track: the
gather **emulation**, not the fabric — HCCL has no `gather`, so `_gather` broadcasts
the per-rank sizes first and the non-destination ranks unpickled an empty payload;
the V0 notice above it is upstream `_warn_or_fallback` on non-CUDA hardware and
`VLLM_USE_V1=1` is a live lever (`supports_v1()` returns True there). The recorded run
read the torch_npu frame by turn 5, then spent nine turns inside torch's own
`distributed_c10d.py` — which this snapshot does not contain — and ended at the output
cap with zero answer text (14 turns, 16 384 completion tokens of reasoning). Public
follow-up: a third party reported the same setup starting with `VLLM_USE_V1=1`, the
reporter said they would try and later reported it **unresolved**, having moved
Qwen3-32B to MindIE on a 310P T18 image. Page (new):
`../startup-failures/profile-run-logits-gather-eof.md`.

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

**#302** image `quay.io/ascend/vllm-ascend:v0.7.1rc1` (`Dockerfile:17` → CANN
`8.0.0-910b`), 910B2 ×8, driver 24.1.rc3, `vllm serve Qwen/Qwen2.5-7B-Instruct
--max_model_len 26240` in a `docker run` that repeats `docs/source/quick_start.md:15-30`
mount for mount (`davinci0`, `davinci_manager`, `devmm_svm`, `hisi_hdc`, `dcmi`,
`npu-smi`, the driver's `lib64/` + `version.info`, `/etc/ascend_install.info`,
`/root/.cache`, `-p 8000:8000`): the engine dies at `init_device` → `torch.npu.set_device`
with `AclSetCompileopt(ACL_PRECISION_MODE) 500001` + `PlatformInfoInit … SocVersion =
Ascend910B2` + `GEInitialize failed` (`ERR99999`), no `vllm_ascend` business frame and no
`Starting to load model`, and the printed SoC matches the card. The recorded run answered
with five ranked causes and the right exclusions (`--max_model_len`, `PYTORCH_NPU_ALLOC_CONF`,
`VLLM_USE_MODELSCOPE`, the `No platform detected … UnspecifiedPlatform` line, the expected
`vllm._C`/Triton absences) but ranked the driver/firmware × image-CANN pairing first and
put the container lane third — as `--privileged` **plus** an extra whole-driver mount, two
variables at once. The reporter's own closing comment the next day was
「已解决，加个--privileged=true即可」: that one flag on the unchanged command, no driver or
CANN change. The ordering is the lesson, not the exclusions — the documented mount list is
a minimum, and the privilege/device set is the first rung
(`../startup-failures/acl-init-500001-and-soc-version.md`).

**#2960** vllm-ascend v0.9.0rc2 (`8dd686d`) + vllm v0.9.0 (`5873877`), `Qwen2.5-32B-Instruct-GPTQ-Int8`
served as a capability question ("请问是否支持…?"): the log's engine args already show
`quantization=ascend` with no `--quantization` on the command line, and startup dies at
`vllm_ascend/quantization/quant_config.py:123 is_layer_skipped_ascend` with
`KeyError: 'model.layers.0.self_attn.q_proj.weight'`. The recorded run derived the chain
correctly — the platform list is `["ascend"]` alone (`platform.py:67`),
`override_quantization_method` renames any declared method once the NPU is up
(`quant_config.py:82-86`), and upstream's `get_quant_config` passes the GPTQ
`quantization_config` dict straight into `AscendQuantConfig` as `quant_description`
(`vllm/model_executor/model_loader/weight_utils.py:154-165`), so
`quant_model_description.json` is never read — and answered "not supported; only
msmodelslim-exported W8A8/W8A8_DYNAMIC", with the model itself supported. Accounting:
15 assistant turns, the verdict only on the last one (3.7 K characters against ~30 K of
reasoning), several calls spent re-deriving that chain from upstream source and reading
`weight_scale` precedents that are a different signature. Section C of
`../startup-failures/quantization-method-gate.md` now carries the two-call version.

**#902** vllm-ascend v0.8.5rc1 (`ec27af3`) + vllm v0.8.5.post1 (`3015d56`), image
`quay.io/ascend/vllm-ascend:v0.8.5rc1`, QwQ-32B-w8a8 TP4, `--max-model-len 32768`:
`vllm serve … --quantization ascend` exits with `vllm serve: error: argument
--quantization/-q: invalid choice: 'ascend' (choose from 'aqlm', 'awq', …, 'torchao',
None)` and nothing else — no `APIServer pid=` line, no worker, no device or weight text.
Parse-time lane, not the platform gate of §A: `--quantization` is added with
`choices=[*QUANTIZATION_METHODS, None]` (`vllm/engine/arg_utils.py:600-604`), a copy of
upstream's static `QuantizationMethods` `Literal`, and the plugin hook that is supposed to
extend it runs *afterwards* (`:1640-1646`). At this pin `NPUPlatform.pre_register_and_update`
takes `parser` and never uses it (`vllm_ascend/platform.py:66-75`), while
`supported_quantization = ["ascend"]` (`:61`) and the offline path both accept the method:
`ModelConfig._verify_quantization` gates on `supported_quantization = QUANTIZATION_METHODS`
(`vllm/config.py:752-753`), which the import-time `@register_quantization_config("ascend")`
(`vllm_ascend/quantization/quant_config.py:41`) has already extended because
`EngineArgs.create_engine_config` calls `pre_register_and_update()` (`arg_utils.py:1096`)
before `create_model_config()` (`:1099`). Failing and fixed states verified across pins:
v0.17.0rc1 (`e20f0b1a`) rewrites the choices through
`parser._option_string_actions.get("--quantization")` (`platform.py:134-147`, pinned by
`tests/ut/test_platform.py:54-100`). The run delivered a complete ranked answer in 9
assistant turns (evidence table, the action ladder — drop the flag when the checkpoint
declares `quant_method`, import the config module before `make_arg_parser`, the offline
`LLM(quantization=...)` spelling, upgrade — and the unverifiable "first fixed release"
left open), and its own closing note flagged the
gap this page then had. Route: `../startup-failures/quantization-method-gate.md` §D.

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

**#4696** vllm-ascend v0.11.0rc0 (`00ba0710`) + vllm v0.11.0rc3
(`8ce5d319`), Ascend 910B: `[Doc]` report that the Qwen2.5-Omni tutorial's
"Graph Mode / full graph" section cannot be enabled, with the serve log carrying
`CUDAGraphMode.FULL cudagraph_mode is not support on NPU. falling back to NONE`.
The pasted line is the plugin's own `else` branch, the legacy `full_cuda_graph`
is what lands there, and the tree's unit test asserts the fallback — so the doc's
command is the defect and `FULL_DECODE_ONLY` is the replacement. The run delivered
a complete ranked answer including the
release note that already recorded the failure on A2 (910B1, #2182); the one
correction the library now carries is the quote-style point (the Python-literal
dict parsed, so it is cosmetic) — `../startup-failures/graph-mode-not-applied.md`
(`FULL` section).

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
