# Image and dependency faults (before any vllm-ascend code runs)

## Scope

Failures owned by the Ascend/vllm-ascend **image or its build recipe**, not by
vllm-ascend Python code: the interpreter aborts inside a native module the image
installed (`mooncake`, `lmcache`, `torch_npu`, a `libascend*` transport), a
system library or tool the project's own tutorials prescribe is absent, or a
dependency list drifted because the Dockerfile runs a *copied* helper script.
Also the **hand-built** environment that follows the documented install steps
instead of using the image: the image installs and sources components (CANN,
NNAL/ATB, driver) that a manual build must reproduce itself, and a missing
`source` line looks exactly like a missing package.
The same ownership test covers the **Triton-Ascend compiler pair**: a kernel that
fails to *compile* (`MLIRCompilationError`, `bishengir-compile … returned non-zero
exit status 1`) belongs to the wheel/compiler that was installed into the image,
not to vllm-ascend Python or the kernel's shapes. The same wheel's **runtime**
behaviour is here too: a Triton-Ascend extension op (`extract_slice`) or a compiled
kernel (a per-head reduction whose scalar scale becomes 0) that runs and returns wrong
values is owned by the wheel and its test coverage, with the tree's own kernels audited
separately (`triton-extension-op-silent-miscompute.md`).
It also covers the **host C++ compile of a generated translation unit** that fails
inside a *torch_npu* or *CANN* header (`use of undeclared identifier
'aclOpExecutor'` at `AclOpCompileInterface.h`, `fatal error: 'cstdint' file not
found` in `kernel_macros.h`): the failing line is in the wheel's headers, the
owner is the wheel × CANN header pair, and the compiler is the messenger.
It also covers **published-artefact availability**: when the tag or wheel the
reporter needs was never built for their CPU architecture or SoC variant, the
owner is the image/wheel pipeline, and the answer is the tag suffix plus one
registry manifest read — not a vllm-ascend code path.
The same pipeline owns the **host CPU's instruction-set level**: an aarch64 wheel
built on an ARMv8.2 baseline dies with SIGILL on an ARMv8.0 core before any Python
frame — `host-cpu-isa-sigill-at-import.md`, not a vllm-ascend defect and not the card.
It also owns the **interpreter's own resolution residue**: what `import vllm`
actually binds on the reporter's box when a checkout, a stray directory or an
editable `.pth` wins over the install (the `(unknown location)` form is
`wrong-vllm-module-shadowed.md`; the missing-module form is
`plugin-load-module-missing.md`).
Inherit the rules in SKILL.md; this topic is the counterpart of
`../startup-failures/index.md` for failures that never reach a worker.

## Routes

- A **hub id is served on an image that sets `VLLM_USE_MODELSCOPE`**, the weights
  load, and the engine then dies with the hub's own
  `modelscope.hub.errors.NotExistError: The model: <id> has no revision: main !`
  from `modelscope/utils/hf_util.py` while building the multimodal dummy prompt
  (`profile_run`). vllm patches the hub globally
  (`vllm/transformers_utils/__init__.py:3-19`) and every download it makes itself
  passes `revision`, but the **processor** load does not
  (`processor.py:53-69,101-111`; `config.py:686-693` "ModelScope does not provide an
  interface for image_processor"), so modelscope mints the HF default branch `main`
  against a `master` repo. Rung 1 is the snapshot directory the log itself prints
  (`Downloading Model to directory: /root/.cache/modelscope/hub/<id>`), not a flag:
  `modelscope-hub-patch-processor-revision.md` (recorded #829, v0.7.3).

- The follow-up is a **sourcing question**, not a new stack: "0.10.1 需要安装新的东西吗",
  "用相同环境把 vllm 从 0.10.x 改回 0.9.x 就能正常启动", "请问哪里可以看到你们镜像制作的
  依赖，我对比下", often with "只能访问 docker 容器，接触不到主机". The image's dependency
  set is four artefacts at the pin — the `FROM` CANN base image (which is what supplies
  CANN **and** NNAL/ATB), `ARG VLLM_TAG`, the `--extra-index` URLs that serve
  `torch-npu`, and the `devlib` `LD_LIBRARY_PATH` append no doc reproduces — while the
  old-version success is a control only if `torch`/`torch_npu` moved with it
  (`nnal-atb-missing-library.md` for the missing component itself) →
  `image-recipe-inventory-and-diff.md`; one call:
  `scripts/image-recipe-inventory.sh <vllm-ascend-root> --soc 910b --image-tag <tag>`.
- `python -c "import <third-party module>"` (or the first real use of it) dies
  with a glibc allocator message — `corrupted size vs. prev_size`,
  `free(): invalid pointer`, `double free or corruption`,
  `munmap_chunk(): invalid pointer`, `Aborted (core dumped)` — with no Python
  frame and no NPU/collective text → `native-import-abort.md`.
- `import torch` (or any image-bundled `.so`) dies with **`Illegal instruction
  (core dumped)`** — SIGILL, no Python frame, no engine log, usually a 310P box on a
  Phytium D2000/FT-2000-class ARMv8.0 host — and the reporter's control is a lower
  torch line that imports → `host-cpu-isa-sigill-at-import.md`. The owner is the
  wheel's ARMv8.2 baseline against an ARMv8.0 core (reporter-confirmed on #13936:
  「d2000只支持armv8 … torch.2.10默认使用armv8.2的指令集编译的」). One host-side
  `dmesg -T | grep -i 'trap invalid opcode'` names the `.so`, `/proc/cpuinfo`
  `Features` (or `LD_SHOW_AUXV=1`) names the missing extension, and the rung is the
  last **release** line before the 2.10 bump (`release_notes.md:316`, v0.20.2rc1) — never
  `pip install torch==2.9.0` under the newer `torch_npu`; a nightly or `-openeuler` tag
  of one release re-dispatches that branch's own `requirements.txt`, so a newer/nightly
  tag is not an alternative rung. The device-side aicore
  `Illegal instruction, which is usually caused by unaligned UUB addresses`
  (`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`) shares the words and
  nothing else.
- The startup dies in a vllm-ascend **inductor fusion pass** with
  `'_OpNamespace' 'vllm' object has no attribute '<op>'` (`qknorm_rope_fusion_pass.py`,
  `torch/_inductor/pattern_matcher.py` frames), and `pip list` does show a triton wheel →
  not a missing package: the official images **remove triton on purpose** (every
  `Dockerfile*` at this ref has the same `python3 -m pip uninstall -y triton` under
  the comment "in Ascend, triton doesn't work correctly"), so `HAS_TRITON` is false and every `if HAS_TRITON:`-registered custom op
  is absent by design while its ungated consumer still runs. Grep the log for
  `Triton not installed or not compatible`, then turn the pass off →
  `../startup-failures/missing-triton-op-in-compile-pass.md`.
- The reporter compares **two implementations of one op in the tree** rather than
  reporting a crash — "rotary_embedding.py 中 Triton 实现和 ascend 实现结果不一致，结果应该是一样
  的，但是这里会有乱码", both `triton` and `triton-ascend` listed in the environment, no
  traceback. The branch is `if HAS_TRITON:` at the call site (install state: one importable
  `triton` **and** exactly one active backend driver), the recipe removed triton on purpose
  and pinned triton-ascend, and the Triton kernel's own TODO names a two-model envelope —
  answer with that boolean (`python -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"`),
  `pip show triton triton-ascend`, and the same-tensor A/B, and keep the
  output-garble cause a hypothesis → `triton-install-state-op-dispatch.md`.
- `import torch` itself dies with `AttributeError: module 'triton' has no
  attribute 'language'` at `torch/_dynamo/utils.py` (`common_constant_types.add(
  triton.language.dtype)`), reached from `torch_npu/utils/_dynamo.py`, with **no
  `vllm_ascend` frame** and a `-310p`/310P box in the dump → `triton-module-residue.md`.
  A complete triton or triton-ascend binds `language` in its own `__init__`
  (`python/triton/__init__.py:26` / `:25`), and the `-310p` Dockerfile uninstalls
  both on purpose (`docs/source/tutorials/hardwares/310p.md:7`), so the module
  that resolved is a leftover/shadow, not a version-pin problem: one `python -c`
  printing `triton.__file__` / `__path__` / `__version__` decides it, and the
  repair is removal, never `pip install triton-ascend`.
- The log carries `Failed to import Triton kernels. Please make sure your triton
  version is compatible. Error: No module named 'triton.<x>'` (recorded
  `triton.language.target_info`), often beside the reporter's own history of
  `module 'triton' has no attribute 'next_power_of_2'` and a `kernel[grid]` launch
  raising `TypeError: 'function' object is not subscriptable`, on a box where
  triton/triton_kernels was hand-installed → `triton-kernels-import-error.md`.
  That title sentence is upstream vllm's `logger.error` around a swallowed
  `triton_kernels` import (`fused_moe/config.py:25-34` at vllm v0.18.0), **not** the
  crash, and its advice is the action already tried: on an Ascend box the only valid
  `triton` is the one triton-ascend's wheel provides, so the fix is removal of the
  foreign distributions plus one `python -c` probe, never another triton version.
- The **platform plugin fails to import**: `vllm serve` prints
  `ERROR Failed to load plugin ascend` + `ModuleNotFoundError: No module named
  'vllm_ascend'`, then `No platform detected, vLLM is running on
  UnspecifiedPlatform`, and dies with `RuntimeError: Failed to infer device type`
  although `pip show`/the env dump lists `vllm_ascend <ver>` →
  `plugin-load-module-missing.md`. The title error is downstream (verified at vllm
  v0.7.3 `config.py:1654,1660` / `platforms/__init__.py:210-212` /
  `plugins/__init__.py:26-46`), and entry-point discovery and `plugin.load()` share
  one `sys.path`, so it is a metadata-vs-module residue: editable install whose
  checkout moved, two interpreters on one host, or a source-built `vllm`
  (`0.7.3+empty`) beside another env's install. One `python -c "import
  vllm_ascend, sys; print(sys.executable, vllm_ascend.__file__)"` in the serving
  env decides it; the device, CANN and driver are innocent.
- The same title error, but the plugin **loaded and then its hook raised**:
  `INFO … plugin ascend loaded.`, no `ERROR Failed to load plugin ascend`, no
  `Platform plugin ascend is activated`, then `No platform detected …` →
  `platform-plugin-register-raised.md`. vllm v0.7.1 calls the hook inside
  `except Exception: pass` (`platforms/__init__.py:153-161`), so the traceback is
  gone from the paste; at v0.7.1rc1 the hook's first statement imports the quant
  config, whose bare `import torch_npu` makes the serving shell's CANN env the
  first suspect. One command recovers it:
  `python -c "import vllm_ascend; print(vllm_ascend.register())"`. The pin's
  `release_notes.md:33` lists `No platform detected` / the `vllm._C` warning as
  harmless — quote it to pre-empt the `vllm._C` follow-up, never to close the raise.
- A serve installs from a **wheel** and dies *after weights load* with
  `ModuleNotFoundError: No module named 'vllm_ascend.<sub>'` while a container of the
  same release works — the frames are `vllm/compilation/backends.py:298` →
  `vllm/utils.py::resolve_obj_by_qualname`, i.e. a qualname the engine itself asked
  for. The subpackage has no `__init__.py`, so `find_packages()` leaves it out of the
  wheel; two greps at the pin decide it, `--enforce-eager` is only the bridge, and the
  reporter-confirmed remedy is the next rc or a source install →
  `missing-subpackage-after-wheel-install.md` (#1156). Distinguish it from the
  top-level `No module named 'vllm_ascend'` two bullets above and from
  `vllm_ascend.vllm_ascend_C` (never built, `../soc-platform-paths/kernel-package-vs-card.md`).
- The whole image boots and the **first use** of `torch.ops._C_ascend.<op>` raises
  `AttributeError: '_OpNamespace' '_C_ascend' object has no attribute '<op>'` after a
  startup `Failed to import vllm_ascend_C … Sleep mode will be disabled` warning →
  `custom-op-extension-missing-from-image.md`. The first use is a request **or** the
  memory-profiling dummy run (`determine_available_memory` → `profile_run`, weights
  already loaded, `/v1/models` never reached), and the **tail of that warning** picks
  the owner before any source read: `No module named 'vllm_ascend.vllm_ascend_C'` = no
  artefact at all (the csrc-cache hit running `COMPILE_CUSTOM_KERNELS=0`,
  `.github/workflows/_schedule_image_build.yaml:225-250`, or the `develop`-only install
  prefix), while `/…/vllm_ascend_C*.so: undefined symbol: <internal symbol>` = the
  artefact is present and `dlopen` refused it, because a SoC branch dropped the TU that
  defines a symbol the unconditional host binding still references (v0.14.0rc1 on 310P:
  `CMakeLists.txt:85-93` against `:116` plus `csrc/torch_binding.cpp:31,1400-1401`;
  recorded #6527). One grep decides whether either lane is fatal here
  (`rg -c 'torch\\.ops\\._C_ascend' vllm_ascend/_310p/`: at v0.23.0.post1 the 310P
  GDN files hold 6 unguarded call sites, so a Qwen3.5/Qwen3-Next checkpoint on 310P is
  the fatal case), and the reporter's own per-tag `vllm_ascend_C*.so` table or one
  `nm -DC … | grep '<symbol>'` is the strongest artefact in the report.
- A `python -c "import <pkg>"` (or the first real use) fails **at symbol
  resolution**, not at open: `ImportError: …/<pkg>/_ops.cpython-310-…so: undefined
  symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs` and the ENV dump selects
  `atb/cxx_abi_1` (or `cxx_abi_0`) → `abi-undefined-symbol.md`. The mangled
  signature names the side (`Ss` = old `std::string` ABI, `NSt7__cxx11` = new), and
  pointing `ATB_HOME_PATH`/`LD_LIBRARY_PATH` at the matching ATB root is the
  fix once step 3 shows torch_npu's own `libop_plugin_atb.so` is on the same
  side — reporter-confirmed on #1144. One call:
  `scripts/abi-symbol-check.sh <paste>`.
- An `ImportError` whose module is **`vllm` itself** and whose tail reads
  `(unknown location)` — `cannot import name 'PoolingParams' from 'vllm'`,
  `from vllm import <X>` raised from an upstream frame such as
  `vllm/engine/multiprocessing/__init__.py`, beside an env dump listing a
  `vllm …+empty` install → `wrong-vllm-module-shadowed.md`. That tail means a bare
  `vllm` directory (no `__init__.py`) was bound and no regular `vllm` package was
  reachable on that `sys.path`; the fix is deleting/renaming the stray directory,
  and `+empty` is the docs' own source-build marker, not a cause.
- A documented command or import fails because something is **missing** —
  `cannot open shared object file`, `No module named`, `command not found` — and
  `docs/source/tutorials/*.md` installs it in its own startup block → same page,
  steps 1 and 3: identify the copy that is loaded, then compare the tutorial's
  and the Dockerfile's ingredient lists.
- The missing `.so` is raised while a **KV-connector module is imported** in a
  deployment started by a launcher (K8s/kserve/operator `args:`/`command:`,
  compose `entrypoint`) — `ImportError: ascend_transport.so: cannot open shared
  object file` from `from mooncake.engine import TransferEngine`
  (`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:24`) — and
  the ENV dump or a profile script claims the directory is already on
  `LD_LIBRARY_PATH` → `launch-env-and-library-path.md` first. Reporter-confirmed
  on #7352: the effective environment of the launched `vllm serve` is the launch
  command's, so the fix is the export inside that command, not a rebuild.
- The missing library is a **separate Ascend component with its own
  `set_env.sh`** — `libatb.so` inside
  `Could not load this library: .../torch_npu/lib/libop_plugin_atb.so`, raised
  from `_register_atb_extensions()` at worker init — → `nnal-atb-missing-library.md`.
  The owner is NNAL/ATB, not CANN: each install root has its own `set_env.sh`,
  and sourcing the CANN one does not export the ATB paths (a later comment on
  #10525 reports that test; the pinned docs list the two `source` commands as
  separate steps). The dump's own `ATB_HOME_PATH`/`LD_LIBRARY_PATH` lines decide
  "installed but not sourced" before any round trip — vllm-ascend's
  `collect_env.py:307` prints every `ASCEND_*`/`ATB_*` variable, so an absent
  `ATB_HOME_PATH=` is a finding, not a gap. Same page carries the reporter-confirmed
  `CANN: None` `collect_env.py` blind spot, the non-fatal `triton_kernels` log
  noise, and — for the 「从零构建这条路很难走通？」 follow-up — the from-zero stage
  order with a gate per stage (the manual CANN block, NNAL included, is a collapsed
  dropdown under the pip tab, which is why a package-by-package follower misses it).
- Engine start (or plugin activation) aborts with a C++ exception and no Python
  frame — `terminate called after throwing an instance of 'std::logic_error'`,
  `what(): basic_string::_S_construct null not valid`, then
  `RuntimeError: Engine core initialization failed … Failed core proc(s): {}` ->
  `native-string-abort.md`: **step 0 is the tutorial tab / image-tag match** — was
  the pasted block the A2 or the A3 section of that model page, and does the log's
  `EngineCore_DP<N>` prefix agree with the pasted `--data-parallel-size` (#11300
  v0.22.1rc1 was closed by its reporter as "A3机器跑了A2的命令"; the same paste
  carried `EngineCore_DP1`/`DP2` against a dp1 command). Run it before writing
  anything when a launch script is in hand: #8745's reporter later posted theirs,
  it is the A2 tab verbatim ("A2单机", unsuffixed `v0.13.0rc3`), and the abort came
  back — that case's tab lane is closed, and at v0.13.0rc3 both tabs export
  `USE_MULTI_BLOCK_POOL`/`OMP_*`, so "missing the A3 exports" is not a cause there.
  Then the stock-model discriminator, the one-env bisect of the recipe
  (`USE_MULTI_BLOCK_POOL=0` first — it selects six extra patch imports right after
  the last printed log line), the three `python -c` native probes, and
  `gdb -ex 'break std::__throw_logic_error'` to name the `.so`. The checkpoint's
  config, processor and architecture are not the owner (#8745 v0.13.0rc3
  DeepSeek-V4-Flash and #10943 v0.20.2rc1 print the same two lines; #10943's
  reporter confirmed the unmodified checkpoint crashes identically and eight
  model-side changes moved nothing).
- The report **is a pip resolution error about a requirement**, with no engine and no
  crash: `ERROR: Could not find a version that satisfies the requirement
  torch-npu==2.5.1.post1.dev20250619`, `No matching distribution found for <pkg>`, or
  a title like "unmatched torch version between vllm=X and vllm-ascend=Y" — at install
  time from the published wheel, or inside a deployed image when adding an extra
  (`ImportError: Please install vllm[bench] …` then `pip install pandas` → `No matching
  distribution found`) → `pip-index-visibility.md` first, **before** any version
  survey (the track decision and its no-failing-frame variants — a pairing story, the
  `[build-system] requires` conflict, the Python floor, a dangling build path, the
  pybind11 rename — are `../dispatch/no-frame-install-shapes.md`):
  `bash scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]` prints the
  `torch` / `torch-npu` / `torchvision` pins of both sides and the contradictions, and
  the page carries the three-way split (mirror-only metadata pin vs unsatisfiable pin
  set vs unreachable index) plus both answer shapes (#1796, #4737) and the **paraphrase**
  face where no pip message is pasted at all — 「安装 vllm 0.9.1 提示卸载 torch 2.5.1 装 2.7.x」,
  then vllm-ascend "could not find torch 2.5.1" on a rented box whose CANN/torch were
  preinstalled (#1363, §A0: four reads, and rung 1 is vllm from source with
  `VLLM_TARGET_DEVICE=empty`, not another pip command). Its sibling case
  is the one where **nothing failed**: a two-command wheel install that only prints
  pip's post-install block (`vllm 0.18.0 requires torch==2.10.0, but you have torch
  2.9.0`, plus torchaudio/torchvision/opencv lines) → `pip-resolver-warning-block.md`
  (which requirement file supplied each pin, and why raising torch is the one thing
  that breaks the install).
- The **title is a Python-version requirement in the reporter's own words** —
  「python version should above 3.10」, "requires python 3.10+" — with `npu-smi`, a
  `conda`/`pip` transcript and a screenshot but **no printed line**, and the install
  itself succeeded. The string is nowhere in either tree (`rg -n 'above 3\.10'`), so the
  deciding line is the screenshot's OCR: at #3657 it is
  `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` inside
  `vllm/model_executor/models/registry.py`'s class body, i.e. a PEP 604 annotation
  evaluated at import on Python 3.9, and the pin's own metadata
  (`requires-python = ">=3.9,<3.14"`, `python_requires=">=3.9"`,
  `installation.md:8` `>= 3.9, < 3.12`) is why pip never refused it. Verdict: recreate the
  env on 3.10/3.11 — and the paste's `conda create --clone` of an old `PyTorch-2.1.0` env
  is where the 3.9 came from → `python-floor-import-error.md`.
- The complaint is a **dependency conflict that asks for a pin change** — "A 依赖 B，
  B 依赖 C …", "能否把 vllm-ascend 依赖的 torch-npu 升级为 2.8.0？", no pip error quoted →
  `dependency-pin-bump-request.md`. Verdict first (no, not for this release), then the
  rc's own row of `docs/source/community/versioning_policy.md` (v0.10.2rc1 `:25`:
  `v0.10.2rc1 | v0.10.2 | … | 2.7.1 / 2.7.1.dev20250724`). Do not grep
  `installation.md` for the version — it prints `|pip_vllm_version|` substitutions and
  returns nothing — and treat the reporter's premise about the *paired* package's pin as
  the open half: one `importlib.metadata.requires('vllm')` line decides it (#3266).
- The complaint is that an artefact **does not exist for the reporter's machine**:
  "找不到支持 linux on arm64 的 310p 镜像", the quay tag page shows an arm64 tag
  while the pull lands on amd64 (`WARNING: The requested image's platform
  (linux/amd64) does not match the detected host platform (linux/arm64/v8) and no
  specific platform was requested`), or "有没有可 pip 装的 <soc> 轮子" →
  `image-arch-and-tag-availability.md`. Published-artefact availability, not a
  vllm-ascend defect: the decisive check is
  `docker buildx imagetools inspect quay.io/ascend/vllm-ascend:<ver>-<soc>` (plus
  the same on the `FROM` CANN base), and the tree's own intent comes from the tag
  suffixes, the conditional `platforms:` matrix (the same expression as the step's
  `push:`, so the single-arch fallbacks — which differ per workflow — only cover
  check builds) and the wheel-release matrix — one call:
  `bash scripts/image-arch-matrix.sh <vllm-ascend-root> --soc 310p`.
- `pip install` **from source** in a pod without public network fails in the
  custom-ops step ("pulls dependencies from the public network", sometimes only
  `fetch failed`): the build chain fetches a `catlass` git submodule plus four
  third-party tarballs from `gitcode.com` at configure time →
  `offline-source-build.md` for the inventory, the pre-seed gates and the two knobs
  that do not work (`CANN_PKG_SERVER`, an exported `CANN_3RD_LIB_PATH`).
- The build **finished** and the whole report is one dangling path —
  `…/csrc/build/compat/liboptiling.so -> lib/linux/aarch64/libcust_opmaster_rt2.0.so 指向无效`,
  often with "the real file is under `_CPack_Packages/…/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/`"
  or "should it be `./build/libcust_opmaster_rt2.0.so`?" — no failing frame, no exit
  code. `compat/` is a **build-tree-only** relative link whose layout exists only after
  `install()` (`csrc/CMakeLists.txt:195-209` vs the absolute `$<TARGET_FILE:optiling>`
  link at `:186-189`), so the staged copy resolves and nothing needs `ln` or a rebuild:
  answer that first, and map the `_CPack_Packages/…/op_tiling/lib/linux/<arch>/` copy the
  body names onto the staged link rather than asking for it back — the one escalation is
  whether that staged link and the `.run` exist (a `cat` on the build-tree link must fail).
  → `custom-op-build-stage.md` §Step 2b (v0.19.1rc1, #10792).
- A **compiler or linker** error in that same source build, with no engine start:
  `ld: … multiple definition of '<symbol>'` / `first defined here`, `collect2: error:
  ld returned 1 exit status`, `ccec_compiler`/`ascendc_pack_kernel` frames, then
  `CalledProcessError` from CANN's `recompile_binary.py` and from `setup.py`
  `cmake --build` — with `COMPILE_CUSTOM_KERNELS=1` exported → `custom-op-build-stage.md`.
  Its sibling is a **missing pta header**, not a linker fault:
  `csrc/torch_binding.cpp:20: fatal error: torch_npu/csrc/core/npu/NPUStream.h: No such
  file or directory` under `CMakeFiles/vllm_ascend_C.dir/`, no `COMPILE_CUSTOM_KERNELS`
  needed, usually `pip install -e .` inside build isolation with the body saying
  「comment the torch-npu in requirements.txt and pyproject」 → same page, §Step 1b:
  `setup.py:203-212` derives `-DTORCH_NPU_PATH` from `python3 -m pip show torch-npu`
  with no error path (empty → `/torch_npu`), `CMakeLists.txt:63` is the only torch_npu
  include root, and `CMakeLists.txt:54` already prints `TORCH_NPU_PATH is …` — that line
  is the check, `--no-build-isolation` is the rung (#3403). Keep the
  "wheel ships no `include/`" lane open until `ls <torch_npu>/include/torch_npu/csrc/core/npu/`
  says which it is.
  The paste's last frames name the **stage**, and the stage names the owner
  (vllm-ascend `CMakeLists.txt`/`setup.py` vs CANN's AscendC toolchain, which is what
  a snapshot does not export); check the reporter's CANN/torch/torch-npu trio against
  the pin's own requirements table **before** any version advice, and read a
  `multiple definition` from the `recompile:` line's own object list — one
  `csrc/kernels/*.cpp` under two absolute prefixes means the tree was compiled under
  two path spellings (one `stat -c '%d:%i'` on the two printed prefixes says alias or
  copy), not that the kernel source is wrong (#1774, reporter-confirmed
  「Thanks a lot. Solved!」 on the skip-the-build rung). Do not open `csrc/<op>/` or the
  CANN toolchain source here.
- The build dies **before any CMake work** in pip's isolated env: `usage: __main__.py
  [--version] [--includes] [--cmakedir] …` + `__main__.py: error: unrecognized arguments:
  --cmake` + `CalledProcessError: Command '[…] '-m', 'pybind11', '--cmake']`, above
  `/tmp/pip-build-env-*/overlay/…`, while the host line reads `Requirement already satisfied:
  pybind11 (3.0.0)` → `pybind11-cmake-option-rename.md`: pybind11 3.0 renamed the option,
  vllm-ascend's own `setup.py` calls the old spelling, and both requirement files leave
  `pybind11` bare, so the isolated env re-resolves it. The pinned `v0.9.2rc1` already uses
  `--cmakedir` (`setup.py:178-180`); a host-side downgrade is inert without
  `--no-build-isolation`. Do not rank the reporter's `cmake` 4.x (`CMakeLists.txt:1` is `3.16`).
- You suspect the image at all: the Dockerfile `cp`s a helper from the repo
  (`Dockerfile.a3:37` copies `tools/mooncake_installer.sh`) and upstream ships
  its own copy of that script. Diff the two dependency lists before blaming the
  component's code; the image builds against the stale copy.
- A request dies in a **Triton-Ascend compile** although the engine started:
  `triton.compiler.errors.MLIRCompilationError`,
  `[ConvertLinalgRToBinary] encounters error: … operation's operand is unlinked`,
  `bishengir-compile … returned non-zero exit status 1`, deepest `vllm_ascend`
  frames in `vllm_ascend/ops/triton/fla/**`, under `triton/runtime/autotuner.py` —
  and the report/launch script contains a `pip install triton-ascend` or an extra
  BiSheng/AscendNPU-IR `.run` over the image → `triton-ascend-pair.md`. The phrase
  "Failed to compile custom operators with Triton Ascend" is usually the reporter's
  paraphrase; it is not the AscendC custom-op lane. **The repair is decided by that
  release's `triton-ascend` pin, not by the launch script**: absent from
  `requirements.txt`/`pyproject.toml` (≤ v0.13.0rc1) means the hand-install `.dev`
  wheel its docs name is required, a final pin (v0.14.0rc1 `==3.2.0`) means the image
  already has it and the manual install is the damage — so read the pin before
  offering "remove the install", and put the release move first.
- A Triton-Ascend **extension op or compiled kernel returns wrong values** although
  the engine is healthy, in either of two forms → `triton-extension-op-silent-miscompute.md`.
  **(a) slicing**: bf16 `extract_slice` at a non-32-byte-aligned start yielding foreign
  values and zeros, `insert_slice`/`extract_slice` pairs in a hand-written kernel, fp16
  "works" but bf16 does not. **(b) a zeroed reduction**: `squares` and
  `tl.sum(…, axis=1)` are correct in the reporter's own dump, a non-identity scalar
  scale of that result (`/ head_size`, `* 2.0`, `* 0.5`) comes back `0` while `* 1.0`
  is fine, it only happens while a later `extract_slice(<that tile>) * cos` consumer
  exists, and only at one head count (kv heads 4 wrong, 2 right) — the reporter often
  arrives with a working temporary-store patch and the wrong owner named. Owner is the
  pinned wheel (`triton-ascend==3.2.1` at v0.23.0rc1, `==3.2.2` at v0.23.0), not
  vllm-ascend Python; the page carries the doc-versus-UT coverage gap, the
  print-path-versus-data-path check, the dtype-versus-offset discriminator, the
  shape × consumer bisect, the in-tree exposure grep and
  `scripts/qknorm-mrope-exposure.sh` (exposure + CI coverage in one call).
- A **host C++ compile of a generated TU** fails inside torch_npu's or CANN's own
  headers: `RuntimeError: Worker failed with error 'Failed to compile
  /tmp/tmpXXXX/launcher_cxx11abi1.cxx …'`, an include chain
  `…/.triton/cache/<hash>/precompiled.h → torch_npu/…/OpCommand.h →
  OpParamMaker.h → interface/AclOpCompileInterface.h:146: error: use of undeclared
  identifier 'aclOpExecutor'; did you mean 'aclopExecute'?`, a note quoting
  `/usr/local/Ascend/cann-<ver>/include/acl/acl_op.h`; or `kernel_macros.h:… fatal
  error: 'cstdint' file not found` (#3135) → `host-cxx-compile-header-conflict.md`.
  `launcher_cxx11abi1.cxx` is the Triton backend's ABI probe, **not** a
  multi-process launcher, and the engine is still in `determine_available_memory`;
  one call: `scripts/acl-header-guard-check.sh --torch-npu-root … --cann-root …`
  decides the two colliding `acl_base.h` copies. Do not triage the checkpoint's
  quantisation or the model here.
- The engine dies inside the **compiled model forward**, naming a symbol from an
  optional accelerator wheel — `torch._dynamo.exc.Unsupported: Graph break due to
  unsupported builtin mindie_turbo._ops.PyCapsule.rms_norm`, frames going up
  through `vllm/compilation/decorators.py` `patched_inline_call` and
  `torch/_dynamo/**` — with `mindie-turbo` installed (often asked as
  "mindie_turbo 算子不支持图模式?") → `mindie-turbo-graph-break.md`. Same wheel as
  the ABI route above, opposite failure mode: it imports fine and both worker
  classes load it **unconditionally** (`worker_v1.py:78-81`,
  `utils.py:172-182`, no flag), so the trigger is [installed] × [compile
  enabled]; the `MindIE Turbo is installed. vLLM inference will be accelerated
  with MindIE Turbo.` INFO line is the probe for *which* process had it, decisive
  when the reporter says another entry path worked, and that ref's compatibility
  matrix has no MindIE Turbo column filled (answer the pairing question from it).
- The **same wheel, no crash, and a quality complaint** — "some results have no
  content / 概率性没有 content" under concurrency (a 40-way eval run is the recorded
  setting), accuracy below the plugin-free run, or "not reproduced after uninstalling
  mindie_turbo", on a release whose compatibility matrix **does** name a plugin
  version → `mindie-turbo-pre-0.8-eager-and-quantizer.md` (#1954, v0.7.3.post1). Read
  the version gate first: at ≤ v0.7.3.post1 `vllm_ascend/platform.py:110-114` forces
  `NO_COMPILATION`, so the graph-break page's compile lanes are impossible there and
  what is left is eager op substitution plus the plugin-owned w8a8 quantizer
  (`quantization/quantizer.py:39-46`; at 0.7.3 w8a8 exists only through the plugin,
  `faqs.md:82`). Rank the response body (`content` vs `reasoning`, `finish_reason`)
  before any plugin mechanism, then the pinned `mindie-turbo==2.0rc1` contract
  (`setup.py:371`, `conf.py:76`, `versioning_policy.md:70-71`) that the issue's own
  bare `pip install mindie_turbo` breaks.
- Nothing crashes and the question is a **version string**: `pip show
  vllm-ascend` reports `0.9.0rc2` while the reporter installed "0.9.1dev", asks
  whether that version exists at all, or asks why the official
  `vllm-ascend:vX.Y.Z-dev` image reports `vllm_ascend 0.1.dev1+g<sha>` /
  `vllm 0.9.1+empty` instead of `X.Y.Z` → `installed-version-vs-checkout.md`
  (the `-dev` images `COPY` the branch and `pip install -e` both projects, so the
  number is a build-time SCM derivation and the image tag/commit is the release
  identity). The
  number is build-time metadata (`setup.py:305-311` `setuptools_scm`, no constant in
  `vllm_ascend/`), no runtime reader consults it, and the pairing gates read
  `vllm.__version__` — so the deliverable is the two output paths, not a survey.

## Pages

- `native-import-abort.md` — the interpreter aborts inside an image-bundled
  native module (glibc allocator text, no Python frame): which copy is loaded,
  allocator interposition, dependency-list correspondence.
- `host-cpu-isa-sigill-at-import.md` — `Illegal instruction (core dumped)` at the
  first import: the host core's ISA level against the wheel's build baseline (D2000 =
  ARMv8.0 vs Kunpeng 920 / D3000 = ARMv8.2), the host-side `dmesg`/`AT_HWCAP`/`gdb`
  round trip, the release line that last carried torch 2.9 (`release_notes.md:658`,
  bump at `:316`), why the whole image is swapped rather than torch, and the
  ppc64le-gated `mcpu=native` line that is not aarch64 evidence (#13936,
  reporter-confirmed).
- `native-string-abort.md` — `terminate called after throwing … std::logic_error`
  / `basic_string::_S_construct null not valid` at engine start; the tutorial tab /
  image-tag match is step 0, the native probes and the `gdb` owner search follow.
- `nnal-atb-missing-library.md` — `libatb.so` / `libop_plugin_atb.so` unsourced
  NNAL roots, the `CANN: None` collector blind spot.
- `abi-undefined-symbol.md` — the same NNAL/ATB component at the other failure
  mode: a symbol is *missing at resolution* because the wheel and the selected ATB
  root are on opposite sides of the `_GLIBCXX_USE_CXX11_ABI` switch (`Ss` vs
  `NSt7__cxx11`), one process can load only one `libatb.so` SONAME, and the
  reporter-confirmed remedy is the `cxx_abi_0`/`cxx_abi_1` export.
- `plugin-load-module-missing.md` — the out-of-tree platform plugin cannot import at
  all: the metadata/dist-info is present, `vllm_ascend` is not, and vLLM degrades to
  `UnspecifiedPlatform` and then `Failed to infer device type`. The log chain with
  its pinned anchors, the two entry-point names both refs register, the PEP 660
  `.pth`/`egg-link` residue check, the two-interpreter signature that the paste's own
  `site-packages` paths already expose, and the `+empty` marker that says vllm was
  source-built. Worked for #540 (v0.7.3rc2), cause left open.
- `platform-plugin-register-raised.md` — the same title error when the plugin *did*
  import: `plugin ascend loaded.` with no `Failed to load plugin ascend` and no
  `Platform plugin ascend is activated` means `register()` raised and vLLM v0.7.x's
  `except Exception: pass` deleted the traceback (`platforms/__init__.py:153-161`).
  The pin's hook body (a quant-config import → bare `import torch_npu`), why
  `platform.py`'s own guard print cannot appear, the un-swallow command, the
  `torch-npu …dev20250218` that the pin itself prescribes, and the released
  known-issue bullet naming both this line and the `vllm._C` warning to pre-empt.
  Worked for #130 (v0.7.1rc1); environment-level resolution, mechanism open.
- `missing-subpackage-after-wheel-install.md` — `No module named 'vllm_ascend.<sub>'`
  from a pip-installed release: the "who asks for the name" grep that splits packaging
  from a version contract, the `find_packages` + missing-`__init__.py` test at the pin
  (verified at v0.9.0rc1 `706de02`: `vllm_ascend/compilation/` is the only such
  directory, `setup.py:357`, `platform.py:263-267`, editable install at `Dockerfile:52`),
  the three reporter-side commands, and the ladder `--enforce-eager` → source install /
  next rc (reporter-confirmed on #1156). Precedent family #415/#592/#1093/#1209/#2929.
- `wrong-vllm-module-shadowed.md` — `cannot import name '<X>' from 'vllm'
  (unknown location)`: the two message tails and what each proves (bare directory
  vs a regular package at a named path), the minimal local reproduction of each,
  the one-line `python -c` that prints the resolved `vllm`, why `+empty` and
  vllm-ascend are innocent, and #527's reporter-confirmed *"delete the `vllm`
  folder in the current path"*.
- `launch-env-and-library-path.md` — `ImportError: <name>.so: cannot open shared
  object file` while a connector module is imported in a launcher-managed
  deployment, when the ENV dump says the path is fine: the launch command's
  effective environment, the tutorials' exports, and the image-owns-it boundary.
- `triton-ascend-pair.md` — a Triton-Ascend kernel that fails to *compile* on the
  first real request: how to tell the paraphrase from the library error, the
  versioned wheel+BiSheng pair the tutorial pins (`multi_npu_qwen3_next.md:48-61`
  at vllm-ascend v0.11.0), and the maintainer-confirmed answer on #5750 — do not add
  triton-ascend to these images, and move the release (v0.14.0rc1 needs none)
  rather than the compiler.
- `triton-extension-op-silent-miscompute.md` — a Triton-Ascend-compiled kernel that runs
  and computes the wrong values, in two forms: (a) `extract_slice` of a bf16 sub-tile
  at an unaligned start — the wheel pin (`requirements.txt:20`), the doc claim (bf16
  supported, 无特殊限制) against a UT that only slices 1-D fp32 at 2048-byte-aligned
  starts, the lowering chain from `vec_ops.py:97-136` to `tensor::ExtractSliceOp`, the
  two runs that separate dtype from offset (#18345); (b) a per-head reduction whose
  non-identity scalar scale becomes 0 under a later slice × cos consumer at one head
  count — the five-row ladder, the `patch/worker/` callers that skip the FX pass's
  `head_size != 128` gate, the nightly test's kv-head tuples, the "break the data
  chain" fix and why a temporary store is a dead store (#12642). Both keep the
  host-side workaround ladder and `scripts/qknorm-mrope-exposure.sh`.
- `triton-install-state-op-dispatch.md` — the reporter compares **two implementations of
  one op** ("triton 实现和 ascend 实现结果不一致", 乱码) on a box holding both `triton` and
  `triton-ascend`: `rope_forward_oot`'s `if HAS_TRITON:` (`ops/rotary_embedding.py:166`)
  picks the Triton kernel or `torch_npu._npu_rotary_embedding`, `HAS_TRITON` is install
  state (`vllm/triton_utils/importing.py:12-48`), the recipe uninstalls triton on purpose
  (`Dockerfile:54-57`) and pins `triton-ascend==3.2.0` (`requirements.txt:39`), and the
  kernel's own envelope is two models (`ops/triton/rope.py:23-25`) — so "they should agree"
  needs the same-tensor A/B before any wheel is blamed (#8047, hypothesis).
- `host-cxx-compile-header-conflict.md` — a *host* clang/gcc compile of a generated
  TU (the triton-ascend `launcher_cxx11abi{0,1}.cxx` ABI probe, or a generated
  kernel source stopping on a missing standard header, #3135) failing inside a
  torch_npu/CANN header: the two `acl_base.h` copies
  share the guard `INC_EXTERNAL_ACL_ACL_BASE_H_` while only torch_npu's typedefs
  `aclOpExecutor` (and newer CANN trees drop `acl_op_compiler.h` entirely), so the
  first copy in the TU wins — verified two-sided at #6429 and refiled as #6271 on the
  published `v0.13.0rc2-openeuler` image (reporter-confirmed). The `torch-npu==2.8.0`
  pin vs the reported `2.8.0.post1` is **not** a reporter mismatch: `Dockerfile:76`,
  `Dockerfile.openEuler:75`, `Dockerfile.310p:61` and `Dockerfile.a3:75` all install
  post1 on a `cann:8.5.0-*` base, so the collision ships in the tag and "use the
  release image" is not a rung. Page carries the `--sources` check, the recipe grep,
  the in-container header hotfix, the stale install-doc body and the "no precedent in
  the corpus" note.
- `image-arch-and-tag-availability.md` — the artefact for this CPU architecture or
  SoC variant does not appear to exist: what the tag suffixes mean (`-310p`,
  `-310p-openeuler`, `-openeuler`, and the unsuffixed generic A2 tags including
  `latest`), why `platforms:` is multi-arch only on an upstream `push` (the two
  310P workflows fall back to *different* single arches, and the openEuler leg runs
  on `ubuntu-24.04-arm`), the `FROM` CANN base an arm64 leg depends on, and the
  wheel lane (`release_whl.yml` builds x86_64 + aarch64, but `Dockerfile.buildwheel`
  is a **910b** base and `SOC_VERSION` is baked in) — verified at v0.9.2rc1 for
  #2178, with the rewritten v0.23.0rc1 layout (`_schedule_image_build.yaml`
  arch matrix + `schedule_image_build_and_push.yaml` suffix list) beside it.
- `pip-resolver-warning-block.md` — the install **succeeded** and pip only warned: a
  two-command `vllm` + `vllm-ascend` wheel install whose gap is the torch line, decided
  by upstream `setup.py:843-864` device-gated requirement files (`common.txt` for
  `VLLM_TARGET_DEVICE=empty`, `cuda.txt` with `torch==2.10.0` for the PyPI wheel), the
  `Dockerfile:52-56` `VLLM_TAG`/`empty` build the project ships, `conf.py:75-76`'s
  documented pip pair with the CI doctest that hand-patches torchvision/torchaudio,
  the three unrelated leftover lines, and the "never raise torch" rule (#8286).
- `dependency-pin-bump-request.md` — the install report asks for a **pin bump** instead
  of reporting a missing artefact: the matrix row that fixes the pair, why a version grep
  of `installation.md` comes back empty (substitutions), which line is the real constraint
  (`torch-npu==<dev>` plus the torch it hard-pins, not the tree's `torch>=`), and the one
  `importlib.metadata` call that tests the reporter's premise about the paired package
  (#3266).
- `pip-index-visibility.md` — the requirement exists but no index pip can see serves
  it: a wheel pin on a mirror-only dev build with the `--extra-index-url`/`--pre` option
  lines dropped from the metadata (`setup.py` skips `--` lines), the torch /
  torch-npu / torchvision line arithmetic that makes a documented pip pair
  unsatisfiable (#1796), the air-gapped image extra (`vllm[bench]` → `pandas` /
  `datasets`, the image's own `global.index-url`, the wheelhouse recipe) (#4737), and the
  no-error paraphrase of a PyPI `vllm` wheel that replaced the box's preinstalled torch
  (§A0, #1363 — the pip leg of that ref's install doc has no Ascend mirror, the
  build-from-source leg does). §C is the `Cannot install <a> and <b> … conflicting
  dependencies` text read as a `+cpu` local-label/visibility row; §D is the same words
  coming from the **build-dependency** subprocess (`× pip subprocess to install build
  dependencies did not run successfully`), where the pins are `pyproject.toml`'s
  `[build-system] requires` resolved into an empty prefix — the reporter's installed torch
  pair cannot satisfy them, so `--no-build-isolation` is rung 1 (#446). That §D text is also
  the **expected** failure when the pin pairs `torch-npu==X` with a `torch>=X` floor and no
  `+cpu`-labelled torch is visible — with no pip output pasted at all, read
  `[build-system] requires` first and treat a title's "x86 cannot install" as a premise
  (#1411, whose own pin tells x86 users to add the CPU index at `installation.md:126,129`).
- `offline-source-build.md` — an offline source build: the own-source invocation
  chain (`setup.py` `build_aclnn` → `csrc/build_aclnn.sh` → `csrc/build.sh` →
  `csrc/CMakeLists.txt`), the five network legs with the existence gate that
  suppresses each, and the report-shape answer (which leg stopped, and where to
  seed).
- `mindie-turbo-graph-break.md` — an op-substituting accelerator wheel versus
  `torch.compile`: the `Unsupported … PyCapsule.<op>` graph break at
  `fullgraph=True`, the unconditional worker-side import and its INFO line, the
  `PIECEWISE`/eager log pair that says whether that process compiled at all, why
  the recorded report's other (working) entry path is the first evidence, and the
  versioning-policy/`faqs.md`/`setup.py` answer to "does this plugin pair with
  this release" (#1534, v0.9.1rc1).
- `mindie-turbo-pre-0.8-eager-and-quantizer.md` — the same wheel on the releases that
  *do* pair with it (≤ v0.7.3.post1), where `platform.py:110-114` forces
  `NO_COMPILATION` and the complaint is empty/partial output rather than a traceback:
  the three log states of `try_register_lib` at that ref, the plugin-owned w8a8
  quantizer, the downstream Qwen3/Qwen3-MoE forward copies it substitutes into, the
  `mindie-turbo==2.0rc1` contract an unpinned `pip install` breaks, and the ranking
  that puts the response body (reasoning routing / truncation) ahead of any plugin
  mechanism (#1954; mechanism open).
- `installed-version-vs-checkout.md` — `pip show` naming a different version than
  the reporter installed ("版本号显示不对"): what `setuptools_scm` can and cannot
  produce, why a shallow dev-branch clone yields `0.0.0` rather than a real number,
  the unsafe `-dev` expectation, and the one `python -c` + `pip show -f` pair that
  separates stale metadata from the wrong checkout (#1784).
- `triton-module-residue.md` — a `triton` that imports but has no `language`
  (`AttributeError: module 'triton' has no attribute 'language'` at `import torch`,
  no `vllm_ascend` frame, usually a `-310p` box where the Dockerfile removes it on
  purpose): the resolved module is uninstall residue or a shadow directory.
- `triton-kernels-import-error.md` — the logged `Failed to import Triton kernels …
  Error: No module named 'triton.language.target_info'` on a hand-built box: the
  `triton_kernels` → `triton.language.target_info` chain, the placeholder forms
  (`no attribute 'next_power_of_2'`, `'function' object is not subscriptable`), why
  the failures concentrate in `ops/layernorm_gated.py` (unguarded import at
  `ops/layernorm.py:24`, `triton.next_power_of_2` at `:139`), and why
  `pip uninstall triton` on a triton-ascend box leaves a gutted tree (#11681).
- `python-floor-import-error.md` — an install report whose title paraphrases a Python
  requirement and whose only failure is an import-time `TypeError` in `vllm/` (PEP 604
  `X | None` on 3.9): the metadata that let pip through, the OCR line that is the real
  evidence, and the clone-the-old-env reason the interpreter stayed 3.9 (#3657).
- `custom-op-build-stage.md` — a `COMPILE_CUSTOM_KERNELS=1` source build that fails
  in the compiler/linker log: which stage the last frames name, whose code it is
  (vllm-ascend `CMakeLists.txt`/`setup.py` vs the CANN toolkit), the version
  envelope read off the pin's own requirements table, and `multiple definition` at
  the AscendC relink as one source under two path spellings (#1774).
- `pybind11-cmake-option-rename.md` — an sdist build that dies at `pybind11 --cmake`
  (`unrecognized arguments`, pybind11 3.0) inside pip's isolated env: the pinned tree's
  `--cmakedir` line, the bare `pybind11` requirement, the four-rung ladder (release /
  patch the sdist / `PIP_CONSTRAINT` / `--no-build-isolation`) and the 310X
  `COMPILE_CUSTOM_KERNELS=1` wall behind it (#1931).
- `image-recipe-inventory-and-diff.md` — the sourcing question behind a hand-built
  environment or an image regression: the four artefacts that *are* the image's
  dependency set, the version-swap control rule, the container-only rung set, and
  `scripts/image-recipe-inventory.sh` (#2812).
- `custom-op-extension-missing-from-image.md` — `Failed to import vllm_ascend_C … Sleep
  mode will be disabled` (logger basename `[camem.py:67]`) followed by an
  `AttributeError` on `torch.ops._C_ascend.<op>` at the first use. **Step 0 reads the
  warning's tail** and splits two lanes with two owners: the `.so` is **absent**
  (`No module named …` — the two silent build lanes, `COMPILE_CUSTOM_KERNELS=0` on a
  csrc-cache hit and the `develop`-only CMake install prefix, the two-command decision
  and the artefact-keyed guard that a `COMPILE_CUSTOM_KERNELS=1`-keyed one misses;
  recorded #17713), or it is **present and refused by `dlopen`** (`…: undefined
  symbol: …` — a SoC source-list gate that drops a defining TU while the host binding
  stays, verified at v0.14.0rc1 for 310P, recorded #6527; the op name in the
  `AttributeError` is arbitrary either way).

## Common rules

1. This is an environment question with a short answer. Deliver it from the
   reporter's reproduction plus the image recipe, and act with the cheap set:
   a later release / the prebuilt wheel, rename the shadowing copy, preload the
   allocator. Do not walk the failing module's C++: the exports pin a *different*
   release than the image built (recorded case: `kvcache-ai/Mooncake` v0.3.8
   export vs `MOONCAKE_TAG=v0.3.7.post2` in the Dockerfile), so any ABI,
   `glog`-duplication, RPATH or static-destructor story is unverifiable there and
   delivers nothing the reporter can act on.
1b. **`AssertionError: PlaceholderModule should not be used when the original module
   can be imported` is a soft-import failure, and the caller frame names the module.**
   vllm wraps an optional dependency's failed import in `PlaceholderModule`
   (`vllm/utils/__init__.py`), so the assertion fires later, on first *use*, and names
   no package — read the frame that touches the placeholder, not the assertion. The
   audio path is the recurring one: `vllm/multimodal/audio.py:15-22` soft-imports
   `librosa` and `soundfile` (`PlaceholderModule("librosa")`) and `:88` calls
   `librosa.load` from `AudioMediaIO.load_bytes`, so any serve whose request carries an
   audio input raises it — while the image installs vllm **without** the `[audio]`
   extra (`vllm-ascend/Dockerfile:43` at v0.8.5rc1; the same line with `[audio]` at
   v0.11.0rc2, `pip-index-visibility.md` §B). Reporter-confirmed on #1087: after moving
   from v0.8.5rc1 to `v0.9.0rc2-openeuler` the Ultravox model declares and answers
   text, and the audio request fails exactly here. The fix is inside the image —
   `python -c "import librosa, soundfile"` shows the real `ImportError`
   (numba/llvmlite/numpy pin, missing `libsndfile`) — plus a wheelhouse when the pod is
   air-gapped; the `--media-io-kwargs '{"audio": {"num_frames": …}}'`/processor flags do
   nothing, because the module object is a placeholder.

2. Find out which file the reporter's `import` actually loads before theorising.
   Ascend images routinely carry a second copy of a Python package under
   `<toolkit>/python/site-packages/`, and the reporter's own `import` line may
   resolve to that one rather than to the package vllm-ascend built. An
   `ImportError` naming an attribute of `vllm` itself usually states it in the
   message's location text — `wrong-vllm-module-shadowed.md`.
3. Separate "this published artefact is bad" from "the build recipe is wrong" by
   asking whether a locally rebuilt image fails the same way; #5609 did, which
   moved the cause into the Dockerfile stage.
4. Version-verify the fix claim: name the release the reporter says works, keep
   the workaround for the broken one, and label the mechanism as a hypothesis
   unless a reporter comment states it.
5. When the environment dump contradicts itself — `CANN: None` while the host
   reports an `ascend-toolkit`/`cann-*` root, or a component version missing
   from a list — suspect the *collector* before the install
   (`collect_env.py` reads hard-coded default paths), and ask for the install
   root rather than concluding the component is absent. The same caution applies
   to a *correct-looking* dump: it is one shell's environment, not the launched
   process's (see `launch-env-and-library-path.md`).
