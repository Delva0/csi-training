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
not to vllm-ascend Python or the kernel's shapes.
It also covers the **host C++ compile of a generated translation unit** that fails
inside a *torch_npu* or *CANN* header (`use of undeclared identifier
'aclOpExecutor'` at `AclOpCompileInterface.h`, `fatal error: 'cstdint' file not
found` in `kernel_macros.h`): the failing line is in the wheel's headers, the
owner is the wheel × CANN header pair, and the compiler is the messenger.
It also covers **published-artefact availability**: when the tag or wheel the
reporter needs was never built for their CPU architecture or SoC variant, the
owner is the image/wheel pipeline, and the answer is the tag suffix plus one
registry manifest read — not a vllm-ascend code path.
It also owns the **interpreter's own resolution residue**: what `import vllm`
actually binds on the reporter's box when a checkout, a stray directory or an
editable `.pth` wins over the install (the `(unknown location)` form is
`wrong-vllm-module-shadowed.md`; the missing-module form is
`plugin-load-module-missing.md`).
Inherit the rules in SKILL.md; this topic is the counterpart of
`../startup-failures/index.md` for failures that never reach a worker.

## Routes

- `python -c "import <third-party module>"` (or the first real use of it) dies
  with a glibc allocator message — `corrupted size vs. prev_size`,
  `free(): invalid pointer`, `double free or corruption`,
  `munmap_chunk(): invalid pointer`, `Aborted (core dumped)` — with no Python
  frame and no NPU/collective text → `native-import-abort.md`.
- The startup dies in a vllm-ascend **inductor fusion pass** with
  `'_OpNamespace' 'vllm' object has no attribute '<op>'` (`qknorm_rope_fusion_pass.py`,
  `torch/_inductor/pattern_matcher.py` frames), and `pip list` does show a triton wheel →
  not a missing package: the official images **remove triton on purpose** (every
  `Dockerfile*` at this ref has the same `python3 -m pip uninstall -y triton` under
  the comment "in Ascend, triton doesn't work correctly"), so `HAS_TRITON` is false and every `if HAS_TRITON:`-registered custom op
  is absent by design while its ungated consumer still runs. Grep the log for
  `Triton not installed or not compatible`, then turn the pass off →
  `../startup-failures/missing-triton-op-in-compile-pass.md`.
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
  separate steps). Same page carries the reporter-confirmed `CANN: None`
  `collect_env.py` blind spot and the non-fatal `triton_kernels` log noise.
- Engine start (or plugin activation) aborts with a C++ exception and no Python
  frame — `terminate called after throwing an instance of 'std::logic_error'`,
  `what(): basic_string::_S_construct null not valid`, then
  `RuntimeError: Engine core initialization failed … Failed core proc(s): {}` ->
  `native-string-abort.md`: the stock-model discriminator, the three `python -c`
  native probes, and `gdb -ex 'break std::__throw_logic_error'` to name the
  `.so`. The checkpoint's config, processor and architecture are not the owner
  (#8745 v0.13.0rc3 DeepSeek-V4-Flash and #10943 v0.20.2rc1 print the same two
  lines; #10943's reporter confirmed the unmodified checkpoint crashes
  identically and eight model-side changes moved nothing).
- The report **is a pip resolution error about a requirement**, with no engine and no
  crash: `ERROR: Could not find a version that satisfies the requirement
  torch-npu==2.5.1.post1.dev20250619`, `No matching distribution found for <pkg>`, or
  a title like "unmatched torch version between vllm=X and vllm-ascend=Y" — at install
  time from the published wheel, or inside a deployed image when adding an extra
  (`ImportError: Please install vllm[bench] …` then `pip install pandas` → `No matching
  distribution found`) → `pip-index-visibility.md` first, **before** any version
  survey: `bash scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]` prints the
  `torch` / `torch-npu` / `torchvision` pins of both sides and the contradictions, and
  the page carries the three-way split (mirror-only metadata pin vs unsatisfiable pin
  set vs unreachable index) plus both answer shapes (#1796, #4737). Its sibling case
  is the one where **nothing failed**: a two-command wheel install that only prints
  pip's post-install block (`vllm 0.18.0 requires torch==2.10.0, but you have torch
  2.9.0`, plus torchaudio/torchvision/opencv lines) → `pip-resolver-warning-block.md`
  (which requirement file supplied each pin, and why raising torch is the one thing
  that breaks the install).
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
  paraphrase; it is not the AscendC custom-op lane.
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
- Nothing crashes and the question is a **version string**: `pip show
  vllm-ascend` reports `0.9.0rc2` while the reporter installed "0.9.1dev", or asks
  whether that version exists at all → `installed-version-vs-checkout.md`. The
  number is build-time metadata (`setup.py:305-311` `setuptools_scm`, no constant in
  `vllm_ascend/`), no runtime reader consults it, and the pairing gates read
  `vllm.__version__` — so the deliverable is the two output paths, not a survey.

## Pages

- `native-import-abort.md` — the interpreter aborts inside an image-bundled
  native module (glibc allocator text, no Python frame): which copy is loaded,
  allocator interposition, dependency-list correspondence.
- `native-string-abort.md` — `terminate called after throwing … std::logic_error`
  / `basic_string::_S_construct null not valid` at engine start.
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
- `host-cxx-compile-header-conflict.md` — a *host* clang/gcc compile of a generated
  TU (the triton-ascend `launcher_cxx11abi{0,1}.cxx` ABI probe, or a generated
  kernel source stopping on a missing standard header, #3135) failing inside a
  torch_npu/CANN header: the two `acl_base.h` copies
  share the guard `INC_EXTERNAL_ACL_ACL_BASE_H_` while only torch_npu's typedefs
  `aclOpExecutor` (and newer CANN trees drop `acl_op_compiler.h` entirely), so the
  first copy in the TU wins — verified two-sided at #6429 (v0.13.0rc2, torch_npu
  2.8.0.post1, CANN 8.5.0), with the `--sources` check, the wheel/CANN pairing
  question (`torch-npu==2.8.0` pin vs `2.8.0.post1` reported), the stale install-doc
  body, and the "no precedent in the corpus" note.
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
- `pip-index-visibility.md` — the requirement exists but no index pip can see serves
  it: a wheel pin on a mirror-only dev build with the `--extra-index-url`/`--pre` option
  lines dropped from the metadata (`setup.py` skips `--` lines), the torch /
  torch-npu / torchvision line arithmetic that makes a documented pip pair
  unsatisfiable (#1796), and the air-gapped image extra (`vllm[bench]` → `pandas` /
  `datasets`, the image's own `global.index-url`, the wheelhouse recipe) (#4737).
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
- `installed-version-vs-checkout.md` — `pip show` naming a different version than
  the reporter installed ("版本号显示不对"): what `setuptools_scm` can and cannot
  produce, why a shallow dev-branch clone yields `0.0.0` rather than a real number,
  the unsafe `-dev` expectation, and the one `python -c` + `pip show -f` pair that
  separates stale metadata from the wrong checkout (#1784).

## Common rules

1. This is an environment question with a short answer. Deliver it from the
   reporter's reproduction plus the image recipe, and act with the cheap set:
   a later release / the prebuilt wheel, rename the shadowing copy, preload the
   allocator. Do not walk the failing module's C++: the exports pin a *different*
   release than the image built (recorded case: `kvcache-ai/Mooncake` v0.3.8
   export vs `MOONCAKE_TAG=v0.3.7.post2` in the Dockerfile), so any ABI,
   `glog`-duplication, RPATH or static-destructor story is unverifiable there and
   delivers nothing the reporter can act on.
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
