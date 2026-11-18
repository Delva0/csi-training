# Image and build reports: which chip the artefact targets, and the custom-kernel build

## Scope

Reports about an **artefact** rather than a serving path: "does image `vX` support my card", a chip whose
operators fail to initialise, a locally installed tree missing `libvllm_ascend_kernels.so`, a logged
`Failed to import vllm_ascend_C:libvllm_ascend_kernels.so`, or a `COMPILE_CUSTOM_KERNELS` recipe copied from
an old page. The evidence is that tag's `Dockerfile*`, `setup.py`, `CMakeLists.txt` and
`docs/source/installation.md` — read at the reporter's ref — not a kernel or runner class. Symptom-side
recognition of the first family lives in `../symptoms/device-kernel-faults.md` ("op type … not found in this
op store" and the `AclOpKernelInit failed` variant). A **contributor setup** report (`[Doc]`, filed by
someone following `docs/source/developer_guide/`) is the same kind of artefact question and lives in §4:
the guide's container path, `setup.py`'s generated `_build_info.py`, and what `format.sh ci` actually runs;
the doc-side answer shape is `../symptoms/doc-guide-vs-capability.md`. A **packaging** report — a wheel or
source install that omits whole package directories, or a tree whose directories lack `__init__.py` — is
§5: `setup.py`'s `find_packages` contract, and why such a report must be checked at the branch/sha it names
rather than at the release tag.

Not this page: a doc block that cannot run on the reporter's hardware
(`../symptoms/doc-guide-vs-capability.md`), a missing device (`npu-device-visibility.md`), or an image env
variable / package that never reaches a non-interactive process (`dockerfile-env.md`, which owns the
`~/.bashrc`, `LD_PRELOAD` and missing-package classes).

## Common rules

- **The artefact names its chip.** Every `Dockerfile*` starts from a CANN base whose suffix is the SoC
  family: `…-910b-…` = Atlas 800 A2 (910B), `…-a3-…` = A3 (910C class, `910_93` kernels), `…-310p-…`,
  `…-a5-…`. Quote the `FROM` line at the ref before interpreting any operator error.
- **The set of Dockerfiles is itself a release fact.** `Dockerfile.a3`, `.310p`, `.a5` do not exist in every
  tag: at `v0.7.3` the tree has only `Dockerfile` and `Dockerfile.openEuler`, which is why "the A3 image is
  not supported yet" was the maintainer's answer there.
- **An install question is answered from the build system, not from packaging folklore.** The repo's
  `setup.py` (gate + install prefix) and `CMakeLists.txt` (`target_link_libraries`, `-rpath`, `install`)
  say where a library goes; a recalled setuptools/PEP 660 derivation is both expensive and wrong.
- **These reports still need `status and limits`.** What ships is verifiable; whether the reporter's build
  actually produced the file is not — give them one command that prints it.

## §1 Image → hardware class (worked case: vllm-ascend #1696, `[Usage]`)

Report: image `quay.io/ascend/vllm-ascend:v0.7.3` on a **910C** board, CANN `8.1.RC1`, `example.py` dies at
`LLM(...)` with `RuntimeError: call aclnnArange failed, detail:EZ9999: Inner Error!` /
`Parse dynamic kernel config fail` / `ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed`.

Verified at `v0.7.3` (commit `779eebb`):

- `Dockerfile:18` → `FROM quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10` — the **910B/A2** CANN base.
- `docs/source/installation.md:9` — "A hardware with Ascend NPU. It's usually the Atlas 800 A2 series";
  `:87-89` install the `Ascend-cann-kernels-910b_8.1.RC1_linux-<arch>.run` package — A2 kernels only.
- the tree has no `Dockerfile.a3` at all: that file arrives with `875a920d4` ("[Platform] Add support for
  Atlas A3 series", #1794) and its first tag is **`v0.10.0rc1`** (`Dockerfile.a5` first exists at
  `v0.23.0`). So `v0.7.3` has no A3 image family, and "upgrade to a release from `v0.10.0rc1` on" is the
  durable lever.

Mechanism: 910C is the A3 family whose SoC needs A3 (`910_93`) kernel binaries. On an A2 kernel package the
op has no kernel description for that SoC, so initialisation — not execution — fails, which is why the
traceback carries no shape, dtype or model frame and why the same CANN version is fine on 910B.

Maintainer answer (Yikun, `reported`, 2025-07-10, thread closed as a feature request): "The A3 image
haven't supported yet … replace the base image with `quay.io/ascend/cann:8.1.rc1-a3-ubuntu22.04-py3.11`".
Cost to state: the a3 base is **py3.11** where the A2 one was py3.10, so the torch/torch-npu wheels and a
source rebuild of vllm-ascend must move with it — a rebuild, not an env tweak. Later releases carry the A3
line from `v0.10.0rc1` (`Dockerfile.a3`; the A2 `Dockerfile` itself still uses a `910b` base at `v0.9.1`,
and `docs/source/getting_started/installation/supported_hardware.inc.md` at main has an "Atlas A3 Products"
row), so "upgrade to a release whose image family includes your chip" is the durable lever.

Falsifier: the same `v0.7.3` image runs the same script on an A2 card. If it does, the class is wrong.

## §2 `libvllm_ascend_kernels.so` missing after a local editable install (worked case: #1114)

Report: inside `quay.io/ascend/vllm-ascend:v0.8.5rc1` the user uninstalled `/vllm-workspace/vllm-ascend`,
ran `MAX_JOBS=4 pip install -e . --no-build-isolation` in their own copy, and startup logs

```text
ERROR camem: Failed to import vllm_ascend_C:libvllm_ascend_kernels.so: cannot open shared object file
```

while the image's own tree does have `vllm_ascend/lib/libvllm_ascend_kernels.so`.

Verified at `v0.8.5rc1`:

- **Two libraries, one install rule, and the error names the second one.** `CMakeLists.txt:50` builds
  `libvllm_ascend_kernels.so` through `ascendc_library(vllm_ascend_kernels SHARED …)`, `:77` builds the
  pybind module `vllm_ascend_C`, `:86-96` links the module against the kernels library with
  `-Wl,-rpath,$ORIGIN:$ORIGIN/lib`, and `:98` installs **both** into `${VLLM_ASCEND_INSTALL_PATH}`. So the
  reported string is the dynamic loader's message for a missing **DT_NEEDED dependency of
  `vllm_ascend_C.so`** — a missing/unresolvable dependency next to the wrapper, not an attempt to open that
  filename, not a `sys.path` problem and not proof that nothing was compiled.
- **`libvllm_ascend_kernels` is never spelled in Python.** `git grep -n libvllm_ascend_kernels <tag>` is
  empty at `v0.8.5rc1` by construction: the name exists only as the CMake target. The Python-side strings
  are `vllm_ascend.vllm_ascend_C` and the log text `"Failed to import vllm_ascend_C:%s"`. An empty grep is
  evidence about the spelling, not about the file.
- **The gate.** `vllm_ascend/envs.py:35-36`: `COMPILE_CUSTOM_KERNELS = bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1")))`;
  `setup.py:288-289` leaves `ext_modules = [CMakeExtension(name="vllm_ascend.vllm_ascend_C")]` empty when
  that is falsy, and `setup.py:207-208` returns from `build_extensions()` before any cmake call. So an
  install can report success with no kernels target in it — `COMPILE_CUSTOM_KERNELS=0` (or any value
  `int()` reads as 0) is the first thing to check on "my install has no `.so` and no failure"; a failed
  cmake *configuration* raises (`RuntimeError: CMake configuration failed`), so silence is the env-var
  shape.
- **Where the file goes.** `CMakeLists.txt:98` `install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION
  ${VLLM_ASCEND_INSTALL_PATH})`, and `setup.py:160-164` sets that prefix to `ROOT_DIR/<build_lib>` unless
  `get_command_obj("develop")` is a `develop` instance, in which case it is `ROOT_DIR/vllm_ascend`. The same
  test gates the copy-back loop at `:255-256`, which walks `self.build_temp` and copies every `*.so` into
  `self.build_lib/vllm_ascend`. Consequence for triage: when a reporter's tree lacks
  `vllm_ascend/lib*/…so`, the two candidate outcomes are "cmake installed into the build dir" and "cmake
  never ran"; the file listing decides, not a theory about the package manager. **What the installed
  setuptools' editable-wheel path does with that command object is in the installed wheel, not in this
  clone** — that is a gap to state, and deriving it from memory is what cost a recorded run its whole
  budget (#1114, no answer).
- **Loader sites differ in severity.** `device_allocator/camem.py:56-70` logs it with `logger.error`, sets
  `camem_available = False` in the `except ImportError` branch and rebinds
  `init_module`/`python_create_and_map`/`python_unmap_and_release` to `None`; `platform.py:31` imports the
  extension plainly. A run without sleep mode can therefore serve normally with only that ERROR line; a run
  with `enable_sleep_mode=True` is the one that actually needs the extension. Say which of the two the
  reporter is in before claiming the engine cannot start.
- **The documented recipe and the image use the same command**: `docs/source/installation.md:163` ("vllm-ascend
  will build custom ops by default. If you don't want to build it, set `COMPILE_CUSTOM_KERNELS=0`"),
  `:167-168` (gcc/g++ > 8, C++17, and `pip install --no-build-isolation -e .` to build against the system
  env). `Dockerfile:21` `ARG COMPILE_CUSTOM_KERNELS=1`, `:25` `ENV`, `:52` `python3 -m pip install -v -e
  /vllm-workspace/vllm-ascend/`. When the image has the `.so` and the reporter's tree does not with the
  same command, the difference is their tree (branch, submodules, `csrc/`, `ASCEND_HOME_PATH`), not a
  missing image feature — an uninitialised clone is the §3 case.
- **The env var itself is gone from `v0.12.0rc1` onward.** `37db0844f` "Remove COMPILE_CUSTOM_KERNELS env
  (#4864)" has first tag `v0.12.0rc1`: from that tag `setup.py` always defines
  `ext_modules = [CMakeExtension("vllm_ascend.vllm_ascend_C")]` and runs a `build_aclnn` command
  (`csrc/build_aclnn.sh`, `setup.py:201-217`, `:426`) before `build_ext`. A `COMPILE_CUSTOM_KERNELS=0`
  copied from an old page is inert there — check the tag before promising it as a way to skip the build.

Thread lever (attributed, `Yikun` 2025-06-07): follow the installation guide (`docs/source/installation.md`)
or build with the tag's `Dockerfile`; a later comment closes the conversation to a new issue. Membership of
the support matrix is the reporter's real question, so quote the doc lines and the `Dockerfile` build line
above rather than a code window.

Cheapest checks to ask for, in order: `COMPILE_CUSTOM_KERNELS` in `env -0` / `docker exec printenv` (must
be unset or `1`); `find <tree> -name '*.so'` and `ls vllm_ascend/*.so vllm_ascend/lib/` (does
`vllm_ascend_C*.so`, `vllm_ascend/lib/libvllm_ascend_kernels.so`, or a `build/lib*/` copy exist?); the tail
of the pip log for `Copy: … -> …` and `cmake --install` lines, which the `develop` branch prints.

Workaround ladder: rebuild in place with `COMPILE_CUSTOM_KERNELS=1`, the image's `SOC_VERSION` and `-v` so
the `cmake --build` / `cmake --install` lines are visible, then confirm both files exist — or keep the
image's `/vllm-workspace/vllm-ascend` tree and run their code as a local patch instead of reinstalling over
it (the maintainer's answer pointed at the installation docs and `Dockerfile`, i.e. reproduce the image
build rather than a bare `pip install -e .`). Copying the image's `libvllm_ascend_kernels.so` next to a
separately built wrapper works only if both came from the same commit and toolchain — state it as fragile,
never as the recommendation.

Status and limits: it is **unverified** why that build produced no kernels library (gate off in their
environment, a stale `build/` directory, or an `ascendc` install step that did not run), and the setuptools
version in that image was never established. Falsifier: their `vllm_ascend/lib/` listing plus the `-v`
log's `--target=vllm_ascend_C` / `cmake --install` lines. Do not present a setuptools/PEP 660 internal as
the cause.

The DSO list grows later: at main the rpath is
`$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib` and the install target
is conditional, so re-read `CMakeLists.txt` at the tag in hand.

## §3 The custom-op build needs the `catlass` git submodule (offline source installs)

Report class: "custom ops compilation fails in an offline / no-public-network pod" — vllm-ascend #8788,
`v0.13.0` installed from source inside `quay.io/ascend/vllm-ascend:v0.13.0rc3-a3-openeuler`, A3. The
build step, not the compiler, is what needs the network.

Verified at the tag:

- **The build runs unconditionally once `csrc/build_aclnn.sh` exists.** `csrc/build_aclnn.sh` and
  `setup.py`'s `build_aclnn` command arrive with `554f16ae1` "[Kernel] add custom op
  GmmSwigluQuantWeightNzTensorList" (#3804), first tag `v0.12.0rc1`; `cmdclass` registers it
  (`setup.py:489-492` at `v0.13.0`) and `cmake_build_ext.run()` calls
  `self.run_command("build_aclnn")` **before** `super().run()` (`:425-427`). So the fetch is on the
  default `pip install -e .` path, not behind an opt-in flag.
- **What it fetches.** For `ascend910b` and `ascend910_93` the script requires
  `${ROOT_DIR}/csrc/third_party/catlass/include`; when that directory is absent it runs
  `git submodule update --init --recursive` and `exit 1` if the fetch fails (`build_aclnn.sh:16-23`
  and `:33-40`). `.gitmodules` declares it — `path = csrc/third_party/catlass`,
  `url = https://gitcode.com/cann/catlass.git`, `branch = catlass-v1-stable` — and first exists with
  `ad0607f90` (#3532), first tag `v0.12.0rc1`. A second guard repeats the requirement in CMake:
  `csrc/dispatch_gmm_combine_decode/op_host/CMakeLists.txt:14` `message(FATAL_ERROR "dependency
  catlass is missing …")`.
- **The failure shape is an install abort, not an op error.** `build_and_install_aclnn.run()` turns the
  non-zero exit into `SystemExit(returncode)` (`setup.py:209-217` at `v0.13.0`), so the reporter sees
  `Running bash build_aclnn.sh ...` → `fetch failed` → `Error running build_aclnn.sh:` before any cmake
  compile — and no `vllm_ascend/` frame anywhere.
- **310P is exempt.** The script's first branch is `ascend310*` → `exit 0` ("no custom aclnn ops for
  ASCEND310 series", `:6-11`), so a 310P source install never reaches the submodule.
- **The A3 branch has a second, non-network prerequisite.** After the submodule guard it requires
  `moe_distribute_base.h` somewhere under `$ASCEND_TOOLKIT_HOME` (`build_aclnn.sh:41-46`, copied into
  `csrc/utils/inc/kernel` at `:48`) and `exit 1`s with "cannot find moe_distribute_base.h file in CANN
  env" otherwise — so an A3 build can fail after catlass is resolved, and the message names the CANN
  toolkit rather than the repo.
- **HEAD keeps the requirement and pins the revision.** `.gitmodules` at HEAD carries `branch = master`
  plus `commit = 41bf90da655bba3c66d0acd7e00abe33960ecfd6`, and `setup_catlass_dependency()` runs
  `git fetch origin` + `git checkout <commit>` after the submodule update — but only inside the same
  "include/ is missing" branch, so a pre-seeded tree still short-circuits git entirely.

Workaround ladder (proposed; the thread named no fix):

1. **Pre-seed the tree**, on a networked host or from an internal mirror: put catlass in
   `csrc/third_party/catlass` so `csrc/third_party/catlass/include` exists (branch `catlass-v1-stable` for
   the `v0.12`–`v0.22` era, the `.gitmodules` `commit` for `main`-era tags), then run `SOC_VERSION=910c
   pip install -e .` as usual — the `if [[ ! -d … ]]` guard is false and git is never invoked.
2. **Redirect git to a mirror** before the install:
   `git config --global url."<mirror>/cann/catlass.git".insteadOf https://gitcode.com/cann/catlass.git`.
   The script also runs `git config --global --add safe.directory "$ROOT_DIR"`.

Cost: you pin the revision yourself — headers-only, so no ABI risk, but the ops are built from your copy,
and on a tag without the `commit` pin the branch tip is whatever the mirror served. **Not a lever:**
`FETCHCONTENT_BASE_DIR` / `$ROOT/.deps` (`setup.py:316-318` at `v0.13.0`) — no `CMakeLists.txt` at that tag
uses `FetchContent`, so it pre-seeds nothing.

Thread lever (`reported`): `zhaomingyu13` — use the latest image, "in which all dependencies match
properly"; if you must install from source, paste the error log.

Status and limits: the reporter pasted no log, so the failing command in their pod was never established;
the mechanism above is the tag's build script. Falsifier: their build log's last line names another
fetcher (a `pybind11` download, a pip step) — then this section is not their case.

## §4 Contributor dev/test environment: the mount shadows `_build_info`, and `format.sh ci` is not CI

Worked report: **#17718** — a `[Doc]` issue from a new contributor. The `testing.md` nightly container
starts fine, they add `-v /host/vllm-ascend:/vllm-workspace/vllm-ascend` to edit code, and
`import vllm_ascend` dies with `ImportError: cannot import name '_build_info' from 'vllm_ascend'`.

Verified on the default branch (2026-11-17):

- **The image's package is editable at that exact path.** `Dockerfile:65`
  `python3 -m pip install -e /vllm-workspace/vllm-ascend/` — not a site-packages copy — so mounting a
  host tree on `/vllm-workspace/vllm-ascend` replaces the package code *and* every build artefact that
  was generated inside it. The docs' own container blocks never mount the source (Single-card
  `docs/source/developer_guide/contribution/testing.md:66-87`, Multi-cards `:104-125`), so the mount is
  the contributor's own addition and no page explains the consequence.
- **The two pages name different image families on purpose.** `installation.md` renders the stable tag
  through an include template (`docs/source/getting_started/quick_start/ascend_image/atlas-200i-pro.inc.md:12`,
  `quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-…`), while `testing.md` hard-codes `nightly-main`
  (`:71`, `:106`). "The docs recommend 0.23.0 for contributors" is a real disconnect in the prose, not a
  wrong pin. The two container tabs' only install lines are `pip config set global.index-url
  https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` and `pip install -r requirements-dev.txt`
  (`:91-100`, `:129-137`).
- **One file is generated into the source tree at build time.** `setup.py:157` `gen_build_info()`
  writes `vllm_ascend/_build_info.py` (`:161`, body `__device_type__`), called from
  `custom_develop.run` (`:176`) and `custom_build_info.run` (`:182`, registered as `"build_py"` at
  `:466`). It is gitignored (`.gitignore:212` `/vllm_ascend/_build_info.py`), so a fresh checkout never
  has it. Present since `v0.12.0rc1` (`367720259` "make vllm-ascend work well in developer mode", #4179).
- **The consumer moves between releases, the import shape does not.** `v0.23.0`
  `vllm_ascend/utils.py:836` `from vllm_ascend import _build_info`; default branch
  `vllm_ascend/device/device_config.py:30`, same statement. Absent module → `ImportError: cannot import
  name '_build_info'`, raised at import time, before any NPU call.

Lever (proposed, not reporter-confirmed): keep the mount and re-run the build **in the mounted tree** —
`cd /vllm-workspace/vllm-ascend && pip install -e . --no-build-isolation` — which reruns
`gen_build_info`/`build_py` there; or restore just `vllm_ascend/_build_info.py`. Cost: an editable
rebuild minutes long on a cold cache, and `--no-build-isolation` reuses the image's build deps (a mirror
that 403s turns that into a failure rather than a slow download). `pip install -e .` also re-runs the
custom-op build unless `COMPILE_CUSTOM_KERNELS=0` (§2, and the UT caveat below).

Falsifier: `pip show -f vllm-ascend | grep -i location` inside their container naming a `site-packages`
directory (then the mount shadows nothing and the ImportError has another cause, e.g. a stale
`vllm_ascend` on `PYTHONPATH`).

**The contributor UT sequence** (each step is a doc gap in #17718; the order is what makes it work):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export SOC_VERSION=ascend910b1     # only when `npu-smi` detection fails (setup.py get_chip_type)
export COMPILE_CUSTOM_KERNELS=0    # NPU-less UT host only
cd /vllm-workspace/vllm-ascend && pip install -e . --no-build-isolation
pytest -sv tests/ut
```

- The `COMPILE_CUSTOM_KERNELS` comment the report quotes is real and still there: default branch
  `vllm_ascend/envs.py:39-43` ("should only be set to False when running UT scenarios in an environment
  without an NPU. Do not set it to False in other scenarios.") above the key at `:44`; `v0.23.0` has the
  same text at `:39-42`, key `:43`. The report quotes `envs.py:51` — the line **moved**, it was not
  removed, so widen the `sed` range before reading it as gone.
- `SOC_VERSION` appears in `testing.md` only in the **Local (CPU)** tab (`:57-59`); the two container
  tabs never mention it, and the container's `npu-smi` is a bind-mounted host binary.
- Do not `pip config set` a mirror globally inside the image; pass `-i <index>` per command. A 403 from
  the documented Tsinghua mirror is the reporter's network (`reported`), not a repo fact.
- **`bash format.sh ci` is style, not tests.** Default branch `format.sh:40-44`:
  `if [[ "$1" != 'ci' ]]; then pre-commit run --all-files; else pre-commit run --all-files
  --hook-stage manual; fi` — no pytest anywhere. `contribution/index.md:35-60` presents that command
  under `#### Run CI locally`, which is why a contributor reads it as "the CI tests".

## §5 A missing `__init__.py` silently drops a whole package subtree from the wheel (worked case: #8686)

Worked case vllm-ascend #8686 (`[Bug]` on `releases/v0.13.0`, build `0.13.0rc4.dev3+gc9888eabd`, sha
`c9888eabd`, 2026-04-24): the reporter scans `vllm_ascend/` and lists three directories without
`__init__.py` (`vllm_ascend/models/layer/`, `vllm_ascend/transformers_utils/`,
`vllm_ascend/_cann_ops_custom/`), predicting a producer-mode build that omits them; the observed shape is
`pip install .` succeeding, then
`python -c "from vllm_ascend.models.layer.attention.layer import *"` raising
`ModuleNotFoundError: No module named 'vllm_ascend.models.layer'` — while the same import works in a
developer checkout, because `pip install -e .` reads the source tree and never runs the packaging scan.

- **Root cause is the packaging line, not a runtime path.** At `v0.13.0`, `setup.py:522` is
  `packages=find_packages(exclude=("docs", "examples", "tests*", "csrc"))` (verified locally). A package
  is a directory with `__init__.py` **whose every ancestor is also a package**; a subpackage that has
  one under an ancestor that does not is skipped together with all its modules, with no warning and no
  build error. `package_data`/`include_package_data` do not rescue it. That is why
  `vllm_ascend/models/layer/attention/` (which *does* have `__init__.py`) disappears from the wheel while
  its parent has none.
- **The tree to read is the reporter's sha, not the nearest tag.** At `v0.13.0` neither
  `vllm_ascend/models/layer/` nor `vllm_ascend/transformers_utils/` exists —
  `git ls-tree -r --name-only v0.13.0 -- vllm_ascend/models/layer vllm_ascend/transformers_utils` is
  empty, while `vllm_ascend/_cann_ops_custom/` holds only `.gitkeep`. Both arrive on the
  `releases/v0.13.0` branch with `bbe9bc0d9 [v0.13.0][Feature] Add DeepSeek v4 initial support (#8648)`
  (2026-04-24 11:29), and the reporter's `0.13.0rc4.dev3+gc9888eabd` is branch commit `c9888eabd`
  (2026-04-24 15:09). `resolve_ref.sh` resolves the `+g<sha>` suffix, and
  `git ls-tree -r --name-only <sha> -- <paths>` then shows the four files (two `__init__.py`, two
  modules) that the wheel lost. Reading the tag and reporting "the directory does not exist" would be a
  wrong answer built from a correct command.
- **Fix and its two branches.** #8729 = `58ea73f95` `[v0.13.0][Bugfix] Fix Missing __init__.py` (merged
  2026-04-27, base `releases/v0.13.0`, exactly two added empty files, `git tag --contains` → **no tag**)
  and #9649 = `471f0590b` `[BugFix] Add missing __init__.py to models/layer package` (merged 2026-05-28,
  base `main`, adds only `vllm_ascend/models/layer/__init__.py`) — the same defect landed on both
  branches, so a `fixed-in` answer needs the branch, not just the PR number, and the class recurs
  whenever a new package directory is added under a parent that has no `__init__.py`.
- **Workaround on an affected build.** Add the empty files before building —
  `touch vllm_ascend/models/layer/__init__.py vllm_ascend/transformers_utils/__init__.py` — then
  `pip install .`/`pip wheel .`; or take a build whose tree carries #8729. Cost: a source patch you must
  carry, i.e. not usable when installing the released artefact. Verify with
  `unzip -l dist/*.whl | grep 'vllm_ascend/models/layer'` rather than by importing.
- `vllm_ascend/_cann_ops_custom/` at that ref holds only `.gitkeep`: no Python module is lost today, so
  it is at most a forward-looking request — keep it out of the root cause.
- **Reply shape and scope line.** It is a packaging/build question: the workaround is the empty files
  (with that cost), the root cause is `setup.py:522`'s `find_packages` contract, and the check the
  reporter runs is `python -c "import vllm_ascend.models.layer.attention.layer"` in the built
  environment. The files are absent **at the tag**, present on the branch — do not report "the tag is
  broken". Generalisation: any `[Bug]` whose body is "these package directories lack `__init__.py`" is
  answered from `setup.py`'s `packages=` line plus the tree at the reporter's ref; there is no code path
  to search, and the symptom is always `ModuleNotFoundError` after a non-editable install.

## Check before reuse

```bash
R=/workspace/repos/vllm-project/vllm-ascend
git -C $R ls-tree --name-only <tag>: | grep -i dockerfile            # which image families exist
git -C $R show <tag>:Dockerfile | grep -n '^FROM'                    # the SoC suffix
git -C $R show <tag>:Dockerfile.a3 | grep -n '^FROM'                 # when present
git -C $R show <tag>:docs/source/installation.md | sed -n '150,215p' # kernels build + --no-build-isolation
git -C $R show <tag>:vllm_ascend/envs.py | grep -n -A2 COMPILE_CUSTOM_KERNELS
git -C $R show <tag>:setup.py | grep -n 'ext_modules\|install_path\|develop\|COMPILE_CUSTOM_KERNELS'
git -C $R show <tag>:setup.py | sed -n '150,270p'
git -C $R show <tag>:.gitmodules                                     # catlass submodule (from v0.12.0rc1)
git -C $R show <tag>:csrc/build_aclnn.sh | sed -n '1,45p'            # the submodule guard per SoC branch
git -C $R show <tag>:CMakeLists.txt | sed -n '45,100p'
git -C $R show <tag>:vllm_ascend/device_allocator/camem.py | sed -n '50,72p'
git -C $R grep -n 'Failed to import vllm_ascend_C\|import vllm_ascend.vllm_ascend_C' <tag> -- vllm_ascend
git -C $R show <tag>:Dockerfile | grep -n 'pip install -e'                  # the editable install path (§4)
git -C $R grep -n 'gen_build_info\|_build_info' <tag> -- setup.py vllm_ascend  # generated file + consumer (§4)
git -C $R show <tag>:format.sh | sed -n '36,46p'                            # `ci` = pre-commit only (§4)
git -C $R show <tag>:setup.py | grep -n 'find_packages'                     # the packaging scan (§5)
git -C $R ls-tree -r --name-only <sha-from-collect_env> -- vllm_ascend/models vllm_ascend/transformers_utils  # what the wheel can see (§5)
git -C $R ls-tree -r -d --name-only <ref> -- vllm_ascend | while read d; do git -C $R cat-file -e <ref>:$d/__init__.py 2>/dev/null || echo "no __init__: $d"; done  # every parent must be a package (§5)
```

Falsifier for the whole page: if the report's failing frame is inside a serving path (`model_runner`,
attention, scheduler), it is not an artefact question — route it to `../symptoms/index.md`.
