# The `COMPILE_CUSTOM_KERNELS=1` build failed: which stage, whose code

Pin for every `path:line` here: vllm-ascend **v0.7.3.post1** (`c69ceaca1ecc`), the
tree recorded #1774 built. The custom-op build was refactored from v0.13.0 (per-SoC
`csrc/build_aclnn.sh`, `offline-source-build.md`), so re-grep `CMakeLists.txt` /
`setup.py` at the reported ref before quoting a line.

## Report shape

`[Installation]` / 安装 … 编译失败 with a **compiler or linker log and no engine
start**: `gmake`/`CMakeFiles/… Error 1`, `c++: error:`, `ld: … multiple definition
of <symbol>`, a header `No such file or directory`, then a `CalledProcessError` from
`setup.py` `build_extensions` (`setup.py:250` `cmake --build … --target`). The body's
install command is normally `python setup.py install` or `pip install .` with
`COMPILE_CUSTOM_KERNELS=1` exported, as in the pinned page's sleep-mode note.

Not this page: a pip **resolution** error (`pip-index-visibility.md`), the pybind11 CLI
rename that fails **before** any CMake work (`pybind11-cmake-option-rename.md` — `--cmake`
vs `--cmakedir`, build isolation), the
third-party **fetch** legs of the same build (`offline-source-build.md`), and a host
C++ compile of a generated TU **at engine start**
(`host-cxx-compile-header-conflict.md`).

## Step 0 — the version envelope, before the build

The reporter nearly always states a trio. Check it against the pin's own table
rather than the newest release: at v0.7.3.post1 `docs/source/installation.md:15-17`
is CANN `>= 8.1.rc1`, torch-npu `>= 2.5.1`, torch `>= 2.5.1`, and its manual-install
block even `wget`s **CANN 8.1.RC1** (`:81,87,91`). #1774's
`CANN 8.1.RC1 / torch-npu 2.5.1 / torch 2.5.1 / vllm 0.7.3` is exactly inside that
envelope, so 「版本不匹配」 is refuted in one read and must not be rung 1 — state it
as the first sentence, because "upgrade the stack" is the answer that costs the
reporter a day. (The converse shape, a self-built vllm beside an rc vllm-ascend, is
still a version-contract defect: `../startup-failures/environment-correspondence.md`.)

## Step 1 — the paste names its own stage, and the frames name the owner

| Last frames in the paste | Stage | Owner | First check |
|---|---|---|---|
| `setup.py` in `get_requirements`/`_read_requirements`, `setuptools` | python metadata | vllm-ascend | the requirement line it died on |
| `CMake configuration failed`, `cmake_minimum_required`, `ascendc_kernel_cmake does not exist` | cmake configure | vllm-ascend `CMakeLists.txt`, else a missing/mis-set CANN | `echo $ASCEND_HOME_PATH`; `ls $ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake` |
| `usage: __main__.py [-h] [--version] [--includes] [--cmakedir] …` + `__main__.py: error: unrecognized arguments: --cmake`, under `running build_ext` and after the `Found existing ASCEND_HOME_PATH/PYTHON_EXECUTABLE` lines | cmake configure (pre-CMake), the `vllm_ascend_C` pybind extension's prefix path | vllm-ascend `setup.py`'s `python -m pybind11 --cmake` call plus the bare `pybind11` build requirement — pybind11 3.0 renamed it (`pybind11-cmake-option-rename.md`; pinned `setup.py:178-180` already uses `--cmakedir`) | the sdist's own `setup.py`, then `PIP_CONSTRAINT`/`--no-build-isolation` |
| `ccec_compiler`, `ascendc_pack_kernel`, `[NN%] Building … _aiv_device` | AscendC kernel compile/pack | the CANN toolkit | the kernel file the compiler named |
| CANN `recompile_binary.py` → `CalledProcessError`, `ld returned 1 exit status`, `collect2: error: ld returned 1 exit status` | kernel host **relink** | the CANN toolkit | the `recompile:` line it just printed |
| `…host_stub_obj…`, `Linking CXX shared library lib/libvllm_ascend_kernels.so` | host stub link | the CANN toolkit | the object list on that line |
| `csrc/torch_binding.cpp:<n>: fatal error: torch_npu/csrc/core/npu/NPUStream.h: No such file or directory` inside `CMakeFiles/vllm_ascend_C.dir/…` — often **without** `COMPILE_CUSTOM_KERNELS` | the `vllm_ascend_C` pybind extension, i.e. the torch_npu **include root** | vllm-ascend `setup.py` + `CMakeLists.txt` (or the pta wheel) | the configure line `TORCH_NPU_PATH is …` — step 1b below |

`CMakeLists.txt:35-45` picks the toolkit directory that exists — `tools/tikcpp/
ascendc_kernel_cmake`, then `compiler/…`, then `ascendc_devkit/tikcpp/samples/cmake`
— and includes its `ascendc.cmake`. So a paste whose frame is
`…/tools/tikcpp/ascendc_kernel_cmake/util/recompile_binary.py` came through the
**first** branch: the branch choice is not the discriminator there, and both
`ascendc.cmake` and `recompile_binary.py` are CANN files a snapshot usually does not
export (`SOURCES.json` `unavailable`) — label any mechanism inside them **open** and
do not spend turns reading `csrc/`.

## Step 1b — a missing `torch_npu/…` header is the include ROOT, not the wheel

Verified at vllm-ascend **v0.10.2rc1** (`048bfd5`), the tree recorded **#3403**
(910B / aarch64, CANN 8.1.RC1, `torch_npu-2.7.1.dev20250724…whl` installed by hand,
`pip install -v -e .`): the extension compiles `csrc/torch_binding.cpp:20` and
`csrc/torch_binding_meta.cpp:4`, both `#include <torch_npu/csrc/core/npu/NPUStream.h>`,
and the include root comes from a value computed **without an error path**:

- `setup.py:203-212` runs `python3 -m pip show torch-npu | grep '^Location:' | awk '{print $2}'`
  and appends `/torch_npu`. The pipeline's exit status is `awk`'s, so a missing
  `torch-npu` raises nothing: `torch_npu_path` is `""` → `-DTORCH_NPU_PATH=/torch_npu`.
- `CMakeLists.txt:63` (`${TORCH_NPU_PATH}/include`) is the only torch_npu include root
  (`:82` the only lib root), and `CMakeLists.txt:54` already prints the value:
  `message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")`.

**So the decisive artefact is the configure block's `TORCH_NPU_PATH is …` line**; a
paste that starts at the compiler error (as #3403's does) has not shown it — ask for it
rather than re-deriving the C++ side, and do not open `csrc/`.

Two ways that value comes out empty, and both are visible in the paste:

1. **pip build isolation + torch-npu removed from the build requirements.** The
   traceback runs through `/tmp/pip-build-env-*/overlay/lib/.../setuptools/...`, which
   *is* the isolated env, and the body says so in words — #3403: 「comment the torch-npu
   in requirements.txt and pyproject」 (they did it so pip would not re-fetch the dev
   wheel). `pyproject.toml:15` pins `torch-npu==2.7.1.dev20250724` in
   `[build-system] requires`; removing that line empties the pta path.
2. **`python3` on `PATH` is not the interpreter pip builds with**, so `pip show` queries
   another site-packages and prints a path that holds no `torch_npu`.

Rung 1, which is also the project's own documented remedy (`installation.md:175`,
「encounter a torch-npu version conflict … `pip install --no-build-isolation -e .`」):

```bash
pip install cmake pybind11 "setuptools-scm>=8" decorator einops "numpy<2.0.0" scipy pyyaml wheel
MAX_JOBS=4 pip install -v -e . --no-build-isolation
```

Keep the second lane **open**, do not merge it: a pta wheel that ships no `include/`
would fail identically with a correct root. It is refuted for that branch at
`Ascend/pytorch` (`d33df559`) `setup.py:441-464`, which copies `torch_npu/csrc/**/*.h`
into `build/packages/torch_npu/include/torch_npu` (and the file exists at
`torch_npu/csrc/core/npu/NPUStream.h`), so the check that closes it is one `ls`:

```bash
ls "$(python -c 'import torch_npu,os;print(os.path.dirname(torch_npu.__file__))')"/include/torch_npu/csrc/core/npu/NPUStream.h
python -c "import torch_npu; print(torch_npu.__file__)"   # same interpreter the build used
```

Present → the root was wrong (lane above). Absent → that wheel is not a build of the
pinned pta line; ask which URL it came from.

Same report, different owner: `module 'vllm_ascend' has no attribute 'register'` beside
`No module named 'vllm_ascend._version'` is a **half-built install**, not a second
build fault — the tree defines `register()` at `vllm_ascend/__init__.py:19` and
`_version.py` is generated by `setuptools_scm` (`setup.py:306`
`get_version(write_to="vllm_ascend/_version.py")`), so the module that got imported is
not this tree. One `python -c "import vllm_ascend; print(vllm_ascend.__file__)"`, then
`pip uninstall -y vllm-ascend`, `rm -rf build .deps`, rebuild
(`plugin-load-module-missing.md` for the general residue
case, `wrong-vllm-module-shadowed.md` when the tail is `(unknown location)`).

## Step 1c — `Error 1` with **no** compiler line at all

The paste may start at the `gmake` lines: `CMakeFiles/vllm_ascend_C.dir/csrc/torch_binding.cpp.o`
`Error 1` → `CMakeFiles/Makefile2` → `Makefile:… vllm_ascend_C Error 2` →
`CalledProcessError: … ['cmake', '--build', '.', '-j=<N>', '--target=vllm_ascend_C']`,
then the `setup.py` `build_ext` traceback. That is **#676** (v0.8.4rc1 `086423d`,
x86_64 openEuler 24.03, `python setup.py develop`), and its attached screenshot's
`.ocr.txt` repeated the same traceback — so the screenshot is not extra evidence and
the compiler diagnostic really is missing. Say that in the first line instead of
picking a lane from the paste; then give the ladder that produces the line:

```bash
MAX_JOBS=8 pip install -v -e . --no-build-isolation 2>&1 | tee build.log
grep -nE 'TORCH_NPU_PATH is|error:|fatal error|Killed|Error 1' build.log
```

Three owners fit a paste that shows only `Error 1`, and each has one check:

| Owner | What the full log carries | Check |
|---|---|---|
| empty `${TORCH_NPU_PATH}` (step 1b) | `fatal error: torch_npu/csrc/…: No such file or directory` | the configure block's `TORCH_NPU_PATH is …` |
| `cc1plus`/`c++` killed by the OOM killer | `c++: fatal error: Killed signal terminated program cc1plus`, or **nothing** but `Error 1` when the driver itself is killed | `dmesg -T` and look for `killed process`; rerun with `MAX_JOBS=8` |
| the reporter trimmed the paste | an `error:`/`note:` block he cut | the 20 lines above `gmake[3]` |

Source-verified at v0.8.4rc1: `setup.py:88-100` `compute_num_jobs()` takes `MAX_JOBS`
when set and otherwise `len(os.sched_getaffinity(0))`, and `:237` passes
`-j={num_jobs}` straight to `cmake --build` — so a 192-core box builds with `-j=192`,
and the fan-out is printed inside the traceback's own `CalledProcessError` command.
The toolchain gate is documented, not enforced: `CMakeLists.txt:6`
`set(CMAKE_CXX_STANDARD 17)` plus `docs/source/installation.md:181` "gcc/g++ higher
than 8 and c++ 17 or higher" — an old `g++` (or no `g++` beside a working `gcc`) fails
the first TU, so ask for `gcc --version; g++ --version` in the same request.

One free discriminator, labelled a heuristic: under `-j=<nproc>` a **global** include
or flag fault fails many TUs, so a log with a single `.o Error 1` favours the OOM or a
single-file error; many favours step 1b. #676's paste showed one — and its later
`collect_env.py` (posted two days later) settled the rest: CANN 8.1.RC1, gcc 12.3.1,
192 CPUs, `torch 2.5.1+cpu` / `torch-npu 2.5.1.dev20250320` — i.e. the documented pair
(`installation.md:12-17`) and a modern toolchain, which leaves the 192-way `-j` and
the include root. That issue stayed open in public: `COMPILE_CUSTOM_KERNELS=0` was
rebuffed (step 3) and a later pointer to PR #664 did not fix it either.

## Step 2 — `multiple definition` at the kernel link is one source under two spellings

Recorded **#1774** (v0.7.3.post1, 910B, CANN 8.1.RC1, `python setup.py install`,
`COMPILE_CUSTOM_KERNELS=1`): the build reaches 93 %, `ascendc_pack_kernel` prints

```
recompile: /usr/bin/c++ … -o lib/libvllm_ascend_kernels.so … \
  …/host_bisheng_obj/efs/<user>/projects/vllm-ascend/csrc/kernels/pos_encoding_kernels.cpp.o \
  …/host_bisheng_obj/<HOME>/work/<user>/projects/vllm-ascend/csrc/kernels/pos_encoding_kernels.cpp.o …
/usr/bin/ld: … multiple definition of `rope_custom_true_half'; … first defined here
```

for a whole file of symbols (`rope_custom_*`, `rotary_embedding_impl`), ending in
`recompile_binary.py` `CalledProcessError` and `setup.py:250` `cmake --build` error 2.
**That `recompile:` line is the finding**: the same `csrc/kernels/pos_encoding_kernels.cpp`
appears twice, as two `.o` files under two absolute prefixes of one tree. It says
the tree was compiled under two path spellings, not that the kernel source, the C++
or the version set is wrong.

Why that lands on the build directory, not on the tree (source-verified at the pin):
`CMakeLists.txt:46-47` is `file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/pos_encoding_kernels.cpp)`
— an exact path, not a wildcard — feeding `ascendc_library(vllm_ascend_kernels SHARED …)`
(`:49`), and `setup.py:112` derives the source dir with `os.path.abspath(ROOT_DIR)`
(physical). **One configure therefore yields exactly one source entry**, so two
spellings must come from outside it: a `build/` tree left by an earlier compile under
the other path, or one directory reachable under two names (symlinked/bind-mounted
home). The pin's `setup.py` also configures the same `build_temp` twice (`:205` and
`:210-215`), so stale objects are never re-derived — worth one line in the answer,
labelled an observation, not the cause.

Three lines from the reporter decide which:

```bash
pwd; pwd -P; readlink -f .        # logical vs physical cwd
env | grep '^PWD='
grep -c 'csrc/kernels/pos_encoding_kernels.cpp' \
  build/temp*/CMakeFiles/vllm_ascend_kernels.dir/*/link.txt   # 2 = duplicated
ls build/temp*/vllm_ascend_kernels_host_dir/objects-Release/host_bisheng_obj/  # two roots = stale
```

When `build/` is already gone, the paste is the record and the shape above is enough.

## Step 3 — the ladder (a working install first, the mechanism second)

1. **Skip the custom-op build** — one command, and the rung that ended #1774 — but
   **say whom it does not answer: a reporter who is developing custom ops.**
   Recorded #676: 「Similar to issue #591 … set COMPILE_CUSTOM_KERNELS=0」 was met with
   「We want to use the ability of custom operators for develop」, and a following
   「#664 might be useful」 with 「it still has the same problem using #664」. For that
   reporter the skip is a *diagnostic* rung (it separates "the extension build is
   broken" from "the Python metadata is broken"), never the fix — lead with the missing
   diagnostic and the step 1b/1c lanes, and label this one with its cost.

   ```bash
   cd "$(pwd -P)"
   unset COMPILE_CUSTOM_KERNELS
   rm -rf build
   pip install --no-build-isolation -e .
   ```

   Source-verified consequence at the pin: `build_extensions` returns at once
   (`setup.py:217-218`; `:207-209` at v0.8.4rc1), the extension is never declared
   (`:297-298`; the `ext_modules` block is itself gated by
   `if envs.COMPILE_CUSTOM_KERNELS:` at `setup.py:288` at v0.8.4rc1), and at import
   `vllm_ascend/platform.py:28-38` swallows the `ImportError` and logs
   `Warning: Failed to register custom ops, all custom ops will be disabled`. Say what
   it costs: no custom ops, and the pinned page states the custom-op build is what
   sleep mode needs (`docs/source/installation.md:156`).
2. **Rebuild with custom ops from one canonical path** — `rm -rf build`, then the same
   `pip install -e .` from the physical path; if the tree must live behind a mount,
   build the physical copy and stop entering it through the alias.
3. **Use the documented command.** The pin's page installs with `pip install -e .`
   (`:151`) and `--no-build-isolation` when the env's torch-npu would otherwise be
   replaced (`:157`); `python setup.py install` is what put the reporter on the
   `setup.py:250` traceback in the first place.

## Delivery

A build report is answerable from the paste plus one docs grep: stage, owner, version
envelope and rung 1 need no local source read, so the ranked answer is owed by the
third message (SKILL.md rule 1). #1774's recorded run used 16 assistant turns,
39.7 K characters of reasoning against 4.1 K of message text, and 15.6 K of the
16.4 K completion-token cap — the answer arrived on the last turn and only just.
Its final claim (path alias) stays a hypothesis and the outcome is
reporter-confirmed only at the level of 「Thanks a lot. Solved!」.

## Not verified

- Which side minted the second spelling: CANN's `ascendc.cmake` /
  `recompile_binary.py` are not in the snapshot, so never publish "symlinks are the
  cause"; give the three-line check above instead.
- Whether a `vllm_ascend_C`-less install is acceptable for the reporter's workload is
  theirs to decide — answer with the cost (no custom ops, sleep mode) and let them
  choose.
- The pin predates the v0.13.0 build refactor, so `build_aclnn.sh`, the target names
  and the `--no-build-isolation` advice are version-specific.
