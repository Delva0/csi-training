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

Not this page: a pip **resolution** error (`pip-index-visibility.md`), the
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
| `ccec_compiler`, `ascendc_pack_kernel`, `[NN%] Building … _aiv_device` | AscendC kernel compile/pack | the CANN toolkit | the kernel file the compiler named |
| CANN `recompile_binary.py` → `CalledProcessError`, `ld returned 1 exit status`, `collect2: error: ld returned 1 exit status` | kernel host **relink** | the CANN toolkit | the `recompile:` line it just printed |
| `…host_stub_obj…`, `Linking CXX shared library lib/libvllm_ascend_kernels.so` | host stub link | the CANN toolkit | the object list on that line |

`CMakeLists.txt:35-45` picks the toolkit directory that exists — `tools/tikcpp/
ascendc_kernel_cmake`, then `compiler/…`, then `ascendc_devkit/tikcpp/samples/cmake`
— and includes its `ascendc.cmake`. So a paste whose frame is
`…/tools/tikcpp/ascendc_kernel_cmake/util/recompile_binary.py` came through the
**first** branch: the branch choice is not the discriminator there, and both
`ascendc.cmake` and `recompile_binary.py` are CANN files a snapshot usually does not
export (`SOURCES.json` `unavailable`) — label any mechanism inside them **open** and
do not spend turns reading `csrc/`.

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

1. **Skip the custom-op build** — one command, and the rung that ended #1774:

   ```bash
   cd "$(pwd -P)"
   unset COMPILE_CUSTOM_KERNELS
   rm -rf build
   pip install --no-build-isolation -e .
   ```

   Source-verified consequence at the pin: `build_extensions` returns at once
   (`setup.py:217-218`), the extension is never declared (`:297-298`), and at import
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
