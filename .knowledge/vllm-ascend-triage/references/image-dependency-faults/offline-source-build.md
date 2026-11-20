# Building vllm-ascend from source in an offline pod: the third-party fetch inventory

Pin: vllm-ascend `v0.13.0rc3` (`bbe9bc0d90f3`), the tree the recorded report used;
the chain was re-verified there by reading the files below. Re-check the paths at
another ref — the custom-op build was refactored around v0.13.0 and the same
scripts carry several SoC branches.

## Report shape

"Starting from vX, installing vllm-ascend from source in a pod without public
network fails: the custom ops compilation step pulls dependencies from the
public network." Usually no error text, sometimes `fetch failed` or a cmake
download error.

## The invocation chain (source-verified)

1. `pip install .` / `pip install -e .` with `COMPILE_CUSTOM_KERNELS=1` (the
   default: `vllm_ascend/envs.py:44-45`) declares the extension
   (`setup.py:489`); `cmake_build_ext` does **not** configure `csrc/` — the root
   `CMakeLists.txt` builds only `vllm_ascend_C` + `vllm_ascend_kernels`.
2. `cmake_build_ext.run()` (`setup.py:466-470`) calls `run_command("build_aclnn")`
   first; that command (`setup.py:235-258`) runs
   `bash csrc/build_aclnn.sh <ROOT_DIR> <soc>`.
3. `csrc/build_aclnn.sh` branches on SoC (310P returns early; A2 `:12`, A3
   `:59`, A5 `:137`). Each live branch requires `csrc/third_party/catlass/include`
   and, if absent, runs `git submodule update --init --recursive`
   (`:19/:66/:144`) → `.gitmodules` → `https://gitcode.com/cann/catlass.git`.
   `csrc/third_party` is also in `.gitignore:214`, so a plain source checkout has
   no catlass and this fetch always runs.
4. `csrc/build_aclnn.sh:201` → `bash build.sh --pkg --ops=… --soc=…` →
   `csrc/build.sh` sets `CANN_3RD_LIB_PATH="${CURRENT_DIR}/third_party"`
   (`:73`), overridable with `--cann_3rd_lib_path=<PATH>` (`:1140-1142`), and
   passes `-DCANN_3RD_LIB_PATH=…` (`:1310`) into the configure of
   `csrc/CMakeLists.txt`.
5. `csrc/CMakeLists.txt` includes the four fetchers: `json.cmake:103`,
   `abseil-cpp.cmake:121`, `ascend_protobuf.cmake:122`, `makeself-fetch.cmake:124`
   (plus `gtest.cmake` only when `ENABLE_TEST`, default `OFF` at `:21`).

## What is fetched, and the gate that suppresses it

| Dependency | Source | Offline gate (pre-seed this) |
|---|---|---|
| catlass | `git submodule` gitcode (`gitmodules`) | populate `csrc/third_party/catlass/include` (submodule init) |
| json v3.11.3 (`include.zip`) | `json.cmake:46` gitcode | extracted include at `${CANN_3RD_LIB_PATH}/json/include/nlohmann/json.hpp` (`:27-41`); **no `file://` fallback** — seeding `pkg/include.zip` alone is not the documented gate |
| abseil-cpp 20230802.1 | `abseil-cpp.cmake:42` gitcode | `${CANN_3RD_LIB_PATH}/abseil-cpp/abseil-cpp-20230802.1.tar.gz` or `${CANN_3RD_LIB_PATH}/pkg/<same>` (`:35-40`), or the extracted source found by `find_path` (`:17-31`) |
| protobuf 25.1 (`ascend_protobuf`) | `ascend_protobuf.cmake:40` gitcode | `.../protobuf/protobuf-all-25.1.tar.gz` or `.../pkg/protobuf-25.1.tar.gz` (`:33-38`) — note the two file names differ; or an installed `ascend_protobuf` that satisfies the package check (`:23`) |
| makeself 2.5.0-patch1 | `makeself-fetch.cmake:16` gitcode, via `FetchContent` | `makeself.sh` **and** `makeself-header.sh` present under `${CANN_3RD_LIB_PATH}/makeself` (`:14`) |
| gtest 1.14.0 | `gtest.cmake:68` gitcode | only with `ENABLE_TEST=ON`; skip otherwise |

Two knobs that look like offline switches and are not:

- `CANN_PKG_SERVER` rewrites the URL in `protobuf.cmake` and `secure_c.cmake`
  only, and **neither file is included** by `csrc/CMakeLists.txt` at this pin
  (`secure_c.cmake` also defaults its first URL to an empty string). Setting it
  changes nothing for the four live fetches.
- `export CANN_3RD_LIB_PATH=…` does **not** reach the pip build: `setup.py`
  `configure()` (`:289-390`) forwards only `ASCEND_HOME_PATH`, `PYTHON_*`,
  `SOC_VERSION`, `FETCHCONTENT_BASE_DIR`, `TORCH_NPU_PATH` and the compilers, so
  CMake falls back to `${PROJECT_SOURCE_DIR}/third_party` = `csrc/third_party`.
  Redirect it with `bash csrc/build.sh --cann_3rd_lib_path=<abs>` or seed the
  default directory.
- `COMPILE_CUSTOM_KERNELS=0` exists (`envs.py:44-45`, `setup.py:395,489`) but its
  own comment restricts it to UT without an NPU, and the tree's release notes
  (`docs/source/user_guide/release_notes.md`, v0.12.0rc1 section) state the env was
  *removed* because custom ops are mandatory — do not offer it as the deployment
  workaround; mention it only as "skips the build, engine then lacks the ops".

## Answer shape

1. Confirm the report's premise as fact: name the fetch legs (catlass submodule
   first, then the four cmake fetches) with their files, so the reporter can see
   which one their log stopped on.
2. The workaround ladder: pre-seed `csrc/third_party` (table above) and the
   catlass submodule, or point an internal mirror at those URLs, or run
   `csrc/build.sh --cann_3rd_lib_path=<pre-populated>`. A pip mirror alone does
   not fix a cmake fetch.
3. Ask which leg failed (`fetch failed` from `build_aclnn.sh` vs a cmake
   `Downloading …` line) — that decides between the submodule and the tarballs and
   is one line from the reporter.
4. State that the image path (`Dockerfile*`) has network at build time, so "it
   works in the image" is not evidence about the pod.

## Not verified

- No run of this chain was performed here; the ordering is source-read, and
  whether a pre-seeded cache is honoured depends on CMake's
  `ExternalProject`/`FetchContent` reuse rules for the file names above.
- The A5 branch's op list and the exact SoC map (`setup.py:345-352`
  `910b`/`910c`/`310p`) are version-specific.
