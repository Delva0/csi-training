# sdist build dies at `pybind11 --cmake`: pybind11 3.0 renamed the option

## Scope

An `[Installation]` / 安装 report that fails **before any CMake work**, in `pip`'s
isolated build env, on a `pip install vllm-ascend==<v>` from the **sdist** (recorded
**#1931**: `0.8.5rc1`, aarch64, CANN `8.1.rc1-310p`, `torch-npu 2.5.1`, cmake 4.0.3,
pybind11 3.0.0):

```text
      running build_ext
      Found existing ASCEND_HOME_PATH: /usr/local/Ascend/ascend-toolkit/latest
      Found existing PYTHON_EXECUTABLE: /usr/local/python3.10.17/bin/python3.10
      usage: __main__.py [-h] [--version] [--includes] [--cmakedir] [--pkgconfigdir]
                         [--extension-suffix]
      __main__.py: error: unrecognized arguments: --cmake
      …
      subprocess.CalledProcessError: Command '['…/python3.10', '-m', 'pybind11', '--cmake']' returned non-zero exit status 2
```

inside `/tmp/pip-build-env-*/overlay/lib/python3.10/site-packages/setuptools/…`.

**Not this page**: a pip **resolution** error (`pip-index-visibility.md`), a compiler or
linker fault (`custom-op-build-stage.md`), a missing `torch_npu/…` header
(`custom-op-build-stage.md` §Step 1b), a network fetch at configure time
(`offline-source-build.md`).

## Mechanism

The failing statement is vllm-ascend's own `setup.py` → `cmake_build_ext.configure()`:
it shells out to `python -m pybind11 --cmake` to fill `-DCMAKE_PREFIX_PATH`. pybind11 3.0
dropped that spelling (renamed `--cmakedir`), and the paste **prints the valid list
itself** — that usage line is the proof, not a recall. Nothing has compiled yet, so no
CANN, AscendC, `SOC_VERSION` or `gcc` fact is involved.

Why a current host install does not prevent it: both requirement files leave the package
**unpinned**, so pip's isolated build environment (the `/tmp/pip-build-env-*/overlay/`
in the traceback) resolves the newest release whatever the host has.

| Anchor (vllm-ascend `v0.9.2rc1`, `b5b7e0ecc765`) | Fact |
|---|---|
| `setup.py:178-180` | the same call is `[python_executable, "-m", "pybind11", "--cmakedir"]` — **`--cmakedir` is already in the pinned tree**, i.e. the repair is code-side and shipped before this pin |
| `pyproject.toml:10`, `requirements.txt:8` | bare `pybind11` in `[build-system] requires` and in the runtime list — no cap exists to inherit, so `pip install "pybind11<3"` alone cannot help |
| `setup.py:181-183` | the `except subprocess.CalledProcessError` re-raises as `RuntimeError(f"CMake configuration failed: {e}")` — a wrapper; the argument error above it is the cause |
| `setup.py:90-93` | a **310X** build separately raises `SOC version 310 only supports custom kernels. Please set COMPILE_CUSTOM_KERNELS=1 to enable custom kernels.` — the next wall after the pybind11 one on a `-310p` base |

## Ladder (cheapest first)

1. **A release whose `setup.py` already uses `--cmakedir`** — source-verified at
   v0.9.2rc1. The snapshot holds one ref per repo, so it cannot date the change: say
   "≥ the release that carries `--cmakedir`", not a guessed tag, and give the one
   command that checks the reporter's own sdist (step 2).
2. **Patch the sdist** (`--cmake` → `--cmakedir`) and install the file:
   `pip download --no-deps --no-binary :all: vllm-ascend==<v>`,
   `grep -n -- '--cmake' vllm_ascend-<v>/setup.py`, edit, `pip install ./vllm_ascend-<v>.tar.gz`.
3. **Cap the build-time requirement**: `PIP_CONSTRAINT='pybind11<3' pip install -v vllm-ascend==<v>`.
   That constraints reach build dependencies is pip behaviour (**recall**, not verified
   here) — the acceptance is the log line naming `pybind11-2.x` in the overlay download,
   so ask for it rather than asserting the fix.
4. **`--no-build-isolation`** with the host env pre-populated (cmake, `pybind11<3`,
   `setuptools-scm>=8`, decorator, einops, `numpy<2.0.0`, scipy, pyyaml, wheel) — the
   project's own documented escape for a build-env conflict
   (`docs/source/installation.md:169-175` at this pin). Note this is the only rung that
   makes a host downgrade effective.
5. On 310P/300I, add the hardware contract to the same answer: `COMPILE_CUSTOM_KERNELS=1`
   plus gcc/g++ ≥ 8 and C++17 (same doc note), and the SoC-support timeline
   (`../soc-platform-paths/index.md`: 300I/310P arrives around v0.9.1rc1), so an `-310p`
   box should take the project's 310P image/tutorial rather than a from-source pip build.

## Do not

- Do not rerun `pip install pybind11<3` and call it fixed: build isolation re-resolves
  from `pyproject.toml`, so the host package is invisible. Both halves are in the paste —
  `Requirement already satisfied: pybind11 (3.0.0)` beside the `/tmp/pip-build-env-*/`
  traceback.
- Do not rank `cmake 4.0.3`. `CMakeLists.txt:1` at this pin is
  `cmake_minimum_required(VERSION 3.16)`, which cmake 4 still accepts (only `<3.5` was
  dropped); a cmake-4 failure would print `Compatibility with CMake < 3.5 has been
  removed` instead. (Note `vllm/docker/Dockerfile.rocm_base:48` pins `'cmake<4'` for a
  different build — not a vllm-ascend contract.)
- Do not read the `SetuptoolsDeprecationWarning: License classifiers are deprecated`
  block above it as the failure; it is noise at that `setuptools` version.

## Limits

- The reporter's sdist is in **no** snapshot, so `--cmake`'s line number (`<string>`
  line 153) is theirs, not citable; the pinned `--cmakedir` line is the counterpart and
  the version gap is the finding. Label the first release carrying the rename as open.
- The `PIP_CONSTRAINT` rung is pip behaviour quoted from recall: it stands only with the
  log line that shows which pybind11 the isolated env installed.
