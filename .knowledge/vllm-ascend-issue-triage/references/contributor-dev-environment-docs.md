# Contributor / developer-environment doc reports (`testing.md`, contribution index, nightly-main)

Verified in the read-only clones on 2026-11-01, `main` unless a tag is named. Use for `[Doc]` and
`[Usage]` reports shaped "I started from the contributor guide (`main`), followed the documented
commands to build a dev/test environment or run UT, and they do not work" — mounting a checkout into
the image, `pip install -e .`, `bash format.sh ci`, "installation.html is user-facing only". The
deliverable is a doc-vs-reality inventory (`file:line` per claim), not a code fix, and these pages
move between releases.

## Locate the pages before quoting any command

- `docs/source/developer_guide/contribution/testing.md` — `## Set up a test environment` with
  tab items `=== "Local (CPU)"` and `=== "Single-card"`, then per-scope `pytest` sections.
- `docs/source/developer_guide/contribution/index.md` — the "Run CI locally" section (fetch
  `.github/vllm-main-verified.commit`, `VLLM_TARGET_DEVICE="empty" pip install .` for vLLM, then
  `pip install -r requirements-dev.txt`, `bash format.sh ci`).
- `docs/source/getting_started/installation.md` — an include-based page
  (`installation/{prebuilt_image,install_vllm_ascend,existing_cann_environment,base_environment}.inc.md`).
  A claim about "installation.html" needs the include fragment that renders the quoted block, not
  just the top page.

## The per-tab inventory decides the report — read the tab the reporter followed

`testing.md` at `main`:

- **`Local (CPU)`**: mounts `-v $(pwd):/vllm-project` into `quay.io/ascend/cann:{{ cann_image_tag }}`,
  clones `vllm`/`vllm-ascend`, exports `SOC_VERSION="ascend910b1"` (`:57-59`),
  `VLLM_TARGET_DEVICE=empty python3 -m pip install .` for vLLM, `python3 -m pip install .` for the
  plugin, then `requirements-dev.txt`. The `SOC_VERSION` and from-source install steps live **here**;
  neither testing tab sources `set_env.sh` (it exports `LD_LIBRARY_PATH` only, `:56`/`:155`).
- **`Single-card`** (`:64-100`): `docker run` on `quay.io/ascend/vllm-ascend:nightly-main` with only
  driver/cache mounts — no checkout mount — then `cd /vllm-workspace/vllm-ascend/` and
  `pip install -r requirements-dev.txt`; the UT section runs `pytest -sv tests/ut` (`:149-164`, one
  tab prefixed `TORCH_DEVICE_BACKEND_AUTOLOAD=0`). Nothing here reinstalls the plugin after a mount.

So "the guide never mentions `SOC_VERSION`" is true of the Single-card tab and false of the CPU tab.
Name the tab. The same applies to the reporter's other items: whether a step is missing is a
per-tab statement.

## Before accepting "the docs do not mention X", grep `X` across `docs/` at the ref

The guidance usually exists on a sibling page, in another tab, or in an include fragment. At `main`:

- `installation.md:136-145`, under `### CPU-only build verification` (`:112`), already documents the
  trio `COMPILE_CUSTOM_KERNELS=0 SOC_VERSION=ascend910b1 python -m pip install --no-build-isolation
  --no-deps -e .` (`:138-139`), with a bullet per variable (`:142-145`) and the note (`:147`) that
  the assignments do not persist.
- `installation.md:163` already cross-references contributors: "For CPU unit tests, prepare a
  separate environment using [Testing > Set up a test environment] ... [Contributing > Run CI
  locally] provides the checkout and installation commands."
- `vllm_ascend/envs.py` documents `COMPILE_CUSTOM_KERNELS` in-code (`main:39-44`; a report quoting
  `envs.py:51` is quoting an older line number): "should only be set to False when running UT
  scenarios in an environment without an NPU. Do not set it to False in other scenarios."
- The CANN/NNAL `set_env.sh` pair is on the installation page, not in `testing.md`:
  `installation/base_environment.inc.md:34` (`ascend-toolkit/set_env.sh`) and `:45`
  (`nnal/atb/set_env.sh`), with `existing_cann_environment.inc.md:39-45` naming the
  "`libatb.so` not found" failure and the non-default-path caveat. So "the UT guide never says to
  source them" is true of `testing.md` and false of the docs tree.

A report that frames these as absent is then a **cross-reference / discoverability** request, not
missing content — answer with the page and line that carries each item, and keep only the genuinely
missing pieces (see the mount/reinstall mechanism below) as doc defects.

## Mounting a checkout over the image's plugin: `_build_info.py` is generated and gitignored

- `setup.py:157-165 gen_build_info()` writes `vllm_ascend/_build_info.py`; it runs from
  `custom_develop` (`:172-177`) and `custom_build_info` (`:180-183`, registered at `:466`), and
  `.gitignore:212` excludes the file. It carries `__device_type__` (and older `__soc_version__`).
- `vllm_ascend/device/device_config.py:29-48` (`_device_type_from_build_info`,
  `get_ascend_device_type`) does `from vllm_ascend import _build_info` with **no** try/except, so a
  plain checkout mounted over `/vllm-workspace/vllm-ascend` (where the image installed the plugin)
  shadows the generated module and the first device-type query raises `ImportError` — reported form
  `ImportError: cannot import name '_build_info'`.
- Fix, and the sentence the Single-card tab lacks: reinstall inside the container,
  `pip install -e . --no-build-isolation`. The image already ships the build deps, so build
  isolation only re-downloads them (the reporter reports a mirror 403 doing exactly that).
  A reinstall also re-bakes `SOC_VERSION`; without it the package targets `ASCEND910B1`, which on
  310P is the wrong-variant trap in [310p-atlas-300i-duo.md](310p-atlas-300i-duo.md).

## `bash format.sh ci` is the pre-commit lint entry, not the test suite

`format.sh:36-43` at `main`: `ci` as `$1` runs `pre-commit run --all-files --hook-stage manual`;
anything else runs `pre-commit run --all-files`; `check_command pre-commit` gates both. So
"`bash format.sh ci` does not run CI's UT/E2E" is a naming/scope gap (and a missing cross-link to
`testing.md`), not a wrong command — the recorded `#17718` framing "`format.sh ci` ≠ running CI
tests" is an inference from the name, not a doc error. Tests are `testing.md`'s `pytest` sections.

## Version and image claims in these reports

- The install pages' image tag comes from `mkdocs.yml extra:` substitutions, which at `main` still
  read `vllm_version: v0.23.0` / `vllm_ascend_version: v0.23.0` (`:166-172`). "installation.html
  recommends `v0.23.0` while main is far ahead" is substitution drift, not a pin mistake, and the
  `nightly-main` / `nightly-main-a3` / `-openeuler` variants are the development tags
  ([model-tutorial-recipes.md](model-tutorial-recipes.md)).
- Mirror/timeouts (`mirrors.tuna.tsinghua.edu.cn` 403, a five-minute `requirements-dev.txt`
  resolve) are reporter-side environment state: give the flag that avoids the fetch
  (`--no-build-isolation`, pre-installed deps) and label the mirror as environment, not a defect.

## Boundary

Nothing here was executed in the image; every claim is source-level (`setup.py`, `device_config.py`,
`format.sh`, `envs.py`, the docs) at `main` on the verification date. The recorded thread (`#17718`)
has one non-maintainer comment proposing a plan and no maintainer root cause, so nothing in it
establishes that any of these gaps was accepted as a doc fix.
