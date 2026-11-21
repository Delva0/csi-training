# `check_ascend_device_type()`: the build stamp vs the live SoC

Scope: engine start dies **inside worker construction**, before any weight loads —
`EngineCore failed to start` with the last application line being `Initializing a V1
LLM engine`, and **no** `Starting to load model`. The traceback's deepest frame is
`vllm_ascend/worker/worker.py:113 check_ascend_device_type()` →
`vllm_ascend/utils.py`. Two different failures come out of that one function, and
they are not the same report:

| Message | Meaning | Page |
|---|---|---|
| `AssertionError: Current device type: AscendDeviceType.A5 does not match the installed version's device type: AscendDeviceType.A2, please check your installation package.` | the wheel was **built** for one SoC family and runs on another | steps 1-5 below |
| `RuntimeError: Can not support soc_version: 104.` | the live SoC reports a code the map does not contain (driver/firmware/NNAL pairing, not the wheel) | step 6 |

Verified at vllm-ascend **v0.21.0rc1** (`80610e4`) with vllm **v0.21.0**
(`ad7125a`); the map and the stamp location move between releases, the shape does
not — re-grep before quoting a line number.

## Step 1 — read both sides out of the assertion text

`utils.py:806-809` formats the message as
`Current device type: <live> … installed version's device type: <stamp>`. The
**first** value is `torch_npu.npu.get_soc_version()` classified by the live map;
the **second** is what the installed package was built as. So
`…A5 … AscendDeviceType.A2…` means "A5 silicon (950-class), A2 build" — not the
reverse. Getting this backwards produces a confident wrong hardware diagnosis.

## Step 2 — two one-liners settle it before any rebuild

`bash <skill_root>/scripts/device-type-check.sh <their-paste>` does the parsing for
you: it names which side of the assertion is the card, classifies a raw
`soc_version`, and flags the non-fatal `liboptiling.so` line. Then ask for:

```bash
python -c "import torch_npu; print(torch_npu.npu.get_soc_version())"            # 260 = A5
python -c "import vllm_ascend._build_info as b; print(b.__device_type__)"       # 'A2'/'A3'/'_310P'/'A5'
```

`_init_ascend_device_type()` (`utils.py:778-786`) reads
`_build_info.__device_type__`; only when the stamp is absent does it fall back to
`__soc_version__` and then to the literal default `"ASCEND910B1"` → A2. A package
that has no `_build_info` at all therefore behaves as an A2 build on every SoC —
that fallback is the second way to see this assertion, and it points at the
installer (`pip` built from an sdist without `SOC_VERSION`), not at the card.

Live map (`utils.py:794-804`): `220-225` A2 · `250-255` A3 · `200-205` `_310P` ·
`260` A5 · anything else → `RuntimeError`.

## Step 3 — how the stamp gets baked in

- `setup.py:134` — `if not envs.SOC_VERSION: soc_version = get_chip_type()`: on a
  host where `npu-smi` answers, the chip is read from `npu-smi info -t board`, and
  `setup.py:74-131` returns `(chip_name + "_" + npu_name)` when `"950" in
  chip_name` (it also handles the 310P/910B/910C shapes). **A source build on the
  reporter's own node self-selects its SoC**, which is why the source-build route
  is the robust fix and a copied wheel is not.
- `setup.py:150-190` `gen_build_info()` — one `soc_to_device` table holds both the
  short aliases (`910b`→A2, `910c`→A3, `310p`→`_310P`) and the concrete strings
  (`ascend910b1/b2/b2c/b3/b4/b4-1`→A2, `ascend910_9391/9381/9372/9392/9382/9362`
  →A3, `ascend310p1/p3/p5/p7`+vir→`_310P`); **any** `soc_version` containing
  `ascend950` → A5 is checked *before* the table; an unknown value hits
  `assert soc_version in soc_to_device`, and the result is written to
  `vllm_ascend/_build_info.py` as `__device_type__ = '<X>'`.
- `docs/source/installation.md:236-241` — the documented `SOC_VERSION` values for a
  CPU-only/no-`npu-smi` build: `ascend910b1`, `ascend910_9391`, `ascend310p1`, and
  for A5 *only* "`<value starting with "ascend950">`". The concrete string lives in
  `Dockerfile.a5:60` (`SOC_VERSION="ascend950dt_9582"`, base image
  `quay.io/ascend/cann:9.0.0-950-…`). Use the value the node reports
  (`npu-smi info -t board -i 0`), not the Dockerfile's, when they differ. The
  autodetect path is young and still moving: `#8238` (v0.18.0 release feedback) has
  "Ascend950 compatibility is improved for updated `npu-smi info` behavior and
  `SOC_VERSION` detection", and `#7157` is the A5 roadmap RFC — cite them as
  pointers, not as proof about the reporter's host.

## Step 4 — check whether an artifact for that SoC exists at all

The gap in the recorded case was packaging, not code:

- `.github/workflows/schedule_release_code_and_wheel.yml` builds variant wheels in
  three jobs only — `Dockerfile.buildwheel.a2` (`-l 910b`, ~`:122`), `.a3`
  (`-l a3`, ~`:206`), `.310p` (`-l 310p`, ~`:284`) — then uploads one variant
  index. **There is no A5 job.**
- **PyPI receives exactly one wheel per release, and it is the A2 base build.** At
  v0.21.0rc1 the only wheel job with a `twine upload` is the A2 one
  (`…schedule_release_code_and_wheel.yml:175`, `SOC_VERSION: ascend910b1` at `:78`,
  `Dockerfile.buildwheel.a2:20`); the sole other `twine upload` (`:92`) ships the
  sdist. The A3 (`:245` `-l a3`) and 310P jobs stop at `upload-artifact`. So a
  generic-named `…-manylinux_2_24_aarch64.whl` from pypi.org is A2 **by
  construction**, which is why reported 0.13.0/0.18.0 wheels both carried
  `__device_type__ = 'A2'` (#11160).
- **The non-A2 wheels do exist, as wheel *variants* on the Ascend mirror.** The three
  jobs' `dist/variants/*.whl` are collected, indexed with `variantlib
  generate-index-json` (`:362`) and uploaded to OBS `pypi/packages/ascend/repos/pypi/
  variant/vllm-ascend` (`:376-385`), which is what
  `https://mirrors.huaweicloud.com/ascend/repos/pypi/variant` serves; the variant
  labels are `ascend :: npu_type :: {910b,a3,310p}`
  (`.github/workflows/scripts/wheel/config.json`) and the filenames carry the label as
  a suffix (same dir, `make_variant.py:65` `<base>-<label>.whl`).
  The pin's own install command already passes it — `installation.md:159` (pip tab) and
  `:185` (uv-wheelnext tab) both use `--extra-index-url …/pypi/variant` — so
  **following the docs selects the SoC variant while a bare `pip install vllm-ascend`
  against pypi.org always lands A2**. That asymmetry, not a missing artefact, is the
  recorded #11160 report; answer it with the docs command before proposing a rebuild.
- Variant *selection* needs variant-capable tooling (that is why the second docs tab
  installs `uv-wheelnext`): plain pip sees only the base wheel even with the index
  listed. Check the artefact before promising it, per version:
  `pip download --no-deps -d /tmp/v --extra-index-url
  https://mirrors.huaweicloud.com/ascend/repos/pypi/variant vllm-ascend==<their
  version>` then `ls /tmp/v | grep -i a3`; an empty result means the variant
  machinery postdates that release (only one commit per repo is pinned, so 0.13.0-era
  trees cannot be checked locally) and the source build *was* the right route there.
- `.github/workflows/dockerfiles/` contains `Dockerfile.buildwheel.{a2,a3,310p}`
  and `Dockerfile.nightly.{a2,a3}` only.
- `docs/source/installation.md` prebuilt-image table lists plain (A2), `-a3`,
  `-310p` tags — no `-a5`.
- In-tree, the A5 path does exist: `Dockerfile.a5` / `Dockerfile.a5.openEuler`, and
  `docs/source/user_guide/release_notes.md` "Ascend950 chip is now supported"
  (#7151). So **the source tree supports A5 while the published wheel index cannot
  hand you an A5 build** — say exactly that; it is the difference between "not
  supported" (wrong) and "not shipped as a wheel" (right).

Search the mirror before answering if the reporter insists a wheel exists: the
variant index may have gained a label after the pinned commit, and `SOURCES.json`
pins one commit per repo.

## Step 5 — the action ladder, cheapest first

1. The two one-liners in step 2. If the stamp is the wrong family, nothing about
   the model, dtype, TP size or prompt matters.
2. For A2/A3/310P, re-install with the docs command (variant index on the index
   list) before any rebuild; assert the result with the step-2 one-liner instead of
   trusting the download name. Rung 1.5 is the source build below, which self-selects.
3. Install an A5-capable build, in this order of preference: a newer official
   `-a5` image/wheel if one exists → source build **on the node**
   (`pip install --no-build-isolation -e .`, autodetect via `npu-smi`) → source
   build with `export SOC_VERSION=<ascend950…>` → `docker build -f Dockerfile.a5`.
   Re-run `print(_build_info.__device_type__)`; it must read `A5`.
4. **Do not patch the assertion.** See step 6: the guard is the cheapest place to
   fail, and every downstream op is worse.

## Step 6 — the `RuntimeError: Can not support soc_version: N` branch

Same function, no wheel involved. Recorded in **#10206**: a 910B host whose
`torch_npu.npu.get_soc_version()` returns **104** (outside `220-225`) gets
`RuntimeError: Can not support soc_version: 104.`; the reporter edited `utils.py`
to treat 104 as A2 and the engine then failed in the first `profile_run`
(`qkv_proj` → `torch.nn.functional.linear`, MatMul/TransData). The decisive check
is a framework-free repro on the reporter's host — if a bare
`torch.nn.functional.linear` on two ones-matrices fails, the defect is below
vllm-ascend and the answer is the torch_npu/CANN/driver pairing for that SoC
reporting (the same symptom is tracked as #6796). #10206's title is the argument
against patching the guard; quote it.

## What the A5 answer usually still has to add

Fixing the package does not make the workload validated: at this pin
`docs/source/user_guide/support_matrix/supported_models.md` carries a
**Supported Hardware** column whose values are `A2`/`A3` (Qwen3.5-27B and
Qwen3.5-397B-A17B rows at `:87-88`), and `grep -c A5` on that file returns **0**.
So state two things separately — "the install is wrong, here is the rebuild" and
"this model class has no A5 row at the pinned ref, so expect a second, separate
report after it starts". The usual second report is already known and is not
device-type-specific: on A5 `enable_custom_op()` is `False` by design, so **every**
`torch.ops._C_ascend.<op>` raises `AttributeError` and `build_aclnn.sh` has no
`ascend950` branch — a correct A5 install still has no custom kernels
(`../custom-ops-runtime-faults/custom-ops-disabled-by-device-type.md`).

## Secondary lines in the same log (do not rank them)

`Failed to load custom_transformer library … libcust_opmaster_rt2.0.so … Error
detail: liboptiling.so: cannot open shared object file` is **expected and
non-fatal**: `.github/workflows/scripts/wheel/auto_exclude.py` has
`liboptiling.so` in `BASE_EXCLUDES`, i.e. the wheel deliberately does not ship
CANN's copy; the vendored opmaster `.so` resolves it only if CANN's directory is
on the loader path. It prints "may cause missing functionality", never
`EngineCore failed to start`. An ATB ABI mismatch in the same `collect_env` dump
(`ATB_HOME_PATH=…/cxx_abi_1` while `LD_LIBRARY_PATH` lists `…/cxx_abi_0/…`) is a
separate candidate for a *later* failure; keep it out of the device-type answer
(`../image-dependency-faults/abi-undefined-symbol.md`).

## Recorded run: #12141 (Ascend 950PR, vLLM 0.21.0, vllm-ascend 0.21.0rc1)

`AssertionError: Current device type: AscendDeviceType.A5 does not match the
installed version's device type: AscendDeviceType.A2`, reached from
`worker.py:113` at `EngineCore failed to start`, on an 8-die 950PR host whose paste
also carries the `liboptiling.so` line. The run said the verdict in its third
assistant turn, then spent the session pinning the stamp's origin (`setup.py`), the
wheel matrix, the support matrix and four precedents (#7157 A5 roadmap, #9354
"有支持昇腾950PR的安装包吗", #10206, #8238 `npu-smi`/`SOC_VERSION` detection), and
answered at turn 28 — correct, but a run of this shape can answer at turn 3 and
spend the rest on the two one-liners plus the build route. Note the reporter's
`collect_env.py` proves the driver/CANN (`9.1.0`) side is installed; it does not
prove the wheel.
