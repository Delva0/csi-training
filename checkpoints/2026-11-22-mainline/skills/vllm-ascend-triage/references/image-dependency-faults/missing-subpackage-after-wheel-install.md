# `ModuleNotFoundError: No module named 'vllm_ascend.<sub>'` on an installed release

Scope: a serve started from a **pip wheel** (not the image's editable checkout) dies
after weights load with `ModuleNotFoundError` naming a **pure-Python** `vllm_ascend`
subpackage — `vllm_ascend.compilation`, `.distributed`, `.attention`, `.lora`,
`.models.interfaces`. The traceback is the giveaway: the deepest frames are
`vllm/compilation/backends.py:298` →
`vllm/utils.py::resolve_obj_by_qualname` → `importlib`, i.e. *dynamic* resolution of a
qualname the engine itself asked for, not an import at module top. Recorded #1156
(vllm-ascend v0.9.0rc1, vllm v0.9.0, first request of the default `-O3` serve).

Not this page: `No module named 'vllm_ascend'` (the whole package — plugin metadata
residue, `plugin-load-module-missing.md`), `ImportError: …so: undefined symbol`
(`abi-undefined-symbol.md`), and `vllm_ascend.vllm_ascend_C` / `vllm_ascend._C`
(the compiled extension was never built for that SoC or that install,
`../soc-platform-paths/kernel-package-vs-card.md` "red herring" section,
`launch-env-and-library-path.md`).

## Step 1 — who asks for the name (one grep, decides the lane)

```bash
rg -n "<missing.dotted.name>|vllm_ascend\.<sub>" <pinned vllm-ascend tree> --glob '!docs/**'
```

- **The string is produced by `vllm_ascend/` itself** — the artifact contradicts its
  own source, so the subpackage was left out *of the distributable*. That is a
  packaging defect; go to Step 2. (At #1156's pin: `vllm_ascend/platform.py:263-267`
  returns `"vllm_ascend.compilation.piecewise_backend.NPUPiecewiseBackend"` from
  `get_piecewise_backend_cls()`, and upstream calls it at
  `vllm/compilation/backends.py:298-299`.)
- **The string comes from `vllm/` or another package** — a version contract
  (`installed-version-vs-checkout.md`), not packaging: that library is newer/older
  than the vllm-ascend it runs against.

## Step 2 — the packaging test, at the pinned commit (one call, two greps)

```bash
cd <vllm-ascend>; rg -n 'find_packages|packages=' setup.py
for d in $(find vllm_ascend -mindepth 1 -maxdepth 2 -type d ! -name __pycache__); do
  [ -f "$d/__init__.py" ] || echo "NO_INIT $d"; done
```

`find_packages()` — as opposed to `find_namespace_packages()` — yields **only**
directories carrying `__init__.py`, so any `vllm_ascend/<sub>/` without one is
silently absent from the wheel while remaining importable from a source checkout.
Verified at vllm-ascend `v0.9.0rc1` (`706de0231762`): `vllm_ascend/compilation/`
contains exactly one file, `piecewise_backend.py`, with **no** `__init__.py` — the
only top-two-level directory in that state — while `setup.py:357` is
`packages=find_packages(exclude=("docs", "examples", "tests*", "csrc"))` and
`pyproject.toml` declares plain `setuptools.build_meta` (no namespace override).
The image is unaffected only because `Dockerfile:52` installs it editable
(`python3 -m pip install -v -e /vllm-workspace/vllm-ascend/`), which is why "works in
the container, fails with `pip install`" is the same bug and not a stale checkout.

Two tells in the paste that usually follow: the error names the **parent** package
(`vllm_ascend.compilation`, not `…piecewise_backend`) because the import dies on the
package itself, and it appears **after** `Starting to load model` / weight load —
the first place the qualname is resolved. An empirical
`from setuptools import find_packages` demo is optional (the greps above already
decide it); if you run one, the base env has no setuptools, so use
`uv run --no-project --with setuptools python -c '<snippet>'`.

## Step 3 — three lines that confirm it on the reporter's box

```bash
python3 -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))"   # then: ls <dir>/compilation
pip show -f vllm-ascend | grep -c compilation    # 0 = the wheel really omits it
python3 -c "import vllm_ascend.compilation"      # ModuleNotFoundError = same defect
```

A non-zero file list **with** the directory present moves the case to shadowing /
two installs (`installed-version-vs-checkout.md`, `wrong-vllm-module-shadowed.md`):
print `vllm_ascend.__file__` and `sys.path[:5]` before proposing any reinstall.

## Action ladder

1. **`--enforce-eager` is the bridge, not the fix.** `vllm_ascend/platform.py:167-169`
   sets `CompilationLevel.NO_COMPILATION` and logs
   `Compilation disabled, using eager mode by default`, so the piecewise backend (and
   therefore the import) never runs. It costs graph capture and leaves the defect for
   the next request that uses the default level, so pair it with row 2.
2. **Install from source at the tag, or move to the next rc — reporter-confirmed on
   #1156** (comment, 2025-06-10, by the reporter): `pip uninstall vllm-ascend`,
   `git clone --depth 1 --branch v0.9.0rc2 https://github.com/vllm-project/vllm-ascend.git`,
   `cd vllm-ascend && pip install -v -e .`, and the issue's own resolution line is
   "Upgrade to v0.9.0rc2". A later rc's artifact is the only check that can see the
   repaired wheel; the snapshot carrying one commit cannot.
3. If the wheel must be kept, copy the directory into site-packages **and** add the
   missing marker (`touch …/vllm_ascend/compilation/__init__.py`) — without the
   marker the next wheel built from that tree drops it again.

Report the recipe to the project as release-blocking when the missing module is on a
**default** path (`-O3` piecewise compile is the default at v0.9.0rc1), because every
pip user hits it on the first request.

## Precedent family (title/line evidence only — reuse the shape, not a cause)

`#415` `vllm_ascend.attention` on a local `0.7.3rc2.dev9+g7fbdd01` build; `#592`
`vllm_ascend.lora` on v0.8.4rc1; `#1093` `vllm_ascend.distributed.device_communicators`;
`#1209` `vllm_ascend.distributed`; `#2929` `vllm_ascend.models.interfaces`. None of
those bodies records a confirmed fix, so cite them as "same shape on other releases"
and let Step 2 decide *this* report. `vllm_ascend._version` (#612, #3403) is the
adjacent producer: that module is written by the build step, so its absence means the
artifact was assembled without one — an install question, not an engine question.

## Recheck conditions

`find_packages` vs `find_namespace_packages`, the `__init__.py` set, the qualname in
`platform.py`, and the `-e` install in the `Dockerfile` all move between releases. Run
Step 2 at the pinned commit every time; a page-level "fixed in the next rc" claim is
reporter-confirmed only for the release it was reported against.
