# `pip show` reports a different version than the one installed

Scope: the report is a **version-string question**, nothing crashed — "`pip show
vllm-ascend` shows 0.9.0rc2 but I installed 0.9.1dev", "版本号显示不对", "is there a
`0.9.1dev` version at all", "which version am I actually running". No log, no
traceback, no worker. The owner is the **build's metadata**, not the code that gets
imported, so the deliverable is a decision (which of three causes) plus the
five-second command that proves which code is loaded — not a version survey.

## Verdict first

`pip show` prints the distribution metadata recorded when that wheel was built.
vllm-ascend has **no version constant in its source**: at v0.9.2rc1 (`b5b7e0e`)
`setup.py:305-311` is

```python
try:
    VERSION = get_version(write_to="vllm_ascend/_version.py")
except LookupError:
    VERSION = "0.0.0"      # CI checkout has no tag — the file's own comment
```

so the printed number is `setuptools_scm`'s description of the **git tags of the
tree it was built in**, and `vllm_ascend/_version.py` is generated and gitignored
(`.gitignore:198`). Two consequences belong in the opening paragraph: the number
does not prove which code is imported, and no code in `vllm_ascend/` reads it —
`rg -n "__version__" vllm_ascend/` at that ref has exactly one hit,
`utils.py:283-295` `vllm_version_is()`, which loads `envs.VLLM_VERSION` or
`vllm.__version__` (vLLM's, never vllm-ascend's). The compatibility gates are
keyed on vLLM's version.

## One call decides the cause

```bash
python -c "import vllm_ascend, vllm, sys; print(vllm_ascend.__file__); print(vllm.__version__); print(sys.executable)"
pip show -f vllm-ascend | head -20        # Location + Editable project location
pip list 2>/dev/null | grep -i vllm
cd <checkout> && git describe --tags --always --dirty; git rev-parse HEAD
```

`Location` differing from the printed `__file__`, a second `vllm_ascend-*.dist-info`,
or an `__editable__*vllm_ascend*.pth` ⇒ stale metadata is the answer. `git describe`
printing the reported number ⇒ the checkout really sits there. Everything else is
arithmetic on these two outputs.

## Where a bare version number can come from

- setuptools_scm describes the nearest **reachable tag**: exactly on it →
  `0.9.0rc2`; N commits past it → `0.9.0rc3.devN+g<sha>`. A bare `0.9.0rc2` with no
  `.devN+g…` suffix therefore means the built tree sat *exactly* on that tag, or the
  number is a leftover dist-info. Nothing else produces that string.
- A **shallow** clone of a branch carries no tags: `git clone --depth 1 --branch
  <branch>` ⇒ `LookupError` ⇒ `0.0.0`. So "I cloned the dev branch shallow" cannot
  explain a real-looking number; that row is closed by the derivation, not by a test.
- The documented source build clones a *tag*: `docs/source/installation.md:150-170`
  (`--depth 1 --branch |vllm_ascend_version|`, then `pip install -v -e .`), and
  `docs/source/conf.py:67-76` fills those substitutions with `vllm_version: v0.9.2`,
  `vllm_ascend_version: v0.9.2rc1`, `pip_vllm_ascend_version: "0.9.2rc1"`.
  `Dockerfile:41` clones vllm the same way. The released pip versions are `rc` tags,
  and `vX.Y.Z-dev` is a **branch name** in this project's own prose ("If you are
  building from v0.7.3-dev", `installation.md:173`) — so "vllm-ascend 0.9.1dev" is
  not a version any artifact reports. Correct that expectation explicitly; it is
  half of what the reporter is complaining about.
- A dev build of *vllm* breaks the pairing code rather than the display:
  `vllm_version_is()` raises `ValueError: Invalid vllm version …` whose own message
  is the fix (`Set the environment variable VLLM_VERSION … format of x.y.z`).

## Ranked causes

1. **Stale dist-info — the old distribution is still the installed one.** Images
   ship a preinstalled wheel, and a failed `pip install -v -e .` (custom-op compile
   is the usual failure) or an install into a different interpreter leaves the old
   metadata in place. `pip show` reads dist metadata, not the imported module.
   Decided by the two output paths above. Labelled hypothesis until they are shown.
2. **The checkout really is at/near that tag.** Both recipes clone with `--branch`;
   the reported number is the tag `git describe` reproduces. Source-verified
   mechanism at `b5b7e0e`.
3. **Expectation mismatch.** Released pip versions are `rc` tags (`conf.py:74-76`),
   `-dev` is a branch, and an unpinned `pip install vllm-ascend` installs the newest
   non-prerelease — a different artifact from the branch they meant to build.

## Recorded run: #1784 (vllm-ascend v0.9.2rc1 `b5b7e0e` + vllm v0.9.2 `a5dd03c`)

"installing vllm-ascend 0.9.1dev, `pip show vllm-ascend` shows 0.9.0rc2". Track:
packaging/metadata, no lane. The run ranked the same three causes, cited
`setup.py:305-311`, `.gitignore:198`, `utils.py:283-295`, `conf.py:67-76` and
`installation.md:150-170`, and delivered on its last turn (13 tool calls, ~12 K
completion tokens) — the derivation above is that session's cost, reusable as-is.

Two gaps to state rather than paper over: the case snapshot is a **one-commit
export with no tags** (`git tag` empty, `git describe` → `No names found, cannot
describe anything`), so the tag-ancestry half cannot be reproduced locally and must
be handed back to the reporter as their own `git describe`; and
`scripts/history-titles.sh 'pip show|版本号|setuptools'` has no precedent for a
version-display report, so the answer rests on the build recipe, not on a prior
issue.

Recheck before reuse: read `setup.py`'s version block at whatever ref the report
names. A later tree that hardcodes a version module (`vllm_ascend/version.py`, a
`_build_info.__version__`) supersedes the derivation — `_build_info` already carries
`__soc_version__` / `__sleep_mode_enabled__` at this ref (`utils.py:74-83`), so check
that file before claiming "no version constant anywhere".
