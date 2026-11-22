# "No matching distribution": the requirement exists but no index pip can see serves it

Scope: the report is a **pip resolution error about a requirement**, and nothing has
crashed yet. Four faces of one decision:

- **install time, from the published wheel** — `pip install vllm-ascend==<ver>` ends in
  `ERROR: Could not find a version that satisfies the requirement
  torch-npu==2.5.1.post1.dev20250619` / `No matching distribution found for
  torch-npu==…`, sometimes titled "unmatched torch version between vllm=X and
  vllm-ascend=Y" (#1796);
- **inside a running deployment, adding an optional dependency** — a command from the
  project's own docs dies on `ImportError: Please install vllm[bench] for bench
  support`, and the reporter's next attempt, `pip install pandas`, ends in
  `No matching distribution found for pandas` in the image (#4737);
- **no pip output pasted at all** — the body narrates the substitution instead: installing
  `vllm==<ver>` "prompts you to uninstall torch 2.5.1 and reinstall version 2.7.X", after
  which the vllm-ascend install "reports that torch 2.5.1 could not be found" (#1363) —
  same defect, paraphrase instead of a message: §A0.
- **`ERROR: Cannot install <a> and <b> … conflicting dependencies` from the
  build-dependency subprocess**, ending `× pip subprocess to install build dependencies
  did not run successfully` under `/tmp/pip-build-env-*/` — same words as §C, different
  owner: the conflicting pins are `pyproject.toml`'s `[build-system] requires`, resolved
  into an empty prefix, so the reporter's installed torch pair cannot satisfy them and the
  `+cpu` index ladder of §C is spent: **§D**.

**Navigate by header, not by prefix.** The sections are ordered by paste shape, so §C/§D
sit past line 250 while §A0 — the *other* no-message shape — is inside the first 120 lines:
a `sed -n '1,140p'` read returns §A0 and looks complete, and has now cost two recorded runs
(#446, #1411). One `rg -n '^## '` on this page, then the matching §, is the whole navigation.
**And the paste is not the only trigger.** With **no pip output at all**, read the pin's own
`[build-system] requires` first: when it puts `torch-npu==X` beside `torch>=X` (v0.8.5rc1
`pyproject.toml:14-15`, mirrored at `requirements.txt:12-13`), §D's conflict is the failure
that release produces on any x86 host whose index does not serve the `+cpu` label — so the
first message carries §D's rung 1 and treats "x86 cannot install" as the reporter's premise,
not a finding (#1411, reporter-confirmed).

The sibling shape — the same two-command install ending in pip's *warning* block
rather than an error — is `pip-resolver-warning-block.md`; it owns the "nothing
failed" verdict and the device-gated metadata that decides it.

All three are answered from **three files and one command**, never from a release survey:
the downstream `requirements.txt` (and the `setup.py` that turns it into metadata), the
upstream `requirements/*.txt` for the same three packages, and the reporter's own
`pip config list` / `pip show torch torch-npu`. Inherit `index.md` and the rules in
`SKILL.md`; this page is the install-time counterpart of
`offline-source-build.md` (which owns the *source build* network legs).

## Rule 0 — decide visibility before compatibility

A pip error names the requirement it could not satisfy, and that line covers three
different faults. Tell them apart **before** any version reasoning:

| What the line means | The decisive read |
|---|---|
| the artefact is on an index pip was never given (mirror-only build, extra index that lived in `requirements.txt` options) | does the wheel's `Requires-Dist` pin point at a `.dev`/`.post`/`rc` build, and are the `--extra-index-url`/`--pre` lines still in the **tree** but absent from the **metadata**? |
| the pin set is genuinely unsatisfiable (two torch lines in one graph) | the `torch` / `torch-npu` / `torchvision` pins on both sides — `scripts/pip-pairing-check.sh` prints them, the contradictions, and the release's own matrix row. When no pip error is quoted and the ask is a pin bump, that matrix row *is* the answer: `dependency-pin-bump-request.md`. **Not** every "conflicting dependencies" error is this row: a conflict that only names a `+cpu` local label is section C, and the two specs are compatible by PEP 440 |
| pip cannot reach any index at all (air-gapped pod, image pip.conf pointing at a public mirror) | `pip config list` in that container, plus whether the same install ever reached an index |

Only the middle row is a packaging defect; the first row is a missing index on the
command line; the third is a network fact about the reporter's site and is fixed with a
wheelhouse, not with a version.

## A0. No pip output is pasted — the reporter narrates the torch substitution

Recorded: **#1363**, `[Installation]: How to deploy vllm-ascend in AutoDL's 910B instance`.
A rented aarch64 box whose pasted `npu-smi` (one 910B2 chip, 3335 MB resident HBM, no
process) and `ascend_toolkit_install.info` (CANN `8.1.RC1`) show the CANN and torch side
already installed, and whose whole "error" section is one sentence pair: installing
`vllm 0.9.1` "prompts you to uninstall torch 2.5.1 and reinstall version 2.7.X", after
which the `vllm-ascend 0.9.1rc1` install "reports that torch 2.5.1 could not be found";
`collect_env.py not Found`. No `ERROR:` line, no traceback — so §A's chain cannot be
triggered by a grep, and the report reads like a recipe question ("how will such a
problem be handled?").

It is the §A conflict seen from the reporter's side, answered by **four reads and no
source walk** at vllm-ascend `v0.9.1rc1` (`c30ddb8`) + vllm `v0.9.1` (`b6553be`):

| Read | What it settles |
|---|---|
| `docs/source/developer_guide/versioning_policy.md:25` | the release's own pairing row — `v0.9.1rc1 \| v0.9.1 \| … \| 8.1.RC1 \| 2.5.1 / 2.5.1.post1.dev20250528`: the Ascend line is torch **2.5.1** |
| downstream `requirements.txt:13-14,25-27` with `setup.py:306-316,359` | `torch>=2.5.1`, `torchvision<0.21.0`, and mirror-only `torch-npu==2.5.1.post1.dev20250528` after `--pre` / `--extra-index-url`; `--` lines are dropped from `install_requires`, and there is **no `vllm` requirement at all**, so nothing checks which vllm this sits beside |
| upstream `requirements/cuda.txt:9-12` + `setup.py:436,611-612` | the PyPI `vllm` wheel is a CUDA build whose metadata pins `torch==2.7.0` / `torchvision==0.22.0`; `_no_device()` reads `common.txt` only, which has **no torch** — so `VLLM_TARGET_DEVICE=empty` is the one vllm install that leaves the torch line alone |
| `Dockerfile:18,40-44,49-53` | the project's own recipe: CANN base image (`:18`), `ARG VLLM_TAG=v0.9.1` (`:40`), `VLLM_TARGET_DEVICE="empty" … pip install -e vllm` (`:43`), `pip uninstall -y triton` (`:44`), then vllm-ascend after `export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi` (`:49,53`) |

Verdict for the first paragraph: **do not take `vllm` from PyPI on that release** — no pip
command gives both wheels one torch line; install vllm v0.9.1 from source with
`VLLM_TARGET_DEVICE=empty` as the Dockerfile does, keep torch 2.5.1, pull
`torch-npu==2.5.1.post1.dev20250528` from the Huawei mirror, then vllm-ascend. The
reporter's second error is that mirror, not a second defect — the paraphrase does not say
whether pip named `torch`, `torchvision` or `torch-npu`, so ask for the exact line
(`pip install --dry-run` / `pip show` on their box) instead of guessing which of the three
it was.

The doc defect this exposes at the same ref: the **pip leg** of `installation.md`
(`:126-141`) renders `pip install vllm==0.9.1` / `vllm-ascend==0.9.1rc1` from
`conf.py:75-76` and carries **no Ascend mirror** — its only extra-index is
`download.pytorch.org/whl/cpu` for x86 torch (`:126-129`) — while the **build-from-source**
leg is the one with `VLLM_TARGET_DEVICE=empty` (`:153`) and
`export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi` (`:159`).
A reader who follows the pip leg therefore cannot satisfy the tree's own
`torch-npu==…dev20250528`. Quote both line ranges when filing the `[Doc]` item and hand
the reporter the source leg.

Ladder (on a box where CANN and torch are preinstalled — reinstall neither):

1. `pip show torch torch-npu torchvision vllm` plus
   `python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"`
   — which torch line the aborted install left on the box.
2. `export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi`, then
   `pip install "torch-npu==<the row's dev pin>"` (and `torch==` its line if pip moved it
   to 2.7).
3. vllm from source at the paired tag: `VLLM_TARGET_DEVICE=empty pip install -v -e <vllm>`,
   then `pip uninstall -y triton`.
4. `pip install -v -e . --no-build-isolation` in vllm-ascend (`installation.md:169` names
   `--no-build-isolation` as the escape for a torch-npu conflict in the build); the
   prebuilt image `quay.io/ascend/vllm-ascend:v0.9.1rc1` is the alternative.
5. `python -c "import vllm, vllm_ascend, torch, torch_npu; print(...)"`, then one small
   serve. `pip check` still lists the torchvision pair on a source vllm — that residual is
   the design (`pip-resolver-warning-block.md` §D), not a second fault.

Do not read the one-chip `npu-smi` or the 3335 MB resident HBM as the symptom, and do not
reinstall CANN: `ascend_toolkit_install.info` already matches the row's `8.1.RC1`.

## A. The wheel pin lives on a mirror, the metadata cannot say so

Recorded: #1796, vllm-ascend `v0.9.2rc1` (`b5b7e0e`) + vllm `v0.9.2` (`a5dd03c`),
`pip install vllm==0.9.2` then `pip install vllm-ascend==0.9.2rc1` as printed in
`docs/source/installation.md:144-147`.

Source-verified chain at that ref:

1. `requirements.txt:25-27` carries `--pre`, `--extra-index-url
   https://mirrors.huaweicloud.com/ascend/repos/pypi` and
   `torch-npu==2.5.1.post1.dev20250619`.
2. `setup.py:331-351` `get_requirements()` → `_read_requirements()` **skips every
   line that starts with `--`** (`:341`) and returns the rest; `setup.py:388`
   `install_requires=get_requirements()`. So the published wheel's metadata keeps
   `torch-npu==2.5.1.post1.dev20250619` and drops the two option lines that would
   have made it findable.
3. pip still accepts that prerelease, because an exact `==` pin with a pre-release
   component allows prereleases for that requirement (`packaging`:
   `Specifier("==2.5.1.post1.dev20250619").prereleases` is `True`) — so **the missing
   `--pre` is not the cause and the missing index is**. Do not answer "add `--pre`".
4. The same requirement set is not satisfiable even with the mirror visible:
   `requirements.txt:13-14` is `torch>=2.5.1` + `torchvision<0.21.0`, while vllm
   `v0.9.2` pins `torch==2.7.0` / `torchvision==0.22.0`
   (`requirements/cuda.txt:9,12`, `pyproject.toml:9`, `requirements/build.txt:7`).
   A `torch_npu` wheel hard-pins the exactly matching torch: `Ascend/pytorch`
   `setup.py:625` is `requirements = ['torch==2.7.1+cpu' if platform.machine() ==
   'x86_64' else 'torch==2.7.1']` with `install_requires=requirements` (`:650`), so
   the `2.5.1` line pins torch 2.5.1 and `torchvision<0.21.0` is that same 2.5.x line
   (0.22.0 is excluded by the cap). The reporter's title is literally right.

The reporter's second message is the same defect from the other side: installing
`torch-npu==2.5.1` by hand uninstalls torch 2.7.1 and pip then only *warns* —
`vllm 0.9.2 requires torch==2.7.0 … but you have torch 2.5.1 which is incompatible`
(plus torchvision/torchaudio lines). pip does not hold the installed vllm's metadata
when it resolves a later command, so a downgraded torch is a warning, not an error.
That warning block is the cheapest proof of the torch line the reporter landed on.

Later trees show the invariant that was missing: at `v0.11.0rc2` the same file is
`torch==2.7.1` (`:13`) + `torch-npu==2.7.1` (`:28`, mirror lines commented out) +
`torchvision` uncapped (`:14`) — the `torch-npu` version prefix **equals** the torch
version prefix. Check that equality first; it is the whole pairing in one line.

### Answer shape (A)

State it in the first paragraph: the wheel's pin is a dev build that only the Ascend
mirror serves, the option lines that point at that mirror are dropped from the wheel
metadata, and the requirement set sits on two torch lines at once — so **no pip
command makes `vllm-ascend==0.9.2rc1` + `vllm==0.9.2` consistent**; the docs' pip
block is broken at that RC (the single unverified link is which torch that dev wheel
pins — see *Not verified*; say so in the same paragraph). Then the ladder:

1. Prove which line you are on, one command: `pip show torch torch-npu torchvision
   vllm`, and `python -c "import torch, torch_npu; print(torch.__version__,
   torch_npu.__version__)"`.
2. Reproduce that the index, not the version, is the blocker:
   `pip install --dry-run --no-deps --extra-index-url
   https://mirrors.huaweicloud.com/ascend/repos/pypi 'torch-npu==2.5.1.post1.dev20250619'`.
   A resolution here plus a conflict with torch 2.7.0 names cause 2 exactly.
3. Prefer the prebuilt image (`quay.io/ascend/vllm-ascend:v0.9.2rc1`) or a release
   whose pins agree, and verify it with step 1 — do not hand-assemble the pair from
   `--no-deps` installs unless the reporter asks for that.
4. If they must stay on that RC: install `vllm-ascend` with `--no-deps` and give
   torch-npu/torch/torchvision explicitly from the line they choose, then `pip check`
   to show the residual inconsistency. Label this a workaround, not supported.
5. `pip config set global.extra-index-url "<url> <url>"` is **not** the bug: pip
   splits the value (`ConfigOptionParser._update_defaults`, `action == "append"` →
   `val.split()`), so both indexes are used. The reported "keeps looking forever" is
   resolution cost across every configured index; the fix is a per-command
   `--index-url`/`--extra-index-url` instead of a global multi-index.

## B. A missing optional extra inside an air-gapped image

Recorded: #4737, image `vllm-ascend:0.11.0rc2`/`0.11.0rc3`, `vllm bench serve …`
dies on `ImportError: Please install vllm[bench] for bench support`; the reporter's
later public comment adds the follow-up attempt: `pip install pandas` + `datasets`
in the image → `No matching distribution found for pandas`.

Source-verified chain at `v0.11.0rc2` (`a2e4c3f`) + vllm `v0.11.0` (`b8b302c`):

1. `vllm/setup.py:658` defines `"bench": ["pandas", "datasets"]`; the image installs
   vllm with the **audio** extra only — `vllm-ascend/Dockerfile:43`
   (`pip install -v -e /vllm-workspace/vllm/[audio]`), the same line in
   `Dockerfile.a3:43`, `Dockerfile.310p:43`, `Dockerfile.openEuler:41`,
   `Dockerfile.a3.openEuler:41`, `Dockerfile.310p.openEuler:41`. So both packages are
   absent **by construction**, not by accident.
2. The message is upstream's rewrite: `vllm/benchmarks/datasets.py:44-52` soft-imports
   `datasets` and `pandas` into `PlaceholderModule`, and
   `vllm/utils/__init__.py:2582-2587` converts the real
   `ModuleNotFoundError` into `Please install vllm[bench] for …`, which is why the
   pasted text names no module. Read the frame that touches the placeholder
   (`pd.read_json` at `datasets.py:1612` for `--dataset-name custom`,
   `:1708` spec-bench, `:1838` BurstGPT `pd.read_csv`, `:1919`
   `load_dataset` for the HF family) to name the actual missing module.
3. The project's own recipe already says these are hand-installed:
   `benchmarks/requirements-bench.txt` (`pandas`, `datasets`, `modelscope`,
   `tabulate`), cited by `benchmarks/README.md:38` and
   `docs/source/developer_guide/performance/performance_benchmark.md:37`. So the
   answer is "install the extra", not "the image is broken" — and the reporter's own
   opinion that shipping the bench toolkit in the image is debatable is a valid
   follow-up, not the diagnosis.
4. `No matching distribution found for pandas` is **not** a pandas version problem:
   the image's pip carries a persistent index from build time —
   `Dockerfile:20` `ARG PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"`
   and `Dockerfile:36` `RUN pip config set global.index-url ${PIP_INDEX_URL}` — so in
   an air-gapped pod every requirement is looked up on an unreachable host. Two lines
   decide it: `pip config list` (or `pip config debug`) and whether a wheel is already
   on disk (`ls /root/.cache/pip`).

### Answer shape (B)

First paragraph: the message is upstream's placeholder text, the two real modules are
`pandas` and `datasets`, the image installs `vllm[audio]` only, and in an offline
container `pip install pandas` can only end in `No matching distribution found` — the
fix is a wheel carried in, not a different pandas. Ladder:

1. Say which dataset name needs what: `--dataset-name custom` (and `spec_bench`) needs
   **pandas** only; `random`, `sharegpt`, `sonnet`, `prefix_repetition` touch neither;
   the `hf`/`conversation`/`vision_arena`/`instruct_coder`/… family needs
   **datasets**. So a run that only needs custom data is one wheel away, and a
   no-dependency smoke test is `--dataset-name random`.
2. Build the wheelhouse on a networked host with the **image's** interpreter and arch:
   `pip download pandas datasets -d /tmp/wh --only-binary=:all:
   --platform manylinux2014_aarch64 --python-version 3.11 --abi cp311` (v0.11.0rc2's
   base is `quay.io/ascend/cann:8.3.rc2-910b-ubuntu22.04-py3.11`, `Dockerfile:18`), then
   inside the pod `pip install --no-index --find-links=/tmp/wh pandas`. Add
   `--no-deps` only if the transitive closure is already present; without
   `--only-binary=:all:` pip may pick an sdist that cannot build offline.
3. Keep the tree's own constraint: `requirements.txt:5` is `numpy<2.0.0` and
   `:16` `opencv-python-headless<=4.11.0.86 # Required to avoid numpy version conflict
   with vllm`. Pulling a pandas that upgrades numpy breaks the engine — check
   `python -c "import numpy; print(numpy.__version__)"` after the install.
4. A client-side escape hatch that needs no install: `vllm bench serve` is a client,
   `--base-url http://<npu-host>:8000` from any machine that has pandas, with
   `--tokenizer /local/tokenizer` when that machine is also offline.
5. Only then the upstream-facing point: the image could either ship `vllm[bench]` or
   document the offline wheel path; that is a packaging decision for the maintainers.

## C. `Cannot install <a> and <b> because these package versions have conflicting dependencies` — the `+cpu` local label

Recorded: **#7089**, `pip install vllm-ascend==0.15.0rc1 torch-npu==2.9.0` →
`ERROR: Cannot install vllm-ascend and vllm-ascend==0.15.0rc1 because these package
versions have conflicting dependencies.` / `The user requested torch==2.9.0` /
`torch-npu 2.9.0 depends on torch==2.9.0+cpu`. Precedent one torch line earlier:
**#4621** (`v0.11.0rc1`) with `torch-npu==2.7.1` / `torch==2.7.1` and
`ResolutionImpossible`.

pip's own advice ("loosen the range of package versions") points at the compatibility
row; PEP 440 says this is the **visibility** row. Two rules, one probe:

- a specifier with **no** local label ignores a candidate's local label, so `==2.9.0`
  *accepts* `2.9.0+cpu`;
- a specifier **with** a local label requires it, so `==2.9.0+cpu` *rejects* plain
  `2.9.0`.

So the two requirements in #7089 are satisfiable together and no version needs changing:
pip failed because no index it was given served a `+cpu`-labelled torch. That label is
built per machine upstream — `Ascend/pytorch` `setup.py:626` is
`['torch==2.8.0+cpu' if platform.machine() == 'x86_64' else 'torch==2.8.0']` (`:625` on
the 2.7.1 line) — and only `download.pytorch.org/whl/cpu` serves it. At `v0.15.0rc1`
(`3d43ed9`) the tree pins `torch==2.9.0` (`requirements.txt:27`) beside
`torch-npu==2.9.0` (`:42`), and the documented way to make it visible is
`pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"`
(`docs/source/installation.md:135-139`) marked **[Optional]** and scoped to x86 — so a
non-x86 reader skips it and hits this error. The reverse direction is why a hand-installed
plain `torch` breaks `torch_npu` afterwards, and why the working environments' own dumps
print `torch==2.9.0+cpu` (#6509, #6842).

Probe it instead of asserting it:
`uv run --no-project python <skill_root>/scripts/pep440-local-check.py '==2.9.0' '==2.9.0+cpu'`.
`import packaging` fails in the no-project environment and `uv run --no-project --with
packaging` needs the network, which is closed; the script falls back to
`pip._vendor.packaging`, which is installed.

## D. The same words from the build-dependency subprocess — `[build-system] requires`

Recorded: **#446**, `[Installation]: why v0.7.3rc2 install failed, but v0.7.3rc1 install
successful ?` — `pip install vllm-ascend==0.7.3rc2 -i <internal mirror>`, vllm-ascend
`v0.7.3rc2` (`00459ae`). Two screenshots, no text; the failing one ends

```
INFO: pip is looking at multiple versions of torch-npu to determine which version is
compatible with other requirements. This could take a while.
ERROR: Cannot install torch-npu==2.5.1rc1 and torch>=2.5.1 because these package versions
have conflicting dependencies.
    The user requested torch>=2.5.1
    torch-npu 2.5.1rc1 depends on torch==2.5.1+cpu
× pip subprocess to install build dependencies did not run successfully.
```

The tell is the last line plus the `/tmp/pip-build-env-*/` paths above it. Three reads
settle it at this ref:

| Read | What it settles |
|---|---|
| `pyproject.toml:3-13` | `requires = [… "torch_npu >= 2.5.1rc1" (:11), "torch >= 2.5.1" (:12)]` — the torch pair lives **only** here |
| `requirements.txt` (6 lines, no torch) + `setup.py:308,318,361` | `install_requires` carries **no** torch line, so a **wheel** install resolves nothing torch-shaped; this fails only on the **sdist** path. Both rcs arrived as `vllm-ascend-0.7.3rcN.tar.gz` on that mirror, which is why the box ever ran a build-deps step |
| the two rcs' own build-deps logs | rc1's list is `Requirement already satisfied: decorator / pyyaml / scipy / setuptools`, `Collecting setuptools-scm`, then `numpy` / `packaging` from those — **no torch line**, so rc1's build requires never named torch. That is the answer to "why did rc1 install": the release moved the torch pair into the build requirements, not the runtime ones. Tags cannot be diffed here (`one_commit_git`), so the reporter's two pastes *are* the control |

**The same words recur with a released pin, and there the pair also rides into the wheel's
metadata.** #1411 (`pip install vllm_ascend==0.8.5rc1` on a 910B2 x86 host, internal
`-i http://mirrors.aliyun.com/pypi/simple/ --trusted-host …`) is reporter-confirmed
(`issuecomment-3059849201`): `ERROR: Cannot install torch-npu==2.5.1 and torch>=2.5.1 …
torch-npu 2.5.1 depends on torch==2.5.1+cpu`, again under `× pip subprocess to install build
dependencies did not run successfully`. At that pin `pyproject.toml:14-16` and
`requirements.txt:12-14` are identical and `setup.py:359` is
`install_requires=get_requirements()`, so the torch pair is in **both** legs — unlike
v0.7.3rc2 above, where the wheel's metadata never named torch. Two consequences: `-i
<mirror>` alone (no `--extra-index-url …/whl/cpu`) cannot resolve the pair from either leg,
and the reporter's `pip list` showing `torch 2.5.1+cpu` beside `torch-npu
2.5.1.post1.dev20250626` is again the installed pair saying nothing — the build env forces
released `2.5.1`, while the dev wheel they wanted is *newer* than the exact pin but still
satisfies the docs' `torch-npu | >= 2.5.1` table (`installation.md:15`), i.e. the same
doc-versus-metadata drift, one pin later. Before ranking anything, run
`bash <skill_root>/scripts/pip-pairing-check.sh <ascend-root> [vllm-root]`: its first
finding restates the `torch-npu==X` / `torch>=X` pair, names the `+cpu` label as the only
joint solution, prints the tree's own x86 extra-index line, and predicts this error text —
which is also the check that rules out "no wheel was published for my arch": an
`installation.md` that tells x86 users to add `pip config set global.extra-index-url
https://download.pytorch.org/whl/cpu/` (`:126,129` at v0.8.5rc1) is the project saying the
x86 pip install is a supported path, and that index exists precisely because the `+cpu`
label is otherwise invisible.

**Build requirements are resolved into an empty prefix, so the reporter's `pip list` can
never satisfy them.** pip installs them with `--no-user --no-warn-script-location
--disable-pip-version-check --no-compile --prefix <tmp>` (plus `--ignore-installed` when the
prefix is not a venv) and forwards the finder flags — `-i`, then one `--extra-index-url`
per extra index, `--find-links`, `--trusted-host`, `--cert` (`pip/_internal/build_env/installer.py:104-131`;
read it with `python -c "import pip,os;print(os.path.dirname(pip.__file__))"`, which works
where `import packaging` does not; verified at pip 26.2.1 — the reporter's version is not in
the paste, so re-check on theirs). Two consequences: an env dump showing `torch 2.5.1+cpu`
**and** `torch-npu 2.5.1rc1` installed says nothing about this error (#446's later comments
are exactly that pair — "torch 2.5.1 and torch 2.5.1+cpu both don't work" — so the version
axis is spent), and `--extra-index` is an accepted abbreviation of `--extra-index-url`
(pip long-option prefix matching; the tree's own `installation.md:131` and `pta_install.sh:12`
write it that way), so never blame the flag's spelling.

**Then read the build floor against the release's own torch_npu, because that is what forces
the `+cpu` dependency.** The same tag's table says `torch-npu | >= 2.5.1.dev20250320`
(`docs/source/installation.md:15`) and its release note bumps to `dev20250320.3`
(`user_guide/release_notes.md:15`, PR #406), while the build floor is `torch_npu >=
2.5.1rc1` — and a dev build is *below* rc1: `2.5.1.dev20250320` fails `>=2.5.1rc1`
(`scripts/pep440-local-check.py '>=2.5.1rc1' '>=2.5.1' --candidates
2.5.1.dev20250320,2.5.1rc1,2.5.1,2.5.1+cpu`). The build env therefore cannot use the wheel
the release documents; it is pushed onto released `2.5.1rc1`, whose x86_64 wheel pins
`torch==2.5.1+cpu` — a local label only `download.pytorch.org/whl/cpu` serves (§C; the tree
passes that index for the same wheel at `pta_install.sh:12`). The doc-versus-metadata drift
is the defect; the internal mirror is only where it surfaced.

Ladder:

1. `pip install vllm-ascend==<ver> --no-build-isolation` — the host already carries the pair,
   and the tree names this escape in its own install note (`installation.md:168` at v0.8.5rc1:
   "If you're using `pip install -e .` and encourage a torch-npu version conflict, please
   install with `pip install --no-build-isolation -e .`"), and at v0.7.3rc2 the build needs
   nothing from torch when the custom ops are off
   (`setup.py:289-291`: `ext_modules` is appended only `if envs.COMPILE_CUSTOM_KERNELS is not
   None`, and the release note says compilation is disabled by default). One line, and it
   also answers the literal "why rc1 and not rc2".
2. Keep isolation and *prove* the index reached the subprocess: `pip install -v` prints the
   build-deps command, which must carry `-i <mirror> --extra-index-url
   https://download.pytorch.org/whl/cpu/`; `PIP_EXTRA_INDEX_URL` is the env form. If the
   flag is there and the conflict persists, ask for that command line plus the candidate
   versions pip lists, instead of adding a third torch pin.
3. A release whose build requires do not name torch (rc1 here), a **wheel** rather than the
   sdist (`--only-binary vllm-ascend`), or the release's image.

Open: the reporter's `pip --version` (arg forwarding is version-dependent), whether their
internal mirror proxies `download.pytorch.org` (a different host, so the CPU index has to be
named explicitly, which the first command did not do), and the full `pip install -v`
build-deps command.

## Common rules

1. **One command per claim, and the claim is the verdict.** These reports are two
   `grep`s and one `pip` command deep; a session that spends the output budget on the
   mirror's contents, on release archaeology, or on which torch `2.5.1.post1` was
   built against delivers nothing. The unanswerable half (which exact torch a dev
   wheel pins) goes in the answer as the open question with the one command that
   shows it (`pip show`/`pip install --dry-run`), labelled hypothesis.
2. **Never assert a package is missing from an index you cannot see.** Say which index
   pip was given, which line in the tree supplies it, and that wheel metadata cannot.
   The mirror's file listing was never reachable from a recorded session; keep it that
   way rather than guessing that a dev build was "deleted". The reporter's own
   `(from versions: …)` list **is** the set of indexes pip was given, so a `.dev` build
   absent from it is §A0's visibility evidence, not a removal.
3. **Options are not requirements.** `--pre`, `--extra-index-url`, `--find-links`,
   `-r` in `requirements.txt` are pip *options*: the tree honours them, a wheel's
   metadata cannot. Same rule for a `pip config` value: it applies to the whole
   process, not to a command the reporter pasted from the docs.
4. **A deprecation notice is not a version contract.** `README.md:41` at
   `v0.9.2rc1` ("PyTorch >= 2.5.1, torch-npu >= 2.5.1.post1.dev20250619") and
   `installation.md`'s requirements table are prose; the executable contract is
   `requirements.txt` + `setup.py` + the upstream `requirements/*.txt`. The same
   answer settles the report that **quotes two files and asks which to obey** —
   `[Installation][v0.9.1]` "The versions for torch_npu in the README and
   pyproject.toml are different … is it necessary to forcibly pin 2.5.1.post1?",
   no error and no log (#2838): the `==` is the contract and the prose `>=` is the
   defect, so do not relax the pin and do not rank a version mismatch.
   (a) The two exact files name each other as the mirror pair (`pyproject.toml:2`
   "# Should be mirrored in requirements.txt", `requirements.txt:1` "# Should be
   mirrored in pyporject.toml", typo in tree) and the README is **not** in that
   pair, so its `>=` licenses no other wheel. (b) The pin is enforced twice:
   `setup.py:331-341` reads `requirements.txt` (skipping `--` lines) into
   `install_requires` (`:388`), and the same set is in `pyproject.toml`
   `[build-system] requires` (`:3-19`), so build isolation and every wheel
   consumer get `torch-npu==`; `==X` matches only that build, while `>=X` would
   admit a `.post2`/next minor that cannot coexist with the pinned `torch`. (a) and
   (b) were verified at v0.10.1rc1 (`7e16b4a7cdb1`), where the identical split
   exists; the reported v0.9.1 tree was not in that snapshot, so quote the pin you
   verified and hand the reporter the one line that decides **their** install —
   `python -c "import importlib.metadata as m; print([r for r in m.requires('vllm-ascend') if 'torch' in r])"`
   (`torch-npu==…` = the README's `>=` is cosmetic there too). The sanctioned way to
   admit a newer torch_npu is the next release line, never a hand-edited pin.
5. **`pip check` is the closing evidence.** Whatever ladder the reporter takes, the
   end state is `pip check` clean plus `python -c "import torch, torch_npu, vllm,
   vllm_ascend"`; a warning-only environment "works" until the first NPU op.

## Recorded runs and their cost

- **#1363** (v0.9.1rc1 + vllm v0.9.1, AutoDL aarch64, §A0): 16 assistant turns, 20 tool
  calls and **15 721 of the 16 384 completion tokens**, the whole answer typed on the last
  turn — the run never opened this topic and re-derived the same conclusion from
  `conf.py`, `release_notes.md` and the docs' FAQ. The four reads of §A0 were in hand by its
  sixth message, and message text carried no verdict until the end; the paraphrase in the
  body is the trigger, so route it here on the first pass.
- **#1901** (`v0.9.2rc1` + vllm `v0.9.2`; 「源码安装0.9.2rc1，torch npu版本问题」): the
  same defect with the `ERROR: Could not find a version …` text pasted and with
  `VLLM_TARGET_DEVICE=empty` already in the reporter's vllm step. This page's §A chain plus
  `setup.py`'s `--`-line skip answered it in 7 assistant turns / 6 005 tokens — the shape a
  session reaches §A0's evidence with when it opens the topic early.
- **#1796** (`v0.9.2rc1` + `v0.9.2`): 39 tool calls, ~16 K completion tokens of
  reasoning, **every assistant message a bare tool call, cut with no answer** — the
  delivery failure, on a report whose answer is `requirements.txt:25-27` +
  `setup.py:341` + the two upstream pins. The session's own reasoning reached the
  right facts (the metadata pin, the `torchvision<0.21.0` contradiction) and then
  spent the budget on whether `torch-npu 2.5.1.post1` was rebased onto torch 2.7.1,
  which no local file can settle.
- **#2838** (snapshot v0.10.1rc1 `7e16b4a7cdb1` + vllm v0.10.1.1 `1da94e67`; the
  report's own `v0.9.1` is not in that tree): the "README says `>=`, pyproject says
  `==`" question, answered in 13 tool calls with the mirror-comment pair, the
  `install_requires` chain and the metadata one-liner — and it flagged that v0.9.1
  is absent instead of quoting it. The cost was scope, not delivery: it then
  re-derived upstream vLLM's torch pin to explain *why* the pin is exact, which the
  tree in hand cannot settle. Ask the reporter for their metadata line instead.
- **#4737** (`v0.11.0rc2`): answered in 11 tool calls — the `[audio]`-only install,
  the `bench` extra, the placeholder rewrite and the pandas-free dataset names were
  all delivered. The gap the reporter's later comment exposes: the answer offered
  `pip download … ; pip install --no-index --find-links=wheelhouse pandas` as ladder
  item 1 without first naming the **index hypothesis**, and the reporter's next
  message is exactly `No matching distribution found for pandas`. Put the
  `pip config list` / unreachable-index check in the opening paragraph when the
  deployment is described as offline.

- **#446** (v0.7.3rc2 `00459ae` + vllm v0.7.3; `[Installation]`, two screenshots, the
  rc1 one a success log and the rc2 one the build-deps conflict): 580 K cumulative prompt
  tokens — most of it the ~100 KB skill body re-sent on each of its 15 later turns — and a
  reasoning allowance spent whole, **every assistant message a bare tool call and no
  message text at all** — the §D facts were in hand by the last turn
  (`pyproject.toml:11-12`, `requirements.txt`, the release-note floor). Two openings were
  missed: the page was read as `sed -n '1,140p'`, so §C/§D never entered context, and the
  fix was re-derived by hand instead of by the two commands above. The remaining turns went
  on PEP 440 semantics (`import packaging` and `uv --with packaging` both fail with the
  network closed — use `scripts/pep440-local-check.py`) and on pip's build-env internals,
  which the *installed* pip answers in one grep and which no local tree can settle for the
  reporter's version. Verdict first, one rung, then stop.

- **#1411** (vllm-ascend v0.8.5rc1 `ec27af3` + vllm v0.8.5.post1 `3015d563`; title
  「x86 systems cannot be installed directly through `pip install vllm_ascend==xxx`」, env block
  literally two lines, no pip output): 547 K prompt tokens and every assistant turn after the
  first spent on the artefact question — `.github/workflows/` has no wheel-release file at that
  pin, so the answer shipped "no x86_64 wheel is published" as cause 1, with the correct §D
  conflict demoted to cause 2 and `--no-build-isolation` buried as ladder step 3. It had the
  decisive pair in context by its 7th message (`pyproject.toml:14-16`, from one `sed -n '1,40p'`)
  and the docs' x86 CPU-index line by its 6th (`installation.md:126`), and it read this page as
  `sed -n '1,120p'` — so §C/§D never entered context, the second time that exact prefix read has
  decided a run. The reporter's next comment is §D's verbatim text. Opening move for the
  no-paste shape: `pip-pairing-check.sh` (one call, prints the predicted error and rung 1), and
  rank "the arch has no artefact" **below** the `+cpu` label whenever the pin pairs an exact
  `torch-npu==` with a `torch>=` floor.

## Not verified

- The huaweicloud Ascend mirror was not reachable from any recorded session, so
  neither run confirms which `torch` a `torch-npu 2.5.1.post1.dev*` wheel pins; the
  claim rests on the sibling `2.7.1` branch's `setup.py:625` and on the reporter's
  own `pip` warning block. Treat "the pin set is unsatisfiable" as source-verified and
  "that dev wheel would have installed" as untested.
- Whether the released `v0.9.2rc1` **image** is internally consistent: its
  `Dockerfile` runs the same editable install, so the build would have accepted the
  torch downgrade with only a warning. Check `pip show torch torch-npu` inside the
  image before recommending it.
- `scripts/pip-pairing-check.sh` compares the trees given on its command line. Read
  the drift it prints as a question, not a verdict: at `v0.11.0rc2` the tree lands on
  torch 2.7.1 while the `v0.11.0` vllm release pins torch 2.8.0, and the image builds
  vllm from that tag with `VLLM_TARGET_DEVICE="empty" -e` — which tree the reporter's
  wheel was actually resolved against is the open fact there.
