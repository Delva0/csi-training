# "No matching distribution": the requirement exists but no index pip can see serves it

Scope: the report is a **pip resolution error about a requirement**, and nothing has
crashed yet. Two faces of one decision:

- **install time, from the published wheel** — `pip install vllm-ascend==<ver>` ends in
  `ERROR: Could not find a version that satisfies the requirement
  torch-npu==2.5.1.post1.dev20250619` / `No matching distribution found for
  torch-npu==…`, sometimes titled "unmatched torch version between vllm=X and
  vllm-ascend=Y" (#1796);
- **inside a running deployment, adding an optional dependency** — a command from the
  project's own docs dies on `ImportError: Please install vllm[bench] for bench
  support`, and the reporter's next attempt, `pip install pandas`, ends in
  `No matching distribution found for pandas` in the image (#4737).

Both are answered from **three files and one command**, never from a release survey:
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
| the pin set is genuinely unsatisfiable (two torch lines in one graph) | the `torch` / `torch-npu` / `torchvision` pins on both sides — `scripts/pip-pairing-check.sh` prints them and the contradictions |
| pip cannot reach any index at all (air-gapped pod, image pip.conf pointing at a public mirror) | `pip config list` in that container, plus whether the same install ever reached an index |

Only the middle row is a packaging defect; the first row is a missing index on the
command line; the third is a network fact about the reporter's site and is fixed with a
wheelhouse, not with a version.

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
   way rather than guessing that a dev build was "deleted".
3. **Options are not requirements.** `--pre`, `--extra-index-url`, `--find-links`,
   `-r` in `requirements.txt` are pip *options*: the tree honours them, a wheel's
   metadata cannot. Same rule for a `pip config` value: it applies to the whole
   process, not to a command the reporter pasted from the docs.
4. **A deprecation notice is not a version contract.** `README.md:41` at
   `v0.9.2rc1` ("PyTorch >= 2.5.1, torch-npu >= 2.5.1.post1.dev20250619") and
   `installation.md`'s requirements table are prose; the executable contract is
   `requirements.txt` + `setup.py` + the upstream `requirements/*.txt`.
5. **`pip check` is the closing evidence.** Whatever ladder the reporter takes, the
   end state is `pip check` clean plus `python -c "import torch, torch_npu, vllm,
   vllm_ascend"`; a warning-only environment "works" until the first NPU op.

## Recorded runs and their cost

- **#1796** (`v0.9.2rc1` + `v0.9.2`): 39 tool calls, ~16 K completion tokens of
  reasoning, **every assistant message a bare tool call, cut with no answer** — the
  delivery failure, on a report whose answer is `requirements.txt:25-27` +
  `setup.py:341` + the two upstream pins. The session's own reasoning reached the
  right facts (the metadata pin, the `torchvision<0.21.0` contradiction) and then
  spent the budget on whether `torch-npu 2.5.1.post1` was rebased onto torch 2.7.1,
  which no local file can settle.
- **#4737** (`v0.11.0rc2`): answered in 11 tool calls — the `[audio]`-only install,
  the `bench` extra, the placeholder rewrite and the pandas-free dataset names were
  all delivered. The gap the reporter's later comment exposes: the answer offered
  `pip download … ; pip install --no-index --find-links=wheelhouse pandas` as ladder
  item 1 without first naming the **index hypothesis**, and the reporter's next
  message is exactly `No matching distribution found for pandas`. Put the
  `pip config list` / unreachable-index check in the opening paragraph when the
  deployment is described as offline.

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
