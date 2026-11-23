# "A 依赖 B，B 依赖 C … 能否把 torch-npu 升级为 X": a pin-bump request, not a missing wheel

Scope: the report is an **install-time dependency complaint whose deliverable is a
verdict on relaxing a pin**, and nothing has failed on a device. Canonical body
(recorded #3266, vllm-ascend `v0.10.2rc1`):

```
pip install vllm==0.10.2          # reporter claims this needs torch 2.8.0
pip install vllm-ascend==0.10.2rc1  # declared: torch-npu==2.7.1.dev20250724 → torch 2.7.1
能否把vllm-ascend依赖的torch-npu升级为2.8.0的版本？
```

Not this page: an actual pip error (`No matching distribution found …`) →
`pip-index-visibility.md`; an install that only *warned* →
`pip-resolver-warning-block.md`; a build that died in a compiler (`… No such file or
directory` under `pip install -e .`) → `custom-op-build-stage.md`.

## Answer shape first

One sentence, in the report's language, before the table: **no — do not raise the
`torch-npu` pin for this release; the pair is fixed by the release and the extension
is compiled against the installed pta.** Then the matrix row that says so, then the
one command that decides whether the reporter's premise about the *other* package is
even true. The bump itself is a release-line / feature-request question (#1390
`[Feature]: Request vllm-ascend to support torch_npu>=2.6` is the recorded precedent
for that shape: the answer there is a version plan, not a pin edit).

## Step 1 — read the rc's own row, and do not grep the install page for versions

The pair is recorded twice, and the first file a version grep reaches is the one that
cannot answer: `docs/source/installation.md` prints `|pip_vllm_version|` /
`|pip_vllm_ascend_version|` **substitutions**, so `rg -n '0\.10\.2|torch-npu==' docs/source/installation.md`
returns *nothing* (recorded in #3266's run) — that empty result is a template, not an
absent contract. The literal numbers are in:

| read | at `v0.10.2rc1` (`048bfd5`) |
|---|---|
| `docs/source/community/versioning_policy.md:23` header, `:25` row | `| v0.10.2rc1 | v0.10.2 | >= 3.9, < 3.12 | 8.2.RC1 | 2.7.1 / 2.7.1.dev20250724 |` |
| `docs/source/conf.py:68,75-76,80` | `vllm_version: v0.10.2`, pip pair `0.10.2rc1`/`0.10.2`, `ci_vllm_version: v0.10.2` |
| `docs/source/installation.md:15-17` (no substitutions here) | `CANN >= 8.2.RC1`, `torch-npu >= 2.7.1.dev20250724`, `torch >= 2.7.1` |
| `requirements.txt:13,25-27` | `torch>=2.7.1`; `--pre`; `--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi`; `torch-npu==2.7.1.dev20250724` |
| `pyproject.toml:15-17` | the same set inside `[build-system] requires` |

One call finds the row: `rg -n '<the ascend version from the report>' docs/source`.
`scripts/pip-pairing-check.sh` prints this block beside the pin arithmetic.

## Step 2 — which line is the constraint

`torch>=2.7.1` is unbounded, so the downstream tree does **not** forbid torch 2.8.0 by
metadata; what forbids it is the exact `torch-npu==<dev build>` pin plus the pta
wheel's own hard pin of the torch it was built against (`Ascend/pytorch`
`setup.py:625`: `requirements = ['torch==2.7.1+cpu' if x86_64 else 'torch==2.7.1']`,
`install_requires` at `:650`). Say that in one line: the `torch-npu` line, not the
`torch` line, is the constraint, and a hand-relaxed pin moves the whole install onto
an ABI the release never built against — `vllm_ascend`'s extension takes its include
root and target from the **installed** pta (`setup.py:203-212` `-DTORCH_NPU_PATH`), so
a bump is a rebuild plus a numerics risk, never a free upgrade. Give the correct
pairing command instead of a pin edit:

```bash
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
pip install vllm==0.10.2 && pip install vllm-ascend==0.10.2rc1 && pip check
```

If they insist on `--no-deps` to keep their own torch-npu, keep it labelled a
workaround and require the `pip check` residue in the same breath.

## Step 3 — the reporter's premise about the paired package is the open half

The claim "vllm 0.10.2 依赖 torch 2.8.0" contradicts the matrix, and the snapshot may
not carry that vllm ref at all (#3266's capsule exported vllm **v0.9.1**,
`version_role: dated_reference_only`, so its `requirements/cuda.txt` cannot decide
0.10.2). Do not fill the gap from memory and do not call the reporter wrong — hand
back one line that decides it:

```bash
pip show torch torch-npu torchvision vllm vllm-ascend
python -c "import importlib.metadata as m; print([r for r in m.requires('vllm') if r.split()[0].split('[')[0].lower().startswith('torch')])"
```

`torch==2.7.1` there refutes the premise and the rung-1 reinstall closes the report; a
real `torch==2.8.0` line means the wheel is not the one the matrix was validated
against (mirror repack, self-built vllm, or residue from another release) and the same
reinstall-plus-`pip show` answer still stands. Label the local half **unverified**:
the compatibility row is downstream evidence, the upstream pin is the reporter's.

## Delivery

Two reads (matrix row + `requirements.txt`) and one grep answer this shape; a tenth
turn of source reading buys nothing, because the owner is a release pairing, not code.
Do not ladder `torch-npu` patch releases into the answer unless the paste's own pta
line shows one (`.post1`/`.post2` fixes belong to
`../startup-failures/environment-correspondence.md` step 3, #6299) — a 2.8.0 bump
conversation is version-space, not a patch-release defect.
