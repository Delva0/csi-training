# Install-track classification: a report that fails before any engine exists

Scope: the routing decision inside the **install track** — an `[Installation]` / 安装
report that fails before any engine exists — including the shapes that carry **no failing
frame at all**. Inherit the rules in `index.md` and SKILL.md; this page says which stage
owns the failure and which rung answers it. It was SKILL.md's *Classify the evidence*
install paragraph, kept whole so the no-frame variants stay one read from the track
decision.

An `[Installation]` / 安装 report that fails before any engine exists — a compiler or
linker log, a pip resolution error, a missing library at import — is its own track: the
paste's last frames name the failing *stage* and the stage names the owner (vllm-ascend's
`CMakeLists.txt`/`setup.py` vs the CANN toolkit vs the image), the reporter's
CANN/torch/torch-npu trio is checked against the pin's own requirements table before
any version advice, and the cheapest working install is rung 1. When the body carries **no
failing frames at all** — prose only, "installing vllm 0.9.1 prompted me to uninstall torch
2.5.1 and install 2.7.x, then vllm-ascend could not find torch 2.5.1" (#1363) — it is a
**pairing** question: the release's own versioning-policy row, the two requirement trees and
`_no_device()` decide it in four reads, and rung 1 is the project's own recipe (vllm from
source with `VLLM_TARGET_DEVICE=empty`, then the tree's mirror), not another pip command
(`../image-dependency-faults/pip-index-visibility.md` §A0). **Read the pin's
`[build-system] requires` before accepting that route.** When it puts `torch-npu==X` beside a
`torch>=X` floor (v0.8.5rc1 `pyproject.toml:14-15`), the failure that release produces on an
x86 host whose index does not carry the `+cpu` label is the §D build-dependency conflict —
`ERROR: Cannot install torch-npu==X and torch>=X … torch-npu X depends on torch==X+cpu` under
`× pip subprocess to install build dependencies did not run successfully` — whose cure is one
flag (`--no-build-isolation`, which the same pin's `installation.md:168` names, or
`--extra-index-url …/whl/cpu`). **A report that says it also cannot *compile from
source* has not added a second fault when both legs carry the same pin** — at `v0.9.2rc1`
`pyproject.toml:15` (`[build-system] requires`) and `requirements.txt:27` (→
`install_requires` through `setup.py:341,388`) name the same
`torch-npu==2.5.1.post1.dev20250619`, so the sdist/editable leg re-raises the identical
`Could not find a version …` from the build-deps subprocess and the levers are the wheel
leg's two (`PIP_EXTRA_INDEX_URL`, or `--no-build-isolation`); only §D's shape, where the
torch pair lives *only* in the build requirements, is a build-specific conflict
(`../image-dependency-faults/pip-index-visibility.md` §A). Then the first
message carries that rung, and the title's
"x86 systems cannot install through pip" is the reporter's **premise, not a finding**: that
pin's own pip tab tells x86 users to add the CPU index (`installation.md:126,129`), so the
project documents the path and the `+cpu` label — not a missing arch wheel — is the usual
holder. One call decides which: `bash <skill_root>/scripts/pip-pairing-check.sh
<ascend-root> [vllm-root]` prints the pin pair, the predicted error text and the rung, and
`rg -n '^## ' <page>` before reading any § (two recorded runs read `sed -n '1,140p'` of that
page, landed in §A0 and never saw §C/§D — #446, #1411; the second one ranked "no x86_64
wheel is published" first and shipped `--no-build-isolation` as ladder step 3).
**Not every no-frame shape is a pairing question.** A title that paraphrases a *Python* requirement
(「python version should above 3.10」) with its printed line only in the screenshot's OCR
is an interpreter question: the string is in no tree, pip was told `>=3.9` by
`pyproject.toml` / `python_requires` / `installation.md:8`, and the failure is a PEP 604
`X | None` annotation evaluated at import in `vllm/` — recreate the env on 3.10/3.11
(`../image-dependency-faults/python-floor-import-error.md`, #3657). A body that is
nothing but one dangling build path (`csrc/build/compat/liboptiling.so 指向无效`) is a
build-tree artefact answered from the paste — the `_CPack_Packages/…/op_tiling/lib/linux/<arch>/`
copy such a body usually names is that link's own resolution target — not a stage to walk
(`../image-dependency-faults/custom-op-build-stage.md` §Step 2b, #10792: one run spent 60
tool calls, 1.65 M prompt tokens and emitted no message; the next spent 14 calls and
answered). A `multiple
definition` at the AscendC kernel relink is one source under two path spellings, not a
kernel defect: `../image-dependency-faults/custom-op-build-stage.md`. A build that dies
**before** any CMake work with `__main__.py: error: unrecognized arguments: --cmake` is pybind11
3.0's option rename against a bare `pybind11` build requirement — the pinned tree's `setup.py`
already says `--cmakedir`, and a host-side downgrade is inert under build isolation:
`../image-dependency-faults/pybind11-cmake-option-rename.md`.
