# The install succeeded and pip only *warned*: `vllm 0.18.0 requires torch==2.10.0, but you have torch 2.9.0`

Scope: the report is a **two-command wheel install** (`pip install vllm==X`, then
`pip install vllm-ascend==Y`) whose output is pip's post-install warning block, not
an `ERROR: Could not find a version`:

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
vllm 0.18.0 requires opencv-python-headless>=4.13.0, but you have opencv-python-headless 4.11.0.86 …
vllm 0.18.0 requires torch==2.10.0, but you have torch 2.9.0 which is incompatible.
vllm 0.18.0 requires torchaudio==2.10.0 … / torchvision==0.25.0 … which is incompatible.
te 0.4.0 requires ml-dtypes, which is not installed.
outlines 0.1.11 requires outlines_core==0.1.26, but you have outlines-core 0.2.11 which is incompatible.
```

Worked case #8286 (`vllm & vllm-ascend torch version conflict`, vllm-ascend
v0.18.0rc1 `99e1ea0` + vllm v0.18.0 `bcf2be9`). The sibling page
`pip-index-visibility.md` owns the hard-error family; this page owns the shape where
**nothing failed** and the deliverable is *which statement supplied each pin* plus why
the obvious "fix" is the one thing that breaks the install.

## Rule 0 — the block is a set of statements about *metadata*, one per line

`pip` re-reads the installed distributions after the second command and prints one
line per unsatisfied `Requires-Dist`. It exits 0. So:

- answer **"what does this prove"**, never "which wheel is wrong";
- the only torch line worth chasing is a mismatch *inside* the torch trio
  (`python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"`);
- **never** propose `pip install torch==2.10.0` / `torchvision==0.25.0` to silence it.
  A `torch_npu` wheel pins its exactly matching torch (mechanism and `Ascend/pytorch`
  `setup.py` line in `pip-index-visibility.md` §A.3, `abi-undefined-symbol.md` for the
  C++ side), so raising torch is what turns a cosmetic warning into a dead
  `import torch_npu`.

## A. Which file put `torch==2.10.0` in the graph — one grep decides it

Upstream `vllm/setup.py:843-864 get_requirements()` chooses the requirement file by
target device: `_no_device()` → `common.txt` **only** (no torch pin at all);
`_is_cuda()` → `cuda.txt`, which is `torch==2.10.0` / `torchaudio==2.10.0` /
`torchvision==0.25.0` (`requirements/cuda.txt:7-10`). Therefore:

| What the reporter installed | Which torch pin its metadata carries |
|---|---|
| vllm **from PyPI** (a cuda wheel) | the 2.10 line — this is what produced the block |
| vllm **from source** with `VLLM_TARGET_DEVICE=empty` | none; `common.txt` only |

Downstream ships exactly the second form: `Dockerfile:52-56`
`ARG VLLM_TAG=v0.18.0` then `VLLM_TARGET_DEVICE="empty" python3 -m pip install -v -e
/vllm-workspace/vllm/[audio] --extra-index https://download.pytorch.org/whl/cpu/`,
with vllm-ascend installed after it (`:68`). That is the project's supported artefact
and it pairs vllm 0.18.0 with this tree's torch 2.9.0 without any warning — so "is
this pair intended" is `Dockerfile:52-56` + `setup.py:843-864`, not a release survey.
The downstream side of the pin is `requirements.txt:16-18` (`torch==2.9.0`,
`torchvision==0.24.0`, `torchaudio==2.9.0`), `:35 torch-npu==2.9.0`, and
`setup.py:459-476,517` (`_read_requirements` → `install_requires`; `--` lines skipped),
which is why the second command downgrades torch and pip can only warn.

## B. The pair the tag *documents* is a third fact

`docs/source/conf.py:75-76` (`pip_vllm_ascend_version: 0.17.0rc1`,
`pip_vllm_version: 0.17.0`) feeds `docs/source/installation.md:148`, and the CI doctest
`tests/e2e/doctests/002-pip-binary-installation-test.sh:40-56` installs precisely that
pair, then hand-patches the trail:

```bash
pip install vllm=="${PIP_VLLM_VERSION}"        # 0.17.0
pip install vllm-ascend=="${PIP_VLLM_ASCEND_VERSION}"   # 0.17.0rc1
if [ "$PIP_VLLM_ASCEND_VERSION" == "0.17.0rc1" ]; then
    pip install torchvision==0.24.0 torchaudio==2.9.0
fi
python3 ./offline_inference_npu.py            # the actual acceptance gate
```

Two consequences to state in the answer: the project's own pip test **expects** the
block and fixes `torchvision`/`torchaudio` by hand; and the reporter's pair
(0.18.0 + 0.18.0rc1) is not the tag's documented pip pair, so a metadata-clean
environment is the documented pair or the image — not a hand-assembled torch set.
One more check before trusting the paste: the doc line the reporter copied reads "The
newest supported version is v0.18.0" (the `|vllm_version|` substitution), but this
tag's `conf.py:68` renders `v0.17.0` — so they were reading a docs build from another
ref, which is how a reporter ends up pairing a version the tag does not substitute.
`conf.py:80 ci_vllm_version: v0.18.0` is the **source-build** CI pin
(`VLLM_TARGET_DEVICE=empty`), not a pip pair; reading it as "vllm 0.18.0 is the pip
partner" is what leaves "does vllm 0.18.0 run on torch 2.9.0?" unanswerable (#8286).

## C. Which lines are not about this pair

- `te 0.4.0 requires ml-dtypes` — `transformer_engine` is a GPU-only leftover, absent
  from both trees' requirements: `pip uninstall transformer-engine`.
- `outlines 0.1.11 requires outlines_core==0.1.26` — upstream pins
  `outlines_core == 0.2.11` (`common.txt:23`) and the stale `outlines` is older than
  that; `pip install -U outlines` (or drop it and use `xgrammar`, downstream
  `requirements.txt:20 xgrammar>=0.1.30`).
- `opencv-python-headless` — a deliberate, permanent divergence: upstream floor
  `>= 4.13.0` (`common.txt:35`) against downstream cap `<=4.11.0.86 # Required to avoid
  numpy version conflict with vllm` (`requirements.txt:22`) under `numpy<2.0.0`
  (`:5`). Raising it drags numpy ≥ 2 in. Leave it and say why.

## D. Answer shape

First paragraph: nothing failed, the block is pip's post-install warning set, which
line supplied each pin, and the one command that shows whether anything is actually
broken — `python -c "import torch, torch_npu, vllm, vllm_ascend; print(...)"` plus the
tree's own `examples/offline_inference_npu.py` (the CI gate above). Ladder:

1. that import + offline example;
2. the two unrelated leftovers of §C;
3. if a warning-free environment is required: the tag's documented pip pair
   (`conf.py:75-76`) or the prebuilt image, or vllm from source with
   `VLLM_TARGET_DEVICE=empty` as `Dockerfile:52-56` does;
4. never raise torch to make `pip check` quiet; `pip check` will keep listing the
   torch trio and opencv on this pairing, and that residual is the design, not a fault.

## Evidence limits

- The two trees in the snapshot are v0.18.0rc1 + vllm v0.18.0 only; the documented pip
  pair (0.17.0 / 0.17.0rc1) is not present, so "0.17.0 pairs cleanly" is the docs' and
  CI's claim, not a read pin.
- Whether vllm 0.18.0 *runs* on torch 2.9.0 is upstream's call: the image builds it
  that way with an editable install, which is strong but indirect evidence. Say it is
  unverified and give the offline example as the test.
- The reported run (#8286, 10 assistant turns / 18 tool calls, answer delivered in full) reached the
  `conf.py` pip pair and the "torch_npu pins exactly" mechanism but never opened
  `Dockerfile:52-56` or upstream `setup.py:843-864`, so it left "is this pair
  intended" open when two greps closed it. Read those two first next time.
