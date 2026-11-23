# `[Installation]` titled with a Python-version requirement: the import-time `TypeError`

Scope: an `[Installation]` / 安装 report whose **title is the reporter's own summary of a
requirement** — "python version should above 3.10", "能不能用 python3.9", "python 版本不对"
— while the body pastes `npu-smi`, a `conda`/`pip` transcript and a screenshot instead of
the printed line. No engine ever starts, so `npu-smi` being healthy, CANN version, driver
and every device question are irrelevant to it.

Verified at vllm-ascend **v0.11.0rc0** (`00ba0710c1`) with vllm **v0.11.0rc3**
(`8ce5d3198d`), the tree recorded **#3657** (910B4, CANN `8.2.RC1` aarch64, conda env
cloned from an old `PyTorch-2.1.0`, `pip install vllm==0.11.0` +
`vllm-ascend==0.11.0rc0`, torch_npu `2.7.1.dev20250724` **`cp39`** wheel). The union, file
and frame move between releases; the shape — install report, paraphrased title,
import-time `TypeError`/`AttributeError` whose deepest frame is in `vllm/` — is what is
stable, and the answer below needs three reads.

## Step 1 — the title is a paraphrase, so the deciding line is in the attachment

Neither tree prints that sentence: `rg -n 'above 3\.10|should above'` across the vllm and
vllm-ascend checkouts returns nothing, and the only `sys.version_info < (3, 10)` sites are
`importlib_metadata` **fallbacks** (`vllm/plugins/__init__.py:19`,
`vllm/v1/sample/logits_processor/__init__.py:50`, `requirements/common.txt:34`) — i.e. the
project still intends to support 3.9. So open the screenshot's OCR
(`/workspace/attachments/<issue>/*.ocr.txt`); #3657's carries, after the garbled
`vllm.platform_plugins` INFO lines:

```
File ".../envs/vllm-0.11.0/lib/python3.9/site-packages/vllm/model_executor/models/registry.py", line 426, in <module>
  class LazyRegisteredModel(BaseRegisteredModel):
File ".../vllm/model_executor/models/registry.py", line 442, in LazyRegisteredModel
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

Two frames with the **second inside the class body** means a `def` annotation evaluated at
definition time: that file has no `from __future__ import annotations`, and the annotated
return type is `_ModelInfo | None` (`registry.py:440-441` at the pin, `:442` in the
reporter's `0.11.0` final), where `_ModelInfo` is itself a frozen dataclass
(`registry.py:346-347`). Python 3.9 has no `type.__or__`, so importing `vllm` raises before
any vllm-ascend code, plugin registration or device open.

## Step 2 — pip cannot catch it, and the metadata is not the answer

Pip's gate is metadata, and the metadata at that pin admits 3.9:

| File at the pin | Line | Value |
|---|---|---|
| vllm `pyproject.toml` | `:34` | `requires-python = ">=3.9,<3.14"` |
| vllm-ascend `setup.py` | `:387` | `python_requires=">=3.9"` (`:376` still classifies 3.9) |
| vllm-ascend `docs/source/installation.md` | `:8` | `- Python: >= 3.9, < 3.12` |

So the honest classification is **a per-file inconsistency in the released code**, not
"版本不匹配" and not a wrong pip command: say that in one line (it is what the reporter
would file next), then give the interpreter fix. Do not propose patching
`registry.py`, and do not propose downgrading vllm — the same file carries many such
annotations and 3.9 is a lane upstream is leaving.

## Step 3 — the answer, and the reason the env was 3.9 in the first place

Recreate the environment on **3.10/3.11**; 3.11 is the safe choice because the docs cap at
`< 3.12` and `setup.py:374` says `# TODO: Add 3.12 back when torch-npu support 3.12`.

The paste itself explains the 3.9: `conda create -n PyTorch-2.7.1 --clone PyTorch-2.1.0`
clones the **interpreter** along with the packages — the same
`envs/PyTorch-2.1.0/lib/python3.9/...` path is the one in #152 and #340, so
"clone my old env" keeps a 3.9 base forever.

```bash
conda create -n vllm-0.11.0 python=3.11 -y            # no --clone
pip install torch==2.7.1
pip install torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl   # a cp39 wheel does not cross interpreters
pip install vllm==0.11.0 vllm-ascend==0.11.0rc0
python -V; python -c "import vllm, vllm_ascend; print(vllm.__version__, vllm_ascend.__version__)"
```

That last import is the acceptance test: the copy step the reporter needs is the wheel
tag (`cp39` → `cp311`), which is the one thing a clone cannot fix for them.

## Recheck conditions

- Quote both line numbers when the reporter's release differs from the pin (`:442` in
  `0.11.0` final vs `:440-441` at the rc3 pin) — the pair is the evidence, one alone reads
  as a mis-transcription.
- The *file* is version-specific: any `X | None` annotation evaluated at import in
  `vllm/` reproduces it, e.g. a `TypeError: unsupported operand type(s) for |` on a
  different module is the same lane, not a different report.
- Not verified locally: whether the reporter's `torch 2.7.1` itself installs on 3.9 (the
  pytorch export carries no `pyproject.toml` at that path), so never claim torch is the
  blocker — the vllm frame is, and it fires before torch matters.
- A `Requires-Python`-honest release would have refused the install outright; if a future
  pin raises the floor, this page's answer collapses into "pip already told them".
