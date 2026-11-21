# `module 'triton' has no attribute 'language'` at `import torch`: a `triton` that is not a triton

## Scope

Nothing of vllm-ascend has loaded yet; the interpreter itself dies, so there is
**no `vllm_ascend` frame anywhere**:

```
File "/usr/local/python3.12.13/lib/python3.12/site-packages/torch/__init__.py", line 2935, in _import_device_backends
File ".../torch_npu/__init__.py", line 41, in <module>            import torch_npu.npu
File ".../torch_npu/utils/_dynamo.py", line 6                     from torch._dynamo.utils import tensortype_to_dtype
File ".../torch/_dynamo/utils.py", line 2503, in <module>         common_constant_types.add(triton.language.dtype)
AttributeError: module 'triton' has no attribute 'language'
```

recorded on image `vllm-ascend:v0.23.0rc1-310p`, 310P3, x86, Qwen3-VL-8B
(#13199). Owner: **the `triton` the interpreter resolves on that box**, not
torch, not torch_npu, not vllm-ascend, not the model. `import triton` succeeded
(so nothing raised `ModuleNotFoundError` and torch's presence check passed), yet
the module object has no `language` submodule bound.

Sibling lanes: a *complete* triton-ascend whose ascend backend cannot map the SoC
fails later and differently (`SystemError: <built-in function get_arch> returned
NULL` at the worker import — `../soc-platform-paths/atlas-300i-310p.md`); a
complete pair that compiles bad IR fails on the first request
(`triton-ascend-pair.md`); and an upstream triton or a `triton_kernels` package
installed beside triton-ascend logs `Failed to import Triton kernels … No module
named 'triton.language.target_info'` and then fails later in
`ops/layernorm_gated.py` (`triton-kernels-import-error.md`). This page is the
**import-time resolution** form.

## Why a normal install cannot produce it

Both candidate wheels bind the submodule eagerly in their own `__init__`, so a
complete install of either always has `triton.language`:

| Tree at this pin | Anchor |
|---|---|
| upstream triton v3.7.1 (`f797708c`) | `python/triton/__init__.py:26` `from . import language` |
| triton-ascend v3.2.1 (`2badfc89`) | `python/triton/__init__.py:25` `from . import language` |

Therefore the module that got imported is **not a complete package**: it is a
leftover directory, a shadowing directory earlier on `sys.path`, or a wheel whose
`__init__` never reached that line. Which one is a property of the reporter's
filesystem and **cannot be derived from the paste** — see "One command" below.

## What the image is supposed to contain (v0.23.0rc1 `f4a08bd`)

- `docs/source/tutorials/hardwares/310p.md:7`: "Atlas 300I DUO does not support
  `triton` or `triton-ascend`."
- `Dockerfile.310p:45` carries the reason ("in x86, triton will be installed by
  vllm. But in Ascend, triton doesn't work correctly. we need to uninstall it"),
  `:47` `python3 -m pip uninstall -y triton`, `:63`
  `python3 -m pip uninstall -y triton-ascend triton` — and **no reinstall**.
- Contrast, same pin: `Dockerfile:70-71` and `Dockerfile.a3:72-73` uninstall then
  `pip install triton-ascend==3.2.1`. So `triton` present is normal on A2/A3
  images and abnormal on `-310p`; ask which image before ranking anything.
- `requirements.txt:16-20` still *pins* the triton stack the image removes
  (`torch==2.10.0`, `torch-npu==2.10.0.post2`, `triton-ascend==3.2.1`), and
  triton-ascend's own `requirements.txt:1` asks for `torch-npu==2.7.1` — so a
  hand `pip install triton-ascend` on such an image both violates the SoC
  statement and drags a torch_npu the rest of the stack does not match.

## Ranked causes, each with its one check

1. **Uninstall residue left as a namespace package** (hypothesis, and the usual
   one on a `-310p` image): `pip uninstall` deletes only the paths in the wheel's
   `RECORD`, so a `site-packages/triton/` that keeps e.g. `__pycache__/` or a
   `backends/` subtree remains importable as a PEP 420 namespace package with no
   attributes. Check: the resolution probe below prints `None` for `__file__`
   and a list for `__path__`, and `find <site-packages>/triton -maxdepth 2`
   shows no `language/`, no `runtime/`.
2. **A stray `triton` directory earlier on `sys.path`/`PYTHONPATH`** (a checkout,
   a copied tree, a different interpreter's site-packages): the probe's
   `triton.__file__` names a path outside the serving interpreter's
   site-packages.
3. **A foreign or hand-installed triton wheel** whose `__init__` never binds
   `language` (an old triton, or a distribution that ships the name only
   partially). Check `pip list | grep -i triton` — **do not** trust
   `triton.__version__`: the 3.2.1 triton-ascend tree prints `3.2.0`
   (`python/triton/__init__.py:2` vs `version.txt`).

## One command (the whole diagnosis)

```
python -c "import triton, sys; print(sys.executable); print(triton.__file__); \
print(list(triton.__path__)); print(getattr(triton,'__version__','NO VERSION'))"
```

`None` + a `__path__` list + `NO VERSION` ⇒ residue/shadow directory (cause 1 or
2, repair below). A real `__file__` with a version ⇒ cause 3: uninstall that
distribution and re-check.

## Repair (in this order)

1. Remove every `triton*` distribution: `pip uninstall -y triton triton-ascend`,
   then delete/rename any surviving `site-packages/triton` directory (the residue
   is exactly what `pip` cannot see).
2. Verify the intended state, which on `-310p` is **no triton at all**:
   `python -c "import triton"` must raise `ModuleNotFoundError`, and
   `python -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"` must
   print `False` (310P runs eager; the Triton branches are `HAS_TRITON`-gated).
3. Re-run the untouched `vllm serve` line. The model command needs no change:
   `--dtype float16`, `--max-model-len`, `--enforce-eager` are all irrelevant to
   an `import torch` failure.
4. **Do not "fix" it by installing triton/triton-ascend on this SoC.** The doc
   line above forbids it, and the first Triton kernel then dies in BiSheng —
   `Cannot find option named 'Ascend310P3'` / `--target=Ascend310P1`
   (#7421, #7423 record that pair; #10641 is the mirror failure,
   `No matching distribution found for triton-ascend==3.2.1`). If the reporter
   needs Triton kernels, the answer is a non-310P card/image, not a wheel.

## Labels and gaps

- The traceback shape, the uninstall/keep split and the two `__init__` lines are
  reporter-paste plus source-verified at the commits named above.
- The **torch_npu side is citable at the pin**: `Ascend/pytorch`
  (`v26.0.1-pytorch2.10.0`, `8751b36d`) `torch_npu/utils/_dynamo.py:6` is exactly
  `from torch._dynamo.utils import tensortype_to_dtype`, so torch_npu drags
  `torch._dynamo` in while `import torch` is still running its device-backend
  loop — that is the mechanism, verified.
- `torch/_dynamo/utils.py` **is not in that case's pytorch export** (the
  `pytorch/pytorch` snapshot is `v2.12.1` and its `torch/_dynamo/` holds only a
  placeholder, `1714/21642` files), so `:2503` and the
  `common_constant_types.add(triton.language.dtype)` line are citable only as the
  reporter's paste — and the reporter's torch is 2.10, not that snapshot's 2.12.
  Do not spend a call grepping for them.
- No recorded issue body in that corpus carries this exact string
  (`grep -rln "has no attribute 'language'" history/` ⇒ 0 hits), so say the
  token has no precedent instead of implying one. The *nearest* precedents are
  the 310P-triton family above.
