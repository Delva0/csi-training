# `Failed to import Triton kernels … Error: No module named 'triton.<x>'`: a logged line, and the triton stack under it

## Scope

The report leads with this sentence (recorded #11681, 2026-07-09; vllm-ascend
`v0.18.0rc1` `99e1ea0f`, vllm `v0.18.0` `bcf2be96`, triton-ascend 3.2.0, torch /
torch-npu 2.9.0, 910C):

```
Failed to import Triton kernels. Please make sure your triton version is compatible.
Error: No module named 'triton.language.target_info'
```

and the recorded body also volunteers a ladder of further triton errors — *"之前尝试过
triton 3.5.0，3.6.0，出现过其它问题，比如 `AttributeError: module 'triton' has no
attribute 'next_power_of_2'`，修改后有 `triton.csdiv` 等等"*, ending
`_layer_norm_fwd_1pass_kernel_npu[grid]` launch raising `TypeError: 'function' object is not
subscriptable`, with the failures *"集中出现在 `ops/layernorm_gated.py`"*. Every one of
these is a statement about **which `triton` the interpreter resolved**, and the
sentence in the title is the least important of them.

Sibling lanes: `triton-module-residue.md` (a resolved `triton` with no `language` at
`import torch`), `triton-ascend-pair.md` (a complete pair that miscompiles at the
first request), `../soc-platform-paths/atlas-300i-310p.md` (the triton-less 310P
box where the *placeholder* is the intended state). This page is the **hand-built
box where an upstream triton or a `triton_kernels` package sits beside
triton-ascend**.

## Step 0 — the title sentence is upstream vLLM's `logger.error`, so it is not the crash

vllm `v0.18.0` emits it in exactly two places, both around a **swallowed import**:

- `vllm/model_executor/layers/fused_moe/config.py:25-34` — `if has_triton_kernels():`
  → `try: from triton_kernels.matmul_ogs import PrecisionConfig` →
  `except (ImportError, AttributeError) as e: logger.error("Failed to import Triton kernels. …")`
- `vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py:60-65` — same
  text around the gpt-oss MoE kernel imports.

It is `logger.error`, **not a raise**: the process continues and `PrecisionConfig`
simply stays unbound (only the gpt-oss MoE path reads it). So the sentence is a
*wrapper* (SKILL.md rule 11) and its advice — "make sure your triton version is
compatible" — names the action the reporter already tried and that made the chain
worse. The fatal frame is elsewhere and usually absent from the paste.

`has_triton_kernels()` (`vllm/utils/import_utils.py:420-427`) is true when **either**
`triton_kernels` or `vllm.third_party.triton_kernels` is importable;
`import_triton_kernels()` (`:59-84`) prefers the site-packages one and falls back to
the wheel's vendored copy (`setup.py:302-311`; a source checkout has no
`vllm/third_party/triton_kernels`, so local `rg` cannot tell you which copy ran —
`python -c "import triton_kernels; print(triton_kernels.__file__)"` can).

## Where the missing name comes from (verified chain)

- The importer is `triton_kernels`, not vllm-ascend: triton v3.7.1
  `python/triton_kernels/triton_kernels/target_info.py:5` is literally
  `from triton.language.target_info import (…)`.
- `triton.language.target_info` ships with **upstream** triton (v3.7.1
  `python/triton/language/__init__.py:139 from . import target_info`, file
  `python/triton/language/target_info.py`) and exists **nowhere** in triton-ascend
  v3.2.1 (`rg -l target_info` over that tree → 0 hits).
- Therefore: the interpreter resolves **triton-ascend's** `triton`
  (`python/triton/__init__.py:2 __version__ = '3.2.0'`) while a **newer
  `triton_kernels`** (from a hand-installed upstream triton ≥3.5, or a separate
  `triton_kernels` wheel) imports a submodule only the newer triton has. The triton
  tree itself can be perfectly self-consistent — "mixed tree" is the second reading,
  and the one that needs no evidence from an export nobody has.

## The other three errors are the same axis

- **`AttributeError: module 'triton' has no attribute 'next_power_of_2'`** has two
  readings and only one is verifiable here. (i) **The placeholder**: with
  `HAS_TRITON` false, `vllm/triton_utils/__init__.py:7-18` binds `TritonPlaceholder`,
  which defines only `jit/autotune/heuristics/Config/language`
  (`vllm/triton_utils/importing.py:74-90`) — so `triton.next_power_of_2` cannot exist,
  and this form arrives *after* a triton uninstall. (ii) An upstream triton that
  dropped the top-level name: **not** true of v3.7.1
  (`python/triton/__init__.py:73 def next_power_of_2`), and the 3.5/3.6 trees are not
  in any snapshot — label it unverified.
  `HAS_TRITON` is not "triton is installed": it needs `find_spec("triton")` **and**
  `triton.backends` importable **and** exactly one active backend driver
  (`importing.py:12-57`), which an overwritten/gutted tree fails while `import triton`
  still succeeds.
- **`TypeError: 'function' object is not subscriptable` at the `kernel[grid]` launch** is the
  same placeholder one step later: the dummy decorator returns the raw function
  (`importing.py:84-90` `return args[0]` / `lambda f: f`), so `@triton.jit` yields a
  plain function and the launch subscript raises — for the recorded paste that is
  `vllm_ascend/ops/triton/layernorm_gated.py:16` (the kernel) and `:148`
  (`_layer_norm_fwd_1pass_kernel_npu[grid]`, the name in the reporter's message). That is why the reporter's chain
  "never ends": each repair moves the same absence to the next line.
- **Why it lands in `ops/layernorm_gated.py`**: `vllm_ascend/ops/layernorm.py:24`
  imports `layer_norm_fwd_npu` from `vllm_ascend/ops/triton/layernorm_gated.py`
  **unconditionally** (no `HAS_TRITON` guard), and that module is written against the
  triton-ascend dialect — `:10 from vllm.triton_utils import tl, triton`,
  `:139 BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))`. So an
  upstream triton is never the fix: **triton-ascend's own wheel provides the `triton`
  package** (triton-ascend v3.2.1 `python/setup.py:784-796` packages `triton`,
  `triton/language`, …; `:844-845` `name=get_package_name()` → `triton_ascend`), which
  is why `pip uninstall triton` on such a box deletes `triton/*` paths the triton-ascend
  dist also owns and leaves a gutted tree — the recorded "只使用 triton-ascend，卸载
  triton，行不通".

## One command

```
python -c "
import triton; print(type(triton).__name__, getattr(triton,'__file__',None), getattr(triton,'__version__','-'))
from vllm.triton_utils import HAS_TRITON; print('HAS_TRITON', HAS_TRITON)
from vllm.utils.import_utils import has_triton_kernels; print('triton_kernels', has_triton_kernels())"
pip list | grep -i triton
```

- `TritonPlaceholder` ⇒ no usable triton (the last two error strings above); repair
  the stack, do not touch vllm-ascend.
- a real module with `__version__ 3.2.0` and `triton_kernels True` ⇒ this page's
  chain: drop `triton_kernels` and any upstream `triton`.
- `collect_env.py` filters `pip list` by `triton` and the reporter's dump was left as
  the template, so a paste without those lines is a **gap to ask for**, not a
  triton-free box.

## Repair ladder

1. `pip uninstall -y triton triton_kernels` — every name `pip list | grep -i triton`
   shows except `triton-ascend` — then delete any surviving `site-packages/triton`
   directory that triton-ascend did not write (`pip uninstall` removes only its own
   `RECORD` paths), then `pip install triton-ascend==3.2.0` for this pin
   (`requirements.txt:40`; `torch==2.9.0` `:16`, `torch-npu==2.9.0` `:35`).
2. Verify with the three probes above (`HAS_TRITON True`, `triton_kernels False`) plus
   `python -c "import vllm_ascend.ops.triton.layernorm_gated"`, then re-run the same
   serve line unchanged.
3. **Do not offer an upstream triton version as the fix** and do not debug hand-edited
   kernels: a box whose `site-packages/triton` was patched is not a state to reason
   from — first re-run on the image / a fresh venv, then read whatever is left.
4. Ask back for the *first* traceback (the `EngineCore pid=…` block above the logged
   sentence — the paste carried only the wrapper), the model directory, and the two
   probe lines. The model matters: a dense VL model serving fine on the same box
   proves the import path works and makes the failing model the one that reaches
   `layer_norm_fwd_npu` at forward time (`ops/layernorm.py:131`), i.e. a hybrid
   linear-attention family.

Say plainly that cleaning the stack is not "then it will work" (SKILL.md rule 3): the
fatal frame was never in the paste, so the deliverable is the verdict, the probe and
the ask.

## Labels and gaps

- Source-verified at the pins above: the two `logger.error` sites and the
  `has_triton_kernels`/`import_triton_kernels` pair, the `triton_kernels` →
  `triton.language.target_info` import, `target_info` absent from triton-ascend,
  `next_power_of_2` still present in upstream v3.7.1, the placeholder's
  `jit`/`__version__` behaviour, and `ops/layernorm.py:24`'s unguarded import.
- Hypothesis, and stated as such: that the gutted tree came from `pip uninstall`
  overlap; that the 3.5/3.6 runs removed `next_power_of_2`.
- Unconfirmed: no resolution appears in this snapshot — the reporter's last public
  comment is a mention-only ping — so the ranking is a ranked diagnosis, not a
  reporter-confirmed fix. Recheck the `config.py` line numbers per vllm release; the
  two sentences are the stable anchor.
