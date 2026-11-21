# The patch ran, but the consumer module kept the pre-patch name

A third dispatch failure beside `patch-import-side-effect.md` (hook never ran in
that process) and `oot-class-registration-drift.md` (replacement keyed by class
name): here the patch **did** run in the failing process, rebound the defining
module, and the model module that calls the name had already cached it. Downstream
still owns it — the patch is incomplete, not absent.

Recorded #15815: vllm-ascend `v0.26.0rc1` (`f2f74a16`), vllm `v0.26.0`
(`568afb3a`), Qwen3-Coder-30B-A3B BF16, TP4 with EP on and off.

## Symptom

Startup fails inside `profile_run()` (Lane B, memory-profiling phase), before any
request, with an `AttributeError` whose message contradicts the code:

```text
vllm_ascend/worker/worker.py:546 determine_available_memory -> self.model_runner.profile_run()
vllm_ascend/worker/model_runner_v1.py:3590 profile_run -> self._dummy_run(...)
vllm/model_executor/models/qwen3_moe.py:232 forward -> final_hidden_states = self.experts(...)
vllm/model_executor/layers/fused_moe/runner/moe_runner.py:830 _forward_impl
vllm/model_executor/layers/fused_moe/runner/moe_runner.py:564 _apply_quant_method
AttributeError: 'UnquantizedFusedMoEMethod' object has no attribute 'is_monolithic'.
Did you mean: 'apply_monolithic'?
```

The property exists. Calling the descriptor directly on the same object raises
`AttributeError: 'NoneType' object has no attribute 'is_monolithic'` — that inner
sentence is the evidence, the outer one is `torch.nn.Module.__getattr__` masking a
property that raised. Two conclusions follow immediately: a `None` sub-object is
the owner (not a missing method), and the frame file is upstream only because the
*wrong runner* is installed.

## Mechanism (both trees read at the pins above)

- Consumer caches the factory at import: `vllm/model_executor/models/qwen3_moe.py:45`
  `from vllm.model_executor.layers.fused_moe import FusedMoE`; constructed at
  `:204`, called from `forward` at `:231-233` (`self.experts(...)`).
- The patch rebinds only the two defining modules:
  `vllm_ascend/patch/platform/patch_fused_moe.py:87-88`
  `_fused_moe_layer.FusedMoE = _ascend_FusedMoE` and
  `_fused_moe_pkg.FusedMoE = _ascend_FusedMoE`. Nothing sweeps `sys.modules`.
- The ordering assumption is the project's own, in writing: the same file's header
  (`:20-27`) says the patch must land "before any model is imported", with the
  intended sequence `adapt_patch() -> FusedMoE patched -> model loading ->
  deepseek_v2 imported -> gets patched FusedMoE`; `vllm_ascend/patch/__init__.py:185-191`
  repeats it ("before any model is imported").
- The stale path is a **guaranteed** crash, not a slow path: the upstream
  unquantized method picks its kernel class at construction
  (`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:50-53`
  `self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(...)`)
  and a non-CUDA platform returns `(UnquantizedMoeBackend.OOT, None)`
  (`fused_moe/oracle/unquantized.py:203-210`; CPU and TPU return `None` too;
  `is_out_of_tree` is `PlatformEnum.OOT`, `vllm/platforms/interface.py:211-212`).
  The base property then dereferences it:
  `fused_moe_method_base.py:163-170` — `moe_kernel is None` -> `hasattr(self,
  "experts_cls")` is **true** (the attribute exists, value `None`) ->
  `self.experts_cls.is_monolithic()` -> `AttributeError` on `None`.
  So on Ascend the upstream runner/method cannot work at all; the Ascend runner is
  what makes `experts_cls` irrelevant, and a stale binding sends the call to the
  upstream one.

## Discriminator: three one-line checks

Run them in the process that failed (or reproduce the import order in a fresh one);
the reporter's own reproducer is the cheapest form:

```python
import vllm.model_executor.layers.fused_moe as pkg          # package
import vllm.model_executor.layers.fused_moe.layer as layer  # defining module
import vllm.model_executor.models.qwen3_moe as qwen3        # consumer
original = layer.FusedMoE
import vllm_ascend.patch.platform.patch_fused_moe as patch
print(layer.FusedMoE is patch._ascend_FusedMoE)   # True  -> patch ran
print(pkg.FusedMoE   is patch._ascend_FusedMoE)   # True  -> package rebound
print(qwen3.FusedMoE is patch._ascend_FusedMoE)   # False -> stale consumer alias
print(qwen3.FusedMoE is original)                 # True  -> it holds the original
```

How to read the three neighbouring pages with it:

| Observation | Owner |
|---|---|
| `layer.FusedMoE` is the Ascend wrapper, `qwen3_moe.FusedMoE` is the original | this page — incomplete patch, fix by sweeping `sys.modules` |
| the patch module is not in `sys.modules` at all (`patch-import-side-effect.md`) | the hook never ran in that process |
| the symbol is registered by class name and the name is missing (`oot-class-registration-drift.md`) | registry drift |
| the property's own body raised because it **selected a different module at runtime** (`model_instance.model.layers[i]…`) and a feature such as `--enable-lora` had substituted that module — the message names a class that is not the object holding the gap | `../lora-adapters/fusion-reader-vs-lora-wrapper.md` |

## Where the order breaks, and where it merely looks fragile

- Plugin import installs the platform set first: `vllm_ascend/__init__.py:32-34`
  calls `adapt_patch(is_global_patch=True)`, so a stock `vllm serve` normally
  patches before any model module is imported. The reported repro needs something
  to import a model module earlier — the reporter's own environment is a `runpy`
  launcher with frontend changes, and they say a pristine CLI launch was not
  verified. Do not promote that to "stock startup is broken".
- Worker init only reuses it: `vllm_ascend/worker/worker.py:119` `adapt_patch()`,
  and `patch/worker/patch_fused_moe.py:20` merely imports the platform module
  ("avoids wrapping an already patched factory").
- The in-tree order that *looks* like the bug is not one: the worker set imports
  `patch_qwen3vl` at `patch/worker/__init__.py:38`, and that module does
  `from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention` at
  `patch_qwen3vl.py:5`, while `patch_fused_moe` is imported at `:63`. It is safe
  only because the parent package's `__init__` already ran the global patch in
  that same process. Worth one comment in the PR, not a defect claim.

## Breadth and the fix

At the vllm pin, **15** modules do the package-level import and inherit the same
hazard — `qwen3_moe`, `glm4_moe`, `ernie45_moe`, `bailing_moe`, `aria`, `jamba`,
`laguna`, `lfm2_moe`, `openpangu`, `param2moe`, `cohere2_moe`, `sarvam`,
`transformers/moe.py`, `models/inkling/nvidia/moe.py`, and
`layers/quantization/inc/schemes/inc_scheme.py`
(`rg -ln 'from vllm\.model_executor\.layers\.fused_moe import FusedMoE' vllm/`).
A qwen3-only repair leaves the same class of failure for the others, so propose
the sweep after the two existing assignments:

```python
import sys
for _mod in list(sys.modules.values()):
    if getattr(_mod, "FusedMoE", None) is _original_FusedMoE:
        _mod.FusedMoE = _ascend_FusedMoE
```

The identity guard is sound here because `_original_FusedMoE` is captured at
`patch_fused_moe.py:38` before assignment and the worker module only re-imports
the platform module (idempotent). The in-tree precedent for exactly this style is
`vllm_ascend/patch/worker/patch_process_weights_after_loading.py:58-66`
(`sys.modules.get(loader_module)` then rewrite the attribute), so cite it — the
change reads as a consistency fix rather than a new mechanism.

## Limits to state in the answer

- A sweep repairs **module attributes** only; a factory captured as a default
  argument or in a closure before the patch is not fixed. `RoutedExperts` needs no
  second rebind — the factory resolves it per call.
- The exports carry one commit per repo, so "is it fixed in a later release" is
  unanswerable here; give the recheck (grep `patch_fused_moe.py` for `sys.modules`).
- Traceback line numbers matching the pinned vllm (`moe_runner.py:564` is literally
  `if self.routed_experts.quant_method.is_monolithic`) is the version evidence —
  the issue's version table alone is not.
- Not causes, do not pursue: the checkpoint, BF16, EP (the reporter reproduced with
  EP on and off), TP size, the MoE kernel, and `is_monolithic` itself.
- Missing artefact that settles severity: one stock `vllm serve` run at these pins
  with no launcher, plus the worker log line showing whether `vllm_ascend` was
  imported before the model module.
