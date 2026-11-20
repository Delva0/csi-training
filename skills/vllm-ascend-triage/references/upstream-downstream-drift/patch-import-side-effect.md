# The Ascend op was never installed in *that* process

Recorded #253: vllm-ascend `v0.7.1rc1` (`f17417f2`), vllm `v0.7.1` (`4f4d427a`),
DeepSeek-R1-BF16 on the reporter's `2x16P` machine, `--tensor-parallel-size 16
--pipeline-parallel-size 2 --distributed_executor_backend ray --enforce-eager
(32 ranks, so the second node is reached only through Ray):

```
vllm_ascend/worker.py:226 determine_num_available_blocks -> self.model_runner.profile_run()
vllm_ascend/model_runner.py:1139 execute_model -> model_executable(...)
vllm/model_executor/models/deepseek_v3.py:158 self.experts(...)
vllm/model_executor/layers/fused_moe/layer.py:584 layer.forward -> quant_method.apply(...)
vllm/model_executor/custom_op.py:23 forward -> self._forward_method(*args, **kwargs)
vllm/model_executor/custom_op.py:63 forward_oot -> return self.forward_native(...)
vllm/model_executor/layers/fused_moe/layer.py:156 forward_cuda -> return fused_experts(...)
TypeError: 'NoneType' object is not callable
```

The reporter's own analysis — "`current_platform.is_cuda_alike()` is false, so
`fused_experts` is set to None" — is **correct about the `None` and wrong about
the fix**: `fused_experts = None` is the intended non-CUDA gate, and the missing
thing is vllm-ascend's MoE replacement, which never got installed in that
worker process.

## Mechanism (verified in both pinned trees)

Upstream gate and dispatch (vllm `v0.7.1`):

- `vllm/model_executor/layers/fused_moe/layer.py:17-20`
  `if current_platform.is_cuda_alike(): from .fused_moe import fused_experts` /
  `else: fused_experts = None`. Ascend is `PlatformEnum.OOT`, so the module-level
  name is `None`.
- `:118` `UnquantizedFusedMoEMethod.apply` → `self.forward(x=x, ...)`;
  `:156` `forward_cuda` → `fused_experts(...)`; `:218` `forward_native = forward_cuda`.
- `vllm/model_executor/custom_op.py:22-23` `forward` → `self._forward_method(...)`;
  `:60-63` `forward_oot` → `return self.forward_native(...)`. `_forward_method` is
  bound once, in `CustomOp.__init__`/`dispatch_forward`, at layer construction.

Downstream install (vllm-ascend `v0.7.1rc1`):

- `vllm_ascend/ops/fused_moe.py:170` (**last line** of the file):
  `UnquantizedFusedMoEMethod.forward_oot = forward_oot` — a class-attribute
  monkeypatch, applied when that module is imported.
- The **only** importer of `vllm_ascend.ops` in the whole package is
  `NPUPlatform.check_and_update_config` (`vllm_ascend/platform.py:100`,
  `from vllm_ascend import ops  # noqa: F401`, next to the
  `worker_cls = "vllm_ascend.worker.NPUWorker"` assignment).
- `check_and_update_config` has exactly one caller upstream:
  `vllm/config.py:3253` inside `VllmConfig`, and `VllmConfig` is constructed once,
  in `vllm/engine/arg_utils.py:1260` — i.e. **in the process that parses the CLI**.
- `vllm_ascend/worker.py` imports `vllm_ascend.patch` inside
  `init_worker_distributed_environment` (`:461`) but never `vllm_ascend.ops`;
  `vllm_ascend/model_runner.py` has no `vllm_ascend` import at all (checked by
  `rg -c 'from vllm_ascend|import vllm_ascend'`).

So the patch exists only where a `VllmConfig` was built. Workers in a **fresh
interpreter** — Ray actors, or `VLLM_WORKER_MULTIPROC_METHOD=spawn` — never run
the hook and keep upstream's `forward_oot`. The default start method is `fork`
(`vllm/envs.py:50`), so the default backend inherits the already-patched class
from the driver and the same deployment works; that asymmetry is why the report
looks backend-specific and is easy to misread as "DeepSeek BF16 unsupported".

## The discriminator: read the file name in the `forward_oot` frame

- Frame is `vllm/model_executor/custom_op.py:63 forward_oot` → the **upstream**
  body ran; downstream's patch was not installed in that process.
- If it had been installed, the frame would read
  `vllm_ascend/ops/fused_moe.py:<line> forward_oot`.
- The printed next frame is `layer.py:156 forward_cuda` even though the call came
  from `forward_native`, because `:218` aliases the two — do not read that as
  "it took the CUDA path on purpose".

Two-line check on the reporter's host, in the same kind of process that failed:

```
python -c "import vllm_ascend.worker, sys; print('vllm_ascend.ops' in sys.modules)"
python -c "from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod as M; import vllm_ascend.ops; print(M.forward_oot.__module__)"
```

The first prints `False` under Ray and `True` under the default fork executor; the
second prints `vllm_ascend.ops.fused_moe` once the import is forced.

## Fixes, labelled

- **Patch proposal (the real fix):** import `vllm_ascend.ops` at module scope of
  `vllm_ascend/worker.py` (or inside `init_worker_distributed_environment` beside
  the existing `from vllm_ascend import patch`), so every worker process installs
  it; report it upstream with the traceback above.
- **Configuration action (same day, no source edit):** if the model fits one node,
  drop `--distributed_executor_backend ray` and use the default fork executor — the
  patch is inherited. On two nodes Ray is required, so this is not available.
- **Not causes, do not pursue:** the checkpoint (BF16 checkpoints are the intended
  target of this very patch), the device, `--enforce-eager` (already set), and
  `is_cuda_alike()` itself.

## Version scope and the same shape elsewhere

- This page is the vllm 0.7.x mechanism: `forward_oot` **monkeypatched by an import
  side effect**. Later vllm dispatches through the name-keyed
  `CustomOp.register_oot` / `op_registry_oot`
  (`oot-class-registration-drift.md`), where the replacement is missed because the
  class *name* is not in the map, not because the import never ran. Before quoting
  either page, check which dispatch is live
  (`hasattr(CustomOp, 'op_registry_oot')`) and where the patch is installed.
- Generalise: a downstream monkeypatch installed as an import side effect of a
  **driver-only** hook is absent from any worker that gets a fresh interpreter.
  Ask which process ran the hook (fork / spawn / Ray actor) before believing any
  patch is live in the process that crashed.
- Version pinning trick that worked: the reporter's frames
  (`worker.py:226` `determine_num_available_blocks`, `model_runner.py:1139`,
  `custom_op.py:63`, `layer.py:156`) matched this commit exactly, which is how the
  installed pairing was confirmed without `pip show`. `SOURCES.json` for this case
  marks the ref `release_publication_unverified` (`version_role:
  dated_reference_only`), so a frame-line match is the evidence, not the issue's
  version string.
