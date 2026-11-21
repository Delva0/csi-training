# `'_OpNamespace' … has no attribute '<triton op>'` inside an inductor fusion pass

## Scope

Engine start dies while `torch.compile` runs vllm-ascend's own inductor passes —
the deepest frames are `vllm_ascend/compilation/passes/**` plus torch's pattern
matcher, and no model weight has been touched:

```text
  File ".../vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py", line 214, in __call__
    self.matched_count = self.pattern_match_passes.apply(graph)
  File ".../torch/_inductor/pattern_matcher.py", line 1982, in apply
  File ".../torch/_inductor/pattern_matcher.py", line 1520, in check_fn
    match.replacement_graph = trace_fn(replace_fn, args)
  File ".../torch/_inductor/pattern_matcher.py", line 2115, in fwd_only
    gm = make_fx(fn, decompositions, tracing_mode="real")(*args)
AttributeError: '_OpNamespace' 'vllm' object has no attribute 'qkv_rmsnorm_rope'
```

Recorded as **#6578** at vllm-ascend **v0.14.0rc1** (`52d4acfa`) + vllm **v0.14.1**
(`d7de043d`), Qwen3-32B on 910B, image built from the tree (both `Dockerfile` and
`Dockerfile.310p` at that ref uninstall triton). Re-grep per release: the pass list
and the gate move.

## Mechanism: registration is gated, its consumer is not

- The op exists only behind an optional import:
  `vllm_ascend/ops/__init__.py:25-26` — `if HAS_TRITON: import
  vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope`, and that module ends
  with `direct_register_custom_op(op_name="qkv_rmsnorm_rope", …,
  dispatch_key="PrivateUse1")` (`ops/triton/linearnorm/split_qkv_rmsnorm_rope.py:279-286`)
  → `torch.ops.vllm.qkv_rmsnorm_rope`.
- The consumer is ungated: `compilation/graph_fusion_pass_manager.py:57-60` appends
  `QKNormRopeFusionPass` whenever `ascend_compilation_config.fuse_qknorm_rope` is
  not `False` (default `True`, `ascend_config.py:192`), and
  `compilation/passes/qknorm_rope_fusion_pass.py:181-215` gates only on
  `dtype in (bf16, fp16)` and `layer.head_size == 128`. There is **no**
  `HAS_TRITON`/SoC check anywhere in the pass.
- The replacement is traced when the pass manager first runs
  (`__call__` → `pattern_match_passes.apply`), so **any** bf16/fp16 model with
  head_dim 128 fails on the first compile — the model does not have to contain a
  matching subgraph.
- `HAS_TRITON` is *not* "triton is installed": `vllm/triton_utils/importing.py:12-57`
  requires `find_spec("triton")` **and exactly one active** triton backend driver
  (0 or >1 → `HAS_TRITON = False`, `ImportError`/any exception in `triton.backends`
  likewise). "I have triton 3.6.0" therefore does not contradict this cause.

## One call confirms it

Grep the startup log for `Triton not installed or not compatible` (emitted by
`importing.py:69` whenever `HAS_TRITON` is false), or run
`python -c "from vllm.triton_utils import HAS_TRITON; print(HAS_TRITON)"` in the
serving env. `False` → this page. If the log instead shows
`Triton is installed but N active driver(s) found (expected 1). Disabling Triton…`
(`importing.py:47-52`), the package is present and the *driver check* is the gate —
that is the pairing question (`../image-dependency-faults/triton-ascend-pair.md`).

## Fix ladder

1. **Disable that one pass** — the project's own instruction for this environment:
   `--additional-config '{"ascend_compilation_config": {"fuse_qknorm_rope": false}}'`
   (`docs/source/user_guide/configuration/additional_config.md:79`: "If Triton is
   not in the environment, set it to False"). Metadata-only; no accuracy change.
2. **`--enforce-eager`** as the ablation: it prints
   `Compilation disabled, using eager mode by default` (`platform.py:235`) and skips
   the pass manager entirely, which proves the lane is graph compilation and not the
   model. On 310P this is also the only supported mode.
3. **Only if the fusion is wanted**: install a working triton/triton-ascend pair,
   confirm `python -c "import torch, vllm_ascend; print(torch.ops.vllm.qkv_rmsnorm_rope)"`
   resolves, then drop the override. Report the missing gate upstream:
   `QKNormRopeFusionPass.__init__` should register only `if HAS_TRITON`, the way
   `ops/__init__.py:25` imports the op.

## Two traps

- **`'_OpNamespace' … has no attribute 'X'` is not a claim about a `.so`.** The
  namespace object is created lazily on attribute access, so the message says only
  "this op was never registered in this process". It cannot prove `_C`/`_C_ascend`
  is absent, and it is not evidence about which wheel is installed.
- **A headline error naming another namespace is a second paste.** #6578's title
  says `qkv_rmsnorm_rope` while the body's first line is
  `AttributeError: '_OpNamespace' '_C_cache_ops' object has no attribute 'reshape_and_cache'`.
  `_C_cache_ops` has **zero** hits in vllm-ascend v0.14.0rc1 — the Ascend path is
  `torch_npu._npu_reshape_and_cache` (`vllm_ascend/device/device_op.py:26-27`) — and
  the name belongs to upstream's CUDA cache kernels (`vllm/_custom_ops.py:2374` at
  vllm v0.14.1), i.e. a different launch. Answer the traceback you can attribute and
  ask which command printed the other line; do not merge the two into one cause.
  When the hardware string is undefined shorthand ("910B A+X"), say so: if part of
  the box is a 300I/310P lane, that lane is eager-only and never runs this pass
  (`../soc-platform-paths/atlas-300i-310p.md`).

## Limits

- The override is documented, not reporter-confirmed on #6578; whether that run had
  `HAS_TRITON == False` is inferred from the mechanism, and the log grep above is
  the check.
- The same coupling can exist for other `if HAS_TRITON:`-registered ops
  (`vllm_ascend/sample/rejection_sampler.py`, `ops/rotary_embedding.py`,
  `patch/worker/__init__.py`, `spec_decode/eagle_proposer.py` all gate on it at this
  ref): when a pass or patched path calls one of them unguarded, the error text
  carries the op name and the fix is the same shape — find its registration gate.
