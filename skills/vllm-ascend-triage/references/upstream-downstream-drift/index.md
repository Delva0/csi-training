# Upstream / downstream drift in patched code

## Scope

Python-level failures raised inside a vllm-ascend module that subclasses or
re-implements an upstream vLLM or transformers class: `AttributeError: '<Ascend…>'
object has no attribute '<x>'`, `TypeError: ... unexpected keyword argument`,
`ImportError: cannot import name ...`, a signature or return-shape mismatch, or a
forward body that no longer matches the library it mirrors.

Inherit the rules in SKILL.md. This topic is the right entry when:

- the failing frame is in `vllm_ascend/ops/`, `vllm_ascend/patch/`,
  `vllm_ascend/models/` or `vllm_ascend/attention/`
  (collectively `vllm_ascend/{ops,patch,models}` below);
- **or the failing frame is an upstream vllm file whose class vllm-ascend is
  supposed to replace** — either because the Ascend implementation is registered
  by class *name* (`CustomOp.register_oot` / `REGISTERED_ASCEND_OPS`), so an
  upstream refactor that adds or renames a class leaves that class running
  upstream code, or because the patch is installed as an **import side effect**
  that never ran in that process. The owner is still downstream in both cases; do
  not answer "upstream bug" (recorded #6784, #253);
- the named class subclasses a `transformers`/`vllm` class or is registered as a
  drop-in replacement for one (`PluggableLayer`, `ModelRegistry.register_model`,
  monkey-patched `forward`);
- or the upstream frame is a **generic consumer** of `static_forward_context` (a
  connector/spec filtering layers by an `isinstance` check) that fails on the class
  hierarchy itself: `KeyError` on a `<layer>.self_attn.attn` key at
  `offloading_connector.py register_kv_caches` while an `NPUOffloadingSpec` is being
  created, asked as "MLA 没适配吗 / vllm 还是 vllm-ascend 的问题" →
  `kv-offloading-layer-coverage.md`. This is the one shape where "upstream's own bug"
  is the right answer, and the discriminator is the two builders: when upstream's own
  wrapper registers the same class at the same prefix, the `isinstance` filter is
  upstream's; when only the downstream replacement creates the failing object, the
  bullet above stands;
- or the report blames an upstream **kernel/backend selector flag** for an NPU
  failure and the pasted line is not a traceback: `--<x>-backend auto` "chose
  flashinfer", a quoted layout error that greps to nothing in either tree, a
  `UserWarning` from `vllm_ascend/ops/triton/fla/chunk.py` on the path the reporter
  thought was the fix, and "prefill fixed, decode still on flashinfer" → grep the
  quoted fragment in **both** trees first, then check whether `patch/**` rebinds the
  method that reads the flag: upstream decides the flag, downstream decides the
  kernel → `dead-selector-flag.md` (recorded #15206, GDN prefill backend on 910B);
- a downstream **fork of a whole upstream file** lags the upstream *state machine*
  rather than its signatures — `vllm_ascend/core/scheduler.py` (the `AscendScheduler`
  fork of `vllm/v1/core/sched/scheduler.py`) rejects a request status upstream added
  (`RuntimeError: Invalid request status: WAITING_FOR_FSM`),
  `scheduler-fork-missing-status.md` (the `npu_input_batch.py` copy below is the
  other direction: the copy grew a branch upstream does not have);
- a downstream **verbatim copy** of an upstream worker module has grown an extra
  branch the upstream file does not have, and that branch is what fails — e.g.
  `vllm_ascend/worker/npu_input_batch.py` at v0.11.0 (a copy of
  `vllm/v1/worker/gpu_input_batch.py`) with an `is_pooling_model` fast path in
  `add_request`/`remove_request`/`condense`: pooling/embedding crash under
  concurrency, `../pooling-embedding/batch-index-desync.md`. Same method as
  `patch-copy-vs-upstream.md`, different directory, and the diff is what proves
  ownership;
- the error is a Python attribute/type/state error (including a `RuntimeError`
  raised by a fork's own state handling), not a device dump, not a collective.

The one `NotImplementedError` that belongs to startup instead: a **custom op**
registered by the downstream C++ extension without a Meta/fake implementation
(`_C::rotary_embedding` at v0.9.2rc1, `_C_ascend::<op>` after the rename) fails the
compile-phase fake-tensor trace and is answered by the registration and the dtype
gate, not by a class drift — `../startup-failures/meta-kernel-missing.md`.

It is the wrong entry for an aclnn/AscendC device fault
(`../custom-ops-runtime-faults/index.md`), for a collective or `ERR00100`
(`../hccl-and-hardware/index.md`), for "model type not recognised"
(`../startup-failures/model-type-not-recognized.md`), and for a raise that never
touches downstream code at all — a traceback over `vllm/**` plus
`site-packages/**` with no Ascend class involved (`RuntimeError: Already
borrowed` in a fast tokenizer, a thread race in the OpenAI frontend) is
`../serving-frontend-faults/index.md`.

## Rules

1. **Split the question in two and answer the first one at once.** (a) Does the
   object the reporter ran actually lack the attribute/signature? A reporter
   traceback settles this — that is a reporter-confirmed fact, no further
   evidence needed; say "yes, the crash exists on that image" straight away.
   (b) *Why* it lacks it — the drift analysis below. Do not hold the answer to
   (a) hostage to (b), and do not treat "the upstream devs tested it" as
   evidence against (a).
2. **Downstream owns the drift when upstream delegates and downstream
   re-implements.** Compare the two call sites at their pinned commits. If the
   upstream class calls `super().forward(...)` and lets the installed library run
   its own body, upstream can never disagree with the library; a downstream copy
   of that body can, and that copy is the bug owner. Record both line numbers.
3. **Do not date a library change from a truncated export.** Large third-party
   repos (`huggingface/transformers` and friends) arrive as `content_export`
   with a small fraction of their files; the per-model source is usually the
   first casualty.
   Never write "transformers removed X in 5.x" from such an export. State the pin
   you do have (`requirements.txt` line) and phrase the version claim as an open
   question with the check that settles it.
4. **Mine the dependency's docs and utils when its source is gone.** Content
   exports retain `docs/`, `utils/` and tests far more often than model code, and
   library modeling-rule docs state the *sanctioned* pattern. In the recorded
   case the answer came from `docs/source/en/modeling_rules.md`, not from the
   missing `modeling_qwen2.py`.
5. **Give the reporter both halves**: the one-line check that proves the drift
   in their image, and the cheap image/version action. Repairing a downstream
   re-implementation is a patch proposal; a version change is a same-day fix.
   Label which is which.
6. **State the invalidating condition.** A drift claim is bound to the
   vllm-ascend commit, the dependency pin and the image tag. If the reporter is
   on a different image than the one named in the issue, the claim has to be
   re-derived.
7. **Decide which layer the live function object belongs to, not which layer the
   text names.** Three different keying schemes produce the same symptom ("the
   Ascend op was not used"): a class-*name* registry miss (later vllm), a
   monkeypatch installed by an import side effect in a process that never ran the
   hook (vllm 0.7.x, and any driver-only hook with `spawn`/Ray workers), and a
   patch that ran in that process but rebound only the *defining* module while the
   consumer module had already cached the name in its own namespace (vllm 0.26's
   `FusedMoE` factory). The traceback's **file name** at the
   `forward_oot`/`apply` frame settles the first two — an upstream `custom_op.py`
   frame means the downstream replacement is absent from that process
   (`patch-import-side-effect.md`). For the third, the file name is upstream *and*
   the patch is installed: compare the defining module's attribute with the
   consumer module's (`qwen3_moe.FusedMoE is not patch._ascend_FusedMoE`) and see
   `stale-module-binding.md`. Check `hasattr(CustomOp, 'op_registry_oot')` before
   quoting the name-keyed page.
   A fourth form raises nothing at all: the downstream shim
   `<Class>.forward_oot = …` is **unreachable** because the upstream class defines
   its own `forward`, so `CustomOp.dispatch_forward` never runs. At v0.9.0rc2,
   `vllm_ascend/ops/rotary_embedding.py:273` binds
   `RotaryEmbedding.forward_oot = rope_forward_oot`, while
   `MRotaryEmbedding.forward` (`vllm/model_executor/layers/rotary_embedding.py:1008`
   at vllm v0.9.0) bypasses it — mRoPE keeps upstream's native body. Two greps
   settle it (`grep -n 'class <X>\|def forward' <upstream file>`); run them before
   repeating anyone's claim that a flag "gates the fused op", including a
   code-change request's own premise
   (`../code-change-requests/env-flag-retirement.md`, step 4).
8. **A `patch/**` frame is a copy, so attribute it with a diff.** Every module
   under `vllm_ascend/patch/` assigns a verbatim upstream method onto an upstream
   class under a `vllm_version_is('<ver>')` gate, so the innermost traceback frame
   names vllm-ascend while the failing statement may be upstream text. Extract the
   class-level definition from the pinned upstream file (dedent by 4), diff it
   against the module-level copy, and classify the hunks: formatting-only means
   the owner is upstream at that pin, and only an *added statement* can be
   vllm-ascend's — then check whether that statement is even live for the
   reporter's command (PD `is_kv_consumer`, `use_async_scheduling`, an
   `additional_config` key). Doing this first is 2-3 calls; the recorded #8724 run
   derived the same conclusion from proxies over 29 turns and shipped nothing
   (`patch-copy-vs-upstream.md`).
9. **A traceback file that is absent from the pinned upstream tree is a snapshot
   fact, not an upstream bug.** Say what it proves (the image ships a different
   upstream ref), get the real pairing from the project's compatibility matrix
   (`docs/source/community/versioning_policy.md`) and CI `vllm_version` matrix,
   and ask for `pip show vllm`. Do not keep grepping the pinned tree for the
   missing line — that loop cost a recorded run its whole session.

10. **A config flag is not the capability test, and a stack through the graph
   wrapper is not attribution.** The downstream fork replaced upstream's
   capability predicate with a model-level flag in its dummy/capture path:
   `vllm_ascend/worker/model_runner_v1.py:2265,2431` (v0.18.0 `e18643f`) branches on
   `self.is_multimodal_model` where upstream `gpu_model_runner.py:5152` uses
   `self.supports_mm_inputs` + `_prepare_mm_inputs()` + `**_dummy_mm_kwargs(...)`.
   Upstream `is_multimodal_model` is only `multimodal_config is not None`
   (`vllm/config/model.py:1424`), true for every VL-capable architecture even with
   `language_model_only=True`, while `multimodal_registry.supports_multimodal_inputs`
   (`vllm/multimodal/registry.py:103-111`) is false in text-only mode. The
   observable is `AttributeError: 'NoneType' object has no attribute 'size'` at
   `torch/_dynamo/utils.py::call_size`, reached through `compilation/acl_graph.py` —
   the graph wrapper is the call chain, not the owner. Before theorising, check
   whether the *release branch* simply lacks a main-only fix (recorded #14638:
   fixed on main by PR #8556, not merged into the v0.18.0 branch, backport PR
   #14618): `rg -n '<flag>' <module>` in the image's tree answers "predates the
   fix" in one call, and the answer is a backport request, not a code walk.
   Behaviour page: `../accuracy-benchmarks/first-generate-degenerate.md`.

11. **An attribute assigned unconditionally in the upstream `__init__` cannot be
   "missing" — the constructor was interrupted.** When the message is
   `AttributeError: '<Ascend…>' object has no attribute '<x>'`, downstream calls
   `super().__init__()` and the pinned upstream `__init__` binds `<x>` on every
   path (even to `None`), do not look for a rename or a capability gap: check the
   `with <wrapper>():` around the `super()` call for a `@contextmanager` whose
   `try:` encloses the `yield` and whose `except Exception:` returns — that
   suppresses the first exception and leaves the half-built object. The message
   itself separates the two neighbours: a mere `None` gives `AssertionError` from
   the next `assert`, not `AttributeError`. Two greps decide it; the reporter's
   fix is a `raise` in that branch to recover the real error
   (`swallowed-init-exception.md`, recorded #5571 `mm_budget`).

## Pages

- `kv-offloading-layer-coverage.md` — worked case vllm-ascend #7971 (v0.17.0rc1,
  GLM-5 / Kimi-K2.5 with `OffloadingConnector`): `KeyError:
  'language_model.model.layers.0.self_attn.attn'` because
  `get_layers_from_vllm_config(vllm_config, Attention, …)` drops `MLAAttention`
  (a sibling of `Attention` under `AttentionLayerBase`) and the connector indexes
  the empty dict; the model→class table, the three greps that put ownership on
  upstream's filter (`vllm/.../layers/mla.py:95` builds the same `.attn` layer),
  and the downstream handler's hardcoded K/V unpacking that makes a filter-only
  patch silently corrupt MLA caches.
- `patched-forward-api-drift.md` — worked case vllm-ascend #11661 (v0.22.1rc1,
  DeepSeek-OCR-2): `AscendQwen2DecoderLayer` has no `attention_type`, the
  evidence table, the decisive check, and what the report still leaves open.
- `stale-module-binding.md` — worked case vllm-ascend #15815 (v0.26.0rc1,
  Qwen3-Coder-30B-A3B TP4): `AttributeError: 'UnquantizedFusedMoEMethod' object
  has no attribute 'is_monolithic'. Did you mean: 'apply_monolithic'?` inside
  upstream `moe_runner.py`, because `patch/platform/patch_fused_moe.py:87-88`
  rebinds only the layer and package modules while `qwen3_moe.py:45` cached the
  factory first; why the stale path is a guaranteed crash (`OOT, None` ->
  `self.experts_cls.is_monolithic()` masked by `nn.Module.__getattr__`), the
  three-line discriminator against the two pages above, the 15 upstream modules
  with the same hazard, and the `sys.modules` sweep plus its in-tree precedent.
- `patch-import-side-effect.md` — worked case vllm-ascend #253 (v0.7.1rc1,
  DeepSeek-R1-BF16 under `--distributed_executor_backend ray`):
  `TypeError: 'NoneType' object is not callable` at upstream
  `fused_moe/layer.py:156 forward_cuda`, because `vllm_ascend/ops/fused_moe.py`
  installs its MoE replacement by class monkeypatch and the only importer of that
  module is `NPUPlatform.check_and_update_config` — which runs once, where
  `VllmConfig` is built. Ray ranks and `spawn` workers keep upstream's
  `forward_oot`; the default `fork` inherits the patched class. Includes the
  frame-file discriminator and the two-line host check.
- `scheduler-fork-missing-status.md` — worked case vllm-ascend #3979
  (v0.11.0rc0 `00ba071` / vllm v0.11.0rc3 `8ce5d31`): a `guided_choice` request
  raises `RuntimeError: Invalid request status: WAITING_FOR_FSM` at
  `vllm_ascend/core/scheduler.py:274` and kills EngineCore. The fork's status
  whitelist (`:269-274`) never learned upstream's FSM state
  (`vllm/v1/request.py:75`, gate at `vllm/v1/core/sched/scheduler.py:357-366`),
  the fix has three sites because the fork also hardwires
  `structured_output_request_ids={}` / `grammar_bitmask=None` (`:475-476`), and
  the first move is a grep on `platform.py`: rc0 **force-enables** the fork for a
  default dense serve (`platform.py:139-150`, so an explicit `enabled: false` is
  overwritten) and the block is gone by `v0.11.0rc1`, which flips the escape
  between a gate-input flag and dropping the key. Includes the method-set diff
  recipe, the CI-coverage gap, the shutdown `nanobind` noise, and the recheck
  command.
- `patch-copy-vs-upstream.md` — the `patch/**` frame: the extract-and-diff recipe
  with exact line ranges, the formatting-vs-added-statement classification, the
  version gate that binds a copy to one upstream release, and worked #8724
  (v0.13.0rc3/v0.13.0): the patch is upstream `gpu_model_runner.py:753-1018`
  re-indented plus one `is_kv_consumer`-gated hunk the reporter's non-PD command
  never runs, so the `KeyError` at `patch_model_runner.py:154` is upstream's — plus
  the one-provenance reading of its three tracebacks.
- `oot-class-registration-drift.md` — worked case vllm-ascend #6784
  (v0.14.0rc1-a3-openeuler, LongCat-Flash): `'ZeroExpertFusedMoE' object has no
  attribute 'select_experts'`, raised inside an upstream file. The name-keyed
  OOT registry, why the missing upstream file is version evidence, where the
  vllm↔vllm-ascend pair is actually recorded, and the checks that settle it.
- `swallowed-init-exception.md` — worked case vllm-ascend #5571
  (v0.13.0rc1 `1b5d5ab` / vllm v0.13.0 `72506c9`, multimodal): upstream binds
  `self.mm_budget` unconditionally at `gpu_model_runner.py:565-573` and the
  `profile_run` read at `:4434` raises `AttributeError`, because
  `_torch_cuda_wrapper` (`model_runner_v1.py:3453-3488`) wraps the
  `super().__init__()` call (`:191-193`) in a `try/except Exception` around its
  `yield` and contextlib suppresses the exception. The 6-line proof that such a
  wrapper swallows, the `MultiModalBudget` construction as the prime raise site,
  `--skip-mm-profiling` as a smoke test that is not a fix, and the v2
  `try/finally` wrapper as the discriminator.
- `unsupported-path-raise.md` — a downstream backend's own `raise`
  (`RuntimeError: Prefix cache and chunked prefill are currently not supported`, beside
  `# TODO: Will support … soon.`) reached from an RL trainer that builds `vllm.LLM(...)`
  itself: the metadata state the branch refuses (any prefill batch carrying a
  `block_tables` tensor), the V0 defaults that prove a caller enabled it, the multimodal
  auto-disable row, and the "fixes" that cannot work. Recorded #133 (v0.7.1rc1).
