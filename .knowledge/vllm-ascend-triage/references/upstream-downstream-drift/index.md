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
  supposed to replace** — the Ascend implementation is registered by class
  *name* (`CustomOp.register_oot` / `REGISTERED_ASCEND_OPS`), so an upstream
  refactor that adds or renames a class leaves that class running upstream code.
  The owner is still downstream; do not answer "upstream bug" (recorded #6784);
- the named class subclasses a `transformers`/`vllm` class or is registered as a
  drop-in replacement for one (`PluggableLayer`, `ModelRegistry.register_model`,
  monkey-patched `forward`);
- the error is a Python attribute/type error, not a device dump, not a collective.

It is the wrong entry for an aclnn/AscendC device fault
(`../custom-ops-runtime-faults/index.md`), for a collective or `ERR00100`
(`../hccl-and-hardware/index.md`), and for "model type not recognised"
(`../startup-failures/model-type-not-recognized.md`).

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
7. **A traceback file that is absent from the pinned upstream tree is a snapshot
   fact, not an upstream bug.** Say what it proves (the image ships a different
   upstream ref), get the real pairing from the project's compatibility matrix
   (`docs/source/community/versioning_policy.md`) and CI `vllm_version` matrix,
   and ask for `pip show vllm`. Do not keep grepping the pinned tree for the
   missing line — that loop cost a recorded run its whole budget.

## Pages

- `patched-forward-api-drift.md` — worked case vllm-ascend #11661 (v0.22.1rc1,
  DeepSeek-OCR-2): `AscendQwen2DecoderLayer` has no `attention_type`, the
  evidence table, the decisive check, and what the report still leaves open.
- `oot-class-registration-drift.md` — worked case vllm-ascend #6784
  (v0.14.0rc1-a3-openeuler, LongCat-Flash): `'ZeroExpertFusedMoE' object has no
  attribute 'select_experts'`, raised inside an upstream file. The name-keyed
  OOT registry, why the missing upstream file is version evidence, where the
  vllm↔vllm-ascend pair is actually recorded, and the checks that settle it.
