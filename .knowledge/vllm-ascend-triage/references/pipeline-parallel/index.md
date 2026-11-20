# Pipeline-parallel (PP > 1) deployments

Scope: the launch command carries `--pipeline-parallel-size 2` (or more) and the
symptom is PP-shaped — the engine comes up and the **first request** dies on a
non-last rank, a rank-specific `AssertionError`/`TypeError` with no device error,
or the question is "is PP validated on this model/hardware at all?". Not the lane
for a collective error on a PP cluster (that is `../hccl-and-hardware/index.md`)
and not the lane for one-sided PP sizing/卡数 arithmetic
(`../deploy-sizing/index.md`).

Inherit the rules in SKILL.md. Three that decide this topic:

1. **Read the docs' own PP statements before any code: they disagree, and the
   older one is the useful one.** One call —
   `rg -ni "pipeline.parallel" docs/` — returns, at `048bfd5` (v0.10.2rc1):
   `supported_features.md:22` `| Pipeline Parallel | 🟢 Functional | Write
   official guide and tutorial. |`, `release_notes.md:182` "Pipeline Parallelism
   is supported in V1 now (#1800)", and **`release_notes.md:359` (v0.9.2rc1
   Known Issues) "Pipeline parallel does not work with ray and graph mode:
   #1751 #1754"**. There is *no* PP tutorial page, so the usual first move of
   this skill (diff the reporter's command against the matching tutorial) has
   nothing to compare with — and the known-issue line names the same axis the
   reported failure lands on. Quote it with its release, do not present the
   🟢 row as the project's validated position for graph mode.
2. **CI coverage is one grep and it is part of the answer.** The e2e test exists
   (`tests/e2e/multicard/test_pipeline_parallel.py`: `Qwen/Qwen3-0.6B`, tp 1,
   pp 2, `mp`+`ray`, `VllmRunner` whose `enforce_eager` default is `False` —
   `tests/e2e/conftest.py:96`) but **no workflow invokes it** at `048bfd5`:
   `rg -n "test_pipeline_parallel" .github/workflows/` → no hit, while
   `vllm_ascend_test_full.yaml:213-221` lists the other multicard files. So "PP is
   🟢 Functional" is a documentation claim, not a CI-backed one — quote both.
3. **A non-last-rank failure is an interface question between the runner and the
   worker, not a model or checkpoint question.** `worker_v1.py` sends
   `output.tensors` when `not is_last_rank` (`:217-222`), so whatever came back
   from `model_runner.execute_model` must still be an `IntermediateTensors`. When
   the paste proves it is not, the owner is the code that returned it, and the
   model file is exonerated (upstream's own body returns `IntermediateTensors` for
   a non-last rank — e.g. `vllm/model_executor/models/qwen2.py:359-365`).

Conditional routes:

- First request on a non-last PP rank dies with a bare `AssertionError`
  (`assert isinstance(output, IntermediateTensors)`) immediately after
  `acl_graph.py … Replaying aclgraph`, while the engine started cleanly → the
  graph-mode × PP output contract → `pp-nonlast-rank-output-contract.md` (rung 1
  is a one-flag ablation, not a source walk).
- `AssertionError`/`TypeError` raised from an upstream graph wrapper or
  compilation backend that the downstream `compilation/` fork replaced → also
  `../upstream-downstream-drift/index.md` for the drift pattern (fork copied from
  an older upstream design, so the upstream file at the pinned vllm is not the
  reference implementation).
- The reporter asks whether model M supports PP: `vllm/config.py` raises
  `NotImplementedError: Pipeline parallelism is not supported for this model.
  Supported models implement the SupportsPP interface.` before any worker starts
  (vllm `b6553be`, `:1063-1069`); absence of that line in a startup log means the
  model passed the gate and PP support is not the issue.

Limits: no PP e2e run is reproducible from this snapshot (single-commit export,
no NPU devices), so every PP verdict here is source-verified at the pinned commit
plus the reporter's own ablations. State that instead of implying a measured fix.
