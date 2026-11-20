# Deployment sizing: "can this run on N cards?" and "why does my stage layout fail?"

Scope: the report is a *plan*, not a crash — "the GPU docs do a 2-device
stage-based deploy, mine does not work on 2 NPUs", "the bundled `platforms.npu`
yaml has three stages and I only have two cards", "does model M fit on 2×64 GB",
"how do I pin stage 1 to one card". Nothing has crashed yet, or the crash text is
an allocation failure that the layout predicts.

Leave this topic when there is a real startup traceback with a lane
(`../startup-failures/index.md`), a collective/device error
(`../hccl-and-hardware/index.md`), or a long-run memory report
(`../runtime-memory/index.md`); when the question is a latency/throughput claim
(`../accuracy-benchmarks/index.md`); or when the deliverable is a report about a
tutorial page (`../tutorial-verification/index.md`).

Inherit the rules in SKILL.md, plus these:

1. **Answer with arithmetic, then options.** One engine's weights per card
   (`params × bytes-per-param ÷ TP`) against the HBM that engine may use
   (`cards' HBM × gpu_memory_utilization`), and the sum of
   `gpu_memory_utilization` over every process that shares a device. Report both
   numbers; reproduce them with
   `bash <skill_root>/scripts/card-budget.sh` instead of by eye. A sum above
   1.0 on any device is **infeasible by construction** and is not a scheduler,
   connector or device-visibility bug.
2. **The device-visibility env var is the second half of every such report.**
   On Ascend the control variable is `ASCEND_RT_VISIBLE_DEVICES`
   (`vllm_ascend/platform.py:101` at v0.19.1rc1; every multi-node launch script
   in the same docs tree does `export ASCEND_RT_VISIBLE_DEVICES=$1`, e.g.
   `GLM5.md`, `DeepSeek-V3.1.md`, and `CUDA_VISIBLE_DEVICES` occurs nowhere in
   `docs/source/`). A command copied from a GPU page therefore exposes **all**
   cards to **every** process, which looks exactly like a stage-collision bug.
3. **A sibling project's config that the snapshot does not carry is a gap, not a
   hypothesis.** vllm-omni's stage CLI, `qwen3_omni_moe.yaml` and the
   `platforms.<vendor>` merge are not in `SOURCES.json`; you cannot verify their
   parsing. Say what is unverifiable, answer what the arithmetic and the pinned
   vllm-ascend docs do settle, and ask for the yaml plus the failing log.
   Precedent: #9336 files this exact absence as a `[Doc]` gap ("Missing NPU
   device allocation strategy for multi-NPU deployments") and asks for the NPU
   equivalent of `CUDA_VISIBLE_DEVICES` — so "the docs do not answer this" is a
   publishable finding.
4. **Quantify the options you offer** (more cards / quantised weights / drop a
   stage / pin devices) with the same script, and state the limit: the fit check
   excludes activations, KV growth and fragmentation, so a PASS is "not yet
   excluded" while a FAIL is decisive.

Conditional routes:

- Two serve commands that differ by one resource knob (`--tensor-parallel-size`,
  `--max-model-len`, `--gpu-memory-utilization`) where the smaller one comes up and
  the larger OOMs at engine start → `tp-scaling-oom-at-startup.md` first: the
  tutorial's validated serve line, the OOM line split against `gmu × total`
  (`../../scripts/oom-budget-split.sh`), the per-rank terms that do not shrink with TP,
  and the stop rule against an allocator-mechanism hunt (#6572).
- The worked #9335 case: the 30B-A3B Omni model on 2×64 GB 910B, the three-stage
  NPU yaml, why the 2-NPU squeeze is infeasible, the evidence table and the
  answer shape → `card-budget-and-device-visibility.md`.
- The model's own documented envelope and CI config (which TP the project
  actually runs) → `../startup-failures/max-model-len-budget.md` and
  `../tutorial-verification/tutorial-command-audit.md` (checklist item 5).
- The layout is right and the process still dies during profiling/load →
  `../runtime-memory/index.md` (capacity vs growth vs a dead config key).
