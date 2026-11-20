# Worked case #9335: 2-NPU stage deploy of Qwen3-Omni-30B-A3B on 910B

Report: vllm-ascend alone serves text models with `tensor_parallel_size=2` on the
same two 910B cards; a **vllm-omni** stage-based deploy (stage 0 = thinker, stages
1+2 = talker/code2wav) works from the GPU docs' 2-device example but not on NPU;
the bundled NPU yaml ships a 3-card layout (`stage 0: tp 2, devices "0,1"`,
`stage 1: devices "2"`, `stage 2: devices "2"`). Pinned: vllm-ascend v0.19.1rc1
(`da421af`), vllm v0.19.1; vllm-omni is **absent** from the snapshot.

## The arithmetic that decides it

```bash
bash <skill_root>/scripts/card-budget.sh --params 30.5 --tp 2 --gmu 0.9 --cards 2
bash <skill_root>/scripts/card-budget.sh --params 30.5 --cards 2 \
     --stage-gmu 0.9,0.6,0.3 --stage-devices '0,1;0;0'   # the 2-card squeeze
```

- ~30.5B params at 2 B/param ≈ **61 GB** for one full copy: larger than the
  **57.6 GB** a single 64 GB card may use at `gpu_memory_utilization 0.9`, so
  stage 0 cannot be TP=1 on 910B. It is TP=2 (both cards) — which is what the
  project's own tutorial documents (`docs/source/tutorials/models/
  Qwen3-Omni-30B-A3B-Thinking.md:19` "requires 2 NPU Cards (64G × 2)";
  `:191` `vllm serve … --tensor-parallel-size 2 --enable_expert_parallel`).
- Stage 0 at 0.9 on both cards plus stages 1 and 2 at 0.6/0.3 on either card
  gives a per-device `gpu_memory_utilization` sum of 1.8: infeasible by
  construction. That is why the shipped yaml has a third device, not an
  oversight in the yaml.
- A CI data point on the same family, not a contradiction: the Instruct variant's
  e2e config runs `tensor_parallel_size: 4`, `gpu_memory_utilization: 0.7`,
  `max_model_len: 8192`, `enable_expert_parallel: true`
  (`tests/e2e/models/configs/Qwen3-Omni-30B-A3B-Instruct.yaml`); scaling TP out
  is the project's own answer to the same wall.

## The second defect in the reporter's commands

`CUDA_VISIBLE_DEVICES=0 vllm serve …` on an NPU host is inert: the control
variable is `ASCEND_RT_VISIBLE_DEVICES` (`vllm_ascend/platform.py:101`,
v0.19.1rc1; `CUDA_VISIBLE_DEVICES` appears in no file under `docs/source/`, while
multi-node scripts export `ASCEND_RT_VISIBLE_DEVICES`) — so each stage the
reporter launched saw every card, and `devices:` in the yaml was the only thing
pinning them. Precedent, from the issue titles: `#9336` requests exactly this
NPU device-allocation documentation, including *which* visibility variable to use.

## Answer shape that fits this class

1. Direct answer: the 2-NPU three-stage layout is memory-infeasible on 64 GB
   cards, and `CUDA_VISIBLE_DEVICES` was inert; neither is an Ascend defect.
2. The two numbers (61 GB per copy vs 57.6 GB per card; device gmu sum 1.8) with
   the script line that reproduces them.
3. Ranked options, cheapest first: keep 3 cards; drop the speech stages and serve
   the thinker only at TP=2 (supported and documented); quantise so stage 0 fits
   one card; or 2 cards with stages 1+2 sharing the card the thinker leaves
   room on — only after the arithmetic says the sum is ≤1.0.
4. The gap: vllm-omni is not in this snapshot, so `platforms.npu` merging,
   `--stage-id/--headless`, and how `devices:` maps to
   `ASCEND_RT_VISIBLE_DEVICES` cannot be verified here — send
   `qwen3_omni_moe.yaml`, both stage commands and the first failing log lines.
5. Confidence labels: doc/CI lines are source-verified at the pinned commits;
   the GPU-side layout details are hypothesis (the GPU yaml is not in the
   snapshot).

## Limits

- The fit check ignores activations, KV and fragmentation; TP=2 fitting at 0.9
  is not a promise that the KV pool is adequate at a long `--max-model-len`.
- "Feasible" is about memory, not scheduling: a layout that fits can still fail
  in the orchestrator, which is exactly the part that cannot be read here.
