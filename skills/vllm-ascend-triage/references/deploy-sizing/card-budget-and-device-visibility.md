# Omni stage deploys on N NPUs: the stage yaml's `devices`, the visibility variable, the card budget

Two reports meet the same layout — one as a plan, one as a startup crash:

- **#9335**: the 30B-A3B Omni model in a three-stage vllm-omni deploy
  (stage 0 = thinker, stages 1+2 = talker/code2wav) on 2×64 GB 910B.
  vllm-ascend alone serves text models with `tensor_parallel_size=2` on the same
  cards; the stage deploy works from the GPU docs' 2-device example but not on
  NPU. Pinned vllm-ascend v0.19.1rc1 (`da421af`), vllm v0.19.1.
- **#8068**: the same shape as a crash — `EngineCore failed to start` with
  `aclInit, error code is 107001 … Invalid device ID` in the first worker of
  `vllm serve <Qwen3-Omni-30B-A3B-Instruct> --omni --tensor-parallel-size 2`
  (2×910B3, vllm 0.18.0 / vllm-ascend 0.18.0rc1 / vllm-omni 0.18.0), no
  `Starting to load model`. **Reporter-confirmed fixed on the same two cards.**

A **310P** omni question is not this page: no `-310p` omni image exists, and the
source build plus the per-stage `platforms.npu` block are
`../soc-platform-paths/310p-omni-model-deploy.md`.

## The reporter-confirmed recovery, before any card arithmetic

Recorded on #8068 from the original reporter after the triage reply. Three steps,
no new hardware:

1. Pin the visible set with the **runtime** variable, ascending, indexing the
   driver's device list: `export ASCEND_RT_VISIBLE_DEVICES=0,1` (why `0,1` and
   not the `npu-smi` labels: next section).
2. Copy the bundled NPU stage config out of the installed vllm-omni and edit the
   `devices` entry of **every** stage so the ids stay inside that set:
   `vllm_omni/platforms/npu/stage_configs/qwen3_omni_moe.yaml` →
   `/path/to/stage_config.yaml`.
3. Start with the override flag — it is `--stage-configs-path`, not
   `--deploy-config` — and note the working line carries **no**
   `--tensor-parallel-size`, because per-stage TP and devices come from the yaml:
   `vllm serve <model dir> --omni --stage-configs-path /path/to/stage_config.yaml`

The omni launch log already names the values to edit, before anything loads:
`stage_init_utils.py:222 … [stage_init] Stage-0 set runtime devices: 0,1`,
`Stage-1 … : 2`, `Stage-2 … : 2` (verbatim in #8068). Those ids are the stage
config's own `devices` values, so the log says the shipped default gives stages 1
and 2 a **third** device — on a two-card host that id is outside the visible set,
and the launch dies at the first device open
(`vllm_ascend/worker/worker.py:109 check_ascend_device_type` →
`utils.py:723 get_soc_version` → `aclInit 107001`), before weights, KV, capture or
any request. Editing the yaml is therefore the fix, not "get a third card".

Still unverified, because vllm-omni is in no snapshot: the merge order of
`platforms.<vendor>` with the stage yaml, whether `devices` are logical or
physical ids, and whether two stages may share one id. Say so, and ask for the
edited yaml plus each stage's own log line whenever an attempt disagrees.

## The visibility variable, and the three traps in it

The control variable on Ascend is `ASCEND_RT_VISIBLE_DEVICES`
(`vllm_ascend/platform.py:101` at v0.19.1rc1; every multi-node launch script in
the same docs tree exports it — `GLM5.md`, `DeepSeek-V3.1.md` — while
`CUDA_VISIBLE_DEVICES` occurs nowhere in `docs/source/`). A command copied from a
GPU page therefore exposes **all** cards to **every** process, which looks exactly
like a stage-collision bug.

- **It has no alias.** #8068's env dump exported `ASCEND_VISIBLE_DEVICES=2,1`;
  that spelling appears nowhere in the `Ascend/pytorch` Python tree nor in
  `cann/runtime`, and `_parse_visible_devices()` reads
  `ASCEND_RT_VISIBLE_DEVICES` only (`torch_npu/npu/__init__.py:341`, the same
  line at both `v7.1.RC1.alpha002-pytorch2.7.1` and
  `v7.2.0.1-pytorch2.8.0`). A reporter who "set the visible devices" under that
  name has set nothing.
- **The list is ascending, or it is empty.** `_device_count_ascend_hal()`
  returns **0** devices as soon as an entry is not greater than its predecessor
  (`torch_npu/npu/__init__.py:375-398` at the 2.7 pin; `:386-409` at the 2.8 pin,
  comment `` `rts` need ascending order``), so `2,1` is not a narrower view, it is
  no devices at all.
- **The ids index the driver's device list, not `npu-smi`'s NPU labels.** The same
  function trims at the first entry `>= drvGetDevNum()` (raw count at `:369-383`,
  trim at `:392-393` in the 2.7 pin / `:403-404` in the 2.8 pin), so the
  reference is `ls /dev/davinci*` (0…N-1), not the host labels `npu-smi` prints.
  #8068 shows both halves: `npu-smi` listed NPUs **1** and **2** while the working
  export was `0,1`, i.e. the container's `davinci0,1` are those two cards.

One line settles this lane before any vLLM flag is audited:
`python -c "import torch, torch_npu; print(torch_npu.npu.device_count(), torch_npu.npu.get_soc_version())"`
— a `107001` there ends the vllm-ascend part of the case
(`../startup-failures/device-id-invalid-at-worker-init.md`).

## Then the arithmetic: does the layout you just wrote fit?

```bash
bash <skill_root>/scripts/card-budget.sh --params 30.5 --tp 2 --gmu 0.9 --cards 2
bash <skill_root>/scripts/card-budget.sh --params 30.5 --cards 2 \
     --stage-gmu <stage0>,<stage1>,<stage2> --stage-devices '0,1;0;1'
```

- ~30.5B params at 2 B/param ≈ **61 GB** for one full copy, against the
  **57.6 GB** a 64 GB card may use at `gpu_memory_utilization 0.9`: the thinker
  stage cannot be TP=1 on 910B, it is TP=2 on both cards — which the project's own
  tutorial documents (`docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md:19`
  "requires 2 NPU Cards (64G × 2)", `:191` `vllm serve … --tensor-parallel-size 2
  --enable_expert_parallel`).
- The feasibility test is the **sum of `gpu_memory_utilization` over every stage
  that shares one device**: above 1.0 is infeasible by construction *for those
  values*. The shipped yaml's per-stage values are in no snapshot, so that sum is
  a number the reporter supplies (`--stage-gmu`), never one to assume. #8068's
  three-stage serve came up on two cards, so its edited yaml does **not** stack
  0.9 + 0.9 + 0.9 on a single device — "2 cards is infeasible" is a claim that
  requires reading the yaml, and it is not the first answer.
- A CI data point on the same family, not a contradiction: the Instruct variant's
  e2e config runs `tensor_parallel_size: 4`, `gpu_memory_utilization: 0.7`,
  `max_model_len: 8192`, `enable_expert_parallel: true`
  (`tests/e2e/models/configs/Qwen3-Omni-30B-A3B-Instruct.yaml`); scaling TP out is
  the project's own answer to the same wall.

## Answer shape that fits this class

1. Direct answer: the stage yaml's `devices` plus the visibility variable first,
   then whether the per-device `gpu_memory_utilization` sum fits — with numbers.
2. Name the three moves that need no new hardware (the recipe above) before
   offering more cards, dropped stages or quantised weights.
3. Quantify every alternative with the script, and state the limit: the fit check
   excludes activations, KV growth and fragmentation, so a PASS is "not yet
   excluded" while a FAIL is decisive.
4. State the remaining gaps — the yaml merge semantics and the
   logical/physical meaning of `devices` — instead of inventing them, and ask only
   for what is still missing (the edited yaml, per-stage log lines).
5. Confidence labels: doc/CI lines and the `ASCEND_RT_VISIBLE_DEVICES` /
   ascending-order source lines are verified at the pinned commits; the recovery
   recipe is reporter-confirmed on #8068; GPU-side layout details of the shipped
   GPU yaml remain hypothesis.

## Limits

- The fit check ignores activations, KV and fragmentation; TP=2 fitting at 0.9 is
  not a promise that the KV pool is adequate at a long `--max-model-len`.
- "Feasible" is about memory, not scheduling: a layout that fits can still fail in
  the orchestrator, and the orchestrator's parsing is the part that cannot be read
  in these snapshots.
- A confirmed 2-card deploy of *this* model is not a card budget for a larger
  Omni checkpoint or for a different `gpu_memory_utilization`; re-run the script
  with the reporter's numbers.
