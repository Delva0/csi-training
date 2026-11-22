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
`utils.py:723 get_soc_version` at the recorded #8068 pin v0.18.0rc1 — re-grep the
symbol, the line moves: at the #9335 pin v0.19.1rc1 it is `worker.py:114` and
`utils.py:745 check_ascend_device_type()`, which calls
`torch_npu.npu.get_soc_version()` inline, there being no vllm-ascend
`get_soc_version` at that ref → `aclInit 107001`), before weights, KV, capture or
any request. Editing the yaml is therefore the fix, not "get a third card".

Still unverified, because vllm-omni is in no snapshot: the merge order of
`platforms.<vendor>` with the stage yaml, whether `devices` are logical or
physical ids, and whether two stages may share one id. Say so, and ask for the
edited yaml plus each stage's own log line whenever an attempt disagrees.

## Two visibility variables at two moments — pin both, not one

The card set reaches a process through **two** variables consumed at **two**
different stages, and a report that set one has pinned neither:

| Variable | Read by | When it takes effect |
|---|---|---|
| `ASCEND_VISIBLE_DEVICES` | the **Ascend Docker Runtime prestart hook**, out of the OCI spec's own `Process.Env` — `component/ascend-docker-runtime/runtime/process/process.go:545` (`checkVisibleDevice` → `getValueByDeviceKey(spec.Process.Env)`), devices + cgroup entries added at `:426-443`, orchestrated by `processDevicesAndHooks` at `:672-688` | container **creation**, before PID 1 exists |
| `ASCEND_RT_VISIBLE_DEVICES` | `torch_npu._parse_visible_devices()` (`torch_npu/npu/__init__.py:341-366`, `getenv("ASCEND_RT_VISIBLE_DEVICES")` only) | every vLLM process, at torch_npu import |

The creation-time one does three things in that hook and nothing afterwards
(`Ascend/mind-cluster` `v7.2.RC1.SPC2` `c5694eb`, `component/ascend-docker-runtime/README.md:21-23`):
mounts the named `/dev/davinciN` into the container namespace, appends the
`devices.allow` cgroup entry for the same majors/minors, and mounts the host CANN
runtime libraries. Name constant `component/ascend-common/api/default_name.go:133-134`.
So `export ASCEND_VISIBLE_DEVICES=…` **inside** a running container reaches no
reader: the hook has already run, `checkVisibleDevice` returns empty when the name
is absent from the spec env (`:546-548`, then "will not mount any ascend device"),
and no Python in the pinned trees parses it. The half that has to be at creation is
`docker run -e ASCEND_VISIBLE_DEVICES=…` (in K8s the Ascend device plugin writes it
for you — `component/ascend-device-plugin/pkg/common/common.go:97`,
`component/ascend-operator/pkg/controllers/v1/env.go:99-105`); the in-container half
is `export ASCEND_RT_VISIBLE_DEVICES=…`.

Reporter-confirmed on **#6942** (image `v0.14.0rc1-openeuler`, 2 cards on a node
already running another container; `docker run --privileged --device /dev/davinci6
--device /dev/davinci7 …`, then `vllm serve /data/model/Qwen3-30B-A3B
--tensor-parallel-size 2 --enable-expert-parallel`, OCR paste stopping at
`Waiting for 1 local, 0 remote core engine proc(s) to start.`). The reporter's own
account: `export ASCEND_RT_VISIBLE_DEVICES=6,7` **alone** did not clear it,
`export ASCEND_VISIBLE_DEVICES=6,7` **alone** did not clear it, and
`docker run -e ASCEND_VISIBLE_DEVICES=6,7` at creation **plus** `export
ASCEND_RT_VISIBLE_DEVICES=6,7` inside did. Neither half alone tests the other, so
"I already set the visible devices" is not evidence until it says *which* variable
and *where*: `docker inspect -f '{{.Config.Env}}' <container>` and
`tr '\0' '\n' < /proc/1/environ` (creation env, inherited by every process) against
the failing shell's own `export` is one call and separates the two.

Why a hand-built command is the usual source of this: **the vllm-ascend docs never
carry the creation-time variable.** At v0.14.0rc1 `ASCEND_VISIBLE_DEVICES` occurs
nowhere under `docs/`, the install block (`docs/source/installation.md:60-73`) lists
the `--device` lines with no `-e`, and the 13 pages under `docs/` that mention a
visibility variable mention only the runtime one (`DeepSeek-V3.1.md:298`,
`Qwen3-Dense.md:158`, …).
A `docker run` assembled from that block gives the hook no input by construction, so
the container's device set is only what `--device`/`--privileged` handed it. A
project outside vllm-ascend that does pass both at creation:
`LMCache/LMCache-Ascend` `docs/deployment.md:23-24` / `README.md:63-64`
(`-e ASCEND_VISIBLE_DEVICES=${DEVICE_LIST} -e ASCEND_RT_VISIBLE_DEVICES=${DEVICE_LIST}`).

Left open by these sources, so do not close it for the reporter: *why* #6942's
missing creation-time half only bit while a neighbouring container ran (the
reporter's own control), and whether the hook's list also restrains a **privileged**
container — `--privileged` bypasses `devices.allow`, and this runtime has no
privileged branch. On that paste the HBM co-tenancy reading
(`../runtime-memory/kv-budget-per-instance-vs-device-global.md`) is a candidate the
reporter never confirmed; the cure they reported is the visibility pin, which is also
the cheaper check.

## The runtime variable's three traps

The control variable every launch script in the vllm-ascend docs exports is
`ASCEND_RT_VISIBLE_DEVICES` (`vllm_ascend/platform.py:101` at v0.19.1rc1; every
multi-node launch script in the same docs tree exports it — `GLM5.md`,
`DeepSeek-V3.1.md` — while `CUDA_VISIBLE_DEVICES` occurs nowhere in
`docs/source/`). A command copied from a GPU page therefore exposes **all** cards to
**every** process, which looks exactly like a stage-collision bug.

- **No Python alias.** `_parse_visible_devices()` reads
  `ASCEND_RT_VISIBLE_DEVICES` only (`torch_npu/npu/__init__.py:341`, the same line
  at both `v7.1.RC1.alpha002-pytorch2.7.1` and `v7.2.0.1-pytorch2.8.0`), and neither
  the `Ascend/pytorch` tree nor `vllm_ascend` mentions the other spelling. #8068's
  env dump exported `ASCEND_VISIBLE_DEVICES=2,1` while its working recovery exported
  the `RT_` name; the first had no in-process effect (its only reader is the
  creation-time hook above), which is why "the reporter set the visible devices" is
  not a finding yet.
- **The list is ascending, or it is empty.** `_device_count_ascend_hal()`
  returns **0** devices as soon as an entry is not greater than its predecessor
  (`torch_npu/npu/__init__.py:375-398` at the 2.7 pin; `:386-409` at the 2.8 pin,
  comment `` `rts` need ascending order``), so `2,1` is not a narrower view, it is
  no devices at all.
- **The ids index the device list the container exposes, not `npu-smi`'s NPU
  labels.** The same function trims at the first entry `>= drvGetDevNum()` (raw
  count at `:369-383`, trim at `:392-393` in the 2.7 pin / `:403-404` in the 2.8
  pin), so the reference is `ls /dev/davinci*` inside that container. Which names
  those are depends on the mapping: #8068's container held `davinci0,1` for the
  cards `npu-smi` calls NPU **1** and **2**, so the working export was `0,1`; on
  #6942's `--privileged` container, which carries all eight host nodes under their
  host names, the export that worked was `6,7`. `device_count()` printing `0` or a
  `107001` is how you find out which case you are in.

One line settles this lane before any vLLM flag is audited:
`python -c "import torch, torch_npu; print(torch_npu.npu.device_count(), torch_npu.npu.get_soc_version())"`
— a `0` there means the RT list is out of range or not ascending (not that a card is
missing), and a `107001` ends the vllm-ascend part of the case
(`../startup-failures/device-id-invalid-at-worker-init.md`).

## Then the arithmetic: does the layout you just wrote fit?

```bash
bash <skill_root>/scripts/card-budget.sh --params 30.5 --tp 2 --gmu 0.9 --cards 2
bash <skill_root>/scripts/card-budget.sh --params 30.5 --cards 2 \
     --stage-gmu <stage0>,<stage1>,<stage2> --stage-devices '0,1;0;1' \
     --stage-params 30.5,<talker>,<code2wav>
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
- **The per-device `weights` column is a worst case, and the gmu sum is the
  decision.** Without `--stage-params` the script charges *every* stage a full copy
  of `--params` — right for three separate checkpoints, wrong for vllm-omni, whose
  stages 1+2 are the same checkpoint's talker/code2wav sub-models. The recorded
  #9335 run priced a 2-card split without it, read `weights 91.5 GB` per device,
  and had to disclaim that number in its own answer before the ladder was usable;
  one `--stage-params 30.5,3,0.5` on that `'0,1;0;1'` layout prints the honest
  `36.5 GB` for the card that also hosts the talker and `31.5 GB` for the other
  (`34.0 GB` averaged over the two; the sub-model sizes come from the checkpoint's
  own tensor index, not from the model name). A gmu sum above 1.0 fails with or
  without that flag.
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
