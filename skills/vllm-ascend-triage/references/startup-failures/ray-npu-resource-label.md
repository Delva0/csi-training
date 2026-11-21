# Ray worker start: `Current node has no NPU available` — the cluster advertises `GPU`

Verified at vllm-ascend `v0.10.2rc1` (`048bfd5`, `platform.py:52`) with vllm
`v0.9.1` (`b6553be`) as the ray-executor reference (that pin is
`version_role: dated_reference_only` in `SOURCES.json`; the reporter ran vllm
0.10.2, whose own traceback puts the same two messages at `ray_utils.py:352`
and `:368`). Recorded for #3114: two A2 nodes × 8×910B3,
`v0.10.2rc1-openeuler` image, Qwen3-235B-A22B, `--tensor-parallel-size 8
--pipeline-parallel-size 2 --enable-expert-parallel
--distributed-executor-backend ray` (world size 16 = the cluster's 16 devices).

## The signature

```
(EngineCore_DP0 pid=14400) INFO 09-23 03:01:22 [worker.py:1951] Connected to Ray cluster.
(EngineCore_DP0 pid=14400) WARNING [ray_utils.py:352] The number of required NPUs exceeds the total number of available NPUs in the placement group.
(EngineCore_DP0 pid=14400) ERROR [core.py:718] EngineCore failed to start.
ValueError: Current node has no NPU available. current_node_resource={'memory':
  2151287372800.0, 'object_store_memory': 10000000000.0, 'node:x.x.x.201': 1.0,
  'CPU': 192.0, 'node:__internal_head__': 1.0, 'GPU': 8.0}. vLLM engine cannot
  start without NPU. Make sure you have at least 1 NPU available in a node
  current_node_id='…' current_ip='x.x.x.201'.
```

Read two things in that paste and the lane is decided: the engine **did** connect
to the ray cluster (the port/GCS side is fine), and the driver node's own
resource map carries `'GPU': 8.0` and **no** `NPU` key. The two log lines are the
two lookups that key returns 0 for.

## Mechanism (source-verified at the pins above)

- vllm-ascend tells vLLM which resource name to use:
  `NPUPlatform.ray_device_key: str = "NPU"` (`vllm_ascend/platform.py:52`), plus
  `device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"` (`:53`).
- Upstream keys **everything** on that literal string:
  `device_str = current_platform.ray_device_key` (`vllm/executor/ray_utils.py:300`);
  the cluster total `num_devices_in_cluster =
  ray.cluster_resources().get(device_str, 0)` (`:336`) — which is what emits the
  *warning* in the paste; the driver-node check
  `available_resources_per_node()[current_node_id]` +
  `current_node_resource.get(device_str, 0) < 1` (`:355-361`) — the `ValueError`;
  the placement-group bundles `{device_str: 1.0}` (`:344-347`); the worker actors
  `resources={current_platform.ray_device_key: num_gpus}`
  (`vllm/executor/ray_distributed_executor.py:209-224`, the non-`"GPU"` branch);
  and the accelerator ids handed back to the worker
  (`ray_utils.py:59-63`, `get_accelerator_ids()[device_key]`).
- So a node that registers its eight devices under `GPU` satisfies none of them:
  there is no card shortage, no driver fault and nothing wrong with the model —
  the resource **name** does not match.

A `ray start … --num-gpus 8` in the tutorial being followed is the usual producer
of `GPU: 8.0` on an NPU-only host; that line is not in the paste, so ask for it
(hypothesis, one line settles it).

## The one check, and the fix

```bash
# on each node, in the shell that started the raylet
ray status          # Resources block: it must list `NPU: 8` per node, not `GPU: 8`
```

If the block lists `GPU`, restart both raylets so each node advertises the key
the engine asks for — Ray's own doc at `ray-2.47.1`
(`doc/source/ray-core/scheduling/accelerators.rst:36-37` lists Huawei Ascend as
`NPU`, community/experimental, and `:110-117` shows the declaration form):

```bash
# node 0
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --head --port=6379 --resources='{"NPU": 8}'
# node 1
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ray start --address=<node0-ip>:6379 --resources='{"NPU": 8}'
```

Ray snapshots the environment at `ray start`, so `ASCEND_RT_VISIBLE_DEVICES`
(the platform's `device_control_env_var`) has to be exported in **that node's**
shell, not in the driver's — same per-process rule as
`environment-correspondence.md`'s ray section. Then `ray status` shows
`NPU: 16` for two 8-card nodes and the same serve command starts.

## Not the causes, and say so

- **Card count / TP size**: the paste's `ray status` already shows `0.0/16.0`
  devices for world size 16 — the arithmetic fits; only the key differs.
- **A driver, CANN or image fault**: the engine never reached a device; every
  frame is ray placement-group setup.
- **The model or `--enable-expert-parallel`**: nothing about the checkpoint is
  read before this raise.
- **Ray "not supporting NPU"**: NPU is a community accelerator in Ray, so it must
  be declared; that is a launch line, not an install.

## The doc gap is part of the answer

The reporter followed `https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_node_ray.html`
— a page **absent from the pinned release**: `docs/source/tutorials/` at
`048bfd5` holds `multi_node.md`, `multi_node_kimi.md`,
`multi_node_pd_disaggregation.md`, `multi_npu*.md` and `single_npu*.md`, and no
`ray start` line exists anywhere in that tree
(`rg -n "ray start|--resources|ray_device_key" -g '!tests/**'` returns only
`platform.py:52`). So the launch commands the reporter copied cannot be verified
here: state that, give the `ray status` check and the `--resources` fix, and ask
for their two `ray start` lines. The reporter's own follow-up on #3114 was a docs
PR ("可以合并这个，优化手册"), i.e. the manual, not the engine, was the defect.

## Recorded run (#3114)

38 messages, one 81-character greeting and no answer text at all before the
16 384-token cap: the discriminator (`ray status` → `0.0/16.0 GPU`, no `NPU`
line) was in the paste from the first turn, and the walk went instead into ray's
accelerator docs, upstream `ray_utils`, release notes and the precedence corpus.
The reply owed on turn 2-3 was: "your cluster advertises `GPU`; vllm-ascend asks
for `NPU`; restart each raylet with `--resources '{"NPU": 8}'`, then re-check
`ray status`" plus the doc gap.
