# Ray + DP: `Current node has no NPU available` with `NPU_group_<i>_<pgid>` in the dict

The **second** owner of that ValueError. Same message as
`ray-npu-resource-label.md`, different mechanism, and the resource dict decides
between them in one read:

| `current_node_resource={…}` shows | owner |
|---|---|
| `'GPU': 8.0` and **no `NPU` key** | the node advertises the wrong accelerator name → `ray-npu-resource-label.md` |
| `'NPU': 0` (or absent) beside **`'NPU_group_9_<pgid>': 1.0`, `'NPU_group_10_<pgid>': 1.0` …** | a **live placement group already holds this node's devices**, and the process raising is *outside* it → this page |

Verified at vllm `v0.19.1` (`b1388b1f`) + vllm-ascend `v0.19.1rc1` (`da421afad7`),
on vllm's own module path `vllm/v1/executor/ray_utils.py`. Recorded #10910: 2 nodes ×
16 910C, Qwen3-235B-A22B, `--distributed-executor-backend ray --data-parallel-size 2
--tensor-parallel-size 16 --enable-expert-parallel`; `DP=1,PP=1,TP=32` and single-node
`PP=2,TP=8` both work, `DP=2` dies in `initialize_ray_cluster` with the grouped labels.

## Why the dict reads that way (source-verified)

- The check is against the **free** view, not the total:
  `current_node_resource = available_resources_per_node()[current_node_id]`
  (`ray_utils.py:463`, imported at `:35-40`) and `current_node_resource.get(device_str, 0) < 1`
  → `ray_utils.py:466`. Reserved devices stay in the node dict under
  `"<resource>_group_<bundle_index>_<pgid>"` keys, so the plain key reaches 0 exactly
  while a placement group owns them.
- The message also proves **which branch ran**: a process that *is* inside the group
  (or was handed `parallel_config.placement_group`) takes `ray_utils.py:413-419`
  (`Using the existing placement group`) and, if short, raises the *other* text —
  `The number of required NPUs exceeds the total number of available NPUs in the
  placement group` (`:421-438`). So "Current node has no NPU available" means this
  process had neither `parallel_config.placement_group` nor a current PG, and took
  `:441` (`No current placement group found. Creating a new placement group.`).
- DP placement groups are created **only** on the ray DP backend:
  `launch_core_engines` branches on `parallel_config.data_parallel_backend == "ray"`
  (`vllm/v1/engine/utils.py:920-931`) → `create_dp_placement_groups` (`:402`, logs
  `Creating placement groups for data parallel` at `:412`), hands each engine its group
  (`:346-347 dp_vllm_config.parallel_config.placement_group = pg`) and schedules the
  engine core at `placement_group_bundle_index=world_size` (`:370-376`).
- `data_parallel_backend` defaults to **`"mp"`** (`vllm/config/parallel.py:121`), and
  `--distributed-executor-backend ray` does **not** imply it. On `mp`,
  `data_parallel_size_local` defaults to the global DP size (`arg_utils.py:1716-1724`, unless `VLLM_RAY_DP_PACK_STRATEGY=span`),
  so every DP engine starts as a *local process* on the driver node, and each builds a
  `RayDistributedExecutor` → `initialize_ray_cluster` with
  `world_size = pp × tp × pcp` (DP excluded, `vllm/config/parallel.py:632-636`) → the
  first engine's own `strategy="PACK"` group (`ray_utils.py:474-478`, specs `:454-456`)
  packs that many devices onto the driver node, and the **second** engine reads the driver node's free
  NPU as 0. That is the reported shape: "the DP engine creates a placement group and
  then fails" — the group it saw belongs to its sibling.
- `nnodes > 1` is only allowed with `mp` (`arg_utils.py:1630-1632`), so a
  `--data-parallel-backend ray` launch must be started from one node (which is the
  upstream-supported ray DP shape).

## One call each, cheapest first

1. Grep the reporter's own log for the three branch lines above
   (`Creating placement groups for data parallel`, `Using the existing placement group`,
   `No current placement group found …`). Their presence/absence names the backend and
   the branch — no source read needed.
2. Read their launch line: is there `--data-parallel-backend ray`? `nnodes`? Only
   `--distributed-executor-backend ray` + `--data-parallel-size 2` is the `mp` path.
3. `ray status` and `ray list placement groups` on the driver node: `NPU: 0.0/16.0`
   with one live group is the confirmation. A group left by a **crashed earlier run**
   produces the identical dict with no sibling engine alive — restart the raylets (or
   remove the group) and re-run; distinguish the two by whether a second engine is
   starting in the same second.
4. Bundle index arithmetic if they paste the full dict: `<i>` is the bundle index, so
   the highest `NPU_group_*` index + 1 = the devices that group took (`tp×pp`, or
   `tp×pp + 1` when the extra CPU bundle of a DP engine core is visible).

## Fix shape

- Two-node DP from one `vllm serve`: add `--data-parallel-backend ray` (keep
  `--distributed-executor-backend ray`), so the DP manager creates one group per DP
  rank, sizes each to `world_size` and passes it down; then `dp_rank` engines no longer
  race for the node's free devices.
- Two-node DP with the default `mp` backend is the upstream `--nnodes 2 --node-rank i`
  shape: one serve per node, each node's engine owning its own devices. A single serve
  that tries to spawn both DP ranks locally against a ray executor is what fails here.
- Do not answer with a card count, a driver, an HCCL or a `--tensor-parallel-size`
  theory: no device was ever opened, and `world_size` is `tp×pp`, so the second
  engine's demand is a *duplicate* of the first's, not a bigger one.

## Limits

No run is reproducible from this snapshot: the branch structure and the free-resource
read are source-verified at the pins above, and the reporter's own dict is the evidence
that a group holds the devices. Which of the three producers (sibling engine, stale
group, another serve) owns it is decided by check 1/3, not by the message text.
