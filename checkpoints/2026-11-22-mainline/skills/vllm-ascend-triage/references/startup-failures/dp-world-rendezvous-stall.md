# Stall on `Adjusting world_size=N rank=r … for DP`: the world rendezvous

Scope: the pasted log **ends** on a line like

```
INFO … [parallel_state.py:883] Adjusting world_size=16 rank=0 distributed_init_method=tcp://127.0.0.1:42051 for DP
```

with no traceback, often "部署卡住/stuck at this step", on a serve that carries
`--data-parallel-size` > 1 (recorded #1255: vllm-ascend v0.8.5rc1 + vllm
0.8.5.post1, DeepSeek-R1-W8A8, `--data-parallel-size 16 --enable-expert-parallel
--distributed-executor-backend ray`, 8×910B2 one node). This is the DP lane of
startup, not an HCCL/fabric lane and not the offline `VLLM_DP_*` demo
(`../hccl-and-hardware/dp-group-peer-close.md`).

## 1. What the line means (one read, pinned upstream)

`vllm/distributed/parallel_state.py` `init_distributed_environment` (vllm v0.8.5
`ba41cc90e8ef`, lines 871-895): when `parallel_config.data_parallel_size > 1` it
rewrites `rank = data_parallel_rank * world_size + rank`,
`world_size = parallel_config.world_size_across_dp`,
`ip = data_parallel_master_ip`, `port = get_next_dp_init_port()`,
`distributed_init_method = f"tcp://{ip}:{port}"`, logs that INFO, and the **next
statement** is `torch.distributed.init_process_group(backend=…, init_method=…,
world_size=world_size, rank=rank)`.

So the silence after it is by construction: **every TP×PP×DP rank must join that
TCPStore before any collective, weight load or device op happens.** Nothing about
the card, the driver or HCCL can be concluded from the stall; the ranks are
simply not all there. The reporter's `:883` vs the pin's `:884` is a
`.post1` one-line offset.

## 2. Arithmetic first, from the paste alone

- `world_size_across_dp = (PP × TP) × DP` — `vllm/config.py:1646-1647` fields,
  computed at `:1724`; `world_size` is `PP × TP` (`:1710-1712`).
- Count the devices the same paste proves: `npu-smi` blocks, or the cardinality
  of `ASCEND_RT_VISIBLE_DEVICES`. #1255: DP=16 × TP=1 = 16 ranks against 8 cards
  on one node → **8 ranks can never be created**, so the rendezvous waits. The
  model is irrelevant to this leg (but decisive for the replacement launch — §5).
- `--enable-expert-parallel` does not add ranks: EP group sizes are derived from
  the same DP×TP world (`vllm_ascend/distributed/parallel_state.py:get_ep_group`,
  `vllm_ascend/ops/fused_moe.py:59-61`), so it cannot make 16 ranks fit on 8
  devices either.

## 3. Where the missing ranks are

- The DP engine cores are spawned **by the frontend process on this host**:
  `vllm/v1/engine/core_client.py:849-851` — `dp_size = …data_parallel_size; for i
  in range(dp_size): # Multi-node not yet supported so local_dp_rank == dp_rank`.
  16 `EngineCore_DP*` children are created here; `--distributed-executor-backend`
  only decides how each core places its *own* TP×PP workers.
- Each core's ray executor asks for one device as resource `NPU`, not GPU:
  `vllm/executor/ray_distributed_executor.py:159` (`num_gpus =
  envs.VLLM_RAY_PER_WORKER_GPUS`) and `:205-221` — on a non-`GPU`
  `ray_device_key` the actor is `num_gpus=0,
  resources={<ray_device_key>: num_gpus}`; bundles are taken from
  `placement_group.bundle_specs` where `bundle.get("NPU")` is truthy, sliced to
  `world_size` (`:193-198`). `vllm_ascend/platform.py:57` sets
  `ray_device_key = "NPU"`.
- Consequence: on 8 cards, 8 cores get a bundle and 8 stay pending; the 8 that
  started block in step 1 forever. `PG_WAIT_TIMEOUT = 1800`
  (`vllm/executor/ray_utils.py:24`, used by `_wait_until_pg_ready`) is why
  **nothing is printed for up to 30 minutes** — silence is not evidence of a
  deadlock-free run.

## 4. `tcp://127.0.0.1:<random>` is itself the finding

At v0.8.5, `ParallelConfig.__post_init__` takes the CLI branch when
`data_parallel_size > 1` and sets **only** `data_parallel_master_port =
get_open_port()` with a literal `# TODO multi-node`
(`vllm/config.py:1717-1720`); `data_parallel_master_ip` keeps its `127.0.0.1`
default. So a `--data-parallel-size` launch is intra-node-only at this ref: on
two nodes each node's rank 0 binds its *own* loopback and the two stores never
meet. Loopback + random port in the paste therefore rules out "I gave you 16
cards across two hosts"; if the reporter needs multi-node capacity, DP is not the
knob at this version (see step 5).

## 5. Version scope (state it in the answer)

- vllm-ascend v0.8.5rc1 (`ec27af346a`) contributes only process-group plumbing
  for DP — `vllm_ascend/patch/platform/patch_common/patch_distributed.py`
  (`parallel_config_get_dp_port`, `ascend_stateless_init_dp_group` with
  `backend="hccl"`, `destroy_model_parallel`) — plus the offline
  `examples/dp_offline/data_parallel.py` demo. No DP engine, no DP docs: the
  version's own `docs/source/tutorials/multi_node.md:138-164` recipe is
  `--tensor-parallel-size` = NPUs per node, `--pipeline-parallel-size` = nodes.
- So `--data-parallel-size 16` on one 8-card node is an unsupported
  configuration rather than a defect. The deliverable is the working launch plus
  the checks that confirm the diagnosis on their machine — and that launch must be
  **priced before it is named**, because the reporter's DP is not gratuitous: with
  `--enable-expert-parallel` the MoE experts are sharded across the whole DP×TP
  world (16 ranks here), so "drop DP and use `--tensor-parallel-size 8` on your one
  node" halves the shard and cannot load a 671B W8A8 checkpoint at all.
- **Price the replacement at 1 byte/param, from the paste's own HBM and util**
  (`../deploy-sizing/index.md` rule 1). #1255's `npu-smi` reads `65536 MB` and its
  serve line `--gpu-memory-utilization "0.95"`: `bash
  <skill_root>/scripts/card-budget.sh --params 671 --dtype w8a8 --hbm 64 --gmu
  0.95 --tp 8` prints `83.88 GB per card` against a `60.8 GB` budget →
  `VERDICT: FAIL`, while `--tp 16` prints `41.94 GB per card` and fits with
  ~`18.9 GB` left for KV. So the single-node downsize is infeasible on this
  reporter's numbers, and the working answers are the ones that keep 16 cards: the
  pin's `TP=8 × PP=2` two-node recipe, `TP=16` across two nodes
  (`docs/source/tutorials/multi_node.md:138-165`), or the offline `VLLM_DP_*`
  fan-out with one process per rank. `--tp` is the **weight-shard width** (the
  TP×PP product for those two layouts), not the flag the reporter wrote. Never
  quote a per-card weight figure that was not priced this way.

## 6. Checks to request (cheap → decisive)

1. `ray status` — Pending Demands / `NPU` (or `GPU`) total: 8 vs 16 settles it in
   one command; also `ray list actors | grep -c pending`.
2. Per-rank logs: `grep -c 'Adjusting world_size'` should equal
   PP×TP×DP; a smaller count says which rank never started.
3. `ps -ef | grep -c EngineCore_DP` and the `npu-smi` process table — 16 cores
   with 8 devices is the same answer without ray.
4. If the log later shows a store/HCCL timeout (`Timed out after …`,
   `Connection closed by peer`), that is the same stall expiring, not a link
   fault; if it shows `ValueError: Current node has no NPU available …
   'GPU': 8.0`, switch to `ray-npu-resource-label.md` (label mismatch, not
   cardinality).

Leaf: leave the cause open between "too few devices for DP×TP×PP" and "ray
advertises GPU while the platform asks for NPU" only when the paste shows the
`ValueError`; without it the arithmetic answer needs no further source read.
