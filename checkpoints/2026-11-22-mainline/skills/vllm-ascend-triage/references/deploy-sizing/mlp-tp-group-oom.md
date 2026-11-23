# OOM right after enabling `VLLM_ASCEND_ENABLE_MLP_OPTIMIZE`: the MLP group is sized by DP, and the group size is what shards the weights

## Scope

The report is one sentence of configuration — "Qwen3-32B TP4 使能
`VLLM_ASCEND_ENABLE_MLP_OPTIMIZE` 会导致 OOM", `enforce_eager=True`, `vllm-ascend 0.11.0rc0`,
often **no log at all** (#3463). Deliverable: the bytes per card the switch adds, the code path
that decides them, and "unset it" — with the reason the switch cannot pay for itself in that
deployment.

Not this page: an OOM at KV sizing or after hours (`../runtime-memory/index.md`), a TP-scaling
OOM with no flag diff (`tp-scaling-oom-at-startup.md`), an EP load OOM on a many-expert MoE
(`moe-load-oom-expert-shard.md`).

## Mechanism (vllm-ascend `v0.11.0rc0`, `00ba071`)

| site | fact |
|---|---|
| `vllm_ascend/envs.py:156-157` | `VLLM_ASCEND_ENABLE_MLP_OPTIMIZE`, default `'0'` — so it is part of the reporter's diff, not a default |
| `vllm_ascend/utils.py:588-589` | `mlp_tp_enable()` returns that env var |
| `vllm_ascend/distributed/parallel_state.py:65-81` | `mlp_tp = parallel_config.data_parallel_size`; `torch.arange(world_size).reshape(-1, mlp_tp, pipeline_parallel_size, 1).view(-1, mlp_tp)` — the `_MLP_TP` group is **dp ranks** (TP4/DP1, world 4 → `[[0],[1],[2],[3]]`) |
| `vllm_ascend/ops/linear_op.py:67-77` | `CustomTensorParallelOp.comm_group = get_tp_group()`, and `tp_rank`/`tp_size` are `comm_group.rank_in_group` / `comm_group.world_size` |
| `vllm_ascend/ops/linear_op.py:133-153, 219-247` | `MLPColumnParallelOp.comm_group` / `MLPRowParallelOp.comm_group` return `get_mlp_tp_group()`; the column op `all_gather`s the input, the row op `reduce_scatter`s the output, both over that group |
| `vllm_ascend/ops/linear_op.py:423, 447` | the selectors: `"gate_up_proj" in prefix and mlp_tp_enable()` → `MLPColumnParallelOp`; `"down_proj" in prefix and mlp_tp_enable()` → `MLPRowParallelOp` (first match owns the layer, so this is mutually exclusive with the SP / matmul-allreduce / oproj branches below it) |
| `vllm_ascend/ops/linear.py:229, 311` | `input_size_per_partition = divide(input_size, self.tp_size)` and `output_size_per_partition = divide(output_size, self.tp_size)`, and `create_weights(...)` is called with those per-partition sizes |

With `dp == 1` the group has one member, `tp_size == 1`, so `divide(x, 1)` leaves both MLP
linears **full width on every card**. The arithmetic stays correct — a size-1 group makes
`all_gather`/`reduce_scatter` no-ops — which is why nothing else looks broken: the cost is the
footprint, and the promised comm saving is gone at the same time.

## Rule 1 — price an "optimize" switch in bytes before ranking allocator causes

Sharding follows the **comm group**, not `parallel_config.tensor_parallel_size`. A flag that
swaps a layer's group therefore changes weight shapes at construction time, before any
allocator, workspace or KV behaviour is in play. For an OOM report that names an env flag:
read the flag's default (`envs.py`), the selector that consumes it (`linear_op.py`), the group
it builds (`parallel_state.py`), then `divide(...)` in `linear.py`. Only then price.

Arithmetic, Qwen3-32B (`hidden 5120`, `intermediate 25600`, 64 layers, bf16 — #3463's model):

- `gate_up_proj` = 2·h·i = 262.1 M params/layer: 524 MB full vs 131 MB at TP4
- `down_proj` = i·h = 131.1 M params/layer: 262 MB full vs 66 MB at TP4
- 64 layers: **50.3 GB per card full vs 12.6 GB sharded → +37.7 GB**, i.e. `MLP bytes ×
  (1 − 1/dp)`. Recompute with the reporter's own `config.json`
  (`(2·h·i + i·h) × layers × 2 B × (1 − 1/dp)`) when the numbers differ
- Secondary: activations, because `gate_up`'s output is the full `2·i` wide (51200 vs 12800 at
  TP4 → 0.84 vs 0.21 GB at 8192 tokens). Never the main term
- The surrounding budget is `scripts/card-budget.sh`, which prices sharded weights only — this
  delta is invisible there, so state it separately

## Rule 2 — the version scope splits selector from arithmetic

`mlp_tp_enable()` reads an **env var** at this pin; later refs move the same group behind
`--additional-config` `finegrained_tp_config.{mlp,oproj}_tensor_parallel_size > 0`
(`../feature-enable-recipes/sequence-parallel-flashcomm.md`, v0.14.0rc1 `utils.py:800-805`).
Before quoting, re-grep two lines at the reporter's ref:
`rg -n -A8 'def mlp_tp_enable' vllm_ascend/utils.py` and
`rg -n -A10 'mlp_tp =' vllm_ascend/distributed/parallel_state.py`. What transfers is the shape
("weight shard = group size, group sized by dp"); the number is pinned.

## Rule 3 — the answer

1. `unset VLLM_ASCEND_ENABLE_MLP_OPTIMIZE` (default is `0`), rerun unchanged — that is the fix.
   Say why: at `dp=1` the flag builds size-1 groups, so the MLP is replicated and every MLP
   collective becomes a no-op; it is the release note's own "when tensor parallel is enabled …
   in eager mode gets better performance" (`docs/source/user_guide/release_notes.md:107`,
   PR #2120) that invites exactly this misuse, so `enforce_eager=True` is not the cause.
2. The A/B that proves it: one command with `=1` and `=0`, compare the post-load line
   (`Loading model weights took … GB`) or `torch.npu.memory_allocated()` right after
   `LLM(...)`; expect the ~35-40 GB above.
3. A perf lever that keeps the TP group on a TP-only serve is `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`
   (selector branch `linear_op.py:451`) — validate numerics *and* step time, and do not stack it
   with the MLP knob.
4. If the flag is genuinely wanted, its group must have more than one rank, i.e. a real
   `--data-parallel-size > 1` layout: the MLP is then sharded over the dp axis, which is a
   different sharding plan (and needs the `--data-parallel-size-local/address/rpc-port`
   group, see `moe-load-oom-expert-shard.md` §DP group), not a tuning knob for a TP-only serve.
5. `--gpu-memory-utilization` cannot close a ~38 GB deficit on a 64 GB card — do not offer it.
6. For the maintainer: at `dp == 1` the switch degenerates **silently** (full MLP copies, all
   its collectives no-ops), so the guard/warning belongs where `_MLP_TP` is built
   (`init_ascend_model_parallel`), and that is a legitimate `[Bug]` to file from the answer.

## What to send back

Card HBM (`npu-smi info`, 32 vs 64 GB decides feasibility), the OOM text — `NPU out of memory`
during weight load is this page, whereas vLLM's `No available memory for the cache blocks` is
the KV-sizing stage (`../deploy-sizing/tp-scaling-oom-at-startup.md`) — the serve command plus
the exported env block, and the checkpoint's `config.json`
(`hidden_size`, `intermediate_size`, `num_hidden_layers`).

Recorded: #3463 (vllm-ascend `v0.11.0rc0` `00ba071`, Qwen3-32B TP4 eager, no log in the report).
