# PP > 1 on Ray: `KeyError: '<…>.layers.<N>.self_attn.attn'` during kv-cache init

## Scope

Engine start passes weight loading, then every rank (or one rank) dies with

```
vllm_ascend/worker/model_runner_v1.py  initialize_kv_cache -> initialize_attn_backend
vllm_ascend/worker/model_runner_v1.py  get_attn_backends_for_group
vllm/v1/worker/worker_base.py          initialize_from_config
vllm/config/vllm.py                    get_layers_from_vllm_config -> <dictcomp>
KeyError: 'model.layers.30.self_attn.attn'
```

Typical launch: `--distributed-executor-backend ray`, `--pipeline-parallel-size N>1`,
one node per stage (recorded #5724: 4 nodes × 16 910B, TP=16 PP=4, Kimi-K2-Thinking).
Verified at vllm v0.12.0 `4fd9d6a8` + vllm-ascend v0.12.0rc1 `42ceaf08`; the
per-worker-config contract is upstream's and has been stable across v1 releases,
but re-grep the line numbers below at another pin.

**Re-verified at vllm `v0.19.1` (`b1388b1f`) + vllm-ascend `v0.19.1rc1` (`da421afa`),
and the anchors moved.** Recorded #10910: 2 nodes × 16 910C, Qwen3-235B-A22B (94
layers), `--distributed-executor-backend ray`, and the reporter's own matrix decides it
— `DP=1,PP=1,TP=32` works, single-node `PP=2,TP=8` works, `PP=4,TP=8` dies, `DP=2`
fails elsewhere (see `../startup-failures/ray-dp-placement-group-npu-exhausted.md`).
At this pin: `global_rank` frozen at `v1/worker/worker_base.py:199`, the index at
`:303` (`kv_cache_config = kv_cache_configs[self.global_rank]`), **`adjust_rank` is
`v1/executor/ray_utils.py:54-61`** — a `rg "def adjust_rank" v1/worker/` returns
nothing at this pin and costs a turn — the re-rank sort is
`v1/executor/ray_executor.py:247-276` (`0 if ip == driver_ip else 1, ip_counts[ip], ip`,
then `collective_rpc("adjust_rank", …)` at `:276`), and `collective_rpc` walks the
sorted `self.workers`. So TP-only survives (every config carries the same 94 names)
while PP>1 does not, which is exactly the reporter's control pair.

**Not** the offloading page: `../upstream-downstream-drift/kv-offloading-layer-coverage.md`
covers the same *key shape* raised in `offloading_connector.py register_kv_caches`
from a layer-**type** filter. Decide by the frame, not the key: `initialize_kv_cache
→ initialize_attn_backend` (this page) vs `register_kv_caches` after
`Creating offloading spec with name:` (that page). A plain PP launch with no
`--kv-transfer-config` is never the offloading one.

## Mechanism (the only candidate that also explains why one layer index is named)

`get_layers_from_vllm_config` does `forward_context[layer_name]` for every name in
`kv_cache_group_spec.layer_names`; those names come from the **per-worker**
`KVCacheConfig` the engine built, so a KeyError means *the worker was handed a layer
list it never built*. With PP>1 the per-worker lists differ (one PP stage's layers
each), which is what makes a wrong config fatal instead of harmless.

| Claim | Evidence (vllm v0.12.0) |
|---|---|
| The worker picks its config by index: `kv_cache_config = kv_cache_configs[self.global_rank]` | `v1/worker/worker_base.py:316-320` |
| `global_rank` is frozen at construction from `rpc_rank` | `:196` `self.global_rank = self.rpc_rank if global_rank is None else global_rank` |
| Ray passes only `rpc_rank`, and it is the **creation** index | `v1/executor/ray_executor.py:212` / `:222` (`.remote(vllm_config=…, rpc_rank=rank)`) |
| Workers are then **re-sorted** by `(driver node first, ip_counts[ip], ip string)` and re-indexed | `ray_executor.py:244-264` (`sort_by_driver_then_worker_ip`, `adjusted_rank = i`, `self.workers = […]`) |
| Only `rpc_rank` is rewritten by the re-rank; `global_rank` is not | `worker_base.py:217-227` (`adjust_rank`), called at `ray_executor.py:266-269` |
| Result lists follow the **sorted** order, so list index = adjusted rank | `ray_executor.py:482-493` (`collective_rpc` → `[worker.execute_method.remote(...) for worker in self.workers]`) |
| Configs are per-worker layer-name subsets, in merged (ascending) order | `v1/core/kv_cache_utils.py:1299+` (`get_kv_cache_configs`, `group_layer_names_one_worker = [n for n in group.layer_names if n in kv_cache_spec_one_worker]`) |

So on a single node the sort key is constant, Python's stable sort keeps creation
order, and nothing is misindexed; with ≥2 nodes the sort permutes and each worker
can read a sibling stage's config. TP-only deployments survive because every
config carries the same names — **PP>1 is what makes the permutation visible**, and
the first missing name in the wrong list is always that foreign stage's *lowest*
layer index.

## Step 1 — fingerprint the printed layer index (do this first)

`vllm/distributed/utils.py:95-141` `get_pp_indices`: base = `layers // pp`, and the
`layers % pp` remainder is added one layer at a time to partitions `[-2], [-3], …`
— the **last** partition is never enlarged, so the naive `layers//pp` boundary is
wrong whenever `layers % pp != 0`. 61 layers, PP=4 → `[15,15,16,15]`, stage starts
`0,15,30,46`, so `layers.30` is exactly pp_rank 2's first layer and the wrong-config
theory is live; 60 layers → starts `0,15,30,45` (same verdict); 62 → `0,16,32,47`
and the same key would **refute** the theory.

One call: `bash <skill_root>/scripts/pp-layer-partition.sh --layers 61 --pp 4 --key-layer 30`.
Get `layers` from the checkpoint's `config.json` (`num_hidden_layers`), never from
the model name, and prefer the reporter's own log line when the split is uneven —
`INFO … Hidden layers were unevenly partitioned: [15,15,16,15]. This can be manually
overridden using the VLLM_PP_LAYER_PARTITION environment variable.`
(`utils.py:130-135`) is printed at every model build, so one grep on their serve log
beats the arithmetic.

If the key is **not** a stage start, stop here: the requested names are the worker's
own `get_kv_cache_spec()` output, so the gap is in registration, not ordering — go
to step 3. Do not read the executor to "see whether the sort happens"; that read
cannot separate the cases and is where this family burns turns.

## Step 2 — confirm the ordering path with an ablation, not a source walk

- Same model, `--pipeline-parallel-size` unchanged, **one node** (`mp` executor, or
  Ray with all workers on one IP): starts cleanly ⇒ the Ray re-rank path is the
  owner; still fails ⇒ step 3.
- `VLLM_RAY_BUNDLE_INDICES` pinning creation order to the IP order removes the
  permutation on the same cluster (documented workaround shape, `ray_executor.py:170-186`).
- Ask which rank printed the traceback and whether the other stages logged the same
  key: a foreign config means the *reader* is not the stage that owns the layer, so
  the key names a stage the failing worker does not own.

## Step 3 — if the fingerprint refutes ordering: registration, not ordering

`AscendMultiHeadLatentAttention` registers itself under `prefix`, and the inner
`MLAAttention` under `prefix + ".attn"` (`vllm/attention/layer.py:653-656` at
v0.12.0; downstream `vllm_ascend/ops/mla.py` mirrors it) — so a `.attn` key missing
while `…self_attn` exists means the inner layer was never built or is not an
`AttentionLayerBase`. Compare the two builders (`vllm_ascend/ops/mla.py` vs
`vllm/model_executor/layers/mla.py`) and check `models/__init__.py` registry hits
before blaming PP at all.

## Support surface to state in the answer

At v0.12.0rc1 the only Kimi-K2-Thinking recipe is
`docs/source/tutorials/Kimi-K2-Thinking.md`: **one** Atlas 800 A3 node,
`--tensor-parallel-size 16 --enable_expert_parallel --trust-remote-code
--no-enable-prefix-caching`, no `--pipeline-parallel-size` and no Ray block (it also
requires editing `quantization_config.config_groups.group_0.targets` to `["MoE"]`).
So multi-node PP is outside the documented shape: say "unvalidated configuration"
and give the confirmation step, rather than calling the crash a regression.

## Limits

Source-verified only — no PP cluster is reproducible from this snapshot, so the
verdict is "consistent with" plus the ablation, not a measured fix. If the reporter
cannot run the one-node ablation, ship the fingerprint result and the
`Hidden layers were unevenly partitioned` grep as the two facts that decide it.
