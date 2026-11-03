# Observing KV-cache / HBM usage: metrics, log lines, and `num_gpu_blocks="None"`

Usage-question shape (`https://github.com/vllm-project/vllm-ascend/issues/8285`, vLLM 0.18.0 +
vllm-ascend 0.18.0rc1, one A2 node, 8×910B3): "how do I see the total and used KV cache on HBM —
does `/metrics` have it?" The thread answers half of it: a maintainer replies that start-up prints
the total block count and the server-side log prints per-step usage. The reporter then greps
`/metrics` for `block` and gets
`vllm:cache_config_info{...num_gpu_blocks="None",num_cpu_blocks="None",kv_cache_memory_bytes="None",cpu_kvcache_space_bytes="None"...} 1.0`
twice (one per `engine`), and asks which metric carries the total.

This file answers the usage/observability question only. A 0 % prefix-cache hit rate, or an
accuracy change that tracks cache hits, belongs to
[prefix-cache-and-chunked-prefill.md](prefix-cache-and-chunked-prefill.md) (block sizing, the
hit-length alignment, per-runner branches); do not re-derive that from the gauges here.

## Answer from the metrics/log surface, not from code archaeology

Verified in the clone at vLLM `v0.18.0` (the tag matching the report):

- **Used share**: gauge `vllm:kv_cache_usage_perc`, one series per `engine` label, documentation
  "KV-cache usage. 1 means 100 percent usage." (`vllm/v1/metrics/loggers.py:481-488`), set from
  `scheduler_stats.kv_cache_usage` (`:1045`). It is the fraction of that engine's KV blocks in use,
  not a byte count.
- **The same number in the log**, once per stat interval, beside the hit rate:
  `Engine 000: ... GPU KV cache usage: 12.3%, Prefix cache hit rate: 90.9%`
  (`loggers.py:238-241`). This is the maintainer's "log on server side". On NPU the line still says
  "GPU"; that is upstream wording, not a CUDA-only feature.
- **Total**: start-up lines `GPU KV cache size: <N> tokens` and
  `Maximum concurrency for <N> tokens per request: <x.xx>x`
  (`vllm/v1/core/kv_cache_utils.py:1316,1322`), plus the plugin's own
  `Available KV cache memory: <X> GiB` emitted after memory profiling
  (`vllm_ascend/worker/worker.py:358`; `vllm_ascend/_310p/worker_310p.py:97` on 310P). These are
  different quantities: the GiB line is the budget the profiler granted, the token line is what
  block/alignment sizing made of it.

So the short answer to this class: `vllm:kv_cache_usage_perc` for used, the start-up `GPU KV cache
size` / `Available KV cache memory` lines for total, and the periodic
`GPU KV cache usage:` line for a human-readable trace.

## `vllm:cache_config_info` is not the total; `None` is expected on V1

- The info metric is a permanently-1 gauge whose labels are `CacheConfig.metrics_info()` =
  **every** field of the config object stringified (`vllm/config/cache.py:187-190` at v0.18.0), so
  any field the logger's config object never received prints as `"None"`.
- `num_gpu_blocks` is assigned **inside the EngineCore**, after
  `generate_scheduler_kv_cache_config` (`vllm/v1/engine/core.py:268`), while the Prometheus stat
  logger and the `log_engine_initialized()` call that emits the info metric are constructed in the
  **frontend** from the frontend's own `vllm_config` (`vllm/v1/engine/async_llm.py:166-174`;
  `metrics/loggers.py:1006-1030,1203`). The frontend's copy keeps its pre-init `None` values.
- Consequence: do not file "`num_gpu_blocks` is `None`" as a defect for a V1 report and do not send
  the reporter to grep `block` — answer with the gauge and the start-up lines. If a block-count
  metric is genuinely required, that is a feature request against the info metric and should be
  stated as one.
- The same reasoning covers the other `None` labels in that output (`num_cpu_blocks`,
  `kv_cache_memory_bytes`, `cpu_kvcache_space_bytes`): unset config fields, not failed collection.

## HBM (device) versus KV cache

`npu-smi info` HBM-usage / process-memory columns are device totals: weights, activations, workspace
and KV cache together. The KV-cache share is the profiled available-KV-cache figure above, not
`HBM used - weights`; do not subtract device numbers to "measure" KV cache, and do not compare two
runs' HBM-usage numbers as if they were KV-cache usage.

## Why every TP rank's HBM looks full ("the whole model is on each card")

A report that reads `npu-smi` HBM totals and concludes the full model is loaded on every card
usually mistakes the KV-cache reservation for weights. In the V0 `determine_num_available_blocks`
(`vllm_ascend/worker/worker.py:269-272` at v0.7.3.post1) and the V1 `determine_available_memory`
(`worker_v1.py:211-212`), each rank allocates KV blocks from
`total_npu_memory * gpu_memory_utilization - peak_memory`, where `peak_memory` is that rank's
weights plus activations. With the default
`gpu_memory_utilization=0.90` (`vllm/engine/arg_utils.py:121` at vLLM v0.7.3), every TP rank fills
the device to the same fraction regardless of its 1/TP weight shard. The maintainer's formula in
vllm-project/vllm-ascend#1316 is
`kvcache size = (total memory size - model size) * gpu_memory_utilization`.

A V1 rank that computes **zero or negative** free memory logs `Available memory: 0, total memory:
<bytes>` and the engine then dies in upstream `check_enough_kv_cache_memory` with `ValueError: No
available memory for the cache blocks. Try increasing ...` (vLLM `v0.11.0`:
`vllm/v1/core/kv_cache_utils.py:678`, raise at `:698`, reached from `v1/engine/core.py` KV-cache
init). The plugin clamps the subtraction at 0 (`max(..., 0)`, `worker_v1.py:255` at the same tag),
so that line is the profiling budget running out, not a verdict that the cards are too small: the
raised `peak_memory` is a lifetime `allocated_bytes.all.peak` plus the non-torch remainder, so a
weight-loading peak or a large profile-run dummy counts. Raising `gpu_memory_utilization` only helps
if the peak sits between the old and new budgets — the recorded hybrid-model case where the reporter
had already tried 0.9 is in [qwen3-next-family.md](qwen3-next-family.md).

Use the per-rank `Loading model weights took %.4f GB` line
(`vllm_ascend/worker/model_runner.py:907` at v0.7.3.post1) and the engine-init
`tensor_parallel_size` to test sharding: TP>1 should log roughly 1/TP of the full checkpoint,
while a DP replica with no TP loads the full model by design and `tensor_parallel_size` stays 1
(the upstream default, `vllm/engine/arg_utils.py:113` at v0.7.3). HBM totals alone cannot
distinguish the two.

## Boundaries

- Everything here was read from the tag's source and the thread's pasted metrics; no run was
  executed. The metric-name and `cache_config_info` statements apply to the V1 engine line
  (0.18.x); the HBM-full section was read at v0.7.3.post1 and vLLM v0.7.3. Re-read
  `vllm/v1/metrics/loggers.py` at the reported tag before quoting a metric or log-line name — the
  names and the `cache_config_info` construction both move between releases.
- The `None` explanation is a code-path conclusion (frontend logger versus EngineCore assignment)
  that is consistent with the reporter's output; it is not a maintainer statement and not an
  intent claim.
