# "How much KV cache is on the card, and how much is used?" — the surfaces, not a source walk

## When this page applies

No error, no crash: the body only asks **where** KV-cache (kvcache) size and utilization
are observable — `[Usage]: 怎么查看当前HBM上kvcache的使用情况`, `请问怎么查看当前HBM上kvcache的
总量、已使用量？curl metrics 的接口中有相关的统计信息吗？`, "is there a metric for KV cache
usage", or a dashboard owner whose gauge reads `0` / whose metric name is not in the
output. Recorded #8285 (v0.18.0rc1 + vllm v0.18.0, 8×910B3-64G, paste = env dump +
`npu-smi` only) — an answerable lookup that the recorded run turned into 16 tool turns of
`vllm/v1/core` and `vllm_ascend/worker/` reads.

The three numbers exist, they are upstream vLLM's, and the pin's own text settles every
one of them. **Deliver the table plus the two commands on the turn the shape is
classified; cap this page at ~3 calls**, one optional re-grep if the reporter's ref is
older than the pins below.

## Surfaces (verified at vllm v0.18.0 `bcf2be9` + vllm-ascend v0.18.0rc1 `99e1ea0`;
same names at vllm v0.19.1 `b1388b1` / vllm-ascend v0.19.1rc1 `da421af`)

| what is asked for | surface | anchor |
|---|---|---|
| pool size in **bytes** | startup `Available KV cache memory: X.XX GiB`, once per rank (`scope="local"`) | downstream `vllm_ascend/worker/worker.py:357-359` (`:392-394` at v0.19.1rc1); upstream twin `vllm/v1/worker/gpu_worker.py:456-460` |
| pool size in **tokens** | startup `GPU KV cache size: N tokens` and `Maximum concurrency for <max_model_len> tokens per request: X.XXx` | `vllm/v1/core/kv_cache_utils.py:1316`, `:1321-1326`, inside `_report_kv_cache_config` (`:1284`, called at `:1610`) |
| used ratio, **machine-readable** | `curl -s http://127.0.0.1:8000/metrics \| grep vllm:kv_cache_usage_perc` (0…1) | gauge declared `vllm/v1/metrics/loggers.py:481-489`, set from `SchedulerStats.kv_cache_usage` at `:1045` |
| used ratio, **human-readable** | periodic `... Running: N reqs, Waiting: M reqs, GPU KV cache usage: xx.x%, Prefix cache hit rate: yy.y%` | `vllm/v1/metrics/loggers.py:238-239`; `Preemptions: %d` only when non-zero, `:233` |
| the allocation knobs, from the same scrape | `vllm:cache_config_info` — every `CacheConfig` field as a label, incl. `num_gpu_blocks`, `block_size` | `loggers.py:1006-1028` (name `:1012`), labels from `vllm/config/cache.py:187-190` |
| per-block residency (allocated vs used vs evicted) | **off by default**: `--kv-cache-metrics` → `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, `vllm:kv_block_reuse_gap_seconds`; sampling `--kv-cache-metrics-sample` (default `0.01`) | `vllm/config/observability.py:48-53`; `vllm/engine/arg_utils.py:1145-1151`; histograms `loggers.py:934,947,960` |
| queue pressure beside the pool | `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:num_preemptions`, `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits` | `loggers.py:436,446,582,505,516` |

`/metrics` is mounted by `vllm serve` with **no flag**:
`vllm/entrypoints/serve/__init__.py:55-57` → `instrumentator/__init__.py:9-20` →
`instrumentator/metrics.py:41-44`. So "there is no metrics endpoint" is a different
report (a route question — `online-profiling-endpoints.md`), and a missing
`vllm:cache_config_info` is a scrape/log-level question, not a missing feature.

## The arithmetic, from the reporter's own two lines

    used bytes       ≈ kv_cache_usage_perc × Available KV cache memory
    bytes per token  ≈ Available KV cache memory / GPU KV cache size
    pool tokens      = num_blocks // len(kv_cache_groups) × min_block_size   (× pcp × dcp)

The first formula is the whole answer to "已使用量": `kv_cache_usage_perc` is a
**block-pool occupancy ratio** (`KVCacheManager.usage()` → `block_pool.get_usage()`,
`vllm/v1/core/kv_cache_manager.py:156-162`), not a share of device HBM. The third is the
line's own arithmetic (`kv_cache_utils.py:1300-1305`, with the `pcp × dcp` multiplier
logged at `:1306-1315`), which is why a hybrid/multi-group model's `GPU KV cache size`
is not `num_blocks × block_size`.

## Pitfalls that change the answer

1. **`npu-smi` cannot split KV out of HBM, and no metric can at these pins.** #8285's
   paste is 8 cards at ~61.6/65.5 GB whole-card with one 58.2 GB process row: weights +
   KV + activations + op workspace. There is **no** componentized memory metric —
   #2662 (`[Feature]: Expose Componentized GPU Memory Metrics`) is the open request for
   exactly the weights / KV ("allocated vs. used vs. evicted") / framework-overhead
   split, so quote the pool lines instead of a card-level number.
2. **Per rank, not per model.** `info_once(scope="local")` prints one line per worker:
   TP8 with one pool line per rank does not mean a request may use 8× that.
3. **An idle engine neither logs nor updates.** The periodic line is logged at DEBUG
   while idle (`log_fn = logger.debug if self.engine_is_idle else logger.info`,
   `loggers.py:217`), and gauges are only recorded when a step produced outputs
   (`vllm/v1/engine/llm_engine.py:321-330`), so an idle serve shows the last non-idle
   value (`multiprocess_mode="mostrecent"`), not a fresh `0`. Ask for a sample taken
   **while requests are in flight**.
4. **`--disable-log-stats` freezes the usage gauge while the route still answers.**
   `Scheduler.make_stats` returns `None` when stats logging is off
   (`vllm/v1/core/sched/scheduler.py:1910-1911`), the gauge is set only from
   `scheduler_stats` (`loggers.py:1045`), and the pin's own docstring for
   `--kv-cache-metrics` says it "Requires log stats to be enabled (i.e.,
   `--disable-log-stats` not set)" (`vllm/config/observability.py:49-51`). So "the gauge
   is `0.0` / never moves while `Running` > 0" is a flags question before it is an
   engine question — this is the opposite of the recorded #8285 draft, which told the
   reporter that `/metrics` survives the flag. The route survives; the value does not.
5. **The gauge was renamed.** `vllm:gpu_cache_usage_perc` is what recorded #2662's
   reporter saw, and that string occurs in **no** file of the v0.18.0 or v0.19.1
   exports — a dashboard or paste carrying it dates the reporter's release. Answer in
   the name their version uses; an all-zero old dashboard is the rename, not a broken
   engine.
6. **Multi-engine deployments label every vector.** DP / `--api-server-count` give one
   series per `engine` (+ `model_name`); read the labels, never compare ranks as if one
   were the total.
7. **A tracing flag that is force-cleared downstream is the same "nothing printed"
   shape** — `--enable-layerwise-nvtx-tracing` logs `relies on NVTX (NVIDIA Tools) and is
   not supported on Ascend. Resetting to False.` and clears itself
   (`vllm_ascend/platform.py:774-783` at v0.18.0rc1). The answer is "unsupported on this
   backend", not a logging-config hunt (`index.md` rule 3).

## What to send back when the paste has no startup block

#8285's paste had the env dump and `npu-smi` only — no engine log, so no pool size at
all. The first message is still complete: the two greps, the conversion, and the ask for
the three startup lines (`Available KV cache memory`, `GPU KV cache size`,
`Maximum concurrency`) plus one `/metrics` sample taken under load. Offline `LLM()` has
no HTTP surface: the same two startup lines come from stdout, and utilization only from
the periodic line (or `SchedulerStats` in-process), which is worth saying before the
reporter reports "offline has no metrics" as a defect.

Not this page: a KV budget near zero or negative at startup
(`../runtime-memory/kv-budget-per-instance-vs-device-global.md`), a TPOT step at one
concurrency (`../performance-throughput/kv-capacity-step-and-preemption.md`), or
`Event resources are insufficient` / `207007` (`../hccl-and-hardware/acl-event-resource-exhaustion.md`).
