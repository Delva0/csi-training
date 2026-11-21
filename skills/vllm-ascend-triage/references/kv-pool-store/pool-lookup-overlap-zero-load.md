# Pool lookup hits but loads 0 tokens: contract upstream, one blocking RPC downstream

## Scope

A deployment with `AscendStoreConnector` (backend `memcache` or `mooncake`, usually
inside a `MultiConnector`, often PD) where the complaint is **performance**, not a
crash: enabling the pool makes throughput/TTFT worse than the local-prefix-cache or
mooncake-only baseline on a prefix-reuse workload, and the connector log for every
request says some version of

```text
Reqid: …, Total tokens 131072, kvpool hit tokens: 32768, need to load: 0
```

while `vllm:external_prefix_cache_hits_total` stays 0 and
`vllm:external_prefix_cache_queries_total` keeps climbing. Checked at vllm-ascend
`v0.23.0rc1` @`f4a08bdd`, vllm `v0.23.0` @`0fc695fc` (case #13356, 128K prompts).
A different fault — the store has the keys and *nothing* overlaps, hits are
genuinely 0 — is `cross-engine-lookup-zero-hits.md`; decide that first from the
log line above (a `kvpool hit tokens: N > 0` beside `need to load: 0` is this page).

## What is not the bug

`need_to_allocate = 0` when the pool's hit is contained in vLLM's local prefix
cache is the **upstream contract**, not an arithmetic mistake:

- Downstream `get_num_new_matched_tokens` (`pool_scheduler.py:478-565`) returns
  "the number of tokens that can be loaded from the external KV cache **beyond what
  is already computed**" (docstring `:484-493`), implemented as
  `if num_external_hit_tokens < num_computed_tokens: need_to_allocate = 0`
  (`:529-530`).
- Upstream passes `num_new_local_computed_tokens` in and treats the return as
  `num_external_computed_tokens` (`vllm/v1/core/sched/scheduler.py:617-636`), and
  derives the metrics from the same pair:
  `connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens`,
  `connector_prefix_cache_hits = ext_tokens`. Overlap therefore *must* show
  hits 0 / queries counting the whole non-locally-computed remainder — the metric
  panel in such a report is consistent, not contradictory.
- The upstream store scheduler has the identical shape and the identical
  subtraction at `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:74-113`
  (`:95-96`), so "the arithmetic is wrong" is not a downstream defect either.
  Do not spend the session re-deriving it.

## Where the cost actually is (verified differences vs upstream)

1. **The default (non-layerwise) hit check is a blocking ZMQ round trip over the
   whole prompt.** `LookupKeyClient` (`pool_scheduler.py:1082-1112`) opens a
   `zmq.REQ` socket on `ipc://$VLLM_RPC_BASE_PATH/lookup_rpc_port_{port}_dp_rank{N}`,
   encodes **every** `request.block_hashes` entry plus `kv_cache_group_ids`, sends,
   and `recv()`s (`:1107`) inside the scheduler step. At 128K tokens that is
   `131072 / block_size` hashes per request (1024 at `--block-size 128`), and the
   reply is then discarded whenever it lands at or below `num_computed_tokens`.
   Upstream's store scheduler pays the same full-range lookup
   (`store/scheduler.py:81-85` queries `request.block_hashes` from block 0), so the
   *cost model* is shared — what a report can attribute is the per-request RPC on
   the scheduler's critical path, not a vllm-ascend-only mistake.
2. **Only the store-scheduler path skips the locally computed prefix.**
   `_get_store_lookup_hit_tokens` uses
   `query_start_block = num_computed_tokens // self._block_size`
   (`pool_scheduler.py:255`) in non-layerwise mode, i.e. that path never queries
   the prefix it already has, while the ZMQ `client.lookup` path above has no such
   start offset. That asymmetry is the concrete thing to point at when the ask is
   "make the miss cheaper".
3. **Layerwise mode deliberately diverges from upstream: it loads anyway.**
   `force_layerwise_load = self.use_layerwise and num_external_hit_tokens > 0`
   (`:545`) keeps the load alive through `:546` and sets `can_load` in the
   `LoadSpec` (`:552`) even when `need_to_allocate == 0`, on the stated reason that
   layerwise data may not be in HBM. Upstream has no such branch. `use_layerwise`
   comes from `kv_connector_extra_config["use_layerwise"]`
   (`ascend_store_connector.py:80,86`), and memcache additionally routes to the
   GVA key shape (`use_gva_layerwise`, `:89`). So "the pool gives zero marginal
   benefit" is mode-dependent and worth one A/B, not an inference.

## Steps

1. **Classify the workload before reading code: overlap or genuine misses.** Take
   one request's `kvpool hit tokens:` / `need to load:` pair plus
   `external_prefix_cache_hits_total` vs `…_queries_total`. Hits > 0 with
   `need to load: 0` is this page; the design answer is that those tokens are
   already in HBM (re-loading them is wasted work), so the pool's marginal value
   for *this* workload is zero by construction.
2. **Say what the report actually measured.** A perf claim needs the same harness
   three ways — local prefix cache only, pool attached, mooncake-only — with the
   same input/output lengths, request rate and concurrency; a screenshot of the
   Grafana panel is not that. Ask for the `vllm bench serve` invocation and the
   three rows; a report whose "baseline" is a different workload cannot be graded.
3. **Then A/B the two real levers**, both one env/config each:
   `use_layerwise: true` in `kv_connector_extra_config` (loads in overlap, per
   item 3), and the connector removed (isolates the lookup RPC). Keep
   `load_async` (default `False`, `pool_worker.py:133`) in the diff: on the
   non-layerwise path it also decides whether the load blocks
   (`return need_to_allocate, self.load_async and not self.use_layerwise`).
4. **Only then consider a fix.** The reported shape supports "lookup cost is paid
   per request with no benefit in the overlapping regime", i.e. a scheduler/RPC
   design issue (skip locally computed blocks, batch lookups, or answer from the
   prefix-cache state before the RPC), not a correctness bug in the load mask —
   state that distinction, and that no load is ever issued for an overlapping hit.

## What to send back

The three benchmark rows with their exact commands, one request's full log trio
(`Total tokens` / `kvpool hit tokens` / `need to load`) from the pool-enabled run,
the `--kv-transfer-config` JSON (so `use_layerwise`/`load_async` are known), and
whether the decode node is the one printing the log (PD: `kv_role == "kv_consumer"`
short-circuits to `(0, False)` at `:495-496` before any lookup).
