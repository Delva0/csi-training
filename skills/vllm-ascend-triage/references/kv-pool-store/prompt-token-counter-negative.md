# `Counters can only be incremented by non-negative amounts` on a KV-pool serve

## When this page applies

A pool-attached serve (any connector that supplies **external** KV: `AscendStoreConnector`,
Mooncake, a CPU-offload connector) answers requests for a while and then the **APIServer**
(never a worker) raises:

```text
metrics/loggers.py:1120 in record … self.counter_prompt_tokens_by_source[source][engine_idx].inc(…)
prometheus_client/metrics.py:339 in inc … raise ValueError('Counters can only be incremented by non-negative amounts.')
async_llm.py:707 in output_handler … logger.exception("AsyncLLM output_handler failed.")
```

The request streams die with it (`output_processor.propagate_error`), so the reporter files it
as "the service crashes". Do **not** treat that as a device, transport or store fault: the
deepest frame is the metric writer, and nothing was launched. **The line number in the paste
dates the formula** — `metrics/loggers.py:1120` beside a `counter_prompt_tokens_recomputed`
write (v0.18.0) is this page's arithmetic; `:1156` with no `recomputed` counter at all is the
rewritten v0.21.0 accounting, where the negative above cannot be produced (*Version scope*).
The report usually arrives with long prompts, a large `max_tokens`/`--max-model-len`, repeated prefixes and a guess that
`num_cached_tokens + num_external_computed_tokens > prompt_len` (recorded #8664, QWQ-32B
W8A8, TP2, `--kv-transfer-config` `{"kv_role": "kv_both", "backend": "mooncake"}`, 36 requests of
20-60 K tokens at 64 K output).

Not this page when the launch also has `--speculative-config` **and** `--async-scheduling`
(there `num_accepted = -1` produces the same `ValueError`, and the sibling symptoms are a
worker `KeyError` and `'NoneType' … sampled_token_ids`:
`../spec-decode-mtp/async-spec-bookkeeping-desync.md`), or when the deepest `vllm_ascend`
frame is inside `vllm_ascend/patch/`
(`../upstream-downstream-drift/patch-copy-vs-upstream.md` decides that one by a diff).

## The identity, and why the reporter's inequality is not it

At vllm **v0.18.0** (`bcf2be9`), `vllm/v1/metrics/stats.py:270-288`
`PromptTokenStats.update_from_output`:

```python
recomputed = 1 if (num_cached_tokens + 1 == prompt_len) else 0   # :280
self.computed           += prompt_len - num_cached_tokens         # :282
self.external_kv_transfer += num_external_computed_tokens
self.local_cache_hit     += (num_cached_tokens + recomputed
                             - num_external_computed_tokens)       # :284-286
```

So the counter goes negative **iff, for one iteration,
`num_external_computed_tokens > num_cached_tokens + 1`**. `prompt_len` only enters `computed`
and `total`, so the reporter's own `cached + external > prompt_len` cannot produce it — say
that plainly and give the real inequality instead. Two consequences that decide the
candidate list:

- `IterationStats` is rebuilt every iteration, so the negative comes from **one request's
  single output**, not from a cumulative drift.
- `is_prefilling` is cleared after the first processed output
  (`vllm/v1/engine/output_processor.py:621`), so each request contributes exactly once —
  at the output that carried its first new token.

## Why `cached` can lag `external`: the latch, then a preemption

`num_cached_tokens` is a **latch**, not a running value:

- `vllm/v1/request.py:150` initialises it to `-1`;
- `vllm/v1/core/sched/scheduler.py:806-807` (new request) and `:2045-2046` (async remote-KV)
  write it **only while it is still `< 0`** — the first schedule pass latches it, and it is
  never re-armed;
- `:941 _preempt_request` sets `request.num_computed_tokens = 0` and leaves
  `num_cached_tokens` alone;
- the next pass queries the connector again **because** `num_computed_tokens == 0`
  (`:598-602`), so a resumed request can receive a *larger* external hit than the latched
  count — and its first output then feeds the metric `cached < external`.

Worked numbers (prompt 60 000 tokens, block 128): first pass local 0 / external 0 latches
`cached = 0`; preempted under KV pressure mid-prefill; resumed with `external = 24 576` →
`local_cache_hit += 0 + 0 - 24576`. Pump those three numbers through the identity in one
`uv run --no-project python -` block if the reporter wants arithmetic, but the inequality
above is the answer.

Two other ways the latch goes stale, in rank order after preemption: `:2208-2210` truncates
`num_computed_tokens` **to** `num_cached_tokens` and decrements `num_external_computed_tokens`
when a connector reports invalid blocks (a *failed* load), and `:2161` reads the latch as the
"already computed" bound in that same path. Both need an async load or a load failure, so they
are only live when the launcher sets the connector's async option. Recorded precedent for that
lane: `#7871` — KV offloading with `kv_load_failure_policy=fail`, TP ranks disagreeing about the
load result, same `Counters can only be incremented` ValueError with an `External prefix cache
hit rate` line in the log; `#8086` is a GLM-5 report on the identical frame. So grep the
reporter's log for `kv_load_failure_policy` / `load_async` before choosing between the two
lanes — both end in this one counter.

## What the pool connector is not

`KVPoolScheduler.get_num_new_matched_tokens`
(`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:56-111` at
vllm-ascend v0.18.0rc1) is self-consistent: `need_to_allocate = num_external_hit_tokens -
num_computed_tokens`, floored at 0, and it subtracts the full-prompt-hit token by one
(`:86-87`). Fed to the scheduler, that keeps `cached == local + external` on the **first**
pass, which is exactly why the fault needs a second query. Its two defaults also prune the
candidate list in one read (`:42-52`): `load_async` is `False` unless passed in
`kv_connector_extra_config`, and `_discard_partial_chunks` is `True`. The reporter who never
set `load_async` cannot be in the `WAITING_FOR_REMOTE_KVS` lane.

## The two checks

1. **Was anything preempted?** `curl -s <host>:<port>/metrics | grep -E
   'vllm:num_preemptions'` (or the run's own `Preempted`/`preempted` lines). The mechanism
   predicts `> 0` together with the failing requests being the **later** ones under a memory
   bound the reporter's numbers already exceed (36 requests × 20-60 K prompts + 64 K outputs
   at `--gpu-memory-utilization 0.9`). Zero preemptions everywhere after the first
   failures? The latch branch is out — go to the async/invalid-block rows above.
2. **Does `--disable-log-stats` remove the crash?** `self.logger_manager` is built only when
   stats are on (`vllm/v1/engine/async_llm.py:164-166`) and `output_handler` guards the write
   with `if logger_ref[0]:` (`:699`), so the counter never runs and the same workload serves.
   That is the confirmation that this is a **metric-side** arithmetic defect and the request
   path was never wrong — it is not the fix to ship on its own.

## Ownership, fix, workarounds

Upstream vLLM at that pin owns both statements (the latch and the metric); vllm-ascend only
supplies the external count. Name that split before any connector or store read — a pool is
required to *reach* it, not to cause it. Fixes, cheapest first: clamp in
`PromptTokenStats.update_from_output`, re-arm `request.num_cached_tokens = -1` in
`_preempt_request` (or in the resume branch), or feed `min(external, cached + recomputed)`.

For the reporter: rung 1 is **remove the preemption** — cut `--max-num-seqs` (or
`max_tokens` / `--max-model-len`) so the long requests are not re-queued mid-prefill; rung 2
is `--disable-log-stats` while the accounting is fixed; rung 3 is the release that re-arms the
latch (re-read `stats.py:270-288` and `_preempt_request` at the newer pin rather than quoting
this page).

## Version scope

Verified at vllm **v0.18.0** `bcf2be96` + vllm-ascend **v0.18.0rc1** `99e1ea0f` (line numbers
re-read in that case view: `request.py:150`, `scheduler.py:806-807/941/2045-2046/2161/2208-2210`,
`stats.py:280-286`, `loggers.py:1120`). The record of the run that derived the mechanism is
`../evidence-workspace/recorded-cases-pd-kv-and-features.md` (#8664): correct mechanism, no
answer shipped.

**At vllm v0.21.0 `ad7125a4` the accounting was rewritten and this subtraction no longer
exists** — say that before ranking anything when the paste's line numbers are newer:

- `stats.py:243-273` `PrefillStats` + `set()`: `num_cached_tokens = num_local_cached_tokens +
  num_external_cached_tokens`, `assert num_cached_tokens <= num_prompt_tokens` (`:267`), and
  `num_computed_tokens = num_prompt_tokens - num_cached_tokens` (`:270`);
- `PromptTokenStats.update_from_output(prefill_stats)` (`:304-311`) only `+=`s the two
  non-negative fields from the scheduler: no `recomputed` term, no subtraction, and
  `recomputed_tokens` is gone from the dataclass;
- **the latch is gone**: `num_cached_tokens` occurs nowhere in `vllm/v1/request.py` or
  `vllm/v1/core/sched/scheduler.py`; the source counts are the scheduler's own per-iteration
  values (`scheduler.py:612-613`) and the counter write moved to `loggers.py:1156-1157`. The
  `max(…, 0)` clamp at v0.21.0 sits on a different metric (`loggers.py:1199`, the prefill-KV
  histogram over `FinishedRequestStats`).

Consequences, in order: (a) a traceback naming `loggers.py:1156`-era lines is **not** this
page's arithmetic — the candidates are a scheduler/connector count that is negative by itself
or another consumer, so do not hand over the latch fix; (b) `:1120` + `async_llm.py:707` is
v0.18.0-era and the mechanism above stands; (c) "is it fixed upstream" for a v0.18.0 reporter
is "the rewrite exists at vllm v0.21.0", not a PR number. Recheck with the behaviour greps,
never the numbers: `local_cache_hit +=` (what it adds decides the formula) and
`num_cached_tokens < 0` (absent ⇒ post-rewrite).

A DSA/MLA model's **index cache** (GLM-5/5.1, the reporter's "indexcache patch": at
vllm-ascend v0.21.0rc1 only `vllm_ascend/attention/sfa_v1.py`, `dsa_v1.py`,
`context_parallel/dsa_cp.py`, `models/deepseek_v4.py`) changes which blocks are available and
when they are freed, so it can *trigger* an external-vs-latched mismatch — it does not own the
metric arithmetic. Ask for the patch diff instead of reading that code.
