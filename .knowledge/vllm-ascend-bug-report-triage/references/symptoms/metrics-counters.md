# Counter and metrics crashes: the serve process dies in the metrics writer

## Scope

A serving process exits with a `ValueError` raised from a `.inc(...)` on a Prometheus counter, usually
after the service has already answered requests and often only when a KV-transfer/pool/offload feature
is enabled. The crash site is **upstream vLLM accounting** (`vllm/v1/metrics/loggers.py`,
`vllm/v1/metrics/stats.py`), not the NPU compute path, so the deliverable is an accounting defect plus
the release that clamps it — not a kernel, scheduler or connector fault. Worked case: vllm-ascend
#8664 (`issue-ledger/reports-later.md`).

Not this page: a counter metric simply reading 0 or a wrong value with no crash (check the feature's own
gating), a stall after a connector action (`kv-pool-store-hang.md`), and a `ValueError` raised while
parsing `--additional-config` (a config-key fault, `../vllm-ascend/config-flag-lifetimes.md`).

## The signature

```text
ValueError: Counters can only be incremented by non-negative amounts
```

raised from `vllm/v1/metrics/loggers.py`, at the `prompt_tokens_by_source` loop in `record()`
(`vllm:prompt_tokens_by_source_total`). The value being incremented comes from
`IterationStats.prompt_token_stats`, i.e. `PromptTokenStats`, which splits a prompt into
`local_compute`, `local_cache_hit` and `external_kv_transfer`. The serving process dies with it even
though the trigger is only bookkeeping.

Read the frame before blaming the reporter's config: #8664's line was `:1120`, exactly
`self.counter_prompt_tokens_by_source[source][engine_idx].inc(pts.get_by_source(source))` at vLLM
`v0.18.0` — that line number is also the cheapest way to pin the upstream build they run.

## Mechanism (verified at vLLM `v0.18.0`)

```python
# vllm/v1/metrics/stats.py, PromptTokenStats.update_from_output
recomputed = 1 if (num_cached_tokens + 1 == prompt_len) else 0
self.computed += prompt_len - num_cached_tokens
self.external_kv_transfer += num_external_computed_tokens
self.local_cache_hit += num_cached_tokens + recomputed - num_external_computed_tokens   # can be < 0
```

`num_cached_tokens` is a snapshot taken at first scheduling and never reset on preemption, while
`num_external_computed_tokens` (what the KV connector/pool reports as already computed elsewhere) is
overwritten when the request is re-scheduled. As soon as the connector reports more external tokens
than that earlier local snapshot, `local_cache_hit` goes negative and the next `record()` calls
`inc(<negative>)` — the error the reporter sees. `[verified]` for the arithmetic at `v0.18.0`;
`[reported]` for "after several requests" and "not with the KV pool disabled", which is exactly what a
re-scheduling-under-pool explanation predicts.

Which Ascend path makes the external count exceed the snapshot is the part to state carefully: the
vLLM fix names preemption plus CPU offload, and the recompute scheduler / the connector's
`get_num_new_matched_tokens` are the local candidates, but a report that only shows the counter error
does not prove it. Say `hypothesis` and name the falsifying check rather than asserting a module.

## Fix lineage (verified with `git tag --contains` in the vLLM clone)

| ref | `vllm/v1/metrics/stats.py` |
| --- | --- |
| `v0.18.0`, `origin/releases/v0.18.0` | no clamp — the reporter's line |
| `v0.19.0`, `v0.19.1`, `v0.19.1rc0` | `local_cache_hit += max(0, (num_cached_tokens + recomputed - num_external_computed_tokens))` plus a `FIXME` explaining the preemption underflow |
| `v0.20.0rc1`+ | reworked into `PrefillStats` + `assert num_cached_tokens <= num_prompt_tokens` (#37460) |

| commit | PR | what it does | first release tag |
| --- | --- | --- | --- |
| `91e4521f9f` | #37160 | adds `max(0, …)` plus the FIXME that documents the underflow after preemption | `v0.19.1rc0` |
| `d3af8c1831` | #37460 | replaces the fields with `PrefillStats` and the assert; removes the clamp | `v0.19.2rc0`, `v0.20.0` |

**Check the hunk, not the subject.** #37160 is titled "[Feat][v1] Simple yet General CPU KV Cache
Offloading" and looks unrelated; its diff contains the clamp, so a maintainer's "fixed by PR #37160,
try v0.19.1rc1" is correct and checkable. The v0.19 branch carries the clamp as the cherry-pick
`1dbbafd3f3`, which is why `merge-base --is-ancestor 91e4521f9f v0.19.0` says *no* while the tag's tree
does contain it — and why "the real fix is #37460" is not the right correction
(`../workflow/fix-discovery.md`).

## Workaround ladder

- **Upgrade** to a build containing either commit (vLLM ≥ `v0.19.1rc0`, and the vllm-ascend release
  whose paired tag pulls it; for the fundamental fix `v0.19.2rc0`+). Verify the pairing at the plugin
  tag — the plugin's version prefix says nothing about the vLLM it installs, and an image ships its own
  wheel.
- **Disable the pool/offload path** (remove `--kv-transfer-config`), which the reporter themselves
  showed as the control: the crash stops and the cross-instance cache reuse with its TTFT win goes
  with it. A bisect tool that doubles as a workaround.
- **Reduce the trigger** (hypothesis, untested): stop preempting requests whose prefix came from the
  external KV source — lower `--max-num-seqs`, smaller `--max-num-batched-tokens`, or no offload
  connector. The metric only underflows when a resumed request re-reports external tokens.
- Not a workaround: raising `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`, changing `BALANCE_SCHEDULING`, or any
  metrics flag — nothing in the counter path reads them. Do not offer "ignore the metric" either: the
  exception kills the frontend process.

## Falsifying check

Reproduce with the pool on, then print the disagreement instead of inferring it: log
`num_cached_tokens`, `num_external_computed_tokens` and `prompt_len` in `update_from_output` at the
reported ref, or run the same workload with preemption disabled (more KV headroom, lower
`--max-num-seqs`). If `local_cache_hit` never goes negative, this class is not the bug and the negative
increment comes from another counter. A `max(0, …)`/`PrefillStats` construction already present at the
reporter's paired vLLM ref means the trigger needs a source the code does not have there — re-read
`vllm_ascend/core/recompute_scheduler.py` and the connector's `get_num_new_matched_tokens`.

```bash
git -C /workspace/repos/vllm-project/vllm show <paired-vllm-tag>:vllm/v1/metrics/stats.py \
  | grep -n -A4 'local_cache_hit +='
git -C /workspace/repos/vllm-project/vllm log --oneline -S'local_cache_hit += max' \
  origin/releases/<line> -- vllm/v1/metrics/stats.py
```
