# A PD score gap that tracks KV/memory saturation: the decode node hands the request back for recompute

Verified at the case pin vllm-ascend **v0.13.0rc3** (`bbe9bc0d`) + vllm **v0.13.0**
(`72506c98`), recorded for #8844, whose reporter supplied the resolution after the
recorded run had answered. The run's own versions (vllm / vllm-ascend **0.18.0**) are
not in the snapshot; the flag, the scheduler fork and the proxy branch below all exist
at the pin, so the mechanism is source-verified *there* and must be re-grepped at the
reporter's ref (§*Version scope*).

## When this page applies

A **score** report on a P/D-separated deployment — GPQA / MMLU-style aggregate below
the paper or the reference image, nothing crashing, answers fluent — where the
reporter attaches a **state** to the loss: "accuracy degradation occurs when the
output length is long, concurrency is high and GPU memory stays saturated", "a
potential mitigation is to avoid fully saturating GPU memory on the D node". And the
launch's `--additional-config` carries `recompute_scheduler_enable: true`.

Then the recompute hand-back is rung 1, ahead of every harness-measurement row.
#8844's reporter later wrote: *"This issue arises from recalculations after the gpu
memory usage is fully utilized. Disabling recalculations eliminates this problem."*

Not this page: 乱码 / garbled / repeated replies once the decode KV cache is full
(#5438, corruption end, `garbled-long-context-output.md` and the PD pages); a
**latency** step at one concurrency from the same exhaustion
(`../performance-throughput/kv-capacity-step-and-preemption.md`, which prices the pool
and owns the preemption reading); a score gap with no saturation statement in the
report (`harness-vs-engine-attribution.md`, `quantized-sibling-score-gap.md`).

## Step 1 — Read the flag off **both** serve commands

`recompute_scheduler_enable` is an `additional_config` bool, default `False`
(`vllm_ascend/ascend_config.py:91-92`; docs table
`docs/source/user_guide/configuration/additional_config.md:40`). When true,
`vllm_ascend/platform.py:330-334` swaps in `RecomputeSchedulerConfig`, i.e.
`vllm_ascend/core/recompute_scheduler.py` (883 lines, an `Adapted from
vllm/v1/core/sched/scheduler.py` fork); `async_scheduling` selects
`AsyncRecomputeScheduler` instead (`:61-79`). #8844 sets it on **both** nodes.
Quote both nodes, because the behaviour below exists only on the kv **consumer**.

The feature is documented as a PD feature, not as an accuracy lever: "enables the
recomputation scheduler. When the Key-Value Cache (KV Cache) of the decode node is
insufficient, requests will be sent to the prefill node to recompute the KV Cache. In
the PD separation scenario, it is recommended to enable this configuration on both
prefill and decode nodes simultaneously"
(`docs/source/tutorials/DeepSeek-V3.1.md:585`), and the pin's Mooncake multi-node PD
tutorial sets it in both of its blocks (`pd_disaggregation_mooncake_multi_node.md:277,337`).
So a reporter who follows a tutorial has the flag on: turning it off is a deliberate
deviation from the documented recipe, and the answer should say that it costs the
KV-shortage resiliency the flag buys (the request restarts at the prefiller instead of
being preempted locally), not present it as a free win.

## Step 2 — What the fork does when the pool is exhausted

| Site (pin) | Fact |
|---|---|
| `recompute_scheduler.py:217-239` | `kv_cache_manager.allocate_slots(...)` returns `None` while a running request needs one more block. With `kv_transfer_config` set **and** `not transfer_config.is_kv_producer` — i.e. on the **decode** node — the fork does **not** preempt: it `running.pop()`s a request, frees its blocks and appends `RecomputeReqInfo(request_id, output_token_ids, client_index)`. If that request is the one being scheduled, it `break`s, so the request also leaves this step's batch |
| `recompute_scheduler.py:700-712` | `update_from_output` turns each `RecomputeReqInfo` into an `EngineCoreOutput` with `finish_reason=STOP`, `new_token_ids=[output_token_ids[-1]]`, `stop_reason="recomputed"` |
| `recompute_scheduler.py:270` | the `else` branch — a kv **producer**, or any serve without a transfer config — takes the ordinary `_preempt_request` path (upstream semantics, `num_computed_tokens = 0`). So this page's behaviour needs *both* PD-decode role and the flag |
| no `_preempt_request`, no metric write | the branch increments **no** preemption counter and logs nothing above DEBUG, so `Preemptions: N` (which upstream emits only when non-zero) stays silent and the server log is clean. The only surfaces are the response's `stop_reason` and its `completion_tokens` |

The semantic: the decode node **abandons** the request and returns it to the front
end, which sends prompt+generated text back to a prefiller — the "recalculation" the
reporter names — and generation restarts from a fresh prefill. It fires when the
decode pool is full, which is exactly the state the reporter describes.

## Step 3 — Why that lowers a score: three channels, one check each

- **(a) A front end that does not know `"recomputed"`** treats it as a normal stop:
  `finish_reason: "stop"` mid-answer, `completion_tokens` short, the emitted tokens
  correct, no final answer. An extractor scores it wrong, and nothing in an aggregate
  table distinguishes it from truncation.
- **(b) Ascend's own example proxy does know it, and re-issues from *text*** —
  `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:767-782`:
  `prompt = origin_prompt + generated_token` (string concatenation of the detokenized
  continuation) and `max_tokens = origin_max_tokens - completion_tokens + retry_count`,
  then `_handle_select_instance(...)` re-routes it. Two consequences: the continuation
  is re-prefilled from a **re-tokenized string** (token boundaries, chat template,
  special tokens round-trip), and every retry **shrinks** the remaining budget by what
  was already generated, so a long-CoT sample that recomputes twice can be cut before
  it emits its answer.
- **(c) The re-issued request is a new request** — KV and sampler state are rebuilt, so
  at `temperature > 0` the trajectory after the recompute is not the one the run would
  otherwise have had. Not sufficient for garbage; sufficient to move an aggregate.

The report settles *that* the recompute path owns the loss (flag off removes it), not
which channel dominates. Say that rather than picking one.

**The single check**: per-sample dump of the scored run — `stop_reason`,
`finish_reason`, `completion_tokens` — counting samples that ended on `"recomputed"`
(or ended short of the CoT length), correlated with `vllm:kv_cache_usage_perc ≈ 1.0`
on the decode node. Zero recomputes retires this page and the ordinary score ladder
resumes.

## The ablation, in order

1. `"recompute_scheduler_enable": false` on **both** nodes, restart, same harness and
   subset. Reporter-confirmed on #8844; the docs default is `False`, so omitting the
   key is the same arm.
2. Same run with the **pressure** removed and the flag kept: decode node
   `--gpu-memory-utilization` 0.92 → 0.85 and `--max-num-seqs` 128 → 64, i.e. reserve
   headroom so the pool never fills. Recovery in **both** arms is the report's own
   reading — the recompute path owns the loss and the trigger is the pressure. Recovery
   in arm 1 only means the flag does something beyond the full-pool path (it is firing
   before the pool reads 100 %, e.g. on fragmenting allocation): the flag-off arm is
   still the answer, but the mechanism sentence must stay weaker. Recovery in neither
   retires this page.
3. Only then the ordinary score ladder: paper number, sampling params, sample count and
   repeats (`harness-vs-engine-attribution.md`; this lane's 2-3-point run-to-run floor
   `quantized-sibling-score-gap.md`), reasoning-output routing
   (`reasoning-output-routing.md`), MTP + `--async-scheduling`
   (`../spec-decode-mtp/accepted-token-row-snapshot.md`), quant vs BF16 on the same
   checkpoint.

Do **not** ship arm 2 as the fix — it hides the trigger, and any long-context or
high-concurrency run will hit it again.

## Traps, from the recorded run

- "Saturation is a performance concern, not a correctness one" is the wrong default
  **when the launch carries a flag whose documented trigger is that saturation**. The
  #8844 run ranked measurement terms, reasoning-output routing and MTP +
  `--async-scheduling` above it and filed KV saturation as "a correctness trigger, not
  a fix" — while the flag sat in the reporter's own paste on both nodes and the
  reporter's hypothesis named the memory-full state. One rule for the future: **when a
  hypothesis names a state and the config carries a flag gated on that state, the
  flag-vs-state control is rung 1.**
- `Preemptions: 0` and a clean log exclude nothing here (§Step 2, last row).
- The deduplication catch is real but not a cause: #8844's `--additional-config`
  repeats `multistream_overlap_shared_expert` (`false`, then `true` → JSON last-wins =
  `true`). One line in the answer, not a ranked cause.
- A score report on a **w8a8** GLM-5.x PD serve invites the quant lane; keep it as a
  row, since #8844 has no BF16 control and no quoted reference number either.

## Recorded case

**#8844** — reported vllm 0.18.0 / vllm-ascend 0.18.0, 4-node PD (GLM-5/5.1
`/mnt/weight/GLM-new-w8a8`, `MooncakeConnectorV1`, `use_ascend_direct`, DP2 × TP16,
`--enable-expert-parallel`, `deepseek_mtp` ×3, P eager / D `FULL_DECODE_ONLY` +
`--async-scheduling`, `--gpu-memory-utilization` 0.95 P / 0.92 D, both nodes
`"recompute_scheduler_enable": true`): GPQA-diamond 67.17 (title) / 67.8 % (body)
"lower than the result reported in the paper", plus the two reporter hypotheses — TP8/
TP16 hurts, and the loss appears with long output × high concurrency × saturated
memory. Resolution: recompute after full memory utilization; disabling it removes the
problem. The run shipped a ranked answer (measurement terms, truncation/routing,
MTP+async, P/D recipe asymmetry, KV saturation, TP16-vs-DP) without ever naming the
recompute flag, and had already found the flag's name in a grep it then did not follow.

## Version scope and what to send back

Re-grep at the reporter's ref before quoting any line: `rg -n 'recompute_scheduler_enable' vllm_ascend/` (owner: `platform.py`, `ascend_config.py`),
`rg -n 'RecomputeReqInfo|recomputed' vllm_ascend/core/recompute_scheduler.py`, and
`rg -n 'recomputed' examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
or whichever front end the deployment uses. Ask for: both serve commands as executed,
the per-sample `stop_reason`/`completion_tokens` dump, the decode node's
`vllm:kv_cache_usage_perc` during the scored run, the proxy's request log (retries), and
the reference number with the harness's sampling params.
