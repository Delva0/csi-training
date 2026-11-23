# Nothing is ever Put, no error anywhere: the `finalize` half of the connector was not ported (spec decoding)

Recorded vllm-ascend #8002 (title: "KV Pool not doing Put KV Cache in vLLM 0.18.0rc1 when used
together with Spec Decoding"), pins vllm-ascend `v0.18.0rc1` `99e1ea0fe685` / vllm `v0.18.0`
`bcf2be961200` — the reporter's own pair. The reporter had already localised it to a fix
(PR #7874) missing from the published image; the job of the answer is to prove that from the
pinned trees, name the exact missing call, and give the acceptance test.

## Signature — and what separates it from the two neighbours

- Pool attached (`AscendStoreConnector` alone or in a `MultiConnector`, `memcache`/`mooncake`),
  **speculative decoding on**, requests served normally, hit rate 0 %, and **no error text at
  all** — no `HcclBatchPut`, no `error_codes=[-1]`, no master-side `PutRevoke`.
- Ask for the master counters: this lane reads `PutStart` **0** as well as `PutEnd` **0** (the
  recorded report carried none, so treat them as the confirming scrape, not as given evidence).
  That zero pair is the discriminator against
  `hccl-onesided-store-put.md`, whose whole family is `PutStart`/`PutRevoke` > 0 with
  `PutEnd` 0 plus an `HcclBatchPut … hccl_result = 4` line: there the control plane ran and
  the write failed; here the save was never handed to the store at all.
- Also not `pool-lookup-overlap-zero-load.md` (that one is the read path with a healthy write
  path) and not `pool-attached-cold-request-cost.md` (that one has puts, they are just
  expensive).
- **Turning spec decoding off makes puts appear** — that single ablation is what makes this
  page the route instead of a store investigation.

## Mechanism (both sides verified at the pins)

Upstream split the connector lifecycle into a **defer** half and a **finalize** half for spec
decoding, because the drafter must run before the connector may conclude the step:

| | upstream vllm v0.18.0 `bcf2be9` |
|---|---|
| defer flag | `vllm/v1/worker/gpu_model_runner.py:3768` `defer_kv_connector_finalize = self.speculative_config is not None`, passed at `:3784` |
| finalize call | `:4047` `self.finalize_kv_connector()` after the draft proposal |
| what finalize does | `kv_connector_model_runner_mixin.py:81-89`: `kv_connector.wait_for_save()` **then** `kv_connector.clear_connector_metadata()` |
| what deferral switches off | the context manager's `finally`: `:116` `if wait_for_save and not defer_finalize: kv_connector.wait_for_save()` and `:128` `if not defer_finalize: kv_connector.clear_connector_metadata()` |

Downstream `v0.18.0rc1` copied **only the defer half**:

- `vllm_ascend/worker/model_runner_v1.py:1367` `clear_kv_metadata = self.speculative_config is None`,
  passed as `{"defer_finalize": not clear_kv_metadata}` at `:1382-1387` — same defer as upstream;
- after the draft block it restores only the *clear*: `:1580-1581`
  `if has_kv_transfer_group(): get_kv_transfer_group().clear_connector_metadata()`;
- `rg -n 'finalize_kv_connector' <vllm-ascend root>` → **zero hits in the whole tree** (also
  zero in `vllm_ascend/worker/v2/`, so the v2 runner is not the path).

So with spec decoding on, `wait_for_save()` is skipped for every step while
`clear_connector_metadata()` still runs: the step looks complete, the request is freed
normally, and nothing was ever enqueued for the store.

Why the *save* half is the half that matters for a pool: in the per-request (non-layerwise)
path, `AscendStoreConnector.wait_for_save` (`distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:187-195`)
is the **only** caller of `ConnectorWorker.wait_for_save`
(`.../ascend_store/pool_worker.py:329-352`), which is what records the NPU event and calls
`kv_send_thread.add_request(...)` / `add_stored_request(...)` — the actual Put submission.

## The check (two greps, no run)

```bash
# 1. does the downstream tree call the missing half at all?
rg -n 'finalize_kv_connector|wait_for_save' <vllm-ascend root>/vllm_ascend
#    definitions inside the connectors are NOT callers; the runner must call one
# 2. is the defer flag present without it? (both sides, same commit pair)
rg -n 'defer_finalize|clear_kv_metadata|defer_kv_connector_finalize' \
   <vllm root>/vllm/v1/worker/ <vllm-ascend root>/vllm_ascend/worker/model_runner_v1.py
```

A downstream grep that returns `clear_kv_metadata = self.speculative_config is None` plus
connector-side `def wait_for_save` and no runner-side caller is the finding. Check the class
bases before promising a one-line patch: `model_runner_v1.py:222`
`class NPUModelRunner(GPUModelRunner)` inherits `finalize_kv_connector` from the mixin, so the
call is available as `self.finalize_kv_connector()`.

## Repair and acceptance

Rung 1, one line: at `:1580` replace the bare
`if has_kv_transfer_group(): get_kv_transfer_group().clear_connector_metadata()` with
`if has_kv_transfer_group(): self.finalize_kv_connector()` — same position (after
`propose_draft_token_ids`, before `ModelRunnerOutput`), because upstream's order is
`wait_for_save` then clear. Reinstall (`pip install -e .`, or rebuild the image).

Acceptance is not "it starts": with spec decoding still on, the pool master must show
`PutStart` **and** `PutEnd` climbing, and a repeated identical prompt must move
`vllm:external_prefix_cache_hits_total` / `External prefix cache hit rate` off 0. A
spec-decode-off run only scopes the lane; it is not the fix.

## Ranked alternatives (only if the greps do not show the gap)

1. **Layerwise save instead of per-request save.** `ascend_store_connector.py:190-193` returns
   early from `wait_for_save` when `use_layerwise`, and the save then comes from
   `save_kv_layer` per layer — so the missing finalize does **not** explain a zero-put
   layerwise connector. Read the connector's `use_layerwise` / `kv_role` lines from the log.
2. **`kv_role: kv_consumer` without `consumer_is_to_put`** — the same method returns before
   saving for a consumer-only role (`:187-190`).
3. **The store's put data plane** — only with `HcclBatchPut … hccl_result = 4` /
   `Failed to put N keys out of N. error_codes=[-1]` in the log: `hccl-onesided-store-put.md`
   (one call: `scripts/store-put-code-check.sh <paste>`).

Store size, `global_segment_size`, RDMA/RoCE, HCCL ports and `--block-size` cannot repair a
path that never enqueues a save; do not start there.

## The generalisable reading

An upstream refactor that splits one call into **defer + finalize** (or acquire/commit,
begin/end) arrives downstream as two halves, and a main2main port that copies one half
produces a *silent* failure: no traceback, no error code, only an effect that does not happen.
For this shape the evidence is **absence**, so the first move is a grep for the call and its
caller — not a store, transport or scheduler investigation. The same asymmetry answers
"feature X does nothing" reports where the flag is honoured but the closing action is missing.

## Scope and recheck

Bound to vllm-ascend `v0.18.0rc1` (`99e1ea0`) with vllm `v0.18.0` (`bcf2be9`). The reporter's
statement that the published rc1 images predate the fix is consistent with the pin but not
independently verified here (PRs are outside the snapshot). After any bump, re-run the two
greps: a later tree may have added the call, and a *newer* upstream may rename the flag pair.
The layerwise/consumer branches above are the recheck condition — if either applies, this page
is not the answer.
