# P node dies on `assert req_id in self.requests` (scheduler KV-xfer bookkeeping)

## When this page applies

A P/D deployment is serving requests and the **prefiller's** EngineCore dies with a
bare `AssertionError` whose last frames are

```text
vllm/v1/core/sched/scheduler.py:1584, in _update_from_kv_xfer_finished
    assert req_id in self.requests
AssertionError
```

with `vllm/v1/core/sched/scheduler.py:1181, in update_from_output` above it and
`vllm/v1/engine/core.py:443, in step_with_batch_queue` above that; the APIServer
then prints `EngineDeadError: EngineCore encountered an issue`, every
`Worker_TP*` line says `Parent process exited, terminating worker`, and the client
gets 500s. The decode side keeps running. This is not the transport lane and not a
KV miss: nothing failed to transfer — the **scheduler was handed a send-completion
for a request it had already deleted**.

## Step 1 — read the frames as a version and scheduler fingerprint

Verified at vllm v0.12.0 (`4fd9d6a85c00`) against the reporter's paste; re-grep per
version:

- `vllm/v1/core/sched/scheduler.py:1579-1585` — `finished_recving` ids are only added
  to a set, `finished_sending` ids hit `assert req_id in self.requests` and then
  `_free_blocks`.
- `scheduler.py:1353-1372` — `_free_request` asks the connector
  (`_connector_finished` → `delay_free_blocks`); when it is `False` the blocks are
  freed immediately, and `_free_blocks` is the only `del self.requests[request_id]`.
- `scheduler.py:1179-1181` — the call site inside `update_from_output`.
- `engine/core.py:443` — the `update_from_output` call inside
  `step_with_batch_queue`.

All four line numbers matching the pin means the reporter runs **unmodified upstream
`Scheduler`**: vllm-ascend's forks would show `vllm_ascend/core/recompute_scheduler.py`
or `scheduler_dynamic_batch.py` frames instead (`platform.py:300-312` selects them for
`recompute_scheduler_enable` / `SLO_limits_for_dynamic_batch`). Neither fork overrides
`_update_from_kv_xfer_finished` at v0.12.0rc1, so the same assert text can appear
through them — the **frame file**, not the message, tells you which scheduler ran.

## Step 2 — the frame that decides a launch flag, before any connector read

`step_with_batch_queue` only exists when the batch queue exists
(`engine/core.py:177-183`, `:203`), and `batch_queue_size` is
`model_executor.max_concurrent_batches`:

- `MultiprocExecutor.max_concurrent_batches` = `2 if scheduler_config.async_scheduling
  else parallel_config.pipeline_parallel_size`
  (`v1/executor/multiproc_executor.py:412-417`; uniproc: `uniproc_executor.py:57-59`).
- `async_scheduling` defaults to `False` (`config/scheduler.py:131`) and is only
  flipped on when it was passed explicitly (`config/vllm.py:530-580`).

So this traceback implies the prefiller was launched with **`--async-scheduling`**
(or PP>1, which also builds the queue without async scheduling). That is the cheapest
ablation in the whole case — turn it off on P, rerun the same traffic — and it needs
no connector source. vllm-ascend never sets it itself at v0.12.0rc1 (no assignment
outside `scheduler_config` consumers), but several vllm-ascend tutorials pass it, so
"we did not enable it" needs the serve command, not the reporter's recollection.

## Step 3 — the assert names no id: ask for the two DEBUG lines

`AssertionError` carries nothing. The id is only in the DEBUG records around it:

- `scheduler.py:1583` — `Finished sending KV transfer for request <id>` (printed per
  id just before the assert);
- the connector's delay record, e.g. at vllm-ascend v0.12.0rc1
  `vllm_ascend/distributed/mooncake_connector.py:917-919` —
  `Delaying free of N blocks for request <id>`.

One `VLLM_LOGGING_LEVEL=DEBUG` rerun on P until the crash gives both, and the
comparison is the diagnosis:

- **`Delaying free` present for that id** → P took the delayed-free path, kept the
  request in `self.requests`, then something reported its send completion twice (or
  after the first report already freed it). Owner: the worker-side report path
  (`model_runner_v1.py:1534-1545` builds `KVConnectorOutput` from
  `get_finished_kv_transfer`; `worker_v1.py:312-322` re-attaches it for non-last PP
  ranks) or the send thread's `get_and_clear_finished_requests`.
- **`Delaying free` absent for that id** → the request ended with a status other than
  `FINISHED_LENGTH_CAPPED`, so the connector returned `(False, None)` and the blocks
  were freed at once, while a send completion for it still arrived. Then the question
  is which path queued the send, and why the finish status was not the length cap.

## Step 4 — the delay condition is an exact status match

At vllm-ascend v0.12.0rc1 the Mooncake scheduler-side `request_finished`
(`vllm_ascend/distributed/mooncake_connector.py:895-919`) returns `(False, None)`
unless the request carries `kv_transfer_params["do_remote_decode"]` **and**
`request.status == RequestStatus.FINISHED_LENGTH_CAPPED`; only then does it record
`_reqs_need_send[id]`, which `build_connector_meta` hands to the worker and clears
(`:889-891`). A prefill request built by the tutorial proxy is `max_tokens=1`, i.e.
length-capped by construction — so a non-capped status on P means the request was
**aborted** (client disconnect, proxy timeout, `Force freed expired request`), which
is a different owner: pair this page with
`proxy-endpoint-vs-api-server-count.md` §1 and check the 500/timeout rate on the
proxy in front of P. Ignored-field warnings such as
`fields were present in the request but ignored: {… 'vllm_ascend_url_choice'}` are
evidence that a proxy is in the path, not a cause of the assert.

Both branches stay open in a report that pastes only the traceback. State them with
their one check instead of choosing.

## What to send back / what decides it

1. Both serve commands verbatim, in particular `--async-scheduling` on P and the
   `--kv-transfer-config` JSON (which connector: `MooncakeConnectorV1`,
   `MooncakeHybridConnector`, `AscendLLMDataDistConnector`, …).
2. P log at DEBUG from 10 s before the crash, with the `Delaying free` and
   `Finished sending` lines for the same request id.
3. Whether D logged/received a request with that id at the same time (a request D
   never picked up is the abort branch).
4. The rerun with `--async-scheduling` removed — if P survives, the batch-queue
   interleaving is the gate regardless of which connector reports late.

## Scope

Verified in the pinned trees of #6266: vllm v0.12.0 `4fd9d6a85c00` (scheduler, executor
and `core.py` line numbers above) and vllm-ascend v0.12.0rc1 `42ceaf08`
(`mooncake_connector.py:889-919`, `platform.py:300-312`). The batch-queue gate and the
`FINISHED_LENGTH_CAPPED` test are version-specific: re-grep
`_update_from_kv_xfer_finished`, `max_concurrent_batches` and the connector's
`request_finished` before reusing a line number. Cause was left open in the recorded
session — no reporter confirmation of a fix is available for this shape.
