# llmdatadist PD: the 120 s "KV Cache transfer done signal" line, and a slow TTFT

## Scope

PD disaggregation on the in-tree `LLMDataDistCMgrConnector` (LLM-DataDist). The
deployment is up and requests are served, and the report is either **"TTFT extremely
slow"** with the prefiller printing

`Some requests in prefill node fail to receive KV Cache transfer done signal in 120s.
If a greater mean TTFT is acceptable, you can 'export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600'`

or that warning alone, with no other error. Verified at **v0.9.1 `0740d10`** (SOURCES
`3062`); the reporter's `llmdatadist_c_mgr_connector.py:886` is the pinned `:887`, so
their build matches this pin. Not this page: no request ever reaching D
(`proxy-endpoint-vs-api-server-count.md`), connect/handshake failures
(`transport-topology.md`), a KV pool's own timeouts (`kv-pool-store-segments.md`), the
Mooncake connector's separate `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` (480 s) line.

## Rule 1: that line is P-side expiry bookkeeping, not a failed transfer

- The prefiller's scheduler registers the request when prefill finishes:
  `request_finished` (`llmdatadist_c_mgr_connector.py:261-299`) sets
  `delay_free_blocks = len(computed_block_ids) > 0`, logs
  `Delaying free of N blocks for request ...` (`:283`), records
  `_reqs_need_send[req_id] = time.perf_counter() + VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`
  (`:287`) and advertises `remote_host`/`remote_port`/`remote_tp_size` back to D.
- `VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` defaults to **120 s** (`envs.py:140-141`).
- The P **worker**'s `get_finished` (`:875-899`) walks `reqs_to_send` and, for each entry
  whose expiry has passed, prints exactly that warning and then drops the id.

So the line means "this P worker still holds a request whose done signal it has not
seen, and the timeout expired". It fires *at* the timeout — 120 s after the request was
prefill-finished — and it is bookkeeping, not a transfer error. Ask for the measured
TTFT and the warning timestamps before accepting it as the mechanism: if TTFT is a few
seconds while the warnings appear 120 s later, the two are different symptoms and the
slow TTFT is elsewhere.

## Rule 2: which P ranks warn is decided by the port arithmetic

Three offsets, all in `llmdatadist_c_mgr_connector.py`, `VLLM_LLMDD_RPC_PORT` default
5557 (`envs.py:134-135`):

| who | formula | line |
|---|---|---|
| P scheduler advertises `self.port` (goes into `remote_port`) | `dp_rank_local*tp_size + RPC_PORT` else `tp_size + RPC_PORT` — **no `tp_rank`** | `:184` |
| P worker listener | `RPC_PORT + local_dp_rank*tp_size + tp_rank` else `RPC_PORT + tp_size + tp_rank` | `:359` |
| D metadata request | `remote_port + tp_offset`, `tp_offset = tp_rank % remote_tp_size` | `:817-819` |
| D done signal | `send_finish_to_remote(remote_ip, remote_port, request_id)` — the **bare** port, payload `[request_id, tp_rank, tp_size]` | `:871`, `:787-800` |

Every done signal therefore lands on the listener of the P worker with `tp_rank == 0`;
the other `tp_size-1` P workers register the same request in their own `reqs_to_send`
and can never receive one. The recorded #3062 paste is exactly that shape: the same
second, `rank=7`, `rank=6`, `rank=5`, one identical line each.

Read: warnings from the **non-zero** P ranks are expected and are not evidence that D
failed to transfer anything. The informative case is **rank 0 warning too**: then
`_increment_task_count` (`:409-425`) never collected `decode_tp_size` distinct decode tp
ranks — it drops repeats with `Received duplicate done signal for request ... Ignoring`
— which happens when two decode ranks collapse onto one P port
(`tp_offset = tp_rank % remote_tp_size`) or the two sides disagree about tp sizes.

## Rule 3: the cost of the window is held blocks, so it needs a saturated P cache

`delay_free_blocks` keeps the prefilled blocks until the `finished_sending` id reaches
the scheduler (upstream `vllm/v1/core/sched/scheduler.py:1042`, mirrored by
`vllm_ascend/patch/platform/patch_0_9_1/patch_core.py:179`). Until then P cannot reuse
them, so retention alone raises TTFT only while P's kv cache is near saturation and
later prefills queue.

Cheap discriminators, all reporter-side:

- P-side free-block/queue depth across the slow window, and whether TTFT tracks
  `max-num-seqs` / prompt length (queue) instead of staying flat (per-request cost).
- `VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600` is the message's own advice and it *extends*
  retention: if adopting it makes TTFT worse, queueing behind held blocks is the
  mechanism; if TTFT is unchanged, the delay is on the D side and the knob is irrelevant.

## Rule 4: where a D-side delay can come from (checks, cause left open)

- `self.executor = ThreadPoolExecutor(1)` per worker (`:324`): every `pull_blocks` runs
  in one thread, so pulls serialize per worker and the k-th request's first token waits
  for k-1 pulls. Discriminator: TTFT scales with in-flight concurrency and returns to
  baseline with a single request.
- `add_remote_agent` (`:626-760`) builds the rank table and then waits in
  `while True: query_register_mem_status(comm_id)` with `time.sleep(1)` (`:733-745`),
  once per remote cluster (cached in `linked_cluster`), so the cost is paid by the first
  request(s) per link. Discriminator: slow only for the first requests after start, then
  flat. This page owns the *transient* cost; a loop that **never** exits on that same log
  line is `llmdatadist-register-mem-loop.md` (#2498, reporter-confirmed: two container-network
  engines on one host, `--net=host` the fix).
- Version note worth one line, not a cause: at this pin
  `examples/disaggregate_prefill_v1/run_server.sh:29` advertises
  `kv_connector_module_path: vllm_ascend.distributed.llmdatadist_connector_v1_a3`, a
  module absent from `vllm_ascend/distributed/` (the README and the tree carry
  `llmdatadist_c_mgr_connector`).

## Delivery order

1. Verdict: the 120 s line is P-side expiry bookkeeping emitted by design on the
   non-zero tp ranks; it is not itself the TTFT mechanism.
2. Ask for: measured TTFT (p50/p90, and whether the slowest requests are the first ones
   after start), in-flight concurrency, whether **rank 0** also warns, and the P-side
   free-block trend.
3. Then the two discriminators above — concurrency sweep and first-request-only test —
   before any further source reading.
4. Scope the claim to v0.9.1 `0740d10`; the snapshot carries no fix, and whether the
   single-listener design is intended (one P port serving all D ranks, with
   `_increment_task_count` waiting for `decode_tp_size`) is not settled by the pin.
