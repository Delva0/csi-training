# LLMDataDist (llmdatadist) PD connector: the prefill holds blocks until the done-signal timeout

## Scope

PD disaggregation on a build whose `--kv-transfer-config` names **`LLMDataDistCMgrConnector`** (the
`llm_datadist` "C-mgr" connector, which exists up to `v0.11.0`), where the prefill role logs

```text
(VllmWorker rank=7 pid=7692) WARNING 09-20 16:26:44 [llmdatadist_c_mgr_connector.py:886]
  Some requests in prefill node fail to receive KV Cache transfer done signal in 120s. If a greater mean
  TTFT is acceptable, you can 'export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600' (10 minutes) to relax the
  timeout condition.
```

and the user-facing symptom is **"TTFT extremely slow"**, not a crash and not a hard hang. Worked case:
vllm-ascend #3062 (`v0.9.1`, 910C, 16 cards, `llm_datadist` P/D split). The service serves; nothing dies.

Not this page: the Mooncake connectors' connect/data-stage failures (`mooncake-transfer-errors.md`), the
silent mid-run freeze (`pd-collective-hang.md`), a pool load that kills a worker
(`kv-pool-store-hang.md`), or a proxy mismatch (`pd-proxy-examples.md`). The connector **name** in
`--kv-transfer-config` decides which page, so read it first.

## Common rules

- **Locate the file before `git show`.** At `v0.9.1` the module is
  `vllm_ascend/distributed/llmdatadist_c_mgr_connector.py` — **not** under
  `distributed/kv_transfer/`, where every later connector lives. One
  `git ls-tree -r --name-only v0.9.1 | grep -i llmdatadist` answers the layout (it prints the C-mgr
  connector, the plain `llmdatadist_connector.py` and their UT) and costs one call instead of a
  `fatal: path … does not exist in 'v0.9.1'`.
- **The timeout is in the request's critical path, so it *is* the TTFT.** This is not "a warning beside a
  slow transfer": a prefill request is only reported finished after its `request_finished` entry expires
  (§"Mechanism"), so one missing notification costs 120 s **per request** and the prefill's queue behind it.
- **The connector no longer exists from `v0.12.0rc1` on.** `a77045f35` "…Offline the llmdatadist connector
  related parts of the code and files." (#4780, 2025-12-09), first tag `v0.12.0rc1`; the file is present at
  `v0.11.0`/`v0.11.0rc1` and absent at `v0.12.0rc1`. So a report naming this file is on a retired
  connector: the durable lever is *which connector*, not a patch to this module.

## Mechanism (verified at `v0.9.1`)

Two roles in one file; `self.llm_datadist_role` splits them (`LLMRole.PROMPT` = prefill producer).

1. **Prefill keeps its computed blocks and starts a deadline.** `request_finished` (`:261`) returns early
   unless `kv_transfer_params["do_remote_decode"]` and the request reached
   `RequestStatus.FINISHED_LENGTH_CAPPED`; then `delay_free_blocks = len(computed_block_ids) > 0` (`:280`)
   and the request is parked in `self._reqs_need_send[request_id] = time.perf_counter() +
   envs.VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` (`:286`). It advertises `remote_block_ids`, `remote_engine_id`,
   `remote_host=self.local_ip`, `remote_port=self.port`, `remote_tp_size` to the decode side.
2. **Only a decode-side `ReqForFinished` releases it.** The listener
   `listen_for_agent_metadata_req` (`:357`) serves a ZMQ endpoint on
   `VLLM_LLMDD_RPC_PORT + local_dp_rank*tp_size + tp_rank` (`:359`, default port `5557`,
   `envs.py:134-135`). On `LLMDataDistCMgrEvent.ReqForFinished` (`:388`) it reads
   `[request_id, decode_tp_rank, decode_tp_size]` and calls `_increment_task_count` (`:409`), which
   **needs one distinct `tp_rank` per decode TP rank** (`:413` dedupes a repeated rank,
   `:419 len(...) == decode_tp_size`) before the id moves to `finished_reqs` and out of `reqs_to_send`
   (`:393-401`).
3. **The decode side sends it only after a successful pull.** `start_load_kv` (`:602`) →
   `cache_manager.pull_blocks(...)`; an `LLMException` becomes
   `RuntimeError("… Timeout during pull_blocks, you can try to increase the sync_kv_timeout config or
   checking your connect status")` (`:852`/`:869`); only after the pull does `send_finish_to_remote`
   (`:787`) post `[ReqForFinished, [request_id, self.tp_rank, self.tp_size]]` to the prefill's
   `host:port`.
4. **The warning is the expiry branch, not the transfer.** `get_finished` (`:875`) walks
   `reqs_to_send` in insertion order and, when `now >= expires`, logs the line at `:886`, adds the id to
   `finished_reqs` and deletes it — i.e. the prefill gives up and frees the blocks 120 s later.
   `VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` default is `120` (`envs.py:140-141`).

So the warning means: **the decode side never posted the notification, or posted fewer distinct TP ranks
than the `decode_tp_size` it announced.** Both are config/handshake facts, not device faults.

## Levers

| tier | change | cost / caveat |
| --- | --- | --- |
| reporter-confirmed in the log | `export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600` | only stretches the deadline: fewer spurious frees, *worse* tail (a genuinely lost notification holds the blocks ten minutes). `reported`, printed by the code itself |
| the real one | make the decode side's announcement match its behaviour — same `tensor_parallel_size`/`data_parallel_size` on both roles, every decode TP rank pulling and notifying for every request, and `sync_kv_timeout` raised on the pull | `hypothesis` about *their* mismatch; it is what the counters test |
| move connector | `vllm_ascend/distributed/mooncake_connector.py` first ships with #1568, **first tag `v0.10.1rc1`** — so a 0.9.x deployment has no Mooncake connector in-repo at all and `v0.10.1rc1`+ is where the migration starts | different transport and a different handshake; then `mooncake-transfer-errors.md` owns the failure |
| upgrade | `>= v0.12.0rc1`, where this connector is gone (`cat-file -e` is enough to prove it) | the class cannot recur; re-read the replacement connector's own evidence first |

## Why a slow TTFT and not a crash

Nothing in the path raises: `pull_blocks` succeeded (no `RuntimeError`) and the KV bytes did arrive, which
is why the request *works* while the prefill's block pool slowly fills with requests parked on a 120 s
deadline; new prefills then queue, and the measured TTFT grows to the timeout itself. That chain is
`hypothesis`, consistent with the log line the code prints. Discriminator: a prefill-side
`finished_reqs`/`reqs_to_send` count (or `Delaying free of N blocks for request …` INFO repeats) that grows
while the decode side reports no error.

## The thread shape to expect

#3062's only comment is `Tflowers-0129`: "Is this issue still exist?" and the issue is **closed** with no
diagnosis, so there is no lever in the thread. The deliverable is the code-level mechanism above plus the
env lever; do not spend a batch looking for a maintainer confirmation that is not there.

## Falsifier

The warning line appears but `--kv-transfer-config` names a **Mooncake** connector (the message would then
come from a patched/older image); or the decode roles are configured with a *larger* `tp_size` than the
prefill was told (then the count can never reach `decode_tp_size` and the fix is the config, not a code
defect); or `VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` is already large and the delay tracks something else —
measure one request's `Delaying free …` → `finished` delta before attributing the TTFT to it.

## Check before reuse

```bash
R=/workspace/repos/vllm-project/vllm-ascend
git -C $R ls-tree -r --name-only <ref> | grep -i llmdatadist          # layout at THEIR ref
git -C $R show <ref>:vllm_ascend/envs.py | grep -n -A2 LLMDD_ABORT_REQUEST_TIMEOUT
git -C $R show <ref>:vllm_ascend/distributed/llmdatadist_c_mgr_connector.py | sed -n '261,300p;875,895p'
git -C $R cat-file -e <ref>:vllm_ascend/distributed/llmdatadist_c_mgr_connector.py  # exit≠0 = gone
```
