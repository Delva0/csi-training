# Mooncake P/D handshake `Receive timeout` → `Failing N request(s) due to KV load failure` → `Send failed … Operation cannot be accomplished in current state`, then no new request runs

## Signature

Decode node, in this order and within seconds:

```text
[mooncake_connector.py:NNNN] Receive failed. error=Receive timeout, attempts left=2
[mooncake_connector.py:NNNN] Receive failed after all retries. source=<P ip>:<port>, error=Receive timeout
Failed to transfer KV cache for request <id>: Failed to receive data after 3 retries: Receive timeout
   … frames: _handle_request → _transfer_kv_cache_all_groups → _get_remote_metadata → ensure_zmq_recv
[scheduler.py:NNNN] Failing 1 request(s) due to KV load failure (failure_policy=fail, 13348 tokens affected)
[mooncake_connector.py:NNNN] Send failed. error=operation cannot be accomplished in current state, attempts left=2 / =1
[mooncake_connector.py:NNNN] Send failed after all retries.
[mooncake_connector.py:NNNN] Unexpected error occurred in socket. error=Failed to send data to <P ip>:<port> after 3 retries: Operation cannot be accomplished in current state.
```

From then on both sides print ZMQ errors steadily, every new request fails, and the decode node's metrics
settle at `Running: 1 reqs, waiting: 0 reqs, GPU KV cache usage: 99.9%` (the trailing `Preemption:` value
is cut off in the OCR, so read it from the reporter's own log rather than the screenshot). Recorded
#12383 (vLLM-Ascend `0.22.1rc1`, GLM-5.1, 4×A3, 1P+1D, 32 concurrent 100 K-in / 30 K-out; the evidence
is two screenshots whose `.ocr.txt` are the log). The shape is *connected-then-transfer-fails*, so it is
not the connect/handshake-`103900` family: `mooncake-adxl-connect-103900.md`,
`adxl-connect-status-same-host.md`.

## Two owners, ranked

1. **Handshake latency against a one-second budget** — why the timeouts appear at all.
2. **The failure path is not a recovery path** — `failure_policy=fail` kills the request, and the
   done-signal handshake then fails too, so neither side releases anything.

## The 1-second arithmetic (source-verified at v0.23.0rc1 `f4a08bd`)

| fact | location |
|---|---|
| handshake timeout `self.timeout = 1.0` (s) | `kv_p2p/mooncake_connector.py:499` |
| decode-side REQ socket `SNDTIMEO`/`RCVTIMEO` = `timeout*1000` | `:1380-1400` |
| 3 attempts, 0.1 s apart, then `RuntimeError` | `ensure_zmq_send` / `ensure_zmq_recv` `:3651-3689` |
| decode fetches metadata once per peer/port (`GET_META_MSG`, then recv) | `_get_remote_metadata` `:1310-1335` |
| prefiller answers `GET_META_MSG`/`DONE_RECVING_MSG` from **one ROUTER busy loop per rank** — `recv_multipart`, reply, loop | `run_busy_loop` `:322-395` |

So each handshake has ~3 s and the prefiller's responder is serial. At 32 concurrent 100 K-token
prefills one slow turn of that loop is enough to exceed it. **First ablation: repeat at concurrency 1-4
with the same prompt length** — if the ZMQ lines disappear, it is the budget, not a corrupt socket, and
the answer's rung is "lower the test concurrency / raise the handshake timeout", not a connector defect.

## Why it is permanent

`_handle_request` (`:697-739`) wraps the transfer in `try/except Exception`; the **`finally`** then calls
`_send_done_signal_to_free_remote_port` and `_send_done_recv_signal` unconditionally, with the pin's own
comment "Always send the done signal … Failing to do so may cause a memory leak on the remote host". That
second call is the one whose failure prints `Unexpected error occurred in socket` (`:1374`). Hence the
`Send failed` line is the *done-recv* handshake, not the transfer.

`Operation cannot be accomplished in current state` is ZMQ **EFSM**: on a `REQ` socket it means this
socket has already sent and has not consumed the reply, so its only source is a socket that timed out on
`recv` and was then reused. At this pin both handshake helpers close such a socket instead of returning it
to the pool (`_get_remote_metadata` `:1329-1334` on any exception; `_send_done_recv_signal` `:1370-1374`
on `RuntimeError`), so the reuse path is **not visible in this tree** — and their tree is demonstrably not
this tag: the paste logs `attempts left=2` where the pin prints `attempts_left=%d` (`:3665`, `:3684`),
calls `ensure_zmq_recv(sock, self.remote_poller, path)` while the pin's non-layerwise signature is
`ensure_zmq_recv(socket, path)` (`:3672`; the poller variant lives in the **layerwise** connector,
`mooncake_layerwise_connector.py:2062-2083`), and puts `_handle_request` at `:1182` and the helpers at
`:2815`/`:2849` against the pin's `:697`/`:3651`/`:3672`. **One check decides it:** `sed -n` of their
`_get_remote_metadata` and `_send_done_recv_signal` around their `:1180-1240` and `:2800-2860` — if the
error path does not `close()` the socket (or closes only for `RuntimeError` while their recv helper's
chained form raises the bare `zmq.ZMQError("Receive timeout")` first, as their traceback shows), then
`_get_remote_socket` pops that socket from the pool (`:1380-1402`) and every later send fails instantly,
forever. That is the report's "permanent".

## The 99.9 % KV is not proof of a leak, and the OCR decides it

Those metrics lines read `Running: 1 reqs, waiting: 0 reqs, GPU KV cache usage: 99.9%`: with nothing
waiting there is nothing to preempt, and the last surviving long request owns the pool while
it decodes at 10-50 tok/s. **Read the OCR's digits before ranking a leak**: the same paste prints
`Engine 994` and `9 · 9 tokens/s` for `Engine 000` / `0.0 tokens/s`, i.e. glyph `9` is a `0`, so
`waiting: 9` is `waiting: 0`. If it really were ≥ 1 the lane moves to
`../scheduler-admission/waiting-queue-wedge.md` or preemption bookkeeping. The decisive missing artefact is the decode node's own
`GPU KV cache size: N tokens` (or `# GPU blocks`) startup line: the request that failed in the paste
affected only 13 348 tokens, so the paste alone cannot say whether one ~130 K-token request fills the
pool — ask for that line rather than asserting the arithmetic.

## Ladder, cheapest first

1. Concurrency 1-4 at the same prompt length (above) — budget vs broken socket.
2. `"kv_load_failure_policy": "recompute"` in `--kv-transfer-config`: a timed-out handshake then degrades
   to a re-prefill instead of a dead request (mitigation, not a cure).
3. The prefiller's log for the same seconds: `KVCacheSendingThread started listening on path …`,
   `Got DONE_RECVING_MSG for request …`, and `Connection listener encountered exception during message
   processing` — the last one means the serial responder threw and kept looping without ever serving the
   earlier `GET_META_MSG`.
4. With `GPU KV cache size` in hand, compute blocks per request
   (`(prompt + generated) / block_size`) against the test's concurrency; if one request already owns most
   of the pool, the occupancy half is sizing, not a fault.
5. Only then the version: re-run on the current tag before proposing any patch (the helper rewrite above).

## Limits

The recorded case has no serve command, no plog and no `GPU KV cache size`; the EFSM owner in *their*
build is unverified (their file is not the snapshot's) and the cascade (timeout → `failure_policy=fail`
→ done-signal failure → no release) is derived from the pin's structure plus their own log lines.
