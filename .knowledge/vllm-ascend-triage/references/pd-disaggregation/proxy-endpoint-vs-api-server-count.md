# "D 实例未接收到请求" / proxy timeout, with P-side `Force freed expired request`

## When this page applies

A PD (or EPD) deployment on two nodes per side, DP > 1 on both, a load-balance
proxy in front, and the report says the client times out (`proxy通信超时`) while
**the decode instance logs no such request**. The only pasted error lines are
repeated on several *prefiller* workers:

```text
(Worker_DP3_TP7_DCP7_EP31 pid=15787) ERROR [mooncake_connector.py:233] Force freed
expired request: chatcmpl-…-9aef698a. Reason: Request exceeded timeout threshold
(480 seconds). Action: Resources have been forcibly released to prevent memory leak.
```

This is not the transport lane (rank table / `device_ip` / `DestLink`,
`transport-topology.md`) and not a capacity or recompute report. The prefill side
did its work; the hand-off to a decoder never reached an engine, and the cheapest
deciding evidence is a **launch line**, not the connector.

## Step 1 — read the pasted line as bookkeeping, and say so

- `add_delayed_request` → `_retrieve_expired_requests` compares
  `time.time() - delay_start_time` with `envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`
  and force-frees the entry (`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:215-239`;
  the same code is in `mooncake_hybrid_connector.py:176-199`).
- The sending thread is constructed only for `kv_role == "kv_producer"`
  (`mooncake_connector.py:2370-2384`), i.e. **only on the prefiller**. Worker names
  do not say which side; this gate does.
- The entry is created in `request_finished` when the request ended
  `FINISHED_LENGTH_CAPPED` and blocks were computed (`:1823-1860`), which is exactly
  what the example proxy's prefill request produces (`payload["max_tokens"] = 1`,
  `payload["min_tokens"] = 1`, `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
  `build_prefill_request`).

So the lines prove: **P finished the prefill, told the proxy where its KV is, and
waited the full timeout for a decoder that never came.** The number 480 is not a
default and not a deviation either — the tutorial the reporter cites exports it
itself (`docs/source/tutorials/models/GLM5.2.md:1342`, `:1413`). Say this before
reading any connector source: it removes the largest file in the tree from the
session.

## Step 2 — count the HTTP endpoints that actually exist

Only API servers listen. The engine's DP ranks do not.

- `--headless` forces `api_server_count = 0` (`vllm/entrypoints/cli/serve.py:61-70`).
- With `--data-parallel-start-rank` set (hybrid LB, the shape every multi-node
  recipe uses) the default is `data_parallel_size_local`, otherwise
  `data_parallel_size` (`serve.py:95-123`) — and an explicit `--api-server-count 1`
  caps it at one anyway.

The 1M recipe is `--data-parallel-size 4 --data-parallel-size-local 2` with
`--api-server-count 1` on node 0 and `--headless` on node 1
(`docs/source/tutorials/models/GLM5.2.md:1324-1327`, `:1347` for P; `:1395-1398`,
`:1418` for D), so each side exposes **exactly one** address: P `node0:9081`,
D `node0:9900`. `9901-9903` and the headless node's `9081/9900` have no listener at
all.

## Step 3 — diff the reporter's proxy command against the same doc's

`--prefiller-hosts/--prefiller-ports/--decoder-hosts/--decoder-ports` are
independent `nargs="+"` lists paired one-to-one
(`load_balance_proxy_server_example.py:644-649`); nothing validates them against the
servers' topology. The doc's "Recommended proxy command" for this deployment lists
one prefiller host with port `9081` and one decoder host with port `9900`
(`GLM5.2.md:1460-1472`) — i.e. the DP master's IP and the **start** port, which is
the only form that matches `--api-server-count 1`.

A per-DP-rank list (the recorded shape: two prefiller hosts on `9081` *and* eight
decoder entries, `9900 9901 9902 9903` per node) registers addresses where nothing
listens. Then, at v0.23.0rc1:

- `_pop_valid` raises only when a pool is empty (`:291-300`); a bad host/port is
  not filtered there, so nothing fails at proxy start;
- every request to such an entry is retried `max_retries` (default 3) and then
  logged (`:809-868` `All 3 attempts failed for <url>`), so the reporter sees
  connection errors/timeouts on *some* requests only, and the D engine never logs
  the request id.

The reporter-confirmed resolution in #12432 is the doc's form: *"api-count 被限制为
1 了，所以只对外暴露一个 api 口，proxy 里只写 master ip 和 start port 就行"*.

## Step 4 — what to ask for, if the proxy line is already correct

- The literal `python load_balance_proxy_server_example.py …` line, plus each
  side's `--api-server-count` / `--headless` / `--data-parallel-size{,-local,-start-rank}`.
- The proxy's own stdout: `All 3 attempts failed for <url>` names the dead address
  directly; `No available decode servers` would instead mean an empty pool.
- `ss -ltnp | grep -E '9081|9900'` on each node — settles which addresses exist
  without a restart.

If the proxy already carries one host + start port per role, this page does not
apply; go back to `index.md`'s transport bullets and ask for the proxy and D-side
logs.

## What this rules out, and the limits

- Two or a handful of expired entries is not a capacity bug, not a recompute storm
  and not a KV-transfer fault: the timer starts at the P request's *finish* and
  ends when the decoder's pull arrives, so an expiry means "no pull at all", at any
  concurrency. Decode-side recompute preemption has its own D-side log trail
  (`stop_reason="recomputed"`, `vllm_ascend/core/recompute_scheduler.py:878`) and
  the proxy would re-prefill.
- Do not raise `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` as the fix: it only postpones
  a free that the missing pull will never use, and leaves the reporter with the
  same timeouts.
- Recorded limits: #12432's session spent its whole budget inside
  `mooncake_connector.py` and the recompute path and never compared the launch
  lines; the reporter's own resolution came from the proxy/serve commands. This page
  carries that shortcut, not a claim about which request the proxy mis-routed.
