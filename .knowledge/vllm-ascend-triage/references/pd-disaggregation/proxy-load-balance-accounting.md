# The example proxy's load accounting: a release that is not keyed to its reservation

Scope: reports about `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
as a **scheduler** — one prefiller (or decoder) is picked far more often than its
peers, a per-server counter leaks or is released twice, or the body is a
self-diagnosed accounting bug with a stdlib repro and no backend fault at all. Every
request is served, nothing is refused and no worker dies: the symptom is the
*selection*, so no log line carries it and no connector / CANN / graph knob moves it.
Transport, endpoint coverage, the `400`/`max_tokens = 1` leg questions and the
`/v1/messages` route table are the topic index's other rows.

Two counters per `ServerState` are the whole system: `active_tokens` and
`active_kv_cache` (`:177-178` at the pin below). The prefiller priority is
`active_tokens + active_kv_cache * 0.3`; the decoder priority is `active_tokens`
alone. Lower wins, from a per-role heap.

## Step 0 — date the reporter's file before reading any of it

The example is rewritten often and the report usually quotes it by line range, so
first decide whether the tree you can read is the tree they ran:

```bash
F=<root>/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
wc -l $F
rg -n 'class (ProxyState|SharedProxyScheduler)|def (select_prefiller|begin_request|_release_load|reassign_instances)|released_kv' $F
git -C <root> cat-file -t <commit-from-the-body>
```

At vllm-ascend v0.19.1rc1 (`da421af`) the file is **928 lines** with class
`ProxyState`; `SharedProxyScheduler`, `begin_request`, `reserve_prefill_kv`,
`_release_load` and `reassign_instances` are all absent, so a body citing `:991`,
`:1027-1033`, `:1047-1048` or `:1087-1096` names a **later refactor**, and its
"working copy at commit …" is usually not an object in the one-commit snapshot
(`git cat-file -t` → *Not a valid object name*). The body's `_priority()` at `:308`
is `release_decoder` at this pin, and its priority expression
(`active_tokens + active_kv_cache * 0.3`) is `:221` in
`_update_prefiller_priority`. Say which of the two trees each line belongs to in the
first paragraph; do not derive the refactor's semantics from the pin.

Then use the corpus as a **date**, not as source: a neighbouring issue that quotes the
same helper fixes the refactor's shape and marks it post-pin. Recorded #13214 quotes
`_release_load(self, role, key, load, *, active_tokens=False, kv_cache=False)` with
`self._pool(role).servers`, `entry.active_*`, `self._push_heap(role, key)` and the
clamped kv branch `max(0.0, entry.active_kv_cache - load)`. That corroborates the
newer generation and gives the discriminator below; it does not let you claim its
behaviour at the pin.

## Step 1 — the pin's accounting, which is what you can verify

| quantity | site | code |
|---|---|---|
| prefiller reservation | `select_prefiller` `:260`, `:268-269` | `active_tokens += n` **and** `active_kv_cache += n` |
| prefiller priority | `_update_prefiller_priority` `:221` | `active_tokens + active_kv_cache * 0.3` |
| decoder reservation / priority | `select_decoder` `:301`, `:229` | `active_tokens += n`; priority `active_tokens` |
| token release | `release_prefiller` `:276-280`, `release_decoder` `:308-312` | `active_tokens -= n`, **unguarded** |
| kv release | `release_prefiller_kv` `:284-289` | `if active_kv_cache > 0: active_kv_cache -= n` |

The kv guard is a **skip, not a clamp**: at `active_kv_cache = 50` a release of 120
lands on **-70**. Only the post-pin `max(0.0, …)` clamps, which makes one number from
the reporter's own repro the generation test — ends at 0 ⇒ clamped refactor, negative
⇒ a pin-like tree.

On the generation path the reservation is taken in `_handle_select_instance`
(`:659-676`): both counters are charged on the chosen prefiller, the prefill request
is sent, and only the **tokens** are released when it returns — the kv stays charged
until the KV is really consumed, which is the first non-empty decoder chunk
(`released_kv = False` `:733`, helper `:738-742`, call `if not released_kv and chunk:`
`:756` → `:757`), with the same helper repeated in `except` (`:822`) and `finally` (`:823-825`).

## Step 2 — what the pin gets wrong: the release outlives its reservation

`released_kv` is a **boolean per request**, not a key on the reservation. The
`recomputed` branch (`:793-803`) replaces `instance_info` with a fresh
`_handle_select_instance(...)` — a *new* prefiller (new kv reservation) and a *new*
decoder (new token reservation) — and then `break`s. Nothing resets the flag, so for
the rest of that request:

- the **new prefiller's** `active_kv_cache` is never released (all three call sites go
  through the flag), so it stays inflated for the life of the proxy process: that
  prefiller's priority is too high and it is **under**-selected;
- `finally` calls `release_decoder(instance_info.decoder_idx, instance_info.decoder_score)`
  (`:826`), which now names the **new** decoder, so the **old** decoder's
  `active_tokens` is stranded and it is over-selected.

So "the old prefiller's kv was released twice, so it is over-selected" and "the new
prefiller never releases, so it is under-selected" are the same defect read from two
ends: the release is keyed on a flag instead of on the reservation it releases. The
fix is per-reservation idempotence, not another guard variable — record what was
taken and compare it:

```python
old_decoder = (instance_info.decoder_idx, instance_info.decoder_score)
instance_info = await _handle_select_instance(api, req_data, tmp_request_length)
released_kv = False                        # the new prefiller holds a new reservation (:802)
proxy_state.release_decoder(*old_decoder)  # retire the decoder being replaced
```

Acceptance is an invariant, not a clean exit: after N `recomputed` retries,
`sum(active_kv_cache) == sum(outstanding prefill reservations)` and
`sum(active_tokens) == live decoders`, with no counter negative. A repro that asserts
that invariant beats one that prints the dict.

## Step 3 — the sibling example is the control

`load_balance_proxy_layerwise_server_example.py` (618 lines at the same pin) takes and
releases the kv score **inside one function**: the prefill callback's `finally` runs
`release_prefiller` + `release_prefiller_kv` (`:609-610`), and its decode-side
`recomputed` branch (`:515-523`) only re-streams from the same decoder. A
double-release is structurally impossible there, and its `released_kv` flag
(`:444`, `:462`, `:464`) is assigned and never read. A report against the layerwise
variant therefore has a different owner — do not carry this page's verdict across.

## What to answer

1. **Which tree**, in one line: the reporter's `git rev-parse HEAD`, `wc -l` and the
   Step-0 grep. Their repro cannot run against the pin (`SharedProxyScheduler` /
   `begin_request()` do not exist, `prefillers` is a `list[ServerState]` and selection
   returns an index, not a `key`).
2. **The counter dump at the moment of the skew** —
   `{key: (active_tokens, active_kv_cache)}` per role — because it carries the
   generation test (clamped / negative / stranded) in one line.
3. **The fix shape** (per-reservation release + retiring the pre-reassignment decoder)
   and the invariant to assert. Mark the pin's version of the defect as
   source-verified at `da421af` and the refactor's as dated-by-precedent only.

Recheck: the line numbers above are v0.19.1rc1's; the refactor renames the class and
moves them. #8545 ("高并发场景下 proxy 请求概率异常，甚至导致D节点崩溃") and #8867
("proxy 的 load_balance_proxy_server_example.py 负载分发机制有问题") are precedent
for the *question* — neither carries accounting evidence, so neither ranks a mechanism.
