# AscendStore KV pool: availability, connector composition, MemCache init, lookup stalls and freezes

Applies when a report runs with a KV-pool connector — `--kv-transfer-config
'{"kv_connector": "AscendStoreConnector", "kv_role": "kv_both", "kv_connector_extra_config":
{"backend": "mooncake" | "memcache", "lookup_rpc_port": ...}}'`, alone or inside a
`MultiConnector` — and typically also a `mooncake_master`. It opens with the availability and
connector-composition questions, then covers the failure shapes with their distinct evidence: the
scheduler↔worker lookup RPC (a run freezes with no traceback), the Put path under speculative
decoding (the pool stays empty while KV usage climbs), a frontend metrics-accounting crash
(`ValueError: Counters can only be incremented ...`) that is upstream arithmetic and only needs the
connector to make external tokens non-zero, the `backend: memcache` start-up/first-request class
(bare `assert res == 0` in `MemcacheBackend._setup_store`, which init path a model+protocol takes,
and the device-binding / DP-init-barrier commits), and a *performance* report where external hits
overlap the local prefix cache and load 0 tokens.

Verified code facts below are from the read-only clone at tags `v0.13.0rc1` and `v0.14.0rc1`
(with the metrics section read at `v0.18.0rc1` and vLLM `v0.18.0`), `/repos/vllm-project/vllm-ascend`
and `/repos/vllm-project/vllm`; the availability/composition section also uses `v0.11.0`,
`v0.12.0rc1`, `v0.13.0rc2`/`rc3`, `v0.14.0rc1`, `v0.16.0rc1`, `v0.18.0`, `v0.19.1rc1`,
`v0.20.2rc1`, `v0.23.0rc1`, `v0.23.0`, `v0.26.0rc1`, `v0.27.1rc1`, `v0.28.0.rc1`, `v0.30.0` and
`origin/main`, plus the `kvcache-ai/Mooncake` clone at `522b716e`. The recorded report that produced this map is issue
`https://github.com/vllm-project/vllm-ascend/issues/6512` (v0.13.0rc1 image, Qwen3-Embedding-8B,
910B A2, `vllm bench serve --backend openai-embeddings`, 9000 prompts, concurrency 10). Its
root cause stays unverified; what follows is the verified path plus labelled candidate causes.

## Availability and connector composition (the "is it supported?" shape)

A report that only asks whether a connector exists, or whether two connectors can be enabled
together, is answered from three locators — not from the transfer code: the tag's own
`docs/source/user_guide/feature_guide/kv_pool.md` (the configuration it actually prescribes), the
connector registry at that tag (the names that resolve), and `git tag --contains <enabling commit>`.

Store-connector names and paths, read with `git show <tag>:vllm_ascend/distributed/__init__.py`
(empty from `v0.14.0rc1`, when the registry moves to
`vllm_ascend/distributed/kv_transfer/__init__.py`):

| tag | name(s) registered | module |
| --- | --- | --- |
| `v0.11.0` | `MooncakeConnectorStoreV1` (module class `MooncakeConnectorV1`) | `vllm_ascend/distributed/mooncake/mooncake_store_connector_v1.py` |
| `v0.12.0rc1`-`v0.13.0` | `MooncakeConnectorStoreV1` **and** `AscendStoreConnector`, both -> `AscendStoreConnector` | `vllm_ascend/distributed/kvpool/ascend_store_connector.py` |
| `v0.14.0rc1`+ | same pair, after the package move | `.../kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`; PD connectors under `kv_transfer/kv_p2p/` |
| `v0.16.0rc1`+ | the same names plus `MultiConnector` re-registered as `AscendMultiConnector` | as above |

- So "does vllm-ascend support the mooncake store connector" is a version question the table
  settles: the name exists from `v0.11.0`, and from `v0.12.0rc1` it is an alias of the KV-pool
  connector. `ascend_store_connector.py:41` at `v0.12.0rc1` logs "It is recommended to use the
  AscendStoreConnector, as the MoonCakeStoreConnector will be removed in the future."
- `kv_pool.md` is titled "Ascend Store Deployment Guide" at `v0.12.0rc1`/`v0.13.0`, and its PD
  section prescribes exactly one pair: `MultiConnector` = `MooncakeConnectorV1` (`kv_producer` on P,
  `kv_consumer` on D, `kv_port`, `use_ascend_direct`, prefill/decode `dp_size`/`tp_size`) +
  `AscendStoreConnector` (`lookup_rpc_port`, `"backend": "mooncake"`). `MooncakeLayerwiseConnector`
  appears in no combined example at those tags.

### `MooncakeLayerwiseConnector` inside `MultiConnector` needs `v0.16.0rc1` or newer

Reported signature (`v0.13.0`, Qwen3-4B, 8x910B, P node `kv_producer` and D node `kv_consumer`, both
with `MultiConnector[MooncakeLayerwiseConnector, AscendStoreConnector]`):

```text
[patch_core.py:70] AssertionError: len of local trans block ids : 0 not equal to the len of remote trans block ids : 2
  File ".../vllm_ascend/distributed/mooncake_layerwise_connector.py", line 758, in add_tranfer_task
```

- Line pin: the format string sits in `build_connector_meta`'s nested `add_tranfer_task` at
  `mooncake_layerwise_connector.py:760` (`v0.13.0` and `v0.13.0rc3`), `:743` (`v0.13.0rc2`) and is
  **absent** from `v0.13.0rc1`. A reported `line 758` therefore pins the tree to `v0.13.0` within two
  lines, not to rc1.
- What it compares (producer path, `:742-762`): `local_trans_block_ids =
  local_block_ids[(local_transferred_tokens//bs):(local_computed_tokens//bs)]` against
  `remote_trans_block_ids =
  remote_block_ids[((local_transferred_tokens-remote_cache_tokens)//bs):((local_computed_tokens-remote_cache_tokens)//bs)]`.
  `remote_block_ids` are the D node's own block ids handed back through `kv_transfer_params` via the
  layerwise metaserver, and `remote_cache_tokens = ((len(all_token_ids)+bs-1)//bs -
  len(remote_block_ids)) * bs` (`update_state_after_alloc`, `:675-692`). A `0`-vs-`N` mismatch means
  P's block count for the request and the count D reported disagree.
- The two lengths agree only while P's `local_block_ids` covers every block up to
  `local_computed_tokens` **and** D's `remote_block_ids` starts at the same prompt offset, so a `0`-vs-`N`
  assert says the two roles disagree about which blocks this chunk covers (an empty P-side list is
  enough on its own: the local slice comes out empty while the remote one still spans blocks).
  Why pooling is the plausible cause (candidate, not reproduced): upstream `MultiConnector` hands the
  same `blocks`/`num_external_tokens` to every child, so the store connector's pool bookkeeping and the
  layerwise connector's PD bookkeeping act on one request with no isolation. The plugin's own fix
  isolates exactly that call, which supports the reading: `AscendMultiConnector` (`v0.16.0rc1`,
  `kv_transfer/ascend_multi_connector.py:13-20`) overrides `update_state_after_alloc` so only the
  chosen connector **and** `MooncakeLayerwiseConnector` get the real `blocks`, while every other child
  gets `blocks.new_empty()` with `0` external tokens. Discriminator for a live report: the file logs
  both lists plus `local_computed_tokens` in `MooncakeLayerwiseConnector scheduler add transfer task`
  (`:765`) — but *after* the assert, so it needs an instrumented or patched build; otherwise ask
  for the D-side `kv_transfer_params` (the debug line in `get_num_new_matched_tokens`, `:608-612`).
- Release boundary: `kv_transfer/__init__.py:24-27` at `v0.16.0rc1` pops the upstream
  `MultiConnector` registration and re-registers it as `AscendMultiConnector`. That commit is
  `675387f1f` ("[P/D][KVPool]Mooncake Layerwise Connector supports kv_pool", #7032) and
  `git tag --contains 675387f1f` starts at `v0.16.0rc1`. The v0.18.0 release-notes section announces
  it ("KV Pooling ... now MooncakeLayerwiseConnector can be activated together with KV Pooling",
  #7032 among the citations) — that section is absent from `v0.19.1rc1`'s copy of
  `release_notes.md`, which still stops at `v0.18.0rc1`, and first appears at `v0.20.2rc1`
  (`main:docs/source/user_guide/release_notes.md:840-858`). Tags before `v0.16.0rc1` cannot do it at
  all.
- Answers: at `v0.13.0` the supported pooling deployment is the documented `MooncakeConnectorV1` +
  `AscendStoreConnector` pair; a layerwise-PD + pool deployment needs `v0.16.0rc1` or newer on both
  roles (same image on P and D). Do not offer `use_layerwise: true` on the store connector as the
  same thing: that is the pool's own layerwise prefetch mode (`KVPoolScheduler`/`KVPoolWorker` gate,
  see the `load_async` boundary above), a different mechanism from `MooncakeLayerwiseConnector`.

### `-DUSE_ASCEND_DIRECT=ON` is a transport choice, not a dependency

The vllm-ascend docs read as mandatory ("Make sure to build with `-DUSE_ASCEND_DIRECT` to enable
ADXL engine", `kv_pool.md` at `v0.12.0rc1`; `v0.18.0` keeps the cmake line and adds an HIXL
`ascend_direct` FAQ), but the flag selects Ascend's transport at **Mooncake** build time. Mooncake's
own build docs give the default and the fallback: `docs/source/getting_started/build.md` (clone
`522b716e`) lists `-DUSE_ASCEND_DIRECT=ON/OFF` with default **OFF**, "Enable Ascend Direct transport
and HCCS support via the ADXL engine. Recommended for Ascend builds", beside `USE_UBSHMEM` and the
CUDA/HIP switches. The maintainer reply in #5299 says the same in words: without the flag Mooncake
falls back to Host RDMA, or TCP when no RDMA NIC is present, and still works with higher latency and
limited bandwidth. Cite the doc default plus that thread when a reporter asks whether they may drop
it; the connector's own runtime switch is the separate `use_ascend_direct` extra-config key, and the
two config files must not be confused
([pd-disaggregation.md](pd-disaggregation.md) keeps them apart).

## Signature: repeated lookup lines, then nothing

```text
(EngineCore_DP0 pid=10773) INFO [pool_scheduler.py:85] Reqid: embd-bench-...-382-0,
  Total tokens 1182, kvpool hit tokens: 1024, need to load: 1024
... (the same request id many times) ...
  Reqid: ...-383-0, Total tokens 1193, kvpool hit tokens: 1024, need to load: 0
```

- `pool_scheduler.py:85-91` is `KVPoolScheduler.get_num_new_matched_tokens` in
  `vllm_ascend/distributed/kvpool/pool_scheduler.py` at v0.13.0rc1 (the log format
  `"Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d"` matches
  character-for-character — that is the pin). One line is emitted per waiting request per
  scheduling step, so a long run of identical ids means those requests are being re-examined
  without being scheduled, not that the lookup itself is slow.
- The same function computes `need_to_allocate = num_external_hit_tokens - num_computed_tokens`,
  returns `0, False` when it is ≤ 0, and otherwise records a `LoadSpec` and returns
  `need_to_allocate, self.load_async and not self.use_layerwise` (`load_async` defaults False).
  "need to load: 1024" on every step therefore means a positive external hit is found each time
  and the request still never advances.
- The run then stops mid-benchmark with no ERROR and no traceback: the client's progress bar
  freezes (382/9000 in the recorded report) while the service stays up. That is a *silent*
  stall, so do not go looking for an exception in the vLLM log.

## External hits that overlap the local prefix cache: `need to load: 0`, lookup still paid

Report shape (issue `https://github.com/vllm-project/vllm-ascend/issues/13356`, vllm-ascend
`v0.23.0rc1` + vLLM `v0.23.0`, HDK 26.1, DeepSeek-V4, PD split, `MultiConnector` with
`AscendStoreConnector` `backend: memcache`): on a prefix-reuse workload `lookup()` reports hits on
almost every request, the connector then loads 0 tokens, and the throughput line keeps printing
`External prefix cache hit rate: 0.0%`. Memcache therefore adds a per-request lookup RPC and no
reuse, and end-to-end throughput lands below the local-prefix-cache / Mooncake-only baseline. There
is no error and no traceback: this is a *performance* report, not the stall signature above.

- Verified at `v0.23.0rc1`, `pool_scheduler.py::get_num_new_matched_tokens` (`:517`):
  `self.client.lookup(token_len, request.block_hashes, self.kv_cache_group_ids)` runs
  unconditionally, with no local-hit input, so the RPC is paid even when the whole prefix is
  already computed locally. `:529-532` then sets `need_to_allocate = 0` for
  `num_external_hit_tokens < num_computed_tokens` and the `<= 0` path returns `(0, False)` — the
  arithmetic is correct and the maintainer's reply is exactly that ("This is an optimization, not a
  bug, as currently upstream vLLM follows the same behavior", `Pz1116`).
- The optimization is `9e45da191` (#12783, "Optimize AscendStore key construction and miss path"):
  **in `v0.23.0`/`v0.23.0.post1`, not in `v0.23.0rc1`** (`git tag --contains 9e45da191`). At
  `v0.23.0` the non-layerwise branch opens with `if num_computed_tokens >= token_len: return 0,
  False` (`pool_scheduler.py:536-537`) so a fully local prompt costs no lookup at all, and `:544`
  passes `hbm_hit_tokens=num_computed_tokens` to `lookup()`. The worker uses it
  (`pool_worker.py:1796-1807`) to pre-seed `exists` with the locally covered block hashes and to
  start the store probe at `hbm_hit_tokens // effective_block_size * effective_block_size`, so a
  partially local prompt probes only the uncovered tail.
- Answer shape: quote the two rc1 lines, name `9e45da191`/#12783, prove the boundary with
  `git tag --contains`, and point the reporter at the `v0.23.0` tag or the `releases/v0.23.0` head
  — the maintainer's own instruction was "test with the lastest commit of 23.0 branch". Label the
  upstream-parity sentence as the maintainer's statement, and do not present the 0-load arithmetic
  as a defect.
- Applicability: `use_layerwise=True` never takes the short-circuit (`force_layerwise_load =
  self.use_layerwise and num_external_hit_tokens > 0`, `:570-577` at `v0.23.0`), so this covers the
  non-layerwise path only. Disabling local prefix caching flips `num_computed_tokens` to 0 and makes
  external hits get used, but it throws away local reuse — a diagnostic, never the recommendation.
  Thread status `wait-feedback` → stale-close, so nothing in the recorded evidence shows the
  reporter's throughput recovering after the upgrade.

## `load_async`: whether the store `get` blocks the worker step

Second signature from the same function: the hit line and the load-spec line for one request, then
(in the reported run) a worker dying with no Python exception.

```text
Reqid: chatcmpl-<id>, Total tokens 455394, kvpool hit tokens: 453632, need to load: 1024
KV pool load spec created req=chatcmpl-<id> vllm_cached=452608 kvpool_cached=453632
  need_to_allocate=1024 load_async=False use_layerwise=False
ERROR ... Worker proc VllmWorker-5 died unexpectedly, shutting down executor.
```

- The format strings are `"Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d"`
  and `"KV pool load spec created req=%s vllm_cached=%d kvpool_cached=%d need_to_allocate=%d
  load_async=%s use_layerwise=%s"`, both in `get_num_new_matched_tokens`
  (`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:535` and `:555` at
  `v0.23.0rc1`; `:560`/`:581` at `v0.23.0` — the body shifts ~25 lines between the rc and the
  release). Grepping either string in both trees names the emitting component outright, and the
  printed `load_async=` is the value actually in force for the request.
- `load_async` comes from the connector's extra config with default `False`:
  `pool_scheduler.py:93` and `pool_worker.py:133` at `v0.23.0rc1` both do
  `...get("load_async", False)`. A log showing `load_async=False` is therefore the default path,
  not a misconfiguration. The tag's docs say the same ("`load_async` | Whether to Enable
  Asynchronous Loading. The default value is false." in
  `docs/source/user_guide/feature_guide/kv_pool.md`) and set `"load_async": true` in
  `docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md`.
- The flag is a scheduler contract, not a tuning knob. `get_num_new_matched_tokens` returns
  `(need_to_allocate, self.load_async and not self.use_layerwise)` (`:565`). Upstream
  `vllm/v1/core/sched/scheduler.py` at vLLM `v0.23.0` unpacks it at `:617-621`, then at `:675-684`
  treats `load_kv_async=False` by scheduling `num_new_tokens = request.num_tokens -
  num_computed_tokens` in the same step (the external tokens are counted as already computed),
  while `True` schedules no new work and lets the recv thread finish.
- Worker side at `v0.23.0rc1`: `pool_worker.start_load_kv` (`:745`, reached from the connector's
  `start_load_kv` in `ascend_store_connector.py:200-217`) hands the request to `kv_recv_thread`
  when `self.load_async` (`:787-791`); otherwise it builds the key/addr/size lists and calls
  `self.m_store.get(key_list_c, addr_list_c, size_list_c)` **inline on the worker thread** (`:840`)
  with no timeout, then only classifies missing blocks (`:841-870`).
- So a silent worker death right after these two lines is consistent with a blocking store `get`
  on the model-runner thread (a device-side abort or a backend hang leaves no Python traceback),
  not with the scheduler or the lookup RPC. Candidate workaround, needing a runtime check: add
  `"load_async": true` to the same `kv_connector_extra_config` map and compare. It moves the
  transfer to `kv_recv_thread` instead of waiting inside the step, but it also changes scheduling
  (no new tokens while loading), so report whether the crash moved or disappeared rather than
  claiming a fix.
- Boundary: `and not self.use_layerwise` makes the return `False` in layerwise mode whatever the
  flag says, so the toggle proves nothing for a layerwise/MemCache report. Layerwise transfer is
  gated instead by `vllm_ascend/memcache_comm_fence.py`'s `AttentionComputeStartGate` (reset in
  `start_load_kv:750`; worker threads wait for the NPU event the attention op records) — that gate
  is the *layerwise prefetch* gate and is not the non-layerwise sync-load mechanism, so do not read
  it as the defect for a plain `load_async=False` run.
- A D-node pool that never loads can also be the documented short-circuit
  `if self.kv_role == "kv_consumer" and not self.consumer_is_to_load: return 0, False` (`:495`).
  `consumer_is_to_load` / `consumer_is_to_put` both default to false in the same doc table, so check
  them before reading the transfer code.

## Why this RPC can block forever

- Scheduler side, `class LookupKeyClient` (`pool_scheduler.py:285-309` at v0.13.0rc1): a
  `zmq.REQ` socket, `bind=False`, and `lookup()` does `send_multipart(...)` then a bare
  `self.socket.recv()` — **no timeout, no retry, no error branch**. The scheduling thread that
  calls it is the EngineCore thread, so a missing reply stops every later scheduling step.
- Socket path: `ipc://{envs.VLLM_RPC_BASE_PATH}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}`
  from `get_zmq_rpc_path_lookup` (`:312-327`). `lookup_rpc_port` is the key to use;
  `mooncake_rpc_port` still works there but logs a deprecation warning. The path embeds
  `dp_rank`, so check that the scheduler and worker resolve the same string before suspecting the
  backend — a mismatch serves a socket nobody connects to.
- Worker side, `class LookupKeyServer` (`ascend_store_connector.py:153-192`): `zmq.REP`,
  `bind=True`, a daemon thread whose loop is `recv_multipart → decode → pool_worker.lookup_scheduler
  → send` with **no try/except around the body**. Any exception in `lookup_scheduler` kills the
  daemon thread silently (its traceback goes to the worker's stderr, which the report may not
  include) and the REQ client then waits forever.
- `KVPoolWorker.lookup_scheduler` (`pool_worker.py:471-527`) calls `self.m_store.exists(keys)`;
  `MooncakeBackend.exists` is a thin `self.store.batch_is_exist(keys)` call into the mooncake
  client (`backend/mooncake_backend.py:53-54`). A stall inside mooncake looks exactly like a
  stall inside the connector, so separate the two before blaming either.
- Error-handler defect worth quoting: both `lookup` and `lookup_scheduler` end with
  `except Exception as e: logger.error(...); return start`. `start` is the loop variable, so
  (a) a failure before the first `process_tokens` yield raises `UnboundLocalError` inside the
  handler, and (b) a mid-loop failure returns the *last* block start, not the first partial hit —
  a wrong hit count rather than an error. Present this as a code defect, not as this report's
  root cause.

## Release window and file moves

The whole `vllm_ascend/distributed/kvpool/` package moves at `v0.14.0rc1` to
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/` (refactor `295018ec0`, #5719):
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/{pool_scheduler,pool_worker,
ascend_store_connector,kv_transfer}.py` plus `backend/{mooncake_backend,memcache_backend}.py`.
Read the path that exists at the reported tag, not the one you remember.

The three lookup-side regions above (`LookupKeyClient`, `LookupKeyServer`, `lookup_scheduler`) read
the same at `v0.13.0rc1` and `v0.14.0rc1` (line numbers shifted, statements did not), so a release
in that window did **not** fix this by adding a receive timeout or an error handler. Commits
touching the old kvpool path in
`v0.13.0rc1..v0.14.0rc1` — the candidate set, each needing its own read:

| commit | subject | note |
| --- | --- | --- |
| `ff4c1a47b` | [bugfix] Fixing KV Pool Memory Retention and Performance Degradation Issues (#5751) | touches `kv_transfer.py`, `pool_worker.py`, `mooncake_backend.py`; message says "missing PUT operations" + refactor-induced perf degradation |
| `db7cf9b0c` | [bugfix] A2 Environment Pooling for Memcache Compatibility (#5601) | A2 memcache memory registration + early HCCS link setup; also touches `pool_scheduler.py`/`pool_worker.py` |
| `473431e7e` | [P/D] Remove mooncake kvpool unused parameter `local_hostname` (#5574) | config surface |
| `46c2fc6a3` | [KVPOOL] decode save kvcache (#5168) | save path, MLA only |
| `295018ec0` | [Refactor] Refactor of vllm_ascend/distributed (#5719) | the move above |
| `b94d58976` | [MM][Bugfix] Update `hf_config` to `hf_text_config` (#5319) | unrelated config rename |

## Pool never receives Puts: PD + spec decode (`finalize_kv_connector`, stale output)

Same connector family, opposite symptom: nothing freezes, but the P-side KV usage climbs and later
requests stay `waiting`. Requires PD split + KV pool + **speculative decoding** (MTP/EAGLE).
Recorded report: issue #8808, `v0.18.0rc1`, GLM-5.1 W4A8, A2 1P1D, `MultiConnector`
(`MooncakeConnectorV1` + `AscendStoreConnector`), `--speculative-config deepseek_mtp`,
`--enforce-eager`.

Cheapest discriminator, and the one the maintainer asked for: the `mooncake_master` metrics line.

```text
Master Metrics: Mem Storage: 0 B / 32.00 GB (0.0%) | Keys: 0 | Clients: 32
  PutStart=0/0, PutEnd=0/0, PutRevoke=0/0, Get=0/0, ExistKey=5/5 (Item=665/665)
```

Lookups happen (`ExistKey` traffic) while `PutStart`/`PutEnd` stay at 0 and storage stays empty: the
producer never finalizes the request, so the pool is never written and a decode-side lookup can only
miss. Do not read `0 B` as a pool-sizing or mooncake-server problem.

Two distinct defects on the same `model_runner_v1.py` `sample_tokens` path:

1. **No finalize call under spec decode at all.** Verified by tag:
   `git grep -n finalize_kv_connector v0.18.0rc1 -- vllm_ascend` → empty, while
   `v0.18.0:vllm_ascend/worker/model_runner_v1.py:1554: self.finalize_kv_connector()` sits inside
   `if self.speculative_config is not None:` after the draft-token block. The maintainer wrote
   "IIRC, #7874 is not merged in 0.18.0rc1"; PR #7874 (`[BugFix][0.18.0][KV Pool] Fix KV Pool not
   putting kv cache for vllm v0.18.0`, merged 2026-04-02, base `releases/v0.18.0`) names the
   mechanism: vLLM v0.18 defers connector finalization during target-model forward when speculative
   decoding is on, and the plugin dropped its own finalize call in the version bump. Workaround:
   cherry-pick #7874 or use an image that contains it — the reporter confirmed it ("usable", the
   waiting hang disappeared). Boundary: one reporter, no bisect, and no statement of what else the
   cherry-pick changed.
2. **Finalize runs but the scheduler still gets the pre-draft connector output.** Upstream re-reads
   the field after finalizing; the plugin does not. Verified at `v0.18.0`:

   | repo | line | statement |
   | --- | --- | --- |
   | upstream `vllm/v1/worker/gpu_model_runner.py` | 4047 | `self.finalize_kv_connector()` |
   | upstream | 4052-4054 | `# self.kv_connector_output may be modified during drafting` then `kv_connector_output = self.kv_connector_output; self.kv_connector_output = None` |
   | plugin `vllm_ascend/worker/model_runner_v1.py` | 1443-1444 | same snapshot taken **before** the draft-token block |
   | plugin | 1554 | `self.finalize_kv_connector()` (after drafting) |
   | plugin | 1569 | `ModelRunnerOutput(... kv_connector_output=kv_connector_output ...)`, i.e. the stale snapshot |

   The shape repeats at `v0.19.1rc1` (1809/1922), `v0.21.0rc1` (2341/2459) and `v0.23.0`
   (2407/2527/2535), with upstream still re-reading (v0.23.0: 4388, 4561, 4567), so it is not a
   0.18-only slip. Consequence for a pool: worker-side state updated during drafting/finalization
   never reaches the scheduler, finished-send is not observed, the delayed blocks are not released,
   and usage climbs until newly waiting requests cannot be scheduled — the reported symptom. PR
   #8063 (`[0.18.0][BugFix][KV Pool] preserve drafted connector output on v0.18.0`) proposes exactly
   this fix and was **closed without merging**, so it stays a code-verified candidate derived from
   the upstream/plugin diff, not an accepted fix.

Keep the thread's negative results: removing `VLLM_ASCEND_BALANCE_SCHEDULING=1` and raising
`VLLM_NIXL_ABORT_REQUEST_TIMEOUT` were tried for this hang and reported ineffective. After #7874 the
reporter still measured pooled-PD TTFT at 150 s (vs 40 s at 0.17.0rc1, 127k in/1k out, 4-way) and
then wrote that enabling flashcomm1 on the P node restored it — inconsistent with the pasted P
script, which already exports `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, so treat it as an unverified perf
workaround and confirm the flag's live value plus the FlashComm support matrix
([env-vars-and-additional-config.md](env-vars-and-additional-config.md),
[context-parallel-moe-paths.md](context-parallel-moe-paths.md)).

Boundary: both mechanisms are read from the two trees at the named tags; nothing recorded here
establishes which defers to which in the reporter's second symptom, and no runtime experiment was
run. The confirming check is a spec-decode run that logs `kv_connector_output.is_empty()` before and
after `finalize_kv_connector()` on the worker and the `PutStart`/`PutEnd` counters on the master.

## Thread statements (not code facts)

- The reporter verified the discriminator themselves: removing `--kv-transfer-config`
  (no `AscendStoreConnector`, no mooncake) let all 9000 requests complete. Treat that as the
  strongest applicability test in the thread — the KV-pool path is required to reproduce.
- The same reporter later wrote that vLLM 0.14.0rc1 solves it. That is a user report with no
  bisect, no commit, and no explanation; it narrows the window but does not name the fix. Do not
  convert it into "fixed in 0.14.0rc1" without `git tag --contains` on a specific commit.

## Next checks, cheap to most expensive

1. Ask for the worker-side stderr and `MC_LOG_DIR` logs: a dead `LookupKeyServer` thread leaves a
   `lookup_scheduler` traceback there, which decides between "exception" and "mooncake stall".
2. Look at the `pool_scheduler.py:85` block again: if the same request id repeats while other ids
   never appear, the scheduler is not draining; if the last line is a *new* id, the stall is
   inside that lookup.
3. A `py-spy dump`/`faulthandler` of the EngineCore process settles it in one shot: a frame in
   `LookupKeyClient.lookup → socket.recv` proves the blocking RPC.
4. A/B `"backend": "memcache"` at the same version separates connector code from the mooncake
   client; also try a single prompt with `--max-num-seqs 1` to see whether concurrency is needed.
5. Only then read the candidate commits, asking `git merge-base --is-ancestor <sha> <reported-ref>`
   and `git tag --contains <sha>`.

Boundary: the recorded investigation never reached a root cause and was cut off by the output
budget before answering; the path facts above were re-verified in the clone, the cause was not.

## `ValueError: Counters can only be incremented by non-negative amounts` is frontend metric math

Report shape (issue `https://github.com/vllm-project/vllm-ascend/issues/8664`, vLLM 0.18.0 +
vllm-ascend 0.18.0rc1, QWQ-32B (quantized), TP=2, `--max-model-len 131072`, KV pool =
`AscendStoreConnector` + `backend: mooncake`, 20k-60k-token prompts with a 64k-token output;
crash after ~24 of 36 requests, and not reproducible with `--kv-transfer-config` removed):

```text
(APIServer pid=16393) ERROR [async_llm.py:707] AsyncLLM output_handler failed.
  File "/vllm-workspace/vllm/vllm/v1/metrics/loggers.py", line 1120, in record
    self.counter_prompt_tokens_by_source[source][engine_idx].inc(
ValueError: Counters can only be incremented by non-negative amounts.
```

### Verified mechanism (upstream vLLM, not the connector)

- vLLM `v0.18.0` `vllm/v1/metrics/loggers.py:1118-1122` loops `PromptTokenStats.ALL_SOURCES`
  (`local_compute`, `local_cache_hit`, `external_kv_transfer`) and calls
  `counter_prompt_tokens_by_source[source][engine_idx].inc(pts.get_by_source(source))` — a
  Prometheus counter, which raises on a negative argument.
- `vllm/v1/metrics/stats.py:240-289` `PromptTokenStats.update_from_output` computes
  `self.local_cache_hit += num_cached_tokens + recomputed - num_external_computed_tokens`, where
  `recomputed = 1` only when `num_cached_tokens + 1 == prompt_len`. Nothing clamps the result.
- The project's own explanation is a FIXME in the fixing diff (vLLM commit `91e4521f9f`, PR #37160):
  "local_cache_hit can go negative after preemption. num_cached_tokens is a one-time snapshot from
  first scheduling and is never reset on preemption, while num_external_computed_tokens is
  overwritten on re-scheduling. ... If CPU offload finds more tokens on the second pass than the
  original total, the subtraction underflows." The patch wraps the subtraction in `max(0, ...)` and
  the FIXME says the fundamental fix is to track the first-time external count separately.
- Both schedulers overwrite that field on each scheduling attempt: upstream
  `vllm/v1/core/sched/scheduler.py:622` and the plugin's own
  `vllm_ascend/core/recompute_scheduler.py:501` (mirrored in `core/scheduler_dynamic_batch.py` and
  `patch/platform/patch_balance_schedule.py`) at `v0.18.0rc1`, each then emitting
  `num_external_computed_tokens=request.num_external_computed_tokens` into `EngineCoreOutput`.
  So this is not a plugin-only or connector-only defect: the connector only makes external tokens
  non-zero, and the frontend then trusts them against a stale cached-token snapshot.

### Release boundaries (checked with `git tag --contains` on the vLLM clone)

| change | commit | first release | absent from |
| --- | --- | --- | --- |
| clamp `max(0, ...)` (PR #37160) | `91e4521f9f` main, `1dbbafd3f3` cherry-pick | `v0.19.0rc0` (also `v0.19.0`, `v0.19.1`) | `v0.18.0`, `v0.18.1`, `v0.18.2rc0` |
| structural `PrefillStats` rework (#37460, separate `num_local_cached_tokens` / `num_external_cached_tokens`) | `d3af8c1831` | `v0.19.2rc0` (and `v0.20.0`+) | `v0.19.0*`, `v0.19.1` |

Consequences to state in an answer: no 0.18.x patch release can fix this, the clamp removes the
crash but leaves the attribution wrong in the preemption case (the FIXME), and the plugin releases
that pair with vLLM ≥ 0.19 were **not** verified here — that vllm-ascend clone carries only
`v0.19.1rc1` and `v0.20.2rc1` in that range — so name the vLLM boundary and ask for the paired image
tag.

### Thread map and claims that need testing

- #8664's comments: a duplicate pointer to
  `https://github.com/vllm-project/vllm-ascend/issues/7489`, then a maintainer comment naming
  vllm#37160 and suggesting `v0.19.1rc1`, then auto-`stale`/close with no reporter confirmation.
- #7489 (vLLM 0.16.0 + vllm-ascend 0.16.0, DSR1 W8A8 with MTP, KV pool, high concurrency, same
  `ValueError`) carries an *earlier* maintainer claim "already fixed in v0.18.0" — the ancestry
  check above contradicts it, so cite the later comment (#37160 / `v0.19.1rc1`) and say the
  v0.18.0 claim does not hold for the clamp or for the rework.
- The report's own hypothesis (`num_cached_tokens + num_external_computed_tokens > prompt_len`) is
  the right shape but the wrong variable — the underflow is against `local_cache_hit`, and it is
  driven by the snapshot/overwrite asymmetry, not by prompt length alone.
- Workarounds backed by evidence: upgrade past the vLLM boundary above, or drop
  `--kv-transfer-config` (the reporter's own A/B, and the strongest applicability test in the
  thread). A metrics-only mitigation (scrape/disable that logger) was not tested.

Boundary: the source facts and the tag ancestry were re-read in the clone; no NPU run reproduced the
crash, and the recorded investigation was cut off by the output budget before writing an answer.

## CPU offloading: `OffloadingConnector` + an MLA model dies in the layer lookup (upstream, not the plugin)

Report shape (issue `https://github.com/vllm-project/vllm-ascend/issues/7971`, vLLM 0.17.0 +
vllm-ascend 0.17.0rc1, GLM-4.7 and Kimi-K2.5 on A2, `--kv-transfer-config` with
`"kv_connector": "OffloadingConnector"`, `spec_name: NPUOffloadingSpec`, `num_cpu_blocks` set):
start-up dies during KV-cache initialisation with
`KeyError: 'language_model.model.layers.0.self_attn.attn'` at `offloading_connector.py:600`, inside
`attn_backends = {layer_name: layers[layer_name].get_attn_backend() for layer_name in layer_names}`,
reached from `vllm_ascend/worker/model_runner_v1.py:2578 initialize_kv_cache ->
get_kv_transfer_group().register_kv_caches`.

Verified at vLLM `v0.17.0`:

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:596-600` calls
  `get_layers_from_vllm_config(self.spec.vllm_config, Attention, layer_names)`, which filters
  `compilation_config.static_forward_context` by `isinstance(..., Attention)`.
- `vllm/model_executor/layers/attention/mla_attention.py:267` declares
  `class MLAAttention(nn.Module, AttentionLayerBase)` — it shares only `AttentionLayerBase` with
  `Attention`, so the filter drops every MLA layer, the returned dict is empty, and the next line
  indexes it. The split follows the layer class, not the model name: a model whose layers are
  `Attention` passes, any MLA model (Kimi-K2.5, DeepSeek-V2/V3-style, GLM-5's MLA path) cannot.
  The report itself is inconsistent about GLM-4.7 — one paragraph says it failed the same way, a
  later one says it did not — so state the class rule and let the reporter's stack decide.
- The plugin does not own the decision: `vllm_ascend/kv_offload/npu.py:17
  NPUOffloadingSpec(OffloadingSpec)` only supplies `get_manager`/`get_handlers`
  (`CpuNpuOffloadingHandler`) and never selects layer classes. Answer "upstream hardcodes the
  layer type", not "vllm-ascend has not adapted MLA" — the reporter's own `isinstance` diagnosis is
  the correct one.

Documented workaround (same thread, maintainer): "The original offloading connector doesn't work on
vllm-ascend, could you please have a try with `AscendStoreConnector` instead?" The issue stays open
with `kv-cache-pool`/`core-features` labels, so the connector switch is the only maintainer-endorsed
path recorded — point at the AscendStoreConnector sections above rather than at the plugin's NPU
handler.

Later upstream: `525f2eeb0b` ("[kv_offload+HMA][6/N]: Split offloading_connector.py", #37405) splits
the file, and the resulting `vllm/distributed/kv_transfer/kv_connector/v1/offloading/` package works
with MLA — `offloading/config.py:19-23, 221, 241, 246, 273` and
`offloading/canonical_mapping.py:23, 282` import/test `MLAAttentionSpec`; `ca1954d58c` (#37090)
disables cross-layer KV cache for MLA backends. Those are `main` observations only: run
`git tag --contains` per commit before promising an upgrade, which the recorded pass did not.

Check order: grep the reported vLLM tag's `offloading_connector.py` for `get_layers_from_vllm_config`
and `class MLAAttention`'s bases at that same tag. Two greps settle the report; the plugin's handler
code cannot change the outcome.

## P-side backlog after hours with pooling and `--mamba-cache-mode align`: duplicate free-batch ids

Signature: pooling on (`AscendStoreConnector` inside `MultiConnector`, mooncake master), a
hybrid/Mamba model, `--mamba-cache-mode align` on both P and D, and the **producer** node's
waiting queue grows monotonically while the decode side stays healthy (#18030: Qwen3.8-27B,
2P4D TP4, 910B4, backlog after ~6 h). There is no traceback; it looks like a scheduler stall.

Verified at `2c59ff1047573f40a471cf7535f01a8877344415` (`releases/v0.23.0`, the reporter's
commit):

- `vllm_ascend/patch/platform/patch_kv_cache_utils.py:46-56` — `_dedupe_free_blocks` drops a
  repeated `block_id` from one free batch after only `_swa_block_diag("duplicate_free_batch",
  block, where)`, a WARNING. **It does not release the reference that duplicate carried.**
  `_ascend_free_blocks` (`:83+` onward) then filters `ref_cnt <= 0` and delegates to the
  original `BlockPool.free_blocks`, which decrements each surviving block once.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:996`
  `touch_sending_mamba_blocks` (`self._block_pool.touch(...)` over the mamba groups' block ids;
  called from `:946`, `:974`, `:982`) raises the refcount of every block scheduled for an
  external send, and with mamba-align the same block id can be listed more than once in one
  free batch.
- Combined: pool touch takes the count to 3, the batch lists the block twice, the dedupe drops
  one copy without a decrement and `free_blocks` decrements once → `ref_cnt` 2, the block never
  re-enters the free queue, and P-side KV blocks leak until the queue backs up. With pooling off
  the count is 1 and the same dedupe path frees correctly, which is why the report correlates
  with enabling pooling.

Reporter's fix (thread, `huangdong2022`): inside `_dedupe_free_blocks`, when a dropped duplicate
has `ref_cnt > 1`, decrement once (`block.ref_cnt -= 1`) and leave the last decrement to
`BlockPool.free_blocks`; they report two stress runs / ~15 h without recurrence (thread claim,
not independently verified).

Applicability tests before proposing it: remove the `AscendStoreConnector` entry from
`MultiConnector` (keeping `MooncakeConnectorV1` PD) or stop passing `--mamba-cache-mode align`,
and watch whether the waiting queue stops growing. Instrument with the existing hook: count
`SWA_BLOCK_DIAG duplicate_free_batch where=BlockPool.free_blocks` warnings against
`SWA_BLOCK_DIAG nonzero_ref_cnt_free_queue_insert` — a rising duplicate count with a stalled
free queue is the shape.

Forward state: on `main` `_dedupe_free_blocks` no longer exists (`git grep -n
"_dedupe_free_blocks\|touch_sending_mamba_blocks" main -- vllm_ascend` finds only
`pool_scheduler.py:826`), so the helper was removed or reworked and "upgrade to main" is not an
established fix. `releases/v0.23.0` has 7 commits after the reporter's build and none touch
`patch_kv_cache_utils.py`; `2bcf78b37` (#17667, cherry-pick of #14132) changes `config_data.py`
to avoid co-queueing save with load and cites "double delayed-free" — the opposite direction
(double free), not this leak. No release tag is verified to contain a refcount fix.

## MemCache backend (`ock.mmc.*`, `memcache-hybrid`): a bare `assert res == 0`, the two init paths, DP init and device binding

The MemCache backend reaches a separate SDK: `MemcacheBackend` (`backend/memcache_backend.py`)
calls `memcache_hybrid.DistributedObjectStore.init(<device id>, init_bm=<bool>)` with the config file
named by `MMC_LOCAL_CONFIG_PATH` (`mmc-local.conf`; the tag's
`docs/source/user_guide/feature_guide/kv_pool.md` tells every vLLM process to export it — at
`v0.26.0rc1` `:786`, and its protocol key table at `:510` recommends `device_rdma` on A2 and
`device_sdma` on A3/HCCS). Main later renames that page to `kv_cache_pool_ascend_store.md` and adds
the defaults the rc page omits: `ock.mmc.local_service.protocol` defaults to `host_rdma`, and
`host_shm` requires DRAM > 0 with HBM = 0 (`:807` there). Reports reach the backend as
`MultiConnector` + `AscendStoreConnector` `backend: memcache`; a launcher's `memcache` block is just
the place the reporter puts that connector plus the service config (its `serve_params` carries
`--no-disable-hybrid-kv-cache-manager`, its `config` the `ock.mmc.*` keys), so read the connector
config for the flag values rather than inferring them from the block.

**The start-up assert.** `_setup_store()` wraps `res = store.init(...)` in a `try` with two
`except` handlers (a `ValueError` = config load, anything else = store init fail; both log the
exception), then reaches a bare `assert res == 0` with no message, then `time.sleep(0.1)`. Verified
`v0.23.0` (`:87`/`:95`, with `:58` = `self.store = self._setup_store()` in `__init__`),
`v0.26.0rc1` (`:88`/`:96`) and still on `origin/main` (`:246`, now `store.init(self.device_id,
...)`). So an `AssertionError` here means the MemCache SDK returned non-zero for that rank: the
reason lives in the SDK's own log, not in Python. Two uses:

- As a **version pin**: a recorded report (`issues/13418`, GLM-5.1, A3 560T 2P+1D, P = DP4TP8
  D = DP8TP4, PP, CANN 8.5.1, vllm-ascend 0.23.0) names `memcache_backend.py` `__init__` line 58 and
  `_setup_store` line 95, which is `v0.23.0` exactly.
- As the **first thing to say**: the assert carries no error text, so ask for the worker's stderr
  and the mmc logs (`mirror_client.cpp` lines, `MC_LOG_DIR`) before theorising; then check the
  memcache-hybrid/memfabric-hybrid versions (main documents `>= 1.3.0` for both, and `device_sdma`
  on A3 additionally needs LingQu Computing Network >= 1.5).
- PP is not itself an unsupported combination: at `v0.23.0` `KVPoolWorker` carries `pp_size`/
  `pp_rank` (`:122-123`, rank-aware pool keys) and the connector's
  `set_xfer_handshake_metadata_pp_aware` ignores generic PP xfer metadata because "AscendStore
  handles PP via pool keys" (`#12762` = `df54ea61f`, in `v0.23.0`). PP + MTP finalization and
  backend-error surfacing are part of that same commit.

**Which init path a run takes.** `self._lazy_init = lazy_init and _is_device_sdma()`, where
`_is_device_sdma()` parses `MMC_LOCAL_CONFIG_PATH` for `ock.mmc.local_service.protocol ==
device_sdma` and raises `ValueError("The environment variable 'MMC_LOCAL_CONFIG_PATH' is not set.")`
when the variable is missing. `lazy_init` comes from `pool_worker._init_backend`
(`backend_kwargs["lazy_init"] = self.use_compress` — DSV4-compress models only; its comment adds
that mooncake needs `ASCEND_ENABLE_FABRIC_MEM=1` for the same flag). Therefore:

- non-compress model **or** a `host_*`/`host_shm`/`device_rdma` protocol → eager init inside
  `__init__`, so a failure is a *start-up* failure at the assert above;
- DSV4/V4-Flash compress model **+** `device_sdma` → init is deferred to `ensure_initialized()`
  inside the first `get`/`put`, i.e. into the serving path. That is the mechanism behind "the
  service starts, then the first/second request crashes", and it is what a reporter means by
  "懒加载" (`issues/16111`, DSV4-Flash-0731-w8a8, DP4TP4, crash on the second curl). Before that
  gating commit, memcache was passed `lazy_init=True` unconditionally: at `v0.23.0rc1`
  `_init_backend` has a dedicated `if self.backend.lower() == "memcache": ... lazy_init=True`
  (`pool_worker.py:248-252`), and `a1b574089` (#13028, the `v0.23.0` backport of the same change;
  `v0.26.0rc1` is where the `main` cherry-pick `b6fe31d11` first lands) replaces it with
  `backend_kwargs["lazy_init"] = self.use_compress`. So the eager-vs-lazy answer is version-gated:
  `v0.23.0rc1`'s memcache path lazy-initialised for *every* model, while the newer text leaves lazy
  init to DSV4-compress runs only (read earlier rc tags the same way rather than assuming). Read `_init_backend` at the reported ref before
  quoting either behaviour. The deferred path announces itself at request time with
  `Initializing Memcache store. local_rank=%d` (the field is `device_id` on `main`), so that line
  appearing after the model is up is itself the proof that lazy init was taken; the backend's
  `ImportError` names `memfabric_hybrid` and the memcache install page.

**Device binding: the id handed to the SDK was a process-group local rank.** Before `c970e5892`
(#16287, first tag `v0.28.0.rc1`) the backend passed `self.local_rank` — default
`get_world_group().local_rank`, `create_scheduler_client(local_rank=0, init_bm=False)` — as the
device id, and its PR body names the failing shape directly: "KV pool clients can create NPU
contexts on the wrong device when a worker's process-group local rank differs from its bound
visible device, such as DP=2 / TP=4", plus "Memcache's metadata-only scheduler client can also
create an unnecessary EngineCore NPU context". The fix renames the ctor argument to `device_id`
(default `torch.npu.current_device()`), resolves the scheduler side with
`get_scheduler_device_id(parallel_config)` from `assigned_physical_gpu_ids`, and skips
`set_device()` when `init_bm=False`. Its recorded check (A3, DP=2/TP=4) is start-up/context
placement only — workers appeared on their assigned physical NPUs and the duplicate EngineCore
contexts disappeared; no requests were run. So for a MemCache "device id conversion"/wrong-device
report on a multi-DP-per-node launch, treat the local-rank mapping as the candidate and this commit
as the fix; the exception text itself is the SDK's.

**DP initialization barrier (`memcache_dp_init_barrier`).** `93c49251f` (#17289, 2026-09-29)
"Synchronize MemCache initialization across DP workers" adds a CPU process-group barrier after a
successful `_setup_store()`, exposed as `kv_connector_extra_config.memcache_dp_init_barrier`
(code at `v0.30.0`: `ascend_store_connector.py:115` `extra_config.get(..., True)`, passed through
`pool_worker.py:459` as `dp_init_barrier`; put it on the `AscendStoreConnector` child inside
`MultiConnector`; skipped for DP=1, metadata-only scheduler clients and actual lazy init). Its stated failure is "an early DP rank can
proceed to NPU work while peers are still establishing their MemCache channels" — the class behind
an `ERR02005 DIST internal error` / HCCL watchdog abort on the first request after the channels
open (`issues/16111` case 1: DSV4-Flash + `device_sdma` + DP4TP4, second curl). Boundaries:
`93c49251f` itself is only in `v0.30.0`/`v0.30.0rc*`; the `v0.27.1rc` cherry-pick is `832e63c85`
(#17308), first tag `v0.27.1rc1`; and `9ccd283e5` (#19370) reverts it on `main` only (no tag at
review time) as no longer needed, documenting the `memcache-hybrid >= 1.3.0` +
`memfabric-hybrid >= 1.3.0` requirement instead. The option is present from `v0.27.1rc1`
through `v0.30.0` (and the `v0.30.0rc*` tags); do not recommend it on a `main` build at or after
`9ccd283e5`, where it is gone.

**Answer shape for a MemCache start-up or first-request crash.** Pin the tag from the traceback (or
say plainly that no traceback survived), state which init path the reported model + protocol takes,
and give the one commit that changed that path with its `git tag --contains` result. `#16111`'s
thread has no maintainer diagnosis (a ping and an extra `device id转换异常` log link), so the thread
contributes no root cause — label the mechanism a candidate and name the SDK log as the confirming
check. The reporter's own "合并这些 PR" style citation (`#13349`, `#12762`, `#13028`, `#12643` on
`#13418`) is not a fix list: `#13349` is an unmerged, diagnostic-only draft (PP ranks 2/3 passing
zero sizes to `batch_put_from_layers` on GLM-5.1 `memcache + use_layerwise=false + PP4`), `#12762`
is the PP/MTP finalization + error-surfacing commit already inside `v0.23.0`, and the lazy-init one is inside `v0.23.0` on that branch
(`a1b574089`), reaching `main`-derived tags at `v0.26.0rc1` (`b6fe31d11`). Resolve each cited number to a commit and ask `git tag --contains` before
repeating any of them.
