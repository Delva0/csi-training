# PD-disaggregated deployment failures

Scope: reports where prefill and decode run as separate instances (or separate Pods) with a KV
connector, and the failure is at P start-up, at the KV-transfer handshake, or as a decode-side
hang. Verified against the read-only clones on 2026-11-01: `vllm-project/vllm` (HEAD
`5423fb8b93`), `vllm-project/vllm-ascend` (HEAD `1bd23a0a9`), `LMCache/LMCache-Ascend` (HEAD
`d241090`, tag `v0.4.4`) and `cann/hixl` (HEAD `51b7dd84b`). Line numbers are per tag; re-anchor
with `git grep -n <symbol> <ref> -- <paths>` before quoting.

## Locate the owning component before reading any code

| symptom stage | code owner | first file |
| --- | --- | --- |
| P or D start-up crash, `MessageQueue` `ZMQ bind ... Address already in use` | upstream vLLM | `vllm/distributed/device_communicators/shm_broadcast.py` (writer path) |
| D-side silent hang, AICore 0 %, port listening, curl no reply | vllm-ascend worker/scheduler | `vllm_ascend/worker/model_runner_v1.py`, `vllm_ascend/ascend_forward_context.py` |
| `HIXL Connect failed`, `HIXL Error Code: 103900`, rank table wants `device_ip` | LMCache-Ascend + cann/hixl | `lmcache_ascend/v1/transfer_channel/hixl_engine.py`, hixl endpoint/rank-table generators |
| `HCCL error: ...` while pushing weights in RL serving | trl `vllm_serve` + `vllm_ascend/.../pyhccl.py` | see [log-signatures.md](log-signatures.md) |

A single report then splits into a vllm-ascend issue and an upstream vLLM (or cann/hixl) issue.
State which repository owns the raising statement instead of filing everything against the plugin.

## The forwarding proxy must match `kv_connector` (docs + script routes verified)

A PD deployment has two example proxies under `examples/disaggregated_prefill_v1/`, and the
choice follows the KV connector, not the hardware generation:

| `kv_connector` in the P/D launcher | proxy | request order |
| --- | --- | --- |
| `MooncakeLayerwiseConnector` | `load_balance_proxy_layerwise_server_example.py` | proxy → D first; D posts to the proxy's `/v1/metaserver`; proxy → P |
| `MooncakeConnector` (`MooncakeConnectorV1`/`V2`) | `load_balance_proxy_server_example.py` | proxy → P first, then D |

Verified at `v0.23.0`:

- `docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md:770-772` maps each
  proxy to its connector ("Requests are first routed to the D nodes ... This proxy is designed
  for use with the MooncakeLayerwiseConnector" vs "routed to the P nodes ... with the
  MooncakeConnector"); `main` keeps the same pair at `:1168-1170`.
- `docs/source/user_guide/feature_guide/layerwise_kv_pool.md:238-241` states the failure mode:
  with `kv_producer` / `kv_consumer` "the dedicated layerwise proxy
  (`load_balance_proxy_layerwise_server_example.py`) is required — the standard disagg proxy
  does not serve the `/v1/metaserver` endpoint."
- The scripts themselves show the protocol split. The layerwise proxy registers
  `@app.post("/v1/metaserver")` (`:600`), fills
  `kv_transfer_params["metaserver"] = "http://<host>:<port>/v1/metaserver"` (`:434`) and logs
  `Decoder hosts will access Proxy host:port/metaserver` at start-up (`:273-277`). Its `--host`
  must therefore be an address the D nodes can reach: the default is `localhost`, and a
  wildcard `0.0.0.0`/`::` raises `ValueError: ... the Wildcard Address ... is not allowed for
  proxy` (`:278-284`). The standard proxy exposes only `/v1/completions`,
  `/v1/chat/completions`, `/reset_prefix_cache`, `/healthcheck`, `/instances/add`
  (`:1146-1188`) — nothing serves `/v1/metaserver`.
- Connector side: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
  carries `metaserver` in its params (`:112`, `:683`) and posts to it (`:830-834`, `:953`).
- Prose mechanism, current tree:
  `main:docs/source/developer_guide/Design_Documents/disaggregated_prefill.md`, section
  "Mooncake Layerwise Connector", numbers the six steps (proxy selects D, D calls
  `kv_connector_no_forward` against the metaserver, the proxy's metaserver endpoint selects P,
  P pushes KV layer by layer).

**The trap is the tutorial, not the connector.** In
`docs/source/tutorials/models/Qwen3.5-397B-A17B.md` at `v0.23.0`, §5.4 (A3) starts both roles
with `MooncakeLayerwiseConnector` (`:521` prefill, `:597` decode) and §5.5 (950DT) starts both
with `MooncakeConnector` (`:703`, `:786`). Both §5.6 tabs open with "Use the layerwise proxy
script", but the A3 tab then runs `python3 load_balance_proxy_server_example.py` (`:817`, `:823`)
while the 950DT tab runs the layerwise script (`:844`, `:850`) — each tab names the other
deployment's proxy. The same swapped text survives at `v0.23.0.post1` and `v0.30.0rc1`
(`:747`-`:753`, `:771`-`:777`). `main` restructured the section (now §5.7) into a connector→proxy
table and its A3 recipe moved to `MooncakeConnectorV1` (same file, `:553`, `:668`; table at `:1202-1208`), so main's
command is correct for main's connector and wrong for the 0.23 recipe. Read the `kv_connector`
value at the reported tag; neither the proxy script name nor the hardware tab alone identifies
the proxy.

Diagnostic use: when a P node dies or a D node hangs immediately after the forwarding proxy
starts on a layerwise deployment, check the pairing and whether the chosen proxy serves
`/v1/metaserver` before opening `mooncake_layerwise_connector.py`. Issue #15389 has this shape
(`v0.23.0` A3, Qwen3.5-122B-A10B deployed from the 397B tutorial's step 5.6, "P node shutdown"
after running the example, the layerwise script starting the proxy successfully) and has zero
comments, so no maintainer confirmation exists: the doc mismatch is verified, the exact
shutdown chain is not.

A layerwise deployment that also enables a KV-pool connector is a support-boundary question before
it is a proxy question: `MultiConnector[MooncakeLayerwiseConnector, AscendStoreConnector]` dies in
the layerwise producer's block-slice assert until `v0.16.0rc1` (then `AscendMultiConnector` isolates
the children), and the connector pairs each tag documents for pooling are in
[kv-pool-ascend-store.md](kv-pool-ascend-store.md).

### The example proxy serves only the routes it registers (API-surface gap)

"The proxy answers `/v1/chat/completions` but `/v1/messages`, `/v1/models` and `/metrics` return 404"
is answered from the script's own route list at the reported tag, not from the engine:

| ref | routes in `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` |
| --- | --- |
| `v0.18.0` (917 lines) | `POST /v1/completions` (`:881`), `POST /v1/chat/completions` (`:887`), `GET /healthcheck` (`:893`), `POST /instances/add` (`:902`), `POST /instances/remove` (`:907`) — nothing else, so every other path is FastAPI's 404 |
| `main` | those five plus `GET /v1/models` (`:1379`) and `POST /reset_prefix_cache` (`:1384`) |

- `GET /v1/models` arrived in `8a6214ac3` ("[Feature][Disagg] Proxy /v1/models in load balance proxy
  server example (#14129)", whose own body states the cause: the proxy "only forwards
  /v1/completions and /v1/chat/completions, so OpenAI-compatible clients calling GET /v1/models get a
  404"); first tag containing it: **`v0.27.1rc1`**. `/v1/messages` and `/metrics` are still absent on
  `main`, so a 0.18/0.2x report is not "one release behind the fix".
- The layerwise proxy has its own shorter list at `main`: `/v1/completions`, `/v1/chat/completions`,
  `/healthcheck`, `/reset_prefix_cache`, `/v1/metaserver` — and no `/v1/models` either.
- The backend does serve the Anthropic API: at the paired vLLM `v0.18.0`,
  `vllm/entrypoints/anthropic/api_router.py` registers `POST /v1/messages` and
  `/v1/messages/count_tokens`, and `vllm/entrypoints/openai/generate/api_router.py:38-42`
  (`register_generate_api_routers`) attaches it unconditionally through
  `attach_router` -> `app.include_router(router)` (`anthropic/api_router.py:135-136`). A proxy 404
  therefore says nothing about the engine's capability.
- State the workaround as deployment glue: call a P/D instance's own port for those endpoints (the
  proxy is a PD router, not an API gateway), or put a gateway/nginx rule in front that forwards
  non-generation paths to one backend. Adding a `/v1/messages` route to the PD handler is *not*
  drop-in: `_handle_completions` (`:710`) computes the request length from the OpenAI body, calls the
  prefiller through `_handle_select_instance` (`:659`) and chains on
  `response_json.get("kv_transfer_params")` (`:678`) to pick the decoder — an Anthropic response is
  not that shape. Label such a patch a candidate, not a fix.

## Start-up `ValueError: KV transfer '<role>' config has a conflicting data parallel size. Expected 1, but got N`

This is a **plugin consistency check on `kv_connector_extra_config`**, not the connector, not KV
transfer and not a launch-order problem. Raising statement at `v0.18.0rc1`:
`vllm_ascend/utils.py::check_kv_extra_config` (`:1125`, message at `:1142`), called from
`vllm_ascend/platform.py:249::check_and_update_config`; `_check("prefill", ...)` runs only when
`kv_transfer_config.is_kv_producer` (`:1147`), `_check("decode", ...)` only when `is_kv_consumer`
(`:1149`). Each compares `tp_size`/`dp_size` from the `prefill`/`decode` sub-dict against the
**engine-local** `parallel_config.tensor_parallel_size`/`data_parallel_size`.

Why "Expected 1" for a node launched with `--data-parallel-size 2`: in the child process, upstream
`EngineCoreProc.run_engine_core` (vLLM `v0.18.0` `vllm/v1/engine/core.py:1029,1040,1062-1072`)
treats DP ranks of a **non-MoE** model as independent — it resets
`parallel_config.data_parallel_size = data_parallel_size_local = 1`, `data_parallel_rank = 0` and
builds a plain `EngineCoreProc` instead of `DPEngineCoreProc`. The plugin check then runs a *second*
time because `_perform_handshakes` re-runs `vllm_config.__post_init__()` after the handshake
(`core.py:950`) -> `current_platform.check_and_update_config` (`config/vllm.py:1086`). The report's
traceback is exactly that chain (`core.py:1082 -> :950 -> vllm.py:1086 -> platform.py:262 ->
utils.py:1252`, ~10 lines ahead of the tag), which is how you know the model took the non-MoE branch.

Consequences to hand the reporter:

- For a dense model (Qwen3-32B and friends) the extra-config `dp_size` can only ever be `1`. A recipe
  copied from a MoE/DeepSeek PD tutorial — `"prefill": {"dp_size": 2, "tp_size": 8}`,
  `"decode": {"dp_size": 32, "tp_size": 1}` in
  `docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md` at `v0.18.0rc1` — fails
  on the first node. Fix the deployment: set that node's own `dp_size` to `1` and keep `tp_size`
  equal to `--tensor-parallel-size`.
- Do not just delete the keys: the Ascend mooncake connector requires them —
  `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py::_get_prefill_decode_size`
  asserts `"tp_size" in ...` and `"dp_size" in ...` for both `prefill` and `decode`
  (`:1164-1185` at `v0.18.0rc1`), and `vllm_ascend/ascend_config.py:96-97` reads the two `tp_size`
  values for `pd_tp_ratio`.
- `--data-parallel-rank` without distributed DP is itself the thing to question: the maintainer reply
  on this shape was to drop the flag unless distributed DP is used, and otherwise to follow the
  multi-node PD tutorial.
- Boundary: the comparison is unchanged on `main` (`vllm_ascend/utils.py:1512-1538`), so this is a
  live config constraint, not a fixed bug. Whether the plugin should compare against the global DP
  size (or skip the check for a dense model) has no maintainer statement; do not report it as a
  confirmed defect.

## Start-up: ZMQ probe-then-bind race (verified)

At vLLM tag `v0.23.0` the MessageQueue writer does:

```text
shm_broadcast.py:428   remote_subscribe_port = get_open_port()
shm_broadcast.py:433   socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
shm_broadcast.py:434   self.remote_socket.bind(socket_addr)
```

`vllm/utils/network_utils.py:190-216` `_get_open_port` binds an ephemeral port inside a
`with socket.socket(...)` block and returns the number after the block closes — the port is
released before the caller binds it, so two workers starting together can be handed the same
port and one dies with `Address already in use`. `get_open_port()` (`:150-166`) only steers away
from a 10-port window around `VLLM_DP_MASTER_PORT` when that env var is set; it is not a fix.

Two consequences decide the workaround. **Do not offer `VLLM_PORT`.** `_get_open_port`
(`:190-204`) takes `envs.VLLM_PORT` as its `start_port` whenever that variable is set and
increments from there, so every MQ-creating process starts from the same number and the first
free port is handed out repeatedly — configuring it makes the collision likelier, not rarer.
And **both binders are on one host**: the writer falls back to `connect_ip = get_ip()`
(`:423-434`), so the address in the traceback is the colliding node's own IP; a multi-node or
PP report is not evidence of a cross-node clash, and the count to ask for is how many ranks on
that node create an MQ (one writer per MQ, `create_from_process_group`). Only two routes are
evidence-backed: the reporter-side bind retry on a fresh port, and moving to a tag that carries
`f7ef489e93` (first `v0.28.0rc1`). "Start the nodes one after another" was never tested and would
not address binders on the same host, so do not offer it as a workaround.

The race is fixed upstream, not in the plugin:

| evidence | value |
| --- | --- |
| fix commit | `f7ef489e93` (2026-08-08) `[Bugfix] Fix ZMQ port TOCTOU race in shm_broadcast MessageQueue (#50960)` |
| fix shape | `bind(f"tcp://{ip}:0")` then `getsockopt(zmq.LAST_ENDPOINT)` (`main:573-576`) |
| in the reported build? | `git merge-base --is-ancestor f7ef489e93 v0.23.0` → NO |
| first release tag containing it | `v0.28.0rc1`; absent from all `v0.23.x`–`v0.27.x` tags |

So a 0.23.0-era report can only be worked around (retry the bind on a fresh port in the worker,
as the reporter's local patch does) or upgraded past v0.28.0. `vllm_ascend/patch/platform/
patch_shm_broadcast.py` at `v0.23.0` patches only the reader side
(`ReadTimeoutWithWarnings.timeout_ms`, `MessageQueue.acquire_read`) and leaves the writer bind
alone — the plugin does not fix this.

The report's own quoted `get_open_port()` is not this source: it shows an extra process-level
`_used_ports` set that never existed in `vllm/utils/network_utils.py` (`git grep -n '_used_ports'
v0.23.0 -- vllm/` and `git log -S'_used_ports' -- vllm/utils/network_utils.py` are both empty).
The frame that decides the pin, `shm_broadcast.py:434 self.remote_socket.bind(socket_addr)`,
does match v0.23.0 exactly. Check quoted snippets against the tag before reasoning from them;
report them as the reporter's paraphrase when they differ.

Triage hygiene for this signature: `Address already in use` needs a **TCP** LISTEN owner.
`TIME_WAIT` entries with no LISTEN are consistent with repeated retries of the race, and a
same-numbered **UDP** listener (e.g. `haproxy`) is unrelated because the ZMQ socket here is TCP.

## Decode hang: per-rank MoE comm type / DP all-reduce divergence (candidate)

Verified code structure at `v0.18.0rc1`; the causal step needs a runtime check.

The report shape behind it is #8975 (`v0.18.0rc1`, A2, 16 chips, DP/EP decode, body = the full
`collect_env.py` + `npu-smi` + a Feishu link and a QR image, **no serve line**). #8974 is the same
reporter's duplicate of the same body, closed pointing at #8975, so read #8975 (thread + comments)
directly when handed either number instead of re-fetching both.

- `vllm_ascend/ascend_forward_context.py:203 select_moe_comm_method(num_tokens, vllm_config,
  is_draft_model)`: with expert parallel on A3, `num_tokens <= mc2_tokens_capacity` selects
  MC2/FUSED_MC2 and a larger batch selects **ALLTOALL** (`:250-272`); on A2 MC2 needs
  `num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= capacity`, else
  ALLGATHER (`:239-248`); without EP it is always ALLGATHER (`:237-238`); A5 is MC2 only when
  `num_tokens <= capacity` **and** `world_size_across_dp > 1`, else ALLTOALL (`:275-279`). The
  fallback capacity is capped at 512 tokens (`:177-178`); when capture sizes exist it is instead
  `max_cudagraph_capture_size` (derivation per tag in
  [moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md)), while the op's own limitation stays 512
  tokens/rank, matching `ops/fused_moe/prepare_finalize.py:119`
  ("used when num_tokens exceed mc2's limitation (512 tokens/rank)").
- `vllm_ascend/worker/model_runner_v1.py:472-510 _skip_all_reduce_across_dp_group` is computed
  per rank: it returns False for non-KV-consumers (`:490-491`, `is_kv_consumer` comes from
  `kv_transfer_config` at `:342-345`), then True only when decode must use MC2 and (prefill must
  use MC2 or `recompute_scheduler_enable`). Its call site `:1891` decides whether the DP
  `all_reduce` that equalises `num_tokens`/cudagraph mode across DP ranks runs at all.
- Consequence to test, **not** established: ranks that disagree on `num_tokens`,
  `is_kv_consumer` or `recompute_scheduler_enable` enter different collectives — MC2 vs ALLTOALL,
  or `all_reduce` vs none — inside the same HCCL group and block forever. In the recorded report
  all 32 decode ranks stopped at the same operator count in `group_name_233` with
  `RunAlltoAllDirectFullmesh_device` in the tag name, and no ERROR preceded it.

Documented constraint for the scheduler involved (v0.18.0rc1 docs, still present at
`v0.28.0.rc1` in 8 files): "Setting the environment variable
`VLLM_ASCEND_BALANCE_SCHEDULING=1` enables balance scheduling ... enabling this feature is not
recommended in scenarios where PD is separated"
(`docs/source/tutorials/models/DeepSeek-R1.md:123`, `DeepSeek-V3.1.md:134`, `Kimi-K2.5.md:131`).
The maintainer's first reply asked the reporter to remove exactly that env var, and a second
maintainer agreed a mutual-exclusion check would make sense.

Flag lifecycle — matters as soon as the report is newer than 0.26:

| ref | state |
| --- | --- |
| `v0.23.0`, `v0.26.0rc1` | env var read in `vllm_ascend/envs.py`; `patch/platform/__init__.py` imports `patch_balance_schedule` only when it is set |
| `v0.27.1rc1` and later (`v0.28.0.rc1`, `v0.30.0`) | env fallback removed (`ascend_config.py` comment "the legacy environment fallback has been removed"); configured as `--additional-config '{"enable_balance_scheduling": true}'`; patch imported unconditionally and gated internally |
| `v0.30.0` | the env var survives only in `patch/__init__.py` comments as "(deprecated)" |

A guard exists for `profiling_chunk_config` × balance scheduling (`ascend_config.py:505-509` at
`v0.27.1rc1`); no guard pairing balance with `recompute_scheduler_enable` was found at
`v0.27.1rc1`/`v0.30.0`, and no commit adding one turned up under
`git log --all --grep='balance.sched'` — an absence in that search, not proof.

Boundary for this class: in the recorded thread the reporter changed a dozen settings at once
and said so themselves, so the maintainer hypothesis plus the code above stays a **candidate**.
Confirm or refute with the same topology and one variable changed (balance scheduling on/off),
while logging each decode rank's selected comm type and whether its DP `all_reduce` ran.

## Decode ranks keep `Running > 0` while generation stops (verified fix, missing from v0.23.0rc1)

Report shape (issue `https://github.com/vllm-project/vllm-ascend/issues/13964`, vLLM 0.23.0
`0fc695f` + vllm-ascend 0.23.0rc1 `f4a08bdd`, DeepSeek-V4-Flash W8A8 + 1-token MTP, A3, Prefill
3 nodes × DP4/TP4, Decode DP16/TP1 with EP, `MooncakeHybridConnector`, `--async-scheduling`,
`FULL_DECODE_ONLY`): scraping `vllm:num_requests_running`, `vllm:generation_tokens_total` and
`vllm:request_generation_tokens_count` per `engine` label shows 15 of 16 decode ranks with
`Running > 0` and **zero** delta in every 15-s interval, while one rank keeps serving and the router
sends all new work to it. Health and metrics stay 200; no `EngineDeadError`, traceback, HCCL or
vector-core timeout precedes it, and some ranks resume hours later. Treat that metric signature as
the classifier: it separates this class from a crash and from a slow kernel.

- The later worker-exit burst (MQ `RuntimeError("cancelled")` plus cascading `EngineDeadError`) is
  downstream of the stall, not its cause — order the timestamps before quoting it.
- The only warning near the first freeze is `temperature 0.0001 ... maxed it out to 0.01`. That is
  the hook for the MTP-NaN theory; the run never inspected tensor contents, so NaN stays a candidate.
  The thread also asks maintainers to distinguish it from an ACLGraph mode-sync issue — that remains
  a named candidate, not a finding.

Mechanism (maintainer close, corroborated in code): premature reuse of sliding-window /
chunked-local KV blocks under async scheduling — `num_computed_tokens` could include tokens from
batches that were scheduled but whose outputs had not been processed, so the prune freed blocks an
in-flight attention step still referenced.

- Upstream decision statement at vLLM `v0.23.0`:
  `SingleTypeKVCacheManager.remove_skipped_blocks(request_id, total_computed_tokens)`
  (`vllm/v1/core/single_type_kv_cache_manager.py:448`; the chunked-local override at `:1018`) turns
  that count into `num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)` and
  frees the blocks below it.
- Plugin fix `42a70b7ba` ("[BugFix][v0.23.0][P/D] Prevent premature SWA KV reuse under async
  scheduling (#13518)", PR merged 2026-08-06 into `releases/v0.23.0`) adds
  `vllm_ascend/patch/platform/patch_async_swa_kv_lifetime.py` (144 lines, registered at
  `patch/platform/__init__.py:35` on that branch) plus 3 lines in `core/recompute_scheduler.py`.
  It wraps `Request.__init__` (adds `num_in_flight_tokens`), `Scheduler.update_after_schedule` /
  `update_from_output` / `connector_finished`, `KVCacheManager.allocate_slots`, and
  `SingleTypeKVCacheManager.remove_skipped_blocks`, where the in-flight count is subtracted from
  `total_computed_tokens` before the original prune for `ChunkedLocalAttentionSpec` /
  `SlidingWindowSpec` only. It also patches both specs' `max_memory_usage_bytes` (admission sizing) —
  that half is what produced the follow-up capacity regression #14678.
- Availability: `42a70b7ba` is in `v0.23.0` and `v0.23.0.post1` and absent from `v0.23.0rc1`;
  `36fcb76ed` ("[Bugfix] Guard async SWA KV free queue handling (#14301)") shares those tags. Neither
  is in `main` and the patch file is absent at `v0.24.0rc1` — that absence is not "unfixed later":
  upstream vLLM gains the same accounting natively, first in `v0.26.0rc1`
  (`num_in_flight_tokens` at `kv_cache_manager.py:565,920`, `scheduler.py:353,1858,2311`,
  `single_type_kv_cache_manager.py:1940-1948`; absent at `v0.24.0rc1`/`v0.25.0rc1`). That is why a
  release-branch-only backport was the right shape for 0.23.0, and why the upgrade answer for a
  0.23.0rc1 report is `v0.23.0`/`v0.23.0.post1` (or a 0.26+ line carrying upstream's version).
- Thread traps in this report: #10237 "Guard against all-placeholder speculative output" is a
  **closed, unmerged** PR on `releases/v0.20.2rc` — not a workaround; #13863 is an issue, not a PR
  (`/pulls/13863` returns 404) and is titled "Async MTP can reuse stale accepted-token counts after
  batch condense"; #14678 is the capacity regression that follows the fix and does not request a
  revert. Do not merge #10237 with the *merged* v0.20.2rc placeholder pair — `7cb5d6180` (#10259,
  sampler rejects `PLACEHOLDER_TOKEN_ID`/`-1` drafts) and `91e6b539d` (#10308, `input_ids`
  sanitized before forward). Both sit on `origin/releases/v0.20.2rc` after the `v0.20.2rc1` tag and
  in no tag; applicability (KV-consumer node + MTP + recompute scheduler), code lines and the
  boundary that a single-node MTP report never sees a placeholder are in
  [sampling-and-output-quality.md](sampling-and-output-quality.md).
- If a 0.23.0+ log prints the patch's own `SWA_BLOCK_DIAG negative_in_flight_tokens
  where=... request_id=... num_in_flight_tokens=...` warning, the accounting went negative
  (over-decrement) — a different defect from the frozen-rank signature; do not merge the two.
- Workarounds named in the earlier duplicate thread (#12725) — disable async scheduling, disable MTP,
  add sampling-stream synchronisation — were never A/B-verified there. The release upgrade is the
  verified path; present the flag changes as candidates.

### Stray tokens / mixed-language insert with a block-aligned boundary (the same defect, reported as a quality regression)

A *quality* report rather than a stall — coherent replies with inserted meaningless characters or
Chinese fragments mid-reasoning, GPQA a few points under the usual score, a small `sliding_window`,
`block_size` 32/128, `--async-scheduling`, decode side a KV consumer — is the same premature-free
defect read from the text side (`vllm-ascend#13501`: DeepSeek-V4-Pro W4A8, `sliding_window=128`,
`block_size=32`, filed 2026-08-04, still open with zero comments).

- **The discriminator is the first-anomaly phase, so ask for it.** `SlidingWindowSpec.get_num_skipped_tokens`
  is `max(0, num_computed_tokens - sliding_window + 1)` (vLLM `v0.23.0`,
  `vllm/v1/core/single_type_kv_cache_manager.py:793`) and the manager frees
  `num_skipped_tokens // block_size` blocks (`:472`), so a block is released exactly when
  `num_computed_tokens ≡ sliding_window - 1 (mod block_size)` — `31` for a 128/32 pair, i.e.
  `block_size - 1` whenever `sliding_window` is a multiple of `block_size`. A first-anomaly boundary
  at `(prompt_tokens + tokenizer(output_before_first_anomaly).length) % block_size == block_size - 1`
  repeated over independent anomalies (17/17 in that report, 15 token-exact and 2 BPE-merged) is this
  race; sampler, detokenizer and MTP-placeholder corruption have no such alignment, so do not route
  the case to [sampling-and-output-quality.md](sampling-and-output-quality.md) on symptom words alone.
- **Check the fix at the reported *sha*, not at the tag it names.** `#13501` cites
  `releases/v0.23.0` commit `390a07b97` (2026-08-04); that tree has no
  `vllm_ascend/patch/platform/patch_async_swa_kv_lifetime.py`, while tag `v0.23.0` (`5cb98caaa`, with
  `42a70b7ba`/#13518) does. `git ls-tree -r --name-only <reported-sha> | grep async_swa` settles it in
  one call — a branch tip filed days before the merge is not the released artifact, and the answer is
  the version upgrade already recorded above, not a new defect.
- The upstream anchor that a report citing the vLLM PR resolves to: **#47728 = `cd0de48d0`** "Free
  out-of-window blocks on the processed-token basis under async scheduling", in `v0.26.0rc1` and every
  `v0.26.0`+ tag (assert with `git merge-base --is-ancestor cd0de48d0 v0.26.0rc1`), absent from
  `v0.23.0`. It is the native replacement for the plugin patch and adds
  `tests/v1/core/test_swa_inflight_window_free.py`.
- The plugin seam any backport must keep symmetric is `RecomputeScheduler.update_from_output`
  (`vllm_ascend/core/recompute_scheduler.py:831` at `390a07b97`, the class actually instantiated with
  async scheduling being `AsyncRecomputeScheduler(AsyncScheduler, RecomputeScheduler)` at `:1122`):
  it `continue`s past `failed_kv_load_req_ids` (`:905`) and finished/`None` requests (`:909`) without
  settling anything, so an in-flight counter added around `allocate_slots` has to be decremented
  on those exits too.

### Same frozen-rank signature with speculative decoding on: the recovered-token path (candidate)

Report shape (`https://github.com/vllm-project/vllm-ascend/issues/13648`, image
`quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731-openeuler`, vllm-ascend source in the image
`50e0ce608` = "[v0.25.1rc] ... (#13314)", vLLM 0.25.1, DeepSeek-V4-Flash-0731 W8A8 + DSpark with 7
draft tokens, Prefill DP2/TP8, Decode DP16/TP1 with EP, 4 nodes × 8×910B4, `MooncakeHybridConnector`):
after tens of real multi-turn agent/tool-calling requests, individual decode requests stop producing
tokens for minutes while `/health` stays 200, all processes stay alive and the NPU stays busy. The
proxy log shows prefill and the KV transfer succeeded (`KV cache transfer ... took 885.13 ms`) and
then `Engine 001: ... Running: 1 reqs, Waiting: 0 reqs` with `Avg generation throughput: 0.0
tokens/s` — the same per-engine metric signature as #13964 above, so classify it the same way and
then use the version + spec-decode presence to pick the candidate.

A commenter on that thread names a mechanism (their own decode service, not the reporter's
reproduction): a request spins until `num_computed_tokens` exceeds `max_model_len`, because the
rejection sampler returned all `-1` token IDs (including the base token), so `parse_output` produced
an empty list and the `num_computed_tokens` rollback in `update_from_output` was bypassed;
`sample_recovered_tokens_kernel` is named as the source, and `recovered_token_ids.clamp_(min=0)` as
a temporary local patch. Treat this as a thread-named **candidate**: nobody reproduced it against
this deployment and the clamp is an untested local edit.

Verified anchors at the reported revision (`50e0ce608`) and upstream vLLM `v0.25.1`:

- The Ascend rejection sampler owns the kernel and the wrapper:
  `vllm_ascend/sample/rejection_sampler.py:903 def sample_recovered_tokens` allocates
  `recovered_token_ids = torch.empty_like(draft_token_ids)` — **uninitialised** — and fills it with
  `vllm_ascend/ops/triton/reject_sample.py:316 sample_recovered_tokens_kernel`; there is no
  `recovered_token_ids.clamp` and no `parse_output` override in that file (1651 lines), so the
  parsing is upstream's.
- Upstream filtering is exactly what turns an all-`-1` row into an empty list:
  `vllm/v1/sample/rejection_sampler.py:30 PLACEHOLDER_TOKEN_ID = -1`, and `:249 parse_output`
  keeps only `(ids != -1) & (ids < vocab_size)`. The rollback the comment names lives at
  `vllm/v1/core/sched/scheduler.py:1603 request.num_computed_tokens -= num_rejected`.
- Log hook worth asking for: `[sample/rejection_sampler] Using fallback (non-reduce-sample) path in
  rejection_sample. This path should not be used in the new distributed flow.` at
  `vllm_ascend/sample/rejection_sampler.py:720` (the report pastes `:719`), a `warning_once` gated on
  `get_ascend_config().enable_reduce_sample` / a missing `target_indices`. Its presence proves the
  fallback path ran on that decode rank; it does not prove causality, and the reporter's copy is 41
  minutes before the stall.
- Next checks, in order: scrape the per-`engine` counters to confirm the frozen rank (as above);
  grep the decode rank's log for that fallback warning and for `SpecDecoding`/acceptance lines;
  instrument `num_computed_tokens` against `max_model_len` for the stuck request; then re-run with
  DSpark off to separate the spec-decode sampler from the scheduler. No released fix is established —
  do not offer an upgrade tag.

## Speculative decoding in PD: what the tag's docs prescribe (v0.18.0rc1)

An "acceptance length collapsed in PD but not colocated" report is a configuration question before
it is a hidden-state question.

- At `v0.18.0rc1` the PD tutorials do not prescribe an EAGLE draft model at all: the linked
  `docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md` has no
  speculative/eagle/draft content, and
  `docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md` only shows
  `--speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}'` (`:292`, `:351`,
  `:409`, `:467`, `:535`). A `draft_model`/`eagle3` config with `num_speculative_tokens: 5` on a
  PD split is therefore outside the documented recipe for that tag — say so before theorising.
- EAGLE3 for co-located serving is documented in
  `docs/source/user_guide/feature_guide/speculative_decoding.md` (EAGLE-3 needs `"method":
  "eagle3"`, and its note says EAGLE draft models must run **without** tensor parallelism), and
  `Multi_Token_Prediction.md:80` lists n-gram/EAGLE/EAGLE3/MTP as the supported methods. Those
  pages are the reference for the draft config itself, not for the P→D boundary.
- Acceptance-related fixes in the same tag's release notes are claims to locate, not matches for
  the report: "Fix Eagle3 acceptance rate for QuaRot quantized models" (#6914), "Fix Eagle
  speculative decoding with Context Parallel enabled" (#6981/#7079) and `v0.17.0rc1`'s "Fixed spec
  acceptance rate problem in vLLM 0.15.0" (#6606) all target other configurations; check whether
  the reporter's build carries them (`git merge-base --is-ancestor`) before citing.
- The P->D transfer channel itself carries **KV cache blocks only**, so no acceptance question is
  answerable from the connector. At `v0.18.0rc1` the only spec-decode accommodation in
  `mooncake_connector.py:516-520` is `end_layer_index = end_layer_index + 1` when
  `speculative_config is not None` and `prefill_pp_rank == prefill_pp_size - 1` ("support MTP layer
  kv transfer"); `mooncake_layerwise_connector.py:289-292` shifts the Mamba block index by
  `num_speculative_tokens + 1`. Nothing there transfers hidden states or draft-model state, so read
  the decode-side draft path, not the connector, if the mechanism has to be argued.
- What such a report still needs is the colocated run on the same build as the control plus the
  decode rank's `SpecDecoding` per-position line; two recorded investigations of this shape (same
  knowledge state) both stopped before establishing a mechanism, so none is established here. The
  delivered finding is the documented-recipe boundary plus that control — do not spend a forward
  deriving a hidden-state theory you cannot check.
- A no-traceback *throughput* complaint about spec decoding (`mtp=2`/`mtp=3` slower than `mtp=1`)
  is a different question: start from the mixed FULL-graph/eager decode split in the next section.

## `mtp=2`/`mtp=3` slower than `mtp=1` in PD: decode ranks split between FULL graph and eager

Report shape: DeepSeek-R1-W8A8 on A2 910B2, 1P1D and a two-node PD layout, `mtp=2`/`mtp=3`
measured *slower* than `mtp=1`; no traceback, no error, only two throughput charts (issue #5605,
`vllm-ascend 0.13.0rc1` / vLLM `0.13.0`, CANN 8.3.RC2). The thread holds one maintainer sentence —
"have you tried 0.13.0rc2? we improved spec decode in this release. And use `mtp=3` is recommended,
since we improved the algorithms for it as well" — then a stale close with no reporter feedback, so
no maintainer root cause exists.

One call finds the candidate fix before any proposer code is opened:
`git log --oneline v0.13.0rc1..v0.13.0rc2 --grep='mtp\|spec'` returns
`895d32c0e` "[v0.13.0][Bugfix] Support ALL D-Nodes in fullgraph when running MTP in PD (#5786)"
(`2026-01-13`), whose `git tag --contains` is `v0.13.0rc2`, `v0.13.0rc3`, `v0.13.0` and **not**
`v0.13.0rc1`. Its message states the mechanism: under PD + MTP + full graph + async scheduling the
KV a decode node pulls from prefill carries no spec tokens, so the decode scheduler's
`total_num_scheduled_tokens` is short of them, the `uniform_decode` gate
(`scheduler_output.total_num_scheduled_tokens == input_batch.num_reqs * max_query_len`) fails, and
that instance is never enqueued into the FULL graph. FULL-graph and eagle-mode decode instances
then coexist in one decode service, and because `MoeDispatch` synchronises across them the
full-graph instances are slowed to the eager instance's rate. The fix pads the spec tokens on the
decode node and lets sampling reject them.

Boundary for the answer: the commit, its message and its ancestry are verified here; the reporter's
per-`mtp` curve was never reproduced, and the mechanism explains a decode-wide slowdown rather than
why the deficit grows with `mtp`. So quote `v0.13.0rc2` as the first tag containing the fix (the
same advice the maintainer gave, now source-backed), leave "MTP ≥ 2 is intrinsically slower in PD"
unproven, and name the two runs that settle it: the same load at `mtp=1/2/3` after the upgrade, and
the same comparison with `--enforce-eager` (no FULL-graph instances, so no mixed-rate group).

## v0.18.0 D-node hang naming `rejection_sampler.py:278`

A PD decode hang (vllm-project/vllm-ascend#9938) with a server log reading
`File ".../vllm/v1/sample/rejection_sampler.py", line 278, repeat_indices = repeat_indices_cpu.to(`
and `RuntimeError: SUSPECT REMOTE ERROR` is not a plugin rejection-sampler defect. At upstream
vLLM `v0.18.0`, `vllm/v1/sample/rejection_sampler.py:277-280` is inside
`RejectionSampler.apply_logits_processors`: `original_indices.repeat_interleave(num_draft_tokens)`
then a non-blocking copy to the logits device. The plugin patches only the module-level
`apply_sampling_constraints`, `rejection_sample` and `expand_batch_to_tokens`
(`vllm_ascend/patch/worker/patch_rejection_sampler.py` at `v0.18.0`), not that method.
`SUSPECT REMOTE ERROR` is a *remote*-device fault surfaced at the local copy, so the frame names
neither the failing operator nor the failing node. Semantics, verified at clone HEAD
(`cann/runtime` `582cb7c3e`, `Ascend/pytorch` `988eaa3220` — not the reporter's torch_npu wheel, so
confirm the code exists in the version shipped before quoting a line): the runtime returns
`ACL_ERROR_RT_SUSPECT_REMOTE_ERROR 507057` (`cann/runtime`
`include/external/acl/error_codes/rt_error_codes.h:148`) for `RT_ERROR_REMOTE_MEM_ERROR`
("device remote mem error"), the CCU-HCCL remote form and `RT_ERROR_SUSPECT_REMOTE_ERROR`
(`src/runtime/core/src/common/errcode_manage.cc:111,115,118`), and it is also what
`TS_ERROR_SDMA_LINK_ERROR` maps to (`src/runtime/core/src/common/error_code.cc:150`) — a failing
SDMA link on the peer/transfer path arrives as this string. torch_npu prints it via
`handleSuspectRemoteError` (`torch_npu/csrc/core/npu/NPUException.cpp:123`; the two matched string
forms `reason=[suspect remote error]` / `reason=suspect remote error` are `NPUException.h:100-101`),
marks the task queue `SUSPECT_REMOTE_EXIT` (`NPUQueue.cpp:427-429`), re-peeks the last error
(`OpParamMaker.cpp:466,575`), and HCCL collectives can report it too
(`distributed/ProcessGroupHCCL.cpp:732`).

Triage consequence: 507057 indicts a peer device's memory or the inter-device link, not the code at
the frame — in a 1P1D run the other role's node or a sibling TP rank of the same replica. Ask for
the peer's own plog and `current working operator name`, check whether that peer logged
`HCCL OP RETRY FAILED` / `LINK ERROR` / `SUSPECT MEM ERROR` first, and re-run graph-free with
`ASCEND_LAUNCH_BLOCKING=1` to move the surface onto the failing op. Never quote the local sampler
frame as the cause.

The report's only other discriminator is `max_num_seqs=128` failing under 60 concurrency while
`32` passes; that is a batch-size condition, not evidence of a sampler bug. The thread's only
redirect is "try MooncakeConnectorV1", and the issue closed stale with no reproduction. Treat the
sampler frame as a surface, not a cause; this is a candidate boundary, not a verified mechanism.

## Mooncake connector: which `start_load_kv`, and the masked transfer error

Both connector files are large (`mooncake_connector.py` 3710 lines, `mooncake_layerwise_connector.py`
2090 at `v0.23.0rc1`), and names repeat inside them. Name the class before reading.

- **The layerwise file has two `start_load_kv` at the same tag** (`v0.23.0rc1`, `v0.23.0`,
  `v0.23.0.post1` and `v0.24.0rc1` share one layout): `:766 def start_load_kv(self,
  forward_context, **kwargs)` on the connector class, and `:1603 def start_load_kv(self, metadata:
  MooncakeLayerwiseConnectorMetadata)` on the connector-worker class, which dispatches per KV-cache
  group — `:1623 self._get_kv_split_metadata_for_mamba(...)`, `:1627
  self._get_kv_split_metadata(req_meta, req_idx, req_id, i)`. A frame
  `mooncake_layerwise_connector.py:1627, in start_load_kv -> _get_kv_split_metadata` therefore pins
  0.23.x exactly; grepping the def name lands on `:766`, where 1627 is out of range, and manufactures
  a false "line does not exist" mismatch. Pick the definition by its signature, not its name.
- Raising sites inside `_get_kv_split_metadata` before it returns: `assert len(selected_p_cp_groups)
  == len(selected_d_cp_groups)` (`:1496`) and `context_parallel_parameters_check(...)` (`:1516`),
  which asserts `(p.pcp*p.dcp) % (remote_pcp*remote_dcp) == 0` and, off MLA,
  `ceil(kv_heads/d.dcp) % ceil(kv_heads/p.tp) == 0`
  (`vllm_ascend/distributed/kv_transfer/utils/utils.py:88-100` at v0.23.0). The mamba sibling asserts
  `self.tp_size >= req_meta.remote_tp_size` (`:1553`) before dividing. A P-side crash whose traceback
  stops at `:1627` has not named its exception yet: ask for the final frames and the exception type
  instead of choosing one of these. Reported instance: issue #15394 (`v0.23.0-a3` image, Qwen3.5
  122B W8A8 + MTP, TP16 per role, `--enforce-eager` on P). That issue has **zero comments** — no
  maintainer guidance and no workaround to quote.
- **Two plugin log statements pin the tag better than the report's version string.** Line numbers of
  `Error in KVCacheTransferThread` (a) and `Mooncake transfer failed for request` (b) in
  `mooncake_connector.py`:

  | ref | a | b |
  | --- | --- | --- |
  | `v0.20.2rc1` | 464 | 597 |
  | `v0.21.0rc1` (= `origin/releases/v0.21.0rc` tip `b4884b42`) | 563 | 780 |
  | `v0.22.1rc1` | 588 | 817 |
  | `v0.23.0rc1` | 626 | 940 |
  | `v0.23.0`, `v0.23.0.post1` | 634 | 965 |
  | `v0.24.0rc1` | 632 | 963 |
  | `v0.25.1rc1` | 623 | 937 |
  | `v0.26.0rc1/rc2` | 642 | 1004 |

  Issue #13175 is titled `0.23.0rc1` and pastes `:565` and `:781`: that is 1-2 lines off
  `v0.21.0rc1` (563/780, a near-identical tree) and 61/159 lines off `v0.23.0rc1`. So the pasted
  D-node frames came from a 0.21-line plugin build, while the curl's
  `system_fingerprint: vllm-0.23.0-5e9d40ec` names the **upstream** vLLM version. Read the two
  numbers as separate facts; do not silently pin the plugin to 0.23.0rc1.
- **In ≤0.22 a failed transfer is reported as a masking `TypeError`.** Chain at `v0.21.0rc1`:
  `ret = self.engine.batch_transfer_sync_read(...)`; `ret < 0` logs `Mooncake transfer failed for
  request ... ret=-1` and raises (`:777-784`) -> `_handle_request`'s inner `except` calls
  `_mark_failed_recv_request(request_id, req_meta["local_block_ids"])` (`:583-586`) ->
  `self.invalid_block_ids.update(local_block_ids)` (`:545`) while the field is
  `BlockIds = tuple[list[int], ...] | list[list[int]]`
  (upstream `vllm/distributed/kv_transfer/kv_connector/utils.py:31` at vLLM v0.21.0), i.e. a list of
  lists -> `TypeError: unhashable type: 'list'`, which escapes `_handle_request` and is logged by
  `run()`'s handler as `Error in KVCacheTransferThread. error=unhashable type: 'list'` (`:562-563`).
  From `v0.23.0rc1` the helper is typed `BlockIds` and does `.update(local_block_ids[0])`
  (`:605-608`), so that masking error disappears; the transfer failure does not. Read the unhashable
  line as secondary and ask for the line before it and the transport error
  (`ascend_direct_transport.cpp:1140 Failed to connect to target: <ip>:<port>, status: 103905`), not
  the Python `TypeError`. A different status is not a failed dial: `:1107 Connect timeout to:
  <ip>:<port>, you can increase the timeout duration to reduce the ASCEND_CONNECT_TIMEOUT environment
  variable` followed by `:836 Transfer slice failed with status: 103902` means the peer was reached
  and the slice transfer failed after ~10 s of connect cost (reporter paste in a `0.20.2.rc1`
  DeepSeek-V4-Flash 2P1D comment), so keep `103902` transport-level and out of whatever capture or
  kernel diagnosis the same thread is about. In #13175 both roles run on one host, so the dial target
  is the peer role's port on the same IP; check the two serve lines' `kv_port`/handshake pairing
  before blaming the network (candidate — the P side logged `Application startup complete`).
- Garbled decode text in such a report (`External prefix cache hit rate: 100.0%` on D while P shows
  0.0%) is downstream of KV that never loaded; do not route it to the sampler references.
- **`use_ascend_direct` is read only from `kv_connector_extra_config`.** At `v0.11.0`,
  `mooncake_connector.py:905-906` is
  `vllm_config.kv_transfer_config.get_from_extra_config('use_ascend_direct', True)`, so a sibling
  key at the top of `--kv-transfer-config` is ignored — and the connector's default is already
  `True`, so a top-level placement does not turn it off. The maintainer advice in #5635 ("add
  `use_ascend_direct: true` to `kv_connector_extra_config`") is a placement rule, not an
  explanation of that failure. A second, unrelated surface defaults the other way: the store config
  file (`MooncakeStoreConfig.from_file`, read from `MOONCAKE_CONFIG_PATH`) uses
  `config.get("use_ascend_direct", False)` (`distributed/mooncake/config_data.py:427,441`), and
  `mooncake_store.py:42-44` takes the hostname-based `store.setup(...)` path when
  `protocol == "ascend"` and it is false. Keep the two configs apart when reading a report's JSON.
- **`Sync batch data transfer timeout after <N>ns` is the C++ batch deadline, not a Python
  connector fault.** Mooncake's `mooncake-integration/transfer_engine/transfer_engine_py.cpp:1192`
  (and `:1314`) waits `transfer_timeout_nsec_ + total_length` (1 GiB/s allowance) and returns
  `-1`; the plugin's `ret < 0` then logs `Mooncake transfer failed for request` and raises
  `Mooncake transfer failed, ret: -1`. Read `<N>` against the transfer timeout the launcher set —
  `30018875442ns` is the ~30 s default. In #5635 (`v0.11.0`, 1P1D on one A2 host) the P log shows
  the send completing (`Number of completed KV cache send requests: 1`) while D repeats
  `is still in WAITING_FOR_REMOTE_KVS state` until that deadline: the recv side never got the
  blocks. Next checks are the two `kv_port`/`engine_id` pairings and whether both roles ran the
  same plugin build; a commenter's raised-timeout config is an unvalidated workaround, and the
  thread's close does not name a cause.

## Side-channel handshake: the DONE_RECVING failure is swallowed

Signature: a PD service starts, runs a stress test, then stops answering new requests while both
roles log continuous ZMQ/transport errors and one side's KV-cache occupancy stays pinned at ~99%
with no recycling (issue #12383, vllm-ascend `v0.22.1rc1`, GLM-5.1 on 4 A3 nodes, P and D each
two nodes, 32-concurrency 100k-in/30k-out; the thread has two comments and no maintainer root cause).
Read this path before any scheduler or sampler code:

- The side channel is REQ/ROUTER over `side_channel_port + device_index`; the handshake helpers are
  `ensure_zmq_send` (`mooncake_connector.py:2801` at `v0.22.1rc1`) and `ensure_zmq_recv` (`:2822`),
  both defaulting to `max_retries = 3` and both raising `RuntimeError` after the last retry;
  `ensure_zmq_recv` polls first with a 1 s default timeout (a `DONE_RECVING` call passes
  `timeout=self.timeout`). The connector file is 2866 lines at that tag — count
  and grep before opening windows.
- `_send_done_recv_signal` (`:1204` at `v0.22.1rc1`; `:1162` at `v0.21.0rc1`, `:1344` at
  `v0.23.0rc1`) sends `(DONE_RECVING_MSG, request_id, remote_port_send_num)` and waits for `ACK`, but
  its `except RuntimeError` only closes the socket and logs
  `logger.warning("Unexpected error occurred in socket. error=%s.")` — nothing is retried at request
  level and nothing propagates. Its caller's own comment states the stake: "Always send the done
  signal to the remote host to ensure proper resource cleanup. Failing to do so may cause a memory
  leak on the remote host." The same `except RuntimeError` → close → warn shape is still at
  `v0.23.0rc1`, `v0.24.0rc1` and `main:1545-1549`.
- On the prefill side the ROUTER loop counts `DONE_RECVING` per handshake port and calls
  `task_tracker.update_done_task_count(request_id)` only once the expected per-device count arrives
  (`:350-364`); `_get_remote_metadata` (`:1178`) does *not* catch, so a metadata-exchange failure
  propagates while a done-signal failure does not.
- `KVCacheTaskTracker` (`:157-230`) force-frees only requests sitting in `delayed_free_requests`,
  under `envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` ("Force freed expired request", default from
  `envs.py`; the PD tutorials in the same tag set `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480`). A
  request still in `reqs_to_process` that never receives its `DONE_RECVING` is outside that timeout,
  and `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` — the name several reporters set — is not this variable.
- The ACK itself is sent from the listener thread in an unbounded `while True` around
  `send_multipart(..., flags=zmq.NOBLOCK)` with `except zmq.Again: time.sleep(0.01)` (`:369-373`), so
  a peer that closed its REQ socket (exactly what `_send_done_recv_signal` does on failure) is
  answered from the same single thread that serves every later handshake.

Candidate chain, labelled: transport error → `DONE_RECVING` dropped after 3 retries and only warned
→ prefill side never marks the request done → its KV blocks are never returned to the free pool →
occupancy climbs to the cap and new prefills stall, with further ZMQ errors as the two sides keep
disagreeing. Not established: which role's occupancy pins (a decode-side report can also be the
consequence of stalled handshakes rather than leaked blocks). Confirm with per-role plogs around the
`Unexpected error occurred in socket` warnings, a `py-spy` dump of the listener thread, and a rerun
with `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` set low to see whether the stuck requests are the ones
with a lost done-signal.

## KV transfer: `engine_id` is listen address, dial target and HCCL server identity

Verified in code (this is where the design question lives, not in vllm-ascend):

- `LMCache-Ascend v0.4.4 lmcache_ascend/v1/transfer_channel/hixl_engine.py:37-42`:
  `ip = get_ip(); port = _find_free_port(); self.engine_id = f"{ip}:{port}";
  self.engine.initialize(self.engine_id, {"BufferPool": buffer_pool})`. Identical on `main`, and
  `git grep -n 'HIXL_ENGINE\|ENGINE_IP\|getenv\|environ' main -- 'lmcache_ascend/v1/transfer_channel/*'`
  is empty — no configuration entry for the engine id.
- `get_ip()` is LMCache's (`lmcache/v1/rpc_utils.py`): a UDP `connect(("8.8.8.8", 80))` and
  `getsockname()`. Under CNI that is the Pod IP, under `hostNetwork` the node IP — two Pods on
  one host therefore advertise two different engine ids.
- `HixlChannel.lazy_init_peer_connection` exchanges `engine_id` over ZMQ and then
  `engine.connect(remote_engine_id)` (`hixl_channel.py:100-124`), so the same string is both this
  process's identity and the dial address.
- In `cann/hixl` HEAD the string is parsed as listen `ip:port` at Initialize
  (`src/hixl/engine/hixl_engine.cc:53`, `endpoint_generator.cc:712 GetHostIpFromLocalEngine`) and
  as the remote address at Connect (`src/hixl/engine/client_manager.cc:59`). The Initialize option
  table (`docs/zh/api/cpp/HIXL-interface.md:100-106`) lists `BufferPool`, `RdmaTrafficClass`,
  `RdmaServiceLevel`, `AutoConnect`, `GlobalResourceConfig`, `LocalCommRes` — there is no
  listen-address or server-identity key. `_find_free_port()`
  (`lmcache_ascend/v1/rpc_utils.py:43-46`) is the same probe-then-release pattern as vLLM's
  `get_open_port`, so the engine port can also be taken between probe and use.

Workarounds and their standing: running both engines with `hostNetwork: true` makes both report
the node IP, the rank table collapses to one server and HCCS is selected — reporter-observed, and
consistent with `get_ip()` above. The reporter's `0.0.0.0:<port>` identity plus a per-Pod TCP
forwarder also worked in their environment (full 1P1D run, RoCE counters at zero), but that is
their unverified-by-us scheme, and it leaves two open problems they name: `_find_free_port`
picks the port at runtime, and multiple P/D pairs on one node need distinct ports because the
HCCL comm name is `<local_eid>_<remote_eid>`.

Newer HIXL does expose a settable `server_id` inside `LocalCommRes`
(`docs/zh/api/cpp/HIXL-interface.md:482-536`), but the doc scopes that field to
`protocol_desc` containing `ub_ctp:host` (H2H, "用于判断传输任务是否跨机"). It does **not**
establish decoupling for the device-HCCS case in the reported CANN/HIXL 8.5.1 runtime. The
thread's maintainer answer ("engine_id is the instance id, supports separate configuration,
please ask in cann/hixl") conflicts with the code read above; take it as routing advice to
`cann/hixl`, not as proof the version in use has the switch.

Recorded environment boundary for that report: vllm-ascend `v0.18.0rc1`, LMCache-Ascend 0.4.4,
CANN/HIXL 8.5.1, 8×910B3 on one node with all eight `hccn_tool -i N -link -g` = `DOWN` (no
optical modules), so RoCE is not an option and HCCS is the only data path.

## Hybrid Mamba + KV connector: the per-group prefix-hit contract (verified)

A P/D deployment of a hybrid (Mamba/state) model can die on the **first request** with
`TypeError: 'int' object is not iterable` raised at
`vllm/v1/core/sched/scheduler.py:701  num_new_local_computed_tokens = max(per_group_hits)`,
reached through `vllm_ascend/patch/platform/patch_balance_schedule.py:152`
(`return super().schedule(throttle_prefills)` — that frame means balance scheduling is *off* and the
upstream scheduler is running on the plugin's coordinator; with balance on, the plugin's copied
`scheduler()` has the same `max(per_group_hits)` at `patch_balance_schedule.py:456`).

Verified chain at `vllm-ascend v0.24.0rc1` + `vllm v0.24.0`:

- Upstream v0.24.0 added `HybridKVCacheCoordinator.find_longest_cache_hit_per_group`
  (`vllm/v1/core/kv_cache_coordinator.py:734`, returning
  `(blocks_per_group, tuple[int, ...])` at `:738`) and its scheduler calls it in the
  `self.connector is not None and self.has_mamba_layers and isinstance(coordinator,
  HybridKVCacheCoordinator)` branch (`scheduler.py:675-701`), then takes `max()` of the per-group
  hit lengths. `vllm v0.23.0` has no such method at all.
- The plugin defines the same-named method on `AscendHybridKVCacheCoordinator`
  (`patch_kv_cache_coordinator.py:70`), which its own `get_kv_cache_coordinator` installs as
  `vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator`, with the 0.23 contract:
  `-> tuple[tuple[list[KVCacheBlock], ...], int]` (`:365-369`), already present at
  `v0.23.0rc1:352`. Its purpose is the opposite of upstream's: return the FullAttention hit, not
  per-group lengths, so a D node's zero Mamba hit does not collapse the FA prefix
  (`vllm_ascend/patch/__init__.py:425`).
- The upstream call site now receives one `int`, so `max(int)` raises. Same model/config on
  `0.23.0rc1` works because 0.23's scheduler never calls that name.
- Fix `87ba94102` "[1/N][Refactor][Core] remove function find_longest_cache_hit_per_group in
  patch and support DSV4 decode partial group hits (#12761)" removes the patched function and
  queries the upstream per-group API instead. `git tag --contains 87ba94102` = `v0.27.1rc1`,
  `v0.28.0.rc1`, `v0.29.0.rc1`, `v0.30.0rc1/rc2`, `v0.30.0`; it is **not** in `v0.24.0rc1` and
  **not** on `origin/releases/v0.24.0rc` (tip `b19127c7e` keeps the `int` annotation), so "move to
  the next 0.24 rc" is not a supported statement.

Applicability: hybrid/Mamba model **and** a KV connector on that node **and** the hybrid KV cache
manager (`--no-disable-hybrid-kv-cache-manager`; default on) **and** prefix caching on. Drop the
connector and the scheduler takes the other branch, where the int is correct. Reported instance
#13998 (`0.24.0rc1`, Qwen3.5-35B-A3B-w8a8-mtp, A2 910B2C, 1P1D with `MooncakeConnectorV1` on
both roles, `--mamba-cache-mode align`): the pasted frame `scheduler.py:701` plus the plugin frame
`:152` pins this class.

Two independent checks support it. The plugin's `get_kv_cache_coordinator` delegates to upstream
when `len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching`
(`patch_kv_cache_coordinator.py:501-518`); with caching off that is upstream's
`KVCacheCoordinatorNoPrefixCache`, which is not a `HybridKVCacheCoordinator`, so the scheduler's
isinstance guard fails and the other branch runs — and the reporter found that turning prefix
caching off on the P node lets inference run. The reporter also confirmed 0.23.0 works. The thread
is still **open** with no maintainer root cause: a would-be contributor guessed "config field type
change", and the two workaround comments are the reporter's own.

## Intermittent corruption on a C8/MXFP8 KV cache under concurrent PD transfer (`v0.30.0`)

Report shape: PD split on a checkpoint whose KV cache is quantized to C8/MXFP8, one request
corrupting and the next clean (or a fixed fraction of churn requests), NaNs or repeated-exclamation
output, no traceback. Two defects share this surface and they are not both on `main`:

- **K-scale write race.** C8 K-scale assignment used NPU advanced indexing, which can materialize a
  packet-strided cache view and write it back through `IndexPutV2`/`ViewCopy`, overwriting a
  concurrent KV transfer to an unselected page; the resulting invalid E8M0 scales produce NaNs and
  repeated exclamations. Fixed by a stride-aware Triton byte scatter in
  `vllm_ascend/ops/triton/mxfp_k_scale.py` — `main`: `e6c0f4b7d` (#19064), release: `14c6c3f43`
  (#19017). `git tag --contains e6c0f4b7d` is empty (main-only); `git tag --contains 14c6c3f43` is
  exactly `v0.30.0`, so `v0.30.0` is the first tag carrying it and `v0.30.0rc1` is not.
- **V-scale restore after a partial async import.** A pull can populate three of the five kernel
  blocks of a decode-manager page while bypassing the ordinary allocation preparation, so static V
  scales must be restored for eligible active C8 consumer requests after Pull receive completion
  and the Mamba copies, before forward. Scope is the exact built-in `MooncakePullConnector`
  including its registration aliases, with no synchronous loads. This half rides in the `v0.30.0`
  backport only; the commit body says the V-scale main PR #19016 "remains on hold" and #19065
  tracks it, so do not promise it on `main`.

The alternating pattern is the defect's own recorded signature: the #19017 body reports a 397B
four-request A/B/A going **4/4 → 0/4 → 4/4** corruption with the restore (GPQA 87.37%, 173/198
after) and a 35B A/B/A of **27/512 → 0/512 → 3/512** for the K-scale half, plus "2048/2048 churn
requests without corruption". Those are the authors' runs in a PR body, not a re-run here: treat the
counts as the shape of the defect, not as a pass criterion.

Applicability, and why an older tag is a different case:

- **The code both fixes touch exists only from `v0.30.0rc1`.**
  `vllm_ascend/attention/attention_c8_mxfp.py` is absent at `v0.23.0`, `v0.24.0rc1`, `v0.26.0rc1`,
  `v0.28.0.rc1` and present at `v0.30.0rc1`; `ops/triton/mxfp_k_scale.py` is present at `v0.30.0`
  only (`git cat-file -e <tag>:<path>`). A C8 report on an earlier tag is not this mechanism: at
  `v0.23.0` the C8 path keeps per-head `k_cache_scale`/`v_cache_scale` parameters
  (`vllm_ascend/quantization/methods/kv_c8.py`) instead of per-token E8M0 scales.
- **C8 is selected by the checkpoint, not by a CLI flag.** `ModelSlimConfig._add_kvcache_quant_metadata`
  (`v0.23.0`, `vllm_ascend/quantization/modelslim_config.py:934-952`) sets
  `enable_c8_quant = quant_description.get("kv_cache_type") == "C8"` and collects the C8 layers from
  `k_proj.kv_cache_scale` keys, so a `...-FULL_QUANT` weight directory can be on the C8 path with a
  serve line that says only `--quantization ascend`. Ask for that description (or the log line
  naming the C8 attention method) before attributing anything to this class.

Recorded instance: `vllm-ascend#12339` — Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT on `releases/v0.23.0`
(`6cee2e163`), 1P1D with `MooncakeConnectorV1` on both roles,
`--no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager`,
`additional_config.recompute_scheduler_enable`, "one request normal, the next not, alternating".
The thread is closed with three comments — a ping and the reporter's "已解决" — so **no maintainer
mechanism** exists: answer with the class, the `v0.30.0` boundary and the checkpoint check, not with
a root cause at `v0.23.0`. The hybrid per-group prefix-hit defect above cannot apply there (prefix
caching is off in that launcher).

## Legacy `llmdatadist` connector (0.9.x-0.11.x): retired, and what its 120 s warning means

`vllm_ascend/distributed/llmdatadist_c_mgr_connector.py` exists at `v0.9.1` (923 lines) and is gone
from `v0.12.0rc1` onward: `a77045f35` "[P/D][main]Offline the llmdatadist connector related parts of
the code and files. (#4780)" states that mooncake support replaced it and it is no longer maintained.
So any report naming `llmdatadist_c_mgr_connector.py` (or `llmdatadist_connector.py`) is answerable
only on the 0.9-0.11 line, and the durable advice is to move to the mooncake connector instead of
patching the retired one.

Its transfer model at `v0.9.1` explains a "TTFT is enormous / prefill node never gets the done
signal" report without any kernel reading:

- the worker owns a single-thread executor (`self.executor = ThreadPoolExecutor(1)`, `:324`) and
  submits every request's `_read_blocks` to it (`:606`), so decode-side pulls of different requests
  queue behind one another;
- each task first does a **per-request** ZMQ `REQ` metadata round-trip to the P node
  (`connect_to_remote_agent`, `:770`, new socket per call) and then one or two blocking
  `cache_manager.pull_blocks` calls with `llm_config.sync_kv_timeout = 20000` ms (`:431`,
  timeout messages `:847-869`).
- the warning belongs to the **P node**, not to a P-side wait for the first token: when a prefill
  request finishes, its scheduler `request_finished` (`:261`, deadline at `:286`) delays freeing the
  blocks and records `deadline = now + VLLM_LLMDD_ABORT_REQUEST_TIMEOUT` in the scheduler class's
  `_reqs_need_send` (`:187`); that dict is deep-copied into the metadata's `reqs_to_send` (`:253`),
  so the symptom is drained on the **worker** side — its `get_finished` (`:875`) walks
  `self.reqs_to_send` and logs the warning (`:887`; offset 886 in the reporter's build)
  "Some requests in prefill node fail to receive KV Cache transfer done signal in {...}s" for entries
  whose deadline passed. The handshake that cancels it is the decode node's per-TP-rank
  `ReqForFinished` (event id `:48`, sent by the D worker at `:792`, handled on P at `:388` and
  popped from `reqs_to_send` at `:401`), so the warning means the D node never confirmed within the
  window — not that a token was delayed on P. Grep both dict names when checking a build; a paste
  that shows only one of them is still this connector.
- `export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600` (default **120 s**: `vllm_ascend/envs.py:141`) is
  what the log itself suggests and only postpones that fallback; it neither speeds up D nor removes
  the serialization, and it keeps P's blocks held longer.

- **An `EngineDeadError` with only the API-server stack is not one of this connector's
  exceptions.** At `v0.9.1` `start_load_kv` (`:602`) submits each request's `_read_blocks` to the
  single-thread executor (`:324`) and attaches `handle_exception`, which only logs
  `KV transfer task failed: {exception}` — so a failed pull (`pull_blocks` timeout, `:847-869`), a
  failed `link` (`:741`) or the metadata listener's raise (`:405`) never propagates into the engine
  step. A stress report ending at `EngineDeadError: EngineCore encountered an issue. See stack trace
  (above) for the root cause` has the engine-side traceback outside the pasted body; answer with the
  retirement, the migration and the timeout knob, and label the death mechanism unknown. Recorded
  forwards on `vllm-ascend#3055` (0.9.1, 910C, aisbench) read `:596-700` and `:720-902` looking for a
  raise and found none; the thread closed `stale` by bot with no maintainer answer (`wait-feedback`,
  one collaborator "may have been resolved in the latest version").

Boundary: the single-worker serialization, the per-request metadata round-trip and the timeout are
read from source; "TTFT grows with concurrency because pulls serialize" is the causal reading and was
not measured, and the matching thread has no maintainer root cause (one collaborator comment asks
whether the issue still exists).

## Reading order that saves steps

1. Pin each side to its tag (plugin, upstream vLLM, LMCache-Ascend, and `cann/hixl` HEAD for the
   transport layer) — see [revision-pinning.md](revision-pinning.md).
2. Decide the stage first: start-up crash, connect failure, or steady-state hang. Only the hang
   case needs the worker/scheduler code.
3. For a maintainer-named suspect flag, `git grep -n '<FLAG>' <tag> -- 'docs/*'` before reading
   code; the tag's tutorials often already exclude the reporter's topology.
4. Reports in this class frequently paste only `collect_env.py`, `npu-smi` and a Feishu/QR link
   instead of the serve command. Do not reconstruct the config from the body — the maintainer
   comments (or a duplicate issue) are the only place it appears.

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **A second exception can be the mask, not the fault.** `Error in KVCacheTransferThread. error=
  unhashable type: 'list'` is raised while *marking* a failed recv request (`set.update(<nested
  BlockIds>)`) after the transfer already returned `ret<0`; read the line before it
  (`Mooncake transfer failed ... ret=-1`, `ascend_direct_transport ... status: 103905`) and treat the
  `TypeError` as secondary.

- **"One request good, the next bad" on a quantized KV cache is the C8/MXFP8 KV-scale transfer
  class before it is a sampler or scheduler question.** Check the checkpoint's `kv_cache_type`
  (`"C8"` → on the C8 path regardless of the serve line) and the tag: the K-scale fix is
  `e6c0f4b7d`/#19064 on `main` and `14c6c3f43`/#19017 in `v0.30.0` (not `v0.30.0rc1`), the V-scale
  restore is `v0.30.0`-only, and the `attention_c8_mxfp.py` path does not exist before
  `v0.30.0rc1`, so an older tag's alternating report is a different defect. Section above.

- **A PD spec-decoding report (acceptance collapsed, or an `mtp` throughput curve) is a recipe
  question before a hidden-state one.** Check what the tag's own PD tutorials prescribe (at
  `v0.18.0rc1` only `deepseek_mtp` with `num_speculative_tokens: 1`; `eagle3` is documented for
  colocated serving and its drafter runs without TP) and whether all decode ranks share one graph
  mode, then ask for the colocated control on the same build. Both shapes have a section above.

- **A one-version regression on the *first request* of a hybrid P/D model can be a plugin override
  of a method upstream just added.** `TypeError: 'int' object is not iterable` at
  `scheduler.py:701 max(per_group_hits)` is that shape at `0.24.0rc1`: the plugin's
  `AscendHybridKVCacheCoordinator.find_longest_cache_hit_per_group` kept the old `int` return
  while the paired upstream scheduler started unpacking per-group lengths. Compare the method's
  return annotation in `vllm_ascend/patch/` against `vllm/v1/core/kv_cache_coordinator.py` at the
  paired tags before reading attention or KV-cache kernels; the same name with two contracts is
  the defect (section above).
