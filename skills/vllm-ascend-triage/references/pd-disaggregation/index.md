# PD/EPD disaggregation: transport plane and KV-state transfer

Scope: a P/D (or EPD) deployment whose *requests or KV* do not arrive: the peers
cannot connect, they connect and the decode node produces wrong tokens, or the
deployment never receives a request at all (proxy/serve wiring), or one side's
EngineCore dies on its own connector/scheduler bookkeeping while the other keeps
running — *not* the generic startup lane. The families below are first split by the
pasted stack (transport, output corruption, request routing, or a bookkeeping assert
on one side), then by two questions: **is a KV pool in the deployment?** and **did
the peers fail to connect, did they connect and the decode node then produce wrong
output, or did no request ever reach a decode engine?**

- A KV pool (MemCache/Mooncake `AscendStoreConnector`) on a **single node, no PD**,
  or a store that initialises and then fails while saving: this topic's transport
  rules do not apply — `../kv-pool-store/index.md` (A3 + DeepSeek-V4 +
  `device_sdma`: `../kv-pool-store/memcache-a3-dsv4-init.md`).
- Connect/handshake failures **with a KV pool in the config**
  (`AscendStoreConnector`, alone or inside a `MultiConnector`, `MOONCAKE_CONFIG_PATH`,
  `pool_worker.py` / `pool_scheduler.py`): `Failed to connect to target … 503900`
  / `E19999` / `HcclCommPrepare … ret: 0x13` on the first transfer, garbled output
  after it, `kv_load_failure_policy: recompute` recomputing every request → the
  **store segment size** (`global_segment_size`) before anything else, then
  `kv-pool-store-segments.md`.
- Connect/handshake failures **without** a pool: `HIXL Connect failed | HIXL
  Error Code: 103900` (or 503900 and friends), `LMCache ERROR: wait_for_save
  failed ... skipping save`, HCCL `RanktableCheck` / `'device_ip' is not set
  correctly`, `get dev list error:serverId[...]`, P/D pairs that only work with
  `hostNetwork: true`, `--net=host`, or one shared IP →
  `transport-topology.md`.
- The same *no-pool* failure with the **Mooncake** connector instead of HIXL:
  `E … ascend_direct_transport.cpp:820] Failed to connect to target: <P ip>:<port>,
  status: 103900` + `:581] Failed to connect to segment` + `mooncake_connector.py:410]
  Mooncake transfer failed … ret: -1`, first proxied request, every TP rank within
  milliseconds, while both sides answer their own `/v1/completions`. Ask for the
  decode node's plog **before** any `ss`/`nc`/`hostNetwork` test (a cross-node P/D is
  multi-server by construction, so `hostNetwork` cannot collapse it) and do not let
  an "add `use_ascend_direct`" hint spend a round — at `v0.11.0rc2` it already
  defaults `True` → `mooncake-adxl-connect-103900.md`.
- Connected, but the answer is corrupt/empty/alternating, or the decode node's
  first token is wrong, on a **hybrid linear-attention (GDN/Mamba)** model —
  Qwen3.5/Qwen3.6, DeepSeek-V4 — under PD → `hybrid-state-transfer.md`.
- Connected, and an **image** request returns a fluent but unrelated short answer on a
  **VL/OCR** model (Qwen2.5-VL family, DotsOCR, olmOCR) — while a single-instance serve
  of the same body, image and flags answers it — is not a transport case: the control
  run, `usage.prompt_tokens` on a good and a bad run, and the `--no-enable-prefix-caching`
  flag the 1P1D recipe pins on both sides are the checks →
  `../accuracy-benchmarks/vl-vision-tower-degeneracy.md` §*PD/EPD split*.
- Connected, and the decode node **repeats** the same token or phrase to the model
  cap (`finish_reason: length`, e.g. `1 + 1 = 2.` ×2048) or otherwise answers wrong
  text on a **DSA/MLA sparse** model — GLM-5/5.2 (DSA), DeepSeek-V3.x (MLA), which are
  not GDN/Mamba hybrids whatever the connector name suggests — with both serve commands
  in the report; **or** the same deployment dies as `ACL stream synchronize failed`
  only when the batch / sequence / capture-size knobs are raised, with no log pasted
  → do not start in `kv_p2p/`: diff both commands against
  `docs/source/tutorials/models/<model>.md` first (connector class,
  `--speculative-config` on **both** sides, the HMA flag, DSA sparse gates) →
  `recipe-contract-diff.md`.
- Connected, the answer degenerates only at long context **and** concurrency on a
  **dense-attention MoE** model (Kimi-K2.x MLA, Qwen3-MoE) with spec decode and/or
  prefix caching, and both serve commands are in the report: do not start in
  `kv_p2p/`. Diff the two commands against
  `docs/source/tutorials/models/<model>.md` first and hand back that diff plus the
  ablation rows (ranked, none an established cause; #13002's first run never
  reached a diagnosis, and its replay reached one from that page alone) →
  `../accuracy-benchmarks/garbled-long-context-output.md`.
- The client times out and the decode instance logs **no request at all**, while the
  only pasted error lines sit on the *prefiller* (`mooncake_connector.py … Force freed
  expired request … exceeded timeout threshold (480 seconds)`, several ranks, same
  ids): that line is P-side delayed-free bookkeeping, and the deciding evidence is the
  proxy's `--decoder-hosts/--decoder-ports` against how many API servers the DP layout
  actually starts (`--api-server-count 1` on the head node plus `--headless` on the
  others = one address per side) → `proxy-endpoint-vs-api-server-count.md`.
  Reporter-confirmed on #12432: the fix was writing only the DP master's IP and the
  start port per role.
- `/v1/chat/completions` works through the example proxy while `/v1/messages`
  (Anthropic), `/metrics`, `/v1/models`, `/tokenize` or `/start_profile` cannot be
  called ("需要对所有 vllm 的接口都做兼容"): an **endpoint-coverage gap of the example
  script**, inherited from upstream's `toy_proxy_server.py`, not a serve or
  transport fault. The answer is the route table plus the three tiers — P/D-split
  generation endpoints, `kv_transfer_params` check first; stateless passthrough with
  `/metrics` and the profilers fanned out; docs — and two silent traps on
  `/v1/messages` (the `recomputed` marker is erased by the Anthropic stop-reason map,
  and the retry rewrite assumes string `content`): `proxy-example-endpoint-coverage.md`.
- Requests **are** served on `LLMDataDistCMgrConnector` but the prefiller prints
  `Some requests in prefill node fail to receive KV Cache transfer done signal in 120s.
  … 'export VLLM_LLMDD_ABORT_REQUEST_TIMEOUT=600' …`, alone or alongside "TTFT extremely
  slow": that line is P-side expiry bookkeeping and the non-zero tp ranks emit it **by
  design** (the advertised port carries no `tp_rank`), so it is not the slow-TTFT
  mechanism — settle TTFT with the two discriminators in
  `llmdatadist-done-signal-and-ttft.md` before any connector or rank-table theory.
- The deployment never gets past engine start because the *KV-transfer config
  block itself* is rejected: `ValueError: KV transfer 'prefill' config has a
  conflicting data parallel size. Expected 1, but got 2.` (or `tensor parallel
  size`, or the `'decode'` role) from `vllm_ascend/utils.py check_kv_extra_config`
  → `../startup-failures/arg-validation-errors.md`. Nothing about transport is
  wrong yet; the compared numbers are `parallel_config.data_parallel_size` /
  `tensor_parallel_size` after upstream has rewritten a **dense** engine's DP to
  1, so the fix is the block's number, not the launch's DP flag.
- **No deployment yet**: the question is a capability verdict about the *transport
  plane* before any run — "Atlas 800T A3 双机 PD 分离，两台机器只能走参数面 RoCE，
  超平面不通、没有灵衢交换机，是否支持", usually with no log and no serve command →
  `plane-selection-a3-parameter-plane.md`. This is not the failure lane above: the
  deciding reads are the A3 `device_ip` branch of the rank-table generator and the
  `server_count` / `super_pod_list` the connector hands HCCL.
- The engine started and the **first request** kills one side's EngineCore while
  the APIServer reports only `EngineDeadError: EngineCore encountered an issue.
  See stack trace (above)`, on a deployment that sets
  `--decode-context-parallel-size` > 1 (a KV pool may or may not be attached):
  DCP is a **matched** launch parameter — the prefill side advertises its
  `dcp_size` in the handshake and the peer asserts divisibility against its own,
  so a dcp that differs between the sides (or was reduced on one side only)
  raises on the first transfer. Compare the two serve commands before any source
  walk → `../feature-enable-recipes/context-parallel-pcp-dcp.md` §3
  (reporter-confirmed #12677: "PD 需要同并行 dcp", the 1M recipe is `dcp=8` on
  both sides, and the MemCache pool is not the blocker).
- The **P node's** EngineCore dies with a bare `AssertionError` in upstream
  `vllm/v1/core/sched/scheduler.py` `_update_from_kv_xfer_finished`
  (`assert req_id in self.requests`, `update_from_output` and
  `step_with_batch_queue` above it), the APIServer then reports
  `EngineDeadError` and D stays alive: a send-completion arrived for a request the
  scheduler already freed. The stack is also a launch-flag fingerprint — a batch
  queue means `--async-scheduling` (or PP>1) on the prefiller — and the assert
  names no id, so the two DEBUG lines (`Delaying free …`, `Finished sending …`)
  decide the branch → `prefill-scheduler-kv-xfer-assert.md`.
- The **prefiller's** worker dies inside `start_load_kv` at the
  `_get_kv_split_metadata(...)` call (`mooncake_layerwise_connector.py:1627` at
  v0.23.0), the traceback cut at that line with no exception class, while D stays
  alive — physical TP/DP the same on both sides, typically a hybrid attn+mamba
  model (Qwen3.5/3.6) with MTP under chunked prefill. That paste carries no
  message, so the answer is the throw-site inventory (`:1496` on this path;
  `:1553` belongs to the MambaSpec sibling, `kv_transfer/utils/utils.py:96/98-100/108/118/129/290/291-294/304`), the
  `VLLM_LOGGING_LEVEL=DEBUG` line that prints its inputs, and the three
  single-axis ablations → `layerwise-split-metadata-crash.md`. Symmetric TP means
  `asymmetric-tp-pull-count.md` (different TP sizes, non-layerwise connector) is
  not this page.
- The **prefiller's** EngineCore dies with `RuntimeError: cancelled` out of the
  response queue (`patch_shm_broadcast.py:70` / `shm_broadcast.py:701`) under the
  `dump_input.py` dump block, with no connector frame anywhere and no worker
  traceback — "短请求 curl 正常，百k输入 prefill 节点挂了" is one recorded title.
  This is **not** a transport, connector or KV-ownership fault: one worker process
  died and the executor cancelled its queue, so grep `died unexpectedly (exit code:
  N)` and rank the environment (driver/firmware × CANN) plus the long-chunk kernel
  lanes → `../hccl-and-hardware/cancelled-executor-teardown.md` (reporter-confirmed
  #16023: a CANN/HDK firmware+driver mismatch, solved by the upgrade, not a patch).
  A `start_load_kv` frame or a connector assert in the same paste outranks this
  bullet.
- The **decode** worker dies before any byte moves when P and D use different TP
  sizes (P `TP=8`, D `TP=2`) on an MLA (DeepSeek/Kimi) or DSA-sparse model:
  `chosen_rank_list([…]) does not match num_group_pulls(…) and prefill pp size(1)`
  — the group pull helper is the one pull-count derivation that skips the
  MLA/sparse single-pull rule → `asymmetric-tp-pull-count.md`.
- The **decode** worker dies on the first request with a bare `IndexError: list index
  out of range` in `kv_p2p/mooncake_connector.py` `_get_kernel_block_ids`
  (`meta.remote_block_ids[kv_cache_group_id]`, `_get_kv_split_metadata` above it):
  the decode side walks its **local** KV-cache groups against the prefiller's list,
  so a group **count** that differs between the sides — a `--speculative-config`
  (DSpark/MTP draft) on one side only, or a group-layout flag moved on one side —
  raises before any byte moves, and the handshake only DEBUG-logs the mismatch. Diff
  the two `--speculative-config` blocks before any connector read →
  `kv-group-count-mismatch.md` (reporter-confirmed #16177). A symmetric-TP pull-count
  assert and a layerwise traceback cut at the same call are the two neighbours.
- The **decode node** logs `Error in KVCacheTransferThread. error=unhashable
  type: 'list'.` during a loaded P/D deployment and keeps serving: that sentence
  is the recv thread's catch-all, and the traceback it replaces was the real
  failure — the downstream `_mark_failed_recv_request` does
  `set[int].update(<list[list[int]]>)` before the `logger.exception` that would
  have named it, so the paste carries **no** transport information, and the
  request may go on to decode on partial KV → `kv-transfer-thread-masked-error.md`
  (unmask with the one-line flatten or the layerwise connector, then classify).
- The same thread prints a **bare** `AttributeError` on a metadata field —
  `Error in KVCacheTransferThread: 'ReqMeta' object has no attribute 'block_ids'` —
  and the issue carries no environment block: the pin's own class has that field as a
  compat property over `block_ids_by_group`, so this is a **build** mismatch between
  the nodes, and the first artefact is per-node `vllm_ascend.__file__` plus that
  class's source, not a reader hunt → `node-build-skew-attribute-error.md`
  (reporter-confirmed #10398: P and D were not on one code version).
- Nothing connects at all because the connector module itself fails to import —
  `ImportError: ascend_transport.so: cannot open shared object file` at
  `kv_p2p/mooncake_connector.py:24`, `No module named mooncake` — or the decode
  side dies in `register_kv_caches` with an `AttributeError` on a config scalar →
  `../image-dependency-faults/launch-env-and-library-path.md` and
  `../startup-failures/connector-config-attributes.md` respectively; no transport
  or rank-table theory applies before the module imports.
- The **whole deployment** hangs at once after minutes of healthy serving, with no
  error line anywhere, the API server still listening, `curl` silent and `npu-smi`
  showing `AICore 0 %` on every chip: neither the proxy nor the KV connector is the
  suspect, and AICore 0 % does not make it a host stall — a rank blocked inside a
  multi-node MC2 all-to-all looks identical. Take
  `../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md` (plog, heartbeat
  `STUCK`, `count of currently executed operators`) before `py-spy` or a `/metrics`
  read, and return to this list only if the decode node's own log shows no request
  arriving at all.
- The **decoder** dies with `OverflowError: out of range integral type conversion
  attempted` (host Python, inside the tokenizer) on a request carrying
  `prompt_logprobs`, or `echo: true`: the decode node never runs the prompt forward,
  so prompt-derived outputs cannot exist there and the fix is a split — strip the
  field for the decoder and/or harvest it on the prefiller. The mechanism (upstream's
  uninitialised `torch.empty(int32)`), the `echo` alias on both endpoints, what the
  prefiller can return, and the three patch sites in the example proxy →
  `prompt-side-outputs-on-decode-node.md`.

Inherit the rules in SKILL.md. Rules specific to this topic:

1. **The transport identity is a topology input, not a tunable.** Any Ascend
   transport that hands HCCL one `ip:port` string per process (HIXL/ADXL through
   LMCache-Ascend or Mooncake's asc-end-direct transport) uses that same string
   as the HCCL rank-table `server_id`. Two processes on one physical node with
   two different addresses are therefore *two servers* to HCCL, which then
   demands `device_ip` and a NIC data path. Before reading connector code,
   establish what the rank table says and run
   `bash <skill_root>/scripts/ranktable-check.sh <table-or-log>`; it prints the
   effective `server_count`, the per-server ids and the decision that follows.
2. **A transport error code is not a diagnosis.** HIXL/HCCL numeric codes
   (`103900`, `503900`, `507057`) carry almost no information and two different
   codes are usually different failure classes. The reason lives in
   `~/ascend/log/debug/plog/plog-<pid>_*.log`; ask for that file, not for a
   better Python traceback. The exception is a pool deployment, where the
   deciding number is already in the worker log (`register mem … mem type:host`) —
   see rule 4 and `kv-pool-store-segments.md`.
3. **"Not in the documented recipe" is not "unsupported".** The LMCache-Ascend
   examples state RoCE as a prerequisite and mark `hixl` experimental, while
   reporter evidence shows the same channel completing over HCCS once the rank
   table converges to one server. Keep the two claims apart: no validated path
   vs. impossible on that hardware.
4. **`503900` is two faults wearing one code.** Decide with one question: is a
   KV pool configured? The pool's store segment is allocated with
   `aclrtMallocHost` and registered into the ADXL engine of that process — the
   same engine the p2p connector dials with (`transfer_executor_base.cpp:404-452`,
   single `global_te`), so an oversized `global_segment_size` breaks the p2p
   connector that dials first and the reported owner is the wrong one.
   Reporter-confirmed on #13439: both
   `MooncakeConnectorV1` and `MooncakeHybridConnector` fail identically *with*
   the pool, and PD without the pool is fine, which no p2p-connector change can
   explain. Ask for the file behind `MOONCAKE_CONFIG_PATH`, or run
   `bash <skill_root>/scripts/kvpool-segment-check.sh <json|log>` on what you
   have; then use `kv-pool-store-segments.md`.
5. **Name the owning repository.** vllm-ascend's `LMCacheAscendConnector` is a
   shim that re-exports upstream `LMCacheConnectorV1`
   (`vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py:1-5`
   at v0.23.0); an HIXL engine-id / rank-table change belongs to
   `LMCache/LMCache-Ascend` or `cann/hixl`, not to vllm-ascend.

## Pages

- `prompt-side-outputs-on-decode-node.md` — the decoder dies with `OverflowError: out
  of range integral type conversion attempted` on a request carrying `prompt_logprobs`
  (or `echo: true`): the node that skips the prompt forward cannot produce
  prompt-derived outputs, so the deliverable is a split. Upstream's
  `torch.empty(int32)` mechanism with its `path:line`, the `echo`→logprobs alias on
  both endpoints, the prefiller-harvest check in one curl, and the three sites in the
  example proxy that has no handling at all (recorded #13363).
- `proxy-endpoint-vs-api-server-count.md` — the deployment runs and the *requests* do
  not arrive: "D 实例未接收到请求" / `proxy通信超时` with P-side `Force freed expired
  request … (480 seconds)` lines. Why that line is a prefiller-only bookkeeping record
  (`kv_role == "kv_producer"`) and the recipe's own export, how `--api-server-count` /
  `--headless` / `--data-parallel-start-rank` decide that each side has exactly one
  listening address at the **start** port, the recorded per-DP-rank proxy list that
  dials closed ports and the proxy's quiet failure mode, plus the two-part request for
  logs. Worked for #12432 (reporter-confirmed) and pairs with
  `../accuracy-benchmarks/long-output-repetition.md`'s note on the same log line.
- `plane-selection-a3-parameter-plane.md` — a *pre-deployment* plane question with
  no log (A3 双机, parameter-plane RoCE only, no UB/灵衢 switch): what the tree
  validates (an A2-only PD tutorial, no SoC-qualified feature row), the A3 branch of
  `examples/disaggregated_prefill_v1/gen_ranktable.py` (`device_ip` from
  `hccn_tool -vnic -g` where A2 uses `-ip -g`), the `server_count` /
  `super_device_id` / `super_pod_list` the connector declares to HCCL and the
  `None == None` trap of a hand-edited A3 table, the `hccn_tool` ping block plus the
  minimal 1P1D ladder, and why `cann/hccl`, `cann/hcomm` and the Mooncake export
  cannot answer the rest. Worked for #3959 (v0.11.0rc0).
- `mooncake-adxl-connect-103900.md` — a Mooncake (`MooncakeConnectorV1`, no pool)
  deployment that serves on both sides but fails the **first** KV pull with
  `ascend_direct_transport.cpp:820 Failed to connect to target … 103900` and
  `Mooncake transfer failed, ret: -1`: what the paste already fixes (handshake
  reached, nothing timed out, 103900 is a compute-stack id, not an errno), the
  D-side plog grep that decides (recorded #4866: `'device_ip' is not set correctly`
  → `HcclCommInitClusterInfo failed, rankNum[0], server[]` → no segment port bound),
  why cross-node `server_count = 2` makes `device_ip`/`hccn_tool` the fix and
  `hostNetwork` the wrong instrument, the `use_ascend_direct` gate at `:905-911`
  (default `True`, engine-name only, extra-config placement a no-op at that pin),
  the ranked causes and the evidence limits (Mooncake's ascend transport absent from
  the export; #4866 left open).
- `transport-topology.md` — P/D has two Pod IPs on one node and the transport
  will not connect: how the engine id becomes HCCL's `server_id`, the plog grep
  set and rank-table decision, `HCCL_INTRA_ROCE_ENABLE` / `HCCL_INTRA_PCIE_ENABLE`
  semantics (source-verified in the `cann/hcomm` export), port-band collisions,
  and what is verifiable in the snapshot when `cann/hixl` is absent.
- `kv-pool-store-segments.md` — a KV pool is in the deployment and ADXL channel
  creation fails (`503900` / `E19999` / `HcclCommPrepare ret: 0x13`) with garbled
  decode output: `global_segment_size` per rank versus the HDK ceiling
  (< 20 GB on 25.2.3, reporter-confirmed), the shared-`global_te` mechanism that
  makes a store fault surface on the p2p connector, the log's
  `register mem … mem type:host, length:N` fingerprint, the two ablations, and
  the documented 1 GB alignment / fabric-mem budget. Also says which knobs are
  *not* this one (`kv_buffer_size`, `HCCL_INTRA_ROCE_ENABLE`, the timeouts).
- `recipe-contract-diff.md` — the decode node corrupts or the deployment dies only
  at raised scale: the recipe contract to diff **before** any `kv_p2p/` walk, with
  the v0.23.0rc1 anchors for GLM-5.2 / Qwen3.5-397B / DSv4 (connector class, MTP on
  both sides, the HMA flag being the hybrid families' flag, DSA sparse-gate parity),
  the divergences that are *not* the mechanism (decode `--max-num-batched-tokens`,
  `--block-size`, `enforce_eager` inside `--speculative-config`), the one-launch-per-row
  ablation ladder, and the no-log scale-fault variant with two verified non-answers
  (GDN chunk-meta buffer arithmetic, #7863's 310P-only masked assign). Worked for
  #13381 and #9578 at v0.23.0rc1 / v0.19.1rc1.
- `hybrid-state-transfer.md` — PD on a hybrid linear-attention model: the
  connector matrix (`MooncakeConnectorV1` / `MooncakeHybridConnector` /
  `MooncakeLayerwiseConnector` / `AscendStoreConnector`), the
  `mamba_cache_mode` gate that only the KV-store connectors satisfy, which
  runner paths that skips, the MTP-under-PD accounting sites (`num_draft_layers`,
  the MambaSpec `num_speculative_tokens` block index, the `"mtp" in layer_name`
  predicates) together with the refuted "our method name is not `mtp`" theory,
  and the ablation matrix to hand the reporter when the symptom is alternating
  good/bad output.
- `prefill-scheduler-kv-xfer-assert.md` — the prefiller dies on
  `assert req_id in self.requests` (upstream `v1/core/sched/scheduler.py`): how the
  frame set identifies the scheduler instance and the version, why
  `step_with_batch_queue` present means `--async-scheduling`/PP>1, the
  `delay_free_blocks` path and the connector's exact `FINISHED_LENGTH_CAPPED` test,
  why the assert carries no request id, and the two DEBUG lines that separate a
  duplicate/late send report from an abort-status free. Verified at vllm v0.12.0
  `4fd9d6a85c00` / vllm-ascend v0.12.0rc1 `42ceaf08`, cause left open (#6266).
- `kv-transfer-thread-masked-error.md` — the D node's
  `Error in KVCacheTransferThread. error=unhashable type: 'list'.` (recorded #10962,
  v0.21.0rc1, A3 4-node 2P1D): the mask in `mooncake_connector.py:542-586`
  (`invalid_block_ids: set[int]` fed a grouped `BlockIds`, mark before the
  `logger.exception`), the same-commit sibling that flattens
  (`mooncake_layerwise_connector.py:1350`), the `finally` that lets a half-filled
  request decode anyway, and the unmask-then-classify ladder (MambaSpec
  block-count mismatch, transfer-engine pull, the *unmasked* pull-count assert).
  A 2-3 call answer; upstream has no such bookkeeping.
- `node-build-skew-attribute-error.md` — the pool thread's
  `Error in KVCacheTransferThread: 'ReqMeta' object has no attribute 'block_ids'`
  (recorded #10398, v0.20.2rc1 `367b8e62da79`): which of the three sites owns the
  basename and why the paste's `kv_transfer.py:84` versus the pin's `:91` is already a
  build signal, the pin's `ReqMeta` (`config_data.py:557`, `block_ids_by_group` `:563`,
  compat `block_ids` property `:620-626`, sibling at `:697`/`:731`) that makes the tag
  unable to raise it, the per-node commands that end the case in one reply, the mirror
  message (`no attribute 'block_ids_by_group'`) that gives the direction of the skew,
  the `run()` handler that skips `task_done`/`set_finished_request` against
  `pool_worker.py:599`'s `request_queue.join()` (a stall, not log spam), and why the
  group-0-only shim is not a fix. Reporter-confirmed: the nodes were not on one code
  version.
- `asymmetric-tp-pull-count.md` — P/D with different TP sizes on an MLA/DSA-sparse
  model: `chosen_rank_list(…) does not match num_group_pulls(…)`, the three
  derivations of the pull count and the one that omits the MLA/sparse rule, the
  ambiguous `num_kv_heads` of a `UniformTypeKVCacheSpecs` group, which groups make
  `_is_hma_required` true (and why turning Eagle3 off is not predicted to help),
  and why the assertion is load-bearing. Verified at vllm-ascend v0.22.1rc1
  `5f6faa0cb883` / vllm v0.22.1 `0decac0d96c4` (#11877).
- `kv-group-count-mismatch.md` — the **decode** worker dies on the first request with
  `IndexError: list index out of range` at
  `mooncake_connector.py:2740` `meta.remote_block_ids[kv_cache_group_id]`: the two
  lists that must agree (local `_build_kv_group2layeridx` vs the prefiller's
  `remote_block_ids`, built at `:1989-1993` and shipped in `kv_transfer_params`), why
  only local>remote raises, why a drafter on one side adds a group, the tutorial line
  that states the rule (Kimi-K3 §5.2 / FAQ `:886`), the DEBUG-only handshake report
  that hides it, the ranked causes (spec asymmetry, draft-config geometry, HMA parity,
  peer image skew) and the four-row ladder. Reporter-confirmed #16177 at v0.26.0rc1
  `f2f74a16`.
- `layerwise-split-metadata-crash.md` — the **producer's** `start_load_kv` dies in the
  layerwise connector and the paste stops at the call line: how `:1627` (full
  attention) differs from the MambaSpec sibling `:1623-1625`, the nine throw sites with
  their exception texts, why a symmetric 16/16 deployment makes the bare head-group
  asserts unlikely and the block-count `IndexError`/`KeyError` family the active one
  (`remote_tp_size` read from the reporter's own `decode.tp_size`), the
  `chunk_finish = True` mutation at `:1479-1480`, the DEBUG lines that print every
  input (`build_connector_meta` `:1066-1079`, `:1485`, `:1508-1514`, `utils.py:253-262`,
  `:305`), and the attn+mamba truncation/trim contract (`need_truncate` `:819`,
  `_trim_hybrid_remote_block_ids` `:881-899`) that the ablations isolate. Verified at
  v0.23.0 `5cb98caaadef`; #15394's cause left open.
- `llmdatadist-done-signal-and-ttft.md` — `LLMDataDistCMgrConnector` serves but TTFT is
  slow and/or the prefiller logs `fail to receive KV Cache transfer done signal in 120s`
  (`VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`, default 120): where the warning is emitted, the
  three port formulas that decide which P ranks can ever see a done signal and which
  duplicates `_increment_task_count` drops, the held-block (`delay_free_blocks`) cost and
  why `=600` makes it worse when queueing is the mechanism, and the D-side checks
  (single-thread `pull_blocks`, `query_register_mem_status` link loop) with the cause
  left open. Verified at v0.9.1 `0740d10` (#3062).
