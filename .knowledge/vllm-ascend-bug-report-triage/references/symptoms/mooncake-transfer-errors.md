# PD KV transfer that *fails* instead of hanging: Mooncake ADXL status codes

## Scope

A deployment with a `--kv-transfer-config` Mooncake connector (`MooncakeConnector`,
`MooncakeConnectorV1`, `MooncakeLayerwiseConnector`) where the first request produces an explicit error
and the deepest frame is the connector, not a scheduler stall. Two halves, two owners:

- **connect-time** — `Failed to connect to target: <engine-name>, status: <code>` and, on older
  builds, `Failed to connect to segment: <name>`, then the Python line
  `Mooncake transfer failed for request <id>, ret: -1`. The local ADXL engine refused the peer's
  *engine name*; nothing was transferred.
- **store-time** — a `MultiConnector` whose `AscendStoreConnector` child cannot PUT (`503900`, then
  `103902`, `Failed to put key … res:…`) while the requests still answer: same ADXL table, different child
  and a different lever (§"The store child of a `MultiConnector`").
- **transfer-time** — `Sync batch data transfer timeout after <ns>` in `transfer_engine_py.cpp` beside
  the scheduler repeating `… is still in WAITING_FOR_REMOTE_KVS state` and
  `Number of completed KV cache send requests: 0, receive requests: 0`. The handshake succeeded and the
  data movement never completes.
- **accuracy-time (no error at all)** — two distinct owners, so read the *pattern* before the code:
  wrong block **range** (garbage at block-boundary prompt lengths; §"The silent sibling", #9430 →
  #8908/#9810, first tag `v0.20.2rc1`) versus wrong **reformat timing** (good/bad answers alternating
  request by request on `tp_size > 1`; §"The TP-inequality shape", #12339 → #12371/#12359/#12382, the
  branch fix first in `v0.23.0`).

Also answers the **build-flag half of a `[Usage]` question** — "does release R support the Mooncake store
connector, and must Mooncake be built with `-DUSE_ASCEND_DIRECT=ON`?" — because the answer is a gate in the
fork's store backend, not a configuration (`§"The store backend is ADXL-only"`).

Not this page: the silent mid-run freeze with HCCL `STUCK` (`pd-collective-hang.md`), a store/pool
connector that never PUTs or whose lookup line repeats (`kv-pool-store-hang.md`), a hybrid model whose
recurrent state no connector carries (`hybrid-gdn-pd.md`), and the proxy↔connector pairing defect
(`pd-proxy-examples.md`).

## Read the status code before any code

Verified at Mooncake `main` (`522b716e`),
`mooncake-transfer-engine/include/transport/ascend_transport/ascend_direct_transport/adxl_compat.h:41-49`
(`namespace adxl`, `using Status = uint32_t`):

| code | constant | code | constant |
| --- | --- | --- | --- |
| `0` | `SUCCESS` | `103904` | `NOTIFY_FAILED` |
| `103900` | `PARAM_INVALID` | `103905` | `UNSUPPORTED` |
| `103901` | `TIMEOUT` | `203900` | `RESOURCE_EXHAUSTED` |
| `103902` | `NOT_CONNECTED` | `503900` | `FAILED` |
| `103903` | `ALREADY_CONNECTED` | | |

These are **Mooncake's ADXL wrapper constants, not CANN/HCCL codes**: the CANN family in the rest of
this skill is `1xxxxx`/`507xxx` plus the `EZ`/`EI`/`EE` prefixes (`cann/runtime` `rt_error_codes.h`,
`hcom … error_code.cc`). So a report quoting `status: 103900` is answered with the Mooncake peer-name
contract below — do not attribute it to a CANN version, a kernel fault or a device fault, and do not
grep `vllm_ascend/` for the code.

Message sites at Mooncake `main` (`mooncake-transfer-engine/src/transport/ascend_transport/
ascend_direct_transport/transfer_executor_base.cpp`): `:377` `Connect timeout to:` for the
`adxl::TIMEOUT` branch, `:382` `Failed to connect to target: … , status: … , errmsg:
<aclGetRecentErrMsg()>`, `:388` the success line `Connected to segment:`. The reporters' builds log the
same messages from `ascend_direct_transport.cpp` (`:820`, `:581` in #5635) because the file was split
later; `Failed to connect to segment` was removed by `b375c663` ("Adapt to adxl connection auto release
feature", #1072). Both are therefore a *fingerprint of an older Mooncake*, not a different fault.

## Why a name can be rejected: the two hostname forms

`Connect()` is given the **target engine name the peer advertised** — the transfer engine's segment
name, not the model server's `--port`. At `v0.11.0` vllm-ascend names its engine one of two ways
(`vllm_ascend/distributed/mooncake_connector.py:905-911` at `v0.11.0`):

```python
if vllm_config.kv_transfer_config.get_from_extra_config('use_ascend_direct', True):
    hostname = self.side_channel_host                      # "<ip>"
else:
    hostname = f"{self.side_channel_host}:0:npu_{self.device_id}"
self.engine = get_global_te(hostname, device_name=None)
```

So both roles must register the **same form**, or the peer's advertised name is one the local ADXL engine
cannot resolve — and a P node and a D node on different builds (or one of each connector class) is exactly
how one side advertises `<ip>` while the other connects to `<ip>:<rpc_port>`. The defaults disagree across
classes at that tag (table below).

What ADXL itself does with a name it cannot resolve is inside the closed `adxl`/`hixl` library and no
clone shows it — keep that step `hypothesis` and let the P/D logs decide.

### The two readers, side by side (`verified` at `v0.11.0`)

| reader | line at `v0.11.0` | default | engine name it builds |
| --- | --- | --- | --- |
| store config (`MOONCAKE_CONFIG_PATH` json) | `distributed/mooncake/config_data.py:441` `config.get("use_ascend_direct", False)` | **False** | `distributed/mooncake/mooncake_store.py:42-52`: for `protocol == "ascend"` and the key false → `get_ip():{BASE_PORT + device_id}:npu_{device_id}`, else a global TE at plain `get_ip()` and `local_seg = ip:rpc_port` |
| connector (`--kv-transfer-config` extra config) | `distributed/mooncake_connector.py:906` `get_from_extra_config('use_ascend_direct', True)` | **True** | `:906-911`: `hostname = self.side_channel_host` (plain IP) when true, else `f"{side_channel_host}:0:npu_{device_id}"` |

A deployment that sets `protocol: "ascend"` in `mooncake.json` but no `use_ascend_direct` **anywhere** runs
the connector on the plain-IP form while the store takes the legacy `ip:port:npu_N` form — one rank
advertises a name the peer's ADXL engine does not know, and the connect returns `PARAM_INVALID`; setting the
key on one side only reproduces the same split.

- The `ASCEND_BASE_PORT` the messages expose (default `20000`, mooncake
  `docs/source/design/transfer-engine/transport/ascend_direct_transport.md:60`) is the ADXL engine port
  base; that is why the maintainer's advice pairs the key with a question about `/etc/hccn.conf` — a
  missing/mismatched HCCN configuration or a filtered port produces the identical `Failed to connect`
  text, and the two causes are only separable by running the `103900` again after the config change.
- Evidence to ask for either way: the **two** reconstructed engine names (one per node) plus each node's
  `mooncake-transfer-engine` version. The connector prints only `Initializing Mooncake work <engine_id>`
  (`logger.info`, `:910` at `v0.11.0`), never the hostname, so the name must be reconstructed from the
  config key, the node's own `HCCL_IF_IP`/side-channel host and its device id — or read out of Mooncake's
  own engine log.

## Worked report #5635 (`v0.11.0` image, 1P1D, single machine, A2 910B)

Config: `MooncakeConnector`, `kv_role: kv_consumer` on the D node, `kv_port 30001`,
`kv_connector_module_path: vllm_ascend.distributed.mooncake_connector`, extra config
`{"prefill": {"dp_size": 1, "tp_size": 1}, "decode": {"dp_size": 1}}` (no `tp_size` on the decode side) —
no `use_ascend_direct` key. Environment: image `v0.11.0-openeuler`, vLLM/vllm-ascend `0.11.0`, CANN
`8.3.RC2`, torch_npu `2.7.1`, A2 910B single machine with one card per role, `ASCEND_RT_VISIBLE_DEVICES=0`,
`HCCL_IF_IP`/`VLLM_HOST_IP` set, `engine_id 1`; the served model is Llama2-Chinese-7b — irrelevant to the
diagnosis, the handshake fails before any attention runs. Symptom on the first request:

```text
E… ascend_direct_transport.cpp:820] Failed to connect to target: [NETWORK_ADDRESS_1]:25482, status: 103900
E… ascend_direct_transport.cpp:581] Failed to connect to segment: [NETWORK_ADDRESS_1]:16584
(EngineCore_DP0 …) ERROR mooncake_connector.py:419] Mooncake transfer failed for request cmpl-…
(EngineCore_DP0 …) ERROR mooncake_connector.py:330] Failed to transfer KV cache for request cmpl-…: ret: -1
```

- **The Python error is the echo.** At `v0.11.0` `mooncake_connector.py` raises
  `RuntimeError(f"Mooncake transfer failed, ret: {ret}")` when `self.engine.batch_transfer_sync_read(
  session_id, src_list, dst_list, length_list)` returns `< 0` (`:417-419`; the reporter's `:419` matches
  that tag's file). The cause is the C++ line above it, so quote the status code, not the raise.
- **The thread's lever** (`LCAIZJ`, 2026-01-07): 「Add `"use_ascend_direct": true` in
  kv_connector_extra_config. Also, do you have the file `/etc/hccn.conf`?」 — advice, not a confirmed
  fix, and at that tag `use_ascend_direct` already defaults to `True` **for this connector**, so the
  explicit key changes nothing by itself. Do not look for the name in vLLM's log: the connector only
  prints `Initializing Mooncake work <engine_id>` (`logger.info`, `:910` at `v0.11.0`), never the hostname — the
  name has to be **reconstructed per node** from that config key, the value of `use_ascend_direct`, and
  the node's own `HCCL_IF_IP`/side-channel host, or read out of Mooncake's own engine log.
- **A second reporter in the same thread** (`feifei0323`, 2026-01-08) shows the transfer-time half
  instead: `Sync batch data transfer timeout after 30018876016ns`, `WAITING_FOR_REMOTE_KVS` repeated,
  `completed KV cache send requests: 0` — and their attempted mitigation was timeout inflation
  (`ASCEND_TRANSFER_TIMEOUT=600000000000`, `MOONCAKE_TRANSFER_TIMEOUT=120`, `HCCL_EXEC_TIMEOUT=3000`,
  `HCCL_CONNECT_TIMEOUT=1800`), which is a diagnostic, not a fix: a raised timeout turns a visible
  failure into a longer stall.
- Both halves are unverified beyond the thread: no maintainer stated a mechanism, and no run printed
  the two engine names from the same failing pair.

## The store backend is ADXL-only: `-DUSE_ASCEND_DIRECT=ON` is a code requirement

`AscendStoreConnector` (`kv_role: kv_both`, `backend: mooncake`) is a different reader from the P2P connector
this page is about: it constructs `MooncakeBackend`, which accepts one `mooncake.json` `protocol` — `"ascend"` —
and drives the KV bytes through Mooncake's ADXL/HIXL engine, so a Mooncake built without
`-DUSE_ASCEND_DIRECT=ON` has no engine for it to use. That is a startup refusal or a `[Usage]` question, not a
transfer error: the per-release code states, the line numbers, the docs/cmake lines and the answer shape are
`kv-pool-store-hang.md` §D. What belongs *here* is only the boundary — a **successful** handshake followed by
stalled or failing data movement is this page's §1/§2, and the thread's "without the flag Mooncake falls back
to host RDMA/TCP" sentence describes Mooncake's own transports, not a fallback path in the pool backend.

## The store child of a `MultiConnector` fails to PUT: `503900` / `103902`, `Failed to put key … res:[-800]` (#7401)

A third shape, and the one most easily mis-merged with the connect mismatch above: `MultiConnector` with a
`MooncakeConnectorV1` child **and** an `AscendStoreConnector` child (`backend: mooncake`) on a 1P1D
GLM-5-w4a8 deployment, `v0.16.0rc1`. Requests keep answering `200 OK`; the KV *store* half fails per key.

```text
Failed to connect to target: <P node's own ip>:20064, status: 503900        # ascend_direct_transport.cpp:1111
Transfer failed for key GLM-5-w4a8-mtp-QuaRot@…: TRANSFER_FAIL (Transfer 0 failed)
Failed to put key ['GLM-5-…@…'],res:[-800]                                  # mooncake_backend.py:85
Connect timeout to: <peer ip>:20028 … reduce the ASCEND_CONNECT_TIMEOUT environment variable  # :1107
Connected to segment: <peer ip>:20028, cost:10010262 us                      # :1119 — connected AT the timeout
Transfer slice failed with status: 103902; Failed to disconnect … 103902     # :836 / :1138 (NOT_CONNECTED)
Successfully revoked failed put for key GLM-5-…@…                          # client_service.cpp:1227
```

- `503900` is the same ADXL table's **`FAILED`**, not `PARAM_INVALID`: the target in that line is the P
  node's *own* store segment, so this is the pool backend, not the P2P peer-name question above. Read the
  Python line to confirm the child: `Failed to put key …,res:<n>` is
  `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py:85` at
  `v0.16.0rc1` (its `memcache_backend.py:85` twin prints `error:` instead of `res:`).
- `cost:10010262 us` beside a `Connect timeout` line is the discriminator against the `103900` row: the
  connect *did* complete, 10 s late (that is `ASCEND_CONNECT_TIMEOUT`), and the slice then failed
  `103902`. So the engine name resolved and the connect-stage mismatch is not the defect; what follows the
  10 s is `hypothesis` (a store segment that is not answering in time), while the retry-then-revoke tail is
  `verified` in the log.
- Consequence, and why it is not the crash in the same log: the request path is intact — the block is
  dropped and the engine keeps serving, with external prefix-cache reuse at 0 % and the puts re-attempted.
  A device-side death in the same report (`execute_dummy_batch`, `507011`, MTE DDR out of range) is a
  **second** symptom with its own owner; do not let one mechanism span both (`index.md`'s two-row rule).
- Lever while it is open: drop the `AscendStoreConnector` child from `connectors` (keep
  `MooncakeConnectorV1`) so PD transfer works and only the pool offload is lost, or raise
  `ASCEND_CONNECT_TIMEOUT` and re-read `cost:`; composition legality and the ordering rules for two
  children are `multi-connector-composition.md`, and the store backend's `-DUSE_ASCEND_DIRECT=ON` gate is
  `kv-pool-store-hang.md` §D. **Status: open, no maintainer diagnosis of this half.**

## The silent sibling: PD + MTP answers fluent garbage, no error at all (#9430)

Scope note: this page also owns the Mooncake-connector defect whose symptom is **wrong tokens**, not a
failed transfer. Signature: a P/D deployment on an MTP `--speculative-config` answers normally-shaped text
with stray/misplaced tokens ("经常出现无意义字符"), nothing fails and plog is clean — the reporter of
**#9430** (DeepSeek-V4-Pro-w4a8, `deepseekv4` named image,
2P2D on 4 nodes, `MooncakeHybridConnector`, `deepseek_mtp` 1, `--block-size 128`,
`recompute_scheduler_enable`, `FULL_DECODE_ONLY`) closed their own half with 「v0.20的版本问题解决了」.

Mechanism (`verified` code, `reported` cause — the two fixes are titled as accuracy bugs and the reporter's
own 「v0.20」 note agrees): the P-side `request_finished` ships **every** computed block id, while the D side
allocates only the blocks the prompt needs. With MTP the producer also allocates the speculative block, so
when `len(prompt_token_ids)` is a multiple of the block size the two lists differ in length; the receiver
zips `remote_block_ids` against its own local ids (`group_concurrent_contiguous` / `zip(...)` in the
receiving thread), so one extra trailing id misaligns the mapping and the decoder reads another block's
KV. Intermittent by construction — only prompt lengths that land on a block boundary.

- Plain connector: at `v0.18.0`, `mooncake_connector.py:1065` sends
  `remote_block_ids=computed_block_ids` although `num_prompt_blocks` is computed at `:1060` and shipped in
  the same dict at `:1075`. Fix `015af05d6` (PR #8908, "Fix Mooncake Connector MTP accuracy bug") cuts it —
  `remote_block_ids=computed_block_ids[:num_prompt_blocks]`, `:1121` at `v0.20.2rc1`. **First tag
  `v0.20.2rc1`.**
- Hybrid connector (the one #9430 runs): the same cut arrives as `_compute_transfer_block_ids` (per KV-cache
  group, block size **and** `compress_ratio`, adding `and not local_block_ids[i]` to the receiving thread's
  skip) in `31860b0d7`, PR #9810 (`[0.20.2][BugFix][P/D] Add compress ratio and block_ids cutting for
  mooncake hybrid connector`). **First tag `v0.20.2rc1`** too, and its own testing note is "Tested by P/D
  accuracy test".
- `mooncake_hybrid_connector.py` **does not exist at `v0.18.0`**: the file arrives with `7bce23cc0`
  (`[Feature] Support DeepseekV4`, #9270), first tag `v0.20.2rc1`, so a report naming
  `MooncakeHybridConnector` on an image *labelled* `0.18.0` is running a **model-specific named image**
  built off that feature branch. Do not localise its `file:line` against the `v0.18.0` tree and do not call
  the connector "unknown"; say the label does not carry the tree.

Levers, cheapest first: (1) move to the `v0.20.2rc1`+ line — this is the reporter's own confirmed fix and
carries both cuts; (2) interim, drop MTP on the P/D pair (`--speculative-config` removed on both roles):
removes the extra block, costs decode throughput; (3) nothing at `v0.18.0` fixes it — no env, no
`--block-size` change (a different block size only changes which prompts straddle the boundary).
`hypothesis` to state with its falsifier: exact-prompt-length sensitivity ("garbage only when
`len(prompt)*1 % 128 == 0`") — falsify with one fixed prompt repeated; if garbage appears at every length,
it is not this class (`flashcomm1-and-moe-reduce.md` §B is the EP/flashcomm twin, `benchmark-accuracy.md`
the measurement one).

## The TP-inequality shape: the KV **reformat** ran before every pull arrived (#12339)

A third silent shape, distinct from the two above: the deployment answers **correctly on one request and
garbage on the next**, alternating, with no error, no stall and a clean plog. Report **#12339**
(Ascend950 B130, `vLLM v0.23.0`, `vLLM Ascend releases/v0.23.0 6cee2e16308ae34aedb7c3b7ed2df0d42284713e`,
torch `2.10.0`, CANN `9.1.0.B103`; Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT, 1P1D, `MooncakeConnectorV1` on
both roles with `prefill/decode dp_size 1, tp_size 8`, `--no-disable-hybrid-kv-cache-manager`,
`--enable-expert-parallel`, `recompute_scheduler_enable`, **MTP off**). Its two samples are the
fingerprint: one request returns `completion_tokens: 1` with empty content, `finish_reason: stop`,
`stop_reason: 248044`; the next returns fluent text with stray `</think>` fragments. Prompt lengths are
identical (45 tokens), so this is not the block-boundary class above.

Thread state: closed, **no lever in it** — `weiguihua2` 2026-07-21 「@LoganJane plz take a look」,
`q664171689` 2026-08-03 「已解决」. So the answer has to come from the reporter's own build commit.

Mechanism (`verified` code, attribution `inference` — the PR body states it and the timing matches the
thread's own resolution date). The fix `86217325b`
(**#12371**, `[BugFix][v0.23.0]`, 2026-07-20) is titled "Fix precision issues caused by incorrect
reordering timing leading to TP inequality" and its body is explicit: *"Defer KV cache reformatting in
MooncakeConnector until all pull tasks for a request are completed, preventing premature reordering
that leads to TP inequality."*

- Before it (the reporter's build), `KVCacheRecvingThread._transfer_kv_cache_all_groups` ends by
  building `ready_attention_group_reformat_block_ids` from the `is_group_transfer_end` flags of **this**
  pull message and applying the ND→NZ / `cat` reformat inline
  (`git show 6cee2e163:vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py | nl -ba |
  sed -n '915,965p'`: the list is built, `if not … : return`, then `reformat_kv_cache_hybrid_linear_torch`
  / `reformat_kv_cache`). Request-level completion is a **different** test —
  `_mark_request_task_done(request_id, all_task_done)` in the handler's `finally` (`:701`) — so with
  `tp_size 8` the reformat could run while another TP-offset pull for the same request was still
  writing into the same blocks. Transposing / concatenating a region that is still being written leaves
  the per-shard rows in different orders, which is exactly "TP inequality": the model still decodes,
  fluently, from inconsistent K/V.
- After it, the thread stashes the ready list per `(request_id, shard_idx)` in `self.pending_reformat`
  and applies it only after all pulls finish (`_stash_pending_reformat` / `_reformat_pending_kv_caches`,
  called from the `all_tasks_done` branch). At `v0.23.0`+ the same members exist; a build **before**
  `86217325b` has none of them.

Fix map — one change, three shas, and the tag that matters is the branch one:

| sha | PR | where | first tag that carries it |
| --- | --- | --- | --- |
| `86217325b` | #12371 (`[BugFix][v0.23.0]`, "Backport of #12359") | `releases/v0.23.0` | `v0.23.0`, `v0.23.0.post1` |
| `b6826ec33` | #12382 (cherry-pick to `releases/v0.24.0rc`) | `releases/v0.24.0rc` | `v0.24.0rc1` |
| `1db626510` | #12359 (mainline original, merged 2026-08-16) | `main` | `v0.27.1rc1` |

**Deliverable for this shape:** the reporter's build id (they paste it next to
`vLLM Ascend releases/vX.Y.Z`) is the needle — `git merge-base --is-ancestor 86217325b <their sha>`
answered *not an ancestor* while the sha is an ancestor of tag `v0.23.0`, i.e. their build predates the
branch fix by eight days and any `releases/v0.23.0` rebuild after 2026-07-20 carries it. Never quote
only the mainline first tag (`v0.27.1rc1`): for a 0.23 reporter that reads as "wait for 0.27".
Levers: rebuild from `releases/v0.23.0` at/after `86217325b` (or `>= v0.23.0`); interim, the failure
needs `tp_size > 1` reformat with `num_group_pulls > 1`, so a TP-per-pull-free shape (one pull per
group) loses throughput but removes the race. `hypothesis` to hand the reporter with its falsifier:
alternation follows transfer timing, not prompt content — the same single prompt repeated 10× at
`temperature=0` should still alternate; if it is stable per prompt, read the block-boundary class above
instead.

## Levers, ordered by what they cost

1. **Make both roles name their engine the same way.** Set `"use_ascend_direct": true` (or `false` on
   *both*) and restart both sides; the connector's `hostname` branch above is the only thing the key
   changes. Falsifier: both names reconstructed as above are equal, and the connect still returns
   `103900`.
2. **Same Mooncake build (and protocol) on both nodes.** The image tag hides the wheel/`.so`: compare
   `mooncake-transfer-engine` version and `MOONCAKE_CONFIG_PATH`'s `protocol`/`device_name` on P and D —
   a `103900` that survives (1) is a version/name-form mismatch, and the `Failed to connect to segment`
   line dates the peer's build.
3. **Check the control plane files the advice names** — `/etc/hccn.conf`, `HCCL_IF_IP`,
   `ASCEND_RT_VISIBLE_DEVICES` on both nodes — because the advertised name embeds the local IP/device
   id; a container on `172.17.x` with a bridged NIC advertises an address the peer cannot reach.
4. For the transfer-time half, do **not** ship timeout inflation as the answer: bisect instead —
   `"use_ascend_direct"` on both roles, then `MOONCAKE_TRANSFER_TIMEOUT`, then the connector class
   (`MooncakeConnectorV1` vs layerwise) — and report the first knob that changes the log line.

## Path drift that costs a batch

`vllm_ascend/distributed/kv_transfer/` **does not exist at `v0.11.0`**; the connector is
`distributed/mooncake_connector.py` (plus `distributed/mooncake_layerwise_connector.py`) beside a
`distributed/mooncake/` package (`config_data.py`, `kv_transfer.py`, `mooncake_engine.py`,
`mooncake_store.py`), and the PD rework moves everything under `distributed/kv_transfer/kv_p2p/…` by
`v0.26.0rc1`. Read `git ls-tree -r --name-only <ref>:vllm_ascend/distributed | grep mooncake` before
quoting a `file:line`, and treat the reporter's own line numbers as the cheapest fingerprint: they
matched `v0.11.0` here, which is what makes the tag trustworthy.

## Answer shape

symptom → stage (connect vs data) → the two config readers and their opposite defaults → lever
(`"use_ascend_direct": true` in `kv_connector_extra_config`, then re-check `/etc/hccn.conf` and the ADXL
port range) → what it trades (nothing on the wire; it selects the engine-name form) → status (advice on the
thread, no code fix) → falsifier: the same `Failed to connect … 103900` after both sides use one form means
name construction is not the cause and the remaining suspects are the HCCN/port path and the ADXL build
(`USE_ASCEND_DIRECT=ON`), not vllm-ascend.

## Status and limits

#5635 was answered with the advice above and never confirmed; the mechanism (name-form mismatch inside
ADXL) is `hypothesis` with the two-engine-name falsifier. Nothing here was reproduced on hardware, and
the closed ADXL library is not in any clone — so the deliverable is the status-code table, the selector
line, and the two log lines to request.

The `mooncake_layerwise_connector.py` twin carried the same
`get_from_extra_config('use_ascend_direct', True)` at `:668` at `v0.11.0`, so re-grep both readers
(`git grep -n "use_ascend_direct" <ref> -- vllm_ascend`) before quoting a line, and re-check the Mooncake
`Status` table in the clone rather than reusing `103900` for another code.
