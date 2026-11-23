# AscendStore put failed: `Failed to invoke HcclBatchPut, hccl_result = 4` / `error_codes=[-1]`

Symptom class: the deployment is up, the OpenAI request **succeeds**, and the KV
pool never stores anything. Prefill logs

```text
E19999 ... Failed to invoke HcclBatchPut, hccl_result = 4
... Failed to put 8 keys out of 8. error_codes=[-1]. Check memory and store capacity.
```

and `mooncake_master` shows `PutStart` > 0, `PutRevoke` > 0, `PutEnd` 0, with
`Mem Storage: 0 B / 32.00 GB`. Worked case **#11454**: one node, 4×910B, P on
NPU 0,1 and D on 2,3 (TP=2 each), Qwen2.5-Coder-14B, `AscendStoreConnector`
`backend: mooncake`, `mooncake-transfer-engine-npu==0.3.11.post1`, 1K prompt, no
embedded SSD offload. Pins: vllm-ascend `v0.22.1rc1` = `5f6faa0`, vLLM `v0.24.0`,
Mooncake `v0.3.11.post1` = `e9c6107`, `cann/hccl` `ce20062d`, `cann/hcomm`
`818068c8`.

Different from `../pd-disaggregation/kv-pool-store-segments.md`: that page is the
`503900` / `HcclCommPrepare ret: 0x13` family, where the channel never got
prepared because the store segment was too large. Here the segment mounted, the
master handed out replicas, and the **write** failed. Check the size first only if
`error_codes` shows `-200` (see §1).

## 1. The two numbers, and what they exclude

**HCCL `hcclResult_t`** — `cann/hccl/docs/api/context/HcclResult.md:10-35`:
`0 HCCL_SUCCESS`, `1 HCCL_E_PARA`, `2 HCCL_E_PTR`, `3 HCCL_E_MEMORY`,
**`4 HCCL_E_INTERNAL`**, `5 NOT_SUPPORT`, `6 NOT_FOUND`, `7 UNAVAIL`,
`8 SYSCALL`, `9 TIMEOUT`, … `19 NETWORK`, `20 AGAIN`, `21 REMOTE`,
`22 SUSPENDING`. So `4` is the generic rejection bucket: **not** a bad
parameter, **not** an allocation failure, **not** a timeout. Answering "bad
parameters" or "out of memory" from `4` is unsupported by the number.

**Store `ErrorCode`** — Mooncake `mooncake-store/include/types.h:254-310`:
`-1 INTERNAL_ERROR` (generic), `-10 BUFFER_OVERFLOW`, `-100..-103` segment,
**`-200 NO_AVAILABLE_HANDLE` = "memory allocation failed due to insufficient
space"**, `-400 INVALID_KEY`, `-500 WRITE_FAIL`, `-800 TRANSFER_FAIL`,
`-900 RPC_FAIL`. `error_codes=[-1]` is therefore the store's *unclassified*
bucket, and an out-of-space put would have said `-200`. With master at
`0 B / 32 GB`, capacity is excluded by the code **and** by the counters.

**The advice in the log is canned text.** `mooncake_backend.py:180-200`
(`put()` → `batch_put_from_multi_buffers` at `:188`) prints
`"Check memory and store capacity"` for *any* negative result (rule 11 of
SKILL.md); it never checked capacity.

**The master's `SSD Storage: X / Y` line is not a client SSD-offload switch.**
`Y` is master-side file/DFS accounting, not the `enable_ssd_offload` /
`ssd_offload_path` keys of the client's `mooncake.json`: master init runs
`if (!root_fs_dir_.empty()) { use_disk_replica_ = true;
inc_total_file_capacity(global_file_segment_size_); }`
(`mooncake-store/src/master_service.cpp:300-304`) and the metric prints
`file_allocated / file_capacity` as `SSD Storage`
(`master_metric_manager.cpp:2016-2017`). At #11454 the master was launched with
`--root_fs_dir /workspace/mooncake_file_root --global_file_segment_size 858993459200`,
i.e. exactly the `800.00 GB` denominator the paste prints, while no client enabled
offload; the docs say the same line reads `0 B / 0 B` without `--root_fs_dir`
(`docs/source/design/mooncake-store.md:804-811`). So a non-zero SSD denominator is not
evidence of offload and does not move this report into the offload or segment-size
families.

**That same master flag adds a replica the store client never writes.** With
`use_disk_replica_` true, every `PutStart` appends a DFS replica
(`master_service.cpp:1238-1242`), but a client writes it only when it has a local-disk
backend — `if (storage_backend_)` around `PutToLocalFile`
(`client_service.cpp:1249-1263`, `:1603-1613`; the backend is created only by
`PrepareStorageBackend`, `:2695-2701`). So the DFS replica stays `PROCESSING`, and even a
successful `PutEnd(ReplicaType::MEMORY)` leaves the key in the processing set because
`AllReplicas(&Replica::fn_is_completed)` fails (`master_service.cpp:1353-1412`). This is
**not** the owner of `hccl_result = 4`: the DFS copy is a host
`CopyDeviceToHost`/`StoreObject` (`client_service.cpp:2711-2750`), never an HCCL
transfer, and a failed disk copy revokes only `ReplicaType::DISK`. It is one launch
difference the reporter's own isolation list never removed — and it is cleared by the
control in §3.

One call turns a paste into these statements:
`bash <skill_root>/scripts/store-put-code-check.sh <paste|->`.

## 2. Who owns the failure, and who does not

| Fact (pinned) | Consequence |
|---|---|
| `mooncake-store/src/client_service.cpp:1692-1761` `FinalizeBatchPut`: `BatchPutEnd` for operations whose transfers completed, `BatchPutRevoke(failed_keys)` for "operation failed but has allocated replicas" (`:1719-1723`, `:1760-1761`) | `PutStart` > 0 says the client reached the master and got replicas: the object/key path worked. `PutRevoke` > 0 with `PutEnd` 0 says the **data write** failed and the client cleaned up. Not a registry, key-format or connector-config fault |
| `client_service.cpp:1683-1687` logs `Transfer failed for key <k>: <toString(code)> (Transfer N failed)` before the revoke | This is the authoritative line to request: it carries the classified code the reporter's `-1` threw away |
| `vllm_ascend/.../ascend_store/kv_transfer.py:391` (layerwise `:648`) calls `self.m_store.put(...)` on the per-rank save thread, and `mooncake_backend.py:190-200` only counts and logs failures | vllm-ascend owns *when* and *what* is put, not the transfer; the failure never reaches the request path, which is why the bench reports `Successful requests: 1` |
| `kvcache-ai/Mooncake` `multi_transport.cpp:331-345`: `protocol == "ascend"` maps to `AscendDirectTransport` (`USE_ASCEND_DIRECT`), `HcclTransport` (`USE_ASCEND`) or `HeterogeneousRdmaTransport` (`USE_ASCEND_HETEROGENEOUS`) depending on **build flags** | `protocol: "ascend"` in `mooncake.json` does not name an engine. The log does: `HcclBatchPut` ⇒ a one-sided HCCL transport is compiled in; `AscendDirectTransport register mem addr:` lines (`transfer_executor_base.cpp:452`, `ascend_direct_transport.cpp:83`) ⇒ ADXL |
| `cann/hcomm/pkg_inc/hccl/hccl_one_sided_services.h:63,85,95,116`: `HcclRegisterMem(comm, remoteRank, …)` → `HcclExchangeMemDesc` → `HcclEnableMemAccess` → `HcclBatchPut(comm, remoteRank, desc, descNum, stream)` | `HcclBatchPut` writes to a peer's registered VA. Any unmet precondition for that `{comm, remoteRank}` returns the generic `HCCL_E_INTERNAL`; the API does not distinguish "peer buffer not enabled" from "peer not in the comm" |

So the owner of the failure is **the store's transfer path inside Mooncake's
Ascend build**, between the client's `PutStart` and its `BatchPutEnd`. Name the
Mooncake store client + transport, not `AscendStoreConnector`.

## 3. Procedure (3 checks, one control, then the answer)

1. **Decode the paste** with `scripts/store-put-code-check.sh` (§1, one call).
2. **Rule out the segment-size family** with the same numbers the reporter
   already gave: `mooncake.json` here is the pinned doc example
   (`kv_pool.md:128-136`: `metadata_server` `P2PHANDSHAKE`, `protocol` `ascend`,
   `device_name` `""`) with `global_segment_size: "8GB"`, inside the
   reporter-confirmed `< 20 GB` band of
   `../pd-disaggregation/kv-pool-store-segments.md`, and no `-200` code. Say so
   explicitly — it is the one lead this symptom invites and it is wrong.
3. **The control the report's own isolation list is missing** — relaunch `mooncake_master`
   without `--root_fs_dir` and without `--global_file_segment_size`, nothing else changed,
   and repeat the same request. That removes the extra DFS replica and the `SSD Storage`
   denominator together (§1), so it is the cheapest way to show the put path on a master
   with no file tier. Rank it **below** the transport causes: it cannot mint an HCCL code
   on its own, and it is listed because it is one flag the reporter never varied.
4. **Name the two missing artefacts** and stop: the store client's
   `Transfer failed for key …` line (or the full prefill worker log above the
   first error) and the transport marker lines in it (`install <X>Transport
   for: …`, `register mem addr: …`). With those two, the next step is a plog on
   the prefill host in the failure minute.

Answer shape for this report: `hccl_result = 4` = `HCCL_E_INTERNAL` (generic, not
a parameter or memory code), `error_codes=[-1]` = the store's unclassified
`INTERNAL_ERROR` (a space failure is `-200`), the master counters prove the
control plane succeeded and the data write failed, the canned capacity sentence
is not a finding, and the request's success is expected because vllm-ascend only
counts put failures.

## 4. What not to chase

- **Capacity / store size**: contradicted by `-1` and by `Mem Storage 0 B / 32 GB`.
- **Key count arithmetic**: `8 keys` in the worker against `16` on the master is
  TP=2 × 8, i.e. consistent, not a duplication bug.
- **SSD offload**: the reporter already reproduced with it off — and the master's
  `SSD Storage: X / Y` line is the master's own DFS layer (§1), not that flag, so its
  presence is not a counter-example.
- **Prompt length**: 1K fails, so the 8K+ long-context framing in the report is
  not the mechanism.
- **`AscendStoreConnector` registration / `lookup_rpc_port` / `MultiConnector`
  composition**: `PutStart` proves the client is a live store client with
  replicas.
- **Grepping the pinned trees for `HcclBatchPut`**: absent from every export in
  this snapshot (Mooncake `v0.3.11.post1`, vllm-ascend, `cann/*`; the only hits
  are the API declaration in `cann/hcomm` and LMCache-Ascend's
  `csrc/hcomm_onesided/bindings.cpp:285`). One scoped grep, then a gap
  statement.

## 5. Evidence limits

- **The emitting code is not in the snapshot.** The `Failed to invoke
  HcclBatchPut, hccl_result = …` string lives in the
  `mooncake-transfer-engine-npu` wheel (an Ascend-side transfer-engine build),
  not in `kvcache-ai/Mooncake` `v0.3.11.post1`, whose `hccl_transport.cpp` does
  its copies through `transportMemTask`/`transportMemAccept`. Do not state which
  precondition failed; say the wheel decides it and ask for its log line.
- **Rank/peer mapping is a hypothesis here.** In the *exported* `HcclTransport`
  a peer is identified by `rankId = devicePhyId` plus host/device address from
  the segment descriptor (`hccl_transport.cpp:106-116`, `:364-371`, `:593-599`),
  so two processes on one host each mounting a segment can collide. That is a
  plausible mechanism for this deployment (P on phy 0,1; D on phy 2,3), not a
  verified one. One rerun discriminates it from a registration/descriptor fault: the
  same 1K request with **TP=1 per role** (`ASCEND_RT_VISIBLE_DEVICES=0` / `=2`, ports
  unchanged). Put succeeds ⇒ peer identity/rank mapping; still `hccl_result = 4` ⇒
  registration/`EnableMemAccess` for that `{comm, remoteRank}`, and the wheel's
  `Transfer failed for key` line is the next artefact.
- **The store's version is the reporter's wheel, not the pin.** `SOURCES.json`
  carries the GitHub tree; the installed package is
  `mooncake-transfer-engine-npu==0.3.11.post1`, which is a different artifact
  with the same number. Ask `pip show mooncake-transfer-engine-npu` before
  quoting any of its behaviour.
- Recheck conditions: if a later Mooncake source carries the `HcclBatchPut`
  call (grep the wheel's tag upstream), §1 and §5 change — the precondition that
  failed becomes readable and belongs in this page.

## 6. Recorded failure mode of this page's absence

On #11454 the recorded session reached the enum question at its second turn and
then never sent a message: 22 assistant turns, every one a bare tool call, the
full 16K output budget spent inside reasoning blocks, and three or four
whole-`/workspace/repos` greps hunting an emitter that is absent from every
export. It was cut inside a reasoning block, so the reporter received nothing —
while the decisive facts (the `HcclResult` enum, `mooncake_backend.py:194` as
canned text, the `-1` vs `-200` split) were one or two calls away. The lesson is
`../evidence-workspace/delivery-discipline.md`: when the question is "what does
this code mean", the enum line *is* the answer.

A later session of the same issue loaded this page, confirmed the enum, the one-sided
call declarations and the canned-text line in the pinned trees (three greps), and
delivered the verdict, four ranked causes and the send-back list — the route in this
page is what worked. What that pass did not resolve, and this section now does, is the master's
`SSD Storage: 192.00 MB / 800.00 GB` line (§1) and the `--root_fs_dir` disk replica it
implies (§3 step 3).
