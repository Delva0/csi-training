# An AICPU task exception (`retCode=0x2a`, `errcode=1`) — including the empty kernel descriptor

## Report shape

A serve that ran for hours dies with a plog block like this and no Python frame that owns it
(verbatim from #12776, vllm-ascend v0.23.0rc1 `f4a08bdd` + vllm v0.23.0, DeepSeek-V4-Pro-w4a8
MTP, TP8/DP4, `FULL_DECODE_ONLY` + `enable_npugraph_ex`):

```
[ERROR] RUNTIME(216,):[stars_engine.cc:1548] ProcLogicCqReport:Task run failed, device_id=0,
  stream_id=5, task_id=40187, sqe_type=1(aicpu), errType=0x1(task exception), sqSwStatus=0x1
[ERROR] RUNTIME(216,):[device_error_proc.cc:1644] ProcessStarsAicpuErrorInfo:… serial number is 6,
  an exception occurred during AICPU execution, stream_id:5, task_id:40187, errcode:1,
  msg:execute kernel param invalid.
[ERROR] RUNTIME(216,):[davinci_task.cc:209] SetStarsResultForDavinciTask:AICPU Kernel task happen
  error, retCode=0x2a.
[ERROR] RUNTIME(216,):[davinci_kernel_task.cc:1435] PrintAicpuErrorInfo:Aicpu kernel execute failed,
  …, soName=, funcName=, kernelName=, errorCode=0x2a.
… flip_num=148, kernel_type=5, fault op_name=, extend_info=.
```

The engine-side text is only a messenger (`RuntimeError: cancelled` from `shm_broadcast.py`, or
`Worker proc VllmWorker-0 died unexpectedly`) — `cancelled-executor-teardown.md`.

## Step 1 — decode the two numbers before any mechanism (one call)

```bash
bash <skill_root>/scripts/cann-error-lookup.sh 0x2a 1
```

| field in the paste | decodes to | where |
|---|---|---|
| `retCode=0x2a`, `[aicpu exception]` | `TS_ERROR_AICPU_EXCEPTION = 0x2A` — the bracketed text **is** the enum's own comment, so it adds no mechanism; siblings `0x25` aicore timeout, `0x26` aicore exception, `0x27` trap, `0x28` aicpu timeout | `cann/driver/src/sdk_driver/ts_agent/inc/tsch/task_scheduler_error.h:25,65-68` (mirrored in `cann/hcomm/externel_depends/tsch/`) |
| `errcode:1, msg:execute kernel param invalid` | `KERNEL_EXECUTE_PARAM_INVLID = 1, // kernel execute param invalid` in `enum rtAICPUErrorType` | `cann/runtime/src/runtime/feature/inc/device/device_error_info.hpp:142-146` |
| `sqe_type=1(aicpu)` | the runtime prints the task type's own name — one fact, no lookup | — |

**The `errcode` family decides the owner, and the numbers collide with `HcclResult`:**

- `1/2/3` (`KERNEL_EXECUTE_PARAM_INVLID` / `_INNER_ERROR` / `_TIMEOUT`) — an **AICPU-executed
  kernel** rejected its own execution parameters, failed internally, or timed out. That is the
  kernel reached the CPU-side execution unit and the unit refused; it is not a host-side
  scheduling or tiling complaint unless another line says so.
- `7/8/9-12` `AE_STATUS_SILENT_FAULT` / `_TASK_ABORT` / `_STRESS_DETECT_FAULT*` — a processor
  fault, i.e. the device-health lane (`device-health-checks.md`).
- `1000/1001` are HCCL's own retry/link failures inside the AICPU channel; `11001-11006` are the
  **aicpu kernel API** layer (`AE_OPEN_SO_FAILED`, `AE_GET_KERNEL_NAME_FAILED`,
  `AE_KERNEL_API_PARAM_INVALID`); `21001+` the **aicpu scheduler/model/tool** layers
  (`AICPU_PARAMETER_NOT_VALID`, `AICPU_TASK_EXECUTE_FAILED`, …); `22001+` the *custom* aicpu sd.
  So `errcode=21001` and `errcode=1` are different statements: never merge them.

**Two readings offered by reporters on this shape are not established by any export.**
`kernel_type=5` has no enum anywhere in the snapshots (no `KernelType`/`kernel_type` definition
prints 5), so the "= KFC (kernel function call)" reading introduced on #12776 is unverifiable
here — and unnecessary, because `sqe_type=1(aicpu)` on the line above already names the family.
`flip_num` is a 16-bit field of `tag_ts_flip_task_t` (`cann/hcomm/externel_depends/tsch/
task_struct.h:899-902`; driver mirror `…/task_struct.h:904-905`), the *flip* task descriptor:
nothing in the exports ties the printed value to a graph-replay iteration, so "148 graph replays"
is a reading, not a counter. Do not grow a per-replay (capture-metadata / stale-workspace)
mechanism out of it before a second line corroborates the replay count.

## Step 2 — the first AICPU record is the hint; the rest of the block is the cascade

After the exception the runtime marks the context failed and prints the failure mode on every
later submission. These lines are one fault with N prints, not N faults:

```
GetError:Stream Synchronize failed, retCode=0x2a
EnterFailureAbort:stream_id=5 enter failure abort
AllocTaskAndSendStars:context is abort, status=0x715002a
SubmitTaskStars:… streamId=46, taskId=0, taskType=3, taskName=EVENT_WAIT, retCode=0x715002a
StarsResumeRtsq:stop scheduling in abort failure mode: stream_id=5, sq_id=44, sq_head=1127
StreamWaitEvent / rtStreamWaitEvent:ErrCode=507018, desc=[aicpu exception], InnerCode=0x715002a
APP:WaitEventFunc:OpParamMaker.cpp:539: "aclrtStreamWaitEvent error! ret = 507018 …"
APP:ReadQueue:NPUQueue.cpp:454: "device = 0, write_idx = 3648, read_idx = 879, status = 1, ret = 507018"
```

- `507018` is `ACL_ERROR_RT_AICPU_EXCEPTION` — the *category* of the first record, not a new
  fault, and `0x715002a` is the same abort status with the module prefix.
- `OpParamMaker.cpp:539` is **not** a second owner: `WaitEventFunc` (`:529-547`) overwrites the
  call's own status with `AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)` before printing, so the
  `507018` there is the thread's latched code — the masking mechanism of
  `latched-runtime-error-masking.md`, reaching the torch_npu dispatch queue. The wait is where a
  poisoned stream surfaces.
- `NPUQueue … status = 1` is `RepoStatus::RUN` (`Ascend/pytorch/torch_npu/csrc/core/npu/
  NPUQueue.h:19-28`), and `read_idx`/`write_idx` are the ring's tail/head (`:114`, `:135-136`),
  so `write_idx − read_idx` is how many queued host tasks sat between them at the print: 2769 in
  #12776, 25 in the recorded #10829 paste. Use it as a *corroborating* backlog figure — it agrees
  with a throughput collapse that started before the first ERROR — and not as proof: the ring
  capacity and the reader's stop-on-error behaviour are not in the export.
- `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!` on another
  pid is the TBE subprocess of a worker whose parent died (`../startup-failures/silent-worker-exit.md`);
  teardown, never a second cause.

The ordering question is rule 13's: if the metrics show one DP engine decaying tens of seconds
before the record prints (#12776: collapse at 02:50:15, record at 02:51:31), then the record is
the tail. `error-report-timestamps.md`; `bash <skill_root>/scripts/plog-timeline.sh --since …`.

## Step 3 — can this record name a kernel at all?

"`soName`/`funcName`/`kernelName` all empty ⇒ nothing was launched" is a hypothesis, not a
finding. Two facts bound it:

- A device-side AICPU task that runs **tiling** has no kernel identity to print: vllm-ascend's own
  `csrc/common/include/tiling_sink/tiling_aicpu_task.h` declares
  `struct TilingAicpuTask { gert::TilingContext *tilingContext; const char *opType; uint64_t
  notifyAddr; uint64_t workspaceAddr; uint64_t workspaceSize; }` — op type and addresses, no
  `soName`/kernel name. Its consumers are **not** in the export (only that header and
  `csrc/common/src/tiling_sink/CMakeLists.txt`), so which ops submit it stays unverified.
- The artefact that *can* name the task is the device DFX dump / `[AIC_INFO] fault kernel_name=`
  segment for the printed `task_id` (with `origin_op_name` when the plog carries it). Ask for it
  by id; it is not derivable from the record.

Named descriptors decide the lane immediately, so read the field once it is populated:
`RunAicpuRpcSrvLaunchV2_reduce_scatter` in `libccl_kernel.so` is HCCL's own RPC launch and
usually arrives after `notify wait timeout … timeout=<N>s` expiries — the collective/device lane
of `error-report-timestamps.md` §*When the expiry bound is not 1836* (#12487); `ScatterElements`
or `IndexPut` in `libcpu_kernels.so` / `libpt_kernels.so` with `funcName=RunCpuKernel` is a
CPU-fallback op, so the owner is the Python statement that launched it
(`../soc-platform-paths/310p-cpu-fallback-op-fault.md`).

An **empty** descriptor with no notify-wait records, where the AICPU record is the *first* ERROR
of the process (#12776), is neither of those yet. The honest deliverable is: the two decodes
above, the cascade read, the ladder below, and the per-rank plog + DFX ask — not a mechanism.

## Step 4 — the reproducer's pre-flight, because the ask depends on peer plogs

#12776 shipped **one** readable plog (the crashed worker's); the other seven were 0 bytes. In
#12487 the lane split came from the *peers'* `notify wait timeout` serials, so empty peer plogs
delete the discriminator. Before spending a reproduction run (these clusters reproduce after
~20 h), have the reporter verify collection: one `ASCEND_PROCESS_LOG_PATH` with per-rank
sub-directories, `ASCEND_GLOBAL_LOG_LEVEL=1`, plog rotation/`…_PRINT_TO_STDOUT` left at the
default that writes files, then `ls -l <plog-dir>/*` — every rank non-zero — plus the device
DFX/AICPU dump directory for the failing `task_id`.

## Levers, cheapest first

1. `--enforce-eager` (or `cudagraph_mode` `PIECEWISE`) — removes the captured-replay path; use
   the **global** flag, since a `--speculative-config` `enforce_eager` key is dead on some
   families (SKILL.md rule 15).
2. `"enable_fused_mc2": false` in `--additional-config` — removes the MC2 MoE-dispatch path that
   submits on-device AICPU/tiling work, which is the one in-tree producer of AICPU tasks.
3. `ASCEND_LAUNCH_BLOCKING=1` — only *after* step 1 (refused with `ValueError: ACL graph is
   incompatible with ASCEND_LAUNCH_BLOCKING=1`, `vllm_ascend/platform.py`).
4. `npu-smi info -t health -i <id>` and per-rank RSS/HBM **at the failing minute**, not the
   startup table.

## Do not re-derive these at v0.23.0rc1 `f4a08bdd` (verified absences)

- `0x2a` is in no `rt_error_codes.h` / `hccl_types.h`; the lookup script now resolves it through
  the TS table (that header ships with `cann/driver`, so a snapshot without the driver tree
  cannot resolve a TS code — quote it as text).
- No `KernelType` enum and no `flip_num` semantics in any export (Step 1).
- The **W4A8 MC2 aliased-workspace sizing is already fixed at this pin**:
  `csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp:253-260`
  carries `// ptrC/ptrC2 alias reuse: allocate max(N, n2), not N + n2` and
  `std::max(info.N, n2)` / `std::max(info.K, k2)`. So a w4a8 MoE crash here is **not** the
  aliasing bug `../custom-ops-runtime-faults/mc2-variant-workspace-and-strides.md` records for
  the W8A8 variant at another pin — that page's W4A8 "already fixed" row is what this line
  proves, and it removes the whole theory for a w4a8 report at this ref.
- The param-check sources behind the message (`cann/opbase`, `cann/ops-*`) are not exported; the
  phrase exists only as the enum comment above.

## A long-run variant the pin itself documents

`v0.23.0rc1` `docs/source/user_guide/release_notes.md:85` (#12345): DeepSeek V4 Pro on A3/A5
"can show continuously increasing memory usage in both P/D-disaggregated and co-located
deployments, eventually causing OOM or **service instability**". A crash after ~20 h of healthy
service is that shape, and a reporter's `kv_cache_usage 7.8% (not OOM)` excludes **KV-cache** OOM
only — that row is about growth outside the KV budget. Keep it as a ranked cause beside the
device record, with the check being per-rank RSS, host memory and `npu-smi` HBM for the minutes
before the print (`../runtime-memory/host-ram-and-per-rank-rss.md`), and state the ladder so the
device record is not the only branch.

## Scope

Verified at vllm-ascend v0.23.0rc1 (`f4a08bdd`) + vllm v0.23.0 (`0fc695fc`); the TS and
`rtAICPUErrorType` enums are driver/runtime exports, so re-check by enum **name** rather than by
line number at another pin. #12776 remains open at this writing: its reporter's follow-up
(2026-07-24) restated the empty-descriptor and KFC readings and reported the seven empty plogs,
which is exactly the evidence gap Step 4 addresses.
