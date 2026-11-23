# Which block in a pasted Ascend error report is this incident

Scope: a pasted plog / serve log where several error families arrive together —
`EE9999 ... rtMemcpy ... ErrCode=107030`, `EZ9999 ... chipId: ... fftsplus
aivector error`, `507011 aclrtSynchronizeEvent`, `ERR00100 PTA call acl api
failed`, `ERR02005 DIST internal error`, an HCCL watchdog terminate. This page
answers one question that must be settled before any mechanism: *which of those
blocks happened when the reporter says the service failed.* Inherit the rules in
`index.md` and SKILL.md.

## The report is a merge of three kinds of text

1. **Live lines**, written at failure time: the vLLM traceback, the
   `aclrtSynchronizeEvent` / `event wait to be complete failed` failures, the
   teardown (`ERR02005`, `HCCL watchdog thread terminated`).
2. **A previously recorded error block, re-printed with its own original
   timestamp** — observed in #9183 as
   `EE9999[PID: 1229] 2026-05-12-03:50:17.907.457 (EE9999): rtMemcpy execution
   failed ...` sitting inside a traceback whose own log prefix reads
   `05-13 15:29:07`, i.e. ~36 h earlier. That a re-print keeps the record's date is
   the observation; the exact runtime bookkeeping behind it is not documented in
   the snapshot, so describe it as a re-printed record, not as "the same crash
   happened again".
3. **Consequences** of (1) or (2): a stream already in an error state makes every
   later `aclrtSynchronizeEvent` / `rtMemcpy` / query fail with its own code.

So the sentence a reporter quotes in the title can be a day older than the
failure. Adopting it names the wrong owner *and* the wrong ablation set.

## Procedure

1. One call decides it —
   `bash <skill_root>/scripts/plog-timeline.sh --since <YYYY-MM-DD of the failure> <pasted-log>`.
   It prints, per error family, the span of the embedded device records and of the
   log-line prefixes, and how many lines carry a record older than their own log
   line, then names the replayed families and the first in-window error.
   Without `--since` it uses the newest log-line date in the paste, which is
   enough when the paste really does cover the failure minute.
   The script needs a timestamped record: long-form `YYYY-MM-DD-HH:MM:SS` device
   records (the usual `[PID: n]` headers) or `MM-DD HH:MM:SS` line prefixes. A
   paste of ordinary `W<MMDD>`/`E<MMDD>` lines yields its "no timestamped error
   line found" verdict, so do not rebuild it into a file first (#18315).
2. If it reports *no* record inside the incident window, do not conclude yet: read
   its offset `NOTE` and its record clusters and re-run with `--since` at the
   cluster nearest the reported time — a multi-hour or multi-incident dump can hold
   the fault on the other time base, or a day earlier near local midnight, and that
   is a timezone, not a replay. Only when no cluster comes near the reported time
   does the paste not contain the fault: say so in one sentence, quote the older
   block only as history, and ask for the plog covering the failure minute
   (`~/ascend/log/debug/plog/plog-<pid>_<timestamp>.log`, every reporting rank) plus
   the vLLM log's first traceback. Do not build a mechanism from the older block —
   that is the recorded #9183 failure mode.
3. Search every rank for the device line: `grep -n 'EZ9999\|chipId' <log>` — another
   rank's block may own the fault (`collective-error-chain.md` step 4). Convert ids
   before comparing them: the PTA prefix's `Device:N`, a record's `device_id=N` and a
   dump's `(chipId:C, dieId:D)` are three different fields, and the last two are the
   *same* physical die (`device-health-checks.md` §*Which device actually failed*).
4. Order the in-window families by timestamp and classify the *first* one:
   - `EZ9999 ... (chipId:N, dieId:M) ... fftsplus aivector error ... error code =
     0x800000 ... The DDR address of the MTE instruction is out of range` →
     device-side kernel/pre-launch exception: `../custom-ops-runtime-faults/index.md`
     (and `device-health-checks.md` for the hardware/software split).
   - `507011` and `rtEventSynchronize ... reason=the model stream execute failed`
     → a synchronization on a stream that already failed. It is the messenger,
     never the owner.
   - `checkUceErrAndRepair: … NPU error, error code is 507899` with
     `rtGetMemInfoByType … invalid handle`, `rtGetMemUceInfo execution failed,
     reason=driver error:internal error`, `dsa task submit failed … 507053`,
     `launch failed for DSARandomNormal, errno:561000` → torch_npu's UCE
     (uncorrectable-error) *query/repair* path: the c10 exception names
     `checkUceErrAndRepair` itself, so these lines date the trap and report a
     device context the trap left broken; the random-number submit failure is a
     later submit on that context, **not** an RNG or `aclnnInplaceNormal` defect.
     Two readings stay open — a real HBM UCE, or the query failing because the
     trap killed the context — settled by `npu-smi info -t health -i <id>`, the
     plog's UCE record, and one rerun on another card.
   - the deepest Python frame sits in `profile_run` /
     `determine_available_memory` (e.g. `_dummy_sampler_run`'s
     `hidden_states[logit_indices]`, `vllm_ascend/worker/model_runner_v1.py:2158`
     at v0.14.0rc1) → a **sync point** of the last dummy forward, not the owner:
     the paste's own async caveat says the "current working operator" line names
     the last *submitted* op. Attribute with `--enforce-eager` +
     `ASCEND_LAUNCH_BLOCKING=1` before opening any kernel's shape code.
   - `hccl fftsplus task timeout occurred during task execution ... stuck notify
     num:1, timeout=<N>s` (`[FUNC:ProcessStarsHcclFftsPlusTimeoutErrorInfo]`) and
     the `[TaskExecStage][HeartbeatAbnormal] ... Heartbeat Lost Occurred` lines
     around it → **the timeout is an expiry, not a timestamp**: the stall began
     about `<N>` seconds earlier. Subtract before ordering anything; see "A
     long-run hang" below.
   - `ERR02005 DIST internal error`, `HCCL watchdog thread terminated`,
     `terminate called after throwing ...` → teardown after a process/device error.
   - `shm_broadcast.py:... No available shared memory broadcast block found in 60
     seconds` → an EngineCore has been waiting on a worker for ≥60 s, so it dates
     the stall relative to the device block; it says nothing about its cause
     (`../startup-failures/acl-graph-capture.md` step 2 reads the same line at
     startup).
5. Hardware vs software when several ranks fail: compare the device blocks. In
   #9183 three workers on three different chips (`chipId:3 dieId:1`,
   `chipId:5 dieId:0`, `chipId:1 dieId:1`) reported the same `error code =
   0x800000`, the same `mte error info` word and the same `pc start:
   0x12c0c3cec7b8`, with differing `blk:` — one kernel image executing an
   out-of-range address on every rank is a software/address-computation shape; a
   single card that keeps failing across runs is the hardware shape. Do not ask
   for a card replacement on the first shape.

## A long-run hang: `timeout=` dates the stall, not the fault (#16652)

Recorded #16652: vllm-ascend v0.23.0 (`5cb98ca`), GLM-5.2 W8A8C8, Atlas 800 A2,
2 nodes × 16×910B **co-located** (混部), three such instances, "4 times crashed
after running a long time, always `hccl fftsplus task timeout`", with four
attached dumps of 0.5-5.5 MB. The paste:

```
The error from device(chipId:0..7, dieId:0) ... hccl fftsplus task timeout occurred
during task execution, stream_id:9, sq_id:9, task_id:..., stuck notify num:1, timeout=1836s
[FUNC:ProcessStarsHcclFftsPlusTimeoutErrorInfo][FILE:device_error_proc.cc][LINE:1788]
```

- `timeout=<N>s` is the **armed notify-wait bound**, i.e. `HCCL_EXEC_TIMEOUT`, not
  a measured elapsed time. Its compiled default in this snapshot *is* 1836 s:
  `constexpr s32 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;   // notifywait默认1836等待时长`
  (`cann/hcomm/src/pub_inc/hccl_common.h:84`), which `SetDefaultParams` assigns to
  `execTimeOut` (`src/platform/common/externalinput.h:131`) and the init line
  reports as `HCCL_EXEC_TIMEOUT set by default to [1836]s`
  (`src/platform/common/externalinput.cc:542`); `DispatcherPub::GetNotifyWaitTime`
  then arms `hrtNotifyWaitWithTimeOut(...)` with `execTimeOut_`
  (`src/platform/task/dispatcher.cc:293-316`). The EI0002 catalogue text's
  "default 1800s" (`src/common/error_code/error_code.cc:38`) is stale doc wording,
  not the constant — **do not add a ~36 s "overhead" to reach 1836**. So the
  printed clock time is the expiry: **stall start = report time − `timeout=`**.
  In dump 1, `21:43:11 − 1836 s = 21:12:35`, and that instance's last
  `POST /v1/chat/completions ... 200 OK` is `21:12:11` — a 24 s match, which is
  what makes the arithmetic worth one call instead of a mechanism hunt.
- Two consequences of it being a *configured* bound, not an elapsed one. (a) Read
  the paste's own HCCL init line before subtracting: a reporter who exported
  `HCCL_EXEC_TIMEOUT` gets **their** value here (`set by environment to [%.2f]s`,
  same file `:1000`), so `timeout=600s` is not evidence of a 10-minute stall.
  (b) The accepted range is SoC-dependent — on 910B/910_93 `0`–`2147483647` s with
  `0` documented as never timing out, elsewhere 1–17340 s
  (`externalinput.cc:961-984`) — so raising it changes when the hang surfaces, not
  whether it does (`aiv-mc2-env-ablation.md`).
- The device block is not the only clock in the paste. Two vLLM timers date the
  same stall independently: the 300 s RPC watchdog
  (`TimeoutError: RPC call to <method> timed out`, `rpc-watchdog-timeouts.md`) and
  the `No available shared memory broadcast block found in 60 seconds` line from
  each EngineCore. Recorded #16646 (same shape as #16652: GLM-5.3 W8A8C8, 2×16
  910B co-located, v0.23.0) reports the fftsplus block at `19:54:23` and
  `19:54:23 − 1836 s = 19:23:47`, exactly the reporter's own "开始出现 ≈19:23:47";
  the same log's `TimeoutError: RPC call to sample_tokens timed out` is stamped
  `19:28:38` (= `19:23:38 + 300 s`) and its `60 seconds` warnings start `19:24:37`.
  So when a paste has no `200 OK` boundary, use those two lines: they bracket the
  stall to a few seconds and turn a single subtraction into a corroborated one.
- Eight chips reporting the same instant (`chipId:0..7`) is one process-group
  stall, not eight faults.
- The code beside it is a messenger, and its **count is bookkeeping**: in #16646
  the same chips print 80 × `error code is 507048`, each paired with
  `rtEventSynchronize execution failed, reason=fftsplus timeout` /
  `rtStreamSynchronizeWithTimeout execution failed, reason=fftsplus timeout`
  (`FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]`) in
  `multiproc_executor.py` — i.e. each worker's host-side synchronize reporting
  the device's expiry, ten lines per chip. Read it as "this rank was waiting
  when the bound expired", never as 80 faults and never as an owner: the
  device block above it owns the mechanism.
- `[TaskExecStage][HeartbeatAbnormal] Cluster Exception Location[IP/ID]:[...],
  Arrival Time:[...], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:
  1. Process has exited, 2. Network Disconnected` arrives at a **single** instant
  (13:17:49/50 in dump 1, 19:40:37/38 in dump 2) about 5 min *after* the derived
  stall start, repeated per peer device (`there are 16 abnormal device(s)`).
  Peers that are stuck in the same collective also stop heartbeating, so this
  line does not date the fault and does not prove a node died.
- The rest of the chain is downstream: `[shutdown] MPClient: start timeout=0s` on
  every `ApiServer_n`, then `Process ApiServer_1 (PID: …) died with exit code
  None`, then `Process group work ALLGATHER, seq_num …, dispatch timeout. device
  N: the current working operator name is aclnnAbs` (torch.distributed — the
  collective never dispatched; the named op is the last submitted one), then the
  fftsplus expiry and `EngineCore` exit — and then a supervised restart, ~40 s
  later in dump 2, which is why the earlier "crash" appears only as a gap between
  `200 OK` lines.
- `(Worker_DP0_TP5_DCP5_EP5 pid=…)` carries **rank indices**, not sizes — read
  sizes from the group lines (`group_name_… rankSize[N]`) or the serve command.

What the timestamps settle for that report: the stall starts ~30 min before the
first error text, while the instance was still answering, so "heartbeat lost" and
"EngineCore abnormal exit" are not the beginning. What they do not settle: why it
stalled — the record holds no maintainer diagnosis, and the honest deliverable is
that gap plus: the plog window around the derived stall start
(`grep -rn 'error cqe'`), per-card health (`device-health-checks.md`), whether the
crash follows one card/instance or moves, and whether the same instance's stall
recurs at a fixed interval (a leak/slow-rank shape) or at random (a fabric shape).

## When the expiry bound is not 1836: notify waits beside an AICPU `param invalid` (#12487)

The same subtraction has a second form, and the bound in it is **not** `HCCL_EXEC_TIMEOUT`.
Recorded #12487: vllm-ascend v0.22.1rc1 A3 3P1D DeepSeek-V4-Flash, healthy for 2.5 days,
then a P-node worker died. Its paste is, in order: `ERR00100 PTA call acl api failed` +
`[Error]: The aicpu execution is abnormal`; a Python traceback whose deepest frame is
`torch.repeat_interleave` in the MoE dispatch preprocess
(`vllm_ascend/ops/fused_moe/token_dispatcher.py:612`, right after the
`output_splits = … .to("cpu").numpy()` readback) with
`RuntimeError: … LocalScalarDenseNpu.cpp:23 … AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507018`;
a re-printed EI0019 record dated three days earlier (see above); then the device block:

```
serial number is 1 … notify wait timeout occurred during task execution,
  stream_id:142, sq_id:146, task_id:46645, notify_id=155, timeout=204s
… serial number is 2,3,4 (streams 143,144,182, same 204s) …
  tag[AllGather_group_name_7ringAlignedAllGatherDoubleRingFor91093Executor_device],
  AlgType(level 0-1-2):[ring-ring-NHR], Waiting peer end: 4294967295, remote rank:[local]
serial number is 5 … an exception occurred during AICPU execution, stream_id:91,
  task_id:57859, errcode:1, msg:execute kernel param invalid  →  retCode=0x2a
Aicpu kernel execute failed, device_id=8, stream_id=91, so_name=libccl_kernel.so,
  funcName=RunAicpuRpcSrvLaunchV2, kernelName=RunAicpuRpcSrvLaunchV2_reduce_scatter
```

- `507018` is `ACL_ERROR_RT_AICPU_EXCEPTION`
  (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:100`), reported at a
  host readback in the MoE dispatch path — a **sync point**, not the owner (step 1).
- **The block is a record list, not a causal chain.** Serials 1-4 are the waits and the
  AICPU record is last, so "the AICPU `param invalid` came first" is not something this
  paste establishes. Two readings stay open and both are lanes: (a) a rejected
  `reduce_scatter` RPC launch never signalled, so the AllGather waits expired after their
  bound — the AICPU record owns the case (CANN/HCCL, `soName=libccl_kernel.so`); (b) the
  peer had already stopped and the AICPU record is a later symptom on a poisoned stream.
- **Read the descriptor before the serials.** Here the AICPU record *names* a kernel
  (`RunAicpuRpcSrvLaunchV2_reduce_scatter` in `libccl_kernel.so`), which is what makes it
  HCCL's launch and not a compute kernel. When the same `msg:execute kernel param invalid`
  arrives with `soName=`/`funcName=`/`kernelName=` **empty**, this page's serial arithmetic
  is not enough — decode `0x2a`/`errcode` and read the cascade in
  `aicpu-task-exception-empty-kernel.md` (#12776).
- `timeout=204s` is that Notify Wait task's **armed bound**, and it expired, so
  `stall start ≈ print time − 204 s` (⇒ ≈`06:24:20` for a `06:27:44` block) — the same
  arithmetic as the 1836 s block above. What it is *not*: 204 ≠
  `NOTIFY_DEFAULT_WAIT_TIME = 27 * 68` (`cann/hcomm/src/pub_inc/hccl_common.h:84`), so
  this wait was not armed by `HCCL_EXEC_TIMEOUT` and the paste says nothing about the
  reporter's `HCCL_EXEC_TIMEOUT` setting. Which class armed 204 s is not in the exports —
  a stated gap, not a finding.
- `Waiting peer end: 4294967295` (`0xFFFFFFFF`) with `remote rank:[local]` is HCCL's
  invalid-rank value, not a rank id: `INVALID_VALUE_RANKID = 0xFFFFFFFF` /
  `INVALID_UINT = 0xFFFFFFFF` (`cann/hcomm/src/pub_inc/hccl_common.h:54,57`) — the dump
  could not name the peer it was waiting on. So the
  paste cannot name the owner, and no local read can. The deliverable is the request:
  every other rank's plog for `06:20`-`06:28` (first ERROR, and whether any rank issued
  the same `tag`/`reduce_scatter`), the twin DP engine's and the P-side logs for the same
  second, and the plog of the die the dump names (`chipId:4`/`device_id=8`, one die — see
  `device-health-checks.md` §*Which device actually failed*). The 7 minutes of
  `No available shared memory broadcast block found in 60 seconds` before the print date
  the stall the same way (`rpc-watchdog-timeouts.md`), and the reporter's title line
  (`port 65536 … already been bound`) is out of window.

## Where `107030` belongs

`107030` is `ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT`, "stream is captured, not support
current oper" (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:51`), so
the code is minted **only while a stream is in capture** — that is a definition, not an
inference from timing, and it rules out "an rtMemcpy inside graph *replay*" however the
reporter words it. `rtMemcpy:ErrCode=107030, desc=[the current capture mode does not support this
operation]` with `synchronized memcpy failed, kind = 1` and small `cnt` (72 and
456 bytes in #9183) is a real capture-path family in the history — #6002 (v0.11.0,
DS-V3.1-w8a8, MTP + `FULL_DECODE_ONLY`), #4208, #8502 — where #6002's reporter
reads it as a framework-side blocking copy during capture and proposes a
capture-mode switch (reporter's reading; no confirmed fix in the record), and #8502
is reporter-confirmed to serve normally with `multistream_overlap_shared_expert`
off (`../startup-failures/acl-graph-capture.md`).

Three conditions before you adopt it:

- **It must belong to this process.** A record older than the failure *minute* is
  either the previous crash (#9183: 36 h and a different process) or **this
  process's own init/capture window** (#10700: recorded at `10:01:23`, inside
  `init engine (profile, create kv cache, warmup model)`, reported at `10:28:19`).
  Compare it with the log's own startup lines before discarding it — see
  "A record from this process's own init is not a stale replay" below.
- **It can also be re-printed by a different op, with the op's own words and the
  record's code** — read the traceback's shape before treating the block and the
  `RuntimeError` as the same event. PTA's `NPU_CHECK_ERROR` replaces a failing
  call's printed code with `AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)` (peeked,
  never cleared) and takes the text from `getDeviceErrorMessage`, so an op that
  dies two hours later on the same thread prints `call aclnnX failed, error code
  is 107030`: the sentence is this call's, the number and its description belong
  to the capture window, and the op's real status is masked. That report is a
  *diagnosability* finding, not the replay mechanism, and the artefact is the plog
  at the failure second — `latched-runtime-error-masking.md`.
- The same block prints `threadCaptureMode=0, exchangeCaptureMode=0,
  contextCaptureMode=0`, i.e. its own diagnostic fields show no capture in
  progress. The sentence therefore does not explain itself; ask for a re-run with
  `--enforce-eager` (graph off) rather than reasoning from the wording.

## A record from this process's own init is not a stale replay

The #9183 lesson ("older than the failure -> the previous crash") has a twin that
reads identically on the page and is the opposite case: the record predates the
failure **minute** because the fault happened during this process's own engine
init, and only a later launch/synchronize reported it.

Recorded #10700 (vllm-ascend v0.21.0rc1 `80610e4`, 2x Atlas 800I-A2, GLM-5.1,
DP2 x TP8 + EP, `deepseek_mtp` 3, `FULL_DECODE_ONLY`, `enable_npugraph_ex`):

| Fact | Evidence in the paste |
|---|---|
| Every rank's `EE9999` inner record is stamped `2026-06-24-10:01:23.68x`, sub-10 ms apart across chips (`.685`, `.686`, `.688`, `.689`, `.690`) | one device-wide event at that instant, not eight independent ones |
| `10:01:23` sits inside the same log's `init engine (profile, create kv cache, warmup model) took 144.14 s (compilation: 52.37 s)`, printed `10:02:16`; the graph-capture line is `10:02:13` | the record is from this process's init window |
| The Python traceback reporting it is stamped `10:28:19`, and its frames are `worker.py execute_dummy_batch -> model_runner_v1._dummy_run -> _model_forward -> acl_graph.__call__ -> torch.npu.current_stream().synchronize()` | the *report* is the first launch/sync that surfaced it, tens of minutes later |
| The service had answered `200 OK` between `10:02:24` and `10:13`, i.e. after the record and before the report | "ran fine for a while" is compatible with a fault recorded at init |
| A later release reproduced it: v0.22.1rc1, record `2026-07-01-01:19:08`, report `03:00:17` (~101 min), same frames, same `MTE` / `0x800000` / `507011` family | the gap is a report delay, not a fixed bug |

So the discriminator is not the gap's size but **process identity plus position in
the log**: a record inside `init engine ...` / `Graph capturing finished` /
`Capturing ACL graphs` of the same log belongs to this incident; a record from
before the process started (or with a different pid, container or dataset) does
not. Say "recorded at init, reported at <time>" rather than "crashed at <time>",
and ask for the plog covering *both* ends.

Mechanism still open at that report: the record text is
`rtMemcpy execution failed, reason=the current capture mode does not support this
operation`, so a host/device copy was submitted while the stream was in capture
mode; what then surfaced 27 min later is an `fftsplus aicore/aivector ... error
code = 0x800000 ... The DDR address of the MTE instruction is out of range`. Name
the owner from the plog's `[AIC_INFO] fault kernel_name=`, not from the wording —
route `../spec-decode-mtp/draft-graph-capture-device-fault.md` when the launch
carries MTP + `FULL_DECODE_ONLY`.

## Worked case #9183 — the answer that quoted the previous day's crash

vllm-ascend v0.18.0rc1 `99e1ea0`, vllm v0.18.0 `bcf2be9`; A3, 1P1D (2+2),
DeepSeek-V3.2-w8a8, failure during an HLE evaluation.

| Fact | Evidence |
|---|---|
| The reporter's first paste is a plog excerpt only: 400 lines, 45 timestamped lines, **all** `2026-05-12-03:50:22`, all `rtMemcpy`/`MemCopySync`/`107030`; no Python traceback, no second date | first comment (25 KB paste) |
| The recorded session ranked a capture-window H2D copy (sfa `.item()` loops, `multistream_overlap_shared_expert`, `--async-scheduling`, tight HBM) from that block, and asked for the plog | the run's answer |
| The plog the reporter then posted has **two** dates: 6 `EE9999` lines carrying `2026-05-12-03:50:17` records inside `2026-05-13 15:29:07` tracebacks, and the live block | comment `#issuecomment-4539074892` |
| In-window first error: `EZ9999 ... 2026-05-13-15:29:05.818 ... error from device(chipId:3, dieId:1) ... fftsplus aivector error ... The DDR address of the MTE instruction is out of range` | same |
| Then `507011` at 15:29:07 from `_get_valid_sampled_token_count` / `get_output`, then `ERR02005` + HCCL watchdog terminate; `shm_broadcast ... 60 seconds` at 15:28:39 on all four EngineCores | same |
| Three chips hit the same `pc start` in the same two seconds (see step 5) | same |
| What each ask produced: the request for versions got 「vllm-ascend和vllm版本 这些The output of `python collect_env.py`输出的信息里不都有吗？」 (the body's own block already prints `vLLM Version: 0.18.0`, `vLLM Ascend Version: 0.18.0rc1`); the request for the fuller first error produced the two-date traceback above; the request for the plog produced `plog-188432_20260515021316221.log` | reporter comments 2026-05-26 `#issuecomment-4539003016`, 2026-05-18 `#issuecomment-4474949456` |
| The body's block and the traceback share **pid 983** (`RUNTIME(983)` in the first, `Worker_DP2_TP2_EP10 pid=983` in the second): one long-lived worker spans both days, which is why the re-printed record reads as "the same crash again" | body `<!-- Failed to upload "plog-983_20260512035017922.log" -->`; `#issuecomment-4539074892` |

State of the conclusion: **unresolved** — no maintainer diagnosis or fix is in the
record. What the timestamps settle is the ownership split: the reported failure is
a device-side aivector/MTE-address exception inside the window, and the `107030`
capture-mode block is the previous day's crash and cannot anchor the diagnosis.
The case is a *repeat*, not one bad process: pid 983 appears on both days and the
reporter later posted `plog-188432_20260515021316221.log`, a third run starting
`20260515-02:13:16` — read that run's `chipId` lines before crediting one card. A
plog file name is `plog-<pid>_<YYYYMMDDHHMMSS>.log`, so the pid and start time **in
the name** select the run: the body's `Failed to upload
"plog-983_20260512035017922.log"` is exactly the plog of its own pasted block, so
re-ask for that file by name instead of for "a plog", and for the other runs'
pids likewise.
Next asks for the reporter, in this order: which kernel/op ran at 15:29:05 on that
chip (plog lines above the `EZ9999` block, tiling `I`-line if the op is an AscendC
custom op), the same failure with `--enforce-eager`, then with
`ASCEND_LAUNCH_BLOCKING=1` to attribute the op, and whether the three chips repeat
the same `pc start` on every attempt (software) or one chip does it alone on
re-runs (hardware).

## A two-node paste carries two environments (#6275)

Worked case #6275: vllm-ascend **v0.14.0rc1** (`52d4acf`) + vllm **v0.14.1**
(`d7de043`), 2×A2 910B2 64G, `DeepSeek-V3.2-Exp-w8a8`, DP2 × TP8 + EP + MTP
`num_speculative_tokens: 2`, `FULL_DECODE_ONLY`. The dump is the
`fftsplus aivector … MTE DDR out of range` trap during `profile_run`
(`Worker_DP0_TP0_EP0`, `determine_available_memory`), followed by the UCE-repair
block above.

Two findings that change the answer, both from the paste rather than from source:

- The launch command is **verbatim** the tree's own A2 multi-node block
  (`docs/source/tutorials/DeepSeek-V3.2.md`: TP=8, DP=2, `cudagraph_capture_sizes:
  [3,6,…,48]`, `num_speculative_tokens: 2`, `--data-parallel-rpc-port 13389`,
  `HCCL_INTRA_PCIE_ENABLE=1`). Config-diff route exhausted — "your command
  deviates from the tutorial" is a wrong answer for this report, and the
  remaining deviations to ablate are the framework ones (`--enforce-eager`,
  MTP off, `VLLM_ASCEND_ENABLE_MLAPO=0`).
- The two nodes' `collect_env` dumps disagree: master `npu-smi 24.1.rc3` on
  openEuler 5.10 versus worker `npu-smi 25.3.rc1` on Ubuntu 5.15 (both CANN
  8.5.0). On a 2-node HCCL deployment, diff the two dumps first and align
  driver/firmware/CANN/OS before any kernel theory — the containers mount the
  *host* driver (`-v /usr/local/Ascend/driver/lib64`, `npu-smi`,
  `version.info`), so the two ranks really are running different stacks.

The recorded session reached this list and shipped **no message text at all**
(rule 1: 14 tool turns, then the output cap).
