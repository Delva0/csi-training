# Per-device health checks (asys) and hardware fault codes

Goal: decide hardware vs software for a collective/device failure, using tools
the reporter can run themselves.

## Run the check on every device

```bash
for i in $(seq 0 7); do asys health -d=$i; done
# brief all-device summary instead of per-device detail:
asys health
```

The range is the **host's physical device ids**, not `ASCEND_RT_VISIBLE_DEVICES`: recorded
#7477 exposed `0,1,2,3` to the job while the flagged chips ran to 7, so a scan bounded by
the visible list under-reports the host (and reports nothing at all when the job's own
chips are the clean ones). The health table is a host fact; the launch's device list is not
its scope.

`asys` ships in CANN `oam-tools` (`cann/oam-tools/src/asys/`), subcommands:
`help, collect, launch, info, diagnose, health, analyze, config, profiling`.
Related escalation: `asys info` (inventory), `asys diagnose` (device diagnostics),
`asys collect` (bundle logs for a hardware case).

## Read the output correctly

Source (cann/oam-tools, `src/asys/health/asys_health.py` and
`src/asys/common/device.py`):

- `Overall Health` comes from `dsmi_get_device_health`
  (0 Healthy, 1 Warning, 2 Alarm, 3 Critical).
- The `ErrorCode Num` rows come from `dsmi_get_device_errorcode` +
  `dsmi_query_errorstring` — a separate source of truth.

So **`Overall Health: Healthy` together with `ErrorCode Num: 1` is normal and not
a contradiction.** Do not stop at the health word; read the code rows. On screen
only the first five error records are printed; the full list is written to
`health_result.txt` in the asys timestamped output directory.

The string next to each code is the driver's own description (node type, sensor
type, event state). Quote it, and look the code up in the hardware
health-management fault definitions for that product/CANN line — the required
action depends on the product, so prefer the vendor document over inference.

## Which device actually failed

The id in the PTA line is the **reporting** rank's device, not necessarily the faulting
chip — but read the three id fields apart before saying so, because a paste carries all
three and they are different kinds of id:

| Field | Who mints it | What it means |
|---|---|---|
| `Device:N` in `(PID:…, Device:N, RankID:N)` | torch_npu's own error prefix | *this process's* device handle: `device = torch_npu._C._npu_getLocalDevice()` → `c10_npu::GetLocalDevice()` (`torch_npu/utils/_error_code.py:53-75`, `torch_npu/csrc/npu/Module.cpp:933-938`, `Ascend/pytorch` v26.0.1); `rank = dist.get_rank()` or `-1` when that raises |
| `device_id=N` in `[DFX_INFO] …` / `PrintAicpuErrorInfo` | runtime / plog | the **physical** device |
| `(chipId:C, dieId:D)` in a dump header | runtime | the same physical device as card+die |

The last two are one device in two forms: the driver maps a physical device id to
exactly one chip+die pair (`hal_kernel_get_chip_die_id(phy_id, &chip_id, &die_id)`,
`cann/driver/src/sdk_driver/inc/ascend_kernel_hal.h:1218-1221`; DCMI's
`dcmi_get_card_id_device_id_from_phyid` and `dcmi_get_device_phyid_from_logicid` in
`cann/driver/docs/zh/Ascend_910_93_DCMI_API/`). So `device_id=8` beside
`(chipId:4, dieId:0)` is **one** die (4×2+0 = 8 on a two-die card) and `Device:0` beside
either contradicts nothing — that pair is not evidence that the fault sat on another
rank's chip. Locate the die with `npu-smi info -t board -i <id>` / `asys health -d=<id>`
or the plog file for that id, not by comparing the fields.

In #10829 (A3, two nodes) node0 printed `ERR00100 PTA call acl api failed` and, in the
same output, an `EZ9999` dump of `The error from device(chipId:4, dieId:0), … there is
an exception of fftsplus aivector error, … error code = 0x800000`; node1 printed only
the bare `SUSPECT REMOTE ERROR`, with no chip dump at all. That difference between
ranks — not the `Device:0`/`chipId:4` pair — is what puts the fault on a card that may
never print it, on either node.

Order that costs one paste instead of a card swap: gather all ranks' logs, grep for the
`EZ9999` / `chipId:` line and for the earliest ERROR in every rank's plog, then run the
per-device health check below on **both** nodes and correlate that chip id (not the
reporting `Device:N`, which is only the process's own handle). Statement to make in the answer: the reporting rank's device is
correlated, the faulting chip is named only by the `chipId` dump, and if no dump exists
the faulting card is still unknown.

## Recorded case (vllm-ascend issue #7477)

- TP=4 with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, Qwen3.5-27B,
  vllm-ascend v0.17.0rc1, driver 25.2.0, CANN 8.5.1, torch_npu 2.9.0.
- Failure: first collective of `profile_run` died with `HcclAllreduce` /
  `hcclRet -> 4`, `AllReduceOpbasePipeline ... intraRankSize_[4]`, on
  `Device:2`; see `collective-error-chain.md`.
- `asys health -d=$i` for 0..7: devices 0 and 1 clean; devices 2,3,4,5,6,7 each
  reported `ErrorCode Num: 1` with `0x81a3880c`, string
  `node type=HCCS_PORT, sensor type=Extend Sensor, event state=Link diagnosis
  exception`, while `Overall Health` still printed `Healthy`. Note the spread: the job's
  visible set was `0,1,2,3` and the TP group was those four, yet chips 4-7 were flagged
  too — so the flagged set is wider than the failing group, and the group's own two
  flagged chips (2,3) are what correlated with the `Device:2` traceback.
- Correlation that closed the case: the failing rank's `Device:2` is inside the
  TP group and is one of the unhealthy devices. The reporter recorded the cause
  as a hardware problem and cross-checked `0x81a3880c` against the vendor health
  management fault definitions.

## Device lost / heartbeat lost (`507010`)

Signature in the container log: `RuntimeError: ... aclrtLaunchKernelWithHostArgs
failed ... error code is 507010` (or `ACL_ERROR_RT_LOST_HEARTBEAT`), `npu-smi info`
shows one chip `Critical`, `ping` to the host is fine and the host never rebooted. The
title is usually 「设备失联 / Device Lost / Heartbeat Lost」, often with a second
「共享内存不足」 complaint.

- `507010 = ACL_ERROR_RT_LOST_HEARTBEAT` "ts lost heartbeat"
  (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:92`, vendor row
  `cann/runtime/docs/api_docs/aclError.md:163` whose action is 收集日志并联系技术支持). It
  is a **runtime/device** code, not an `HcclResult` enum: no vllm-ascend branch mints it,
  and no source read turns it into a software cause. Look it up, do not recall it —
  `bash <skill_root>/scripts/cann-error-lookup.sh 507010 507011 107030` with
  `timeout_ms` ≈ 30 s: a scan of `/workspace/repos` takes ≈2 s, past the 1 s foreground
  cap, so an unsized call returns as a background job and costs a `job_output` round trip
  (`first-checks/scripts-catalogue.md`; the #11171 replay paid that turn).
- One read decides the owner: the **host `dmesg`**. The driver names the dying die before
  the container reports anything (`receive ts exception message from device(N)`,
  `device(N) is not working`, `ts is down`, then `hdcdrv_*`/mailbox callbacks suppressed).
  `asys health -d=N` / `asys collect` on that id is the reporter-side confirmation.
- The `No available shared memory broadcast block found in 60 seconds` line is the
  EngineCore ringbuffer watchdog waiting on the worker that stopped stepping — a
  *consequence* of the lost device, never a `/dev/shm` sizing problem, so `--shm-size`
  is not a rung (`rpc-watchdog-timeouts.md`; `error-report-timestamps.md`).
- Rank, and say it in that order: (1) the device/TS failure on that die — hardware
  escalation with the dmesg block and per-device health; (2) driver/HDK vs the image's own
  CANN base (host `cat /usr/local/Ascend/driver/version.info` against
  `rg -n 'FROM ' Dockerfile*` at the reported tag); (3) an older tag (the version rung).
  A lost-heartbeat code is not repaired by (2) or (3) alone, so do not let a
  version-looking paste demote (1) — see the case below, where (1) was the answer.

## Recorded case (#11171, 310P1, image v0.14.0rc1-310p-openeuler)

Two 310P1 dies, Kunpeng 920S host, `CANN 8.1.RC1`, Qwen2.5-7B, TP=2, `--enforce-eager`;
the engine came up (init engine 3.04 s), served one streaming request at ~6.8 tok/s, the
throughput line fell to 0.0 tok/s at 01:34:35, and device 1 lost heartbeat at 01:34:48.
The reporter's two complaints (「心跳丢失」 and 「共享内存不足」) are **one fault reported in two
sessions** — answer with one cause and say the second symptom is that same event seen
later; never open a `/dev/shm` sizing lane from it.

- Session A, the attached container log (`vllm-ascend.txt`, start 06-16 01:30:48):
  `507010` on `[rank1]` / `Device:1` at 01:34:48-49 (`aclrtLaunchKernelWithHostArgs`,
  `EL9999 … Device[1] fault, ret=0x711000b`), and at 01:35:11 the EngineCore dies out of
  the ringbuffer read — `multiproc_executor.py:336 mq.dequeue` → `shm_broadcast.py:616
  dequeue` → `:531 acquire_read raise RuntimeError("cancelled")` → `EngineDeadError` at
  `async_llm.py:546` — then the API server shuts down. That `cancelled` block is the
  worker's death notice, not a queue defect (`cancelled-executor-teardown.md`); the
  worker's own `507010` block above it is the cause, and here both ends of that chain sit
  in one file — that page's usual gap (the dying rank's log) is not a gap in this case.
- Session B, a screenshot dated 06-25 03:13-03:17 (`APIServer pid=353`): the same loss
  surfaced only as the 60 s `No available shared memory broadcast block` line repeating
  while the request never returned. Its `shm_broadcast.py:542` is **that build's** emit
  line: the reporter ran vLLM **0.14.1** (`vLLM API server version 0.14.1` in the log)
  while this case's `SOURCES.json` pins vllm v0.24.0 `dated_reference_only` (message text
  at `shm_broadcast.py:96`), so quote neither number as the other's.
- `Failed to register custom ops, all custom ops will be disabled` is printed by **both**
  workers at load and the engine served afterwards — context, like the boot-time
  `vdavinci` table miss below. The case's earlier session already had the right verdict
  and then spent three turns on that caught `ImportError`, so open a custom-op lane from
  it only when the failure itself is an op error.
- Do not quote the pasted `deploy.sh` as the launch that ran: the log's own
  `non-default args:` dict gives `max_model_len: 32768` (the default) against the
  script's `--max-model-len 8192`, and `Asynchronous scheduling is enabled.` appears with
  no `--async-scheduling` — which is why the traceback steps through
  `step_with_batch_queue`.
- `Set PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` — the deploy script exports
  `PYTORCH_NPU_ALLOC_CONF` twice, so the second value won and vllm-ascend's
  `expandable_segments` default never applied — and no `vllm_ascend` frame at the fault.
- Host `dmesg`: `receive ts exception message from device(1)` → `device(1) is not
  working` / `ts is down` → mailbox noise; `npu-smi` marked that die `Critical`. The
  boot-time `[vascend_drv] hw_get_vdavinci_type ... can not find any match device, device
  id: 0xd500` / `Failed to init vdavinci types, ret: -524` is a driver table miss printed
  before any engine started and both dies served afterwards — context, not the fault, so
  do not build the answer on it.
- The pairing was genuinely off — image base `quay.io/ascend/cann:8.5.0-310p-...`
  (`Dockerfile.310p:18` at v0.14.0rc1 `52d4acfa`, `SOC_VERSION="ascend310p1"` `:21`)
  against host driver `24.1.t52.b060` — and the
  maintainer's first reply asked for a ≥0.18.0 image and a driver/CANN check.
  **The outcome was neither**: the reporter closed it as 目前定位为硬件问题 (2026-07-02),
  i.e. rank 1 above. Offer (2) and (3) as rungs, but keep the device verdict first and
  say plainly that no vllm-ascend code path is implicated.

## Operational notes

- `HCCS_PORT` errors are intra-node NPU-to-NPU links: single-card work can pass
  while every multi-card collective in that group fails. Do not describe such a
  failure as a model or operator bug.
- For link-level detail the driver ships `hccn_tool` (the vllm-ascend Qwen3-VL
  tutorial mounts `/usr/local/Ascend/driver/tools/hccn_tool`); verify it exists
  and check its own `-h` before quoting subcommands or output format.
- Match this evidence against the failing device id from the PTA log
  (`device = N`) and the visible-device list before concluding: a fault on a
  device outside the group does not explain the failure.
- Absence of an error code is not proof of a healthy device — it narrows the
  software side of the checklist (HCCL knobs, versions, config), so say
  explicitly which side remains open.
- If `asys` is not installed on the reporter's host, fall back to `npu-smi info`
  for per-device health and to the driver/HCCL logs, and state in the answer that
  the extended error codes were not collected — do not imply the device is clean.
