# Per-device health checks (asys) and hardware fault codes

Goal: decide hardware vs software for a collective/device failure, using tools
the reporter can run themselves.

## Run the check on every device

```bash
for i in $(seq 0 7); do asys health -d=$i; done
# brief all-device summary instead of per-device detail:
asys health
```

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

## Recorded case (vllm-ascend issue #7477)

- TP=4 with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`, Qwen3.5-27B,
  vllm-ascend v0.17.0rc1, driver 25.2.0, CANN 8.5.1, torch_npu 2.9.0.
- Failure: first collective of `profile_run` died with `HcclAllreduce` /
  `hcclRet -> 4`, `AllReduceOpbasePipeline ... intraRankSize_[4]`, on
  `Device:2`; see `collective-error-chain.md`.
- `asys health -d=$i` for 0..7: devices 0 and 1 clean; devices 2,3,4,5,6,7 each
  reported `ErrorCode Num: 1` with `0x81a3880c`, string
  `node type=HCCS_PORT, sensor type=Extend Sensor, event state=Link diagnosis
  exception`, while `Overall Health` still printed `Healthy`.
- Correlation that closed the case: the failing rank's `Device:2` is inside the
  TP group and is one of the unhealthy devices. The reporter recorded the cause
  as a hardware problem and cross-checked `0x81a3880c` against the vendor health
  management fault definitions.

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
