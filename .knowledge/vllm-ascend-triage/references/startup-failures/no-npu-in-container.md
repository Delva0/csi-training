# The container has no NPU at all: `npu-smi` itself fails (dcmi / `rtGetDeviceCount`)

Scope: "容器中获取不到NPU信息" reports — the container's `npu-smi` output plus
`collect_env.py`, or **the same driver-probe chain arriving during a serve's engine
start**: config lines and `Platform plugin ascend is activated` above it, then
`get platform info failed, drvRet=87` and `507899`, and no `Starting to load model`.
Route by the **chain** (`drvRet=87` + `507899` + `g_deviceLogicId[-1]`), not by which
command failed first — a serve-time paste is not lane B either, because nothing ever
reaches a `vllm_ascend` frame and no weight byte is loaded. Two recorded cases: **#632**
(below) and **#1905** (image `quay.io/ascend/vllm-ascend:v0.9.2rc1-310p`, 300I Duo /
310P, pasted as two OCR screenshots of one log — `…/attachments/<issue>/*.ocr.txt` is
the log). Recorded for **#632** (image `quay.io/ascend/vllm-ascend:v0.8.4rc1` → vllm-ascend
`086423dc352e`, vllm 0.8.4; CANN `8.0.0` `V100R001C20SPC001B251` aarch64, torch
2.5.1, torch-npu `2.5.1.dev20250320`, 910 host). The issue carries no resolution in
the snapshot: every claim below is about what the pasted text means, not about the
reporter's card.

This is the stage **before** `acl-init-500001-and-soc-version.md` (which needs
torch_npu to have opened a device) and before
`device-id-invalid-at-worker-init.md` (which needs an `107001` from an ACL open).

## The signature set, and what each line does and does not prove

```text
DrvMngGetConsoleLogLevel failed. (g_conLogLevel=3)
dcmi model initialized failed, because the device is used. ret is -8020
[ERROR] ATRACE … get platform info failed, drvErr=87.
[ERROR] RUNTIME … CheckHaveDevice:Call halGetDeviceInfo failed: drvRet=87, module type=0, info type=1.
[ERROR] RUNTIME … GetDeviceCount:Call drvGetDevNum, drvRetCode=87.
[ERROR] RUNTIME … rtGetDeviceCount:ErrCode=507899, desc=[driver error:internal error], InnerCode=0x7020010
[ERROR] ASCENDCL … aclrtGetDeviceCount: get device count failed, runtime result = 507899.
[ERROR] APP … device_count:25: "[PTA]:"get device count of NPU failed""
[ERROR] TBE … AscendCheckSoCVersion:cur soc version unknowsoctype not found.
[INFO] HCCL … [adapter_rts.cc][CallBackInitRts] g_deviceType [6] g_deviceLogicId [-1] g_devicePhyId [-1]
[ERROR] DVPP … [PlatformInfo.cpp:37] [GetHardwareVersion] get device num fail, ret = 507899
INFO … Platform plugin ascend is activated
```

The two middle lines are how the chain looks from the other consumers: HCCL's adapter
prints the unbound ids, DVPP re-uses the same count call. Both belong to this page.

| Line | Reading |
|---|---|
| `dcmi … the device is used. ret is -8020` | the driver's own userspace (DCMI, i.e. the `libdcmi` mounted from the host) refused to initialise the device model **before any Python**. Quote it as text; the numeric meaning is in `cann/driver`, which is **not** in this snapshot's `SOURCES.json` (`unavailable`), so do not publish a decoded enum |
| `halGetDeviceInfo` / `drvGetDevNum` / `rtGetDeviceCount` all `87` → `507899` | one refusal walking up the stack: device probe → device count → ACL. `507899` is the generic `[driver error:internal error]`; the **verb** names the operation, and a device-count failure before any model op means the device stack is already unusable (`../hccl-and-hardware/transport-destlink-ipc-import.md` carries the same reading for the HCCL context) |
| `get device count of NPU failed` from `device_count` | torch_npu's own wrapper around the same call — the consequence, not a second fault |
| `TBE … AscendCheckSoCVersion: cur soc version unknown` | downstream of the failed probe (SoC detection goes through the same driver). Inference, not proof — but it is *not* the `PlatformInfoInit … SocVersion <name>` line that `acl-init-500001-and-soc-version.md` asks you to read; in that lane the name **is** resolved and matches the card |
| `HCCL … CallBackInitRts g_deviceType[6] g_deviceLogicId[-1] g_devicePhyId[-1]` | the same refusal seen from HCCL's `adapter_rts`: `-1` logical **and** physical id means no card was bound to the process. Recorded #248 (whose own trigger, quoted from its body, is one card's `/dev` node mapped into two containers) and #1905 |
| `DVPP … PlatformInfo.cpp:37 GetHardwareVersion … get device num fail, ret = 507899` | a second *consumer* of the failed count (DVPP's platform info), not a second fault — it carries no new code |
| `DrvMngGetConsoleLogLevel failed (g_conLogLevel=3)` | a console-log-level getter printing the level it could not set; it carries no device identity and no error code path. Not a lead (that the getter is harmless is read from the message; the driver body is not in the snapshot) |
| `Platform plugin ascend is activated` | vllm-ascend's plugin loaded fine: nothing in this paste implicates the wheel, the model or a serve flag |

The whole failure therefore sits **below** vllm-ascend: DCMI + the container's
device nodes, or the host driver state. Say that first; then rank.

## The one command that halves the search

Run the **same** `npu-smi info` on the host, outside the container:

- host lists the cards, container does not → the host stack is up and the
  *container's* view is the owner (rungs 1-3 below);
- host prints the same `-8020` / `drvRet=87` → the card or the host driver is the
  owner and the image is not implicated (rung 4).

Ask for the host output verbatim (header versions + process table), because it also
supplies the driver version the version-mismatch rung needs.

## A responding `npu-smi info` is not a mount check (recorded #4847)

This page's lane normally shows the probe *failing*. The inverse is a finding too: the
container answers both `npu-smi info` commands, the engine starts, reads config and
tokenizer, and only then dies — recorded #4847 reported a **weight-load** failure on a
serve (image `v0.11.0rc2`, pin `a2e4c3f`) whose pasted `docker run` carried hccn_tool,
`npu-smi`, driver `lib64/`, `version.info`, `ascend_install.info` and `.cache` but **not**
`-v /usr/local/dcmi:/usr/local/dcmi`, which is line `:30` (and `:59`) of that release's
own `docs/source/quick_start.md`. Adding the dcmi mount made weight loading work
(reporter-confirmed, one variable); the start then moved on to an unrelated torchair
profiling crash (`torchair-graph-profile-run-kv-cache-indexerror.md`). Recorded #302's
`--privileged=true` rung below is the same lesson one layer up: the documented `docker
run` is a *starting point*, and the reporter's copy is where the diff is read.

Two consequences for the ranking:

- `npu-smi info` exercises the `npu-smi` binary plus the driver libraries that *are*
  mounted; it does not exercise every userspace path the engine's later phases take. A
  green `npu-smi` therefore does not clear the mount list: diff the paste's `docker run`
  against the pin's `quick_start.md` (or `installation.md`) block line by line and treat a
  missing `dcmi` line as a finding by itself.
- The phase such an omission surfaces in is not diagnostic — weight load here, a
  device-count probe in the case above — so do not wait for `drvRet=87` before checking
  the mounts.

Mechanism is not verified: no exported source says which library call the weight-load
path makes through `/usr/local/dcmi`, so quote the reporter's outcome and the doc diff,
not a causal chain.

## Ladder, cheapest first

1. **What the container actually maps.** `ls -l /dev/davinci* /dev/davinci_manager
   /dev/devmm_svm /dev/hisi_hdc`, then compare the `docker run` with the pinned
   `docs/source/quick_start.md:18-34` block (at v0.8.4rc1 it is `export
   DEVICE=/dev/davinci0`, the four `--device` lines, `dcmi` + `npu-smi` + driver
   `lib64/` + `version.info` + `/etc/ascend_install.info`, `-p 8000:8000`). A paste
   that matches character for character (the recorded case below) exonerates "a missing mount"
   but the list is a **minimum**: it carries no `--privileged`/`--cap-add`. A paste that
   **omits** a line is a finding on its own even when the container answers `npu-smi` —
   see *A responding `npu-smi info` is not a mount check* below. Inference
   to keep the ranking straight: `get platform info` / `drvGetDevNum` refuse **before**
   any `/dev` node is opened, so a `/dev` list shorter than the tutorial's cannot be the
   owner of a *count* failure — when the host's own `npu-smi` is healthy, go to rung 3
   (host vs container user-space driver) before re-mapping devices.
2. **Another holder of the same device.** `docker ps` for a second container on those
   devices and the host's process table (`npu-smi info`, `fuser -v /dev/davinci0
   /dev/davinci_manager`). A card that is busy is a container/host-state finding, and
   the cheapest falsification is a different, idle `--device /dev/davinciN` — one
   variable.
3. **Host vs container user-space driver.** The image mounts the host's `dcmi`,
   `npu-smi`, `driver/lib64/` and `version.info`, so read
   `cat /usr/local/Ascend/driver/version.info` and the `npu-smi info` header **on both
   sides**: only a same-origin set is self-consistent. This is the pairing rung, and
   the image's own CANN half is fixed by its base (`Dockerfile:18` →
   `quay.io/ascend/cann:8.0.0-910b-ubuntu22.04-py3.10` at this ref, matching the
   reporter's `install.info`).
4. **Host driver / card state.** `dmesg | grep -iE 'davinci|devdrv'`, then the
   vendor-side levers (`npu-smi set -t reset -i N`, host reboot) only when no user
   process holds the card — and hand a shared machine to its administrator rather
   than guessing.
5. **`--privileged=true` on the otherwise unchanged command** is a rung, not the
   mechanism: #302's reporter-confirmed fix is one flag on the *same* command, but
   #302's text is `SetPrecisionMode … 500001` at `torch.npu.set_device` — a later
   stage, after the device was visible. Cite it as the container-permission
   precedent and keep it from being read as an explanation of `-8020`.

## Deliver

Headline: this is a device-visibility/driver report, not a vllm-ascend bug — nothing
in the paste reaches an engine. Then the evidence table above, the host-vs-container
split as the next single action, the ladder, and the ask (host `npu-smi info` with its
header and process table, `driver/version.info` from both sides, `ls -l /dev/davinci*`
inside the container). State the unverified part explicitly: the numeric meanings of
`-8020` and `87` are not in any pinned tree here, and whether *this* reporter's card
was occupied, mis-mounted or driven by a mismatched userspace is not derivable from
the paste.

## Recorded run (#632)

The pasted `docker run` is the pinned quick-start block verbatim (v0.8.4rc1
`docs/source/quick_start.md:18-34`, including `-p 8000:8000`), which removed the
doc-typo/missing-mount hypothesis for that report; the session confirmed the image's
CANN base from `Dockerfile:18`, ruled out the TBE SoC line as a cause, searched the
prior-month corpus for `-8020`/`dcmi` (12 body hits, none with this text) and
answered with the ranked ladder above. It spent its first passes in
`acl-init-500001-and-soc-version.md` and the HCCL IPC page, both of which are the
*later* stages — enter here instead when the paste's first failure is `npu-smi`.

## Recorded run (#1905)

Two OCR screenshots of one 310P serve log (`attachments/1905/*.ocr.txt`, identical
bytes for both). The session pinned v0.9.2rc1 (`b5b7e0ec`) / vllm v0.9.2
(`a5dd03c1`) by matching three pasted line numbers (`platform.py:161`,
`vllm/config.py:1472`, `:2285`), classified it as the device probe **before** weight
load, matched it to this page plus the `#248` HCCL-id line, ranked host-driver ×
image-CANN pairing first, card occupation second, container mapping and
`--privileged=true` third (citing #302 as the permission precedent at a *later*
stage), and asked for host + container `npu-smi info`, both `driver/version.info`
and `ls -l /dev/davinci*`. The reporter's public follow-up (2025-07-22) is just
「已解决」 with **no mechanism**, so nothing here may be published as "what fixed it";
the ladder stands on the driver/user-space reading alone. It also confirmed Qwen3-8B
single-die is not in the pin's 300I validated set (`single_node_300i.md:50`) while the
lane itself is labelled experimental (`:3-5`) — a separate, model-side gap.
