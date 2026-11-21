# The container has no NPU at all: `npu-smi` itself fails (dcmi / `rtGetDeviceCount`)

Scope: "容器中获取不到NPU信息" reports whose **first** failing command is `npu-smi
info`, not a serve command — the paste is the container's `npu-smi` output plus
`collect_env.py`, there is no `EngineCore`, no worker and no `vllm_ascend` frame.
Recorded for **#632** (image `quay.io/ascend/vllm-ascend:v0.8.4rc1` → vllm-ascend
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
INFO … Platform plugin ascend is activated
```

| Line | Reading |
|---|---|
| `dcmi … the device is used. ret is -8020` | the driver's own userspace (DCMI, i.e. the `libdcmi` mounted from the host) refused to initialise the device model **before any Python**. Quote it as text; the numeric meaning is in `cann/driver`, which is **not** in this snapshot's `SOURCES.json` (`unavailable`), so do not publish a decoded enum |
| `halGetDeviceInfo` / `drvGetDevNum` / `rtGetDeviceCount` all `87` → `507899` | one refusal walking up the stack: device probe → device count → ACL. `507899` is the generic `[driver error:internal error]`; the **verb** names the operation, and a device-count failure before any model op means the device stack is already unusable (`../hccl-and-hardware/transport-destlink-ipc-import.md` carries the same reading for the HCCL context) |
| `get device count of NPU failed` from `device_count` | torch_npu's own wrapper around the same call — the consequence, not a second fault |
| `TBE … AscendCheckSoCVersion: cur soc version unknown` | downstream of the failed probe (SoC detection goes through the same driver). Inference, not proof — but it is *not* the `PlatformInfoInit … SocVersion <name>` line that `acl-init-500001-and-soc-version.md` asks you to read; in that lane the name **is** resolved and matches the card |
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

## Ladder, cheapest first

1. **What the container actually maps.** `ls -l /dev/davinci* /dev/davinci_manager
   /dev/devmm_svm /dev/hisi_hdc`, then compare the `docker run` with the pinned
   `docs/source/quick_start.md:18-34` block (at v0.8.4rc1 it is `export
   DEVICE=/dev/davinci0`, the four `--device` lines, `dcmi` + `npu-smi` + driver
   `lib64/` + `version.info` + `/etc/ascend_install.info`, `-p 8000:8000`). A paste
   that matches character for character (the recorded case below) exonerates "a missing mount"
   but the list is a **minimum**: it carries no `--privileged`/`--cap-add`.
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
