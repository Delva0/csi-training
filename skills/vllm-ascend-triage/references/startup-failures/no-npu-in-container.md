# The container has no NPU at all: `npu-smi` itself fails (dcmi / `rtGetDeviceCount`)

Scope: "容器中获取不到NPU信息" reports — the container's `npu-smi` output plus
`collect_env.py`, or **the same driver-probe chain arriving during a serve's engine
start**: config lines and `Platform plugin ascend is activated` above it, then
`get platform info failed, drvRet=87` and `507899`, and no `Starting to load model`.
Route by the **chain** (`drvRet=87` + `507899` + `g_deviceLogicId[-1]`), not by which
command failed first — a serve-time paste is not lane B either, because nothing ever
reaches a `vllm_ascend` frame and no weight byte is loaded. Three recorded cases:
**#632** (below), **#1905** (image `quay.io/ascend/vllm-ascend:v0.9.2rc1-310p`, 300I Duo /
310P, pasted as two OCR screenshots of one log — `…/attachments/<issue>/*.ocr.txt` is
the log) and **#8809** (the vendor-image variant: `ret is -8005` from a `mis-tei`
container beside a working vllm-ascend TP8 engine, closed by an **image tag rollback** —
last section). Recorded for **#632** (image `quay.io/ascend/vllm-ascend:v0.8.4rc1` → vllm-ascend
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
| `dcmi … the device is used. ret is -8020` | the driver's own userspace (DCMI, i.e. the `libdcmi` mounted from the host) refused to initialise the device model **before any Python**. `-8020` is `DCMI_ERR_CODE_RESOURCE_OCCUPIED` (`DCMI_ERROR_CODE_BASE (-8000) - 20`) and the pasted English *is* that enum's reading: the vendor's action for it is 「设备资源已被占用，请排查设备是否被其他容器、进程等占用后再重试」, i.e. **something else holds the card** — so this code, unlike `-8005`, already names a lane (ladder rung 2) while leaving the holder unnamed. Decode it with `bash scripts/cann-error-lookup.sh -8020` **against any exported `cann/driver` in the workspace, not only the reported snapshot's** (§*cheapest classifier*): the table is a return-code ABI and the 910B and 910_93 doc sets agree row for row, so the v0.8.4/v0.9-era snapshots — which export no driver tree at all — still get the name; cite the pin you read it at |
| `halGetDeviceInfo` / `drvGetDevNum` / `rtGetDeviceCount` all `87` → `507899` | one refusal walking up the stack: device probe → device count → ACL. `507899` is the generic `[driver error:internal error]`; the **verb** names the operation, and a device-count failure before any model op means the device stack is already unusable (`../hccl-and-hardware/transport-destlink-ipc-import.md` carries the same reading for the HCCL context) |
| `get device count of NPU failed` from `device_count` | torch_npu's own wrapper around the same call — the consequence, not a second fault |
| `TBE … AscendCheckSoCVersion: cur soc version unknown` | downstream of the failed probe (SoC detection goes through the same driver). Inference, not proof — but it is *not* the `PlatformInfoInit … SocVersion <name>` line that `acl-init-500001-and-soc-version.md` asks you to read; in that lane the name **is** resolved and matches the card |
| `HCCL … CallBackInitRts g_deviceType[6] g_deviceLogicId[-1] g_devicePhyId[-1]` | the same refusal seen from HCCL's `adapter_rts`: `-1` logical **and** physical id means no card was bound to the process. Recorded #248 (whose own trigger, quoted from its body, is one card's `/dev` node mapped into two containers) and #1905 |
| `DVPP … PlatformInfo.cpp:37 GetHardwareVersion … get device num fail, ret = 507899` | a second *consumer* of the failed count (DVPP's platform info), not a second fault — it carries no new code |
| `DrvMngGetConsoleLogLevel failed (g_conLogLevel=3)` | a console-log-level getter printing the level it could not set; it carries no device identity and no error code path. Not a lead (that the getter is harmless is read from the message; the driver body is not in the snapshot) |
| `Platform plugin ascend is activated` | vllm-ascend's plugin loaded fine: nothing in this paste implicates the wheel, the model or a serve flag |

The whole failure therefore sits **below** vllm-ascend: DCMI + the container's
device nodes, or the host driver state. Say that first; then rank.

## The `-8xxx` code is the cheapest classifier — read it before ranking, from *any* exported `cann/driver`

At `cann/driver` `src/custom/include/dcmi_interface_api.h:1677-1698` (verified at pin
`a7ceccf700e0`; the vendor rows below are that tree's
`docs/zh/Ascend_910B_DCMI_API/<…>/return_codes.md` and the 910_93 set, identical row for
row) the DCMI return codes are `DCMI_ERROR_CODE_BASE (-8000)` minus an offset, so the
offset is the whole lookup — no table memory is needed for the common ones.

**Decode the number from whichever snapshot has the driver tree, not the reported one.**
A return-code table is an ABI: it does not move between releases, and the reports that
need it most are exactly the ones whose snapshot exports no `cann/driver`
(v0.8.4/v0.9-era, e.g. #632 — a `SOURCES.json` row of
`cann/driver no_verified_public_reference_by_opening` is *not* a reason to hedge the
name, only the holder). Run one call against the whole workspace
(`bash scripts/cann-error-lookup.sh -8020`, default root `/workspace/repos`) and cite the
pin you read it at.

The code then picks the rung: `-8008` → mapping (rung 1), `-8020` → another holder
(rung 2), `-8002` → a non-root `-u`, `-8013` → the interface is container-limited,
`-8012`/`-8017`/`-8006` → a device-state wait, `-8007` → the named `davinciN` is not on
this host, `-8005`/`-8009`/`-8004` → no lane at all. The rows that change a ranking:

| Code | Enum | What it rules in / out |
|---|---|---|
| -8020 | `DCMI_ERR_CODE_RESOURCE_OCCUPIED` (vendor: 设备资源被占用；处理：请排查设备是否被其他容器、进程等占用后再重试) | **another holder** — for a vllm-ascend container this is the code that promotes rung 2 to first, and the reporter's own text (`because the device is used`) is its translation. The vendor files it under 内部执行异常, so it names the lane but not the holder: a host process, a second container, a vNPU partition, or a device-side process the `dcmi_set_traceroute` doc says needs `param_info.reset_flag=1` to end |
| -8002 | `DCMI_ERR_CODE_OPER_NOT_PERMITTED` | a **permission** refusal — so a non-root `-u <user>` run would print this, not -8005 |
| -8005 | `DCMI_ERR_CODE_INNER_ERR` (vendor row: 内部错误，请提供日志) | the generic internal error: it excludes nothing, so it cannot carry a mechanism on its own |
| -8008 | `DCMI_ERR_CODE_DEVICE_NOT_EXIST` | the node the container maps is not there (`--device` / mount list) |
| -8009 | `DCMI_ERR_CODE_IOCTL_FAIL` | the ioctl into the driver failed |
| -8013 | `DCMI_ERR_CODE_NOT_SUPPORT_IN_CONTAINER` (vendor: 在容器中不支持该接口) | container detection is an **explicit** DCMI contract, so "in a container" is a first-class lane, not a modifier |
| -8007 | `DCMI_ERR_CODE_INVALID_DEVICE_ID` (vendor: 无效的device ID) | the `--device /dev/davinciN` names a card this host does not have — a mapping error, the mirror of -8008 |
| -8006 | `DCMI_ERR_CODE_TIME_OUT` (vendor: 响应超时，请稍后重试) | a transient: retry the same command before ranking anything |
| -8012 | `DCMI_ERR_CODE_NOT_REDAY` (vendor: 设备尚未就绪，请稍后重试) | a **wait**, not a fix: the device is still coming up (just after a reset or boot) |
| -8017 | `DCMI_ERR_CODE_IS_UPGRADING` (vendor: 设备正在升级中，请稍后再试) | a firmware upgrade is in flight — no container or mount change helps |
| -8022 | `DCMI_ERR_CODE_PARTITION_NOT_RIGHT` (enum only: no vendor row at that pin) | the vNPU partition config is wrong — check `npu-smi set -t create-vnpu …`/the partitioning mode before any container theory |

Two traps this table settles in one call: `src/custom/lqdrv/kernel/ioctl_comm_def.h`
carries a **separate** `LQ_DCMI_ERR_CODE_*` numbering where `-8005` *is* `IOCTL_FAIL`
and `-8008` *is* `VERSION_IMCOMPATIBLE` — never cross-apply the two (the recorded
#8809 run quoted the lqdrv row, shipped it, then had to correct itself mid-answer);
and no row of the DCMI table is an HBM or capacity code — `-8020`'s 设备资源 is the
*device*, not HBM — so "显存被 vllm 吃满" is refuted by the number itself.

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
   see §*A responding `npu-smi info` is not a mount check* above. Inference
   to keep the ranking straight: `get platform info` / `drvGetDevNum` refuse **before**
   any `/dev` node is opened, so a `/dev` list shorter than the tutorial's cannot be the
   owner of a *count* failure — when the host's own `npu-smi` is healthy, go to rung 3
   (host vs container user-space driver) before re-mapping devices.
2. **Another holder of the same device — and a `-8020` puts this rung first for a
   vllm-ascend container, while co-tenancy is only a lane for a *second* one.** The
   driver's own action text for `-8020` (「请排查设备是否被其他容器、进程等占用」) is the
   vendor telling the reporter to check exactly this, so say it with the code attached.
   `docker ps` for a second container on those devices and
   the host's process table (`npu-smi info`, `fuser -v /dev/davinci0
   /dev/davinci_manager`); the cheapest falsification is a different, idle
   `--device /dev/davinciN` — one variable. Do **not** rank it first when the failing
   container is a *vendor serving image* (mis-tei / MindIE TEI, an inference-engine
   image that bundles its own CANN userspace): in recorded #8809 the confirmed working
   bge-m3 `mis-tei` command runs on `--device /dev/davinci4` **while the vllm container
   holds `davinci0-7` with TP8**, so a chip shared by two containers is not by itself a
   DCMI blocker at this stack; the reporter attributes the cure to the image tag (ladder rung 3).
   **Before ranking a co-tenant, read what the failing container was *created*
   with.** A `docker run` assembled from the vllm-ascend install block copies the
   `--device` lines and never the Ascend Docker Runtime's own input: when
   `ASCEND_VISIBLE_DEVICES` is absent from the creation env the prestart hook mounts
   nothing and writes no device-cgroup entry (`checkVisibleDevice` returns empty,
   `Ascend/mind-cluster` `component/ascend-docker-runtime/runtime/process/process.go:545-548`),
   so the container's device set is only what `--device`/`--privileged` gave it. One
   call separates the two readings: `docker inspect -f '{{.Config.Env}}' <container>`
   (creation env) against the failing shell's own `export`. Reporter-confirmed on
   **#6942**: `export ASCEND_RT_VISIBLE_DEVICES=6,7` alone and `export
   ASCEND_VISIBLE_DEVICES=6,7` alone both failed, `docker run -e
   ASCEND_VISIBLE_DEVICES=6,7` plus the in-container RT export fixed it — the paste
   itself (`Waiting for 1 local, 0 remote core engine proc(s) to start.`, so the
   child is alive and stuck) is equally consistent with HBM co-tenancy, which is why
   the env half is checked first: it is free and it was the actual cure
   (`../deploy-sizing/card-budget-and-device-visibility.md`).
3. **Host vs container user-space driver — for a vendor serving image this is rung 1,
   and the lever is the image tag.** The image mounts the host's `dcmi`, `npu-smi`,
   `driver/lib64/` and `version.info`, so read
   `cat /usr/local/Ascend/driver/version.info` and the `npu-smi info` header **on both
   sides**: only a same-origin set is self-consistent. For a **vllm-ascend** image the
   CANN half is fixed by its base (`Dockerfile:18` →
   `quay.io/ascend/cann:8.0.0-910b-ubuntu22.04-py3.10` at this ref, matching the
   reporter's `install.info`). For a **third-party image** (mis-tei / MindIE TEI,
   `swr…/ascendhub/mis-tei:<tag>`) nothing in this snapshot defines its userspace
   contract, so the tag *is* the variable: its version is the Ascend release line it was
   built against (`mis-tei:7.2.RC1-800I-A2-aarch64` ↔ a host at the 7.2.RC1 line, which
   `SOURCES.json`'s `Ascend/mind-cluster v7.2.RC1.SPC2` row reflects for that snapshot;
   a `26.0.0-800I-A2` tag is the MindStudio-26 line). Run the **identical** `docker run`
   with the older tag before attributing anything to cards, users or mounts —
   reporter-confirmed on #8809 (same host, same coexisting TP8 vllm engine; keep the
   failing command's own `--device /dev/davinci0` so the tag is the only variable
   changed, which is what the reporter's own pair does *not* do — see the recorded run).
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
inside the container). State the unverified part explicitly: `87` is the driver's own
refusal walking up the stack; `-8020` = `DCMI_ERR_CODE_RESOURCE_OCCUPIED` is the one name
in this paste that *does* select a lane (someone holds the card — say so and quote the
vendor's action text), while `-8005`/`-8009` exclude nothing; and no code on the chain
names *which* holder, so "occupied — by whom" is what the host's process table answers,
not the paste.

## Recorded run (#632)

The pasted `docker run` is the pinned quick-start block verbatim (v0.8.4rc1
`docs/source/quick_start.md:18-34`, including `-p 8000:8000`), which removed the
doc-typo/missing-mount hypothesis for that report; the session confirmed the image's
CANN base from `Dockerfile:18`, ruled out the TBE SoC line as a cause, searched the
prior-month corpus for `-8020`/`dcmi` (12 body hits, none with this text) and
answered with the ranked ladder above. It spent its first passes in
`acl-init-500001-and-soc-version.md` and the HCCL IPC page, both of which are the
*later* stages — enter here instead when the paste's first failure is `npu-smi`.

**Correction a later re-run forces.** That run hedged the code (「`-8020` 的枚举名与精确语义
我只能引用这句文案，不能替你断定是占用」) because the classifier used to say *decode it
whenever `cann/driver` is in this snapshot's `SOURCES.json`* — and this snapshot has none.
The row above now carries the answer, read from the #8809-era export: the name is
`RESOURCE_OCCUPIED` and the vendor's action is the co-tenancy check, so the same evidence
supports rung 2 first with the code attached rather than a hypothesis. What the run got
right and keeps: chain-first routing (`drvRet=87` + `507899` + `g_deviceLogicId[-1]`, not
"which command failed first"), the verbatim-quickstart match as the mount-list
exoneration, the `--privileged` rung placed *below* the driver readings, and the explicit
statement that the paste names no holder.

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

## Recorded run (#8809, and the correction its follow-up forces)

Paste: a 910B4×8 Atlas 800I A2 host whose **vllm-ascend `v0.18.0rc1` (`99e1ea0fe685`)
container running Qwen3.6-35B-A3B TP8 works and lists all 8 cards**, while a
`swr…/ascendhub/mis-tei:26.0.0-800I-A2-aarch64` container serving bge-m3 (that command:
`-u HwHiAiUser`, `--privileged`, `-v /dev:/dev`, `--device /dev/davinci0`) prints
`DrvMngGetConsoleLogLevel failed. (ret=4)` + `dcmi module initialize failed. ret is
-8005` and enumerates nothing. The reporter's framing ("只能用
完全空一张 910B4") reads as memory pressure.

The run answered it in-lane and rank-ordered **same-card co-tenancy first**, non-root
`-u <user>` and a `-v /dev:/dev` + `--device` mix next, and **image/driver pairing
third**, citing `#248` (one card's `/dev` node mapped into two containers → the
`-8020`/`87`/`507899` chain), `#632` (same chain) and the DCMI enum (in that snapshot
`cann/driver` *was* exported, so `-8005 = DCMI_ERR_CODE_INNER_ERR` was decoded
correctly — but only after quoting the **lqdrv** `-8005 = IOCTL_FAIL` row first and
posting a mid-answer correction, which is what the two-table warning above exists to
prevent).

**The reporter's 2026-04-29 follow-up closes it on the image side**: `问题在mis-tei侧，
镜像回退到 mis-tei:7.2.RC1-800I-A2-aarch64 后解决`, and the working command keeps
`--device /dev/davinci4` for the TEI container **while the vllm container still maps
`davinci0-7` with `--tensor-parallel-size 8`**. So:

- one chip mapped into two containers was **not** the blocker → ladder rung 2 does not lead for
  a vendor serving image (§2), and "一张卡只能进一个容器" must not be published as the
  shape;
- the image tag was, per the reporter. What also differs between the two commands is why
  this is an *attribution*, not a single-variable result: `-u root` instead of
  `HwHiAiUser`, no `-v /dev:/dev`, `--privileged=true`, `ENABLE_BOOST=True`,
  `ASCEND_VISIBLE_DEVICES=4` / `TEI_NPU_DEVICE=4`, and the mapped card
  (`/dev/davinci0` → `/dev/davinci4`, i.e. the "different idle `--device`" rung was also
  moved). Say "reporter-confirmed on the image tag", never "the driver's userspace was
  mismatched";
- what the run got right and should keep: the lane (below every vllm-ascend frame, no
  engine or serve flag implicated), the `-8005` reading (a generic internal error, not a
  memory code), and the ask (`npu-smi info` on the host with its process table,
  `driver/version.info` from both sides, `/dev/davinci*` inside the TEI container, and
  the vendor image's **original** `docker run` — the pasted one had a duplicated
  `-v /data/models/` and an unreplaced `/[HOME_DIRECTORY_1]/` placeholder, i.e. hand-edited).
