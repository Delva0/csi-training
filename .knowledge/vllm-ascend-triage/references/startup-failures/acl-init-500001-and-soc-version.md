# `SetPrecisionMode … ACL_PRECISION_MODE … 500001`: ACL/GE bring-up, not vllm-ascend

## Scope

`EngineCore failed to start` / `WorkerProc failed to start` whose deepest frames are
`torch_npu` device open, **before** any `vllm_ascend` frame and before
`Starting to load model`:

```text
  File ".../torch_npu/npu/utils.py", line 80, in set_device
    torch.npu._C.npu_setDevice(device_id)
RuntimeError: SetPrecisionMode:.../torch_npu.dir/compiler/...:aclrtSetCompileOpt(
  aclCompileOpt::ACL_PRECISION_MODE, precision_mode), error code is 500001
  ERROR ... NPUFunctions.cpp:216] aclrtSetCompileOpt failed,ret:500001
  [Error]: The internal ACL of the system is incorrect.
  EH9999: [pid: 357] ... [Inner] [PLATFORM]PlatformInfoInit runtime platform info failed.
    SocVersion Ascend@910B[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
  GELIB: [INNER]Initialize failed. / GEInitialize failed / [Init][Compiler]Init compiler failed
  ... Engine core initialization failed.        [ERROR] ... ERR99999 UNKNOWN application exception
```

Recorded as **#2315** (image `v0.9.2rc1-310p`, host `npu-smi` 24.1.0.1, two `310P3`
chips, Qwen3-32B TP2). The same `AclSetCompileopt(ACL_PRECISION_MODE …) error code
is 500001` plus `PlatformInfoInit … SocVersion` pair recurs in **#302**, **#564**
and **#692** on **910B2** hosts — so the shape is an environment family, not a 310P
or a Qwen3 one.

**#302 is the only case here with a reporter-confirmed resolution, and the fix is a
container flag.** Image `quay.io/ascend/vllm-ascend:v0.7.1rc1` (`Dockerfile:17` → CANN
`8.0.0-910b` base), 910B2 ×8, driver 24.1.rc3, `vllm serve Qwen2.5-7B-Instruct
--max_model_len 26240`; the `docker run` carries exactly the documented device/mount
list (`docs/source/quick_start.md:15-30` at that pin — `davinci0` + `davinci_manager` +
`devmm_svm` + `hisi_hdc`, `dcmi`, `npu-smi`, the driver's `lib64/` + `version.info`,
`/etc/ascend_install.info`), and the printed `SocVersion = Ascend910B2` matches the card.
The reporter closed it himself the next day: 「已解决，加个--privileged=true即可」 — that
one flag on the otherwise unchanged command, with no driver, CANN or image change
reported (#302 comment 2716169193, 2025-03-12).

## What the two log facts mean

- `SetPrecisionMode` is torch_npu's own init step; no vLLM flag reaches it. It
  derives the string from the SoC it believes it is on — `GetSocVersion() >=
  SocVersion::Ascend910B1 ? "must_keep_origin_dtype" : "allow_fp32_to_fp16"` — and
  hands it to `AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, …)`. Both are
  **parameters of an init that already failed**: `500001` is ACL's internal error,
  so no `--dtype`, `--compilation-config`, `--enforce-eager` or precision knob the
  reporter passes can cause or repair it. Do not rank them.
- **Quote the emitting site at the report's pin: the file moves between release
  lines, and a path/line borrowed from another export is a wrong claim about the
  reporter's image.** A recorded replay of #302 answered a `v0.7.1rc1` paste with
  `LazyInitAclops.cpp:150-157`, which is not that snapshot's file at all. The three
  steps, per `Ascend/pytorch` export:

  | pin | where the init lives |
  |---|---|
  | `v6.0.0.alpha001-pytorch2.5.1` (`a3b75fc0a6d6`; the export behind the `v0.7.1rc1` / #302 image) | `torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp` — `:234` `AclGetSocName()` → `:236` `SetSocVersion(soc_name)`, `:245-246` the `Ascend910B1` ternary, `:247` `NPU_CHECK_ERROR(at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, precision_mode))`, `:252` `GetAndSetDefaultJitCompileByAcl()` (`:81`). `rg -l 'LazyInitAclops' <that export>` → nothing, and the sequence there is complete; the export is `1133/1134` files, so what is verified is the sequence, not the later file's non-existence in the release |
  | `v7.1.RC1.alpha002-pytorch2.7.1` (`d33df559`) | the same steps in `torch_npu/csrc/framework/LazyInitAclops.cpp` (`:150-157`, call at `:156`) |
  | `v7.2.0.1-pytorch2.8.0` (`da848bbc39f4`) | same file, shifted: `:167-175`, the `AclSetCompileopt` at `:175` — so between these two, quote the function, not the number |

  The `SocVersion <name>` in the `PlatformInfoInit` line is **what this same init
  resolved for the device it opened** (`:234-236` at the 2.5.1 pin,
  `npu_sys_ctrl.cpp:157-159` in the 2.7.1 export), and it is the first thing to read.
- **At the v0.7.1-era pin the code is the call's own status.** `NPU_CHECK_ERROR` is
  `NPU_CHECK_ERROR_CHECK_UCE`, which binds `auto Error = err_code;` and prints it
  (`torch_npu/csrc/core/npu/NPUException.h:146-168`), and `AclrtPeekAtLastError`
  occurs nowhere in that export — so the thread-latch masking of
  `../hccl-and-hardware/latched-runtime-error-masking.md` §1 does not reach a
  v0.7.x paste and `500001` needs no second-guessing. The two lines under
  `ERR00100 PTA call acl api failed` are that same macro's table lookup for the code —
  `NPUErrorCodes.h:194-195` holds exactly one entry for `500001`,
  "The internal ACL of the system is incorrect." plus "Rectify the fault based on
  the error information in the ascend log.", so they are one report, not three.

## Read the SoC line against the card

| `npu-smi info` chip | `PlatformInfoInit … SocVersion` | Reading |
|---|---|---|
| `310P3` | `Ascend@910B…` (or any name the card cannot report) | the runtime did not bind to this card: the mounted host driver/firmware, or a toolkit/opp inside the container that is not the card's line. Name the conflict before anything else |
| matches the card | `500001` still | the container lane first: re-run the **same** `docker run` with `--privileged=true` added and nothing else changed — the reporter-confirmed fix on #302 (910B2, documented mounts, SoC matching). Only if that clears nothing: host driver/firmware older than the image's CANN base, or a device unusable inside the container. A matching line exonerates nothing, and the flag does not say *which* device or capability was short |

## Sibling variant: the same phase without `500001` is a FILE question

If the paste carries `RealPath:Path '<…>/<arch>-linux/lib64/../data/platform_config' does not
exist.` + `InitializePlatformInfo:File path[] is not valid.` and a final
`RuntimeError: Get ACL JitCompile default value failed.` — with the SoC name resolved and **no**
`SetPrecisionMode … 500001` — this page's lane ends: the failure is a missing directory/path in
the container's CANN tree, the mechanism is readable at
`cann/runtime/src/platform/platform_info.cpp` (exported, unlike `cann/ge`), and neither
`--privileged` nor the driver pairing is the lever →
`platform-config-file-missing.md`. Two reads, and the `lib64/…` prefix in the message is the FE
`.so`'s own real directory.

## Containerized run: one flag, one variable, before any pairing theory

The documented `docker run` (`docs/source/quick_start.md:15-30` at v0.7.1rc1,
`docs/source/installation.md:53-65` at the later pins) is a **minimum** device/mount
list — four `/dev` nodes, the driver's `lib64/` + `version.info`, `dcmi`, `npu-smi`,
`/etc/ascend_install.info` — and it passes no `--privileged` and no `--cap-add`. It is
not a sufficiency proof: a sibling Ascend deployment doc asks for more from its first
line (`LMCache-Ascend v0.3.3 docs/deployment.md:17-19`: `--privileged
--cap-add=SYS_RESOURCE --cap-add=IPC_LOCK`, read in the same snapshot). So when a paste
like #302 dies at device open inside a container whose run line matches the docs, the
cheap discriminating ablation is the failing command plus **one** flag:

```bash
# the reporter's own line (mounts abbreviated) with `--privileged=true` added, nothing else changed
docker run --privileged=true --name vllm-ascend --device /dev/davinci0 ... \
  quay.io/ascend/vllm-ascend:v0.7.1rc1 vllm serve Qwen/Qwen2.5-7B-Instruct --max_model_len 26240
```

Reporter-confirmed on #302 with the driver unchanged. Do **not** bundle a second change
into that re-run — a whole-driver mount (`-v /usr/local/Ascend/driver:/usr/local/Ascend/driver`),
a driver upgrade, a different image tag: two variables, and the case that closed needed
only the flag. Widening privilege is not a mechanism either, so keep the pairing lane
open: if the flag changes nothing, run the checks below; if it fixes the case, the
container's device/capability set was short, and narrowing it afterwards (`--cap-add`,
per-device `--device`) is a follow-up experiment, not this diagnosis.

## Checks, in this order (all on the reporter's host)

1. **The `--privileged=true` re-run above**, when the deployment is a container whose
   `docker run` matches the docs (#302). Then the framework-free probe.
2. **Framework-free probe in the same container**, no vLLM:
   `python -c "import torch, torch_npu; torch.npu.set_device(0); print(torch.npu.get_device_name(0))"`.
   The same `500001` there ends the vllm-ascend part of the case.
3. The version set plus the image tag: `cat /usr/local/Ascend/driver/version.info`,
   `npu-smi info` (header version + chip), `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`,
   `pip show torch torch-npu vllm vllm-ascend`. The container's CANN is the
   **image's** — `docs/source/installation.md:53-65` mounts only the driver's
   `lib64/` + `version.info`, `dcmi`, `npu-smi` and `/etc/ascend_install.info`,
   never `ascend-toolkit/`. What the host contributes is driver/firmware, and that
   is the pairing under test.
4. Compare against the image's own base: at v0.9.2rc1 `Dockerfile.310p:18` is
   `FROM quay.io/ascend/cann:8.1.rc1-310p-ubuntu22.04-py3.10` and
   `requirements.txt:27` pins `torch-npu==2.5.1.post1.dev20250619`. A host driver
   older than that CANN release is the top-ranked candidate here (#2315 reports
   `NPU-SMI 24.1.0.1`); it is a candidate, not a confirmed fix — see Limits.
   The row is per tag, so read the tag the report names: at v0.8.5rc1
   `Dockerfile.openEuler:18` is
   `FROM quay.io/ascend/cann:8.1.rc1-910b-openeuler22.03-py3.10` with
   `requirements.txt:12` `torch-npu==2.5.1` — and #975's own env block already prints
   exactly that in-container CANN 8.1.RC1 (and `torch-npu 2.5.1`), so the *container*
   side of the correspondence is verified by the paste and only driver/firmware is
   still open. Say that instead of re-deriving what the image ships; `git show HEAD:`
   the tag's Dockerfile only when the dump disagrees with it.
5. **A second repro on a different model is the model-independence proof — use it before
   any capacity arithmetic.** #975's reporter also ran `LLM()` on Qwen2.5-0.5B at TP=1
   and got the same three lines, which retires the 235B/MoE/`--max-model-len`/bf16-weight
   theories in one sentence, in the answer's first paragraph rather than a later section; a
   Qwen3-235B-A22B
   bf16 footprint note stays a one-line follow-up for the run that gets past device open.
6. **Conform the container to the 300I tutorial before blaming the driver**: the
   documented `docker run` also mounts `-v /usr/local/dcmi:/usr/local/dcmi` and
   `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi`
   (`docs/source/tutorials/single_node_300i.md:28-29`); #2315's command omits both,
   so its in-container DCMI/`npu-smi` view is not the documented environment.
   One re-run with the full mount list is cheaper than a driver theory.

## Do not

- **Do not walk `torch_npu`/GE C++ past the two frames above.** They state what the
  init does; why ACL's internal state is broken is a driver/firmware question with
  no source in any export. Cap it at ~4 calls — a recorded run (#2315) spent its
  whole session in `LazyInitAclops.cpp`, the `SocVersion` enum and vllm-ascend's
  `is_310p()` and never sent a message.
- **Do not rank the neighbouring Triton / `vllm._C` lines — but give the reason at
  the report's pin, because it changes.** `Failed to import from vllm._C with
  ModuleNotFoundError("No module named 'vllm._C'")` and `Triton not installed or
  not compatible; certain GPU-related functions will not be available.` are normal
  output, and which mechanism makes them normal depends on the tag:
  - **v0.9.2rc1 → v0.14.0rc1**: the Dockerfile removes triton on purpose —
    `Dockerfile.310p:42-44` and `Dockerfile:42-44` ("in Ascend, triton doesn't work
    correctly"), the same block at `:45-47` / `:53-55` by v0.14.0rc1.
  - **v0.7.1rc1 — the #302 image on this page — there is no removal step to cite.**
    `Dockerfile:40` installs vLLM with `VLLM_TARGET_DEVICE="empty"` and
    `requirements.txt:6` is `torch==2.5.1` alone, so triton was never installed and
    upstream's `vllm/triton_utils/importing.py:8-15` prints the line whenever
    `find_spec("triton") is None`; the missing `vllm._C` is the same empty-target
    install, warned at `vllm/_custom_ops.py:15-19` (the paste's `_custom_ops.py:19`).
    Telling a v0.7.1rc1 reporter "the Dockerfile uninstalls triton" is a wrong claim
    about their image — a recorded replay of #302 shipped it.
- **Do not date the install from an OCR'd path.** #2315 states its log came from a
  screenshot. `worker_vl.py` is `worker_v1.py`, `vllm/ascend/…` is `vllm_ascend/…`,
  and `vllm-0.4.0-…egg` is not a vLLM release: pinned vllm v0.9.2 has no
  `vllm/engine/core.py`, no `vllm/entrypoints/vllm_cli.py` and no
  `vllm/executor/multiprocess_executor.py`, and vllm-ascend v0.9.2rc1 registers
  `ascend = vllm_ascend:register` (`setup.py:392-394`). Ask for the log as text plus
  `pip show` before asserting a wrong version.

## Limits

- "Match the host driver/firmware to the image's CANN" is the mechanism plus four
  precedents, **not** a reported fix: the only reporter-confirmed resolution for this
  startup phase is #302's `--privileged=true`, and it shipped no version change, so the
  pairing stays hypothesis-class — it may still be what #2315's 310P host (CANN base
  newer than the host's `NPU-SMI 24.1.0.1`) or #564/#692 met. The *runtime* variant of
  the same pairing does have a reported fix —
  a worker killed during a long-context step, upgraded to a matched set
  (`../hccl-and-hardware/cancelled-executor-teardown.md`, #16023) — but do not
  carry that upgrade into a `500001` at device open: different phase, and no
  report ties the two.
- `Ascend/pytorch` here is a content export (`1190/1240` files) whose ref is the
  **pytorch 2.7.1** line, while this image pins `torch-npu==2.5.1.post1.dev20250619`;
  and #2315's frame spells the call as the free function `aclrtSetCompileOpt` in
  `NPUFunctions.cpp:216` rather than the pinned `at_npu::native::AclSetCompileopt`
  in `LazyInitAclops.cpp`. The two-frame shape is the same across both spellings; the
  reporter's exact torch_npu build is not verifiable here, so quote the mechanism and
  the version files, not the pinned line numbers, when answering.
- Keep the model-side verdict separate and one line long: Qwen3-32B on 2×310P3 is
  outside the 300I validated set and `-310p` is eager + fp16 only
  (`../soc-platform-paths/atlas-300i-310p.md`). It is a second, later finding, not
  this traceback's cause.
