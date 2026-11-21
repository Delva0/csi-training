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

## What the two log facts mean

- `SetPrecisionMode` is torch_npu's own init step; no vLLM flag reaches it. At the
  pinned export (`Ascend/pytorch` `v7.1.RC1.alpha002-pytorch2.7.1`, `d33df559`)
  `torch_npu/csrc/framework/LazyInitAclops.cpp:150-157` derives the string from the
  SoC it believes it is on — `GetSocVersion() >= SocVersion::Ascend910B1 ?
  "must_keep_origin_dtype" : "allow_fp32_to_fp16"` — and hands it to
  `AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, …)` at `:156`. Both are
  **parameters of an init that already failed**: `500001` is ACL's internal error,
  so no `--dtype`, `--compilation-config`, `--enforce-eager` or precision knob the
  reporter passes can cause or repair it. Do not rank them.
- The same init just resolved the SoC:
  `torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp:157-159` `AclGetSocName()` →
  `SetSocVersion(soc_name)`, then `:167-169` `InitAclops()` (when `ACL_OP_INIT_MODE`
  is 0, its default). So the `SocVersion <name>` in the `PlatformInfoInit` line is
  **what the runtime resolved for the device it opened**, and it is the first thing
  to read.

## Read the SoC line against the card

| `npu-smi info` chip | `PlatformInfoInit … SocVersion` | Reading |
|---|---|---|
| `310P3` | `Ascend@910B…` (or any name the card cannot report) | the runtime did not bind to this card: the mounted host driver/firmware, or a toolkit/opp inside the container that is not the card's line. Name the conflict before anything else |
| matches the card | `500001` still | the matching-SoC precedents (#302/#564/#692, all 910B2): host driver/firmware older than the image's CANN base, or the device is unusable inside the container — so a *matching* line does not exonerate the pairing |

## Checks, in this order (all on the reporter's host)

1. **Framework-free probe in the same container**, no vLLM:
   `python -c "import torch, torch_npu; torch.npu.set_device(0); print(torch.npu.get_device_name(0))"`.
   The same `500001` there ends the vllm-ascend part of the case.
2. The version set plus the image tag: `cat /usr/local/Ascend/driver/version.info`,
   `npu-smi info` (header version + chip), `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg`,
   `pip show torch torch-npu vllm vllm-ascend`. The container's CANN is the
   **image's** — `docs/source/installation.md:53-65` mounts only the driver's
   `lib64/` + `version.info`, `dcmi`, `npu-smi` and `/etc/ascend_install.info`,
   never `ascend-toolkit/`. What the host contributes is driver/firmware, and that
   is the pairing under test.
3. Compare against the image's own base: at v0.9.2rc1 `Dockerfile.310p:18` is
   `FROM quay.io/ascend/cann:8.1.rc1-310p-ubuntu22.04-py3.10` and
   `requirements.txt:27` pins `torch-npu==2.5.1.post1.dev20250619`. A host driver
   older than that CANN release is the top-ranked candidate here (#2315 reports
   `NPU-SMI 24.1.0.1`); it is a candidate, not a confirmed fix — see Limits.
4. **Conform the container to the 300I tutorial before blaming the driver**: the
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
- **Do not rank the neighbouring Triton lines.** `Failed to import triton_utils_v2
  with ModuleNotFoundError("No module named 'vllm._C'")` and `Triton not installed
  or not compatible; certain GPU-related functions will not be available.` are
  normal in this image: `Dockerfile.310p:42-44` and `Dockerfile:42-44` remove triton
  on purpose at v0.9.2rc1 ("in Ascend, triton doesn't work correctly"; the same block
  is at `:45-47` / `:53-55` by v0.14.0rc1).
- **Do not date the install from an OCR'd path.** #2315 states its log came from a
  screenshot. `worker_vl.py` is `worker_v1.py`, `vllm/ascend/…` is `vllm_ascend/…`,
  and `vllm-0.4.0-…egg` is not a vLLM release: pinned vllm v0.9.2 has no
  `vllm/engine/core.py`, no `vllm/entrypoints/vllm_cli.py` and no
  `vllm/executor/multiprocess_executor.py`, and vllm-ascend v0.9.2rc1 registers
  `ascend = vllm_ascend:register` (`setup.py:392-394`). Ask for the log as text plus
  `pip show` before asserting a wrong version.

## Limits

- The history corpus carries issue bodies only: "match the host driver/firmware to
  the image's CANN" is the mechanism plus four precedents, **not** a reported fix
  for this startup phase. The *runtime* variant of the same pairing does have one —
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
