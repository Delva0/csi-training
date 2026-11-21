# `set default device id failed, ret: 107001` at the first worker: which devices the container really has

## Scope

`EngineCore failed to start` plus one `WorkerProc failed to start` per rank
(`multiproc_executor.py:852`), **no `Starting to load model`**, and the traceback's
only `vllm_ascend` frames are the first statement of `Worker.__init__`:

```text
vllm_ascend/worker/worker.py:109        check_ascend_device_type()
vllm_ascend/utils.py:723                soc_version = torch_npu.npu.get_soc_version()
torch_npu/npu/_backends.py:97           torch_npu.npu._lazy_init()
torch_npu/npu/__init__.py:263           torch_npu._C._npu_init()
RuntimeError: Initialize: …/compiler_depend.ts:165 … aclInit, error code is …
[ERROR] (PID: N, Device:0, RankID:-1) ERR99999 PTA call acl api failed
set default device id failed, ret: 107001 [FUNC:SetDefaultDeviceId]
  rtSetDefaultDeviceId execution failed, reason=device id error
Invalid_Argument(EE1001): The … invalid device ID. Check whether the device ID is valid.
[Process][DefaultDevice] process HandleDefaultDevice failed … ctx is NULL!
  [FUNC:GetDevErrMsg][FILE:api_impl.cc] … the context is a null pointer
```

Recorded **#10481** (vllm-ascend `v0.18.0rc1` `99e1ea0fe685`, Atlas A2, one visible
card, `ASCEND_RT_VISIBLE_DEVICES=7`, Qwen3.5-0.8B; the whole log exists only as two
screenshot OCRs, so the `aclInit` code digits are lost). Line numbers above are that
pin: re-grep `check_ascend_device_type` if the row is a different release.

The `aclrtSetDevice` failure happens **inside the SoC query**, before the device-type
guard compares anything — so the model, the quant config and every serve flag are
unreached, and nothing about them can be blamed or exonerated from this log.

## Three owners share this frame — classify on the text, not the frame

| Text in the paste | Owner | Page |
|---|---|---|
| `AssertionError: Current device type … does not match the installed version's device type` / `RuntimeError: Can not support soc_version: <n>` | the SoC the runtime resolved vs the type baked into the wheel by `SOC_VERSION` | `../soc-platform-paths/device-type-build-vs-runtime.md` |
| `SetPrecisionMode … 500001` + `PlatformInfoInit … SocVersion <name>` + `GEInitialize failed` | ACL/GE bring-up: mounted driver/firmware × the image's CANN | `acl-init-500001-and-soc-version.md` |
| `set default device id failed, ret: 107001` / `invalid device ID` / `reason=device id error` | **this page**: `aclrtSetDevice` rejected the id — the container/`ASCEND_RT_VISIBLE_DEVICES` device set | here |

`107001 = ACL_ERROR_RT_INVALID_DEVICEID` (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:22`;
doc row `cann/runtime/docs/api_docs/aclError.md:98` "无效的 Device ID，请检查 Device ID 是否合法"),
and `107002` is `ACL_ERROR_RT_CONTEXT_NULL` (`:23`, doc `:99`) — so the `ctx is NULL` /
`context is a null pointer` / `InnerError … log-inner.cpp` lines are the cascade of the
failed set-device, not a second fault.

## Why an id can be "not valid" in a container

torch_npu resolves the variable before it hands any id to ACL:
`_parse_visible_devices()` (`torch_npu/npu/__init__.py:341`, `Ascend/pytorch`
`v7.2.0.1-pytorch2.8.0` `da848bbc39f4`) parses `ASCEND_RT_VISIBLE_DEVICES`, and
`_device_count_ascend_hal()` (`:386-405`) trims that list against `drvGetDevNum()`
(`:369-383`), returning the index of the first entry `>= raw_cnt` (`:403`). So when
the container exposes fewer devices than the id you exported, the id is outside the
process's device set and the open returns `107001` — the same outcome as a card that
is physically present but held by another container/process. This does not establish
by itself *which* of the two cases a report hit; the checks below do.

## Checks, in this order (all on the reporter's host)

1. **Inside the failing container**: `ls /dev/davinci*` and `npu-smi info`. The
   `docker run` must map the **same ids** you export. #10481's own paste maps
   `--device /dev/davinci0` while exporting `7`; the pinned installation block
   enumerates `--device /dev/davinci0` … `/dev/davinci7` plus `davinci_manager`,
   `devmm_svm`, `hisi_hdc` (`docs/source/installation.md:434-441`, the same list
   again at `:475-482` in the source-build tab, `davinci0` also at `:226`),
   identical at v0.18.0rc1 and v0.18.0 — compare the paste with the block, don't
   assume the block.
2. **Framework-free probe**, one line, no vLLM:
   `python -c "import torch, torch_npu; print(torch_npu.npu.device_count(), torch_npu.npu.get_soc_version())"`.
   The same `107001` there ends the vllm-ascend part of the case; a printed count ≥1
   and an A2 code (220–225) means the device lane is fixed.
3. **Fix by making env and mapping agree**: `--device /dev/davinci7` with
   `ASCEND_RT_VISIBLE_DEVICES=7`, or `ASCEND_RT_VISIBLE_DEVICES=0` with the
   `davinci0` mapping. A card that exists but is busy shows in `npu-smi info`'s
   process table (`docker ps` for a stale container on the same card).
4. **Only if `npu-smi info` inside the container is broken or empty** go to the
   driver × image-CANN pairing (`cat /usr/local/Ascend/driver/version.info`, image
   tag); `acl-init-500001-and-soc-version.md` has the four-file list. An `aclInit`
   failure rather than a set-device failure keeps that cause open.

## A `docker run` can silently map one card

`\` followed by **spaces** before a `#` is not a line continuation: the backslash
escapes the space, `#` starts a comment, and bash ends the command there — the
remaining `--device`/`-v` lines run as separate commands (`-v: command not found`,
exit 127) while the container that *did* start has only the nodes named before the
comment. Verified by executing the block shape; `bash -n` does **not** flag it, so a
reporter who "ran the block" can still have a one-card container. Treat "which
`/dev/davinciN` are really inside" as open and settle it with check 1.

Do **not** turn this into a docs defect: at both `v0.18.0rc1` and `v0.18.0`,
`docs/source/installation.md` lists `davinci0` … `davinci7` on clean continuations
(`rg -n 'davinci0' docs/source/installation.md`), so such a paste came from
somewhere else (an older revision, a blog, a reporter's own edit) — ask where it was
copied from instead of claiming the page is wrong.

## Do not audit serve flags on a paste with no `Starting to load model`

The cheap exoneration is one grep in the pinned tree, not a flag walk: #10481's whole
line (`--async-scheduling`, `--distributed-executor-backend mp`,
`--tensor-parallel-size 1`, `--max-model-len 5500`, `--max-num-batched-tokens 40960`,
`{"cudagraph_mode": "FULL_DECODE_ONLY"}`,
`{"pa_shape_list":[48,64,72,80],"weight_prefetch_config":{"enabled":true}}`,
`--block-size 128`, `--gpu-memory-utilization 0.9`, `TASK_QUEUE_ENABLE=1`,
`HCCL_OP_EXPANSION_MODE=AIV`) is verbatim the project's own e2e config
`tests/e2e/multicard/2-cards/test_qwen3_performance.py:57-82` (Qwen3-8B, TP1) —
`rg -n 'pa_shape_list' <vllm-ascend tree>` lands on it. That one call removes the
flags *and* the model from the suspect list, and it is the general check for any
report whose arg set looks exotic (`pa_shape_list`, `weight_prefetch_config`,
`enable_cpu_binding`, a `FULL_DECODE_ONLY` block, a block size that is not 128).

## Limits

- The `aclInit` error code was lost in OCR; whether `aclInit` itself failed (driver/
  toolkit) or only `aclrtSetDevice` did changes the ranking. Ask for the plain-text
  log, `ls /dev/davinci*`, `npu-smi info` and `version.info` rather than guessing.
- Whether a *specific* reporter's container was truncated, mis-mounted or merely busy
  is not derivable from a screenshot; the two candidates are stated, not merged.
- No hardware ran here: every statement above is about what the pasted texts and the
  pinned sources mean.
