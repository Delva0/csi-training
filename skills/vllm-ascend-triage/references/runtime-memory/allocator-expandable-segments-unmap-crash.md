# `expandable_segments:True` crashing inside `torch.npu.empty_cache()` (unmapHandles / 507899)

Reporter-confirmed on #12534 (Recorded section). The engine never becomes ready,
and the failing call is a **free/unmap**, not an allocation — so no memory figure
is short and nothing was served. It belongs to the allocator lane, not to the OOM
track and not to the collective/device track, even though the text is loud.

## Signature

```text
Worker_TPn  multiproc_executor.py:982          output = func(*args, **kwargs)
            vllm_ascend/_310p/worker_310p.py:75    with memory_profiling(
            vllm/utils/mem_utils.py:261           torch.accelerator.empty_cache()
            vllm_ascend/patch/platform/patch_torch_accelerator.py:5
                                                  torch.npu.empty_cache()
            torch_npu/npu/memory.py:180           torch_npu._C._npu_emptyCache()
RuntimeError: unmapHandles:../torch_npu/csrc/core/npu/NPUCachingAllocator.cpp:595
  NPU function error: c10_npu::acl::AclrtUnmapMem((char *)ptr_ + segment_size_ * i,
  getHcclComm()), error code is 507899
[ERROR] (PID:408, Device:0, RankID:-1) ERR00100 PTA call acl api failed
[Error]: An internal error occurs in the Driver module.
        Invalid_Argument(EL0003) … rtUnmapMem execution failed, reason=driver error:internal error
        unmap memory failed, runtime result = 507899
```

Tells: **every** rank prints the same `unmapHandles … 507899`; the deepest frame is
`empty_cache()`; the failing call carries `getHcclComm()`; `507899` is
`ACL_ERROR_RT_DRV_INTERNAL_ERROR` (`cann/runtime`,
`include/external/acl/error_codes/rt_error_codes.h:140`), i.e. the driver refused,
and `ERR00100` + `EL0003` are its own wrapping, not an OOM code. It arrives right
after the last *successful* startup milestone (weight load, `Compiling a graph for
compile range (1, 2048) takes N s`), which is why reporters title the issue after
some other line (on #12534: "unable to save AOT compiled function", a caught
warning) and why the shutdown-time repeats (`RuntimeError: bad optional access`,
a second `empty_cache` in `gpu_model_runner.shutdown`) are echoes, not causes.

`memory_profiling` calls `gc.collect(); torch.accelerator.empty_cache()` **twice** —
on entry and on exit (`vllm/utils/mem_utils.py:284-285` and `:298-299`, vllm
`ee0da84`) — and the 310P worker opens it in `determine_available_memory`
(`vllm_ascend/_310p/worker_310p.py:75`, v0.23.0rc1 `f4a08bbd`). A `contextlib.__exit__`
frame in the paste means the **exit** call (profiling already ran); an entry-side hit
would land before `profile_run`.

The code alone is not the diagnosis: `507899` also appears in torch_npu's **UCE
query/repair** path (`checkUceErrAndRepair: … NPU error, error code is 507899`
beside `rtGetMemUceInfo … driver error:internal error`) — a device-fault lane with a
different owner (`../hccl-and-hardware/error-report-timestamps.md`,
`../hccl-and-hardware/device-health-checks.md`). Read the *API* in the same line:
`AclrtUnmapMem` from `empty_cache` is this page, `checkUceErrAndRepair` /
`rtGetMemUceInfo` is the device lane.

## Mechanism: the knob selects a map/unmap mode

`expandable_segments` in torch_npu does not merely stop splitting large blocks: it
reserves virtual address space and maps handles into it, and releases them with
`aclrtUnmapMem` — the allocator has explicit `SEGMENT_MAP` / `SEGMENT_UNMAP` states
documented "used with expandable_segments"
(`Ascend/pytorch`, `torch_npu/csrc/core/npu/NPUCachingAllocator.h:174-175`).
`empty_cache()` walks the freed segments and unmaps each one — exactly the
`(char *)ptr_ + segment_size_ * i` argument — so on a host/driver pair that rejects
that unmap the process dies at the first `empty_cache()` and never reaches the KV
allocation.

## The injection trap: unset is not off

vllm-ascend writes `PYTORCH_NPU_ALLOC_CONF` itself (`vllm_ascend/platform.py:761-779`
at v0.23.0rc1): unless the current value already contains `expandable_segments`,
`max_split_size_mb` or `garbage_collection_threshold`, it appends
`,expandable_segments:True`, then logs `Set PYTORCH_NPU_ALLOC_CONF=%s`. So:

1. `unset PYTORCH_NPU_ALLOC_CONF`, or omitting the export, leaves the crashing value
   in place. Switch it off by writing an accepted key yourself —
   `expandable_segments:False` or `max_split_size_mb:256` (the 300I tutorial's value).
2. Confirm from the log, not the flags: `grep -m1 'Set PYTORCH_NPU_ALLOC_CONF' <log>`
   must show your value; an appended `…,expandable_segments:True` means the export
   never reached that process.
3. The same block makes `expandable_segments` and `max_split_size_mb` mutually
   exclusive — do not stack them "for safety".

## Ranked causes and one check each

| # | Candidate | One check | Strength |
|---|---|---|---|
| 1 | `expandable_segments:True` cannot be unmapped on this driver/CANN pair | relaunch with `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` | **reporter-confirmed** on #12534: the deployment started with exactly this change |
| 2 | the value came from vllm-ascend's injection, not the launcher | `grep -m1 'Set PYTORCH_NPU_ALLOC_CONF' <serve log>` | source-verified at v0.23.0rc1 |
| 3 | host driver ≠ image CANN (the container bind-mounts `/usr/local/Ascend/driver`) | `cat /usr/local/Ascend/driver/version.info` + `npu-smi info` vs the image tag | unverified; only if row 1 does not clear it |
| 4 | graph compile left allocator state inconsistent (`enable_npugraph_ex`, `FULL_DECODE_ONLY`, AOT) | `--enforce-eager`, `VLLM_USE_AOT_COMPILE=0` | low: the unmap is the allocator's own call, and the AOT line above it is caught and logged as a warning |

Rule out by name before answering: capacity shrinkage (`--gpu-memory-utilization`,
`--max-model-len` — no allocation failed), the KV-init page
(`Memory_Allocation_Failure(EL0004)` + `207001` + SDMA/SMMU inside
`_convert_torch_format`, `kv-init-and-format-cast-alloc.md`), and any card/collective
theory (`ERR00100` here wraps the allocator's failing unmap; there is no `hcclRet`).

## If the same reporter returns with a different failure after the fix

The allocator change unblocks startup; it proves nothing about the rest of the
command. On #12534 the follow-up was TP=2 working and **TP=4 stalling at
`shm_broadcast`** on a `-310p` image with
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,4]}'`.
Read that pair as the reporter's **prose**: the command in the same reply carries
`--tensor-parallel-size 8` (with `--max-model-len 12800`, `--gpu-memory-utilization 0.5`,
a different `--additional-config` and no `--compilation-config` line), so the TP
number and the graph mode of the stalling run are unconfirmed until the reporter
sends that launch verbatim.
Two cheap splits, in this order:

1. `No available shared memory broadcast block found in 60 seconds` is the
   EngineCore writer's INFO watchdog (`VLLM_RINGBUFFER_WARNING_INTERVAL`), and in a
   *startup* log it fires while workers compile: the original #12534 log has it at
   10:22:53 with the ranks saving compiled graphs at 10:22:59-10:23:01, i.e. it
   recovered. Check whether **every** rank printed its compile/AOT line; the rank
   missing from that set is the one that stopped, and its per-rank log is the next
   artefact (`../hccl-and-hardware/rpc-watchdog-timeouts.md`).
2. On 310P the documented graph budget shrinks with TP: "the number of capturable
   graphs is limited and depends on the model depth. For example, Qwen3-32B can
   capture and replay 2 graphs … no such limitation when TP=1"
   (`docs/source/tutorials/hardwares/310p.md:20-30` at v0.23.0rc1). A TP=4 stall with
   two capture sizes is therefore a candidate before any HCCL theory: run
   `--enforce-eager`, then `cudagraph_capture_sizes: [1]`
   (`../soc-platform-paths/atlas-300i-310p.md`,
   `../startup-failures/acl-graph-capture.md`).

## What this page does not establish

- Which driver builds reject the unmap, and why. The exporter's
  `torch_npu/csrc/core/npu/NPUCachingAllocator.cpp` (the `:595` in the message) is
  **not** in the pinned `Ascend/pytorch` export: the header carries the mechanism,
  the implementation is unread. The export is `v26.0.1-pytorch2.10.0` while the
  reporter runs `torch_npu 2.10.0.post2`.
- That `expandable_segments:False` is the right setting elsewhere. The
  anti-fragmentation default exists for a reason (`hbm-growth-and-oom.md` step 3);
  here it is the value that fails, so state the trade rather than recommending
  `False` on every host.
