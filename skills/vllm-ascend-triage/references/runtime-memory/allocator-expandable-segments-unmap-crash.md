# `expandable_segments:True` crashing inside `torch.npu.empty_cache()` (unmapHandles / 507899)

Reporter-confirmed on #12534 (Recorded section). The engine never becomes ready,
and the failing call is a **free/unmap**, not an allocation — so no memory figure
is short and nothing was served. It belongs to the allocator lane, not to the OOM
track and not to the collective/device track, even though the text is loud.

**Not the same lane as a memory-starvation report**: a paste where this variable was
offered as the cure for two engines on one die (`Available memory: 0`, `No available
memory for the cache blocks`) is the profiling-arithmetic family, and the setting
changes nothing there — reporter-confirmed on `#5707` with it set in the container
before launch (`kv-budget-per-instance-vs-device-global.md`). Do not read this page
onto that symptom.

**And a paste whose only crash text is `Fatal Python error: …` is not this page
either**: `none_dealloc: deallocating None …` (or `_Py_NegativeRefcount`) is the
interpreter aborting on a C extension's reference count, later in a run that had been
serving for weeks — the value of this variable is irrelevant to it and the naming
evidence is the block CPython prints under that line
(`../hccl-and-hardware/cpython-fatal-error-abort.md`).

The opposite verb is a **sibling, not a copy of this page**: a long-running engine
that dies on `map:… aclrtMallocPhysical … 507899` with `halMemCreate … drvRetCode=17`
(§*The map direction* below) has had an allocation fail, so the headroom arithmetic
comes first and this page's "no figure is short" shortcut is the wrong way round.

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
command. On #12534 the follow-up reads: with `expandable_segments:False` it deploys,
"2 cards run, 4 cards stall at `shm_broadcast`" — and the serve line in that same
reply is `--tensor-parallel-size 8 --max-model-len 12800 --gpu-memory-utilization 0.5
--max_num_seqs 4 --no-enable-prefix-caching --dtype float16
--additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false,
"enable_npugraph_ex":false,"enable_static_kernel":false},"enable_cpu_binding":"true",
"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false}'` plus
the export block below. It carries **no** `--compilation-config`: the
`FULL_DECODE_ONLY … cudagraph_capture_sizes: [1,4]` block belongs to the *first*,
crashing launch. Four reads, in this order:

1. **Count dies, not cards, before any theory.** A 300I Duo is two 310P chips on one
   physical card (`../hccl-and-hardware/transport-destlink-ipc-import.md`; recorded
   #4148's `npu-smi` puts devices 0,1 on one Bus-Id and 2,3 on another), so the first
   launch's `--device=/dev/davinci4`…`7` is four dies = **two** cards at TP=4, and the
   follow-up's TP=8 is eight dies = **four** cards. Prose and command then agree
   (`2 卡能部署 / 4 卡不行`) and the stall is a genuine eight-rank one — do not spend
   turns proving a rank asked for a device that does not exist; `npu-smi info` settles
   the count in one command (the SoC index's #12985 entry carries the same
   reconciliation and the reporter's two-command diff:
   `../soc-platform-paths/index.md`). Only if it really is eight ranks on four dies does
   the survivor reading apply: grep each rank for `107001` / `invalid device`
   (`../startup-failures/device-id-invalid-at-worker-init.md`) or `Worker … died
   unexpectedly`, and stop there.
2. **This launch still captures graphs, so `enable_npugraph_ex:false` clears nothing.**
   Without `--compilation-config`, vllm's default optimisation level is O2
   (`vllm/config/vllm.py:364`) whose table sets `cudagraph_mode =
   FULL_AND_PIECEWISE` (`:252`); at v0.23.0rc1 that mode satisfies
   `requires_piecewise_compilation()` and takes the piecewise branch
   (`vllm_ascend/platform.py:595`), which itself writes
   `enable_npugraph_ex = False` (`:622`) and applies the v1 splitting ops. Both
   `ascend_compilation_config` keys in the paste are therefore what the platform does
   anyway (`enable_static_kernel` additionally asserts `npugraph_ex`,
   `vllm_ascend/ascend_config.py:627-628`), and eight ranks still capture piecewise
   ACL graphs. Rank the capture lane first and make the ablation real:
   `--enforce-eager` (= `mode:none` + `cudagraph_mode:none`,
   `vllm/config/vllm.py:1064-1067`), then a reduced `cudagraph_capture_sizes`; the
   log's `Capturing ACL graphs (mixed prefill-decode, PIECEWISE): k/N` says whether
   capture was progressing (`../startup-failures/acl-graph-capture.md`,
   `../startup-failures/graph-mode-not-applied.md`).
3. `No available shared memory broadcast block found in 60 seconds` is EngineCore's
   writer watchdog (`VLLM_RINGBUFFER_WARNING_INTERVAL`) — the consequence, never the
   cause: it fires while a worker is inside `profile_run`/capture. In the *first*
   #12534 log it recovered (the line at 10:22:53, ranks saving compiled graphs at
   10:22:59-10:23:01); a follow-up that repeats it for minutes is a stall. Ask for the
   per-rank stop line **and** `npu-smi info` taken while it stalls (rising AICore = a
   slow forward, flat with no HBM growth = a hang). `../startup-failures/mm-encoder-profile-stall.md`
   §*Scope variant* is the recorded shape, reached from the same `2 卡能部署 / 4 卡不行`
   wording (#12985), and that section also owns the "do not re-derive the checkpoint's
   multimodal-ness" instruction — do not ask for it as a separate round.
4. On 310P the documented graph budget shrinks with TP: "the number of capturable
   graphs is limited and depends on the model depth. For example, Qwen3-32B can
   capture and replay 2 graphs … no such limitation when TP=1"
   (`docs/source/tutorials/hardwares/310p.md:23-26` at v0.23.0rc1), and the tree backs
   the eager-first order twice over: the 4×310P MoE tests that run are eager
   (`tests/e2e/pull_request/four_card/_310p/test_moe_model_310p.py:24-38`
   Qwen3-30B-A3B TP4 fp16, `:41-56` TP2-W8A8, both `enforce_eager=True`,
   `max_model_len=16384`, `max_num_batched_tokens=2048`), while the file's
   `Qwen/Qwen3.5-35B-A3B` TP4 case — this reporter's model — is
   `@pytest.mark.skip("Probabilistic failure, need fix")` (`:59-74`). So TP=4/TP=8
   310P runs of this model have **no passing validation at this pin**: say that as
   scope and put eager first, rather than promising a flag that makes it work.

Then read the follow-up's **export block** (v0.23.0rc1, `f4a08bbd`): of its six names
only two are read by this project's Python — the allocator value and the AIV switch — one
more (`HCCL_BUFFSIZE`) is a real knob whose consumer is CANN/HCCL, and two are inert
spellings:

| export | what it is at this pin |
|---|---|
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` | the cure above (rung 1); the `:True` that crashed is the value the model's own **A2/A3** tab exports (`Qwen3.6-35B-A3B.md:187`) |
| `HCCL_OP_EXPANSION_MODE="AIV"`, `HCCL_BUFFSIZE=512` | live, HCCL-side. Both are in this model's tutorial **A2/A3** tab (`docs/source/tutorials/models/Qwen3.6-35B-A3B.md:184-192`, there with `HCCL_BUFFSIZE=1024`), and vllm-ascend only reads `HCCL_OP_EXPANSION_MODE` itself for MiniMax-M2 graph capture (`vllm_ascend/patch/platform/patch_minimax_m2_config.py:118-124`). AIV changes the selected collective path, so an AIV-off relaunch is a real ablation, not a placebo (`../hccl-and-hardware/aiv-mc2-env-ablation.md`) |
| `USE_MULTI_GROUPS_KV_CACHE=1` | **not an env read**: `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:35` is a module constant `USE_MULTI_GROUPS_KV_CACHE = True`, and no other **code** file in either tree mentions it |
| `USE_MULTI_BLOCK_POOL=1` | no reader at this pin in vllm-ascend or vllm; the only hit is a commented line in `docs/source/tutorials/models/GLM5.2.md:316`. Version-scoped: at `bbe9bc0` (v0.13.0rc3) it is a **live gate** — `vllm_ascend/patch/platform/__init__.py:28-34` imports six platform patch modules under it — so an old-pin "patch 未生效" report is this env (`../upstream-downstream-drift/patch-import-side-effect.md`) |
| `ACL_OP_INIT_MODE=1` | docs/CI-only shell export (GLM5 / DeepSeek-V4-Pro tutorial blocks, weekly e2e yaml) — a CANN-side name, read by neither Python tree |

The model's own **310P tab** (`Qwen3.6-35B-A3B.md:234-293`, serve block `:236-259`)
carries none of those exports: `--tensor-parallel-size 2`, `--max-model-len 16384`,
`cudagraph_capture_sizes: [1,2,4,8,16]`, `--no-enable-prefix-caching`,
`--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'`,
`--dtype float16`. Diff the reporter's block against that tab before ranking a knob
(general rule: `../feature-enable-recipes/per-soc-recipe-and-quantized-weights.md`
§*Prune the flags that provably do nothing*; any paste, one call:
`scripts/env-knob-liveness.py`).

## The map direction: `halMemCreate` / `aclrtMallocPhysical` (same allocator, opposite verb)

```text
[drv api]halMemCreate failed drvRetCode=17, size=20971520(bytes), flags=0, ErrCode=507899,
  desc=[driver error:internal error], InnerCode=0x7020022[FUNC:MallocPhysical][FILE:npu_driver_mem.cc][LINE:345]
torch.OutOfMemoryError: map:../torch_npu/csrc/core/npu/NPUCachingAllocator.cpp:457
  NPU function error: aclrtMallocPhysical, error code is 507899
```

Recorded #16649: v0.23.0 image, GLM-5.3 W8A8C8, 2 × 16-card A2 mixed deployment
(DP4 × TP8 × EP16 × DCP8, `--max-model-len 1048576`), 33 h after a healthy start — one
rank (`Worker_DP2_TP0_DCP0_EP16`) fails creating a 20 MiB physical segment, its
EngineCore takes the fatal error and the rest are killed. Four reads, in this order:

1. **Three codes, three different facts** (v0.23.0, `5cb98caaadef`;
   `cann/runtime/include/external/acl/error_codes/rt_error_codes.h`): `507899` = `:140`
   `ACL_ERROR_RT_DRV_INTERNAL_ERROR`; `207001` = `:62` `ACL_ERROR_RT_MEMORY_ALLOCATION`
   ("only used by out of memory"); `107030` = `:51` `ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT`.
   `drvRetCode=17` is `DRV_ERROR_IOCRL_FAIL`
   (`cann/driver/pkg_inc/ascend_hal_error.h:51`), **not** `DRV_ERROR_OUT_OF_MEMORY`
   (`:40`, value 6) — so `torch.OutOfMemoryError` here is torch_npu's text predicate
   (`oom-class-is-not-a-cause.md`), and the driver's own word is "ioctl failed". Order the
   lines by timestamp (`scripts/plog-timeline.sh`) before deciding which is first.
2. **Physical headroom, from the log's own profile line** — the identity in `index.md`
   §*Reading a v0.21+ startup memory block*: `free_memory` minus the five categories. On
   #16649 that leaves ~3.6 GiB **unused** while a 20 MiB segment fails, so "the ratio target
   was already oversubscribed" is not the answer; a small map failing with GiBs free points
   at the segment/driver layer or at growth since startup.
3. **Growth or one-off**: two `npu-smi info` samples across the window. Without them a 33 h
   report cannot separate a climb from a flat ceiling (`hbm-growth-and-oom.md`).
4. **The launch block is the artefact** — ask for it in the first reply. #16649's reporter
   supplied exactly that script, and the flags decide every branch here: a written-out
   `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` (so the question is the *value*, not
   the injector at `vllm_ascend/platform.py:772-788`, `"expandable_segments" not in` at
   `:784`), `--max-model-len 1048576`, `--max-num-batched-tokens 1024`,
   `--gpu-memory-utilization 0.90`, `HCCL_BUFFSIZE=256`, `TASK_QUEUE_ENABLE=1`,
   `enable_cpu_binding`, `multistream_overlap_shared_expert`, `enable_dsa_cp` /
   `enable_sparse_li_c8`, `cudagraph_mode: FULL_DECODE_ONLY` + `enable_npugraph_ex`, MTP
   `num_speculative_tokens: 3`. Reconcile its `--gpu-memory-utilization 0.90` with the
   log's own `Desired GPU memory utilization is (0.92, 56.08 GiB)`: that script is a
   *later* copy (the reporter edited the suggested value in), so every number in the
   arithmetic above comes from the log, never from the pasted command. The neighbouring
   `107030 … synchronized memcpy failed, kind = 1` in the same paste is the capture lane
   (`../startup-failures/acl-graph-capture.md`), not the allocator.

`expandable_segments:False` — this page's cure in the unmap direction — is an **ablation
here, not the answer**: under the segment map an allocation that cannot reuse a cached
block has to create a physical segment, so a map failure can be a real shortage or a
driver/segment limit, where the unmap failure had nothing to allocate at all. Keep it as
rung 2 after the headroom and growth reads, and say which of the two it turned out to be.

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
