# Allocation failure at KV-cache init, surfacing inside a format cast

## Signature (reporter-confirmed for #4812)

Startup dies in the KV-cache initialisation phase, and the deepest Python frames
are a **format cast**, not an operator the model asked for:

```
worker_v1.py  initialize_from_config → model_runner.initialize_kv_cache
model_runner_v1.py  initialize_kv_cache → initialize_kv_cache_tensors
model_runner_v1.py  k_cache = self._convert_torch_format(k_cache)
model_runner_v1.py  tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT)
torch/_ops.py       return self._op(*args, **(kwargs or {}))
RuntimeError: npuSynchronizeDevice … AclrtSynchronizeDeviceWithTimeout, error code is 507013
ERR00100 PTA call acl api failed / [Error]: System Direct Memory Access (DMA) hardware execution error
[PID …] … Memory_Allocation_Failure(EL0004): Failed to allocate memory. Possible Cause: Available memory is insufficient.
alloc device memory failed, runtime result = 207001 … there is a sdma error, sdma channel is 0 …
The SMMU returns a Terminate error during page table translation … CQE status is 2
rtDeviceSynchronizeWithTimeout execute failed, reason=[sdma copy error]
```

Read it in this order: **`EL0004` + `207001` are an allocation failure** (the
CANN runtime could not get device memory), **507013 + `sdma copy error` +
SMMU Terminate are what the synchronizing copy reported afterwards**, and the
`npu_format_cast` frame is simply where that copy was requested. The posted dump's
timestamps support that ordering: EL0004 at `…16:44:35.945.058`, the SDMA/sync
report at `…36.054.123` (the two lines use different time zones in the paste —
compare seconds, not hours). So do not let the loudest line (SDMA hardware error)
become the cause without checking the memory numbers first.

A second recorded paste is the same signature on a **source build**, so a rebuild is
not the rung that ends it: #1993's 300I Duo thread (original report v0.7.3,
Qwen2.5-VL-3B offline `LLM()`) had its reporter answer "ready to install again from
the source code" (2025-07-30) and then, on 2025-09-17 from `/vllm-workspace`:

- frames `worker_v1.py:293` → `model_runner_v1.py:2396 initialize_kv_cache` →
  `:2579 initialize_kv_cache_tensors` → `:2377 _convert_torch_format` →
  `npu_format_cast`;
- `Memory async copy failed … length=2654994432` (2.47 GiB — the same order as the
  two 310P rows below, i.e. one or two layers' K+V, not a second whole pool),
  `sdma channel is 0 … SMMU Terminate … CQE status is 2`, then
  `EL0004 … Failed to allocate memory` with the canned advice
  `Possible Cause: Available memory is insufficient`, ending
  `Engine core initialization failed … Failed core proc(s): {}` in the frontend;
- and, just above it, the torch_npu warning `Cannot create tensor with NZ format`
  for a rank<2 tensor, which continues `tensor will be created with ND format`. That
  line is **torch_npu declining an NZ cast, not the failure**, and not evidence the
  cast is wrong: it only dates the run to the `ACL_FORMAT_FRACTAL_NZ` lane
  (`is_enable_nz` above).

Its `:2396` / `:2579` / `:2377` match neither this pin's numbers nor #4812's, so quote
them as "the reporter's tree", and with no `npu-smi` and no `Using max model len` in
the paste ranked action 1 below is still the deliverable. Do not let "rebuilt from
source" promote a device-fault theory: the failed copy is an allocation, and the
ladder that decides it is `--max-model-len` then `--gpu-memory-utilization`.

This is a different lane from `../hccl-and-hardware/index.md` (there the frame is
a sync point for a pre-existing collective/device fault and no allocation failed)
and from `../custom-ops-runtime-faults/index.md` (there an op is named). It is
also different from `Memory_Allocation_Failure(EL0004)` on a long-running server
(`hbm-growth-and-oom.md`): here the engine **never became ready**.

## Why the frame is an allocation site (verified at vllm-ascend v0.11.0rc3, b6d63bb)

- `initialize_kv_cache` (`vllm_ascend/worker/model_runner_v1.py:2723`) dispatches
  to `initialize_kv_cache_tensors` (`:2954`; MLA `:2869`, sparse-SFA `:2763`).
- The plain path first allocates one raw **int8** device buffer per
  `kv_cache_tensor` — `torch.zeros(kv_cache_tensor.size // 2, dtype=torch.int8,
  device=self.device)` for k and v (`:2961-3005`), i.e. the whole KV pool.
- Then, per layer, it makes a typed view and converts it:
  `k_cache = raw_k_tensor.view(dtype).view(shape)` → `k_cache =
  self._convert_torch_format(k_cache)` (`:3075-3076`), same for `v_cache`. The
  sparse-SFA path does the same for `rope_cache` / `nope_cache` / `k_cache`
  (`:2819-2826`).
- `_convert_torch_format` (`:2716-2721`) calls
  `torch_npu.npu_format_cast(tensor, ACL_FORMAT)`; `ACL_FORMAT` is
  `ACL_FORMAT_FRACTAL_NZ` on 310P and `ACL_FORMAT_FRACTAL_ND` elsewhere
  (`:146-154`), and it returns early when `is_enable_nz(dtype)` is False —
  fp16/bf16 are exempt (`vllm_ascend/utils.py:74-84`), int8 is not
  (`VLLM_ASCEND_ENABLE_NZ` default 1, `vllm_ascend/envs.py:168-169`).
- The source itself flags this window: `# TODO: remove this after the OOM issue
  is located and fixed, otherwise, some model may # encounter OOM issue`
  (`:3033-3034`).

So the pool is allocated once as raw int8 storage, and the format-cast copy adds a
second buffer while the first is still live; if the budget was sized for the pool
alone, the cast is exactly where the peak exceeds HBM. State this as the
mechanism with its scope (the copy is real on 310P and for int8 caches; on fp16
caches at this pin the cast returns early, so a frame that really executed
`npu_format_cast` means a non-fp16/bf16 cache **or** a tree whose gate differs
from this pin — the #4812 line numbers `2682` / `3038` belong to neither pinned
function, so the reporter's tree is not the pinned one).

## Peak arithmetic: do it from the log, before the theory

The serve log already carries every term (#3479, vllm-ascend v0.11.0rc0 +
Qwen3-4B, put in one pass):

| Term | Where it comes from |
|---|---|
| `total` | `Available memory: <N>, total memory: <M>` → `M` |
| `weights` | `Loading model weights took X GB` |
| `pool` | `Available memory: <N>` on that same line — the post-profiling HBM left for KV, i.e. after the weights — or `GPU KV cache size: <T> tokens` × per-token bytes |
| per-token KV bytes | `layers × 2 (K,V) × kv_heads × head_dim × dtype_bytes` from the checkpoint's `config.json`; `pool / T` must reproduce it to a rounding step (147,464.7 vs 147,456 in #3479), which is how you prove the geometry you assumed is the one that ran |
| the failing copy | the `length=<L>` in the `sdma copy` / `memcpy` record — `L = 2 × T × kv_heads × head_dim × dtype_bytes` is **one layer's K+V**, i.e. the per-layer cast loop, not one big allocation |

Verdict rule, and the one place a recorded answer overreached: the raw int8 pool stays
live while the NZ copy is written, but the copy is **per layer**, so the peak is
`weights + pool + window`, where `window` is the `length=` of the failing memcpy record —
one layer's K+V in the two later observations, not a second whole pool. Two 310P
environments of #3316 measure it (both `Qwen`, fp16, default `gmu`):

| env | card (`total memory:`) | weights | pool | failing `length=` | nominal slack |
|---|---|---|---|---|---|
| 310P3, reporter's first box | 46.20 GiB (49,602,220,032 B) | 0.9299 GB | 37.42 GiB (3,269,760 tok × 12,288 B) = 40.18 GB | `3348234240` = **2 layers'** K+V | ~5 GiB |
| 310P1, official `v0.11.0rc0-310p` image, another host | 93.96 GB (89,608 MB `npu-smi`) | 1.1214 GB | 79.95 GB (697,088 tok × 114,688 B) | `2855272448` = **1 layer's** K+V | ~10 GB |

Both died with `EL0004` + SMMU Terminate at that per-layer copy. Two consequences, and
the second is the correction: the `weights + 2 × pool` screen used for #3479
(`7.56 + 2 × 26.95 = 61.46 GB` against a 46.43 GB card) **overstates the demand by about
2×**, and the usable HBM is nevertheless *smaller* than the printed total (CANN
reservations, a pool that large fragments) — so 5-10 GB of arithmetic headroom is not
evidence against the capacity lane, and a computed "gmu X fits to the GiB" prediction is
not deliverable. Say what the ladder does instead: halve `gmu` (or keep ≥20 % of the
card free), re-run, and let the outcome decide. The `length=` term is still the useful
check — reproducing `2 × tokens × kv_heads × head_dim × dtype_bytes` exactly (as both
rows above do) proves the checkpoint geometry you assumed is the one that ran.

## Ranked actions

1. **Get the memory numbers before the theory**: `npu-smi info` on the reporter's
   host while it fails, plus the serve log's `Using max model len` and any
   `NPU out of memory … tried to allocate … (… total capacity; … free)` line. If
   free HBM is genuinely small, this is capacity, and no code reading is needed.
   A source install (`/vllm-workspace/...`) changes none of this — the #1993
   follow-up above was exactly that host and the ask was still these three.
2. **Shrink the pool**: `--max-model-len` (a VL model's default context is huge),
   then `--gpu-memory-utilization`. Two full pools (raw + NZ copy) have to fit.
   This is the cheapest one-line test and it decides capacity vs defect.
3. **The 310P environment line**: `export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`
   is what `docs/source/tutorials/single_node_300i.md:47-48` prescribes for
   300I. Note it disables vllm-ascend's own `expandable_segments` injection
   (`hbm-growth-and-oom.md`, step 3) — mention both effects rather than selling it
   as a fix.
4. **If the cache is int8/quantised and you want to test the cast hypothesis**:
   `VLLM_ASCEND_ENABLE_NZ=0` makes `is_enable_nz` False so the NZ conversion is
   skipped at this pin (verified path above). It trades layout/performance for
   memory; present it as a one-command discriminator, not as the recommended
   configuration.
5. **Only if EL0004 persists at a small `--max-model-len` with plenty of free
   HBM**: treat the SDMA/SMMU Terminate as primary and go to the device lane —
   every rank's first `EZ9999` / `chipId:` line, the plog, `npu-smi info`
   health/`asys health -d=<id>` (`../hccl-and-hardware/device-health-checks.md`).
   An SMMU page-table Terminate does not follow from a user-space OOM.

## What this page does not establish

- It does not prove the #4812 host was out of memory; the posted dump carries no
  free-HBM figure and no `Using max model len` line.
- Line numbers are version-specific. Re-grep `_convert_torch_format`,
  `initialize_kv_cache_tensors` and `is_enable_nz` in the reporter's version
  before quoting any of them.
- A repeated failure at the *same* KV-cast frame across a much smaller
  `--max-model-len` is evidence for a driver/hardware or allocator-fragmentation
  problem, not for the cast being wrong.
