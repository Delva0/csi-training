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

## Ranked actions

1. **Get the memory numbers before the theory**: `npu-smi info` on the reporter's
   host while it fails, plus the serve log's `Using max model len` and any
   `NPU out of memory … tried to allocate … (… total capacity; … free)` line. If
   free HBM is genuinely small, this is capacity, and no code reading is needed.
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
