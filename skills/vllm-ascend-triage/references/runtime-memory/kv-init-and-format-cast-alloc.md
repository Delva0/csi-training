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
- `Memory async copy failed … length=2654994432` (2.47 GiB — the chunk in flight,
  one or two layers' K+V like the two 310P rows below; the second whole pool is a
  separate live requirement, see the verdict rule),
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
ladder that decides it is the `2 × pool` test below, then a **halved**
`--gpu-memory-utilization` (`--max-model-len` does not move the pool in V1).

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

So the pool is allocated as raw int8 storage for **every layer**, and the
format-cast copy adds a second buffer that is kept too while the first is still live;
if the budget was sized for the pool alone, the cast is exactly where the peak exceeds
HBM. State this as the mechanism with its scope: the copy is real on 310P and for int8
caches; at the v0.11.0rc3 pin fp16 caches return early (`is_enable_nz`), so a frame
that really executed `npu_format_cast` **there** means a non-fp16/bf16 cache **or** a
tree whose gate differs from that pin (the #4812 line numbers `2682` / `3038` belong to
neither pinned function, so the reporter's tree is not the pinned one) — while
v0.11.0rc0 and `048bfd555` carry **no** gate at all, so an fp16 KV cache goes through
the cast there and this page's arithmetic applies as written.

## Peak arithmetic: do it from the log, before the theory

The serve log already carries every term (#3479, vllm-ascend v0.11.0rc0 +
Qwen3-4B, put in one pass):

| Term | Where it comes from |
|---|---|
| `total` | `Available memory: <N>, total memory: <M>` → `M` |
| `weights` | `Loading model weights took X GB` |
| `pool` | `Available memory: <N>` on that same line — the post-profiling HBM left for KV, i.e. after the weights — or `GPU KV cache size: <T> tokens` × per-token bytes |
| per-token KV bytes | `layers × 2 (K,V) × kv_heads × head_dim × dtype_bytes` from the checkpoint's `config.json`; `pool / T` must reproduce it to a rounding step (147,464.7 vs 147,456 in #3479), which is how you prove the geometry you assumed is the one that ran |
| the failing copy | the `length=<L>` in the `sdma copy` / `memcpy` record — one or two layers' K+V, i.e. the per-layer cast loop, not one big allocation; `pool / L` tells you which (24 layers on #3316 box 1, 28 on box 2), so divide before quoting it |

Verdict rule — **the second buffer is a whole pool, not one layer.** Verified at
`048bfd555` (the #3316 reporter's own sha): `initialize_kv_cache_tensors` allocates one
raw int8 buffer per layer into `kv_cache_raw_tensors` (`model_runner_v1.py:2504-2527`;
`torch.zeros(kv_cache_tensor.size, dtype=torch.int8)` at `:2513`) and keeps that dict
alive through the whole loop, while each converted tensor goes into a second dict that
the function returns (`kv_caches[layer_name] = kv_cache`, `:2580`). `npu_format_cast` is
out-of-place (`at::Tensor npu_format_cast(const at::Tensor&, int64_t)`,
`Ascend/pytorch → torch_npu/csrc/core/npu/NPUFormat.h:24`), so at the last layer's
conversion the live set is `peak + 2 × pool`. The `length=<L>` in the sdma record is
only the chunk in flight (one or two layers), which is why it is the geometry check and
not the total.

`pool` is `gmu × total − peak` by construction (`worker_v1.py:194-197`), so the cast
completes only while

    peak + 2 × (gmu × total − peak) ≤ total   ⇔   gmu ≲ (total + peak) / (2 × total)

i.e. **just over 0.5**. A 0.9 → 0.8 trim moves 10 % of the card against a demand of one
extra pool, so it cannot work, and the three recorded runs all sit above the bound and
all died at this frame:

| run (310P, fp16 KV, cast unconditional) | `total memory:` | `peak` = gmu×total − `Available memory:` | `pool` | `2 × pool + peak` | vs `total` → gmu bound |
|---|---|---|---|---|---|
| #3316 box 1, `048bfd555`, 310P3, Qwen2.5-0.5B, default 0.9 | 49.60 GB | 4.46 GB | 40.18 GB (3,269,760 tok × 12,288 B); failing `length=3348234240` = **2 layers'** K+V | 84.8 GB | **1.71 ×** → ≤ 0.55 |
| #3316 box 2, official `v0.11.0rc0-310p`, 310P1, Qwen3-0.6B, default 0.9 | 93.96 GB | 4.61 GB | 79.95 GB (697,088 tok × 114,688 B); `length=2855272448` = **1 layer's** K+V | 164.5 GB | **1.75 ×** → ≤ 0.52 |
| #3479, same `-310p` tag, 310P3, Qwen3-4B, **`--gpu-memory-utilization 0.8`** | 46.43 GB | 10.19 GB | 26.95 GB; `length=748683264` | 64.1 GB | **1.38 ×** → ≤ 0.61 |

The third row is the one that prices the rung: a sibling at the same image tag, card and
pin had **already run 0.8** and died here, so "0.9 → 0.8" is not a test and a
"fits to the GiB" prediction built on that 10 % step is refuted before it is written.
Pass the bound rather than a predicted fit: `--gpu-memory-utilization 0.4-0.5` puts
`2 × pool + peak` at 0.5-0.6 of the card on all three boxes, and usable HBM is smaller
than the printed total (CANN reservations, fragmentation), so the rerun's own
`Available memory:` line is the confirmation. The `length=` term stays useful —
reproducing `2 × tokens × kv_heads × head_dim × dtype_bytes` exactly (as all three rows
do) proves the checkpoint geometry you assumed is the one that ran.

## Ranked actions

1. **Run the 2 × pool test on the paste's own lines before asking for anything**:
   `Available memory: <N>, total memory: <M>` and `GPU KV cache size: <T> tokens` plus
   the checkpoint's `config.json` give `peak = gmu×M − N` and `pool = N`, so the verdict
   is arithmetic on numbers the reporter has already posted (#3316 box 2's paste carried
   all of them, and that is the whole answer to "is this the card or the code").
   `npu-smi info` is for co-tenancy and health — another process holding HBM, a
   `Critical` row — not for this sum, and a source install (`/vllm-workspace/...`)
   changes none of it; the #1993 follow-up above was exactly that host.
2. **Halve `--gpu-memory-utilization` (0.4-0.5), and spend nothing on `--max-model-len`**:
   in V1 the pool is sized from available memory (`worker_v1.py:194-197`), so the flag
   bounds per-request context and concurrency only and cannot shrink the pool this frame
   needs. This is the cheapest one-line test and it decides capacity vs defect — at the
   default 0.9 the pool is ~0.9 × total and the second pool can never fit. #4812 is the
   case where nothing was sized by hand: its serve line is the 300I tutorial's
   Qwen2.5-VL-3B tab verbatim, with neither flag, so both ran at their defaults
   (`../soc-platform-paths/atlas-300i-310p.md`).
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
5. **Only if EL0004 persists with the 2 × pool test passing** (halved `gmu`, and the
   rerun's `2 × pool + peak` printed below `total memory:`): treat the SDMA/SMMU
   Terminate as primary and go to the device lane — every rank's first `EZ9999` /
   `chipId:` line, the plog, `npu-smi info` health / `asys health -d=<id>`
   (`../hccl-and-hardware/device-health-checks.md`). An SMMU page-table Terminate does
   not follow from a user-space OOM. **Then the release lane, which is a real answer
   for this shape**: the pin's 310P doc from v0.11.0rc3 on carries this failure as a known
   issue and points at v0.10.0rc1 (quote and pin scope:
   `../soc-platform-paths/atlas-300i-310p.md`, ranked actions), and #3316's reporter
   later confirmed a `v0.10.0rc1` 310P image serving on the same class of host
   (2025-11-13, ~2.5 tok/s) — so the title `v0.11.0rc0不支持Atlas 300I` is never the
   verdict at any of these pins.

## What this page does not establish

- It does not prove the #4812 host was out of memory; the posted dump carries no
  free-HBM figure and no `Using max model len` line, its serve line leaves
  `--max-model-len` and `--gpu-memory-utilization` at their defaults, and no rerun at a
  smaller pool was ever posted. The ranking on the zero-log form
  of this shape is `../soc-platform-paths/atlas-300i-310p.md`, its startup section.
- Line numbers are version-specific. Re-grep `_convert_torch_format`,
  `initialize_kv_cache_tensors` and `is_enable_nz` in the reporter's version
  before quoting any of them.
- A repeated failure at the *same* KV-cast frame **after the 2 × pool test passes**
  (`pool` halved, `2 × pool + peak` printed under `total memory:`) is evidence for a
  driver/hardware or allocator-fragmentation problem, not for the cast being wrong. A
  repeat at a larger `--max-model-len`-style change proves nothing, since that flag does
  not move the pool.
- Whether the two-box #3316 evidence settles the mechanism: it settles the arithmetic
  (both demands are 1.7 × the card), not the layer at which each run died — neither paste
  prints the layer index, and the sdma record names only the chunk.
