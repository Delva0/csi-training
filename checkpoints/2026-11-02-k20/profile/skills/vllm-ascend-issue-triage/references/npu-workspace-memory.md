# NPU op-workspace memory: `NPUWorkspaceAllocator` OOM and the per-stream ratchet

Two other OOM classes are answered elsewhere: an allocation whose size is quadratic in the prompt is
the attention-mask builder ([attention-mask-oom.md](attention-mask-oom.md)), and an OOM at
`acl_graph` replay synchronize is in [log-signatures.md](log-signatures.md). This file covers the
one whose message names `NPUWorkspaceAllocator`, and — because they are the same question of device
memory that nothing returns by itself — sleep-mode reclamation across `sleep`/`wake_up` cycles
(section at the end).

Reading scope: `torch_npu` from the `Ascend/pytorch` clone HEAD (the clone carries
`v7.x-pytorch2.x` tags; a report names a torch_npu version such as `2.10.0`, so confirm the file
still matches before quoting a line), `Ascend/op-plugin` HEAD, and vllm-ascend tag `v0.20.2rc1` for
the plugin side. Everything below was read from source plus one reporter log; no NPU run was made.

## Is the report this class?

```text
NPU out of memory. NPUWorkspaceAllocator tried to allocate 216.00 MiB
  (NPU 0; 61.27 GiB total capacity; 295.32 MiB free)
[PID:4806] Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by APP module
... current working operator name is SelfAttentionOperation
```

Two facts come from the message itself: the failing request is **operator workspace**, not a tensor
the model holds, and the free figure is the device's total, not a pool's. The named operator is
where the *asynchronous* fault surfaced and can be one op later than the allocation —
`ASCEND_LAUNCH_BLOCKING=1` cannot be combined with a graph mode, so making it synchronous needs a
graph-free rerun ([full-decode-only-graph-mode.md](full-decode-only-graph-mode.md)).

## Two allocators, and which one the message names

`torch_npu/csrc/framework/utils/OpPreparation.cpp` decides it by overload:

| call site | allocator | governed by |
| --- | --- | --- |
| `unsafe_empty_workspace(size)` (:413) | `NPUCachingAllocator::get()` | `PYTORCH_NPU_ALLOC_CONF` (`expandable_segments`, `page_size`, …) |
| `unsafe_empty_workspace(size, acl_stream)` (:431) | `NPUWorkspaceAllocator::get()` via `malloc_with_stream` | nothing in `PYTORCH_NPU_ALLOC_CONF` — separate pool |

Which one an op gets depends on the aclnn call macro it was dispatched through
(`Ascend/op-plugin`, `op_plugin/utils/op_api_common.cpp`): the legacy path's
`GetWorkSpaceAddr(size)` (:1630) uses the caching allocator, while `GetWorkSpaceAddrV2(size,
acl_stream)` (:1651) — the one the `EXEC_NPU_CMD_V2` macro expands to, `op_api_common.h:410` — uses
the workspace allocator. `_npu_fused_infer_attention_score_get_max_workspace` goes through
`allocate_workspace(size, stream)` → `unsafe_empty_workspace(size, stream)`, i.e. the workspace
pool, which is why the failing op in the recorded report names `NPUWorkspaceAllocator`.
Consequence for triage: `expandable_segments:True` in the reporter's environment cannot fix this
OOM, and "raise `--gpu-memory-utilization`" is not the lever the message points at.

## What the workspace allocator does

`torch_npu/csrc/core/npu/NPUWorkspaceAllocator.cpp` (`DeviceWorkspaceAllocator`):

- `blocks` is keyed by **stream**; each stream owns exactly one `WorkspaceBlock`.
- `malloc` grows the block only when `size + 32` exceeds it: `aclrtFree` the old pointer (after
  `AclrtSynchronizeDeviceWithTimeout`) and re-allocate rounded up to 2 MiB (`kRoundLarge`) with
  `AclrtMallocAlign32(..., ACL_MEM_MALLOC_HUGE_ONLY)`; otherwise the block is reused as-is.
- `free()` adjusts statistics only — no device memory is returned.
- `empty_cache()` walks `blocks`, `aclrtFree`s each and clears the map.

So per stream the workspace footprint ratchets to the **largest request the process has issued on
that stream** and stays there, and only `torch.npu.empty_cache()` (or a later, larger request)
changes it. `ACL_MEM_MALLOC_HUGE_ONLY` asks the driver for large-page memory, so free memory made of small
fragments need not satisfy it — one plausible reading of the recorded "295 MiB free and a 216 MiB
request failed" (that the device was fragmented rather than simply full). It is an interpretation of
the allocation policy, not a measured result.

## The plugin side: per-`num_tokens` bookkeeping, weak refs

At `v0.20.2rc1`, `vllm_ascend/compilation/acl_graph.py` holds `GraphParams.workspaces:
dict[int, torch.Tensor]` (:262-264), seeded by `set_graph_params(capture_sizes)` (:275) and filled by
`update_graph_params_workspaces(num_tokens, workspace)` (:290, plus draft and draft-prefill
variants); `weak_ref_workspaces` (:217) converts every entry to a weak ref after a capture, so the
dict itself retains no device memory — do **not** read it as "one retained tensor per distinct
sequence length". Callers in `vllm_ascend/attention/attention_v1.py` (:738, :845, :942 — and
`context_parallel/attention_cp.py:599`, `mla_v1.py:1531`) look the entry up first and only call
`torch_npu._npu_fused_infer_attention_score_get_max_workspace` when it is `None`, so the current
step's shapes decide whether the allocator's per-stream block has to grow.

## Recorded report and the open part

`vllm-ascend#12692` (Qwen3.5-397B-A17B W8A8 + MTP, `v0.20.2rc1`, TP16 on A3,
`FULL_DECODE_ONLY`, diverse 40-80k / 20-40k-with-image / 4k load): the device climbs to a
high-water mark over ~37 min, holds it, then dies on a 216 MiB `SelfAttentionOperation` workspace
request. The thread held one ping and no maintainer mechanism.

- Verified here: allocator semantics above, the caller chain, and that the report's
  `expandable_segments` setting cannot govern this pool.
- Candidate, not established: that the ratchet over a diverse-shape load is the growth driver, or
  that the multimodal `_merge_multimodal_embeddings` step (which reshapes the following attention
  call) is the final push. No measurement supports either as a root cause.

Next evidence that would settle it, cheapest first: `torch.npu.empty_cache()` plus `npu-smi`
between load phases (does the high-water mark drop, i.e. is it the workspace pool at all?); a
fixed-sequence-length profile run of the same length (no OOM?); and the pLog lines
`NPUWorkspaceAllocator free by aclrtFree: size=<n>` (`ASCEND_LOGI` on every grow) and
`NPUWorkspaceAllocator malloc by AclrtMallocAlign32: size=<n>` (debug level) — they give the block
size at each grow without any instrumentation.

## Sleep mode: `CaMemAllocator.sleep`, and the tag that first calls `empty_cache`

A report of the shape "after `sleep` the device memory is not fully returned; repeated
`sleep`/`wake_up` grows until OOM; can `empty_cache()` be called after sleep?" is settled by one
`git tag --contains` at the reported tag, not by reading the allocator's unmap path.

- Entry points at `v0.18.0`: `NPUWorker.sleep` (`vllm_ascend/worker/worker.py:190`) sleeps
  `CaMemAllocator` with `offload_tags=("weights",)` at level 1 and `tuple()` at level 2, then logs
  `Sleep mode freed ... GiB memory` and asserts `freed_bytes >= 0`; `wake_up` re-maps and, from
  `"weights"` in the tags, re-transposes `w2_weight`/`w13_weight` weights.
- `CaMemAllocator.sleep` (`vllm_ascend/device_allocator/camem.py:177` at `v0.18.0`) only pins a CPU
  backup for the offload tags and `unmap_and_release`s everything else: **no `gc.collect()`, no
  `torch.npu.empty_cache()`** — so whatever the caching/workspace pools keep is kept across cycles.
- `e3c2d42e4` (#7709, "add gc in sleep mode", referring to upstream vLLM #15248) appends
  `gc.collect()` + `torch.npu.empty_cache()` to the end of `CaMemAllocator.sleep` (lines 207-208 at
  `v0.19.1rc1`). First tag containing it: **`v0.19.1rc1`**; `v0.18.0` and `v0.18.0rc1` do not have it.
- So for a `v0.18.0` report the deliverable is: the change the reporter is asking for *is* #7709,
  first shipped in `v0.19.1rc1`, and the reporter-confirmed workaround is calling
  `torch.npu.empty_cache()` after `sleep()` (recorded #9563: forcing reclamation is what worked).
  A reporter who says they already carry #7709 on `v0.18.0` is contradicted by
  `git tag --contains e3c2d42e4` — say which tag you checked instead of dropping the report.
- Later commits on the same path: `b6de3df77` (#13757, "Fix CaMem sleep race with in-flight graph
  replay") first in `v0.27.1rc1`; `cfc40b79b` (#18338, "Add synchronize NPU in sleep and wakeup") only
  on `releases/v0.30.0` and first in `v0.30.0`; and two commits no tag contains —
  `23e286f9f` (#17766, the `releases/v0.23.0` cherry-pick of the same fix) and `f558b58c1` (#17666,
  "Synchronize NPU worker device", on `main` and `releases/v0.31.0rc`). Check
  `git branch -a --contains` as well as `git tag --contains` before promising a release contains
  one; a branch-only cherry-pick is not an upgrade path.
- The residual "~N MiB re-appears on `wake_up`, probably workspace" part is the allocator section
  above: `NPUWorkspaceAllocator` never returns a per-stream block on its own, and graph-captured
  workspaces are weak-referenced (`vllm_ascend/compilation/acl_graph.py:213` `weak_ref_workspaces`
  at `v0.18.0`), so a leftover that survives `sleep` is consistent with workspace held outside the
  CaMem pool. Candidate, not measured: ask whether the residual drops when `torch.npu.empty_cache()`
  is called between cycles, and compare `npu-smi` before/after — that separates fragmentation from a
  leak.
