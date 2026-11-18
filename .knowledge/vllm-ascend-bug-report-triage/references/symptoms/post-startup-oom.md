# OOM after a healthy start: `NPUWorkspaceAllocator`, the per-stream workspace cache, and who filled the card

## Scope

A service **started cleanly** (`/health` 200, `profile_run` peak printed, KV cache sized) and then died of
`NPU out of memory` minutes-to-hours later under load, with no Python traceback above the allocator line.
Signature (verbatim, from #12692):

```text
[rank0]:[E723 03:19:38] NPU out of memory. NPUWorkspaceAllocator tried to allocate 216.00 MiB
  (NPU 0; 61.27 GiB total capacity; 295.32 MiB free).
torch.OutOfMemoryError: ... current working operator name is SelfAttentionOperation.
[PID:4806] Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by APP module.
```

Route here when the OOM names `NPUWorkspaceAllocator` / `EL0004` / a `current working operator name` and the
report carries a *timeline* (`npu-smi` climbing to a plateau, then the crash). Not here: an OOM during weight
creation (`quantization-load.md`), an OOM while the KV cache is allocated at start (`startup-and-stall.md`
§3/§3b), an OOM on a **long prompt** from an `n²` mask (`attention-mask/prefill-dense-mask-a2-a3.md`), a
`107020`/`207001` pair whose OOM label is cosmetic (`graph-capture-and-custom-ops.md` §C2), a per-card number
that is simply `util × total` (`memory-sizing-and-parallelism.md`), or the sleep-mode `CaMem` allocator
(`sleep-mode-and-allocator.md`).

## Step 1 — the pool that failed is torch_npu's, not the fork's and not a Python leak

`vllm_ascend/` contains no `NPUWorkspaceAllocator`. The message is produced inside torch_npu, verified in the
`Ascend/pytorch` clone at `v26.1.1-pytorch2.10.0` and at HEAD (`988eaa3220`):

| what | where |
| --- | --- |
| the OOM string | `torch_npu/csrc/core/npu/NPUWorkspaceAllocator.cpp:438` (`:454` at HEAD) |
| the caller | `torch_npu/csrc/framework/utils/OpPreparation.cpp:436-440` — `NPUWorkspaceAllocator::malloc_with_stream(workspace_size, stream)` for an ACL op's workspace |
| `current working operator name is …` | `torch_npu/csrc/core/npu/NPUQueue.cpp:178-188` — the **task queue** prints the op it had enqueued |

So the operator name is where the last workspace request happened, not the owner of the memory, and it moves
with `TASK_QUEUE_ENABLE`. Read the message as *"the card was full at that instant"*, then answer **who filled
it after the profile run**.

## Step 2 — the cache is grow-only and asks for huge pages (this is the mechanism half)

Same file, `DeviceWorkspaceAllocator`: **one block per `aclrtStream`** (`blocks: flat_hash_map<aclrtStream,
WorkspaceBlock*>`, `:388`), and `malloc(size, stream)` (`:72`):

```cpp
alloc_size = size + 32
if block->size < alloc_size:                       # :89  grow only; there is no shrink branch
    ... AclrtSynchronizeDeviceWithTimeout(); aclrtFree(block->data_ptr)
    block->size = kRoundLarge * ceil(alloc_size / kRoundLarge)   # kRoundLarge = 2 MiB (:36)
    AclrtMallocAlign32(&block->data_ptr, block->size,
                       aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_ONLY)   # :132
```

`NPUWorkspaceAllocator::malloc` (`:74` at `v26.0.0-pytorch2.10.0`) reuses a per-stream block while it is
large enough, `aclrtFree`s and re-mallocs when the request is larger (`:90`; the destructor path at `:232`),
and rounds up to 2 MiB (`kRoundLarge`, `:36`). Reserved bytes therefore end at the **largest single request
per stream**, and nothing hands memory back to the card before process exit.

Three consequences, all of them reportable:

1. **A per-stream high-water mark that never comes back down.** Every distinct op/stream holds its largest
   workspace for the process lifetime, so a load that keeps asking for a bigger attention workspace
   *ratchets* chip 0 to a plateau and leaves it there — exactly the shape of #12692's table (61767 → 64148 →
   65222 MB, then 12 min flat, then OOM). Sum the streams before blaming a leak.
2. **Growth is a free + `aclrtSynchronizeDeviceWithTimeout()` + realloc**, so a workload that keeps
   increasing the requested size pays device synchronises on the hot path as well as memory.
3. **The allocation is `HUGE_ONLY`.** `free 295 MiB` versus a failing `216 MiB` request is what a
   *contiguous huge-page* refusal looks like; it does not have to mean a Python tensor leaked, and
   `expandable_segments` cannot help — this pool bypasses the caching allocator and calls
   `aclrtMallocAlign32` directly.

The message's own advice (`take a try to set the environment variable TASK_QUEUE_ENABLE=1`) is already spent
in any report whose env sets it — say so instead of repeating it.

## Step 3 — what the fork requests per shape (and one thing it does *not* do)

In the **graph** paths (`full_graph_fia` `:682`, `full_graph_fia_v2` `:825`, `full_graph_pa` `:919` at
`v0.20.2rc1`) `vllm_ascend/attention/attention_v1.py` asks torch_npu for the maximum workspace of the
fused-infer-attention / paged-attention call
(`torch_npu._npu_fused_infer_attention_score_get_max_workspace` `:738`, `…_v2_…` `:845`,
`torch_npu._npu_paged_attention_get_workspace` `:931`) and holds the tensor per `num_tokens`, so the largest
scheduled-token count in a step sizes a device buffer the process keeps. The **eager** path
(`forward_fused_infer_attention` `:1042`, the `# Get workspace from cache or calculate it if not present.`
comment at `:1080`) keeps no such Python-side tensor and lets the op request its own.

**Do not report that cache as an unbounded per-sequence-length ladder.** Verified at `v0.20.2rc1`:
`vllm_ascend/compilation/acl_graph.py:262 GraphParams` is created by `set_graph_params(aclgraph_capture_sizes)`
(`:275`) with **every** dict pre-seeded from the capture sizes, `update_graph_params_workspaces` (`:290`) only
writes `workspaces[num_tokens]`, and the same functions immediately do
`graph_params.events[num_tokens].append(event)` (`attention/attention_v1.py:767`, `:875`, `:950`) — an
unguarded index, so a `num_tokens` outside the seeded set raises `KeyError` in the same call instead of
silently adding a rung. The rungs are the configured `cudagraph_capture_sizes`, decided at startup; a
"thousands of shapes ⇒ thousands of workspaces" story needs a different owner.

The reader's line, if one is needed: `attention_v1.py:713` reads `graph_params.workspaces.get(num_tokens)`
and on a miss `:737-759` calls `torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)` and stores
the tensor through `update_graph_params_workspaces(num_tokens, workspace)`; a speculative deployment keeps
**two** such dicts (the MTP draft uses `update_draft_graph_params_workspaces` /
`get_draft_graph_params()` / `get_draft_graph_prefill_params()`), the fiav2 twin is `:842-867`, and the replay
call sites pass the cached tensor at `:540`/`:670`. At `HEAD` the same shape survives as
`vllm_ascend/attention/utils.py:223 cache_graph_workspace`, which only upgrades to the *largest* candidate for
mixed layer shapes — still one tensor per graph/batch size, still never evicted. So the per-shape cache is a
*cache, not a leak*, and its size is bounded by the capture-size buckets: it explains a ladder **within** those
buckets, not one rung per observed sequence length.

## Step 4 — the arithmetic and the evidence to ask for

Compute, never reason: `util × device total` is the KV+weights envelope; anything the workspace pool or a
runtime cache holds is **on top of** it. For #12692: `0.90 × 61.27 GiB ≈ 55.1 GiB` of weights+KV, 25.05 GiB
weights and 25.27 GiB KV reported, plus a `VLLM_ASCEND_ENABLE_FUSED_MC2=1`/`HCCL_BUFFSIZE=1024` env — the
levers are the ones that move that sum.

Ask for (cheap for the reporter, decisive here):

- `torch.npu.memory_summary()` (and a `torch.npu.memory_snapshot()` if the build has it): the workspace
  allocator's **reserved** figure separates "workspace ratchet" from "something else filled 3 GB".
- the `# NPU blocks` / `Maximum concurrency for …` lines from start-up (the KV budget actually in force) and
  `npu-smi info`'s per-pid table.
- whether the plateau survives an idle period — a caching-allocator block is reused, an `aclrtMalloc` block is
  held; and whether the same run with **one** uniform prompt length still climbs (a ratchet needs growth
  events; a leak does not).

## Levers (all cost throughput or context, none measured on hardware here)

1. Cap the largest request: `--max-num-batched-tokens` down, and prune `cudagraph_capture_sizes` to the sizes
   the traffic needs. Fewer/smaller growth events ⇒ lower per-stream high-water marks.
2. Leave headroom: `--gpu-memory-utilization` below the value that fills the card (0.90 here), so a workspace
   growth can find a huge page.
3. Drop optional memory-resident features for the test — `VLLM_ASCEND_ENABLE_FUSED_MC2=1`,
   `--enable-expert-parallel`, `--speculative-config` — one at a time; they change both the number of streams
   and the sizes requested.
4. Constrain the shape mix (one or few sequence-length buckets) — fewer pinned `num_tokens` entries and fewer
   growth events — cost is the workload you actually want; and drop the MTP draft (`--speculative-config`),
   which stops one of the two workspace dicts from filling, at the cost of the speculation speedup.
5. `--enforce-eager` is **not** a memory lever for this class. It removes the cached per-`num_tokens`
   max-workspace tensors (`attention_v1.py:738` sits in the graph path), but every ACLNN attention call still
   asks the same allocator for a workspace, so the failure can persist — it is a diagnostic, not the fix.
6. Re-exporting `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is **not** a lever: at `v0.20.2rc1`
   `platform.py:563-581` already appends it unless the user set it, `max_split_size_mb` or
   `garbage_collection_threshold`; and the variable is *forbidden* on the CaMem path
   (`device_allocator/camem.py:150-151` asserts `"expandable_segments:True" not in conf`), because the two
   allocators are mutually exclusive. Quote both lines before blaming the reporter's `export`. At
   `v0.23.0rc1` the same block runs `platform.py:761-779` — the guard at `:761-777` (`model_config and not
   model_config.enable_sleep_mode`, the three-key exclusion and the `Set PYTORCH_NPU_ALLOC_CONF=…` log
   line) — so check that ref too before implying the default was in force.
7. **A user-supplied `max_split_size_mb` is the lever that turns that default *off* — the inverse of 6, and
   the one a copied eval script hits.** The append is guarded on all three spellings, so
   `max_split_size_mb` (or `garbage_collection_threshold`) does not merely duplicate the default: it
   **suppresses** it, and the block's own comment says why (「`max_split_size_mb` or
   `garbage_collection_threshold` cannot be enabled together with `expandable_segments=True`」). A serve
   script carrying `-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` — the value the tag's own blocks use:
   `.github/workflows/schedule_vllm_e2e_test.yaml:163`, `:278`, `:392` and all four
   `docs/source/developer_guide/evaluation/using_*.md` guides (`using_ais_bench.md:32`) — therefore runs
   **without** the fork's default, while `docs/source/faqs.md:149` prescribes `expandable_segments:True` for
   OOM. So "the fork's default is not applied here" is a `verified` doc/code disagreement and a reportable
   line. A/B it by **removing the variable** (not by adding `expandable_segments`), and keep the reading
   `hypothesis`: the timeline has to show the climb (#16022 shipped only a screenshot of it), and for an
   `NPUWorkspaceAllocator` OOM neither variable reaches the workspace pool.

## Answer shape

symptom → *which pool* (torch_npu workspace, per stream, grow-only, `HUGE_ONLY`) → the report's own timeline
as the ratchet evidence → the one arithmetic number that shows the envelope was exceeded → levers above with
their cost → `status and limits`: the fork's request sites are `verified`; the *size* of the climb is
`hypothesis` with `memory_summary()` as the falsifier; nothing was reproduced on hardware.

## Worked report #12692 (`v0.20.2rc1`, 16 × 910A3, Qwen3.5-397B-A17B w8a8 + MTP, TP16, `FULL_DECODE_ONLY`)

Thread: **one comment**, `@LoganJane Please check this issue` — no lever, no diagnosis, so the reply owns the
whole localisation. The report is unusually complete (start script, env, `npu-smi` timeline, repro script),
which is why the deliverable is the arithmetic plus the pool's behaviour rather than a fix commit. Two claims
in the body to treat as *reported*, not verified: "碎片化 (no contiguous large page)" — directionally right
for `HUGE_ONLY`, but the requested size and the fragmentation are not separable without `memory_summary()` —
and "多模 image token 改序列组成 → 新 workspace shape", which is plausible (a different `num_tokens` grows the
block) but unnamed in any code line: the mm embedding frame is where the *report* observed the failure, and
the `_merge_multimodal_embeddings` frame is upstream vLLM, not the owner.

Also worth stating in the reply: `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` in that start script is
**the fork's own default**, not a tuning choice — `platform.py:563-581` appends it whenever
`model_config and not model_config.enable_sleep_mode` and none of `expandable_segments`/`max_split_size_mb`/
`garbage_collection_threshold` is present, and `device_allocator/camem.py:150-153` *asserts* it is absent
when the sleep-mode `CaMemAllocator` is used, so the two allocator paths are mutually exclusive
(`sleep-mode-and-allocator.md`).

## Recheck conditions

The workspace allocator and the fork's request sites both move: re-read
`torch_npu/csrc/core/npu/NPUWorkspaceAllocator.cpp` at the **installed** torch_npu tag (the numbers above are
from the `pytorch2.10.0` line and HEAD), and re-grep
`git grep -n "_get_max_workspace\|_get_workspace" <vllm-ascend-ref> -- vllm_ascend` before quoting a line —
`attention_v1.py`'s method names and line numbers differ per release.
