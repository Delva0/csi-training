# Memory: growth/OOM after a healthy period, and allocation failure before ready

Scope, part 1: the service came up, served requests, and later died or degraded on
memory — "跑了 X 时间后显存不释放，最后崩溃", "fails to release VRAM, eventually
crashes", `NPU out of memory`, `torch.OutOfMemoryError` /
`Memory_Allocation_Failure(EL0004)` on a long-running server, HBM pinned at
100 % in `npu-smi`, a P node that keeps KV it should have released.

Scope, part 2: the engine never became ready because an allocation failed —
`Memory_Allocation_Failure(EL0004)` / `alloc device memory failed, runtime result
= 207001` while the KV cache or the model weights are being put on the device.
Same decision (capacity vs growth vs dead key), different phase and different
cheapest test (a smaller `--max-model-len` instead of two timestamped samples).

Scope, part 3: the **host** ran out, not the card — the reporter says the machine
crashed (ssh lost, box unusable) or pastes a `top`/`free` with every python3 rank
holding tens of GiB RES, and the NPU log carries no device or collective error.
Host RAM is invisible to `npu-smi`, so it is the first thing a "machine crashed"
report needs a number for (`host-ram-and-per-rank-rss.md`). When there is no
`top` to sample because the box died **during engine start** — the log's last
lines are the per-rank `rank N in world size M is assigned ...` set and no
`Starting to load model` follows — the suspect is one eager host allocation, the
`max_model_len²` attention mask every worker builds
(`attention-mask-host-allocation.md`, `scripts/attn-mask-budget.sh`).

Scope, part 4: the host did **not** run out — a container-scoped counter climbed
for days while the service stayed healthy and `npu-smi` stayed flat:
`container_memory_working_set_bytes` / `container_memory_rss` /
`container_memory_cache` "持续增加，7 天增加 80G". That is host memory in a cgroup,
so the device-side knobs in the report's environment block are irrelevant and the
deliverable is which process class and which term (anonymous RSS vs page cache vs
allocator retention) owns the rise — `container-working-set-growth.md`.

Not an accuracy or corruption report, and not a collective error even when the
crash text names one (`HcclAllGather` in a late OOM is the allocation site); go
to `../hccl-and-hardware/index.md` first only when a real collective failure
accompanies the OOM. The one HCCL form that *is* this topic: `EI0007 … Failed to
allocate resource[qp] … Memory resources are exhausted` at graph capture with a
`[Create][DestLink]` line for every peer — budget, not link
(`../hccl-and-hardware/qp-resource-exhausted-at-graph-load.md`), and note the KV
cache is allocated before capture, so a start-up that fails there already holds
its `--gpu-memory-utilization` share.

Inherit the rules in SKILL.md. Rules that decide success here:

1. **Split the class before investigating**: capacity (pool or transient peak was
   wrong from the start), growth (reserved or requested HBM rises while the
   workload is steady), or a dead configuration key (the feature the reporter
   thinks is on never engaged). Each has a different owner and a different
   one-command check; the report's narrative rarely says which it is.
2. **The crash text is an allocation site, not a cause.** An async OOM names the
   *current* operator (`current working operator name is SelfAttentionOperation`,
   `HcclAllGather` failure) — the same sync-point trap as SKILL.md rule 11. The
   cause is whatever grew; say that explicitly before the reporter patches an op.
   The startup counterpart is a `npu_format_cast` frame in KV-cache
   initialisation: it is where a second buffer was requested, not an operator
   defect (`kv-init-and-format-cast-alloc.md`); an *unmap* failure is the same
   class of statement about a release site
   (`allocator-expandable-segments-unmap-crash.md`).
3. **Ask for two timestamped samples**, not one `npu-smi` dump: monotonic growth
   and a flat near-full pool look identical in a single sample.
4. **Check flag liveness before attributing behaviour to a flag.** On pre-migration
   refs vllm-ascend silently ignores unknown `--additional-config` keys (since the
   `@config` migration a pydantic `ValidationError` rejects them before any worker:
   `../startup-failures/additional-config-schema.md`), and graph/scheduler keys
   were removed or renamed across releases; `rg -n '<key>' vllm_ascend/` at the
   pinned commit is one call and prevents a whole wrong hypothesis. A multimodal
   model's `--mm-processor-cache-gb` / `--disable-mm-preprocessor-cache` size a host
   shm buffer, so their state says nothing about HBM (`hbm-growth-and-oom.md` 4b).
5. Everything here is about the reporter's host: local `npu-smi` output is not
   evidence about their memory behaviour.
6. **A KV budget that comes out at ~0 or negative is a bookkeeping error before it
   is a small card.** When a device with free HBM hands the engine
   `Available KV cache memory: 0.09 GiB (halved for workspace)` (310P) or
   `Available memory: 0` / `-1.33GB` (the second of two co-tenant engines), the
   worker has charged usage that is not this instance's. One subtraction of the
   logged `Total non KV cache memory` against the requested budget names the
   offending term; `gpu_memory_utilization` is a **per-instance** limit
   (`vllm/config/cache.py:66-74` at v0.22.1), so "raise the ratio, it includes the
   other instance" is not the fix — `kv-budget-per-instance-vs-device-global.md`.

## Reading a v0.21+ startup memory block

Since v0.21.0 the workers profile the ACL graph pool, so a startup log carries new
lines beside `Available KV cache memory` (#10700, 2× A2, DP2 × TP8, `gmu=0.92`):

| Line | Read it as |
|---|---|
| `ACL graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.9103 without ACL graph memory profiling. … increase … to 0.9297. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.` | `--gpu-memory-utilization` no longer maps 1:1 onto KV: ~1 % of the budget goes to the graph pool. On that log the equivalent ranged `0.9101`-`0.9104` across TP ranks. |
| `ACL graph pool memory: 0.44 GiB (actual), 0.59 GiB (estimated), difference: 0.15 GiB (33.8%).` | estimate error of the pool (30-40 % of a sub-GiB number is normal here); it is the delta any KV arithmetic has to absorb, not a defect. |
| `Free memory on device (60.6/60.96 GiB) … Desired GPU memory utilization is (0.92, 56.08 GiB). Actual usage: 30.16 GiB for weights, 2.0 GiB for peak activation, 2.07 GiB for non-torch memory, 0.44 GiB for NPU graph memory. Replace gpu_memory_utilization with --kv-cache-memory=22829425254 (21.26 GiB) to fit into requested memory, or … (25.78 GiB) to fully utilize NPU free memory.` | a complete budget split plus a byte-exact alternative to gmu. Quote it instead of re-deriving weights/activation/KV with `scripts/oom-budget-split.sh`, and prefer `--kv-cache-memory` when the reporter needs a deterministic KV size. |

These lines are also the **presence test** for whether the graph pool was reserved
in *that* log: at v0.23.0rc1 neither pinned tree carries them (upstream still says
`CUDA graph …` and gates `profile_cudagraph_memory()` on `current_platform.is_cuda()`,
which is false on NPU), and vllm-ascend's `determine_available_memory`
(`worker/worker.py:527-597`) has no graph term, so a v0.23-era paste without them has no
reservation at all — capture then competes with the KV cache for the same budget
(`../startup-failures/acl-graph-capture.md`, the v0.21–v0.23 section). Absence of the
line is money; do not read it as "the profile just did not print".

Caveat for cross-version work: the same `--gpu-memory-utilization` yields a ~1 %
smaller KV on v0.21+ than on an older release, and the per-rank `21.26` / `25.78`
GiB numbers differ by ~0.02 GiB between ranks. Neither is growth.

Conditional routes:

- Any memory-growth or "VRAM not released then crash" report →
  `hbm-growth-and-oom.md` (the split, the allocator environment gate, the KV /
  profile-peak arithmetic, the dead-key check, the precedent family).
- The crash instead happens during `profile_run`/`determine_available_memory`
  before the server is ready → `../startup-failures/index.md`
  (`max-model-len-budget.md`) — that is a configuration budget, not a leak.
- The failure is a **free/unmap**, not an allocation: `unmapHandles …
  AclrtUnmapMem(… getHcclComm()) error code is 507899` with `ERR00100` /
  `EL0003`, raised in `torch.npu.empty_cache()` inside the `memory_profiling`
  window on every rank, right after graph compile → the expandable-segments
  allocator lane, `allocator-expandable-segments-unmap-crash.md` (reporter-confirmed
  #12534: `expandable_segments:False` starts; note vllm-ascend injects `True`
  unless the variable already carries another of the three keys, so "unset" is
  not "off").
- Start-up dies **at KV sizing, before any allocation fails**: `Available KV cache
  memory: … (halved for workspace)` then `ValueError: To serve at least one request
  with the model's max seq len …`, or `Available memory: 0` / `-1.33GB` on the
  **second** engine sharing one card, while `npu-smi` shows free HBM →
  `kv-budget-per-instance-vs-device-global.md`: device-global HBM charged to one
  instance, the `#7308` → `#7427` → `#7372`/`#14274` propagation chain, the
  per-instance `gpu_memory_utilization` contract, and the one-line patch.
- The failure is an allocation error inside KV-cache initialisation, with the
  deepest frame in `_convert_torch_format` / `torch_npu.npu_format_cast` and
  `507013` / SDMA / SMMU text around it → `kv-init-and-format-cast-alloc.md`
  first. The format cast is the allocation site, not the defect; a 310P host adds
  the `FRACTAL_NZ` copy and the 300I allocator prescription
  (`../soc-platform-paths/atlas-300i-310p.md`).
- "The machine crashed" / ssh lost / a `top` paste whose python3 ranks each hold
  tens of GiB RES on one host, with no NPU-side error → `host-ram-and-per-rank-rss.md`
  and `bash <skill_root>/scripts/host-ram-check.sh <paste|->` (worked: #3902,
  16 ranks × 88-100 GiB = 1472 of 2014 GiB on a 2 TB host, swap 0).
- The host died with **nothing** to sample: the log stops at the per-rank
  `rank N in world size M is assigned ...` lines, there is no
  `Starting to load model`, and `--max-model-len` is 128K+ → the eager
  `max_model_len²` host attention mask, `attention-mask-host-allocation.md` and
  `bash <skill_root>/scripts/attn-mask-budget.sh --log <serve.log> --host-gib N`
  (#2873 reporter-confirmed: 262144 at TP=8, 0.10.1rc1, 0.10.0rc1 works).
- The report is about score/accuracy dropping under memory pressure, or about
  garbled output after long runs → `../accuracy-benchmarks/index.md`.
- The paste is a **container metric** (`container_memory_working_set_bytes`,
  k8s/cAdvisor dashboard, "容器内内存持续增加") on a service that never crashed,
  with or without `npu-smi` in the body → `container-working-set-growth.md` first
  (`scripts/working-set-arithmetic.sh` splits rss from cache and converts the
  slope into bytes/request). Contrast part 3: there the box is dead and `top` is
  the artifact.
