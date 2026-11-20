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
accompanies the OOM.

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
   defect (`kv-init-and-format-cast-alloc.md`).
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

Conditional routes:

- Any memory-growth or "VRAM not released then crash" report →
  `hbm-growth-and-oom.md` (the split, the allocator environment gate, the KV /
  profile-peak arithmetic, the dead-key check, the precedent family).
- The crash instead happens during `profile_run`/`determine_available_memory`
  before the server is ready → `../startup-failures/index.md`
  (`max-model-len-budget.md`) — that is a configuration budget, not a leak.
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
