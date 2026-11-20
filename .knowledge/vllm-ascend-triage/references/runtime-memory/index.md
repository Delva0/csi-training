# Runtime memory growth / OOM after the server was healthy

Scope: the service came up, served requests, and later died or degraded on
memory — "跑了 X 时间后显存不释放，最后崩溃", "fails to release VRAM, eventually
crashes", `NPU out of memory`, `torch.OutOfMemoryError` /
`Memory_Allocation_Failure(EL0004)` on a long-running server, HBM pinned at
100 % in `npu-smi`, a P node that keeps KV it should have released. Not the
load/profile-time OOM of `../startup-failures/index.md`, not an accuracy or
corruption report, and not a collective error even when the crash text names one
(`HcclAllGather` in a late OOM is the allocation site); go to
`../hccl-and-hardware/index.md` first only when a real collective failure
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
3. **Ask for two timestamped samples**, not one `npu-smi` dump: monotonic growth
   and a flat near-full pool look identical in a single sample.
4. **Check flag liveness before attributing behaviour to a flag.** vllm-ascend
   silently ignores unknown `--additional-config` keys, and graph/scheduler keys
   were removed or renamed across releases; `rg -n '<key>' vllm_ascend/` at the
   pinned commit is one call and prevents a whole wrong hypothesis.
5. Everything here is about the reporter's host: local `npu-smi` output is not
   evidence about their memory behaviour.

Conditional routes:

- Any memory-growth or "VRAM not released then crash" report →
  `hbm-growth-and-oom.md` (the split, the allocator environment gate, the KV /
  profile-peak arithmetic, the dead-key check, the precedent family).
- The crash instead happens during `profile_run`/`determine_available_memory`
  before the server is ready → `../startup-failures/index.md`
  (`max-model-len-budget.md`) — that is a configuration budget, not a leak.
- The report is about score/accuracy dropping under memory pressure, or about
  garbled output after long runs → `../accuracy-benchmarks/index.md`.
