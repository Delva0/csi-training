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
- The failure is an allocation error inside KV-cache initialisation, with the
  deepest frame in `_convert_torch_format` / `torch_npu.npu_format_cast` and
  `507013` / SDMA / SMMU text around it → `kv-init-and-format-cast-alloc.md`
  first. The format cast is the allocation site, not the defect; a 310P host adds
  the `FRACTAL_NZ` copy and the 300I allocator prescription
  (`../soc-platform-paths/atlas-300i-310p.md`).
- The report is about score/accuracy dropping under memory pressure, or about
  garbled output after long runs → `../accuracy-benchmarks/index.md`.
