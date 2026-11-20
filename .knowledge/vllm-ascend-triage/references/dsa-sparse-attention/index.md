# DeepSeek-V3.2 / DSA sparse attention on Ascend

## Scope

Reports on the DeepSeek-V3.2-Exp family (DSA / SFA — `vllm_ascend/models/deepseek_v3_2.py`,
`vllm_ascend/attention/sfa_v1.py`, the `npu_lightning_indexer` top-k selection, MLAPO):
- a device fault whose Python frame is `indexer_select` / `_sfa_preprocess` and whose
  dump names a scatter/gather kernel — `indexer-cache-scatter-fault.md`;
- "服务崩溃 / 崩溃" under concurrency or high load **with no error text pasted** —
  `concurrency-crash-with-no-log.md`.

Not this topic: startup failures before the engine is ready (see
`../startup-failures/index.md`), generic collective/HCCL faults
(`../hccl-and-hardware/index.md`), and a silent numerical gap with no fault
(`../accuracy-benchmarks/index.md`). PCP/DCP enabling for the same model is
`../feature-enable-recipes/context-parallel-pcp-dcp.md`.

## Rules

- **The indexer cache is written from Python, so its faults are index bugs before
  they are hardware.** `kv_cache[2]` (the DSA indexer K cache) is filled by
  `torch_npu.npu_scatter_nd_update_` in `sfa_v1.py`, and the same function also
  writes the latent cache through `npu_kv_rmsnorm_rope_cache`. When the paste's
  dump says `vector core exception` / `The DDR address of the MTE instruction is
  out of range` with **one `pc start` repeated on every core**, and the fault kernel
  is `ScatterNdUpdate_…`, the question is which index tensor the call passed and
  how many rows it had — not which card is bad. A bad card does not depend on batch
  composition.
- **Batch composition is the free discriminator.** These faults follow mixed
  prefill+decode batches, chunked prefill boundaries and CP padding, so the
  confirmation is a run with one request in flight versus the failing concurrency;
  it costs one launch and no code. Do not ask for a plog to *decide* it — the plog
  only confirms the kernel name and chip id the paste already carries.
- **`--enforce-eager` plus `ASCEND_LAUNCH_BLOCKING=1` strengthens attribution but
  does not move ownership.** Both were already set in the #4296 deployment and the
  answer was still a host-side slicing bug at the caller.
- **A model-family concurrency crash without a paste is a config-diff answer,** not
  a mechanism hunt: deliver the differences against the project's own validated
  command at the pinned commit, the 1-2 cheapest A/Bs, and the known-family list,
  inside the normal delivery gate. The recorded run that did not do this spent its
  whole output cap.
