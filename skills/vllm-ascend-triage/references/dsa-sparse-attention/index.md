# DeepSeek-V3.2 / DSA sparse attention on Ascend

## Scope

Reports on the DeepSeek-V3.2-Exp family (DSA / SFA — `vllm_ascend/models/deepseek_v3_2.py`,
`vllm_ascend/attention/sfa_v1.py`, the `npu_lightning_indexer` top-k selection, MLAPO):
- a device fault whose Python frame is `indexer_select` / `_sfa_preprocess` and whose
  dump names a scatter/gather kernel — `indexer-cache-scatter-fault.md`;
- "服务崩溃 / 崩溃" under concurrency or high load **with no error text pasted** —
  `concurrency-crash-with-no-log.md`;
- a `npu_kv_rmsnorm_rope_cache` `cos or sin shape is invalid` /
  `do tiling failed, ret is -1` after a healthy run, i.e. the op's attribute check refusing
  the rows this batch handed it (graph bucket, DSA-CP slice, chunked-prefill chunk) —
  `../custom-ops-runtime-faults/tiling-attribute-rejection.md`.
- an engine-start `AttributeError` on `aclnn_input_scale` / `aclnn_input_scale_reciprocal`
  / `aclnn_input_offset` at `sfa_v1.py:707` (`_init_o_proj_tp_full_params`) with
  FLASHCOMM1 or shared-expert-DP on and a W4A8/BF16 checkpoint —
  `o-proj-quant-attribute-crash.md`.
- an SFA/LI **C8** deployment (`enable_sparse_sfa_c8`, `enable_sparse_li_c8`,
  `c8_enable_reshape_optim` in `--additional-config`) whose **prefill node stops
  answering**, whose first request is garbled, or which only loses accuracy — the flag
  algebra, the two `store_kv_block*` custom ops and the pinned per-weights recipes —
  `c8-flags-and-kv-store-path.md`.
- the engine dies **at KV cache construction** with a bare
  `AssertionError:  assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0`
  (`_reshape_kv_cache_tensors`), on a DSA model with C8 and/or DCP/MTP — a page-size
  arithmetic/spec-consistency question, not memory or a card —
  `c8-page-size-reshape-assertion.md`.

Not this topic: startup failures before the engine is ready (see
`../startup-failures/index.md`), generic collective/HCCL faults
(`../hccl-and-hardware/index.md`), and a silent numerical gap with no fault
(`../accuracy-benchmarks/index.md`). PCP/DCP enabling for the same model is
`../feature-enable-recipes/context-parallel-pcp-dcp.md`; FLASHCOMM1 on a DSA model
is a DSA-CP enabler whose o_proj weight switch is W8A8-static-only —
`o-proj-quant-attribute-crash.md`.

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
- **"Turning the C8 flag off fixes it" is three flags, not one.** `enable_sparse_sfa_c8`
  and `enable_sparse_li_c8` are each `additional_config.get(...) and use_sparse`, and
  `c8_enable_reshape_optim` is gated on `enable_sparse_li_c8`. Disabling the reshape flag
  leaves LI C8 on and only reverts the indexer-cache write from
  `torch.ops._C_ascend.store_kv_block` to `npu_scatter_nd_update_`, which is what makes it
  the one-variable ablation; say that instead of walking the KV connector
  (`c8-flags-and-kv-store-path.md`).
- **`--enforce-eager` plus `ASCEND_LAUNCH_BLOCKING=1` strengthens attribution but
  does not move ownership.** Both were already set in the #4296 deployment and the
  answer was still a host-side slicing bug at the caller.
- **DSA-CP is gated on the model, not on the flag.** `enable_dsa_cp()` is
  `index_topk in hf_text_config` × `enable_sp()` (FLASHCOMM1, legacy FLASHCOMM, or
  shared-expert-DP forcing it), and on this branch `process_weights_after_loading`
  switches `o_proj` between TP and full weights. That switch assumes a W8A8-static
  o_proj, so a FLASHCOMM1 report on this family can be a quantization-combination
  failure at weight load rather than an attention or HCCL problem — read the
  traceback's `sfa_v1` frame before any attention theory
  (`o-proj-quant-attribute-crash.md`).
- **The gate's own startup form is a plain `ValueError`, not a device error.** A
  launch whose `--additional-config` carries `"enable_dsa_cp": true` while SP is
  off dies during **model construction** (before any weight load) with
  `ValueError: DSA CP requires SP to be enabled. Please enable SP(set VLLM_ASCEND_ENABLE_FLASHCOMM1=1) to use DSA CP.`
  — raised by `enable_dsa_cp()`
  when the model has `index_topk` and `enable_sp()` is false
  (`vllm_ascend/utils.py:1515-1536`, called from
  `vllm_ascend/models/deepseek_v4.py:739` at v0.23.0 `5cb98caa`) and wrapped as
  `EngineCore failed to start`. Answer: add
  `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, or put `"enable_flashcomm1": true` in
  the same `--additional-config` (`ascend_config.py:84-89`), or drop
  `enable_dsa_cp`; the tutorial's A2 block carries the export
  (`docs/source/tutorials/models/DeepSeek-V4-Flash.md:196,228`, whose notes say
  "DSA-CP depends on FlashComm1, and both options must be enabled explicitly",
  `:331`). Do not patch out the check — SP also decides how DSA-CP shards and
  communicates. Warn the reporter that the same doc block's `--async-scheduling`
  is a separate axis (spec-decode bookkeeping), so it goes on after the serve
  starts.
- **A model-family concurrency crash without a paste is a config-diff answer,** not
  a mechanism hunt: deliver the differences against the project's own validated
  command at the pinned commit, the 1-2 cheapest A/Bs, and the known-family list,
  inside the normal delivery gate. The recorded run that did not do this spent its
  whole output cap.
