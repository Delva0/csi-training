# DeepSeek-V3.2 / DSA sparse attention on Ascend

## Scope

Reports on the DSA / SFA sparse-attention families — DeepSeek-V3.2-Exp
(`vllm_ascend/models/deepseek_v3_2.py`), DeepSeek-V4 and GLM-5/5.1
(`model_type: glm_moe_dsa`, same `vllm_ascend/attention/sfa_v1.py`, the
`npu_lightning_indexer` top-k selection, MLAPO):
- a device fault whose Python frame is `indexer_select` / `_sfa_preprocess` and whose
  dump names a scatter/gather kernel — `indexer-cache-scatter-fault.md`;
- "服务崩溃 / 崩溃" under concurrency or high load **with no error text pasted** —
  `concurrency-crash-with-no-log.md`;
- a DCP (`--decode-context-parallel-size` > 1) × C8 deployment whose **first real
  request** dies with `MTE accesses an invalid GM address` (`0x800000`) in a
  KV-reading kernel under `FULL_DECODE_ONLY`, and a benchmark client that only
  reports `ConnectionResetError` afterwards — `dcp-c8-kv-address-fault.md`;
- the same DCP (`--decode-context-parallel-size` > 1) × `enable_sparse_sfa_c8`
  deployment that starts cleanly and returns **garbled output with no fault**, reported
  as a config matrix ("dcp=8 乱码 / dcp=16 正常 / dcp=8 且 sfa_c8=false 正常") — the
  `dcp == tp` gate, the skipped interleave override, the pin's "not recommended" line
  and the in-tree coverage — `dcp-c8-accuracy-garble.md`;
- a `npu_kv_rmsnorm_rope_cache` `cos or sin shape is invalid` /
  `do tiling failed, ret is -1` after a healthy run, i.e. the op's attribute check refusing
  the rows this batch handed it (graph bucket, DSA-CP slice, chunked-prefill chunk) —
  `../custom-ops-runtime-faults/tiling-attribute-rejection.md`.
- an engine-start `AttributeError` on `aclnn_input_scale` / `aclnn_input_scale_reciprocal`
  / `aclnn_input_offset` at `sfa_v1.py:707` (`_init_o_proj_tp_full_params`) with
  FLASHCOMM1 or shared-expert-DP on and a W4A8/BF16 checkpoint —
  `o-proj-quant-attribute-crash.md`.
- the first request dies at `torch.ops._C_ascend.npu_sparse_flash_attention`
  (`sfa_v1.py:1049`, `AttributeError: '_OpNamespace' '_C_ascend' object has no
  attribute …`) on an **A5/950** host or with `VLLM_BATCH_INVARIANT` exported — the
  namespace is empty because the device-type/BI gate never imports the extension, so
  no op-specific build question exists; the three-gate verdict (device gate →
  `build_aclnn.sh` branch → `AddConfig` list) is
  `../custom-ops-runtime-faults/custom-ops-disabled-by-device-type.md`.
- the same op with the tiling's **two-number** shape refusal —
  `sparse indices layout is TND, shape is [4996, 1, 2048], expected shape is
  [512, 1, 2048]` + `SparseFlashAttention do tiling failed, ret is -1` — on a serve
  with `enable_dsa_cp: true`, where the two T values are a per-rank query row count and
  a metadata/budget row count (the index-cache branch at `sfa_v1.py:1328` slices
  `topk_indices_buffer` by `num_input_tokens`): `sfa-sparse-indices-row-mismatch.md`.
- an SFA/LI **C8** deployment (`enable_sparse_sfa_c8`, `enable_sparse_li_c8`,
  `c8_enable_reshape_optim` in `--additional-config`) whose **prefill node stops
  answering**, whose first request is garbled, or which only loses accuracy — the flag
  algebra, the two `store_kv_block*` custom ops, why a command that never sets
  `c8_enable_reshape_optim` is **already on the fallback scatter write** (so that
  custom-op surface is not evidence about its fault) and the pinned per-weights recipes —
  `c8-flags-and-kv-store-path.md`. When the accuracy loss is a **comparison against
  another quantized checkpoint** ("w8a8c8 below w4a8c8"), the attribution order (control
  row, recipe/node-count confound, the lane's 2-3 point floor, model validation) is
  `../accuracy-benchmarks/quantized-sibling-score-gap.md`, which defers to Rule 3 here for
  the flags.
- the engine dies **at KV cache construction** with a bare
  `AssertionError:  assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0`
  (`_reshape_kv_cache_tensors`), on a DSA model with C8 and/or DCP/MTP — a page-size
  arithmetic/spec-consistency question, not memory or a card —
  `c8-page-size-reshape-assertion.md`.
- a DSv4 / hybrid **multi-group** serve with `--enable-prefix-caching` whose
  EngineCore dies at the scheduler step inside the **prefix-cache lookup**
  (`kv_cache_manager.get_computed_blocks` → `coordinator.find_longest_cache_hit`),
  nothing device-side, paste often cut at a `@@@` marker — the ownership and the
  contract exclusions at the pin are
  `../scheduler-admission/prefix-cache-lookup-crash.md` (recorded #16760, DSv4
  Vision on 910B3 with `--block-size 32`); the compressed-group page-size twin is
  `c8-page-size-reshape-assertion.md` below.
- a `glm_moe_dsa` (GLM-5/5.1) **w4a8** serve that starts cleanly and emits repeated junk
  from the first token, on a **newer image tag** whose older tag answered the same
  weights correctly, while a plain-text prompt on the broken tag is correct — a
  per-model branch (`sfa_v1.py` `model_type == "glm_moe_dsa"`), the W4A8 shared-expert
  overlap stream, and a launch diff against the tutorial's own multi-node w4a8 row:
  `glm-moe-dsa-tag-regression-corruption.md`.

Not this topic: startup failures before the engine is ready (see
`../startup-failures/index.md`), generic collective/HCCL faults
(`../hccl-and-hardware/index.md`), and a silent numerical gap with no fault and no
DCP/C8 flag in the command (`../accuracy-benchmarks/index.md`). **With DCP > 1 and
`sfa_c8` on, an accuracy-only or 乱码 report is this topic's
`dcp-c8-accuracy-garble.md`, not the accuracy topic.** PCP/DCP enabling for the same model is
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
- **A `CompareShape` refusal that prints both Ts is an arithmetic check, not a source
  hunt.** The first `T` is the tensor the caller passed, the second the query rows the
  tiling derived, so divide each by the flags you were given (`tp`,
  `--max-num-batched-tokens`) and stop: on #11080 `512 = 4096/8` was a per-rank count
  while `4996` was not a multiple of it and exceeded the token budget, i.e. two
  independent counts, one local and one from metadata. No `pcp`/`dcp` factor explains
  such a pair, and the tiling source is in no export —
  `sfa-sparse-indices-row-mismatch.md` carries the owner table and the op-spy.
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
- **A fault in a KV-reading kernel under DCP is an index-space question, and the
  recipe row is checked first.** With `--decode-context-parallel-size` > 1 the
  metadata handed to the attention op is a *derived* view
  (`sfa_cp.py` `local × dcp + rank`), which no code compares against the KV cache's
  capacity; #14320 is reporter-confirmed for `LightningIndexerQuant`, and the same
  class faults the SFA C8 kernel (#15906). Before the mechanism, match the serve
  command against the pinned recipe row and its validation statements — the 1M
  block is A3-only (`release_notes.md:259`) — one launch at a smaller
  `--max-model-len`, then `enable_sparse_sfa_c8: false`, then DCP off
  (`dcp-c8-kv-address-fault.md`).
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
- **A per-token KV footprint on this family is a sizing question, not a dtype
  theory.** When the measured bytes/token/card are several times the reporter's own
  arithmetic from the compressed latent layout, the missing terms are the indexer K
  and indexer-scale caches and the SWA/compressor pages — each with a dtype that
  flips by SoC — plus a DCP replication factor, and the context target is answered
  by the tutorial's per-SoC envelope: `../deploy-sizing/dsv4-kv-footprint-terms.md`.
- **A model-family concurrency crash without a paste is a config-diff answer,** not
  a mechanism hunt: deliver the differences against the project's own validated
  command at the pinned commit, the 1-2 cheapest A/Bs, and the known-family list,
  inside the normal delivery gate. The recorded run that did not do this spent its
  whole output cap. The one code fact worth carrying is the one that **orders** the
  A/Bs — which candidate covers *both* reported load cases with a single mechanism;
  on the V3.2 shape that is the MLAPO token-gate arithmetic in
  `concurrency-crash-with-no-log.md`, not a walk through the model's forward.
- **A corruption report on this family gets two evidence steps before any source read,
  and only one of them is free.** Match the launch against the pinned tutorial row for
  that exact topology (the GLM-5 page ships a multi-node w4a8 row, so "off-matrix" is
  usually several named rows, not a guess), and buy the content-vs-numerics split with
  **one** request: a plain-text prompt padded to the failing prompt's own
  `usage.prompt_tokens`. When the reporter's control differs in token content *and*
  length, that single request is the discriminator — the recorded #10984 run spent its
  turns on code rows and offered it third
  (`glm-moe-dsa-tag-regression-corruption.md`).
