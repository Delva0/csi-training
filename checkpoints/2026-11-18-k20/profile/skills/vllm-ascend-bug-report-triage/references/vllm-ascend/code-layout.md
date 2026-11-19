# Code layout per release, and the drift that burns investigations

The package moves between releases, so list the ref's tree before grepping a remembered path.
Companion to `repo-map.md` (path → question) and `version-pairing.md` (release pairing).

## `vllm_ascend/` subpackages

| ref | notable packages |
| --- | --- |
| `v0.18.0rc1` | attention, worker, ops, quantization, distributed, core, patch, spec_decode, device, eplb, kv_offload, compilation, model_loader, `_310p` — **no `models/`, no `model_executor/`** |
| `v0.21.0rc1` | adds `models/`, `model_executor/`, `profiler/`, `logger.py`, `simple_kv_offload` |
| `v0.26.0rc1` | adds `transformers_utils/`, `memcache_comm_fence.py`; still `kv_offload` |
| main (HEAD) | adds `ec_manager/`, `observability/`, `weight_switch/`, `config_utils.py`, `mrv2_utils.py`; `kv_offload/` and `simple_kv_offload/` are gone |

## Verified path drift (why you must grep at the ref)

- Quantisation: `vllm_ascend/quantization/quant_config.py` at `v0.11.0` became
  `vllm_ascend/quantization/{configs,methods,quant_parser.py}` at HEAD (full table in
  `repo-map.md`). Quantisation unit tests moved with it.
- PCP metadata: `vllm_ascend/worker/pcp_utils.py` (`class PCPManager`, tested by
  `tests/ut/worker/test_pcp_manager.py`) at `v0.18.0rc1`; at HEAD that module and class are gone
  (`git grep -ln "class PCPManager" HEAD` → nothing), and PCP state is reached through
  `vllm_ascend/attention/context_parallel/*`, `vllm_ascend/ascend_forward_context.py` and
  `vllm_ascend/ops/fused_moe/prepare_finalize.py` (`_EXTRA_CTX.max_tokens_across_pcp`). The path
  `vllm_ascend/worker/v2/pcp_manager.py` appears in some notes but has never existed — do not cite
  it.
- `vllm_ascend/models/` does not exist at `v0.18.0rc1`; model overrides lived only upstream then.
  From `v0.21.0rc1` it holds per-model files; at HEAD some grew into packages
  (`models/deepseek_v4.py` at `v0.23.0rc1` → `models/deepseek_v4/` at HEAD, alongside
  `deepseek_v41/`, `glm5next/`). A model-specific traceback path from an older report is therefore
  expected to be a *file* that no longer exists — grep the model name, do not conclude removal.
- `vllm_ascend/attention/attention_mask.py` changed shape: at `v0.11.0rc0` (through `v0.11.x`)
  `AttentionMaskBuilder(max_seq_len, dtype, device)` builds a dense `max_seq_len²` cache and selects
  the compressed `2048²` int8 splitfuse mask only when `torch.version.cann.startswith("8.3")`; at
  HEAD the class is a `@singleton` with no CANN test and `get_splitfuse_attn_mask()` returns the
  `2048²` mask unconditionally (the dense path survives only for non-causal/pooling). The un-gating
  is `c33150367` (#4779, 2025-12-09), first in `v0.12.0rc1`. Constructor arguments moved with it
  (`max_model_len` vs `max_num_batched_tokens`), which changes how much is allocated at startup —
  see `../symptoms/attention-mask/index.md`.
- KV pool / AscendStore connector: at `v0.13.0rc1` the family is `vllm_ascend/distributed/kvpool/`
  (`ascend_store_connector.py`, `pool_scheduler.py`, `pool_worker.py`, `config_data.py`,
  `kv_transfer.py`, `backend/{memcache,mooncake}_backend.py`), and the `kvpool hit tokens ... need to
  load` log line is `kvpool/pool_scheduler.py:86`. `vllm_ascend/core/pool_scheduler.py` exists at **no**
  ref (`git log --all -- vllm_ascend/core/pool_scheduler.py` is empty) — do not guess it. The package
  moved in `295018ec0` `[Refactor]Refactor of vllm_ascend/distributed module (#5719)`: from
  `v0.14.0rc1` it is `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/` (`pool_scheduler.py`
  plus a `layerwise/` variant), which is still its home at HEAD. Grep `pool_scheduler` at the ref.
- `store_kv_block` / `c8_enable_reshape_optim`: `attention/sfa_v1.py` up to ~`v0.29`, then
  `attention/indexer.py` at `v0.30.0rc*`, gone in `v0.30.0` final and at HEAD
  (`config-flag-lifetimes.md`).
- Attention backends: at `v0.18.0rc1` `vllm_ascend/attention/` has `attention_mask.py`,
  `attention_v1.py`, `mla_v1.py`, `sfa_v1.py` (`AscendSFABackend`, `AscendSFAImpl`), `utils.py`,
  `context_parallel/` (`attention_cp.py`, `common_cp.py`, `mla_cp.py`, `sfa_cp.py`). HEAD adds
  `dsa_v1.py`, `dsa_v41.py`, `indexer*.py`, `sparse_flash_mla.py`, `fa_v1.py`/`fa3_v1.py`,
  `attention_c8_mxfp.py`, `sfa_kv_offload.py`, and `context_parallel/` gains `dsa_cp.py`,
  `dsa_v41_cp.py`, `sfa_dcp_utils.py`. `fa3_v1.py` is reached only through
  `attention_config={"backend": "FLASH_ATTN"}` **plus** `VLLM_BATCH_INVARIANT=1` and an installed
  `flash_attn_npu_v3` (`platform.py:804-857`; without the env var the gate silently leaves the registered
  backend in place), and it indexes the batch twice — token-indexed `query`, request-indexed
  `block_tables`/`query_start_loc`/`seq_lens` — which is where the mixed prefill+spec-decode crash lives
  (`../symptoms/fa3-backend.md`).
- Disaggregation: the DSpark-era `kv_transfer/kv_p2p/sfa_pd_rd2h/` package exists from
  `v0.26.0rc1`; its `dspark_context.py` appears only at HEAD. Neither is present in v0.18-era
  trees.

## 310P

`vllm_ascend/_310p/` first exists at **`v0.14.0rc1`** (`1ffca8673`, #5776, 2026-01-17, "Add basic 310p
support. Only dense models work with eager mode now"; absent at `v0.12.0rc1` and `v0.13.0rc1`), and
from there carries the 310P overrides: `attention/{attention_v1,attention_mask,metadata_builder}.py`,
`ops/`, `worker_310p.py`, `model_runner_310p.py`, and later `fused_moe/` and `quantization/` (incl.
`modelslim_config.py`, first tag `v0.15.0rc1`, `39e77fb9e`/#6454 "[Feat.]: support 310p w8a8"),
`sharded_state_loader_310p.py` (`v0.16.0rc1`, `5899438a8`/#6878) and `spec_decode/` (`v0.21.0rc1`,
`969baeda8`/#10309, Qwen3.5 MTP + graph mode). Do not read a later subpackage as present at
`v0.14.0rc1`: the tag's tree is the list (`git ls-tree --name-only <tag>:vllm_ascend/_310p`), and
`hardware-support-by-release.md` carries the timeline and `../symptoms/310p-capability.md` the 310P failure classes. Before that tag there is **no** 310P branch in
the tree at all: 310P was delivered as the separate image `quay.io/ascend/vllm-ascend:<version>-310p` that
`docs/source/tutorials/single_node_300i.md` prescribes (the file exists from at least `v0.9.1rc1`), so
a 310P report on a 0.9/0.10/0.11/0.12/0.13 tag must be answered by *image and CANN generation*, not
by "read the 310P override" — there is none to read, and mainline code paths dispatch to ATB/aclnn
ops that build may not carry (`../symptoms/graph-capture-and-custom-ops.md` §B). At HEAD the device
family is selected through `vllm_ascend/device/{hardware_profile,device_op,device_config}.py`
(`Ascend310PDeviceAdaptor`, `is_310p()`), and `_310p/worker/v2/` exists. On any ref that has `_310p/`,
a 310P report must be checked against the 310P override first — it diverges from the 910 paths.

## Docs that ship with the code

`docs/source/faqs.md`, `docs/source/tutorials/models/<Model>.md` (per-model deployment + FAQ),
`docs/source/user_guide/release_notes.md` and, at HEAD, `release_notes/fragments/<PR>.yaml`. Read
them at the reporter's ref: they state support boundaries such as "w8a8 is supported from
v0.8.4rc2" or "run DeepSeek with BF16 for now". A feature documented for release N may not exist
in release N-2 — check `git ls-tree` at the tag before assuming the flag was honoured.
