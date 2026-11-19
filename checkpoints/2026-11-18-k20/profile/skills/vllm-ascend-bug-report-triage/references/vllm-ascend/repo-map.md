# vllm-ascend repo map: path or symbol → which question it answers

`$REPO` = `/workspace/repos/vllm-project/vllm-ascend`; `$VLLM` =
`/workspace/repos/vllm-project/vllm`. Read at a ref with `git -C "$REPO" show "$TAG":<path>`,
search with `git grep -n -F '<symbol>' "$TAG"`, list before guessing with `git ls-tree`. Release
layout drift is in `code-layout.md`; treat every path here as a search target, not a fact.

The workspace also holds ordinary clones of the rest of the stack — `Ascend/{pytorch,torchair,…}`,
`cann/*`, LMCache, Mooncake, ray, transformers — read the same way when a frame or a connector name
lands outside the two mains repos (`git -C /workspace/repos/kvcache-ai/Mooncake show <ref>:<path>`,
`git -C /workspace/repos/vllm-project/vllm grep -n <symbol> v0.7.3`). Do not enumerate `/workspace`
or `repos/` to find them; go to the owner/repo the traceback names.

## Configuration and platform

| path | answers |
| --- | --- |
| `vllm_ascend/ascend_config.py` | every `--additional-config` key, its defaults table, and the derived booleans (gating conditions such as `and use_sparse`, `and is_prefill_node`, `and is_kv_consumer`). Env-var fallbacks go through `_get_config_value(additional_config, key, ENV_NAME, default)`. First file to grep when a report quotes a config key. |
| `vllm_ascend/envs.py` | `VLLM_ASCEND_*` environment switches referenced from `ascend_config`. |
| `vllm_ascend/platform.py` | device-type gating (A2/A3/A5/310P), backend selection, cudagraph/breakable-graph decisions, what is force-disabled per hardware. Entry points in the log (`__init__.py:207 Platform plugin ascend is activated`) map here, not to model code. |
| `vllm_ascend/device/`, `vllm_ascend/device/hardware_profile.py` | per-device operator dispatch (`device_op.py` has the c8/no-c8 cache-write branches), hardware capability queries. |
| `vllm_ascend/patch/` | monkey patches onto upstream vLLM (`patch/worker/patch_qwen3_5.py` — `worker/patch_qwen3_5.py` in some refs — implements `qwen3_5_mtp_forward`; `platform/patch_kv_cache_utils.py`; `platform/patch_mamba_config.py`). |

## Quantisation

| ref | layout |
| --- | --- |
| `v0.7.3.post1` | `vllm_ascend/quantization/{quant_config.py,quantizer.py}` |
| `v0.11.0` | `vllm_ascend/quantization/{quant_config.py,utils.py,w8a8.py,w8a8_dynamic.py,w4a8_dynamic.py,w4a4_flatquant_dynamic.py}` |
| `v0.23.0rc1` | generic `vllm_ascend/quantization/methods/*.py` plus the 310P family `vllm_ascend/_310p/quantization/{modelslim_config.py,methods/{registry,w8a8_base,w8a8_dynamic,w8a8_static,w8a8s,w8a8sc}.py}`; `platform.py:202` imports `AscendModelSlimConfig310` for the side effect, so a 310P quantisation report is answered by the `_310p/` scheme, not the generic one |
| `v0.27.1rc1` | per-scheme packages under `vllm_ascend/quantization/methods/` (`w4a8/{w4a8,w4a8_mxfp4}.py`, `w4a4/…`, `w8a8/…`, `kv_cache/…`) with checkpoint configs under `configs/`; the W4A8 **linear** scheme no longer exists there (removed by `efda0446d`, #13713 — `../symptoms/quantization-load.md` §9), so `wc -l` against a `0.26` path compares two layouts |
| HEAD | `vllm_ascend/quantization/{configs,methods,method_adapters.py,quant_parser.py,quant_type.py,utils.py}`; per-checkpoint-format configs under `configs/` (`modelslim_config.py`, `fp8_config.py`, `compressed_tensors_config.py`, `modelopt_mxfp8_config.py`), per-scheme methods under `methods/`, KV-cache quantisation under `methods/kv_cache/` (TurboQuant; at HEAD `turboquant/config.py` rejects `enable_sparse_sfa_c8` together with `turboquant_4bit_nc`) |

Symbols worth grepping at any ref: `is_layer_skipped`, `quant_description`,
`get_quant_method`, `ASCEND_QUANTIZATION_METHOD`, `quant_model_description`. 310P overrides in
`vllm_ascend/_310p/quantization/`; for a device-kernel fault, read which operator and weight
layout the scheme feeds (`npu_dynamic_quant`, `npu_quant_matmul`, `maybe_trans_nz`) in
`_310p/quantization/methods/w8a8_dynamic.py` and the NZ policy in `vllm_ascend/utils.py`
(`../symptoms/device-kernel-faults.md`). Unit tests: `tests/ut/quantization/` (`test_quant_config.py`,
`test_w8a8*.py`, …).

## Attention, caches, sparse

| path | answers |
| --- | --- |
| `vllm_ascend/attention/sfa_v1.py` | sparse flash attention (DSA/SFA) backend + `AscendSFAMetadataBuilder`; KV write path (`npu_scatter_nd_update_` for sparse C8), spec-decode metadata, C8 reshape-optim hooks (at v0.23.0rc1: metadata build `:474`, write path `:1837`/`:1855`). |
| `vllm_ascend/attention/indexer.py` | Lightning Indexer cache (`AscendSFAIndexerBackend`, `enable_sparse_li_c8`), `write_cache`, C4 variant; owns the reshape-optim call site at `v0.28.0.rc1`+ (`:277`). |
| `vllm_ascend/attention/{dsa_v1,dsa_v41,mla_v1,attention_v1,flashmla,sfa_kv_offload,attention_mask}.py` | other attention backends; paged-attention (`pa_shape_list`); SFA KV offload. `attention_mask.py` owns the prefill mask — dense `[rows, max(seq_lens)]` vs the compressed 2048² int8 one, gated at older refs on `torch.version.cann`; its caller is `worker/model_runner_v1.py::_make_attention_mask`. Sizing and workarounds: `../symptoms/attention-mask/prefill-dense-mask-a2-a3.md`. |
| `vllm_ascend/attention/context_parallel/` | `sfa_cp.py`, `dsa_cp.py`, `dsa_v41_cp.py`, `mla_cp.py`, `attention_cp.py`, `common_cp.py`, `sfa_dcp_utils.py` — CP/DCP/PCP slicing, LSE merge, `pcp_prolog_global_slots`, local-range arithmetic. |
| `vllm_ascend/core/{kv_cache_interface,kv_cache_placement}.py`, `worker/block_table.py` | cache specs, cache-group placement, block tables and slot mapping. |
| `vllm_ascend/_310p/attention/{attention_v1,attention_mask,metadata_builder}.py` | 310P overrides, including the compressed-mask gate. |

## MoE and parallel groups

| path | answers |
| --- | --- |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | communication strategies per mode. Its module docstring states the order (`Attn → TP AR → DP AG → MoE → DP RS → TP AR`, plus sequence-parallel and EP variants) — read that first, then the `_prepare_with_dp_group` / `_prepare_with_ep_group` / `_finalize_*` pair: an asymmetry between pad/gather size in prepare and slice/scatter size in finalize is a recurring accuracy-bug shape. |
| `vllm_ascend/ops/fused_moe/{fused_moe,moe_comm_method,moe_mlp}.py`, `_310p/fused_moe/` | MoE kernels and per-device variants. |
| `vllm_ascend/ascend_forward_context.py` → `select_moe_comm_method` | which collective family a MoE forward uses (`ALLGATHER` / `MC2` / `ALLTOALL` / `FUSED_MC2`) as a function of `num_tokens` vs `get_mc2_tokens_capacity()`, EP world size, arch and quant type — the branch behind cross-rank collective mismatches. Its argument must be a **DP-max** token count; trace how the caller derives it (`../symptoms/pd-collective-hang.md`). |
| `vllm_ascend/distributed/parallel_state.py` | `get_pcp_group()`, DP/EP/TP groups. |
| `vllm_ascend/attention/context_parallel/*`, `ascend_forward_context.py` | where PCP state is reached at HEAD (`_EXTRA_CTX.max_tokens_across_pcp`); the old `worker/pcp_utils.py` `PCPManager` is gone. |

## Schedulers, spec decode, sampling

| path | answers |
| --- | --- |
| `vllm_ascend/core/recompute_scheduler.py` | recompute/preempt scheduling; at v0.20.2rc1 `:152` injects `[PLACEHOLDER_TOKEN_ID] * num_spec_tokens` into `spec_token_ids`. |
| `vllm_ascend/core/{dyntra_lb_scheduler,scheduler_profiling_chunk}.py` | load-balancing and chunked-prefill scheduling. |
| `vllm_ascend/sample/rejection_sampler.py`, `vllm_ascend/ops/triton/reject_sample.py` | speculative rejection sampling and placeholder masking (the placeholder-token bug class lives here). `PLACEHOLDER_TOKEN_ID` is imported from `vllm.v1.sample.rejection_sampler`. |
| `vllm_ascend/worker/v1/model_runner.py` (older refs: `worker/model_runner_v1.py`), `vllm_ascend/worker/v2/` (incl. `worker/dcp_utils.py`, `worker/v2/spec_decode/`) | model runners (MRV1 vs MRV2); input preparation, placeholder sanitising, attention metadata assembly, spec-decode bookkeeping. |
| `vllm_ascend/worker/dspark_utils.py`, `vllm_ascend/models/glm5_dspark.py`, `models/glm5next/` | DSpark draft model/aux layer handling. |
| `tools/spec_decode_metrics.py` | acceptance-rate tooling referenced by reports. |

## KV transfer / disaggregation

| path | answers |
| --- | --- |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/` | `mooncake_connector.py`, `mooncake_hybrid_connector.py`, `mooncake_layerwise_connector.py`, `mooncake_d2rh_connector.py` (HEAD only), `sfa_pd_rd2h/` (`connector.py`, `protocol.py`, `read_thread.py` from v0.26.0rc1; `dspark_context.py` at HEAD only: P-to-D draft-KV readiness, `uses_sfa_dspark_kv_transfer`, `get_pd_dspark_aux_layer_ids`). |
| `vllm_ascend/distributed/kv_transfer/kv_pool/` | AscendStore / KV pool connectors (MemCache backend). |
| `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/`, `ascend_multi_connector.py` | layerwise/sparse offload and connector composition. |

## Native operators

`csrc/` holds AscendC / AICPU operators registered as `torch.ops._C_ascend.*` with bindings in
`csrc/torch_binding_meta.cpp` and per-op directories `csrc/attention/<op>/{op_api,op_host,op_kernel_aicpu}`.
Custom AICPU tiling operators matter for host-level hangs: they must be launched in the same
process as the pooling machinery (issue #15896). When an operator is removed upstream, the Python
call site, bindings, `build_aclnn.sh` entries, tests and docs move together (#18128 is an example).

## Tests

- `tests/ut/**` — unit contracts; names encode invariants
  (`tests/ut/attention/a2/test_sfa_v1.py::test_li_c8_reshape_optim_requires_layer_li_c8`,
  `tests/ut/kv_offload/test_dspark_pd_context.py`, `test_dspark_transfer_status.py`,
  `tests/ut/core/test_kv_cache_interface.py`, `tests/ut/worker/test_pcp_manager.py`).
- `tests/e2e/pull_request/{one,two,four}_card/**` — PR-gating e2e organised by hardware count;
  model × feature matrix (e.g. `four_card/spec_decode/test_dspark_deepseekv4.py`).
- `tests/e2e/nightly/**` — nightly accuracy/perf coverage, including operator-level cases.

## Docs and recipes

| path | answers |
| --- | --- |
| `docs/source/user_guide/release_notes.md` | aggregated per-release notes: `Highlights`, `Features`, `Performance`, `Dependencies`, `Deprecation and Breaking Changes`, `Ready to Deprecate`, `Stability and Bug Fixes`, `Known Issues`. `†` marks backported entries. The `Known Issues` section is the first place to look for a reported symptom. Grep it, never print it (~300 KB). |
| `docs/source/user_guide/configuration/additional_config.md` | the flag table with defaults (flags are deleted from it when removed, e.g. sparse C8 keys are absent at HEAD). |
| `docs/source/tutorials/models/<Model>.md` | the "official configuration": prerequisites, single-node and PD commands, KV pooling, accuracy/perf eval, FAQ. Reporters copy these lines. |
| `docs/source/user_guide/feature_guide/*.md` | feature semantics: `context_parallel.md`, `kv_cache_pool_ascend_store.md`, `layerwise_and_sparse_kv_cache_offloading.md`, `mooncake_hybrid_attention.md`, `mooncake_layerwise_adaptation_and_optimization.md`, `preempt_offload_connector.md`, `quantization.md`, … |
| `docs/source/faqs.md` | public FAQ: vllm-ascend↔vllm policy, supported combinations, support boundaries (e.g. "w8a8 is supported from v0.8.4rc2"). |
| `docs/source/locale/zh_CN/LC_MESSAGES/**/*.po` | Chinese mirrors of every doc string; a second copy of doc text (noisy for pickaxe — exclude `:(exclude)**/locale/**`). |
| `release_notes/fragments/<PR>.yaml` | one fragment per user-visible PR (present at HEAD). |
| `recipes/overlays/<org>/<Model>.yaml`, `recipes/README.md` | curated deployment recipes/overlays per model. |

## Tooling notes

- The clones are read-only: no worktrees, no checkouts. Read with `git show <ref>:<path>`, search
  with `git grep -n <pat> <ref>`, materialise a tree with `git archive <ref> | tar -x -C …`.
- For worktree file lookup use `search_files (target='files')`; for ref-scoped content use
  `git grep`. Do not run either over a whole clone as a first move.
- A reported path under `/vllm-workspace/vllm/…` (or any frame in `vllm/…`) is **upstream's** file, not
  the plugin's: the image clones `VLLM_TAG` into that directory (`ARG VLLM_TAG=` in the release
  `Dockerfile`, e.g. `v0.23.0:Dockerfile:42`) and copies `vllm_ascend` over it. Read it in the `vllm`
  clone at that tag. `git grep <symbol> <vllm-ascend-ref> -- vllm_ascend` returning nothing is expected
  there and is not evidence the code is missing; a maintainer answer for that class is "fix upstream or
  backport into the image" (`../symptoms/tool-call-streaming.md` §4b is a worked case).
