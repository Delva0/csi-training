# OffloadingConnector + MLA: `KeyError` in `register_kv_caches` (layer-type coverage)

## Scope

Engine start on a **CPU-offload** deployment: `Creating offloading spec with name:
NPUOffloadingSpec`, then `WorkerProc hit an exception` with
`KeyError: 'language_model.model.layers.0.self_attn.attn'` raised in
`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` at the
`attn_backends` dict comprehension, through `vllm_ascend/worker/model_runner_v1.py
initialize_kv_cache`. The reporter usually asks three things at once: is MLA not
adapted, is it vllm's or vllm-ascend's problem, and does every MLA model fail.
Checked at vllm `v0.17.0` @`b31e9326`, vllm-ascend `v0.17.0rc1` @`e20f0b1a` (case
#7971); the reporter's own traceback already proves the crash, so answer it in the
first paragraph and spend the rest on scope.

## Mechanism (3 reads, in this order)

1. `register_kv_caches` (`offloading_connector.py:594-604`) calls
   `get_layers_from_vllm_config(self.spec.vllm_config, Attention, layer_names)`.
   `get_layers_from_vllm_config` (`vllm/config/vllm.py:1764-1787`) keeps only entries of
   `static_forward_context` that satisfy `isinstance(obj, layer_type)`. `Attention`
   (`vllm/model_executor/layers/attention/attention.py:176`) and `MLAAttention`
   (`…/attention/mla_attention.py:267`) are **sibling** subclasses of
   `AttentionLayerBase`, not parent/child, so the filter drops every MLA layer; the dict
   is empty and the comprehension raises on `layers[layer_name]`.
2. Those MLA objects are genuinely the KV-cache layers: `MLAAttention` registers itself
   at `static_forward_context[prefix]` (`mla_attention.py:385`) and implements
   `get_attn_backend` / `get_kv_cache_spec` (`:783`, `:786`) — the only two methods the
   connector wants. So the defect is the filter's class, not an unsupported model.
3. The trigger is the layer **class**, not "MLA" as a model label:
   - works — plain `Attention`: `glm4_moe.py:293` (`Glm4MoeForCausalLM`,
     `registry.py:117`), i.e. the GLM-4.x that succeeded in the report.
   - fails — through `MultiHeadLatentAttentionWrapper` (`layers/mla.py:34`), which builds
     `MLAAttention` at `prefix=f"{prefix}.attn"` (`:95-105`): `deepseek_v2.py:1026-1031`
     selects `DeepseekV2MLAAttention` for the MLA path, which is what `GLM-5`
     (`GlmMoeDsaForCausalLM`, `registry.py:119`) and `Kimi-K2.5`
     (`kimi_k25.py:390` instantiates `DeepseekV2ForCausalLM` under
     `language_model`, `:328` maps `language_model.layers.` →
     `language_model.model.layers.`) both reach — hence the traceback's key.
   - also excluded — any `AttentionLayerBase` that is not `Attention` and still owns KV,
     e.g. `MambaBase` (`vllm/model_executor/layers/mamba/abstract.py:15`) on hybrid
     GDN/linear-attention checkpoints.

## Ownership: vendor drift or upstream gap

The class swap does not create the failure. Upstream builds the same inner layer:
`vllm/model_executor/layers/mla.py:95-105` creates `MLAAttention(prefix=f"{prefix}.attn")`
unconditionally, and vllm-ascend only replaces the outer class by name
(`vllm_ascend/utils.py:651` → `AscendMultiHeadLatentAttention`,
`vllm_ascend/ops/mla.py:67`) which mirrors it (`:100-111` registration `:137-139`). Same
key, same hierarchy on CUDA → the failing filter is upstream's, and the fix belongs there
(`Attention` → `AttentionLayerBase`, as other call sites already do). Three greps settle
it; do not answer "downstream owns it" just because the traceback passes through an
upstream file class vllm-ascend also replaces — compare the two builders first.

## Second gap that decides the workaround

Upstream's handler derives each layer's layout from the backend map it just built
(`vllm/v1/kv_offload/worker/cpu_gpu.py:237-310`: `attn_backends[layer_name]`,
`split_k_and_v`, unbind along the K/V dim). The downstream handler does not:
`CpuNpuOffloadingHandler.transfer_async` unpacks `src_tensor[0], src_tensor[1]` as K/V
(`vllm_ascend/kv_offload/cpu_npu.py:99,141`) and ignores `attn_backends`, while the
Ascend MLA cache is one tensor `(num_blocks, block_size, num_kv_heads, head_size)`
(`vllm_ascend/attention/mla_v1.py:84-85`) — dim 0 is the block id. So a filter-only patch
would move wrong slices silently. The honest verdict on this ref: MLA + offloading is not
adapted end-to-end (e2e keeps only `Qwen/Qwen3-0.6B`,
`tests/e2e/singlecard/test_cpu_offloading.py:161`), and a local monkeypatch of the filter
is a two-file fix or nothing.

## Answer shape

1. Yes, the crash is real on that image (their traceback), and it is upstream's
   `Attention` filter — reproducible on CUDA for the same checkpoint, so not a vllm-ascend
   defect.
2. The criterion is the class registered in `static_forward_context` for the KV-cache
   layer, not "MLA": list works (`Attention`: GLM-4.x, Qwen dense) vs fails
   (`MLAAttention` via the wrapper: DeepSeek-V3.x/V4, GLM-5, Kimi-K2.x, LongCat; and
   `MambaBase` hybrids).
3. One-line probe for their box: iterate the `kv_caches` keys and print
   `type(static_forward_context[name]).__name__`; every `MLAAttention` entry is a fail.
4. Action ladder: don't enable `OffloadingConnector` for those checkpoints on this ref;
   if patching, change the filter to `AttentionLayerBase` **and** make the downstream
   handler consume `attn_backends` (or port upstream's `cpu_gpu.py` handler); a version
   bump only counts if the newer ref's filter can be shown to be `AttentionLayerBase`.
