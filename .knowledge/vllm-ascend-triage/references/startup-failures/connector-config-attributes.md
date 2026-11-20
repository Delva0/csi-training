# KV-connector init raised on a model-config attribute

Lane B variant: the worker dies during `init_device` →
`ensure_kv_transfer_initialized` with a Python `AttributeError`, and the deepest
frame is inside `vllm_ascend/distributed/*connector*.py`. Nothing is loaded yet,
so no kernel, HCCL or memory theory applies.

```
worker_v1.init_device -> ensure_kv_transfer_initialized ->
  KVConnectorFactory.create_connector -> MooncakeConnector.__init__ ->
  MooncakeConnectorWorker.__init__ ->
  AttributeError: 'Qwen3VLMoeConfig' object has no attribute 'num_key_value_heads'
```

## Cause class

Multimodal HF configs are **nested**: the language-model scalars
(`num_key_value_heads`, `head_dim`, `num_hidden_layers`, `num_attention_heads`,
`vocab_size`, `sliding_window`) live on `text_config`, not on the top-level
config class. vllm-ascend connectors that read `hf_config.<scalar>` flat work for
flat configs (DeepSeek-V3, dense Qwen) and raise for every VL/MoE-VL config.
vLLM already exposes the nested-aware accessors — prefer them over hand-written
`getattr(x, "text_config", x)` chains:

- `model_config.hf_text_config` — resolved once at `ModelConfig.__init__`
  (`vllm/config/model.py:167,511` in v0.12.0)
- `get_total_num_kv_heads()` (`:1265`), `get_head_size()` (`:1233`),
  `get_total_num_hidden_layers()` (`:1359`), `get_num_layers(parallel_config)` (`:1396`)

## Procedure

1. **Match the traceback to the pinned release.** Print the named ranges with line
   numbers (`sed -n '965,1005p'` then `nl -ba …`) and compare with the numbers in
   the log. Matching frames prove the reporter runs the pinned tree, so you can
   quote and patch those exact lines. Record the match as evidence.
2. **Enumerate every site of the same shape before answering.** Fixing the one
   line in the traceback only moves the failure to the next one on the same code
   path:
   `grep -n "hf_config\.\(num_key_value_heads\|head_dim\|num_hidden_layers\|num_attention_heads\)" vllm_ascend/distributed/*.py`
3. **Separate MLA-only attributes.** `kv_lora_rank` / `qk_rope_head_dim` are read
   inside an `is_deepseek_mla` branch and are flat on `DeepseekV3Config`; they
   must not be rewritten. The text-model scalars do move.
4. **Grep the repo's own idiom** before proposing an API: `get_total_num_kv_heads`
   is already used in `vllm_ascend/distributed/kvpool/pool_worker.py`,
   `get_head_size` in `vllm_ascend/attention/sfa_v1.py` and `mla_v1.py`. A patch
   that matches its neighbours is a proposal a maintainer can accept.
5. **Offer a config-only workaround for the same versions.** A sibling connector
   that derives shapes from tensors instead of hf config lets the reporter test
   without patching: at v0.12.0rc1 `MooncakeLayerwiseConnector`
   (`vllm_ascend/distributed/mooncake_layerwise_connector.py`) uses
   `model_config.get_num_layers(...)` and reads no `hf_config` scalar, and it is
   registered under that name in `vllm_ascend/distributed/__init__.py`, so the
   reporter changes only `--kv-transfer-config`. Say that it has not been validated
   for their traffic.
6. **Name the sites that will break later but are off the reported path.**
   `vllm_ascend/ascend_config.py:120-138` reads `hf_config.num_key_value_heads`
   inside `try/except` and only when PD prefill_tp / decode_tp > 1; for a nested
   config it surfaces as `AssertionError: Can not get num_key_value_heads from
   model_config`, not as an `AttributeError`. Not the reported failure, but the
   next one if the reporter changes the tp ratio — and it tells you the same
   `hf_text_config` fix is needed there too.
7. **State the version limit.** These are one-commit exports: you cannot see
   whether main already fixed it. Say "present at <ref>; later fixes not
   verifiable from this snapshot".

## Verified case (vllm-ascend v0.12.0rc1 = `42ceaf08`, vllm v0.12.0)

Qwen3-VL-30B-A3B-Instruct, PD disaggregation with `MooncakeConnector`, vLLM
0.12.0 + vllm-ascend 0.12.0rc1 → `AttributeError: 'Qwen3VLMoeConfig' object has
no attribute 'num_key_value_heads'` at `mooncake_connector.py:972`, reached from
`:707`.

Flat sites in that file: `:328,:329` (`head_dim`), `:331` (`num_key_value_heads`),
`:443` (`num_hidden_layers`), `:508` (`head_dim`), `:511`
(`num_key_value_heads`), `:972` (the reported one). `Qwen3VLMoeConfig` comes from
transformers and keeps them on `text_config`, which is what upstream
`vllm/model_executor/models/qwen3_vl_moe.py` builds the text model from. The
scheduler-side connector path does not read the head count, which is why the log
shows exactly one failure and not two.

Scope statement to include: the support matrix
(`docs/source/user_guide/support_matrix/supported_models.md`) has a
"Prefill-decode Disaggregation" column, and at v0.12.0rc1 the Qwen3-VL-MOE row is
blank in every capability column (Qwen2.5-VL by contrast shows PD ✅ and
`max-model-len 30k`). So PD for that model is unvalidated in that release — the
defect is real, and the combination is also outside the project's tested set.
