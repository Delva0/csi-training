# KV-connector init raised on a model-config attribute

Lane B variant: the worker dies during `init_device` / `initialize_from_config` →
`ensure_kv_transfer_initialized` with a Python `AttributeError`, and the deepest
frame is inside `vllm_ascend/distributed/*connector*.py`. Nothing is loaded yet,
so no kernel, HCCL or memory theory applies. Two stack shapes, same class:

```
# A. scheduler/worker-side connector ctor (v0.12.0rc1)
worker_v1.init_device -> ensure_kv_transfer_initialized ->
  KVConnectorFactory.create_connector -> MooncakeConnector.__init__ ->
  MooncakeConnectorWorker.__init__ ->
  AttributeError: 'Qwen3VLMoeConfig' object has no attribute 'num_key_value_heads'

# B. KV-cache registration, consumer side only (v0.14.0rc1 … v0.17.0rc1)
worker_v1.initialize_from_config -> ... -> model_runner_v1.initialize_kv_cache ->
  register_kv_caches -> KVCacheRecvingThread.__init__ ->
  AttributeError: 'Qwen2Config' object has no attribute 'head_dim'
```

## Cause class: two independent axes

1. **Nesting.** Multimodal HF configs are **nested**: the language-model scalars
   (`num_key_value_heads`, `head_dim`, `num_hidden_layers`, `num_attention_heads`,
   `vocab_size`, `sliding_window`) live on `text_config`, not on the top-level
   config class. vllm-ascend connectors that read `hf_config.<scalar>` flat work
   for flat configs (DeepSeek-V3, dense Qwen) and raise for every VL/MoE-VL
   config. vLLM exposes the nested-aware accessors — prefer them over hand-written
   `getattr(x, "text_config", x)` chains:
   - `model_config.hf_text_config` — resolved once at `ModelConfig.__init__`
     (`vllm/config/model.py:167,511` in v0.12.0)
   - `get_total_num_kv_heads()` (`:1265`), `get_head_size()` (`:1233`),
     `get_total_num_hidden_layers()` (`:1359`), `get_num_layers(parallel_config)` (`:1396`)
2. **Attribute presence — `hf_text_config` is necessary, not sufficient.** Moving a
   read to `hf_text_config` fixes nesting and still leaves the attribute to exist.
   The scalar set is *not* uniform across transformers config classes:
   `num_hidden_layers`, `num_attention_heads`, `hidden_size`,
   `num_key_value_heads` are near-universal, while **`head_dim` is optional** and
   is derived when absent (`hidden_size // num_attention_heads`). The Qwen2/Qwen2.5
   family (`model_type: qwen2`, `Qwen2ForCausalLM`) carries no `head_dim` key at
   all; Qwen3 dense, DeepSeek-V3 and most newer configs set it. So a code path that
   reads `hf_text_config.head_dim` works on the models it was written for and
   raises on the Qwen2 family with the *same* file, image and P/D split. Later
   releases often repair it without touching the config at all, by deriving the
   geometry from the registered KV tensors (see the version arc in case B).
   One look decides which axis you are on: the reporter's `config.json` paste, or
   ```bash
   python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('<ckpt>'); print(hasattr(c,'head_dim'), c.hidden_size//c.num_attention_heads)"
   ```

## Procedure

1. **Match the traceback to the pinned release.** Print the named ranges with line
   numbers (`sed -n '965,1005p'` then `nl -ba …`) and compare with the numbers in
   the log. Matching frames prove the reporter runs the pinned tree, so you can
   quote and patch those exact lines. Record the match as evidence.
2. **Name the attribute's own status before theorising.** `hasattr` on the actual
   config class (command above), or the `config.json` key list. "Nested vs flat"
   and "present vs derived" are different findings and give different fixes.
3. **Enumerate every site of the same shape before answering.** Fixing the one
   line in the traceback only moves the failure to the next one on the same code
   path:
   `grep -n "hf_config\.\(num_key_value_heads\|head_dim\|num_hidden_layers\|num_attention_heads\)" vllm_ascend/distributed/*.py`
4. **Separate MLA-only attributes.** `kv_lora_rank` / `qk_rope_head_dim` are read
   inside an `is_deepseek_mla` branch and are flat on `DeepseekV3Config`; they
   must not be rewritten, and the MLA branch also needs a *different* layout
   (`k_head_dim = kv_lora_rank`), so it must not be replaced by a generic
   `get_head_size()`.
5. **Grep the repo's own idiom** before proposing an API: `get_total_num_kv_heads`
   is already used in `vllm_ascend/distributed/kvpool/pool_worker.py`,
   `get_head_size` in `vllm_ascend/attention/sfa_v1.py` and `mla_v1.py`. A patch
   that matches its neighbours is a proposal a maintainer can accept. For a
   derived scalar the upstream accessor is the whole fix:
   `ModelConfig.get_head_size()` → `vllm/transformers_utils/model_arch_config_convertor.py:44-65`
   (`head_dim` only if not `None`, then `hidden_size_per_head`, then
   `hidden_size // num_attention_heads`).
6. **Offer a config-only workaround for the same versions.** A sibling connector
   that derives shapes from tensors instead of hf config lets the reporter test
   without patching: at v0.12.0rc1 `MooncakeLayerwiseConnector`
   (`vllm_ascend/distributed/mooncake_layerwise_connector.py`) uses
   `model_config.get_num_layers(...)` and reads no `hf_config` scalar, and it is
   registered under that name in `vllm_ascend/distributed/__init__.py`, so the
   reporter changes only `--kv-transfer-config`. Say that it has not been validated
   for their traffic. Second option, untested: extra keys in `config.json` become
   attributes on the `PretrainedConfig` instance, so adding `"head_dim": <n>` is a
   plausible no-patch fix — check with the `hasattr` probe before recommending it.
7. **Name the sites that will break later but are off the reported path.**
   `vllm_ascend/ascend_config.py:120-138` reads `hf_config.num_key_value_heads`
   inside `try/except` and only when PD prefill_tp / decode_tp > 1; for a nested
   config it surfaces as `AssertionError: Can not get num_key_value_heads from
   model_config`, not as an `AttributeError`. Not the reported failure, but the
   next one if the reporter changes the tp ratio — and it tells you the same
   `hf_text_config` fix is needed there too.
8. **State the version limit** and, when you have two snapshots of the same file,
   the version arc. These are one-commit exports: say "present at <ref>; later
   fixes not verifiable from this snapshot" — unless another pinned case in the
   workspace carries a newer tree, in which case quote both and say which ref each
   line comes from.

## Verified case A (vllm-ascend v0.12.0rc1 = `42ceaf08`, vllm v0.12.0) — nesting

Qwen3-VL-30B-A3B-Instruct, PD disaggregation with `MooncakeConnector`, vLLM
0.12.0 + vllm-ascend 0.12.0rc1 → `AttributeError: 'Qwen3VLMoeConfig' object has
no attribute 'num_key_value_heads'` at `mooncake_connector.py:972`, reached from
`:707`.

Flat sites in that file: `:328,:329` (`head_dim`), `:331` (`num_key_value_heads`),
`:443` (`num_hidden_layers`), `:508` (`head_dim`), `:511`
(`num_key_value_heads`), `:972` (the reported one). `Qwen3VLMoeConfig` comes from
transformers and keeps them on `text_config`, which is what upstream
`vllm/model_executor/models/qwen3_vlm_moe.py` builds the text model from. The
scheduler-side connector path does not read the head count, which is why the log
shows exactly one failure and not two.

Scope statement to include: the support matrix
(`docs/source/user_guide/support_matrix/supported_models.md`) has a
"Prefill-decode Disaggregation" column, and at v0.12.0rc1 the Qwen3-VL-MOE row is
blank in every capability column (Qwen2.5-VL by contrast shows PD ✅ and
`max-model-len 30k`). So PD for that model is unvalidated in that release — the
defect is real, and the combination is also outside the project's tested set.

## Verified case B (#7352, v0.14.0rc1 = `52d4acf`) — attribute presence, consumer side only

DeepSeek-R1-Distill-Qwen-1.5B (`Qwen2ForCausalLM`, `model_type: qwen2`), 910B4,
1P1D Mooncake PD, vLLM 0.14.1 + vllm-ascend v0.14.0rc1: **prefill starts, decode
dies** in `initialize_kv_cache` → `register_kv_caches` →
`KVCacheRecvingThread.__init__` with
`AttributeError: 'Qwen2Config' object has no attribute 'head_dim'`.

- The read is `mooncake_connector.py:366-367`
  (`self.k_head_dim = self.v_head_dim = self.model_config.hf_text_config.head_dim`)
  — note it **already uses `hf_text_config`**, so the nesting fix of case A was
  applied and did not help. The reporter's later `config.json` paste
  (2026-03-23) is the deciding evidence: `qwen2`, `hidden_size 1536`,
  `num_attention_heads 12`, `num_key_value_heads 2`, `max_window_layers` /
  `sliding_window` present, **no `head_dim`** → derived 1536//12 = 128.
- Numbers to cross-check in the log: the connector prints
  `block_shape: torch.Size([128, 2, 128])` from the registered tensors in the same
  init, i.e. the engine already knows `head_dim = 128` and only the config read is
  wrong. Say that; it makes the answer arithmetic instead of a story.
- **Why P survives and D does not (the fingerprint).** `KVCacheRecvingThread`
  (`:300`) is built only in the `kv_consumer` branch (`:1270`); the producer
  builds `KVCacheSendingThread` (`:184`, built at `:1256`) and that class never
  reads `head_dim` (the only reads are `:362-367` in the receive ctor and
  `:598-599,:660-663` on its buffers). A same-image P/D asymmetry at
  `initialize_from_config` is therefore evidence of a consumer-only code path, not
  of topology, rank tables or the transport — do not spend the session on
  `kv_port` / `hostNetwork`.
- Same model, same deployment: **Qwen3-1.7B works** (its config sets `head_dim`)
  while the Qwen2-arch checkpoint does not — reporter-confirmed.
- Version arc, both ends in the workspace: still present at **v0.17.0rc1** (the
  reporter's 2026-03-17 log shows the same expression at
  `mooncake_connector.py:367`, `register_kv_caches` at `:1260`), and gone by
  **v0.21.0rc1** (`80610e4`) where the connector takes the geometry from the
  registered KV tensors — `_get_kv_cache_dims_from_tensors`
  (`:981`, used at `:996,:1025`) — and `mooncake_layerwise_connector.py:1092` uses
  `model_config.get_total_num_kv_heads()`. The reporter later confirmed a
  maintainer fix (2026-04-03, "hope to see the official fix ❤️") without
  publishing the mechanism, so for a report on a ref between the two, offer the
  accessor / tensor-derived patch and ask them to check the ref.
