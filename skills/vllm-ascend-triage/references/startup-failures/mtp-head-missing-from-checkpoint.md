# `ValueError: Following weights were not initialized from checkpoint` on an MTP drafter

## Signature

The engine dies while loading the **drafter**, not the target model, and the
message is a completeness check rather than a failed tensor read:

```
ValueError: Following weights were not initialized from checkpoint: {
  'model.pre_fc_norm_hidden.weight', 'model.pre_fc_norm_embedding.weight',
  'model.fc.weight', 'model.layers.0.input_layernorm.weight',
  'model.layers.0.self_attn.q_proj.weight', …, 'model.norm.weight'}
ERROR … [core.py] EngineCore failed to start.
```

- the traceback runs `executor/uniproc_executor.py __init__` → `worker/worker.py
  load_model` → `worker/model_runner_v1.py load_model` → `self.drafter.load_model(…)`
  → `spec_decode/llm_base_proposer.py load_model` → `_get_model` → upstream
  `get_model(...)` → `model_loader/default_loader.py`, and ends at the `raise`
  inside `track_weights_loading`;
- the missing set is the **whole MTP module**: `pre_fc_norm_hidden`,
  `pre_fc_norm_embedding`, `fc`, `norm` and `layers.0.*` — note `layers.0` is the
  *drafter's* first layer, not the trunk's, so this reads as "the head is not in
  the file", not "one tensor is misnamed";
- the launch carries `--speculative-config '{"method": "qwen3_5_mtp"|"mtp", …}'`,
  the checkpoint is a **fine-tune export**, and the base checkpoint of the same
  model starts fine with the same command;
- nothing in the log says "file not found", "unexpected key" or "shape mismatch".

## Why config alone enables it (v0.23.0)

| Anchor (vllm-ascend `5cb98ca` = v0.23.0, vllm `0fc695f` = v0.23.0) | Fact |
|---|---|
| vllm `config/speculative.py:460-471` | `model_type` `qwen3_5`/`qwen3_5_moe` → draft `model_type` `qwen3_5_mtp`, `n_predict = mtp_num_hidden_layers`; **read from `config.json`, never from the shards** |
| `config/speculative.py:550-553` | every `MTPModelTypes` alias (`qwen3_5_mtp`, `glm4_moe_mtp`, `deepseek_mtp`, …) is rewritten to `method = "mtp"` with a deprecation warning |
| `config/speculative.py:565` (+ `:567-570`) | for `mtp` the draft model path **is the target's path** (`self.model = self.target_model_config.model`) and it inherits the target's quantization → the drafter reads the same shards |
| vllm-ascend `patch/platform/patch_speculative_config.py:106`, assigned at `:137` | the downstream `hf_config_override` performs the same rewrite; it sees only the `hf_config` |
| vllm `models/qwen3_5_mtp.py:59-118` | the drafter's parameters: `embed_tokens`, `fc`, `layers[0..num_mtp_layers)` (default 1 at `:73`), `norm`, `pre_fc_norm_hidden`, `pre_fc_norm_embedding` |
| `models/qwen3_5_mtp.py:440-455` | `Qwen3_5MTP.load_weights` remaps **only** checkpoint keys starting `mtp.` (→ `model.`) plus `embed_tokens`/`lm_head`, and **skips every other key** → for this family the head is exported under the `mtp.` prefix |
| `model_loader/default_loader.py:403-412`, `:414-434` | tracking is on only when `model_config.quantization is None` **and** `load_weights` returned a set; `weights_to_load - loaded_weights` → the `ValueError` at `:434` |

So the decision to enable MTP is made entirely from `config.json`, and the first
thing that looks at the key set is the post-load tracking check. That is why the
report's own wording — "the framework tried to load a tensor that does not exist"
— is close but not the mechanism: nothing was attempted for that module.

**Why the base model works and the SFT export does not**: the training/export
class was the trunk (no MTP submodule in its `state_dict`), so the tensors were
never written, while `config.json` was copied verbatim and still carries
`mtp_num_hidden_layers` (or `num_nextn_predict_layers`).

**The quantized look-alike hides it**: with `--quantization ascend` / w8a8 the
tracking check is skipped (`:403-405`), so the same checkpoint *starts* and serves
with a randomly initialised head — the symptom becomes `Accepted: 0`,
`acceptance length: 1.00`, degenerate continuations
(`acceptance-collapse.md`), not a `ValueError`. Ask which checkpoint (quantized or
not) before deciding which of the two shapes the reporter is in.

## One check

```sh
uv run --no-project python <skill_root>/scripts/mtp-head-check.py <model dir>
```

It reads `model.safetensors.index.json` (or each shard's header), then looks for
the three head layouts it knows — `mtp.*` keys (Qwen3.5/Qwen3-Next), layer ids
`>= num_hidden_layers` (DeepSeek family), and the `enorm`/`hnorm`/`pre_fc_norm*`
norms — and prints PRESENT/ABSENT plus the quantized branch above. Give the
reporter the command *and* its verdict; a bare "your weights are missing" is not
checkable on their side.

## Answer shape

1. **Cause**: the checkpoint carries no MTP tensors; the drafter is built from
   `config.json` alone and the loader's completeness check raises. Cite the
   reporter's own missing set as the evidence that the *whole* module is absent.
2. **Immediate**: drop `--speculative-config`, or re-export with the head. Do
   **not** offer "delete `mtp_num_hidden_layers` from `config.json`" as a
   mitigation: the draft class defaults `num_mtp_layers` to 1
   (`qwen3_5_mtp.py:73`), so a drafter is still built from the same path and still
   fails (code reading, no run).
3. **Ownership**: the enable decision is `SpeculativeConfig`'s, from `config.json`;
   a warn-and-disable pre-check is a behaviour change the maintainers own. Name
   the two candidate sites rather than patching by hand: vllm-ascend
   `platform.py:422-432` (`_is_mtp_speculative_config`, currently only
   `"mtp" in method.lower()`) at config time, or upstream next to the alias
   rewrite at `speculative.py:550`. The scan above is the implementation of the
   check either way.
4. **Version correspondence**: compare the paste's *symbol names*, not just line
   numbers. #15201's frames matched the pin line for line except two names —
   `self.get_model()` at `:325` against the pin's `self._get_model()` (`:296`,
   called at `:325`) and `model.pre_fc_norm.embedding.weight` against
   `pre_fc_norm_embedding` — so ask for the exact commit instead of "correcting"
   the report from the tag.

## Scope

Verified at vllm-ascend **v0.23.0** (`5cb98ca`) + vllm **v0.23.0** (`0fc695f`).
Re-grep after an upgrade: the alias list, the draft-model path and the tracking
gate all move between releases, and the on-disk key prefix is
**family-specific** (`mtp.` for Qwen3.5, extra trunk layer for DeepSeek). Not
established here: whether a warn-and-disable patch exists upstream at any later
ref, and whether a checkpoint with `mtp.*` keys but a *different*
`mtp_num_hidden_layers` than `config.json` behaves (the loader would remap more
keys than the module has and drop the excess).
