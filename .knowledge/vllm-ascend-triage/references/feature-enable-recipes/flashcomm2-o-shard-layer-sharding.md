# FlashComm2 × `layer_sharding: ["o_proj"]`: `AssertionError: layer_idx must be consecutive`

Scope: a worker dies **while loading weights**, before any request, with

```
File "/vllm-workspace/vllm-ascend/vllm_ascend/ops/layer_shard_linear.py", line 73, in post_process_after_loading
    assert layer.layer_idx == layer_idx, "layer_idx must be consecutive"
AssertionError: layer_idx must be consecutive
```

on a launch that both exports `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` > 0 **and** carries
`"layer_sharding": ["o_proj"]` in `--additional-config`. Verified at vllm-ascend
v0.17.0rc1 (`e20f0b1a`, the #7508 image `quay.io/ascend/vllm-ascend:v0.17.0rc1-a3`).
The assert text, the series bookkeeping and both call sites are ref-specific — re-grep
before quoting the line numbers.

## What the assert means (one read)

`SeriesMetadata.post_process_after_loading` sorts one shard series and then indexes it
positionally: `for layer_idx in range(start_layer, end_layer): layer =
self.layers[layer_idx - start_layer]` (`ops/layer_shard_linear.py:57-73`, loop `:71`, assert `:73`). With exactly
one entry per decoder layer that indexing is exact. The assert fires when a series holds
**two entries for the same `layer_idx`**: after the sort the list is `[0,0,1,1,…]`, so
at `layer_idx=1` the slot still says `0`. So the message is "one linear layer registered
twice into one series" — not a shape, HCCL, weight-format or SoC complaint, and not the
model failing to load.

## The two registrants, at this ref

The series name is the module's own name (`curr_layer.prefix.split(".")[-1]`), which is
why two independent paths collide on `"o_proj"`:

| path | call chain | gated by |
|---|---|---|
| the linear's row-parallel op | `Flashcomm2OProjRowParallelOp.update_attrs` (`ops/linear_op.py:381-386`) → `flashcomm2_oshard_manager.register_layer` (`ops/flashcomm2_oshard_manager.py:37-59`) → `register_layer_to_shard_weight_series(series_name="o_proj", …)`; invoked from `AscendRowParallelLinear.__init__` (`ops/linear.py:304-305`) | `flashcomm2_oshard_enable()` = `flashcomm2_enable()` (size > 0) **and** `o_shard_enable()` (`utils.py:981-991`, "o_proj" in `layer_sharding`) |
| the attention impl's `layer_sharding` kwargs | `mla_v1.py:719-727` (DSA sibling `sfa_v1.py:441-450`) → `register_all_layers_to_shard_weight_series` (`ops/layer_shard_linear.py:279-289`) → same series name | `additional_config.layer_sharding` non-empty |

Both filter through `is_hidden_layer`, so both fire for exactly the decoder layers and
only when the two switches are on together — i.e. the reported command. The model side
is not implicated: keep the attention-impl check only to confirm the second path exists at
all — the impl classes live in `vllm_ascend/attention/` (`AscendMLAImpl` in `mla_v1.py`,
`AscendSFAImpl` in `sfa_v1.py`, the non-MLA `AscendAttentionBackendImpl` in
`attention_v1.py`, which never registers `layer_sharding` kwargs) and the model file itself
is upstream's. A run whose backend is the non-MLA one registers nothing from that side, so
the falsifier below applies.

`layer_sharding` with more than `["o_proj"]` cannot be the trigger: with FlashComm2 on,
`get_flashcomm2_config_and_validate` raises `FLASHCOMM2 only supports 'o_proj' as the
sole layer sharding configuration!` (`utils.py:1007-1009`) before any load, so a
`["q_b_proj","o_proj"]` report dies at a *different* line.

## The pin's own docs ship this combination

`docs/source/user_guide/feature_guide/layer_sharding.md` §*FlashComm2-enabled* is exactly
`export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1` plus
`--additional-config '{"layer_sharding": ["o_proj"]}'` (lines 53-59 at this ref), and
`utils.py:1004` even logs "Enable FLASHCOMM2 with o_proj layer sharding for reduced
memory consumption." So at this ref the **documented recipe is the failing one**: the
deliverable is a defect report (one layer, two registrants), not a corrected launch line,
and it should say so in the first sentence rather than asking the reporter to keep
tuning flags.

## Procedure

1. **Answer first**: both switches are on, so both registrants fire — the failure is the
   combination, not the number after `FLASHCOMM2_PARALLEL_SIZE`, not the checkpoint, not
   the Mooncake/PD settings in the same block.
2. **Two one-line ablations**, each removing exactly one registrant. Send both; the pair,
   not either one alone, is what confirms the mechanism:
   - `export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0`, keep `layer_sharding` — expected to
     reach serving, with `o_proj` still layer-sharded through the attention path;
   - drop `"layer_sharding"` from `--additional-config`, keep FlashComm2 at 4 — expected
     to reach serving, at higher weight memory per card (the manager's own registration
     is gated off with `o_shard_enable()`).
   **Falsifier**: if the assert survives *both*, the duplicate is not this pair — dump the
   series (`for m in series.layers: print(m.layer_idx, m.layer.prefix)`) and re-read every
   `register_layer_to_shard_weight_series` caller at the reporter's ref: a draft/MTP module
   built from the same layer indices (`_EXTRA_CTX.is_draft_model`) or a prefix whose
   `extract_layer_index` collides gives the same signature.
3. **The value question, answered from the validator, not prose**: the number is the
   width of the FlashComm2 o-proj TP group. It must be > 0 to enable, `< tensor-parallel
   size` and divide it (`utils.py:1018-1028`), cannot coexist with
   `finegrained_tp_config.oproj_tensor_parallel_size > 0`, is refused on a KV consumer
   (the D scenario, `utils.py:1033-1038`) and only warns when FLASHCOMM1 is off. TP8 with
   4 satisfies all of these, so the reporter's value is **not** the fault — say that
   before they start tuning it.
4. `recompute_scheduler_enable` (also in that command) is a documented `additional_config`
   bool: `docs/source/user_guide/configuration/additional_config.md:40`, and
   `docs/source/tutorials/models/DeepSeek-V3.1.md:579` — when the decode node's KV cache
   is insufficient, requests are sent back to the prefill node to recompute the KV cache;
   recommended on both sides in PD. It plays no part in this crash; answer it in one line
   so the doc-gap complaint ("我根本不知道…需要填几") does not pull the session into
   `additional_config` schema work.

## Recorded case

#7508 (A3, Kimi-K2.5 W4A8, TP8 × DP2 + EP, P node with `MooncakeLayerwiseConnector`,
`VLLM_ASCEND_ENABLE_MLAPO=1`, `recompute_scheduler_enable: true`). The run reached the
double-registration reading only in its last two turns, by then spending 5-8 K characters
of reasoning per turn, and ended `output_truncated` at 16 384 output tokens: 15 assistant
turns, 26 tool results, 48 K characters of reasoning, **zero** characters of message text.
The three inputs above are one read each: the assert's own loop, the two registrants'
gates, the doc page that ships the pair. A verdict ("both switches register the same
`o_proj`; disable one") plus the two ablations was shippable as soon as the traceback was
mapped to `post_process_after_loading` — no mechanism needed.
