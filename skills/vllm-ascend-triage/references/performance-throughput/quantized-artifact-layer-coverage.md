# Quantized artifact, unquantized layer class: "W8A8 is no faster than BF16"

## Scope

The reporter serves a *quantized* artifact (an msmodelslim `-w8a8` / `-w8a8-mtp`
checkpoint with `--quantization ascend`) and reports that its throughput/latency
equals the BF16 model — backed by a profile in which one whole layer class has no
quant kernel. On an interleaved hybrid model (Qwen3.5 / Qwen3-Next: GDN
linear-attention layers plus every-Nth full-attention `self_attn`) the
full-attention `qkv_proj`/`o_proj` rows come out as plain matmul while the
linear-attention/MoE rows carry `…QuantBatchMatmul…`. Recorded #9443
(Qwen3.5-122B w8a8-mtp, vllm-ascend v0.19.1rc1, 1P1D).

Not this page: a collection that errored or reports a working decode loop as
"NPU 95 % idle" (the table is the suspect first, `profiler-opcount-evidence.md`);
a low rate with no quantization claim (parent `index.md`); a converter's per-layer
loss table (`../accuracy-benchmarks/quantized-checkpoint-conversion.md`).

Pins verified below: vllm-ascend **v0.19.1rc1** (`da421af`), vllm **v0.19.1**
(`b1388b1`). Re-grep per release — the quantization package was restructured in
this window (`quant_config.py` no longer exists; the reader is
`quantization/modelslim_config.py`).

## Rules

1. **There are two different quantizations here; name which one the row is
   about before interpreting it.** (a) int8 weight/activation matmul on a linear
   layer: a per-layer entry in the artifact's own
   `quant_model_description.json`, applied on every node. (b) int8 attention /
   int8 KV cache (`fa_quant_type`, `fa_k*.scale`): separate top-level keys read
   in `_add_kvcache_quant_metadata` (`modelslim_config.py:709-722`) and gated to a
   **KV-consumer (decode) instance only** — `enabling_fa_quant`
   (`:566-575`) demands `is_kv_consumer and not is_kv_producer`, wrapped by
   `enable_fa_quant` (`quantization/utils.py:202-206`). So an absent int8
   attention kernel on a **prefill** node proves nothing; first ask which side
   the table was dumped on. At v0.19.1rc1 `rg -n 'fa_quant' docs/source` is
   empty — there is no documented flag for (b).

2. **A linear layer cannot silently lose its quantization — the artifact's json
   decides, and a *missing* key is loud.** `get_quant_method`
   (`modelslim_config.py:465`) hands a `LinearBase` to
   `is_layer_skipped_ascend` (`:530-557`) and only a `"FLOAT"` entry returns
   `AscendUnquantizedLinearMethod` (`:499-505`). The fused lookups are bare dict
   reads on the *shard* keys (`:540`, `:291`, mapping at `:66-78` for
   `qwen3_5`/`qwen3_5_moe`), so a prefix the json does not carry raises
   `KeyError` at startup, and shards that disagree raise `Detected some but not
   all shards of … are quantized`. Consequences: an engine that started with a
   bf16 layer class has a json that says `FLOAT` for it, and no serve flag will
   change that. One command settles it —
   `jq -r 'to_entries[]|select(.key|test("self_attn"))|.value' $MODEL/quant_model_description.json | sort | uniq -c`.
   Also rules out "the downstream patch dropped `quant_config`": upstream passes
   it to both projections (`vllm/model_executor/models/qwen3_next.py:235,245`)
   and vllm-ascend patches only `forward`
   (`patch/worker/patch_qwen3_5.py:28-75`, rebound at `:139`).

3. **"That is the artifact's choice" is not the verdict when the class is
   architectural.** The skipped class is fixed by the architecture (every Nth
   layer is full attention) and the json is the output of one converter recipe,
   so the same hole is in every artifact that recipe produced — every size of
   the family, not the reporter's file. Say that in the answer, and give the two
   checks that separate it: the **sibling size's** json (same one `jq`), and the
   **same-request bf16-vs-w8a8 A/B** (equal `--max-num-seqs`, dataset,
   `--num-prompts`). Both artifacts FLOAT ⇒ the report belongs to the conversion
   pipeline (Ascend msmodelslim / the artifact publisher), *and* every user of
   the family inherits the missing gain, so it is worth a public issue rather
   than a per-artifact reply. Recorded #9443: the triage answer closed on "the
   artifact most probably never quantized them … the fix belongs to the
   converter owner"; the reporter's next public message was "This issue should be
   a common problem that affects performance gains" — the per-artifact framing
   under-served a family-wide gap.

4. **Variant coverage is a second, separate fact.** At v0.18.0 the matrix lists
   `Qwen3.5-397B-A17B` and `Qwen3.5-27B` and nothing else
   (`docs/source/user_guide/support_matrix/supported_models.md:84-85`; `:82-83` at
   v0.19.1rc1), and the only Qwen3.5 w8a8 nightly configs are those two
   (`tests/e2e/nightly/single_node/models/configs/`), with no quant case in
   `tests/e2e/multicard/4-cards/test_qwen3_5.py`. Report it, but do not merge it
   with rule 3 — and check the report's own names: #9443's title said
   `122B-A3B` while its weights URL said `122B-A10B`; ask which is on disk.

5. **Price the gap before promising a fix.** Decode cost is dominated by weight
   bytes, so an unquantized class costs *its share of the weights* ×
   (unquantized ÷ quantized bytes per weight), spread over the one-in-N layers that
   are full attention — two numbers from `config.json` and the json of rule 2, not a
   profiler rerun. Say plainly that the lever is a re-converted artifact (or a
   family that ships one), not a serve flag.

## What to ask for

The `quant_model_description.json` slice for the class in question; the per-layer
kernel names from the dump (quant matmul vs plain matmul) and **which node**
produced them; both serve lines (P and D) including `--quantization ascend` on
both; and the two checkpoint ids if the title and the weights URL disagree.

## Limits

The ModelScope/Eco-Tech artifact cannot be inspected from the case snapshot, so
"the json says FLOAT" is a check for the reporter, not a finding you can assert;
and whether the int8 FA path (rule 1b) even exists for that SoC at that release
is not documented in the pinned tree.
