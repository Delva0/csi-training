# Load-phase `KeyError: '<param>.weight_scale'`: the checkpoint's scale tensors vs the engine's quant config

Lane B, every rank, after `Starting to load model` (the shard bar is usually
visible). The traceback's deepest frames are the model loader:

```text
(Worker_PP0_TP7_EP7 pid=125) ERROR ... vllm_ascend/models/deepseek_v4.py", line 1535, in load_weights
    param = params_dict[name]
KeyError: 'model.layers.2.self_attn.indexer.wq_b.weight_scale'
```

and the other ranks name *different layers* from a **small set of module
families** — `self_attn.wkv`, `self_attn.indexer.wq_b`, occasionally
`self_attn.compressor.wkv` — with `Multi-thread loading shards: 2% Completed |
1/46` above the first traceback. Recorded #12675 (image v0.23.0rc1,
DeepSeek-V4-Flash, TP8 + `--enable-expert-parallel`, DP1/PP1).

This is the **mirror** of `quantization-method-gate.md` section B. There, the
checkpoint has int8 *weights* a dense model cannot hold; here the checkpoint
carries *scale* tensors the engine built no parameter for. Classify on the key:
`.weight_scale` / `.scale` → this page; a plain `.weight` → the gate page.

## What the pin does (verified at v0.23.0rc1 `f4a08bd`, one-commit snapshot)

1. `vllm_ascend/models/deepseek_v4.py:1381-1382` renames every checkpoint tensor
   ending `.scale` to `.weight_scale` before the lookup, so a modelslim export
   (`…wkv.weight_scale`) and a native one (`…wkv.scale`) meet the same
   `params_dict` name.
2. There is **no** `if name not in params_dict: continue` guard: `:1535`
   (`param = params_dict[name]`) subscripts directly, and the only skips in that
   loop are `.bias` and PP-missing parameters, plus the FP8 kv-scale remap at
   `:1528`. Any extra scale tensor the model does not own is therefore a hard start
   failure, not a warning.
3. A `weight_scale` parameter exists only if that linear was built with a quant
   method that allocates it. At that ref `Attention.wkv` and `Indexer.wq_b` pass
   `quant_config` unconditionally (`:758`, `:554`), so a missing `weight_scale`
   there is a statement about **the engine's quantisation**, never about those two
   modules being special-cased in code.

## Procedure — two calls before any source walk

1. **Get the contract first: the exact `vllm serve` line (or launch script) and
   the served model directory.** In the recorded case that alone decided it; the
   investigation instead spent its whole budget in `modelslim_config.py` and never
   asked.
2. **Census the checkpoint's quantisation tensors** — one call, header-only, no
   device, reuses `../../scripts/checkpoint-key-prefix.py`:
   `uv run --no-project python ../../scripts/checkpoint-key-prefix.py --scales <model dir>`
   It prints every `.scale` / `.weight_scale` / `.weight_scale_inv` /
   `.quant_bias` key grouped by module family, plus whether
   `quant_model_description.json` and a `quantization_config` in `config.json`
   exist. Read the **family count**, not the tensor count:
   - scales across the main linear/MoE families (`self_attn.wq_a`/`wq_b`/`wo_a`/
     `wo_b`, `mlp.gate`, `mlp.experts.*`) and/or a declared quantisation → a
     quantised export; the question is which method the engine selected
     (`--quantization ascend` present? description file matching?) →
     `quantization-method-gate.md` section B.
   - scales confined to a **handful of sub-projections** (`self_attn.wkv`,
     `self_attn.indexer.wq_b`, `compressor.wkv`) with nothing declaring a
     quantisation → the directory is a **full-precision release** whose few
     FP8-stored projections ship a per-tensor scale. The engine builds those
     linears unquantised, the loader meets the scale first and raises. Nothing in
     the reporter's tree needs patching: the served weights are not the variant the
     release documents.
   - **no** scale tensors at all → the name in the traceback came from another
     tree: re-check the served path, the bind mount (`vllm serve` on a mount root
     instead of on the checkpoint directory) and any stale cached copy.
3. **Compare against the tutorial's weight list, not a sibling model's command.**
   `docs/source/tutorials/models/DeepSeek-V4-Flash.md:24-28` lists **one** weight
   variant — `DeepSeek-V4-Flash-w8a8-mtp` (quantised) — and every documented
   `vllm serve` line (`:161`, `:209`, `:435`, `:505`, `:691`, `:769`) carries
   `--quantization ascend`; the PR e2e test uses
   `gdydems/DeepSeek-V4-Flash-w4a8-mtp` with `quantization="ascend"`
   (`tests/e2e/pull_request/four_card/test_deepseek_v4.py`). Serving a plain
   `deepseek-ai/DeepSeek-V4-Flash` directory
   (`-v /data/models/DeepSeek-V4-Flash:/root/.cache` + `vllm serve /root/.cache`)
   with no `--quantization` flag sits outside that list — that was the recorded
   cause.

## Traps this page exists for

- **A different failing layer on every rank is not a per-layer code bug.** The
  loader is multithreaded/prefetched (`--safetensors-load-strategy prefetch`,
  `--model-loader-extra-config '{"enable_multithread_load": "true", …}'`), so each
  rank dies on whichever unmatched key its threads reach first. The stable fact is
  the set of **module families**, not the layer index — say so instead of reading
  the spread as layer-specific behaviour.
- **Do not infer "so exactly these two modules are skipped in code" from a short
  key set.** That inference is what pulls an investigation into
  `modelslim_config.is_layer_skipped_ascend` (`:681-708`),
  `create_scheme_for_layer` (`:417`) and `quant_prefix_mapper` (`:569-602`): real
  code, but load-bearing only *after* step 2 shows scales on the main families.
  Taken in the wrong order it explains a modelslim export to a reporter who is not
  serving one.
- The A5 branch `quant_config=None if get_ascend_device_type() in
  {AscendDeviceType.A5}` (`deepseek_v4.py:626`, `:634`) belongs to the
  **compressor's** `wkv`/`wgate`, so it produces a key containing
  `.compressor.wkv`. The key text tells you whether that branch is in play.
- The raise is load-phase: `--enforce-eager`, `--max-model-len`,
  `--compilation-config` and every graph knob are irrelevant, and a rank that never
  printed `Loading model weights took` never issued a device op.

## Scope and limits

Verified at vllm-ascend v0.23.0rc1 (`f4a08bd`): the remap site, the unguarded
`params_dict[name]`, and the two `quant_config` sites. The recorded case's launch
script arrived from the reporter after the investigation. The native checkpoint's
exact FP8 sub-projection list is **not** in the snapshot — step 2 prints it from the
reporter's own directory, so report "the census shows X", never a recalled list. Not
verified: whether this release accepts the full-precision variant at all (no
conversion path for it is documented).
