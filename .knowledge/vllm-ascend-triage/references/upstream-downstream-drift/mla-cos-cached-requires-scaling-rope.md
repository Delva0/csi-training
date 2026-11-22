# `'<Ascend…>RotaryEmbedding' object has no attribute 'cos_cached'`: the MLA/SFA builder needs the scaling rope

Verified at vllm-ascend `v0.11.0rc1` (`c5fe179`) + vllm `v0.11.0` (`b8b302c`) —
the pin recorded **#5385** reported (910B3, a hand-reduced "DeepSeekV3-like"
checkpoint reduced for single-card inference, `vllm==0.11.0` /
`vllm-ascend==0.11.0rc1`, question form "how do I fix this?"). Every `path:line`
below was re-read there; the reporter never returned their `config.json`, so the
config half is the ranked candidate with its one check, not a confirmation.
Re-grep at another pin (section *Later pins* — at `v0.13.0rc3` this read is gone).

## Signature

The engine starts, loads weights and answers nothing: the **first request** raises
inside attention-metadata building.

```text
vllm_ascend/attention/mla_v1.py:318, in AscendMLAMetadataBuilder.build(...)
      self.cos_cache = model.model.layers[
          model.model.start_layer].self_attn.rotary_emb.cos_cached
AttributeError: 'AscendRotaryEmbedding' object has no attribute 'cos_cached'
```

`#5385` spelled the class `AscendRotartEmbedding`; **no such class exists** at the
pin (`rg -n 'AscendRotart' .` is empty) — it is `AscendRotaryEmbedding`, and the
typo is a transcription slip, not a missing class. Treat the in-tree class name as
the message.

Same idiom, four sites — the file in the frame names the backend, so "switch
backend" only moves the traceback: `attention/mla_v1.py:318-322`,
`attention/sfa_v1.py:308-312`, `torchair/torchair_mla.py:409`,
`torchair/torchair_sfa.py:422`. It fires from `build()`, not from graph capture:
`--enforce-eager` is irrelevant here.

## Two other failures at the same line — discriminate before ranking

| object named in the message | owner | page |
|---|---|---|
| `'PPMissingLayer' object has no attribute 'self_attn'` | pipeline-parallel stages > 0: `layers[...]` is not a real layer | `../pipeline-parallel/pp-blind-first-layer-lookup.md` |
| `'Ascend…RotaryEmbedding' object has no attribute 'cos_cached'` (this page) | the rope object exists; the **buffer** is missing because a non-scaling rope class was built | here |
| the same sentence **with** `torch._dynamo` / a plugin-patched module in the traceback | a patched/traced consumer reading an old attribute name | `impl-and-oot-signature-drift.md` Rule 3 |

## Mechanism

| claim | anchor (v0.11.0rc1 / vllm v0.11.0) |
|---|---|
| the read is unconditional and unguarded — no `getattr`, no `hasattr`, no `if rope_scaling` | `vllm_ascend/attention/mla_v1.py:318-322` (lazy fill under `if self.cos_cache is None`, fields at `:240-241`) |
| exactly one Ascend rope class registers the two buffers, and it is a **downstream-only invention** (upstream's `DeepseekScalingRotaryEmbedding` inherits `cos_sin_cache` and registers nothing else) | `vllm_ascend/ops/rotary_embedding.py:374-375` inside `_set_cos_sin_cache` (`:342`) of `AscendDeepseekScalingRotaryEmbedding` (`:198`); upstream `rotary_embedding/base.py:48-49`, `deepseek_scaling_rope.py:22` |
| the plain and YaRN Ascend classes never register them | `AscendRotaryEmbedding` (`:112`) and `AscendYaRNRotaryEmbedding` (`:157`) get only `cos_sin_cache` from `RotaryEmbedding.__init__` (upstream `base.py:45-49`) |
| which class runs is decided by the OOT **name** mapping, so the printed class names the upstream class `get_rope` built | `vllm_ascend/utils.py:537,541,545` — `"RotaryEmbedding": AscendRotaryEmbedding`, `"YaRNScalingRotaryEmbedding": AscendYaRN…`, `"DeepseekScalingRotaryEmbedding": AscendDeepseek…` |
| upstream reaches the scaling class **only** through a truthy `rope_scaling` | `vllm/model_executor/models/deepseek_v2.py:377-378` (`if rope_scaling: rope_scaling["rope_type"] = 'deepseek_yarn'`); dispatch `rotary_embedding/__init__.py:71-72` (`elif not rope_scaling: RotaryEmbedding(...)`) and `:171-184` (`elif scaling_type == "deepseek_yarn"`) |

So: **`AscendRotaryEmbedding` in the message = the checkpoint made `get_rope`
build an unscaled rope**, while the MLA/SFA builder was written assuming the
DeepSeek scaling rope. Causes, one check each:

1. **`config.json` has no `rope_scaling` (or it is `null`)** — the reported
   checkpoint was hand-reduced, and the block is easy to drop while keeping
   `architectures`/layer counts. Check: one line on the model directory (below).
2. **the block was overridden away** at launch (`--hf-overrides` is a real
   `ModelConfig` field at this pin, `vllm/engine/arg_utils.py:374`) — check the
   serve command and the engine's printed `rope_scaling`/config dump.
3. **the model is not upstream `deepseek_v2`** (a custom/reduced model file,
   `vllm_ascend/models/deepseek_v3_2.py`-style copy, or a non-DeepSeek MLA
   architecture) that builds a plain or YaRN rope while asking for the MLA
   backend. Check: `architectures` in `config.json` against
   `vllm/model_executor/models/registry.py`; a `AscendYaRNRotaryEmbedding` in the
   message is this cause (the YaRN class is *also* registered, `utils.py:541`).

## Cheapest order

1. One line, their machine, decides cause 1:
   `python -c "import json;c=json.load(open('<model_dir>/config.json'));print(c.get('rope_scaling'), c.get('max_position_embeddings'))"` → `None` confirms it.
2. Restore the checkpoint's own block (a copy of `config.json`, or
   `--hf-overrides '{"rope_scaling": {...}}'`). Only these keys are read by the
   `deepseek_yarn` branch: **required** `factor` and
   `original_max_position_embeddings`; optional `extrapolation_factor`,
   `attn_factor`, `beta_fast`, `beta_slow`, `mscale`, `mscale_all_dim`
   (`rotary_embedding/__init__.py:174-184`). The block is not cosmetic: with it
   truthy, `deepseek_v2.py:387-390` also multiplies the attention scale by
   `mscale²`, so inventing numbers changes logits — restore the checkpoint's own.
3. Size it before the next run: the table has
   `ceil(original_max_position_embeddings × factor)` rows
   (`ops/rotary_embedding.py:235-236`; `get_rope` passes the rope_scaling original
   value as `max_position`, not the config-level one) and it is indexed by
   **absolute positions** (`mla_v1.py:379-384,416-421,435-439`), so
   `--max-model-len` above that row count trades this error for an index failure.
   Real DeepSeek-V3 numbers give `4096 × 40 = 163840` rows.
4. Close on output, not on the exception: one completion compared with a run that
   had the block (or a GPU reference).

## The fallback trap

Do **not** wrap the read in `getattr(...)`/`try`. The two layouts differ:
`cos_sin_cache` is `cat([cos, sin])` over `freqs = outer(t, inv_freq)`
(`ops/rotary_embedding.py:371-375`) while `cos_cached`/`sin_cached` are
`cat([freqs, freqs]).cos()/sin() * mscale` (`:366-369`) — a substitution starts
serving and silently corrupts the MLA rope. The honest downstream answer is either
"serve with the scaling rope" or "this is a defect to report": the same unguarded
read exists in the DSA/SFA path (`sfa_v1.py:308-312`) and both torchair builders,
so no config flag avoids it on that pin.

## Later pins: the read was refactored away

At v0.13.0rc3 (`bbe9bc0`) neither builder reads `rotary_emb.cos_cached` any more:
`mla_v1.py:545,642` call `get_cos_and_sin_mla(...)` and the cos/sin tables live in
module globals in `ops/rotary_embedding.py`, recorded by each rope class at
construction — `AscendRotaryEmbedding` records `cos_sin_cache` **and** the
interleaved cos/sin split (`:268-269`, via
`_record_cos_and_sin_cache_interleaved` `:155-165`), `AscendDeepseekScalingRotaryEmbedding` records its own `cos_cached`/
`sin_cached` (`:504-507`). Consequences for a report against a newer tag:

- the unscaled-rope MLA model **works** there, so "upgrade past the refactor" is a
  legitimate rung and the config surgery above is not needed;
- `AscendYaRNRotaryEmbedding` records only `_cos_sin_cache` (`:313`) and never the
  micro cos/sin pair, so a plain-YaRN model on an MLA backend is still a candidate
  failure at that pin (code path only — no recorded report);
- the exact refactor commit is not in this snapshot: say "absent at `bbe9bc0`, not
  fixed by a named release", and keep the one grep below as the recheck.

## Recheck / what to send back

- `rg -n 'cos_cached' vllm_ascend/attention/` — 4 hits at `c5fe179`, **0** at
  `bbe9bc0`; that grep dates the reporter's tree in one call.
- From the reporter: `config.json`'s `rope_scaling` + `max_position_embeddings` +
  `architectures`, the full traceback (which of the four files), and the serve
  command if `--hf-overrides` is in play.
- Older recorded precedents for this buffer name hit older pins and other shapes
  (`#965` row 4 at `0.8.5post1`, `#5111` `'…RotaryEmbedding' object has no
  attribute 'cos'`) — one line each, never the answer
  (`../evidence-workspace/delivery-recorded-sessions.md`).
