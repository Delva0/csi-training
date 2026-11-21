# `--load-format <x>`: `AttributeError: '<Y>ModelLoader' object has no attribute '<method>'`

## Scope

The engine dies during the **load** phase — often the line right before it is
`Loading drafter model...` — with an attribute error whose object is a **model
loader**, not a model or layer class:

```text
  File "vllm_ascend/worker/mtp_proposer_v1.py", line 258, in load_model
    self.model.load_weights(loader.get_all_weights(…))
AttributeError: 'DummyModelLoader' object has no attribute 'get_all_weights'
```

Seen with `--load-format dummy` plus `--speculative-config` (recorded #2346,
DeepSeek-V3 MTP, v0.10.0rc1); the same mechanism covers every non-default loader
(`gguf`, `runai_streamer`, `tensorizer`, `bitsandbytes`, `sharded_state`) and any
downstream site that pulls weights out of the loader by hand.

Not this page: an `AttributeError` whose object is an `Ascend*` model/layer/config
class (`index.md` rows), a `KeyError` on a
`<param>.weight_scale` (`../startup-failures/weight-scale-keyerror-quant-variant.md`),
or a load that stalls without raising
(`../startup-failures/weight-load-stall-in-shard-loop.md`).

## Verdict first (the G1 sentence)

Both facts are already in the paste. (a) **The crash is real on that image** — a
reporter traceback settles it. (b) **The object it names is the loader**, so the
checkpoint, the model code and the quantization config are exonerated: the
downstream call site uses an API that only *one* loader class implements.

The contract is `BaseModelLoader` (`vllm/model_executor/model_loader/base_loader.py:16-46`
at vllm v0.10.0 `6d8d0a24`): abstract `download_model` + `load_weights`, plus a
concrete `load_model(vllm_config, model_config)` that builds the module
(`initialize_model`), calls `self.load_weights(...)`, then
`process_weights_after_loading`. `DefaultModelLoader.get_all_weights`
(`default_loader.py:227`) is **not** part of it, and `DummyModelLoader`
(`dummy_loader.py:11-26`) implements only `download_model` +
`load_weights → initialize_dummy_weights(model)`.

The upstream counterpart fixes the direction of the drift: upstream's drafter is
built through the loader abstraction,
`get_model(vllm_config=self.vllm_config, model_config=draft_model_config)`
(`vllm/v1/spec_decode/eagle.py:337-346`), and `use_eagle()` routes `deepseek_mtp`
into that family (`vllm/config.py:3045-3046`) — so the **same serve runs upstream**
and only vllm-ascend's hand-rolled weight pull breaks. Owner: downstream.

## Anchors (vllm-ascend v0.10.0rc1 `4604882a`, vllm v0.10.0 `6d8d0a24`)

| fact | anchor |
|---|---|
| loader obtained, then used by iterator | `vllm_ascend/worker/mtp_proposer_v1.py:236` `get_model_loader(self.vllm_config.load_config)`, `:257-259` `self.model.load_weights(loader.get_all_weights(draft_model_config, self.model))` |
| `get_all_weights` lives on one loader only | `vllm/.../default_loader.py:227` |
| dummy loader's whole surface | `vllm/.../dummy_loader.py:11-26` |
| the contract | `vllm/.../base_loader.py:16-46` |
| construct the model, then load | Base loader does `initialize_model` → `load_weights` → `process_weights_after_loading`, i.e. the downstream `CustomDeepSeekMTP(...).to(device)` block duplicates half of it |

The same call already exists one release earlier — `mtp_proposer_v1.py:200` at
v0.9.1rc1 `c30ddb8` — so this is not new in the reported release; grep the method
name rather than trusting a line number, because the proposer file is downstream-
only and moves.

**Free pin check:** the reporter's own `model_runner_v1.py:2107
self.drafter.load_model()` sits one call below this tree's
`logger.info("Loading drafter model...")` at `:2100`, so their image is the pinned
release and the anchors above are citable as they stand.

## Workaround ladder

1. **Drop the flag if the directory already has weights.** `ls <model dir>/*.safetensors`
   (a `…-dummy-…` checkpoint name often means a materialized export). With
   `--load-format auto` the loader is `DefaultModelLoader`, which has the method —
   zero code change, and it settles whether the flag was even needed.
2. **Guard the call** when dummy loading is what they need:

   ```python
   if isinstance(loader, DefaultModelLoader):        # or hasattr(loader, "get_all_weights")
       self.model.load_weights(loader.get_all_weights(draft_model_config, self.model))
   else:
       loader.load_weights(self.model, draft_model_config)
   ```

   Do **not** unify to `loader.load_weights(...)` alone: `DefaultModelLoader.load_weights`
   adds a strict `weights_to_load - loaded_weights` check (`default_loader.py:256-270`,
   applied only while `model_config.quantization is None`), and
   `DeepSeekMTP.load_weights` `continue`s on every non-spec-layer name
   (`vllm/model_executor/models/deepseek_mtp.py:180-201`), so the bf16 path can
   turn a working load into a `ValueError`.
3. **Serve without spec decode** at the same `--load-format dummy` as a control —
   it proves the loader choice is the only difference, at the cost of the drafter's
   throughput.
4. Why it shipped: the MTP e2e case uses real random weights
   (`tests/e2e/singlecard/spec_decode_v1/test_v1_mtp_correctness.py`,
   `wemaster/deepseek_mtp_main_random_bf16`) and carries no `load_format` variant,
   so dummy loading has no coverage — quotable as coverage, not as the cause.

## General recipe for the next loader-method message

Two calls, no model walk:

1. `rg -n "class .*ModelLoader|def <missing_method>" vllm/model_executor/model_loader/`
   — which class defines the method, and which class the message named.
2. `rg -n "<missing_method>" vllm_ascend/` — every downstream site that would break
   under a different `--load-format`; sweep them all before answering, since the
   fix is per-call-site.

Then say which load formats the reporter may use and which they may not. State the
invalidating condition: the anchors are bound to both commits, and a later vllm
release that moves `get_all_weights` into `BaseModelLoader` retires this page.

No precedent exists in the corpus for `dummy` / `get_all_weights` (0 title hits at
the time of #2346) — a lexical gap, not evidence of absence.
