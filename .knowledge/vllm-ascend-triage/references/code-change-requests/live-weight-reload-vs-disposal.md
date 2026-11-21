# Live weight reload (RL rollout) vs post-load weight disposal

## Scope

The report is a **change request** even when it carries a traceback: an RL rollout
worker (VERL, ms-swift, trl `vllm_serve`, Megatron actor) cannot receive updated
weights, and the issue asks for the disposal to be gated. Recorded #15463 (GLM-5.2,
TP4, `load_format: dummy`, vLLM/vLLM-Ascend v0.23.0, Megatron-Bridge + VERL v0.9.0,
SFA/sparse model). The load raises one of:

```text
AssertionError: param_data.shape == loaded_weight.shape          # column/row shard loader
AssertionError: Attempted to load weight (…) into parameter (…)  # default_weight_loader
```

Verified at vllm-ascend `5cb98caa` (v0.23.0) + vllm `0fc695fc` (v0.23.0). Re-grep
the anchors when the pin differs; the mechanism, not the line numbers, is the rule.

Not this page: an RL weight sync that dies in a **collective** (`HcclBroadcast`,
`EI0009`, `DestLink`, ms-swift's `update_named_param`) —
`../hccl-and-hardware/weight-sync-hccl-cascade.md`; a load-phase
`AttributeError: '<X>ModelLoader' object has no attribute …` —
`../upstream-downstream-drift/load-format-loader-contract-drift.md`; a plain
checkpoint-vs-parameter shape mismatch (no disposal anywhere in the tree) — that is a
quantization-variant or TP-sharding question.

## Rule 1 — the assertion names the loader, and the empty destination owns it

Both texts above mean the **registered destination has no storage**, not that the
checkpoint's shape is wrong. Which one fired narrows only *which loader* ran
(`vllm/model_executor/layers/linear.py:564` for the column-parallel shard loader,
`vllm/model_executor/model_loader/weight_utils.py:1393` for the default one, at
vllm v0.23.0), so take it from the paste's own frame list and do not spend a call on
it. The loaders compute a shard size from `param_data.shape` first — a `[0]`
destination therefore yields a zero-row slice or an immediate assert.

## Rule 2 — one call decides: print the destination's shape

`model.get_parameter("model.layers.0.self_attn.kv_b_proj.weight").shape` (or any
name in the failing key set) → `torch.Size([0])` confirms this page; a two-dimensional
shape refutes it and moves the case to checkpoint/sharding. Do this before any
source walk: it is the only check that separates "disposed" from "mismatched".

## Rule 3 — the disposal inventory at v0.23.0 (`5cb98caa`)

| Site | What it does |
|---|---|
| `vllm_ascend/utils.py:686-687` | `dispose_tensor(x)` = `x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))` → 1-D `[0]` **in place**, so the `Parameter` object and its name survive |
| `vllm_ascend/utils.py:1422-1426` | `dispose_layer(layer)` disposes **every** `torch.Tensor` attribute of the module (`weight`, `bias`, any already-cast scale tensor), so "keep the weight" is not the same as "keep the layer usable" |
| `vllm_ascend/attention/sfa_v1.py:782` | `AscendSFAImpl.process_weights_after_loading` derives `W_UK_T`/`W_UV` then **unconditionally** `dispose_layer(self.kv_b_proj)`; the same method asserts the 2-D `kv_b_proj` shape at `:746-752` and has a reload branch at `:771-780` that `copy_`s into `W_UV`/`W_UK_T` to keep graph-captured addresses |
| `vllm_ascend/attention/sfa_v1.py:912-913` | on a **decode** worker (`is_kv_consumer`) the same method disposes `fused_qkv_a_proj` and `q_proj` after the MLAPO layouts are copied — the P/D decode side of the same defect |
| `vllm_ascend/ops/layer_shard_linear.py:15,120,246` | the DSA-CP layer-shard path disposes non-source layers' weights with a 0-dim `torch.empty([])` |

## Rule 4 — the tree already contains the reload-aware form; quote it as the fix shape

`vllm_ascend/ops/layer_shard_linear.py:244-247`: for a layer it disposes, the shard
path also takes over `quant_method.process_weights_after_loading` and installs
`layer.weight.weight_loader = lambda *args, **kwargs: None`. So "dispose + neutralize
(or rehydrate) the loader" is in-tree practice, while `sfa_v1.py:782` disposes with
the standard loader still installed. That one contrast is the defect statement and the
shape of the accepted fix, and it is cheaper than arguing about RL semantics.

## Rule 5 — `load_format: dummy` is an amplifier; `weight_transfer_config` is not the capability

- `dummy` builds real-shape parameters and post-processing then empties them, so the
  assert exists **only after** `process_weights_after_loading`. A rollout worker that
  loads a real checkpoint first and then receives updates hits the same state, so
  `dummy` is neither necessary nor sufficient — offer it as the fastest reproducer,
  never as the condition.
- The in-tree live-update engine is `vllm_ascend/worker/worker.py:303` /
  `:324` / `:359` (`start_weight_update` → `initialize_layerwise_reload`,
  `update_weights` → `weight_transfer_engine.receive_weights(load_weights=model.load_weights)`,
  `finish_weight_update` → `finalize_layerwise_reload`), gated on
  `weight_transfer_config` (`:282-286`, built at `:694-701`) and refusing NZ
  (`:295-301`) — the same RL-guard family as `weight_nz_mode` in `wake_up`
  (`worker.py:242-249`).
- A trainer shipping its own ZMQ/IPC transport and calling `model.load_weights()`
  directly never sets `weight_transfer_config`, so the capability gate has to be a
  separate `--additional-config` key. Say that in the change plan: gating on
  `weight_transfer_config is not None` alone leaves that caller broken.

## Rule 6 — do not credit the layerwise reload as the recovery without naming the recorder

`initialize_layerwise_reload` restores each layer from the metadata snapshot taken by
`record_metadata_for_reloading` (`vllm/model_executor/model_loader/reload/layerwise.py:70`,
`reload/meta.py:115-133`) and `finalize_layerwise_processing` reprocesses
attention/MLA layers last (`layerwise.py:217-242`). If a snapshot exists, the disposed
parameter is materialized with its recorded shape and the `copy_` branch at
`sfa_v1.py:779` works. At vllm v0.23.0 the recorder's only callers in the export are
`tests/model_executor/model_loader/test_reload.py:202,243`, while
`docs/training/layerwise.md:108` claims "Called by `BaseModelLoader`" and
`base_loader.py:60-80` calls only `process_weights_after_loading`. State it as "no
production caller found in this export; the export may be partial", and turn it into
one grep on the reporter's tree instead of promising the route.

## Rule 7 — the memory tradeoff belongs in the answer, with the default

`7168 × 512 × 2 B = 7 MiB` per local SFA layer per rank (the recorded report's own
arithmetic, and it is right) — so keep disposal as the default and gate it behind a
capability flag, or rehydrate on the first update. A capability flag defaults off for
inference-only deployments; a patch that always retains the weight is a memory
regression for every non-RL serve, and a reviewer will ask.

## Answer shape

1. Verdict: the report is accurate, the disposal is unconditional, and the line that
   asserts is the shard loader's own — with the two `path:line` anchors (disposer,
   assert).
2. The one check (Rule 2) so the reporter can confirm on their box in one command.
3. Change plan in preference order: capability flag settable from
   `--additional-config` (applies to `kv_b_proj` **and** the decode-side
   `fused_qkv_a_proj`/`q_proj`), then dispose-and-neutralize like
   `layer_shard_linear.py:247`, then rehydrate-on-reload using the shape the method
   already asserts.
4. Scope and gaps: pins verified, whether a later release carries a fix is not in the
   snapshot, and what to send back (the full frame list, the parameter shape before the
   sync, whether `weight_transfer_config` is set, the rollout worker's `vllm_ascend`
   commit, and the exact parameter names that failed — they say which disposer ran).
