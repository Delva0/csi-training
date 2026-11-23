# PP-blind `layers[0]` lookup: `'PPMissingLayer' object has no attribute 'self_attn'`

Verified at vllm-ascend `v0.11.0rc0` (`00ba0710`) + vllm `v0.11.0rc3`
(`8ce5d319`), recorded #3524. Re-grep before quoting at another pin (see *Traps*).

## Signature

The engine **starts cleanly** — every rank loads weights (the log prints
`Loading model weights took 38.9 GB [repeated Nx across cluster]`, one per PP
stage), KV cache sizes are reported, `Starting vLLM API server` appears — and then
the **first request** dies:

```text
vllm_ascend/attention/mla_v1.py:311, in AscendMLABackend.build(...)
      self.cos_cache = model.model.layers[
          0].self_attn.rotary_emb.cos_cached
AttributeError: 'PPMissingLayer' object has no attribute 'self_attn'
```

reached from `vllm_ascend/worker/model_runner_v1.py:1547` (`_prepare_inputs` →
`builder.build(...)`). On a Ray launch the paste wraps it as
`ray.exceptions.RayTaskError(AttributeError)` from a `RayWorkerWrapper` on the
**remote** node: Ray is transport, the worker's own frame is the finding — do not
open the ray/`compiled_dag` path (#1751's shape is a different failure).
`--enforce_eager` is irrelevant here: this is attention-metadata build, not graph
mode.

## Mechanism

| claim | anchor (v0.11.0rc0 / vllm v0.11.0rc3) |
|---|---|
| non-first PP stages get a `PPMissingLayer()` at every index outside `[start_layer, end_layer)` | `vllm/model_executor/models/utils.py:615-633` `make_layers` (`[PPMissingLayer() for _ in range(start_layer)] + … [PPMissingLayer() for _ in range(end_layer, num_layers)]`) |
| the MLA metadata builder caches RoPE tables from **layer 0** | `vllm_ascend/attention/mla_v1.py:309-313` inside `build()` (`:283-288`, takes `model: nn.Module`) |
| the caller passes `model=self.model` (only the non-GDN branch; the GDN builder branch above omits it) | `vllm_ascend/worker/model_runner_v1.py:1536-1552` |
| the same idiom is copied into the SFA and torchair backends | `vllm_ascend/attention/sfa_v1.py:308-312`, `vllm_ascend/torchair/torchair_mla.py:408-412`, `vllm_ascend/torchair/torchair_sfa.py:421-425` |
| every MLA decoder layer builds its own `get_rope(...)`, so any present layer carries the same buffers | `vllm/model_executor/models/deepseek_v2.py:355,514`; downstream copy `vllm_ascend/models/deepseek_v2.py:195,353` |
| the tree already uses the PP-correct index elsewhere | `vllm_ascend/ascend_forward_context.py:127-128` (`model_instance.model.start_layer`) |

Consequences, and both are part of the answer:

- **Only non-first stages fail** — on PP rank 0 `start_layer == 0`, so `layers[0]`
  is a real layer and `build()` succeeds; stage 1 (+2 …) raises. So one request
  produces one traceback on one worker even though the cluster is "up".
- **Only the first request fails** — `self.cos_cache is None` is the lazy-fill
  guard (`mla_v1.py:231`), and the process has already crashed by then; every
  restart reproduces it, which is why the reporter's "works until the first
  request" reading is the correct one.

## The same read, a second missing attribute

The identical line also raises the same `AttributeError` naming a rope class and
`cos_cached` instead, with **no** PP involved: the layer is real, and the rope class the
checkpoint built registers no such buffers (only `AscendDeepseekScalingRotaryEmbedding`
does). The object named in the message is the discriminator — `PPMissingLayer` ⇒
this page, a rope class ⇒
`../upstream-downstream-drift/mla-cos-cached-requires-scaling-rope.md` (recorded
#5385, v0.11.0rc1: unscaled `config.json`).

## Cheapest order

1. **The traceback already names the owner**; one command on their tree confirms
   the idiom is still there (`rg -n "cos_cache is None" -A3 vllm_ascend/attention
   vllm_ascend/torchchair` → four sites, all `layers[` + `0].self_attn`), and
   `sed -n '283,315p' vllm_ascend/attention/mla_v1.py` shows the `build()` header
   that receives `model`.
2. **One-run ablation that removes the axis:** relaunch with
   `--pipeline-parallel-size 1`, cards absorbed by `--tensor-parallel-size 8
   --data-parallel-size 2` (or TP 16 cross-node). Error gone ⇒ PP-only defect and
   the diagnosis is closed. This is cheaper than any toolkit/serialization theory
   and it is what the reporter can run without editing the image.
3. **Patch** (image mount, 4 files): replace the constant with the stage's own
   first layer; `start_layer` is a `SupportsPP` model attribute that is `0` on the
   first stage, so the same line works everywhere and is a no-op under PP=1:

   ```python
   start = getattr(model.model, "start_layer", 0)
   self.cos_cache = model.model.layers[start].self_attn.rotary_emb.cos_cached
   self.sin_cache = model.model.layers[start].self_attn.rotary_emb.sin_cached
   ```

   Safe because `cos_cached`/`sin_cached` are per-layer buffers with identical
   values (`get_rope` per layer, table row above) — say that, because the reviewer
   question is "does layer 0's rope differ from layer 31's?".

## Traps

- **The naive grep misses every hit.** `rg -n "layers\[0\]" vllm_ascend/` returns
  nothing at this ref: the source is line-broken as `layers[` newline
  `0].self_attn…`. Grep `cos_cache is None` (4 hits) or `layers\[$`. A run that
  greps the single-line form concludes "the tree does not have this code".
- **There is no backend escape.** `vllm_ascend/platform.py:318-332` maps
  `(use_mla, use_sfa, use_torchair)` onto `AscendMLABackend` / `AscendSFABackend` /
  `AscendMLATorchairBackend` / `AscendSFATorchairBackend` — all four carry the
  same lookup, so "switch backend" or "enable torchair" only moves the traceback
  to another file (and graph mode adds its own PP failure,
  `pp-nonlast-rank-output-contract.md`).
- **CI cannot catch it, and that is a citable gap, not a cause.**
  `tests/e2e/multicard/test_pipeline_parallel.py` is `Qwen/Qwen3-0.6B`, tp 1, pp 2,
  `mp` + `ray` — a dense model, which uses `attention_v1` and never enters this
  builder — and the workflow line is commented out
  (`.github/workflows/_e2e_test.yaml:192`). Quote both with the ref.
- **Do not answer "PP is unsupported for this model".** The model passes the
  upstream `SupportsPP` gate (no `NotImplementedError`, and upstream's own body
  returns `IntermediateTensors` for a non-last rank), so the failure is one
  PP-blind lookup, not a missing capability.

## Scope and recheck

Single exported pin per case: `v0.11.0rc0` for vllm-ascend, `v0.11.0rc3` for vllm,
so this verdict is source-verified, never reproduced on NPU. At a newer pin the
first check is whether the idiom still exists (`rg -n "cos_cache is None"
vllm_ascend/`): if upstream/downstream replaced it with a stage-aware read, retire
this page rather than citing it. Precedent ids that look adjacent but are not this
bug: #1751 (PP + ray serialization), #2077 (PP=2 Qwen3-Coder-480B startup) —
mention them only as PP history, never as the mechanism.
