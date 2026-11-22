# `NotImplementedError: No compressed-tensors compatible scheme was found for quant_type=<X>, layer_type=<Y>`

Verified at vllm-ascend v0.15.0rc1 (`3d43ed997e71`) with vllm v0.15.0. Recorded
#7076: Kimi-K2.5 on one A3 node, TP 16 + `--enable-expert-parallel`,
vllm 0.15.0 + vllm-ascend 0.15.0rc1. Lane B, **model construction**, before any
weight shard is read:

```
multiproc_executor.py:743 worker_main → :578 self.worker.load_model()
vllm_ascend/worker/worker.py:429 → model_runner_v1.py:2341 get_model
vllm/model_executor/model_loader/__init__.py → base_loader.py:50 initialize_model
kimi_k25.py:331 MoonViT3dPretrainedModel → kimi_k25_vit.py:576/485/367/315 MLP2.fc0
vllm_ascend/ops/linear.py:351 AscendLinearBase.__init__ → :83 quant_config.get_quant_method
vllm_ascend/quantization/compressed_tensors_config.py:160 → :202 → :325 raise
```

`WorkerProc failed to start.` follows it, and the title is usually the message
itself. The failing layer is the *first* linear the model builds, so its family
(vision tower, `mm_projector`, dense attention) says nothing about scope — every
dense linear behind it fails the same way.

## What the message means (one read)

`_create_scheme_for_layer_type` (`compressed_tensors_config.py:298-328`) does
`get_scheme_class(quant_type, layer_type)` and **raises** on `None`. Upstream
vLLM returns `None` from `get_quant_method` for a layer its config matched but it
has no method for, and that layer runs `UnquantizedLinearMethod`. This fork has
no such fallback, so the message means exactly one thing: *the checkpoint's
quant config selected this layer and the release registers no scheme for that
`(quant_type, layer_type)` pair*. Nothing about the image, device, CANN, TP/EP,
serve flags or weight files is involved — do not rank them.

## Two reads decide which pairs exist at the pin

```bash
rg -n "@register_scheme" vllm_ascend/quantization/methods/*.py
rg -n -A25 "Currently Supported Quantization Algorithms" docs/source/*/feature_guide/quantization.md
```

At v0.15.0rc1 the registry holds `W4A16` → `moe` only (`methods/w4a16.py:102`);
`W8A16`, `W8A8`, `W8A8_DYNAMIC`, `W4A8_DYNAMIC`, `W4A4_DYNAMIC`,
`W4A4_FLATQUANT_DYNAMIC`, `W8A8_MXFP8`, `W8A8_MIX` carry `linear`, and only
`W8A8_DYNAMIC` / `W4A8_DYNAMIC` / `W8A8_MIX` carry `moe` as well. The
docs row states the same intent in words: `W4A16 … specifically designed for MoE
model expert layers` (`docs/source/developer_guide/feature_guide/quantization.md:104`).
So `layer_type=linear` + `W4A16` on a dense layer is a **documented support gap**
at this ref, and the project's own table is the citation.

The older generator of the same class lives in `quantization-method-gate.md`
(*per-layer coverage*): the pre-registry form prints `Currently, vLLM Ascend
doesn't support <type> for <layer>` from `ASCEND_QUANTIZATION_METHOD_MAP`. Read
which file the paste's frame names before quoting either message.

## Do not spend turns on *how* the layer received a quant_config

This is the recorded detour: #7076's session spent ~10 turns and most of its
output budget proving that the pinned tree could not hand a `quant_config` to
the vision tower (`MLP2` passes none; `AscendLinearBase` is `None` →
`AscendUnquantizedLinearMethod`) and never delivered a message.

The pin's **upstream** commit is a guess, not the reporter's tree:
`SOURCES.json` records it with `basis: plugin_dockerfile_default` and
`runtime_commit_verified: false`. Check frame alignment before using a pinned
upstream file to rule a path out — compare the paste's `<file>:<line>` with the
pinned files:

- in #7076 every **downstream** frame matched exactly
  (`ops/linear.py:83` and `:351`, `compressed_tensors_config.py:160/:202/:325`),
  so the vllm-ascend pin is right;
- the upstream model frames drifted and the drift grew down the file
  (`kimi_k25.py` 331 vs 325; `kimi_k25_vit.py` 315 vs 313, 367 vs 362,
  576 vs 565), i.e. the reporter runs a slightly different vllm build that does
  thread `quant_config` into the vision tower.

Downstream matches + upstream drifts ⇒ answer from the downstream frame; the
"impossible" code path is the drift. Verify with `grep -n` on the pinned files,
never by counting `sed` output.

## Answer and ladder

Verdict first, in the report's language: the checkpoint's W4A16 quant config
reaches **dense** linears and this release implements W4A16 for MoE experts
only, so the model cannot load at vllm-ascend 0.15.0rc1 — no flag changes that.

1. **Checkpoint census, one command** (it bounds everything else):
   `python -c "import json;c=json.load(open('<ckpt>/config.json'))['quantization_config'];print(c.get('config_groups'));print(c.get('ignore'))"`.
   Targets covering `Linear`/the vision tower, with an `ignore` list that does
   not name them, mean every dense linear is in scope.
2. **Confirm the gap in the installed tree**:
   `python -c "from vllm_ascend.quantization.methods import get_scheme_class as g;print(g('W4A16','linear'), g('W4A16','moe'))"`
   printing `None` plus a class is the whole proof.
3. **Then choose**: a vllm-ascend ref that registers a `W4A16`/`linear` scheme (or
   an equivalent dense-4-bit scheme for that SoC), or a checkpoint whose `ignore`
   list leaves the non-MoE linears unquantized. Escalate with the registry grep
   and the docs row — a request for dense W4A16 support is a real feature ask,
   not a config bug.
4. **Do not** offer `--enforce-eager`, `--compilation-config`, a TP/EP change, a
   re-downloaded checkpoint or `--dtype` as rungs: the failure precedes weight
   loading, so none of them can move it.

Send back: the `config.json` quant block, the full per-rank traceback (to
confirm the constructor chain rather than a weight-load error), and the two
`get_scheme_class` values. Keep the escalation one paragraph — the recorded
session emitted **no** message at all in 25 turns and hit the 16 384-token cap
inside its own reasoning, so the reporter got nothing.
