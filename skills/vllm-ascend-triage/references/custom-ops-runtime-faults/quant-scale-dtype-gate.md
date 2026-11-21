# Quant-matmul scale dtype gate: `161002`, `aclnnQuantMatmulWeightNz` / `aclnnGroupedMatmulWeightNz`

## Signature

```
RuntimeError: npu_quant_matmul_symint: NPU function error: call aclnnQuantMatmulWeightNz
  failed, error code is 161002
AclNN_Parameter_Error(EZ1001): Tensor scale not implemented for DT_FLOAT16,
  should be in dtype support list [DT_UINT64, DT_BFLOAT16, DT_INT64, DT_FLOAT].
```

The MoE twin is the same gate on the grouped op, and states the contract outright:

```
aclnnGroupedMatmulWeightNz failed, error code is 161002
per-token quant case only supports scale data type bfloat16 with output data type bfloat16,
or scale with data type float32 when output is float16,
but now scale[0] has data type FLOAT16 and output has data type FLOAT16!
```

This is the third member of the `EZ1001` family (`index.md`): the op is registered
and the *launch* was rejected because a **tensor dtype** input is outside the
kernel's list. The owner is the Python that built the scale tensor — a vllm-ascend
quantization method — never the CANN package, the driver, or the
`--data-parallel-size`/`--tensor-parallel-size`/`--enable-expert-parallel` flags.
Recorded #7443 (v0.17.0rc1, DeepSeek-V3.1 W8A8, `--quantization ascend`): the
linear hit fires inside `determine_available_memory() → profile_run()` on the first
quantized dense linear (`deepseek_v2.py` `gate_up_proj` → `ops/linear.py` →
`method_adapters.py` → `w8a8_dynamic.py`), i.e. before any request and before any
flag can matter.

## Rule 1 — a float32 copy existing is not the path *using* it

At v0.17.0rc1 (`e20f0b1a0d2f`) the linear method already creates the right tensor
and never passes it:

| Fact | Anchor |
|---|---|
| scale is allocated in `params_dtype` (= the **model** dtype) | `vllm_ascend/quantization/methods/w8a8_dynamic.py:69` |
| `apply()` hands CANN that tensor | `w8a8_dynamic.py:86-89` — `npu_quant_matmul(quantized_x, layer.weight, layer.weight_scale, …)` |
| the float32 copy is created and unread | `w8a8_dynamic.py:102-103` — `weight_scale_fp32 = weight_scale.data.to(torch.float32)` |

So `rg -n '<param>_fp32'` finding the name proves nothing about the call site: read
`apply()` (the launch), not `process_weights_after_loading()` (the declaration).

## Rule 2 — sweep the parameter family before answering; the sibling site is the second round trip

A dtype contract violated at one call site is normally violated at its siblings,
because the schemes were copied from each other. Full sweep at v0.17.0rc1:

| Site | What it passes | float32 companion |
|---|---|---|
| linear `AscendW8A8DynamicLinearMethod.apply` `:86-89` | `layer.weight_scale` | `weight_scale_fp32` `:103`, **unused** |
| `W8A8_MIX` linear `w8a8_pdmix.py:72-82` | delegates to the dynamic method on the prefill / KV-producer side | `weight_scale_fp32` `:84`, **unused** |
| MoE w1 `AscendW8A8DynamicFusedMoEMethod.apply:247` | `[layer.w13_weight_scale_fp32]` | yes, `:278` |
| MoE w2 `:249` | `[layer.w2_weight_scale]` | **none at this ref** — w2 only gets the reshape at `:280` |
| dynamic-EPLB lists `:242-244` | `w13_weight_scale_fp32_list` / `w2_weight_scale_list` | fp32 list only for w13 (`:290-293`) |

Reporter-confirmed consequence (#7443, public follow-up comment, fix pushed as PR
#7444): the one-line linear repair was applied and **the identical error came back
in the MoE path** on the next run — `aclnnGroupedMatmulWeightNz … 161002`, w2 scale
fp16 with fp16 output. An answer that names only the reported method buys the
reporter a second crash and forces a second round trip, so the deliverable lists
every site above (and the `_list` variants, which the fix must cover too).

One call for the sweep, at the pinned tree:
`uv run --no-project python <skill_root>/scripts/quant-scale-dtype-audit.py <vllm-ascend-root>`
— prints every scale argument handed to a quant/grouped matmul, each `_fp32`
companion, the copies nobody reads, and the sites whose output dtype is derived from
a scale tensor.

## Rule 3 — read *both* operands of the gate: the output dtype is derived from the scale

The gate compares the scale dtype **and** the output dtype, so a scale-only repair is
incomplete wherever the output dtype comes from the scale. In `vllm_ascend/ops/fused_moe/moe_mlp.py`
at the same ref: `:144` `_output_dtype = w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype`,
`:216` `fallback_output_dtype=w2_scale[0].dtype`, then `:229` / `:244` pass it as
`output_dtype=`. With an fp16 model the grouped op therefore receives scale fp16
*and* output fp16 — the exact pair the paste rejects. The existing rescue is one
operand at one site: `:175` `if w1_scale[0].dtype != torch.float32: w1_scale[0] = w1_scale[0].to(torch.float32)`
covers gmm1's scale only. The durable repair therefore has two halves — a float32
scale companion (plus its `_list` form for dynamic EPLB) **and** `_output_dtype`
taken from the model/hidden dtype (`input_hidden_dtype`) instead of from the scale.
State both; the reporter's own patch needed both.

## Rule 4 — the gate is dtype-conditional, which is why it survives in CI and why "only I see it" is not a local fault

`DT_BFLOAT16` and `DT_FLOAT` are in the support list, `DT_FLOAT16` is not, and the
scale is allocated in the model dtype — so a default `bfloat16` DeepSeek run
executes the same line and succeeds — the bf16 default is the reporter's own control
path, which is `SKILL.md` rule 15 read in the other direction: a sibling that works on
the same code narrows the cause to the one value that differs. The error text itself proves the reporter's model dtype; the
artefact to ask for is `<ckpt>/config.json` `torch_dtype` and whether `--dtype
float16` is in the serve command, not hardware state. A bf16 re-run of the same
command is the one-line confirmation of the mechanism.
Precedent: a titles-only search for `DT_FLOAT16|scale not implemented` over the
prior-month corpus returns 0 hits — a corpus gap, not evidence the defect is new.

## Do not chase the neighbours in the same paste

`Node[TransposeKvCacheByBlock]: failed to initialize … input2.blockIDs. Tensor scale
not implemented for DT_FLOAT16` appearing beside the quant-matmul error is CANN
re-printing the aborted stream's gate: that node has no scale operand. Treat it as
context noise; only a block that survives the repair is a second fault (then ask for
the full plog block with `ASCEND_LAUNCH_BLOCKING=1`).

## Scope and recheck

Source-verified at vllm-ascend v0.17.0rc1 `e20f0b1a0d2f` (vllm v0.17.0
`b31e9326a7d9`); the MoE second occurrence is reporter-confirmed on #7443, not
executed here (no NPU in the case view). Recheck condition: re-run the audit script
or `rg -n 'weight_scale_fp32|w2_weight_scale' vllm_ascend/quantization/methods/w8a8_dynamic.py`
on the reporter's own tree — if `apply()` now reads a `_fp32`/`_fp32_list` tensor and
the MoE output dtype is no longer taken from the scale, the repair has landed and
this page describes an older ref.
