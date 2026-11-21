# `Promotion for Float8 Types is not supported … Float8_e4m3fn and Bool` at the MoE dispatch

Lane B, every rank, on the **first MoE forward** — in the recorded case inside
`determine_available_memory` → `profile_run` → `_dummy_run`, i.e. immediately after
the weights load:

```text
... vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py:494 in forward_impl
... vllm_ascend/quantization/methods/w4a8_mxfp4.py:205 in apply
... vllm_ascend/ops/fused_moe/moe_comm_method.py:148 in fused_experts
... vllm_ascend/ops/fused_moe/token_dispatcher.py:393 in token_dispatch
    topk_weights = topk_weights * mask
RuntimeError: Promotion for Float8 Types is not supported, attempted to promote
Float8_e4m3fn and Bool
```

Recorded #11498 (vllm-ascend release/v0.23.0 `79928c2`, DeepSeek-V4-Flash w4a8
recipe, TP4 + `--enable-expert-parallel`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` and
`enable_flashcomm1: true`). No device op failed: two torch dtypes met in one
multiply, and the multiply is optional.

## Three conditions, all needed (verified at `79928c2`)

| Condition | Site | Effect |
|---|---|---|
| routing weights are cast to the **activation** dtype | `quantization/methods/w4a8_mxfp4.py:202` `topk_weights = topk_weights.to(x.dtype)`, then `:205` `moe_comm_method.fused_experts(…)`; the method is `AscendW4A8MXFPDynamicFusedMoEMethod`, `quant_type = QuantType.W4A8MXFP` (`:103`) | `topk_weights` inherits whatever dtype the hidden states carry |
| the hidden states are **fp8** | `ops/fused_moe/prepare_finalize.py:369-373` — `_prepare_with_ep_group` runs `npu_dynamic_mx_quant(…, dst_type=torch.float8_e4m3fn)` for `QuantType.W4A8MXFP`/`MXFP8`. That branch is reached only when `enable_sp() or enable_sp_by_pass()` (`:358`, class `PrepareAndFinalizeWithAllGather` at `:321`), and `enable_sp()` **is** the flashcomm1 flag (`utils.py:877-900`, `additional_config.enable_flashcomm1` or `VLLM_ASCEND_ENABLE_FLASHCOMM1`). `_prepare_with_dp_group` (`:428`) never quantises | W4A8-MXFP **+ flashcomm1** ⇒ `x.dtype == torch.float8_e4m3fn`; the same launch without SP keeps bf16 activations |
| the **bool mask multiply** runs | `ops/fused_moe/token_dispatcher.py:390-393` in the AllGather dispatcher: `mask = expert_map[topk_ids] != -1; topk_weights = topk_weights * mask` | an fp8 × bool multiply, which torch refuses. `expert_map` is non-`None` only under EP. The 310P twin is `_310p/fused_moe/token_dispatcher.py:54-56` |

The comm method matters as well because only the AllGather dispatcher multiplies by
that mask: `select_moe_comm_method` (`ascend_forward_context.py:303`) with
`_select_a5_moe_comm_method` (`:285`) returns MC2 while
`num_tokens <= mc2_tokens_capacity`, then ALLGATHER when
`world_size <= num_experts_per_tok`, else ALLTOALL. A large prefill/profile batch is
exactly where ALLGATHER wins — which is why the crash surfaces at `profile_run` and
not on a small decode.

Not the owner: `select_experts`' own
`topk_weights = topk_weights.to(hidden_states.dtype)`. It runs at
`fused_moe_0_23_0.py:453`, before the `prepare` call at `:476`, so it sees the bf16
states. Do not spend turns there.

## Procedure

1. Three `sed` reads in the reporter's tree settle the mechanism:
   `sed -n '196,210p' vllm_ascend/quantization/methods/w4a8_mxfp4.py`,
   `sed -n '386,396p' vllm_ascend/ops/fused_moe/token_dispatcher.py`,
   `sed -n '362,380p' vllm_ascend/ops/fused_moe/prepare_finalize.py`. If all three
   match the table above, the report is a dtype interaction, not a hardware,
   CANN, checkpoint or EP-config fault — say which of the three is the *new* one
   in the reporter's recipe (usually flashcomm1).
2. Confirm the three launch facts from the pasted script: a quantised variant with
   `--quantization ascend`; flashcomm1 on (env **or**
   `additional_config.enable_flashcomm1` — both spellings exist); and
   `--enable-expert-parallel`.
3. Cheapest discriminating ablations, in this order:
   - **flashcomm1 off** (`VLLM_ASCEND_ENABLE_FLASHCOMM1=0` and drop
     `enable_flashcomm1`): `_prepare_with_dp_group` leaves bf16 activations, so
     `topk_weights` stays bf16 and the mask multiply is legal. The W4A8-MXFP grouped
     matmul still runs — it quantises internally (`w4a8_mxfp4.py:221-222` passes
     `mxfp_act_quant_type=torch.float8_e4m3fn` / `mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2`,
     and `:225` `mxfp_use_bf16=(x.dtype == torch.bfloat16)` is the branch taken).
     Startup past this point confirms the mechanism; the cost is the SP feature.
   - **`--enable-expert-parallel` off**: `expert_map` is `None`, the mask multiply
     never runs — same confirmation, different cost (and a different MoE comm path).
   - `--enforce-eager` and the graph/compilation knobs are **not** levers here: the
     raise is a Python op on the first forward and reproduces eagerly.
4. What a fix looks like, stated as a direction rather than a patch: make the mask
   multiply dtype-safe (`topk_weights * mask.to(topk_weights.dtype)`, or
   `masked_fill(~mask, 0.0)`), and raise the deeper question — casting *routing*
   weights to `float8_e4m3fn` costs a 3-bit mantissa (weights below ~2⁻⁹ flush to
   0), while the MC2 path deliberately casts `topk_weights` to fp32 for its
   `expert_scales`. Unverified in this snapshot: whether
   `npu_moe_token_unpermute` requires `probs` to match the token dtype. If a local
   mask-cast patch then fails inside `token_combine`, that constraint is the next
   thing to read — not a second bug.

## Scope and limits

Present at `79928c2` (`release/v0.23.0`). `w4a8_mxfp4.py` is shared by
`ops/fused_moe/fused_moe.py` and `ops/fused_moe/fused_moe_0_23_0.py`, so the defect
is not specific to the 0.23 model file. The snapshot is a single commit: whether a
later release changed the cast at `:202` or the mask site is **not** verified here —
re-grep both lines before quoting them for another release. The trigger depends on
device type through the comm-method selector above (A5/A2/A3 branches), so a report
on another SoC can take a different dispatch path to the same mask multiply.
