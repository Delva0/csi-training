# `Not support mat2 n = 1 or k = 1 when format is FRACTAL_NZ`: a narrow linear cast to NZ kills the load

Verified at vllm-ascend `v0.20.2rc1` (`367b8e6`) + vllm `v0.20.2` (`bc150f5`),
re-read at vllm-ascend `v0.22.1rc1` (`5f6faa0`) + vllm `v0.22.1` (`0decac0`), and
again at vllm-ascend `v0.23.0` (`5cb98ca`) + vllm `v0.23.0` (`0fc695f`): same
signature and same gate, but that ref **does** carry a shape guard — scoped to
310P, which changes the reader's first check (see *The owner* below).
Recorded for #10014 (Qwen3.5-35B-A3B, TP4, A2/A3-class 910B, npu-smi 25.3.rc1,
CANN 9.0.0, `--additional-config '{"multistream_overlap_shared_expert": true}'`) and
for #11621 (Qwen3.6-35B-A3B bf16, TP2 + `--enable-expert-parallel`, MTP spec,
`VLLM_ASCEND_ENABLE_NZ=2`, npu-smi 25.5.0, CANN 9.0.0, same flag; the paste's own
frames `fused_moe.py:488/:493/:500/:501` match the v0.22.1rc1 pin line for line).
#11621 is **reporter-confirmed cleared by `export VLLM_ASCEND_ENABLE_NZ=0`**
(2026-07-08, "I worked around this issue by setting" it) — the same escape as
#10014, on the newer stack (npu-smi 25.5.0).
The third `EZ1001` text family — not the aclnn **registry** family
(`../custom-ops-runtime-faults/op-not-registered-for-soc.md`) and
not the **dtype × layout** family
(`../pooling-embedding/fp32-head-nz-format-crash.md`): here the op is registered,
the dtype is supported, and the **shape** cannot be stored in `FRACTAL_NZ`.

## The signature

```
RuntimeError: matmul_implement_npu:... call aclnnMatmulWeightNz failed, error code is 161002
AclNN_Parameter_Error(EZ1001): Not support mat2 n = 1 or k = 1 when format is FRACTAL_NZ, mat2: [2048, 1]
  File ".../vllm_ascend/ops/fused_moe/fused_moe.py", line 497, in _validate_shared_expert_consistency
    integrated_out = self._shared_experts(test_input)
  File ".../vllm/model_executor/models/qwen2_moe.py", line 120, in forward
    out = F.sigmoid(self.expert_gate(x)[0]) * out
  File ".../vllm_ascend/ops/linear.py", line 52, in unquantized_gemm
    return torch.nn.functional.linear(x, weight, bias)
```

The text is the reporter's; aclnn is not in the snapshot, so the constraint is
quoted, not verified in source. `mat2` is the *transposed* weight, so
`n = 1` is `out_features == 1` and `k = 1` is `in_features == 1`: any linear whose
weight has a 1-length dimension is unmatmul-able in NZ, and the fix is always to
leave that weight ND — no shape of NZ weight helps.

## Read the resolved mode out of the paste, not out of the environment

The mode decides this page, and the serve log prints it. `ascend_config.py:290-302`
(`_get_config_value`) logs
`AscendConfig.weight_nz_mode is set from additional_config with value <v>` when the
key is present, otherwise
`AscendConfig.weight_nz_mode falls back to environment variable
VLLM_ASCEND_ENABLE_NZ with value <v>. Please use additional_config.weight_nz_mode
instead …` when the env var is exported (`:295` in the #11621 log, value `2`), and
prints **nothing** when neither is set — then the mode is the `envs.py:92` default
`1`. So `grep 'weight_nz_mode'` on the paste settles it in one line: no question to
the reporter, and no migration story to write.
`Unknown vLLM environment variable detected: VLLM_ASCEND_ENABLE_NZ` is upstream
`envs.py`'s vocabulary check, which flags `VLLM_USE_V1` too: noise, not evidence
that the switch was ignored.

## Where it fires (which is why the obvious ablations miss)

- `shared_expert_gate` is a **one-output** `ReplicatedLinear`:
  `vllm/model_executor/models/qwen2_moe.py:149-155` (`ReplicatedLinear(hidden, 1,
  bias=False)`; the same builder at `qwen3_next.py:130-136`), applied at
  `qwen2_moe.py:112/:120`. It is replicated, so the tensor-parallel size changes
  nothing about it — a title that blames TP4 is the wrong lever.
- The raise happens **during weight load**, inside a self-check that only exists
  under one `--additional-config` key: `multistream_overlap_shared_expert`
  (`ascend_config.py:131`, default `False`) makes FusedMoE wrap
  `quant_method.process_weights_after_loading` and run
  `_validate_shared_expert_consistency()` on a `torch.rand(10, hidden_size)`
  probe (`ops/fused_moe/fused_moe.py:385`, `:474-497`, `:522-523` for the same
  gate in the real forward). So the failure lands before `profile_run` and before
  graph capture: `--enforce-eager`, `--compilation-config` and `--max-model-len`
  are not levers.
- Dropping the flag removes the **check**, not the defect: the real forward on
  the first request hits the same weight.

## The owner: the NZ gate has no shape test

- `AscendUnquantizedLinearMethod.process_weights_after_loading`
  (`vllm_ascend/ops/linear.py:78-82`) casts **every** non-`conv1d` linear weight
  through `maybe_trans_nz`; the quantised methods do the same for their weights
  (`quantization/methods/w8a8_static.py:155`, `_310p/quantization/methods/*`).
- `maybe_trans_nz` → `_should_trans_nz` (`vllm_ascend/utils.py:203-235`) tests
  dtype, SoC and mode **only**: fp32 → ND (`:205-206`), 310P → always NZ
  (`:209-210`), mode 0 → ND (`:217-218`), bf16/fp16 → NZ **only at mode 2**
  (`:220-222`), any other dtype → NZ (`:224-225`). No `shape` test exists.
- Mode at this ref: `weight_nz_mode`, `0` off / `1` quantised only (default) /
  `2` bf16+fp16 too, resolved from `additional_config["weight_nz_mode"]` then
  `VLLM_ASCEND_ENABLE_NZ` (`ascend_config.py:216-223`; env default `1`,
  `envs.py:92`; at v0.23.0 the same resolution is `ascend_config.py:247-254`,
  and the gate reads `get_ascend_config().weight_nz_mode` rather than the env
  var — `utils.py:274-276`).
- **v0.23.0 adds the shape guard, and it is SoC-scoped.** `_should_trans_nz`
  still has no shape test (`utils.py:261-287`), but
  `AscendUnquantizedLinearMethod.process_weights_after_loading` now computes
  `keep_nd_weight = _should_keep_nd_for_310p_weight(layer.weight.data)`
  (`ops/linear.py:88`) — `is_310p() and weight.ndim >= 2 and (weight.shape[-1] ==
  1 or weight.shape[-2] == 1)` (`:79-80`; `is_310p()` is
  `get_ascend_device_type() == AscendDeviceType._310P`, `utils.py:148-149`) — and
  skips the cast for such a weight (`:93-98`, comment "310P torch_npu rejects
  FRACTAL_NZ matmul when the weight-side matrix has n=1 or k=1 … leaving
  non-310P policy intact"). So at v0.23.x a 310P `[1, H]` gate is clear while a
  **910B/A2/A3 one still converts and still fails identically**: read the SoC
  off `npu-smi` / the paste before quoting either half.

So for a **bf16** `[1, 2048]` gate the crash is a *mode* question, and the three
modes are not interchangeable: `2` is the aggressive one and makes this certain,
`1` and `0` leave the weight ND.

## Answer ladder (cheapest first)

1. `--additional-config '{"weight_nz_mode": 0}'` (same for `VLLM_ASCEND_ENABLE_NZ=0`)
   — the project's own escape, and on an unquantised checkpoint it costs nothing
   except NZ speed on the big GEMMs. Report the line. **Reporter-confirmed on
   #11621** (npu-smi 25.5.0, mode 2): the deployment loads after it.
2. If they only want the narrow weights off: `VLLM_ASCEND_ENABLE_NZ=1` works
   **when the gate weight is bf16/fp16** — and the paste's own "(or 2) may avoid
   it" is inverted, say so.
3. Ask for `env | grep -i VLLM_ASCEND_ENABLE_NZ` plus the full
   `--additional-config` (the paste carried neither): mode `2` and a
   non-bf16 gate weight are the two ways the default state reaches the cast, and
   the two need different answers.
4. Code-side fix: one guard at the top of `_should_trans_nz`
   (`if weight.shape[0] == 1 or weight.shape[1] == 1: return False`), or skip the
   `[1, H]` probe in `_validate_shared_expert_consistency`. NZ is a perf layout
   for real GEMMs, so the guard trades nothing. At v0.23.0 that guard exists
   already but is gated on `is_310p()` (`ops/linear.py:79-80,88,97-98`), so the
   ask at a 910B ref is one line — drop the SoC test, or move the test into
   `_should_trans_nz` — not a new guard. Do not offer "go to mode 1" as that fix:
   the 310P branch returns `True` before the mode is read (`utils.py:270-272`),
   so on 310P the switch is inert in both directions.

## Scope, and what stays open

- **Version-scoped.** Reporter-confirmed: the same MoE model works with NZ on
  v0.19.1rc1 and fails on v0.20.2rc1 (only the v0.20.2rc1 tree is in this
  snapshot, so the older gate body is not re-read here — grep
  `rg -n "def is_enable_nz|_should_trans_nz|maybe_trans_nz" vllm_ascend/utils.py`
  at their ref before quoting the mechanism).
- **A dense sibling is the control.** Qwen3.5-27B has no shared-expert gate and
  runs on the same box, which is why "MoE architecture" and "HDK version" were
  both reported as causes; the gate's shape is the difference.
- **The mode is the discriminator, so resolve it before positing a driver.**
  An earlier version of this page filed a runtime contribution (`aclnnMatmulWeightNz`
  behaviour between npu-smi 25.3.rc1 and 25.5.0) as open, on a later run that served
  the model "with NZ on" plus a bare `F.linear` whose `w.shape == (1, 2048)`
  succeeded under `VLLM_ASCEND_ENABLE_NZ=1`. That control cannot discriminate: at
  mode 1 a bf16 weight is never cast (`utils.py:231-232`). #11621 is the same shape
  on npu-smi 25.5.0 **at mode 2**, it crashed, and `VLLM_ASCEND_ENABLE_NZ=0` cleared
  it — so the conversion is necessary *and*, for a bf16 1-length weight at mode 2,
  sufficient, and no driver difference needs to be postulated. **Corrected at
  v0.23.0:** that ref does ship a shape guard, but 310P-only
  (`ops/linear.py:79-80,88,97-98`), so the mechanism above is unchanged on
  910B/A2/A3 and what stays open is the guard's *scope*, not its absence —
  aclnn is not in the snapshot, so nothing here says whether a non-310P
  `aclnnMatmulWeightNz` would accept a 1-length NZ weight; the 310P-only comment
  is the only evidence that the authors saw the refusal as SoC-specific.
- **Same body at v0.22.1rc1** (#11621): wrapper
  `ops/fused_moe/fused_moe.py:477-491` (`wrapped_process_weights` calls
  `_validate_shared_expert_consistency()` at `:488`), probe `:493-503`
  (`torch.rand(10, hidden_size)` at `:495-497`, `integrated_out =
  self._shared_experts(test_input)` at `:501` — the pasted frame), gate
  `utils.py:209-236` (fp32 `:211-212`, meta `:215-216`, 310P `:219-220`, mode 0
  `:227-228`, bf16 `:231-232`), mode `ascend_config.py:218-226` / `:290-302`,
  default `envs.py:92`, cast `ops/linear.py:78-89`. Nothing added a shape test, and
  the four pasted frame numbers date the tree to this rc.
- **The mode and the gate are newer than the 0.11 line.** At v0.11.0rc0 there is
  no `weight_nz_mode`, no `VLLM_ASCEND_ENABLE_NZ` and no gate function in
  `utils.py` at all — the casts are inline in the quant methods and in
  `worker/model_runner_v1.py:2633-2643`, which is why `weight_nz_mode: 0` is not
  an escape there. The capability-side answer (which weights are already NZ on
  which SoC, and the `is_310p()` polarity) is
  `../feature-enable-recipes/weight-nz-layout-by-soc.md`.
- One call classifies the paste and reads the gate at their ref:
  `bash ../../scripts/nz-format-gate-check.sh <paste|-> <vllm-ascend-root>` — it
  now says "no NZ gate function at this ref" when the gate is absent, instead of
  reporting a missing shape guard inside a function that does not exist, it
  suppresses the mode advice when the switch is not in the tree, and it prints the
  paste's own resolved-mode line.
