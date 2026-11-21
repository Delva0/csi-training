# `Not support mat2 n = 1 or k = 1 when format is FRACTAL_NZ`: a narrow linear cast to NZ kills the load

Verified at vllm-ascend `v0.20.2rc1` (`367b8e6`) + vllm `v0.20.2` (`bc150f5`);
recorded for #10014 (Qwen3.5-35B-A3B, TP4, A2/A3-class 910B, npu-smi 25.3.rc1,
CANN 9.0.0, `--additional-config '{"multistream_overlap_shared_expert": true}'`).
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
  `envs.py:92`).

So for a **bf16** `[1, 2048]` gate the crash is a *mode* question, and the three
modes are not interchangeable: `2` is the aggressive one and makes this certain,
`1` and `0` leave the weight ND.

## Answer ladder (cheapest first)

1. `--additional-config '{"weight_nz_mode": 0}'` (same for `VLLM_ASCEND_ENABLE_NZ=0`)
   — the project's own escape, and on an unquantised checkpoint it costs nothing
   except NZ speed on the big GEMMs. Report the line.
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
   for real GEMMs, so the guard trades nothing.

## Scope, and what stays open

- **Version-scoped.** Reporter-confirmed: the same MoE model works with NZ on
  v0.19.1rc1 and fails on v0.20.2rc1 (only the v0.20.2rc1 tree is in this
  snapshot, so the older gate body is not re-read here — grep
  `rg -n "def is_enable_nz|_should_trans_nz|maybe_trans_nz" vllm_ascend/utils.py`
  at their ref before quoting the mechanism).
- **A dense sibling is the control.** Qwen3.5-27B has no shared-expert gate and
  runs on the same box, which is why "MoE architecture" and "HDK version" were
  both reported as causes; the gate's shape is the difference.
- **Open, do not overclaim:** the same reporter later ran an equivalent stack
  (CANN 9.0.0, npu-smi 25.5.0) with NZ on and served the model, and a bare
  `F.linear` with `w.shape == (1, 2048)` under `VLLM_ASCEND_ENABLE_NZ=1`
  succeeded — so the NZ conversion is necessary but not by itself sufficient, and
  a runtime/driver contribution (`aclnnMatmulWeightNz` behaviour between npu-smi
  25.3.rc1 and 25.5.0) is still possible. State that as the open question; do not
  grade it as the cause.
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
  reporting a missing shape guard inside a function that does not exist, and it
  suppresses the mode advice when the switch is not in the tree.
