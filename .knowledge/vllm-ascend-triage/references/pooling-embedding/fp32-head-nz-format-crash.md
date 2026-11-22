# An fp32 pooling head cast to NZ kills engine start (`aclnnMatmulWeightNz`, EZ1001)

Verified at vllm-ascend `v0.11.0rc3` (`b6d63bb`) + vllm `v0.11.0` (`b8b302c`);
recorded for #4679 (910B, Qwen3-Reranker-8B, `--task score`, reporter says
v0.10.2rc1 worked and v0.11.0rc3 does not). Cross-checked at vllm-ascend
`v0.23.0` (`5cb98ca`) + vllm `v0.23.0` (`0fc695f`), where the same head no longer
trips (see the version split below).

## The signature

```
RuntimeError: matmul_implement_npu:... NPU function error: call aclnnMatmulWeightNz
  failed, error code is 161002
ERR00100 PTA call acl api failed.
AclNN_Parameter_Error(EZ1001): Float32 weight NZ is unsupported by the current SOC
  version [Ascend910B], now self is DT_FLOAT, mat2 is DT_FLOAT .
```

The deepest `vllm_ascend` frames are `ops/linear.py` `forward` →
`ops/linear_op.py` `apply_impl` → upstream `UnquantizedLinearMethod.apply` →
`default_unquantized_gemm` → `torch.nn.functional.linear`. Above them the caller
is the **pooler/classifier** (`vllm/model_executor/models/adapters.py`
`_classifier`), and the engine dies inside `determine_available_memory` →
`profile_run` → `_dummy_pooler_run` (`worker/model_runner_v1.py:2580-2582`,
`:2603-2652`, `:2655-2668`), i.e. before any request and before graph capture.
Both halves matter: the failing op is a stock matmul, and the *first* caller is
the startup dummy pooler run, so `--enforce-eager` cannot move it.

One call decides both halves:
`bash <skill_root>/scripts/nz-format-gate-check.sh <paste|-> [vllm-ascend-root]`
— it classifies the `EZ1001` text, prints the phase of the first hit, and reads
the NZ gate's fp32 policy out of the checked-out `vllm_ascend/utils.py`.

**Leave the custom-op/build lane immediately.** `EZ1001` has two disjoint text
families and only one of them is an install question:

| Text after `EZ1001` | Meaning | Owner |
|---|---|---|
| `Get regInfo failed`, `binary_info_config.json of socVersion [...] does not support opType [...]` | aclnn registry has no binary for this SoC | build/install chain — `../custom-ops-runtime-faults/op-not-registered-for-soc.md` |
| `<Dtype> weight NZ is unsupported by the current SOC version [...]`, `now self is DT_<x>, mat2 is DT_<x>` | this SoC has no kernel for that **dtype × layout** pair; the tensor really arrived in that layout | the Python code that produced the layout (here the weight-format gate) |

## Mechanism at v0.11.0rc3

1. Upstream builds the classifier/score head in **fp32 by design**:
   `vllm/model_executor/models/adapters.py:306-313` —
   `ReplicatedLinear(..., params_dtype=torch.float32, ...)` — and calls it with a
   float input (`:341-343`, `x, _ = self.score(x.float())`).
2. The layer takes vllm-ascend's unquantised method
   (`vllm_ascend/ops/linear.py:79-81`, `quant_config is None` →
   `AscendUnquantizedLinearMethod`), whose
   `process_weights_after_loading` casts the weight to `FRACTAL_NZ` whenever a
   dtype gate passes (`ops/linear.py:43-51`).
3. That gate excludes only half precision: `vllm_ascend/utils.py:74-84` returns
   `False` for `torch.float16`/`torch.bfloat16` and otherwise returns
   `_ENABLE_NZ = VLLM_ASCEND_ENABLE_NZ and model_type != "qwen3_next"`;
   `VLLM_ASCEND_ENABLE_NZ` defaults to `1` (`vllm_ascend/envs.py:167-169`). fp32
   was never excluded, so the head's weight becomes NZ while every bf16 model
   weight stays ND.
4. `aclnnMatmulWeightNz` has no fp32 kernel on `Ascend910B`; the aclnn message
   says so verbatim, so this is not an inference from the traceback.

That is why only pooling/score/classify serves break on an otherwise fine image:
the gate was written for int8 weights, and the classification head is the one
routinely built in fp32. The same tree treats the switch as precision-affecting
from the other side too: `worker/worker_v1.py:183-186` refuses `wake_up` under it
with "FRACTAL_NZ mode is enabled. This may cause model parameter precision issues
… Please set VLLM_ASCEND_ENABLE_NZ=0". Quote that as the flag's own precision
caveat, **not** as project advice about pooling heads — both sites guard RL sleep
and weight-update paths and say "in the RL scenarios", so a reporter can fairly
answer that it does not cover their crash (recorded #4679 was given it that way
anyway, correctly: it is the flag's only in-tree warning). The spelling moves with
the mode refactor: at v0.23.0 the same pair is `worker/worker.py:241-247`
(`weight_nz_mode` truthy → "Please set weight_nz_mode=0 via `--additional-config`")
and `:295-301` (`_check_nz_disabled`, env-var spelling), so name whichever of the
two the reporter's ref accepts.

## Escape and fix

- **Reporter-side, one call:** `VLLM_ASCEND_ENABLE_NZ=0` in the serve
  environment. On an unquantised checkpoint it costs nothing (only the fp32 head
  was being converted). Ask for the confirmation line rather than a log dump.
- **Code-side:** exclude fp32 at the gate (`ops/linear.py:48`) or in
  `is_enable_nz` — NZ helps quantised weights only. Do not "fix" it by forcing
  the head to bf16: vLLM keeps the head fp32 for score fidelity, so that trades a
  crash for a numeric change.
- Not the causes, and say so: an incomplete custom-op package (no custom op runs
  here), a wrong SoC/image (the message names the right SoC and the rest of the
  model loaded), graph capture (`--enforce-eager` is irrelevant — the first hit
  is in eager profile code), and an OOM (`gpu_memory_utilization` will not help).

One family is not about dtype at all: `Not support mat2 n = 1 or k = 1 when format
is FRACTAL_NZ` is the **shape** refusal, and it hits decoder MoE shared-expert gates
(`ReplicatedLinear(hidden, 1)`) as readily as a pooling head —
`../startup-failures/nz-weight-format-refusal.md`. Same gate, different missing test.

## The version split (this is the load-bearing part)

The same fp32 head is built at both pins — at v0.23.0 through
`params_dtype=model_config.head_dtype` (`adapters.py:321-329`), where
`head_dtype` is fp32 for every pooling model when the platform supports it
(`vllm/config/model.py:1675-1709`, `:2088-2106`, opt out with
`--hf-overrides '{"head_dtype": "model"}'`). What changed is the downstream gate:

| Pin | Gate | fp32 outcome |
|---|---|---|
| v0.11.0rc0 (`00ba071`) | **no gate function and no dense hook**: `ops/linear.py` has no `process_weights_after_loading` (the classes only), `utils.py` has no `is_enable_nz`, and the sole unquantized cast is the 310P-gated one in `worker/model_runner_v1.py:2633-2643` where `ACL_FORMAT` is `FRACTAL_ND` off 310P | not converted — so an `EZ1001 … Float32 weight NZ` paste **cannot** come from this rc; check the rc before the release line (`../feature-enable-recipes/weight-nz-layout-by-soc.md`) |
| v0.11.0rc3 | `is_enable_nz(dtype)` (`utils.py:74-84`), cast in `AscendUnquantizedLinearMethod.process_weights_after_loading` (`ops/linear.py:43-51`) | **converted** → crash |
| v0.20.2rc1 | `_should_trans_nz` / `maybe_trans_nz` (`utils.py:203-235`): fp32 → ND (`:205-206`), 310P → always NZ, mode 0 → ND, bf16/fp16 → NZ **only at mode 2** (`:220-222`), other dtypes → NZ; cast in `AscendUnquantizedLinearMethod.process_weights_after_loading` (`ops/linear.py:78-82`) | not converted (fp32 is the first rule) — but the same gate has **no shape test**, so this ref fails instead on a 1-length weight: `../startup-failures/nz-weight-format-refusal.md` |
| v0.23.0 | `_should_trans_nz` / `maybe_trans_nz` (`utils.py:261-298`): first rule `FP32 cannot use NZ` (`:262-264`), `weight.is_meta` → ND (`:266-268`), 310P → NZ (`:270-272`), then `weight_nz_mode` from `get_ascend_config()` (`:274-287`, not the env var); the **unquantised** method still casts (`ops/linear.py:86-98`, `maybe_trans_nz` at `:92`/`:98`) and is safe only because of that first rule — a grep that stops at the cast site reads as if nothing changed; quantised methods cast their own weights (`quantization/methods/w8a8_static.py:155`, `w4a8.py:195`, `w8a8_dynamic.py:149`, …) | not converted, and the caller adds a **310P-only** keep-ND guard for `n = 1` / `k = 1` weights (`_should_keep_nd_for_310p_weight`, `ops/linear.py:79-80`, used at `:88`/`:97-98`) — the shape family is fixed on 310P and untouched on 910B/A2/A3 (`../startup-failures/nz-weight-format-refusal.md`) |

`VLLM_ASCEND_ENABLE_NZ` also changed meaning: from v0.20.2rc1 on it is a
three-valued mode (`envs.py:92` default `1`; `ascend_config.py:216-223` also
accepts `additional_config.weight_nz_mode`; `0` off, `1` only quantised weights,
`2` also bf16/fp16) and `2` is the *aggressive* one, so advice quoted from one ref
can invert on another — including a reporter's own "(or 2) may avoid it", which is
backwards whenever the weight is bf16. Grep the gate at the
reporter's ref (`rg -n "def is_enable_nz|_should_trans_nz|maybe_trans_nz"
vllm_ascend/utils.py`) before naming either the mechanism or the fix, and state
which ref the verdict is verified on.

## Precedent

`#1351`: the NZ weight switch itself (PR #1409, release note) — the flag exists
and is on by default. `#1563`: ND→NZ conversion shows up as `transData` cost on
310P, i.e. the conversion is real work, not bookkeeping. Neither covers an fp32
head; do not stretch them into the mechanism.

## Recheck conditions

- The gate function name and its dtype list (above) change between releases; the
  page's line numbers are the v0.11.0rc3 tree unless the table says otherwise.
- 310P is a different owner: at v0.11.0 the conversion for parallel linears is
  an unconditional loop over the model at load time with `ACL_FORMAT =
  ACL_FORMAT_FRACTAL_NZ` (`worker/model_runner_v1.py:149-153`, `:2685-2693`), and
  at v0.23.0 `_should_trans_nz` short-circuits to `True` on 310P — the env flag is
  not the escape there. Route a 310P report to
  `../soc-platform-paths/atlas-300i-310p.md`.
- The reporter's "it worked on the older release" claim is unverified here (no
  v0.10.2rc1 tree in the snapshot); the bisect is the two gate bodies, not a
  release survey.
