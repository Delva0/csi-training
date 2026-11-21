# `Cannot find bin of op <Op>, integral key <dtypes>` — no kernel binary for that dtype key

Verified at vllm-ascend `v0.10.0rc1` (`4604882a3e2c`) and vllm `v0.10.0`
(`6d8d0a24c02b`), recorded for **#2387**: 4×910B4, `Qwen3-32B-w8a8`,
`quay.io/ascend/vllm-ascend:v0.10.0rc1`, `vllm serve <ckpt> --max-model-len 10240
--port 8000 -tp 4` (no `--dtype`). All four ranks die in the same second in the
first forward after `torch.compile takes 12.21 s in total`, i.e. before the engine
is ready.

## The signature

```text
[rank0]:[E815 02:22:49] operator():…/torch_npu.dir/compiler_depend.ts:3785
         NPU function error: call aclnnAddRmsNormQuant failed, error code is 561000
[ERROR] … (PID:14228, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
EZ9999: Inner Error!
EZ9999: [PID: 14228] … Cannot find bin of op AddRmsNormQuant, integral key
        0/1/|float16/ND/float16/ND/float16/ND/float16/ND/int8/ND/int8/ND/float16/ND/.
        Cannot find binary for op AddRmsNormQuant.
        Kernel Run failed. opType: 31, AddRmsNormQuant
        launch failed for AddRmsNormQuant, errno:561000.
```

Read the **clause**, not the number, and the clause is a *fourth* pre-launch form
(after the `EZ1001` registry text, the `<Dtype> weight NZ is unsupported` dtype ×
layout text and the `161002` `Tensor scale not implemented` text):

| Wording | What it proves | Owner / page |
|---|---|---|
| `EZ1001 … Get regInfo failed`, `binary_info_config.json of socVersion […] does not support opType […]` | no registry entry for the op on this SoC | build/install — `op-not-registered-for-soc.md` |
| `EZ9999 … does not has any binary` / `Parse dynamic kernel config fail`, `AclOpKernelInit failed opType` | no kernel entry **for the op at all** on the running SoC | image-CANN vs card — `../soc-platform-paths/kernel-package-vs-card.md` |
| **`EZ9999 … Cannot find bin of op <Op>, integral key <key>` + `Kernel Run failed. opType: N`** | the op is registered; **no binary matches this dtype/format key**. Nothing launched | the caller's dtypes first, then the image — this page |
| `EZ1001 … Tensor scale not implemented for DT_<x>, should be in dtype support list […]` (`161002`) | the op is registered; this launch was **rejected** on an operand dtype | the quant method's allocation — `quant-scale-dtype-gate.md` |

`561000` + `ERR00100` is the generic aclnn/PTA wrapper: the same pair prefixes the
MC2 dispatch rows (`../hccl-and-hardware/link-establish-timeout.md`) and #6082's
`does not has any binary`. A grep on `561000` alone therefore lands in the wrong
lane — search the distinctive clause (`Cannot find bin`, `integral key`).

## Read the integral key against the model dtype

The segment after `|` is the requested `dtype/format` list in signature order —
here four `float16/ND`, two `int8/ND`, one `float16/ND`. The `float16` entries are
the **model dtype**, and the model dtype is on one line of the paste: the engine
init config prints `dtype=torch.float16`. `--dtype` defaults to `auto` (the
checkpoint's `transformers` `torch_dtype`), so with no `--dtype` on the command
line the checkpoint decided the key.

Which caller allocates those dtypes, in the recorded shape (source-verified at
`4604882a3e2c`):

| Evidence | Claim |
|---|---|
| `vllm_ascend/models/qwen3.py:47-57` | both norms of a decoder layer become `AddRMSNormW8A8Quant` when that layer's `qkv_proj` / `gate_up_proj` quant method is the **static per-tensor** `AscendW8A8LinearMethod` |
| `vllm_ascend/ops/layernorm.py:49-54` | `torch_npu.npu_add_rms_norm_quant(x, residual, weight, self.layer.aclnn_input_scale, self.layer.aclnn_input_offset, epsilon=…)` → `aclnnAddRmsNormQuant` |
| `vllm_ascend/quantization/w8a8.py:60-64` | `input_scale` is allocated in `params_dtype` = the model dtype; `input_offset` loads as `int8` |
| `vllm_ascend/quantization/w8a8.py:125-135` | `aclnn_input_offset` is cast to the **scale's** dtype, so both fused operands carry the model dtype into the key |
| `vllm_ascend/quantization/quant_config.py:62-64` | `get_supported_act_dtypes()` returns `[int8, float16, bfloat16]`, so vLLM's own gate **admits** an fp16 W8A8 checkpoint and the mismatch only surfaces at the op; the same class also reads `quant_model_description.json` (`:70-72`) |
| `tests/ut/quantization/test_w8a8.py:52-54` | the tree's own UT asks only for `get_pertensor_param(torch.bfloat16)` (`input_scale` bf16, `input_offset` int8); no fp16 case exists, and every tutorial/e2e w8a8 line is bf16/auto — an fp16 w8a8 checkpoint is outside what the release exercises |
| `csrc/` has no `AddRmsNormQuant`; `vllm_ascend/envs.py` has no fused-norm/RMSNorm knob; `is_310p()` is the only branch in that path | it is a **CANN built-in**, and there is no switch to turn the fusion off: the dtype (or the image) has to change. `_cann_ops_custom`, `ASCEND_CUSTOM_OPP_PATH`, `SOC_VERSION` and `binary_info_config.json` are the same wrong drawer as for the EZ1001 lane |

## Ranked causes, one check each

1. **No fp16 kernel for that key in this image's CANN** (leading; explains every
   observation, including why the *same* image serves other w8a8 models).
   Cheapest check is the serve itself: add `--dtype bfloat16`. No-vLLM probe in
   the same container, once per dtype:

   ```bash
   python3 -c "import torch, torch_npu
   h = 5120; f = lambda: torch.randn(8, h, dtype=torch.float16, device='npu')
   x, r, w = f(), f(), torch.ones(h, dtype=torch.float16, device='npu')
   s, o = torch.ones(h, dtype=torch.float16, device='npu'), torch.zeros(h, dtype=torch.float16, device='npu')
   print(torch_npu.npu_add_rms_norm_quant(x, r, w, s, o, epsilon=1e-6)[0].dtype)"
   ```

   Repeat with `torch.bfloat16` everywhere. Same `Cannot find bin` on fp16 and
   clean on bf16 ⇒ confirmed, and the answer is "serve this checkpoint as bf16",
   not a patch.
2. **The image's CANN kernel package does not cover the card at all.** Only if
   *both* dtypes fail: switch to
   `../soc-platform-paths/kernel-package-vs-card.md` (standalone probe with a
   built-in unrelated op, `npu-smi info`, `driver/version.info`, image tag vs
   SoC) — no serve flag helps there.
3. **Checkpoint-side mismatch** (cheap, one grep): `quant_model_description.json`
   in the model dir naming `w8a8_dynamic` would mean the static fused path should
   not have been taken at all; an older `msmodelslim` conversion is what produces
   an `input_offset`-bearing static w8a8. Ask for the file's `w8a8*` token, not a
   whole dump.

## Ablations that do not answer this

- `--enforce-eager`: the failing call is ordinary eager `torch_npu` code inside the
  layer forward. Eager removes torch.compile and the ACL graph, **not** the fused
  op, so "still fails with `--enforce-eager`" rules nothing out.
- `ASCEND_LAUNCH_BLOCKING=1`: the error already names the op synchronously and no
  kernel was launched; it adds nothing here (contrast a device fault, where it is
  the attribution step).
- `--max-model-len`, TP size, `chunked_prefill`, capture sizes and every `HCCL_*`
  variable: unrelated — the op fails on dtypes, not on shapes or the fabric.

## What this snapshot cannot settle

- No `cann/ops-*` tree is exported, so *which* dtype keys each SoC's CANN ships
  cannot be read locally: "no fp16 binary exists for this key" is inferred from
  the paste, and the A2-vs-A3 inventory of the image's CANN is unverified.
- `npu_add_rms_norm_quant` / `AddRmsNormQuant` is absent from the exported
  `Ascend/pytorch` and `Ascend/op-plugin` trees at that ref, so the op's own dtype
  list and the key's slot order are unverified — do not derive slot-by-slot claims;
  the count of `float16` entries and the model dtype are enough to localise it.
- #2387's resolution is not in the snapshot. State the bf16 line as the
  one-command experiment that settles it.

## Deliver

Verdict first (a kernel-binary gap for the *dtype key* in the image's CANN —
nothing was launched; not the card, not HCCL, not a vllm-ascend config bug), then
the `--dtype bfloat16` re-serve plus the two-dtype probe, then the ladder and the
"what to send back": the model's `config.json` `torch_dtype`,
`quant_model_description.json`, all four rank logs around the failure, `npu-smi
info` and the image tag.
