# Ascend 950 (A5): an MX-scale device-op refusal, and which die a build covers

Scope: two A5-only report shapes that look alike from the title:

| Paste | What decides it |
|---|---|
| an **A5 device op refusing an MX dtype** — `npu_transpose_quant_batchmatmul` (or a fused A5 attention/MoE op) rejecting `x1_scale` / `x2_scale` as `DT_FLOAT8_E8M0` / `DT_FLOAT4_E2M1FN` with a **one-entry** supported list | the installed kernel's dtype table × the die, **not** the caller's Python — rule 0 of `../custom-ops-runtime-faults/quant-scale-dtype-gate.md`, then steps 2-3 here |
| "does 950PR work / was validation done on 950DT or 950PR" | the build stamp cannot answer it (step 1); the reply must say so and name the two checks that can |

Pins below: vllm-ascend **v0.23.0rc1** (`f4a08bddd0cc`), vllm v0.23.0
(`0fc695fc`). The A5 branch and the doc set move between rcs — re-grep the
`path:line` before quoting.

## Step 1 — one `AscendDeviceType.A5`, so the tree cannot tell the dies apart

| Fact | Anchor (v0.23.0rc1) |
|---|---|
| the enum has a single A5 member; there is no 950PR/950DT axis anywhere in code | `vllm_ascend/utils.py:824-828` |
| the live map sends `soc_version == 260` to A5 and nothing else to 950 | `utils.py:845-862` (`:857-858`) |
| the **build stamp** collapses them too: any `soc_version` containing `ascend950` is mapped to A5 before the concrete table | `setup.py:181-187` |
| the A5 recipe's default is the **DT** die: `ARG SOC_VERSION="ascend950dt_9582"` | `Dockerfile.a5:52` |
| the install doc only says "a value starting with `ascend950`" | `docs/source/installation.md:238` |
| `950PR` / `950DT` appear **nowhere** under `docs/` at this pin (`rg -c` empty) | — |
| the model row that usually matters is `A2/A3`, not A5 | `docs/source/user_guide/support_matrix/supported_models.md:20` |

So `SOC_VERSION=ascend950pr_9579` produces the same `AscendDeviceType.A5` stamp
as the DT default, and **`_build_info.__device_type__ == 'A5'` is not evidence
about the die**. State the two lines separately: "A5 is claimed and your build is
A5" and "no doc, tutorial, test or support-matrix row at this pin enumerates the
PR die" — an enumeration gap, never "950PR is unsupported".

## Step 2 — where the E8M0 paste comes from, and why no flag moves it

At this pin the DeepSeek-V4 o_proj is an **A5-only MX path chosen by device type
alone**, testable in one read:

| Fact | Anchor |
|---|---|
| the branch is `if get_ascend_device_type() in {AscendDeviceType.A5}:`, first in the chain, with the comment "A5 (Ascend950) uses an FP8-quantized o_proj path (dynamic MX quant + quantized batch matmul)" | `vllm_ascend/attention/dsa_v1.py:1560-1563` |
| it dynamic-quantises, then launches with `group_sizes=(0, 0, 32)`, `x1_scale=…view(torch.float8_e8m0fnu)`, `x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu)` | `dsa_v1.py:1564-1577` (`:1566`, `:1572-1573`) |
| the same call exists a second time on the DSA-CP lane — a one-site patch buys a second crash | `attention/context_parallel/dsa_cp.py:1231-1241` (`:1234`, `:1240`) |
| the MX availability shims test **symbols only** (`hasattr(torch_npu, …)`) and are called from the quant methods / `quant_parser`, never from this path | `device/mxfp_compat.py:21-40` (`_get_missing_symbols` `:22`), callers `quantization/methods/w8a8_mxfp8.py:55,202`, `quantization/quant_parser.py:63-65` |
| nothing covers the op | `rg -n 'transpose_quant' tests/` empty |

Consequence for the answer: a checkpoint whose `quantization_config` is
`{"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [128,128]}` passes every
`hasattr` gate and still dies here, so "symbol present" is not "kernel
supports it". `--dtype`, `--data-parallel-size`, `--tensor-parallel-size`,
`--enable-expert-parallel`, `--enforce-eager` and the API server are all outside
this line — say that once instead of ranking them.

**The phase tells you the gate, not the mode.** On the profiling run the branch is
entered only under `oproj_tp_enable()`; otherwise the code fills the output with
zeros and never launches (`dsa_v1.py:1678-1686`). So "engine starts, **first
request** dies at `dsa_v1.py:1566`" is the expected shape for a launch with no
`finegrained_tp_config.oproj_tensor_parallel_size`, and with OTP > 0 the same
failure would surface at start inside `profile_run` instead. A report whose
startup passed is not evidence that the op is fine.

**A capability gate is not a drop-in fix.** The remaining branches of the same
chain — `elif oproj_tp_enable():` (`:1580`, all-to-all then
`npu_transpose_batchmatmul(..., scale=None)`), `elif olora_tp_enable():` (`:1647`,
plain `self.wo_a(...)`) and `else:` (`:1650-1662`, the same unquantized
transpose-matmul) — all consume an **unquantized** `wo_a`. Falling through from an
MX checkpoint therefore needs a dequant path; present it as a source change with
that prerequisite, never as a flag, and accept it only on one real completion
against a bf16 reference.

## Step 3 — the two checks that separate die from CANN build

1. **Build from the tree with the node's own SoC** —
   `docker build -f Dockerfile.a5 --build-arg SOC_VERSION=ascend950pr_9579 .`, then
   the reporter's standalone repro inside it. This removes a hand-installed
   toolkit from the comparison.
2. **The identical repro on the sibling die with the identical CANN package** —
   the one variable that separates "the PR die's kernel package has no MX
   variant" from "this CANN build has none". If no DT card exists, ask for
   `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` (both
   `version=` and `innerversion=`) from the image and from the host, plus the
   plog second under `ASCEND_LAUNCH_BLOCKING=1`.

An fp32 rewrite of the scale is an **experiment, not a repair**: the printed list
`[DT_FLOAT,]` means the kernel has the non-MX scale dtype, and nothing in the
tree or the message says it applies MX block-32 grouping to an fp32 scale — an
E8M0→fp32 cast launches something whose numerics are unverified.

## Recorded run: #13904 (v0.23.0rc1, DeepSeek-V4-Flash w8a8 MXFP8, 4× Ascend950PR)

The report already carried the minimal `torch_npu` repro and the serve shape
(`Model architecture detection: PASS`, `Checkpoint loading: PASS (46/46 shards)`,
KV init and API up, `GET /v1/models` 200, `POST /v1/chat/completions` 500), and
asked four maintainer questions. The measured answer is the pair of verdicts
above: A5 is claimed / the PR die is enumerated nowhere, and the A5 o_proj branch
is ungated with no fallback, so the failure is real and no launch flag routes
around it. The standalone repro is the whole budget — do not walk
`Ascend/op-plugin`, `Ascend/pytorch` or `cann/ops-*`: the export holds 103
op-plugin files with no `opapi/` directory, so `StructKernelNpuOpApi.cpp` and the
aclnn kernel's dtype table are a stated gap.
