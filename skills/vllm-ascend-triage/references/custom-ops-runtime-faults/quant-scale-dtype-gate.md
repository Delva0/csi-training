# Quant/grouped-matmul dtype gate: `161002`, `aclnnQuantMatmulWeightNz` / `aclnnGroupedMatmulWeightNz` / `aclnnGroupedMatmulV5`

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

A second signature is the same gate on an **MX operand**, and it is the one case where
the caller is right (rule 0 below):

```
RuntimeError: npu_transpose_quant_batchmatmul:
  ../third_party/op-plugin/op_plugin/ops/opapi/StructKernelNpuOpApi.cpp:4005
NPU function error: device error type 0, error code is 161002
AclNN_Parameter_Error(EZ1001): Tensor x1Scale not implemented for DT_FLOAT8_E8M0,
  should be in dtype support list [DT_FLOAT,].
```

## Rule 0 — the printed support list's length decides which side of the gate you are on

The same `161002` / `EZ1001` wrapper covers both, and the *number of entries* in
`should be in dtype support list [...]` separates them before any source read:

| Paste | Owner |
|---|---|
| a multi-entry list whose only missing member is the model dtype — `Tensor scale not implemented for DT_FLOAT16, should be in dtype support list [DT_UINT64, DT_BFLOAT16, DT_INT64, DT_FLOAT]` | the Python that built the scale: rules 1-5 below, the caller is wrong |
| a **one-entry** list with an MX operand — `Tensor x1Scale not implemented for DT_FLOAT8_E8M0, should be in dtype support list [DT_FLOAT,]` on `npu_transpose_quant_batchmatmul` (`x1_scale=` / `x2_scale=` arriving as `….view(torch.float8_e8m0fnu)`), or a `DT_FLOAT4_E2M1FN` weight on an A5 fused op | the installed kernel × the running SoC die: this build carries the non-MX scale variant only, while the caller is doing exactly what an MX checkpoint requires |

For the second row, **do not run rules 1-3**: they would "repair" a correct MX
launch. An E8M0 block-32 scale cast to fp32 makes the call leave this gate, but
nothing in the tree or the message says the non-MX path applies block-32
grouping to a float scale, so the cast is an experiment whose acceptance is one
real completion against a bf16 reference — never the recommendation. The
`hasattr`-only probes in `vllm_ascend/device/mxfp_compat.py` (`:21-40`, callers in
`quantization/methods/w8a8_mxfp8.py:55,202` and `quantization/quant_parser.py:63-65`)
all pass in that case, so "the symbol exists" is not "the kernel has it".
Recorded #13904 (v0.23.0rc1, DeepSeek-V4-Flash w8a8 MXFP8, 4× Ascend950PR): the
A5 o_proj branch (`attention/dsa_v1.py:1560-1577`, twin at
`attention/context_parallel/dsa_cp.py:1231-1241`) is selected by device type
alone with no capability probe and no fallback, and no launch flag reaches it —
the die-vs-CANN-build ladder is
`../soc-platform-paths/a5-mxfp-refusal-and-die-coverage.md`.

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
The pair is old, not a v0.17.0rc1 regression: at v0.8.5rc1 (`ec27af34`) the same
declaration/launch split sits at `vllm_ascend/quantization/w8a8_dynamic.py:367`
(creates `weight_scale_fp32`) vs `:352` (passes `layer.weight_scale`), so a report at
an older ref is the same gate and the audit script finds it there too.

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
— its `START HERE` block lists, in order, the `_fp32` companions that are created
and never read (with the launch line that still passes the non-fp32 twin), every
quant-method launch site whose scale is not float32, and every site that takes the
op's output dtype from a scale tensor; the full inventory of scale arguments and
the companion list follow. Three reading rules for its output:

- **It resolves one alias hop**, which is where the MoE twin hides: the launch is
  `fused_experts(w2_scale=w2_scale)` at `:251`, and `w2_scale` is bound at `:249`
  (`[layer.w2_weight_scale]`, or `layer.w2_weight_scale_list` on the EPLB branch).
  Without that hop the sweep reports only the local variable and misses the defect
  — grepping `weight_scale` by hand is what turns `:249` up.
- **Read it whole.** It is ~114 lines at v0.17.0rc1 and the inventory plus
  companion list are at the end, so `| head -60` keeps the candidate block and
  drops the sweep (a recorded run truncated it and then re-grepped by hand for the
  exact lines the tail already held).
- `CREATED BUT NEVER READ` is the v0.17.0rc1 linear shape and is a **same-file**
  fact; `no _fp32 companion in this file` is not yet a defect. Three benign shapes
  sit in the same list: the static method's `deq_scale` is allocated `float32` for
  bf16 and **`int64`** for fp16 (`w8a8_static.py:67-69`), so it is never a scale-dtype
  candidate; `device_op.py` takes the scale from its caller; and `w8a8_pdmix.py:84`
  delegates the launch to the dynamic method it wraps. Read the allocation and the
  delegate before citing a site.

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

## Rule 5 — read which operand the gate names: a rejected **`y`** is an activation-dtype fault, not a scale one

The other `161002` on a grouped matmul rejects the **output**. Recorded #2807
(v0.10.1rc1 official image, Qwen3-30B-A3B-W8A8, `--quantization ascend`,
`-tp 4 --enable-expert-parallel`, aclgraph default):

```
RuntimeError: npu_grouped_matmul … NPU function error: call aclnnGroupedMatmulV5 failed, error code is 161002
EZ1001: [PID: 7114] … Tensor … dtype is DT_BFLOAT16 but found DT_INT32.
GMM:y dtype does not match with required dtype[DT_BFLOAT16]
```

`y` is required to be the *activation's* dtype here, and the code that asks for
`int32` is found in one grep at the launch, not in the scale:

| Fact | Anchor (v0.10.1rc1 `7e16b4a7cdb1`) |
|---|---|
| gmm1 asks for the quantized-output dtype unconditionally for W8A8 | `vllm_ascend/ops/common_fused_moe.py:99-107`, `output_dtype=torch.int32 if use_int8_w8a8 else None` (`:106`) |
| `x` is whatever `moe_comm_method.permute(...)` returned | `:95-97`, called with `use_int8_w8a8 or use_int4_w4a8` as `apply_a8_quantization` |
| that flag decides the activation dtype only in **MC2**: `quant_mode = 2 if apply_a8_quantization else 0` | `vllm_ascend/distributed/moe_comm_method.py:398` |
| `AllGatherCommImpl.permute` takes `apply_a8_quantization` (`:166`) and **never reads it**; it hardcodes `quant_mode=-1` (`:195`) and returns `None` as the dynamic scale (`:202`) | same file, class at `:101` |
| a single-node serve lands there regardless of the flag: `if self.moe_config.ep_size < 16: moe_comm_method_name = "allgathercommimpl"` | `ops/common_fused_moe.py:487-490` |
| the quant method picks its branch **once, in `__init__`, from config alone**: `use_aclgraph = level == PIECEWISE and not enforce_eager and not torchair_graph_config.enabled` | `vllm_ascend/quantization/w8a8_dynamic.py:291-294` |
| the graph body is `unified_fused_experts` (the `ops/common_fused_moe.fused_experts` alias) at `:390`; the eager twin is `unified_fused_experts_eager` at `:418` → `ops/fused_moe.py:55` → `w8a8_dynamic.py:113 apply_mlp` | same file, imports `:30-32`, branch `:389-390` / `:418` |
| `--enforce-eager` sets `NO_COMPILATION` before that constructor reads the level (log: `Compilation disabled, using eager mode by default`) | `vllm_ascend/platform.py:145-147`; the whole flag→level table is `../performance-throughput/graph-mode-from-the-serve-line.md` |
| the only W8A8 e2e case at this ref runs the **eager** path: TP2 + `enforce_eager=True` (the ACL-graph cases are bf16) | `tests/e2e/multicard/test_qwen3_moe.py:59-71` vs `:90+` |

So the contract has two owners and the gate compares them: the launch's `output_dtype`
and whatever actually quantized the activations. Which branch produced the call is in
the paste itself, one frame below the top: `apply` → `unified_fused_experts` is the
graph body, `apply` → `unified_fused_experts_eager` is eager — so a paste whose deepest
`vllm_ascend` frames read `quantization/w8a8_dynamic.py` (`…:390 return
unified_fused_experts(`) → `ops/common_fused_moe.py:99 npu_grouped_matmul` has already
answered "does rung 1 apply?", while a frame in the eager body means the rung is spent
and the open question is `quant_mode` below. Because `use_aclgraph` is fixed in
`__init__` from the config, `--enforce-eager` switches the path for the profiling dummy
forward and for every request alike: it is a path switch, not a phase-conditional
ablation (`SKILL.md` rule 15's test), and no other flag reaches this line.

On the graph path the eager MoE is the working control
(`quantization/w8a8_dynamic.py:113 apply_mlp`, which quantizes per token first and takes
`_output_dtype` from `w2_scale`, `:157`) — which is exactly why `--enforce-eager` (and,
per the validated test, TP2) is rung 1, before any patch. Label the rung
**source-inferred**: the branch switch is verified at the pin, but nobody has reported
that rerun back, so the answer says "this changes the code path the launch takes", not
"this is the confirmed fix", and asks for the new traceback if it does not.

**The one thing not knowable locally: `quant_mode` semantics.** `-1` versus `2` is a
torch_npu/CANN attribute meaning, and no export carries it — in this case view
`rg -ln 'moe_init_routing' Ascend/` is empty and the only `quant_mode` hit in
`Ascend/` is an unrelated HIVM fixpipe attr (`HIVM_FixpipePreQuantModeEnum`). Do not
spend turns deciding whether
`-1` means "dynamic quant" or "input already quantized" (#2807's first run spent its
entire output budget there and shipped no answer; its second run answered from this page
in ~12 turns with rung 1 in the first paragraph): state the two readings, note the
local contrast above, and give the reporter the one-line check that ends it —
`print(permuted_hidden_states.dtype)` at the failing launch, or
`python -c 'import torch_npu; help(torch_npu.npu_moe_init_routing_v2)'` on their image.
Everything else in the ranked answer (dtype mismatch at `:106` vs the permute, the
`ep_size < 16` selection, the eager control, the CI coverage gap) is already verified.
The same `ep_size < 16` selector is also what a *performance* report citing this model
pair has to be priced from (EP off puts every MoE layer on that all-gather path):
`../performance-throughput/moe-comm-path-when-ep-off.md`.

## Do not chase the neighbours in the same paste

`Node[TransposeKvCacheByBlock]: failed to initialize … input2.blockIDs. Tensor scale
not implemented for DT_FLOAT16` appearing beside the quant-matmul error is CANN
re-printing the aborted stream's gate: that node has no scale operand. Treat it as
context noise; only a block that survives the repair is a second fault (then ask for
the full plog block with `ASCEND_LAUNCH_BLOCKING=1`).

## Scope and recheck

Rule 0's MX member is source-verified at vllm-ascend v0.23.0rc1
(`f4a08bddd0cc`) with vllm v0.23.0 (`0fc695fc`), where the aclnn kernel's own dtype
table is not in the export (`Ascend/op-plugin` carries 103 files and no `opapi/`
directory, and `cann/ops-transformer` is unavailable) — so the die-versus-CANN
question is answered on hardware, not locally.
Source-verified at vllm-ascend v0.17.0rc1 `e20f0b1a0d2f` (vllm v0.17.0
`b31e9326a7d9`); the MoE second occurrence is reporter-confirmed on #7443, not
executed here (no NPU in the case view). Recheck condition: re-run the audit script
or `rg -n 'weight_scale_fp32|w2_weight_scale' vllm_ascend/quantization/methods/w8a8_dynamic.py`
on the reporter's own tree — if `apply()` now reads a `_fp32`/`_fp32_list` tensor and
the MoE output dtype is no longer taken from the scale, the repair has landed and
this page describes an older ref. Rule 5's anchors are a different, earlier ref
(v0.10.1rc1 `7e16b4a7cdb1`): there `ops/common_fused_moe.py` is the only MoE path and
`quantization/methods/` does not exist yet, so re-locate `output_dtype=` and the
`ep_size < 16` line in whatever layout the reported tree has, and re-grep
`apply_a8_quantization` — if the AllGather `permute` reads the flag, rule 5's mechanism
is repaired. Rule 5's anchors (the `use_aclgraph` gate, the `--enforce-eager` →
`NO_COMPILATION` line and the eager `apply_mlp` twin) were re-verified there by a later
#2807 session, which shipped the rung in its first paragraph after four verification
calls — for that case this page plus one locate call is the investigation, so
re-deriving them is the detour.
