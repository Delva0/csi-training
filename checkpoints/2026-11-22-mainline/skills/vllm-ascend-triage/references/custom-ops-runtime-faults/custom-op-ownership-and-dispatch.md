# "算子侧反馈非cann算子" — whose op is it, and which framework path dispatched it

Verified at two pins of the same op: vllm-ascend `v0.13.0rc3` (`bbe9bc0d90f3`) + vllm
`v0.13.0` (`72506c9834`), where the sampling mask is plain PyTorch and the name exists
nowhere, and vllm-ascend `v0.18.0` (`e18643f8a4d5`) + vllm `v0.18.0`
(`bcf2be961200`), where it is a `csrc/` AscendC op named `ApplyTopKTopPCustom`.
Recorded for #8730: vllm-ascend `0.14.0.RC1`, "强化学习场景用户设置
top 后报错，调用了 `ApplyTopPTopCustom` 算子，算子侧反馈非cann算子，需要框架侧定位为什么
会下发这个". The reported release sits **between** the two pins, which is exactly why the
older pin could only date the report and the newer one names the op (rule 3).

## Scope — the question this answers

An operator owner returns a case with a bare op name and no framework traceback:
*this is not a CANN operator, why is the framework dispatching it?* Or the reporter
pastes an op name that greps to nothing in the pinned tree. The deliverable is two
things and no diagnosis of a device fault: which framework call site dispatches the
op, whether "not in the CANN op list" is a defect or the design, and — once the name is
identified — which of the gates in rule 4 decides whether it runs at all.

Distinguish from the neighbours by the evidence, not the title: an actual
`AclNN_Parameter_Error(EZ1001) … Get regInfo failed … does not support opType [...]`
is the build/install lane (`op-not-registered-for-soc.md`); an `AclNN_Runtime_Error`,
dump or tiling text is the kernel lane (`index.md`); a tiling refusal returned as
"调度问题" is `tiling-attribute-rejection.md`.

## Rule 1 — a `…Custom` op that CANN does not list is usually vllm-ascend's own

vllm-ascend builds its AscendC kernels from `csrc/` and installs them as a vendor
package, so they never appear in CANN's `ops-*` inventory:

- At v0.18.0 the sampling mask is one of them: `csrc/apply_top_k_top_p_custom/` holds
  the kernel, `csrc/build_aclnn.sh:28` (A2) / `:59` (A3 array) build
  `apply_top_k_top_p_custom`, and `csrc/torch_binding.cpp:876-877` exposes it as
  `npu_apply_top_k_top_p` inside `TORCH_LIBRARY_EXPAND(_C_ascend, ops)` (`:702`) — i.e.
  `torch.ops._C_ascend.npu_apply_top_k_top_p`, registered by the compiled extension.
- At v0.13.0rc3 the arrays are `csrc/build_aclnn.sh:27-56` (A2 `ascend910b`) /
  `:102-134` (A3 `ascend910_93`) / `:152-170` (A5 `ascend910_95`) and carry
  `grouped_matmul_swiglu_quant_clamp`, `sparse_flash_attention`,
  `moe_init_routing_custom`, `moe_gating_top_k`, `add_rms_norm_bias`,
  `quant_lightning_indexer`, … and no sampling op; v0.18.0 has no A5 branch at all.
  The naming convention is lower_snake in the build list and CamelCase at the
  aclnn/opType surface (`MoeInitRoutingCustom`, `AddRmsNormBias`,
  `SparseFlashAttentionCustom`, `ApplyTopKTopPCustom`).
- At v0.13.0rc3 `:201` runs `bash build.sh --pkg --ops=… --soc=…` and `:203-219` install it into
  `vllm_ascend/_cann_ops_custom` (the git tree carries only `.gitkeep` there — the
  binaries exist only when that step ran, i.e. under `COMPILE_CUSTOM_KERNELS`), and
  `platform.py:393-399` prepends the installed path to `ASCEND_CUSTOM_OPP_PATH`.
- Therefore "非cann算子" is the **expected** answer for a framework-owned op. Say so
  in the first sentence, and spend the rest of the answer on the dispatch site.

## Rule 2 — the sampling top-k/top-p surface is the dispatcher for a "设置 top" report

Three call sites, each with its own trigger. Name the one that matches the reporter's
flags instead of listing all three as candidates (v0.18.0 lines, v0.13.0rc3 in
parentheses):

| Call site | Trigger |
|---|---|
| `vllm_ascend/sample/sampler.py:92,106` — `AscendTopKTopPSampler` binds its own `apply_top_k_top_p` and calls it (`:68,78` at v0.13.0rc3); implementations `:121-155` (pytorch) and `:157-165` (ascendc), selector `:167-171` | the main sampler path; any request with `top_p` set (or `top_k` alone), subject to rule 4 |
| `vllm_ascend/sample/rejection_sampler.py:21,79` (`apply_sampling_constraints` → `apply_top_k_top_p`), installed into upstream by `vllm_ascend/patch/worker/patch_rejection_sampler.py` (index comment `patch/__init__.py:222` at v0.13.0rc3, `:333` at v0.18.0) | speculative decoding / MTP / Eagle rejection sampling — the usual RL rollout shape, and the only path where the mask is applied to draft-verified tokens |
| `vllm_ascend/worker/v2/sample/sampler.py` (imports **upstream's** `apply_top_k_top_p`) | Model Runner V2 — never reaches the AscendC op |

So "只在我设置 top 时报错、greedy 时不报" is explained by construction: greedy takes
`all_greedy` → argmax and never reaches the mask. The single question that picks the
file is *MTP/spec-decode on or off*. `top_k` is not the discriminator — both k-only
and k+p enter the same function; the p-only case is the one the release notes call
out ("Added support for advanced apply_top_k_top_p without top_k constraint",
`docs/source/user_guide/release_notes.md:65` at v0.13.0rc3, #6098).

The surface has changed hands twice, so grade a name against the release, not the
family: v0.11-era `torch_npu.npu_top_k_top_p` (release note "Used fused ops
npu_top_k_top_p in sampler", `release_notes.md:664` at v0.13.0rc3, #1920) → plain
PyTorch for both branches at v0.13.0rc3 (`_apply_top_k_top_p_pytorch`, `:93-133`
there) → the `_C_ascend` custom op at v0.18.0.

## Rule 3 — the name is the framework's own, and one pin spells it: `ApplyTopKTopPCustom`

`ApplyTopPTopCustom` is `ApplyTopKTopPCustom` with K/P transposed — the opType of
`csrc/apply_top_k_top_p_custom/op_host/apply_top_k_top_p_custom_def.cpp:18`
(`class ApplyTopKTopPCustom : public OpDef`) — i.e. the lower_snake
`apply_top_k_top_p_custom` build entry CamelCase'd, exactly per rule 1. So "why does
the framework dispatch it" is `sample/sampler.py:106`, and "not a CANN op" is by
design; do not spend the answer on a misdispatch theory.

Do not carry an older "no ref in this snapshot has this name" verdict forward either:
it was true of the v0.13.0rc3 tree (no `csrc/apply_top_k_top_p_custom/`, and both
branches of the mask are plain PyTorch there) and false at v0.18.0. The two pins date
the report — absent at v0.13.0rc3, present at v0.18.0, so a `0.14.0.RC1` report is in
the gap — and the answer says which half was verified and on which commit.

When the name still greps to nothing at the pins you have, hand over these reads
instead of guessing:

```bash
P=$(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))")
rg -n "ApplyTop|apply_top_k_top_p|top_k_top_p" "$P" | head -30
ls "$P/_cann_ops_custom/vendors" ; echo "$ASCEND_CUSTOM_OPP_PATH"
python -c "import torch_npu;print(torch_npu.__version__, hasattr(torch_npu,'npu_top_k_top_p'))"
```

## Rule 4 — three gates decide whether the op is dispatched at all

1. **SoC.** `sample/sampler.py:167-171` binds `apply_top_k_top_p =
   _apply_top_k_top_p_ascendc if get_ascend_device_type() in [AscendDeviceType.A2,
   AscendDeviceType.A3] else _apply_top_k_top_p_pytorch`, and `build_aclnn.sh` has no
   A5 branch at v0.18.0 — so 310P/A5 take the PyTorch mask and never dispatch the op,
   whatever the flags. Ask for the SoC before grading a "why was it dispatched" claim.
2. **Batch-wide `top_p < 1`.** Upstream decides `p` per *batch*, not per request:
   `vllm/v1/worker/gpu_input_batch.py:359-365` adds a request to `top_p_reqs` only when
   `sampling_params.top_p < 1`, `no_top_p` is `len(top_p_reqs) == 0` (`:1015-1016`), and
   sampling metadata carries `top_p=None if self.no_top_p else self.top_p[:num_reqs]`
   (`:848`; the same trio at vllm v0.13.0 `:363-365,833,969-970`). One concurrent
   request with `top_p < 1` therefore gives the **whole** batch a mask, and `top_p: 1.0`
   really does remove it — which makes it the correct off-switch, but only if no
   neighbour in flight sets it. Run that ablation at concurrency 1 (or with every
   request at `top_p: 1.0`), or it proves nothing.
3. **Batch-invariant mode skips the AscendC path.** `sample/sampler.py:104-105` returns
   `super().forward_native(...)` under `vllm_is_batch_invariant()`, so a report that
   also exports `VLLM_BATCH_INVARIANT=1` is on the upstream mask and the custom op is
   not in the picture (`../accuracy-benchmarks/batch-invariant-coverage.md`).

## Rule 5 — the shape this op fails in when the extension did not load

`AttributeError: '_OpNamespace' '_C_ascend' object has no attribute
'npu_apply_top_k_top_p'` at the first sampled step of a **non-greedy** request (greedy
never reaches the mask) is `SKILL.md` rule 11's wrapper family, not an op defect: the
`_C_ascend` namespace is registered by the compiled `vllm_ascend_C`, so read the
`Failed to import vllm_ascend_C: …` warning above it and its tail — `No module named …`
(no `.so` in the image) vs `undefined symbol: …` (present, refused by a SoC source-list
gate) → `../image-dependency-faults/custom-op-extension-missing-from-image.md`. With
the extension present but the vendor package missing or lacking the running SoC's
entry, the failure is instead `EZ1001 … does not support opType` and the lane is
`op-not-registered-for-soc.md`.

## What to send back

The error line verbatim (opType, socVersion, any `EZ` code), the SoC type (A2/A3 vs
310P/A5 — rule 4), image tag, `collect_env.py`, the `_cann_ops_custom` listing, whether
MTP / Eagle / speculative decoding is enabled (picks the table row), and the
**batch-wide** ablation of rule 4: the same request with `top_p: 1.0` served alone,
then the same request with `temperature: 0` — if the op still dispatches, the trigger is
not the top-p mask and the next read is the rejection-sampler row.

`cann/ops-nn` and `cann/ops-transformer` are `unavailable` in this snapshot's
`SOURCES.json`, so "CANN does not have this op" cannot be established locally — only
the operator side can confirm that from its own CANN version.
