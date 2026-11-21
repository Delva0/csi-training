# "算子侧反馈非cann算子" — whose op is it, and which framework path dispatched it

Verified at vllm-ascend `v0.13.0rc3` (`bbe9bc0d90f3`) + vllm `v0.13.0`
(`72506c9834`). Recorded for #8730: vllm-ascend `0.14.0.RC1`, "强化学习场景用户设置
top 后报错，调用了 `ApplyTopPTopCustom` 算子，算子侧反馈非cann算子，需要框架侧定位为什么
会下发这个". The reported release is **newer than the snapshot**, which is part of the
answer (rule 4).

## Scope — the question this answers

An operator owner returns a case with a bare op name and no framework traceback:
*this is not a CANN operator, why is the framework dispatching it?* Or the reporter
pastes an op name that greps to nothing in the pinned tree. The deliverable is two
things and no diagnosis of a device fault: which framework call site dispatches the
op, and whether "not in the CANN op list" is a defect or the design.

Distinguish from the neighbours by the evidence, not the title: an actual
`AclNN_Parameter_Error(EZ1001) … Get regInfo failed … does not support opType [...]`
is the build/install lane (`op-not-registered-for-soc.md`); an `AclNN_Runtime_Error`,
dump or tiling text is the kernel lane (`index.md`); a tiling refusal returned as
"调度问题" is `tiling-attribute-rejection.md`.

## Rule 1 — a `…Custom` op that CANN does not list is usually vllm-ascend's own

vllm-ascend builds its AscendC kernels from `csrc/` and installs them as a vendor
package, so they never appear in CANN's `ops-*` inventory:

- `csrc/build_aclnn.sh:27-56` (A2 `ascend910b`) / `:102-134` (A3 `ascend910_93`) /
  `:152-170` (A5 `ascend910_95`) hold the `CUSTOM_OPS_ARRAY`
  (`grouped_matmul_swiglu_quant_clamp`, `sparse_flash_attention`,
  `moe_init_routing_custom`, `moe_gating_top_k`, `add_rms_norm_bias`,
  `quant_lightning_indexer`, …); the naming convention is lower_snake in the build
  list and CamelCase at the aclnn/opType surface (`MoeInitRoutingCustom`,
  `AddRmsNormBias`, `SparseFlashAttentionCustom`).
- `:201` runs `bash build.sh --pkg --ops=… --soc=…` and `:203-219` install it into
  `vllm_ascend/_cann_ops_custom` (the git tree carries only `.gitkeep` there — the
  binaries exist only when that step ran, i.e. under `COMPILE_CUSTOM_KERNELS`), and
  `platform.py:393-399` prepends the installed path to `ASCEND_CUSTOM_OPP_PATH`.
- Therefore "非cann算子" is the **expected** answer for a framework-owned op. Say so
  in the first sentence, and spend the rest of the answer on the dispatch site.

## Rule 2 — the sampling top-k/top-p surface is the dispatcher for a "设置 top" report

Three call sites at the pinned ref, each with its own trigger. Name the one that
matches the reporter's flags instead of listing all three as candidates:

| Call site | Trigger |
|---|---|
| `vllm_ascend/sample/sampler.py:64-78` — `AscendTopKTopPSampler` overrides upstream `TopKTopPSampler.apply_top_k_top_p` (`:68`), called at `:78`; the downstream implementation is `:93-133` (k branch `:104`, **p branch `:116+`**) | the main sampler path; any request with `top_p` set (or `top_k` alone) |
| `vllm_ascend/sample/rejection_sampler.py:16,74` (`apply_sampling_constraints` → `apply_top_k_top_p`), installed into upstream by `vllm_ascend/patch/worker/patch_rejection_sampler.py:1-12` | speculative decoding / MTP / Eagle rejection sampling — i.e. the usual RL rollout shape, and the only path where the mask is applied to draft-verified tokens |
| `vllm_ascend/worker/v2/sample/sampler.py:19,48` (imports **upstream's** `apply_top_k_top_p`) | Model Runner V2 |

So "只在我设置 top 时报错、greedy 时不报" is explained by construction: greedy takes
`all_greedy` → argmax and never reaches the mask. The single question that picks the
file is *MTP/spec-decode on or off*. `top_k` is not the discriminator — both k-only
and k+p enter the same function; the p-only case is the one the release notes call
out ("Added support for advanced apply_top_k_top_p without top_k constraint",
`docs/source/user_guide/release_notes.md:65`, #6098).

## Rule 3 — a name-order mismatch is a hypothesis, not a finding

`ApplyTopPTop…` does not read like CANN's `aclnnApplyTopKTopP`. That is tempting and
unverified: at every ref in this snapshot neither `ApplyTop*` nor `TopPTop` occurs in
vllm-ascend or vllm, so there is nothing to compare against `0.14.0.RC1`. Label it a
hypothesis with its one check, and do not build the answer on it.

## Rule 4 — the op name may be newer than the snapshot; say which part is unverifiable

Absence in the pinned tree dates the report against the release and is itself a
finding: state it, then hand the reporter the in-image reads that decide everything
the snapshot cannot:

```bash
P=$(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))")
rg -n "ApplyTop|apply_top_k_top_p|top_k_top_p" "$P/vllm_ascend" | head -30
ls "$P/_cann_ops_custom/vendors" ; echo "$ASCEND_CUSTOM_OPP_PATH"
python -c "import torch_npu;print(torch_npu.__version__, hasattr(torch_npu,'npu_top_k_top_p'))"
```

If the first line shows the op in the tree, the dispatch site is named in one open.
If `_cann_ops_custom/vendors` is empty or the `opType` is missing from the running
SoC's `binary_info_config.json`, the case has moved to `op-not-registered-for-soc.md`.
`cann/ops-nn` and `cann/ops-transformer` are `unavailable` in this snapshot's
`SOURCES.json`, so "CANN does not have this op" cannot be established locally — only
the operator side can confirm that from its own CANN version.

## What to send back

The error line verbatim (opType, socVersion, any `EZ` code), whether MTP / Eagle /
speculative decoding is enabled (picks the table row), image tag, `collect_env.py`,
the `_cann_ops_custom` listing, and whether the same request with `top_p: 1.0` (or
`temperature: 0`) still dispatches the op — the ablation that proves the trigger is
the top-p mask and not "the framework always dispatches it".
