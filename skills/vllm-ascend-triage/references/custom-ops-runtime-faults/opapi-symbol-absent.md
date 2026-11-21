# `aclnn<Op> … not in libopapi.so`: the op_api symbol search, not the CANN version

Verified at vllm-ascend v0.26.0rc1 (`f2f74a16c3c5`) and vllm v0.26.0
(`568afb3a1380`). Recorded for #16379: `DeepSeek-V4-Flash-0731-w8a8` on one
Atlas 800 A3 node, CANN 9.1.0 (`innerversion V100R001C11SPC001B243`), the pinned
doc's single-node "A3 series with DSpark" recipe (`--data-parallel-size 4
--tensor-parallel-size 4 --enable-expert-parallel`). The reporter had already
located the mHC `hc_pre` path and a vendor table that lists `aclnnHcPre` for A5
only.

## The signature

```
File ".../vllm_ascend/models/deepseek_v4.py", line 974, in hc_pre
  y = torch.ops._C_ascend.npu_hc_pre_v2(
File ".../torch/_ops.py", line 1209, in __call__
  return self._op(*args, **kwargs)
RuntimeError: aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so, or libopapi.sonot found.
```

Lane B, first forward of **memory profiling** (`determine_available_memory →
profile_run → _dummy_run → model forward → DecoderLayer.forward`), every worker
in the same second. Nothing launched: no device dump, no plog line, no tiling
text, no `EZ`/`ER` code. Do not route it to the collective, tiling or
`ESC`-style lanes on the strength of the word `aclnn`.

## The text names one library while three are searched

- `csrc/aclnn_torch_adapter/op_api_common.h:687-700` — `EXEC_NPU_CMD` dlsyms
  `<aclnn_api>` and `<aclnn_api>GetWorkspaceSize` and, on failure, prints
  `#aclnn_api " or " #aclnn_api "GetWorkspaceSize not in " GetOpApiLibName()`.
  `GetOpApiLibName()` is the constant `"libopapi.so"` (`:259`), so the message is
  a **fixed string**: it names the toolkit library even when the miss is
  somewhere else.
- `GetOpApiFuncAddr` (`:267-317`) searches in this order: each
  `$ASCEND_CUSTOM_OPP_PATH` vendor directory's `op_api/lib/libcust_opapi.so`
  (`:267-290`), then the vendors listed after `load_priority=` in
  `$ASCEND_OPP_PATH/vendors/config.ini` (`:214-257`), and only then the toolkit's
  `libopapi.so` (`:308-317`). "Not in libopapi.so" therefore means *not found in
  any of them*, and the first two roots are vllm-ascend's **own** custom-op
  package (`vllm_ascend/_cann_ops_custom/vendors/custom_transformer`,
  `vllm_ascend/utils.py:74-75`).

That is why "the CANN build is missing the op for this SoC" is the *last*
candidate, not the first: the op is normally supplied by the vendored package,
not by CANN.

## The pinned tree builds this op for A3 — check before granting the vendor table

- `vllm_ascend/models/deepseek_v4.py:973-975` calls
  `torch.ops._C_ascend.npu_hc_pre_v2` unconditionally (no SoC branch, no Python
  fallback), twice per layer (`:993`, `:999`), so MTP-only and DSpark-only
  configs both pass through it.
- `csrc/torch_binding.cpp:1410` → `EXEC_NPU_CMD(aclnnHcPre, …)`.
- `csrc/moe/hc_pre/op_host/hc_pre_def.cpp:65-73` declares `ascend910b`,
  `ascend910_93` and `ascend950` (the 950 arm with arch35 tiling,
  `hc_pre_tiling_arch35.h`); `csrc/build_aclnn.sh:173` lists `hc_pre` in the
  **ascend910_93** `CUSTOM_OPS_ARRAY` (`SOC_ARG="ascend910_93"` at `:193`).
- A vendor table that lists `aclnnHcPre` for A5 only is about **CANN's own copy**
  of the op — the fallback root of the search — not about vllm-ascend's. Say this
  explicitly instead of adopting the table as the verdict.
- Nothing in `vllm_ascend/envs.py` / `ascend_config.py` selects the composite
  path: `npu_hc_pre` (non-v2, `run_hc_pre_composite` =
  `aclnnHcPreInvRms` + `aclnnHcPreSinkhorn`) is reached only from
  `tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py:41`.
  There is no knob the reporter can flip; the fix is the package or the image.

`check_hc_pre_shape_and_dtype` runs *before* the macro (`torch_binding.cpp:1420`,
`:1428`, definition `:1336`), so the paste also proves the input shapes, dtypes
and the checkpoint's `hc_*` weights were accepted — do not spend turns on
weights, quant format or serving flags.

## Two error texts separate "no extension" from "extension without the vendor package"

| Text | Meaning | Lever |
|---|---|---|
| `AttributeError: '_OpNamespace' object has no attribute 'npu_hc_pre_v2'`, or `Failed to register custom ops, all custom ops will be disabled` in the log | `vllm_ascend.vllm_ascend_C` was not built/imported (`setup.py:369,440,463` build only under `envs.COMPILE_CUSTOM_KERNELS`; `enable_custom_op()` `vllm_ascend/utils.py:401-455` turns **all** custom ops off on `ImportError`, and on A5 or `VLLM_BATCH_INVARIANT` by design at `:418-420`) | the install/wheel |
| **this page's** `… not in libopapi.so` at `torch.ops._C_ascend.<op>` | the extension loaded and dispatched, but no searched `op_api` library exports the aclnn symbol for the running SoC | the container's custom-op package / `ASCEND_CUSTOM_OPP_PATH` / image tag |

`bootstrap_custom_op_env()` (`vllm_ascend/utils.py:311-317`) prepends
`ASCEND_CUSTOM_OPP_PATH` **only when the vendor directory exists**, so an
environment that never carries `ASCEND_CUSTOM_OPP_PATH` in `collect_env.py` is
consistent with the package being absent — a hint, not proof (the dump is a
separate process and the env var is set inside the engine).

## Checks, cheapest first

```bash
S=$(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))")
ls -d "$S"/_cann_ops_custom/vendors/*/op_api/lib/ ; echo "opp=$ASCEND_CUSTOM_OPP_PATH"
nm -D "$S"/_cann_ops_custom/vendors/*/op_api/lib/libcust_opapi.so | grep -i 'hcpre'
cat "$S"/_cann_ops_custom/vendors/*/config.ini            # SoC / op list actually installed
nm -D /usr/local/Ascend/<cann>/lib64/libopapi.so | grep -i hcpre   # fallback root, second question
grep -n "Failed to register custom ops" <server log>
```

Then the artefact comparison that settles image-vs-host: the same two `nm`
commands inside the image the model's pinned tutorial names for this SoC
(`docs/source/tutorials/models/DeepSeek-V4-Flash.md:40` — `…:DeepSeekV4-flash-0731-a3`
for A3, `…:DeepSeekV4-flash-0731` for A2; the A3 docker block at `:42-48`, the
"A3 series with DSpark" launcher at `:267`).

## Answer shape

Verdict first: not `--speculative-config`, not DSpark, not `enable_dsa_cp`, not
DP4/TP4 — the fused op is resolved at run time from the custom-op package, and
that package has no `HcPre` export for this SoC in that container. Then, in this
order: (1) run the tutorial's A3 image, or rebuild with
`COMPILE_CUSTOM_KERNELS=1 SOC_VERSION=ascend910_9391`; (2) make
`ASCEND_CUSTOM_OPP_PATH` point at an existing vendor directory; (3) only if the
official image shows the same symbol list, it is a support gap — escalate with
the model's own coverage lines (`docs/source/user_guide/support_matrix/supported_models.md`
A2/A3 rows, `tests/e2e/nightly/single_node/models/configs/DeepSeek-V4-Flash-W8A8-A3.yaml`)
rather than with the CANN `version.cfg` diff.

Do **not** rank "your CANN 9.1.0 differs from the CI CANN" or "install
`Ascend-cann-910b-ops`" above the custom-op package: `docs/source/installation.md`
already pins CANN 9.1.0 for v0.26.0rc1, and the missing symbol is a vllm-ascend
custom op in the first two search roots.

## Open in the recorded case

- Whether the reporter's container had the vendor package at all: unverified.
  The recorded answer ranked the CANN build first and the A3 support gap third
  and never listed `_cann_ops_custom` or `libcust_opapi.so`; that ranking is the
  reason this page exists.
- Which A3 sub-SoC the wheel was built for (`ascend910_9391` vs a bare
  `ascend910_93`) is not visible from the paste; `SOC_VERSION` is recorded by
  `setup.py` at build time, so ask for the build log or read `config.ini`.
