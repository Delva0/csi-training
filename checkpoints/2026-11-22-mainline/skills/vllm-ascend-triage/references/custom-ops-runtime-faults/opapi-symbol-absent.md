# `aclnn<Op> … not in libopapi.so`: the op_api symbol search, not the CANN version

Verified at vllm-ascend v0.26.0rc1 (`f2f74a16c3c5`) and vllm v0.26.0
(`568afb3a1380`), and — for the *dlsym mechanics only* — at vllm-ascend
v0.15.0rc1 (`3d43ed997e71`) with vllm v0.15.0, where the lookup is by bare
soname rather than by vendor directory (see *Version boundary* below; recorded
#7095). Recorded for #16379: `DeepSeek-V4-Flash-0731-w8a8` on one
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

## Version boundary: at the v0.15-era pin the lookup is a bare `dlopen`, and its
## root is an RPATH, not `ASCEND_CUSTOM_OPP_PATH`

The vendor-directory iteration and the `config.ini load_priority` list above are
the **newer** implementation. At v0.15.0rc1 (`3d43ed997e71`):

- `csrc/aclnn_torch_adapter/op_api_common.h:140-160` — `GetOpApiLibHandler`
  is `dlopen(libName, RTLD_LAZY)` with the bare names `"libcust_opapi.so"`
  (`:132`) then `"libopapi.so"` (`:130`); no environment variable is read.
- `CMakeLists.txt:141` bakes the search root into the extension itself:
  `-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/vllm-ascend/op_api/lib`.
  So the vendored library is found *only* when it physically exists at
  `vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/` inside the
  installed package; `ASCEND_CUSTOM_OPP_PATH` (set by
  `vllm_ascend/platform.py:461-467`, and only when that directory already
  exists) is not what makes the symbol visible at this pin.
- The package is produced by `csrc/build_aclnn.sh:101-102`
  (`./output/CANN-custom_ops*.run --install-path=$ROOT_DIR/vllm_ascend/_cann_ops_custom`)
  and reaches the wheel only through `setup.py:396-405`.

Same message, same two commands, different attribution: at this pin a **present**
vendor package whose `op_api/lib/libcust_opapi.so` was built for another SoC
fails exactly like an **absent** one. Do not re-derive the search mechanics —
`ls` the directory and `nm` the library; that pair is the whole diagnosis at
every pin.

## The published wheel is built for one SoC: A2

At v0.15.0rc1 the release workflow builds **one** wheel per OS/python
(`.github/workflows/schedule_release_code_and_wheel.yml`, matrix `os` ×
`python-version`, no SoC axis) inside
`.github/workflows/dockerfiles/Dockerfile.buildwheel`, whose
`ARG SOC_VERSION="ascend910b1"` is exported before `python3 setup.py
bdist_wheel`. `bdist_wheel` runs the `build_aclnn` command (`setup.py:409`,
`build_and_install_aclnn:187-203`), and `csrc/build_aclnn.sh` selects the
`^ascend910b` branch (A2 op set, `SOC_ARG="ascend910b"`) for that value — the A3
branch (`^ascend910_93`, `SOC_ARG="ascend910_93"`, `add_rms_norm_bias` in its
`CUSTOM_OPS_ARRAY`) is never taken.

Consequence for triage: `pip install vllm-ascend==<tag>` on an **A3 / 910_93**
host carries an A2 custom-op package, while the `…-a3` image
(`Dockerfile.a3`, `quay.io/ascend/vllm-ascend:<tag>-a3`) is built for A3. So for
this message on an A3 host with a site-packages install, rank the SoC/packaging
mismatch **first**, and rung 1 is the A3 image or a source build with
`SOC_VERSION=ascend910_9391` (`setup.py:142-147` lists the A3 sub-SoCs) — not a
serve flag, not `--compilation-config`, not `--enforce-eager`. `enable_custom_op()`
(`vllm_ascend/utils.py:243-265` at this pin) returns `False` only on `ImportError`
of `vllm_ascend_C`, so a wheel-built extension dispatches the fused op on any
SoC; there is no runtime switch to fall back to `torch_npu.npu_add_rms_norm`.

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
| `AttributeError: '_OpNamespace' object has no attribute 'npu_hc_pre_v2'`, or `Failed to register custom ops, all custom ops will be disabled` in the log | `vllm_ascend.vllm_ascend_C` was not built/imported (`setup.py:369,440,463` build only under `envs.COMPILE_CUSTOM_KERNELS`; `enable_custom_op()` `vllm_ascend/utils.py:401-455` turns **all** custom ops off on `ImportError`, and on A5 or `VLLM_BATCH_INVARIANT` by design at `:418-420`) | the install/wheel; if the reporter's per-tag table shows the `.so` missing from the published image, `../image-dependency-faults/custom-op-extension-missing-from-image.md` — but when the log carries **no** `Failed to register custom ops` line and the host is A5/950 or `VLLM_BATCH_INVARIANT` is exported, the gate never attempted the import: `custom-ops-disabled-by-device-type.md` |
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
python -c "import vllm_ascend,os;p=os.path.dirname(vllm_ascend.__file__)+'/_cann_ops_custom/vendors';print(os.listdir(p))" 2>&1
```
The last line is the A2/A3 question in one call: `['vllm-ascend']` plus a
`version.info`/`config.ini` naming `ascend910b*` on a 910_93 host *is* the
finding (see the wheel section above).

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

## Recorded #7095 (Kimi-K2.5, A3, vllm 0.15.0 + vllm-ascend 0.15.0rc1)

`aclnnAddRmsNormBias … not in libopapi.so` from
`vllm_ascend/ops/layernorm.py:55` (`npu_add_rms_norm_bias`, reached only when
`residual is not None` **and** `enable_custom_op()`), surfaced as
`RuntimeError: Worker failed with error …` out of
`determine_available_memory → profile_run`: memory-profiling phase, lane B, no
collective involved. The traceback path is `/opt/miniconda3/…/site-packages/
vllm_ascend/`, i.e. a non-editable install, and the reported SoC is A3 — the
shape the wheel section above ranks first. The recorded session shipped no
message in 24 turns while reading `setup.py`, the wheel workflow and
`Dockerfile.buildwheel` to decide *how* the package got there; that question
does not change the two `ls`/`nm` checks or rung 1, so it is not the answer.

## Open in the recorded case

- Whether the reporter's container had the vendor package at all: unverified.
  The recorded answer ranked the CANN build first and the A3 support gap third
  and never listed `_cann_ops_custom` or `libcust_opapi.so`; that ranking is the
  reason this page exists.
- Which A3 sub-SoC the wheel was built for (`ascend910_9391` vs a bare
  `ascend910_93`) is not visible from the paste; `SOC_VERSION` is recorded by
  `setup.py` at build time, so ask for the build log or read `config.ini`.
