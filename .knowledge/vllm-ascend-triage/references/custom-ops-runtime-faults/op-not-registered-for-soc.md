# Custom op not registered for this SoC (`EZ1001 Get regInfo failed`)

Verified at vllm-ascend `v0.18.0` (`e18643f8a4d5`) and vllm `v0.18.0`
(`bcf2be961200`). Recorded for #11519: Step-3.5, "VLLM 0.18.0 / VLLM-Ascend
0.18.0" on an Ascend 910_93 (A3) host.

## The signature

```
RuntimeError: call aclnnAddRmsNormBias failed, detail:[PID: 364676] ... AclNN_Parameter_Error(EZ1001):
Get regInfo failed, The binary_info_config.json of socVersion [ascend910_93]
does not support opType [AddRmsNormBias].
```

`EZ1001` / `Get regInfo failed` / `binary_info_config.json of socVersion [...]`
is aclnn's **registry lookup failing before any kernel launch**. The op is not
absent from the source tree and the kernel was never entered: the custom-op
package that is loaded in that container has no binary entry for the SoC the
process runs on. Owner: the build/install chain, not `csrc/<op>/`.

Separate the three surfaces before reading any kernel code:

| Text in the report | Phase | Owner |
|---|---|---|
| `call aclnnX failed` + `AclNN_Runtime_Error` (+ tiling / UB / DDR text, often a device dump or plog) | kernel launch | op kernel, shape-dependent — `add-rms-norm-bias-op-map.md` (recorded #8944) |
| `AclNN_Parameter_Error(EZ1001)`, `Get regInfo failed`, `binary_info_config.json of socVersion [...] does not support opType [...]` | aclnn op-registry lookup | this page: build SoC, stale vendor package, wrong image tag |
| `undefined symbol`, `op not found`, or *no* custom-op error at all because `import vllm_ascend.vllm_ascend_C` failed | Python import / `enable_custom_op()` fallback | install completeness |

## The source is not the suspect (checked, do not re-derive)

At v0.18.0 the tree builds this op for **both** A2 and A3, so "the A3 op list
forgot `add_rms_norm_bias`" is refuted before you assert it:

- `csrc/add_rms_norm_bias/op_host/add_rms_norm_bias_def.cpp:66-67` —
  `AddConfig("ascend910b")` and `AddConfig("ascend910_93")`.
- `csrc/build_aclnn.sh:12-29` A2 branch (`^ascend910b`): `CUSTOM_OPS` at `:28`
  contains `add_rms_norm_bias`, `SOC_ARG="ascend910b"` at `:29`.
- `csrc/build_aclnn.sh:30-67` A3 branch (`^ascend910_93`): `CUSTOM_OPS_ARRAY` at
  `:45`, `"add_rms_norm_bias"` at `:58`, `SOC_ARG="ascend910_93"` at `:67`.
- `:79` runs `build.sh -n "$CUSTOM_OPS" -c "$SOC_ARG"`; `:82` installs the
  result to `$ROOT_DIR/vllm_ascend/_cann_ops_custom`.

## The chain that actually decides which binaries exist

1. **Build-time SoC.** `setup.py:133-146` takes `SOC_VERSION` from the env or
   from chip detection (`npu-smi`), with the documented values
   (`ascend910b1` A2, `ascend910_9391` A3, `ascend310p1`, `ascend950*`);
   `:223` runs `csrc/build_aclnn.sh` but `:433` only when
   `envs.COMPILE_CUSTOM_KERNELS`; `:419-427` copies the *repo's*
   `vllm_ascend/_cann_ops_custom` into `build_lib`. A wheel built on a CPU-only
   host or for another SoC therefore ships no binaries for the reporter's SoC.
2. **Runtime vendor path.** `vllm_ascend/platform.py:573-579` prepends
   `ASCEND_CUSTOM_OPP_PATH` with `<site-packages>/vllm_ascend/_cann_ops_custom/
   vendors/vllm-ascend` **only if that directory exists**. When it is missing or
   stale, a toolkit-side `/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/*`
   package from an earlier release is what aclnn resolves against.
3. **Image tags are per-SoC.** `docs/source/installation.md:197-204` lists
   `…-a3` / `…-a3-openeuler`; `Dockerfile:22` is `ascend910b1` while
   `Dockerfile.a3:22` is `ascend910_9391`. Source builds need the catlass
   submodule and `export SOC_VERSION=ascend910_9391`
   (`docs/source/installation.md:169-186`).

## Why the op runs at all (for an RMSNorm/Bias report)

`vllm/model_executor/models/step3p5.py:524-525` builds `GemmaRMSNorm` (Step-3.5);
`vllm_ascend/ops/layernorm.py:91-107` `AscendGemmaRMSNorm.forward_oot` calls
`torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, 1.0 + self.weight, None,
eps)` whenever a residual is present (otherwise `npu_gemma_rms_norm`), gated by
`enable_custom_op()` (`vllm_ascend/utils.py:257-279`: no env override — off only
for batch-invariant, A5, or a failed `vllm_ascend_C` import). So there is **no
launch flag** that moves the reporter to the `torch_npu` fallback at v0.18.0;
the fix is the package, not a serve option.

## Ranked causes and the one check each

1. **Image or build targets another SoC** (A2 image on an A3 host, or a source /
   pip build without `SOC_VERSION`). Check on the reporter's host: the image
   tag; `echo $SOC_VERSION`; then
   `P=$(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))")`
   and look for `AddRmsNormBias` under `$P/_cann_ops_custom`.
2. **Stale / shadowing vendor package** from an earlier release (`op added
   later`). Check `echo $ASCEND_CUSTOM_OPP_PATH` and
   `ls /usr/local/Ascend/ascend-toolkit/latest/opp/vendors`; compare the op
   lists between the two `binary_info_config.json` trees.
3. **Partial install**: `$P/_cann_ops_custom` empty or missing (a wheel built
   without `COMPILE_CUSTOM_KERNELS`, or a `pip install` that skipped the aclnn
   step).
4. **Genuine op gap for that SoC at that release** — only after 1-3 are excluded,
   and only with the version's own `build_aclnn.sh` branch and `def.cpp`
   `AddConfig` list as evidence.

## Deliver

Headline first: this is a *registry/build* fault, not the device fault of the
same op in the precedent issue; the kernel never ran. Then the source evidence
that the tree does build it for both SoCs, the four checks in order, and the
cheapest action (use the `-a3` image, or rebuild with `SOC_VERSION` and the
catlass submodule). Ask for what the report omitted: `collect_env.py` output
(the environment block was empty in #11519), image tag, install method, and the
`binary_info_config` grep result. Do **not** ask for a tiling I-line: no tiling
decision was reached.
