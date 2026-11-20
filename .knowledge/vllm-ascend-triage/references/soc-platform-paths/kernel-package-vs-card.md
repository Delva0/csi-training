# Built-in aclnn op has no kernel for the running SoC (`EZ9999 … Parse dynamic kernel config fail`)

Verified at vllm-ascend `v0.7.3` (`779eebb8021e`) and vllm `v0.7.3` (`ed6e9075d31e`),
recorded for #1696: 910C1 host, `quay.io/ascend/vllm-ascend:v0.7.3` image,
host CANN 8.1.RC1, `LLM("Qwen/Qwen2.5-0.5B-Instruct")`. The reporter's literal
question was "does this image support 910C, or did I misconfigure something?".

## The signature

```
[rank0]:   File ".../vllm/model_executor/layers/rotary_embedding.py", line 109, in _compute_inv_freq
[rank0]:     inv_freq = 1.0 / (base**(torch.arange(
[rank0]: RuntimeError: call aclnnArange failed, detail:EZ9999: Inner Error!
[rank0]: EZ9999: [PID: 2075] 2025-07-09-06:39:54.780.150 Parse dynamic kernel config fail.
[rank0]:        AclOpKernelInit failed opType
[rank0]:        ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.
[rank0]: [ERROR] ... ERR01100 OPS call acl api failed
```

`Parse dynamic kernel config fail` + `AclOpKernelInit failed opType` + a
`<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed` line means the **op's kernel could
not be initialised at all** — no kernel-config entry matched the SoC the process
runs on, so nothing was launched. `ERR01100` is the toolkit's "acl api failed"
wrapper, not a device report.

Two things make this a *pre-engine, image-side* question rather than a code walk:

- **The failing op is a CANN built-in** (`torch.arange` through `torch_npu`'s
  `transfer_to_npu` shim), not one of vllm-ascend's custom ops. `csrc/`,
  `_cann_ops_custom`, `ASCEND_CUSTOM_OPP_PATH` and `binary_info_config.json`
  (the `EZ1001` page) are all the wrong drawer here.
- **The pasted traceback has no `vllm_ascend` frame.** In #1696 it was upstream
  `rotary_embedding.py` building the RoPE cos/sin cache during model construction,
  i.e. the first NPU touch — every model dies at a comparable spot, which is why
  the report looks model- and VRAM-independent.

Escape hatch of the whole investigation: one standalone probe inside the same
container, no vLLM involved —

```bash
python3 -c "import torch, torch_npu; print(torch.npu.get_device_name(0)); print(torch.arange(8, device='npu:0'))"
```

The same `EZ9999` there closes the case on the image/CANN-vs-card pairing.

## Split the two causes before advising a fix

| Cause | Evidence that decides it |
|---|---|
| the image's CANN kernel package does not cover this card (wrong image line for the SoC) | the pinned docs exclude the card (below) — then no flag helps, the image has to change |
| the kernels install is incomplete / partial on a *supported* card | docs include the card, so reinstall the matching `Ascend-cann-kernels-<soc>` run and check the container really has it |

The docs answer it in three lines at the pinned ref (all v0.7.3):

| Fact | Location |
|---|---|
| image base is the **910b** CANN | `Dockerfile:16` → `FROM quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10`; `docs/source/conf.py:78` `'cann_image_tag': "8.1.rc1-910b-ubuntu22.04-py3.10"` |
| supported hardware at this release is A2/910b only | `docs/source/faqs.md:14-17` — "Currently, **ONLY Atlas A2 series** (Ascend-cann-kernels-910b) are supported"; `docs/source/quick_start.md:6-7` lists the same A2 families |
| the manual path installs the same single kernel package | `docs/source/installation.md:87-89` → `Ascend-cann-kernels-910b_8.1.RC1_*.run` |

**The container's CANN is the image's, not the host's.** The documented
`docker run` (`docs/source/installation.md:54-65`) mounts only `/usr/local/dcmi`,
`npu-smi`, the driver's `lib64/` + `version.info` and `/etc/ascend_install.info` —
never `ascend-toolkit/`. So the host toolkit version in the report is not the
deciding factor and "upgrade the host CANN" is not the fix; the image tag is.

## Red herring: "Failed to register custom ops" / `No module named 'vllm_ascend.vllm_ascend_C'`

At v0.7.3 this pair appears in a normal official image and is **not** the fault:

- `setup.py:296-298` appends `CMakeExtension(name="vllm_ascend.vllm_ascend_C")`
  only `if envs.COMPILE_CUSTOM_KERNELS is not None` (`vllm_ascend/envs.py:11-12`),
  and the Dockerfile never sets that variable — so the extension is simply absent.
- `vllm_ascend/platform.py:29-38` prints
  `Warning: Failed to register custom ops, all custom ops will be disabled` for
  *any* `ImportError` whose text is not exactly the `PyInit_vllm_ascend_C` one, so
  a plain "No module named …" always produces the warning.

Re-verify before reusing: by vllm-ascend `v0.23.0rc1` the string is gone from
`platform.py` and the import moved to a lazy path (`vllm_ascend/utils.py:435`), so
quote the red herring only for the older release line.

## What the snapshot cannot settle

- The `npu-smi` **name** (`910C1`) → `soc_version` string mapping is not in any
  pinned tree. Ask the reporter for the mapping rather than asserting it; the
  supported/unsupported verdict in #1696 came from the docs, not from it.
- Which later release/mirror tag first covers 910C/A3: absent when `SOURCES.json`
  pins only the old tree (`cann/driver`, `cann/ops-*` are `unavailable`). Say it is
  unverifiable here and send the question to the maintainers with the probe output.
- Do not name the kernel package that "should" be installed (`-910_93`? `-910c`?)
  without a pinned source; offer the check instead:
  `ls -d /usr/local/Ascend/ascend-toolkit/latest/opp/*/*/*/*/kernel 2>/dev/null`
  and compare the SoC-named directories against `npu-smi info` (path varies per
  CANN release — label it as such).

## Deliver

Verdict first: this is the image/CANN kernel package not covering the card, not a
configuration mistake and not a vllm-ascend bug. Then the standalone probe (the
reporter's own next action), the three doc lines, the host-vs-container split, the
two causes above with the docs statement that separates them, and the red herring.
Ask for: image tag, `npu-smi info`, the probe output, and whether the card is on
the release's documented list.
