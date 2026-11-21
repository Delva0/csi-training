# Built-in aclnn op has no kernel for the running SoC (`Parse dynamic kernel config fail`, `does not has any binary`)

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

## Second wording of the same fault: `EZ9999 … Op <Op> does not has any binary`

Verified at vllm-ascend v0.10.0rc1 (`4604882a3e2c`) and vllm v0.10.0 (`6d8d0a24c02b`),
recorded for #6082: `quay.io/ascend/vllm-ascend:v0.10.0rc1-310p` on a 300I Duo
(driver 25.2.0, firmware 7.7.0.6.236), `OpenGVLab/InternVL3-8B`, `--dtype float16
--enforce-eager --trust-remote-code --compilation-config '{"custom_ops":["none",
"+rms_norm","+rotary_embedding"]}'`. Text prompts answer; a request carrying an image
dies:

```
RuntimeError: NPU function error: call aclnnIsNegInf failed, error code is 561000
[ERROR] ERR00100 PTA call acl api failed
EZ9999: [PID: …] Op IsNegInf does not has any binary.
```

The paste's own attribution is `_execute_mm_encoder` → `scatter_mm_placeholders`, with
torch_npu's "the operator is called asynchronously, the stacktrace may be inaccurate"
line beside it. Same drawer as above, different wording: `Parse dynamic kernel config
fail` is kernel-*config* initialisation, `does not has any binary` is the op-package
inventory; both mean nothing was launched, both belong to the image's CANN, and
`csrc/`, `_cann_ops_custom`, `ASCEND_CUSTOM_OPP_PATH` and `binary_info_config.json` are
all the wrong drawer here too.

**The caller is not in any pinned tree — do not hunt it.** `rg -ni 'isneginf'` over the
pinned vllm and vllm-ascend trees at that ref returns **nothing**; the transformers and
pytorch exports are partial and show only a test file and a header; op-plugin declares
`isfinite` (`op_plugin/op_plugin/ops/OpInterface.h:452`) and no `IsNegInf` anywhere. A
torch_npu built-in of this family is reached from ATen composites (the SDPA math path's
`nan_to_num` / `isfinite`) or from the checkpoint's remote code, neither of which the
snapshot ships — so "who calls it" is an **open question from the turn you notice it**,
not a grep to keep running. #6082 spent its whole output budget on five-tree `isneginf`
greps plus a `is_310p()` call-site survey and delivered nothing
(`../evidence-workspace/delivery-discipline.md`).

Two cheap facts explain the shape:

- vLLM's vision tower is the only code an image request adds, and at vllm v0.10.0 it
  calls `F.scaled_dot_product_attention(q, k, v, scale=…)` with no mask
  (`vllm/model_executor/models/intern_vit.py:275`). Confirm the split by serving one
  text request and one image request on the same engine instead of reasoning about it.
- The 310P op set is the image's: `Dockerfile.310p:18 FROM
  quay.io/ascend/cann:8.2.rc1-310p-ubuntu22.04-py3.11`, and the 300I page says the
  platform is experimental, eager-only and float16-only
  (`docs/source/tutorials/single_node_300i.md:4-5`). A host driver bump does not add an
  op the image's CANN lacks. The tree's own precedent for gating a missing 310P op in
  Python is `vllm_ascend/sample/sampler.py:25` (`aclnnApplyTopKTopP … does not support
  310P`, guarded by `is_310p()`).

Decisive probe — same container, no vLLM, the reported op and the model's dtype
(`float16` in that image):

```bash
python3 -c "import torch, torch_npu; x=torch.tensor([float('-inf'),0.,1.],dtype=torch.float16).npu(); print(torch.isneginf(x)); print(torch.isfinite(x))"
```

- It fails the same way ⇒ SoC op-coverage gap: no serve flag helps. Deliver that, the
  image-CANN line, and the two honest limits — whether a later `-310p` image covers the
  op is unverifiable when `SOURCES.json` pins one release, and the op's absence is a
  maintainer/CANN question (no `cann/ops-*` tree in the snapshot).
- It passes ⇒ the failure is shape/dtype/format specific, so it is **not** an op-gap
  claim: rerun the serve with `ASCEND_LAUNCH_BLOCKING=1` (eager is already on) so the
  traceback names the real caller, and plog the request window.

Also state that the signature is new rather than citing a sibling: `history-titles.sh -i
'IsNegInf'` and the prior-month issue search both return 0 hits. Ask for the full
container log around the failing request (not the tail), the probe output, `npu-smi
info`, and whether a text-only request on the same engine succeeds.

## Third wording that is **not** this page: `Cannot find bin of op <Op>, integral key <key>`

`EZ9999 … Cannot find bin of op AddRmsNormQuant, integral key
0/1/|float16/ND/…/int8/ND/int8/ND/float16/ND/.` + `Kernel Run failed. opType: 31`
(#2387, v0.10.0rc1, 910B4, an fp16 W8A8 Qwen3 checkpoint) also fails before any
launch and also carries `call aclnn<Op> failed, error code is 561000` +
`ERR00100`, but the op **is** in the package and only the requested dtype key has
no binary. The SoC/image lane here (`npu-smi` name, `kernels-<soc>` run, image tag)
is the answer only when **both** dtypes fail; test the dtype first —
`../custom-ops-runtime-faults/cannot-find-bin-integral-key.md`.

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

**Scope of the red-herring verdict: only while no code path on that SoC needs the
extension.** At vllm-ascend v0.23.0.post1 (`1cdb8c4`) the 310P lane calls it
unguarded — `rg -c 'torch\.ops\._C_ascend' vllm_ascend/_310p/` returns 4 in
`_310p/ops/fla/gdn_310.py` and 2 in `_310p/ops/fla/chunk_gated_delta_rule.py` — so a
GDN / Qwen3.5 / Qwen3-Next family checkpoint on 310P dies at the first prefill with
`AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'npu_causal_conv1d_310'`,
and an image whose per-tag `vllm_ascend_C*.so` table shows the extension *missing*
is a real image regression, not a red herring:
`../image-dependency-faults/custom-op-extension-missing-from-image.md`. Run the grep
before quoting either verdict.

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
