# `'_OpNamespace' '_C_ascend' object has no attribute '<op>'`: the whole namespace is empty

Verified at vllm-ascend **v0.19.1rc1** (`da421afad719`) with vllm **v0.19.1**
(`b1388b1fbf5a`). Recorded for **#10074**: `torch 2.10 / torch_npu 2.10 / a5 /
cann 9.0.0 / vllm 0.19.1 / vllm_ascend 0.19.1rc1`, a DSA/sparse-MLA serve whose
first request died in `mla_forward` → `sfa_v1._execute_sparse_flash_attention_process`
→ `torch.ops._C_ascend.npu_sparse_flash_attention`, with the reporter's own
diagnosis being "the op name has not been added into the build list for a5".

Line numbers move between releases; re-grep the two names below before quoting them.

## What the message does and does not say

`torch.ops.<ns>` is a **lazy** `_OpNamespace`: `torch/_ops.py`
`_OpNamespace.__getattr__` raises `AttributeError: '_OpNamespace' '<ns>' object has
no attribute '<op>'` for any name that was not registered **in this process**. So the
text states only that the name is absent at that moment. It does **not** say the
`.so` is missing, and it does not say the op is unbuilt for this SoC — those are two
other owners with two other texts (`EZ1001 … does not support opType`,
`aclnn<Op> … not in libopapi.so`).

Two owners produce "**every** op of the namespace is absent", and both are silent
switches, not installs:

| Switch | Where | What it means |
|---|---|---|
| `get_ascend_device_type() == AscendDeviceType.A5` | `vllm_ascend/utils.py:293-295` | A5/950 runs with **all** custom ops off **by design** (FIXME in the tree cites vllm-ascend#7157) |
| `envs.VLLM_BATCH_INVARIANT` | same line | batch-invariant mode needs every kernel to be BI-implemented, so the whole extension is off — `../accuracy-benchmarks/batch-invariant-coverage.md` |

`enable_custom_op()` (`utils.py:276-307`) memoises in `_CUSTOM_OP_ENABLED` (`:54`),
returns `False` at `:293-295` **before** the import, and only the `except ImportError`
arm at `:307-308` logs `Failed to register custom ops, all custom ops will be
disabled`. **The absence of that warning is therefore not evidence that the
extension is installed** — on A5 and under BI the import never runs and nothing is
printed. Check the device type and the env line before the wheel.

## Why a working-looking op call can still reach an empty namespace

The import is **lazy on purpose**: `AscendPlatform.import_kernels()`
(`vllm_ascend/platform.py:575-596`) sets `ASCEND_CUSTOM_OPP_PATH` but its
`enable_custom_op()` call is commented out (PR #884 — importing early breaks
`ASCEND_RT_VISIBLE_DEVICES` applied at runtime), so registration happens at the
first `enable_custom_op()` call site: `ops/layernorm.py:71,100`,
`compilation/passes/norm_quant_fusion_pass.py:495`, `ops/fused_moe/moe_mlp.py:38`,
`distributed/kv_transfer/kv_p2p/mooncake_connector.py:575`, `utils.py:91`. Any code
path that calls `torch.ops._C_ascend.<op>` **without** consulting the gate inherits
whatever state the gate is in.

The SFA path is exactly that: `vllm_ascend/attention/sfa_v1.py` holds five ungated
`torch.ops._C_ascend.*` calls (`:839 batch_matmul_transpose`, `:875 mla_preprocess`,
`:998`/`:1028 npu_lightning_indexer{_quant}`, `:1049 npu_sparse_flash_attention`) —
the only guard in the file is a `hasattr(torch.ops._C_ascend,
"batch_matmul_transpose")` at `:834`, which returns **False** on an empty lazy
namespace instead of raising, i.e. the tree already knows the namespace can be empty.
`AscendPlatform.get_attn_backend_cls` selects that backend for
`(use_mla=True, use_sparse=True)` with **no** A5 exclusion (`platform.py:599-604`).
So on A5 a sparse-MLA model dies at whichever call comes first; the op named in the
traceback is the messenger, not the finding. `meta_registration` is skipped by the
same branch, so `torch.compile` on A5 sees no meta kernels either
(`../startup-failures/meta-kernel-missing.md`).

## "Is op F usable on SoC X at this pin?" — three gates, in order

This is the capability verdict a "支持了吗 / is the feature ready now" report needs.
Each gate is one grep in the reporter's own ref:

1. **Runtime device gate** — `utils.py:293-295` above. On A5 it alone answers "no",
   for every custom op.
2. **Build list** — `csrc/build_aclnn.sh:6-72`: `^ascend310` (`:8`), `^ascend910b`
   (`:26`, which does contain `sparse_flash_attention`), `^ascend910_93`
   (`CUSTOM_OPS_ARRAY`, `:28-68`), then `else` (`:70-72`)
   `# currently, no custom aclnn ops for other series` + `exit 0`. There is no
   `ascend950` branch, so an A5 build produces no `_cann_ops_custom` package at all.
3. **The op's own SoC config** — `csrc/<op>/op_host/<op>_def.cpp`
   `this->AICore().AddConfig("<soc>", …)`. For SFA at this ref:
   `csrc/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp:85-86` lists
   only `ascend910b` and `ascend910_93`. Adding the name to an A5 branch without a
   matching `AddConfig` would not yield a runnable kernel.

**Do not hunt the binding.** `csrc/torch_binding.cpp` registers
`npu_sparse_flash_attention` unconditionally in the non-310P block
(`TORCH_LIBRARY_EXPAND` at `:1026`; schema `:1135`, impl `:1142`; the 310P block is
`:992`), so "the op was never added to the binding" is refutable in one grep and is
never the missing piece in this lane.

## Deliver

For a reporter on a released wheel/image on A5 the answer is a **verdict plus what
would change it**, because all three gates are outside their control:

- Name the gate that actually stops them (the device-type switch) and say the
  symptom is not op-specific — the next `torch.ops._C_ascend.*` on the same path
  raises the same `AttributeError`.
- Their cited PR/issue (#9491 in the recorded case, plus the #7157 tracker the FIXME
  names) is the right place to watch; a build-list patch on their machine is not a
  fix, and neither is `SOC_VERSION=ascend950…` — that yields an extension with no
  A5 aclnn binaries, i.e. the failure just moves to a runtime text
  (`aclnn<Op> … not in libopapi.so`, `EZ1001`).
- On A2/A3 the same empty namespace is a **launch** question first: read the env
  block for `VLLM_BATCH_INVARIANT`; unsetting it restores every custom op. If the log
  *does* carry `Failed to register custom ops, all custom ops will be disabled`, the
  owner is the install/wheel
  (`../image-dependency-faults/custom-op-extension-missing-from-image.md`, the 310P
  GDN family) or a genuine per-SoC build gap
  (`op-not-registered-for-soc.md`), not this page.

## Recorded run

#10074 asked the question directly ("the pull request of SFA to support a5 … is the
feature ready now?") and the session reached the device-type gate only after ten
source reads of `build_aclnn.sh`, the op's `AddConfig` list, `torch_binding.cpp`,
`setup.py` and `sfa_v1.py`. The order to keep: **device-type gate first** (one grep on `utils.py`,
which changes what the rest of the audit means), then gates 2-3 as corroboration for
the answer text, then stop — the reporter's premise ("one name missing from the
build list") is refuted without a kernel walk.
