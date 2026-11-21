# `libatb.so` missing: NNAL/ATB is a separate install with its own `set_env.sh`

Worked case: #10525 (vllm-ascend v0.20.2rc1 `367b8e6`, vllm v0.20.2 `bc150f50`,
CANN 9.0.0, 910B4, Ubuntu 22.04 hand-built instead of the prebuilt image).
Symptom, verbatim (per rank, so all TP ranks print the same block):

```text
File ".../torch/_ops.py", line 1442, in load_library: ctypes.CDLL(path)
OSError: libatb.so: cannot open shared object file: No such file or directory
  ... from torch_npu/op_plugin/atb/_atb_ops.py:66  torch.ops.load_library(atb_so_path)
OSError: Could not load this library: <venv>/site-packages/torch_npu/lib/libop_plugin_atb.so
  ... from vllm_ascend/worker/worker.py:108  _register_atb_extensions()
OSError: [Errno None]      <- re-raised NNAL_EX wrapper; the first line is the cause
WorkerProc failed to start. -> EngineCore failed to start. -> APIServer exits
```

Read it as a **Lane B** worker error whose owner is the environment, not vllm-ascend:
the failure is at worker init, before any `Starting to load model` line, and it is
identical on every rank.

## Why this is the environment, not a defect

At the pinned ref, ATB registration is unconditional on everything except A5:

| claim | evidence (v0.20.2rc1 `367b8e6`) |
|---|---|
| NNAL is a required component and owns `libatb.so` | `docs/source/installation.md:18` (`NNAL == 9.0.0 — Required for libatb.so`) |
| the **image** gets NNAL for free, a hand-built env does not | `docs/source/installation.md:51` ("The CANN prebuilt image includes NNAL … No additional installation is required when using the prebuilt image") |
| the docs already warn about this exact error | `docs/source/installation.md:81` ("libatb.so not found … ensure NNAL is properly installed") |
| manual install = `Ascend-cann-nnal_9.0.0_linux-$(uname -i).run --install` then a **separate** `source /usr/local/Ascend/nnal/atb/set_env.sh` | `docs/source/installation.md:97,107` — note `:97` is the CANN `ascend-toolkit/set_env.sh`, `:107` is NNAL's |
| the FAQ lists them as two independent steps | `docs/source/faqs.md:87-93`, item 1 = `source /usr/local/Ascend/nnal/atb/set_env.sh`, item 2 = `source …/ascend-toolkit/set_env.sh`, item 3 = `npu-smi info` |
| every official image sources it at build/run time | `Dockerfile:69`, `Dockerfile.a3:71`, `Dockerfile.310p:54`, `Dockerfile.openEuler:69` |
| why every worker hits it | `vllm_ascend/worker/worker.py:106-108`: `if get_ascend_device_type() != AscendDeviceType.A5: _register_atb_extensions()` — 910B4 is A2, so it always runs |

**The rule the docs imply, and which a later public comment on the same issue
(2026-08-11) states from another user's own test — the reporter could not retest,
having lost access to the machine, so treat the causal claim as
comment-confirmed rather than reporter-confirmed:** CANN's `set_env.sh` does
**not** export the ATB paths. `source <ASCEND_HOME>/cann*/set_env.sh` and
`source <ASCEND_HOME>/nnal/atb/set_env.sh` must each be executed once; a CR
container or a `~/.bashrc` that only sources the CANN one leaves `ATB_HOME_PATH`
and the nnal `lib64` out of `LD_LIBRARY_PATH`. `<ASCEND_HOME>` is the Ascend
install root: `/usr/local/Ascend`, or a conda prefix such as
`$CONDA_PREFIX/Ascend` — it is **not** necessarily derived from where the
reporter's CANN `set_env.sh` lives. The structural evidence for the same rule is
in the tree: `faqs.md:91-92` lists the two `source` commands as separate steps,
and `installation.md:97` (CANN) precedes `:107` (NNAL) without either implying
the other.

## "安装过程中没有报错" does not cover NNAL

NNAL is a `.run` installer outside pip, and the docs' **two install routes do not
contain it**: at v0.7.1rc1 the CANN manual-install block carries
`Ascend-cann-nnal_8.0.0_linux-aarch64.run --install` plus both `source` lines
(`docs/source/installation.md:88-93`), while the pip block (`:118-141`) and the
"Build from source code" block (`:145-171`, `VLLM_TARGET_DEVICE=empty pip install .`
→ `pip install -e .` → the `torch_npu-*.whl`) never mention NNAL. So a reporter who
answers "followed the doc, installation finished without errors" with a paste of
only pip commands has told you NNAL was **never installed** — #152's reporter did
exactly that (issuecomment-2677722481, 2025-02-24: the source-build block verbatim,
his `libatb.so` ImportError unchanged). Read the pasted command list for a `.run`
line and a `nnal/atb/set_env.sh` source before accepting the clean install as a
refutation of cause 1. The same asymmetry means "the doc worked for someone else"
proves nothing here: it works only for the reader who also ran the CANN block.

## Steps, in order

1. `ls -l <ASCEND_HOME>/nnal/atb/set_env.sh` and
   `find <ASCEND_HOME> -name 'libatb.so' 2>/dev/null` — this splits "not
   installed" from "installed, not sourced", and both fixes are cheap.
2. Not installed → install NNAL **matching the CANN version** (`== 9.0.0` here;
   the docs' `wget … Ascend-cann-nnal_9.0.0_linux-$(uname -i).run` needs
   `--header="Referer: https://www.hiascend.com/"`). Then source its
   `set_env.sh`.
3. Installed but absent from `LD_LIBRARY_PATH` → add the `source` line to
   `~/.bashrc` **after** the CANN one, and verify in a fresh shell
   (`echo $ATB_HOME_PATH; echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -i nnal`).
   If the deployment is started by a launcher (`args:`/`command:` in a Pod or
   operator CR) rather than by the reporter's shell, the profile line never
   reaches the engine process — put the `source`/export in the launch command and
   read `launch-env-and-library-path.md`; the ENV dump will look correct either
   way.
4. Confirm with the one-line repro/verification, which is exactly the worker's
   own call:
   `python -c "import torch, torch_npu; from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions; _register_atb_extensions(); print('ATB OK')"`.
   It must print without a traceback before `vllm serve` is retried. The FAQ's
   own probe is weaker — `import torch` / `import torch_npu` / `import vllm`
   (`faqs.md:97-101`) can pass while the worker still fails, because
   `libop_plugin_atb.so` is only `load_library`-ed by `_register_atb_extensions()`;
   do not accept a clean triple-import as evidence that ATB resolves.
5. Choosing the `cxx_abi_0` vs `cxx_abi_1` root is *sometimes* the fix and never
   a hand-copied file: do **not** copy one `libatb.so` into `/usr/lib64`, symlink
   the two roots or preload one `.so`. `nnal/atb/set_env.sh` picks the root to
   match the torch ABI, so a plain missing-library case is solved by sourcing it.
   But when a **prebuilt third-party wheel** was compiled against the other side,
   overriding the root *is* the reporter-confirmed remedy (#1144: `export
   ATB_HOME_PATH=…/atb/cxx_abi_0` + its `lib` first on `LD_LIBRARY_PATH` made
   `import mindie_turbo` work). Decide it from the failing `.so`'s mangled symbol
   — `Ss` = old `std::string` ABI, `NSt7__cxx11` = new — not by guessing, and
   check that torch_npu's own `libop_plugin_atb.so` does not need the opposite
   side (both roots share the `libatb.so` SONAME, one per process). Steps in
   `abi-undefined-symbol.md`; classifier: `scripts/abi-symbol-check.sh <paste>`.

`ldd .../torch_npu/lib/libop_plugin_atb.so | grep -i atb` shows the unresolved
dependency directly when the reporter cannot run the Python probe.

## The two distractors in the same log

- **`CANN: None` / empty CANN version in the environment dump is a script
  assumption, not a missing CANN.** `collect_env.py:300` reads
  `/usr/local/Ascend/ascend-toolkit/latest/<arch>-linux/ascend_toolkit_install.info`;
  an install rooted at `/usr/local/Ascend/cann-9.0.0` (or any custom root) is
  simply not at that path. Reporter-confirmed on #10525 ("cann的路径在安装完成cann后
  已有，但这个检查脚本确实没检查到"). Cross-check `ASCEND_HOME_PATH` and
  `python3 -c "import acl; print(acl.get_soc_name())"` before saying CANN is
  absent, and ask for the real install root.
- **`ERROR … Failed to import Triton kernels … cannot import name
  'constexpr_function' from 'triton.runtime.jit'` is non-fatal noise.** It comes
  from upstream vllm's fused-MoE config when an unrelated `triton_kernels`
  package is importable but incompatible with the installed triton
  (triton-ascend 3.2.1 is the pinned, correct pin for v0.20.2rc1 —
  `requirements.txt:20`); it is logged and the process continues. Rank it below
  the environment fix and suggest `pip list | grep -i triton` +
  `pip uninstall triton_kernels`, not a rollback.

Recheck conditions: the `worker.py` gate (`AscendDeviceType.A5`), the
`installation.md` component table and the FAQ entry are all version-scoped to
v0.20.2rc1 here; re-grep them if the report names another release, and check
`requirements.txt`/`pyproject.toml` for the companion `triton-ascend` pin rather
than recalling one.

Formatting-of-the-report note: this reporter pasted the same 4-rank traceback
four times, so the paste has no model-load section at all. Ask for one rank's
log plus `npu-smi info` if the next attempt fails differently.
