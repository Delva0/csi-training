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

1. **The report's own env dump already answers the first half.** vllm-ascend
   ships its own `collect_env.py` and prints every variable whose name starts
   with `TORCH`/`PYTORCH`/`ASCEND_`/`ATB_` (`collect_env.py:307`) plus vllm's
   own list, which includes `LD_LIBRARY_PATH`. So a dump with **no
   `ATB_HOME_PATH=` line** and an `LD_LIBRARY_PATH` holding no `nnal`/`atb`
   entry is positive evidence that the ATB env was never sourced — quote it
   instead of asking `echo $ATB_HOME_PATH`, which is the same fact a round trip
   later. #10525's dump is that shape: only `<cann root>/lib64*` and
   `/usr/local/Ascend/driver/*` on `LD_LIBRARY_PATH`, no `ATB_*` line at all.
   Then `find <root> -name 'libatb.so' 2>/dev/null` splits "not installed" from
   "installed, not sourced"; both fixes are cheap.
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

## When the follow-up is the recipe question

A reporter in a self-installed CANN + conda env (or a container they launched
themselves, with no host access) usually answers the NNAL finding with 「哪里可以看到
你们镜像制作的依赖，我对比下」. That comparison is `image-recipe-inventory-and-diff.md`:
the image sources `nnal/atb/set_env.sh` at `Dockerfile:51` and never installs NNAL, so
the base image is where it comes from, and the same recipe also carries the `devlib`
`LD_LIBRARY_PATH` append (`:48,52`) and the `torch-npu` index (`requirements.txt:25-27`)
that a hand-built env must reproduce. Put those paths plus the in-container `pip list`
diff in the same message as this page's step 1, not in a later turn.

## When the report is a whole from-zero hand build

The follow-up after this page's fix is a documentation question, not a stack.
#10525's reporter answered 「cann的路径在安装完成cann后已有，但这个检查脚本确实没检查到，不过
这个影响是不是不大，主要是从零开始构建环境，没有一个详细的资料或说明吗？工程下的说明比较简单，
按顺序一个一个包的装和核对，依然到启动过程中会报上面的错误。从零构建这条路很难走通？」
— so the verdict is "viable, and the step you missed is in the doc but collapsed",
delivered with the ordered path and a gate per stage, not only the one missing
package.

**Where the pin hides it.** The whole manual CANN sequence sits inside one
collapsed `dropdown "Install CANN manually"` (`installation.md:76`) under the
**pip** tab, while that tab's *visible* text says the CANN base image "includes
NNAL … No additional installation is required when using the prebuilt image"
(`:49-52`). Inside the dropdown: toolkit `.run --full` `:94-96`,
`source …/ascend-toolkit/set_env.sh` `:97`, the SoC ops package
`.run --install` `:99-101`, NNAL `.run --install` `:103-105`, then
`source …/nnal/atb/set_env.sh` `:107`. The component table `:12-18` lists NNAL as
required, and FAQ §6 (`faqs.md:87-95`) leads with the NNAL `source` and puts the
CANN one second. A reader who follows the numbered, visible steps therefore
never sees stages 2-4 — that is the documentation defect to name in the same
message, and it is why "装完每个包都核对过" is compatible with this traceback.

**The order, one gate each** (line refs at v0.20.2rc1):

| stage | gate that must pass before the next stage | the paste's frame when it was skipped |
|---|---|---|
| firmware/driver | `npu-smi info` lists the dies (`installation.md:34`) | no driver version, `npu-smi` missing |
| CANN toolkit + its `set_env.sh` (`:94-97`) | `python3 -c "import acl; print(acl.get_soc_name())"` → `Ascend910B4-1` (the probe #10525's reporter ran) | ACL/GE init codes, `libascend*.so: cannot open shared object` |
| SoC ops/kernels package (`:99-101`) | no startup gate — an unresolved aclnn op at the first launch, i.e. *not* this traceback's stage | — |
| NNAL + its own `set_env.sh` (`:103-107`) | `python -c "import torch, torch_npu; from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions; _register_atb_extensions(); print('ATB OK')"` | this page's `libatb.so` `OSError` |
| python env + `torch`/`torch-npu`/`triton-ascend` pins | `python -c "import torch, torch_npu; print(torch.npu.is_available())"` (the tree's own check, `vllm_ascend/worker/worker.py:294`) | the SIGILL / ABI / `undefined symbol` forms → the sibling pages |
| `pip install vllm==<pin>` then `vllm-ascend==<pin>` (`:148-163`) | `pip show vllm vllm-ascend torch torch-npu` against the tag's own pair (see below) | plugin/import and pairing shapes |

**A relocated install root moves both documented `set_env.sh` paths.** #10525's
dump reads `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0` and the reporter
sourced `/usr/local/Ascend/cann/set_env.sh`, while the doc block's paths are
`/usr/local/Ascend/ascend-toolkit/set_env.sh` (`:97`) and
`/usr/local/Ascend/nnal/atb/set_env.sh` (`:107`). On such a box all three
documented absolute paths are wrong, so print the reporter's own roots (the CANN
root from `ASCEND_TOOLKIT_HOME`, the sibling `nnal/atb` root, `$ASCEND_OPP_PATH`)
and let `find` locate `libatb.so` — do not hand back the doc paths verbatim. The
pair to check the stack against is the tag's own (`installation.md:12-18`,
`requirements.txt:17,20,38`), not the neighbouring row of
`versioning_policy.md` — at an `rc` tag that table can still describe the
previous release (`../startup-failures/environment-correspondence.md`).

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
