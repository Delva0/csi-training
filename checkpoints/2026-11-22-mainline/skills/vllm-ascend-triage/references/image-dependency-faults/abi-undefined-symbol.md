# `ImportError: …/_ops*.so: undefined symbol: …` — the `cxx_abi_0` / `cxx_abi_1` build choice

Worked case: vllm-ascend **v0.7.3.post1** (`c69ceac`), vllm v0.7.3, PyTorch/torch_npu
2.5.1, aarch64 EulerOS. Symptom, verbatim:

```text
运行 import mindie_turbo 时会出现链接库不匹配的错误:
ImportError: /…/site-packages/mindie_turbo/_ops.cpython-310-aarch64-linux-gnu.so:
undefined symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs
```

The same report pasted its own environment, and that is where the answer is:

```text
ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
LD_LIBRARY_PATH=…:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:…
```

**Reporter-confirmed (2025-06-10, same issue): "Thanks, the env setting works for
me."** — i.e. the reporter applied the `cxx_abi_0` override below and the import
resolved. This is an environment/packaging fault that fires *before* any
vllm-ascend code, and it is answered from the mangled symbol plus the env dump.

## Not this page

Only a **third-party / prebuilt** `.so` whose mangled name carries a `std::string`
form (`…ERKSs`, `…ERKNSt7__cxx11…`) is this ABI question. When the `.so` is
vllm-ascend's **own** `vllm_ascend_C*.so` and the unresolved name is an internal
namespace symbol (`_ZN9pp_matmul17GetPpMatmulTilingE…`, any `vllm_ascend::` /
`bmm_trans::` symbol), no `LD_LIBRARY_PATH` or `ATB_HOME_PATH` value can fix it: a
build source list left the defining translation unit out for that SoC →
`custom-op-extension-missing-from-image.md` (Step 0 / Lane B, verified on #6527 at
v0.14.0rc1 for 310P).

## What the symbol itself proves

Demangle first; two forms of `std::string` exist and they are different symbols:

| mangled | demangled (local `c++filt`) | build |
|---|---|---|
| `…GetOperationByNameERKSs` | `AsdOps::Ops::GetOperationByName(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) const` | **old ABI**, `_GLIBCXX_USE_CXX11_ABI=0` |
| `…GetOperationByNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE` | same signature with `std::__cxx11::basic_string` | new ABI, `_GLIBCXX_USE_CXX11_ABI=1` |

Because the mangled names differ, the loader reports *undefined symbol*, never
"wrong signature": the defining library on the path was compiled on the other
side of the ABI switch. NNAL/ATB ships both builds on purpose —
`<ASCEND_HOME>/nnal/atb/latest/atb/cxx_abi_0/` and `…/cxx_abi_1/`, each with its
own `lib/` — and `nnal/atb/set_env.sh` exports one of them (`ATB_HOME_PATH` plus
its `lib` on `LD_LIBRARY_PATH`). A prebuilt wheel whose extension was compiled
against the other one cannot resolve at import.

`sh scripts/abi-symbol-check.sh <paste>` does the demangling, the old/new-ABI
classification, the env-marker count and prints the three `nm` commands to hand
the reporter (verified on the #1144 paste).

## Steps, in order

1. `bash <skill_root>/scripts/abi-symbol-check.sh <paste>` — confirms this class and
   which side the paste's environment selects. If the paste instead carries
   `cannot open shared object file` or `GLIBCXX_3.4.2x not found`, it says so:
   those are the missing-library and old-toolchain classes, not this one.
2. On the reporter's host, name the side each object needs and the side that
   defines the symbol (both are one-liners):
   ```bash
   nm -D --undefined-only …/mindie_turbo/_ops.cpython-310-aarch64-linux-gnu.so | grep GetOperationByName   # U …ERKSs
   for f in <ASCEND_HOME>/nnal/atb/latest/atb/cxx_abi_{0,1}/lib/*.so*; do
     echo "== $f"; nm -D --defined-only "$f" 2>/dev/null | grep GetOperationByName; done                        # T in one of them
   ```
3. Check that the torch stack does not need the opposite side, because both roots
   carry the **same SONAME** (`libatb.so`, `libasdops.so`) and one process loads
   one of them:
   ```bash
   nm -D --undefined-only <venv>/site-packages/torch_npu/lib/libop_plugin_atb.so | grep -c 'NSt7__cxx11'
   ```
   `0` → the torch stack is on the old ABI too, so the override in step 4 is
   consistent. Non-zero → torch_npu's own plugin wants the cxx11 definitions and
   no `LD_LIBRARY_PATH` ordering can satisfy both; the answer is a wheel built for
   the ABI the stack uses, not a path fix.
4. The fix, when 2 and 3 agree (this is what the #1144 reporter confirmed):
   ```bash
   export ATB_HOME_PATH=<ASCEND_HOME>/nnal/atb/latest/atb/cxx_abi_0
   export LD_LIBRARY_PATH=<ASCEND_HOME>/nnal/atb/latest/atb/cxx_abi_0/lib:$LD_LIBRARY_PATH
   python -c "import mindie_turbo"
   ```
   `<ASCEND_HOME>` is the Ascend install root (`/usr/local/Ascend`, or a conda
   prefix such as `$CONDA_PREFIX/Ascend`). Keep the `export` **in front** of the
   inherited path, and put it in the process that actually runs the engine: a
   launcher-managed deployment needs it in the launch command
   (`launch-env-and-library-path.md`), not in an interactive shell.
5. Do **not** hand-copy one `libatb.so`/`libasdops.so` into `/usr/lib64`, symlink
   the two roots together or preload one of them "so both resolve": the SONAMEs
   collide, so the second consumer of the other ABI turns this load error into a
   wrong-symbol crash later. Choosing the matching root (step 4) is the supported
   move; mixing them inside one process is not.
6. Record what it costs when left alone, because the two consumers of
   `mindie_turbo` degrade differently (verified at v0.7.3.post1):
   - `vllm_ascend/worker/worker_v1.py:77-82` calls `try_register_lib("mindie_turbo", …)`
     and `vllm_ascend/utils.py:23-35` is `try/except Exception` +
     `logger.warning_once(…)`, so a server started without it still works, minus the
     acceleration: the log line is *"MindIE Turbo is installed but unable to
     `import mindie_turbo`, skip MindIE Turbo acceleration."*
   - `vllm_ascend/quantization/quantizer.py:39-44` turns the same `ImportError`
     into `NotImplementedError: There is no available ascend quantizer.`, so a
     quantised run fails with *that* text and never shows the ABI line (precedent
     #633; #780 has the same package failing deeper, in
     `mindie_turbo/adaptor/vllm/weight_utils.py`).
   The **other** failure mode of the same wheel needs no ABI at all: when the
   import succeeds and the plugin replaces ops, the first compiled forward dies
   with `Unsupported: Graph break due to unsupported builtin
   mindie_turbo._ops.PyCapsule.<op>` — that shape is
   `mindie-turbo-graph-break.md` (v0.9.x: `extras_require={}`, the worker import
   logs an INFO line instead of the `warning_once` above, and the compatibility
   matrix has no MindIE Turbo entry).
7. Ownership answer for the report: `mindie_turbo` is an optional extra of
   vllm-ascend (`setup.py:371`, `extras_require={"mindie_turbo": ["mindie-turbo==2.0rc1"]}`)
   loaded with `importlib.import_module` — the C++ ABI pairing belongs to that wheel
   plus the installed NNAL, not to the vllm-ascend tree. Say so, and give the two
   consumers' behaviour so the reporter can decide whether to fix or ignore it.

## Scope, limits, and what would change the answer

- The assertion is about the *reference* in the failing `.so`: `Ss` proves it was
  compiled with `_GLIBCXX_USE_CXX11_ABI=0`. Which ATB file *exports* the matching
  definition is the reporter's `nm` line (step 2) — the fix is reporter-confirmed,
  the per-file mapping was never printed in the thread.
- ATB/NNAL source is not in the snapshot for this case (`cann/ascend-transformer-boost`
  is listed `unavailable` in `SOURCES.json`), so the `cxx_abi_0`/`cxx_abi_1`
  directory semantics come from the env dump, the mangling and the confirmed fix,
  not from reading NNAL.
- Reverse-direction precedents, same class, different object:
  `torch_npu/lib/libop_plugin_atb.so: undefined symbol: _ZN3atb15CreateOperationINS_5infer14GroupTopkParamEEE…`
  (#750, #117) and `libop_plugin_atb.so: undefined symbol: _ZN3atb15CreateOperation…`
  (#750 again on MindSpore). Use them to show the class is ATB-build pairing rather
  than a driver fault, not as a root cause for this report.
- If the same wheel fails on a host whose `ATB_HOME_PATH` already points at
  `cxx_abi_0`, the remaining owner is the NNAL **version** the wheel was built
  against: ask for `pip show mindie-turbo`, `ls -l <ASCEND_HOME>/nnal/atb/` and
  `cat <ASCEND_HOME>/nnal/atb/latest/version.info`.
