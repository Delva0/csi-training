# Host C++ compile of a torch_npu-touching TU: the wheel/CANN header pair

## Scope

The engine is starting (or the first kernel is being built) and a **host clang/gcc
compile of a generated C++ translation unit** fails inside a *torch_npu* or *CANN*
header — no device is touched, no request is served. The recorded shape
(vllm-ascend #6429, v0.13.0rc2 + torch_npu 2.8.0.post1 + CANN 8.5.0):

```
RuntimeError: Worker failed with error 'Failed to compile /tmp/tmpXXXX/launcher_cxx11abi1.cxx, error:
In file included from <built-in>:1:
In file included from /root/.triton/cache/<hash>/precompiled.h:38:
In file included from .../site-packages/torch_npu/include/torch_npu/csrc/framework/OpCommand.h:5:
In file included from .../torch_npu/csrc/framework/OpParamMaker.h:7:
 .../torch_npu/csrc/framework/interface/AclOpCompileInterface.h:146:55:
   error: use of undeclared identifier 'aclOpExecutor'; did you mean 'aclopExecute'?
  146 | ACL_FUNC_VISIBILITY  aclError AclDestroyAclOpExecutor(aclOpExecutor *executor);
/usr/local/Ascend/cann-8.5.0/include/acl/acl_op.h:267:30: note: 'aclopExecute' declared here
2 errors generated.'
```

Sibling in the same lane: `kernel_macros.h:24:10: fatal error: 'cstdint' file not
found` (#3135) — that one is the host compiler's own headers missing from the
image, and the same reading applies (the compile is the reporter's toolchain, not
vllm-ascend and not the kernel's shapes).

Not this page: a Triton **kernel** that fails to compile (`MLIRCompilationError`,
`bishengir-compile … non-zero exit status`, `*.ttadapter.mlir`) →
`triton-ascend-pair.md`; an AscendC/aclnn custom-op build or device fault →
`../custom-ops-runtime-faults/index.md`; a multi-process `external_launcher`
startup argument failure → `../startup-failures/arg-validation-errors.md`.

## Read the paste correctly before theorising

1. **`launcher_cxx11abi1.cxx` is not a launcher script.** The path is a generated
   TU under the Triton cache, and the `abi0`/`abi1` twin names say it is the
   cxx11-ABI probe the backend compiles before any kernel (the emitter is not in
   the snapshot — point 2); `precompiled.h` is that cache's precompiled header.
   Nothing about `--data-parallel-size`, multiprocessing or the reporter's shell
   script is involved.
2. **The backend that emits it is not in the snapshot.** At the recorded refs
   `triton-lang/triton` v3.5.1 carries `make_launcher` only for the `nvidia` and
   `amd` backends (`third_party/nvidia/backend/driver.py:128`,
   `third_party/amd/backend/driver.py:221`); the ascend backend ships inside the
   `triton-ascend` wheel, so the exact `-I` order of that TU is **not
   verifiable** from the export. Say so instead of deducing it.
3. **The phase is engine start.** The paste's outer stack ends at
   `EngineCore._initialize_kv_caches → determine_available_memory →
   collective_rpc → get_response`, which re-raises the worker's error
   (`vllm/v1/executor/multiproc_executor.py`); `patch_core.py`'s `', please check
   the stack trace above for the root cause` is the downstream wrapper, and the
   `torch/_ops.py … __call__` frames belong to the worker's own stack. So it is
   neither a request-path kernel nor an `aclnn` device fault.
4. **The checkpoint is incidental.** QuaRot/W8A8, the model family and the parallel
   flags appear in the report because that is the serve line; the failing TU names
   them nowhere. Do not price/triage the model.

## The verified header facts (this is the whole diagnosis)

At the recorded refs (`Ascend/pytorch` v7.2.0.1-pytorch2.8.0 `da848bb`,
`cann/runtime` export `0b16a5e`):

| Fact | Where |
|---|---|
| torch_npu **vendors** its own ACL headers, ships them in the wheel, and builds against them | `third_party/acl/inc/acl/`; `setup.py:449-451` packages `third_party/acl/inc/*/*.h`, `setup.py:547-549` adds `third_party/acl/inc` as an include dir |
| the vendored `acl/acl_base.h` typedefs the type | `third_party/acl/inc/acl/acl_base.h:61` `typedef struct aclOpExecutor aclOpExecutor;` |
| `AclOpCompileInterface.h` uses it **without** including `acl_base.h` itself | `torch_npu/csrc/framework/interface/AclOpCompileInterface.h:4` includes only `acl_op_compiler.h`; the use is at `:146` |
| the include that is supposed to supply it comes *one line earlier*, from `OpParamMaker.h` | `OpParamMaker.h:6` `#include "third_party/acl/inc/acl/acl_base.h"` (quoted → resolved through the wheel's include root), then `:7` includes `AclOpCompileInterface.h` |
| **newer CANN's `acl/acl_base.h` uses the same include guard and no longer declares the type** | `cann/runtime/include/external/acl/acl_base.h:11-12` guard `INC_EXTERNAL_ACL_ACL_BASE_H_`, body is three includes, no `aclOpExecutor`; the type survives only in `runtime/src/inc/external/aclnn/acl_meta.h:48` |
| that CANN export has no `acl/acl_op_compiler.h` anywhere (read it as "not in this export": both exports are byte-limited) | `find cann/runtime -name acl_op_compiler.h` → empty |

Both copies therefore define `INC_EXTERNAL_ACL_ACL_BASE_H_`, and only one of them
typedefs `aclOpExecutor`. In **one** translation unit the first copy included wins
and the other body is skipped — which is exactly the observed error, in a TU that
demonstrably had CANN's ACL include root on its path (the compiler's own note
quotes `/usr/local/Ascend/cann-8.5.0/include/acl/acl_op.h:267`).

Two consequences worth stating in the answer:

- The failing *line* is in torch_npu, but the **owner is the wheel × CANN header
  pair**, i.e. the same class as `triton-ascend-pair.md`: an artefact pairing, not
  vllm-ascend Python. Nothing in `vllm_ascend/` can fix it.
- It is **not** "the include is missing": a quoted include that does not resolve is
  `fatal error: 'third_party/acl/inc/acl/acl_base.h' file not found`, and the
  compiler printed exactly two errors and reached `:146`. Say this out loud,
  because "torch_npu's headers are not installed" is the intuitive wrong answer.

## One call

```bash
bash <skill_root>/scripts/acl-header-guard-check.sh \
     --torch-npu-root "$(python -c 'import os,torch_npu;print(os.path.dirname(torch_npu.__file__))')/include" \
     --cann-root /usr/local/Ascend/ascend-toolkit/latest       # or the wheel's own root
```

It prints both `acl_base.h` paths, each include guard, whether each typedefs
`aclOpExecutor`, whether `acl_op_compiler.h` sits beside it, and the collision
verdict (exit 1 = collision). Run it **on the reporter's host**: it decides between
"two colliding copies" and "the wheel ships a copy without the typedef", and those
have different owners. `--sources <repos-root>` runs the same check against the
pinned exports (verified on #6429: collision, exit 1).

If both roots are unavailable, the two facts to ask for are the output of
`ls $(python -c 'import torch_npu,os;print(os.path.dirname(torch_npu.__file__))')/include/third_party/acl/inc/acl/`
and of `grep -n 'aclOpExecutor'` on that `acl_base.h` — plus the same two for the
CANN root in `ASCEND_TOOLKIT_HOME`.

## Which pairing is supposed to hold

- The ref's own pin is `torch-npu==2.8.0` and `triton-ascend==3.2.0`
  (`requirements.txt` at v0.13.0rc2). The report says **`torch_npu==2.8.0.post1`** —
  a different build, so ask which wheel and treat the pair as unconfirmed; the
  vendored ACL header set is baked at that wheel's build time.
- `docs/source/installation.md:14-18` states the contract (CANN `== 8.5.0`,
  torch-npu `== 2.8.0`, NNAL `== 8.5.0`) while the same page's manual steps
  (`:92-104`) still download CANN **8.3.RC2**. The table is the contract; the body
  is stale. Quote the table, and do not use the body to claim the reporter is on
  the wrong CANN.
- No precedent exists for this string in the prior-month corpus (0 hits for
  `aclOpExecutor`, `AclOpCompileInterface`, `launcher_cxx`, `precompiled.h` across
  794 issues). Say the lane is new rather than implying a known issue.

## Answer shape

1. Owner first: a **host C++ compile** of a triton-ascend launcher TU that puts
   CANN's ACL headers and torch_npu's vendored ones in one translation unit;
   engine start, before any request; not QuaRot/W8A8, not the model, not vllm-ascend
   Python.
2. The two-sided evidence (vendored typedef at `acl_base.h:61` + `OpParamMaker.h:6-7`;
   CANN copy with the same guard and no typedef) and the log line that proves CANN's
   root was on the TU's path.
3. The cheapest check: the script above on their two roots; plus the wheel version
   they actually have.
4. Options, cheapest first: use the release image / the wheel the ref pins (rebuilds
   nothing, changes one artefact); point `ASCEND_TOOLKIT_HOME`/`ASCEND_HOME_PATH` at
   one consistent CANN root so only one ACL header set can be reached; if the wheel
   itself ships the new `acl_base.h`, that is a torch_npu packaging defect (vendored
   headers must not reuse CANN's guards) — file it there rather than in vllm-ascend.
5. Gaps: the `-I` composition lives in the installed `triton-ascend` backend and is
   not in the snapshot; the exact `torch_npu` build; whether the reporter's CANN root
   and `ASCEND_TOOLKIT_HOME` are the same tree.

## Stop rule

After the header facts and the wheel version there is nothing left to derive: the
`-I` order is in a wheel the snapshot does not carry, so a page-by-page hunt through
`triton-lang` or `Ascend/pytorch` cannot settle which `#include` fired first. #6429
spent the whole recorded session exactly there (grep for the token in the skill,
then the `precompiled.h` producers, then the CANN header set) and never emitted the
answer, although both header copies and the guard collision were in hand by the
middle of it.

## Scope and recheck

Verified at vllm-ascend v0.13.0rc2 (`a17059a`), `Ascend/pytorch`
v7.2.0.1-pytorch2.8.0 (`da848bb`), `cann/runtime` (`0b16a5e`), `triton-lang/triton`
v3.5.1 (`0add682`). **Both upstream exports are byte-limited content exports**
(`complete_regular_files: false`), so a file missing from them is "not in the
export", not proof about the real tree: the guard and the typedef are present in
both copies (a positive read), while the absence of `acl_op_compiler.h` is weaker
evidence — the reporter-side script settles that on their install. Re-read before
reuse on another release: the guard names and whether CANN still declares
`aclOpExecutor` in `acl_meta.h` only, whether `AclOpCompileInterface.h` gained its
own include of `acl_base.h` (that would end this lane), the `torch-npu` pin in
`requirements.txt`, and the install doc's version table.
