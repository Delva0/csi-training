# `Get ACL JitCompile default value failed` + `RealPath … data/platform_config`: a FILE missing from the CANN tree

## Scope

The paste (recorded **#6336**: image `vllm-ascend:v0.14.0rc1`, CANN 8.5.0, 8×910B2,
SPOD/K8s, `torch_npu==2.9.0`, vLLM 0.14.1; reporter titled it "path resolution
failure … across container mount points"):

```text
[ERROR] FE(2595,python3):… [platform_info.cpp:970]2595 RealPath:Path
  '/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/../data/platform_config' does not exist.
[ERROR] FE(2595,python3):… [platform_info.cpp:1003]2595 InitializePlatformInfo:File path[] is not valid.
[ERROR] ASCENDCL(…) [op_compiler.cpp:124]2595 QueryPlatformInfoValue:init runtime platform info failed, SocVersion = Ascend910B2
[INFO] GE(…) [error_manager.cc:397]2595 ReportInterErrMessage:report error_message, error_code:EH9999
RuntimeError: Get ACL JitCompile default value failed.
[ERROR] … (PID:2595, Device:0, RankID:-1) ERR00100 PTA call acl api failed
```

Same **phase** as `acl-init-500001-and-soc-version.md` — framework-free ACL/GE
bring-up, no `vllm_ascend` frame, no `Starting to load model` — but a different
**lane**, and the difference changes the answer:

| | `500001` lane (`acl-init-500001-and-soc-version.md`) | this lane |
|---|---|---|
| `SetPrecisionMode … ACL_PRECISION_MODE … 500001` | present | **absent** |
| `PlatformInfoInit … SocVersion <name>` | present | present, and the name matches `npu-smi` |
| failing statement | `aclrtSetCompileOpt` on acquired ACL state | `realpath()` on a path in the toolkit tree |
| lever | container privilege (`--privileged=true`), then pairing | **the file tree**: the directory the `.so` points at |

So do not carry `--privileged`, the driver/firmware pairing or any dtype/precision
flag into this shape.

## Mechanism (source-verified)

Pins for this page: `cann/runtime` `0b16a5e98a71`, `Ascend/pytorch`
`v7.2.0.1-pytorch2.8.0` (`da848bbc39f4`), vllm-ascend `v0.14.0rc1` (`52d4acfa51fb`).
**`cann/runtime` is exported** — `src/platform/platform_info.cpp` is the `FE` file that
prints both of the first two lines — while `cann/ge` (which owns the
`op_compiler.cpp:124` frame) is in `SOURCES.json` `unavailable`. Two reads of the
runtime file settle the whole chain:

```bash
rg -n 'PLATFORM_RELATIVE_PATH|GetSoFilePath|RealPath' \
  /workspace/repos/<issue>/cann/runtime/src/platform/platform_info.cpp
```

| Anchor | Fact |
|---|---|
| `platform_info.cpp:27` | `const std::string PLATFORM_RELATIVE_PATH = "../data/platform_config";` — the exact suffix in the paste |
| `:955-973` `GetSoFilePath()` | `dladdr(&PlatformInfoManager::Instance)` → `dli_fname` → `RealPath(so_path)` → keep the dirname. **The printed base directory is the loaded FE `.so`'s own real directory**, so the message names the CANN tree that owns the failure |
| `:933-953` `RealPath()` | `realpath()`; on any failure prints `Path '%s' does not exist.` — the sentence is not a `stat` result, an `EACCES` or a shadowed path prints it too |
| `:975-1000` `InitializePlatformInfo()` | `RealPath(GetSoFilePath() + PLATFORM_RELATIVE_PATH)`, then `File path[%s] is not valid.` — the paste's two lines, in order; what failed is the **directory** |
| `:1196-1216` `InitRuntimePlatformInfos()` | the same directory **plus** `cfg_file_real_path + "/" + <SoCVersion> + ".ini"` — a different function, reached later; a directory that exists but has no `Ascend910B2.ini` fails *there*, not here |
| `Ascend/pytorch` `torch_npu/csrc/framework/LazyInitAclops.cpp:116-134` | `GetJitCompileMode()`: the **size** query tolerates failure (`ASCEND_LOGW … use PTA default value`, returns `""`), the **value** query is a hard `TORCH_CHECK(ret == ACL_SUCCESS, "Get ACL JitCompile default value failed.")` — the paste's `RuntimeError` |
| `:181-190` `LazyInitAclopsCore()` | order is `SetPrecisionMode()` → `SetHF32DefaultValue()` → … → `GetAndSetDefaultJitCompileByAcl()`: `SetPrecisionMode` passed, which is why there is no `500001` — the absence is a **lane marker**, not a missing line |

`EH9999` / `ERR99999` are the inner-error report wrappers re-printed by vllm-ascend;
neither is a second cause.

## What the reporter's own workaround proves

`ln -s ../data ./data` inside `aarch64-linux/lib64/` creates `lib64/data → ../data`.
The code never resolves `lib64/data`: it resolves `lib64` (a real directory — the
printed path has no symlink resolution) and then `..` to the parent, so the link is
never consulted. Their explanation ("`realpath()` cannot cross a container mount
boundary") is also wrong: POSIX `realpath()` crosses mount points. **The correct place
is `<dir-of-lib64>/data/platform_config`, i.e. `…/aarch64-linux/data/platform_config`
alongside `lib64`, not inside it.**

## Checks, in order (inside the *worker* container, as the job's UID)

1. `ls -ld <printed-base>/../data/platform_config && ls <printed-base>/../data/platform_config | head`
   — with `#6336`'s paste `<printed-base>` is `/usr/local/Ascend/cann-8.5.0/aarch64-linux/lib64/`.
2. `namei -l <that path>` — a permission gap on any component prints the same
   "does not exist" sentence, and in SPOD/K8s the job's UID is not the image's user.
3. `ls <that path>/<SoC>.ini` (`Ascend910B2.ini` for 910B2) — the *next* call site
   (`:1213`) needs it even after the directory is fixed.
4. **Is the tree partial?** A host install mounted piecewise (`lib64`, `opp`, `tools`,
   `compiler`) leaves `data/` behind, and a `volumeMount`/`subPath` **on a parent path
   shadows** whatever the image had there. One call decides:
   `docker run --rm <image> ls /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/data`
   vs the same path in the running pod.
5. Framework-free probe in the same container, no vLLM:
   `python -c "import torch, torch_npu; torch.npu.set_device(0)"` — the same two lines
   end the vllm-ascend part of the case.

## Action ladder (cheapest first)

1. The five checks above; report which of the three facts fails (directory absent /
   unreadable / shadowed).
2. **Probe that skips the failing query, not a fix**: `torch_npu.npu.set_compile_mode(jit_compile=False)`
   sets option `jitCompile="disable"` (`torch_npu/npu/npu_config.py:88-92`), which makes
   `IsJitCompileModeSetted()` true (`LazyInitAclops.cpp:107-114`) so
   `GetAndSetDefaultJitCompileByAcl()` returns before calling `AclGetCompileopt`. It
   must run before NPU init (a `sitecustomize.py` on `PYTHONPATH`). Use it only to see
   the *next* error: an op compile without platform info is not a validated numeric
   path, so never ship it as the repair and accept any run only on a real request whose
   output matches a working deployment.
3. **Repair**: make `<printed-base>/../data/platform_config` exist *and* be readable in
   the container — mount/copy the complete CANN tree (never split `lib64` from `data`),
   or `ln -s /real/complete/cann/aarch64-linux/data <printed-base>/../data` with a
   target that is itself mounted; then check the `<SoC>.ini`. Remove the inert
   `lib64/data` link.
4. **Preferred**: drop the hand-assembled host-CANN mount and use the image's own
   toolkit with `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (as the image does,
   `Dockerfile:61`), so `.so`, `data/`, `opp/` come from one tree. #6336's env block
   shows the tell of a hand-assembled env: `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0`
   while `LD_LIBRARY_PATH` also carries `ascend-toolkit/latest/lib64`.

## Do not

- **Do not walk `cann/ge`** (`op_compiler.cpp` is `unavailable` — the caller side is
  where the answer is) or `Ascend/pytorch` past `LazyInitAclops.cpp`: two files, not a
  session.
- Do not rank `--privileged`, the driver pairing, `--enforce-eager`, dtype or any serve
  flag; and do not read `EH9999`/`ERR99999` as an independent fault.
- Do not accept "the file exists, I checked with `ls`" as a refutation — the paste's
  path is `<lib64>/../data/…`, so the `ls` must be of **that** resolved path, in the
  worker container.

## Limits

- The `op_compiler.cpp:124` frame's own body cannot be read here (`cann/ge`
  `unavailable`); the chain is quoted from its CANN-runtime callee, which is enough
  because the two error lines are printed there.
- `Ascend/pytorch` is exported at the **pytorch 2.8.0** line while #6336 runs
  `torch_npu==2.9.0`, so `LazyInitAclops.cpp` line numbers may shift by a few: quote the
  message string and the function names, and label line numbers as
  version-specific.
- `cann/runtime` is a truncated content export (`1985/4097` files). `platform_info.cpp`
  was present in #6336's snapshot; where a later snapshot lacks
  `src/platform/platform_info.cpp`, say the mechanism is unverified there rather than
  reconstructing it from the paste.
