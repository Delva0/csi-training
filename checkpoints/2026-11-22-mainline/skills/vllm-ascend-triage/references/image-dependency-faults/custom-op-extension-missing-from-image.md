# `Failed to import vllm_ascend_C` — extension absent, or present and refused

## Signature

`vllm serve` starts — or, in the unresolved-symbol lane of Step 0, never gets past
memory profiling — and the *first use* of `torch.ops._C_ascend.<op>` raises an
`AttributeError` naming the extension's op namespace. The startup log carries one
warning that is easy to read as cosmetic:

```text
WARNING ... [camem.py:67] Failed to import vllm_ascend_C: No module named
'vllm_ascend.vllm_ascend_C'. Sleep mode will be disabled.

File ".../vllm_ascend/_310p/ops/fla/gdn_310.py", line 282, in _forward_core
    mixed_qkv_non_spec = torch.ops._C_ascend.npu_causal_conv1d_310(...)
AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'npu_causal_conv1d_310'
```

The reported file is often *present* in the image; only the compiled `.so` is gone.
The log prints the logger's **basename** (`[camem.py:67]`), so grep the message
string — at v0.23.0.post1 the file is `vllm_ascend/device_allocator/camem.py:67`.

## Step 0 — read the tail of the warning: absent, or present and refused

What follows `vllm_ascend_C:` picks the lane *and* the owner, before any source read:

| tail | what it means | where it goes |
|---|---|---|
| `No module named 'vllm_ascend.vllm_ascend_C'` | no compiled `.so` reached the image | Steps 1-4 below (csrc-cache and `develop`-install-prefix lanes) |
| `/…/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so: undefined symbol: _ZN…` | the `.so` exists; `dlopen` loaded it and refused it because one symbol it references was left out of the build **for this SoC** | *Lane B* at the end of this page |
| `undefined symbol: …ERKSs` / `…NSt7__cxx11…` in **another** package's `.so` (`mindie_turbo/_ops…so`, a connector) | not this extension, not this page | `abi-undefined-symbol.md` |

Either lane makes the later `AttributeError`'s op name arbitrary — one module
registers the whole `_C_ascend` namespace, so *nothing* was registered and the name
that happens to be called first is the one printed. The corpus family is
`weak_ref_tensor` (#3342, the project's own "custom kernels fail to compile without
raising any errors", and #3090), `npu_causal_conv1d_310` (#17713) and
`moe_gating_top_k` (#6527): do not read the name as "this op is unsupported at this
pin", and do not look for a precedent carrying that op.

## Step 1 — is the extension load-bearing here? One grep decides

```bash
rg -c 'torch\.ops\._C_ascend' vllm_ascend/_310p/          # 310P-only call sites
rg -n 'torch\.ops\._C_ascend' vllm_ascend/_310p/ops/fla/gdn_310.py
```

Verified at vllm-ascend v0.23.0.post1 (`1cdb8c4db6e5`): `_310p/ops/fla/gdn_310.py`
holds 4 call sites (`:133` `npu_recurrent_gated_delta_rule_310`; `:265`, `:282`,
`:296` `npu_causal_conv1d_310`) and `_310p/ops/fla/chunk_gated_delta_rule.py` 2
(`:552`, `:567`), with **no** `enable_custom_op()` guard,
and registration lives only in the extension (`csrc/torch_binding.cpp:2216-2227`).
So on **310P with a GDN / Qwen3.5 / Qwen3-Next family checkpoint** the missing
extension is fatal at the first prefill — this page.

On A2/910B and on any path that reaches `torch_npu`/aclnn instead, the *same*
warning is the documented red herring (`../soc-platform-paths/kernel-package-vs-card.md`
§*Red herring*). Do not quote this page without the grep. A **VL/OCR model on 310P**
is not automatically in the fatal set either: check whether its forward reaches
`_310p/ops/fla/` (`../soc-platform-paths/310p-cpu-fallback-op-fault.md`).

## Step 2 — why the build never fails: two silent lanes at the pin

| Fact | Location (v0.23.0.post1) |
|---|---|
| `ext_modules = []` unless `COMPILE_CUSTOM_KERNELS`, and `build_extensions()` returns immediately when it is falsy — `pip install -e .` exits 0 with no artefact and **nothing checks that a `.so` exists** | `setup.py:462-464`, `:368-370` |
| CMake installs into `$ROOT/vllm_ascend` **only** for the legacy `develop` command; otherwise into `$ROOT/<build_lib>` (and the `.so` copy-back loop is behind the same test) | `setup.py:311-315`, `:417` |
| The caller of both import sites swallows `ImportError`: one disables *all* custom ops, the other only warns about sleep mode | `vllm_ascend/utils.py:409-465`, `vllm_ascend/device_allocator/camem.py:65-72` |
| Image builds set `COMPILE_CUSTOM_KERNELS=${{ cache-hit == 'true' && '0' \|\| '1' }}` and restore `vllm_ascend/_cann_ops_custom`, `vllm_ascend/*.so`, `vllm_ascend/lib`, `vllm_ascend/include` before `docker/build-push-action` | `.github/workflows/_schedule_image_build.yaml:225-234`, `:250` |
| The image `COPY . /vllm-workspace/vllm-ascend/` then `pip install -e` — so a cache **hit** compiles nothing and its only extension is the restored file carried by `COPY` | `Dockerfile.310p:57`, `:62` |
| The cache is written by a separate workflow that runs `uv pip install -e .` and saves the same four paths, keyed by `IMAGE_TAG` (the restore keys on the Dockerfile's `FROM` tag) | `.github/workflows/push_build_csrc_cache.yaml:177-186`, `:189-209` |
| The restore is the **same job for nightly and release**: the release workflow calls `_schedule_image_build.yaml`, and the key is `vllm-ascend-build-v1-<ARCH>-<BASE_IMAGE_TAG>-<CSRC_HASH>` — so "nightly restores the cache, release does not" is not a pipeline difference at this pin, and the failing run's own `cache-hit` value is the evidence | `.github/workflows/_schedule_image_build.yaml:222-233`, `schedule_image_build_and_push.yaml:98,145` |

Two mechanisms follow, and the `ls` below separates them:

1. **Cache hit without a usable `.so` in the archive** (save/restore path or key
   asymmetry, save race): the image publishes with no extension at all.
2. **Cache miss whose compile lands outside the source tree** (the `develop`-only
   install prefix): the build succeeded, the artefact is under `build/lib…/vllm_ascend/`.

**The Dockerfile's `ARG` is not inert, and that is what keeps mechanism 1 live.**
`Dockerfile.310p:52` declares `ARG COMPILE_CUSTOM_KERNELS=1` and the
`RUN … python3 -m pip install -e` at `:62` never names the variable again — but an
`ARG` declared in a stage **is** in the environment of every later `RUN` of that stage,
which is exactly how the workflow's `build-args: COMPILE_CUSTOM_KERNELS=…` line reaches
`envs.COMPILE_CUSTOM_KERNELS` — the same file proves the pattern two lines up, where
`ARG VLLM_COMMIT=""` (`:37`) is read straight inside `RUN … if [ -n "$VLLM_COMMIT" ]`
(`:38`) with no `ENV` anywhere, and the workflow passes it only as a `build-arg`
(`_schedule_image_build.yaml:251`). Recorded #17713's replay read that missing
`ENV`/inline assignment as "the build-arg cannot suppress the compile at this pin",
demoted mechanism 1 and built a "what changed in the 310P recipe between 0.23 and 0.26"
story on top of the demotion; the failing build's `cache-csrc` output decides mechanism
1 directly, with no Docker-semantics reasoning at all.

## Step 3 — the two commands that decide it

```bash
docker run --rm --entrypoint bash quay.io/ascend/vllm-ascend:<tag> -c \
  'ls -l /vllm-workspace/vllm-ascend/vllm_ascend/*.so /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom 2>&1; ls -l /vllm-workspace/vllm-ascend/build 2>&1 | head'
```

plus the release build job's `cache-csrc` step output (`cache-hit`) and its resolved
`build-args` line. A `.so` under `build/` only is mechanism 2; no `.so` anywhere is
mechanism 1; and the reporter's own control — rebuilding *inside the same container*
with the same `pip install -e . --no-deps --no-build-isolation` — refutes mechanism 2
when it produces an importable extension, so ask for it before ranking. The
`_cann_ops_custom` listing rides along because the save/restore steps name both paths
(`:230-231`): the aclnn directory present with no `.so` anywhere is a **partial archive**
of mechanism 1, not a third cause — and on a tag newer than the snapshot's pin it is
also the first place a *separate prebuilt ops package* would show (recorded #17713 cites
#17390 "installs a prebuilt ops package into the 310P image" for that possibility; this
tree cannot show it, so say so instead of ranking it above the `ls`).

## Step 4 — the fix belongs in the pipeline, keyed on the artefact

- **Assert the artefact, not the flag.** A guard that fails the install when
  `COMPILE_CUSTOM_KERNELS=1` but no `.so` appeared passes on exactly the broken lane,
  because the cache-hit lane runs with `0`. Require
  `glob(vllm_ascend/vllm_ascend_C*.so)` to be non-empty **regardless** of the flag.
- **Assert after the restore, before `docker/build-push-action`**, so an empty cache
  hit cannot reach `COPY .`.
- **Do not guard the 310P call site.** At the pin there is no non-`_C_ascend` path in
  `_310p/ops/fla/`, so a `getattr`/`try` around `gdn_310.py:282` starts the server and
  leaves GDN numerics unvalidated — the same rule as
  `../upstream-downstream-drift/patched-forward-api-drift.md` rule 6.
- Acceptance is one real request on the rebuilt image with an oracle (embedding
  similarities against the model card, or a golden completion), not a clean start.

# Lane B — the `.so` is present and `dlopen` refuses it: a SoC source-list gate

Recorded #6527: image `v0.14.0rc1-310p` (arm64), vllm-ascend v0.14.0rc1
(`52d4acfa51fb`), Qwen3-Coder-30B-A3B on 4×310P1, TP4. All four ranks finish the
load (`Loading model weights took 14.2651 GB`), then the worker dies **inside
memory profiling** — so a report of this lane never reaches `/v1/models` or a
request:

```text
WARNING [camem.py:66] Failed to import vllm_ascend_C:/vllm-workspace/vllm-ascend/
  vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so: undefined symbol:
  _ZN9pp_matmul17GetPpMatmulTilingERKNS_10MatMulInfoERKNS_12HardwareInfoERjRNS_18PpMatmulTilingDataE.
  Sleep mode will be disabled.
...
(Worker_TP0 pid=273) ERROR [multiproc_executor.py:822]   File "…/vllm_ascend/worker/worker.py", line 262, in determine_available_memory
    self.model_runner.profile_run()
...
AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'moe_gating_top_k'
```

Here that last line sat in the reporter's own Chinese summary (`后面是类似的报错，中间夹带
AttributeError: …`), with the stack above it being the `profile_run` chain — a terminal
symptom quoted in prose is still the terminal symptom, and it is the only line in the
paste that names the extension.

The chain is all build configuration at the pin, and its last row means "compile the
missing file in" is not the cure:

| fact | location (v0.14.0rc1) |
|---|---|
| the module globs `csrc/*.cpp` unconditionally, and `torch_binding.cpp` is in it | `CMakeLists.txt:116` `pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})` |
| that file includes the header and registers the host op with no SoC guard | `csrc/torch_binding.cpp:31`, `:697-720`, `:1400-1401` |
| the header's own tiling body calls the symbol | `csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.h:100` → `pp_matmul::GetPpMatmulTiling(...)` |
| the symbol's **only** definition is the TU the 310P branch drops (the `else` branch adds it back) | `csrc/batch_matmul_transpose/op_host/tiling/tiling_data.cpp:135`; excluded by `CMakeLists.txt:85-93` |
| the official 310P image selects that branch | `Dockerfile.310p:21,25` `ARG SOC_VERSION="ascend310p1"` → `ENV`; `setup.py:297-298` passes `-DSOC_VERSION=` |
| the *kernel* side was excluded correctly, which is why nobody noticed the host side | `CMakeLists.txt:65-78` `VLLM_ASCEND_CUSTOM_OP_EXCLUDE` (bgmv/mla/batch_matmul kernels) |
| the op that dies dispatches through **aclnn**, and 310P builds no custom aclnn ops at all | `csrc/torch_binding.cpp:1298` `EXEC_NPU_CMD(aclnnMoeGatingTopK, …)`; `csrc/build_aclnn.sh:6-11` `exit 0` for `^ascend310` ("currently, no custom aclnn ops for ASCEND310 series") |

Consequences, in the order they decide the answer:

1. **One unresolved symbol empties the namespace.** The extension is a single module,
   so a resolution failure anywhere means `_C_ascend` registers none of its ops. Why
   the build did not fail: `-shared` without `--no-undefined` links fine and the
   loader is the first thing to complain.
2. **Nothing checks.** `device_allocator/camem.py:55-72` catches the `ImportError`
   and only disables sleep mode; `utils.py:243-266 enable_custom_op()` records
   `_CUSTOM_OP_ENABLED = False` with its own warning; and the consumer that dies
   reads neither — `ops/fused_moe/experts_selector.py:102-129
   check_npu_moe_gating_top_k` gates on `scoring_func`, `topk_group` and shape
   arithmetic only, so the `_native_select_experts` fallback at `:90` is unreachable
   while the namespace is empty (the presence gate is what is missing, not a flag).
3. **On 310P this op has no kernel at this pin even after a repaired build.** The
   aclnn custom-op package that carries `moe_gating_top_k` is not built for
   `^ascend310` (`build_aclnn.sh:6-11`), which is what `release_notes.md:9` means by
   "only basic dense and vl models are supported with eager mode" (#5776) —
   `../soc-platform-paths/atlas-300i-310p.md` states the same fact from the doc side.
   So the structural fix is to exclude the host include and registration on 310P too
   (build hygiene: the image must import), **and** to route the consumer to its
   native implementation; adding `tiling_data.cpp` back only moves the failure to the
   kernel lookup.

## Lane B steps

1. Confirm on the reporter's box, one command — `U` is the proof, `T` means the
   symbol is there and the story is wrong:
   ```bash
   P=$(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))")
   nm -DC "$P"/vllm_ascend_C*.so | grep -Ei '<symbol fragment>|<op name>'
   python -c "import vllm_ascend.vllm_ascend_C"   # full failure text; dlopen stops at the first symbol
   ```
2. Name the owner with two greps at the reporter's ref, not a source walk:
   `rg -n '<symbol>' <root>/csrc/` (only definition site?) and
   `rg -n 'ascend310p|SOC_VERSION' <root>/CMakeLists.txt <root>/Dockerfile*` (which
   source list that SoC takes). If the definition is inside the compiled list, the
   lane is ABI/third-party instead — Step 0's third row.
3. Ladder, cheapest first: a newer `-310p` tag re-checked with the same `nm` (a
   one-commit snapshot cannot show which tag fixed it — say so); a presence gate at
   the consumer so the native path is reachable (`hasattr(torch.ops._C_ascend,
   "<op>")` → `return False`), with **a real request compared against a known-good
   box as the acceptance test**, never a clean start, because this changes numerics;
   then the build-side exclusion.
4. Expect the second symptom: with the fused selector bypassed the paste may hit
   another unguarded `torch.ops._C_ascend.<op>` in the same model path
   (`ops/fused_moe/moe_mlp.py`, `token_dispatcher.py`, `ops/layernorm.py`,
   `ops/rotary_embedding.py` at this pin) — answer that case by case, and say the
   perf will sit below the fused path.
5. Report the model's own support row beside the verdict when the SoC is 310P: the
   row's `Supported Hardware` column (here `A2/A3`, `supported_models.md:23`) is a
   coverage statement, not the cause — the `.so` failure would fire on any model on
   that tag.

## Scope and recheck

Line numbers are v0.23.0.post1's (`1cdb8c4db6e5`) for Steps 1-4 and v0.14.0rc1's
(`52d4acfa51fb`) for Lane B. On Lane B, 910B/910_93 builds compile `tiling_data.cpp`
and are unaffected, so the unresolved `pp_matmul` symbol is itself the fingerprint of
a 310P build — not a wrong image, a mount problem or a reporter-modified package —
and re-grepping its two sites (the SoC branch in `CMakeLists.txt`, the include and
registration in `csrc/torch_binding.cpp`) at the reporter's tag is the reuse check.

On Steps 1-4 the workflow **names** move between releases (recorded #17713 reported
`_ensure_csrc_cache.yaml` / `_build_csrc_cache.yaml`, which do not exist at that pin —
it has `push_build_csrc_cache.yaml` and the restore step inside
`_schedule_image_build.yaml`), and a reported tag newer than the snapshot cannot be
shown to be fixed or broken by this tree. Re-grep the three files at the reporter's
tag and say which release the evidence came from.

The reporter's own table (which earlier `-310p` tags carry
`vllm_ascend_C.cpython-*.so`) is the strongest artefact in such a report: it is
per-tag evidence from images, and one row per tag is cheaper than any source walk. Read
it as evidence about the **failing** tag, though: the good rows went through the same
restore step (the job is shared), so "this release has the `.so` and that one does not"
does not by itself rank the two mechanisms — reproduce the failing row with the one `ls`
above rather than opening a release-to-release diff the pin cannot answer.
