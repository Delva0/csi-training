# A published image ships without the compiled `vllm_ascend_C` extension

## Signature

`vllm serve` starts, `/v1/models` answers, and the *first* request dies with an
`AttributeError` naming the extension's op namespace; the startup log carries one
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

Two mechanisms follow, and the `ls` below separates them:

1. **Cache hit without a usable `.so` in the archive** (save/restore path or key
   asymmetry, save race): the image publishes with no extension at all.
2. **Cache miss whose compile lands outside the source tree** (the `develop`-only
   install prefix): the build succeeded, the artefact is under `build/lib…/vllm_ascend/`.

## Step 3 — the two commands that decide it

```bash
docker run --rm --entrypoint bash quay.io/ascend/vllm-ascend:<tag> -c \
  'ls -l /vllm-workspace/vllm-ascend/vllm_ascend/*.so; ls -l /vllm-workspace/vllm-ascend/build 2>&1 | head'
```

plus the release build job's `cache-csrc` step output (`cache-hit`) and its resolved
`build-args` line. A `.so` under `build/` only is mechanism 2; no `.so` anywhere is
mechanism 1; and the reporter's own control — rebuilding *inside the same container*
with the same `pip install -e . --no-deps --no-build-isolation` — refutes mechanism 2
when it produces an importable extension, so ask for it before ranking.

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

## Scope and recheck

Line numbers are v0.23.0.post1's. The workflow **names** move between releases
(recorded #17713 reported `_ensure_csrc_cache.yaml` / `_build_csrc_cache.yaml`, which
do not exist at that pin — it has `push_build_csrc_cache.yaml` and the restore step
inside `_schedule_image_build.yaml`), and a reported tag newer than the snapshot
cannot be shown to be fixed or broken by this tree. Re-grep the three files at the
reporter's tag and say which release the evidence came from.

The reporter's own table (which earlier `-310p` tags carry
`vllm_ascend_C.cpython-*.so`) is the strongest artefact in such a report: it is
per-tag evidence from images, and one row per tag is cheaper than any source walk.
