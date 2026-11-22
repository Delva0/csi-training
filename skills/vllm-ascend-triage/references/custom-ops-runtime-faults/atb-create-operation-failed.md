# `RuntimeError: <Op>Operation CreateOperation failed!` — ATB could not *create* the op (not the caller's pack)

## Scope

Engine start (or the first ATB call) dies inside a `torch_npu._npu_*` op with
`RuntimeError: LinearOperation CreateOperation failed!` — also `GroupTopkOperation …`,
`SelfAttentionOperation …`. This is ATB's (`libop_plugin_atb.so` →
`atb::CreateOperation(param, &op)`) **op-construction** refusal: it happens before any
tensor is described, so nothing about shapes, masks, dtypes, KV cache, weights or the
model can own it. Recorded #5170: vllm-ascend `v0.11.0` (`2f1aed98ccdb`), Atlas 300I /
310P3, CANN 8.3.RC2, torch_npu 2.7.1 — the engine loaded weights, profiled memory and
died at `vllm_ascend/worker/worker_v1.py:341`
`torch_npu._npu_matmul_add_fp32(x, weight, c)` inside `_warm_up_atb`.

**Do not confuse it with the `setup failed!` family.** `setup failed!` fires at
`OperationSetup`, i.e. after the caller handed over tensors, and is owned by the call
site's own pack (`setup-failed-call-site.md`). `CreateOperation failed!` is the
create-time twin and is owned by the **ATB build × SoC** — reading the caller's shape
arguments, branch or mask is a wrong turn, and so is the import-time ABI page
(`../image-dependency-faults/abi-undefined-symbol.md`, whose message is
`undefined symbol: _ZN3atb15CreateOperation…` and never reaches Python's op call).

## Where the answer is, and where it is not

- **The message's source is in no export.** `cann/ascend-transformer-boost` is listed
  `unavailable` in every case's `SOURCES.json`, and the `Ascend/pytorch` export carries
  no `op_plugin/atb/` (`rg -ln "matmul_add_fp32" /workspace/repos/<issue>` at the #5170
  view returns **only** the vllm-ascend call site). One grep, then stop — the wording
  is a create-time refusal, so its text cannot discriminate causes anyway.
- **The prior corpus carries the family's resolutions.** #19's FAQ #8 lists
  `RuntimeError: GroupTopkOperation CreateOperation failed.` among the DeepSeek-V3
  issues *fixed by* vllm-ascend `v0.7.3rc1` **with CANN `8.1.RC1.alpha001`** — i.e. the
  installed ATB was too old to create the op, and the fix moved the CANN/NNAL side, not
  the model code. #206 is the same wording on `0.7.1rc1`. #19 FAQ #4 gives the pairing
  rule for the neighbouring failures (`Failed to infer device type`,
  `ImportError: libatb.so: cannot open shared object file` = wrong `torch_npu` or a
  missing/incomplete CANN **NNAL** package); #117 and #750 are the link-time sibling
  (`undefined symbol … CreateOperation<GroupTopkParam>`). So the ranked cause is
  **ATB/NNAL build × SoC**, and an upgrade/re-pairing is a rung — never a code patch.
- **vllm-ascend's own contribution is one unconditional call, and it is worth quoting
  because it means the box cannot start at all.** At v0.11.0:
  `worker_v1.py:332` calls `self._warm_up_atb()` from `compile_or_warm_up_model`
  (after the dummy runs and capture), `:337-341` builds `(2,4) fp16 × (2,4) fp16 →
  (4,4) fp32` and calls `torch_npu._npu_matmul_add_fp32`; ATB is registered at
  `:27,88` (`_register_atb_extensions`). There is **no SoC gate and no env switch** at
  this ref (`rg -n -i atb vllm_ascend/envs.py` empty). The function's own comment says
  it exists so the first `ReshapeAndCache` "may cause performance degradation at
  runtime" — a *performance* warm-up, so short-circuiting it is a start-only
  diagnostic, never the fix.

## Dead levers — check the paste before offering them

- `--enforce-eager`: the recorded paste already carries
  `cudagraph dispatching keys are not initialized. No cudagraph will be used`, so eager
  was the running lane.
- Model/weights/KV (`--max-model-len`, `--gpu-memory-utilization`, `--kv-cache-dtype`):
  the paste shows weight load, `GPU KV cache size: N tokens` and the memory profile all
  succeeding before the failure.
- Any `ATB_*` env var in the dump (`ATB_HOME_PATH`, `ATB_MATMUL_SHUFFLE_K_ENABLE`): they
  select kernels/tiling, not whether `CreateOperation` returns an op; label such a
  probe as an unverified one-launch hypothesis and keep it below the pairing rung.

## The three checks, cheapest first

1. **One-op repro outside vllm-ascend** — reproduces ⇒ ATB build/SoC; passes ⇒ the
   vllm-ascend build (SoC stamp, custom kernels):
   ```bash
   python - <<'PY'
   import torch, torch_npu
   from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
   _register_atb_extensions()
   x = torch.rand((2, 4), dtype=torch.float16).npu()
   w = torch.rand((2, 4), dtype=torch.float16).npu()
   c = torch.rand((4, 4), dtype=torch.float32).npu()
   torch_npu._npu_matmul_add_fp32(x, w, c); print("atb linear ok")
   PY
   ```
2. **ATB identity, both boxes**: `ls /usr/local/Ascend/nnal/atb/`,
   `cat /usr/local/Ascend/nnal/atb/latest/version.info`, plus the dump's
   `ATB_HOME_PATH` and the `cxx_abi_0`/`cxx_abi_1` entries in `LD_LIBRARY_PATH`, read
   against the same two lines in the release's own image.
3. **The SoC stamp** of a source install: `python -c "from vllm_ascend import
   _build_info as b; print(b.__soc_version__)"` and
   `python -c "from vllm_ascend.utils import is_310p; print(is_310p())"`
   (`setup.py:85-102` writes `__soc_version__` from the build's `SOC_VERSION`;
   `utils.py:67-71` matches `ascend310p*`; `setup.py:90-92` refuses a 310 build without
   `COMPILE_CUSTOM_KERNELS=1`).

The pin's own 310P recipe is the control that the combination is *meant* to work and
what an artefact must contain: `Dockerfile.310p:18`
`FROM quay.io/ascend/cann:8.3.rc2-310p-ubuntu22.04-py3.11`, and `:47-53` sources both
`ascend-toolkit/set_env.sh` and `nnal/atb/set_env.sh`, exports
`SOC_VERSION=ASCEND310P3`, then `pip install -e .`. A `/vllm-workspace` source install
that skipped that block has neither the matching ATB nor the stamp.

## Answer shape

- Verdict in the first line: real crash, create-time ATB refusal, owner = the ATB/NNAL
  build × SoC pairing; the engine cannot start at this ref because the warm-up is
  unconditional. Label the pairing *hypothesis pending check 1*, not a finding.
- Rung 1: re-run the identical command on the release's own `-310p` image (base
  CANN+NNAL, `SOC_VERSION`, custom kernels in one artefact) — or, staying on the source
  build, `source /usr/local/Ascend/nnal/atb/set_env.sh && SOC_VERSION=ASCEND310P3
  COMPILE_CUSTOM_KERNELS=1 python -m pip install -e .`.
- Rung 2: the one-op repro (splits cause 1 from cause 2 before any reinstall).
- Rung 3: short-circuit `_warm_up_atb` **as a diagnostic only**, and attach the
  acceptance check (one real request whose output is compared with a working run) —
  a box that starts with a broken SoC stamp still takes every 310P-conditional path
  (`is_310p()`, NZ layout, fp16-only KV) the wrong way.
- Doc rung, quoted not paraphrased: `docs/source/tutorials/single_node_300i.md:4-9` —
  "This Atlas 300I series is currently experimental", "only supports eager mode and the
  float16 data type", known issues `#3316`/`#2795`, "You can use v0.10.0rc1 version
  first"; the recipe line is `:21` `IMAGE=quay.io/ascend/vllm-ascend:v0.10.0rc1-310p`.
  **Those two ids are not in the case's history corpus and a GitHub lookup of them may
  return 404** (recorded #5170 spent a call on both): quote the doc's own line and
  spend nothing else.
- Send back: the repro's output, `nnal/atb` listing + `version.info` on both boxes,
  `_build_info.__soc_version__` / `is_310p()`, the exact serve line, the plog.

Pins: vllm-ascend `v0.11.0` `2f1aed98ccdb` / vllm `v0.11.0` (case #5170, opened
2025-12-18). Re-grep `_warm_up_atb` per ref: later releases may gate it by SoC, and a
snapshot holds one commit, so "is it fixed upstream" is not answerable here.
