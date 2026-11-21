# Triton-Ascend kernel compile failure: the installed wheel/compiler pair

## Scope

The engine starts, the weights load, and then the **first real request** dies
inside a Triton-Ascend kernel compile:

```
triton.compiler.errors.MLIRCompilationError:
///------------------[ERROR][Triton][BEG]------------------
[ConvertLinalgRToBinary] encounters error:
loc("/tmp/tmpXXXX/kernel.ttadapter.mlir":42:20): error: operation's operand is unlinked
subprocess.CalledProcessError: Command '['/usr/local/Ascend/tools/bishengir/bin/bishengir-compile',
  '/tmp/tmpXXXX/kernel.ttadapter.mlir', '--enable-auto-multi-buffer=True', ...]' returned non-zero exit status 1
```

Recorded on a hybrid GDN model (Qwen3-Next-80B-A3B) with the deepest
`vllm_ascend` frames in the FLA chunked-delta-rule kernels
(`vllm_ascend/ops/triton/fla/chunk.py` → `chunk_scaled_dot_kkt.py`), under
`triton/runtime/autotuner.py`: the traceback runs through the autotuner and the
error escapes it into `execute_model`, so the request dies on the first config it
tries — the failure is not confined to one bench candidate.

Owner: the **triton-ascend wheel × BiSheng / AscendNPU-IR compiler pair that was
installed into the image**, not vllm-ascend Python and not the kernel's shapes.
`bishengir-compile` is a separate executable fed the backend's `*.ttadapter.mlir`;
a wheel and a compiler from different builds is exactly how that conversion step
rejects valid IR. Sibling pages cover AscendC/aclnn device faults
(`../custom-ops-runtime-faults/index.md`); this page is the Triton compiler lane.

## Reading the report correctly

- **"Failed to compile custom operators with Triton Ascend" is usually the
  reporter's paraphrase, not a library string.** In the recorded #5750 paste the
  phrase does not appear anywhere (`grep -c "Failed to compile"` = 0); the real
  error is the `MLIRCompilationError` block above. Do not go looking for the
  AscendC extension build (`vllm_ascend_C`, `Failed to register custom ops`):
  nothing in this failure touches it. Same family as SKILL.md rule 11.
- **It is not a startup failure.** The engine initialised, KV cache was allocated,
  then the request entered `execute_model` (`vllm/v1/worker/worker_base.py` →
  `vllm_ascend/worker/worker.py` → `model_runner_v1.py`, line numbers move per
  release; #5750 was on the 0.13.0rc1 line). So "the server came up but any
  inference kills it" is the signature — the opposite of the lane A/B workers in
  `../startup-failures/index.md`.
- Which kernel is chosen (`chunk_scaled_dot_kkt_fwd`, `chunk_gated_delta_rule`,
  `layer_norm_fwd_kernel`, …) dates the release but does not identify the defect: it
  is release code that the release's own pair is exercised with, so a per-kernel MLIR
  rejection says nothing about its shapes. Do not compute tiling or shapes here.

## The install axis (what the pair is supposed to be)

The project prescribes a **versioned pair**, never "latest from a mirror". At
vllm-ascend v0.11.0 (`2f1aed9`), `docs/source/tutorials/multi_npu_qwen3_next.md`:

| Lines | Step |
|---|---|
| `:48-54` | `wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/Ascend-BiSheng-toolkit_aarch64.run`, `--install`, then `source /usr/local/Ascend/8.3.RC2/bisheng_toolkit/set_env.sh` |
| `:60-61` | `triton_ascend-3.2.0.dev20250914-cp311-…whl` — an exact dev build, not a range |
| `:78` | `set_env.sh` is sourced again after the wheel install |

The recorded reporter's container instead logged (from its own attachment):

```
[AscendNPU-IR] ... AscendNPU-IR-1.0.0 install success       # into /usr/local/Ascend/tools
Downloading .../triton_ascend-3.2.0rc4-cp311-…whl
Successfully installed triton-ascend-3.2.0rc4
...
bishengir-compile: returned non-zero exit status 1
```

i.e. a standalone AscendNPU-IR package plus whatever the mirror served, on top of
an image that already carried its own pair. The differencing items to ask for:
`pip show triton-ascend`, `which bishengir-compile`, and whether `set_env.sh`
from the *doc's* toolkit root was sourced (a `RUN source …` in a `Dockerfile`
does not persist into later layers or the container).

## Reporter/maintainer-confirmed resolution (#5750, 2026-01-29)

> 0.13.0rc1 uses the triton-ascend dev package. Installing it via `pip install
> --trusted-host mirrors.xfusion.com -i https://example.invalid/pypi/simple
> triton-ascend` will result in an unusable version of triton-ascend. We
> recommend using our latest version, v0.14.0rc1, which works correctly and does
> not require additional installation of triton-ascend.

Two rules follow, and they outrank any compiler archaeology:

1. **Do not add or replace triton-ascend in these images.** The release that needs
   it ships it; a mirror "latest" makes it unusable. Delete the `pip install` line
   and re-run the untouched image first (cheapest single action).
2. **When the pair is mismatched, move the release rather than fix the compiler.**
   v0.14.0rc1 needs no separate triton-ascend install at the recorded date. If the
   reporter must stay on the broken release, install the pair from *its* pinned
   tutorial verbatim (paths and file names above), including `set_env.sh`.

The mechanism sentence to keep labelled as hypothesis: the log shows the IR
rejection, not which of the two artefacts is older or newer; "mispaired
wheel/compiler" is the reading that fits both artifacts, and the maintainer's
answer settles the *action*, not the ABI detail.

## Answer shape

1. Name the lane in the first sentence: Triton-Ascend kernel compile, engine up,
   first request dies — not an AscendC custom-op build.
2. Cheapest action: fresh container from the image, **no** `pip install
   triton-ascend`, **no** extra BiSheng/AscendNPU-IR `.run`; then serve the same
   command. Print `pip show triton-ascend` / `which bishengir-compile` as evidence.
3. If it still fails: try the newer release whose image carries a usable pair
   (v0.14.0rc1 at the recorded date), which also removes the install step.
4. Missing artefacts to ask for: the `pip list | grep -i triton`, the toolkit root,
   and whether the failing request is the very first one after startup.
5. Close with the version gap when it applies: the pinned snapshot tree may not
   contain the release under test (for #5750 it is v0.11.0, which has no
   `vllm_ascend/ops/triton/` directory at all), so `path:line` for the 0.13 line
   can only come from the reporter's own traceback — say so.

## Related

- Same pair at **startup**, before any kernel runs: an `Error in inspecting model
  architecture '<Arch>'` whose logged tail is an import-time
  `AttributeError: module 'triton.language' has no attribute '<x>'` (a triton is
  installed and it is the wrong one — an absent triton only gives vllm's
  placeholders) → `../startup-failures/model-type-not-recognized.md`, inspection
  section.
- Same pair, performance side: `../performance-throughput/qwen3-next-decode-throughput.md`
  (a hybrid GDN model is slow when the Triton FLA kernels are absent/replaced).
- `../custom-ops-runtime-faults/index.md` for the aclnn/AscendC lane that this
  page is *not*.
- `offline-source-build.md` for a wheel that has to be built, rather than
  installed, on a host without public network.
