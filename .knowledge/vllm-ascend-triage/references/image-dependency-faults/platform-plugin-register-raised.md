# `Failed to infer device type` with `plugin ascend loaded.`: the hook raised and vLLM discarded it

Scope: `vllm serve` dies before any worker with `RuntimeError: Failed to infer device type`, and the
log carries **`plugin ascend loaded.`**, **no** `ERROR Failed to load plugin ascend`, **no**
`Platform plugin ascend is activated`, then `No platform detected, vLLM is running on
UnspecifiedPlatform`. The sibling shape — the plugin module itself cannot import — is
`plugin-load-module-missing.md`; there the residue/`sys.path` candidates are the answer. Here the
module imported, so those candidates are out, and the real traceback is not in the log at all: vLLM
threw it away.

Verified at vllm **v0.7.1** (`4f4d427`) + vllm-ascend **v0.7.1rc1** (`f17417f`); recorded #130.

## The chain, from the lines the paste does show

| Step | Anchor |
|---|---|
| entry point `ascend = vllm_ascend:register` | vllm-ascend `setup.py:102` |
| `plugin.load()` imports `vllm_ascend` and, on success, logs `plugin ascend loaded.` (a module import failure would print `Failed to load plugin ascend` instead) | vllm `plugins/__init__.py:39-44` |
| `resolve_current_platform_cls_qualname()` calls that function as `func()` inside `try:` … `except Exception: pass` | vllm `platforms/__init__.py:153-161` |
| not appended to `activated_plugins` ⇒ no OOT platform ⇒ `UnspecifiedPlatform` + `No platform detected …` | `platforms/__init__.py:186-188` (the paste's `:187`) |
| `Platform plugin ascend is activated` is the **success** branch — its absence is the branch marker | `:172-174` |
| `DeviceConfig(device="auto")` then finds `current_platform.device_type` empty and raises | `config.py:1602-1608` (`:1608` in the paste ⇒ v0.7.1; v0.7.3 moved it to `:1660`) |

Read it as: the *entry point* imported (otherwise `Failed to load plugin ascend`), and the
*hook call* raised (otherwise the activation line). Everything below `config.py` is downstream.

## First suspect at this pin, and the one command that prints the traceback

`register()` at v0.7.1rc1 is not a registry lookup: its first statement imports the quantization
config module (`vllm_ascend/__init__.py:19-25`), and that module does a bare `import torch_npu`
(`vllm_ascend/quantization/quant_config.py:22`) plus several upstream vLLM symbols. So the swallowed
exception is most likely an import failure on the reporter's own box — CANN not sourced in the shell
that runs `vllm serve`, `libascendcl.so`/`libtorch_npu.so` off the loader path, a wheel for another
arch — not a device, driver-permission or model fault.

Two facts keep that inference honest:

- `vllm_ascend/platform.py:23-26` guards **its own** `import torch_npu` with
  `try/except ImportError: print("Failed to import torch_npu.")`. That print cannot appear in this
  failure, because `platform.py` is imported only *after* `register()` returns its qualname — its
  absence says nothing about torch_npu.
- The reporter's `torch-npu 2.5.1.dev20250218` is **not** a mismatch at this pin: the pin's
  `docs/source/installation.md:141,171` prescribes exactly that wheel, and
  `docs/source/user_guide/release_notes.md:32` explains the release depends on an unreleased
  torch_npu (manual install for a non-container environment). Do not rank the dev-build version
  string as a cause.

Un-swallow it in one command, in the same shell/container that runs `vllm serve`:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh     # the CANN root on that box
python -c "import vllm_ascend; print(vllm_ascend.register())"
```

Healthy output is `vllm_ascend.platform.NPUPlatform`; anything else prints the traceback the serve
deleted. Follow with `python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"`
and `npu-smi info` in that same shell. `--device npu` is not a fix — expect the same failure a few
lines later, because the *platform object* is still `UnspecifiedPlatform`.

## The pin's own known-issues page names this exact log line — scope it before quoting

`docs/source/user_guide/release_notes.md:33` (v0.7.1rc1 §Known issues) calls logs "like `No platform
deteced, vLLM is running on UnspecifiedPlatform` or `Failed to import from vllm._C with
ModuleNotFoundError(\"No module named 'vllm._C'\")`" harmless, and says they are fixed by vllm PR
#12432 in v0.7.3 — the same PR `register()`'s TODO comment waits for (`vllm_ascend/__init__.py:21-22`).

- It is the maintainers' answer for a **working** serve whose log happens to carry the line. It does
  **not** close a paste that ends in `RuntimeError: Failed to infer device type`: at this pairing that
  raise means the platform never resolved in that process, and the `UnspecifiedPlatform` fallback is
  exactly what `config.py:1608` trips on. Establish whether `/v1/models` ever answered before calling
  the line cosmetic.
- Its second half is the one to **pre-empt**. After the platform activates, the same serve on a
  `VLLM_TARGET_DEVICE=empty` vllm build still prints
  `WARNING _custom_ops.py:19] Failed to import from vllm._C with ModuleNotFoundError("No module named
  'vllm._C'")`. State that in the first answer with this citation, or the reporter files it as a new
  fault: recorded #130, where the original failure cleared ("vllm is working fine now"), the reporter
  then reported the `vllm._C` line, and a maintainer's entire answer was a link to this page.
- The third bullet on that page (`# CPU blocks:` printed where `# NPU blocks:` is meant, vllm PR
  #13378) is cosmetic for the same reason.

## Recorded outcome (#130) and what stays open

vllm-ascend v0.7.1rc1, vllm `0.7.1+empty`, torch `2.5.1+cpu`, torch-npu `2.5.1.dev20250218`, CANN
`8.0.0.beta1`, no log text (one OCR'd screenshot). The reporter's next comment is "vllm is working
fine now" with no change to the build or the version stated, which fits an environment-level cause on
the `torch_npu` import path and does not identify it: keep the import and the `+empty`-vs-checkout
question as hypotheses with the two commands above, not as the verdict. The recorded run reached this
branch from the OCR alone and did lead with the un-swallow command, but its answer never mentioned the
`vllm._C` warning the corrected run would print nor the pin's known-issue bullet — the reporter
filed exactly that line next, and a maintainer closed it with the release-notes link.
