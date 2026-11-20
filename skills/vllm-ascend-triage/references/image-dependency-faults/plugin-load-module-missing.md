# Platform plugin cannot import: metadata present, `No module named 'vllm_ascend'`

Worked for #540 (vllm-ascend v0.7.3rc2 / vllm v0.7.3 (`0.7.3+empty`), one host,
`vllm serve <model> --max-model-len 4096 --port 8000 -tp 4`).

## The shape

`vllm serve` dies before any worker, and the *title* error is the last line of a
chain. In log order:

```
INFO  Available plugins for group vllm.platform_plugins:
INFO  name=ascend, value=vllm_ascend:register
ERROR Failed to load plugin ascend
ERROR   … File ".../importlib/__init__.py", line 126, in import_module
ERROR ModuleNotFoundError: No module named 'vllm_ascend'
INFO  No platform detected, vLLM is running on UnspecifiedPlatform
INFO  Available plugins for group vllm.general_plugins:
INFO  name=ascend_enhanced_model, value=vllm_ascend:register_model
ERROR Failed to load plugin ascend_enhanced_model        (same ModuleNotFoundError)
ERROR engine.py:400] Failed to infer device type
      … DeviceConfig ... config.py:1660 … RuntimeError: Failed to infer device type
```

`Failed to infer device type` is **always downstream**. Verified at vllm v0.7.3
(`ed6e907`):

| Step | Anchor |
|---|---|
| entry points are read from whatever `*.dist-info` is on this interpreter's path, then `plugin.load()` imports the module | `vllm/plugins/__init__.py:26,30,42,44` |
| the import error is caught and logged, the engine keeps going | `vllm/plugins/__init__.py:46` (`logger.exception("Failed to load plugin %s")`) |
| with no activated platform plugin, `current_platform` becomes `UnspecifiedPlatform` | `vllm/platforms/__init__.py:210-212` |
| `DeviceConfig(device="auto")` then has no `device_type` and raises | `vllm/config.py:1654,1660` |

So classification needs three log lines, not the env dump: the platform plugin
failed to **import**, therefore no platform, therefore no device type. `-tp 4`
(`arg_utils.py:421-422`, a valid alias) and the model are irrelevant — nothing
reached a device.

vllm-ascend registers exactly those two names, which is how the paste's
`name=…` lines identify the package that is missing both entry points' module:

- v0.7.3rc1 `setup.py:102-106`, v0.7.3rc2 `setup.py:365-369`:
  `vllm.platform_plugins: ascend = vllm_ascend:register`,
  `vllm.general_plugins: ascend_enhanced_model = vllm_ascend:register_model`.

## The owner question: which interpreter, and what left the metadata behind

Entry-point discovery and `import vllm_ascend` use the **same** `sys.path`, so
"metadata found, module not" is a residue case, never a networking/CANN/model one.
Three candidates, one check each, all run inside the environment that runs
`vllm serve`:

1. **Editable install whose source tree moved or was deleted.** The rc-era docs
   ship no wheel ("**Unreleased yet**, please build from source code") and the
   source route is `pip install -e .` (`docs/source/installation.md`, "Build from
   source code"); with PEP 660 the dist-info survives in site-packages while the
   code stays in the checkout.
   `python -c "import vllm_ascend, sys; print(sys.executable, vllm_ascend.__file__)"`,
   then `pip show -f vllm_ascend` and `cat site-packages/__editable__*vllm_ascend*.pth`
   (or `*.egg-link`) — a `.pth` naming a path that no longer exists is the whole
   fault.
2. **Two interpreters on one host.** Metadata was written by one env's `pip`, the
   serve ran from another. The traceback already names the interpreter:
   #540's is `/root/miniconda3/envs/hzy_dev_vllm/lib/python3.10/site-packages/…`.
   Compare it with `which -a python vllm` and with any other vllm log from the same
   box — in this corpus #541's run (opened a day later) resolves `vllm_ascend`
   from `/usr/local/python3.10/lib/python3.10/site-packages/vllm_ascend/…` with
   upstream vllm imported from a `/workspace/vllm` checkout, i.e. that host
   demonstrably serves traffic from a different interpreter than #540's.
3. **A source-built vllm paired with a vllm_ascend installed for another stack.**
   `0.7.3+empty` is vllm's own marker for `VLLM_TARGET_DEVICE=empty`
   (`vllm/setup.py:506-507`), the docs' source-build instruction, so this env was
   hand-built; re-check the pairing with
   `bash <skill_root>/scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]`
   before adding any code theory.

Answer shape: not a device, CANN, driver or model fault; name the missing module and
the interpreter-level check, ask for the one-line `python -c` output. Cap it — this
is an environment question with a two-command answer.

## Open

#540 has no reporter follow-up in the corpus, so which of the three candidates
actually applied is **unverified**; the chain above (`plugin import → no platform →
no device type`) is source-verified at vllm v0.7.3, the mechanism of the missing
module is not. Do not present the editable-move theory as the cause without the
`python -c`/`.pth` output.
