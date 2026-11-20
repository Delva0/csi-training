# Image and dependency faults (before any vllm-ascend code runs)

## Scope

Failures owned by the Ascend/vllm-ascend **image or its build recipe**, not by
vllm-ascend Python code: the interpreter aborts inside a native module the image
installed (`mooncake`, `lmcache`, `torch_npu`, a `libascend*` transport), a
system library or tool the project's own tutorials prescribe is absent, or a
dependency list drifted because the Dockerfile runs a *copied* helper script.
Also the **hand-built** environment that follows the documented install steps
instead of using the image: the image installs and sources components (CANN,
NNAL/ATB, driver) that a manual build must reproduce itself, and a missing
`source` line looks exactly like a missing package.
Inherit the rules in SKILL.md; this topic is the counterpart of
`../startup-failures/index.md` for failures that never reach a worker.

## Routes

- `python -c "import <third-party module>"` (or the first real use of it) dies
  with a glibc allocator message — `corrupted size vs. prev_size`,
  `free(): invalid pointer`, `double free or corruption`,
  `munmap_chunk(): invalid pointer`, `Aborted (core dumped)` — with no Python
  frame and no NPU/collective text → `native-import-abort.md`.
- A documented command or import fails because something is **missing** —
  `cannot open shared object file`, `No module named`, `command not found` — and
  `docs/source/tutorials/*.md` installs it in its own startup block → same page,
  steps 1 and 3: identify the copy that is loaded, then compare the tutorial's
  and the Dockerfile's ingredient lists.
- The missing `.so` is raised while a **KV-connector module is imported** in a
  deployment started by a launcher (K8s/kserve/operator `args:`/`command:`,
  compose `entrypoint`) — `ImportError: ascend_transport.so: cannot open shared
  object file` from `from mooncake.engine import TransferEngine`
  (`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:24`) — and
  the ENV dump or a profile script claims the directory is already on
  `LD_LIBRARY_PATH` → `launch-env-and-library-path.md` first. Reporter-confirmed
  on #7352: the effective environment of the launched `vllm serve` is the launch
  command's, so the fix is the export inside that command, not a rebuild.
- The missing library is a **separate Ascend component with its own
  `set_env.sh`** — `libatb.so` inside
  `Could not load this library: .../torch_npu/lib/libop_plugin_atb.so`, raised
  from `_register_atb_extensions()` at worker init — → `nnal-atb-missing-library.md`.
  The owner is NNAL/ATB, not CANN: each install root has its own `set_env.sh`,
  and sourcing the CANN one does not export the ATB paths (a later comment on
  #10525 reports that test; the pinned docs list the two `source` commands as
  separate steps). Same page carries the reporter-confirmed `CANN: None`
  `collect_env.py` blind spot and the non-fatal `triton_kernels` log noise.
- Engine start (or plugin activation) aborts with a C++ exception and no Python
  frame — `terminate called after throwing an instance of 'std::logic_error'`,
  `what(): basic_string::_S_construct null not valid`, then
  `RuntimeError: Engine core initialization failed … Failed core proc(s): {}` ->
  `native-string-abort.md`: the stock-model discriminator, the three `python -c`
  native probes, and `gdb -ex 'break std::__throw_logic_error'` to name the
  `.so`. The checkpoint's config, processor and architecture are not the owner
  (#8745 v0.13.0rc3 DeepSeek-V4-Flash and #10943 v0.20.2rc1 print the same two
  lines; #10943's reporter confirmed the unmodified checkpoint crashes
  identically and eight model-side changes moved nothing).
- You suspect the image at all: the Dockerfile `cp`s a helper from the repo
  (`Dockerfile.a3:37` copies `tools/mooncake_installer.sh`) and upstream ships
  its own copy of that script. Diff the two dependency lists before blaming the
  component's code; the image builds against the stale copy.

## Pages

- `native-import-abort.md` — the interpreter aborts inside an image-bundled
  native module (glibc allocator text, no Python frame): which copy is loaded,
  allocator interposition, dependency-list correspondence.
- `native-string-abort.md` — `terminate called after throwing … std::logic_error`
  / `basic_string::_S_construct null not valid` at engine start.
- `nnal-atb-missing-library.md` — `libatb.so` / `libop_plugin_atb.so` unsourced
  NNAL roots, the `CANN: None` collector blind spot.
- `launch-env-and-library-path.md` — `ImportError: <name>.so: cannot open shared
  object file` while a connector module is imported in a launcher-managed
  deployment, when the ENV dump says the path is fine: the launch command's
  effective environment, the tutorials' exports, and the image-owns-it boundary.

## Common rules

1. This is an environment question with a short answer. Deliver it from the
   reporter's reproduction plus the image recipe, and act with the cheap set:
   a later release / the prebuilt wheel, rename the shadowing copy, preload the
   allocator. Do not walk the failing module's C++: the exports pin a *different*
   release than the image built (recorded case: `kvcache-ai/Mooncake` v0.3.8
   export vs `MOONCAKE_TAG=v0.3.7.post2` in the Dockerfile), so any ABI,
   `glog`-duplication, RPATH or static-destructor story is unverifiable there and
   delivers nothing the reporter can act on.
2. Find out which file the reporter's `import` actually loads before theorising.
   Ascend images routinely carry a second copy of a Python package under
   `<toolkit>/python/site-packages/`, and the reporter's own `import` line may
   resolve to that one rather than to the package vllm-ascend built.
3. Separate "this published artefact is bad" from "the build recipe is wrong" by
   asking whether a locally rebuilt image fails the same way; #5609 did, which
   moved the cause into the Dockerfile stage.
4. Version-verify the fix claim: name the release the reporter says works, keep
   the workaround for the broken one, and label the mechanism as a hypothesis
   unless a reporter comment states it.
5. When the environment dump contradicts itself — `CANN: None` while the host
   reports an `ascend-toolkit`/`cann-*` root, or a component version missing
   from a list — suspect the *collector* before the install
   (`collect_env.py` reads hard-coded default paths), and ask for the install
   root rather than concluding the component is absent. The same caution applies
   to a *correct-looking* dump: it is one shell's environment, not the launched
   process's (see `launch-env-and-library-path.md`).
