# Image and dependency faults (before any vllm-ascend code runs)

## Scope

Failures owned by the Ascend/vllm-ascend **image or its build recipe**, not by
vllm-ascend Python code: the interpreter aborts inside a native module the image
installed (`mooncake`, `lmcache`, `torch_npu`, a `libascend*` transport), a
system library or tool the project's own tutorials prescribe is absent, or a
dependency list drifted because the Dockerfile runs a *copied* helper script.
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
- You suspect the image at all: the Dockerfile `cp`s a helper from the repo
  (`Dockerfile.a3:37` copies `tools/mooncake_installer.sh`) and upstream ships
  its own copy of that script. Diff the two dependency lists before blaming the
  component's code; the image builds against the stale copy.

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
