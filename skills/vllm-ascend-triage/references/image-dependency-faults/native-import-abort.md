# Interpreter abort inside an image-bundled native module

Worked case: #5609 (v0.12.0rc1, A2 and A3 images). Symptom, verbatim:

```text
python -c "from mooncake.store import MooncakeDistributedStore"
corrupted size vs. prev_size
Aborted (core dumped)
```

No Python traceback, no NPU op, no collective, nothing from vLLM. The reporter
then rebuilt v0.12.0rc1 locally from the pinned `Dockerfile.a3` and reproduced
the same abort, which makes it the build recipe rather than one bad artefact —
and not A2/A3 hardware.

Reporter-confirmed outcome (later public comments on the same issue):

- Workaround A: rename the CANN-toolkit copy of the package
  (`mv /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake
  .../mooncake.disabled`) and the import stops crashing.
- Workaround B (the one they kept):
  `apt update && apt install libjemalloc2` then
  `echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc`.
- Fix: "vllm-ascend v0.13.0rc1 mooncake can work normally."

## Steps, in order

1. **Which copy is loaded.** `python -c "import mooncake; print(mooncake.__file__)"`,
   then `find / -name 'mooncake*' -name '*.so' 2>/dev/null` and
   `pip show mooncake-transfer-engine` if it is installed. The official image
   builds the package from source (`Dockerfile.a3:21,36-40`: clone
   `MOONCAKE_TAG=v0.3.7.post2`, run `tools/mooncake_installer.sh -y`,
   `cmake .. -DUSE_ASCEND_DIRECT=ON`, `make install`) *while* the CANN toolkit
   publishes its own copy under `python/site-packages`, which the vllm-ascend
   tutorials themselves reference for library paths
   (`docs/source/tutorials/pd_disaggregation_mooncake_multi_node.md:224`,
   `DeepSeek-V3.1.md:400`). Two copies with different builds is the default
   state, not an exotic one.
2. **Allocator interposition** — the decisive cheap action for this symptom class.
   `apt install libjemalloc2` (Ubuntu) or `yum install jemalloc` (openEuler), then
   preload it: `LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2` on
   Ubuntu, `/usr/lib64/libjemalloc.so.2` on openEuler. A glibc
   heap-consistency abort that disappears when one allocator is preloaded
   process-wide is an allocation-ownership mismatch inside the bundled module
   (memory from one allocator freed by another), not a vllm-ascend bug — say so
   with the mechanism labelled as inference and the fix labelled
   reporter-confirmed. The same recipe is already in this repo's tutorials as a
   prerequisite for multi-NPU serving
   (`docs/source/tutorials/Qwen-VL-Dense.md:385-395`, and
   `DeepSeek-V3.1.md:98` as "[Optional] jemalloc ... for better performance"),
   so `docs/source/tutorials/*.md` is where the list of ingredients the image is
   expected to provide actually lives.
3. **Dependency-list correspondence.** The Dockerfile builds with the repo's
   *copy* of upstream's installer, and the copy drifts: Mooncake v0.3.8
   `dependencies.sh:123-124` installs `liburing-dev` **and** `libjemalloc-dev`,
   while v0.12.0rc1's `tools/mooncake_installer.sh` installs neither. So the
   image has no jemalloc at all, which is consistent with step 2 and is
   checkable on the reporter's box with `dpkg -l | grep -i jemalloc`. Diff the two
   lists rather than reading the component's CMake. Note the honest limit:
   Mooncake's own `STORE_USE_JEMALLOC` is `OFF` by default and only affects the
   `mooncake_master` target (`CMakeLists.txt:43`,
   `mooncake-store/src/CMakeLists.txt:73-78`), so the missing `-dev` package is
   not itself a build defect — the *runtime* library is what the fix installs.
4. **Split load-time from teardown, and pin the module** (cheap, one call each):
   `python -X faulthandler -c "import mooncake.store"`;
   `python -c "import mooncake.store, os; os._exit(0)"` — a clean exit there means
   the corruption happens in static destructors after interpreter finalisation,
   which changes what the reporter should report but not the two actions above;
   `gdb --batch -ex run -ex bt --args python -c "..."`; `ldd` / `LD_DEBUG=libs`
   on the imported `store*.so` to see which libraries won. Ask for that output
   instead of guessing.
5. **Version correspondence.** Answer with the release that works (here
   v0.13.0rc1) and keep the workaround for the broken one, since the reporter is
   often stuck on the broken image. This is also the one place to mention that
   `mooncake.store` is imported lazily inside `try/except ImportError`
   (`vllm_ascend/distributed/kvpool/backend/mooncake_backend.py:24`), so a heap
   abort kills the process instead of raising and that handler cannot explain
   the missing traceback.

## Do not spend the session on these (each was falsified in the recorded triage)

- **Python-headers/ABI mismatch** ("built against 3.10 headers, imported by
  3.11"). The extension suffix would then be `cpython-310` and the import would
  raise `ImportError`, not abort. Discarded in one step.
- **Two `glog` copies / duplicate flag registration.** That aborts with a glog
  `CHECK` message, not a malloc-consistency message. Reading
  `mooncake-common/FindGLOG.cmake` and the pybind module's link list to chase it
  buys nothing.
- **RPATH / `LD_LIBRARY_PATH` archaeology before step 1.** `$ORIGIN` vs
  `RUNPATH` ordering only matters once you know which package file is loaded.
- **`PYBIND11_MODULE` bodies.** Class registrations allocate nothing; an abort at
  import comes from static initialisers or from a library the module pulls in.

A recorded triage of this case spent both output budgets on exactly those four
lines of inquiry and delivered no answer, *after* it had already surfaced the
`libjemalloc-dev` omission and ranked it low. Cap the source work at about six
tool calls: steps 1-3 are the whole answer.

## Answer shape for this class

- Direct answer: yes, the crash is real, it is inside the native module the image
  installed, and it happens before any NPU/vLLM code — so mooncake is unusable on
  that image, not misconfigured.
- Then, cheapest first: the later release that works; rename the toolkit copy
  (one command, reversible); `libjemalloc2` + `LD_PRELOAD` and the exact paths for
  both distro families.
- What to send back if it still aborts: `mooncake.__file__`, the `find` output,
  `dpkg -l | grep -i jemalloc`, whether `os._exit(0)` also aborts, the image tag
  and whether it was rebuilt from the Dockerfile.
- Scope: image tag / release; state that the export pins a different component
  release than the image built, and that the allocator mechanism is inference on
  top of a reporter-confirmed fix.
