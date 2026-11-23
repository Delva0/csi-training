# `Fatal Python error: …` — a host-side interpreter abort whose owner the paste omits

## When this page applies

The report's last lines (or its title) carry an interpreter-emitted fatal error and
the process died after a long healthy run — recorded #6863:

```text
Fatal Python error: none_dealloc: deallocating None: bug likely caused by a
refcount error in a C extension
```

Siblings that belong here: `Fatal Python error: Segmentation fault` on a
*single* line with no `!!!!!!! Segfault encountered !!!!!!!` dump
(`worker-segfault-no-python-frames.md` owns that marker), `_Py_NegativeRefcount`,
`GC object already tracked`, `cannot allocate memory in static TLS block`.
The tell is the prefix: **the interpreter, not the engine, printed it**, and it
`abort()`ed afterwards (`SIGABRT`), so there is no `EngineDeadError`, no device code
and no plog entry.

**Not this page**: `!!!!!!! Segfault encountered !!!!!!!` + `Worker proc … died
unexpectedly (exit code: None)` (a signal kill whose dump must be read,
`worker-segfault-no-python-frames.md`); `RuntimeError: cancelled` at
`patch_shm_broadcast.py` (a worker death surfacing as a queue error,
`cancelled-executor-teardown.md`); `terminate called after throwing an instance of
'std::logic_error'` at engine start (`../image-dependency-faults/native-string-abort.md`);
`Illegal instruction (core dumped)` at `import torch`
(`../image-dependency-faults/host-cpu-isa-sigill-at-import.md`).

## Step 0 — the naming evidence is in the block the report cut

CPython prints, right after that line, the running thread's frames
(`Current thread 0x… (most recent call first):`) and closes with an
`Extension modules: …` inventory, plus one `Thread 0x…` section per other thread
that has a Python frame. Between them they name the extension that was executing,
which is the only owner this family has. So the first ask is **everything printed
after the `Fatal Python error:` line, verbatim, from the rank that died** — that
rank's own stderr / worker log, not `plog` (a clean plog is expected and is not
evidence of anything here).

Phrase that ask so one reply can satisfy it: name the file and the extraction
(`grep -n -A 300 'Fatal Python error' <worker stderr file>`, or `tail -300` of it),
say a screenshot of that region is fine, and ask for the **end** of the log rather
than its head. This report's own later follow-ups are the evidence: the reporter
answered with a single image attachment and then re-pasted the launch command the
opening body already contained, so the block still never arrived as text. A reply
like that keeps this ask open — the next turn is one narrower command for the same
file, never another mechanism pass.

If the block carries no Python frames at all, nothing was executing Python at that
moment: treat it as native memory corruption and go straight to a core dump
(`ulimit -c unlimited` before the next run, `gdb "$(command -v python)" core`,
`thread apply all bt`) — the C stack, not the Python one, names the library.

Do not spend the budget deriving the mechanism before this ask: every reading
below ends at the same artefact.

### What the text does and does not prove

- Proves: a C extension handed CPython an object whose reference count is wrong, or
  overwrote a `PyObject` header. vllm-ascend is Python almost end to end, so a
  `vllm_ascend/*.py` frame in the printed stack is where the **call** was, not the
  defect.
- Does not prove: which extension, which request, or anything about the model, the
  checkpoint, the card, HCCL or capacity. Rule out by name before ranking:
  plog clean + no `EL0004`/`207001` + no `hcclRet`/`EI`/`EJ` code ⇒ not the device,
  not a collective, not an OOM (`../runtime-memory/index.md` covers the case where a
  number is actually short).
- It is not the `expandable_segments` crash: that knob's recorded text is
  `unmapHandles … 507899` inside `empty_cache()` at **startup**
  (`../runtime-memory/allocator-expandable-segments-unmap-crash.md`), and
  vllm-ascend writes the value itself (`vllm_ascend/platform.py:404-422` at
  v0.14.0rc1 `52d4acfa51fb`) unless the variable already carries
  `expandable_segments`/max_split_size/gc threshold — so a launch line exporting it
  is the default, not a deviation.

## Step 1 — which interpreter (changes the mechanism, not the owner)

The project's own images are CPython 3.11: `FROM
quay.io/ascend/cann:8.5.0-…-py3.11` in all six Dockerfiles at v0.14.0rc1
(`Dockerfile:18`, `Dockerfile.a3:18`, `Dockerfile.310p:18`, and the three
`*openEuler` variants), and vllm's own floor is `requires-python = ">=3.10,<3.14"`
(`vllm/pyproject.toml:34` at v0.14.1).

- **≤3.11 (the image default):** `None` is an ordinary object, so one extra
  `Py_DECREF` on it reaches `none_dealloc`. The plausible shape is a missing
  `Py_INCREF` on a borrowed `None` in a rarely-taken branch — which is exactly why
  the engine served for weeks first.
- **3.12+:** `None` is immortal, so a plain decref is a no-op and this message
  instead implies a *written* object header — use-after-free/overrun in native code,
  or an extension built against different interpreter headers.

Confirm with `python -V` (a trimmed `collect_env` `<details>` block does not carry
it — #6863's block had only the two vllm lines), but do not spend a turn on it
first: it decides which bug class you look for, not what you ask the reporter.

## Step 2 — owners to name, one check each

| Candidate | Why it is on the list | One check |
|---|---|---|
| `torch_npu` (`Ascend/pytorch`) C++ | every device call and every stream/event wrapper passes through it; many wrappers return `None` | the `Current thread` frames + `Extension modules:` line |
| vllm's `_C` extension | still imported and dispatched on Ascend for some ops | same block |
| the engine IPC serializer `msgspec` | every frontend↔EngineCore message is encoded/decoded through `vllm/v1/serial_utils.py:14`, and the pin is a bare `msgspec` (`vllm/requirements/common.txt:32` at v0.14.1), so a rare message shape is one release upgrade from a refcount bug | same block, or a py-spy/`gdb` C stack |
| `tokenizers` (pyo3) / any other in-process extension | prompt encode path; the `Extension modules:` list is the complete inventory of what could have done it | same block |

Nothing in the tree says which one; that is why the ranking — not a cause — is the
deliverable when the block is missing.

## Step 3 — the instrumentation ladder if it recurs

1. `ulimit -c unlimited` and a `kernel.core_pattern` that keeps the file; reproduce;
   read the C stack. This is the only artefact that names the library without the
   fatal block.
2. Restart with `PYTHONFAULTHANDLER=1` and `PYTHONMALLOC=debug` (3.6+): pymalloc's
   debug mode turns the overrun/duplicate free into a message at the offending
   allocation instead of weeks later; `PYTHONDEVMODE=1` adds the dev-mode warnings.
3. `dmesg -T | tail -50`, and each rank's RSS at the failure minute. Flat RSS keeps
   the rare-path refcount reading; a steady climb moves the case to the growth lane
   (`../runtime-memory/index.md`, `../runtime-memory/hbm-growth-and-oom.md`) — a leak that ends in a
   native abort is still a leak.
4. Ablate the launch env one variable at a time, cheapest first: drop
   `LD_PRELOAD=…/libjemalloc.so.2` (the images append it to `~/.bashrc` —
   `Dockerfile:79`, `.a3:78`, `.310p:63`, `.310p.openEuler:59` — and the tutorials
   list jemalloc as an optional performance item
   (`docs/source/tutorials/Qwen3-Dense.md:163-164`), so it is a lever, never the
   finding), then
   `ATB_OPERATION_EXECUTE_ASYNC=2` off, then `HCCL_OP_EXPANSION_MODE=AIV` off, then
   `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` (writing the key is the only
   way to switch it off; unsetting it re-adds it).
5. Ask what preceded the death: long prompts, client disconnects/timeouts/cancels,
   or a `--max-num-batched-tokens` step that captured a new graph bucket. A rare
   path is a *request shape*, so the last few requests before the abort are evidence.
6. Operationally, keep an auto-restart supervisor on the worker: this is a process
   abort, not a hang, and the supervisor restores service whether or not the owner is
   ever found. Probe liveness with a real request rather than a readiness flag —
   which flag notices a dead worker differs between releases, so do not present that
   as the fix.

## Precedent status

At this snapshot the library and the `history/` corpus hold **no**
`none_dealloc` / `Fatal Python error` precedent — `bash <skill_root>/scripts/
history-titles.sh -b 'none_dealloc|deallocating None|Fatal Python error'` returned
nothing, and the title-only search does not add any (the SKILL.md rule that a
body search is the one that counts applies here and was already run). Recorded
#6863 = vllm 0.14.1 + vllm-ascend v0.14.0rc1 (`52d4acfa51fb`), Atlas 800T A2
single node ×8, Qwen3-Next-80B-A3B TP8, `--max-model-len 131072`,
`--gpu-memory-utilization 0.7`, `--max-num-batched-tokens 81920`,
`{"cudagraph_mode":"FULL_DECODE_ONLY"}`, ~半个月 uptime, plog clean; the recorded run
answered with a mechanism-derived ranking plus the ask list, and nothing since has
changed that shape. Say so in the answer instead of implying a known bug.
