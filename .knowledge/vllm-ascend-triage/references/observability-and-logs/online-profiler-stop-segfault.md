# `/stop_profile` kills the process: a `torch_npu` patch-release lane, not the engine path

Scope: profiling was enabled (`--profiler-config '{"profiler":"torch",
"torch_profiler_dir":…}'`) and `POST /start_profile` / `POST /stop_profile` takes the
engine down **without a Python exception**: `!!!!!!! Segfault encountered !!!!!!!`, a
`Worker proc VllmWorker-<n> died unexpectedly (exit code: None)` line, and on the
EngineCore side `[…] Invocation of profile method failed` raised out of
`vllm/v1/engine/core.py` `invoke_utility_method`. Recorded #15983 (vllm-ascend
`v0.26.0rc1` `f2f74a16`, vllm `v0.26.0` `568afb3a`, `torch_npu==2.10.0.post4`);
precedents **#14860** (main-nightly 2026-08-24, same dump) and **#14974**
(vllm `0.23.0` + `torch_npu 2.10.0.post4`). Not this page: the endpoint answering
`404` / `500 Profiling is not enabled`
(`online-profiling-endpoints.md`), or a collection that finishes and then has no
`kernel_details.csv` (`../performance-throughput/profiler-no-kernel-details-graph-mtp.md`).

## Rule 1 — a cross-patch control in the body is the answer, not a hint

When the body already says *works on `torch_npu==A`, fails only on `B`* (recorded
#15983: "在 `2.10.0.post6`/`2.10.0.post2` 中正常运行，仅在 `2.10.0.post4` 中出现同样错误"),
that is the strongest control the report will ever carry: one variable, same
command, same card. The deliverable is rung 1 = **change the pta patch level**, the
reason = the reporter's own comparison, and the mechanism stays open. Do not derive
the stop-path defect from pta source to earn the verdict — the wheel is not in the
snapshot at that granularity (Rule 7), and the issue's own later public reply
(2026-09-17) settled it the same way: "likely related to a regression in the
torch_npu profiler backend rather than the normal vLLM execution path … use a
previous torch_npu version as a workaround … track this as a compatibility issue
with torch_npu 2.10.0.post4 and check whether a workaround is needed in the vLLM
profile RPC path."

## Rule 2 — read the release's own pin before advising the patch change

A patch-level answer is only useful if it names the pin the reporter installed from:

| Site (v0.26.0rc1) | Fact |
|---|---|
| `requirements.txt:17` | `torch-npu==2.10.0.post4` |
| `pyproject.toml:21` | `"torch-npu==2.10.0.post4"` |

So an install that follows the release table **is** the failing patch: the finding is
"the release pins the broken `.post4`", not "your env drifted". One grep
(`rg -n 'torch-npu' requirements.txt pyproject.toml Dockerfile*`) separates the three
cases — pinned-and-broken, image-overridden, hand-installed — exactly as the
version-contract rule in `SKILL.md` rule 5 requires for any dependency line.

## Rule 3 — the chain proves vllm-ascend is only a forwarder

Quote these instead of walking the profiler sources:

- `vllm/v1/engine/core.py:761-762` — `EngineCore.profile()` →
  `self.model_executor.profile(is_start, profile_prefix)`; the RPC raises in
  `multiproc_executor` and surfaces as `Invocation of profile method failed`.
- `vllm_ascend/worker/worker.py:1034-1062` — `is_start` builds
  `TorchNPUProfilerWrapper` and calls `start()`, otherwise `self.profiler.stop()`
  (`:1062`). With no start in the same session it only logs `Profiler was not
  started, nothing to stop.` — so ask whether `/start_profile` was actually issued
  before `/stop_profile`; a bare stop cannot reach pta.
- `vllm_ascend/profiler/torch_npu_profiler.py:50-72` — the wrapper only chooses
  pta's options: `ExportType.Text`, `data_simplification=True`, `with_stack=False`,
  `on_trace_ready=tensorboard_trace_handler(...)`. The stop call therefore drives
  pta's finalize/parse/export, i.e. the crash site is inside the wheel.

## Rule 4 — `Incorrect schedule: Stop profiler while current state is RECORD` is the normal exit action

That warning is hard-coded in pta
(`torch_npu/profiler/_profiler_action_controller.py:78-82`): the
`(RECORD, None)` exit action prints it and then runs `stop_trace`,
`finalize_trace`, `_trace_ready`. Every run that stops while recording prints it —
#14860 shows it on four workers before the dump, and a *successful* export prints it
too. So it dates the stop, it does not name a cause, and "stop while RECORD" is not
an ablation to propose.

## Rule 5 — the dump says nothing; the paste's dotted lines do

The `Segfault encountered` block in #15983's screenshot OCR is CPython's own free
path (`_PyInterpreterState_GET` → `_PyObject_Free` → `cfunction_call` → … →
`Py_BytesMain`), frame-for-frame the block in #14860: no `vllm_ascend/*.py` frame, no
op name, no thread id. Take the mechanism from the paste's *other* lines — the
`vllm/v1/engine/core.py` / `multiproc_executor.py` sentences — and note that the OCR
is lossy (`core.py:IS48` for `core.py:1548`), so grep it for dotted `path:line`
tokens rather than re-reading it.

## Rule 6 — the other lane: the *frontend* profiler on a non-owner thread

`vllm/v1/engine/async_llm.py:908,914` runs `asyncio.to_thread(self.profiler.start)` /
`(self.profiler.stop)`. That `self.profiler` is the API server's own profiler object —
a different object from the worker's `NPUWorker.profiler` — and it is the path #14974
describes, with pta's thread-affinity sentence
`External init callback must run in same thread as registerClient` in the log (pta
exposes `_enable_profiler_in_child_thread` / `_disable_profiler_in_child_thread`,
`torch_npu/csrc/profiler/init.cpp:98-100`, precisely because the profiler is
thread-affine). Discriminator, one grep of the paste: that sentence present ⇒ rank the
frontend/thread lane (and a possible RPC-path workaround); absent while the
EngineCore→`model_executor.profile` chain is present ⇒ the worker-side pta export
lane of Rule 1. Keep it as a second candidate: on #15983 the reply only promised to
"check whether a workaround is needed in the vLLM profile RPC path", so this lane is
neither confirmed nor refuted.

## Rule 7 — the snapshot cannot resolve a `.postN`

`SOURCES.json` exports `Ascend/pytorch` by CANN pairing tag — `v26.0.1-pytorch2.10.0`
(`8751b36d`, #15983) — while the report is about PyPI wheels `2.10.0.post2/post4/post6`.
The exported tree shows only the *shape* of the stop path
(`torch_npu/profiler/profiler.py:274-279` `stop()` →
`action_controller.transit_action(current_action, None)`,
`_profiler_action_controller.py:78-82`, `csrc/profiler/init.cpp:98-100`), never which patch
introduced the defect. Say that: "the difference between `.post2`/`.post4`/`.post6` is
not in the snapshot; the reporter's control is the evidence".

## Delivery

Verdict (pta patch-release lane / upstream compatibility issue; vllm-ascend forwards
and adds no logic to the stop path) + the release's own pin as the reason the report
is not env drift + rung 1 "install `torch_npu==2.10.0.post6` (or `.post2`), keep the
vllm-ascend wheel, re-run the same serve + `/start_profile` + `/stop_profile`" + the
one-line verification (`python -c "import torch_npu; print(torch_npu.__version__)"`
before the serve) + what to attach when filing upstream (the `Segfault encountered`
block, the `Incorrect schedule …` line, `pip show torch-npu vllm vllm-ascend`, CANN
`version.cfg`) + the minimal bisect below. Do **not** send a patch proposal against
`vllm_ascend/profiler/torch_npu_profiler.py` as the fix: `data_simplification` /
`export_type` are candidates to *test*, not the located cause (upstream said
"whether a workaround is needed" was still open).

Minimal reproducer for the patch comparison — copy `templates/profiler-patch-bisect.py`
and run it once on `.post4` and once on the working patch. It uses the wrapper's own
`_ExperimentalConfig` and a `with`-block start/stop on a few NPU matmuls, so a crash
there reproduces the lane without vLLM, and a clean run redirects the search to the
RPC/thread lane (Rule 6) or the reporter's launch shape.

## Limits to keep stated

- #15983 pastes no serve command, no CANN version and no full log — the two
  screenshots are its only artefacts, so "does `/start_profile` succeed first" and
  "which of start/stop dies" come from asking, not from the paste.
- A `torch_npu` wheel is not rebuilt by the reporter in the recorded case; whether an
  off-pin `pip install` satisfies the installed vllm-ascend ABI is unverified here —
  ask for `pip show torch-npu` after the change rather than asserting it is safe.
- #15869-style "profile 采集" reports on other releases may be this lane or the
  missing-table lane; decide by the marker (`Segfault encountered` vs an analysed
  directory with no table), not by the word "profile".
