# Silent worker exit: reading the vLLM v1 multiproc failure lanes

When the engine reports a worker failure but the pasted log contains no worker
traceback, the worker was killed (signal, host OOM, device/ACL abort) rather than
raising. That distinction decides the next question.

## How to tell the lanes apart (vllm v0.11.0rc3,
`vllm/v1/executor/multiproc_executor.py`)

- Parent side: `_init_executor` calls `WorkerProc.wait_for_ready(...)` (line
  106). `wait_for_ready` (line 477) waits on each worker's `ready_pipe`:
  - worker sent `status != READY` → raises
    `Exception("WorkerProc initialization failed due to an exception in a
    background process. See stack trace for root cause.")`;
  - worker closed the pipe (`EOFError`, i.e. died) → the same exception with the
    context suppressed. This is a **pipe read** in the parent, not the
    `EOFError: Ran out of input` raised *inside* a rank's own traceback: that one is
    the torch_npu gather emulation's object broadcast during `profile_run` —
    `profile-run-logits-gather-eof.md`.
  So the parent message alone cannot distinguish "raised" from "killed" — look
  for the worker-side traceback.
- Worker side: `worker_main` wraps init in `try/except Exception` and logs
  `WorkerProc failed to start.` with the traceback **before** reporting failure
  (line 597). Newer versions log `WorkerProc hit an exception.` (seen at
  `multiproc_executor.py:880` in the v0.17 log). A raised worker therefore leaves
  a rank-prefixed traceback; a killed worker leaves nothing.
- `Parent process exited, terminating worker` (line 558) is emitted by each
  worker's `WorkerDeathMonitor` thread when the *parent* channel closes. It is a
  consequence of the engine core going away, never the cause — even when it
  appears above the parent's own traceback in a pasted log.
- `[ERROR] TBE Subprocess[task_distribute] raise error[], main process
  disappeared!` is printed by TBE op-compile helper processes noticing their
  parent died. Also a consequence. Older issue reports (e.g. vllm-ascend issue
  #198, vllm 0.7.1, "TBE Subprocess Task Distribute Failure When TP>1") used
  this string as a title; do not inherit that conclusion for a different version.
- `!!!!!!! Segfault encountered !!!!!!!` above the parent's `RuntimeError: Engine
  core initialization failed … Failed core proc(s): {'EngineCore': -11}` is the
  same lane seen from the child: `-11` is SIGSEGV, so there is no Python
  traceback to wait for. Read the dump through
  `../hccl-and-hardware/worker-segfault-no-python-frames.md`, section
  `## The startup variant` — a trivial CPU `torch.zeros` at the top of the dump
  is where the damage surfaced, not an arithmetic bug, and the TBE block above is
  already accounted for as a messenger.

## What the timing tells you

A silent gap (the recorded v0.11.0rc0 log had ~14 minutes) between
`rank N in world size M is assigned ...` and the failure spans weight loading,
memory profiling (`determine_available_memory` / `profile_run`) and graph
capture. Use it to bound the phase, not to name a cause.

## Requests that resolve this lane fastest

1. Full log per rank (worker lines carry a rank prefix such as
   `(Worker_TP2 pid=...)`; an excerpt without them is unusable).
2. Host OOM check: `dmesg -T | grep -iE 'out of memory|oom-kill'`.
3. `npu-smi info` (per-card HBM model and free memory).
4. The effective `max_model_len` (`Using max model len N`) and the exact command.
5. Then run the config-first check in `max-model-len-budget.md`.

Recorded outcome for this lane (vllm-ascend issue #3543, image v0.11.0rc0,
910B3, TP=4, Qwen3-VL-30B-A3B-Instruct): no worker traceback; the reporter
reduced `max_model_len`, the model loaded and the service started.
