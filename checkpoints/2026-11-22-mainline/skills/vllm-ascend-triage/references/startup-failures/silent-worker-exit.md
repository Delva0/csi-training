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
  **The traceback follows that message, it does not precede it.** At vllm v0.14.1
  `d7de043` the site is `except Exception:` → `if ready_writer is not None:
  logger.exception("WorkerProc failed to start.")` (`multiproc_executor.py:744-751`),
  and `logger.exception` attaches `exc_info`, so the rank's whole stack is printed
  *below* the message with the **same** `[multiproc_executor.py:749]` prefix on every
  line. A traceback whose lines all repeat one wrapper's `file:line` is that one
  logging call's block, so asking for "the lines above `WorkerProc failed to start.`"
  asks for the wrong block — ask for the lines below it (per rank), or a paste of
  `grep -A80 'WorkerProc failed to start' <log>`.
- **A kill after the engine is ready looks different from a kill during init:**
  there is no `wait_for_ready` message at all. The EngineCore instead dumps the
  in-flight `SchedulerOutput` and dies with `RuntimeError: cancelled` raised from
  the response queue's `acquire_read`, because `monitor_workers` already saw the
  dead process and shut the queue down. Read that text as the same lane (a
  killed worker, no cause in the paste) and continue in
  `../hccl-and-hardware/cancelled-executor-teardown.md`.
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

## When the traceback exists but the paste cannot carry it

A screenshot whose `*.ocr.txt` is noise (`0 ， 以 ' 彐 《 27i 力 …`) is a **data gap**,
not a lane: the wrapper line is all the paste holds, so classify from the other
tokens the launch line and the model path do carry, and make the traceback request
the deliverable. Recorded #7744 (v0.14.0rc1 `52d4acf` ↔ vllm v0.14.1 `d7de043`,
Atlas 800I A2 ×8, title `使用v0.14.0rc1 … Qwen3-VL-235B-A22B-Instruct-W8A8模型，报错
WorkerProc failed to start.`, body = the `vllm serve` line plus one OCR-garbage
screenshot) is the shape: the recorded answer read the OCR as "no traceback" and led
with **"your serve line is missing `--quantization ascend`"** — the aux-key branch
second, `--mm_processor_cache_type shm` last, against the `shm` page's then-first rung
("drop the flag") — and the reporter's later posts show the failure was
load-phase `KeyError: 'layers.37.mlp.experts.w2_weight_offset'` in `qwen3_vl_moe.py`,
with the same weights and command working on `v0.17.0rc1`. So the flag lane was the
wrong lever, and it was wrong in the direction that costs an answer: before ranking it,
read the model dir's own `quantization_config` (that lane's contract) — item 4 of
`weight-scale-keyerror-quant-variant.md` §*The VL-MoE offset form* carries the
cross-pin reason a flag-omitted engine cannot produce a newer-tag success.

What to read instead, in this order:

1. **The model directory's quant suffix.** `…-W8A8`, `…-w8a8_dynamic`,
   `…-bf16-hfd-w8a8` in the serve path means the engine loads a *quantised variant*,
   and the load-phase quant aux-key `KeyError` (`…weight_scale`, `…weight_offset`) is
   the first hypothesis — `weight-scale-keyerror-quant-variant.md`, whose "no
   traceback reaches the paste" section holds the two-call procedure (the line below
   the wrapper, plus `scripts/checkpoint-key-prefix.py --scales <dir>`).
2. **The support matrix cell for that quant.** `supported_models.md` at the pin can
   mark the model `✅` and leave the W8A8 column empty (Qwen3-VL-MOE, line 81 at
   v0.14.0rc1) — an unvalidated combination, which makes the newer tag a rung rather
   than a mechanism question.
3. **A flag that is merely absent from the tutorial is not a cause.** `--mm_processor_cache_type
   shm` killed workers only when its own error string appeared (`Data for address:id …
   has been modified or is invalid.` on a request, `../multimodal-inputs/shm-mm-cache-invalid-object.md`)
   or when `/dev/shm` could not hold the 4 GiB segment (`MemoryError: Not enough space
   in the data buffer`). Neither text is in the paste, so the flag is a hypothesis at
   the bottom of the ladder with its own grep, never rung 1.
4. **The newer tag when the pin is one or more rcs behind and the combination is
   unvalidated.** #7744's resolver was exactly that (`v0.17.0rc1`, reporter-confirmed),
   not a code fix on the failing host.

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
