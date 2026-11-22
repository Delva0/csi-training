# `execute_model` returned `None`: `'NoneType' object has no attribute 'sampled_token_ids'` on the v0.8/v0.1x line

## Signature

- EngineCore dies on the **first** request inside upstream
  `vllm/v1/core/sched/scheduler.py::update_from_output` with
  `AttributeError: 'NoneType' object has no attribute 'sampled_token_ids'`. The stack
  above it is all `vllm/**` plus the caller, with **no `vllm_ascend` frame**:
  `LLM.generate` → `v1/engine/llm_engine.py::step` → `core_client.py::get_output` →
  `self.engine_core.step()` → `v1/engine/core.py::step` → `update_from_output`.
- It arrives through an RL trainer that builds `vllm.LLM(...)` itself, so it is wrapped
  in the framework's own error type — recorded #960 is `RayTaskError(AttributeError)`
  inside verl's `vllm_rollout_spmd.generate_sequences` (DAPO recipe), on
  vllm-ascend **v0.8.5rc1** with `VLLM_USE_V1=1` and a self-built **vllm main** build
  (`0.8.5.post2.dev0+g3015d5634`). There is no serve command to read: the flags live in
  the trainer's `LLM(**kwargs)`.
- **Same message, different cause — read the command before following this page.**
  `../spec-decode-mtp/async-spec-bookkeeping-desync.md` has it with a worker
  `KeyError: '<id>'` in `_update_states` beside it on a newer release, and its flags are
  `--async-scheduling` **plus** `--speculative-config`. With neither flag the runner's
  bookkeeping cannot be the source: the whole output object is `None`.

## Two statements can produce it; one is refuted at the pin

`sampled_token_ids = model_runner_output.sampled_token_ids`
(`vllm/v1/core/sched/scheduler.py:630` at vllm v0.8.5 `ba41cc90e8ef`) receives `None`.
Only:

1. the worker's rank gate — vllm-ascend v0.8.5rc1
   `vllm_ascend/worker/worker_v1.py:170-175`:
   `output = self.model_runner.execute_model(scheduler_output)` then
   `return output if self.rank == 0 else None`, where upstream at the same pin is
   `return output if self.is_driver_worker else None`
   (`vllm/v1/worker/gpu_worker.py:264-269`);
2. the model runner returned `None` — **refuted at this pin**, two reads, no sweeps:
   `vllm_ascend/worker/model_runner_v1.py:631-641` returns
   `EMPTY_MODEL_RUNNER_OUTPUT` for an empty batch and a `ModelRunnerOutput` otherwise,
   exactly like upstream `vllm/v1/worker/gpu_model_runner.py:1003-1016`.

## The gate is equivalent only in the executor that derives one from the other

`is_driver_worker` is *not* a second predicate in the V1 multiproc executor:
`vllm/v1/executor/multiproc_executor.py:288` passes `"is_driver_worker": rank == 0` for
the ranks built at `:79-83`, and `WorkerBase.__init__` stores both values
(`vllm/v1/worker/worker_base.py:44-48`). So "downstream replaced `is_driver_worker` with
`rank == 0`" dies **there** — which is what recorded #960 concluded: correct for that
executor, and only an answer once the reporter's own executor is excluded (next
paragraph). Do not stop at the multiproc read: one grep over
every executor of the pin (`vllm/v1/executor/`, `vllm/executor/`) is what decides it.

**They diverge in the one executor an offline trainer actually uses.**
`ExecutorWithExternalLauncher(UniProcExecutor)` — `vllm/executor/uniproc_executor.py:68-120`,
selected by `distributed_executor_backend="external_launcher"`, docstring
"torchrun-compatible launchers, … offline inference with tensor parallelism" — builds its
single worker with `rank = int(os.environ["RANK"])` (`:110`) and
`is_driver_worker = True` (`:112`) **unconditionally**. Upstream every process is a
driver; vllm-ascend's `self.rank == 0` passes on **one** of them. Every other engine's
`execute_model` returns `None`, its own EngineCore raises in `update_from_output`, and
the trainer reports it as its rollout task failing. That is the shape of #960, and the
reporter's own sentence describes it — *"Theoretically, it should be executed when
rank==0, but in fact it return None"* is a driver process whose `rank` is not 0.
Confirm the backend in their tree before asserting it (Deliverable 1/3).

A free corroborating line is already in the paste: the client frame is
`get_output` → `return self.engine_core.step()`, i.e. `InprocClient`
(`vllm/v1/engine/core_client.py:200-201` at the pin), and the external-launcher branch is
the V1 configuration that *requires* it — `uniproc_executor.py:97-99` asserts
`not envs.VLLM_ENABLE_V1_MULTIPROCESSING`. An engine that steps in-process **and** hands
`None` to `update_from_output` is this branch until the reporter's
`distributed_executor_backend` says otherwise.

## Deliverable

1. Verdict + one line of mechanism (as above), then the check that binds it to their
   process: `print(os.environ["RANK"])` and the `distributed_executor_backend=` line of
   their `LLM(...)` kwargs — grep the trainer for `external_launcher`.
2. Fix, cheapest first: change `worker_v1.py:175` to upstream's
   `return output if self.is_driver_worker else None` (the worker already receives the
   flag — `NPUWorker.__init__` takes `is_driver_worker` and forwards it, `:51-73`); or
   take a release whose worker has no gate at all — verified at v0.20.2rc1
   (`vllm_ascend/worker/worker.py:435-441` returns the runner's output whenever it is a
   `ModelRunnerOutput` / `AsyncModelRunnerOutput` / `NoneType`), paired with that
   release's vllm.
3. Ask for the one grep in *their* tree before either:
   `rg -n 'is_driver_worker' vllm/executor/uniproc_executor.py` — the flag's value at
   their ref is the premise, and #960's vllm is a main build, not the pin.

## Ablations that do not test this

- **`--tensor-parallel-size 1` does not remove it.** The failing `rank` is the
  launcher's `RANK` (`uniproc_executor.py:110`), not a TP index: with
  `external_launcher` every process still reads its own `RANK`, so a TP=1 re-run keeps
  the same failure on the same ranks. The ablation that removes the gate is a
  **single-process** launch (plain `python`, world 1, `RANK` 0 or unset).
- `--enforce-eager` and `ASCEND_LAUNCH_BLOCKING=1`: the return value is host-side; there
  is no device work to attribute and no capture region involved.
- A longer/shorter prompt, a different checkpoint, a `--gpu-memory-utilization` change:
  the first request of any shape reaches the same statement.

## When the pinned upstream is not the reporter's upstream

The mechanism above is readable at the pin, but #960's other vllm frames were not:
pasted `v1/engine/core.py:294 def step` is `:192` at the pin, `scheduler.py:639
update_from_output` is `:625`, and the screenshot's `EngineCore.execute_model` wraps the
executor call in `try/except BaseException` + `dump_engine_exception(...)`, a helper that
exists in **no** file of the pinned tree. Some frames did match
(`entrypoints/llm.py:473,1423`, `v1/engine/llm_engine.py:218`), so the test is 2-3
frames, not one — and a partial match means "nearby main build", not "the pin"
(`index.md` rule 9). Say that in the message and keep the mechanism, which does not
depend on the drifted lines.

## Limits

- verl's `LLM(**kwargs)` and its `external_launcher` string are not in this snapshot; the
  page states the mechanism from vllm's own branch, and the reporter confirms the backend.
- Whether the **main** build's external-launcher branch still sets
  `is_driver_worker = True` is unverified here (pin is v0.8.5); one grep in their tree.
- Which process was `RANK=0` in their run, and whether the trainer's Ray actors share one
  torchrun world, is only in their launch script — ask for the `RANK` line rather than
  reconstructing the topology.
