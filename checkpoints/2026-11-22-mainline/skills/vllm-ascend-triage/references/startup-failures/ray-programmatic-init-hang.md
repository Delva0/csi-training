# An engine built from a program over Ray hangs in silence (`AsyncLLM.from_engine_args`)

Verified at vllm-ascend **v0.8.5rc1** (`ec27af3`) with vllm **v0.8.5.post1**
(`3015d563`); recorded for **#1364** (vllm 0.8.5.post1 / vllm-ascend 0.8.5rc1 / torch
2.5.1 / torch_npu 2.5.1 / CANN 8.1.RC1 / ray 2.47.1, `enable_sleep_mode=True`, and no
`vllm serve` line anywhere in the report: the engine is constructed **inside an RLHF
critic server** with `AsyncEngineArgs(distributed_executor_backend="ray", …)`).
Re-grep the line numbers per release; the ordering below does not move.

## The shape

The first HTTP request to that server hangs (the reporter's `POST /setup`), and the
only evidence is a screenshot whose lines are: several `(RayWorkerWrapper pid=…)`
processes (four pids in the OCR), `plugin ascend enhanced model loaded`, one model-architecture override
warning per vllm-ascend class, then one line per pid —

`[utils.py:2522] Methods add_lora,cache_config,…,scheduler_config not implemented in
<vllm_ascend.worker.worker_v1.NPUWorker object>`

— and nothing after it: no `Starting to load model`, no traceback, no
`Engine core initialization failed`. The reporter names
`AsyncLLM.from_engine_args` (`vllm/v1/engine/async_llm.py:161`) as the blocked
function. That is true, and it is not where the fault is.

## Three readings decide the lane before any Ray theory

1. **`from_engine_args` is the waiter, not the failing frame.** It builds the config,
   picks the executor and constructs `AsyncLLM` → `EngineCoreClient.make_client`
   (`vllm/v1/engine/core_client.py:52-79`) → `AsyncMPClient` → `MPClient.__init__` →
   `_wait_for_engine_startup` (`:412-443`). That loop polls the core-engine
   `proc_handle` and the input socket, and its *only* failure path is a core process
   **exiting** (`RuntimeError("Engine core initialization failed. See root cause
   above.")`, `:430`). A child that is alive and stalled raises nothing, ever. The
   success line is `logger.info("Core engine process %d ready.", eng_id)` (`:439`),
   so its absence is the same fact as the hang; the loop's only progress message is
   `logger.debug("Waiting for %d core engine proc(s) to start: %s")` (`:425`) —
   invisible at the default level, so `VLLM_LOGGING_LEVEL=DEBUG` is the cheapest way
   to see whether the parent is still in that loop. The child's own order is
   `init_device` **then** `load_model`
   (`vllm/executor/ray_distributed_executor.py:395-396`; the same order at
   `vllm/v1/executor/multiproc_executor.py:305-306`), and `load_model`'s first line is
   `Starting to load model %s...` (`vllm/v1/worker/gpu_model_runner.py:1329`).
2. **The last pasted line dates the stall to the worker constructor.** `utils.py:2522`
   is `warn_for_unimplemented_methods` (`vllm/utils.py:2488-2529`): it walks the
   instance inside `__init__` and warns about every base-class method whose body still
   says `NotImplementedError`. It is introspection — not a missing implementation and
   not a defect — and it prints as the worker object is created. So imports, plugin
   load and worker construction all succeeded, and the stall sits in `init_device`
   (device open, then the HCCL/distributed rendezvous) or the start of `load_model`.
3. **The Ray scheduling lane is refuted by the same paste.** Each actor exists and
   constructed an `NPUWorker`, so the cluster did advertise the resource vllm-ascend
   asks for (`ray_device_key = "NPU"`, `vllm_ascend/platform.py:57`) and no actor is
   parked waiting for resources — that is `ray-npu-resource-label.md` and
   `ray-dp-placement-group-npu-exhausted.md`, a different paste (they carry
   `ValueError: Current node has no NPU available … 'GPU': 8.0`, or `'NPU': 0`).

## `VLLM_WORKER_MULTIPROC_METHOD=spawn` is a discriminator here, not a rung

At this pin `_maybe_force_spawn` (`vllm/utils.py:2363-2388`, called from
`get_mp_context`) **already** overrides fork when `is_in_ray_actor()`, exporting
`RAY_ADDRESS` and logging `We must use the 'spawn' multiprocessing start method.
Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. … Reason: In a Ray actor and can
only be spawned`. So read that line out of the paste instead of recommending the env
var:

- present → the **driver** runs inside a Ray actor (the trainer calls the engine from
  an actor/task), and the engine core is a spawned child of an actor;
- absent → the driver is an ordinary process and only the vLLM workers are actors.

A reporter whose engine is already in a Ray actor gains nothing from the env var, and
a report without that warning should not be filed under a fork-after-Ray theory.
(`vllm-ascend`'s own V1 example sets it explicitly anyway —
`examples/offline_inference_npu_v1.py:25`.)

## Ranked candidates, each with the one check that settles it

1. **A worker stalls in `init_device`** — device open, or the HCCL rendezvous with
   the ranks not agreeing on device/rank. Check `npu-smi info` **during** the hang,
   and print from inside each actor
   `ray.get_runtime_context().get_accelerator_ids()["NPU"]` — the value vllm itself
   uses as the device (`vllm/executor/ray_utils.py:55-63`) — plus `RANK`,
   `LOCAL_RANK`, `WORLD_SIZE` from `/proc/<pid>/environ`. Two ranks on one card
   blocks without printing anything.
2. **The core-engine child never reaches `init_device`** — spawn/import fault in the
   actor's environment, or the `RAY_ADDRESS` handoff. Check: is there one process per
   rank at all (`ps -ef | grep -c python`), and does the driver log
   `Waiting for 1 core engine proc(s) to start: {0}` under `VLLM_LOGGING_LEVEL=DEBUG`.
3. **The V1-over-Ray path itself** — real, but ranked below 1-2. At this pin
   `vllm/v1/executor/ray_distributed_executor.py:27` subclasses the V0 executor, which
   forces `VLLM_USE_RAY_SPMD_WORKER=1` and `VLLM_USE_RAY_COMPILED_DAG=1` for V1
   (`vllm/executor/ray_distributed_executor.py:75-78`), and the compiled DAG is built
   lazily on the first batch (`vllm/v1/executor/ray_distributed_executor.py:50-53`).
   vllm-ascend v0.8.5rc1 documents Ray only for multi-node `vllm serve`
   (`docs/source/tutorials/multi_node.md:92-147`; `ray start --num-gpus=8` at `:110`,
   `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` at `:108`, which
   `vllm_ascend/platform.py:48` also sets at import) and ships no recipe or e2e test
   that builds an engine from inside a trainer over Ray — say that as the coverage
   statement. The ablation that isolates this lane is the same work on one node with
   `distributed_executor_backend="mp"`: if that comes up, Ray is the lever; if it is
   equally silent, candidates 1-2 own it.
4. **`enable_sleep_mode=True` is not a start-phase candidate.** Nothing in the stall
   window calls it — `sleep`/`wake_up` are `EngineCoreClient` methods used after the
   engine is ready. Keep it for a report where the engine serves and *then* hangs on
   sleep/wake-up.

## Delivery

Answered, not investigated. The first message owes the correction (the named frame is
the wait; the stalled phase is between the constructor warning and
`Starting to load model`), the ranked list above, and four artefacts:
`py-spy dump --pid <driver> --pid <each RayWorkerWrapper pid>` **during** the hang,
`npu-smi info` during the hang, the per-pid accelerator/rank ids, and the full actor
logs (`ray logs <node>`; `ASCEND_PROCESS_LOG_PATH=<dir>` for the plogs,
`multi_node.md:128`). Recorded #1364 spent 21 assistant turns and 36 tool calls,
~45 K characters of reasoning and **0** characters of message text, and was cut at the
16 384-token cap — with the `_wait_for_engine_startup` reading and the
constructor-warning dating both in hand by its 20th turn and never shipped. None of
the Ray placement-group algebra, the sleep-mode allocator or the V0 executor body
changes that ranked list, so none of it is worth a turn before the answer is out.
