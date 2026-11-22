# A dead background helper subprocess: the request never returns and `/health` stays 200

## Scope

The serve is up, `/health` answers **200 in ~2 ms**, one ordinary completion request
never comes back (the client's own deadline fires), nothing is logged as an error, and
the process table still holds the API and its workers. The engine's own counters show
`Running: 1 reqs` flat and `Avg generation throughput: 0.0 tokens/s` — the request was
admitted and its step never finished.

This page owns the shape where the **step is blocked on a queue fed by a helper
process that is gone**: the Dynamic EPLB planner subprocess at vllm-ascend v0.23.0 is
the recorded instance, and any `multiprocessing` helper a worker waits on without a
deadline has the same anatomy (`Launched … subprocess, pid=` in the log, no consumer
timeout, no sentinel on the helper's exit path).

Not this page: the KV-block-lifetime stall behind `--async-scheduling`
(`kv-block-lifetime-async-scheduling.md` — same counters, no helper in the log), the
admission gate (`waiting-queue-wedge.md` — `Running` 0), a bare `TimeoutError` whose
frames are `multiproc_executor.py`/`shm_broadcast.py` from a *live* worker
(`../hccl-and-hardware/rpc-watchdog-timeouts.md`), and a worker process that died
(`died unexpectedly (exit code: N)`,
`../hccl-and-hardware/cancelled-executor-teardown.md`).

Pins for every `path:line` below: **vllm-ascend v0.23.0 (`5cb98caaadeff`)**, **vllm
v0.23.0 (`0fc695fc6d1d`)**; the `/health` semantics are re-verified at **vllm v0.18.0
(`bcf2be961200`)** and the route is version-dependent — re-grep at the reporter's pin
(Step 0).

## Step 0 — a 200 from `/health` is not a liveness reading, and it is one grep

At vllm v0.23.0 the route calls the *engine client's* `check_health`, and that call
reads one flag:

```python
# vllm/entrypoints/serve/instrumentator/health.py:30-34
await client.check_health()
return Response(status_code=200)
except EngineDeadError:
    return Response(status_code=503)

# vllm/v1/engine/async_llm.py:900-903
async def check_health(self) -> None:
    if self.errored:
        raise self.dead_error

# vllm/v1/engine/async_llm.py:1045-1055  (condensed; expressions verbatim)
def is_running(self):  return self.output_handler is None or not self.output_handler.done()
@property
def errored(self):     return self.engine_core.resources.engine_dead or not self.is_running
```

So the 200 means only: the EngineCore resources are not marked dead **and** the API
process's own `output_handler` task is still alive. A worker blocked inside
`execute_model` satisfies both, and stays healthy-looking for as long as the RPC
deadline (Step 4). The executor *has* a real probe — `check_health` doing
`collective_rpc("check_health", timeout=10)` (`multiproc_executor.py:491-493`) — but
`rg -n 'check_health' vllm/` at this pin returns only that definition, the
`AsyncLLM` method and the route: **nothing calls the executor's probe**, so no
`RPC call to check_health timed out` line can appear while a worker is wedged.
`vllm v0.18.0` has the same pair (`async_llm.py:874-877`, `multiproc_executor.py:464-466`)
and the same single-caller result.

Consequences for the answer, both cheap to state:

- "`/health` should have gone 503" is a **frontend-design claim, not the mechanism** —
  say that, with the two lines, instead of ranking a health defect. The recorded
  reporter's Expected section asked for exactly this: the gap is real, the mechanism
  is not a defect in their run.
- `/health` is not the discriminator between "serving" and "wedged". Use the engine
  counters (`Running` level and the generation-token *delta*) or `py-spy dump`.
- Do not transfer the older `check_health`-based reading of this route across releases:
  `../hccl-and-hardware/rpc-watchdog-timeouts.md` records a pin where a worker inside a
  long dump trips a 10 s `check_health`; at v0.18.0/v0.23.0 there is no such caller.

## Step 1 — name the helper from the log, then look for its pid

Two lines, both INFO, both easy to miss because they sit in the *worker* block:

```
(Worker pid=108) INFO [eplb_updator.py:77] [eplb/updator] Launched EPLB subprocess, pid=312
(Worker pid=109) INFO [eplb_updator.py:77] [eplb/updator] Launched EPLB subprocess, pid=324
```

`rg -n 'Launched .* subprocess, pid=|Dynamic EPLB is True' <serve log>` gives you the
helper pids and the config that started them (`eplb_config.dynamic_eplb`, in
`--additional-config`, which `ascend_config.py:885-890` asserts only together with
`DYNAMIC_EPLB`/`EXPERT_MAP_RECORD` — so the key on its own raises at config time, and a
launch that got this far had both). Each **worker** launches its own planner, so a TP=2 serve
shows two pids and one dead planner wedges one rank — and because the next step's
collective cannot complete, every rank then blocks, which is why the report reads as a
whole-service stall. Ask for the planner pids from the log the reporter already pasted
and for `/proc/<pid>/stat` state (`Z` or missing = gone), which is the one command that
turns "the forward is waiting" into "the helper it waits on is dead".
Then `py-spy dump --pid <worker pid>` names the wait directly — a blocked worker
answers no RPC, so the stack is the only in-place look (it needs ptrace permission:
same UID, or `--privileged`/`CAP_SYS_PTRACE` inside the container).

## Step 2 — the consumer has no deadline (source-verified)

```python
# vllm_ascend/eplb/eplb_updator.py:106-109, called from
# vllm_ascend/worker/model_runner_v1.py:2269 and :3424 inside execute_model
def forward_before(self):
    if self.get_update_info_flag():                                  # :91
        self.update_info_all = self.eplb_process.block_update_q.get()   # :109  no timeout
```

The flag fires on a fixed iteration count
(`cur_iterations == expert_heat_collection_interval + algorithm_execution_interval - 1`,
`:91-92`), i.e. the wait is entered on the *next* forward after one heat-collection
window — which is why the baseline request can succeed and the *next* one hang on a
serve that already recorded a full expert-weight update.

The producer's exit path emits nothing the consumer can see:

```python
# vllm_ascend/eplb/core/eplb_worker.py:383-401  (condensed)
while True:
    try:
        planner_q.get(); packed_update_info = self.worker.do_update()
        ... block_update_q.put(packed_update_info) ...
    except Exception as e:
        logger.warning("[eplb/worker] Subprocess crashed, EPLB optimization will stop. error=%s", e, exc_info=True)
        break
```

So **both** death modes are silent to the consumer: a natural policy/update exception
(the warning above, `break`, no sentinel, no error object into `block_update_q`) and an
external SIGKILL/SIGTERM (`Z` in `/proc`, nothing logged at all). The only liveness
check in the file guards shutdown (`if self.process.is_alive(): terminate()` at `:189`),
so the gate needed in `get_update_info_flag`/`forward_before` already has a known
idiom in the same class.

## Step 3 — what upstream does, and why the patch is not a second code owner

Upstream vLLM has its own EPLB control plane and it fails *differently*:
`vllm/distributed/eplb/async_worker.py:25-48` starts a **daemon thread**
(`threading.Thread(target=thread_target, daemon=True)`) and the transfer side
rendezvouses on `CpuGpuEvent` (`:133`), never on a blocking `Queue.get()` inside the
model step. A dead upstream worker therefore degrades rebalancing (the log line says
"EPLB optimization will stop") and leaves generation running; a dead downstream planner
blocks the step. That difference is the finding to state when a fix is requested — the
downstream migration (RFC #14178, whose body reads like a full executor fork) is about
this, not about a missing health probe.

Do not go hunting for a fork that duplicates the monitor/timeout: at this pin
`vllm_ascend/patch/platform/patch_multiproc_executor.py` is **211 lines** and
`class AscendMultiprocExecutor(MultiprocExecutor)` overrides only `_init_executor`
(a copy of the upstream body) plus `AscendWorkerProc.make_worker_process` with
`daemon=False` (`:199`). It does **not** override `execute_model`, `check_health`,
`start_worker_monitor` or `collective_rpc`, so the upstream lines the report cites
(`multiproc_executor.py:268-299` monitor, `:307-330` the RPC deadline) are the code that
runs. One read of the file settles it; a search for a second copy costs the session.

## Step 4 — the RPC deadline cannot be lowered from the launch line on Ascend

The wedge ends only when the step RPC's deadline expires, and that deadline has a
**floor** here:

```python
# vllm_ascend/platform.py:1275-1282  (log line: platform.py:1277)
if envs_vllm.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS < 1836:
    envs_vllm.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS = 3000
```

with the launch log carrying exactly that sentence. So `export
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=8` before `vllm serve` is **inert**, and only an
in-process assignment after platform init (the recorded reporter patched an executor
subclass) can shorten it. Two things follow:

- The dump lane's "raise `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`" lever
  (`../hccl-and-hardware/rpc-watchdog-timeouts.md`) and this lane's "the service takes
  3 000 s to notice" are the same variable seen from opposite sides. Here the answer is
  **never** "lower the timeout": the floor exists because the HCCL op timeout is
  1 836 s (`platform.py:1275`), and a lower value converts a wedged step into an
  `EngineDeadError` teardown.
- What the shortened-deadline diagnostic proves is only the generic error path: the RPC
  timeout raises `RPC call to execute_model timed out` → `EngineDeadError` → `/health`
  503 → API stop in ~10 s. It does not measure recovery and it does not place the
  cause — the cause is Step 2. Say which part is a diagnostic experiment: the report's
  own numbers (HTTP 500 after 8.0876 s) read like a recovery, and they are not one.

## Ranked answer this page supports

1. **The planner died and the worker is waiting on `block_update_q.get()` with no
   deadline** (source-verified at the pin: `eplb_updator.py:106-109`, `:91`;
   producer exit at `eplb_worker.py:383-401`). Confirming check: planner pids from the
   log + `/proc/<pid>/stat`, then `py-spy dump` on the worker pid. The observed stall is
   the reporter's own recording (both pids killed, next request timed out at 45.0075 s);
   the blocking site is source-verified, the "this is what also happens on a natural
   crash" part is not.
2. **`/health` cannot report it** (source-verified: `health.py:30-34`, `async_llm.py:900-903`,
   `:1045-1055`). This is the design gap in the report's Expected section, not a second
   defect; it also means any readiness prober is blind to this stall.
3. **A natural planner exception is the same wedge without any operator action**
   (source-verified: `eplb_worker.py:383-401` logs and breaks; nothing is put back).
   Untested end-to-end in the recorded report — say so.
4. **Cross-rank exposure**: each worker owns its planner, and the expert-weight d2d
   transfer is collective, so one dead planner leaves ranks at different map states
   (hypothesis for the multi-planner/partial-death case; the recorded run killed both).
   The eventual failure for the surviving ranks is the HCCL op timeout, not an EPLB
   error.

## Ladder, cheapest first

1. **Prove the wait, in place**: `py-spy dump --pid <worker pid>` (a wedged worker still
   accepts the dump; it accepts no RPC). `block_update_q.get()` in the stack is the
   verdict. Same call: planner pids and `/proc` state.
2. **Separate EPLB from everything else in one launch**: same command with
   `eplb_config.dynamic_eplb: false` **and** without `DYNAMIC_EPLB=1` (drop the pair
   together — the assert above means neither alone is a valid flag state). The baseline
   request then serves through the same fault and the wedge is an EPLB-lane statement;
   keep the rest of the recipe identical, since `weight_nz_mode`/`fused_mc2` also come
   from that `--additional-config` block.
3. **Bound the wait** (the fix to propose, not a launch flag): `block_update_q.get(timeout=…)`
   plus an `is_alive()` gate in `get_update_info_flag` (`:91`), or have the planner's
   `except` path put an error/sentinel object before `break`, so
   `forward_before` raises a usable `RuntimeError` instead of blocking. Both are local
   to `vllm_ascend/eplb/`, both keep the RFC's managed-thread direction open.
4. **Regression seam**: `tests/ut/eplb/test_eplb_updator.py` at this pin builds a real
   `EplbUpdator` with mocked dist/adaptor and exercises only
   `compute_and_set_moe_load` (`def test_compute_and_set_moe_load_*`); nothing in it
   mentions `block_update_q` or `forward_before`, so a "planner exits before returning
   a plan" unit case is a small addition there — name the file, do not claim existing
   coverage.

## What to ask the reporter for

The serve log's `Launched EPLB subprocess, pid=` lines and the planner pids' state
*after* the stall, `py-spy dump` of the worker pid, the engine log block across the
stall (`Running:` / `Waiting:` per 10 s line), and — for a natural-crash claim — the
planner's own stderr, since `[eplb/worker] Subprocess crashed …` is on the worker's
inherited stdout and the traceback that precedes it is what names the policy exception.

## Scope limits of the recorded conclusion

Recorded #18691 was a **controlled failure injection**: both planners SIGTERM'd, a tiny
dummy-weight Qwen3-MoE on 2×910B3, TP=2/EP=2, eager, `--max-model-len 64`. So the wait,
the missing sentinel and the `/health` semantics are general to the pin, while the
45 s/3 000 s timings, the per-rank exposure and the "no fallback" statement are that
run's. A production report of the same shape needs the planner's own exit reason before
the cause is labelled: SIGTERM from an OOM killer, a policy exception and a container
restart all land here.
