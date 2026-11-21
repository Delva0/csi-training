# `IndexError: tuple index out of range` at `new_computed_blocks.blocks[0]` in the Ascend scheduler

EngineCore dies inside `scheduler.schedule()` on the **first schedulable waiting
request**, with no device involvement:

```text
File ".../vllm/v1/engine/core.py", line 283, in step
    scheduler_output = self.scheduler.schedule()
File "/vllm-workspace/vllm-ascend/vllm_ascend/core/scheduler.py", line 209, in schedule
    blocks = new_computed_blocks.blocks[0]
IndexError: tuple index out of range
```

Recorded for **#4207** (`vllm-ascend:v0.11.0rc0` image, no model/serve line in
the body). Line-anchors below are `v0.11.0rc1` (`c5fe179cef7f`) + vllm
`v0.11.0` (`b8b302cde434`); the same line exists at `v0.10.1rc1`
(`7e16b4a7cdb1`, `scheduler.py:177`), so the window spans both releases — re-grep
`grep -n 'blocks\[0\]' vllm_ascend/core/scheduler.py` before quoting a number.

## Step 1 — why the forked body runs at all (two lines decide it)

- `AscendScheduler.schedule()` delegates to upstream **only** when chunked
  prefill is on: `if self.scheduler_config.chunked_prefill_enabled: return
  super().schedule()` (`vllm_ascend/core/scheduler.py:65-66`). Everything below
  that line is the fork.
- `AscendSchedulerConfig.initialize_from_config` **forces**
  `enable_chunked_prefill = False` (`vllm_ascend/core/schedule_config.py:48`)
  unless the reporter's `--additional-config` carries
  `{"ascend_scheduler_config": {"enabled": true, "enable_chunked_prefill": true}}`;
  the CLI `--enable-chunked-prefill` is overwritten by that dict and cannot flip
  it. It also refuses (`ValueError`, `schedule_config.py:66-75`) a
  `max_num_batched_tokens` below `max_model_len` when chunked prefill is off.

So the crash **proves** the reporter ran `ascend_scheduler_config.enabled: true`
with chunked prefill off, and the `[0]` line executes for every request that
gets as far as the prompt-limit check (`scheduler.py:209` in the `else` arm of
`load_kv_async`).

## Step 2 — what `blocks[0]` is, and why the guard is safe

`blocks` here is **not** used for allocation (that passes the whole
`new_computed_blocks` object to `allocate_slots`, `scheduler.py:234-240`); it is
the first KV-cache group's block list handed to the fork's own watermark check
`scheduler.py:223-224 → _check_watermark_for_prefill` — a function upstream does not
have at all. Two facts make the repair exactly one expression:

- the callee already normalises emptiness: `computed_blocks = computed_blocks or []`
  (`scheduler.py:512`);
- the caller already passes `None` on the async-load branch (`scheduler.py:178`).

So `[]` is the value the consumer is written for, and
`blocks = new_computed_blocks.blocks[0] if new_computed_blocks.blocks else []`
changes no behaviour. Treat "the fork reaches into the upstream container's
tuple instead of using its API" as the ownership statement: upstream's
`KVCacheBlocks` (`vllm/v1/core/kv_cache_manager.py:19-33`) is free to represent
"no blocks" however it likes, and upstream's own `Scheduler.schedule` never
indexes it (`vllm/v1/core/sched/scheduler.py:382-476` passes the object through).

## Step 3 — stop deriving the producer; print it

At the pinned vllm `v0.11.0` the empty tuple is **not reachable**:
`create_empty_block_list()` returns one list per `num_kv_cache_groups`
(`kv_cache_manager.py:396-399`, count at `:129`), and a zero-memory KV config is
refused at startup (`kv_cache_utils.py:696-702`, `raise ValueError` — the only
path that skips it is an *empty* kv-cache spec set). Two candidate producers
therefore remain, and both are the **reporter's build**, not the pin:

1. their bundled vllm returns an empty `KVCacheBlocks` (a 0-length `blocks`
   tuple) for "no computed blocks", which is a legal representation in some
   vLLM generations and exactly what `[0]` cannot survive;
2. their `kv_cache_config.kv_cache_groups` is empty in that process.

One command on their box settles which, and it costs one sentence in the answer:

```bash
python - <<'PY'
import inspect, vllm, vllm_ascend
from vllm.v1.core.kv_cache_manager import KVCacheManager
print(vllm.__version__, vllm_ascend.__version__)
print(inspect.getsource(KVCacheManager.create_empty_block_list))
print(inspect.getsource(KVCacheManager.get_computed_blocks).splitlines()[-6:])
PY
```

**Stop rule.** The rc0 image pairs with an rc-versioned vllm that no snapshot
carries (`#3979` recorded `v0.11.0rc0` + vllm `0.11.0rc3`). Deriving that
release's container semantics from the final release is unbounded: after the
pin's code has answered "this cannot be empty here", the next artefact owed is
the reporter's print plus the guard, not a third theory. The recorded #4207 run
spent its whole 16 384-token output budget on precisely that derivation and
shipped nothing.

## Escape ladder

1. Drop `ascend_scheduler_config` from `--additional-config` (upstream
   `Scheduler`, no `[0]` line) or add `"enable_chunked_prefill": true` to the
   same dict, which routes `schedule()` to `super().schedule()` — keep
   `max_num_batched_tokens >= max_model_len` or the config raises at startup.
   Label it reporter-side: one restart, same traffic.
2. The guard above, with the one-line reproducer "send any prompt twice with the
   same `system` message" — the crash does not need a cache hit, only a
   schedulable request.
3. Only then look at the reporter's `pip show vllm` / the print above; if their
   vllm is a generation where empty is legal, this is a downstream compatibility
   bug to file, not a configuration error.

## Noise that is not evidence

- The traceback's `core.py:710` is the log-emitter line, not a frame: read the
  frames below it, and note the `scheduler.py:209` line number matches the pin
  to the line, which is what says the reporter's vllm-ascend is the release
  tree rather than a hand patch.
- A `_check_watermark_for_prefill` refusal (`skip_cur_request`) and a
  `blocks[0]` IndexError both stop scheduling, but only one raises; if the
  report is "the queue never drains" with no traceback, it is
  `waiting-queue-wedge.md`, not this page.
