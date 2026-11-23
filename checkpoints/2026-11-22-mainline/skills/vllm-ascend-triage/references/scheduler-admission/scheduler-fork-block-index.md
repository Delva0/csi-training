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

Recorded for **#4207** (`vllm-ascend:v0.11.0rc0` image, no model/serve line in the
body) and **#3462** (same `v0.11.0rc0` + vllm `0.11.0rc3`, **bge-m3 =
`XLMRobertaModel`** under a k8s `--task embed` serve — the case that names the
producer, Step 3). Line anchors below are `v0.11.0rc0` (`00ba07102212`; `:209`,
`:520`, `platform.py:139-150`, `schedule_config.py:48,51` and both vllm sites were
re-verified there) and `v0.11.0rc1` (`c5fe179cef7f`) + vllm `v0.11.0rc3`
(`8ce5d3198d00`); the same line exists at `v0.10.1rc1` (`7e16b4a7cdb1`,
`scheduler.py:177`), so the window spans both releases — re-grep
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
  it. Its `__post_init__` also raises (`ValueError` at `schedule_config.py:68`)
  when `max_num_batched_tokens < max_model_len` and chunked prefill is off.

**The crash proves the fork was in play, not that the reporter configured it.**
`Platform.check_and_update_config` force-enables the fork for a default dense
serve — `if model is not None and not use_mla and not async_scheduling:` →
`if not is_multimodal_model and structured_outputs_backend == "auto" and
scheduler_delay_factor == 0 and not send_delta_data and policy == "fcfs":
ascend_scheduler_config.enabled = True` (`vllm_ascend/platform.py:139-150`,
gate input at `:149`) — overwriting an explicit `enabled: false`; the swap itself
is `platform.py:287-292`. So the paste's `Using configured V1 scheduler class
vllm_ascend.core.scheduler.AscendScheduler` is that gate's output, and asking
which `--additional-config` they passed answers nothing (recorded #3979, #3462).
What the traceback does prove is the *pair*: fork in play **and** chunked prefill
off — and the second is forced too, so **nothing about the reporter's flags
survives in this signature**. The `[0]` line then runs for every request that
reaches the prompt-limit check (`scheduler.py:209`, the `else` arm of
`load_kv_async`), i.e. the first request after a clean start.

## Step 2 — what `blocks[0]` is, and why the guard is `[]`

`blocks` here is **not** used for allocation (that passes the whole
`new_computed_blocks` object to `allocate_slots`, `scheduler.py:234-240`); it is
the first KV-cache group's block list handed to the fork's own watermark check
`scheduler.py:223-224 → _check_watermark_for_prefill` — a function upstream does not
have at all. Two facts make the guard behaviour-neutral at that site:

- the callee already normalises emptiness: `computed_blocks = computed_blocks or []`
  (`scheduler.py:512`);
- the caller already passes `None` on the async-load branch (`scheduler.py:178`).

So `blocks = new_computed_blocks.blocks[0] if new_computed_blocks.blocks else []`
changes no behaviour — **but it is not the whole repair when the model has zero
KV-cache groups**: `_check_watermark_for_prefill` indexes a second empty-by-
construction tuple at `scheduler.py:520`
(`num_new_blocks = (num_required_blocks - len(req_blocks[0]) - …)`) over
`self.kv_cache_manager.coordinator.get_blocks(request_id)`, and that returns one
list per KV-cache group — `single_type_managers` is built by enumerating
`kv_cache_config.kv_cache_groups` (`vllm/v1/core/kv_cache_coordinator.py:38-44`),
`get_blocks` returns `tuple(manager.req_to_blocks.get(rid) or [] for manager in
single_type_managers)` (`:176-182`). Zero groups ⇒ `()` ⇒ the same IndexError one
call later. Patch **both** sites, and make the acceptance criterion a real
`POST /v1/embeddings` whose vector matches transformers/HF (`cos ≈ 1`,
`check_embeddings_close` tol 1e-2), not "the service starts".

Treat "the fork reaches into the upstream container's tuple instead of using its
API" as the ownership statement: upstream's `KVCacheBlocks`
(`vllm/v1/core/kv_cache_manager.py:19-33`) is free to represent "no blocks"
however it likes, and upstream's own `Scheduler.schedule` never indexes it
(`vllm/v1/core/sched/scheduler.py` passes the object through).

## Step 3 — the empty tuple *is* reachable at the pin: encoder-only pooling

Earlier revisions of this page asserted that the pinned upstream "cannot produce
an empty tuple" and sent the run to the reporter's box to print
`create_empty_block_list`. That is wrong, and it is why #4207's recorded run spent
its whole output budget deriving container semantics: the producer is an
**encoder-only pooling checkpoint**, a supported serve shape.

- bge-m3 / `BertModel` / `XLMRobertaModel` carry **no** KV cache, so the runner
  reports an empty spec, and `is_kv_cache_spec_uniform({})` returns `True` with the
  tree's own comment `# Encoder-only models do not have KV cache`
  (`vllm/v1/core/kv_cache_utils.py:768-771` at vllm `8ce5d31`). That same empty
  spec is the one path that skips the `No available memory` `raise`
  (`:696-702`), so startup says nothing about it.
- Groups are built *from* the spec (`kv_cache_utils.py:744-752`), so 0 layers ⇒
  `kv_cache_groups == []` ⇒ `num_kv_cache_groups = 0`
  (`vllm/v1/core/kv_cache_manager.py:129`) ⇒ `create_empty_block_list()` returns
  `KVCacheBlocks(())` (`:396-399`) ⇒ `get_computed_blocks` hands that object over
  whenever prefix caching is off (`:169-172`; a pooling serve disables both prefix
  caching and chunked prefill).
- A **decoder-based** embedding checkpoint (`Qwen/Qwen3-Embedding-0.6B`, the only
  case in `tests/e2e/singlecard/test_embedding.py:28-31` at `00ba071`) has ≥1
  group, so `blocks[0]` is `[]` and nothing raises. That coverage gap is why the
  project's own green CI does not contradict the report — and a paste carrying
  `num_gpu_blocks is: 1` is that placeholder, not a block pool.

One read decides which side of the split the report is on: the checkpoint's
`architectures` (`BertModel` / `XLMRobertaModel` / a `…ForSequenceClassification`
over an encoder ⇒ zero groups; `Qwen3Embedding*`/decoder ⇒ ≥1), or the pooling
serve lines in the paste (`Only "last" pooling supports chunked prefill and prefix
caching; disabling both`, `Disabling chunked prefill for model without KVCache`).

Verdict per side: encoder-only ⇒ downstream defect in the release tree, file it
(this is not a configuration error); a decoder-based serve whose group count is
nevertheless 0 is the only case where printing `create_empty_block_list` on their
box still pays.

## Escape ladder

1. **Leave the fork: `--scheduling-policy priority`.** The force-enable gate needs
   `scheduler_config.policy == "fcfs"` (`platform.py:149`) and
   `SchedulerPolicy = Literal["fcfs", "priority"]`
   (`vllm/config/scheduler.py:21`, flag `--scheduling-policy`,
   `vllm/engine/arg_utils.py:902` at v0.11.0rc3), so a non-`fcfs` policy leaves
   `enabled` at its default `False`, `platform.py:287` does not swap the class, and
   upstream `Scheduler` serves — the `[0]` line does not exist there. Keep
   `--additional-config` free of `ascend_scheduler_config`: an explicit
   `enabled: true` substitutes `AscendSchedulerConfig`, which forces `policy`
   back to `"fcfs"` (`schedule_config.py:51`). **This replaces the older "just drop
   the key" rung, which is inert at rc0 precisely because of the force-enable.**
   Label reporter-side: one restart, same traffic; a single embedding request is
   insensitive to the policy, and the effect on a PD / `--async-scheduling` serve is
   not verified.
2. **Route the fork to `super().schedule()`**: `--additional-config
   '{"ascend_scheduler_config": {"enabled": true, "enable_chunked_prefill": true}}'`
   — `enable_chunked_prefill` can be flipped **only** from that dict. Keep
   `max_num_batched_tokens >= max_model_len`, or `schedule_config.py:68` raises at
   startup (turning chunked prefill on also satisfies that check).
3. **Both-site guard** (Step 2), with the one-line reproducer "send any prompt
   once" — the crash needs no cache hit, only a schedulable request — and the
   vector-match acceptance criterion. Say in the answer that a one-site guard is what
   a drive-by patch ships, so the reviewer should expect the `:520` site too.
4. Ask whether a later rc carries the guard; do not claim it does. The judgement is
   "same curl after the upgrade", not the release-notes line.

## Noise that is not evidence

- The traceback's `core.py:710` is the log-emitter line, not a frame: read the
  frames below it, and note the `scheduler.py:209` line number matches the pin
  to the line, which is what says the reporter's vllm-ascend is the release
  tree rather than a hand patch.
- `ERR99999 UNKNOWN application exception` beside it is CANN's teardown noise
  while the process dies around the already-raised `IndexError`; it is not a
  second fault. `num_gpu_blocks is: 1` in a no-KV-cache paste is a placeholder.
- A `_check_watermark_for_prefill` refusal (`skip_cur_request`) and a
  `blocks[0]` IndexError both stop scheduling, but only one raises; if the
  report is "the queue never drains" with no traceback, it is
  `waiting-queue-wedge.md`, not this page.
