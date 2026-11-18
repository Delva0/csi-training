# vllm-ascend's own scheduler subclasses: the opt-in `AscendScheduler`, signature drift, and a copied `schedule()` that goes stale

## Scope

A report whose traceback or log names a scheduler class this fork ships — a `vllm_ascend/core/*scheduler*.py`
subclass or the `patch/platform/patch_balance_schedule.py` rebind — a crash inside it, a `ValueError` out of
its config, a scheduling behaviour that only exists when it is enabled, a
`TypeError: <Subclass>.__init__() got an unexpected keyword argument '<kw>'` raised while upstream vLLM
constructs the scheduler, or a request that is silently never scheduled because the subclass copied
`Scheduler.schedule()` and upstream moved on. Three members, three failure shapes, all worked below:

| member | selected by (at its tag) | its crash |
| --- | --- | --- |
| `AscendScheduler` (`core/scheduler.py`, `v0.10.x`–`v0.11.x`) | `additional_config.ascend_scheduler_config.enabled` (default `false`) | `IndexError: tuple index out of range` at `.blocks[0]` for a zero-KV-group (pooling) model, a `ValueError` out of `AscendSchedulerConfig`, or a long-prefill/P-D-transfer behaviour → §A |
| `SchedulerDynamicBatch` (`core/scheduler_dynamic_batch.py`, present through `v0.23.0`, gone from `v0.24.0rc1`) | `additional_config.SLO_limits_for_dynamic_batch != -1` (default `-1`) | `TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument 'hash_block_size'` at engine startup → §B |
| `BalanceScheduler` (`patch/platform/patch_balance_schedule.py`, monkeypatched in by a module-level `vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler`) | env `VLLM_ASCEND_BALANCE_SCHEDULING=1` (default unset) | **no error at all**: a guided/structured-output request never leaves upstream's `skipped_waiting` queue, so a streamed `guided_choice`/`regex`/`grammar` request never emits a first token while the engine stays up → §C |

§B and §C are *config-selected*; **§A is not always** — on the `v0.11` line the platform enables
`AscendScheduler` itself for a non-MLA model whose scheduler still has the default shape (see "How it
gets enabled"), so its marker can appear in a run whose serve command never mentions it. The first
question is therefore whether it was enabled at all — the marker is one upstream warning line (`Using configured V1 scheduler class …`) for the two
`scheduler_cls` members, or the env var and the module-level `Scheduler = BalanceScheduler` rebind for §C —
never the reporter's command. The second question is whether the subclass still matches the upstream
`Scheduler.__init__` at that tag. Siblings worth naming in an answer, because they show the two intended
patterns at the same tag: `core/scheduler_profiling_chunk.py` takes `hash_block_size: int | None = None` and
forwards it (with a comment naming the upstream PR), and `core/recompute_scheduler.py` takes `*args,
**kwargs` and passes them straight through.

Not this page: scheduler crashes on the default upstream scheduler (`IndexError`/`KeyError` in
`vllm/v1/core/sched/scheduler.py`), the prefix-cache unit mismatch (`prefix-cache-granularity.md`), or the
pooling model's *wrong vectors* with the service healthy (`pooling-and-embedding.md`). Drift arguments land
on the one subclass that spells the signature out, which is why the siblings above are worth naming.

## Fingerprints

| line | owner at the ref |
| --- | --- |
| `WARNING … [core.py:112] Using configured V1 scheduler class vllm_ascend.core.scheduler.AscendScheduler. This scheduler interface is not public and compatibility may not be maintained.` | upstream `vllm/v1/engine/core.py:112` (`v0.11.0rc3`), printed whenever `scheduler_config.scheduler_cls` is not vLLM's own `V1Scheduler`. Its presence proves the opt-in scheduler is in force even if the pasted serve command does not show `--additional-config`. |
| `INFO … [core.py:121] Disabling chunked prefill for model without KVCache` | upstream `core.py:119-121`: `if len(kv_cache_config.kv_cache_groups) == 0`. This is the zero-KV-group fingerprint that decides the §"blocks[0]" crash. |
| `IndexError: tuple index out of range` at `vllm_ascend/core/scheduler.py:209 blocks = new_computed_blocks.blocks[0]` | the AscendScheduler waiting loop, `:209` at `v0.11.0rc0`–`v0.11.0` (`:201` at `v0.10.2rc1`). |
| `ValueError: Ascend scheduler is enabled without chunked prefill feature. Argument max_num_batched_tokens (A) is smaller than max_model_len (B) …` | `core/schedule_config.py:66-75` at `v0.11.0rc0`. |
| `TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument 'hash_block_size'` (wrapped as `Worker failed with error …` / `Engine core initialization failed`) | upstream `vllm/v1/engine/core.py:149` constructs the scheduler by keyword at the paired tag; the subclass is `vllm_ascend/core/scheduler_dynamic_batch.py` and its `__init__` has no `hash_block_size` (§"The 0.19–0.23 line"). |

## How it gets enabled (`verified` at `v0.11.0rc0`)

- `vllm_ascend/ascend_config.py:168-174`: `AscendSchedulerConfig.enabled = additional_config[
  "ascend_scheduler_config"].get("enabled", False)`; every other key in that dict becomes an attribute.
- `vllm_ascend/platform.py:287-292` (`check_and_update_config`): only when `enabled` is true does it call
  `AscendSchedulerConfig.initialize_from_config`, which rewrites `vllm_config.scheduler_config` with
  `scheduler_cls = "vllm_ascend.core.scheduler.AscendScheduler"`, `enable_chunked_prefill = False`,
  `policy = "fcfs"`, `max_long_partial_prefills/long_prefill_token_threshold = None` → and then
  (`core/schedule_config.py:56-59`) re-applies any of those keys the user passed inside
  `ascend_scheduler_config`, `scheduler_cls` included.
- The docs at that release show the block with `"enabled": True`
  (`docs/source/user_guide/configuration/additional_config.md:63-73`, example `:95-100`), so a tutorial or
  doc example is the usual way a reporter lands on it while pasting only `vllm serve … --enforce-eager`.
  Treat the warning line as authoritative and say the flag is in force, rather than disputing their command.
- **A is also auto-enabled (verified at `v0.11.0rc0`, `platform.py:139-163`).** When
  `not model_config.use_mla and not scheduler_config.async_scheduling`, the platform logs
  `Non-MLA LLMs forcibly disable the chunked prefill feature …` and — if the model is not multimodal,
  `structured_outputs_config.backend == "auto"`, `scheduler_delay_factor` is not positive,
  `send_delta_data` is off and the policy is `fcfs` — sets `ascend_scheduler_config.enabled = True`
  (`:150`), then raises `max_num_batched_tokens` to `max_model_len` (`:160-163`, Trap D in
  `../vllm-ascend/config-flag-lifetimes.md`). The later `initialize_from_config` gate (`:287-292`) then
  sees `enabled` and installs `AscendScheduler`. Consequences for triage: on that line a Qwen/GLM-class
  (non-MLA) model runs the fork's scheduler by default; its `max_num_batched_tokens` is the model's
  `max_model_len`, which is what makes a dummy `profile_run` expensive enough to print
  `Available memory: 0` (`startup-and-stall.md` §3c); and `--max-num-batched-tokens` alone cannot bound
  it. Escape hatches: `--async-scheduling` (skips the whole block) or
  `additional_config.ascend_scheduler_config.enable_chunked_prefill: true`.
- Version status: present in `v0.10.2rc1` and the whole `v0.11` line; **dropped before `v0.12.0rc1`**
  (`Drop ascend scheduler` #4623, `ea54388e1`, after the #4498 drop and the #4580 revert), so `v0.12.0rc1`
  has no `vllm_ascend/core/scheduler.py` at all. "The file is absent at a later tag" is the removal, not a
  fix in place.

## A. The `blocks[0]` crash: a pooling model has zero KV-cache groups

Chain, all read at the refs:

1. A pooling/embedding model (bge-m3, `--runner pooling`, `/v1/embeddings`) gets a `kv_cache_config` with
   **no groups**; upstream `vllm/v1/engine/core.py:119-121` (`v0.11.0rc3`) notices and logs
   `Disabling chunked prefill for model without KVCache`.
2. `vllm/v1/core/kv_cache_manager.py:396-399` (`v0.11.0rc3`):
   `create_empty_block_list()` returns `KVCacheBlocks(tuple([] for _ in range(self.num_kv_cache_groups)))`
   — with zero groups that is an **empty tuple**. `get_computed_blocks` returns exactly this object whenever
   caching is disabled (`:154-170`, `if not self.enable_caching … return self.create_empty_block_list(), 0`),
   which is the pooling model's forced state, and `num_kv_cache_groups` is
   `len(kv_cache_config.kv_cache_groups)` (`:129`), i.e. 0.
3. `AscendScheduler.schedule()`'s waiting loop evaluates `blocks = new_computed_blocks.blocks[0]`
   (`:209` at `v0.11.0rc0`) **unconditionally** for a request that is not loading remote KV, inside its own
   watermark check (`_check_watermark_for_prefill`, `:507`, called at `:223`), before any `blocks`-shaped
   value is used. `.blocks[0]` on `()` raises `IndexError: tuple index out of range` on the first schedule of
   the first request.
4. Upstream's own waiting loop never indexes `.blocks[0]`; it keeps the `KVCacheBlocks` object and passes it
   to `allocate_slots`. That is why the same model serves on the upstream scheduler and dies here.

Levers, in order of cost:

| lever | exact change | cost |
| --- | --- | --- |
| drop the opt-in scheduler | remove `ascend_scheduler_config` from `--additional-config` (or `"enabled": false`); confirm the `core.py:112` warning is gone | lose the Ascend scheduler's long-prefill/P-D-transfer behaviour |
| keep the block, change the class | `--additional-config '{"ascend_scheduler_config": {"enabled": true, "scheduler_cls": "vllm.v1.core.sched.scheduler.Scheduler"}}'` — `core/schedule_config.py:56-59` copies any key the block carries over the forced value | untested on hardware; the rest of the block's settings still apply |
| upgrade | `v0.12.0rc1`+, where the AscendScheduler is dropped | version jump; re-check every other `additional_config` key |

## B. The subclass fell behind upstream `Scheduler.__init__`: `TypeError … unexpected keyword argument 'hash_block_size'` (#18688)

`v0.23.0` release image (its `.github/vllm-release-tag.commit` says `v0.23.0`, so the paired vLLM is the same
version), one 910B3, Qwen3-0.6B BF16, TP=PP=DP=1, `additional_config={"SLO_limits_for_dynamic_batch": 1.0,
"weight_nz_mode": 0}`, eager, prefix caching off. `LLM(...)` loads the weights and sizes the KV cache, then
EngineCore dies constructing the scheduler (`vllm/v1/engine/core.py:149` → `Scheduler(`) with
`TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument 'hash_block_size'`; the parent
exits 1, `generate()` is never reached, HTTP serving is not involved.

Chain, all read at the tags:

1. **Gate.** `vllm_ascend/platform.py:713-718` at `v0.23.0`: `if ascend_config.SLO_limits_for_dynamic_batch !=
   -1:` sets `scheduler_cls = "vllm_ascend.core.scheduler_dynamic_batch.SchedulerDynamicBatch"` and forces
   `enable_chunked_prefill = True`; `vllm_ascend/ascend_config.py:238` defaults that key to `-1`, so only an
   explicit `additional_config` entry reaches this path. Removing the key is therefore the off-switch,
   verified by the gate line and not by a hardware test.
2. **Upstream side.** vLLM `v0.23.0` `vllm/v1/engine/core.py:149` calls
   `Scheduler(..., block_size=scheduler_block_size, hash_block_size=hash_block_size)` by keyword, and
   `vllm/v1/core/sched/scheduler.py:66-75` declares `hash_block_size: int | None = None` after `block_size`
   (`if hash_block_size is None: hash_block_size = block_size`). The kwarg was introduced by vLLM `#29143`
   (`48ddb02b79`, "[Hybrid Allocator] Support KV cache groups with different block_size"). Earliest release
   tag containing it: **`v0.12.0`** (`git grep -n hash_block_size v0.12.0 -- vllm/v1/core/sched/scheduler.py`
   → `:199`; absent at `v0.11.0`), so every Ascend release from then until the removal shipped without it.
   Do not read a *filtered* `git tag --contains` list as the introduction point: a recorded run filtered with
   `grep -E '^v0\.(2[0-9])'`, saw `v0.20.0` first and concluded "added by v0.20.0".
3. **The Ascend subclass never learned it.** `v0.23.0:vllm_ascend/core/scheduler_dynamic_batch.py:126-144`
   declares `__init__(vllm_config, kv_cache_config, structured_output_manager, block_size=None, mm_registry=…,
   include_finished_set=False, log_stats=False)` and forwards `block_size` positionally, so the keyword raises
   before `super().__init__` runs. `git log -S'hash_block_size' v0.23.0 -- <that file>` is empty: no fix ever
   landed.
4. **The in-tree pattern is one directory away.** `v0.23.0:vllm_ascend/core/scheduler_profiling_chunk.py:56-74`
   (comment at `:62-64`) carries `# hash_block_size was added in vLLM #40946; keep it optional so the subclass
   works on both pinned vllm and main` and forwards it. So "which sibling already solved it" is a single grep,
   and that file is the model for a local patch.
5. **Status.** The feature was removed from `main` by PR #11933 (`2b115d02dd`, 2026-07-15, "Remove dynamic
   batch slo"). That commit is **not** an ancestor of `v0.23.0`/`v0.23.0.post1` (the file is still present
   there) and **is** an ancestor of `v0.24.0rc1`, where the file is gone (`git merge-base --is-ancestor
   2b115d02dd v0.24.0rc1`, `git ls-tree`). Cite `v0.24.0rc1`, not `v0.24.0`: this clone has no `v0.24.0` tag,
   and `git cat-file -e <tag>:<path>` fails for a missing **tag** exactly as it does for a missing path, so
   the empty result proves nothing either way. The same caveat covers `v0.19.0`, `v0.20.0` and `v0.22.0rc1`,
   which this clone's tag line (`v0.18.0`, `v0.19.1rc1`, `v0.20.2rc1`, `v0.21.0rc1`, `v0.22.1rc1`, `v0.23.0`,
   `v0.24.0rc1`) does not carry. The report's own scope statement ("already merged on main, still in the
   pinned release image") is therefore correct: this is a release-scope answer, not an open bug hunt.

Answer shape:

| line | content |
| --- | --- |
| workaround (proposed) | delete `SLO_limits_for_dynamic_batch` from `--additional-config` (keep the rest, e.g. `weight_nz_mode`) → default vLLM `Scheduler` → startup succeeds. Cost: lose the dynamic token-budget refinement |
| workaround if the feature is required | local patch adding `hash_block_size: int \| None = None` and forwarding it, exactly as `scheduler_profiling_chunk.py` does at the same tag; with one KV group and prefix caching off, `hash_block_size` defaults to `block_size` upstream anyway, and forwarding — not "merely swallowing" the keyword — answers the objection a reviewer will raise — *proposed, not hardware-tested* |
| upgrade | `v0.24.0rc1`+ where the class is removed (#11933) — a version jump plus a re-check of every other `additional_config` key |
| root cause | `vllm_ascend/core/scheduler_dynamic_batch.py` `SchedulerDynamicBatch.__init__` at the reported tag against the paired `vllm/v1/core/sched/scheduler.py:66-75`; gate `platform.py:713-718` at `v0.23.0` |

Falsifier: the same command with the key removed still dies in the scheduler constructor ⇒ the selector is
not `platform.py:713` at that build, and the first step is to read the `non-default args` / `Using configured
V1 scheduler class …` lines rather than the pasted command.

## C. The env-gated `BalanceScheduler` copies `schedule()`: a guided request is never scheduled (#8328)

`vllm-ascend 0.18.0rc1` + vLLM `0.18.0`, 2×910B3, Qwen3.5-27B-W8A8-MTP, TP2, `FULL_DECODE_ONLY`, MTP 3,
`--async-scheduling`, `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`; the failing deployment
differs from a working one only by its env block, which carries `-e VLLM_ASCEND_BALANCE_SCHEDULING=1`. The
official `structured_outputs.py --stream` example hangs after printing `choice (Streaming):` — no first
token, no error, no timeout, the engine stays up and non-guided traffic serves.

Chain, verified at the two refs:

1. `vllm_ascend/patch/platform/__init__.py:38` imports `patch_balance_schedule` only when
   `envs.VLLM_ASCEND_BALANCE_SCHEDULING` is set, and that module ends with
   `vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler` (`:706`) — a process-wide rebind of the
   upstream symbol, not a `scheduler_cls` string.
2. `BalanceScheduler.schedule` (`:58`) is a **verbatim copy** of an older upstream `Scheduler.schedule`: its
   waiting loop is `while self.waiting and token_budget > 0` (`:265`), skipped requests go into a *local*
   `skipped_waiting_requests` queue (`:263`) that is prepended back into `self.waiting` (`:518-519`). The
   file contains **no** reference to `self.skipped_waiting` (grep: 0 hits at `v0.18.0rc1`, `v0.19.1rc1`,
   `v0.20.2rc1`, `v0.21.0rc1`).
3. The paired vLLM `v0.18.0` added a second queue: `scheduler.py:168 self.skipped_waiting =
   create_request_queue(...)`, `add_request` → `_enqueue_waiting_request` (`:1722`) routes by status
   (`:1538-1542`), and **`Request.__init__` sets `status = WAITING_FOR_FSM` for any request carrying
   structured outputs** (`request.py:83` builds the `StructuredOutputRequest`; `:104-105` the status).
4. Upstream promotes it from inside the method the subclass replaced: its loop is
   `while (self.waiting or self.skipped_waiting)` (`:557`) and `_try_promote_blocked_waiting_request`
   (`:568-570` → `:2067-2072`) flips the request to `WAITING` once its FSM has compiled.
   `get_num_unfinished_requests` still counts `skipped_waiting` (`:1825-1830`), so the engine keeps stepping
   and returns empty steps forever.
5. Net: a guided request is born in a queue the override never reads, so every schema in the example hangs
   from the first request, and only where the env var is set. The same trap catches the other two blocked
   statuses (`WAITING_FOR_REMOTE_KVS`, `WAITING_FOR_STREAMING_REQ`) — a KV-transfer or streaming-input
   deployment with this env var on those tags stalls the same way.

Same class, same fix family, three cherry-picks of one change: `0a66a5c4a` (#10509, "0.18" branch,
2026-06-17 — in **no tag** of this clone, i.e. after `v0.18.0rc1` and after `v0.18.0`), `fc9230c42`
(#10559, main, first tag `v0.23.0rc1`) and `de3256fa1` (#10699, first tag **`v0.22.1rc1`**). Each is titled
"Align custom schedulers with vLLM <x> skipped_waiting queue for structured output" and each patches
`patch_balance_schedule.py` **and** `core/scheduler_dynamic_batch.py` (≈70 lines apiece) — so a
`SLO_limits_for_dynamic_batch` deployment on the same tags stalls identically.

| tier | change | cost / status |
| --- | --- | --- |
| drop the env var | unset `VLLM_ASCEND_BALANCE_SCHEDULING` → upstream `Scheduler` | loses the DP balance scheduling; this is the reporter's own A/B (`proposed` for them, `verified` as the gate) |
| port the fix | the `de3256fa1`/`0a66a5c4a` hunk: replace the copied waiting loop with `while (self.waiting or self.skipped_waiting)` plus `_select_waiting_queue_for_scheduling()` and `_try_promote_blocked_waiting_request`, and collect into `self.skipped_waiting` | local patch, two files if `scheduler_dynamic_batch` is in use; `proposed` |
| upgrade | a build with #10509 (0.18 line), or `v0.22.1rc1`+ (#10699), or `v0.23.0rc1`+ (#10559) | version jump |

Falsifier: with `VLLM_ASCEND_BALANCE_SCHEDULING` unset the same guided request still hangs ⇒ not this class
(then it is the bitmask path — `grammar-bitmask-and-dep-skew.md` — or the FSM compile itself).
Recheck: `git show <ref>:vllm_ascend/patch/platform/patch_balance_schedule.py | grep -c skipped_waiting` —
`0` before the fix, `2` after; and read the *paired* vLLM tag's `Request.__init__` status assignment, since
the trap needs a request born in a blocked status.

## Worked case #3462 (bge-m3 on `v0.11.0rc0`, 910B4, CANN `8.2.RC1`)

- Reported: `vllm serve /[HOME_DIRECTORY_1]/bge-m3 --enforce-eager` (1 card visible), then
  `curl /v1/embeddings` returns 400 `EngineCore encountered an issue`; the server log has the `core.py:112`
  warning, `core.py:121`, `Only "last" pooling supports chunked prefill and prefix caching; disabling both.`
  and dies at `vllm_ascend/core/scheduler.py:209` with `IndexError: tuple index out of range` (identical
  frames from the `EngineCore_DP0` process traceback). The trailing `tbe/…/repository_manager_log.py:60`
  `IndexError` is CANN's teardown noise.
- [verified] the line is exact at `v0.11.0rc0`, `v0.11.0rc1`, `v0.11.0rc2`, `v0.11.0rc3`, `v0.11.0` and the
  classes above; absent from `v0.12.0rc1` because the scheduler is gone.
- [reported] thread: one comment (`Tflowers-0129`, 2026-05-28) 「请问在新版本中重新尝试过吗」 — a
  "please retry on a newer version" hint, no diagnosis. Relay it as the pin lever, not as the mechanism.
- Why the `enabled` flag is true although the paste has no `--additional-config`: the pasted command is
  abbreviated and the warning line is the only proof needed; do not answer "this scheduler is not in use".

## Other gates worth knowing

- `max_num_batched_tokens < max_model_len` with chunked prefill off raises the `ValueError` above at config
  time — the opt-in scheduler forces `enable_chunked_prefill = False` unless the same block sets it true, so
  a small `--max-num-batched-tokens` cannot be combined with a large `--max-model-len`.
- `decode_max_num_seqs` and `enable_pd_transfer` are read only by the Ascend scheduler
  (`docs/source/user_guide/configuration/additional_config.md:68-69`), so a report that toggles them while
  the `core.py:112` warning is absent is testing nothing.
- `policy` must stay `fcfs`: `core/schedule_config.py:100` raises `AscendScheduler only supports fcfs policy`.

## Falsifier and recheck

Falsifier for the `blocks[0]` class: the same model on the same tag with `ascend_scheduler_config` removed
serves embeddings — if it still dies at `:209`, the model has KV groups and the empty-tuple premise is wrong
(then check `num_kv_cache_groups` on the failing rank instead of the pooling story).

Recheck at the reported tag: `git show <ref>:vllm_ascend/core/scheduler.py | grep -n
'new_computed_blocks'`, `git show <ref>:vllm_ascend/platform.py | grep -n -A6
'ascend_scheduler_config.enabled'`, and the upstream `kv_cache_manager`/`core.py` lines from the **paired**
vLLM tag (`../vllm-ascend/version-pairing.md`) — both move between releases. For the signature member the
pair is `git show <ref>:vllm_ascend/core/<subclass>.py | sed -n '/def __init__/,/)/p'` against the paired
tag's `vllm/v1/core/sched/scheduler.py` `def __init__` block: a subclass that lists keywords by hand drifts
whenever upstream inserts one, and the siblings' `*args, **kwargs` (or an optional keyword with a comment
naming the upstream PR) is the pattern to point at.
