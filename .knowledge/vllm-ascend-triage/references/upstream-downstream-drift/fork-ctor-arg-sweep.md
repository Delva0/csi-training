# A downstream fork of an upstream base class lags a new constructor keyword

Worked case #18688, vllm-ascend `v0.23.0`
(`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`) + vllm `0.23.0`
(`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`), image
`quay.io/ascend/vllm-ascend:v0.23.0`, one 910B3, Qwen3-0.6B, BF16, TP=PP=DP=1,
eager, prefix caching off, `additional_config={"SLO_limits_for_dynamic_batch":
1.0}`. Recheck every `path:line` at another pin.

```text
TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument 'hash_block_size'
```

`LLM(...)` loads the model, sizes the KV cache, then EngineCore dies building the
scheduler; the parent exits with status 1 and no `generate()` runs. The reporter
had already read the constructor correctly. Confirm it and answer — this shape is
**deterministic on that image**, not a consequence of the reporter's flag values.

## The answer shape

The engine always passes `hash_block_size` to whatever `scheduler_cls` names, and
exactly one fork in the tree never learned the parameter. So: (a) yes, the crash is
real at the pin, for every configuration that selects that scheduler; (b) the
repair is the three-line signature change its **sibling fork in the same release**
already carries; (c) the reason it shipped is a coverage gap — nothing in `tests/`
constructs that class. Do not present "swallow the keyword with `**kwargs`" or a
`getattr` guard as the fix; the sibling's own comment states the intended contract.

## Procedure (each step one call)

1. **The message is the claim.** `TypeError: <Class>.__init__() got an unexpected
   keyword argument '<kw>'` names the callee and the argument. Read the callee
   `path:line` in the reporter's vllm-ascend tree and confirm neither `<kw>` nor
   `**kwargs` is in its `__init__` — that closes question (a) without any
   version archaeology.
2. **Split the caller from the callee, in the vllm tree.** Find who passes it at
   that pin: `rg -n '<kw>' vllm/ | head`. Two different answers, and they change
   the wording of the verdict:
   - passed **unconditionally** by the engine's construction call (below) ⇒ every
     run of the image fails, whatever `additional_config`/`--flag` the reporter
     used. Say "deterministic", and do not repeat the reporter's "this
     configuration" framing;
   - passed only under a branch ⇒ the branch is the gate; name it and stop
     guessing at the config.
   The class the engine instantiates is the one the config string resolves to
   (`get_scheduler_cls` → `resolve_obj_by_qualname`), so confirm the reported class
   name *is* that target and not a base it happens to print. Also read the **base**
   class the fork inherits: if upstream declares `<kw>` with a default
   (`int | None = None`), a missing argument is a lag, not a required contract
   change — which is what makes the optional-default repair safe. Upstream says the
   same thing in one line worth quoting (`config/scheduler.py:177-184`: *"This
   scheduler interface is not public and compatibility may not be maintained."*).
3. **Sweep every sibling fork of the same base — this is the step that pays.**
   `bash <skill_root>/scripts/fork-ctor-sweep.sh <vllm-ascend-root> --base
   '^Scheduler$' --kwarg <kw>` (the script follows subclasses of a subclass, so an
   `Async…` twin is not missed; the base regex is matched on the tail name of each
   base expression). It prints, per subclass, `OK` (declares it, with the
   declaration's line), `IMMUNE` (`**kwargs`), `MISSING` (neither — the bug set)
   or `NO-CTOR`, plus the in-tree fix pattern. Two findings come out of that one
   call and both belong in the answer:
   - the **fix pattern**, quoted from the sibling that already does it — here
     `vllm_ascend/core/scheduler_profiling_chunk.py:62-64` with its own comment
     *"`hash_block_size` was added in vLLM #40946; keep it optional so the
     subclass works on both pinned vllm and main"*, and
     `patch/platform/patch_balance_schedule.py:63,73`. This is the project's own
     statement of intent and it turns "please redesign" into "one branch of a
     sweep was missed";
   - the **bounded repair set**: the subclasses taking `*args, **kwargs`
     (`core/recompute_scheduler.py:95,98` and `:1124,1125`) are immune and need no
     edit. Without the sweep you cannot say which other fork the reporter will hit
     next.
   The sweep is only meaningful for the base the caller in step 2 actually
   instantiates: pointed at an abstract base whose constructor the engine never
   calls with that kwarg, it will still print `MISSING` for stubs that merely
   declare `__init__(self)`. Read the caller's `Scheduler(...)`/factory call, not
   the count, and report the subclasses the engine can select.
4. **Ask whether the tree is the producer of the value.** Here it is:
   `vllm_ascend/patch/platform/patch_kv_cache_utils.py:112-147` re-implements
   `resolve_kv_cache_block_sizes` (because vLLM PR #40860 forbids multiple KV
   groups with CP, which Ascend supports — the rationale is written out in
   `vllm_ascend/patch/__init__.py:237-250`), and `:345` plus `:356-359` install it
   on `vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes` **and** on the
   reference `vllm.v1.engine.core` imports directly. So downstream manufactures
   `hash_block_size` and hands it to the very call that fails to bind it on a
   downstream class. That is the ownership sentence; it also rules out "upstream
   changed an interface we don't control".
5. **Check which fork the reporter actually selected, and whether anything tests
   it.** `platform.py:713-718` sets `scheduler_cls` to the dotted string when
   `ascend_config.SLO_limits_for_dynamic_batch != -1` (default `-1`,
   `ascend_config.py:238`) and force-enables chunked prefill; the next block
   (`:721-725`) can overwrite `scheduler_cls` with `ProfilingChunkScheduler`, so
   the last assignment decides — read both before naming the class. Then
   `rg -ln '<ClassName>' tests/` (0 hits here) and the one UT that touches the
   flag pins the **disabled** value (`tests/ut/test_platform.py:58`
   `SLO_limits_for_dynamic_batch = -1`): CI only ever exercises the path where the
   fork is never constructed. That is "why did this ship", and the regression test
   you recommend constructs the class with the upstream caller's kwargs instead of
   going through the flag.
6. **After the constructor, read the class's own degrade path before promising the
   feature works.** Here `BudgetRefiner.__init__` sets `self.enabled = slo_limit >
   0` (`:39-41`) and `_read_lookup_table` (`:54-68`) logs
   `The dynamic batching feature requires the lookup table 'profile_table.csv'`
   and sets `self.enabled = False` when `vllm_ascend/core/profile_table.csv` is
   absent; the observable afterwards is `Table miss for ctx=… dnum=…` (`:98`).
   The documented prerequisites are `docs/source/user_guide/feature_guide/dynamic_batch.md:12-20`
   (download `A2-B3-BLK128.csv`, rename into that path, install `pandas`) and the
   flag is documented as an **integer** with `-1`/`0` disabled and `> 0` enabled
   (`:22-30`), which is worth one line when the reporter passed `1.0`. So
   "the engine starts after the fix" is not "dynamic batch is active".

## Facts at this pin (all re-verified)

| Claim | Anchor |
|---|---|
| The engine instantiates the *named* class and passes the kwarg unconditionally | `vllm/v1/engine/core.py:136` `Scheduler = vllm_config.scheduler_config.get_scheduler_cls()` → `:149-157` (`block_size=scheduler_block_size, hash_block_size=hash_block_size`) |
| Upstream states the fork contract itself | `vllm/config/scheduler.py:168-188`: the dotted string is resolved by `resolve_obj_by_qualname`, and `logger.warning_once("Using custom scheduler class %s. This scheduler interface is not public and compatibility may not be maintained.")` (`:177-184`) — quote it: nobody upstream promised a frozen constructor, so a fork that stops matching the base is downstream's to fix |
| Values come from a resolver that a downstream patch also replaces | `vllm/v1/core/kv_cache_utils.py:593-618`; `vllm_ascend/patch/platform/patch_kv_cache_utils.py:109-147` |
| Upstream base accepts it, optional | `vllm/v1/core/sched/scheduler.py:66-76` (`block_size: int`, then `hash_block_size: int \| None = None`) |
| The fork does not, and has no `**kwargs` | `vllm_ascend/core/scheduler_dynamic_batch.py:122` (`__init__` at `:126-135`) |
| Sibling fork already fixed, with the upstream PR in a comment | `vllm_ascend/core/scheduler_profiling_chunk.py:56-79` |
| Second fixed sibling | `vllm_ascend/patch/platform/patch_balance_schedule.py:57-75` |
| Two immune siblings (`*args, **kwargs`) | `vllm_ascend/core/recompute_scheduler.py:95-99`, `:1124-1126` |
| Selector | `vllm_ascend/platform.py:713-725`, `vllm_ascend/ascend_config.py:238` |
| No test constructs the class | `rg -ln 'SchedulerDynamicBatch' tests/` → no hits; `tests/ut/test_platform.py:58` pins `-1` |

## Boundaries

- `<Ascend…Impl>.__init__` and `forward_oot` mismatches are the attention-impl
  routing question, not this page: `impl-and-oot-signature-drift.md`.
- A fork whose *state machine* lags (a new request status, a changed return
  container) rather than its argument list: `scheduler-fork-missing-status.md`,
  `legacy-wrapper-contract-audit.md`.
- The same sweep generalises to any base class a downstream module subclasses
  (`BaseModelLoader`, `AttentionBackend`, a coordinator, an executor): the base
  regex is the only thing that changes.

## Recheck

`rg -n 'hash_block_size' vllm_ascend/core/scheduler_dynamic_batch.py` at the ref the
reporter deploys; whether the upstream addition had a PR of its own (the sibling
comment names #40946) and whether the release branch received the same three lines;
and, when the report asks about *intended handling* of a feature whose removal has
already merged on main, answer from the pinned release (the class exists, the docs
still document it, no test covers it) and label main's status as not settleable
from a release snapshot — PRs and their merge dates are outside it.
