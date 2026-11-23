# `VLLM_ASCEND_BALANCE_SCHEDULING=1` admits nothing: the patch's cross-rank guard

## Scope

A synchronous, single-device serve starts cleanly, `/health` answers 200, and an
ordinary generation request is **never admitted**: zero computed tokens, the request
sits in `waiting` until the client gives up, nothing is logged, no traceback and no
device fault. Two matched controls finish normally — balance scheduling off at the same
capacity, or the same env var at capacity 2. The launch may set a small
`--max-num-seqs` deliberately, for latency.

Pins for every `path:line`: **vllm-ascend v0.18.0rc1 (`99e1ea0fe685`)** with
**vllm v0.18.0 (`bcf2be961200`)**. The guard exists only in this downstream patch, so
grep the reporter's own tree for it before quoting a line — the file has moved and the
comparison has changed since (rule 6).

## Step 1 — the flag pair is the first question, not the KV pool

Recorded #17953 (one 910B3, aarch64, Qwen3-0.6B bf16, TP1/DP1, eager, synchronous
scheduling, `max-model-len` 4096): `VLLM_ASCEND_BALANCE_SCHEDULING=1` +
`--max-num-seqs 1` wedges; env unset at `--max-num-seqs 1` serves; env set at
`--max-num-seqs 2` serves. Three launches settle it, and the extra discriminators are
cheap: **every** request fails (a 7-token one too), not just a long one, and the
startup `GPU KV cache size` / `Maximum concurrency … : kx` lines show a pool that
could hold the request. That is what separates this page from
`waiting-queue-wedge.md`, whose gate is per-request demand against the pool.

Also read *which* scheduler the engine actually built — do not infer it from the env
var: `patch/platform/__init__.py:38-39` imports the module when the env flag is truthy
(`envs.py:105`, default `0`), and the module's tail rebinds the class globally,
`vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler` (`patch_balance_schedule.py:706`).

## Step 2 — the guard, source-verified at the pin

```python
while self.waiting and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:      # :266  legitimate capacity gate
        break
    balance_flag = max(t.item() for t in self.balance_queue) >= self.max_num_running_reqs - 1   # :269
    if balance_flag:                                        # :270
        break                                               # before peek_request() at :273
```

`balance_queue` is constructed as **zeros**, one CPU int tensor per DP rank
(`:49-52`). At `max_num_seqs=1` the comparison is `0 >= 0` → `break` on every call, so
the waiting loop is dead for every request. The `:266` gate does not fire on an idle
scheduler (nothing is running), so nothing else can admit the request.

## Step 3 — check the writer before calling the value transient

`balance_queue`'s only writer is `balance_gather` (`:54-56`, an `all_gather` of
`len(self.running)` over `dp_group`), and it is called from exactly one place:
`BalanceDPEngineCoreProc.run_busy_loop` (`:631`). The patched `run_engine_core`
instantiates that class **only** when
`parallel_config.data_parallel_size > 1 or dp_rank > 0` (`:673-681`) and otherwise keeps
plain `EngineCoreProc`. On a DP=1 serve, therefore:

- the queue stays zeros for the life of the process, so the capacity-1 `break` is
  permanent — not a first-step artifact; the reporter's own late invocation capture
  (`"balance_queue": [0]` when the request was already thousands of steps old) is the
  confirmation;
- at `max_num_seqs ≥ 2` the guard is **inert** (`0 >= N-1` is never true and nothing
  writes the queue), so on DP=1 the env var buys no balancing at all and only adds the
  capacity-one hazard. That is why the capacity-2 control matches the env-off arm
  exactly rather than merely happening to.

This is the transferable method for any downstream admission guard that reads shared
state: prove (a) the writer is reachable in the configured topology, and (b) the
initial value does not already satisfy the break condition, before ranking thresholds
or calling the state stale.

## Step 4 — scope it: not DP slot arithmetic

`max_num_running_reqs` is `scheduler_config.max_num_seqs` verbatim
(`vllm/v1/core/sched/scheduler.py:105` at the pin), not divided by
`--data-parallel-size`, and each rank builds its own zeros — so the capacity-one wedge
reproduces per rank at any DP size, not only in the DP=1 evidence the report carries.
Nothing rejects `--max-num-seqs 1`: `vllm_ascend/platform.py` touches `max_num_seqs`
only to match ACL-graph capture sizes — the default capture bound
`min(max_num_seqs * decode_query_len, 512)` (`:182-191`) and the sequence-parallel size
filter (`:326-334`); read both, since the first is the one that sets the bound. Say
both, because "capacity 1 is a legitimate latency setting" is the reason the
consequence matters.

## Step 5 — correspondence: downstream-owned, so no upstream fix will arrive

At the vllm pin the upstream scheduler has no balance/wave logic at all
(`rg -n 'balance|wave' vllm/v1/core/sched/scheduler.py` returns nothing), and
`patch/__init__.py:82-91` records this file as a downstream patch of upstream vllm PR
29721 ("Future Plan: Remove this patch when vLLM merge the PR") — the threshold is the
fork's own number. The doc note that goes with it
(`docs/source/tutorials/models/DeepSeek-R1.md:123`, same text in the V3.1 and GLM4.x
tutorials) says the feature "may help increase output throughput and reduce TPOT …
However, TTFT may degrade", i.e. the cross-DP trade is documented, and it is the trade
a fix has to make explicitly.

## Step 6 — the repair and what stays unverified

The filing states that a later PR (#8674, into `releases/v0.18.0`) reverted the
comparison to `== self.max_num_running_reqs`, motivated by a W4A8 TTFT degradation.
That form does repair capacity 1 — `0 == 1` admits, and the `:266` gate still enforces
the slot — and leaves the DP=1 `≥2` path inert. What it changes is **cross-DP**: a
remote rank at full capacity now stops local admission instead of keeping one slot
free. Two limits: the snapshot is a single commit, so the diff cannot be read here, and
PRs/comments are unavailable — label the merged fix as reported-by-the-filing, not
verified, and give the check that settles the trade on the reporter's box (same
concurrency and same model on the two builds, compare TTFT and admission).

Coverage at the pin is four nightly e2e configs and nothing else —
`tests/e2e/nightly/single_node/models/configs/{Kimi-K2.5,GLM-4.7}.yaml` and
`tests/e2e/nightly/multi_node/config/{DeepSeek-V3.1-BF16,DeepSeek-R1-W8A8-A2}.yaml`,
all at `--max-num-seqs 16` or the default, with no unit test in `tests/ut/` — so "a
capacity-one regression test" is a supportable ask and "nightly e2e covers it" is not.

## Redirects

- The request **is** admitted but generation never advances → the per-request KV-state
  page (`kv-block-lifetime-async-scheduling.md`), not this one.
- Only long/oversized prompts fail and short ones are served → the pool gate
  (`waiting-queue-wedge.md`).
- `Running` > 0 with a queue that drains slowly → `../performance-throughput/index.md`.
- The env var appears in a *different* failure (a crash, a wrong answer, a hang under
  load): it is a config axis to ablate, not this lane — the scheduler patch only
  decides admission. Recorded cases of that kind: `../pd-disaggregation/pd-vs-colocated-ttft.md`
  (TTFT claim), `../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` (hang under
  load), `../observability-and-logs/env-var-validation-warning.md` (validation warning).

## Answer shape

Verdict first: `real, pin-scoped code defect in the downstream balance-scheduling
patch, admission-only — workaround --max-num-seqs ≥ 2 or unset the env var at that
release`. Then the guard line with the zeros initialiser, the DP=1 writer gate that
makes it permanent (and the inertness above capacity 1, which explains the control
arm), the DP-size widening, the reported fix with its cross-DP semantics labelled
unverified, and the coverage gap that justifies the regression test.
