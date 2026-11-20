# Downstream scheduler fork missing an upstream request status

Recorded #3979: vllm-ascend `v0.11.0rc0` (`00ba07102212`) + vllm `v0.11.0rc3`
(`8ce5d3198d00`), QwQ-32B-w8a8 on Atlas 800I A2, "其他模型有同样问题". A plain chat
request is fine; the same request with `extra_body={"guided_choice": [...]}` kills
the server:

```
File ".../vllm/v1/engine/core.py", line 283, in step
  scheduler_output = self.scheduler.schedule()
File ".../vllm_ascend/vllm_ascend/core/scheduler.py", line 274, in schedule
  raise RuntimeError(f"Invalid request status: {request.status}")
RuntimeError: Invalid request status: WAITING_FOR_FSM
...
vllm.v1.engine.exceptions.EngineDeadError
INFO: POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
```

This is **downstream-owned drift**, not a structured-output/xgrammar, model,
quantization or hardware fault. `AscendScheduler` is a fork of upstream's v1
scheduler; upstream added a request status and the fork never learned it, so the
first request carrying a grammar reaches a whitelist that only knows `WAITING` and
`PREEMPTED` and raises inside EngineCore's step loop.

## Two-gap mechanism (verified at the pins above)

| Fact | Anchor |
|---|---|
| The whitelist accepts only `WAITING` / `PREEMPTED`; everything else raises | `vllm_ascend/core/scheduler.py:269-274` |
| The same function **does** handle the P/D wait state, so the omission is selective, not "the fork has no status logic" | `vllm_ascend/core/scheduler.py:128-136` (`WAITING_FOR_REMOTE_KVS`) |
| Upstream marks any request with `sampling_params.structured_outputs` as this state | `vllm/v1/request.py:75`, enum at `:208` |
| Upstream's gate: grammar compiled → demote to `WAITING`, else skip and retry next step | `vllm/v1/core/sched/scheduler.py:357-366` |
| Upstream fills the SchedulerOutput payload: `structured_output_request_ids` + `grammar_bitmask` | `vllm/v1/core/sched/scheduler.py:579-598`, helper `get_grammar_bitmask` at `:831-859` |
| Upstream advances the FSM after sampling | `vllm/v1/core/sched/scheduler.py:946`, `:1086` |
| The fork hardwires the payload empty in the SchedulerOutput it builds | `vllm_ascend/core/scheduler.py:475-476` |
| The fork has **no** `get_grammar_bitmask` and no `should_advance` call (`grep -n '    def '` on both files shows the method-set gap) | fork `def` list: `__init__`, `schedule`, `_check_watermark_for_prefill`, `_get_prompt_limit`, `finish_requests`, `update_from_output` |
| The worker half is already ported and gated on the payload | `vllm_ascend/worker/model_runner_v1.py:1706 apply_grammar_bitmask`, applied only `if scheduler_output.grammar_bitmask is not None` at `:2022-2023` |

**The second gap is the one to warn about.** Adding `WAITING_FOR_FSM` to the
whitelist stops the crash but leaves the payload empty and the FSM unadvanced, so
guided decoding silently stops constraining anything — a correctness bug that is
harder to find than a traceback. Any fix has three sites (status gate, bitmask
computation before building the SchedulerOutput, `should_advance` in
`update_from_output`), and the general rule holds beyond this case: **when a fork
drops an upstream state, enumerate every field that state fills, not just the branch
that reads it.**

## Is the fork even in play? Yes by default at this pin

The natural first guess — "the reporter must have enabled the Ascend scheduler" — is
**wrong at v0.11.0rc0**, and the code says so:

- `vllm_ascend/platform.py:125` builds `AscendConfig` from `--additional-config`
  (`enabled` defaults `False`, `ascend_config.py:169`);
- `platform.py:139-150` then **forces** `ascend_scheduler_config.enabled = True`
  whenever the model is not MLA, `async_scheduling` is off, the model is not
  multimodal, `structured_outputs_config.backend == "auto"` (the default),
  `scheduler_delay_factor` is 0, `send_delta_data` is off and `policy == "fcfs"`.
  The assignment is unconditional, so an explicit `"enabled": false` in
  `--additional-config` is overwritten — a config-only opt-out does not exist there;
- `platform.py:288-293` then swaps `vllm_config.scheduler_config` for
  `AscendSchedulerConfig`, whose dataclass sets
  `scheduler_cls = "vllm_ascend.core.scheduler.AscendScheduler"`
  (`core/schedule_config.py:32-33`);
- the same block logs `Non-MLA LLMs forcibly disable the chunked prefill feature …`
  — that startup line is the marker that the block ran, and the cheapest thing to
  ask for when the reporter's build is unknown;
- **v0.11.0rc1 has no such block**: `grep -n "ascend_scheduler" vllm_ascend/platform.py`
  at `c5fe179cef7f` (read from the #4309 snapshot, not this case's) shows only the
  opt-in gate at `:288`. So the
  force-enable is a short-lived rc0 behaviour: while it exists, a *default* dense
  serve hits this bug; afterwards the fork must be requested (the tutorials and
  `feature_guide/graph_mode.md:48,55`, `tutorials/multi_node.md:139,180`,
  `multi_node_pd_disaggregation.md:131,171,210`, `multi_node_kimi.md:96,136` are
  where reporters do request it).

**Cheapest next action, first paragraph of the answer.** Get the fork out of the
path, using the escape that matches their build:

1. One grep decides which world they are in:
   `grep -n "ascend_scheduler_config.enabled = True" <site-packages>/vllm_ascend/platform.py`.
   Present → step 2; absent → step 3.
2. Build carries the force-enable (rc0): changing `ascend_scheduler_config` cannot
   help, so change a *gate input* — `--structured-outputs-config
   '{"backend": "xgrammar"}'` (the default is `"auto"`; flag registered at vllm
   `vllm/engine/arg_utils.py:938`) or `--async-scheduling`. Label these two as
   reporter-side flags to verify, not as measured fixes, and re-run the same
   `guided_choice` script.
3. Fork is opt-in: drop `ascend_scheduler_config` from `--additional-config`
   (or set `enabled: false`) and restart; upstream's `Scheduler` has the FSM branch.

Only if the reporter needs the fork (graph-mode performance, a recipe that requires
it) does the three-site patch matter — and say plainly that a partial patch degrades
to the silent non-enforcement above.

## Capability verdict and coverage

- `docs/source/user_guide/support_matrix/supported_features.md:20` —
  `Guided Decoding | 🟢 Functional`, and `feature_guide/structured_output.md`
  documents xgrammar/guidance usage, **with no note that it excludes the Ascend
  scheduler**. So: works on upstream's `Scheduler`, broken whenever the fork is in
  the path — forced at rc0 for a default dense serve, opt-in afterwards — and the
  doc/matrix should say so.
- `tests/e2e/singlecard/test_guided_decoding.py` runs the default config, and the
  e2e files that do enable the fork (`test_ascend_scheduler.py`, `test_chunked.py`,
  `test_prefix_cache*.py`) never use `guided_*` — CI cannot see the combination. That
  is the coverage answer when the reporter asks why a released version does this.

## Noise that looks like evidence

The shutdown tail of the same paste carries
`nanobind: leaked 2 instances! … GrammarMatcher / CompiledGrammar` and leaked
functions. That is xgrammar teardown noise after EngineCore is already dead, and it
also proves the grammar **did** compile, so the backend is not the owner. Same family
as rule 11 in `SKILL.md`: name the line you trust and say why. Do not build a theory
on it, and do not send the reporter to xgrammar or the model.

## Procedure (3 calls after the classification)

1. `grep -n '    def '` on the pinned upstream scheduler and on the fork; the fork is
   missing every method whose name belongs to the feature (`get_grammar_bitmask`), and
   its `schedule()` has fewer status branches. Print both lists.
2. `grep -n "WAITING_FOR_FSM\|structured_output\|grammar" vllm_ascend/core/scheduler.py`
   and the upstream call sites; record where the state is set, gated, filled and
   advanced.
3. Confirm the worker half is ported but payload-gated
   (`grep -n "grammar_bitmask" vllm_ascend/worker/model_runner_v1.py`) — that is what
   turns "patch the raise" into "patch three places", and it is the sentence the
   maintainer needs.

## Version scope and recheck

- Mechanism verified at vllm-ascend `v0.11.0rc0` / vllm `v0.11.0rc3`; the rc1
  comparison comes from the `v0.11.0rc1` tree in the #4309 snapshot. The reporter was
  on `0.11.0` final, which this snapshot does not carry, so whether the final release
  keeps the rc0 force-enable or the rc1 opt-in behaviour — and whether the FSM status
  is handled there — is **open**: phrase it as the grep in step 1 of the escape,
  never as a finding.
- Recheck in one command at the reporter's ref:
  `grep -n "WAITING_FOR_FSM" $(python -c "import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))")/core/scheduler.py`
  — a branch plus a non-empty `grammar_bitmask` computation means fixed; only a
  passed-through status means the silent variant; no hit at all means still crashing.
- Sibling forks with the same hazard: the `npu_input_batch.py` copy of upstream's
  `gpu_input_batch.py` (`../pooling-embedding/batch-index-desync.md`) and every
  `vllm_ascend/patch/**` copy (`patch-copy-vs-upstream.md`). A whole-file fork in
  `vllm_ascend/core/` is a third site; a state/status/enum that upstream grew after
  the fork was taken is the recurring failure mode.
