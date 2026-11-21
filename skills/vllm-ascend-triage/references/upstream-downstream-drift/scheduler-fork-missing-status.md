# Downstream scheduler fork missing an upstream request status

Recorded twice, at two pins, with the same gap underneath (the *signature* variant of the same family — a fork that stops binding the base's constructor keyword — is `fork-ctor-arg-sweep.md`, #18688 at v0.23.0):

- **#3979** — vllm-ascend `v0.11.0rc0` (`00ba07102212`) + vllm `v0.11.0rc3`
  (`8ce5d3198d00`), QwQ-32B-w8a8 on Atlas 800I A2, "其他模型有同样问题";
- **#3904** — vllm-ascend `v0.10.2rc1` (`048bfd555319`) + vllm `v0.10.2`
  (`01efc7ef7813`), GLM-4.5-Air (non-MLA MoE), 8×910B3, TP8, serve command not
  pasted; the reporter's traceback ends at `scheduler.py, line 260`, which is the
  `raise` in this tree. Same shape: ordinary chat fine, a 500 on the requests that
  carry `response_format`/guided parameters, which is why it reads as 偶发.

A plain chat request is fine; the same request with `extra_body={"guided_choice":
[...]}` kills the server:

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

Same two gaps at `048bfd5`: the whitelist `raise` is
`vllm_ascend/core/scheduler.py:255-260`, the payload is hardwired empty at
`:459-460` (`structured_output_request_ids={}, grammar_bitmask=None`), the `def` set
is the same six methods, and the worker half is ported but gated
(`vllm_ascend/worker/model_runner_v1.py:1537 apply_grammar_bitmask`, called only
`if scheduler_output.grammar_bitmask is not None` at `:1814-1815`). So this is not a
one-release regression: it predates v0.11.

**The second gap is the one to warn about.** Adding `WAITING_FOR_FSM` to the
whitelist stops the crash but leaves the payload empty and the FSM unadvanced, so
guided decoding silently stops constraining anything — a correctness bug that is
harder to find than a traceback. Any fix has three sites (status gate, bitmask
computation before building the SchedulerOutput, `should_advance` in
`update_from_output`), and the general rule holds beyond this case: **when a fork
drops an upstream state, enumerate every field that state fills, not just the branch
that reads it.**

## Is the fork even in play? Yes by default at both recorded pins

The natural first guess — "the reporter must have enabled the Ascend scheduler" — is
**wrong at v0.11.0rc0 and at v0.10.2rc1**, and the code says so:

- `vllm_ascend/platform.py:125` builds `AscendConfig` from `--additional-config`
  (`enabled` defaults `False`, `ascend_config.py:169`);
- the force-enable block then **assigns** `ascend_scheduler_config.enabled = True`
  whenever the model is not MLA, the model is not multimodal, the guided-decoding
  backend is `"auto"` (the default), the delay-factor input is 0,
  `send_delta_data` is off and `policy == "fcfs"` (`:139-150` at `00ba0710`,
  `:136-145` at `048bfd5`; the delay/backend spellings differ per pin — next
  table). The assignment is unconditional, so an explicit `"enabled": false` in
  `--additional-config` is overwritten — a config-only opt-out does not exist there;
- `platform.py:288-293` then swaps `vllm_config.scheduler_config` for
  `AscendSchedulerConfig`, whose dataclass sets
  `scheduler_cls = "vllm_ascend.core.scheduler.AscendScheduler"`
  (`core/schedule_config.py:32-33`);
- the same block logs `Non-MLA LLMs forcibly disable the chunked prefill feature …`
  — but that INFO sits on the **outer** `not use_mla` condition
  (`platform.py:141-144` at `00ba0710`, `:136-140` at `048bfd5`), *before* the gate
  above. It prints for every non-MLA model even when the gate fails, so **it does
  not prove the fork is in play** — and it still prints after the reporter applies
  the `backend` escape below, so its continued presence is not evidence the escape
  failed. The gate test is the assignment line
  (`grep -n "ascend_scheduler_config.enabled = True" …/platform.py`) or the
  reporter's own successful `guided_*` request;
- **v0.11.0rc1 has no such block**: `grep -n "ascend_scheduler" vllm_ascend/platform.py`
  at `c5fe179cef7f` (read from the #4309 snapshot, not this case's) shows only the
  opt-in gate at `:288`. So the
  force-enable spans v0.10.2rc1 → v0.11.0rc0 and is gone by rc1 (#3904 and #3979
  are the same bug on either side of that window): while it exists, a *default*
  dense serve hits this bug; afterwards the fork must be requested (the tutorials and
  `feature_guide/graph_mode.md:48,55`, `tutorials/multi_node.md:139,180`,
  `multi_node_pd_disaggregation.md:131,171,210`, `multi_node_kimi.md:96,136` are
  where reporters do request it).

**Cheapest next action, first paragraph of the answer.** Get the fork out of the
path, using the escape that matches their pin. **Which gate inputs exist — and what
the escape flag is called — follows the installed *vllm* version, not the
vllm-ascend one**, because the same vllm-ascend tree serves both:

| Pin | backend gate input | escape that clears the gate |
|---|---|---|
| `048bfd5` (v0.10.2rc1) + vllm 0.10.2 | `decoding_config.backend == "auto"` (`platform.py:141-142`) | `--guided-decoding-backend xgrammar` (vllm 0.10.2 `config/__init__.py` `DecodingConfig.backend = "auto"`, flag `arg_utils.py:615`), or `--scheduler-delay-factor 0.1` — `delay_factor` **is** a real 0.10.2 `SchedulerConfig` field (`config/scheduler.py:85`), and the gate reads it directly (`not scheduler_config.delay_factor > 0`). `--async-scheduling` is **not** in the gate at this pin |
| `00ba0710` (v0.11.0rc0) + vllm 0.11.x | `platform.py:134-137` picks the object by version: `if vllm_version_is("0.10.2"): …decoding_config else …structured_outputs_config`, then tests `.backend == "auto"` | `--structured-outputs-config '{"backend": "xgrammar"}'` on vllm 0.11 (or `--guided-decoding-backend …` if the image pairs it with vllm 0.10.2), or `--async-scheduling` (it removes the whole block at this pin). The `--scheduler-delay-factor` route is **dead here**: vllm 0.11.0rc3's `SchedulerConfig` has no delay-factor field and `arg_utils.py` no such flag, while the gate asks `getattr(scheduler_config, "scheduler_delay_factor", 0)` — a name nothing sets, so it always evaluates 0 |

1. One grep decides which world they are in:
   `grep -n "ascend_scheduler_config.enabled = True" <site-packages>/vllm_ascend/platform.py`.
   Present → row of the table; absent → fork is opt-in.
2. Fork is opt-in: drop `ascend_scheduler_config` from `--additional-config`
   (or set `enabled: false`) and restart; upstream's `Scheduler` has the FSM branch.
3. An explicit `"enabled": false` is overwritten wherever the force-enable exists,
   so *changing `ascend_scheduler_config` is not an escape there* — move a gate
   input instead, and label it a reporter-side flag to verify rather than a measured
   fix: re-run the same `guided_choice`/`response_format` request after it.

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

- Mechanism verified at vllm-ascend `v0.11.0rc0` / vllm `v0.11.0rc3` and at
  `v0.10.2rc1` / vllm `v0.10.2` (#3904); the rc1 comparison comes from the
  `v0.11.0rc1` tree in the #4309 snapshot. The reporter was
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
