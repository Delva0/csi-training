# Operator-substituting accelerator plugin vs the compiled graph (`mindie_turbo`)

## Symptom this page answers

The engine dies while the **model forward is traced by Dynamo**, and the message
names a symbol from an optional acceleration wheel:

```text
torch._dynamo.exc.Unsupported: Graph break due to unsupported builtin
mindie_turbo._ops.PyCapsule.rms_norm. This function is either a Python builtin
… or a third-party C/C++ Python extension (perhaps created with pybind).
```

The frames above it are `torch/_dynamo/**` and, near the top,
`vllm/compilation/decorators.py:231-238 patched_inline_call` →
`return inline_call(parent, func, args, kwargs)` — that is upstream
`support_torch_compile`'s `__call__` on its first traced call, so the failure is
*compile-time*, not a device op. Recorded #1534 (vllm-ascend 0.9.1rc1, vllm
0.9.1, 910B, CANN 8.1.RC1): offline `LLM("Qwen/Qwen2-7B-Instruct").generate(...)`
failed this way with `mindie_turbo: 2.0rc2` installed, while
`vllm serve Qwen/Qwen2-7B-Instruct` on the same host was reported to work.

**Variant B — nothing raised, and the claim is "turbo 没有生效".** Recorded
#2451 (vllm-ascend v0.9.1rc2 `b9f715d6ce30` + vllm v0.9.1 `b6553be1`, 910B,
`VLLM_USE_V1=1 … --enforce-eager`, Qwen2.5-32B TP4): "vllm 0.9.1 + turbo 没有吞吐
提升, vllm0.7.3 + turbo 2.0.rc2 有提升", and the reporter's own note is the answer
— the `INFO … utils.py:33] MindIE Turbo is installed. vLLM inference will be
accelerated with MindIE Turbo.` line they saw on 0.7.3 is **absent** from the
0.9.1 serve log. Per §1 that sentence is the only probe, so the wheel never
loaded in that process and both 0.9.1 runs measured one baseline (their 756.7 vs
739.0 tok/s is noise). Do not answer this variant from §2/§3 alone, and do not
accept "the model is not supported": the read is the missing probe line, then the
two causes in §1 (import raised and was swallowed — #1144 is the recorded form,
`ImportError: … undefined symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs`,
and `ImportError` *is* an `Exception` — or the wheel is not visible to that
interpreter), each settled by one `python -c "import mindie_turbo"` **in the
launching interpreter**. Second, independent finding in that case: the 0.9.1
baseline was measured with `--enforce-eager`, which removes the very path
`faqs.md:72` names as the 0.9.x performance source, so re-run without it before
comparing against a 0.7.3 number.

Verified anchors below are at vllm-ascend `c30ddb8` = v0.9.1rc1 and vllm
`b6553be` = v0.9.1 (the #1534 case pins); variant B's pins are vllm-ascend
`b9f715d6ce30` = v0.9.1rc2 + the same vllm. Re-grep before quoting at another ref.

**Third shape — a *quality* complaint on a paired pre-0.8 release**, e.g. "some results
have no content" under concurrency with `mindie_turbo` installed on v0.7.3.post1, does
**not** belong on this page. Check the one gate first: at ≤ v0.7.3.post1
`vllm_ascend/platform.py:110-114` forces `NO_COMPILATION` for every config, so §2's
"compile is on by default" and §4's log pair cannot apply and there is no capture lane to
rank. The possible lanes there are eager op substitution and the plugin-owned w8a8
quantizer → `mindie-turbo-pre-0.8-eager-and-quantizer.md` (verified at `c69ceaca1ecc`,
#1954).

## 1. Which process imported the plugin

- Both workers **unconditionally** try to import it at init:
  `vllm_ascend/worker/worker_v1.py:78-81` (V1) and
  `vllm_ascend/worker/worker.py:84-87` (V0), through
  `vllm_ascend/utils.py:172-182` — `importlib.util.find_spec` then
  `importlib.import_module`, then
  `logger.info("MindIE Turbo is installed. vLLM inference will be accelerated with MindIE Turbo.")`
  (`utils.py:180`), and `except Exception: pass` around all of it (at ≤ v0.7.3.post1 the
  helper takes a third argument and logs `warning_once(exception_info)` instead —
  `utils.py:23-35` there, see `mindie-turbo-pre-0.8-eager-and-quantizer.md`). The helper
  **moves**: the same `try_register_lib` is `utils.py:172-182` at v0.9.1rc1
  `c30ddb8` and `utils.py:75-85` at v0.9.1rc2 `b9f715d6ce30` (worker call sites
  `worker_v1.py:78` / `worker.py:85` at rc1, `worker_v1.py:97-100` /
  `worker.py:84-87` at rc2) — grep `def try_register_lib` instead of quoting a
  line number, and never read "not at that line" as "no such helper".
- No flag, env var or config gates it: `grep -rn mindie vllm_ascend/` at this ref
  returns only those two worker sites and
  `vllm_ascend/quantization/quantizer.py:49-55` (the w8a8 quantizer path). So if
  the wheel is installed in an interpreter, **every** model and both entry points
  load it.
- That INFO sentence is the probe for "did *this* process have the plugin?".
  Absent from a log ⇒ the plugin never reached that process (another interpreter,
  another `HOME`/user site — `pip install --user` puts it in
  `~/.local/lib/python3.x/site-packages`, another container), and the same run
  says nothing about the plugin.

## 2. Why the graph break is fatal instead of a fallback

- Compile is on by default **at these 0.9.x pins**: `vllm_ascend/platform.py:162-182`
  keeps `CompilationLevel.PIECEWISE` and logs
  `PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode`
  (`:177-182`); `enforce_eager` / `NO_COMPILATION` instead logs
  `Compilation disabled, using eager mode by default` (`:163-164`). The same method at
  v0.7.3.post1 (`platform.py:110-114`) is the opposite gate — it rewrites every level to
  `NO_COMPILATION` — so read `check_and_update_config` before assuming either default. ACL-graph
  *capture* is a second, narrower gate:
  `vllm_ascend/worker/model_runner_v1.py:325-327` (`level == PIECEWISE and not
  model_config.enforce_eager`), so "capture is off" does not imply "compile is
  off".
- Upstream compiles the forward with
  `fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE`
  (`vllm/compilation/wrapper.py:51`), whose default is `"1"`
  (`vllm/envs.py:311-313`). With `fullgraph=True` a graph break is raised as
  `Unsupported`; it would otherwise be a silent graph break with an eager
  fallback for that region.
- The traced object is the model class itself (`vllm/model_executor/models/qwen2.py:265`
  `@support_torch_compile`, i.e. the same forward for both entry points), so at
  these pins the code has no serve-vs-offline branch: an asymmetry has to come
  from the *process* (§4), not from the engine path.
- The substituted attribute is a pybind `PyCapsule`, i.e. not a
  `torch.library`/`custom_op` registration, so Dynamo has no rule for it
  (`torch/_dynamo/variables/functions.py`, the `unimplemented(msg)` at the end of
  the pasted trace). The trigger is therefore **[plugin installed] × [compile
  enabled]**, and nothing in vllm-ascend turns either off selectively.

## 3. The pairing verdict (the reporter's literal question)

- `docs/source/developer_guide/versioning_policy.md:23-30` at v0.9.1rc1 (the
  file moves to `docs/source/community/versioning_policy.md:23-32` at v0.9.1rc2,
  same rows): the MindIE Turbo column is **empty** for every row from `v0.8.4rc2`
  upward, and `2.0rc1` appears only beside `v0.7.3` / `v0.7.3.post1`. At v0.9.1rc1 there is no matched
  mindie-turbo (maintainer, same thread: *"0.9.1rc1 has no matched mindie-turbo
  now"*).
- `setup.py:362` at v0.9.1rc1 is `extras_require={}` — the
  `mindie_turbo: ["mindie-turbo==2.0rc1"]` extra exists only at the older refs
  (`setup.py:371` at v0.7.3.post1, see `abi-undefined-symbol.md`).
- `docs/source/faqs.md:72` at this ref: *"From 0.9.0rc2, Qwen and Deepseek works
  with graph mode to play a good performance. What's more, you can install
  `mindie-turbo` with `vllm-ascend v0.7.3`"* — on 0.9.x the plugin adds nothing
  this tree documents.
- **A newer plugin against a newer release is answerable, and the answer is no.**
  A pairing question about the *reported* release is read from a **newer** export,
  because both artefacts accumulate: at vllm-ascend v0.11.0rc3
  (`docs/source/community/versioning_policy.md:25-39` — the file has left
  `developer_guide/`, so `rg --files docs | rg versioning_policy` locates it) every
  row from `v0.8.4rc2` up still carries an **empty** MindIE Turbo cell —
  `v0.9.1rc2` (`:32`) included — and `2.0rc1` still sits only beside `v0.7.3` /
  `v0.7.3.post1`; the same pin's `docs/source/user_guide/release_notes.md:141`
  (under `## v0.9.1 - 2025.09.03`) retires the wheel by name: *"Mindie Turbo is not
  needed with this release. And the old version of Mindie Turbo is not compatible.
  Please do not install it. Currently all the function and enhancement is included
  in vLLM Ascend already."* So "is 2.0rc2 the pair for 0.9.1rc2?" is a **no**, in
  the project's own words — not the vendor's upstream table, and never "the pinned
  tree cannot answer this". `../startup-failures/environment-correspondence.md`
  carries the same fact from the other direction (a 0.9.1+ stack carrying the wheel
  is a compatibility question, not an old release line).

## 4. When another entry path works with the same install

#1534's follow-up is the important half: the reporter says the offline example
fails but `vllm serve` runs, and asks whether serve simply does not enable ACL
graph. Do **not** answer "the plugin breaks graph mode" without accounting for
that; rank the asymmetry itself as evidence.

**Answer that hypothesis in the same message — it is refutable at the pin, and
leaving it as "send me the log" is what drew the re-ask.** The entry point is not
a compilation knob: `check_and_update_config` reads only
`model_config.enforce_eager` + `compilation_config.level` (`platform.py:151-182`),
V1-vs-V0 comes from the same `EngineArgs._is_v1_supported_oracle` for `LLM()` and
for `vllm serve` (vllm `envs.py:616-617` defaults `VLLM_USE_V1` to `"1"`; a V0
fallback prints `… is not supported by the V1 Engine. Falling back to V0.` from
`arg_utils.py:1633-1642`, so that line's absence proves V1 ran), and the only
entry-point-dependent defaults at vllm 0.9.1 are `max_num_batched_tokens` /
`max_num_seqs` (`vllm/engine/arg_utils.py:1554-1567`). A timing argument then
closes the lane: both paths run a dummy forward **inside engine init** — the V1
worker's `profile_run()` (`worker_v1.py:144`) → `_dummy_run(self.max_num_tokens)`
(`model_runner_v1.py:1789-1815`, its own comment *"Trigger compilation for general
shape"*) and `compile_or_warm_up_model` (`worker_v1.py:194-205`) → `capture_model()`
(`model_runner_v1.py:2036-2072`) — so a serve process that came up at all has
already traced that forward under the same `fullgraph` setting. "在线推理默认没开 acl graph" therefore cannot be the
difference; what is left is the process (§1), an eager serve, or a different
model/tag.

Candidates, cheapest first, each decided by one grep of the **working** process's
log:

1. That process never imported the plugin — no `MindIE Turbo is installed. vLLM
   inference will be accelerated with MindIE Turbo.` line.
2. That process ran without compile — `Compilation disabled, using eager mode by
   default` (`platform.py:163-164`), where the offline run shows `PIECEWISE
   compilation enabled on NPU`; a serve that carries `torchair_graph_config`
   without MLA takes the narrower fallback `Torchair graph mode is still
   experimental … Fallback to eager mode.` (`platform.py:153-159`).
3. It compiled but the traced region never contained the substituted op (a
   different model/config family) — only then are the two runs about different
   code.

The discriminating unit is the pair of log lines, so ask for both. Meanwhile the
offline ablation `LLM(..., enforce_eager=True)` reproduces the serve shape and
proves the trigger is the compile step rather than the plugin on its own.

## 5. Answer ladder

0. Variant B first, because it is one grep: is the `MindIE Turbo is installed.`
   line in the serve log at all? Absent ⇒ the plugin never loaded, so there is
   nothing to measure; then `python -c "import mindie_turbo"` in the launching
   interpreter, and drop `--enforce-eager` before quoting any 0.9.x throughput
   (that flag selects the pre-graph baseline, not the release's performance).
1. `enforce_eager=True` / `--enforce-eager` on the failing path — the accepted
   workaround at this ref, and the one-line test of the mechanism.
2. `VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE=0` — turns the hard error into a graph
   break (hypothesis, not observed in the recorded thread).
3. Uninstall `mindie-turbo`, or move to the release where the plugin is paired
   (≤ v0.7.3.post1) — for Qwen-family graph mode on 0.9.x this costs nothing
   (`faqs.md:72`).
4. If the plugin is wanted anyway: `pip show mindie-turbo`,
   `python -c "import mindie_turbo"` in *that* interpreter, plus the two log
   greps of §4.

## Scope and limits

- `mindie_turbo`'s own source is not in the case export, so *which* operators it
  replaces is known only from the symbol in the message; do not enumerate them.
- The `Unsupported` text and the `fullgraph` default are upstream torch/vllm
  behaviour at these pins; a ref using `VLLM_COMPILE_...` or an upstream default
  flip changes rung 2, so re-read `vllm/compilation/wrapper.py` there.
- #1534's serve-works observation is reporter-supplied and was never explained in
  the thread: the mechanism of the asymmetry stays **open**, but the *entry point*
  is not one of its candidates (§4: one config gate, same V1 selection, same
  init-time dummy forward), so §4's three remaining lanes are the check that
  settles it, not a conclusion.
