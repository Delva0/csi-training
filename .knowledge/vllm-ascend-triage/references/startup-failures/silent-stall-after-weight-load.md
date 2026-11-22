# The log simply stops after `Loading model weights took`

Scope: the pasted log ends at `Loading safetensors checkpoint shards: 100%
completed` + `[model_runner.py:NNN] Loading model weights took X.XXXX GB`, nothing
is raised, nothing else is printed, and `curl` to the port answers
`connection refused`. The process may still be alive and working, or it may have
been killed without a message — the paste cannot tell, so the deliverable is the
phase bound plus the one artefact that separates the two, not a mechanism.

Not this topic: `EngineCore failed to start` / `WorkerProc initialization failed`
(→ `silent-worker-exit.md`), a paste whose last line is `Using OOT custom backend
for compilation` (→ `compile-backend-banner-then-silence.md`: that banner is
printed before the shards are read, not after the weights are down), a traceback
(→ the matching page in `index.md`), or a box that dies *before* `Starting to
load model`
(→ `../runtime-memory/attention-mask-host-allocation.md`).

## Step 1 — bound the phase from the pinned worker, then say it

At vllm-ascend **v0.8.4rc1** (`086423d`, the pin for recorded #698) with the V0
engine, the weight-load line is `vllm_ascend/worker/model_runner.py:904` and the
next code is deterministic:

| Order | Site | What can stall there |
|---|---|---|
| 1 | `vllm_ascend/worker/worker.py:240-282` `determine_num_available_blocks` → `self.model_runner.profile_run()` (`:256`) | one dummy **prefill** of the whole model, then `torch_npu.npu.synchronize()` (`model_runner.py:1078`) — a device op that never completes shows as silence here. Size that batch from the serve line instead of by feel: its **total** token count is `max_num_batched_tokens`, split across `max_num_seqs` sequences (`model_runner.py:1001-1036`, each `seq_len = max_num_batched_tokens // max_num_seqs + …` at `:1031` sums back to `max_num_batched_tokens`), and when unset it is `max(max_model_len, 2048)` on this pin's V0 path (`vllm/config.py:1865-1882`; `_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` at `:67`) because V0 leaves chunked prefill off by default (`vllm/engine/arg_utils.py:1568-1600` `_set_default_args_v0`, whose >32K auto-enable is CUDA-only). So `--max-model-len 10000` is a 10 K-token dummy prefill while `--max-model-len 131072` is a 131 K-token one: the same paste means two different phases |
| 2 | `worker.py:260-282` — `mem_get_info()`, then `assert peak_memory > 0` (`:264`) | an `assert` message would be the next line; silence means it has not reached it |
| 3 | `worker.py:284` `initialize_cache` → KV allocation | an allocation failure prints `EL0004`/`207001`, so silence is not this |
| 4 | `worker.py:325-326` `_warm_up_model` | **nothing** — the comment reads "model capture is not supported, thus we just set seed here", so on this version the capture lane is not a candidate |
| 5 | upstream `LLMEngine`/`EngineCore` `init engine (profile, create kv cache, warmup model) took … s` (`vllm/engine/llm_engine.py:449`, `vllm/v1/engine/core.py:163`), then `Starting vLLM API server on http…` (`vllm/entrypoints/openai/api_server.py:1081`) | the port opens only after this line, so "refused" is expected while any of 1-4 runs |

That table *is* the answer's first paragraph: "your last line is X, the next
expected line is Y, so you are inside the profiling forward (or the process is
gone) — two commands tell which". A report of this shape that is answered with a
mechanism instead of a phase bound is a guess.

## Step 2 — the V0/V1 fingerprint is free evidence; label it as evidence

The reporter's own line prefix fingerprints the engine generation at that ref:
`vllm_ascend/worker/model_runner.py:904` is the **V0** runner and
`worker/model_runner_v1.py:781` the V1 one, and `vllm/worker/model_runner.py` of
vllm v0.8.4 does not carry the string at all (only `hpu_model_runner.py`,
`xpu_model_runner.py` do). So `[model_runner.py:904]` says V0 — which V1 evidence
would have contradicted.

The pin's own docs agree that V1 is opt-in here, so vllm's `envs.py:76,527-528`
default of `VLLM_USE_V1: bool = True` is not the operative one: the release note
says "By default, vLLM will fallback to V0" (`docs/source/user_guide/release_notes.md:9`)
and `pytest.ini:60` records "VLLM_USE_V1=1 is not supported with device type=npu".

Caveats to state with it: the log prints the **basename** only, line numbers move
between releases, and a partial export weakens an absence claim. Use it to pick
which post-load path to bound (V0: `profile_run` → `synchronize`; V1:
`_dummy_run` → `GPU KV cache size: N tokens`, `vllm/v1/core/kv_cache_utils.py:634`),
never as proof of the exact image.

## Step 3 — one artefact request, then the ladder

Ask for, in this order (the recorded maintainer reply to #698 was exactly the
first one, and the reporter supplied it the next day):

1. **The full `vllm serve` line** — it decides model, `--max-model-len`, `--dtype`,
   TP, and which of the flags the reporter believes are active. Without it the
   report is not answerable; ask for it instead of ranking causes.
2. Liveness and death, one command each: the container's exit code / `docker ps -a`
   (was it killed?), `dmesg -T | grep -iE 'out of memory|oom-kill'`, and
   `npu-smi info` taken **while** it is stalled.
3. A rerun with `VLLM_LOGGING_LEVEL=DEBUG` and, only for attribution,
   `ASCEND_LAUNCH_BLOCKING=1`. Do **not** lead with `--enforce-eager`: at this pin
   the platform already sets `simple_compile_backend = "eager"`
   (`vllm_ascend/platform.py:59`), so on this version it is a no-op and reads as
   not having read the code.
4. The effective length budget (`Using max model len N`) and a rerun with a smaller
   `--max-model-len`, or the same length with an explicit small
   `--max-num-batched-tokens` (e.g. 2048) — those two are the only knobs that shrink
   the Phase-1 forward, because they are what sets the dummy batch's token count
   (previous section). Two knobs that do **not**: `--max-num-seqs` only re-splits
   that same total across more or fewer sequences (`model_runner.py:1029-1036`,
   default 128 at `vllm/config.py:1768`), and `--gpu-memory-utilization` only scales
   the KV block count computed *after* the peak is measured (`worker.py:271-274`).
   Offering either as "the stall fix" is a wrong lever and wastes the reporter's
   round trip. Upstream states the first lever itself: a long `max_model_len` with
   chunked prefill off "may cause OOM during the initial memory profiling phase"
   (`vllm/engine/arg_utils.py:1602-1607`).

## Step 4 — what the paste cannot support

- Not "graph capture": step 4 above shows this path does not capture at this ref.
- Not "the model is unsupported": the weights loaded, so the model class, the
  device and the checkpoint all worked.
- Not an OOM and not a dead process until 2 above says which.
- Not "it hangs on request 1": the port never opened, so no request exists yet.

## Recorded case

**#698** image `vllm-ascend:v0.8.4rc1`, 910B4, DeepSeek-R1-Distill-Qwen-1.5B
(`--served-model-name deepseek-r1-distill-qwen-1.5b`), `--max-model-len 10000`,
`--dtype float16`, `--tensor-parallel-size 1`, `--enable-reasoning
--reasoning-parse deepseek_r1`: `Loading model weights took 3.3461GB`, no further
line, HTTP port refused. Both flags are valid at vllm v0.8.4
(`vllm/engine/arg_utils.py:216-217,979-994`; `--reasoning-parse` is an argparse
abbreviation of `--reasoning-parser`, whose `choices` are the registered parser
names), so the command is not the failure.

That serve line became public the next day (reporter comment, 2025-04-29), so read
it as a **size** before ranking anything:

```bash
vllm serve /llm --served-model-name deepseek-r1-distill-qwen-1.5b --enable-reasoning \
    --reasoning-parse deepseek_r1 --max-model-len 10000 --api-key REDACTED \
    --tensor-parallel-size 1 --dtype float16
```

No `--max-num-batched-tokens` + `--max-model-len 10000` ⇒ the dummy prefill is
`max(10000, 2048) = 10000` tokens, split over the default `max_num_seqs` into 39-40
tokens per sequence at 256, 78-79 at 128. Which divisor applies has to be read off
the pin: `SchedulerConfig`'s dataclass fallback is `128` (`vllm/config.py:1961-1965`),
but the V0 `EngineArgs` path overwrites it with `256` before the config is built
(`vllm/engine/arg_utils.py:1561-1563` `_set_default_args_v0`, verified at vllm
v0.8.5.post1), so a `// 128` split is the dataclass value, not the running one
(the same trap as the multimodal profile batch,
`../deploy-sizing/vl-profile-run-multimodal-budget.md`). A 10 K-token prefill of a
1.5 B model whose weights are 3.35 GB is a small batch beside any 910B4's HBM budget (32 GB on the common variant, and the
paste does not state which), so on this report "the profiling forward is too big
for the box" is the weakest row of the ranking; the live pair is a device op
that never returns versus a process killed with no line printed, which is why the
next turn asks for liveness (`docker ps -a` exit code, `dmesg`, `npu-smi` **during**
the stall) rather than for a smaller run. `--api-key` and `--served-model-name` play
no part in the phase.

The phase is therefore the profiling forward of `determine_num_available_blocks`
(or a killed process), and the artefact that moved the case was the serve line
itself. The mechanism was never established in the original session: it shipped no
answer (14 turns, 11 of them bare tool calls, ~47.6 K characters of reasoning
against 322 characters of text) and its last blocks were re-reading the V0/V1
question it had already settled. The replay with this page in the library shipped
the complete ranked answer on its 5th assistant turn in 4 tool blocks — owning page,
then one pin-verification call — which is the shape this page exists to make cheap:
when the paste matches this recorded case, the ladder *is* the message.
