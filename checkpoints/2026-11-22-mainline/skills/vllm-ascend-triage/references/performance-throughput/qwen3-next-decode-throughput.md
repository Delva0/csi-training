# Qwen3-Next-80B-A3B at ~5 tokens/s: single-stream decode on the documented command

## When this page applies

A hybrid GDN MoE (Qwen3-Next-80B-A3B and siblings) at TP=4 with `Running: 1`
and `Avg generation throughput` in the low single digits — recorded 3.6-4.7 tok/s
across the sampled windows at v0.11.0, 5.3 tok/s at v0.11.0rc2. The report is
either a bare "太慢 / too slow" with one metrics paste (#4357) or the performance
half of a two-part issue whose failing half is a Triton kernel compile (#5750 →
`../image-dependency-faults/triton-ascend-pair.md`).

Read `index.md` first for the metric rules. This page exists because the usual
"diff the command against the tutorial" answer is *nearly* empty here: the
tutorial carries no `--compilation-config`, so the one honest diff is graph mode,
and whether that diff is worth the restart is a version question with three
readable outcomes (rung 1).

## Arithmetic that decides the shape (do this before any source)

From the #4357 paste: `Running: 1`, `Waiting: 0`, `GPU KV cache usage: 0.5%`,
no `Preemptions` line, `Avg prompt throughput: 0.0` in the sampled windows.

- 5.3 tok/s ⇒ **~190 ms per decode step**, single request, no prefill competing.
  Queueing, capacity, `--gpu-memory-utilization`, `--max-num-seqs`, KV tokens and
  `--max-model-len` are therefore all out — say so and move on.
- `Loading model weights took 37.4525 GB` per card at TP=4 (bf16, ~150 GB total);
  a 3B-active MoE reads ~1.5 GB/card/token ⇒ **~1 ms at ~1.6 TB/s**.
- Two orders of magnitude of headroom means the step is **host-dispatch/latency
  bound**, not compute- or bandwidth-bound: that is the class of cost a captured
  graph removes, and nothing else on the launch line does.

## What the pinned version says (vllm-ascend v0.11.0rc2 `a2e4c3f`, vllm v0.11.0)

| Site | Fact |
|---|---|
| `docs/source/tutorials/multi_npu_qwen3_next.md:89` | the project's **own** serve command at this ref is `--tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.7 --enforce-eager`; the offline example repeats `enforce_eager=True` (`:134`) |
| `vllm_ascend/platform.py:151-153` | `if enforce_eager:` → `Compilation disabled, using eager mode by default` (the paste's `[platform.py:152]`) and `level = NO_COMPILATION` |
| `vllm_ascend/platform.py:207-208` | `FULL_AND_PIECEWISE` (which the upstream hybrid config auto-sets, "in order to optimize performance") is **downgraded to `PIECEWISE`** — "Full graph is fully supported later". This row covers **only** that one mode |
| `vllm_ascend/platform.py:213-215` | the success line to look for once eager is dropped: `PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode` |
| `vllm_ascend/platform.py:224-227` | an **explicit** `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` is *not* downgraded: it logs `FULL_DECODE_ONLY compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode` plus a red experimental warning ("capturing too many batch sizes can lead to OOM … or inference hangs", `:228-242`); any mode other than these two falls to `%s cudagraph_mode is not support on NPU. falling back to NONE` (`:242-246`) |
| `vllm_ascend/platform.py:249-258` | a non-`NONE` graph mode **raises** `ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1` — unset that env before testing the mode |
| vllm `vllm/config/__init__.py:336-341` | with `enforce_eager` off, `level` defaults to `PIECEWISE`, so the gate above leaves the explicit mode alone; with the flag on, `Cudagraph is disabled under eager mode` (`:378-381`) wipes it — which is why `--enforce-eager` and the mode are mutually exclusive |
| `vllm_ascend/worker/model_runner_v1.py:591-592` | `_use_aclgraph()` needs `cudagraph_mode != NONE and level == PIECEWISE and not enforce_eager` — true for that launch |
| `vllm_ascend/worker/model_runner_v1.py:3428-3433` | the refusal to expect when the model's attention builder declares `AttentionCGSupport.NEVER`: `ValueError(… CUDAGraphMode.FULL_DECODE_ONLY is not supported with <Builder> backend … please try cudagraph_mode=PIECEWISE …)`. `attention_v1.py:174-175` declares `UNIFORM_SINGLE_TOKEN_DECODE` — full graphs for uniform single-token decode, i.e. exactly a `Running: 1` run — and `mla_v1.py:181-182` declares `UNIFORM_BATCH` |
| `vllm_ascend/patch/worker/patch_triton.py:16` → `vllm_ascend/ops/fla.py:205` | prefill `chunk_gated_delta_rule` is rebound to `torch_chunk_gated_delta_rule`, a pure-torch fp32 chunk loop (`chunk_size=64`) — prefill/TTFT is dispatch-bound too |
| `vllm_ascend/ops/sigmoid_gating.py:45-46`, launched `:214` | decode keeps a Triton `fused_recurrent_gated_delta_rule_fwd_kernel` |
| `tests/` | **no** `qwen3_next` case at this ref, and the mode's own coverage is worse than "present": `tests/e2e/singlecard/test_aclgraph.py:103-108` marks the `FULL_DECODE_ONLY` variant `@pytest.mark.skip("… it fails intermittently and needs investigation")` (the same skip at v0.11.0rc1 `c5fe179`), and `tests/e2e/multicard/test_full_graph_mode.py` (Qwen3-30B-A3B, TP2, bf16, `max_model_len=1024`) is referenced by **no** `.github/workflows` file. So nothing at this pin ran the mode in CI |

Consequence: the reporter's command is not a misconfiguration, and a config diff
against this release's tutorial finds one thing only — the tutorial carries no
`--compilation-config` at all, so both it and the report run the *default* graph,
and the default full graph is the one this line downgrades. Nor is a published
number being missed: the tutorial has no throughput table. So the flag diff is
**not empty, it is one flag** (`FULL_DECODE_ONLY`), and that flag is an
experiment whose outcome at this pin is readable from a log line — not a promise
and not a dead end.

## The graph lane on this family has a recorded startup failure at this line

#3865 ("Qwen3-Next-80B-A3B-Instruct vllm-ascend0.11.0cr1图模式加载失败") is the only
recorded graph-mode *outcome* for this family on the 0.11 line: vllm-ascend
`0.11.0rc1.dev167+gcd58a643c`, TP=8, `--gpu-memory-utilization 0.5`,
`--additional-config '{"torchair_graph_config": {"enabled": true},
"ascend_scheduler_config": {"enabled": true}}'`. It dies in
`determine_available_memory` → `profile_run` → `_dummy_run` at
`vllm_ascend/models/qwen3_next.py:197` `assert isinstance(attn_metadata, dict)`
(`AssertionError`, then `EngineCore failed to start`), with `cudagraph dispatching keys
are not initialized. No cudagraph will be used.` earlier in the same log. That is
**torchair**, not ACL graph, so it does not predict what `FULL_DECODE_ONLY` does — but
no recorded run of *any* graph mode on this family at this line has loaded, so rung 1
is a probe to send **together with** rung 2, not a promise to send alone. Both lines
sit at the same numbers in the v0.11.0rc2 tree
(`vllm_ascend/models/qwen3_next.py:193` `if attn_metadata is None: return` for the
profile path, `:197` the assert), so when the reporter's graph attempt dies at startup,
look for this assert before reading the failure as a kernel or HBM fault.

## Ranked answer for this shape

1. **The one-flag in-version experiment: drop `--enforce-eager`, add
   `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, change nothing
   else.** At v0.11.0rc2 that mode is accepted (`platform.py:224-227`) and
   `_use_aclgraph()` is true for it (`model_runner_v1.py:591-592`), so this is the
   only in-version launch that touches the dominant term, and it costs one restart.
   With #3865 in hand, say the startup-failure branch is a recorded one for this
   family and keep the rung-2 ask in the same message rather than after the crash.
   Its three outcomes are each one log line, so say which line decides what:
   - `FULL_DECODE_ONLY compilation enabled on NPU …` **and** a faster
     `Avg generation throughput` on the same body ⇒ that is the finding; report the
     delta and note the red capture-size/OOM warning it printed.
   - `ValueError(… CUDAGraphMode.FULL_DECODE_ONLY is not supported with <Builder>
     backend …)` (`model_runner_v1.py:3428-3433`) ⇒ that model's attention builder
     refuses full graphs at this pin; stop experimenting and go to rung 2.
   - The banner prints and the run then **dies or hangs** (`507011`, `507053`,
     `another aivec`/`mte` fault, or a stall at the first replay) ⇒ this is the
     mode's known-broken lane, not the reporter's box: the **next** rc's Known
     Issues row (v0.12.0rc1 `42ceaf0`, `release_notes.md:42` — not this pin) reads
     "DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY graph mode …
     #4990", answered "we'll fix it in next release", and #5076's reporter
     confirmed 0.13.0rc1 fixed it
     (`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md` §0). Report that
     row and make ≥0.13.0rc1 the ask.
   Also flag the likely *first* failure before it happens: graph capture needs HBM
   this launch does not have — 37.45 GB/card of weights at
   `--gpu-memory-utilization 0.7` leaves ~4 GB free — and the notes file at this pin
   still carries the Qwen3-Next HBM row from the previous rc's section
   (`release_notes.md:64`, *under* `## v0.10.2rc1` at `:27`: "The HBM usage of
   Qwen3-Next is higher than expected … You can set `max_model_len` and
   `gpu_memory_utilization` to suitable value … to avoid oom error", #2884 — there is
   no v0.11 section, so quote its section with it). Cap the capture sizes / lower the
   utilization in the same message rather than after an OOM round-trip.
2. **Move the release, and use the newer tutorial command.** At v0.14.0rc1 the
   tutorial's Qwen3-Next command (quoted in `index.md`) has no `--enforce-eager`
   and carries the same `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
   plus `--max-num-batched-tokens 4096` — i.e. rung 1's mode on a tree with real
   coverage for it, plus a FLA path that is not the torch loop. This is the durable
   answer and also the move #5750's maintainer answer prescribes for the same
   model. It is what makes the answer survive rung 1 failing.
3. **Then the generic levers**, in the order of the arithmetic: a multi-prompt
   benchmark (`--num-prompts 8+`) for aggregate throughput, since single-stream
   decode is the worst case for a MoE at TP=4; a quantized W8A8 checkpoint if one
   exists for the model; expert-parallel or TP layout only after a measurement.
4. **Explicitly refuted by the paste**, so they do not belong in the answer as
   candidates: KV capacity/`--gpu-memory-utilization`, `--max-num-seqs`,
   `--max-model-len`, and `Prefix cache hit rate: 0.0%` (hybrid models are
   force-disabled in the config layer at this line —
   `zero-prefix-cache-hit-hybrid.md`).

## What the recorded outcome does and does not support

#4357's reporter answered the thread's 问题解决了吗? with **并没有** on 2025-11-24
(`#issuecomment-3569471787`), two days after opening, and no comment in between
reports a measured speedup. Read that as an *outcome* datum — this shape was still
unsolved on the reporter's box at that date — and **not** as "the graph lever was
tried and failed": the comment names no invocation and no version. So do not write
"removing `--enforce-eager` does not help" as a finding, and do not write "fixed by
X" either; the rung-1 experiment above is still the open, cheap check, and the
owning facts are the two rows this page pins (the downgrade, and the mode's
skipped e2e). What the 并没有 does support is the ordering in §*Ranked answer*: send the
probe and the version move in one message, since the only recorded graph outcome for
this family at this line is a startup failure (above, #3865).

The class-level form of this trap is `SKILL.md` rule 6 (*a page's provenance id is not a
precedent when it is the report you are answering*). One attribution trap, recorded
— a session answering this shape wrote to the reporter
with "同形态的已记录报告 #4357 里，报告者两天后回答并没有" — i.e. it cited the
thread it was replying to as a *different* recorded case and dated its own reporter
two days into the future. When the report in front of you **is** #4357 (or a
duplicate of it), say "as of 2025-11-24 this thread was still open" and never
"a recorded report says …": the reader is the reporter.

## Paste hygiene for this track

- The reporter's own extras are not the cause but are worth one A/B line:
  `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32` appears in #4357 and in no
  tutorial; `--max-model-len 32768` vs the tutorial's 4096 only moves KV size.
- **Confirm the metrics belong to the reported model.** #4357's log carries
  `served_model_name: Qwen3-235B-A22B-Instruct-2507` and a request for
  `Qwen3-Next-80B` that 404s, while the issue's command says
  `--served-model-name Qwen3-Next-80B-A3B-Instruct`: the throughput samples came
  from an instance whose served name differs from the reported command. Say it in
  one line and ask which invocation produced the numbers.
- A hand-built image that adds a bisheng toolkit and a triton wheel over the stock
  image (both recorded reports did) is a second variable on the same run. The
  isolation step is *that release's own install recipe*, not "no triton at all":
  read `requirements.txt` for `triton-ascend` first — absent (≤ v0.13.0rc1) means its
  pinned `.dev` wheel from the obs bucket is required, a final pin means none is —
  and the compile-failure page (`../image-dependency-faults/triton-ascend-pair.md`)
  is where that thread belongs.
