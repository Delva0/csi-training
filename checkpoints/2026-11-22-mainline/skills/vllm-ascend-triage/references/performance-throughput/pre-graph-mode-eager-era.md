# Pre-graph-mode releases: date the engine from the metric line, then price eager decode

Scope: a *low-utilisation / low-throughput* report on a v0.8.x–v0.9.x-era image — "NPU
utilisation < 15 % AI Core", "生成吞吐只有几十 tokens/s", "还有什么参数要调" — where the
paste carries a metrics block and an `npu-smi` screenshot. Recorded #1138: vllm-ascend
v0.8.5rc1 (`ec27af3`), vllm v0.8.5.post1, DeepSeek-R1-Distill-Qwen-32B (dense Qwen2),
TP=4 on 4 of 8 910B, `--gpu-memory-utilization 0.98 --max-num-seqs 1024`.

Second recorded case, later release: #2099 — vllm-ascend v0.9.1rc1 (`c30ddb83`) + vllm v0.9.1
(`b6553be1`), a ~1B Qwen-VL-class model served online on one 910B3, title
"多模态在线推理性能较差，显卡利用率很低", the only log evidence an OCR'd metrics block plus an
`npu-smi` screenshot. The two generations need opposite answers: on v0.8.x there is no usable V1
to move to (step 3), on v0.9.x the serve is V0 only because nobody exported `VLLM_USE_V1=1`
(step 3b).

This is the *older* sibling of `index.md`'s rules: those assume the V1 logger, ACL graphs
and `max_cudagraph_capture_size`. On v0.8.x none of the three exist; on v0.9.x the V1 logger and
the ACL graph do exist but are off until `VLLM_USE_V1=1` (step 3b). Either way the lever list is
different and the newer pages will misfire. **Older than both** — a v0.7.x rc pin, e.g.
vllm-ascend `v0.7.3rc1` — has none of this machinery to switch on (no `enforce_eager`
override, no graph key, no V1 worker): go to `pre-0.8-vl-vs-transformers.md` instead of
reading step 3 or 3b, and do not carry `VLLM_USE_V1=1` backwards.

## Step 1 — the metric line dates the engine, and the engine picks the lever list

- **V0 (legacy) logger:** `… Running: %d reqs, Swapped: %d reqs, Pending: %d reqs, GPU KV
  cache usage: %.1f%%, CPU KV cache usage: %.1f%%.` — `vllm/engine/metrics.py:489` at vllm
  v0.8.5.post1, `:418` at vllm v0.9.1.
- **V1 logger:** `… Running: %d reqs, Waiting: %d reqs, … Prefix cache hit rate: %.1f%%` —
  `vllm/v1/metrics/loggers.py:115-117` at v0.8.5.post1, `:120-124` at v0.9.1.

`Swapped`/`Pending`/`CPU KV cache usage` are printed by V0 only, so their presence in the
paste is proof of the legacy engine — one line, no source walk. A paste that instead
carries `npu is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.` dates itself
the same way, and that line is **not** a platform verdict: it comes from upstream
`_warn_or_fallback` on non-CUDA hardware and an exported `VLLM_USE_V1=1` clears it
(`../startup-failures/profile-run-logits-gather-eof.md` §1) — so "the engine is V0
because V1 does not work on NPU" is never a citable conclusion. It is logged at **INFO**
(`_warn_or_fallback`, `arg_utils.py:1645-1657` at vllm v0.9.1: unset → `logger.info` + `should_exit
= True`, set to 1 → a "considered experimental" warning and V1 continues), which is why it scrolls
past in a long startup log — on a v0.9.x paste it is the *finding*, not background (step 3b).

On **v0.8.x** the same line also tells you the release could not have captured an ACL graph: at
v0.8.5rc1 the tree says `pytest.ini:60` "`VLLM_USE_V1=1` is not supported with device type=npu",
and graph mode is torchair behind an `additional_config` key, not ACL graph (step 3).

## Step 2 — `Pending: 0` closes the "what caps my concurrency" question arithmetically

In V0 `Pending` is the waiting queue and `Swapped` the preemption-swapped count. A paste
with `Running: N`, `Pending: 0`, `Swapped: 0` and `GPU KV cache usage: 0.2 %` says the
engine had **nothing queued** and ~500× KV headroom: no server-side cap, N is what the
client offered. Ask for the load driver (its concurrency setting, number of client
workers) — `--max-num-seqs` is a ceiling, not a target, and raising it moves nothing when
`Pending` is 0.

Then do the per-request arithmetic that the utilisation complaint hides: #1138's block was
`Avg generation throughput ≈ 57 tok/s` with `Running: 6` → **9.5 tok/s per request**, which
is the number to compare against the card's bandwidth floor, not the AICore %.

## Step 3 — on v0.8.x eager is not a flag the reporter chose

At v0.8.5rc1 `vllm_ascend/platform.py:124` sets `enforce_eager = True` **unconditionally**
(comment: "Override the value of enforce_eager to True before the CANN and torch_npu
support NPU compilation"), `:133` logs `Compilation disabled, using eager mode by default`,
and `:126-129` prints a warning naming the only opt-in:
`additional_config={'enable_graph_mode': True}` for **V0**. Two gates silently void that
key, each with its own warning to grep for in the reporter's log:
`supports_dynamo()` i.e. torch < 2.4 (`:151-155`) and `envs.VLLM_USE_V1` (`:156-160`, "NPU
graph mode is still experimental and not supported for V1 … disabled automatically").
Graph mode there is `torchair` over the whole forward in the V0 runner
(`vllm_ascend/worker/model_runner.py:979-997`, `get_npu_backend`/`cache_compile`), with
`use_cached_npu_graph` alongside `enable_graph_mode` in the same `additional_config`.

Consequence for the answer: "low AICore + low per-request tok/s on a release that forces
eager" is dispatch-bound, and the levers are (1) the graph-mode opt-in if the log shows it
is applicable, (2) TP/device layout, (3) a newer release. Do not cite
`max_cudagraph_capture_size`, the ACL-graph capture ceiling or
`sequence-count-cliff-capture-size.md` here — that machinery postdates these pins. Everything in this
step is **v0.8.x**: at v0.9.x `enforce_eager` is read from the model config instead, and the graph
arrives together with V1 (step 3b).

## Step 3b — v0.9.x: V0 only because NPU is a non-CUDA platform, so `VLLM_USE_V1=1` is the lever

Verified at vllm-ascend v0.9.1rc1 (`c30ddb83`) + vllm v0.9.1 (`b6553be1`), the #2099 pins:

- **`enforce_eager` is no longer forced.** `platform.py:146-151` reads it from the model config and
  `:161-181` splits the two modes: eager or `NO_COMPILATION` → `Compilation disabled, using eager
  mode by default` (`:164`), otherwise `PIECEWISE compilation enabled on NPU. use_inductor not
  supported - using only ACL Graph mode` (`:178`). Do not carry step 3's unconditional
  `enforce_eager = True` onto a v0.9.x paste.
- **V1 is available on this platform.** Ascend's `supports_v1()` returns `True`
  (`platform.py:254-258`), and the worker class is chosen by the env (`:186-187` →
  `vllm_ascend.worker.worker_v1.NPUWorker` when `VLLM_USE_V1`). The single reason a default serve
  comes up V0 is upstream's non-CUDA autoselect: `_is_v1_supported_oracle` computes `v0_hardware =
  not any(is_cuda, is_tpu, is_cpu+x86)` (`arg_utils.py:1443-1450`) and, with `VLLM_USE_V1` unset,
  `_warn_or_fallback(device_name)` returns `should_exit = True` → `use_v1 = False`
  (`:1026-1035`, `:1645-1657`).
- **The release asks for the env var in its own words.** `docs/source/user_guide/release_notes.md:52`
  (under `## v0.9.0rc1`): "V1 Engine is recommended to use. The code of V0 Engine is frozen and will
  not be maintained any more. Please set environment `VLLM_USE_V1=1`". The shipped VL tutorial
  (`docs/source/tutorials/single_npu_multimodal.md`) does **not** set it, so following the tutorial
  verbatim lands the reporter on the frozen engine — say that, it is why the paste looks "slow for no
  reason".
- **VL on V1 is the CI-tested path at this ref**, not a gamble to hedge: `.github/workflows/
  vllm_ascend_test.yaml` runs the singlecard e2e suite under `VLLM_USE_V1: 1`, and that suite
  includes `test_multimodal` on `Qwen/Qwen2.5-VL-3B-Instruct`
  (`tests/e2e/singlecard/test_offline_inference.py:39`, `:87-104`); the same file is re-run on V0
  only on the schedule.
- **What V0 costs a VL model** besides eager dispatch: `_set_default_args_v0` turns chunked prefill
  **off** for multimodal (`arg_utils.py:1455-1464`, "Chunked prefill not supported for Multimodal or
  MLA in V0"), so the paste shows one large-throughput prefill window followed by decode-only
  windows.
- **Is the rate abnormal?** The same release's own VL baseline is in-tree:
  `single_npu_multimodal.md:112` prints `output: 20.55 toks/s` for Qwen2.5-VL-7B on one card. A 1B
  model at ~28 tok/s single-stream is that order of magnitude, so the rate alone is not a defect —
  the finding is the engine, and the AICore % is step 4 arithmetic.

Two side notes that keep a v0.9.x answer from chasing the wrong thing: an extra body field such as
`add_generation_prompt` is a real `ChatCompletionRequest` field (`vllm/entrypoints/openai/
protocol.py:284` at v0.9.1), so it is not silently dropped and cannot explain a short answer — "只有
20 token" is a `finish_reason` question (`stop` = model/template, accuracy track; `length` = a cap).
And a `collect_env` that lists torch twice (pip `torch==2.5.1`/`torch-npu==2.5.1` beside conda
`torch 2.7.0`) is a hygiene item to confirm inside the serving env, not the perf cause.

## Step 4 — AICore % is a decode-shaped number, not a defect meter

At batch 6 a dense 32 B model's decode is HBM-bound: small skinny GEMMs, so a 12-13 %
AICore reading taken *while requests are running* is consistent with a healthy step. Say
that before proposing anything, then decide "dispatch-bound or bandwidth-bound" with one
measurement: op-level time and the launch gaps around each step. At v0.8.5.post1 the
online entry point is the env-gated pair — `if envs.VLLM_TORCH_PROFILER_DIR:` at
`vllm/entrypoints/openai/api_server.py:763` registering `POST /start_profile` /
`/stop_profile` (`:768-771`), i.e. the directory must be set **before** the server starts.
That env path is gone by v0.18 (replaced by `--profiler-config`), so quote the pin's own
line rather than the newer gate map in
`../observability-and-logs/online-profiling-endpoints.md`, which is verified only at
v0.18.0. The arithmetic is step time = `Running / avg_generation_throughput` against the
HBM floor for the per-card weights.

## Delivery

All three literal questions of this shape ("do extra flags help?", "what caps concurrency
at N?", "how do I profile?") answer from the metric line plus `platform.py` — two reads.
The recorded #1138 run instead walked the V0 runner, the V1 gating and the model config for
12 assistant turns, in reasoning blocks of 0.8–8.5 K characters, and was cut at the 16 384
token cap with **no answer text** (three status notes of 79/212/217 characters). The G1 sentence was available on its third turn:
`Running: 6 / Pending: 0 / KV 0.2 %` means nothing server-side caps concurrency, this
release runs eager, and the measurable lever is the graph-mode opt-in.

#2099 repeated it on the other side of the split: the engine-dating sentence is three turns of work,
yet the run walked `arg_utils._is_v1_supported_oracle` → `supports_v1` → `_warn_or_fallback` →
`_set_default_args_v0` for ~8 turns with no user-visible text, then wrote the ranked answer last and
was cut inside it. The G1 sentence for its shape was available at turn 2: *the paste runs the frozen
V0 engine because on non-CUDA hardware V1 is off unless `VLLM_USE_V1=1`; export it and confirm the
metric line flips to `Waiting … Prefix cache hit rate` and the log gains `PIECEWISE compilation
enabled on NPU`.* Emit that, then keep digging only for the corrections.
