# Pre-graph-mode releases: date the engine from the metric line, then price eager decode

Scope: a *low-utilisation / low-throughput* report on a v0.8.x–v0.9.x-era image — "NPU
utilisation < 15 % AI Core", "生成吞吐只有几十 tokens/s", "还有什么参数要调" — where the
paste carries a metrics block and an `npu-smi` screenshot. Recorded #1138: vllm-ascend
v0.8.5rc1 (`ec27af3`), vllm v0.8.5.post1, DeepSeek-R1-Distill-Qwen-32B (dense Qwen2),
TP=4 on 4 of 8 910B, `--gpu-memory-utilization 0.98 --max-num-seqs 1024`.

This is the *older* sibling of `index.md`'s rules: those assume the V1 logger, ACL graphs
and `max_cudagraph_capture_size`. None of the three exist on these releases, so the lever
list is different and the newer pages will misfire.

## Step 1 — the metric line dates the engine, and the engine picks the lever list

- **V0 (legacy) logger:** `… Running: %d reqs, Swapped: %d reqs, Pending: %d reqs, GPU KV
  cache usage: %.1f%%, CPU KV cache usage: %.1f%%.` — `vllm/engine/metrics.py:489` at vllm
  v0.8.5.post1.
- **V1 logger:** `… Running: %d reqs, Waiting: %d reqs, … Prefix cache hit rate: %.1f%%` —
  `vllm/v1/metrics/loggers.py:115-117` at the same pin.

`Swapped`/`Pending`/`CPU KV cache usage` are printed by V0 only, so their presence in the
paste is proof of the legacy engine — one line, no source walk. It also tells you the
*release could not have captured an ACL graph*: at v0.8.5rc1 the tree says
`pytest.ini:60` "`VLLM_USE_V1=1` is not supported with device type=npu", and graph mode is
torchair behind an `additional_config` key, not ACL graph (step 3).

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

## Step 3 — on these releases eager is not a flag the reporter chose

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
`sequence-count-cliff-capture-size.md` here — that machinery postdates these pins.

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
