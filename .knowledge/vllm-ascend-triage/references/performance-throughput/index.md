# Throughput, queueing and cache-hit questions on a healthy engine

## Scope

The deployment is up, nothing crashed, and the report is about speed or cache
behaviour: "多并发的时候请求要排队那么久", "为什么缓存命中为 0", "生成吞吐只有个位数",
a benchmark table compared with a vendor number — **or a rate that drops in a
single step as one concurrency number is crossed** (256 running requests fine,
258 collapsed, no error). The evidence is the engine
metrics block (`Avg prompt throughput`, `Avg generation throughput`, `Running`,
`Waiting`, `GPU KV cache usage`, `Prefix cache hit rate`, sometimes
`Preemptions`), the serve command, and the prompt lengths of the load driver.

This is a *measurement* track, not a defect hunt: no `WorkerProc` line, no device
dump, no traceback. The deliverable is arithmetic over the pasted numbers, the
one or two flags that move each number, and the artefacts that are missing.

Not this topic, leave immediately:

- Requests eventually answer but the **scores** differ between runs → accuracy
  track, `../accuracy-benchmarks/index.md`.
- Throughput ~0 with requests staying `Running` and `Accepted: 0` / `Drafted: N` →
  speculative decoding, `../spec-decode-mtp/acceptance-collapse.md`.
- The service dies, OOMs or leaks after a while, or a rank dies under load →
  `../runtime-memory/index.md` / `../dsa-sparse-attention/concurrency-crash-with-no-log.md`.
- The reporter is planning a deployment, not measuring one → `../deploy-sizing/index.md`.

## Rules

- **Read the metric block arithmetically before any source.** Four numbers carry
  the whole answer: `Waiting` (queue depth, and whether it *drains*), `GPU KV
  cache usage` (capacity), the ratio `Avg generation throughput / Running`
  (per-request inter-token rate), and whether `Preemptions` ever appears — that
  line is printed only when it is non-zero (`vllm/v1/metrics/loggers.py:230-233`
  at vllm v0.14.1). Compute the ratios; do not characterise the scheduler from
  memory.
- **`Prefix cache hit rate: 0.0%` is a *switch* question first and a tuning
  question second.** The line is printed unconditionally (`loggers.py:234-245`),
  so its presence proves nothing was attempted, and on a hybrid linear-attention
  model at vllm ≤ 0.14.x the capability is force-disabled in the config layer.
  Check the gate before proposing anything: `zero-prefix-cache-hit-hybrid.md`.
- **Separate the two complaints before answering.** "排队久" is TTFT: total
  prefill work ahead / prefill throughput, and it is *not* improved by
  `--max-num-seqs` or by memory. "生成慢" during the same window is
  prefill/decode interference inside one scheduler budget. A report usually
  contains both, and each has a different lever. When it does, load **both** pages
  in the same `skill_view` turn before any grep: the arithmetic page answers
  "排队", the cache page answers "命中 0", and the recorded #6833 replay lost its
  session by opening only the cache page and re-deriving the queue arithmetic from
  `scheduler.py`.
- The reporter's serve command is the first missing artefact: defaults differ per
  usage context and per device size, and vllm-ascend rewrites some of them
  (`refresh_block_size`, the `ascend_config` alignment). Ask for the startup line
  (`Chunked prefill is enabled with max_num_batched_tokens=%d`,
  `vllm/config/scheduler.py:229-232` at v0.14.1) rather than assuming a default.
- Quote the project's own validated command at the pinned ref as the baseline the
  reporter's command should be diffed against — at vllm-ascend v0.14.0rc1 it is
  `docs/source/tutorials/Qwen3-Next.md:64` (`--tensor-parallel-size 4
  --max-model-len 32768 --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`).
- Never read `npu-smi` AICore%/power from an idle paste as load evidence: those
  numbers were taken while no request was running.
- **A step change at a concurrency number is a fixed-size structure, and the
  first candidate is the ACL-graph capture ceiling.** Running sequences ×
  `(1 + num_speculative_tokens)` is compared with
  `max_cudagraph_capture_size`, which vllm-ascend defaults to
  `min(max_num_seqs * decode_query_len, 512)`; above it
  `cudagraph_dispatcher` returns `NONE` with no log, so every step falls back to
  eager while the metric block stays healthy. One call —
  `bash <skill_root>/scripts/seq-cliff-capture-budget.sh --max-num-seqs N
  --num-spec K` — gives the threshold from the serve command alone, so this
  rule belongs in the *first* answer, not after a source walk
  (`sequence-count-cliff-capture-size.md`). Two traps that make an otherwise
  correct diagnosis look refuted: an ablation that moves the baseline
  (`--enforce-eager` deletes the graph, so it cannot refute a graph-boundary
  claim) and a user-set `max_cudagraph_capture_size` that the log says was
  **truncated** before it took effect. A flat bare-HCCL all-gather benchmark, if
  the reporter offers one, retires the transport lane for the whole family.
- **On a hybrid GDN MoE, a low single-digit `Avg generation throughput` with
  `Running: 1` may be the release, not the flags.** When the project's own
  tutorial command at that ref already carries `--enforce-eager` and caps graph
  mode at `PIECEWISE` (`platform.py:207-208` at v0.11.0rc2), the config diff finds
  nothing and the answer is the version ladder — `qwen3-next-decode-throughput.md`.
  Do not promise the flag flip as a fix: it is a hypothesis with a named log line to
  confirm.

- **A rate that drops when a feature is switched on is a different question
  from a rate that is low on its own.** If the reporter compares two commands
  for the *same* model and one of them adds `--enable-lora` (recorded #1812:
  9 → 2 tok/s on 310P), the answer needs the backend that feature binds and
  whether the SoC can amortise launches, not throughput arithmetic on the metric
  block — `../lora-adapters/decode-overhead-vs-base-model.md`. It is still this
  track's arithmetic: convert both rates to ms/step and price the feature's
  FLOPs before calling the gap overhead.

## Pages

- `prefill-queue-and-throughput.md` — the worked #6833 block (40 concurrent
  requests, `Waiting` 35 → 0, generation throughput 8.5-38.5 tok/s during prefill
  and 1039.6 tok/s after): the arithmetic, why the queue is prefill-bound, the
  `--max-num-batched-tokens` trade-off, and the levers that actually change TTFT.
  If the paste's samples match that table line for line, the arithmetic is
  settled — cite it and spend the remaining turns on the answer, not on
  re-deriving it.
- `qwen3-next-decode-throughput.md` — Qwen3-Next-80B-A3B at 4-6 tok/s on one
  request: the per-step arithmetic that rules capacity and queueing out, the
  verified v0.11.0rc2 map (`--enforce-eager` in the project's own command,
  `FULL_AND_PIECEWISE` degraded to `PIECEWISE`, prefill GDN rebound to a pure-torch
  chunk loop, no `qwen3_next` in `tests/`), why the newer release's
  `FULL_DECODE_ONLY` tutorial command is the real lever, and the paste-hygiene traps
  (#4357: `served_model_name` contradicts the reported command).
- `sequence-count-cliff-capture-size.md` — throughput collapsing in one step as
  running sequences cross a threshold (#14440: 256 → 258 at tp=8, MTP,
  DeepSeek-V4-Flash on 910B2): the `min(max_num_seqs × (1+num_spec), 512)`
  ceiling, the silent `NONE` dispatch, the one startup `warning_once` that names
  both numbers, the `max_cudagraph_capture_size` A/B with its truncation trap,
  which of the reporter's seven "collapses" rows cannot refute it, and what the
  residual host-side candidate class looks like if the A/B does not move.
- `zero-prefix-cache-hit-hybrid.md` — a hybrid GDN model (Qwen3-Next) reporting
  `Prefix cache hit rate: 0.0%`: the `SupportsMambaPrefixCaching` gate that
  disables prefix caching in `MambaModelConfig` at vllm v0.14.1, the model's own
  `assert not cache_config.enable_prefix_caching`, how vllm-ascend's
  `patch_mamba_config.py` inherits that decision, the one log line that confirms
  it on the reporter's host, and the later-version contrast.
