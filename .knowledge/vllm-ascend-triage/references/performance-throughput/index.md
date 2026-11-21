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

**Also this topic:** the report is a *version comparison* instead of a measurement —
"[Bug]: <model>相同条件下性能测试不如0.7.3版本", "升级后吞吐掉了", the new image's env dump
as the only evidence, no numbers on either side. Then the baseline may carry an
accelerator the new image no longer ships and a weight layout whose default flipped
twice inside three releases, so the deliverable is the one-launch A/B list plus the
removed-switch checklist, not a mechanism: `upgrade-vs-older-baseline.md`.

This is a *measurement* track, not a defect hunt: no `WorkerProc` line, no device
dump, no traceback. The deliverable is arithmetic over the pasted numbers, the
one or two flags that move each number, and the artefacts that are missing.

**Also this topic:** the pasted evidence is a *profiler table* instead of a metric
block — "单请求推理时 `QuantBatchMatmulV3_ND` 被调用了 98 万次", "采集出来的数据
NPU 空闲时间 95%", a MindStudio/`msprof` op-statistics screenshot, or an
`msprof analyze compare` NPU-vs-GPU operator-count ratio. There the table itself is
the first suspect (collection health, window, step normalization), not the engine —
read `profiler-opcount-evidence.md` before doing any arithmetic, and never walk the
quantization/linear internals for a count question. **The one profiler report that is
*not* a count question is a coverage claim** — "the W8A8 artifact is no faster than
BF16 because *this layer class* has no quant kernel" (recorded #9443, the
full-attention `self_attn` rows of a Qwen3.5 hybrid): there the layer class, not the
window, is what needs checking, and `quantized-artifact-layer-coverage.md` owns it.

**Also this topic:** the report compares **two runs** — two platforms, two images, or two
releases on one command line — and pastes one number per run ("12s vs 4s", "吞吐高了一倍",
an EvalScope/AISBench table or a screenshot of one). Convert those into rates on a stated
axis before ranking anything: `cross-run-ratio-normalization.md`, one script call.

**And this topic:** a MoE serve whose own command omits `--enable-expert-parallel`
(usually because "开 EP 会胡言乱语") has moved every MoE layer onto the TP all-gather
path — the perf complaint and the garbled-output complaint are one decision:
`moe-comm-path-when-ep-off.md`.

**And this topic:** the two runs differ by one `--speculative-config` and the direction
is **negative** ("开启 mtp 性能下降 30 %", while a sibling model on the same release
improves). Acceptance is healthy there, so this is the token-per-step arithmetic, not
the acceptance-collapse lane: derive tokens per step from `ITL_p50 / TPOT_p50`, compare
it with the step-cost ratio, then price the per-step weight term to decide whether the
model class can gain at all — `spec-decode-gain-accounting.md`.

**And this topic:** a profiler table with a *missing column* — `Input Shapes: N/A`,
`Block Dim: 0`, `OP State: N/A` for one custom op while its neighbours carry shapes,
usually attached to a self-diagnosed `OpCommand::SetCustomHandler` /
`aclprofRangePushEx` fix proposal — is a **launch-lane** question, not a count one:
read `profiler-shape-columns-custom-op-launch.md` (and run its
`scripts/op-shape-lane.sh`) before any of the count arithmetic below.

Not this topic, leave immediately:

- Requests eventually answer but the **scores** differ between runs → accuracy
  track, `../accuracy-benchmarks/index.md`.
- Throughput ~0 with requests staying `Running` and `Accepted: 0` / `Drafted: N` →
  speculative decoding, `../spec-decode-mtp/acceptance-collapse.md`.
- The service dies, OOMs or leaks after a while, or a rank dies under load →
  `../runtime-memory/index.md` / `../dsa-sparse-attention/concurrency-crash-with-no-log.md`.
- The reporter is planning a deployment, not measuring one → `../deploy-sizing/index.md`.
- Nothing answers at all after one oversized prompt — `Running` 0 while `Waiting`
  stays non-zero and KV usage ≈ 0, recovery only by restart → that is an admission
  wedge, `../scheduler-admission/index.md`.

## Rules

- **Date the engine from the metric line before choosing any lever.** `Swapped: N reqs,
  Pending: N reqs` (with `CPU KV cache usage`) is the **V0** logger — `vllm/engine/metrics.py:489`
  at vllm v0.8.5.post1 — while V1 prints `Waiting` and `Prefix cache hit rate`
  (`vllm/v1/metrics/loggers.py:115-117`). On a pre-V1 Ascend image graph mode is torchair
  behind `additional_config={'enable_graph_mode': True}` (opt-in, force-disabled under V1)
  and the platform hard-sets `enforce_eager = True` (`platform.py:124` at v0.8.5rc1), so the
  ACL-graph capture rules below do not apply and must not be quoted:
  `pre-graph-mode-eager-era.md`. It also answers the usual "what caps my concurrency"
  question for free: `Pending: 0` with KV usage near 0 % means nothing was queued and the
  running count is what the client offered.
- **A cross-release comparison is priced from the new pin's own release notes
  before any flag is proposed.** `docs/source/user_guide/release_notes.md` at the new
  commit is cumulative — the v0.13.0rc1 file still holds `## v0.7.3` — so the old
  baseline's documented levers are in-tree, and a removed switch (torchair block,
  `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE`) or an absorbed accelerator (MindIE Turbo,
  "Qwen 2 series", `release_notes.md:728` vs `:342`) makes "same conditions" false
  without any log. Date each hit by its own `## vX` header (the sections are **not**
  in date order) and treat the same env var's contradictory lines
  (`VLLM_ASCEND_ENABLE_NZ` at `:125` vs `:37`) as a multi-value A/B, never as a
  default to quote — `upgrade-vs-older-baseline.md`.
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
  numbers were taken while no request was running. Nor as a host-vs-device
  discriminator: `0 %` is also what a rank waiting inside a collective shows, frozen
  on the same operator count as its peers
  (`../hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md`).
- **A step change at a concurrency number is a fixed-size structure, and the
  paste's own `GPU KV cache usage` at that step says which one.** Near 100 % at the
  stepping concurrency, with `Running` plateauing below the offered count, is the
  **KV pool**: back-solve the pool from two of the reporter's rows
  (`footprint × running / usage`; #5859 gave ≈256 K tokens twice over a 33.8 K-token
  request, a ceiling of ≈7.5 requests, and `Running: 8` behind 12 offered), and note
  that V1 prices the wall by preemption — `_preempt_request` sets
  `num_computed_tokens = 0`, i.e. a full re-prefill, and `Preemptions` is printed
  only when non-zero (`kv-capacity-step-and-preemption.md`). At 81-84 % instead
  (#14440), capacity and admission are out and the graph ceiling below is the
  candidate. Decide from that column before opening any source; the two shapes look
  identical otherwise. **The graph ceiling, when it is the owner:** Running sequences ×
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
- **"Why is this path many ops instead of one — platform limit or engineering
  choice?" is an intent question, and it is answered per phase.** On a hybrid
  linear-attention (GDN) family the prefill/chunk lane and the decode lane have
  different owners, different kernels and different fusion status; a source walk
  can state the current shape and the tiling limits, but the *plan* is a
  maintainer statement that must be labelled as one — and a fusion op announced
  for the chunk/prefill pipeline usually leaves decode on the Triton kernel, which
  is what the reporter asks next. Lane map, per-phase checks and the recorded
  #8940 roadmap reply: `gdn-fusion-lanes-and-roadmap.md`.
- **An op count or an "NPU idle %" is a property of the collection window until
  proven otherwise.** Recover the window's step count from the table's own ratios
  (`scripts/opcount-reconcile.py`), reconcile `count × avg` against the elapsed
  time, and let one same-request end-to-end A/B outrank every kernel-level
  mechanism; a collection that printed an error, or that reports a working decode
  loop as 95 % idle, invalidates the table rather than the engine
  (`profiler-opcount-evidence.md`; reporter-confirmed family, #7351).
- **A rate that drops when a feature is switched on is a different question
  from a rate that is low on its own.** If the reporter compares two commands
  for the *same* model and one of them adds `--enable-lora` (recorded #1812:
  9 → 2 tok/s on 310P), the answer needs the backend that feature binds and
  whether the SoC can amortise launches, not throughput arithmetic on the metric
  block — `../lora-adapters/decode-overhead-vs-base-model.md`. It is still this
  track's arithmetic: convert both rates to ms/step and price the feature's
  FLOPs before calling the gap overhead.

- **A reported ratio between two runs is a rate question first.** Whatever the two
  sides are (two platforms, two images, two releases, P vs D), the number in the body
  is a wall clock, a total throughput or an OCR'd table cell, and none of those is the
  quantity the reporter thinks it is: totals mix input and output lengths, wall clock
  moves with the token count, and the small columns of a screenshot table get
  transposed. One call —
  `uv run --no-project python <skill_root>/scripts/benchmark-table-normalize.py --run … --compare A B` —
  gives per-request token counts, the decode step, the prefill cost per input token
  and which axis actually moved; rank causes on that axis only. Two recorded shapes:
  a cross-platform table whose per-request lengths differed 4 K/100 vs 2 K/1024, and a
  two-release PD report at "12s vs 4s" with `num_speculative_tokens: 3` on the decode
  side and no token counts anywhere (`cross-run-ratio-normalization.md`).
- **A MoE serve with EP off is not a tuning question.** When the command drops
  `--enable-expert-parallel` (or keeps it on one node only), every MoE layer runs the
  TP all-gather path instead of MC2 dispatch/combine at the same card count — one
  grep of the selector plus the two step times is the answer, and the garbled EP run
  is the defect to chase (`moe-comm-path-when-ep-off.md`).

## Pages

- `profiler-opcount-evidence.md` — the evidence *is* a profiler table
  (operator call counts, NPU idle %, NPU-vs-GPU operator compare): validate the
  collection and its CANN/profiler pairing (`libruntime_v100.so` is the
  `runtime_v100` target of `cann/runtime`, `cmake/package.cmake:271`), recover the
  step count from the table's own quotients (the #7351 arithmetic:
  `988224/192 = 329408/64 = 653669/127 = 334555/65 = 5147`, ≈256 matmuls/step), and
  treat per-step parity as the null hypothesis, with the launch-command flags that
  have no reader at the pinned ref (`VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` removed at
  v0.13.0rc1, `release_notes.md:35`) as a correction rather than a cause.
- `profiler-shape-columns-custom-op-launch.md` — one row of the table is `N/A`:
  `SetCustomHandler` is not the discriminator (vllm-ascend's own `EXEC_NPU_CMD` uses
  it too, `csrc/aclnn_torch_adapter/op_api_common.h:584` at v0.18.0); the
  `CUSTOM_OPS` build list decides whether an `aclnn<Op>` entry point exists at all
  (`scripts/op-shape-lane.sh`), the CANN tensor-info record holds 5 tensors × 8 dims
  (`cann/runtime/pkg_inc/profiling/aprof_pub.h:34-35,322-326`), and the aclprof
  tensor-info API is *unverifiable* from a truncated export (`acl_prof.h` stops at
  `aclprofMarkEx`). Recorded for #9026, where the run had every fact and no delivery.
- `prefill-queue-and-throughput.md` — the worked #6833 block (40 concurrent
  requests, `Waiting` 35 → 0, generation throughput 8.5-38.5 tok/s during prefill
  and 1039.6 tok/s after): the arithmetic, why the queue is prefill-bound, the
  `--max-num-batched-tokens` trade-off, and the levers that actually change TTFT.
  If the paste's samples match that table line for line, the arithmetic is
  settled — cite it and spend the remaining turns on the answer, not on
  re-deriving it.
- `gdn-fusion-lanes-and-roadmap.md` — "why isn't linear attention one big op
  on Ascend, and will it be fused" (#8940, Qwen3.5-0.8B on 910B4, v0.15.0rc1):
  the per-lane stage map (`chunk_gated_delta_rule` prefill vs the single
  `fused_sigmoid_gating_delta_rule_update` decode kernel), why "Ascend splits it
  more" is refutable at the pin, the UB/grid constants that really bound each
  kernel, the maintainer roadmap split (mega GDN fusion op #8872 for prefill,
  decode still Triton) and the traps: `main` vs pinned paths, `--enforce-eager`
  moving the baseline, layer types that are not comparable per forward.
- `quantized-artifact-layer-coverage.md` — a **quantized artifact that is no
  faster than its BF16 sibling**, with a dump showing a whole layer class without a
  quant kernel (#9443, Qwen3.5-122B w8a8-mtp on v0.19.1rc1): the two separate
  quantizations (linear int8 per-layer json entry vs `fa_quant_type` int8
  attention/KV, decode-only), why a bf16 linear layer can only come from a `"FLOAT"`
  entry in `quant_model_description.json` (a missing key is a `KeyError`, not a
  silent skip), why an *architectural* layer class is a family-wide converter gap
  and not "the artifact's choice" (the reporter's own follow-up on #9443), the
  sibling-artifact `jq` and the same-request bf16/w8a8 A/B that separate the two,
  and the variant-coverage facts.
- `qwen3-next-decode-throughput.md` — Qwen3-Next-80B-A3B at 4-6 tok/s on one
  request: the per-step arithmetic that rules capacity and queueing out, the
  verified v0.11.0rc2 map (`--enforce-eager` in the project's own command,
  `FULL_AND_PIECEWISE` degraded to `PIECEWISE`, prefill GDN rebound to a pure-torch
  chunk loop, no `qwen3_next` in `tests/`), why the newer release's
  `FULL_DECODE_ONLY` tutorial command is the real lever, and the paste-hygiene traps
  (#4357: `served_model_name` contradicts the reported command).
- `kv-capacity-step-and-preemption.md` — the same step shape owned by the KV pool
  instead of the graph (#5859: 6/7/8/12 concurrent at 79.4/92.4/98.9/98.7 % KV):
  the two-row pool back-solve and the per-request percentage, the pinned
  preemption-recompute sites (`scheduler.py:316-362,759-781`, `Preemptions` printed
  only when non-zero at `loggers.py:221-223`, the downstream SLO fork repeating it),
  why the default capture list already contains 8, the `--max-num-seqs`-as-guard
  ablation, the capacity levers, and the GLM-4.7 MTP-on-main-only note that collapses
  the `×(1+K)` multiplier on a released image.
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
- `pre-graph-mode-eager-era.md` — the v0.8.x/v0.9.x sibling of these rules (#1138,
  v0.8.5rc1, DeepSeek-R1-Distill-Qwen-32B TP4 on 4×910B, "AICore < 15 %, 57 tok/s"):
  how `Swapped`/`Pending` date the paste to the V0 logger, why `Pending: 0` +
  `KV 0.2 %` retires every server-side concurrency cap, the forced-`enforce_eager`
  platform line with its two silent graph-mode gates (`supports_dynamo()`,
  `VLLM_USE_V1`), the torchair opt-in that is the real lever there, and why AICore % is
  a decode-shaped number rather than a defect meter.
- `cross-run-ratio-normalization.md` — two runs, one number each (#2833 Ascend-vs-H800
  EvalScope screenshots; #14291 v0.23→v0.24 PD "12s vs 4s"): the reconcile step and
  what each field survives OCR, `step = concurrency / output tok/s` vs
  `TTFT / input tokens` as the two axes, the wall-clock closure check, the two
  innocent explanations for a release-to-release wall-clock ratio (thinking length,
  the `1 + num_speculative_tokens` budget with `Mean acceptance length` as the
  logged multiplier), the missing-control list, and the version-absent-from-snapshot
  diff command.
- `spec-decode-gain-accounting.md` — "MTP on is slower" with acceptance fine
  (#8967, Qwen3.5-35B-A3B w8a8 on one 910B3, `qwen3_5_mtp` K=3, vllm 0.18.rc1):
  tokens/step ≈ `ITL_p50 / TPOT_p50` = 53.8/16.2 ≈ 3.3 of a 4-token budget, step
  3.62× against 3.32× tokens (per-token +9 %, the "30 %" is E2EL and TTFT), why a
  14.8 ms step against a ~2 ms weight read is dispatch-bound so the verify tokens cost
  ~additive, the slope experiment (`num_speculative_tokens` 3→2→1→0) and concurrency
  row, the two one-grep refutations (`MTPModelTypes` → `"mtp"` normalization;
  spec-scoped `enforce_eager` touching only the drafter graph), and the
  dense-sibling-recipe provenance check.
- `moe-comm-path-when-ep-off.md` — a DP×TP MoE perf report whose command has
  `--enable-expert-parallel` missing or asymmetric (#2833, GLM-4.5 on 2×8×910B2 at
  v0.10.1rc1): the `ep_size`/`_get_fused_moe_state` table (`1` → AllGather,
  `≥16` decode → MC2) at the ref where the choice is logged **nowhere**, the tutorial
  the release note points at and why its `torchair_graph_config` block raises on
  `glm4_moe`, the absent MTP proposer, and the four-rung ladder that puts EP back on
  both nodes before any tuning.
- `upgrade-vs-older-baseline.md` — "新镜像比 0.7.3 慢" with no numbers (#6119,
  910B 32G, Qwen2.5-72B BF16, vLLM 0.13.0rc1 image): the cumulative-notes grep that
  dates every default flip, the silent no-op table for a 0.7.3-era command
  (`torchair_graph_config`, `ascend_scheudler`, `DENSE_OPTIMIZE`, TOPK_TOPP now
  default-on), the MindIE-Turbo pair that makes a Qwen2-series comparison
  incomparable (`:728` vs `:342`, probe line `utils.py:180`), the three-value
  `VLLM_ASCEND_ENABLE_NZ` A/B with its four dated, contradicting note lines, and the
  artefacts both runs owe before any arithmetic.
