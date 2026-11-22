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

**Or the cache hit rate is *low* rather than absent** — "输入 token 数为 1024 倍数时 prefix cache
命中率个位数", 命中率在一个长度上骤降 — which is the lookup arithmetic in
`prefix-cache-hit-rate-at-block-boundary.md` and ends in two log lines plus one
`cached_tokens` read; it is not the gate page's `0.0 %` and not a kernel question.

**Also this topic:** the report is a *version comparison* instead of a measurement —
"[Bug]: <model>相同条件下性能测试不如0.7.3版本", "升级后吞吐掉了", the new image's env dump
as the only evidence, no numbers on either side. Then the baseline may carry an
accelerator the new image no longer ships and a weight layout whose default flipped
twice inside three releases, so the deliverable is the one-launch A/B list plus the
removed-switch checklist, not a mechanism: `upgrade-vs-older-baseline.md`.

This is a *measurement* track, not a defect hunt: no `WorkerProc` line, no device
dump, no traceback. The deliverable is arithmetic over the pasted numbers, the
one or two flags that move each number, and the artefacts that are missing.

**Not this topic: a "首 token 慢" that came with a frontend flag.** When the report is
"加了 `--reasoning-parser` 后首 token 显著变慢 / 思考内容也丢了", the flag lives in the
API server *after* generation and routes every pre-`</think>` delta into
`delta.reasoning`, so a client timing the first non-empty `delta.content` is timing the
end of thinking — open `../accuracy-benchmarks/reasoning-output-routing.md` instead, and
decide it with the engine's own `vllm:time_to_first_token_seconds` histogram (flat ⇒
measurement artifact, and this topic stays closed). The neighbouring route-away is a
streaming complaint on a serve that added `--tool-call-parser` / `--enable-auto-tool-choice`:
a tool parser holds text back until it can classify the boundary, which is why those
runs show silent `delta: {}` chunks — `tool-call-streaming/index.md` owns that, and the
engine's TTFT histogram separates it from decode work the same way.

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
**Its dense-model sibling has no dump at all** — the whole report is one title,
"Qwen2.5-7B-w8a8量化配置，没有性能收益（甚至有降低），精度同样降低", with no serve line
and no numbers — and that one is two claims whose owners differ:
`quantized-dense-no-gain.md` (scheme histogram from the artifact's own json, then the
`2/(1+u)` byte ceiling, before any engine cause).

**Also this topic:** the report compares **two runs** — two platforms, two images, or two
releases on one command line — and pastes one number per run ("12s vs 4s", "吞吐高了一倍",
an EvalScope/AISBench table or a screenshot of one). Convert those into rates on a stated
axis before ranking anything: `cross-run-ratio-normalization.md`, one script call.

**And the cross-*model* form of that shape:** the two sides are two **different models**
on the same box, image and release ("910B 部署 X-35B 为什么推理很慢, 20k 不到 25 t/s, 比
Y-27B 的还慢, 是部署参数问题吗") with a screenshot table of t/s and TTFT across context
lengths. There the second model is a genuine control — same cards, same flags — so the
deliverable is the ratio its bytes predict plus the two startup lines, not a flag hunt:
`cross-model-decode-ratio.md`.

**And the PD variant of that shape:** the two runs are **one model's PD deployment and its
colocated/multi-node deployment** ("the TTFT of PD is 2 times longer than TTFT of
Multi-node Deployment … but the typical model of qwen no this issue"), with no numbers,
no serve commands and an empty `collect_env`. The direction is structural — the PD proxy
returns the first token only after **D** produces one (#3099) — so nothing here is a
transport mechanism until the terms are split, and the two recipes are not the same
configuration (the documented PD prefiller runs `--enforce-eager` + `--max-num-seqs 8`
while the colocated blocks run `--async-scheduling` + `FULL_DECODE_ONLY`). Answer on the
second or third turn with the chain, the pinned tutorial's own PD anchor and the one-flag
A/B on the prefiller: `../pd-disaggregation/pd-vs-colocated-ttft.md`.

**And this topic, one switch away from that one:** the two runs differ by adding
`--speculative-config` to a PD pair and the regression is **TTFT alone** — 64 K/1 K at
"prefix 90 %" going 2.2 s → 22 s while decode throughput is not in the complaint. That
factor is `1/(1−reuse)`, i.e. a full re-prefill, so read the **prefiller's**
`Prefix cache hit rate` before any draft-model theory, and diff P against the model
tutorial's own spec-decode PD recipe in the same turn: DSV4-Flash's prefill block
exports `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` and runs
`--max-num-batched-tokens 8192`, both of which recorded #16997 omitted (and turning
spec decode off on P alone is not a legal rung — the doc requires the same
`num_speculative_tokens` on both sides, and the asymmetric pair is #16177's decode
crash): `spec-on-prefill-ttft-regression.md`.

**And this topic:** a MoE serve whose own command omits `--enable-expert-parallel`
(usually because "开 EP 会胡言乱语") has moved every MoE layer onto the TP all-gather
path — the perf complaint and the garbled-output complaint are one decision:
`moe-comm-path-when-ep-off.md`.

**And this topic:** the reference point is another **engine** or another **card count** —
"8卡 910B3 … 平均 6 token/s", "单卡跑 Qwen3-4B 可以到 17 token/s 也比 mindie 慢了一半多",
"多卡跑更慢 应该是多卡通信机制上没有优化" (recorded #770). A vendor-engine number is a data
point, not "same conditions", and a report that already carries two card counts is an ablation
matrix with the single-card control inside it: the deliverable is the per-request decode rate
plus a one-client TP sweep, and the pin decides whether a comm lever even exists (at v0.8.x the
MC2 MoE decode path is DeepSeek-only and `VLLM_ENABLE_MC2`-gated, and there is no EP flag at
all) — `multicard-scaling-and-vendor-baseline.md`.

**And this topic, on a pre-v0.8 pin:** the baseline is **transformers**, not another
engine image — "[Bug]: [0.7.3rc1] vlm inference … inferring speed is slower than
original transformers" (recorded #1427: vllm-ascend `v0.7.3rc1`, Qwen2.5-VL-3B,
`-tp 2`, client-side latency rows only and no transformers numbers). The 0.8/0.9 pages
above misfire there (no graph key, no V1 worker at that ref), two flags in the pasted
command are inert (`--enable-prefix-caching` for a multimodal model under V0,
`--multi-step-stream-outputs` already default), and the vision tower is the one
component with a pinned up/downstream gap (Qwen2-VL patched, Qwen2.5-VL not) —
`pre-0.8-vl-vs-transformers.md`.

**And this topic:** the two runs differ by one `--speculative-config` and the direction
is **negative** ("开启 mtp 性能下降 30 %", "PD 双机 mtp=2/3 比 mtp=1 差", while a sibling
model on the same release improves). Acceptance is healthy there, so this is the
token-per-step arithmetic, not the acceptance-collapse lane: derive tokens per step from
`ITL_p50 / TPOT_p50`, compare it with the step-cost ratio, then price the per-step weight
term to decide whether the model class can gain at all — `spec-decode-gain-accounting.md`.
When the paste is two EvalScope/AISBench **screenshots at concurrency > 1** (the PD form,
no TPOT row), that derivation is unavailable: normalize the summary fields with
`scripts/benchmark-table-normalize.py` per `cross-run-ratio-normalization.md`, and note
that K>1 on a PD/multi-node DeepSeek recipe is outside the pin's validated matrix
(every published recipe there is K=1).

**And this topic:** a profiler table with a *missing column* — `Input Shapes: N/A`,
`Block Dim: 0`, `OP State: N/A` for one custom op while its neighbours carry shapes,
usually attached to a self-diagnosed `OpCommand::SetCustomHandler` /
`aclprofRangePushEx` fix proposal — is a **launch-lane** question, not a count one:
read `profiler-shape-columns-custom-op-launch.md` (and run its
`scripts/op-shape-lane.sh`) before any of the count arithmetic below.

**And this topic:** the direction is **graph mode slower than eager** and the
reporter has already localized it — cites a `path:line` and says the perf comes
back when that line is deleted ("ACLGraph 模式下 query 被截断，部分算子进不了图") —
is a verdict on *their* patch: find the padding gate one block above the slice
(the state-gated block that pads `seq_lens`/`block_table`/`query_start_loc`), check
who assigns that state enum before trusting its name, and recommend extending the
gate rather than deleting the guard — `graph-mode-slower-than-eager.md`.

**And this topic:** several `vllm serve` processes share one ARM host — one card each,
or several DP instances — and the complaint is that running them *together* is much
slower than one alone, usually with two `The CPU allocation plan is as follows:` blocks
in the paste. Then the host-side affinity policy is the suspect, not the cards:
`topo_affinity` mode (A2/310P) sizes each pool from the **visible** NPU's topo band, so
two processes whose cards sit in one band compute the *identical* pool, the
`average_distribute` guard never fires, and both instances' IRQ cores land on the same
two CPUs — the pin's own release notes call the multi-instance case broken and the
non-overlap fix (`global_slice`) is A3-only. Read
`cpu-binding-multi-instance-overlap.md`, invert the plan lines with
`scripts/cpu-bind-plan-audit.py`, and answer with the card pairing before any tuning
lever. The single-instance `[migrate]` hang on an offload serve is a different page
(`../feature-enable-recipes/engram-offload.md`).

Not this topic, leave immediately:

- Requests eventually answer but the **scores** differ between runs → accuracy
  track, `../accuracy-benchmarks/index.md`.
- Throughput ~0 with requests staying `Running` and `Accepted: 0` / `Drafted: N` →
  speculative decoding, `../spec-decode-mtp/acceptance-collapse.md`.
- The service dies, OOMs or leaks after a while, or a rank dies under load →
  `../runtime-memory/index.md` / `../dsa-sparse-attention/concurrency-crash-with-no-log.md`.
- The benchmark client itself throws — `ConnectionResetError: [Errno 104]`,
  `ClientOSError`, `ConnectionRefusedError`, an `evalscope perf` row of
  `Total / Success / Failed n / k / m` with everything after the first batch failing —
  while the serve log has a device dump: that is **not** a measurement track report.
  The client is reporting a dead worker; go to the serve log's first `ERROR` block
  (`../dsa-sparse-attention/dcp-c8-kv-address-fault.md` Rule 1).
- The reporter is planning a deployment, not measuring one → `../deploy-sizing/index.md`.
- Nothing answers at all after one oversized prompt — `Running` 0 while `Waiting`
  stays non-zero and KV usage ≈ 0, recovery only by restart → that is an admission
  wedge, `../scheduler-admission/index.md`.

## Rules

- **Date the engine from the metric line before choosing any lever.** `Swapped: N reqs,
  Pending: N reqs` (with `CPU KV cache usage`) is the **V0** logger — `vllm/engine/metrics.py:489`
  at vllm v0.8.5.post1, `:418` at v0.9.1 — while V1 prints `Waiting` and `Prefix cache hit rate`
  (`vllm/v1/metrics/loggers.py:115-117`; `:120-124` at v0.9.1). What the V0 date *means* splits
  by generation: on **v0.8.x** graph mode is torchair behind
  `additional_config={'enable_graph_mode': True}` (opt-in, force-disabled under V1) and the
  platform hard-sets `enforce_eager = True` (`platform.py:124` at v0.8.5rc1), so the ACL-graph
  capture rules below do not apply and must not be quoted; on **v0.9.x** `enforce_eager` is read
  from the config, Ascend `supports_v1()` returns `True`, and the serve is V0 purely because
  non-CUDA hardware falls back unless `VLLM_USE_V1=1` is exported — the release notes ask for
  exactly that (`release_notes.md:52`, V0 frozen), so the lever is the env var and the `PIECEWISE`
  ACL graph arrives with it. Both generations: `pre-graph-mode-eager-era.md`. The metric line also
  answers the usual "what caps my concurrency" question for free: `Pending: 0` with KV usage near
  0 % means nothing was queued and the running count is what the client offered.
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
  **At a newer pin that gate is dead** — then, when the model is a documented one
  and the deployment is PD, the pin's own `docs/source/tutorials/models/<model>.md`
  decides before any source: which side the recipe enables caching on (DSV4-Flash
  at v0.22.1rc1: P yes, D `--no-enable-prefix-caching`) and its tuning note
  (§9.1: "4K prefix cache hit in an experimental manner … change `--block-size`
  from 128 to 32"), so a 0 % read off the decode instance is structural. **On a
  prefilling instance a flat 0 % is *not* the envelope**: a ~4 K granularity still
  reuses most of a 64 K shared prefix, so zero means the lookup returned nothing —
  rank the version rung first, because DSV4 prefix-cache hit rate was rewritten in
  v0.21.0rc1 (#10354/#9903/#10297) and improved again by #11107 in v0.23.0rc1,
  which the #12041 reporter confirmed fixed the colocated case (「v0.23.0测试已修复」).
  Then `--block-size 32` on the P node as the on-pin workaround. Diff the reporter's
  two launch scripts against that recipe in the same turn (#12093 burned 60 turns on
  the coordinators instead).
- **A rate that is *nonzero* but low, or that steps with the prompt length, is
  arithmetic and not a defect.** The lookup asks for `num_tokens − 1` tokens and
  every group rounds the hit down to a block multiple, so the ceiling is
  `B·floor((L−1)/B)/L`: exact-multiple prompt lengths lose one whole block, and
  anything shorter than one block is never cached. `B` is 128 on the dense path but
  is *derived* from the state geometry on a hybrid-marked model (the pin's own
  Qwen3.5 tutorials say it can be 2048), which is why the lost block can look like a
  collapse. Two log lines and one `cached_tokens` read close it —
  `prefix-cache-hit-rate-at-block-boundary.md`; reporter-confirmed #7965, and the
  answer is "normal behaviour", so do not rank kernels or config for it.
- **Separate the two complaints before answering.** "排队久" is TTFT: total
  prefill work ahead / prefill throughput, and it is *not* improved by
  `--max-num-seqs` or by memory. "生成慢" during the same window is
  prefill/decode interference inside one scheduler budget. When the driver offers
  a *rate* of long-output requests (`--request-rate` × `--random-output-len`) the
  queue is decode-bound instead: the wall time is `output tokens ÷ aggregate
  rate`, `mean TTFT ≈ wall/2` is one FIFO drain, and the prefill terms are not in
  the report at all (`prefill-queue-and-throughput.md`, second half). A report usually
  contains both, and each has a different lever. When it does, load **both** pages
  in the same `skill_view` turn before any grep: the arithmetic page answers
  "排队", the cache page answers "命中 0", and the recorded #6833 replay lost its
  session by opening only the cache page and re-deriving the queue arithmetic from
  `scheduler.py`.
- **Name the load driver before reading any of its columns.** The same numbers
  mean different things per tool, and one tool's per-request row is not the
  engine's `Output token throughput`. Four families are seen: upstream
  `vllm bench serve` / `benchmark_serving.py` (TTFT from the client send, ITL/TPOT
  pooled, `--request-rate` open loop — `prefill-queue-and-throughput.md`),
  EvalScope and AISBench (the same block plus an `Average … per request` pair),
  and third-party Chinese load generators — the sheet headed
  `LLM 性能测试汇总报告` with `总生成Token` / `总测试时间` / `总体Token生成速率`
  over a per-concurrency ladder of `平均延迟` / `P99延迟` / `首token延迟` /
  `成功率` is **lework/llm-benchmark**, which the #2328 reporter named when asked
  (2025-08-26) after a run had guessed EvalScope/AISBench from the layout. That
  sheet has no ITL/TPOT column, so nothing about the decode step or the in-flight
  count is derivable from it: ask for the tool plus its config (the concurrency
  ladder, input/output length) instead of OCR-reconstructing transposed small
  fields, and compare only the header's aggregate rate.
  EvalScope's `Avg Input Tokens` / `Avg Output Tokens` cells are its own
  accounting, not the engine's: recorded #16079's script asked for
  `--prefix-length 130048` with 1024-in / 2048-out while the summary printed
  `65536` / `1024` (both exactly half, over 2 of 5 requests succeeding), so never
  size a KV pool or an HBM budget from those two rows — take the server's own
  prompt/prefill count or the `PrefixCacheStats` line.
- The reporter's serve command is the first missing artefact: defaults differ per
  usage context and per device size, and vllm-ascend rewrites some of them
  (`refresh_block_size`, the `ascend_config` alignment). Ask for the startup line
  (`Chunked prefill is enabled with max_num_batched_tokens=%d`,
  `vllm/config/scheduler.py:229-232` at v0.14.1) rather than assuming a default —
  but not on a v0.10.x non-MLA serve, which forces chunked prefill off and logs
  `Non-MLA LLMs forcibly disable the chunked prefill feature …` instead
  (`prefill-queue-and-throughput.md`).
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
  `min(max_num_seqs * decode_query_len, 512)`, which upstream then floors to its
  8/16-step capture grid; above it `cudagraph_dispatcher` returns `NONE` with no
  log, so every step falls back to eager while the metric block stays healthy.
  One call — `bash <skill_root>/scripts/seq-cliff-capture-budget.sh
  --max-num-seqs N --num-spec K --observed-cliff M` — gives the threshold from
  the serve command alone *and* reconciles it with the reporter's step index, so
  this rule belongs in the *first* answer, not after a source walk
  (`sequence-count-cliff-capture-size.md`). Read the index before accepting a
  "fixed-size structure indexed by sequence count": with `mns × (1+K) ≤ 512` the
  cliff *is* the per-engine `--max-num-seqs`, so it moves between dp/tp runs
  because the flag does, and removing MTP does not raise it (264 × 1 floors back
  to 256). Three traps that make an otherwise correct diagnosis look refuted:
  an ablation that moves the baseline (`--enforce-eager` deletes the graph),
  `enable_npugraph_ex: false` read as "graphs off" (it is a compile-backend
  switch; only `cudagraph_mode: NONE` zeroes the capture list), and a user-set
  `max_cudagraph_capture_size` that the log says was **truncated** before it took
  effect. A flat bare-HCCL all-gather benchmark, if
  the reporter offers one, retires the transport lane for the whole family.
- **The graph lane is one flag in the reporter's command, and the same flag is a cost on
  one symptom and a cure on the other.** At v0.10.x, `--enforce-eager` (or
  `--compilation-config '{"level":0}'`) forces `NO_COMPILATION`
  (`platform.py:143-145`, log `Compilation disabled, using eager mode by default`), while a
  command without it and without a torchair block runs `PIECEWISE` ACL graph (`:162-168`).
  So a throttled decode whose own command carries the flag is answered by *removing* it,
  and a corruption report on a `--enable-expert-parallel` + `--data-parallel-size` serve
  whose command does *not* carry it is answered by *adding* it — the pinned release lists
  "Aclgraph could not work with DP + EP currently" (`release_notes.md:53`, #2229). Pick the
  direction from the flag before ranking mechanisms; details and both recorded runs:
  `graph-mode-from-the-serve-line.md`.
- **On a hybrid GDN MoE, a low single-digit `Avg generation throughput` with
  `Running: 1` has a one-flag diff even when the tutorial already carries
  `--enforce-eager`.** v0.11.0rc2 downgrades only `FULL_AND_PIECEWISE` to
  `PIECEWISE` (`platform.py:207-208`); an explicit
  `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` is *accepted*
  (`:224-227`) and turns `_use_aclgraph()` on (`model_runner_v1.py:591-592`), so
  "remove the flag and add the mode" is the cheapest check — but it is a check with
  three readable outcomes, not a promised fix: the banner plus a faster rate, a
  `ValueError … not supported with <Builder> backend` (`:3428-3433`), or the
  mode's known-broken lane (v0.12.0rc1 `release_notes.md:42`, #4990 for
  DeepSeek 3/3.1 and Qwen3, fixed 0.13.0rc1). At v0.11.x that mode's own in-tree
  e2e is `@pytest.mark.skip`-ped, so an upgrade stays the durable rung. Recorded
  #4357's own later answer was 并没有, with no invocation named —
  `qwen3-next-decode-throughput.md`.
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
  transposed. **Gate on the fields first:** an OCR that lost the concurrency or the request
  count cannot be normalized at all — say so and ask for the summary text instead of
  multiplying two recovered cells (recorded #770 published an "in-flight ≈ 233 → 602"
  Little's-law count from an OCR'd latency column that no cell supported). One call —
  `uv run --no-project python <skill_root>/scripts/benchmark-table-normalize.py --run … --compare A B` —
  gives per-request token counts, the decode step, the prefill cost per input token
  and which axis actually moved; rank causes on that axis only. Two recorded shapes:
  a cross-platform table whose per-request lengths differed 4 K/100 vs 2 K/1024, and a
  two-release PD report at "12s vs 4s" with `num_speculative_tokens: 3` on the decode
  side and no token counts anywhere (`cross-run-ratio-normalization.md`).
- **When the two sides are two *models* on one box, the fast one is the control and
  the paste's own concurrency row prices the step.** If raising concurrency 1 → 2
  leaves the per-stream step wall (`2 / aggregate_t_s`) within a few percent while the
  two models sit ~2x apart row by row, the step is weight-byte bound and the
  scheduler / KV-pool / `--max-num-seqs` / preemption lanes are closed. The answer is
  then `params × bytes_per_param / TP` against the card's bandwidth class plus the
  per-rank `Model loading took … GiB` line from **both** startups; and when the nominal
  size ratio is smaller than the observed ratio (1.3x vs 2.1x), the remainder is
  dtype/quantization or the model class's decode efficiency — separated by those same
  two logs, never by a registry hunt for an unknown model name
  (`cross-model-decode-ratio.md`).
- **A MoE serve with EP off is not a tuning question.** When the command drops
  `--enable-expert-parallel` (or keeps it on one node only), every MoE layer runs the
  TP all-gather path instead of MC2 dispatch/combine at the same card count — one
  grep of the selector plus the two step times is the answer, and the garbled EP run
  is the defect to chase (`moe-comm-path-when-ep-off.md`).
- **A performance switch that swaps a layer's comm group changes its weight sharding,
  so it can OOM instead of speeding anything up.** `VLLM_ASCEND_ENABLE_MLP_OPTIMIZE`, a
  hand-set fine-grained `mlp`/`oproj` tensor-parallel size and
  `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` are all chosen in
  `get_{column,row}_parallel_op` (`vllm_ascend/ops/linear_op.py`), and the selected op's
  `comm_group` supplies the `tp_size` that `divide(...)` shards the weight by — with
  `data_parallel_size == 1` the MLP group has one member, so the MLP is replicated on
  every card (+~38 GB for a Qwen3-32B at TP4) while its collectives become no-ops. Price
  the flag in bytes before any tuning or leak hypothesis:
  `../deploy-sizing/mlp-tp-group-oom.md`.

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
- `profiler-no-kernel-details-graph-mtp.md` — `kernel_details.csv` **absent** on a
  graph-mode + MTP launch: read the file-existence semantics first (the dir must be
  analysed; torch_npu's own test treats it as the device-collection predicate), the
  profiler flags are the tree's not the reporter's, and an "it collected on an older
  release" statement makes the deliverable a regression plus a bisect — not a CANN
  limitation.
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
  re-deriving it. **Its second half is the opposite shape**: no engine log at all,
  only a `vllm bench serve` summary with `--request-rate`/`--num-prompts` (#3112
  on one 910B4), where the queue is *decode*-bound, `mean TTFT ≈ wall/2` proves a
  single FIFO drain, and in-flight sequences come from the **mean** ITL/TPOT
  (`output tokens × mean ITL / wall`), never from the median.
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
- `quantized-dense-no-gain.md` — a **dense** model where the quantized checkpoint is
  no faster than BF16 (or slower) **and** scores lower, pasted as one sentence with no
  numbers (#10604, Qwen2.5-7B-w8a8 on v0.11.0; the same direction with numbers in
  #1015, QwQ-32B-w8a8 at 10.49 vs 12.54 tok/s): `W8A8` static vs `W8A8_DYNAMIC`
  per-token in the artifact's json (one script call, and the static one is what the
  pin's own dense w8a8 tutorial produces with `--anti_method m1` over a BoolQ
  calibration file), why static-vs-dynamic moves accuracy but not the matmul, the
  `R = 2/(1+u)` ceiling that decides whether "no gain" is arithmetic (and #2514, the
  32B end where `R ≈ 1.9×` with a flat prefill means the int8 matmul is not on its
  fast path), the three asymmetric levers (graph lane, the pin's own weight-layout
  switch — `enable_weight_nz_layout` **off by default** on a v0.9-line tree,
  `VLLM_ASCEND_ENABLE_NZ`/`weight_nz_mode` later, one-sided runs), the two tables
  behind 「是否有模型对应版本表」, and the Qwen2.5-7B-w8a8 coverage gap with the
  published 0.5B artifact as the control.
- `qwen3-next-decode-throughput.md` — Qwen3-Next-80B-A3B at 4-6 tok/s on one
  request: the per-step arithmetic that rules capacity and queueing out, the
  verified v0.11.0rc2 map (`--enforce-eager` in the project's own command,
  `FULL_AND_PIECEWISE` degraded to `PIECEWISE` while an explicit `FULL_DECODE_ONLY`
  is accepted, `ASCEND_LAUNCH_BLOCKING` refusal, prefill GDN rebound to a pure-torch
  chunk loop, no `qwen3_next` case and the mode's e2e skipped), the in-version
  `FULL_DECODE_ONLY` experiment with its three log-line outcomes, the HBM warning
  that usually hits first, why the newer release's command is the durable lever, the
  scope of the recorded 并没有, and the paste-hygiene traps (#4357:
  `served_model_name` contradicts the reported command).
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
  ceiling, the 8/16-step capture grid that floors it (so the cliff is the
  per-engine flag below 512 and `512 // (1+num_spec)` above it), the silent
  `NONE` dispatch, the one startup `warning_once` that names both numbers, the
  `max_cudagraph_capture_size` A/B with its truncation trap, the reporter's
  seven "collapses" rows decided one by one (two of them confirm the grid), the
  TTFT-falls / wall-time-triples shape that retires prefill and admission, and
  what the residual host-side candidate class looks like if the A/B does not
  move.
- `prefix-cache-hit-rate-at-block-boundary.md` — the *nonzero* sibling of that page
  (recorded #7965, Qwen3.5-9B, v0.18.0rc1, "1024·n prompt lengths → 个位数 hit rate"):
  the `max_cache_hit_length = num_tokens − 1` line and the block rounding that make
  `hit = B·floor((L−1)/B)`, the sawtooth it produces (worst at exact multiples, ~100 %
  one token past a boundary, 0 % below one block), the downstream derivation that can
  push `B` far above 128 (`refresh_block_size`'s hybrid early return plus
  `patch_mamba_config.py`'s `attn_block_size` from the ssm page size, and `lcm`
  alignment in align mode), the `IsHybrid` marker question that decides whether that
  branch is reachable at all, and the two-line closure (the `Setting attention block
  size to` log line, then `usage.prompt_tokens_details.cached_tokens` against the
  formula). The reporter's own comment is the verdict: prefill must compute at least
  one token, so the last block never hits. The page also owns the mirror shape in
  which the **field itself is `null`** under `--enable-prompt-tokens-details` while
  the log prints 99.0 %: the writer's truthiness gate (`num_cached_tokens == 0` ⇒ no
  `prompt_tokens_details`, and `cached_tokens: 0` is never emitted, recorded #9030 on
  a 25-token prompt), plus the MiniMax chat usage rebuild whose first-value latch is
  the only patch-side second owner.
- `zero-prefix-cache-hit-hybrid.md` — a hybrid GDN model (Qwen3-Next) reporting
  `Prefix cache hit rate: 0.0%`: the `SupportsMambaPrefixCaching` gate that
  disables prefix caching in `MambaModelConfig` at vllm v0.14.1, the model's own
  `assert not cache_config.enable_prefix_caching`, how vllm-ascend's
  `patch_mamba_config.py` inherits that decision, the one log line that confirms
  it on the reporter's host, and the later-version contrast. Plus the newer-pin /
  PD branch (recorded #12093, DeepSeek-V4-Flash on A2 under PD): the model
  tutorial's own `--block-size 32` / 4K-experimental note, the recipe's P-yes /
  D-`--no-enable-prefix-caching` asymmetry that makes a decode-side 0 % structural,
  and the per-cache-kind block sizes that quantise the hit. The colocated sibling
  (#12041, same pin, TP8, 64 K prompts, TTFT 6.8 → 15.0 s) is the version verdict:
  a flat 0 is a lookup failure, not the envelope, and the fix named in the next
  release's notes (`#11107`, v0.23.0rc1) is what the reporter confirmed.
- `pre-graph-mode-eager-era.md` — the v0.8.x/v0.9.x sibling of these rules (#1138,
  v0.8.5rc1, DeepSeek-R1-Distill-Qwen-32B TP4 on 4×910B, "AICore < 15 %, 57 tok/s"): how
  `Swapped`/`Pending` date the paste to the V0 logger, why `Pending: 0` + `KV 0.2 %` retires
  every server-side concurrency cap, and why AICore % is a decode-shaped number rather than a
  defect meter; then the **generation split** — v0.8.x force-sets `enforce_eager` with the
  torchair opt-in as the only lever (its two silent gates, `supports_dynamo()` / `VLLM_USE_V1`),
  while v0.9.x (#2099, a 1B VL serve on one 910B3) runs V0 only because non-CUDA hardware falls
  back unless `VLLM_USE_V1=1`, which the release notes require and the singlecard CI already
  uses for its `Qwen2.5-VL-3B` multimodal case.
- `multicard-scaling-and-vendor-baseline.md` — the reference point is another engine
  ("比 mindie 慢了一半多") or another card count ("多卡跑更慢"; recorded #770: v0.8.4rc2 reported,
  pin v0.8.5rc1, 8×910B3 Qwen3-30B-A3B ≈6 tok/s against one-card Qwen3-4B 17 tok/s): the OCR
  field gate that decides whether the table can be priced at all, the TP 1/2/4/8 sweep as the
  discriminating check, and the pin's own MoE comm map — a non-DeepSeek MoE goes through
  `ops/common_fused_moe.py:27-67` → `fused_experts` → a per-layer
  `tensor_model_parallel_all_reduce` (`ops/fused_moe.py:668-670`), MC2 is DeepSeek-only and
  env-gated (`:35`, `:496`, `models/deepseek_v2.py:208`), no `enable_expert_parallel` exists,
  everything is eager (`platform.py:124,133`), and the tree holds no 8-card MoE recipe or
  throughput number to grade against.
- `cross-run-ratio-normalization.md` — two runs, one number each (#2833 Ascend-vs-H800
  EvalScope screenshots; #14291 v0.23→v0.24 PD "12s vs 4s"): the reconcile step and
  what each field survives OCR, `step = concurrency / output tok/s` vs
  `TTFT / input tokens` as the two axes, the wall-clock closure check, the two
  innocent explanations for a release-to-release wall-clock ratio (thinking length,
  the `1 + num_speculative_tokens` budget with `Mean acceptance length` as the
  logged multiplier), the missing-control list, and the version-absent-from-snapshot
  diff command.
- `cross-model-decode-ratio.md` — two models on the same box and release (#11260,
  v0.18.0, 2×910B 64G, TP2: 35B-class 22/18.9/14 t/s vs 27B-class 48/40/27, TTFT
  identical to within a few percent, both tables at mtp=2): the TTFT-vs-t/s split, the
  concurrency-row test that makes the step weight-byte bound (45.5 → 48.2 ms for double
  the batch), the `params × bytes/param / TP` floor at the ~1.6 TB/s class (~48 % vs
  ~80 % of class for the two models), the 1.30x-nominal-vs-2.1x-observed gap and the two
  readings it leaves, the per-rank `Model loading took %s GiB memory` line
  (`vllm/v1/worker/gpu_model_runner.py:4567`) plus the `config.json` fields that
  separate them, the table-vs-command tells (mtp=2 with no `--speculative-config`,
  `--served-model-name` of another model), and why the run's zero-text truncation was
  the delivery failure rather than a missing mechanism.
- `spec-decode-gain-accounting.md` — "MTP on is slower" with acceptance fine
  (#8967, Qwen3.5-35B-A3B w8a8 on one 910B3, `qwen3_5_mtp` K=3, vllm 0.18.rc1):
  tokens/step ≈ `ITL_p50 / TPOT_p50` = 53.8/16.2 ≈ 3.3 of a 4-token budget, step
  3.62× against 3.32× tokens (per-token +9 %, the "30 %" is E2EL and TTFT), why a
  14.8 ms step against a ~2 ms weight read is dispatch-bound so the verify tokens cost
  ~additive, the slope experiment (`num_speculative_tokens` 3→2→1→0) and concurrency
  row, the two one-grep refutations (`MTPModelTypes` → `"mtp"` normalization;
  spec-scoped `enforce_eager` touching only the drafter graph), and the
  dense-sibling-recipe provenance check.
  Its **PD 双机** form is a different measurement: EvalScope percentile screenshots at
  concurrency > 1 have no TPOT row to divide (the OCR transposes the small cells), so
  reconcile the summary fields with the script and answer with the K slope, the
  `Mean acceptance length` line and the D-node graph mode; the pin's every PD/multi-node
  DeepSeek MTP recipe is K=1 (`pd_disaggregation_mooncake_multi_node.md:255,315,…`,
  `DeepSeek-R1-W8A8-A2.yaml:33,55`), and #3097 prices the PD MTP gain at +8.6 % and the
  eager step at 4× — worked #5605, which shipped no message text in **either** of its
  two recorded runs (the second identified the case at turn 6 and then deferred to
  re-verify the page's citations at another pin).
- `spec-on-prefill-ttft-regression.md` — the one-switch sibling of that page: adding
  `--speculative-config` to **both** nodes of a PD/EPD pair regresses **TTFT alone**
  (#16997, A5 single-node split, DeepSeek-V4-Flash + `dspark` K=5, 64 K/1 K at "prefix
  90 %": 2.2 s → 22 s): the factor is `1/(1−reuse)` so it is a full re-prefill on the
  prefiller, the deliverable is P's `Prefix cache hit rate` plus the model tutorial's own
  spec-decode prefill recipe (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`,
  `--max-num-batched-tokens 8192`, both omitted on the reported P node), the DSV4
  min-over-groups hit bookkeeping that lets one group zero the prefix, and the P/D
  spec-token parity rule that forbids the P-only ablation.
- `graph-mode-from-the-serve-line.md` — a 2×N 910B MoE serve at v0.10.x whose *command*
  has `--enforce-eager` (or the `torchair_graph_config` key, or a ray backend) and whose
  complaint is a 300-500 ms/token decode: read the compiled level off the command
  (`platform.py:143-168` at v0.10.0rc1, one log line per branch), the depth-divided
  capture clamp that replaces today's `max_cudagraph_capture_size` (`utils.py:47,309-361`),
  and the same page's second half — the *inverse* rung for a DP+EP corruption report whose
  command has no `--enforce-eager`, where the release's own Known-Issues row (#2229) owns
  the lane.
- `moe-comm-path-when-ep-off.md` — a DP×TP MoE perf report whose command has
  `--enable-expert-parallel` missing or asymmetric — **not** a command that carries the
  flag on both nodes, however well the model/card count match this row (recorded #2446
  was routed here by such a match and had to leave the page: `graph-mode-from-the-serve-line.md`)
  (#2833, GLM-4.5 on 2×8×910B2 at
  v0.10.1rc1): the `ep_size`/`_get_fused_moe_state` table (`1` → AllGather,
  `≥16` decode → MC2) at the ref where the choice is logged **nowhere**, the tutorial
  the release note points at and why its `torchair_graph_config` block raises on
  `glm4_moe`, the absent MTP proposer, and the four-rung ladder that puts EP back on
  both nodes before any tuning.
- `cpu-binding-multi-instance-overlap.md` — several single-card instances on one
  A2/310P host whose CPU binding ranges coincide (#8600, v0.18.0rc1): the pin's own
  `Known Issues` rows that admit the multi-instance case is broken
  (`release_notes.md:547`, `:650`) and why the later `global_slice` fix (#6945/#6889)
  is A3-only, the `topo_affinity` pool (the card's topo band **plus** `extend_numa`'s
  whole next NUMA node), the intra-process-only `average_distribute` guard, the
  plan-line inversion that recovers the pool (`main = pool[2:-2]`) and the band table
  that decides which card pairs are safe, the shared IRQ cores and the host-wide
  `irqbalance` stop, the silent `Skip binding cpu` degradation under a narrow cpuset,
  and the ladder (pair different bands → disable and `taskset`/`numactl` by hand →
  group over the full topo table).
- `upgrade-vs-older-baseline.md` — "新镜像比 0.7.3 慢" with no numbers (#6119,
  910B 32G, Qwen2.5-72B BF16, vLLM 0.13.0rc1 image): the cumulative-notes grep that
  dates every default flip, the silent no-op table for a 0.7.3-era command
  (`torchair_graph_config`, `ascend_scheudler`, `DENSE_OPTIMIZE`, TOPK_TOPP now
  default-on), the MindIE-Turbo pair that makes a Qwen2-series comparison
  incomparable (`:728` vs `:342`, probe line `utils.py:180`), the three-value
  `VLLM_ASCEND_ENABLE_NZ` A/B with its four dated, contradicting note lines, and the
  artefacts both runs owe before any arithmetic.
- `graph-mode-slower-than-eager.md` — "ACLGraph 比 eager 慢" on a working serve,
  often with the reporter's own ablation and `path:line` (#5124, Qwen spec decode at
  v0.11.0rc3): the cited block re-read at the pin, what the line compensates for, the
  patch that keeps the contract, and why this axis is not `spec-decode-gain-accounting.md`.
- `pre-0.8-vl-vs-transformers.md` — a v0.7.x-era VL perf report whose reference point
  is transformers (#1427, vllm-ascend `v0.7.3rc1` + vllm `v0.7.3`, Qwen2.5-VL-3B TP2 on
  two 910B4): the two inert flags in the pasted command (`--enable-prefix-caching` is
  forced off for multimodal under V0 at `arg_utils.py:1129-1134`, chunked prefill is off
  at `:1170-1173`, `--multi-step-stream-outputs` is already the default), the
  registry-level vision gap (only `Qwen2VLForConditionalGeneration` is patched:
  `vllm_ascend/models/__init__.py:5-9` → `_npu_flash_attention_unpad` at
  `qwen2_vl.py:87-101`, while Qwen2.5-VL keeps the upstream
  `get_vit_attn_backend` → TORCH_SDPA per-image Python loop at
  `qwen2_5_vl.py:227,304-320`), the per-vision-block TP all-gather both paths pay, and
  why no graph-mode or `VLLM_USE_V1=1` lever exists at that ref.
