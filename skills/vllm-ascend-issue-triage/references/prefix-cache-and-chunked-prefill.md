# Prefix caching on vLLM Ascend: hit rate, chunked prefill, cache-hit correctness

Four recurring report shapes on vLLM Ascend, with the code facts that bound them. Some recorded
investigations reached no confirmed root cause; the pooling/embedding one has a verified code path
and a released fix, and the context-parallel one is corroborated by source plus a reporter's own
resolution. What follows separates verified source behaviour from candidates.

For the observability side — `vllm:kv_cache_usage_perc`, the log lines that print the
KV-cache total and per-step usage, and why `vllm:cache_config_info` labels read `None` on V1 —
use [kv-cache-metrics-and-hbm-usage.md](kv-cache-metrics-and-hbm-usage.md); this file covers
hit rate, block sizing and cache-hit correctness.

## Prefix caching reports "enabled" but never hits

Report template (#10710, `https://github.com/vllm-project/vllm-ascend/issues/10710`,
vLLM 0.20.2 + vllm-ascend 0.20.2rc1, DeepSeek-V4-Flash W8A8, DP=2/TP=4, `--block-size
128`): `vllm:cache_config_info` shows `enable_prefix_caching="True"`,
`vllm:prefix_cache_queries_total` grows after every request, while
`vllm:prefix_cache_hits_total` and
`vllm:prompt_tokens_by_source_total{source="local_cache_hit"}` stay 0 across 8 identical
serial requests of ~10102 prompt tokens. The service is healthy (HTTP 200).

### Verified upstream mechanics

- `vllm/v1/core/kv_cache_utils.py::resolve_kv_cache_block_sizes` (present at v0.20.2)
  returns `(scheduler_block_size, hash_block_size)`. Single KV-cache group:
  `block_size * dcp * pcp` for both. Hybrid groups: `scheduler_block_size` is the **LCM**
  of all group block sizes, while `hash_block_size` is the **GCD** of the group block
  sizes or the `cache_config.hash_block_size` override (v0.20.2; renamed
  `prefix_match_unit` by v0.30/main).
- `vllm/v1/engine/core.py:141-151` (v0.20.2) hands that `scheduler_block_size` to the
  scheduler as its block size. Prompt positions and cache reuse therefore advance in
  LCM-sized blocks even when hashing is finer.
- Consequence to check first: a model whose hybrid groups include a much larger
  compressed block makes the effective reusable-prefix granularity much larger than the
  configured `--block-size`. If that LCM exceeds the reported prompt length (~10102
  tokens in #10710), zero hits are expected behaviour and the real ask is documentation
  of the minimum reusable prefix — not a bug hunt.
- Later releases add per-group gating: `kv_cache_spec.prefix_cacheable` is consulted in
  `kv_cache_manager.py`, `kv_cache_coordinator.py` and `kv_cache_utils.py` on main (the
  attribute does not exist at v0.20.2). On main/`v0.30+` the startup log prints
  `kv cache group sizes ...` and `kv lcm block sizes ...`
  (`vllm/v1/core/kv_cache_utils.py:865-866` on main; only the lcm line at v0.30.0). At
  v0.20.2 those lines are absent — read the resolved values from
  `resolve_kv_cache_block_sizes` and the model's KV-cache group specs instead.

### The documented Qwen3.5 requirement: `--enable-prefix-caching --mamba-cache-mode align`

The Qwen3.5 tutorials state the constraint instead of leaving it to the LCM arithmetic above:
at `v0.18.0` (`docs/source/tutorials/models/Qwen3.5-397B-A17B.md:145`,
`Qwen3.5-27B.md:137`; the line moves by one or two per tag through `v0.21.0rc1`) — "To enable it, for
mamba-like models Qwen3.5, set `--enable-prefix-caching` and `--mamba-cache-mode align`. Notice the
current implementation of hybrid kv cache might result in a very large block_size when scheduling.
For example, the block_size may be adjusted to 2048, which means that any prefix shorter than 2048
will never be cached."

So a Qwen3.5 serve line that enables prefix caching without `--mamba-cache-mode align` is a
documented configuration gap to flag before any kernel reading — the recorded `v0.20.2rc1`
Qwen3.5-35B-A3B report (vllm-ascend#10405) has exactly that, with `--max-model-len 1M` and
`--enable-prefix-caching`. The docs' own "adjusted to 2048" is the same mechanism as the LCM
consequence above, and it answers a hit-rate complaint without reading code; it does **not** explain
wrong or mixed-language text, which belongs to
[sampling-and-output-quality.md](sampling-and-output-quality.md). The requirement follows the
hybrid models to later tags (`--mamba-cache-mode align` in `Kimi-K3.md` at `v0.23.0`,
`Qwen3.8-Flash-Next.md` on `main`).

### Candidate causes and next checks

- **Alignment larger than the prompt** (candidate, best supported): read the group block
  sizes / resolved scheduler block size, compare with the reported prompt length, then
  retest with a prompt above the alignment (or a model/version without a compressed
  group). Recorded A/B data from #10710 (identical requests 2.323 s vs different-system
  control 2.857 s with 0 hits) does not show a cache effect and must not be read as one.
- **Data-parallel routing**: hits are per-engine local state. With DP=2 and request
  round-robin, identical requests alternate engines, which can depress hit rates — but
  both engines reported 0 in #10710, so routing alone does not explain that report.
- `--enable-prefix-caching` plus `--block-size 128` is not itself evidence of anything;
  resolve the actual block sizes before theorising.
- **In a PD deployment, resolve the P engine's per-DP hit counters before using the alignment
  bound.** Hits are per-engine state and only the prefill engine hashes the prompt, so a P node
  served as 2DP×8TP holds two independent caches (a round-robin split caps the ceiling near 50 %,
  it does not zero it). Recorded shape (#12093): vllm-ascend `v0.22.1rc1` on four A2 hosts (P 2DP/8TP,
  D 4DP/4TP), `aisbench --dataset_type prefix_cache --input_len 30720 --repeat_rate 0.9`,
  0 % with and without MTP; the thread (reporter restating the command, bot `wait-feedback`,
  `stale`) carries no maintainer diagnosis, and the attachments holding the serve lines were not
  retrievable. A prompt larger than one alignment unit should still register one hit, so the LCM
  arithmetic is necessary but not sufficient there — ask for `--block-size`, the P serve line and
  `vllm:prefix_cache_hits_total` per P engine.

### Context parallelism multiplies the reusable unit (reporter-resolved, source-corroborated)

Report template (#13750, `https://github.com/vllm-project/vllm-ascend/issues/13750`, image
`quay.io/ascend/vllm-ascend:v0.23.0rc1`, GLM-5.2 W4A8C8, A2/A3, `--tensor-parallel-size 8`, same
chat request repeated, one configuration adding DP=4 + `MooncakeConnectorV1` for the 1M case): with
`--decode-context-parallel-size 8` the identical request never registers a prefix hit, and removing
DCP makes the second identical request hit; the reporter also measured ~1966 tokens with no hit. An
automated-response comment echoed a maintainer's suggestion blaming DCP-inflated prefix units plus
an MTP last-block drop (upstream #50438 / #50897); the reporter rejected the MTP half — their script
has no `--speculative-config` and 2048-token inputs still missed — and later closed with the
mechanism below.

- Verified upstream: `resolve_kv_cache_block_sizes` at vLLM `v0.23.0`
  (`vllm/v1/core/kv_cache_utils.py:593-656`) returns `(bs, bs)` with
  `bs = cache_config.block_size * dcp * pcp` for `len(groups) <= 1`, and its docstring says the
  scheduler block size is the token-alignment invariant for `num_computed_tokens` rounding while
  `hash_block_size` equals it for a single group. So `--block-size 128` with DCP=8 hashes and
  reuses in 1024-token units — the reporter's "128 becomes 128*8 = 1024 before it can hit".
- The same function raises `ValueError("Hybrid KV cache groups with multiple block sizes do not
  support context parallelism ...")` when there is more than one KV-cache group and `dcp`/`pcp` > 1.
  A DCP-enabled run therefore necessarily has a single group, which is what puts it on the
  multiplier branch — check the group count before proposing the hybrid/LCM story for such a report.
- Reporter's own caveat, worth repeating back: judge the input by its **tokenized** length, not by
  characters or by the number typed in the report. That is the cheap first check.
- Open point, do not paper over it: a prompt longer than one multiplied block still contains one
  alignable block, so the multiplier alone predicts a partial hit, not a hard zero. If the
  tokenized prompt is ≥ `block_size * dcp * pcp`, the multiplier is necessary but not sufficient and
  the remaining candidates (request-routing between DP engines, the MTP last-block drop, an
  engine-side cache reset) stay open. The recorded pass stopped there.

### DeepSeek-V4 compressor groups: `block_size × compress_ratio` sets the alignment (v0.20.2rc1)

Report shape (`https://github.com/vllm-project/vllm-ascend/issues/10365`, vLLM 0.20.2 +
vllm-ascend 0.20.2rc1, DeepSeek-V4-Flash W8A8 + MTP, A2, GSM8K perf run): the PD serve reports a
prefix-cache hit rate of ~0 (max ~3 %) where the same model on a single node reports ~16 %. The
prefill node carries `--enable-prefix-caching --block-size 128
--no-disable-hybrid-kv-cache-manager` and `VLLM_ASCEND_APPLY_DSV4_PATCH=1`; the decode node has
`--no-enable-prefix-caching`. Six comments, closed stale: a maintainer pointed at #10354/#10353
("reducing the `block_size`", delaying eviction) and said the PD case was unclear; a reporter who
tried #10298 measured no change.

Verified at `v0.20.2rc1` in the clone:

- The model registers extra KV-cache groups for the DSA compressor state with **hardcoded** block
  sizes: `vllm_ascend/models/deepseek_v4.py:516-531` passes `block_size=8` for
  `compress_ratio == 4` and `block_size=32` for `compress_ratio == 128`, under a TODO saying they
  should follow a configurable block size.
- `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:140-147` computes
  `block_sizes = [spec.block_size * getattr(spec, "compress_ratio", 1) ...]` and
  `self.lcm_block_size = lcm(*block_sizes)`, with the in-file note "NOTE: use 16k as the alignment
  tokens for model with compress ratio". `:217` passes it as
  `alignment_tokens=self.lcm_block_size` to each group's `find_longest_cache_hit`, and `:226-227`
  rescales the resulting hit length by `compress_ratio` again.
- That wrap is opt-in: `patch/platform/__init__.py` imports `patch_kv_cache_coordinator` only
  under `envs.VLLM_ASCEND_APPLY_DSV4_PATCH` (`envs.py:116`, default `0`). A DSV4 run without the
  env var is on the unpatched coordinator and this alignment story does not apply to it.
- Fix availability: `bbeba20ce` "[Feature][DSv4] Support compressor block size [32,64,128] to
  improve automatic prefix cache hit rate (#10354)" first ships in `v0.21.0rc1` (then `v0.22.1rc1`,
  `v0.23.0`, ...). `d376f722e` (#10298, "Fix DeepSeek-V4 compressed prefix lookup") is contained in
  **no tag**, which matches the reporter's "this pr doesn't work" — never offer it as the fix.
  `git log --all --grep='10353'` finds no commit.

So on a DSV4 hybrid run the reusable-prefix granularity is an LCM over `spec.block_size ×
spec.compress_ratio` of **every** attention group, not over the compressor groups alone — 16k tokens
at `--block-size 128`, which is what the in-file note says. Enumerate the groups when you recompute
it, because the terms come from three different files:

- the main compressed KV spec, `MLAAttentionSpec(block_size=cache_config.block_size,
  compress_ratio=self.compress_ratio)` in upstream `vllm/model_executor/layers/deepseek_v4_attention.py:704-718`
  (v0.20.2): `128 × 4 = 512` on C4 layers and `128 × 128 = 16384` on C128 layers — **this is the
  16k term**;
- the indexer cache, `AscendMLAAttentionSpec(block_size=128, compress_ratio=4)` in
  `patch/worker/patch_deepseek_compressor.py:84-96`: `512`;
- the compressor state caches (`AscendCompressorStateCache`, same file `:26-66`): `SlidingWindowMLASpec`
  with `block_size` 8 / 32 and **no** `compress_ratio`, so `getattr(spec, "compress_ratio", 1)` is 1
  and the terms are `8` and `32` — reading only `models/deepseek_v4.py:516-531` and multiplying 8×4 /
  32×128 gives 32/4096 and an LCM of 4096, which under-counts by one order of magnitude.

`find_longest_cache_hit` documents that the returned hit length must be a multiple of
`alignment_tokens` (`vllm/v1/core/single_type_kv_cache_manager.py:346,366` at v0.20.2), so on a run
with a C128 layer a prompt below 16384 tokens gets a hit length of 0 — the arithmetic bound and the
strongest candidate for the ~0 % rate, and why the upgrade answer is "v0.21.0rc1 or newer"
(`--block-size 32` there, or the configurable compressor block size, shrinks the alignment). Still open: the PD-versus-single-node delta is unexplained;
per-DP-rank local caches (hits are per-engine state), the decode node's caching being off, and a
different prompt mix remain candidates. Resolve it by logging `lcm_block_size` at start-up and
re-running with a prompt longer than one alignment unit.

## TTFT changes with --max-num-batched-tokens

Report template (#1886, `https://github.com/vllm-project/vllm-ascend/issues/1886`,
vLLM 0.9.0 + vllm-ascend 0.9.0rc2, `VLLM_USE_V1=1`, chunked prefill, Qwen2.5-14B-Instruct,
same 4962-token prompt, temperature 0): TTFT 936 / 714 / 325 ms for
`--max-num-batched-tokens` 8192 / 4096 / 2048, and the returned first tokens differ per
budget. A second run (llmperf, chunk fixed at 4096) shows TTFT ≈ constant ~700 ms while
the prompt grows 4000 → 8000 → 16000 tokens.

### Verified upstream invariants at the reported versions

- vllm v0.9.0 `vllm/v1/worker/gpu_model_runner.py:661-665`: partial (chunked-prefill)
  requests are sampled "for simplicity" but the note states the sampled tokens "will be
  ignored"; `vllm/v1/core/sched/scheduler.py:814` states the invariant
  "EngineCore returns no partial prefill outputs." The vllm-ascend v1 runner carries the
  same note (`vllm_ascend/worker/model_runner_v1.py:1020-1023` at v0.9.0rc2; line 1443 at
  v0.11.0; `worker/v1/model_runner.py:1755` on main).
- So the first accepted output token should appear only after the prompt's last chunk.
  TTFT that scales with the chunk budget instead of the prompt length conflicts with that
  invariant, and temperature-0 outputs that differ between budgets are a correctness
  signal, not just a timing curiosity.

### Candidate cause and next checks

- **Candidate (untested):** the partial-prefill sampled token leaks into the output stream
  on the Ascend v1 path, so a token produced from the first chunk is returned. Check the
  masking of sampled tokens for requests whose prompt is not fully computed in
  `vllm_ascend/worker/model_runner_v1.py` (the `sample_indices` / partial-request mask
  around lines 1020 and 1357 at v0.9.0rc2) and the scheduler-side discard in
  `vllm/v1/core/sched/scheduler.py` around `update_from_output`. A discriminating test:
  a prompt shorter than one chunk, and a prompt giving exactly 2 chunks, at the same
  `--max-num-batched-tokens`; TTFT should jump at the chunk boundary only if a token is
  emitted before the prompt completes.
- Do not present the "first token right after chunked prefill" explanation as established;
  the recorded trace never confirmed where the token is emitted.

### TTFT at concurrency 2, "I disabled chunked prefill and nothing improved" (#8126)

Report template: vllm-ascend `0.18.0rc1`, Qwen3-8B, C=2, the second request's TTFT degrades; the
body proposes chunked prefill batching the next prefill with the previous decode, then says
disabling chunked prefill changed nothing. No `collect_env.py` output and no serve line.

- First establish that the flag took effect at the reported upstream version (`vLLM v0.18.0`):
  `enable_chunked_prefill: bool | None = None` (`vllm/engine/arg_utils.py:518`) is resolved from
  `model_config.is_chunked_prefill_supported` (`:2065-2069`); disabling a model that supports it
  only logs "This model does not officially support disabling chunked prefill ..." (`:2080-2085`);
  `SchedulerConfig.__post_init__` prints "Chunked prefill is enabled with
  max_num_batched_tokens=%d" (`vllm/config/scheduler.py:230-234`) only while it is on, so the
  absence of that line in the reporter's log is the cheap confirmation. `verify_max_model_len`
  (`:251-263`) raises when `max_num_batched_tokens < max_model_len` and prefill is not chunked, so
  a run that starts with the flag off also proved the token budget was raised.
- Then check the plugin's force-enable gates, which silently override the CLI: at `v0.18.0rc1`
  `vllm_ascend/platform.py:442-447` swaps in `SchedulerDynamicBatch` and sets
  `enable_chunked_prefill = True` when `SLO_limits_for_dynamic_batch != -1` (key default `-1`,
  `vllm_ascend/ascend_config.py:119`), and `ascend_scheduler_config.enabled` does the same
  ([env-vars-and-additional-config.md](env-vars-and-additional-config.md),
  [glm-moe-accuracy.md](glm-moe-accuracy.md)). A reporter who set either key never measured
  "chunked prefill off".
- Boundary: the recorded investigation stopped after these greps (two `Truncated call was not
  executed` steps) and established no cause. With the flag genuinely off, the remaining candidates
  — admitting the second request behind the in-flight decode batch, `max_num_batched_tokens` /
  `max_num_seqs`, and the decode graph mode — are untested here. Ask for the serve line, the
  scheduler log lines and a per-step timeline before naming one.

## Pooling/embedding accuracy: first request matches, repeats diverge

Report template (`https://github.com/vllm-project/vllm-ascend/issues/7192`, vllm-ascend
v0.15.0rc1, Qwen3-Embedding-8B, A2, `--runner pooling --enable-prefix-caching`): cosine
similarity against a GPU reference is ~0.9998 for most requests but 0.28-0.32 for others, and
`dimensions=4096` looked clean while `dimensions=1024` did not. The reporter's own follow-ups
narrow it: the first request of a service matches the GPU, later ones diverge, and
`--no-enable-prefix-caching` makes all of them agree. The same shape is in #5725 (0.6B,
cosine ~0.58 on later requests; disabling prefix caching fixed it) across v0.13.0rc1,
v0.14.0rc1 and v0.15.0rc1.

### Verified code path at v0.15.0rc1

- `vllm_ascend/attention/attention_v1.py:975` routed **every** pooling model to
  `_forward_encoder_attention`, which calls `torch_npu.npu_fusion_attention` with
  `actual_seq_qlen=actual_seq_kvlen=attn_metadata.actual_seq_lengths_q` (`:855-890`). The paged
  `kv_cache` is written by `reshape_and_cache` immediately above (`:972-973`) but never read
  back, so tokens already in the prefix cache are absent from the attention input. Whatever
  pooling vector comes out is then computed from the scheduled suffix only.
- `AscendMetadata.causal` defaults to `True` (`:194`) and is copied from
  `common_attn_metadata.causal` (`:316`), so a decoder-only embedder such as Qwen3-Embedding
  took that branch. Which requests lose similarity therefore tracks which requests hit the
  cache, which is why one file shows both ~0.999 and ~0.28.
- The fix (#7452, commit `bd004043d`, merged 2026-03-28) narrows the branch to
  `model_runner_type == "pooling" and not attn_metadata.causal`, so causal pooling models fall
  through to `forward_impl` (paged/fused infer attention) while true encoders keep
  `npu_fusion_attention`. The PR's own test is `Qwen3-Embedding-0.6B` with/without caching,
  tolerance 1e-2.
- Availability, from `git tag --contains`: `bd004043d` on the main line first ships in
  `v0.19.1rc1`; the v0.18.0 cherry-pick `d3de7333d` (#7894) is in `v0.18.0` and `v0.18.0rc1`.
  No 0.13-0.17 release contains either, so "upgrade within the release line" is wrong for
  those reports.

### Workaround and next check

- `--no-enable-prefix-caching` is reporter-verified on both threads (outputs then match the GPU
  reference). Offer it as the workaround, with the cache-hit rate loss stated.
- Do not turn the dimension statement into a mechanism. Truncation/renormalisation is applied to
  the pooled vector, so it cannot change which tokens the attention saw; the recorded pass noted
  but never verified where `dimensions` is applied at that tag, and the thread's follow-up
  identifies cache hits as the real delta. Treat 1024-vs-4096 as an unreconciled co-variate
  unless the truncation site is read at the reported tag.
- Discriminating test for a new report: the same prompt twice, caching on, and the same pair
  with `--no-enable-prefix-caching`. Only the cached second request diverges, and only for a
  pooling runner, if this is the cause. Wrong second-request output from a non-pooling LLM is a
  different class; do not apply this fix path to it.

## Dense LLM, garbled output only on cache-hit repeats (no root cause in the thread)

Report template (`https://github.com/vllm-project/vllm-ascend/issues/5269`, vLLM 0.10.0 +
vllm-ascend 0.10.2rc1): the same request sent 10 times returns normal text once and garbled text on
the 9 repeats that hit the prefix cache. The thread is closed `stale`/`wait-feedback` with no serve
line, no logs and no maintainer root cause — the only replies ask for a newer version and then the
full start-up command. This is *not* the pooling class above; do not offer
`--no-enable-prefix-caching` + `model_runner_type == "pooling"` as the explanation.

What is verified at the reported tag (`/repos/vllm-project/vllm-ascend`, `v0.10.2rc1`):

- `vllm_ascend/worker/model_runner_v1.py:792-809` `_make_attention_mask` returns
  `attn_mask_builder.get_attn_mask(128, ...)` — a fixed 128-token mask — for
  `AscendAttentionState.PrefillCacheHit`, while the `PrefillNoCache` branch sizes the mask from
  `max(seq_lens)`. So a cache-hit prefill is the one branch whose mask ignores the request length.
- That branch is reachable only from `_build_attn_state`
  (`worker/model_runner_v1.py:1426-1447`), whose fall-through is
  `elif not ascend_config.ascend_scheduler_config.enabled or self.chunked_prefill_enabled:
  ChunkedPrefill ... else: PrefillCacheHit`. `ascend_scheduler_config.enabled` defaults to `False`
  (`vllm_ascend/ascend_config.py:158`), and the runner then forces
  `self.chunked_prefill_enabled = True` (`worker/model_runner_v1.py:264-268`), so **a default run
  never takes this branch**: it needs `--additional-config '{"ascend_scheduler_config":
  {"enabled": true}}'` *and* `chunked_prefill_enabled` false. That is why the serve line is the
  first thing to ask for, and why the fixed-128 mask must not be presented as the cause without it.
- Consumers of the state span the attention, quantized and graph paths
  (`attention/attention_v1.py:555`, `quantization/w8a8.py:399-417` —
  which has its own `PrefillCacheHit` branch — `torchair/torchair_attention.py:380,416`), so code
  there cannot be read as one backend's bug.

Next checks for a fresh report: the full `vllm serve` line (ascend scheduler? chunked prefill?) and
the start-up log lines that name it; whether the garbling tracks the *cached* prefix length
(same prompt, shortened prefix) or the number of repeats; and a same-version run with
`--no-enable-prefix-caching` as the attribution test. Then, if the symptom is a wrong-attention one rather
than a decode-time one, `_make_attention_mask` is the single place worth instrumenting — the mask
width against `num_scheduled_tokens` of the cache-hit prefill.

## Boundaries

- Execution on hardware was not exercised in these investigations; every statement above is
  either read from tagged source or taken from the reporters' logs.
- The vllm-ascend clone's `main` is far ahead of these reports (future-dated release
  tags); mechanisms such as `prefix_cacheable`, `prefix_match_unit` and
  `resolve_cache_hit_alignment_tokens` did not exist at the reported versions and must not
  be used to explain them.
- The hit-rate and TTFT investigations stopped before a verified root cause; their next-check
  items are continuation points, not conclusions. For DCP the multiplier is verified upstream at
  v0.23.0 and the reporter closed the thread with it, but the zero-hit case is not fully explained
  by arithmetic alone (see the open point above).
- The dense-LLM cache-hit section is a read path plus an applicability gate, not a diagnosis: the
  fixed-128 `PrefillCacheHit` mask is real at v0.10.2rc1, and so is the gate that makes a default
  V1 run avoid it. Nothing recorded links it to the reported garbled text.
- The pooling/embedding section is different: the branch at `:975`, the fix diff and its
  `git tag --contains` results were read in the clone, and the workaround is reporter-verified.
  What is *not* verified is the numerical mechanism end to end (no kernel-level reproduction) and
  the 1024-vs-4096 dimension claim.
