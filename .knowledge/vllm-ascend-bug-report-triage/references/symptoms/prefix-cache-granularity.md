# "Prefix caching is on and identical requests never hit": the granularity family

## Scope

`--enable-prefix-caching` is set, the service is healthy, there is no traceback, and the hit rate is
~0 (or far below expectation). Every verified member of this family is a **unit mismatch**, not a
cache bug: the cache is keyed on a coarser unit than the tokens the reporter sends. Read the unit
before reading the code:

| member | hit unit | where |
| --- | --- | --- |
| CP (`--decode/prefill-context-parallel-size` > 1) | `block_size × dcp × pcp` tokens | `pcp-dcp-and-mtp.md` §A — do not re-derive it here |
| DeepSeek-V4 / DSA compressed KV groups | one compressed block covers `block_size × compress_ratio` original tokens; write and lookup must agree | this page §1 |
| sliding-window (SWA) layers | blocks of finished requests are recycled, so only in-flight sharing and selectively retained blocks hit | this page §2 |
| PD with DP fan-out | each prefill DP rank owns its own cache, so a shared prefix only hits the rank that already served it | this page §3 (`hypothesis`) |

## Common rules

- **Divide the reporter's prompt by the unit before suspecting a defect.** A prompt shorter than one
  hit unit cannot hit, whatever the request count. Say the number.
- **The write path and the lookup path must use the same key.** A compression ratio applied on one side
  only gives a cache that is populated and never found — valid-looking metrics, zero hits, no error.
- **Only the prefill side of a PD deployment computes prefixes.** A decode node started with
  `--no-enable-prefix-caching` is the expected configuration, so read hit-rate numbers per instance
  and quote which node produced them.
- **Check the tag before blaming the code**: this family has fixes that ship in a later rc, and one
  backport that exists only on a release branch.

## §1 DeepSeek-V4 compressed KV groups (`compress_ratio > 1`)

Verified at `v0.20.2rc1` (`vllm_ascend/core/single_type_kv_cache_manager.py`):

- write: `CompressAttentionManager.cache_blocks` (`:152-162`) does `num_tokens //= self.compress_ratio`
  before delegating to upstream's full-attention manager, so the stored key is a **logical** block of
  `block_size × compress_ratio` original tokens;
- lookup: `CompressAttentionManager.find_longest_cache_hit` (`:166-208`) iterated the incoming
  **physical**-granularity `block_hashes` with `block_size = kv_cache_spec.block_size` (× `dcp × pcp`),
  so the two key spaces only agree by accident;
- and `block_size` is pinned: `vllm_ascend/utils.py` forces `cache_config.block_size = 128` for
  `model_type == "deepseek_v4"` at that ref.

The group structure matters as much as the ratio: `vllm_ascend/patch/platform/patch_kv_cache_utils.py`
(`group_and_unify_kv_cache_specs`, `:58-89` at that ref, "currently only used for DeepseekV4") groups
`MLAAttentionSpec` by `compress_ratio` and `SlidingWindowMLASpec` by `block_size`, so a DSv4 deployment
has **several** KV-cache groups; with CP on, `:47-53` then returns `lcm(group_block_sizes) × dcp × pcp`
— that CP multiplier is `pcp-dcp-and-mtp.md` §A and is not re-derived here. `:237-244` gives
`MLAAttentionSpec` with `compress_ratio > 1` a `CompressAttentionManager` with an admission cap of
`cdiv(max_model_len // compress_ratio, block_size) + 1`; `:245-255` re-applies the upstream `#40946`
cap for `SlidingWindowSpec`/`ChunkedLocalAttentionSpec` (the vLLM #40863 "long request stuck in
waiting" shape).

Negative result worth keeping: `vllm_ascend/core/scheduler.py` does **not** exist at `v0.20.2rc1` (the
schedulers are `recompute_scheduler.py`, `scheduler_profiling_chunk.py`, `scheduler_dynamic_batch.py`),
so `git show <ref>:vllm_ascend/core/scheduler.py` fails — an empty read there is not evidence of a
missing feature.

Fixed later, in this order:

| change | commit | first tag |
| --- | --- | --- |
| #9903 write path uses `block_size * compress_ratio` | `f5c97749f` (2026-06-08) | `v0.21.0rc1` |
| #10297 lookup converts the hash list to the logical block size (+ hybrid coordinator) | `eef0b36b3` (2026-06-11) | `v0.21.0rc1` |
| #10298 the same fix on `releases/v0.20.2rc` | `d376f722e` (2026-06-11) | release branch only, no tag |
| #10354 compressor `--block-size` 32/64/128, default 128 → **32** | `bbeba20ce` (2026-06-12) | `v0.21.0rc1` |
| #13330 compressed prefix-cache hit-length accounting | `d44278a9b` (2026-08-02) | `v0.25.1rc1` |

#10354's own measurements (32k/1k, concurrency 8, DSv4 w8a8 + MTP, repeat_rate 0.9) are the quotable
number for this class: block size **32 → 87.5 %**, 64 → 75.0 %, **128 → 45.3 %** hit rate, with TTFT
1471 / 2661 / 5430 ms. It also states the PD constraint: **P and D must use the same `--block-size`** or
the D side cannot consume the transferred pages.

Workaround ladder for a report at `v0.20.2rc1` or earlier: upgrade to `v0.21.0rc1`+ and set
`--block-size 32` on both P and D (costs a little throughput per block, quoted above); if the reporter
is pinned to the 0.20.2 rc line, cherry-pick `d376f722e` (#10298) — but say it is untested: #10365's
reporter applied #10298 on their build and reported no change in the hit ratio (`reported`), so a
build with only the release-branch backport is still missing the `--block-size` half and the PD
fan-out question (§3). Never present #10298 as a confirmed fix for that reporter.

## §2 Sliding-window / hybrid retention

DeepSeek-V4 is a DSA/SWA hybrid: `MLAAttentionSpec.compress_ratio > 1` plus SWA layers
(`get_manager_for_kv_cache_spec` at the same ref builds `CompressAttentionManager` for the compressed
groups). SWA blocks are recycled when a request finishes, so a prefix that is not retained cannot hit
on the next request, whatever the block size. The relevant upstream work is vLLM #43447 (block-reuse
free-list insertion + selective prefix-cache retention for sliding-window KV), backported for
vLLM 0.21.0 as vllm-ascend **#10353** (closed, **not merged**), gated by
`VLLM_ASCEND_PREFIX_CACHE_RETENTION_INTERVAL` in that proposal. Treat retention as *not available* on
any tag in this clone unless you can point at the code; do not offer the env var as a workaround.

## §3 PD beats single node by an order of magnitude (open)

A PD deployment whose prefill side fans out over many DP ranks gives each rank a private cache, so a
shared prompt prefix has to have been served by that rank before. That halves, quarters or sixteenths
an otherwise healthy hit rate — but it is a `hypothesis` here, not a verified mechanism: no comment in
#10365 and no code read established it. Falsifier: run the same trace against a single prefill rank
(`--data-parallel-size 1`, same prompt mix) and compare hit rate per rank; if the per-rank rate is
unchanged and the aggregate still collapses, the cause is the granularity members above, not fan-out.
The measured baseline in that report — single node ≈ 16 %, PD ≈ 0–3 % — is the reporter's, on a
`v0.20.2rc1` build that predates every fix in §1.

## Checklist

1. Resolve the ref, then read one window: `git show <ref>:vllm_ascend/core/single_type_kv_cache_manager.py
   | sed -n '25,60p;150,215p'` plus the `deepseek_v4` block-size override in `vllm_ascend/utils.py`.
2. Name the unit in tokens (`block_size × compress_ratio`, × `dcp × pcp`) and compare it with the
   reporter's prompt length; state the number.
3. One ancestry check per candidate fix (`scripts/ref_ancestry.sh`), and quote #10354's measured table
   rather than describing it.
4. Answer with: unit → who computes it → fix and first tag → workaround (`--block-size 32` on both
   sides, or the release-branch backport) → what was not verified.
