# Throughput cliffs at a running-sequence threshold (no error, idle NPUs)

Scope: a `[Performance]`-shaped report where per-request decode time steps up 2-4x at a specific
number of *running* sequences, with no exception, no warning, engine `Waiting: 0`, KV cache below
100%, and the NPUs under-used at the slow end (lower AICore and power per card than on the healthy
side). What the recorded instance excludes (below) is more solid than what it establishes: the
candidate direction is host-side or fixed-size state indexed by the running-sequence count, reached
through a context-parallel collective, and it is neither a kernel, a quantization effect nor the
HCCL driver. Start from the exclusions, not from the candidate.

## First three discriminators, in this order

These are cheap and each removes a whole branch:

1. **Per-engine or per-node?** Re-shard the same node (e.g. `tp=8/dp=1` vs `dp=2/tp=4`) so the
   per-engine sequence count drops below the threshold. If the threshold follows the per-engine
   count, the wall is per-engine state; if it stays at the node total, look at the host or the
   node-level runtime instead. This also yields the only workaround recorded for the class:
   re-shard so no engine crosses the threshold (the reporter measured the healthy shape at
   <=224 running sequences per engine).
2. **KV pressure?** `Waiting: 0` throughout plus a KV peak in the low 80s at the collapse rules out
   admission queueing and cache exhaustion — per [kv-cache-metrics-and-hbm-usage.md](kv-cache-metrics-and-hbm-usage.md),
   read the engine's own `Waiting`/KV fields rather than the client's latency curve.
3. **HCCL driver?** Run the same collective standalone (`dist.all_gather_into_tensor`, torchrun over
   all ranks, the model's dtype and hidden size, across the token counts the failing step uses). A
   flat ~0.3 ms/op across every size says the driver is not the slow path, whatever the model-level
   profile shows.

## A py-spy frame inside a collective marks where the stream stalls

NPU streams execute in order, so a worker sampled in `dist.all_gather_into_tensor` while the step
is 3x slow only says the device stream is waiting at that point, not that the collective is the
cause — the same reasoning as a Python frame that is merely the last one on the stack. Report it as
the stall position and keep the mechanism open.

## Candidate direction after the exclusions

The reporter's own reading of the evidence — a "fixed-size structure in the host-side
/model-execution path indexed by running sequence count" — is a candidate, not a proven cause. If
you pursue it, search for structures sized by `max_num_seqs` / running-request count rather than
for the literal threshold:

- `git grep -nE '(^|[^0-9])256([^0-9]|$)' <tag> -- vllm_ascend` mostly returns dtype and block
  sizes. At `v0.23.0rc1` the only literal cap of that shape is
  `distributed/kv_transfer/utils/utils.py:14 MAX_HCCL_REGISTER_REGIONS = 256`, used by
  `validate_register_region_count` (`:428-445`) to **raise** `RuntimeError: Mooncake register_buffer
  region count N exceeds HCCL per-process limit 256` — a KV-transfer/Mooncake-only validator, so it
  is inapplicable to a run with no KV pool and it fails loudly rather than slowly.
- Look instead at per-batch-size caches, capture-size tables and anything keyed by
  `get_num_seqs`/`num_reqs`, then at the collective call path that the profile landed in.

## Locating the collective named by the profile (verified at v0.23.0rc1)

For DSA-CP (DeepSeek-V4 / sparse-attention models with `enable_dsa_cp`, i.e. `enable_sp()`), a
profile frame naming `_maybe_all_gather_and_maybe_unpad_impl (dsa_cp.py:1296)` splits into two
locators that must be quoted separately:

- `vllm_ascend/attention/context_parallel/dsa_cp.py:1296` is the **call site**:
  `torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states_local, need_gather_q_kv)`.
- `vllm_ascend/ops/register_custom_ops.py:40` is the **implementation**
  (`_maybe_all_gather_and_maybe_unpad_impl`), registered as a custom op at `:230`; it branches on
  `flash_comm_v1_enabled` and reads `_EXTRA_CTX.pad_size` / `dp_metadata.num_tokens_across_dp_cpu`,
  so its cost and shape depend on the FlashComm/SP state of that forward — see
  [context-parallel-moe-paths.md](context-parallel-moe-paths.md) for the per-forward gate.

`enable_cpu_binding` (in the reported `--additional-config`) is on by default since
`v0.18.0rc1` (`docs/source/developer_guide/Design_Documents/cpu_binding.md` at the tag;
`--numa-bind` only maps onto it), so its presence in a serve line is not a delta by itself.

## What this class has already survived (recorded, reporter-run, one report)

One DeepSeek-V4-Flash-w8a8-mtp report (`tp=8`, EP, MTP k=1, FlashComm1, `FULL_DECODE_ONLY`,
`--async-scheduling`, A2 8x910B2; `vllm-ascend#14440`) turned every one of these off in isolation at
N=264 without moving the cliff: MTP, cudagraph replay (`--enforce-eager`), `enable_npugraph_ex`,
`--max-model-len` reduced to 1024, `--async-scheduling`, FlashComm1 + `enable_dsa_cp`, and
`multistream_overlap_shared_expert`. MTP mean acceptance stayed 1.6-1.8 on both sides of the cliff,
so the spec-decode acceptance path is not the differentiator. Those are the reporter's numbers, not
maintainer-confirmed, and they bound this report only — do not propose any of them as a workaround
for a similar report without checking whether the new report's configuration makes that toggle
meaningful.

## Boundary

The thread reached no maintainer mechanism (an automated `/wait-feedback` request, then the reporter
supplied the serve line, benchmark table and plog attachments; still open). State the class-level
finding — per-engine threshold, exclusions, re-shard workaround — and label the root cause unknown.
Do not upgrade the candidate "fixed-size structure" into a mechanism without a source line that
sizes or truncates something by the running-request count.
