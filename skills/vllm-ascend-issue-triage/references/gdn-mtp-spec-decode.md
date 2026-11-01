# GDN + MTP speculative decoding triage (Qwen3.6-35B-A3B w8a8)

Verified in the clones at `v0.20.2rc1` / `v0.21.0rc1` / `v0.22.1rc1`, plus upstream `vllm` tags and
the issue threads below. Qwen3.6-35B-A3B is an MoE model with GDN (gated delta net / linear attention)
layers plus an MTP head; the plugin method name in `--speculative-config` is `qwen3_5_mtp`.
Proposer metadata a release never populates (`seq_lens_cpu` under async spec decode,
`PCPManager.mtp_slot_pad`, `ForwardContext.draft_attn_metadatas`) is mapped separately in
[spec-decode-metadata-failures.md](spec-decode-metadata-failures.md).

## Where the state and acceptance plumbing lives

- `vllm_ascend/patch/worker/patch_qwen3_5.py` — rebinds `Qwen3NextAttention.forward` (fused
  qkv/rmsnorm/mrope via `torch.ops.vllm.triton_split_qkv_rmsnorm_mrope`, gated on `"qwen3_5" in
  config.model_type`) and `Qwen3_5DecoderLayer.forward` (per-layer output buffer, FlashComm1 sizing).
  It does **not** patch the MTP head. `patch_qwen3_next_mtp.py` only replaces
  `vllm.v1.worker.utils.bind_kv_cache` (works around a `NotImplementedError` during kv-cache init),
  and no plugin file patches the MTP-head forward — `Qwen3_5MTP` / `Qwen3NextMTP` run upstream code.
  From `v0.20.2rc1` the method→architecture mapping is re-implemented in
  `vllm_ascend/patch/platform/patch_speculative_config.py`, which assigns
  `SpeculativeConfig.hf_config_override = hf_config_override` (a copy of upstream's function), so the
  MTP class a run instantiates is decided there, not by the method string in
  `--speculative-config` (every MTP method normalizes to `mtp` —
  [first-request-only-output.md](first-request-only-output.md)).
- `vllm_ascend/patch/worker/patch_gdn_attn.py` — patches upstream `GDNAttentionMetadataBuilder`
  (chunk metadata, spec/non-spec indices, accepted-token plumbing); imports
  `vllm.v1.attention.backends.gdn_attn`. At `v0.20.2rc1` the accepted-token consumer is
  `_patched_build` (`:709`) → `_patched_build_spec` (`:806`) → `_build_spec_query_start_loc_cpu`
  (`:454`) plus `_build_spec_causal_conv1d_host_meta` (`:651`), which copies
  `attn_metadata.num_accepted_tokens[:num_spec_decodes]` into a pinned CPU buffer (`:670-672`);
  `_build_spec_sequence_masks_cpu` is at `:420`. The `numAcceptedTokens[0]=4 exceeds varlen segment
  length=3` assertion is about that copy, so read those functions before any GDN kernel.
- `vllm_ascend/ops/gdn.py`, `vllm_ascend/ops/triton/gdn_chunk_meta.py`,
  `vllm_ascend/ops/triton/fused_gdn_gating.py`, `vllm_ascend/ops/triton/fla/utils.py`
  (`clear_ssm_states`).
- 310P overrides: `vllm_ascend/_310p/ops/fla/{gdn_310,chunk_gated_delta_rule,fused_gdn_gating,
  fused_recurrent_gated_delta_rule}.py`.
- Upstream model side: `vllm/model_executor/layers/mamba/gdn_linear_attn.py` at v0.21.0,
  `vllm/model_executor/layers/mamba/gdn/base.py` from v0.22.0; metadata in
  `vllm/v1/attention/backends/gdn_attn.py`.

## Verified plugin/upstream divergence at the spec merge

Upstream `vllm` v0.21.0 `gdn_linear_attn.py` merges speculative and non-speculative outputs with
`merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)` (lines 1304-1305). The 310P plugin
replaces that with dim-0 boolean-index assignment because NPU `index_copy_` fails on some layouts
(see [310p-atlas-300i-duo.md](310p-atlas-300i-duo.md)). When tracing any merge/index bug, read both
sides: the traceback may name the upstream file while the executed override is in `vllm_ascend/`.

## Documented known issue (same model, same feature)

`docs/source/user_guide/release_notes.md` at tag `v0.21.0rc1` (section v0.20.2rc1, "Known Issues"):
"Qwen3.6-35B-A3B may shut down when MTP/speculative decoding is enabled, with
`numAcceptedTokens[0]=4 exceeds varlen segment length=3` reported during shape/dtype processing"
(issue #9956). Cite this rather than re-deriving it for reports against v0.20.2rc1/v0.21.0rc1.
The reported tag's **own** file does not carry the entry (`git grep Qwen3.6 v0.20.2rc1 --
docs/source/user_guide/release_notes.md` returns nothing); it is written into the next rc tag's file,
so grep one tag ahead before calling the symptom undocumented.

## Issue-thread evidence for the acceptance-collapse-and-stall symptom

From vllm-ascend#10495 (0.20.2rc1, A2 2 cards, 64K input, KV cache ~99%):

- Reporter: "If we remove MTP, inference was normal" — i.e. disabling speculative decoding restored
  serving on the reported version. That is the strongest reported workaround; it is a reporter
  observation, not a maintainer-confirmed fix.
- Independent reproduction (comment on v0.22.1rc1-openeuler, `Qwen3.6-35B-A3B-w8a8`, **single 910B4,
  TP=1, low KV usage 22%, no waiting requests, no preemptions**): acceptance fell from ~91% to
  `Per-position acceptance: 0.000, 0.000, 0.000` across all speculative positions within ~10 s and did
  not recover, later degrading to zero generation throughput while the API server stayed healthy. Some
  responses were also malformed/repetitive, but stdout multiplexing prevented ruling out client-side
  interleaving.
- That reproduction used a **local patch clamping** `num_accepted_tokens = torch.minimum(num_accepted_tokens,
  spec_query_lens)` to suppress `numAcceptedTokens[0]=4 exceeds varlen segment length=3`; the author
  flagged it as a possible confounder. Treat the clamp as an unvalidated experiment, not a fix.

Consequences for triage: the failure does **not** require KV-cache pressure, waiting requests,
preemption, or TP > 1, so KV-pressure explanations are weak for this symptom; the transition pattern
(healthy load -> idle -> new batch starts healthy -> abrupt all-position zero acceptance -> stall)
points at request-slot / recurrent-state reuse, bonus-token handling, or graph replay with changing
batch sizes — all still candidates.

## Candidate fixes (main only at the clone snapshot)

`git tag --contains <sha>` returns nothing for these, so no released image contains them:

| commit | date | subject |
|---|---|---|
| `87f2b060a` | 2026-10-22 | Fix GDN recurrent state indices exceeding the token count (#19378) |
| `ccc6564de` | 2026-10-26 | [310P] Avoid bool-mask IndexPut in GDN prefill state init (#19295) |
| `52ba694bb` | 2026-10-15 | Gate fused GDN prefill paths on bfloat16 to fix fp16 models crash (#18522) |

Present them as what a maintainer would need to backport, not as an upgrade the reporter can take.

## Next verification steps

1. Re-read the acceptance lines (`SpecDecoding metrics: ... Per-position acceptance rate`) to separate
   "draft rejected" from "engine not stepping"; the stall shows as static Running/non-zero requests with
   frozen token counters and `/metrics` still 200.
2. Test with speculative decoding off on the reported tag to confirm the reporter's workaround, and with
   prefix caching / async scheduling toggled independently.
3. On 310P, combine with the mask-memory check before blaming GDN state — a 310P report may fail at
   engine init rather than in decode.
