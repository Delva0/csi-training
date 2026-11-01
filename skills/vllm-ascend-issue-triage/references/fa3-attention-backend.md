# FA3 (`FLASH_ATTN`) backend: how it gets selected, and the mixed-batch/spec-decode metadata defect

Applies when a report names `attention_config {"backend": "FLASH_ATTN"}`, `fa3_v1.py`,
`AscendFABackend`, `flash_attn_npu_v3`/`flash_attention_npu`, "training-inference consistency",
or a crash whose traceback ends in `vllm_ascend/attention/fa3_v1.py::forward_impl`. It is a
*dense*-attention backend; MLA/SFA models never take this path (raising
`ValueError: FA3 backend does not support MLA and SFA.` when requested).

Read at `v0.23.0rc1` = `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db` and `v0.23.0` = `5cb98caaad`
of `/repos/vllm-project/vllm-ascend`, plus upstream `v0.23.0`; the fix state was checked on
`main` (`1bd23a0a9`, 2026-11). Nothing was executed on an NPU — the surfacing-point statements
below are labelled as such.

## First: is FA3 even in play?

`platform.py:806-809` (`Platform.get_attn_backend_cls`) returns
`vllm_ascend.attention.fa3_v1.AscendFABackend` only when **all** of these hold:

- `selected_backend == AttentionBackendEnum.FLASH_ATTN`, i.e. the serve line carries
  `--attention-config '{"backend": "FLASH_ATTN"}'` (an `attention_config` dict in the `LLM(...)`
  call is the same object);
- `attn_selector_config.use_batch_invariant`, which is `envs.VLLM_BATCH_INVARIANT`
  (upstream `vllm/v1/attention/selector.py:102`, `envs.py:598` — `VLLM_BATCH_INVARIANT=1`);
- `_validate_fa3_backend` passes (`platform.py:833-856`): dense only (see above) and
  `flash_attn_npu_v3` importable with a `flash_attn_with_kvcache` attribute, else `ValueError`
  naming the missing package/attribute.

Without the env var the request is *not* an error: the plugin logs "FA3 will not be enabled when
not in training-inference consistency scenario ... will use its registered plugin backend
instead" and falls through to `attention_v1.AscendAttentionBackend` (the FIA path). So a serve
line that keeps `FLASH_ATTN` but drops `VLLM_BATCH_INVARIANT=1` (or the reverse) never runs this
code — check both before pinning a `fa3_v1.py` frame. Batch-invariance gating itself, the
`PIECEWISE` requirement and the documented `FULL`/`FULL_DECODE_ONLY` exclusion live in
[batch-invariance.md](batch-invariance.md).

## The defect: prefill rows sliced by token count in a mixed batch (`#17991`)

Recorded report: `v0.23.0rc1`/vLLM `0.23.0`, Qwen3-0.6B BF16, `enforce_eager=True`,
`max_num_batched_tokens=256` (so decode and chunked-prefill requests share a batch),
`attention_config {"backend": "FLASH_ATTN"}`, `VLLM_BATCH_INVARIANT=1`, ngram speculative
decoding (`num_speculative_tokens: 2`), 8 mixed short/long prompts. Crash: aicore exception at
the first mixed batch, `RuntimeError: ACL stream synchronize failed, error code:507015` /
`rtDeviceSynchronizeWithTimeout ... reason=aicore exception`, reported at
`fa3_v1.py:122`, and the same workload completes with speculative decoding disabled.

At `v0.23.0rc1` `AscendFAImpl.forward_impl` (`fa3_v1.py:99-145`) splits one kernel call per part
and mixes the two index spaces — requests vs tokens (`num_decode_tokens > num_decodes` exactly
when the decode part carries spec tokens):

```python
# decode part (line 116-126): block_tables by REQUESTS -- correct
attn_metadata.block_tables[:num_decodes, :],
attn_metadata.query_start_loc[: num_decodes + 1],
attn_metadata.seq_lens[:num_decodes].npu(),
False,                                     # causal=False for a multi-token decode step
# prefill part (line 128-137): block_tables by TOKENS, offsets not rebased
attn_metadata.block_tables[num_decode_tokens:, :],
attn_metadata.query_start_loc[num_decodes:],
attn_metadata.seq_lens[num_decodes:].npu(),
```

Three defects, in the order the fix names them: (1) prefill `block_tables` rows sliced by
`num_decode_tokens` instead of `num_decodes`, so the page table and the `seq_lens`/`query` slices
describe different requests — with `num_decode_tokens + num_prefills > n_requests` the slice is
short or empty; (2) prefill `query_start_loc` handed over unrebased, so the first cumulated
offset is `num_decode_tokens`, not 0; (3) `causal=False` on a decode part that verifies more than
one token. All three are invisible without speculative decoding, because then
`num_decode_tokens == num_decodes` and offsets start at 0 — which is exactly the reporter's own
control ("the same workload completes when speculative decoding is disabled"). The `507015` at
line 122 is the **next** `.npu()` host sync after the bad prefill launch (deferred device-error
reporting — candidate surfacing point, not a measured one; the aicore fault itself is on device).

Independent corroboration in the `#17991` thread (collaborator, 2026-10-14, `v0.23.0` CPU AST
probe of the shipped `forward_impl`, no NPU): with `num_decodes=2, num_decode_tokens=8,
num_prefills=1, n_requests=3` the recorded prefill call passes `block_tables[8:8]` (0 rows)
against `seq_lens[2:3]` (1 row), and with `num_decode_tokens == num_decodes` both start at row 2 —
one mechanism, several failing assertions, not several bugs.

## Fix boundary

`5e1647cf9` — "[BugFix][Attention] Fix FA3 mixed speculative decode semantics (#18068)",
2026-10-15, whose body states **"Fixes #17991"** and names the same three changes
(`block_tables[num_decodes:, :]`, `prefill_query_start_loc = query_start_loc[num_decodes:] -
num_decode_tokens`, `causal = num_decode_tokens > num_decodes`). It is **main-only**: not in
`v0.23.0rc1`, not in `v0.23.0`, and `git tag --contains 5e1647cf9` is empty on this clone (newest
tag `v0.30.0`), so no tag is an upgrade path and an "upgrade fixes it" answer would be wrong.
Quote the commit as the fix on `main` and give a code-path workaround instead.

Separate, still-open in that PR's own words: "the independent physical-page addressing fix,
including `_as_fa3_paged_cache` and page-ID remapping, has been moved to `#18729` and is not
included" — so metadata correctness alone does not promise a real-model layout with strided
physical KV pages. Overlapping work to check before citing a duplicate: `#17774` (request-based
slice), `#18070` (mixed-batch). The PR's test section is unit-level (7 focused CPU/FP32-reference
cases plus a retained Qwen3-0.6B ngram-vs-ordinary regression) and says so.

## Workarounds, cheapest first

1. Drop `VLLM_BATCH_INVARIANT=1` (or the `{"backend": "FLASH_ATTN"}` selection) and keep the
   default FIA backend — verified from the selection gate above: without the env var the plugin
   logs and uses `AscendAttentionBackend`, i.e. a different metadata path (code-path inference,
   not a run). Costs determinism and the FA3 kernels.
2. Disable speculative decoding for that deployment (reporter-confirmed: the same workload
   completes), or keep batches homogeneous (pure prefill / pure decode) — the mixed part is the
   trigger.
3. Keep serving on a build containing `5e1647cf9`; validate the KV-page layout separately against
   `#18729` if the model's cache is not densely paged.

## Boundaries

- Verified in source at the refs above: the selection gate, the four call arguments, the fix
  diff, the ancestry/tag result, the thread's probe numbers. Not verified: any NPU run, the exact
  device fault (aicore vs MTE) for a given reporter, and whether `#18729` is required for that
  model's KV layout.
- Line numbers move: re-anchor with `git grep -n 'num_decode_tokens' <tag> -- vllm_ascend/attention/fa3_v1.py`
  and check `get_name()`'s `VLLM_USE_V2_MODEL_RUNNER` split before quoting a backend name.
- Common misroute: a `507015`/aicore body with an FIA/`attention_v1.py` frame is *not* this class —
  see [spec-decode-metadata-failures.md](spec-decode-metadata-failures.md) for the FIA
  `actualSeqLengthsKv` tiling reject and [log-signatures.md](log-signatures.md) for the shared
  device-error codes.
