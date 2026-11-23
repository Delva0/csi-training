# FA3 mixed prefill+spec-decode batch: the split's index units (#17991)

## Report

vllm-ascend v0.23.0rc1 (`f4a08bdd`) + vllm v0.23.0 (`0fc695fc`), one 8×910B3 host,
`torch_npu 2.10.0.post2`, CANN 9.0.1, FA3 NPU `v0.4.2` (`MinghuasLab/flash-attention-npu`).
Qwen3-0.6B, `enforce_eager=True`, `VLLM_BATCH_INVARIANT=1`,
`attention_config={"backend": "FLASH_ATTENTION"}`, `speculative_config={"method":
"ngram", "num_speculative_tokens": 2, "prompt_lookup_min": 2, "prompt_lookup_max":
4}`, 8 prompts of very different lengths, `max_num_batched_tokens=256`.

```text
File ".../vllm_ascend/attention/fa3_v1.py", line 122, in forward_impl
    attn_metadata.seq_lens[:num_decodes].npu(),
RuntimeError: ACL stream synchronize failed, error code:507015
[Error]: The aicore execution is abnormal.
EE9999: rtDeviceSynchronizeWithTimeout execution failed, reason=aicore exception
Kernel task happen error, retCode=0x26, [aicore exception].
```

"crashes at the first mixed batch; the same workload completes with speculative
decoding disabled."

## Verdict

Real downstream defect, one line away from correct, in the **prefill half** of
`vllm_ascend/attention/fa3_v1.py:116-137`; the reported line 122 is the *decode*
half and only the blocking H2D sync that reports the previous layer's prefill
launch. Source-verified at v0.23.0rc1; the reported line number matches the pin
exactly, so the pin is not in question.

| line | expression | unit problem |
|---|---|---|
| `fa3_v1.py:119` | `query[:num_decode_tokens]` | correct (tokens) |
| `:120-122` | `block_tables[:num_decodes, :]`, `query_start_loc[:num_decodes+1]`, `seq_lens[:num_decodes]` | correct (requests) — the decode half is the reference |
| `:131` | `query[num_decode_tokens:]` | correct (tokens) |
| `:132` | `block_tables[num_decode_tokens:, :]` | **request-indexed tensor sliced by a token count** |
| `:133` | `query_start_loc[num_decodes:]` | **cu_seqlens kept absolute; must be rebased to 0** |
| `:136` | `max(seq_lens[num_decodes:])` | **KV length passed as `max_seqlen_q`** |

`_flash_attn_with_kvcache` forwards these to `_fa3_fn` as
`cu_seqlens_q=actual_seq_lengths`, `page_table=block_table`,
`cache_seqlens=seq_lens`, `max_seqlen_q=max_seq_len` (`fa3_v1.py:83-92`). The FA3
NPU interface derives `batch = cu_seqlens_q.size(0) - 1` and `total_q =
cu_seqlens_q[-1]` from the cumulative array, so with an absolute array the kernel
believes there are `num_prefills + 1` sequences whose first one starts at
`num_decode_tokens`, writes/reads `q`/`out` at absolute token offsets while the
prefill half of `q` has only `num_prefill_tokens` rows, and reads
`page_table`/`cache_seqlens` rows that a `num_decode_tokens`-offset slice no
longer has — the page table can be sliced to zero rows once
`num_decode_tokens (2 × 3 = 6 with ngram k=2) ≥ num_reqs`. Garbage block ids then
index the KV cache at arbitrary addresses: aicore exception 507015.

Why spec decode is the discriminator: with `num_speculative_tokens = 0` the same
four expressions still overshoot but only by `num_decodes` rows, which usually
lands inside the caching allocator's own block (wrong output, no fault). The e2e
test `tests/e2e/pull_request/one_card/test_attention_fa3.py` exercises mixed
prompts and `max_num_batched_tokens=5` but never enables spec decode, so the
multiplied offset is untested at this pin.

## The fix, in the tree's own units

```python
attn_metadata.block_tables[num_decodes:, :],
attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens,
# max_seqlen_q for the prefill half:
(attn_metadata.query_start_loc[num_decodes + 1:] -
 attn_metadata.query_start_loc[num_decodes:-1]).max(),
```

Sanctioned in-tree references for the same split: `attention_v1.py:1843-1845`
(`prefill_seq_qlen = [actual_seq_qlen[i] - num_decode_tokens for i in
range(num_decodes, ...)]`), `attention_v1.py:1864` (`block_tables[num_decodes:]`),
`gdn_attn_builder.py:652-653` (`non_spec_query_start_loc[num_decodes:] -
num_decode_tokens`). Upstream vllm's `v1/attention/backends/flash_attn.py` has no
equivalent split at v0.23.0 (one varlen call for the whole batch), so there is no
upstream body to diff — the convention lives only downstream.

## Secondary defects to state, not to conflate

- `causal=False` on the decode half (line 123) is wrong whenever a decode request
  carries more than one query token, i.e. under spec decode: the KV written for
  the draft tokens is already in the cache when attention runs
  (`attention_v1.py:1532-1543` calls `reshape_and_cache` before `forward_impl`),
  so a non-causal decode attends to its own draft K/V. Numerics, not a fault;
  label it hypothesis until an output/logprob comparison against a non-spec run
  settles it.
- `max_seqlen_q = max(KV length)` (line 136): overprovisioned q tiling.

## Ladder

1. Drop `attention_config={"backend": "FLASH_ATTENTION"}` (or
   `speculative_config`) on the same script — one run, splits FA3 from spec
   decode. Note that `VLLM_BATCH_INVARIANT=1` is what makes FA3 the required
   backend; if batch invariance is the goal, dropping spec decode is the only
   side that keeps it.
2. `num_speculative_tokens=1` — halves the offset; a crash that disappears or
   moves to a later batch confirms the multiplier.
3. Apply the three-line correction above and rerun the reporter's script.
4. Ask for `ASCEND_LAUNCH_BLOCKING=1` output if the faulting kernel still has to
   be attributed to a layer.

## Scope

Source-verified at vllm-ascend v0.23.0rc1 / vllm v0.23.0 / flash-attention-npu
v0.4.2. Recheck `launch_metadata.num_decodes`/`num_decode_tokens` semantics if the
attention metadata builder changes; invalidated if `fa3_v1.py` is rewritten
against a newer FA3 NPU interface that takes per-request query lengths instead of a
cumulative array.
