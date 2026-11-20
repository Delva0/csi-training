# Mixed decode+prefill batches: split sites and index units

## Scope

A step that contains **both** decode requests and prefill (or extend) requests is
split into two attention calls. vllm-ascend implements that split in several
places (`vllm_ascend/attention/fa3_v1.py`, `attention_v1.py`, `mla_v1.py`,
`ops/gdn_attn_builder.py`, `attention/context_parallel/*`), each hand-written, and
each of them mixes two different counting units in adjacent lines:

- `num_decodes`, `num_prefills` — how many **requests**;
- `num_decode_tokens`, `num_prefill_tokens` — how many **tokens**;
- with speculative decoding `num_decode_tokens = num_decodes * (1 + num_spec)`,
  so the two differ by the spec length per request, and with chunked prefill a
  prefill request contributes many tokens but one row.

Enter this topic when the symptom is a **device-side fault or corruption whose
deepest frame is an attention backend** and the batch was mixed: `RuntimeError:
ACL stream synchronize failed, error code:507015` / `EE9999 …
rtDeviceSynchronizeWithTimeout … aicore exception`, `507035`, `MTE … DDR address
… out of range`, or output that is wrong only for batches where a decode and a
prefill coexisted. The traceback line is often a `.npu()` H2D copy, not the
faulting kernel: the aicore error is recorded asynchronously and surfaces at the
next blocking sync, so **the reported line can belong to the decode half while
the fault is in the prefill launch of the previous layer**.

Not this topic: a pure decode-only or prefill-only crash (read that backend's own
page), **the variant the operator itself refuses** — `EZ9999 … When layout is TND,
queryT(N) must be equal to the last element of actualSequenceLengthQ(M)`, which is
padding bookkeeping in a `FULL_DECODE_ONLY` graph rather than a wrong split
(`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`) —, a device fault raised from the
indexer/`npu_scatter_nd_update_` of a sparse-attention model (`../dsa-sparse-attention/indexer-cache-scatter-fault.md`,
which is the same class on the DSA path), a collective (`../hccl-and-hardware/`),
and slow mixed batches with no fault (`../performance-throughput/`).

## Rules

1. **Audit the split before the kernel.** When the op itself names two numbers that
   must be equal, the units are already suspect *and already localised* — take
   `../spec-decode-mtp/fia-tnd-seqlen-mismatch.md` (one script call) instead of this
   audit, which is for the silent variant. Otherwise, `bash
   <skill_root>/scripts/batch-split-audit.sh <vllm-ascend-root>` lists every site
   that slices a request-indexed tensor by a token count and every
   `query_start_loc`/`actual_seq_lengths` slice passed without rebasing. One call,
   whole tree, no reading. Verified on the #17991 pin (v0.23.0rc1): exactly two
   hits — the real defect at `fa3_v1.py:132-133` and one benign hit at
   `gdn_attn_builder.py:320`, where that file rebinds `num_decode_tokens :=
   num_decodes` (request count) and the script says so.
2. **The three tensors have fixed units; check each one separately.**
   `query`/`attn_output` are token-indexed (`[num_decode_tokens : num_tokens]`);
   `block_tables`/`seq_lens`/`state_indices` are request-indexed
   (`[num_decodes :]`); `query_start_loc`/`actual_seq_lengths` is a *cumulative*
   array with one entry per request plus a leading 0, so its prefill half must be
   `query_start_loc[num_decodes:] - num_decode_tokens` and must start at 0 before
   it is handed to a kernel as `cu_seqlens_q`. A slice that keeps the absolute
   values declares the first prefill sequence at element `num_decode_tokens`
   instead of `0`.
3. **`max_seqlen_q` is a query length, not a KV length.**
   `max(seq_lens[num_decodes:])` is the longest context in the prefill half; the
   query length is `max(query_start_loc[num_decodes+1:] -
   query_start_loc[num_decodes:-1])`. Passing the KV length overprovisions the q
   tile loop — usually tolerated, always wrong.
4. **Spec decode is what turns the mismatch from silent into fatal.** Without
   spec, `num_decode_tokens == num_decodes`, so a wrong slice overshoots by a few
   rows and stays inside the allocator block (wrong ids, no fault). With MTP or
   ngram, the offset multiplies by `1 + num_spec`, the page table can start past
   the end of its own storage, the block ids read become garbage, and the gather
   addresses fault. That is why "the same workload completes with spec decode
   disabled" is *consistent with this defect*, not evidence against it — the
   whole point is the multiplier. It is also why the in-tree e2e test can pass:
   `tests/e2e/pull_request/one_card/test_attention_fa3.py` mixes prompts and
   tightens `max_num_batched_tokens` but never enables speculative decoding.
5. **Find the same file's correct half and copy its units.** The decode branch of
   the very same `forward_impl` is usually right, and `attention_v1.py`
   (`:1843-1845,1864`) and `gdn_attn_builder.py:652-653` show the sanctioned
   pattern. Quote the sibling site in the answer; it makes the fix
   non-negotiable and shows the maintainers' own convention.
6. **Say that a mixed batch is unavoidable in the reporter's workload** when it
   is: with N prompts submitted at once and chunked prefill, the short prompts
   start decoding while the long ones are still prefilling, so any
   `--max-num-batched-tokens` that cannot finish every prompt in one step
   produces the trigger. The workarounds are therefore "drop the backend / drop
   spec decode", not "reorder the workload".

## Pages

- `fa3-mixed-batch-split.md` — worked case vllm-ascend #17991 (v0.23.0rc1 +
  FA3 NPU v0.4.2, Qwen3-0.6B, `attention_config={"backend": "FLASH_ATTN"}` +
  `ngram` spec decode): the four wrong expressions at `fa3_v1.py:131-136`, why
  the traceback names the decode half's `.npu()`, the unit table, the derived
  overshoot, the corrected lines, and the two secondary defects
  (`causal=False` on multi-token decodes, `max_seqlen_q` = KV length).
