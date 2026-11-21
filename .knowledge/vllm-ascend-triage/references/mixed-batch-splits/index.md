# Mixed decode+prefill batches: split sites and index units

## Scope

A step that contains **both** decode requests and prefill (or extend) requests is
split into two attention calls. vllm-ascend implements that split in several
places (`vllm_ascend/attention/fa3_v1.py`, `attention_v1.py`, `mla_v1.py`,
`ops/gdn_attn_builder.py`, `attention/context_parallel/*`) and builds the same
split a third time as **batch metadata** in `vllm_ascend/worker/pcp_utils.py`
(`PCPManager.init_batch_info` / `update_tokens_for_pcp`, reached from
`_prepare_inputs`). Each site is hand-written, and each mixes two different
counting units in adjacent lines:

- `num_decodes`, `num_prefills` — how many **requests**;
- `num_decode_tokens`, `num_prefill_tokens` — how many **tokens**;
- with speculative decoding `num_decode_tokens = num_decodes * (1 + num_spec)`,
  so the two differ by the spec length per request, and with chunked prefill a
  prefill request contributes many tokens but one row.

The same two units also meet on a **pure decode** step when the runner padded the
token count up to an ACL-graph bucket — no mixed batch is involved there, only the pad
(third bullet).

Four entry shapes, same class:

- **Device side**: a fault or corruption whose **deepest frame is an attention
  backend** while the batch was mixed: `RuntimeError: ACL stream synchronize
  failed, error code:507015` / `EE9999 … rtDeviceSynchronizeWithTimeout … aicore
  exception`, `507035`, `MTE … DDR address … out of range`, or output that is
  wrong only for batches where a decode and a prefill coexisted. The traceback
  line is often a `.npu()` H2D copy, not the faulting kernel: the aicore error is
  recorded asynchronously and surfaces at the next blocking sync, so **the
  reported line can belong to the decode half while the fault is in the prefill
  launch of the previous layer**.
- **Host side, no device frame at all**: `ValueError: could not broadcast input
  array from shape (a,) into shape (b,)` (or an `np.repeat` length error) whose
  deepest frame is `vllm_ascend/worker/pcp_utils.py`, reached from
  `model_runner_v1._prepare_inputs`, with spec decode on and PCP > 1. numpy is
  reporting the unit confusion for you — two slices of the per-request
  `num_scheduled_tokens` array, one offset by a request count and one by a token
  count. Do **not** route this to the PCP enablement lane
  (`../feature-enable-recipes/index.md`) as a card/flag problem, and do not ask
  for an MTP-only ablation: the offset only exists because MTP is on. Its `TypeError`
  sibling — `'NoneType' object does not support item assignment` at
  `eagle_proposer.py set_inputs_first_pass`, `cad.seq_lens_cpu[-num_prefill_reqs:]`,
  because async spec-decode mode nulls that CPU field while `_seq_lens_cpu` carries it
  — is `../spec-decode-mtp/pcp-async-spec-seq-lens-cpu-none.md`.
- **Host side on a decode-only step, no mixed batch at all**: `RuntimeError: shape
  '[N, 1, H*D]' is invalid for input of size M` raised at `attention_v1.py`
  `_forward_decode_only`, on a serve that had been answering for a long time. `N` is
  `attn_metadata.seq_lens.shape[0]` (requests) and `M/(H*D)` is the **padded token**
  count, so the second unit is the ACL-graph bucket and not a schedule — one division
  classifies it, five concurrent requests confirm it, and no device check applies
  (`decode-pad-request-vs-token-count.md`).
- **PCP padding against the step's own token budget**: neither a mis-sliced unit nor a
  kernel refusal on a mixed batch — the padding the PCP split computes exceeds the
  step's token budget, and the device reports `Index out of range in dimension 0:
  index value N exceeds bounds M` with `M` the scheduler's token budget
  (`pcp-padding-vs-token-budget.md`, recorded #18518). The path moved between refs:
  ≤ v0.23.x this metadata is `vllm_ascend/worker/pcp_utils.py`, while at ≥ v0.26 that
  file is gone and the split is upstream (`vllm/v1/worker/gpu/pcp_manager.py`,
  subclassed by `vllm_ascend/worker/v2/pcp_manager.py`) — an absent `pcp_utils.py`
  under a v0.26 label is a correspondence finding, not a vendored out-of-tree file.

A **Python `UnboundLocalError` in `mla_v1.py`'s forward** (`decode_hs_or_q_c`, then
`decode_k_pe` after a hand patch) under `torchair_graph_config` is *not* this topic even
though a mixed step is what reaches it: no unit is mis-sliced, a branch guarded by
`not self.torchair_graph_enabled` was simply never taken, and the answer is the
torchair × chunked-prefill config contract —
`../feature-enable-recipes/torchair-mla-graph-and-chunked-prefill.md`.

Not this topic: **a step that mixes a spec-decode request with a plain decode request** —
same word "mixed", different question: no unit is mis-sliced, an upstream hybrid
linear-attention builder refuses the composition and the answer is its version window
(`../spec-decode-mtp/mixed-spec-and-plain-decode-assert.md`); a pure prefill-only crash, or a
decode-only crash with no padding in it (read that backend's own page — the
**padded** decode-only variant is the third entry shape above); **the variant the operator itself
refuses** — `EZ9999 … When layout is TND,
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
   audit, which is for the silent variant. When the ValueError already prints both
   slice lengths, decode them instead of auditing (`pcp-schedule-index-units.md`
   §1: the difference is `num_decode_reqs × num_spec`, the target length gives
   `num_reqs`). Otherwise, `bash
   <skill_root>/scripts/batch-split-audit.sh <vllm-ascend-root>` lists every site
   that slices a request-indexed tensor by a token count and every
   `query_start_loc`/`actual_seq_lengths` slice passed without rebasing. One call,
   whole tree, no reading. Verified on the #17991 pin (v0.23.0rc1): exactly two
   hits — the real defect at `fa3_v1.py:132-133` and one benign hit at
   `gdn_attn_builder.py:320`, where that file rebinds `num_decode_tokens :=
   num_decodes` (request count) and the script says so. Verified on the #11231 pin
   (v0.21.0rc1): three unit mismatches — `fa3_v1.py:132`, plus the two PCP
   schedule sites `pcp_utils.py:660` and `:759`, which is why
   `num_scheduled_tokens` is in the script's request-indexed pattern.
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
   Corollary the reporter will test first: with spec decode off the same workload
   completes (for the PCP metadata variant because `num_decode_tokens ==
   num_decode_reqs` makes the two lengths equal), so a clean MTP-only or PCP-only
   baseline is a *prediction* of this defect, not a refutation — say that in the
   answer and do not spend a launch re-confirming it.
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
- `decode-pad-request-vs-token-count.md` — worked case vllm-ascend #2822 (vllm
  `[NETWORK_ADDRESS_1]` + vllm-ascend `0.10.1rc1`, Phi-4-mini-instruct TP1, `lm_eval` gsm8k):
  `shape '[255, 1, 3072]' is invalid for input of size 786432` at
  `_forward_decode_only`'s `self.sliding_window is not None` branch, the
  `M/(H*D)`-vs-`seq_lens` division that names the graph pad
  (`pad_for_cudagraph`, 255 → 256), the bucket list upstream derives and vllm-ascend
  trims, the sibling paged branch that raises nowhere, the call-site guard a later pin
  carries (`seq_lens.shape[0] == query.size(0)`), and the five-concurrent-request
  confirmation.
- `pcp-schedule-index-units.md` — worked case vllm-ascend #11231 (v0.21.0rc1
  `80610e4438`, Qwen3.5-27B, TP2/PCP2, MTP `num_speculative_tokens: 2`):
  `ValueError: could not broadcast input array from shape (29,) into shape (31,)`
  at `pcp_utils.py:667` from `:660`, the broadcast-length arithmetic that names
  the batch (`num_decode_reqs=1`, `num_reqs=32`), the three gates that make the
  combined launch the only failing one, the second site at `:759` that fails
  silently, the same function's correct sibling `:684`, and the fix as it already
  reads in the v0.23.0 pin (`:821`, `:845`, `:938`).
- `pcp-padding-vs-token-budget.md` — worked case vllm-ascend #18518 (v0.26.0rc1 +
  vllm v0.26.0, 910B2 × 8, PD with PCP=2/TP=8/EP, DeepSeek-V3 + MTP K=1):
  `index value 8194 exceeds bounds 8192` at `index_check.h:130` with no Python frame.
  The two padding owners and their forms (per-request `ceil(n/(2*pcp))*(2*pcp)` in
  `pcp_utils.py` before v0.26; `max(per_rank) * pcp` upstream after it), the rank
  arithmetic that yields 8194 from `[6661, 1531]`, the guard that only checks the
  per-rank value, why `_set_max_num_scheduled_tokens` is the right hook yet inert
  (`max_num_new_slots_for_drafting == 0` for MTP K=1), the DEBUG line that confirms it,
  and the two mitigations.
