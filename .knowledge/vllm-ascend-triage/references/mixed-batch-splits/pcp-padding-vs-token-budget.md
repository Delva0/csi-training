# PCP padding vs the token budget: `index value 8194 exceeds bounds 8192` on the P node

Scope: the **prefill** node of a PD deployment dies at request time under real
concurrency (small concurrency is clean) with a *device-side* index assertion and no
Python frame at all:

```
Index out of range in dimension 0: index value 8194 exceeds bounds 8192!
[AIV Block 1/2] [ASSERT] index_check.h:130: Assertion `(idx < bound)' failed.
```

followed by the cascade the reporter will quote next — vector-core exception →
`event synchronize` failure → `WorkerProc hit an exception` → EngineCore 500. The bound
is the engine's own token budget (`max_num_batched_tokens` / its `max_num_scheduled_tokens`
reduction) and the offending index is one past the **PCP-expanded** token count. This is
host-side budget arithmetic surfacing inside a kernel: no HCCL, no card, no checkpoint,
and the cascade is one provenance, three symptoms. Verified against vllm-ascend
**v0.26.0rc1** (`f2f74a16`) with vllm **v0.26.0** (`568afb3a`), i.e. the pin #18518 was
filed on; the numbers below are read from that tree, not executed.

## First: which tree owns the padding (the path decides the page)

| ref | padding site | form |
|---|---|---|
| ≤ v0.23.x | `vllm_ascend/worker/pcp_utils.py` `update_tokens_for_pcp` (`:660`/`:759` at v0.21.0rc1, `:811`- era prints in #18518) | per request, `ceil(n / (2*pcp)) * (2*pcp)`, then `// pcp` per rank |
| ≥ v0.26 | **gone** — PCP is upstreamed: `vllm/v1/worker/gpu/pcp_manager.py`, with `AscendPCPManager(PCPManager)` in `vllm_ascend/worker/v2/pcp_manager.py:62-79` | `padded_num_tokens = max(per_rank_num_tokens)`, `num_expanded_tokens = padded * pcp` (`:282-283`) |

A paste that names `vllm_ascend/worker/pcp_utils.py:NNN` while `collect_env` prints a
v0.26 version is therefore a **correspondence** question before it is an arithmetic one:
one existence check settles it (below), and the history corpus already records the
removal (`#13973`'s own `git diff --stat` table: `worker/pcp_utils.py | −2387 (deleted)`,
`worker/v2/pcp_manager.py | +107 (new)`; `#14320`: "the v0.26 PCP-removal refactor
(#12592 'remove PCP from MRV1')"). See
`../startup-failures/environment-correspondence.md` §*A `vllm_ascend/**` frame the pin
does not contain*.

## The arithmetic that reproduces the reported index at the pin

`_build_batch_layout` splits every request's scheduled tokens across the PCP ranks with
the remainder left on rank 0, then pads to the **largest rank** and expands back:

- reporter's step: `num_scheduled_tokens = [6661, 1531]` (the scheduler packed exactly
  `max_num_batched_tokens = 8192`);
- rank split: `6661 → 3331/3330`, `1531 → 766/765` → ranks hold `4097` and `4095`;
- `padded_num_tokens = max(...) = 4097`, `num_expanded_tokens = 4097 * 2 = 8194` — the
  reported index, exactly, with **no** per-request padding needed;
- the only guard compares the *per-rank* value with the input buffer:
  `if num_local_tokens_padded > input_buffers.max_num_tokens: raise RuntimeError(...)`
  (`vllm/v1/worker/gpu/pcp_manager.py:387-408`), and `max_num_tokens` is
  `scheduler_config.max_num_batched_tokens`
  (`vllm_ascend/worker/v2/pcp_manager.py:102`). `4097 <= 8192` passes while the expanded
  tensor is `8194`.

So the two refs produce the same overflow from different producers: per-request padding
before v0.26, rank-imbalance padding at v0.26. Answer "the padding formula changed, the
8194 quantity did not" — never "v0.26 fixed it" on the strength of the file move alone.

## Procedure

1. **Answer first**, in the reporter's language: real defect, host-side token budget that
   never accounts for the PCP expansion; the `index_check` line is the symptom and the
   vector-core/`event synchronize`/worker sequence is its cascade.
2. **One existence call** when the paste names `pcp_utils.py` and a ≥ v0.26 version:
   `python -c "import vllm_ascend,os;d=os.path.dirname(vllm_ascend.__file__);print(d, os.path.exists(os.path.join(d,'worker/pcp_utils.py')))"`
   plus `pip show vllm vllm-ascend`. Old path present = an older tree than the dump says;
   absent = read the upstream manager before advising a patch site.
3. **Reporter's budget proposal, answered from the source**: `_set_max_num_scheduled_tokens`
   *is* the right hook — the scheduler really uses `max_num_scheduled_tokens` as its token
   budget (`vllm/v1/core/sched/scheduler.py:111-113,445`, asserted at `:1066`) — but its
   only subtraction is `max_num_new_slots_for_drafting * max_num_seqs`
   (`vllm/config/vllm.py:1689-1711`), and that property is `0` for parallel drafting with
   `num_speculative_tokens = 1` and no draft model (`vllm/config/speculative.py:1293-1307`).
   It is inert for MTP, which is why the log shows the full budget. Say the owner is
   upstream vllm and, while PCP has no upstream budget term, the patch lands through
   `vllm_ascend/patch/`.
4. **Confirm with the tree's own debug line** rather than a hand print:
   `VLLM_LOGGING_LEVEL=DEBUG` prints `PCP batch: … per_rank_tokens=[…]`
   (`vllm/v1/worker/gpu/pcp_manager.py:395-398`); `max(per_rank_tokens) * pcp >
   max_num_batched_tokens` is the reproduction.
5. **Mitigation before a fix**: raise `--max-num-batched-tokens` (the bound moves away
   from the packed total) or cap concurrency so the packed total cannot reach it. Both are
   rungs, not repairs, and both cost throughput — say which.
6. Do not attribute it to the model, the attention backend, EP, or MTP itself: MTP only
   decides how many tokens a request contributes to the packed total, and the same
   arithmetic reproduces on a dense prefill.

## Recorded case

#18518 (Ascend 910B2 × 8, vllm 0.26.0 / vllm-ascend 0.26.0rc1, PD with
`PCP_SIZE=2 TP=8 EP` + `deepseek_mtp` `num_speculative_tokens: 1`, 64 K prompts, chunked
prefill, `--max-num-batched-tokens 8192`): 1/2/4 concurrency clean, 8 concurrent from the
first step, 63/64 failing at 16. The reporter's own fix (reserve
`(2*pcp-1) * max_num_seqs` in the budget) measured 64/64 with a peak expanded count of
8008 — their numbers, not verified here; the hybrid-path half of their proposal
(`pcp_utils.py:958`) has no counterpart at the pin, which has a single
`_get_rank_segments` split.
