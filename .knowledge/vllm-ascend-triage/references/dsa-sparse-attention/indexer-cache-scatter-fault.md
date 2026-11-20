# The indexer-cache scatter faults the device: an index-shape bug, not a bad card

Worked case: #4296, DeepSeek-V3.2-Exp-w8a8, vLLM 0.11.0 + vllm-ascend **v0.11.0rc1**
(`c5fe179`), 16×910B2C 64G, `--tensor-parallel-size 16 --max-num-seqs 16
--max-model-len 13000 --max-num-batched-tokens 13000 --gpu-memory-utilization 0.8
--enforce-eager`, `ASCEND_LAUNCH_BLOCKING=1`; crash during 4-concurrent 2048-in /
2048-out load after several successful requests.

## The signature (what makes this one call identifiable)

- Python frame chain, all project code at the bottom:
  `worker/worker_v1.py:276` → `worker/model_runner_v1.py:1947,1550` →
  upstream `models/deepseek_v2.py` → `vllm_ascend/models/deepseek_v3_2.py:387`
  (`self.sfa_attn(...)`) → `models/layers/sfa.py:143,162` →
  `attention/sfa_v1.py:826` (`_sfa_preprocess`) → `sfa_v1.py:690`
  (`indexer_select`) → `sfa_v1.py:969` `torch_npu.npu_scatter_nd_update_`.
- `RuntimeError: RunOpApiV2 … AclrtSynchronizeStreamWithTimeout … error code is
  507035`, `ERR00100 PTA call acl api failed`, `EZ9999 … exception of aivec error`,
  `errorStr: The DDR address of the MTE instruction is out of range`,
  `Kernel task happen error, retCode=0x31, [vector core exception]`, and
  `fault kernel_name=ScatterNdUpdate_…_high_precision_…, program id=310`.
- **The same `pc start` on dozens of cores of the one reporting chip.** That is one
  software address computation, not several failing tiles (`../hccl-and-hardware/error-report-timestamps.md`
  makes the same point for the plog). Nothing in the paste names a collective, so
  lane C collective routes (`ASCEND_LAUNCH_BLOCKING=1`, device health) are the wrong
  next step: they were already in effect.

## Mechanism (verified in the pinned tree, `c5fe179`)

`_sfa_preprocess` splits the merged batch into decode and prefill halves and calls
the indexer once per half, passing the phase's rows only:

| Line | Fact |
|---|---|
| `sfa_v1.py:300-303`, `:425` | `slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]` is stored on the merged `attn_metadata` — **full batch rows** |
| `sfa_v1.py:638` / `:723,727` | the author already computes `decode_slot_mapping` (`[:num_decode_tokens]`) and `prefill_slot_mapping` (`[num_decode_tokens:num_actual_tokens]`) |
| `sfa_v1.py:672` / `:770` | `npu_kv_rmsnorm_rope_cache(..., decode_slot_mapping / prefill_slot_mapping, ...)` — the **latent** KV cache is written with the sliced mapping, which is why it never faults |
| `sfa_v1.py:690` / `:780` | `indexer_select(hidden_states_decode …)` / `indexer_select(x=hidden_states_prefill …)`: `x`, and therefore `k` (`:956-965`), holds only that phase's rows |
| `sfa_v1.py:969-973` | `npu_scatter_nd_update_(kv_cache[2].view(-1, k.shape[-1]), attn_metadata.slot_mapping.view(-1, 1), k.view(-1, k.shape[-1]))` — **the unsliced full-batch mapping** |

In a mixed batch the index tensor has `num_decode_tokens + num_prefill_tokens` rows
while the source `k` has one phase's rows, so the kernel reads past the source
allocation (MTE DDR out of range) — and when no crash occurs, the indexer K of that
phase is written into *other* tokens' slots, so the following
`npu_lightning_indexer` (`sfa_v1.py:977`, `self.index_topk = 2048` at `:524`) picks a
wrong top-k and accuracy degrades silently.

The alternative mechanism for the same dump is the *value* range rather than the row
count: a slot index beyond `kv_cache[2]`'s `num_blocks × block_size` rows would also
raise an MTE DDR error. It is separable in one print — `slot_mapping.max()` against
`kv_cache[2].shape[0] * kv_cache[2].shape[1]` — and the row-count mismatch above is the
one this tree actually contains.

Consequences for ranking, all cheap to state:
- a pure-decode batch (`num_prefill_tokens == 0`) and a pure-prefill batch
  (`num_decode_tokens == 0`) are **safe**; only the mixed batch is not — exactly the
  shape of "many 200 OKs, then one crash".
- `--enforce-eager`, `--no-enable-prefix-caching`, w8a8 quantisation, CANN 8.3.RC1 and
  the 910B2C hardware are all unrelated; say so, because the paste invites a
  hardware answer.

## Confirmation and fix to hand back

1. Cheapest confirmation, no code: re-run the same 2048/2048 load with
   `--max-num-seqs 1` (one request in flight, so the scheduler cannot build a mixed
   batch) — no crash — and with `--max-num-seqs 4` — crash. That single A/B is the
   whole test.
2. Log evidence for the bug report: print `attn_metadata.num_decode_tokens`,
   `num_prefill_tokens`, `attn_metadata.slot_mapping.numel()` and `k.shape[0]` before
   `sfa_v1.py:969`; the faulting step ends with both phase counts non-zero and the
   row counts unequal.
3. Fix: pass the phase mapping into `indexer_select` (both call sites already hold
   `decode_slot_mapping` / `prefill_slot_mapping`) and use it at `:969` instead of
   `attn_metadata.slot_mapping`.
4. The plog adds only the kernel name and `task_id`, which the paste already carries.

## Scope and recheck conditions

- Claim class: **source-verified mismatch plus a matching symptom**; the reporter
  never posted the A/B result, so the fix is not reporter-confirmed. Say that.
- Verified only at `v0.11.0rc1` (`c5fe179`); the snapshot carries one commit per
  release, so "which release fixed it" is unknown. The recheck is one grep for
  whether the second argument of that `npu_scatter_nd_update_` is still the merged
  `attn_metadata.slot_mapping`.
- **The file was refactored by the next releases, so re-derive, do not transplant.**
  At `v0.13.0rc2` (`a17059a`, #6240's pinned tree) it is
  `indexer_select_pre_process` (`:920`) / `indexer_select_post_process` (`:964`), the
  whole batch goes through one path, and the scatter at `:996-997` still passes
  `attn_metadata.slot_mapping` — so the v0.11 mixed-batch mismatch no longer has the
  same shape there, but the same asymmetry survives in a different place: under
  `enable_dsa_cp` the latent write uses `dsa_cp_context.slot_mapping_cp`
  (`:804-812`) while the indexer scatter keeps `attn_metadata.slot_mapping`
  (`:996`). Whenever an indexer write faults or corrupts, compare *those two*
  arguments at the pinned commit.
- Second implementation to flag, not to assert: the graph path
  `vllm_ascend/torchair/torchair_sfa.py:1262-1263` makes the identical call
  (`prefill_slot_mapping = attn_metadata.slot_mapping` at `:995`,
  `decode_slot_mapping = attn_metadata.slot_mapping` at `:1120`). Whether the
  torchair metadata is already phase-split at that point was not verified here;
  list it as an open question if the reporter switches to graph mode.

## Same class outside DSA

The unit rule behind this page — a request-indexed tensor sliced by a token count,
or a cumulative `cu_seqlens` array handed over without rebasing, turning into a
device fault once speculative decoding multiplies the offset — is general to every
hand-written decode/prefill split in the tree, and it was re-recorded on the FA3
backend (#17991: `attention/fa3_v1.py:132-136`, `507015`). When the batch was
mixed and the frame is an attention backend rather than `sfa_v1.py`, start at
`../mixed-batch-splits/index.md`, whose `scripts/batch-split-audit.sh` lists every
such site in one call. What does not transfer: the DSA path's write/read
asymmetry (`slot_mapping` vs `slot_mapping_cp`) is specific to `sfa_v1.py`.
