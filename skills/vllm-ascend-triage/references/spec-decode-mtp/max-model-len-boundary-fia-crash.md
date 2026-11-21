# The drafter asks for one token past the engine's own cap (#9359)

## Signature

A serving deployment with `--speculative-config` (eagle3 here) kills its EngineCore
at request time with an **operator argument rejection**, not a device fault:

```text
RuntimeError: npu_fused_infer_attention_score_symint:… NPU function error:
  call aclnnFusedInferAttentionScoreV3 failed, error code is 107030
ERR00100 PTA call acl api failed.
EE9999: Inner Error!
EE9999… rtMemcpy execution failed, reason=the current capture mode does not support this operation
   synchronized memcpy failed, kind = 1, runtime result = 107030
   the key/value's actual sequence lengths(0) is 133121, it should be in range [0, KV_S(133120)].[FUNC:CheckFeatureActualSeqLensKvData][FILE:fused_infer_attention_score_tiling_check_feature.cpp][LINE:762]
   FusedInferAttentionScore do tiling failed, ret is -1.
   Check NnopbaseExecutorDoTiling(executor) failed
```

The stack is the **drafter**, not the target model: `sample_tokens` →
`propose_draft_token_ids` → `drafter._propose` → `acl_graph.__call__` →
`_run_merged_draft` → `self.model(**model_kwargs)` → `unified_attention_with_output`
→ `attention_v1.py forward_fused_infer_attention` → the op. `EngineCore` then reports
`Worker failed with error '<the same text>'` — that second traceback is the worker
having died (index rule "three tracebacks are one provenance").

## Read the numbers first; the source walk rarely changes the ranking

The reporter's own `dump_input.py` block decides most of the case:

| value | #9359 | meaning |
|---|---|---|
| the engine's effective sequence cap (`max_seq_len=133120` there) | 133120 | `KV_S` in the op message |
| `num_computed_tokens` | 133117 | tokens already resident |
| `num_scheduled_tokens` | 2 | tokens this step: 133117 + 2 = 133119, one bonus + one draft row from the cap |
| `scheduled_spec_decode_tokens={req: [-1]}` | present | the step *did* schedule spec rows; `-1` is the placeholder that drives the drafter past the cap |
| `num_output_tokens` | 120298, `num_running_reqs=1` | one very long generation — this shape needs a request that reaches the cap, so it never reproduces on short prompts |

`N - KV_S == 1`: the value is the cap plus exactly one token, i.e. an off-by-one at
the boundary, not a runaway length and not a corrupted block table. `KV_S` reads as a
KV-side **capacity** — the dumped cap is 1040 blocks × 128, which is exactly 133120 —
so this is a `value vs capacity` comparison, not the "two host tensors must be equal"
test of `fia-tnd-seqlen-mismatch.md` (`queryT` vs `last(actualSequenceLengthQ)`). Route
by the string: `actual sequence lengths(K)` / `KV_S` is this page. Label the capacity
reading an inference — the emitting checker
(`fused_infer_attention_score_tiling_check_feature.cpp`) is in `cann/ops-transformer`,
which no export carries.

Nothing was launched (`do tiling failed, ret is -1`), so no plog timeline, no
`npu-smi`, no collective chain applies.

## Two launcher disagreements to surface before ranking a mechanism

The dumped config is the launch that ran; the pasted script frequently is not.

- **`num_spec_tokens`.** #9359's pasted `--speculative-config` says
  `num_speculative_tokens: 1`, its dump says `SpeculativeConfig(… num_spec_tokens=3)`.
  K is the size of the overshoot, so every ladder row written from the paste would
  test the wrong launch. Ask for the engine's own `args: Namespace(...)` line or the
  `--speculative-config` from `/proc/<pid>/cmdline`.
- **The cap.** The decode node's script says `--max-model-len 172032` while the dump's
  effective cap is 133120, and at the pinned vllm v0.19.1 a `max_model_len` larger than
  the value derived from the checkpoint raises `ValueError … User-specified
  max_model_len (N) is greater than the derived max_model_len`
  (`vllm/config/model.py:2155-2181`) unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. So either
  the length in the script was never passed on this node, or the env carried that flag —
  ask, do not assume. Field spelling is also a fingerprint: the pinned `ModelConfig` has
  `max_model_len` (`vllm/config/model.py:182`), not `max_seq_len`, so a dump line
  spelling `max_seq_len` is worth confirming against the installed wheel before quoting
  it as the pin (#9359 also labels itself `v0.19.0rc1` in the body and `0.19.1rc1` in
  the env dump).
- Related but not the crash: a P/D pair whose prefiller runs `--max-model-len 131072`
  while the decoder's cap is 133120 disagrees about the envelope. Align them, and do not
  present the disagreement as the cause.

## Ranked candidates, one check each

1. **The merged-draft guard is narrower downstream than upstream (source-verified
   difference; hypothesis for this run).** Upstream vllm v0.19.1 masks with a
   *lookahead*: `exceeds_max_model_len = (positions + total_num_drafts) >=
   self.max_model_len`, then `attn_metadata.seq_lens.masked_fill_(…, 1)`
   (`vllm/v1/spec_decode/eagle.py:984,1033-1038`). vllm-ascend's per-step merged-draft
   loop increments `used_update_positions += 1` and then compares only that step's
   position, `exceeds_max_model_len = used_update_positions >= self.max_model_len`
   (`vllm_ascend/spec_decode/eagle_proposer.py:1314,1335-1359`), masking the tensors
   afterwards. A guard that only looks at the current step cannot cover the step whose
   length becomes `cap + 1`. Check: `num_speculative_tokens` 3 → 1, same request —
   collapse with K=1 and crash with K=3 confirms the lookahead width.
   (`num_spec_tokens=1` *is* the cheapest row regardless, so run it even before the
   comparison matters.)
2. **The value that reaches the op is a host list, materialised from a different
   container than the masked tensor (mechanism open, one print settles it).** FIA gets
   `actual_seq_lengths_kv = attn_metadata.seq_lens_list` (`attention_v1.py:805,815,826`,
   DecodeOnly branch), and that list is `seq_lens.tolist()` built in
   `AscendAttentionMetadataBuilder.build` from `_seq_lens_cpu` (preferred), else
   `seq_lens_cpu`, else `seq_lens.to("cpu")` (`attention_v1.py:289-296,320`). The drafter
   masks `common_attn_metadata.seq_lens` / `seq_lens_cpu` / `_seq_lens_cpu`
   (`eagle_proposer.py:1352-1361`). Whether every draft step's metadata is rebuilt *after*
   that masking is the open question: one `logger.info` of `attn_metadata.seq_lens` next
   to `attn_metadata.seq_lens_list` at the failing step settles it, and so does the
   `--speculative-config`-off row if the cap is reached without a crash.
3. **`--async-scheduling` optimistic seq lens (hypothesis, one ablation).** Under async
   spec decode the runner keeps `optimistic_seq_lens_cpu` — "assumes all draft tokens
   from previous iteration accepted" (`model_runner_v1.py:803-811`) — and the code itself
   notes it is stale and must be synced back from the corrected GPU values
   (`:951-970`), while the Ascend metadata build prefers `_seq_lens_cpu`
   (`attention_v1.py:289-296`). One step of lag between the mask and the list makes
   exactly this off-by-one. Check: same launch with `--async-scheduling` off.

Do **not** rank the `rtMemcpy … the current capture mode does not support this
operation` line as the cause. It is the same record's tail: the op was reached through
`acl_graph.__call__` replay, the tiling refusal is what the caller got
(`do tiling failed, ret is -1`), and the `EE9999` line even carries an older wall-clock
date than the `ERROR` frame — apply rule 13 before believing it is this incident. That
line alone routes to `../startup-failures/acl-graph-capture.md` /
`draft-graph-capture-device-fault.md`; the `KV_S` line is the more specific row and
wins.

## Ladder (cheapest first)

1. `num_speculative_tokens: 3 → 1` **in the launch that actually runs**, then delete
   `--speculative-config` (one row, splits drafter from everything else). The
   MTP-off row is index rule 1 of this topic.
2. `--async-scheduling` off (candidate 3, one launch).
3. `"cudagraph_mode": "PIECEWISE"`, then `--enforce-eager`: removes the merged-draft
   graph replay the stack passes through, at a known throughput cost.
4. Align P and D `--max-model-len` and keep `num_spec_tokens + 1` tokens of headroom
   under the checkpoint's own cap if a stop-gap is needed — say plainly that this
   **moves the wall**, it does not fix the guard: the scheduler legitimately schedules
   the step at `cap - 2` and the drafter overshoots it.
5. A patch that adds the lookahead to the merged-draft guard is the fix *shape*, not
   something to hand a reporter as a workaround; state which of the three candidates
   the K=1 / async-off rows eliminated before proposing it.

## Scope and gaps

Verified at vllm-ascend `v0.19.1rc1` (`da421af`) + vllm `v0.19.1` (`b1388b1f`); line
numbers move between releases, the shape does not. Not verifiable in any export: the
tiling checker that emits the message lives in `cann/ops-transformer`, which
`SOURCES.json` lists under `unavailable` — one grep attempt, then report it as the gap
instead of widening the pattern. Not settled here: which container the 133121 came from
(candidate 1 vs 3), and the reporter's real `num_spec_tokens` / `max_model_len`.
Ask back for: the engine's `args: Namespace(...)` (or `/proc/<pid>/cmdline`), the
engine-start line that prints the final max sequence length, the full `dump_input.py`
block for the *failing* DP rank, and whether a request that never reaches the cap is
clean.
