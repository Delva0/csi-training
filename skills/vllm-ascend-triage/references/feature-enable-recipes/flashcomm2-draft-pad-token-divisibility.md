# FlashComm2 × eagle/MTP multi-step draft: `AssertionError: Batch_size(N) must be divisible by chunk_num(TP)`

Scope: the engine loads and profiles, then a worker dies **at request time** with

```
File "/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear_op.py", line 319, in otp_maybe_quant_comm
    assert batch_size % chunk_num == 0, f"Batch_size({batch_size}) must be divisible by chunk_num({chunk_num})"
AssertionError: Batch_size(4) must be divisible by chunk_num(8)
```

whose stack passes `spec_decode/eagle_proposer.py _propose → _run_merged_draft →
self.model(**model_kwargs)` → upstream `llama_eagle3.py` → `llama.py o_proj` →
`vllm_ascend/ops/linear.py forward` (`custom_op.apply`) → `ops/linear_op.py apply_impl`,
on a launch that exports `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` > 0 (usually together with
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`) and carries `--speculative-config` with
`"method": "eagle3"` and `num_speculative_tokens` ≥ 2. Verified against vllm-ascend
**v0.17.0rc1** (`e20f0b1a`); #7809's reporter ran `releases-v0.18.0-a3`, whose paste puts
the `assert` at `:319` and the `apply_impl` call at `:350` against the pin's `:323` /
`:354` (≈4 lines above the pin's numbering), while their `_run_merged_draft` frame at
`:890` is a different code shape from the pin's `:767` — re-grep and compare function
names, never line numbers. Not the load-time sibling (`layer_idx must be consecutive`) and not a
checkpoint or HCCL question: nothing failed to load and no collective was entered.

## The two numbers are TP and rows, so the config value is never the answer

- `chunk_num = len(self.reorgnized_batch_ids) * len(self.reorgnized_batch_ids[0])`
  (`ops/linear_op.py:320`) over the table built by
  `get_flashcomm2_reorgnized_batch_ids` (`utils.py:1042-1058`): `num_oproj_tensor_parallel_groups
  = global_tp_size // flashcomm2_otp_size` rows of `flashcomm2_otp_size` entries, so
  `chunk_num = (tp // otp) * otp == global_tp_size` for **every** legal value
  (the validator at `utils.py:1018-1028` accepts only `0 < otp < tp` with `tp % otp == 0`).
  Say that before the reporter starts tuning: `1 → 2/4` cannot move the divisor.
- The `Batch_size(N)` in the message is the o_proj input's **row count after the pad**,
  not the request count: `apply_impl` pads with `_EXTRA_CTX.pad_size` immediately before
  the assert (`ops/linear_op.py:317-323`). Read the pair as "this many rows, must be a
  multiple of tp", and take the request count from the log, not from `N`.

## Why the pad is the only normalizer, and why the draft path misses it

- `pad_size` is written **only** inside `set_ascend_forward_context`
  (`ascend_forward_context.py:109-112`): it is reset to `0`, then set to
  `(tp_world_size - num_tokens % tp_world_size) % tp_world_size` when
  `flash_comm_v1_enabled or flashcomm_v2_enabled`. Everywhere else the pad is a passive
  read of that one field.
- FlashComm2 has **no draft-model gate**: `_get_row_parallel_op` returns
  `Flashcomm2OProjRowParallelOp(layer)` for any `o_proj` / `out_proj` prefix whenever
  `flashcomm2_enable()` (`ops/linear_op.py:685-688`), so the drafter's attention output
  linear takes the all-to-all too. FlashComm1 *is* explicitly disabled for the drafter
  (`elif is_draft_model: flash_comm_v1_enabled = False`, `ascend_forward_context.py:97-101`,
  two lines above the `TODO(Levi-JQ)` about normalising the sp/fc2 asymmetry) — so
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` gives the drafter nothing while FC2 silently applies.
- Eagle's multi-step loop rewrites the token count **without** recomputing the pad:
  `_EXTRA_CTX.num_tokens = input_batch_size` / `num_accept_tokens = batch_size`
  (`eagle_proposer.py:870-874`), and steps `1 … K-1` then run the model on
  `input_batch_size` rows — `num_input_tokens` only for `method: mtp` or full-graph
  capture, otherwise `batch_size = common_attn_metadata.batch_size()`, i.e. the request
  count (`:870`). The pad in force is the one the first pass computed. Reporter-confirmed root cause for #7809
  (fix PR #7810): *"Eagle only executes set_forward_context (where the pad parameter is
  set) during the first step."*
- Therefore divisibility holds only by coincidence: the first pass pads
  `num_input_tokens = n_req * (K + 1)` while the loop runs on `n_req` rows, so
  `(n_req + pad(n_req*(K+1))) % tp == 0` is satisfied for some request counts and not
  others, and at the reporter's ref the later steps get no pad at all. Do not build the
  answer on which counts happen to pass — that is the version-specific part.

## Procedure

1. **Answer first**: real defect in the FC2 × multi-step-draft combination, at a
   request-time o_proj all-to-all; nothing to fix in the launcher, the checkpoint or the
   `FLASHCOMM2_PARALLEL_SIZE` value.
2. **Rung 1 — drop FC2, keep FC1** (`VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0`,
   `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` unchanged). Expect it to serve. Label it a
   workaround, not a repair — the combination is uncovered (below).
3. **Rung 2 — the discriminator that keeps FC2 on**: `"num_speculative_tokens": 1`. The
   proposer returns before entering the loop (`if self.num_speculative_tokens == 1 or
   self.parallel_drafting: return draft_token_ids.view(...)`,
   `eagle_proposer.py:846-847`), so there is no second pass to lose the pad. Serving at
   K=1 and dying at K=2 is the mechanism confirmation, and it is also the only
   FC2-preserving shape this tree supports.
4. **Ask for the one print, not another source walk**: at the assert site,
   `print("pad", _EXTRA_CTX.pad_size, "rows", x.size(0), "chunk", chunk_num,
   "draft", _EXTRA_CTX.is_draft_model)`. `pad 0 draft True` = the later step never got a
   pad (the reporter's ref); `pad > 0` = the pad belongs to the first pass's token count.
5. Do not offer "raise `--max-num-seqs`" as a lever: it needs `request count ≡ 0 mod tp`
   on top of that request count fitting the token budget, which is not a deployable
   condition.

## Coverage and version scope

- Nothing in `tests/` combines `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` with a
  `speculative_config`: the FC2 e2e rows are `test_qwen3_moe_fc2_tp2` and
  `test_qwen3_moe_fc2_oshard_tp2`
  (`tests/e2e/multicard/2-cards/test_offline_inference_distributed.py:147,166`) and that
  file's only spec-decode row (`:246`, `deepseek_mtp`) sets no FC2. No doc page states
  the token-divisibility requirement either, so quote the code, not the docs.
- At v0.17.0rc1 the whole multi-step loop still runs inside the single
  `set_ascend_forward_context(... num_tokens=num_input_tokens ...)` block
  (`eagle_proposer.py:735-760`), so the literal `Batch_size(4)` may **not** reproduce at
  this pin even though the defect shape does. Reproduce the shape (a draft pass whose
  rows are not a multiple of tp) and say the literal number is ref-specific; #7809's fix
  (`#7810`) is post-pin, so the pin is not evidence that the report was a config error.
- Nearby family, different owner: sequence-parallel token shaping in the drafter is
  `sequence-parallel-flashcomm.md` §*FC1 with a MoE MTP drafter*, and the neighbouring
  history entry is #4011 (`[Bug]: flashcomm1 report shape error in DP>1`) — the
  flashcomm × draft-model shape family, not this assert.
