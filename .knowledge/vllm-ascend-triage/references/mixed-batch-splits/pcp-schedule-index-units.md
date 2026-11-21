# PCP schedule metadata under spec decode: a request offset taken from a token
# count (#11231)

## Report

vllm-ascend **v0.21.0rc1** (`80610e4438`, the pinned tree) — the issue *title*
says 0.21.0rc1 while the body says the service was started on **v0.22.1rc1**, and
only 0.21.0rc1 is in the snapshot. Qwen3.5-27B, one host,
`--tensor-parallel-size 2 --prefill-context-parallel-size 2
--decode-context-parallel-size 1 --max-num-seqs 32
--max-num-batched-tokens 32768 --max-model-len 262144 --async-scheduling
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--speculative-config '{"num_speculative_tokens": 2, "method": "mtp"}'`.
The service **starts**, `aisbench` drives it for ~348 iterations, then all four
workers die in the same millisecond:

```text
(model_runner_v1.py, line 839, in _prepare_inputs)
    num_scheduled_tokens[:num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
(pcp_utils.py, line 667, in update_tokens_for_pcp)
    num_padded_scheduled_tokens[self.num_decode_reqs :] = num_prefill_scheduled_tokens_linear
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not broadcast input array from shape (29,) into shape (31,)
```

There is **no device frame**: lane B, Python/numpy, phase = `_prepare_inputs`
building the PCP schedule, before any kernel or collective.

## Verdict

Downstream defect in `vllm_ascend/worker/pcp_utils.py`, `update_tokens_for_pcp`'s
`pcp_use_hybrid_attn` branch: the **read** offset into the per-request
`num_scheduled_tokens` array is `self.num_decode_tokens` (a *token* count) while
the **write** target is offset by `self.num_decode_reqs` (a *request* count).
Units, from `init_batch_info` (`pcp_utils.py:184-197`, same file):

- `self.num_decode_reqs` = number of leading decode requests;
- `self.num_decode_tokens = num_scheduled_tokens[: self.num_decode_reqs].sum()` —
  tokens, and `decode_threshold = 1 + num_speculative_tokens` (`:68`) makes each
  decode request schedule exactly `1 + num_spec` tokens, so
  `num_decode_tokens = num_decode_reqs × (1 + num_spec)`.

The two agree only when spec decode is off. With `num_spec = 2` the source slice
is short by `num_decode_reqs × num_spec` elements and numpy refuses the
assignment.

| line (v0.21.0rc1) | expression | unit |
|---|---|---|
| `:186-194` | `num_decode_tokens = …[:num_decode_reqs].sum()` | correct definition (tokens) |
| `:660` | `torch.Tensor(num_scheduled_tokens[self.num_decode_tokens :])` | **request-indexed array sliced by a token count** |
| `:684` | `num_prefill_tokens = num_scheduled_tokens[self.num_decode_reqs :].sum()` | correct — the sibling to copy |
| `:667` | `num_padded_scheduled_tokens[self.num_decode_reqs :] = …` | correct (requests); this is where numpy reports |
| `:749-750` | `get_current_rank_positions(...)[self.num_decode_tokens :] - self.num_decode_tokens * pcp_world_size` | correct — a **token** offset into a token-indexed positions array; do not "fix" it |
| `:759` | `np.roll(np.cumsum(num_scheduled_tokens[self.num_decode_tokens :]), 1)` | **same defect, second site** |

## 1. Decode the two numbers before adding a print

`num_padded_scheduled_tokens` is built from the argument array, i.e. length
`num_reqs`, so

```text
len(target) = num_reqs - num_decode_reqs
len(source) = num_reqs - num_decode_tokens
difference  = num_decode_tokens - num_decode_reqs = num_decode_reqs * num_spec
```

`(29,) into (31,)` therefore gives `num_decode_reqs × num_spec = 2` → one decode
request with `num_spec = 2`, and `num_reqs = 32`, which is exactly the launch's
`--max-num-seqs 32`. Cross-check against the two log lines the reporter already
has: the preceding iteration prints `0 context requests, 0 context tokens,
1 generation requests, 3 generation tokens`, and `SpecDecoding metrics: Mean
acceptance length: 3.00` is the same `1 + 2`. If the derived difference is not
`num_decode_reqs × num_spec`, the wrong offset is elsewhere — look for another
`num_decode_tokens` used as a request index in the same call path.

## 2. Why only the combined launch fails

Every gate is in the tree, so a reporter's "MTP alone was clean" is consistent,
not contradictory:

1. `pcp_use_hybrid_attn` must be true — the whole branch is under it, and it is a
   model whitelist (`pcp_utils.py:131-135`: `qwen3_next`, `qwen3_5`,
   `qwen3_5_moe`). The reported line proves a whitelisted model type was in use;
   the exact string is in the checkpoint's own `config.json`, so quote the
   whitelist rather than asserting it. An MLA/DeepSeek PCP+MTP report does **not**
   reach this line (that family's failures are `#7375`, `#8578`, ...).
2. The step must hold **≥1 decode request and ≥1 prefill request**
   (`:659 if self.num_reqs - self.num_decode_reqs > 0`). A pure-decode or
   pure-prefill step passes — hence "runs hundreds of iterations, then dies",
   and hence `--max-num-seqs 32` with `Waiting: 2 reqs` is the trigger shape.
3. Spec decode must be on. Without it `num_decode_tokens == num_decode_reqs` and
   the two lengths are equal, so **PCP alone is clean by this arithmetic** — the
   non-error is a prediction of the mechanism, not evidence against it.

## 3. The repair, and how to know the reporter's tree has it

Fix both sites to the sibling's units (`:684` already has it right):

```python
:660  prefill_tokens_tensor = torch.Tensor(num_scheduled_tokens[self.num_decode_reqs :])
:759  ori_tokens_start_loc = np.roll(np.cumsum(num_scheduled_tokens[self.num_decode_reqs :]), 1)
```

`:759` is not optional: after `:660` is corrected,
`num_prefill_scheduled_tokens_linear` has `num_prefill_reqs` entries while
`ori_tokens_start_loc` still has `num_prefill_reqs - num_decode_reqs × num_spec`,
so `np.repeat(ori_tokens_start_loc, num_prefill_scheduled_tokens_linear)` fails
next. It is also the silent half — wrong `exit_fa_scatter_indices` reorders
tokens without raising, so verify output identity against a PCP-off run after
patching, not just absence of the exception.

**The later pin carries the fix**, so this is not a hypothetical patch: at
vllm-ascend **v0.23.0** (`5cb98caaadef`) the same function reads
`num_scheduled_tokens[self.num_decode_reqs :]` at `pcp_utils.py:821`, `:845` and
`:938`. The two pins bracket the change; the committing release between them is
not visible in this snapshot, and v0.22.1rc1 (the reporter's actual tree) has no
view at all. One `sed -n '655,670p;755,765p' vllm_ascend/worker/pcp_utils.py`
on the reporter's host settles it — ask for that and the `DBG` line below, not
for another launch.

`DBG` print (before `:659`), expected values for the recorded batch:
`num_reqs=32 num_decode_reqs=1 num_decode_tokens=3` and
`num_scheduled_tokens[:32]` = `[3, <31 prefill counts>]`.

## 4. Precedent family

MTP × PCP has failed repeatedly in this repository, which is why a new report in
that combination should not be routed to the MTP lane or the card/HCCL lanes
first: `#7375` (`aclnnInplaceCopy … 4050 and 16 cannot broadcast`, the broadcast
symptom one layer down), `#7122`, `#8402` (eagle3 + PCP), `#8578`
(`'PCPManager' object has no attribute 'mtp_slot_pad'`), `#8926` (v0.19.0rc1
PCP + spec decode), and `#10417` (RFC: PCP/DCP unification — states outright that
`update_tokens_for_pcp` carries two token-split policies and both are
maintained by hand). These are precedent, not the mechanism of this report.
