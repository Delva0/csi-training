# "ACLGraph is slower than eager" — and the reporter already named the line

## When this page applies

A performance claim whose direction is **graph mode slower than eager**, on a
serve that works: "[Bug]: AclGraph mode underperforms eager mode in Qwen
speculative decoding scenario", "开了图模式反而慢 20%", usually with a
`npu-smi`/`collect_env` block and **no** benchmark table. The reporter has often
done the ablation themselves and pastes the mechanism: "在 ACLGraph 模式下，原本
padding 的 query 被截断了，导致部分算子进不了图；把这行删掉就快了" plus a
`path:line` and a one-line patch. Deliverable: a verdict on **their** patch —
whether the line is removable, what it compensates for, and the patch that keeps
the contract — not a tutorial on graph capture.

Route here, not to `spec-decode-gain-accounting.md`: that page prices
MTP-on vs MTP-off on one image (step cost ratio vs tokens per step). This one is
the *graph-mode axis* with the same flags.

## Step 1 — confirm the cited block at the pin, then read one block ABOVE it

Recorded #5124, vllm-ascend v0.11.0rc3 (`b6d63bbd5294`) / vllm v0.11.0: the
report cites `attention_v1.py:670-676`; the tree carries it at
**`vllm_ascend/attention/attention_v1.py:669-676`**:

```python
else:                                    # "Normal V1 situation."
    num_tokens = attn_metadata.query_start_loc[-1]
    query = query[:num_tokens]           # ← the reporter's line (:675)
    output = self._forward_v1_style(query, attn_metadata, output)
```

The comment above the slice states the op contract it protects
(`npu_fused_infer_attention_score` does not support
`query.shape[0] != query_start_loc[-1]`). So the question is not "is the slice
wasteful" but **which branch pads the query and which pads the metadata**:
`AscendAttentionMetadataBuilder.build` pads `seq_lens`, `block_table` and
`query_start_loc` up to `common_attn_metadata.num_input_tokens` only under a
**state gate** — `attention_v1.py:228-248`, condition
`attn_state == AscendAttentionState.DecodeOnly and num_input_tokens > num_actual_tokens`,
with the `arange` extension of `query_start_loc_cpu` at `:242-249`. Every state
that reaches the `else` therefore arrives padded in one field and unpadded in the
other, and the slice is the compensation. **One block one level up owns the
verdict**; the slice is downstream of it.

## Step 2 — check who assigns the state enum before reasoning from its name

The enum member existing is not evidence that this path sets it. At v0.11.0rc3
`worker/model_runner_v1.py:1609-1633` (`_build_attn_state`) assigns
`SpecDecoding` **only for `self.speculative_config.method == 'deepseek_mtp'`**;
every other speculative method (Qwen `mtp`/`qwen3_next_mtp`, ngram, suffix) falls
into `DecodeOnly` (all scheduled tokens == 1) or `ChunkedPrefill`.

That correction matters twice: the recorded answer read "spec decode → the
`SpecDecoding` state → the truncating branch" and proposed widening the `:228`
gate to `(DecodeOnly, SpecDecoding)` — which does not cover the reporter's Qwen
MTP batch at this ref. State the state you verified for their flags
(`grep -n "_build_attn_state" -A 25 vllm_ascend/worker/model_runner_v1.py`), or
say the fix must cover every state that reaches the branch.

## Step 3 — the two cost mechanisms, both cheap to name

1. **A device→host sync per attention layer per step.**
   `query_start_loc: torch.Tensor` (`attention_v1.py:151`), moved to the device at
   `:250` — so `query_start_loc[-1]` at `:674` is a 0-dim NPU scalar used as a
   Python slice bound. Multiply by the layer count to price it.
2. **Non-static tiling after the slice.** `_forward_v1_style`
   (`:501-562`) feeds `npu_fused_infer_attention_score` with
   `actual_seq_lengths=actual_seq_lengths_q` and `actual_seq_lengths_kv=seq_lens_list`,
   both Python lists built on the host at `:269` and `:271` (`...tolist()`), so the TND
   call cannot be a fixed-shape graph region.

The contrast that proves the intent is one branch earlier: `_forward_decode_only`
(`:397-499`) captures the op in a `torch.npu.graph_task_group_begin/end` and
caches per-shape handles — `graph_params.handles[num_tokens].append(handle)` at
`:487` — i.e. the plain-decode path *was* built for graph mode; `mla_v1.py` has
the same handle cache (`:1071`) and handles `SpecDecoding`/`ChunkedPrefill`/
`DecodeOnly` with TND layout (`:988-996`). Read the finding as a **missing
propagation** (SKILL.md procedure step 8), not as a new design: the dense backend
lagged the MLA backend's graph plumbing.

## Step 4 — the verdict shape on a self-diagnosed reporter

- Say **"your diagnosis is right in kind"** in the first sentence, then name the
  gate one level up, so the headline is a corrected patch and not agreement.
- Do **not** endorse the bare deletion. Correctness of the deletion rests on the
  rows past `num_actual_tokens` being dropped (`ori_output[:num_tokens, :, :] =
  output[:num_tokens, :, :]`, `:681-682`), which holds only while
  `actual_seq_lengths`/`seq_lens_list` stay length-aligned with the query rows —
  the contract the comment at `:671-673` describes. Recommend extending the
  padding block to the states that reach the branch (so `query_start_loc[-1] ==
  query.shape[0]` and the slice becomes a no-op) instead of deleting the guard.
- Ablation that prices it: `num_speculative_tokens` K → 2 → 1 → off on the same
  request set (a gap that scales with K is the padded-token path); normalize the
  win as `ITL_p50 / TPOT_p50` tokens per step before quoting any percentage.
- Version scope of the drafter lever: at v0.11.0rc3 the drafter's graph gate is
  **global** `model_config.enforce_eager` (`spec_decode/eagle_proposer.py:46-48`),
  so `--speculative-config {"enforce_eager": true}` is *not* a lever at this ref —
  the opposite of the v0.18.0rc1 forcing rule in SKILL.md rule 15. Re-grep that
  file per ref before repeating either.
- What stays open: which capture mode they ran (`PIECEWISE` vs
  `FULL_DECODE_ONLY`), the padded token count the runner used, and — with a single
  pin in the snapshot — whether a later release already carries the extension.
  Send them `sed -n '665,690p' vllm_ascend/attention/attention_v1.py` to run on
  their own tree.

Cap ~6 turns: the mechanism is four greps, and the reporter's own ablation already
carries the measurement.
