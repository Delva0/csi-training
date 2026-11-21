# FlashAttentionScore tiling refuses a caller argument: mask shape and the TND seq-len tail

## Scope

A lane-C paste whose deepest trainer/TorchNPU frame is
`call aclnnFlashAttentionScore failed` (or `aclnnFlashAttentionVarLenScore failed`)
inside `torch_npu`'s `compiler_depend.ts` wrapper, with `ERR00100 PTA call acl api
failed` / `EZ9999: Inner Error!` and one or two CANN tiling sentences from
`flash_attention_score_tiling_general.cpp`. `FlashAttentionScore` is the **built-in**
CANN operator behind `torch_npu.npu_fusion_attention`, so `ls csrc` finds nothing and
the aclnn name maps to no `vllm_ascend/` kernel: the owner is always the Python
argument list at the launch, and the `FUNC:`/`FILE:` pair names **which** argument was
refused. Read that pair before the error code — `561103`, `161001` and `EZ1001` are
shared across unrelated lanes (`moe-init-routing-tiling.md`,
`tiling-attribute-rejection.md`).

Not this page: the same refusal from **`npu_fused_infer_attention_score`**
(`aclnnFusedInferAttentionScoreV3`, `CheckFAISeqlenDataInTND`,
`queryT(16) must be equal to the last element of actualSequenceLengthQ(13)`) —
`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`; the **rank** of a caller-built
mask (`atten mask dim should be 2 or 4, but got 3` out of the same
`AnalyzeOptionalInput`/same `.cpp`, #115) — `tiling-attribute-rejection.md`
§*Mask rank*, which reaches this page's conclusion on a different rejected property
(a model-side mask, and no mode/dtype/eager knob moves it); a custom-op tiling
attribute rejection (`tiling-attribute-rejection.md`); a device dump naming a kernel
(`aclnn-error-dump-reading.md`).

## The two members

| Tiling block in the paste | Member |
|---|---|
| `get unsupported atten_mask shape, the shape is [16, 1, 1, 1500]. B=[16], N=[20], Sq=[1500], Skv=[1500], supported atten_mask shape can be [B, N, Sq, Skv], [B, 1, Sq, Skv], [1, 1, Sq, Skv] and [Sq, Skv].[FUNC:AnalyzeOptionalInput][FILE:flash_attention_score_tiling_general.cpp][LINE:1606]` | **A — the mask's query axis.** The accepted list has `Sq` in slot 3; this mask has `1` there, i.e. it is an *additive broadcast over the query axis* (`[B,1,1,Skv]`). No batch value makes it legal, so the numbers in `B=`/`Sq=` are a distraction (#6573). |
| `The end of actualSeqQLen & actualSeqKvLen should be 0 in EOD scenario, but got (15) and (15).[FUNC:Analyze3DimLayout]…[LINE:1249]` + `Get unsupported layout: TND[FUNC:AnalyzeLayout]…[LINE:1101]` + `AclNN_Parameter_Error(EZ1001): FlashAttentionScore launch kernel failed` / `Kernel GetWorkspace failed. opType: 29` | **B — the TND tail invariant.** Some rows of the query tensor are not covered by the cumulative `actualSeqQLen`; the tiling calls that trailing region *end-of-data* and demands a `0` terminator, which this caller never writes. The two printed values are the **last** cumulative lengths, and two equal values say q and kv agree — the missing fact is `query.shape[0]` (#10757). |

`--enforce-eager` is not a shared ablation: it changes nothing for A (the mask is
built by the model in every mode) and it is the recorded workaround for B (no token
padding, so the tail invariant holds). Say which member the paste is, then use the
matching route below.

## Member A — a model-side mask reaches the fused kernel

The mask is not vllm-ascend's. At vllm-ascend v0.14.0rc1 every mask the plugin owns
is a `[max_seq_len, max_seq_len]` triangle out of `AttentionMaskBuilder`
(`attention/attention_mask.py:24`, `:57-82`), and the encoder/pooling FA branch passes
**no** mask at all (`attention/attention_v1.py:819-881`) — so a `[B,1,1,S]` mask cannot
come from the attention backend, and walking `attention_v1.py`, `mla_v1.py` and
`ops/mm_encoder_attention.py` cannot produce it.

Where it does come from is the `get_extended_attention_mask` /
`_prepare_4d_attention_mask` family (a 2-D `[B,S]` padding mask expanded to
`[B,1,1,S]`), and the recorded caller is an **audio tower**:

- vllm v0.14.1 `vllm/model_executor/models/ultravox.py:441-467`
  (`ModifiedWhisperEncoder.get_attention_mask_by_audio_len`) builds
  `attention_mask[:, None, None, :]` through `get_extended_attention_mask` and passes
  it to every layer (`:494-506`); `_audio_features_to_embeddings` (`:591-609`) loops
  in `_MAX_ENCODER_BATCH_SIZE` (=16, `:64`) slices, and the ultravox projector does
  the same at `:376-381`. The tower's layers are **transformers'**
  `WhisperEncoderLayer`/`WhisperAttention` (imported at `:19-21`), not vllm's
  `vit`/`MMEncoderAttention` path — so no `vllm_ascend` op override is involved and
  the flattening frame (`flattened_embeddings = embeddings[mask]`, `:675`) is only
  where the async failure surfaced.
- That the same call reaches `aclnnFlashAttentionScore` means the HF attention
  resolved to a fused NPU kernel rather than the math fallback. The trees here cannot
  settle which (torch_npu ships its own `scaled_dot_product_attention`; its older
  `torch_npu.contrib.function.npu_fused_attention` wrapper enforces the same
  `mask.shape[1] == 1 and shape[2] == shape[3]` contract). Cheapest discriminator,
  six lines, no vLLM:
  `uv run --no-project python -c "import torch, torch_npu, torch.nn.functional as F; q=torch.randn(1,20,1500,64,device='npu',dtype=torch.bfloat16); m=torch.zeros(1,1,1,1500,device='npu',dtype=torch.bfloat16); F.scaled_dot_product_attention(q,q,q,attn_mask=m)"`
  — the same `get unsupported atten_mask shape` proves the routing, and repeating with
  `[1,1,1500,1500]` proves the mask is the only defect. Name the mechanism, hand the
  repro over, and do not spend turns on "which line": the producing line is a
  confirmation.

Next steps for the reporter, in order: print `attention_mask.shape` at the tower's
`forward` (proves the `Sq=1` axis and settles the `B=` conflict), then the model-side
patch (pass `attention_mask=None`, or a `[B,1,Sq,Skv]` mask), then the version question. #6573's
reporter later said the same model deployed before vllm-ascend **v0.11.0rc1** and
never after, so the diff that matters is the *pair*: the audio-lens mask plumbing in
vllm and the NPU attention routing in the image — one `rg -n
'get_attention_mask_by_audio_len|audio_lens' ` in the reporter's vllm tree settles
which side moved.

## Member B — the TND tail invariant, and the one call site that forgets to trim

CANN's TND FA call requires `last(actual_seq_qlen) == query.shape[0]` (and the kv
twin). `vllm_ascend/attention/attention_v1.py:863-881`
(`_forward_encoder_attention`) is the only FA (`npu_fusion_attention`) call in the
tree at v0.18.0 outside `ops/mm_encoder_attention.py:161`, and it is the one caller
that passes **untrimmed** `query`:

- the FIA sibling immediately above it trims first —
  `forward_fused_infer_attention` does `num_tokens = attn_metadata.actual_seq_lengths_q[-1];
  query = query[:num_tokens]` (`attention_v1.py:789-790`, `key`/`value` too at
  `:792-795`) — which is why every FIA model survives a padded batch;
- `_forward_encoder_attention` passes `query/key/value` as handed to it with
  `actual_seq_qlen=actual_seq_kvlen=attn_metadata.actual_seq_lengths_q`, so any
  graph-mode token padding (vLLM rounds `num_tokens` up to the dispatched bucket,
  `vllm/v1/cudagraph_dispatcher.py:231-323`) breaks the invariant by construction.
  `_pad_query_start_loc_for_fia` exists exactly for this and says so in its docstring
  (`worker/model_runner_v1.py:527-537`: "the first dimension of `hidden_states` must
  equal the last element of `actual_seq_lengths_q`"), but the real-run call site runs
  only when `cudagraph_mode == FULL` or SP is on (`:1256-1263`, with the maintainers'
  own note that the omitted `num_tokens_padded != num_tokens` case is "way too big and
  the consequences are unpredictable"); the capture path calls it unconditionally
  (`:2401`). Eager never pads, which is the reporter's own control ("eager 模式下不会触发").

So the ranked answer for this shape is: (1) the encoder/pooling FA call does not trim
`query` to `actual_seq_lengths_q[-1]` (fix: trim, as the FIA sibling does) — confirmed
by the reporter's eager-vs-graph split, and by printing `query.shape[0]` and
`attn_metadata.actual_seq_lengths_q[-1]` in the same step; (2) which padding produced
the gap is a *secondary* fact decided by the effective `cudagraph_mode`
(`platform.py:353-355` demotes `FULL_AND_PIECEWISE` to `PIECEWISE`, so the default serve is piecewise) — state it as
the check, not as the cause; (3) `--enforce-eager` / `cudagraph_mode: NONE` is the
workaround until (1) lands.

**The `causal` gate is a version question, not a puzzle.** The encoder branch is opened
by `attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal`
(`attention_v1.py:972`), and at the pinned v0.18.0 **nothing** sets it false: the V1
runner hardcodes `causal=True` (`worker/model_runner_v1.py:2127`) and the V2 builder
omits the field, so it keeps upstream's default `True`
(`vllm/v1/attention/backend.py:351`). The paste still proves the branch ran (only it
launches FA-varlen), therefore the reporter's build — `vLLM Version: dev` in the
collect_env — is newer than the pin. Ask for the vLLM commit; do not derive how
`causal` became false from the snapshot. Recorded cost of deriving it anyway: #10757
spent most of a 16 K output budget on this side question and shipped no answer.

## Rules

- Classify on `FUNC:`/`FILE:` first, then say in the first message which argument was
  refused and whether eager is a control (member A: no; member B: yes).
- A `[B,1,1,Skv]` mask never becomes legal; a legal one has `Sq` in slot 3. Do not
  explain the batch number, and do not re-derive it — the profile batch size in the
  log ("profiled with 1 audio items") and the `B=` in the paste are different numbers
  and the conflict is the check to hand over, not a mechanism to prove.
- A tiling refusal is an argument refusal: the deliverable is the invariant, the one
  producing call site and the discriminating print — never a source walk down every
  candidate caller.
