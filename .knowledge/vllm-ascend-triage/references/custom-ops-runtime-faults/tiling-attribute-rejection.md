# `<tensor> shape is invalid` from `<op>_tiling.cpp`: an attribute rejection, not a split bug

## Scope

A paste whose deepest line is an `EZ9999` sentence about one tensor's shape, carrying
`[FUNC:GetShapeAttrsInfo][FILE:<op>_tiling.cpp]` (or `<op>_base_tiling.cpp`), followed by
`TraceBack … <Op> do tiling failed, ret is -1.` and the
`NnopbaseExecutorDoTiling / TilingAndUpdateBinInfo / MatchCache / RunForWorkspace` chain,
under `call aclnn<Op> failed, error code is 561002`. Nothing launched: there is no device
dump, no `aivec`, no `pc start`, and often no per-rank frame beyond `torch_npu.<op>`.
Recorded on the DeepSeek-V3.x latent-KV write, `npu_kv_rmsnorm_rope_cache` (#6243
v0.12.0rc1, SFA path; #2732 DS-V3.1, torchair MLA path) and on
`npu_interleave_rope` (#2702, non-MLA).

Also this page, on a **built-in** op with no `csrc/` owner: the same argument check
applied to the **rank of a caller-built attention mask** — `atten mask dim should be 2 or
4, but got 3` from `flash_attention_score_tiling_general.cpp` (§*Mask rank*).

Not this page: `EZ1001 … Get regInfo failed`/`does not support opType` (the op was never
resolved for the SoC — `op-not-registered-for-soc.md`), a dump naming a kernel with a
chip id (`../hccl-and-hardware/index.md`, `aclnn-error-dump-reading.md`), a numeric
aclnn code with **no** `EZ9999` body (`moe-init-routing-tiling.md` for `561103` on MoE
routing, which is the same argument-rejection family on another op), and the **shape**
of a caller-built mask (`get unsupported atten_mask shape … [B, 1, 1, Skv]`, same
`AnalyzeOptionalInput`) — `fa-tiling-argument-refusals.md` member A.

## What the line proves, and what the reporter's title usually gets wrong

`GetShapeAttrsInfo` is the tiling entry's **attribute validation**: it runs before any
split arithmetic, so a message from it says the arguments were refused, not that a tiling
split was computed wrongly. Say that in the first sentence — it redirects the reporter
from "算子 tiling 切分有误" to the tensors they can change, and it is decidable from their
own paste without any source.

The check family is a **row-count comparison**: the rotated tensors' first dim against the
operand's first dim. Three recorded wordings, two ops:

| Message | Op | Site in the paste |
|---|---|---|
| `cos shape is invalid` | `npu_kv_rmsnorm_rope_cache` | #2732, `kv_rms_norm_rope_cache_tiling.cpp:298` |
| `cos or sin shape is invalid` | `npu_kv_rmsnorm_rope_cache` | #6243, `kv_rms_norm_rope_cache_base_tiling.cpp:291` |
| `cos or sin batchSize must same as x batchSize` | `npu_interleave_rope` | #2702, `interleave_rope_tiling.cpp:124` |

So the decisive numbers are `cos.shape[0]`, `sin.shape[0]` and the batched operand's
`shape[0]`, plus their dtype — never the block dim, the core count or the card.

## The caller's tensors, at the pin

vllm-ascend v0.13.0rc2 `a17059a2` (`SOURCES.json` ref of #6243; the reporter's own
v0.12.0rc1 tree is not in that view, but the `exec_kv` body they pasted in the follow-up is
identical to this one, so the call form is corroborated for both):

- `vllm_ascend/attention/sfa_v1.py:511-553` `exec_kv` — `kv_no_split.view(B, N, S=1,
  kv_lora_rank + qk_rope_head_dim)` with `B = kv_no_split.shape[0]`, then
  `npu_kv_rmsnorm_rope_cache(kv_no_split, kv_a_layernorm.weight, cos, sin,
  slots.to(int64), kv_cache[1], kv_cache[0], epsilon=…, cache_mode="PA")`; the
  `enable_dsa_cp` branch (`:527`) adds `is_output_kv=True` and returns `k_pe, k_nope`.
- `vllm_ascend/attention/mla_v1.py:1098` (`exec_kv_decode`, `cache_mode="PA_NZ"` when
  `enable_kv_nz`) and `:1126` (`exec_kv_prefill`) — the same argument order, so a shape
  mismatch is reachable on both the SFA and the MLA deployment.
- `cos`/`sin` come from `vllm_ascend/ops/rotary_embedding.py:107-119`
  `get_cos_and_sin_mla`: `_cos_cache[positions].unsqueeze(1).unsqueeze(2)`, i.e.
  `[tokens, 1, 1, rope_dim]`, written into a **process-global** `_cos_mla` buffer sized to
  `max_num_batched_tokens` and returned as `_cos_mla[:num_tokens]`. The rows therefore
  track whatever `positions` was, and the buffer is shared by every layer.
- The rows are set in `build()`: `positions[:num_input_tokens]` (`sfa_v1.py:198-202`), and
  the metadata stores `sin`/`cos[:num_input_tokens]` (`:294-295`) next to
  `num_input_tokens` and `num_actual_tokens` (`:284`).

`num_input_tokens` **includes padding**, `num_actual_tokens` does not
(`sfa_v1.py:101,110`). Any regime where the batched operand's rows are one of those two
and `cos`'s rows the other is the whole bug, and it is a host-side row count.

## Why it is intermittent: four regimes that change the rows at this call

"Ran hundreds of requests, then failed" is the signature of a batch regime that a
different, rarer batch reaches. Check these in order, each with one launch or one grep:

1. **Does the failing call even use this branch?** `sfa_v1.py:763` routes
   `num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS` (1024, `mla_v1.py:52`) through the
   fused `_sfa_preprocess_decode`, which writes the cache without `exec_kv`; on the MLA
   path MLAPO covers **decode only** (`mla_v1.py:1497-1499`: `not has_prefill and
   num_decode_tokens <= 1024`), so a prefill batch always reaches `exec_kv_prefill`. A
   traceback inside `exec_kv` therefore means MLAPO is off, or the batch was a prefill /
   exceeded 1024 tokens — a long prompt's chunked-prefill chunk, not a decode step. Ask
   for the request's prompt length and `--max-num-batched-tokens`.
2. **Graph buckets.** A captured FULL decode bucket pads tokens up to the bucket, so the
   batched operand can hold the bucket's rows while `cos` was built from `num_input_tokens`
   — and the bucket that faults is the one the load happens to hit. #2732 is the graph-mode
   instance: torchair graph on, `graph_batch_sizes: [32]`, DP 2, failing under
   `benchmark_serving`, same op, same `EZ9999` body. `--enforce-eager` is the ablation.
3. **DSA-CP (SP) slicing.** With `enable_dsa_cp`, `build()` pads `cos`/`sin` to
   `_round_up(num_input_tokens, tp_world_size)` and then slices
   `cos[local_start:local_end_with_pad]` to `num_tokens_per_device`
   (`sfa_v1.py:205-236`), while the forward passes `dsa_cp_context.slot_mapping_cp` and
   that sliced `cos`. Rows now depend on the TP world size and on the batch size **mod**
   tp — a regime that only some batches hit. Compare `cos.shape[0]` with `kv`'s `B` and
   with `slots.numel()` in the dump below.
4. **SP / flashcomm gather.** In the non-MLAPO branch, `q_c` and `kv_no_split` go through
   `torch.ops.vllm.maybe_all_gather_and_maybe_unpad` (`sfa_v1.py:790-796`) while `cos`
   is not re-gathered — so a per-rank row count that disagrees with the gathered one is
   another row mismatch of exactly this family.

## The two artefacts to ask for, in this order

1. **The shapes at the failing call** — a five-line dump answers the whole question, and
   it needs no rebuild:

   ```python
   import torch_npu, functools
   _orig = torch_npu.npu_kv_rmsnorm_rope_cache
   @functools.wraps(_orig)
   def _spy(kv, w, cos, sin, slots, *a, **kw):
       if cos.shape[0] != kv.shape[0] or sin.shape[0] != kv.shape[0]:
           print("MISMATCH", kv.shape, kv.dtype, cos.shape, cos.dtype,
                 sin.shape, slots.shape, kw, flush=True)
       return _orig(kv, w, cos, sin, slots, *a, **kw)
   torch_npu.npu_kv_rmsnorm_rope_cache = _spy
   ```

   If nothing prints, widen the spy to log every call (shapes, dtype, `kv` rows) and take
   the run that fails — the condition above is a hypothesis, the dump is the evidence, so
   ask for the whole log rather than only the mismatch lines. Ask alongside it: `num_input_tokens` / `num_actual_tokens`, the
   `--max-num-batched-tokens` value, `VLLM_ASCEND_ENABLE_MLAPO`, `enable_dsa_cp` and the
   TP world size.
2. **The wiki/patch text they already applied.** #6243's body ends at "做出如下修改：" with
   the modification missing — a report whose fix is not in the paste cannot be graded, and
   naming that gap is part of the answer.

Optional, and only as the third item: a **framework-free sweep** — the reporter runs it on
their box, since the op is not in this environment:

```python
import torch, torch_npu
# Take every number and both cache shapes from the reporter's dump: the argument order
# (kv_cache[1] then kv_cache[0]) and the cache dtypes are theirs, not a recalled default.
B, N, D, R, eps = ROWS_FROM_THE_DUMP, 1, 512 + 64, 64, 1e-6   # kv_lora_rank 512 + rope 64
for b in sorted({B, B - 1, 1, 32, 1024, 1025}):
    kv = torch.randn(b, N, 1, D, dtype=torch.bfloat16, device="npu:0")
    w = torch.ones(512, dtype=torch.bfloat16, device="npu:0")
    cos = torch.randn(b, 1, 1, R, dtype=torch.bfloat16, device="npu:0")
    sin = torch.randn(b, 1, 1, R, dtype=torch.bfloat16, device="npu:0")
    slots = torch.arange(b, dtype=torch.int64, device="npu:0")
    caches = (torch.zeros(1, 128, 512, dtype=torch.bfloat16, device="npu:0"),
              torch.zeros(1, 128, 64, dtype=torch.bfloat16, device="npu:0"))
    try:
        torch_npu.npu_kv_rmsnorm_rope_cache(kv, w, cos, sin, slots, caches[1],
                                            caches[0], epsilon=eps, cache_mode="PA")
        print("OK", b)
    except RuntimeError as e:
        print("FAIL", b, str(e).splitlines()[0])
```

A sweep that reproduces it hands CANN exact shapes; a sweep that passes at every `b`
hands the case straight back to the caller, which is the useful verdict either way.

## Ownership when both sides decline

#6243's follow-up is the shape to expect: the operator owner answered "算子侧认为为调度问题"
(the op side reads it as a scheduling issue) and returned the case, while the framework's
own reading of the same line is "the op rejected my argument". Do not settle that with a
source walk — the tiling `.cpp` is in no export of these cases, and the check is
deterministic on the numbers, so the dump above *is* the tiebreaker. Answer with
(a) the stage classification (attribute check, nothing launched), (b) the tensors whose
rows must agree and their pinned construction sites, (c) the two artefacts, and (d) the
`--enforce-eager` / one-request-versus-load A/Bs from the regime list. Do not claim the op
or the caller is at fault before the shape line exists.

## Mask rank: `atten mask dim should be 2 or 4, but got 3` (a built-in op, a VL vision tower)

Verified at vllm `v0.7.1` (`4f4d427ac2ce`) and vllm-ascend `v0.7.1rc1` (`f17417f2d5b5`),
recorded for #115: 8×910B2, `quay.io/ascend/vllm-ascend:v0.7.1rc1`, Qwen2-VL-72B-Instruct,
`--tensor-parallel-size 4 --limit-mm-per-prompt image=6 --gpu-memory-utilization 0.9`,
`ASCEND_LAUNCH_BLOCKING=1`. Engine init, KV profiling and the 87 s warmup all finish; the
**first image request** kills every rank.

```
RuntimeError: call aclnnFlashAttentionScore failed, detail:EZ9999: Inner Error!
EZ9999: [PID: 480] … atten mask dim should be 2 or 4, but got 3[FUNC:AnalyzeOptionalInput]
        [FILE:flash_attention_score_tiling_general.cpp][LINE:1473]
        fail to analyze context info.[FUNC:GetShapeAttrsInfo][…][LINE:835]
        Tiling failed / Tiling Failed. / Kernel GetWorkspace failed. opType: 40
        launch failed for FlashAttentionScore, errno:561103.
[ERROR] ERR01100 OPS call acl api failed
deepest Python frame: vllm/model_executor/models/qwen2_vl.py:345
        output = F.scaled_dot_product_attention(q,
```

`AnalyzeOptionalInput` is the tiling entry's optional-input check, so this is the same
family as the rows above — **an argument was refused, nothing launched** — with three
differences worth typing before any walk: the op is a CANN **built-in** (no `csrc/`, no
`_cann_ops_custom`, no `ASCEND_CUSTOM_OPP_PATH` question); the rejected argument is a
**rank**, so no `--enforce-eager`, `ASCEND_LAUNCH_BLOCKING=1`, graph-mode or dtype knob can
move it; and the mask was built by **upstream vLLM's model file**, which dates the run.

The chain, verified line by line at that pin (the traceback's other frames — `:400`,
`:612`, `:614` — match the same file, so the container is running this tree):

- `vllm/model_executor/models/vision.py:79-103` `get_vit_attn_backend`: an explicit
  `VLLM_ATTENTION_BACKEND` wins, otherwise `current_platform.is_cuda()` decides and **every
  non-GPU platform falls through to `_Backend.TORCH_SDPA`** (`:102-103`, the comment says
  so). Ascend therefore never takes the flash-attn branch.
- `vllm/model_executor/models/qwen2_vl.py:336-349` — the `TORCH_SDPA` branch builds
  `attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool)` and fills
  it per `cu_seqlens` block (`:339-344`), then calls
  `F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)` at `:345` — a
  **3-D** mask. torch_npu lowers SDPA to `aclnnFlashAttentionScore`, whose `atten_mask`
  accepts 2-D or 4-D, hence `got 3`.
- `vllm_ascend/` at `v0.7.1rc1` has **no** `qwen2_vl` and no
  `scaled_dot_product_attention` reference (`rg -n 'qwen2_vl|scaled_dot_product_attention'
  vllm_ascend/` → empty), so there is no downstream patch to read and no plugin defect to
  argue: the owner is the upstream model file. vllm-ascend's own
  `tools/npu-vllm-test.sh:165,191,208` shows the Qwen2-VL ViT tests **commented out** at
  that ref — quotable as coverage, never as the cause.

The two checks that make this a verdict instead of a theory: a **text-only** request must
succeed (the mask exists only on the ViT path) while one image request reproduces the
crash; then
`sed -n '336,350p' /usr/local/python3.10/lib/python3.10/site-packages/vllm/model_executor/models/qwen2_vl.py`
inside their container shows whether that branch is what ran.

Ladder: (1) the version rung — the same request on the current image, because the branch
was rewritten after this pin and the snapshot cannot hold any later fix; (2) the mask
repair, which must keep the block-diagonal semantics: `attention_mask.unsqueeze(1)` to
`[1, 1, S, S]`, acceptance = one image request whose transcription matches a working
reference **plus** a two-image request (a single-image-only trick such as `attn_mask=None`
is equivalent for one segment and silently lets images attend to each other). Do **not**
offer `VLLM_ATTENTION_BACKEND=FLASH_ATTN`/`XFORMERS` as a rung without checking the image:
those branches `import flash_attn` (`qwen2_vl.py:318`) and `xformers` (`:352-353`).

Do not chase: the `current tensor is running as_strided, don't perform inplace operations …
allow_internal_format = False` warning is the `x = x.unsqueeze(1)` at `:612` two frames up
(a layout advisory, not the mask); `Token indices sequence length … 114688 > 32768` and the
`profiling.py:184` multimodal-budget warning come from `--limit-mm-per-prompt image=6` and
are warnings only; and `561103`/`ERR01100` name no lane by themselves.
