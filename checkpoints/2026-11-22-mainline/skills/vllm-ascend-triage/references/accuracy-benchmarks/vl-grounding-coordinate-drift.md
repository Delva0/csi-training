# VL grounding: right structure, right words, **wrong numbers**

## Scope

A vision request whose answer is fluent, on-topic and **schema-valid** (guided
decoding holds), but the *numbers* in it don't land on the image: bbox /
keypoint / absolute-pixel coordinates that miss the objects, measured as IoU
against ground-truth labels, with the same checkpoint through `transformers` as
the control (`Qwen3VLForConditionalGeneration.generate()`, ms-swift
`--infer_backend pt`) scoring far better on the same images. Often paired with a
wrong self-reported `image_size` in *some* TP configs and a correct one in
others, while the IoU row fails in all of them.

Recorded #18197: vllm-ascend v0.23.0 (`5cb98caa`), vllm v0.23.0 (`0fc695f`),
8×910B4, Qwen3-VL architecture with a merged (offline `merge_lora`) grounding
adapter, `--tensor-parallel-size 8 --max-model-len 32768 --enforce-eager
--reasoning-parser qwen3 --mm-processor-kwargs '{"max_pixels": 262144}'`;
transformers 100 % (IoU>0.3) across 6 images, the serve 0 % at TP=8 **and** at
TP=2 (TP=2's self-reported size was numerically right), same result with the
`max_pixels` override removed.

Turn here from `vl-vision-tower-degeneracy.md` (that page owns *repeated*,
*coarse single-element*, *unreadable-OCR* and *mis-structured* answers — where
the **content** degrades) and from `harness-vs-engine-attribution.md` (that page
owns a score gap with **no** reference run). Here the content is right and the
geometry is wrong, so the question is which stage carries positional
information.

## Order, before any model-file walk

1. **The body's own matrix is the classification.** The transformers row retires
   the checkpoint, the weights, the training data, the prompt and the parser;
   TP=8-fails plus TP=2-fails retires TP as the sole owner; with/without
   `max_pixels` retires that key. Say those three sentences, and do not re-derive
   them from source.
2. **Input-side arithmetic, one exact comparison.** Push one of the reporter's
   images through `AutoProcessor` (HF) and read the same engine request's
   `image_grid_thw` / `usage.prompt_tokens`: `smart_resize` rounding and a
   `max_pixels` cap are the two things a harness silently changes, and equal grid
   + equal prompt tokens close the processor lane by arithmetic instead of by
   argument. This is the same certificate `vl-vision-tower-degeneracy.md` uses
   for #6909; here the reporter has already varied `max_pixels`, so compare at
   *their* setting rather than assuming the default.
3. **A content control on the same checkpoint and the same image** — "what is it /
   how many / what colour", no coordinates requested. Content right while the
   coordinates miss ⇒ the *positional* lane (RoPE / m-RoPE / position embedding),
   because the tower feeds both surfaces. Content also coarse or wrong ⇒ tower or
   grid lane: back to `vl-vision-tower-degeneracy.md`. This is the cheapest
   discriminator and it costs one request; the recorded run never sent it.
4. **On v0.23.0 the positional lane is downstream code, and it has no switch.**
   Quote these before theorising about the model file:
   - `vllm_ascend/patch/worker/__init__.py:48-51` imports `patch_qwen3vl` on
     every non-310P serve; `patch_qwen3vl.py:70-71` assigns
     `Qwen3Attention.forward = Qwen3MoeAttention.forward =
     forward_with_split_qkv_rmsnorm_mrope`, so on Qwen3-VL the qkv split, the
     q/k RMSNorm **and** the rope are downstream, in every text layer.
   - When `isinstance(self.rotary_emb, AscendMRotaryEmbedding)` (`:37`) that
     forward calls the fused `torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(...,
     mrope_section=…, is_interleaved=…, rope_dim=…)` (`:43-56`); otherwise it
     takes the plain `split` + `q_norm`/`k_norm` + `self.rotary_emb(...)` path.
   - The same file replaces `Qwen3_VisionTransformer.fast_pos_embed_interpolate`
     with upstream's eager `pos_embed_interpolate_native` (`:77-92`, assigned at `:94`) — same
     math, the triton bilinear kernel bypassed — so "the downstream pos-embed
     kernel" is not a suspect on this pin.
   - `tensor_parallel_wrap` (`:15-31`) chunks `deepstack_input_embeds` by TP
     **only when `_EXTRA_CTX.flash_comm_v1_enabled`**, and flashcomm1 defaults
     off (`vllm_ascend/envs.py:75`, `VLLM_ASCEND_ENABLE_FLASHCOMM1` default
     `"0"`; `additional-config` key `enable_flashcomm1`). A plain TP=8 serve does
     not reach it — do not rank it for a command with no such env/config key.
5. **Correct the plausible-but-wrong elimination.** "Qwen3-VL gets
   `MRotaryEmbeddingInterleaved`, that class is not in the Ascend OOT table, so
   the Ascend rope never runs and m-RoPE is innocent" is false:
   `get_rope`'s `scaling_type == "default"` branch builds a plain
   `MRotaryEmbedding(..., mrope_interleaved=…)`
   (`vllm/model_executor/layers/rotary_embedding/__init__.py:102-113`), while
   `MRotaryEmbeddingInterleaved` is built only by the `openpangu` branch
   (`:337-347`). OOT dispatch is keyed on the exact class name
   (`CustomOp.__new__`, `vllm/model_executor/custom_op.py:109-125`, against
   `op_registry_oot`, filled by `CustomOp.register_oot` at
   `vllm_ascend/utils.py:817-818`), and `"MRotaryEmbedding"` is in that table
   (`utils.py:740`; the same fact is used by
   `../startup-failures/env-write-in-compiled-forward.md`). So Qwen3-VL's rope
   **is** `AscendMRotaryEmbedding` and the fused op receives
   `is_interleaved=True`.
   Recorded #18197 spent its entire 16 384-token output budget on this walk and
   shipped **no** message; the elimination above was its last reasoning block and
   it is wrong.
6. **The mask math is not decided by reading it.** In
   `vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py:114-116`
   `h_mask = (i%3==1) & (i <= 3*section_h)`, `w_mask = (i%3==2) & (i <=
   3*section_w)`, `t_mask = ~(h|w)`; upstream `apply_interleaved_rope`
   (`vllm/model_executor/layers/rotary_embedding/mrope.py:190-198`, writes at `:196-197`) writes
   `x_t[..., 1 : section[1]*3 : 3]` and `2 : section[2]*3 : 3`. The two sets are
   the same for Qwen3-VL's `[24,20,20]` (3·section is a multiple of 3, so
   `<=` vs `<` changes nothing; 24+20+20 = 64 = `rotary_dim/2`), so reading the
   kernel neither opens nor closes the lane. Settle it with the op's own UT or an
   eager comparison.
7. **Coverage, quoted as coverage.** `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_mrope.py:6-14`
   parametrises exactly this model's combination — `mrope_section
   [[11,11,10],[24,20,20]] × is_interleaved {True,False} × head_size {128,256} ×
   {bf16,fp16} × num_tokens {1,4096}` — against a naive reference that does apply
   `apply_interleaved_rope`, but **no scheduled workflow runs that directory**:
   `.github/workflows/configs/nightly_config.yaml` lists only
   `tests/e2e/nightly/single_node/ops/multicard_ops_a2/` for the pytest-driven A2
   single-node case, and `_e2e_nightly_single_node.yaml:241` only `--ignore`s a
   sibling file, showing the path is meant to be passed in. So the sentence is
   "the UT exists and covers this exact combination, and the nightly matrix does
   not run it", plus the one-file run inside the reporter's image.
8. **What to send back**: `image_grid_thw` for one image from both paths (HF
   processor and the engine), the `usage.prompt_tokens` of the failing request,
   the content-control reply, and one IoU row per launch config. A traceback is
   not owed — this lane has none by construction.

## What not to lead with on this profile

- **Graph capture / `--enforce-eager`**: already on in the recorded report, and
  capture would corrupt the whole answer, not only its numbers.
- **Sampling / temperature**: greedy by the report, and a sampler does not have a
  coordinate-shaped failure.
- **The `--dtype bfloat16` restart** (`precision-dtype-degeneracy.md`): that rung
  belongs to a *repeated single token*, not to fluent wrong numbers; the HF
  control is normally bf16 too, so it is not a reference-direction change.
- **The LoRA runtime path**: an offline `merge_lora` checkpoint does not enter
  `--enable-lora`; a *separate* garbage-output report on the runtime path is
  `../lora-adapters/index.md`.
- **The `extract_slice` alignment lane** (`../image-dependency-faults/triton-extension-op-silent-miscompute.md`):
  that page's rule 6 already records the offsets in this same kernel as
  `0`/`HALF_ROPE_DIM`/`COL_BLOCK_SIZE` steps, i.e. aligned for rope_dim 64/128 in
  bf16 — quote it as "not exposed at this pin", not as the mechanism.
