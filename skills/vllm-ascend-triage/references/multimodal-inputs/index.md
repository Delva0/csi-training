# Multimodal input assembly: placeholder count vs feature count

Scope: a VL/omni serve where the **engine rejects or mis-places the multimodal
features** — `ValueError: Attempted to assign N = N multimodal tokens to M
placeholders`, `RuntimeError: shape mismatch: value tensor of shape [N, D] cannot
be broadcast to indexing result of shape [M, D]` out of
`_merge_multimodal_embeddings`, an image in the request with no `<image_pad>`
placeholder (or the reverse), a `tgt_sizes` / `image_bound` disagreement, or a
request that dies inside `embed_input_ids` / `_preprocess` before any kernel runs.
Not this topic: a VL answer that is merely degenerate or wrong with sane counts
(`../accuracy-benchmarks/vl-vision-tower-degeneracy.md`), a **device fault mid-request**
inside the vision tower or in the LM prefill of a long/chunked image prompt
(`fault kernel_name=unpad_flashattention_…` + MTE DDR out of range: read the dumped
`SchedulerOutput` before any source walk, `../custom-ops-runtime-faults/aclnn-error-dump-reading.md`
built-in-kernel section), or a checkpoint architecture question
(`../startup-failures/model-type-not-recognized.md`).

Inherit the rules in SKILL.md.

## Common rules

- **Two counts, two implementations.** The placeholder count comes from the
  prompt-building path (for MiniCPM-V: the *transformers* image processor's
  `get_slice_image_placeholder`, called from
  `vllm/transformers_utils/processors/minicpmv.py:_convert_images_texts_to_inputs`,
  or a chat template's own `<|image_pad|>` run). The feature count comes from the
  vision tower plus the model's merge/merger, driven by `tgt_sizes`. Neither side
  is Ascend code, so no `--enforce-eager`, plog or device-health check can move
  this class — route by **which side is short**, then compute, don't read source.
- **The error is a host-side `ValueError`** raised in
  `vllm/model_executor/models/utils.py::_merge_multimodal_embeddings` (the
  `len(mm_embeds_flat) != is_multimodal.sum()` branch). Its deepest frame is
  upstream `vllm/model_executor/models/<model>.py::embed_input_ids`; a
  `vllm_ascend/worker/model_runner_v1.py:_preprocess` frame above it is a call
  site, and a `_310p/model_runner_310p.py` override carries no mm-merge logic of
  its own (checked at v0.23.0rc1: no `merge_multimodal_embeddings` under
  `vllm_ascend/`).
- **Deliver the ownership verdict and the arithmetic in the first message**
  (SKILL.md rule 1). Every number needed is in the paste: N, M, D, and the
  request's own image/grid sizes. Reading the model file to "see why" is what
  consumes a session — the recorded #14014 run spent its whole budget there and
  emitted no answer text.
- The same error string is recurring in the prior-month corpus — one
  `bash <skill_root>/scripts/history-titles.sh -i 'multimodal tokens'` gives the
  precedent family in one call before any code read.

## Routes

- `placeholder-count-mismatch.md` — the arithmetic of the two counts (area floor
  vs per-stage merge floor), the three levers that make them disagree, the
  direction rule, and the recorded family (#14014, #3286, #5349, #6934, #7023,
  #3092). One call: `uv run --no-project python <skill_root>/scripts/mm-placeholder-count.py`.
