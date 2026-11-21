# Multimodal input assembly: placeholder count vs feature count

Scope: a VL/omni serve where the **engine rejects or mis-places the multimodal
features** — `ValueError: Attempted to assign N = N multimodal tokens to M
placeholders`, `RuntimeError: shape mismatch: value tensor of shape [N, D] cannot
be broadcast to indexing result of shape [M, D]` out of
`_merge_multimodal_embeddings`, an image in the request with no `<image_pad>`
placeholder (or the reverse), a `tgt_sizes` / `image_bound` disagreement, or a
request that dies inside `embed_input_ids` / `_preprocess` before any kernel runs —
**or** an `IndexError: list index out of range` raised while the runner builds
M-RoPE positions for a new request (`get_mrope_input_positions`,
`image_grid_thw[image_index]`), with `EngineDeadError` at the API server — **or** a
serve launched with `--mm_processor_cache_type shm` that dies on a request with
`ValueError: Data for address:id '<addr>:<id>' has been modified or is invalid.` from
upstream's `shm_object_storage.py` in every worker (`shm-mm-cache-invalid-object.md`).
Not this topic: an adapter said to cover the vision tower ("LoRA（with ViT）",
`target_modules` naming `visual.*`) — upstream filters it before any kernel
(`../lora-adapters/multimodal-vit-lora-ignored.md`); a VL answer that is merely
degenerate or wrong with sane counts
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
- **The m-rope shape counts images, not tokens, and counts them a third way.** The
  item count there is taken from `vision_start_token_id` occurrences in the final
  token ids while the feature list is gathered from `mm_features`; the run length
  comes from yet another derivation (`get_num_image_tokens`). So the owning
  variable is the **model directory** (special-token ids in `config.json`, the chat
  template, the preprocessor config), not the engine version, and the cheapest
  evidence is the API server's `Detected the chat template content format to be
  '…'` line plus the ids in `config.json` — `mrope-image-index-error.md`.
- **The error is a host-side `ValueError`/`IndexError`, not a device fault.** The
  merge shape raises in `vllm/model_executor/models/utils.py::_merge_multimodal_embeddings`
  (the `len(mm_embeds_flat) != is_multimodal.sum()` branch); the m-rope shape
  raises in the model file's `get_mrope_input_positions`, called from
  `v1/worker/gpu_model_runner.py::_init_mrope_positions` /
  `_update_states`. A `vllm_ascend/worker/model_runner_v1.py` frame above either is
  a call site, and a `_310p/model_runner_310p.py` override carries no mm-merge
  logic of its own (checked at v0.23.0rc1: no `merge_multimodal_embeddings` under
  `vllm_ascend/`).
- **Deliver the ownership verdict and the arithmetic in the first message**
  (SKILL.md rule 1). Every number needed is in the paste — for the merge shape
  N, M, D and the request's own image/grid sizes, for the m-rope shape
  `prompt_token_ids_len`, the `mm_features` count and each `mm_position`
  offset/length from the dumped `SchedulerOutput`. Reading the model file to "see
  why" is what consumes a session — the recorded #14014 run spent its whole budget
  there and emitted no answer text; #5772 did the same on the m-rope shape (20
  turns, 50 K characters of reasoning, 53 characters of text).
- **A cache-transport `ValueError` has no count arithmetic.** When the paste's leaf is
  `shm_object_storage.py::get` / `get_and_update_features`, the two counts agree by
  construction and the object was recycled: the verdict is the serve flag
  (`--mm_processor_cache_type shm`), not a placeholder/feature mismatch — do not open the
  model file for it (`shm-mm-cache-invalid-object.md`).
- The same error string is recurring in the prior-month corpus — one
  `bash <skill_root>/scripts/history-titles.sh -i 'multimodal tokens'` gives the
  precedent family in one call before any code read.

## Routes

- `placeholder-count-mismatch.md` — the arithmetic of the two counts (area floor
  vs per-stage merge floor), the three levers that make them disagree, the
  direction rule, and the recorded family (#14014, #3286, #5349, #6934, #7023,
  #3092). One call: `uv run --no-project python <skill_root>/scripts/mm-placeholder-count.py`.
- `shm-mm-cache-invalid-object.md` — the `shm` multimodal cache's reader-side staleness
  check (#5117, vllm v0.11.0 `b8b302c`): the ring-buffer free rule
  (`reader_count >= writer_count * n_readers`), why the plugin cannot own it (its only mm
  patch is `_merge_multimodal_embeddings`), and the one-flag ladder
  (`--mm_processor_cache_type` back to `lru`, then `--mm-processor-cache-gb 0`, then one
  buffer name per host).
- `mrope-image-index-error.md` — `IndexError` in `get_mrope_input_positions`
  (#5772, PaddleOCR-VL on v0.13.0rc1): the token-side image count vs the gathered
  `image_grid_thw`, the arithmetic on the dumped `SchedulerOutput`, the
  `'string'` vs `'openai'` chat-template content-format discriminator, the
  one-command special-token-id print, and the reporter-confirmed fix (refresh the
  model files).
