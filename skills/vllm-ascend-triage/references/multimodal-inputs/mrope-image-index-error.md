# `IndexError: list index out of range` inside `get_mrope_input_positions`

Recorded #5772: PaddleOCR-VL-0.9B served from `quay.io/ascend/vllm-ascend:v0.13.0rc1`
(vllm `v0.13.0` `72506c98349d`), 910B4, one card. The engine starts, and the
**first request that carries an image** kills EngineCore:

```
File ".../vllm/v1/worker/gpu_model_runner.py", line 851, in _update_states
    self._init_mrope_positions(req_state)
File ".../vllm/model_executor/models/paddleocr_vl.py", line 1127, in get_mrope_input_positions
    t, h, w = image_grid_thw[image_index]
IndexError: list index out of range
```

The API server answers `500` and then
`EngineDeadError: EngineCore encountered an issue`. No `vllm_ascend` frame is in
the stack, and the line numbers match the pinned vllm tree exactly (v0.13.0
`:1127` is that statement), so the image ships the pinned upstream.

Sibling shape, already covered: a VL request that dies with
`ValueError: Attempted to assign N = N multimodal tokens to M placeholders`
(`placeholder-count-mismatch.md`). Same family — two independent counts — but a
different stage: this one fails **before** any embedding merge, while building
M-RoPE positions for a new request.

## The two counts, and why neither is derived from the other

| Count | Derived from | Site (vllm v0.13.0) |
|---|---|---|
| images claimed by the **token stream** | `vision_start_token_id` occurrences, then `image_token_id` right after each | `paddleocr_vl.py:1097-1102` (`argwhere(tokens == vision_start_token_id)`, `image_nums = (vision_tokens == image_token_id).sum()`) |
| images with **feature data** | `mm_features` → `MultiModalFeatureSpec.gather_kwargs(features, {"image_grid_thw","video_grid_thw","second_per_grid_ts"})`, skipping any feature whose `data` is `None` | `multimodal/inputs.py:343-353`, `paddleocr_vl.py:1077-1088` |

The placeholder **length** is a third, unrelated derivation: the processor's
`_get_prompt_updates` replaces one `image_token_id` with
`[image_token_id] * get_num_image_tokens(...)` (`paddleocr_vl.py:255-290`), and
`get_num_image_tokens` re-does `smart_resize` from `patch_size` /
`spatial_merge_size` / `min_pixels` / `max_pixels` (`:147-177`). The
`<|IMAGE_START|>…<|IMAGE_END|>` wrapper therefore comes from the checkpoint's
chat template, or — when vLLM assembles the prompt itself — from
`PaddleOCRVLForConditionalGeneration.get_placeholder_str` (`:1239-1246`).

Consequence: a change in the **model directory** (`config.json` special-token
ids, `chat_template.json` / `tokenizer_config.json`, `preprocessor_config.json`)
moves one side and not the other. That is why the version of the engine is not
the variable in this class (see the resolution below).

## Do the arithmetic on what the report already carries

vLLM dumps the offending `SchedulerOutput` for V1 on a fatal error
(`dump_input.py:79`), and it holds everything: `prompt_token_ids_len`, the
`mm_features=[…]` list with each feature's `image_grid_thw` and its
`mm_position=PlaceholderRange(offset=…, length=…)`. #5772's dump:

```
prompt_token_ids_len=3577
mm_features=[MultiModalFeatureSpec(data={'image_grid_thw': tensor([1, 178, 80]), 'pixel_values': …},
             modality='image', mm_position=PlaceholderRange(offset=5, length=3560))]
```

`1*178*80 / (spatial_merge_size**2 = 4) = 3560` — the placeholder run and the one
feature agree, and `5 + 3560 + 12 = 3577`, so the prompt holds **one** image run.
A single feature and a single run means the gathered list was either empty or the
token side counted 2+; both are testable from the two facts below, and neither
needs the model file read.

## Ranked candidates and their one check

1. **The checkpoint's special-token ids make the token side count the image run.**
   `vision_start_token_id` is read from `config.json`, never checked against the
   tokenizer: if it equals `image_token_id` — or is any id the image run itself
   contains — then `vision_start_indices` selects every image token,
   `image_nums` ≈ 3560, the first loop pass succeeds and the **second** index
   raises, which is exactly what one feature plus a 3560-token run produces. One
   command on the reporter's box:
   `uv run --no-project python -c "import json;c=json.load(open('<model_dir>/config.json'));print(c.get('image_token_id'),c.get('vision_start_token_id'),c.get('video_token_id'))"`,
   then compare with the tokenizer's `<|IMAGE_START|>` id. Whichever of candidates
   1/2 it turns out to be, the **verified fix** was on this side of the line: the
   reporter's own resolution was *download the latest model files* (#5772,
   2026-01-15, `问题已解决，下载最新的模型文件即可`), i.e. a stale or partial model
   directory — which retires every engine-side explanation, this image tag
   included.
2. **The chat template the checkpoint ships is not the one the model needs.** The
   tell is one INFO line the API server prints at startup, present in both of the
   reporter's runs:
   `Detected the chat template content format to be 'openai'` (the run that
   answered the same receipt image correctly, OCR text returned) versus
   `… to be 'string'` (the run that raised). Detection is AST-only: `openai` iff
   the Jinja template loops `{% for content in message['content'] %}`;
   otherwise `string` (`chat_utils.py:400-444,552-583,585-600`). On `string` the
   image part contributes no text at all and every missing placeholder is
   **prepended** to the joined message text from
   `MODALITY_PLACEHOLDERS_MAP` (`:70-74`, `:1271-1319`), so the prompt layout is
   vLLM's fallback rather than the checkpoint's own branch. Check: that line in
   both logs; then re-serve with `--chat-template-content-format openai` (no code
   change) and repeat the same image.
3. **The gathered list was empty at runner time** — only possible when the
   processor cache is on. `MultiModalReceiverCache.get_and_update_features`
   rewrites `feature.data` from the cache in `WorkerWrapperBase._apply_mm_cache`
   (`v1/worker/worker_base.py:346-358`, `multimodal/cache.py:629-643`). Retire it
   in one line: `--mm-processor-cache-gb 0` makes `_enable_processor_cache`
   false (`cache.py:563-571`), so `worker_receiver_cache_from_config` returns
   `None` (`:802-823`) and the runner sees exactly the dumped data. #5772's serve
   line has that flag, so the mm cache is **not** the mechanism there.

## Ownership and what not to do

- Upstream, not downstream: at `v0.13.0rc1` `grep -rn 'mm_features' vllm_ascend/`
  has no hit and `grep -rn -i paddle vllm_ascend/` has no code hit — the only
  `paddle` mentions in the repo are `docs/source/user_guide/release_notes.md:89`
  and an unrelated ERNIE e2e yaml — so `--enforce-eager`, plog and device health
  cannot move this class.
- **Do not walk the mm/`MultiModalCache` internals to produce a mechanism the
  reporter's directory decides.** The recorded #5772 run read
  `paddleocr_vl.py`, `input_processor.py`, `scheduler/output.py`,
  `worker_base.py` and `multimodal/cache.py` for 20 assistant turns and 50 K
  characters of reasoning, emitted **no message text**, and was cut at the 16 384
  output-token cap with its two candidate mechanisms still unreconciled. The two
  log lines and the config ids above were the answer surface.
- Do not propose an Ascend-side patch: there is nothing downstream in the path,
  and the reporter-confirmed fix is a model-directory refresh.

## Scope and recheck conditions

Verified at vllm `v0.13.0` / vllm-ascend `v0.13.0rc1` (#5772). Line numbers move;
re-check two greps in the pinned model file before quoting them —
`def get_mrope_input_positions` (does the token-side count still come from
`vision_start_token_id`?) and `def _get_prompt_updates` (is the placeholder still
`[image_token_id] * get_num_image_tokens(...)`?). The record does not pin which
of candidates 1/2 was the defect in that directory — the config-id print and the
content-format line settle it, and the reporter's refresh fixed the class.
