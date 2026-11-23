# `KeyError: '<modality>'` in the model's `get_input_embeddings` (V0 placeholder maps)

Shape: an audio/omni model (Ultravox, Qwen2-Audio) whose serve or first request dies
with `KeyError: 'audio'` raised inside the **model file's** `get_input_embeddings`, on
`attn_metadata.multi_modal_placeholder_index_maps["audio"]` — typically once per rank
right after `Loading model weights took …`, i.e. inside memory profiling, not on a
request. Recorded #1087 (vllm-ascend v0.8.5rc1 `ec27af3` + vllm v0.8.5, TP8 910B3,
`vllm serve …ultravox-v0_5-llama-3_1-8b`); the frame was legible only in the
screenshot OCR's **last** block.

## Rule 1 — the two branches are the discriminator

Verified at vllm v0.8.5 (`ba41cc9`):

- V0 branch, `vllm/model_executor/models/ultravox.py:565-569`:
  `if not envs.VLLM_USE_V1: … attn_metadata.multi_modal_placeholder_index_maps["audio"]`.
  The subscript is a bare `[` — an empty dict is a `KeyError`, not a fallback.
- V1 branch, same file `:570-575`: `merge_multimodal_embeddings(input_ids, …,
  _AUDIO_PLACEHOLDER_TOKEN)` — no map, and it cannot raise this key.

So the first check is the engine lane, not the model code: the log's own
`npu is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.` (vllm-ascend
v0.8.5rc1) or the missing V1 engine banner says V0. On v0.9.0rc2 the same serve comes
up and answers text requests (reporter-confirmed, #1087) — the V1 default removes this
subscript entirely.

## Rule 2 — the dict is empty because *that* seq group had no placeholders

The map is per-request plumbing, not a feature count:

| stage | owner | anchor (vllm v0.8.5 / vllm-ascend v0.8.5rc1) |
|---|---|---|
| placeholders → per-modality map | upstream `MultiModalPlaceholderMap.from_seq_group` | `vllm/multimodal/base.py:100-101` — `if not seq_mm_data or not seq_mm_placeholders: return MultiModalKwargs({}), {}` |
| map → per-seq-group input | Ascend V0 runner | `vllm_ascend/worker/model_runner.py:784-799` (`inter_data.multi_modal_placeholder_maps`) |
| map → attention metadata | Ascend metadata builder | `vllm_ascend/attention/attention.py:629-641` (dict comprehension over `self.multimodal_placeholder_maps`) |
| map read by the model | upstream model file | `vllm/model_executor/models/ultravox.py:569` |

An absent modality key therefore means *this batch carried no placeholder list for
that modality*. Two common instances: the **dummy profile batch**
(`vllm_ascend/worker/model_runner.py:1067-1155` `profile_run` → `prepare_model_input`,
the stage the recorded traceback shows: `determine_num_available_blocks` →
`profile_run` → `execute_model`) and a **decode step**, where the builder appends
nothing for an empty positions range (`vllm/multimodal/base.py:111-118`). It is
neither a placeholder/feature count mismatch (`placeholder-count-mismatch.md` — that
raises `ValueError`/`RuntimeError` from `_merge_multimodal_embeddings` *with* numbers)
nor a device fault, so no plog, `--enforce-eager` or health check moves it.

## Rule 3 — report coverage, then the two-stage ladder

- Coverage at that pin: `docs/source/user_guide/supported_models.md:32` is
  `| Ultravox |  |Need test|` — a **coverage gap, not a ❌**: audio as a modality is
  listed ✅ for Qwen2-Audio (`:11`), so "audio unsupported" is wrong and "validated
  for Ultravox" is also wrong.
- Ladder: (1) same command on `VLLM_USE_V1=1` / the newer rc that defaults to V1;
  (2) a text request, which should answer; (3) then the **audio** request — on an
  image whose vllm was installed without the `[audio]` extra
  (`vllm-ascend/Dockerfile:43` at this pin: `pip install -v -e /vllm-workspace/vllm/`,
  no extra) it dies one stage later with
  `AssertionError: PlaceholderModule should not be used when the original module can be
  imported` from `vllm/multimodal/audio.py::load_bytes → librosa.load`. Say so up
  front: fixing the deploy does not make audio work on that image
  (`../image-dependency-faults/index.md` rule 1, and §"PlaceholderModule" there).

## Answer shape

Verdict line: the crash is real and is the **V0** VLM placeholder-map path, not the
audio encoder; `KeyError: 'audio'` states its own remedy as "no placeholder list in
this batch" and the V1 lane does not read the map. Then: the one grep that proves the
lane, the coverage row, the three-rung ladder, and the request for the two artefacts
that settle the dummy-batch variant if (1) does not fix it — the full log around
`init engine` (to see `profile_run`) and `python -c "import librosa, soundfile"`
inside the image.
