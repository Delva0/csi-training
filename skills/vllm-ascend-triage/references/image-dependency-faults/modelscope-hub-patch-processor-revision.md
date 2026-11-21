# `VLLM_USE_MODELSCOPE` patch breaks the one lookup vLLM does not route through it

Scope: a serve by **hub id** on an image that sets `VLLM_USE_MODELSCOPE`, failing
after the weights have loaded with the hub's own exception:

```
INFO ... [model_runner.py:902] Starting to load model Qwen/Qwen2.5-VL-7B-Instruct...
INFO ... Loading model weights took 18.4429 GB
ERROR ... [engine.py:400] The model: Qwen/Qwen2.5-VL-7B-Instruct has no revision: main !
  … modelscope/utils/hf_util.py:247 in from_pretrained
  … modelscope/utils/hf_util.py:172 in get_model_dir
  … modelscope/hub/snapshot_download.py:109/257
  … modelscope/hub/api.py:600 in get_valid_revision_detail
modelscope.hub.errors.NotExistError: The model: … has no revision: main !
```

The paste is full of `Downloading Model to directory: /root/.cache/modelscope/hub/<id>`
lines and `modelscope - INFO - Target directory already exists`, i.e. ModelScope
is already the backend — that, plus the phase, is the classification. Lane B,
**memory-profiling** window (`worker.determine_num_available_blocks` →
`model_runner.profile_run` → upstream `multimodal/profiling.py`
`_get_dummy_mm_inputs` → `info.get_hf_processor`), so the checkpoint, dtype, card
and every serve flag are exonerated: the *processor* is what could not be fetched.

## Why only the processor leg fails (verified at vllm `ed6e9075d31e`/v0.7.3)

- `vllm/transformers_utils/__init__.py:3-19` — when `VLLM_USE_MODELSCOPE` is on,
  vllm imports modelscope (gate: `modelscope>1.18.0`, `:11-14`) and calls
  `modelscope.utils.hf_util.patch_hub()` at `:19` **before any transformers
  import**, i.e. it globally redirects HF-hub lookups to ModelScope.
- vllm's **own** downloads are ModelScope-aware and pass `revision` through, so
  config and weights resolve on ModelScope's default branch —
  `transformers_utils/tokenizer.py:147-160` (`snapshot_download(..., revision=revision)`),
  `transformers_utils/config.py:46` (`from modelscope import AutoConfig`),
  `model_executor/model_loader/loader.py:227-240`. That is why the failure looks
  late: 5/5 shards already loaded.
- The **processor** leg is not: `transformers_utils/processor.py:53-69`
  (`get_processor` → plain `AutoProcessor`/`ProcessorMixin.from_pretrained`),
  `:101-111` (`cached_processor_from_config` passes `model_config.model` and no
  revision), `:118-131` (`get_image_processor`), and
  `transformers_utils/config.py:686-693` returns `{}` for the image-processor
  config with the comment "ModelScope does not provide an interface for
  image_processor". So the call lands in modelscope's `hf_util` wrapper with
  `revision=None`, and that wrapper mints the **HF** default branch name `main`,
  which a ModelScope repo (branch `master`) does not have.
- The pinned tutorial reproduces it, so this is not reporter error:
  `docs/source/tutorials/single_npu_multimodal.md:41` `export
  VLLM_USE_MODELSCOPE=True`, `:152` the same in `docker -e`, `:168`
  `vllm serve Qwen/Qwen2.5-VL-7B-Instruct …`.

Gap: `modelscope` is **not** in the case export (`/workspace/repos/SOURCES.json`),
so the last step — which modelscope build substitutes `main` for `None` — is read
off the pasted frames, not source. Do not name a modelscope fix release as fact.

## Ladder

1. **Serve the snapshot directory the paste already printed** (it exists: the
   weights loaded from it). No hub call, no revision:
   `vllm serve /root/.cache/modelscope/hub/<owner>/<model> <same flags>`.
   The rest of the command is unchanged; `VLLM_USE_MODELSCOPE`
   may stay set.
2. **Pre-download to a known dir, then serve that dir**: `modelscope download
   --model <id> --local_dir /models/<name>` (older CLIs: `uv run --no-project
   python -c "from modelscope import snapshot_download as d; print(d('<id>'))"`).
3. **With HuggingFace reachable**: unset `VLLM_USE_MODELSCOPE` (or `False`) and keep
   the hub id.
4. **Only if 1 also fails**: the snapshot dir may be incomplete for a *processor*
   (`ls <dir>` for `preprocessor_config.json`, `processor_config.json`,
   `video_preprocessor_config.json`, chat template) — that is a different error
   than the one pasted; and report `pip show modelscope transformers`.

## Not the cause

- `--revision main|master`: it cannot reach this leg —
  `cached_processor_from_config` passes no revision, and the value modelscope
  rejects is minted inside modelscope.
- `--max-model-len` / KV arguments: the raise precedes KV sizing.
- vllm-ascend code: no `vllm_ascend` frame appears anywhere in the stack.
- The same paste's `ERROR … Failed to import vllm_ascend_C` /
  `WARNING:root:Warning: Failed to register custom ops, all custom ops will
  be disabled` / `Methods … not implemented in NPUWorker`: separate, pre-existing
  noise at that tag; they do not raise.

## Recorded cost

#829 (`quay.io/ascend/vllm-ascend:v0.7.3`, Atlas 800I A2): the modelscope frames
and the cache-directory line were in the first paste, yet the session spent
~49 K characters of reasoning before answering and its single message was **cut
mid-sentence inside the mechanism** — the ladder above never shipped. Ship rung 1
plus "the path is printed in your own log" in the first paragraph, and keep the
mechanism table behind it (SKILL.md answer shape).

Recheck conditions: anchors are vllm v0.7.3's. Later refs re-grep
`patch_hub|VLLM_USE_MODELSCOPE` in `vllm/transformers_utils/` (the processor
loader has changed file names) and re-check whether `get_processor` finally
forwards a revision — if it does, the class narrows to image processors only.
