# Vision tower (VL) accuracy: multi-card TP, qkv padding, rope

Read this when a multimodal checkpoint's *text* output is garbage or degenerate while text-only
models on the same box are fine, when the reporter blames the ViT ("the qkv weight is all zeros",
"the vision output differs from V100"), or when a VL run is fine at TP=1 and wrong at TP>1. The
text decoder and the sampler are usually not the suspect
([sampling-and-output-quality.md](sampling-and-output-quality.md) covers those); the plugin's
vision-tower subclass is.

## First check: did the image reach the model at all?

An offline (`LLM.generate`) report whose text is garbage from an `<image>`-bearing prompt, on a
checkpoint that works for other people, is often an input-plumbing miss rather than the ViT.
At vLLM `v0.9.2` the chain is (recorded report: Llava-1.5-7b-hf, vllm-ascend
`0.9.2rc2.dev0+gb5b7e0e`, `VLLM_USE_V1=0`, one dict-prompt `generate()`):

- `vllm/inputs/parse.py:130-131` — `elif "prompt" in prompt: return ParsedTextPrompt(type="text",
  content=prompt)`. A dict prompt is accepted on the presence of `"prompt"` alone; no other key is
  validated, so a misspelled multimodal key is not an error.
- `vllm/entrypoints/llm.py:625-627` (beam path `:606-607`) copies `prompt["multi_modal_data"]` and
  nothing else into the request's multimodal kwargs.
- `vllm/model_executor/models/llava.py:681-687` returns `[]` from `get_multimodal_embeddings` when
  no image arrived, and `:695-696` merges only when `multimodal_embeddings is not None and
  len(...) != 0`, so the `<image>`/IMAGE_TOKEN id stays an ordinary token and the decoder emits
  degenerate text.

`git grep -c 'multi_model_data' <vllm-tag> -- vllm` returning 0 is the one-call check; the key must
be `multi_modal_data`. The same silence holds at `v0.11.0` and `v0.13.0` (same `elif "prompt" in
prompt` branch); `vllm/inputs/parse.py` is gone by `v0.19.2rc0`, so locate the file with
`git ls-tree -r --name-only <tag> | grep 'inputs/'` before quoting lines for a modern tag.

In the recorded thread the maintainer states the test script itself is wrong and posts a working
script using `multi_modal_data`, i.e. the image was never loaded. Label it "stated in the thread" +
"verified in vLLM's source": it is a client-side input defect, not a vllm-ascend defect, and the fix
is the key name (plus the per-prompt count via `--limit-mm-per-prompt`). Do not open the plugin's
vision patches for this shape.

## "qkv weight/bias are all zeros" under TP>1 is head-size padding, not a broken checkpoint

Recorded on `v0.8.5rc1` (Qwen2.5-VL-72B-Instruct, 8×910B1, `--tensor_parallel_size 8 --dtype
float16 --enforce-eager`): the reporter printed `qkv.weight`/`qkv.bias` inside
`AscendQwen2_5_VisionAttention.forward` (`x, _ = self.qkv(x)`) and found the tensor tails all zero,
and the replies were garbage; a 7B model on a single card was fine, 4 cards showed the same
problem. The maintainer's answer retires the checkpoint hypothesis: the `0`s are **the padding the
plugin adds for NPU performance**, and the defect is the consumer of that padded tensor, fixed by
PR #805 (`05a471001`, "bugfix for qwen2_5_vl": "the interface of qwen2.5vl changes from column
linear to qkv linear, this makes our weight pad func become abnormal, thus we optimize split_qkv
func to fix this bug").

Code shape at the reported tag (`git show v0.8.5rc1:vllm_ascend/models/qwen2_5_vl.py`):

- `MIN_PAD_SIZE = 64` / `MAX_PAD_SIZE = 128` (`:45-46`); `AscendQwen2_5_VisionAttention.__init__`
  calls the base ctor and then raises `self.hidden_size_per_attention_head` to `MAX_PAD_SIZE` when
  the real head dim falls between the two (`:70-71`, i.e. 80 → 128 for this checkpoint), so the
  padded qkv rows carry the trailing zeros the reporter saw.
- `forward` (`:73-84`) calls `self.split_qkv(x)` — the **inherited** method. At vLLM `v0.8.5`
  (`vllm/model_executor/models/qwen2_5_vl.py:258-280`) that method first
  `all_gather_interleave(qkv, self.qkv.hidden_size, self.tp_size)`, then chunks 3, re-splits the
  last dim per TP rank and views with `num_attention_heads_per_partition ×
  hidden_size_per_attention_head`: it reconstructs the *unpadded* full head layout the padded,
  pre-sharded tensor no longer has.
- The fix adds an `AscendQwen2_5_VisionAttention.split_qkv` override that chunks 3 and views
  directly with `self.num_attention_heads_per_partition, self.hidden_size_per_attention_head` —
  no TP all-gather, no re-split; the reporter then confirmed the garbage was gone on the same box.

Version boundaries:

- `git merge-base --is-ancestor 05a471001 v0.8.5rc1` → **not** in the reported tag; earliest tag
  containing it is **`v0.9.0rc1`** (`git tag --contains 05a471001 | sort -V | head -3` gives
  `v0.9.0rc1`, `v0.9.0rc2`, `v0.9.1`). Answer for a 0.8.x report: upgrade to `v0.9.0rc1`+ or carry
  the `split_qkv` override.
- The modeling file moves: `e52ebf867` (#4349, first tag `v0.12.0rc1`) deletes
  `vllm_ascend/models/qwen2_5_vl.py` and the without-padding variant, and patches
  `Qwen2_5_VisionAttention` from `vllm_ascend/patch/worker/patch_qwen2_5_vl.py` with the padding
  moved into `forward()` before FA. A modern report is checked there, and this defect's exact
  shape does not exist after the rewrite.
- Verify the padding constants at the reported ref before generalising: the class pads only when
  the head dim is in `(MIN_PAD_SIZE, MAX_PAD_SIZE)`, and both breaking branches of the inherited
  method are guarded by `if self.tp_size > 1`. The recorded "7B on one card is fine, 4 cards are
  not" is therefore a TP statement, not a card or checkpoint statement; ask for the serve line's
  TP and the checkpoint's vision head dim rather than the card model.

Boundary: the maintainer statement, the fix commit and both code texts are verified; the padded
tensor's exact arithmetic was not executed here, so present the mechanism as "the inherited
`split_qkv` consumes a padded per-partition layout" and the check as the same prompt on the fixed
tag. Do not turn "zeros at the tail of a fused qkv weight" into "corrupted checkpoint" — the
corrupted-checkpoint signature is garbage from token 1 across several releases, not a padded tail.

## Other vision-tower reports land here too

- **Flash Comm V1 is refused for VL models.** `vllm_ascend/platform.py:408-420` at `v0.17.0rc1`
  asserts `not is_vl_model(...)` when `enable_sp()` (= `VLLM_ASCEND_ENABLE_FLASHCOMM1`), with the
  message "Flash Comm V1 is not supported for VL models ... For optimal performance with VL
  models, we recommend enabling Sequence Parallelism via
  `--compilation-config '{"pass_config": {"enable_sp": true}}'"`. So the EP-mode garbled-output
  workaround in [moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md) (keep EP, set FLASHCOMM1=1) is
  unavailable for a VL checkpoint; the plugin prescribes the compilation pass instead.
- **310P vision/OCR accuracy** is the rope override, not padding:
  [310p-atlas-300i-duo.md](310p-atlas-300i-duo.md) (`AscendApplyRotaryEmb` ignoring
  `is_neox_style` through `v0.29.0.rc1`, fixed `v0.30.0rc1`).
- **Qwen3-VL/Qwen3.5-VL attention binding**: `patch/worker/patch_qwen3vl.py` rebinds
  `Qwen3Attention.forward`/`Qwen3MoeAttention.forward` to the fused
  `triton_split_qkv_rmsnorm_mrope` path; a derived model that does not inherit those classes runs
  the unfused path. Locators in
  [sampling-and-output-quality.md](sampling-and-output-quality.md).
- **Vision-tower weight lookups** (`KeyError`/prefix misses on `vpm.`/`visual.`/`llm.`) are the
  quantization/config path: [quantization-startup-keyerrors.md](quantization-startup-keyerrors.md).
- **A pre-fix tag's numeric tolerance is not a defect by itself.** The same thread notes `0.7.3rc2`
  "works" with ViT differences at 1e-4 and LM logits at 1e-2, which shows up as non-identical
  greedy (`temperature: 0`) replies across runs — that is the determinism question
  ([batch-invariance.md](batch-invariance.md)), separate from this shape's garbage.
