# `Attempted to assign N = N multimodal tokens to M placeholders`

Recorded #14014: MiniCPM-V 4.6, vllm-ascend `v0.23.0rc1` (`f4a08bddd0cc`), vllm
`v0.23.0` (`0fc695fc6d1d`), CANN 9.1.0-beta.1, `310P3`, 16 running requests.
The first multimodal step killed the engine:

```
RuntimeError: shape mismatch: value tensor of shape [186, 1024] cannot be
  broadcast to indexing result of shape [196, 1024]
ValueError: Attempted to assign 186 = 186 multimodal tokens to 196 placeholders
```

Frames: `Worker.execute_model` → `vllm_ascend/worker/model_runner_v1.py:2208
execute_model` → `:1363 _preprocess` → `super()._preprocess`
(`vllm/v1/worker/gpu_model_runner.py:3447`) → `vllm/model_executor/models/
minicpmv4_6.py:1204 embed_input_ids` → `models/utils.py:471-506
_merge_multimodal_embeddings`. No Ascend kernel is in the stack: this is input
assembly, and the two numbers are the whole evidence.

## What each number is

| Number | Produced by | Site (vllm v0.23.0) |
|---|---|---|
| `M` placeholders | the prompt builder. For MiniCPM-V the **transformers image processor's** own formula, inserted per image while the text is split on the literal `(<image>./</image>)` pattern | `vllm/transformers_utils/processors/minicpmv.py:227-238` (`self.image_processor.get_slice_image_placeholder(image_sizes[index][i])`) |
| `N` features | the vision tower + the model's merge, from `tgt_sizes` in the batch | `vllm/model_executor/models/minicpmv4_6.py:1109-1185` (`_process_vision_input`), merge at `:835-890` |
| `D` (`1024`) | the text hidden size the merge writes into | `utils.py:488-492` |

The two formulas, as vLLM mirrors them:

- placeholder side: `source_tokens = W*H // (patch*patch*token_divisor)`, total
  `source_tokens + grids[0]*grids[1]*patch_tokens`, `token_divisor = 4` for
  `downsample_mode="4x"` and `16` for `"16x"`
  (`minicpmv4_6.py:467-537`, `get_num_image_tokens` at `:583-597`) — an **area
  floor** over the resized pixel tile;
- feature side: `MiniCPMV4_6DownsampleMLP` reshapes `(h // m1, m1, w // m2, m2)`
  and permutes, once per stage (`self.times`), so the rows per tile are
  `Π (h_i // 2)(w_i // 2)` — a **per-stage per-dimension floor** over the patch
  grid, with `merge_kernel_size = (2, 2)` by default (`:804-846, :856-890`).

Do not turn that difference into the diagnosis too quickly: a grid that the merge
stages cannot divide raises a `reshape` error of its own, it does not silently
produce a smaller count. At equal geometry the two formulas agree, so a mismatch
means the two passes **disagreed about geometry or mode**, not that one rounds
differently. Compute both for the reporter's tile sizes before theorising —
`uv run --no-project python <skill_root>/scripts/mm-placeholder-count.py --tile 392x392 --mode 16x`
(and `--mode 4x`, `--placeholders M`, `--features N`), which prints the two counts
per tile and which side of the reported pair is short.

## The three levers that make them disagree

1. **Per-request processor kwargs reach only one pass.** The placeholder call
   passes the image size and nothing else (`vllm/transformers_utils/processors/minicpmv.py:235`), so
   `max_slice_nums` / `downsample_mode` / `use_vit_merger` supplied through
   `mm_processor_kwargs` change the vision pass while the prompt keeps the image
   processor's defaults → `mode` mismatch, and `4x` against a `16x` prompt is a
   4× feature surplus. Check the request's `mm_processor_kwargs` first.
2. **`use_vit_merger` false** (or `downsample_mode="4x"`) skips the merger
   (`:1088`, `:1109-1118`: `downsample_mode = "16x" if use_vit_merger else "4x"`),
   so the same image yields `4×` the tokens the `16x` placeholder text promised.
3. **The placeholder text did not come from this processor.** A chat template that
   emits its own `<|image_pad|>` run (transformers ≥5.7 chat templates do; see
   `MiniCPMV4_6ProcessingInfo.image_pattern = "<|image_pad|>"`, `:419-421`), a
   client that pasted a placeholder string, or a `--limit-mm-per-prompt` /
   truncation that dropped one image's run. This is the shape the precedents below
   mostly belong to, and it shows as one number being `1` or an order of magnitude
   off, not as a 10-token gap.

## Procedure

1. Direction first, from the two counts alone: `M > N` = the prompt promised more
   placeholders than the vision path produced (levers 1–3, or one image's features
   missing); `M < N` = the prompt lost placeholders (template/truncation).
   Both numbers are in the paste; say which side is short in the first message.
2. `uv run --no-project python <skill_root>/scripts/mm-placeholder-count.py --tile <HxW> …` on the
   image sizes from the request; compare with the reported `M`/`N`.
3. One history call: `bash <skill_root>/scripts/history-titles.sh -i
   'multimodal tokens'` (add `-b` only if the titles are not enough).
4. Then, and only then, ask for the processed request rather than reading the
   model file: per image `image_size`, `tgt_sizes`, the placeholder string as
   tokenized, and the `mm_processor_kwargs` actually used (`--mm-processor-kwargs`,
   any `downsample_mode` in the request). If those three agree with the formula on
   both sides, the divergence is in the *counts at merge time* and the per-tile
   table from the script is what a maintainer needs.
5. Do **not** propose an Ascend-side patch for this class. `vllm_ascend` has no
   merge implementation (checked: no `merge_multimodal_embeddings`, no `minicpm`
   module besides `tests/e2e/pull_request/one_card/test_minicpm.py`), so the fix
   is a placeholder/processor change upstream or a `mm_processor_kwargs` /
   template change on the reporter's side.

## Recorded family

- #14014 — the case above; **unresolved in the record** (the run that carried it
  was cut before an answer). The discriminating input is the per-image counts.
- #3286 — `<image_pad>` placeholder statistics in `input_ids` reported wrong:
  the placeholder builder itself, before any merge.
- #6934 — kimi-k2.5, `4225 = 4225 multimodal tokens to 1 placeholder`;
  #7023 — Qwen3.5, `65 = 65 multimodal tokens to 3905 placeholders`. Both are
  template/parse failures (lever 3), and they are the shape to expect on a
  release whose chat template changed.
- #5349 — MiniCPM-V-4.5 would not run at all; #3092 — `v0.10.2rc1-310p` on a
  300I Pro card. On a 310P serve, check the SoC capability question
  (`../soc-platform-paths/index.md`, `310p-vl-ocr-garbage-output.md`) in the same
  message as the count verdict — a model the pinned image does not support for
  that SoC does not need a count analysis.

## Scope and what is open

Verified at vllm `0.23.0` / vllm-ascend `0.23.0rc1` in the `#14014` snapshot; line
numbers move between releases, the two formula families and the
`mm_processor_kwargs` asymmetry do not. Not established: why #14014's pair is
`196`/`186` — with `16x` and equal geometry both formulas agree, so either the
request used a non-default `max_slice_nums`/mode on one pass or the placeholder
string did not come from the image processor. That is the open question the
per-image artefact (image size, `tgt_sizes`, tokenized placeholder string,
`mm_processor_kwargs`) settles.
