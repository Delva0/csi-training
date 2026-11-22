# 310P + a VL/OCR model: it serves, it answers garbage

Verified at vllm-ascend **v0.21.0rc1** (`80610e4`) + vllm **v0.21.0**
(`ad7125a4`) while triaging #11936 (image `v0.21.0rc1-310p`, `Ascend310P3`,
CANN 9.1.0-beta.1, torch_npu 2.10.0). The `_310p/` file set and its line numbers
move between releases — re-grep, the shape stays. The v0.22.1rc1 layout of the
same subpackage is in `atlas-300i-310p.md` (same directory).

## When this page applies

A **vision-language or OCR checkpoint** on 310P / Atlas 300I serves, accepts an
image, and returns garbled, short or unrelated text: 乱码, "精度有问题", "随便跑个
图片识别就能复现". No traceback, no stall, weights load, the engine answers. A
text-only model on the same box is fine, and the report usually names a sibling
model that "seems OK" — that split is the most useful thing in the report.

Not this page: the VL request degenerates into one token repeated (`!!!!`) — that
is `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`; the box OOMs or stalls
at start — `atlas-300i-310p.md` and `../runtime-memory/attention-mask-host-allocation.md`.

## Procedure (four turns)

1. **Turn 1 — the support verdict, from the docs, not from a kernel.** Run the
   two lookups in §1 and put the verdict in the message: which of the named
   models has an Atlas-300 statement at the reported ref, and which has no page
   at all. Add the pipeline question of §2 in the same message. **When the paste or
   the thread already says the model serves and answers on the official image, the
   support verdict is one clause of context: lead with the §2/§3 mechanism and the
   two-request pair instead** (§2, first consequence).
2. **Turn 2 — the two-request ablation of §2 plus the response shape of §3.**
   The ranked list is: pipeline stage > unvalidated model/SoC pair > engine
   numerics. Deliver it; do not open `_310p/` yet.
3. **Only if the reporter comes back with "a cropped region reads fine, a text
   prompt reads fine, the whole page is nonsense"** the answer is finished —
   that is the pipeline, not the engine. A reply of "a simple image is garbage too"
   is **not** that reply and not a refutation either: ask what the simple image was
   (one cropped text region, or a whole page / an arbitrary photo). One clause decides
   between §2 and §4, so ask it before ranking the engine leg.
4. **Only if the reporter asks for the code owner** run
   `scripts/310p-rope-branch.py <config.json>` and §4's two readings. Keep it as
   a candidate with one probe, not a walk — and if the report turns out to carry a
   rope traceback rather than garbage text, switch pages
   (`310p-vl-rope-cos-nullptr-startup.md`).

## 1. Support is a per-model doc lookup at this ref

- **The 310P page validates text-only Qwen3 only**:
  `docs/source/tutorials/hardwares/310p.md:96-172` lists
  `Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310-vllm` (TP1/TP1/TP4, W8A8SC, `--dtype
  float16 --quantization ascend --max_model_len 16384|20480`). No VL row.
- **The support matrix column reads `A2/A3` for every VL row it has**
  (`docs/source/user_guide/support_matrix/supported_models.md`): `PaddleOCR-VL
  🔵 A2/A3`, `MiniCPM-V 🔵 A2/A3`, `Gemma3 🔵 A2/A3`, while `GLM-4V` is `❌`
  (#2260). So the matrix never claims a VL model on Atlas 300.
- **Atlas 300 support for a VL model lives in the model's own tutorial page.**
  `docs/source/tutorials/models/PaddleOCR-VL.md:51` "The Atlas 300 inference
  products are supported from version 0.15.0rc1", `:60` "…single-node
  single-card deployment on the 910B4 and Atlas 300 inference products", and its
  own tab (`:93-115`) is exactly the command a reporter copies: `--max_model_len
  16384 --no-enable-prefix-caching --mm-processor-cache-gb 0 --enforce-eager
  --dtype float16` (`:115` explains that `--max_model_len` is there to keep the
  mask allocation working).
- **At v0.21.0rc1 there is no GLM-OCR and no MinerU page**:
  `ls docs/source/tutorials/models/ | grep -i -E 'glm-ocr|mineru'` is empty,
  while `paddleocr_vl.py`, `glm_ocr.py` and `qwen2_vl.py` all exist upstream in
  vllm v0.21.0 — the engine recognises the architectures; **vllm-ascend-on-310P
  has no validated statement for them**. Say that, and say it is a documented
  state, not a diagnosis of the reporter's host (`atlas-300i-310p.md`, rule 2 of
  the topic index).

So "the model that behaved is the one with an Atlas-300 tab, and the two that did
not have no page at all" reproduces the reporter's own split with two greps.

## 2. Document parsing is a pipeline; the VLM alone is one stage

The repo says it in the OCR page: `PaddleOCR-VL.md:202` — "Typically, we also
need to integrate the PP-DocLayoutV2 model to fully unleash the capabilities of
the PaddleOCR-VL model, making it more consistent with the examples provided by
the official PaddlePaddle documentation" — with the combined
layout→recognise pipeline at `:268-280` (`pipeline = PaddleOCRVL(...)`, `:275`) and, for this SoC,
`:263` ("The Atlas 300
inference products support only the OM model inference. For details … see the
guide provided in ModelZoo … PP-DocLayoutV2"): on Atlas 300 the layout stage is
not even a vLLM process.

Consequence: a whole-page image POSTed straight to `vllm serve` exercises the
*element recognition* stage only, on a page it was never given the regions for.
Short, repeated or unrelated text is the expected answer, not a numeric defect.

This is also where a thread on #11936 split, and the split is the part worth keeping.
A commenter (`zyz111222`, 2026-07-15) answered exactly as above — "these two models are
not officially supported yet … GLM-OCR needs to be used with PP-DocLayout-V3" — and the
**reporter** (`Big2Wheel`, minutes later) pushed back on it: both models **do** launch and
infer on the official `v0.21.0rc1-310p` image with no extra operations, 只是输入简单的图片识别
都是输出乱码. Two consequences for the answer:

- **The support verdict of §1 is context, never the resolution.** Closing on "not
  validated / not supported" is the reply this reporter had already rejected, and it
  answers a question they did not ask.
- **"A simple image is garbage" does not confirm the pipeline on its own**, because a
  plain picture that is not a document region is outside the model's trained input too.
  Only the cropped-region vs whole-page pair below separates the two — and if the
  reporter's "simple image" was already one cropped text region, go straight to §4 and
  say which reading you are taking.

Two requests the reporter can run decide it:
1. one **cropped** text region (or one line) as the image;
2. the **whole page** as before.

Cropped reads correctly + whole page nonsense ⇒ pipeline, close the report. Both
garbage ⇒ go to §3/§4.

## 3. Separate "wrong text" from "not enough text" first

Ask for `finish_reason`, `usage.completion_tokens` and 3-5 raw responses (`curl`
body, not a paraphrase):

- a handful of tokens then `stop`, on a **documented** model → EOS / chat
  template, `../accuracy-benchmarks/early-eos-short-output.md`;
- long, fluent, wrong text → the corruption track,
  `../accuracy-benchmarks/index.md`;
- one token repeated (`!!!!`, 感叹号) and "very slow" → vision-tower
  degeneracy, `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`.

A **text-only prompt on the same serve** is the one-call ablation: if plain text
answers correctly, the owner is the vision→text path below and not the sampler.

## 4. If a code owner is still wanted: the 310P-only components

Every VL model in this family is routed through ops that vllm-ascend replaces
**only on 310P** — `vllm_ascend/utils.py:744-759` (`RotaryEmbedding →
AscendRotaryEmbedding310`, `MRotaryEmbedding → AscendMRotaryEmbedding310`,
`MMEncoderAttention → AscendMMEncoderAttention310`, plus the `_310p/` RMSNorm /
FusedMoE / Conv3d set). Which rope implementation runs is decided by the
**checkpoint config**, never by the model family:

- `uses_mrope` ⇔ `mrope_section` inside the text config's `rope_parameters`
  (vllm `vllm/transformers_utils/config.py:508-517`). If it holds, the layers are
  `AscendMRotaryEmbedding310`, whose cos/sin are a **per-forward global** built by
  the runner only when `uses_mrope`
  (`_310p/model_runner_310p.py:627-651` →
  `_310p/ops/rotary_embedding.py:68-118`, 234-247), and whose `forward_oot` raises
  a plain `RuntimeError` when they were not built (`:196-199`).
- Otherwise the layers are `AscendRotaryEmbedding310._rope_forward_oot`
  (`_310p/ops/rotary_embedding.py:121-181`) with three branches:
  (a) `head_size == 128` and `cos_sin_cache.shape[-1] == 128` (i.e. full rotary
  with head 128 — the cache carries `rotary_dim` values, vllm
  `vllm/model_executor/layers/rotary_embedding/base.py:83-92`), `:138`; and
  (b) partial rotary with `rotary_dim == 64`, `:142`, both call
  `npu_apply_rotary_pos_emb(..., cos, sin)` with the globals from
  `get_cos_and_sin_slice()`; (c) everything else calls
  `torch_npu._npu_rotary_embedding(positions, …, self.cos_sin_cache, …)` and
  needs no globals.
- **Do not infer the branch from the model family.** `paddleocr_vl.py` *defines*
  `get_mrope_input_positions` (vllm v0.21.0 `:1097-1135`) while `uses_mrope` is
  still decided only by `mrope_section` in the text config
  (`vllm/transformers_utils/config.py:511-517`), and `glm_ocr.py` carries
  `rope_parameters={"partial_rotary_factor": 0.5}` with no mrope method at all
  (`:287-291`). Read the checkpoint's `config.json` — `scripts/310p-rope-branch.py`
  does exactly that — instead of grouping models by family.
- **The trap in branches (a)/(b):** those globals are allocated only for **non-VL**
  models — `set_cos_and_sin` skips a VL config (`has_rope` is never even consulted
  for a multimodal config, `vllm_ascend/ops/rotary_embedding.py:62-86`) and
  `update_cos_sin` returns early while `_cos is None` (`:129-146`). On 310P a VL
  model in branch (a) or (b) therefore hands the op `cos=None, sin=None`.

  **One gate, two outcomes — and the paste's symptom says which one you are in,
  so no probe is needed to choose between them.** If the engine serves and answers
  wrong text (this page's shape), the op tolerated `cos=None` on the reporter's
  CANN pair, every position is wrong, and "garbled but fluent" is the expected
  symptom. If instead the paste carries a `profile_run` traceback ending
  `call aclnnApplyRotaryPosEmbV2 failed, error code is 161001` +
  `Check cos != nullptr failed`, that is **not** evidence for branch (c) and rope is
  not exonerated: the op refused the `None` and the engine died — that report is
  `310p-vl-rope-cos-nullptr-startup.md` (recorded #13522, v0.23.0rc1, MiniCPM-V-4.6,
  branch (b) at `:158`). Keep the rope reading a candidate here only as long as the
  serve is known to come up.

Verified dead ends — each cost a recorded session turns and none is the split:

- **`scale_value`.** `AscendMMEncoderAttention310` recomputes
  `self.scale_value = self.head_size**-0.5` (`_310p/ops/mm_encoder_attention.py:66`)
  and passes it to `_npu_flash_attention_unpad` (`:124-131`), ignoring the `scale`
  argument. Every OCR/VL model here passes exactly `head_dim**-0.5` (vllm v0.21.0
  `glm_ocr.py:142`, `paddleocr_vl.py:593`, `qwen2_vl.py:320`), so for this family
  the recompute is a no-op; it diverges only for a model that passes a different
  scale.
- **The head-dim pad path.** `64 < head_size < 128` zero-pads q/k/v to 128 and
  slices the output back (`_310p/ops/mm_encoder_attention.py:24-25,65,113-120,135-136`).
  Zero-padded k/v add `q·0 = 0` to the scores, and `context_layer[..., :origin]`
  restores the shape — layout preserved, not an accuracy owner.
- **`ApplyRotaryEmb` is not in the 310P override set** (`utils.py:705` has it for
  the A2 path, `:744-759` does not replace it), but its Ascend implementation is
  `torch_npu.npu_rotary_mul` with full-dim cos/sin
  (`vllm_ascend/ops/rotary_embedding.py:557-590`) — SoC-agnostic. Absence from the
  override set is not evidence of a defect.
- **Memory/shape questions** (`--max-model-len`, the `O(max_model_len²)` fp16 NZ
  mask, `hardwares/310p.md:71-88`) explain OOM and stalls, not wrong text.

## Recorded case (#11936)

`v0.21.0rc1-310p` image, one 310P3 box, CANN 9.1.0-beta.1: GLM-OCR and
MinerU2.5-2509-1.2B "可以正常拉起，但是精度有问题 / 随便跑个图片识别就能复现",
PaddleOCR-VL "好像没问题" — and MinerU2.5 appears in **both** lists in the same
paragraph, so ask which line the reporter means before ranking anything (a
title/body mismatch is common in this family). The paste carries a full
environment dump and only the PaddleOCR-VL serve command (verbatim from the tutorial's Atlas-300 tab), no
response body and no log. Later in the same thread a commenter (`zyz111222`) answered
"these two models are not officially supported yet … GLM-OCR needs to be used with
PP-DocLayout-V3", and the **reporter rejected that framing**: both models do launch and
infer on the official image with no extra operations, 只是输入简单的图片识别都是输出乱码. So the
thread closes the **launch** half only; the artefact stays open. Nothing changed in the
engine and the two-request ablation of §2 was never returned, so §2 remains the leading
hypothesis and §4 the open alternative — do not record this case as resolved by the
support verdict, and do not quote the commenter's line as a maintainer's decision.

The session that produced **no answer** spent 28 assistant turns and its whole
16 K-token output budget inside `_310p/` rope and attention internals with zero
message text, after a mid-session guess that the failing pair is "the models on
the M-RoPE / partial-rotary branches" — a guess this page partly refutes in §4
(`../evidence-workspace/delivery-discipline.md`). For this report shape the
deliverable at turn 3 is the support verdict plus the two-request ablation; the
rope branch is a single probe at turn 4+, never a walk.
