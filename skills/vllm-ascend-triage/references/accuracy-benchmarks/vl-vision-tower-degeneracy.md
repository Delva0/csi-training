# VL degenerate output: repeated single token / `!` runs / "very slow"

Scope: a **vision** request (image or video) whose answer is a long run of one
token — `!!!!!!!!`, 感叹号, `，，，`, a single word — from the first generated
token, with no traceback and a healthy process; often reported together with
"特别耗时 / unusually slow". The first suspect is the downstream vision tower
(`vllm_ascend/models/qwen2_5_vl*.py`, `qwen2_vl.py`), not the sampler — and the
first check is one env var. Text-only prompts answered as word salad are the
`short-prompt-corruption.md` / `garbled-long-context-output.md` profile instead — and a
serve whose **every** text-only request degenerates is `short-prompt-corruption.md`
Step 0b (recorded #6533: `prompt_tokens: 12` on `Qwen3-VL-8B-Instruct`), where the token
count alone excludes the tower; only a **first-call-only** degenerate answer, or the
`is_multimodal_model` crash on a text-only launch, is `first-generate-degenerate.md`.
It also owns the **wrong-but-not-repeated** vision answer — a document/OCR run whose
TP=2 reply collapses to one trivial element (`[{"bbox":[0,0,307,307],"category":
"Picture"}]`, no text) while TP=1 returns the full box+text list. Same owner (what
the vision tower produced), different surface: the answer is present and plausible,
just poorer. Either way the presence of an **image** is what selects this page; a
**text-only** serve whose whole answer is one repeated token (`AAAA…`, `!!!!`) is
`constant-token-degeneracy.md` (word salad on a text prompt stays
`short-prompt-corruption.md`).

## Grade the symptom before theorising

- A repeated single token that never recovers is **logits degeneration**: the
  vision embedding (or the LM conditioning on it) is garbage, so every step
  argmaxes the same id. `finish_reason: length` on such an answer also explains
  the performance complaint — decode time ≈ `max_tokens` × per-step cost, all
  wasted. Cutting `max_tokens` to ~256 is the cheapest stop-loss *and* a
  discriminator: still all-`!` ⇒ not sampling, not the template.
- "`!` is token id 0 in the Qwen tokenizer" is **recalled**, not verified: this
  snapshot's `huggingface/transformers` export contains no Qwen config or
  tokenizer (gpt2 only). Give the one-line check
  (`AutoTokenizer.convert_tokens_to_ids('!')`) and label it a hypothesis; do not
  put it in the headline.
- **Open the screenshots when the report is two runs side by side.** Exported
  attachments carry an `*.ocr.txt` per image (procedure step 2), and the pair is
  the whole symptom description: recorded #7680's single-card OCR shows dozens of
  `bbox`+`text` entries while the two-card one shows a single
  `{"bbox":[0,0,307,307],"category":"Picture"}` — the layout head produced one
  coarse box and no text. That reads as *degraded ViT features*, not as a sampler
  tie or a template problem, so it points at the vision tower and at whatever the
  TP size changed. Reading it costs one call; the recorded run reasoned about
  `head_size` arithmetic for twenty turns without ever opening them.
- Same for the model's `head_dim`: read `vision_config.hidden_size` and
  `num_heads` from the reporter's `config.json` instead of quoting a number.
  Qwen2.5-VL-7B = 1280/16 → 80 is recall; it decides whether the padding branch
  below is even in play, so it must come from their file.

- **If the reporter's request is verbatim a pinned tutorial's sample request, the
  tutorial's sample response is a matched control for the input side — read it before
  ranking any mechanism.** Recorded #6909 (vllm-ascend v0.15.0rc1 `3d43ed9` ↔ vllm
  v0.13.0 `72506c9`): the paste's system message + `qwen.png` + "What is the text in
  the illustrate?" *is* the request in
  `docs/source/tutorials/models/Qwen3-VL-30B-A3B-Instruct.md:112-128` (doc form: a
  modelscope URL for `qwen.png` and `"max_completion_tokens": 100`; reporter: the same
  image via `file:///media/qwen.png`, no cap), whose sample response (`:130`) answers
  correctly in 15 tokens with `prompt_tokens: 107` — and
  the reporter's degenerate answer to the same body also says `prompt_tokens: 107`
  with `completion_tokens: 2`, `finish_reason: "stop"`, `stop_reason: null`. Equal
  `prompt_tokens` certifies that the image reached the model exactly as the validated
  recipe delivers it, which exonerates `--allowed-local-media-path`, `file:///`, the
  processor and MM-token expansion by arithmetic (no `config.json` round trip), and
  leaves the *serving difference* as the subject: diff the launch against that page's
  recipe (`:87-90` — `--tensor-parallel-size 2 --enable-expert-parallel
  --max-model-len 128000` there, versus the reporter's TP4 with neither) before
  theorising about the tower. A `prompt_tokens` that is *smaller* by the image's token
  count is the opposite verdict and sends you to the input side.

## Step 1 — the two-implementation gate (cheapest restart)

At vllm-ascend `v0.9.2rc1` (`b5b7e0e`), Qwen2.5-VL is registered **twice** and
the choice is one env var:

- `vllm_ascend/models/__init__.py:23-33` — `Qwen2_5_VLForConditionalGeneration`
  → `qwen2_5_vl:AscendQwen2_5_VLForConditionalGeneration` when
  `envs.USE_OPTIMIZED_MODEL`, else
  `qwen2_5_vl_without_padding:..._Without_Padding`.
- `vllm_ascend/envs.py:118-119` — `USE_OPTIMIZED_MODEL` defaults to **1**.

So the first ablation is one restart: `USE_OPTIMIZED_MODEL=0 vllm serve ...`.
Same failing request, image unchanged. Symptom flips ⇒ the optimized vision
tower owns it and the issue is submittable with the lines below; no change ⇒ the
padding path is exonerated and you go to rows 2-5.

## Step 2 — what the optimized path does (v0.9.2rc1, `b5b7e0e`)

`vllm_ascend/models/qwen2_5_vl.py` replaces upstream `Qwen2_5_VisionAttention`
with a head-dim-padded, unpadded-flash-attention version:

| Line | Fact |
|---|---|
| `:45-46` | `MIN_PAD_SIZE = 64`, `MAX_PAD_SIZE = 128` |
| `:70-71` / `:216-222` | head_dim in the **open** interval (64, 128) is padded up to 128; `enable_pad = True` on the transformer |
| `:227-231` | cos/sin padded by `half_pad_hidden_size_per_attention_head` |
| `:264`, `:292+` | `load_weights` rewrites `attn.qkv.weight` / `attn.qkv.bias` / `attn.proj.weight` in-place via `pad_qkv_weight`, `pad_qkv_bias`, `pad_proj_weight` |
| `:113-120` | attention call is `torch_npu._npu_flash_attention_unpad(..., scale_value=origin_head_dim**-0.5)` |

Two consequences worth stating in the answer: the padding is done **after**
weight loading on the parameter already sharded by TP, and it is keyed on
weight **names** — a checkpoint whose names miss the `attn.qkv.*` branch keeps a
zero tail. Both make this path sensitive to TP size and to non-official
checkpoints, which is exactly the precedent pattern below.

## Step 3 — ablation matrix (one restart per row, same request)

| Row | Change | Corrupt here means | Clean here kills |
|---|---|---|---|
| 1 | `USE_OPTIMIZED_MODEL=0` | optimized vision tower owns it | the whole padding theory |
| 2 | text-only request, same deployment | LM side, not the vision tower | vision tower |
| 3 | `--tensor-parallel-size 1` | TP-split/pad interaction (#976) | TP |
| 4 | `--enforce-eager` | ACL-graph replay path | graph mode |
| 5 | official HF ckpt at the documented revision | checkpoint/SFT/quant side (#562) | their weights |
| 6 | `max_tokens=256` | — | "it's sampling / it's slow" framing |

Rows 1-3 are the ones that decide ownership; all six are one flag each, so the
reporter can run the matrix in one sitting.

## The PD/EPD split: one instance answers the same body correctly

A VL/OCR deployment behind 1P1D (or EPD) whose image request comes back as **fluent but
unrelated text** is this page's wrong-but-not-repeated surface with the *topology* as the
axis. Recorded #5628: the request (served name `dotsocr`, shell working directory
`…/models/Qwen/Qwen2.5-VL-7B-Instruct`) answered `请告诉我这张图片上有些什么？` with 18
tokens of unrelated Chinese prose at HTTP 200, `prompt_tokens: 1424` (all figures read from
the screenshot OCR), no error and no `null` content. Do not start in the connector or
`kv_p2p/`: the answer is coherent text, so nothing here says the transfer failed, and the
first question is whether the defect survives without the split.

- **The pinned tree ships a 1P1D single-node recipe on Qwen2.5-VL-7B**, so "VL is not
  supported under PD" is not the answer —
  `docs/source/tutorials/pd_disaggregation_mooncake_single_node.md:1-7`, serve flags at
  `:149-154` and `:192-197`: `--no-enable-prefix-caching`, `--tensor-parallel-size 1`,
  `--seed 1024`, `--max-model-len 40000`, `--max-num-batched-tokens 40000`, and the same
  `kv_connector_extra_config` on both sides. The doc's own header says v0.11.0rc1 / vLLM
  v0.11.0, so it is the reference the report is diffed against, not the report's version.
- Rows that change the *answer* while the transport is healthy, cheapest first:
  1. **Same body against a single-instance serve** of the same checkpoint, image and
     flags. A title saying the accuracy dropped *after* switching to PD makes the no-PD
     run the reporter's own control: a cause that also predicts that run failing —
     unpinned temperature/seed, an OCR model asked a generic chat question — ranks below
     it (SKILL rule 15), it does not lead.
  2. **`usage.prompt_tokens` on one good and one bad run.** Different counts mean the two
     sides did not build the same prompt (image fetched twice, a dropped placeholder) →
     `../multimodal-inputs/index.md`, not this page.
  3. **Prefix caching**, on by default (vLLM v0.13.0 `vllm/config/cache.py:76`) while the
     1P1D recipe disables it on **both** sides. On the decode side a local hit changes
     what the prefiller is asked to send: `get_num_new_matched_tokens` returns
     `max(len(request.prompt_token_ids) - num_computed_tokens, 0)` after
     `assert num_computed_tokens % block_size == 0`
     (`vllm_ascend/distributed/mooncake_connector.py:862,864`), and
     `update_state_after_alloc` pulls only `get_unhashed_block_ids()` (`:870-884`).
     Which block set the decode side ends up owning is the open question — hypothesis,
     and `--no-enable-prefix-caching` on both sides is the one-flag launch that tests it.
  4. **Same SoC, dtype and TP on both sides**: heterogeneous P/D nodes are unsupported
     (`docs/source/developer_guide/feature_guide/disaggregated_prefill.md:101`) and the
     recipe pins TP 1. The reporter's `npu-smi` showing three engine processes cannot be
     read as "1P+1D" without both serve commands — ask for them instead of inferring TP
     from HBM.
- An OCR/document model asked a generic chat question answers prose on any topology; check
  the checkpoint's intended prompt and its `config.json` before calling it a defect. Naming
  it is a data gap when the shell directory and the served model name disagree (#5628:
  `Qwen2.5-VL-7B-Instruct` vs `dotsocr`).
- Ask for: both serve commands, `usage.prompt_tokens` of one good and one bad run, and the
  same body against a non-PD instance. Two screenshots of the same image are one
  observation — #5628's export held only the bad-result OCR under two file names.

## Precedent in the issue corpus

- `#976` — **the same file and the same padding logic**, v0.8.5rc1, 8×910B1
  Qwen2.5-VL-72B: output at `qwen2_5_vl.py/AscendQwen2_5_VisionAttention.forward`
  `x, _ = self.qkv(x)` differs completely from 8×V100, and **`qkv.weight` /
  `qkv.bias` tails are all zeros**; 7B single card is fine, 4 cards reproduce;
  `0.7.3rc2` works. Read this body — it is the mechanism, not just the symptom.
- `#131`, `#548`, `#549` — Qwen2.5-VL-**7B** answers 感叹号 while 3B is fine, and
  `0.7.3rc2` still reproduces: a model-size-dependent, not request-dependent,
  report shape.
- `#445` — 72B `tp 8 --max-model-len 32768`, output occasionally all 感叹号.
- `#562` — after LLaMA-Factory SFT, 全感叹号: checkpoint side, row 5.
- `#321` — offline inference fine, **server** inference infinite 感叹号: argues
  for request/template handling (row 2) as the alternative family.

Quote them as `#<id>: <one-line fact>` and keep "same symptom reported" separate
from "same cause confirmed" — the corpus has no maintainer resolution text.

## Upstream vs downstream (why this is a downstream-only path)

Upstream vLLM v0.9.2 (`a5dd03c`) `vllm/model_executor/models/qwen2_5_vl.py` uses
`QKVParallelLinear` with a `("attn.qkv.", "attn.q.", ...)` name map and no head-dim
padding. The downstream file re-implements the same class with padding and a
`("qkv_proj", "q_proj", ...)` map, so an upstream fix or a name change between
releases silently changes which branch loads. State the pairing (downstream
`b5b7e0e` ↔ upstream `a5dd03c`) with any claim.

## Second pin: v0.13.0rc1 moved the pad from weights to activations

Verified on #7689 (vllm-ascend v0.13.0rc1 `1b5d5abf` ↔ vllm v0.13.0 `72506c98`), and
worth quoting because it changes what the older precedent can be used for:

- `vllm_ascend/models/` **does not exist** at this pin, so the `qwen2_5_vl*.py` files
  and their `load_weights`/`pad_qkv_weight` rewrite are gone; the whole VL model is
  upstream and the only downstream piece is `AscendMMEncoderAttention`
  (`vllm_ascend/ops/mm_encoder_attention.py:31`), registered by class name at
  `vllm_ascend/utils.py:697`.
- The head-dim pad is applied to **q/k/v activations** (`:105-108`, then
  `torch_npu._npu_flash_attention_unpad` at `:124`), not to the TP-sharded weights —
  the `MIN_PAD_SIZE = 64` / `MAX_PAD_SIZE = 128` constants keep their stale
  `# … to pad weight` comment (`:27-28`). `enable_pad` still keys on
  `head_size` in the open interval (64, 128) (`:101-103`), so a Qwen2.5-VL-family
  3B/7B vision tower (1280/16 → 80) is still in the branch.
- Consequence: **#976's "multi-card ViT qkv weights are zero"** mechanism cannot be
  quoted at this pin; only its symptom shape (TP-sensitive vision path) transfers.
  Say that in the answer instead of citing the weight-pad fix.

## Third pin: v0.17.0rc1 — the ViT sharding lever is `--mm-encoder-tp-mode`

Verified on #7680 (vllm-ascend v0.17.0rc1 `e20f0b1` ↔ vllm v0.17.0 `b31e932`). At
this pin the ViT of a VL model is **weight-sharded by default**, so "TP=1 clean,
TP>1 wrong" is a config question before it is a kernel question, and the report may
paste no log at all — the two screenshots *are* the evidence.

**The lever is older than this pin — do not answer "upgrade" to a 0.13/0.14
report.** (Scope: a report whose pin is at or after the release that introduced the lever
and within an rc or two of the analysis date. A pin a month / several rcs behind is a
different question — fourth pin below.) Checked on #6525 (vllm-ascend v0.14.0rc1 `52d4acf` ↔ vllm v0.14.1
`d7de043`): `mm_encoder_tp_mode` already defaults to `"weights"`
(`vllm/config/multimodal.py:111`), the CLI flag already exists
(`vllm/engine/arg_utils.py:981`, `--mm-encoder-tp-mode`), and Qwen3-VL already opts
in with `supports_encoder_tp_data = True` (`vllm/model_executor/models/qwen3_vl.py:1246`),
so `--mm-encoder-tp-mode data` is a runnable first ablation there too. What that pin
lacks is any Ascend-side use of it: `rg -c 'mm_encoder_tp_mode' vllm-ascend/` returns
nothing (source, tests, docs), so quote the absence as coverage — never as the cause —
and call the row experimental rather than an official recipe.

| Site | Fact |
|---|---|
| `vllm/config/multimodal.py:143` | `mm_encoder_tp_mode: MMEncoderTPMode = "weights"` — the default shards the vision tower across the TP group |
| `vllm/engine/arg_utils.py:1057` | the flag is `--mm-encoder-tp-mode` (`weights` \| `data`) |
| `vllm/model_executor/models/vision.py:127` | `is_vit_use_data_parallel()` reads `mm_encoder_tp_mode == "data"` |
| `vllm/model_executor/models/dots_ocr.py:223-300` | `DotsVisionAttention`: `use_data_parallel ? tp_size=1 : TP world size`, `num_attention_heads_per_partition = heads // tp_size`, `QKVParallelLinear` / `RowParallelLinear` with `disable_tp=use_data_parallel` |
| `vllm/model_executor/models/dots_ocr.py:644` | `supports_encoder_tp_data = True` on `DotsOCRForCausalLM` — so `--mm-encoder-tp-mode data` is **accepted** for dots.ocr (the models without it get `vllm/config/model.py:571-578`'s `logger.warning_once(… Falling back to … weights)`) |
| `vllm_ascend/utils.py:652` | `AscendMMEncoderAttention` replaces `MMEncoderAttention` by class name, and is constructed with the **per-partition** head count |
| `vllm_ascend/ops/mm_encoder_attention.py:124-136` | the replacement runs `torch_npu._npu_flash_attention_unpad(..., num_heads=self.num_heads, num_kv_heads=…)` (`:124`, `num_heads` at `:130`); its only reshape gate is a head-**dim** pad (`enable_pad = 64 < head_size < 128`, `:65`) — a dots.ocr default tower (`vllm/transformers_utils/configs/dotsocr.py:14,18`: embed 1536 / 12 heads → 128) is outside it |
| `vllm_ascend/platform.py:697-704` | `--mm-encoder-attn-backend` is **reset to `None`** with a warning, so the ViT attention backend is not a lever on Ascend |

So the first ablation row — before any `config.json` request — is

```
--mm-encoder-tp-mode data      # ViT replicated per rank, LLM still TP-sharded
```

One restart, same request, same checkpoint. Clean ⇒ the weight-sharded vision path
owns it (per-rank head count 12 → 6 through `AscendMMEncoderAttention`, or the
sharded qkv/proj load), and `data` is also the workaround to offer. Corrupt ⇒ the
vision tower is exonerated and the text side becomes the suspect. Confirm the
accepted value by grepping `supports_encoder_tp_data` in
`vllm/model_executor/models/<model>.py` first: a model without it silently falls
back to `weights`, which looks like "the flag did nothing".

Ladder for this shape: (1) `--mm-encoder-tp-mode data`; (2) `--tensor-parallel-size 4`
— if a larger TP is *also* wrong, the sharded path is implicated further, if TP=4 is
clean while TP=2 is not, suspect the per-rank head count/divisibility rather than
weights; (3) TP=1 to re-confirm the reporter's own control; (4) `--enforce-eager`
(the `_npu_flash_attention_unpad` call inside an ACL graph); (5) answer-decode
difference: only the boxes are wrong, or the text inside them too — that separates
"layout head degenerated" from "everything after the tower".

**A 双机 / multi-node VL report: settle the topology before ranking causes.** "Two
nodes" has two different owners, and the pin above ships one validated recipe for each:
the tutorial `docs/source/tutorials/Qwen3-VL-235B-A22B-Instruct.md` (MP: DP2×TP8 +
`--enable-expert-parallel --async-scheduling --max-model-len 262144
--max-num-batched-tokens 4096`) and the only multi-node nightly
`tests/e2e/nightly/multi_node/config/Qwen3-VL-235B-disagg-pd.yaml` (PD Mooncake:
P DP2×TP8 / D **DP4×TP4** + `FULL_DECODE_ONLY`, `--max-model-len 8192`,
`--no-enable-prefix-caching`). The decode side's `TP4` versus the prefill side's `TP8`
is the first thing to diff when only the answer quality is wrong, and `双机` alone does
not say which one was run: make that the first question in the answer, then give one
check per branch. A repeat/loop complaint on such a serve ("循环输出", repetition
penalty has no effect) is still this page only if an image is in the request — the
two-row split is a text-only request on the same deployment (text also loops ⇒ the
vision tower is exonerated) and `finish_reason` on the looping answer.

**Coverage statement at this pin:** `rg -ln 'dots' vllm_ascend/ tests/ docs/`
returns nothing (no dots.ocr registration, doc or test), and the only 2-card VL test
in `tests/e2e/multicard/2-cards/` is `test_disaggregated_encoder.py`; the nightly
model configs (`Qwen2.5-VL-7B/32B`, `…-EPD`) are the whole VL coverage. So "no
validated multi-card recipe for this checkpoint" is citable; it is not the cause.

**Delivery note:** this shape is answered by the matrix, not by a source walk. The
recorded #7680 session had both operands (TP=1 works, TP=2 wrong) at turn 1, then
spent the 16 384-token cap deriving `DotsVisionConfig` head arithmetic and the
`mm_encoder_attn_backend` reset — and shipped **no message text** on its last turn,
so the matrix never reached the reporter. Write the verdict plus the rows above in
the first message, then check them.

## Fourth pin: v0.11.0rc0 — Qwen3-VL is a brand-new registration, and the *tag* is rung 1

Verified on #3806 (vllm-ascend v0.11.0rc0 `00ba0710`, committed 2025-09-30 ↔ vllm
v0.11.0rc3 `8ce5d319`; reported 2025-10-27): `Qwen3-VL-30B-A3B-Instruct`, video request,
`prompt_tokens: 11686`, answer = `!` repeated to `finish_reason: "length"`. Pin facts:

- **The Step 1 gate above does not exist for Qwen3-VL at this pin.**
  `vllm_ascend/models/__init__.py:11-19` registers both `Qwen3VLMoeForConditionalGeneration`
  (the reporter's 30B-A3B) and `Qwen3VLForConditionalGeneration` **unconditionally** to
  `qwen2_5_vl_without_padding`, while `USE_OPTIMIZED_MODEL`
  (`vllm_ascend/envs.py:96-97`, default 1) gates only `Qwen2_5_VLForConditionalGeneration`
  (`:21-30`). So `USE_OPTIMIZED_MODEL=0` changes nothing here — offering it as row 1 spends
  an ablation and reads as not having opened the registry.
- **`--mm-encoder-tp-mode data` is a dead lever here too**: `AscendQwen3_VisionBlock.__init__`
  (`qwen2_5_vl_without_padding.py:373-378`) builds its
  `AscendQwen2_5_VisionAttention_Without_Padding` replacement without `use_data_parallel`,
  and that subclass' `__init__` (`:76-90`) has no such parameter, so `disable_tp` stays
  false and the qkv/proj linears stay TP-sharded (upstream `qwen2_5_vl.py:269-304` keys
  them on `use_data_parallel`).
- **Two diffs that look like the mechanism and are not.** (a) The Ascend
  `_process_video_input` (`:518-533`) drops `second_per_grid_ts`; upstream also only
  carries it into `Qwen2_5_VLVideoPixelInputs`
  (`vllm/model_executor/models/qwen3_vl.py:1257,1272`) and never reads it in
  `Qwen3_VisionTransformer.forward` (`:494-505` take `x, grid_thw`). (b) The deepstack
  `torch.cat([...], dim=1)` after `unsqueeze(1)` mirrors upstream (`qwen3_vl.py:259-300`
  against the Ascend `AscendQwen3_VisionTransformer.forward`), so "the Ascend copy mangles
  the merger output" is not available.
- **Coverage at this pin** (citable as coverage, never as the cause):
  `rg -ln 'Qwen3-VL|Qwen3VL|qwen3_vl' tests/ docs/` returns only
  `docs/source/tutorials/multi_node_qwen3vl.md` (235B 双机, no video sample request); there
  is no Qwen3-VL test, and `video` appears only in
  `tests/ut/models/test_qwen2_5_vl_without_padding.py`.
- **The rung the recorded answer missed: the image tag.** `v0.11.0rc0` was committed a
  month before the report while rc2/rc4 tags were current, and the reporter's own next
  comment says the identical `curl` body on `quay.io/ascend/vllm-ascend:latest` returned a
  correct markdown table (reporter-confirmed, same hardware). A fix landed after rc0 cannot
  exist in this snapshot, so **for a degeneration report whose tag is two or more rcs older
  than the analysis date, rung 1 is "re-run the same body on the current image"** — one line
  in the first message — and the ranking below it is the mechanism if the re-run still
  degenerates. Say plainly that the pinned tree cannot answer "is it fixed".

Ladder for this shape at this pin: (1) same body on the current image; (2) the same
deployment with a **text-only** request (clean ⇒ the tower owns it); (3) an **image**
instead of a video (clean ⇒ the video branch, `temporal_patch_size=2` / `grid_thw[:,0]`);
(4) `--tensor-parallel-size 1`; (5) `--enforce-eager` **with its own banner line** in the
paste, otherwise the ablation proves nothing (SKILL rule 15).

## Recheck conditions

`MIN_PAD_SIZE` / `MAX_PAD_SIZE`, the two registrations, the `USE_OPTIMIZED_MODEL`
default, the `mm_encoder_tp_mode` default and the `is_vit_use_data_parallel()` /
`supports_encoder_tp_data` pair, and the padding call sites all move between releases. Re-grep
`MIN_PAD_SIZE|enable_pad|USE_OPTIMIZED_MODEL` at the pinned commit before quoting
any of them; on a release where the class is gone, say the path does not exist
there and ask for the version.

## What to send back

`config.json` `vision_config` (hidden_size, num_heads, depth, patch_size), the
full serve command (TP, `--max-model-len`, `--limit-mm-per-prompt`,
`USE_OPTIMIZED_MODEL`, `--enforce-eager`), one raw request+response JSON with
`usage.prompt_tokens` and `finish_reason`, and rows 1-3 of the matrix with their
outcomes. Compare that `prompt_tokens` with the model tutorial's sample response for
the same request yourself (the grading bullet above) — ask for the tutorial file only
when the report names no model whose page exists at the pin.
