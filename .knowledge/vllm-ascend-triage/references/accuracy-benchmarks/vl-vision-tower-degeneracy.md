# VL degenerate output: repeated single token / `!` runs / "very slow"

Scope: a **vision** request (image or video) whose answer is a long run of one
token — `!!!!!!!!`, 感叹号, `，，，`, a single word — from the first generated
token, with no traceback and a healthy process; often reported together with
"特别耗时 / unusually slow". The first suspect is the downstream vision tower
(`vllm_ascend/models/qwen2_5_vl*.py`, `qwen2_vl.py`), not the sampler — and the
first check is one env var. Text-only prompts answered as garbage are the
`short-prompt-corruption.md` / `garbled-long-context-output.md` profile instead —
unless only the **first** call degenerates or the run is a text-only launch of a
VL-capable checkpoint, which is `first-generate-degenerate.md`.
It also owns the **wrong-but-not-repeated** vision answer — a document/OCR run whose
TP=2 reply collapses to one trivial element (`[{"bbox":[0,0,307,307],"category":
"Picture"}]`, no text) while TP=1 returns the full box+text list. Same owner (what
the vision tower produced), different surface: the answer is present and plausible,
just poorer. Either way the presence of an **image** is what selects this page; the
`AAAA…`/`!!!!` text-only profiles are `short-prompt-corruption.md`.

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
report.** Checked on #6525 (vllm-ascend v0.14.0rc1 `52d4acf` ↔ vllm v0.14.1
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
`usage.prompt_tokens` and `finish_reason` (to prove the image tokens reached the
model at all), and rows 1-3 of the matrix with their outcomes.
