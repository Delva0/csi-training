# `--hf_overrides architectures` is a routing key, not a label

Verified at vllm `v0.23.0` (`0fc695f`) + vllm-ascend `v0.23.0` (`5cb98ca`);
recorded for #14817 (Atlas 800T A2 / 910B2, driver 26.1.1, vllm-ascend 0.23.0,
Qwen3-Reranker-0.6B served with `--runner pooling`). Re-checked at vllm-ascend
**v0.23.0rc1** (`f4a08bd`) + vllm v0.23.0 for #13446 (910B2, driver 26.0.0RC1,
`vllm-ascend:v0.23.0rc1` image, `--hf_overrides` copied from a *later* doc page).

## The signature

```
(APIServer pid=235) File ".../vllm/model_executor/models/qwen3_vl.py", line 883, in get_data_parser
(APIServer pid=235)     self.get_hf_config().vision_config.spatial_merge_size,
(APIServer pid=235) File ".../vllm/model_executor/models/qwen3_vl.py", line 866, in get_hf_config
(APIServer pid=235)     return self.ctx.get_hf_config(Qwen3VLConfig)
(APIServer pid=235) File ".../vllm/multimodal/processing/context.py", line 141, in get_hf_config
(APIServer pid=235) TypeError: Invalid type of HuggingFace config. Expected type:
  <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'>, but found
  type: <class 'transformers.models.qwen3.configuration_qwen3.Qwen3Config'>
(APIServer pid=235) [ERROR] ... ERR99999 UNKNOWN applicaiton exception
```

Read it as: *the model class vLLM built expects a different HF config class than
the checkpoint has*. It is a **routing** verdict, so the search space is the
architecture name in `--hf_overrides`, the checkpoint's `model_type`, and the
registry — never the card, the driver, the tokenizer or `config.json` validity.
The camera-ready tell is the process: the raise happens in the **APIServer**,
before any worker or device exists, and `ERR99999` after it is the generic
uncaught-exception tag, not an NPU error code.

The **`found type:` half is a fingerprint of the checkpoint that actually
loaded**, and it is the fastest way to separate the two ways a reporter arrives
here. `…but found type: Qwen3Config` means the weights on disk were the **text**
reranker whatever the command's model path says; found `Qwen3VLConfig` with a
text class would be the mirror case. So put the found class beside the model path
in one line: if they disagree (`Qwen/Qwen3-VL-Reranker-2B` in the command, text
class found), the directory does not hold what its name suggests or the paste
mixes two recipe rows — check `config.json`'s `model_type` at that path before
believing anything about a document. #13446 is that case: a VL model path beside
a text `--served-model-name`, and the `found` class that says text weights.

## Why the name decides the model class

There is usually no class with the name you typed. `ModelRegistry._normalize_arch`
(`vllm/model_executor/models/registry.py:1134-1158` at v0.23.0) matches the
architecture against the `*ForSequenceClassification` suffix, then rewrites the
suffix to each other default in priority order — `ForCausalLM`,
`ForConditionalGeneration`, `…Model` — and returns the first rewrite that is a
registered class:

| Architecture typed | First registered rewrite | Model class family |
|---|---|---|
| `Qwen3ForSequenceClassification` | `Qwen3ForCausalLM` (`registry.py:203`) | **text** Qwen3 |
| `Qwen3VLForSequenceClassification` | `Qwen3VLForConditionalGeneration` (`registry.py:564`) | **multimodal** Qwen3-VL |

So the suffix after `Qwen3` in the string is what selects a text model versus a
vision-language model — the checkpoint's own `model_type` plays no part in this
step. With the `…VL…` name on a text checkpoint, vLLM builds the multimodal
model, so the multimodal `Qwen3VLProcessingInfo.get_hf_config` asserts
`Qwen3VLConfig` (`qwen3_vl.py:865-866`) and `get_data_parser` dereferences
`vision_config.spatial_merge_size` (`:881-885`); the checkpoint carries
`Qwen3Config`, and the TypeError above is the result.

The `--hf_overrides` *config* side is not the discriminator and needs no change:
both names are registered in `vllm/model_executor/models/config.py:620-621`, and
`Qwen3VLForSequenceClassificationConfig` inherits
`Qwen3ForSequenceClassificationConfig` unchanged (`:511-533`), so
`classifier_from_token` + `is_original_qwen3_reranker` →
`text_config.method = "from_2_way_softmax"` happens either way. The two keys the
reporter usually suspects are innocent — leave them alone and change one string.

## Procedure

One call settles the resolution itself when the vllm checkout is present:
`bash <skill_root>/scripts/seq-cls-arch-check.sh <vllm-root> '<Arch1>,<Arch2>'`
prints, per name, the registered base class the rewrite lands on and whether that
base is text or multimodal.

1. Read the checkpoint's `model_type` (`qwen3`, `qwen3_vl`, `qwen3_moe`, …) and
   pick the family prefix the registry rewrite will land on. If the report says
   "my config.json says `model_type: qwen3`", that is confirmation the override,
   not the checkpoint, is wrong.
2. Grep the pinned tree for the authoritative pairing before believing either
   side: upstream `docs/models/pooling_models/scoring.md` lists one row per
   pooling architecture, and the repo's own examples/tests carry the string it
   actually runs (`examples/pooling/score/qwen3_reranker_online.py:26`,
   `examples/pooling/score/qwen3_reranker_offline.py:57` — both
   `Qwen3ForSequenceClassification`).
3. Check whether the reporter copied a **sibling** page. VL and text variants of
   the same model family share almost all prose, so the override line is a
   common copy-paste casualty; diff the two pages' serve commands in one call.
4. Answer with the corrected line and the one decisive observation (process =
   APIServer, expected-vs-found config classes). If the reporter cannot change
   the launch command, note that no env var or device setting substitutes for
   the architecture name.

## Telling a doc defect from a user error (two recorded cases)

The same family produced the same report twice, from opposite sides:

- **#14817** — the project's own text page carried the VL string. At v0.23.0
  (`5cb98ca`) `docs/source/tutorials/models/Qwen3-Reranker.md:141` (A2/A3 serve),
  `:156` (300I DUO serve) and `:284` (mteb offline `hf_overrides`) all said
  `Qwen3VLForSequenceClassification` on a text-reranker recipe.
- **#13446** — one release earlier the tree is consistent: at v0.23.0rc1
  (`f4a08bd`) the text page is `docs/source/tutorials/models/Qwen3_reranker.md`
  and carries the text name (`:38`, `:116`), while
  `docs/source/tutorials/models/Qwen3-VL-Reranker.md:74`, `:154` carries the VL
  name. The reporter nevertheless quoted
  `docs.vllm.ai/…/v0.24.0rc/tutorials/models/Qwen3-Reranker.html#42-source-code-installation`
  — a **hyphen**-named page with numbered sections, i.e. a generation newer than
  the pinned tree (underscore name, section titles, no `4.2`) and newer than the
  image they ran.

So the page name dates the quoted document: underscore + unnumbered sections =
the rc-era page, which is correct (the name is `Qwen3_reranker.md` at v0.22.1rc1
*and* v0.23.0rc1); hyphen + `4.2` = v0.23.0 or later, the window in which the
string was wrong. Answer in this order: (1) the corrected override;
(2) what the pinned tree shows at the reporter's *release*; (3) that the page they
read is newer than their image, so it is not evidence about their install — and if
their quoted block indeed pairs the VL name with the text checkpoint, say it is the
same doc defect recorded in #14817 and ask for the block only to date the page.
Do not spend calls looking for the newer page in the snapshot; it is not there
(`SOURCES.json` pins one commit per repo) and saying so is part of the answer.

## The worked case (#14817)

- The vllm-ascend tutorial used, on the text page, the **VL** string — three
  sites, all one serve recipe:
  `docs/source/tutorials/models/Qwen3-Reranker.md:141` (A3/A2 serve),
  `:156` (Atlas 300I DUO serve), `:284` (mteb offline `hf_overrides`).
- The same repository's own e2e job for the same weights uses the text name:
  `tests/e2e/pull_request/one_card/lora/test_qwen3_reranker_lora.py:29`
  (`"architectures": ["Qwen3ForSequenceClassification"]`, registered in
  `.github/workflows/scripts/test_config.yaml:689`) — a self-contained
  contradiction that settles the "which name is right" question without the
  reporter.
- `Qwen3VLForSequenceClassification` is the **Qwen3-VL-Reranker** recipe:
  `docs/source/tutorials/models/Qwen3-VL-Reranker.md:172,190,288` and upstream
  `docs/models/pooling_models/scoring.md:100,109`; the text row is
  `scoring.md:52,88`.
- Rendering: `docs/source/user_guide/support_matrix/supported_models.md:102`
  lists Qwen3-Reranker (A2/A3) as supported, so "unsupported model" is not the
  answer — the launch line is.

Delivery for this shape is short: corrected `--hf_overrides`, the process-location
argument, the two `path:line` contradictions, and the score-check reminder that
the tutorial's own query/document template must be used before judging accuracy.
#13446 needed six calls to reach it once the page was loaded — the recorded run
answered in one message with the registry table, both tutorial pages and the
version gap, so treat this shape as a 5-6 call case, not an investigation.

## Recheck conditions

- If a later ref registers a real `<family>ForSequenceClassification` class for
  the checkpoint in question, `_normalize_arch` returns it directly and the
  suffix table stops deciding. One grep:
  `rg -n '"<Arch>"' vllm/model_executor/models/registry.py`.
- The suffix priority order lives in `iter_architecture_defaults()` /
  `_SUFFIX_TO_DEFAULTS` (`vllm/config/model.py:1919` region); a reorder changes
  which rewrite wins.
- Related but different shapes: an architecture the registry does not know at
  all → `../startup-failures/model-type-not-recognized.md`; a `ValueError` naming
  a missing config attribute → `../startup-failures/connector-config-attributes.md`.
