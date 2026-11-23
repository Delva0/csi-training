# `/rerank` serves, and every relevance score is plausible but meaningless

Verified at vllm v0.11.0 (`b8b302cd`) + vllm-ascend v0.11.0rc2 (`a2e4c3fe`); recorded for
#4616 (910B 32G, official image, `Qwen3-Reranker-4B` served with
`--hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token":
["no", "yes"], "is_original_qwen3_reranker": true}'` and no `--task` / `--runner`). It is a
**silent** wrong-score report: HTTP 200, no traceback, six documents, scores 0.807-0.889, and
the exact-match document ranks fourth behind an unrelated one.

## The arithmetic that decides it

`RerankUsage.total_tokens` is **not** a billed aggregate — it is the summed prompt length over
the scored pairs: `num_prompt_tokens += len(final_res.prompt_token_ids)` and
`usage=RerankUsage(total_tokens=num_prompt_tokens)` (`vllm/entrypoints/openai/serving_score.py:462-468,479`).
So `total_tokens / <number of documents>` is the tokens the model actually saw for one pair,
and that number is the whole case. Recorded #4616: `88 / 6 ≈ 15` tokens per pair for Chinese
query + document pairs — no instruction, no delimiter, no role block.

`get_score_prompt` has exactly three branches at this pin (`vllm/entrypoints/score_utils.py:170-203`):

| Branch | Condition | What the pair becomes |
|---|---|---|
| score template | the built model class implements `SupportsScoreTemplate` | `model.get_score_template(query, document)` — the checkpoint's own Judge prompt |
| pad-token pair | `model_config.use_pad_token` | `tokenizer(text=prompt_1, text_pair=prompt_2)` |
| "llm as reranker" | otherwise | `full_prompt = prompt_1 + prompt_2`, i.e. **raw concatenation** |

At v0.11.0 only one model in the tree implements the protocol —
`vllm/model_executor/models/jina_vl.py:77,116`; `rg -n 'get_score_template' vllm/model_executor/models/`
returns `interfaces.py` plus that file. **No Qwen3-family class does**, and the adapter-generated
sequence-classification class inherits nothing of the kind. So a Qwen3-Reranker `/rerank` call at
this pin scores an untemplated pair whichever of the two fallback branches runs, and a relevance
score from it carries no relevance. The `--hf_overrides` keys are *not* the problem here: the
two-way softmax head is built from them at this pin (`vllm/model_executor/models/config.py:202-209`,
`adapters.py:411-441`), which is why the scores are finite, ordered and plausible instead of
raising. This shape differs from `seq-cls-arch-routing.md`, where a wrong architecture string
kills the engine at startup.

## Procedure (five calls, then answer)

1. Divide: `usage.total_tokens / len(results)`. Compare it with the checkpoint's own instruction
   template — the Qwen3-Reranker Judge block plus `<Query>`/`<Document>` is many tens of tokens,
   so a handful of tokens per pair means the template was never applied.
2. Confirm the pin: `rg -n 'get_score_template|SupportsScoreTemplate' vllm/` in the vllm tree the
   paste's release pairs with. A hit list naming only `interfaces.py` and one VL model settles
   it; do not read `adapters.py` line by line to re-derive it.
3. Check whether the release documents a `/rerank` route at all —
   `rg -n -i 'rerank|v1/score' docs/source tests/ --glob '!locale/**'`. At v0.11.0rc2 the answer
   is no chapter: the project's own Qwen3 reranker route is the **embedding** endpoint with the
   instruction prefix built client-side (`docs/source/tutorials/single_npu_qwen3_embedding.md:3,63-64,86`,
   `get_detailed_instruct` = `f'Instruct: {task}\nQuery:{query}'`, documents embedded plain and
   scored by dot product).
4. Answer with the route, in this order: (a) the response's own tokens/pair, (b) that no class on
   this pin applies a score template to this family, (c) the two ways forward — the documented
   embedding+instruct route on the current release, or a release whose tree implements
   `SupportsScoreTemplate` for the family, (d) the acceptance test that needs no baseline: after
   the fix the exact-match document must outrank the unrelated ones and the band 0.80-0.89 must
   separate. Say explicitly that no NPU, precision, quant or `--enforce-eager` row applies — the
   model never saw the prompt the checkpoint was trained to score.
5. Ask for one dumped request/response pair only if the tokens/pair lands near the template
   length; that divergence reopens the pad-token/`use_pad_token` question, nothing else.

Recorded cost of skipping this: #4616's session read the OCR-free paste, opened
`qwen3.py` / `adapters.py` / `serving_score.py` / `pooler.py` / `registry.py` for ~25 calls, had
the `88`-token arithmetic in hand at its own response body and still shipped **no message text**
inside the 16 384-token output budget. A silent scoring report is answered from the response
body plus two greps.

## Recheck conditions

- A later vllm ref where `rg -n 'get_score_template' vllm/model_executor/models/` lists a
  Qwen3-family module: then the template branch runs and this shape is answered by the
  checkpoint's own Judge prompt, not by the route.
- `use_pad_token` decides between the two fallback branches (`vllm/config/model.py:1456-1459`,
  set per-architecture in `adapters.py:426-427`); it changes the input text, never the fact that
  the instruction is missing.
- Requests that already carry the Judge prompt as `query` text (the tutorial's instructed form)
  are *templated*: their tokens/pair is large and this page does not apply — then the ladder is
  `index.md` Rule 1's numeric classes.
