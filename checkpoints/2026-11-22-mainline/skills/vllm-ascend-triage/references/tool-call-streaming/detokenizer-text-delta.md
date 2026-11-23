# Damaged streamed *text*: the incremental detokenizer's two paths

## When this page applies

A `stream: true` response whose **text** is wrong — not only the `arguments`:
characters missing at the start of a value or a sentence, a fragment repeated a
few characters later, the damage clustered where the script changes (ASCII ↔
Chinese ↔ arrows/emoji), while the same body with `stream: false` is clean, the
deployment healthy, nothing in the log raising. Entry shapes recorded in this
family: "流式时 json 格式异常，缺失首 `[`、文本重复、转义符异常，多出现在 ascii
⬅➡ 中文字符交界处" (#14656) and 乱码/重复 in prose that disappears without
streaming.

This is the owner behind `index.md` rule 1's outcome D1. Reach it
when the assembled `arguments` carries *missing leading characters plus a repeated
fragment*: the parser's own final `args_so_far` is then valid, so the schema-type
gate does not explain it, and outcome D's "the text itself is wrong" is not
generation either — the text the tool parser consumed was already damaged one
layer below it.

## Two paths, one symptom — pin the reporter's first

At vllm v0.23.0 (`0fc695fc6d1d`) `vllm/v1/engine/detokenizer.py` chooses per
request (`:49-66`):

- **Fast path** — `FastIncrementalDetokenizer` (`:167-247`) when
  `USE_FAST_DETOKENIZER` and the tokenizer is a `PreTrainedTokenizerFast`.
  `USE_FAST_DETOKENIZER` is `tokenizers >= 0.22.0` (`:24`), while vLLM's own
  requirement floor is `tokenizers >= 0.21.1`
  (`vllm/requirements/common.txt:11`), so *whether this path is installed depends
  on the image*. It streams through `tokenizers.decoders.DecodeStream`, primed
  with `ids=request.prompt_token_ids` (`:184-187`). When the library raises
  `Invalid prefix encountered` — its own "non-monotonic, invalid UTF-8 output"
  error, referenced in the comment to upstream vllm #17448 — `_protected_step`
  (`:223-247`) logs
  `WARNING … Encountered invalid prefix detokenization error for request <id>, resetting decode stream`
  (`:235-243`) and rebuilds `DecodeStream(skip_special_tokens=…)` **without** the
  prompt ids, then steps only the current token (`:243-245`). The text already
  emitted stays in `output_text`, the in-flight bytes of a split character are
  gone, and the re-stepped token can re-emit characters the stream had already
  produced: lost head plus repeated fragment, on the same request.
- **Slow path** — `SlowIncrementalDetokenizer` (`:250-305`) otherwise. Its
  `detokenize_incrementally` (`vllm/tokenizers/detokenizer_utils.py:110-202`)
  re-decodes the last tokens and returns `new_text[len(prefix_text):]` (`:202`)
  guarded only by `len(new_text) <= len(prefix_text)` and a trailing `�` (`:194`).
  The docstring says why the lookback exists — decode "cleanup algorithms …
  decide to add a space or not depending on the surrounding ids" (`:129-130`,
  `INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5` at `:56`). Because the decoded
  form of a fixed token window is not stable while neighbours arrive, a **length**
  slice both drops insertions and re-sends text the client already has. Note the
  file moved: before this tree the same function lived under
  `vllm/transformers_utils/`.

Neither path is Ascend code: `rg -n -i 'detokeniz' vllm_ascend/` is empty at
v0.23.0, and the chat frontend, the reasoning parser and every tool parser
(`glm4_moe`, `glm47_moe`, …) all consume the same `output.text`, so a detokenizer
fault damages **prose and arguments together**.

## Checks, cheapest first

1. **Grep the reporter's server log** for
   `Encountered invalid prefix detokenization error` /
   `resetting decode stream`. Present → the fast-path reset fired and owns the
   damaged chunk, per request, with no fix in the Ascend tree. Absent → the path
   is unproven; go to 2. One log request, and it is the only decisive artefact
   this family has.
2. **Does `content` carry the same damage at the same place?** Ask for one failing
   request's whole SSE (assistant `content` and `arguments` side by side). Identical
   corruption in both → this page. Damage confined to `arguments` → the parser's
   delta (`arg-type-gate-unquoted-value.md`, `_compute_args_diff`).
3. **Which path is installed** — `python -c "import tokenizers;
   print(tokenizers.__version__)"` in the serving image plus `ls <model-dir>/tokenizer.json`.
   `>= 0.22.0` and a fast tokenizer → fast path; anything else → slow path with the
   `:202` slice.
4. **`return_token_ids: true` on the identical body** (rule 4 of `index.md`) and
   decode the returned ids with the same checkpoint tokenizer: ids decode to the
   correct text → the serving text path lost or repeated it; ids reproduce the
   damage → the tokens themselves are wrong, which is the decoding/spec-decode
   lane, not this one.
5. `--tool-call-parser`, `--reasoning-parser` and `--async-scheduling` are **not** on
   this path. Do not spend ablations on them for a text-delta fault.

## What the answer owes

- The classification first: damaged **text delta** (upstream vLLM serving) vs
  parser delta vs decoding, and the one check that separates them (the log grep,
  the `content`-vs-`arguments` comparison).
- The workaround that needs no release, because this lane has no server-side
  switch: validate and repair the assembled arguments client-side — a JSON/JSONL
  check that re-quotes and re-joins the fragment (the #14656 reporter's own fix:
  "通过手动添加jsonl格式校验与补充，可以缓解") — or send `stream: false`.
- Owner and scope: upstream vLLM; fast path gated on `tokenizers >= 0.22.0`; the
  Ascend tree owns neither branch, so a patch proposal belongs upstream, and the
  fix shape is "compare the re-decoded prefix instead of its length" /
  "re-prime the reset `DecodeStream` with the request's ids".
- The reset warning's presence in the log, quoted, with the request id if present.

## Recorded case

**#14656** vllm-ascend v0.23.0 / v0.23.0rc1 images (`-a3`), GLM-5.2-w4a8c8 on
910C 560T, 4P/8D PD with `MooncakeConnectorV1` + `AscendStoreConnector`
(memcache), `--tool-call-parser glm47`, `--reasoning-parser glm45`, MTP
`deepseek_mtp` `num_speculative_tokens` 1 on P / 3 on D, `--max-model-len 524400`,
`--enable-auto-tool-choice`; **no `--async-scheduling` on either side**. Symptom as
filed: streaming only, `arguments` missing the leading `[`, a repeated fragment,
wrong escapes, "出现于字符变化、ascii/中文交界处". The recorded run verified the
parser side at the pin (`_is_string_type` already fixed, see
`arg-type-gate-unquoted-value.md`), named this text-delta path as the alternate
owner, never grepped for the reset warning, and **shipped no answer**
(`../evidence-workspace/delivery-discipline.md`). The reporter's later statements:
the serve commands above, client-side jsonl validation as the mitigation, and "新版本
parser 重构，暂未发现问题" — a later parser refactor removed the recurrence, so on
the versions they ran the parser family stays in scope while the actionable workaround
is the client-side check.
