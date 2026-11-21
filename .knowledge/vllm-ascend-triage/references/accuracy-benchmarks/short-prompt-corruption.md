# Corruption on a short prompt (乱码 / degenerate text without long context)

Scope: the answer comes back as word salad, a repeated token cluster, or empty
text while `usage.prompt_tokens` is a few thousand — no traceback, the process is
alive, no harness is being compared. If the prompt is 100K+ with sparse c8 / DCP /
PCP / PD disaggregation, use `garbled-long-context-output.md` instead; the two
report the same symptom on different owners, and the prompt-token count decides
which one you are in. If the prompt is *tens* of tokens on a hybrid
linear-attention model (Qwen3-Next / GDN) and the content is digit arithmetic, the
owner is that model's numeric path, not this page —
`qwen3-next-gdn-numeric-path.md`. If the prompt is a handful of tokens on a
**DeepSeek-V2-Lite** serve and the answer is *one token repeated* to
`finish_reason: length` (`"....."`), that is the MLA/`q_lora_rank: null` model shape
and #894 already refuted the stack-pairing answer — go to
`deepseek-v2-lite-mla-degeneracy.md` before this page's matrix. And it owns the
**tens-of-tokens** form on a dense model, including a text-only request to a VL-capable
checkpoint (recorded #6533: 12-token chat prompt, `Qwen3-VL-8B-Instruct`, degenerate every
request on one 910B4) — `usage.prompt_tokens` alone then excludes the vision tower, so the
VL page does not apply (Step 0b).

## Grade the symptom first

- Read `usage.prompt_tokens`, not the size of the pasted request. The recorded
  #10503 body looks enormous and is 3598 tokens.
- `finish_reason: length` with `content: null` and the garbage inside
  `reasoning` means the **model** emitted those tokens and the parser routed
  them; it is not a routing bug (`reasoning-output-routing.md` covers the routing
  end). Removing `--reasoning-parser` once shows the same tokens in `content`.
- **Two things to check in the same read before ranking any mechanism: the budget
  and the subject.** If `completion_tokens == max_tokens` the sample is a budget
  truncation, not an accuracy measurement — one resend with a large `max_tokens`
  replaces it (recorded #15939, DeepSeek-V4-Flash w4a8 1P1D: the "verification"
  curl sends `max_tokens: 128` against `你好，请做自我介绍` and the pasted body is
  `content: null`, `finish_reason: "length"`, 128 completion tokens, all of it
  `reasoning`). And if that `reasoning`/`content` answers a *different* question
  than the request — #15939's text works through formalising `∀n (n + 0 = n)`,
  which is not the prompt — stop: whose response this is decides the lane before
  any sampler or quantization theory. Ask for the request whose `id` matches
  (`chatcmpl-…`), one request at a time direct to the engine port (no
  load-balancing proxy, no concurrent traffic), and only then read the text as
  the engine's output. Do not spend an ablation ladder on a body that may belong
  to another in-flight request.
- Degenerate from the first tokens and never recovering — repeated function
  words, one token cluster looping — is corruption. Truncation, a stop-string
  mismatch or a budget limit produce short or empty output, not fluent nonsense.
- **Read the sampling value before any engine mechanism when the checkpoint is a
  reasoning distill.** Reporter-confirmed #7191 (vllm-ascend `0.11.0` / vllm
  `0.11.0`, CANN 8.3.RC1, `torch_npu` 2.7.1, TP4, `enforce_eager=True`, offline
  `LLM.generate` on DeepSeek-R1-Distill-Qwen-7B/32B): the loop
  `'\n\n,\n\n ( (\n ( or'` is what that model emits at a temperature of 0.1 or
  below, and the reporter's own resolution was *"adjust the temperature of
  samplingparams to 0.6 (anything number >0.1), and found problem solved on
  DeepSeek-R1-Distill-Qwen-7B model"* — a sampler value, no code path. On these
  checkpoints `temperature: 0` is the **reproducing** condition, not the control
  the sibling pages below use (`constant-token-degeneracy.md`,
  `repeated-run-nondeterminism.md`), so a ladder whose first rung lowers the
  temperature walks away from the fix. Rung 1: same script, same prompt,
  `SamplingParams(temperature=0.6, top_p=0.95, max_tokens=128)` — one run, and
  fluent text retires prompt templating, dtype, TP sharding and first-call state
  together, because none of them predicts a temperature-only cure. **The same lane is
  reported as "a chain of numbers"** (recorded #892, the same
  DeepSeek-R1-Distill-Qwen-7B, offline `LLM.generate`, `temperature=0.0`,
  `enable_chunked_prefill=True, max_num_batched_tokens=2048, block_size=16`): a digit
  run reads like wrong-KV corruption, which is how it gets ranked below the engine.
  **A version-onset phrase does not promote the version lane** — #7191's own body dates
  the loop "from v0.7.3 to v0.11.0" and its fix was still `temperature=0.6`, so
  "starts occurring from vllm-ascend v0.7.3" is the reporter's version history, not a
  regression datum. The pre-0.8 lanes stay *named and ranked below* rung 1: at the
  v0.7.3 pin only prefix caching refuses a non-128 `--block-size`
  (`vllm_ascend/platform.py:132-134`, so `block_size=16` survives under chunked
  prefill), and the release note ties chunked prefill/APC to CANN 8.1.RC1 while the
  matrix's v0.7.3 row wants Stable CANN 8.1.0 (`release_notes.md:11`,
  `versioning_policy.md:70`) — one flag, one package, each an ablation after the
  sampler run.
- **The pasted `SamplingParams` can be the wrong value.** #7191's body shows
  `temperature=0.6` while the symptom it reports is the low-temperature one — a
  body edited after the fix, or a snippet that does not match the run that
  produced the sample. Ask for the args actually passed plus
  `out.outputs[0].token_ids` / `.finish_reason` instead of reading the pasted
  value as effective; a degenerate run *at* the named temperature is a different
  lane. The recorded triage of that issue ranked chat-template, first-call state,
  `--dtype` and TP sharding above the sampler and offered
  `SamplingParams(temperature=0, ...)` as its cheapest check, i.e. the
  reproducing configuration.

## Step 0 — when the reporter names the trigger, test that claim against the paste

A report that names its own trigger ("any prompt containing these special tokens",
"only with a tool call") narrows the triage, but the naming is usually a
correlation and the paste often refutes it. Recorded #8590 (vllm-ascend
v0.14.0rc1 `52d4acfa`, vllm v0.14.1 `d7de043d`): Qwen2.5-1.5B, repeated 猞猁 from
the first token whenever `<|im_start|>`/`<|im_end|>` (151644/151645) are in the
prompt, "same from 0.14.0rc1 to 0.18.0rc1", "works on CPU/GPU", Qwen3-0.6B fine.

- **Same ids both ways ⇒ not input plumbing.** #8590 prints the tokenizer's ids and
  then passes the rendered *string* to `llm.generate`, so the engine re-tokenizes
  to that same list. Two runs with identical ids cannot differ because of ids; what
  varied was prompt content/length. That sentence belongs in the first paragraph —
  it converts "special tokens corrupt the engine" into "the model degenerates on a
  chat-shaped prompt".
- **A token-id-dependent branch would have to exist.** One grep decides:
  `rg -n '151644|151645|im_start|is_special|added_tokens' vllm_ascend/` is empty at
  `52d4acfa`, and that ref has no `vllm_ascend/models/` at all — the Qwen2 forward
  is upstream vLLM's — so "the downstream re-implementation mishandles those ids"
  is not available as an answer on that version.
- **A TP=1 run cannot silently zero an out-of-range id.** `AscendVocabParallelEmbedding._forward_origin`
  (`vllm_ascend/ops/vocab_parallel_embedding.py:174-184`) skips
  `_get_masked_input_and_mask` when `tp_size == 1`, so an id past the table is not
  masked to zero — it reaches the embedding call itself, where the expectation is a
  loud error rather than silent corruption. Drop the "silent zero embedding" theory
  before asking for vocab arithmetic.
- **`Using max model len N` is the checkpoint's own number.** It is printed from
  `_get_and_verify_max_len(hf_text_config, …)` (`vllm/config/model.py:1545`,
  function at `:1898`) — not from the launch command — so a value that does not
  match the public model card means the local `config.json` differs (edited,
  rope-extended, merged, or paired with another tokenizer). That is the cheapest
  non-engine explanation for degeneration that survives every release, and it costs
  one CPU run: the same directory through `transformers`, same chat template,
  `do_sample=False`. Ask for `config.json`'s `vocab_size`,
  `max_position_embeddings`, `rope_scaling` and `architectures` alongside it.
- **Version-invariance is evidence, not noise.** "Tried 0.14.0rc1 through
  0.18.0rc1, same" removes a version-specific downstream code path from the ranked
  list and promotes the checkpoint/config (and any op shared by both) — so step 5 of
  the procedure confirms the pin but no longer owns the diagnosis.
- **Coverage is not support.** At `52d4acfa` the only Qwen2.5-1.5B reference under
  `tests/` is a **pooling** checkpoint
  (`tests/e2e/singlecard/pooling/test_classification.py:10`), so this dense
  generation config has no e2e/nightly row. State it as a gap: it raises the prior
  for a model-specific defect and proves none.

## Step 0b — a text-only request on a VL-capable checkpoint (#6533)

Recorded #6533 (vllm-ascend `v0.14.0rc1` `52d4acfa` ↔ vllm `v0.14.1` `d7de043d`): one
**910B4**, `Qwen3-VL-8B-Instruct`, `POST /v1/chat/completions` with
`{"messages": [{"role": "user", "content": "你好，请介绍一下你自己"}], "max_tokens": 100}` —
no image, no `temperature`, no serve command in the report. Response: `prompt_tokens: 12`,
`completion_tokens: 100`, `finish_reason: "length"`, content degenerate from about the fifth
token (`你好，，，我是一个AI，我叫Qwen，我是Qwen，，我叫，我，我是，我，我…`). The reporter's
later answer to triage's first question ("请问你的卡是B4还是B4-1?") was **910B4**, i.e. the
32 GB A2 card, not the 64 GB `910B4-1`.

Four readings come from that body alone, before any source read:

- `prompt_tokens: 12` is the chat template's own length — no image tokens — so the vision
  tower, `--mm-encoder-tp-mode` and `USE_OPTIMIZED_MODEL` are out of the path by
  arithmetic, and the VL page's first ablation is not available here.
- The collapse starts *inside* a fluent answer and never recovers: logits degeneration, not
  truncation, not a stop-string mismatch, not a parser.
- The body carries no `temperature`, so the sampler is the OpenAI default and the two
  cheapest discriminators are one request each: the same prompt at
  `temperature: 0, top_p: 1, top_k: 1`, and a trivially different prompt on the same engine.
- The card name is a triage datum, not decoration. `910B4` = 32 GB, `910B4-1` = 64 GB (both
  are real device ids, `cann/hcomm/src/platform/inc/adapter/adapter_hal.h:47-48`; per-card
  HBM is a `npu-smi info` read — `../deploy-sizing/index.md`), and the recorded #7689 pair
  (dots.ocr, **identical** serve command: garbled on 2×910B4 32 G, correct on 910B3 64 G)
  makes the SKU a variable to state explicitly with the sibling card as the control. Ask for
  the chip name in the first answer; the mechanism stays open until a second card or a
  second model on the same box answers cleanly.

**Before ranking a kernel that the whole path shares** (rope/mrope, attention, embedding,
the sampler), read the pin's own accuracy row for that model *and* its CI wiring:
`tests/e2e/models/configs/Qwen3-VL-8B-Instruct.yaml` at `52d4acfa` declares
`hardware: "Atlas A2 Series"`, `mmmu_val acc,none 0.55`, and
`.github/workflows/nightly_test_a2.yaml:120` runs it. An image benchmark at that accuracy
exercises the same rope/attention path on the same hardware, so
"`AscendMRotaryEmbedding`/`triton_mrope` corrupts every Qwen3-VL forward" is refuted — a
surviving kernel hypothesis must name the regime difference (text-only positions vs image,
TP, dtype, batch). Recorded #6533 instead walked `AscendMRotaryEmbedding.forward_oot` →
upstream `triton_mrope` → argument-parity checks for its last four turns, after its own
answer was already available, and shipped **no message text** at all (45 turns, 1.26 M
prompt tokens, output-capped) — `../evidence-workspace/delivery-discipline.md`.

## Step 1 — test the prompt before the engine

The recorded #10503 request runs `temperature: 0`, `presence_penalty` 0,
`frequency_penalty` 0, `max_tokens` 2048, and repeats the same ~1.5K-character
news article more than ten times. Repetition collapse on a pathological prompt is
a non-engine explanation and costs one request: same deployment, a trivially
different short prompt, plus the same text with the duplicates removed. Clean
there ⇒ the kernels are fine and the trigger is prompt shape or prefix cache,
which changes the whole answer; still garbled ⇒ continue with the matrix.

## Step 2 — ablation matrix (one restart per row, same failing prompt)

| Row | Change | Corrupt here means | Clean here kills |
|---|---|---|---|
| trivially different prompt | nothing else | — | pathological-prompt collapse |
| EP off, or `--data-parallel-size 1` on one node | remove cross-node expert sharding | EP/MoE dispatch path owns it | EP as the trigger |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` (+ `HCCL_BUFFSIZE=1024`) | the project's own EP garble advice | — | "missing FLASHCOMM1" |
| `--no-enable-prefix-caching` | what the Kimi-K2.5 reference command uses (`Kimi-K2.5.md:161,181`) | KV reuse/eviction mis-loads blocks | prefix cache |
| `--enforce-eager` | ACL graph capture off | the full-graph decode path owns it | graph mode |
| `--max-model-len 32768` vs the native 262144 | shrink KV/block budget | capacity/length arithmetic | length maths |
| identical image and weights on both nodes: `md5sum config.json`, `collect_env.py` diff, weight file count | rule out a mixed deployment | one node loads different weights → fluent garbage, no error | — |

The last row is the cheap one reporters skip: with EP the experts are sharded
across nodes, so a mismatched weight set, revision or image on one node produces
exactly this symptom without any error line.

## Step 3 — what the project documents at the pinned commit

v0.20.2rc1 (`367b8e6`):

- `docs/source/tutorials/models/Kimi-K2.5.md:187` — `Kimi-K2.5-w4a8` requires at
  least 2 Atlas 800 A2 (64G × 8), so `tp8 + dp2 + dp-size-local 1` is the
  documented topology and "unsupported topology" is not the answer.
- `docs/source/tutorials/models/MiniMax-M2.5.md:449` — FAQ "What should I do if
  the output is garbled in EP mode?" → keep `--enable-expert-parallel` and set
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`. That is a **different model at the same
  topology**: quote it as a precedent row, not as the confirmed fix.
- same file `:465-467`: cross-node `tp=16` is discouraged and `tp=8, dp=2` is what
  the docs recommend for two nodes.
- `vllm_ascend/ascend_config.py:83-87` — `enable_flashcomm1` comes from
  `additional_config.enable_flashcomm1` or `VLLM_ASCEND_ENABLE_FLASHCOMM1`;
  :121-130 realigns `max_num_batched_tokens` to a `tp*pcp` multiple only when
  `pcp > 1`, so that alignment does not explain a `--max-num-batched-tokens 4096`
  deployment at `pcp = 1`.
- `docs/source/user_guide/support_matrix/supported_models.md` lists
  Kimi-K2-Thinking and no Kimi-K2.5 row at this commit while the tutorial exists:
  say "documented in the tutorial, absent from the support matrix", not
  "unsupported".

## Evidence caveats to state in the answer

- `model` in the response echoes a served-model-name (upstream
  `vllm/entrypoints/openai/chat_completion/serving.py:267` and `:1642` at vllm
  `bc150f5` / v0.20.2). When the pasted response's `model` matches neither
  `--served-model-name` nor the request, a gateway rewrote it and the response
  may not come from the pasted deployment: ask for a direct-to-port repro. That
  observation needs one grep or none — a recorded run spent 12 tool calls tracing
  the upstream response builder for it.
- A `w4a8` model directory served under a `w8a8` name, with a redacted model
  path, leaves "which weights are loaded" open. Ask both nodes for
  `config.json`'s `quantization_config`, the weight file list, and `collect_env.py`
  output.
