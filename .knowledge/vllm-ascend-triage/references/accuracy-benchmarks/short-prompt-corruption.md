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
`deepseek-v2-lite-mla-degeneracy.md` before this page's matrix.

## Grade the symptom first

- Read `usage.prompt_tokens`, not the size of the pasted request. The recorded
  #10503 body looks enormous and is 3598 tokens.
- `finish_reason: length` with `content: null` and the garbage inside
  `reasoning` means the **model** emitted those tokens and the parser routed
  them; it is not a routing bug (`reasoning-output-routing.md` covers the routing
  end). Removing `--reasoning-parser` once shows the same tokens in `content`.
- Degenerate from the first tokens and never recovering — repeated function
  words, one token cluster looping — is corruption. Truncation, a stop-string
  mismatch or a budget limit produce short or empty output, not fluent nonsense.

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
