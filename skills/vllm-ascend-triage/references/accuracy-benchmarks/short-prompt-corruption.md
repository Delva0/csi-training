# Corruption on a short prompt (乱码 / degenerate text without long context)

Scope: the answer comes back as word salad, a repeated token cluster, or empty
text while `usage.prompt_tokens` is a few thousand — no traceback, the process is
alive, no harness is being compared. If the prompt is 100K+ with sparse c8 / DCP /
PCP / PD disaggregation, use `garbled-long-context-output.md` instead; the two
report the same symptom on different owners, and the prompt-token count decides
which one you are in. If the prompt is *tens* of tokens on a hybrid
linear-attention model (Qwen3-Next / GDN) and the content is digit arithmetic, the
owner is that model's numeric path, not this page —
`qwen3-next-gdn-numeric-path.md`.

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
