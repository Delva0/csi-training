# Accuracy and score discrepancies

## Scope

Reports of the shape "score with tool A is N points below tool B / the vendor number /
another image", and reports of accuracy regressions after a code change. Covers MMLU-Pro,
gsm8k, ceval and friends as seen through evalscope, ais-bench, lm_eval and vendor blogs.
Also covers the corruption end of the same track — a prompt that comes back as 乱码 /
garbled / repeated / empty text rather than as a wrong answer — and the shortest sibling
symptom, output that is fluent but stops after tens of tokens ("被截断") — because both are
decided by the same first question (is anything being compared, or is the engine wrong per
se), then by `finish_reason`. A report that pastes only a response body — thinking text in
`content`, `reasoning`/`reasoning_content` missing, `reasoning_tokens: 0` — is also this
topic: the answer is the serve flag's presence, not a kernel.

Six entry shapes leave this topic immediately and are not accuracy questions at all:

- Wrong or alternating answers in a **PD/EPD** deployment on a hybrid
  linear-attention model (GDN/Mamba: Qwen3.5/Qwen3.6, DeepSeek-V4, GLM) — the
  suspect is the state the decode node received, not the numerics. Go to
  `../pd-disaggregation/hybrid-state-transfer.md` (connector matrix plus the
  `mamba_cache_mode` gate) and bring back its ablation rows as the deliverable.
- A number printed while the checkpoint is being *converted* (per-layer
  `MSE loss`, an epoch table, a bad quantized artifact) is converter output and
  precedes any engine comparison — go to `quantized-checkpoint-conversion.md`.
- Speculative decoding / MTP owns the symptom — `Accepted: 0` with `Drafted: N > 0`
  and `Mean acceptance length: 1.00`, a stall with drafts still being produced, or a
  first response with `content: ""` and `completion_tokens: 1` — go to
  `../spec-decode-mtp/index.md`; the ordered ablation (MTP off first) lives there.
- A **tool call** the client cannot parse (`JSON Parse error: Unterminated
  string`, `Expecting ',' delimiter`) or that arrives truncated / 格式错误 /
  重复输出: that is protocol assembly against the model's own stop, not the text
  corruption on this page — go to `../tool-call-streaming/index.md` and run its
  audit on the raw SSE first.
- A **vision** request answered as a long run of one token (`!!!!!!!!`, 感叹号,
  `，，，`) with "and it is very slow", on a VL model — that is not a score
  question and not the text-corruption profile: the downstream vision tower owns
  it, and one env var decides it. Go to `vl-vision-tower-degeneracy.md`.
- The comparison is between **embedding vectors** (cosine, `/v1/embeddings`,
  `--runner pooling`, `--task embed|score|classify`) rather than between answers:
  no text is generated, so `finish_reason` and extraction play no part and the
  engine can be wrong per span rather than per token. Go to
  `../pooling-embedding/index.md`; a crash under concurrency on the same serve is
  `../pooling-embedding/batch-index-desync.md`.

## Rules

- A corruption report whose "clean" and failing commands differ by a **scheduling**
  knob — `--block-size`, `--enable-prefix-caching`, `--mamba-cache-mode`,
  `--enforce-eager` — on a **hybrid linear-attention** model (Qwen3.5/Qwen3.6 GDN and
  siblings) is a KV-cache-geometry question, not a kernel-numerics one: on those
  models the cache block is never clamped to 128 and is only ever *raised* to match
  the mamba page size. Separate the knobs (the recorded #13853 "clean" command
  changed two) and go to `hybrid-block-size-corruption.md`. A short *digit-heavy*
  prompt on the same model family (Qwen3-Next) with **no** knob difference is the
  other hybrid shape: `qwen3-next-gdn-numeric-path.md`.
- First question: **did both numbers come from the same server, weights and serve flags?**
  If yes, the engine produced identical logits for identical prompts, so the gap lives in
  prompt template, sampling, decoding budget, output routing, extraction or aggregation — not
  in the kernel. If no, nothing is attributable until the serve flags are matched; list them
  as candidates instead of choosing a culprit.
- Compare the *pattern*, not only the mean. Bidirectional or subset-specific gaps (some
  subsets up, others down several points) are characteristic of harness-level effects
  (truncation, answer extraction, thinking output routing). A uniform drop accompanied by
  garbled/repeated/empty output is the shape of an engine or quantization defect. Say which
  pattern the report actually shows, and reject the other explicitly.
- Recompute aggregates from the reported numbers before comparing them: micro (sample
  weighted) vs macro (subset mean) can differ by a point or more on MMLU-Pro, and the two
  averages are often printed side by side. Show the arithmetic.
- No numeric claim from OCR screenshots without a cross-check (a log line, caption or
  reproducible sum). Screenshot tables lose row/label alignment; annotate the caveat.
- Prefer `finish_reason`, `content`, `reasoning` and dump-file statistics over speculation:
  the discriminating evidence is per-sample, not aggregate.
- Establish the phase before the component: a loss printed *before* the server starts is
  the quantizer's output, so attribute it to the converter (msit/msmodelslim) rather than
  to the engine, and say which of the two documented gates the reporter has not shown.

## Pages

- `harness-vs-engine-attribution.md` — knob-by-knob checklist (thinking, max tokens,
  sampling, extraction, aggregation), where the project's own baselines live, and the
  decisive experiment to run.
- `reasoning-output-routing.md` — output-routing in both directions, and the place to
  answer a pure response-contract question that pastes no score at all: (a) verified
  upstream vLLM v0.19.1 behaviour that turns truncated thinking into `content: null`,
  plus the `enable_thinking` vs `thinking` kwarg trap (which is per-parser: Kimi-K2's
  reads `thinking`, Qwen3's reads `enable_thinking` — vllm v0.23.0
  `kimi_k2_reasoning_parser.py:40-46`); (b) re-verified at vllm v0.22.1
  that "thinking inside `content` + no `reasoning` field + `reasoning_tokens: 0`" is a
  missing `--reasoning-parser` (default `""`), that the chat field is now `reasoning`
  with `reasoning_content` only a deprecated input alias, and that the Qwen3.6 tutorial
  omits the flag while seven sibling pages pass it. (c) A "thinking is in the wrong
  language / 回答中英混杂 / the answer is in English" report, in this order: routing plus
  `finish_reason` budget, then model/template thinking language (`#6848`), then a
  hand-written `--hf-overrides` rope block (`#10405`, hypothesis row), then the version
  gate before recommending `enable_thinking: false` (`#7094` / `#7061`, upstream PR 34779).
- `quantized-checkpoint-conversion.md` — per-layer loss tables and bad artifacts from
  msmodelslim conversion: the two gates (engine version plus the msit branch, which
  differs between doc pages), the published-checkpoint split test, layer-localization by
  shape invariance, and the ownership boundary (Ascend/msit, not vllm-ascend).
- `garbled-long-context-output.md` — long-input 乱码/garbage reports: grade the symptom,
  hand the reporter a feature-ablation matrix (DCP, sparse c8, dsa_cp, KV offload,
  length sweep), the `dcp_size == tp_size` / interleave constraints that make rows
  un-runnable, the tested envelope at the pinned release, and the #12905 / #14320
  precedents. When the report pastes **both** serve commands for a model the tree
  documents, its first section is the move: diff them against
  `docs/source/tutorials/models/<model>.md` (validated P and D commands, §9.1
  `Recommended Configurations`) and make that diff the opening answer — the worked
  Kimi-K2.6 v0.23.0rc1 table is there, with #13002 as the cost of skipping it —
  its five deviations are ranked rows, not proven causes.
- `vl-vision-tower-degeneracy.md` — VL image/video requests that degenerate into one
  repeated token, with the co-reported "very slow": the two-implementation gate
  (`USE_OPTIMIZED_MODEL`, v0.9.2rc1 `b5b7e0e` `models/__init__.py:23-33`), the head-dim
  padding branch (`MIN_PAD_SIZE`/`MAX_PAD_SIZE` 64/128, `enable_pad`, weight-name-keyed
  `load_weights`), the six-row ablation matrix, the #976 / #131 / #445 / #562 precedent
  family, and why "`!` is token id 0" and a recalled `head_dim` must stay labelled.
- `early-eos-short-output.md` — "only tens of tokens / 输出被截断, worked before I added a
  flag": `finish_reason` decides the lane (length = budget, stop = early stop), an
  integer `stop_reason` names the *declared* stop id that ended it (and not why the
  model emitted it), then audit every reader of the toggled key, then the shipped
  support matrix — and check the checkpoint's `generation_config.json`
  `max_new_tokens`, which upstream renames to `max_tokens` and which makes "only a
  large `max_tokens` helps" literal. Worked
  cases #12253 (v0.17.0rc1) and #12278 (v0.22.1rc1), `pass_config.enable_sp` on a
  non-VL MoE w4a8 model: the key is honored, not dropped; the matrix excludes that
  class (at v0.22.1rc1 because non-VL **and** quantized — the VL + MoE cell changed,
  and so did `sp_threshold` → `sp_min_token_num`); upstream disables the flag itself
  at TP=1; FC1 is a complementary feature, not a rename; the key also selects the
  MoE all2all backend, so "just ignore it" is not safe; plus the ablation rows.
- `hybrid-block-size-corruption.md` — 乱码 on a hybrid GDN/Mamba model that follows
  `--block-size` (or prefix caching / `--enforce-eager`): the two-knob confound in the
  reporter's own command pair, the verified clamps at v0.20.2rc1 (`refresh_block_size`
  hybrid early-return, the upward page-size override in `patch_mamba_config`, the 16:1
  physical→logical block split), the five-row matrix whose prefix-caching row decides
  align-mode state block vs attention-side split, and the stop rule that cost #13853
  its session. Recorded case #13853 (Qwen3.5-0.8B, 2048 vs 512 + eager).
- `short-prompt-corruption.md` — the same symptom on a few-thousand-token prompt: test the
  trigger claim in the paste first (identical ids on both input routes refute an id-driven
  mechanism, and `Using max model len N` is the checkpoint's own number, so a mismatch
  points at the local `config.json` before any kernel — #8590), then the
  prompt shape (a repeated body at temperature 0 can collapse on its own), then the
  MoE/EP/quantisation ablation rows (EP off, FLASHCOMM1, prefix caching, `--enforce-eager`,
  `max-model-len`, both nodes' weights and image), and the gateway/model-echo caveat.
  `usage.prompt_tokens`, not the size of the pasted request, decides which of the two
  corruption pages you are in — and a **short** prompt with a tens-of-thousands-token
  completion that repeats sentences is neither: see `long-output-repetition.md`.
- `long-output-repetition.md` — a few responses in a long accuracy run come back with
  sentences repeated ("5 条响应的句子重复"), fluent and on-topic, on a documented model
  with PD + MTP + prefix caching and a large `max_tokens`. The identity row
  (`--max-num-seqs 1`, `temperature: 0`, same request 3×) comes first, then the tutorial
  P/D diff for the decode side, the three traps in reading that diff
  (`--enable-prefix-caching` is default-on at vllm v0.23.0; a Mooncake
  `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` hit is a log line, not a generation bound; the
  block-verify gate is off by default there), and the recorded #13719 diff table.
- `qwen3-next-gdn-numeric-path.md` — the hybrid linear-attention (Gated-DeltaNet) shape:
  a 19-token to few-hundred-token prompt on Qwen3-Next-80B-A3B answered as a digit run or a
  phrase loop while ordinary chat is fine. What vllm-ascend replaces on that path at
  v0.11.0rc0 (the prefill `chunk_gated_delta_rule` swapped for an fp32 pure-torch
  re-implementation with a hard-coded x2 head expansion, while decode keeps the patched
  Triton kernel), the reporter-confirmed serve command that already rules out graph mode,
  MTP, EP and quantisation, the ablation matrix that separates model, platform and kernel
  ownership, and the #3689 / #3571 / #4057 precedents with what remains unestablished.
