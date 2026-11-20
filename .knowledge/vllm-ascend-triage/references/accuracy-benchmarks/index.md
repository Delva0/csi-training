# Accuracy and score discrepancies

## Scope

Reports of the shape "score with tool A is N points below tool B / the vendor number /
another image", and reports of accuracy regressions after a code change. Covers MMLU-Pro,
gsm8k, ceval and friends as seen through evalscope, ais-bench, lm_eval and vendor blogs.
Also covers the corruption end of the same track — a prompt that comes back as 乱码 /
garbled / repeated / empty text rather than as a wrong answer — and the shortest sibling
symptom, output that is fluent but stops after tens of tokens ("被截断") — because both are
decided by the same first question (is anything being compared, or is the engine wrong per
se), then by `finish_reason`.

Three entry shapes leave this topic immediately and are not accuracy questions at all:

- Wrong or alternating answers in a **PD/EPD** deployment on a hybrid
  linear-attention model (GDN/Mamba: Qwen3.5/Qwen3.6, DeepSeek-V4, GLM) — the
  suspect is the state the decode node received, not the numerics. Go to
  `../pd-disaggregation/hybrid-state-transfer.md` (connector matrix plus the
  `mamba_cache_mode` gate) and bring back its ablation rows as the deliverable.
- A number printed while the checkpoint is being *converted* (per-layer
  `MSE loss`, an epoch table, a bad quantized artifact) is converter output and
  precedes any engine comparison — go to `quantized-checkpoint-conversion.md`.
- A **vision** request answered as a long run of one token (`!!!!!!!!`, 感叹号,
  `，，，`) with "and it is very slow", on a VL model — that is not a score
  question and not the text-corruption profile: the downstream vision tower owns
  it, and one env var decides it. Go to `vl-vision-tower-degeneracy.md`.

## Rules

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
- `reasoning-output-routing.md` — verified upstream vLLM v0.19.1 behaviour that turns
  truncated thinking into `content: null`, plus the `enable_thinking` vs `thinking` kwarg trap.
- `quantized-checkpoint-conversion.md` — per-layer loss tables and bad artifacts from
  msmodelslim conversion: the two gates (engine version plus the msit branch, which
  differs between doc pages), the published-checkpoint split test, layer-localization by
  shape invariance, and the ownership boundary (Ascend/msit, not vllm-ascend).
- `garbled-long-context-output.md` — long-input 乱码/garbage reports: grade the symptom, hand
  the reporter a feature-ablation matrix (DCP, sparse c8, dsa_cp, KV offload, length sweep),
  the `dcp_size == tp_size` / interleave constraints that make rows un-runnable, the tested
  envelope at the pinned release, and the #12905 / #14320 precedents.
- `vl-vision-tower-degeneracy.md` — VL image/video requests that degenerate into one
  repeated token, with the co-reported "very slow": the two-implementation gate
  (`USE_OPTIMIZED_MODEL`, v0.9.2rc1 `b5b7e0e` `models/__init__.py:23-33`), the head-dim
  padding branch (`MIN_PAD_SIZE`/`MAX_PAD_SIZE` 64/128, `enable_pad`, weight-name-keyed
  `load_weights`), the six-row ablation matrix, the #976 / #131 / #445 / #562 precedent
  family, and why "`!` is token id 0" and a recalled `head_dim` must stay labelled.
- `early-eos-short-output.md` — "only tens of tokens / 输出被截断, worked before I added a
  flag": `finish_reason` decides the lane (length = budget, stop = early EOS), then audit
  every reader of the toggled key, then the shipped support matrix. Worked case #12253
  (`pass_config.enable_sp` on a non-VL MoE w4a8 model): the key is honored, not dropped;
  the matrix excludes that class; FC1 is a complementary feature, not a rename; plus the
  ablation rows and the version-flip gap.
- `short-prompt-corruption.md` — the same symptom on a few-thousand-token prompt: test the
  prompt shape first (a repeated body at temperature 0 can collapse on its own), then the
  MoE/EP/quantisation ablation rows (EP off, FLASHCOMM1, prefix caching, `--enforce-eager`,
  `max-model-len`, both nodes' weights and image), and the gateway/model-echo caveat.
  `usage.prompt_tokens`, not the size of the pasted request, decides which of the two
  corruption pages you are in.
