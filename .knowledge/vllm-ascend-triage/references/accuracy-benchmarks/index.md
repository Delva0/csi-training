# Accuracy and score discrepancies

## Scope

Reports of the shape "score with tool A is N points below tool B / the vendor number /
another image", and reports of accuracy regressions after a code change. Covers MMLU-Pro,
gsm8k, ceval and friends as seen through evalscope, ais-bench, lm_eval and vendor blogs.
Also covers the corruption end of the same track — a prompt that comes back as 乱码 /
garbled / repeated / empty text rather than as a wrong answer — because that is decided by
the same first question (is anything being compared, or is the engine wrong per se).

One entry shape leaves this topic immediately: a number printed while the checkpoint is
being *converted* (per-layer `MSE loss`, an epoch table, a bad quantized artifact) is
converter output and precedes any engine comparison — go to
`quantized-checkpoint-conversion.md`.

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
- `short-prompt-corruption.md` — the same symptom on a few-thousand-token prompt: test the
  prompt shape first (a repeated body at temperature 0 can collapse on its own), then the
  MoE/EP/quantisation ablation rows (EP off, FLASHCOMM1, prefix caching, `--enforce-eager`,
  `max-model-len`, both nodes' weights and image), and the gateway/model-echo caveat.
  `usage.prompt_tokens`, not the size of the pasted request, decides which of the two
  corruption pages you are in.
