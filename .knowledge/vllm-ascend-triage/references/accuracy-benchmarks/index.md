# Accuracy and score discrepancies

## Scope

Reports of the shape "score with tool A is N points below tool B / the vendor number /
another image", and reports of accuracy regressions after a code change. Covers MMLU-Pro,
gsm8k, ceval and friends as seen through evalscope, ais-bench, lm_eval and vendor blogs.

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

## Pages

- `harness-vs-engine-attribution.md` — knob-by-knob checklist (thinking, max tokens,
  sampling, extraction, aggregation), where the project's own baselines live, and the
  decisive experiment to run.
- `reasoning-output-routing.md` — verified upstream vLLM v0.19.1 behaviour that turns
  truncated thinking into `content: null`, plus the `enable_thinking` vs `thinking` kwarg trap.
