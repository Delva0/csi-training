# Harness vs engine: attribution checklist

## Verified facts in the pinned tree (vllm-ascend `da421afa` = v0.19.1rc1, vllm `b1388b1f`)

- The project's own ais-bench recipe is *sampled*, not greedy: `using_ais_bench.md` shows
  `generation_kwargs = dict(temperature=0.6, top_k=10, top_p=0.95, seed=None,
  repetition_penalty=1.03)` plus `pred_postprocessor=dict(type=extract_non_reasoning_content)`,
  and advises `max_out_len = 32768` with `max_out_len + input < max_model_len`.
- `tools/aisbench.py` rewrites the request config: `temperature=0.6` for `accuracy` tasks and
  `temperature=0, ignore_eos=True` for `performance`; it can set `top_p`, `top_k`, `seed`,
  `repetition_penalty`, and `chat_template_kwargs={"thinking": True}` when `thinking` is on.
- The in-repo accuracy regression is lm_eval based: `tests/e2e/models/configs/*.yaml` +
  `tests/e2e/models/test_lm_eval_correctness.py` with `RTOL = 0.05` (5% relative), tasks like
  `gsm8k` / `ceval-valid`, serve knobs incl. `enable_thinking`. Baselines are per-model yaml values, not MMLU-Pro numbers.
- `docs/source/developer_guide/evaluation/using_evalscope.md` documents only gsm8k; there is
  no evalscope MMLU-Pro baseline anywhere in the tree. So "evalscope vs ais-bench on
  MMLU-Pro" is outside the project's regression coverage — that is a legitimate finding to
  report, and it means the reporter's harness configs are the primary evidence.

## Knobs to align before calling it a defect

1. Thinking mode: on/off per side, and *how* it was switched (see `reasoning-output-routing.md`).
2. Decoding budget: `max_tokens`/`max_out_len` per harness vs the server's `max_model_len`;
   a CoT-heavy model truncated mid-thought can lose the answer entirely.
3. Sampling: temperature/top_p/top_k/repetition_penalty/seed. Greedy vs temp 0.6 changes
   results in both directions; recall that Qwen3-family guidance discourages greedy decoding
   for thinking models — cite it as a hypothesis, not as verified repo fact.
4. Prompt/answer format: official dataset prompt vs harness template; how the letter is
   extracted (first vs last `ANSWER:` match; regex on full text vs post-processed text).
5. Aggregation and coverage: micro vs macro, `--limit`, subset list, retries/failed requests
   scored as wrong.
6. Deployment: same image/model rev, TP/DP/EP, quantization, graph mode (`aclgraph` vs eager),
   `max_model_len`, batch/concurrency.

## Decisive experiments (ask the reporter, or state them as the plan)

- Rerun harness B with harness A's parameters only (one change at a time); the parameter that
  closes the gap is the cause.
- Dump raw requests and responses for both runs and compare on identical prompts: count
  `finish_reason == "length"`, empty/`null` content, missing extracted answer, and repeated
  n-grams per subset.
- Compute the floor: if k of N samples lost their answer, the aggregate damage is ~k/N — check
  whether the observed gap is explained by the measured truncation/extraction failure rate.
- Only after harness effects are excluded, look for engine-side causes (custom-op kernels,
  quantization path, graph mode), and prefer a clue that predicts a *uniform* drop.
