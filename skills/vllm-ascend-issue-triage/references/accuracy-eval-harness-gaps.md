# Score gaps between eval harnesses, or against the published number

Report shape: "harness A scores N points below harness B" / "our MMLU-Pro is 5-6 points under the
model card", with no request that visibly fails. This is a **measurement** question first: the
serving stack may be identical in both runs, and the difference then lives in the prompt the
harness builds, the sampling it sends, or the way it extracts an answer. No failing run means no
traceback to pin, so the finding is the diff between the two harness configurations plus the one
test that separates them. Verified from the harness sources named below; the runtime numbers in
these reports stay unverified here.

## Diff five axes before opening any plugin source

1. **Prompt template and system prompt.** Each benchmark ships its own template. For evalscope the
   authority is one JSON per benchmark:
   `https://raw.githubusercontent.com/modelscope/evalscope/main/evalscope/benchmarks/_meta/<benchmark>.json`
   (read it raw; `_meta/mmlu_pro.json` last modified 2026-10-30). It holds everything that changes
   the score: `few_shot_num`, `subset_list`, `prompt_template`, `few_shot_prompt_template`,
   `system_prompt`, `metrics`, `aggregation`. MMLU-Pro there is **5-shot CoT over 14 subjects,
   10 options, few-shot examples taken from the task's own validation split, and the answer must
   end in a line `ANSWER: [LETTER]`**. A second harness that runs 0-shot (`few_shot_num: 0`) or
   without that answer-format instruction is not measuring the same task, and MMLU-Pro's
   `ANSWER: [LETTER]` requirement is the extraction contract, not decoration.
   For ais-bench, do not go repo-hunting: ask the reporter for the case YAML they ran
   (`configs/datasets/<bench>/…`: few-shot/CoT switch, prompt template, `max_out_len`). The two
   configs side by side answer most of these reports.
2. **Whether a chat template is applied, and with which kwargs.** `lm_eval --model vllm` with
   `--apply_chat_template --fewshot_as_multiturn` sends templated multi-turn chats; without them it
   sends one raw completion prompt. Same model, different input string, different score. On a
   reasoning model this axis also carries `enable_thinking` / `reasoning_effort`, which outrank a
   serve-line default — see [deepseek-v4-flash.md](deepseek-v4-flash.md) for the DeepSeek-V4
   bridge and the per-request override.
3. **Sampling defaults** (`temperature`, `top_p`, `max_tokens`/`max_out_len`) each harness picks
   when the config does not say.
4. **Answer extraction / post-processing.** A harness that strips reasoning content before scoring
   (`extract_non_reasoning_content` and friends) versus one that scores the raw completion; a
   harness whose regex needs `ANSWER: B` versus `B`.
5. **Truncation.** 5-shot CoT MMLU-Pro prompts are long (the per-subject examples push them into
   the thousands of tokens) and the answer sits in the *last* line. A per-request `max_tokens`, a
   `max_out_len`, or a server `--max-model-len` shorter than prompt+reasoning cuts the `ANSWER:`
   line and scores the item wrong. Ask for the share of `finish_reason: length` in the low run —
   that count, not the mean, is the evidence.

## The one test that separates harness from serving

Send the same 10-20 items through **both** harness prompt templates against the same endpoint and
compare per-prompt outputs. If the model text for a given prompt is identical both ways, the score
delta is prompt/extraction, and no plugin source read will change it. If the model text differs,
the harnesses are sending different requests (thinking on/off, different system prompt, different
`max_tokens`) — fix that axis and re-score. A score that is stable across harnesses but below the
published table then moves to model/quant/checkpoint questions.

## When the thread has no maintainer root cause

A stale/`not_planned` close, or a close that lists candidate causes ("the `/chat/completions`
interface and system-prompt differences, the default sampling parameters, and the evaluation
metrics"), is a checklist, not a diagnosis. Answer with the two configs, the axis that differs and
the decisive test above, labelled **candidate needing a runtime check**; do not manufacture a
plugin defect from it. The same discipline applies to a closure that contradicts the reporter's own
numbers ([moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md) has the recorded instance).

## Related classes

- A score gap that appears only with `data_parallel_size` / expert parallel:
  [moe-dp-ep-accuracy.md](moe-dp-ep-accuracy.md).
- A score gap after a version bump with `temperature > 0`, or `<pad>`/id-0 output:
  [sampling-and-output-quality.md](sampling-and-output-quality.md).
- "Same input, different score at higher concurrency":
  [batch-invariance.md](batch-invariance.md).
