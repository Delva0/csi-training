# `num_speculative_tokens >= 3` switches the Ascend acceptance path (block verify)

## When this page applies

A spec-decode (MTP / `deepseek_mtp` / eagle) report in which **the reported number
is 3 or more and 1 works**, and the symptom is exact-form output — a tool call's
JSON, code, ids, numbers — rather than general fluency: truncated arguments,
"格式错误", an occasional duplicated span, a stop token before the JSON closes.
This page supplies the branch, what it does, and the two ablations that confirm
it. It is *not* the audit itself: run
`streamed-arguments-audit.md` first and bring its outcome here.

## The branch (source-verified at vllm-ascend `e20f0b1`, v0.17.0rc1)

| Fact | Location |
|---|---|
| `# When num_speculative_tokens>=3, using block verify.` then `using_block_verify = max_spec_len >= 3` | `vllm_ascend/sample/rejection_sampler.py:113-114` |
| `max_spec_len = max(self.num_draft_tokens)`, i.e. the request's `num_speculative_tokens` | `vllm/v1/spec_decode/metadata.py:27`; passed in as `metadata.max_spec_len` from `vllm/v1/sample/rejection_sampler.py:144` |
| n = 1, 2 → the strict per-token kernels; n >= 3 → the Triton kernel (`rejection_random_sample_block_verify_kernel`, `vllm_ascend/ops/triton/reject_sample.py:366`, dispatched in the `else` of `if not using_block_verify:` at `sample/rejection_sampler.py:228-246`, comment `# MagicMTP: Improving acceptance rate with Block Verify.` at `:229`) or its pytorch twin `rejection_random_sample_block_verify_pytorch` (`sample/rejection_sampler.py:696-768`); the strict path is the `if` at `:193` | as cited |
| No config key or env var selects it: `rg -n 'block_verify' vllm_ascend/` returns only the sampler and the kernel/test files, so the only switches are `num_speculative_tokens` and temperature | verified |
| Greedy rows are untouched by it: `non_greedy_mask = (~is_greedy)[:, None]` gates accept/reject/bonus, and greedy rows are filled by `rejection_greedy_sample_with_triton` in the `if not sampling_metadata.all_random:` branch | `sample/rejection_sampler.py:130-146` (greedy kernels), `:754-762` (`non_greedy_mask`) |

## What block verify does (and why it can shorten a JSON)

From `rejection_random_sample_block_verify_pytorch`:

- per position it computes `pi = target_prob(draft_token) / draft_prob(draft_token)`,
  clamps at 1, and takes the **cumulative product over the whole draft block**;
  `uniform` is likewise a cumulative product; `legal_mask = (draft_prob > 0) &
  (pi_cumprod >= uniform_cumprod) & valid_mask`;
- `last_accept_pos` is the **last** legal position, and then *every* position
  `<= last_accept_pos` is accepted — including positions whose own per-token test
  failed. A recovered token is written at `last_accept_pos + 1`, and the bonus
  token at `num_draft_per_batch` when the whole block was accepted.

Two consequences that matter for a report:

1. **The emitted token is no longer the target model's sample.** Fluent text
   survives a rare wrong token; JSON, ids and code do not — one wrongly accepted
   token inside a string literal is enough for `Unterminated string`, and one
   wrongly accepted token can also make the model end the tool call early (the
   next step predicts from a context the target would never have produced), which
   is the "arguments stop before the closing `"}` while `finish_reason:
   tool_calls` follows" shape.
2. **The tail can be silently emptied.** Upstream `RejectionSampler.parse_output`
   keeps only slots with `id != PLACEHOLDER_TOKEN_ID and id < vocab_size`
   (`vllm/v1/sample/rejection_sampler.py:218-252` at v0.17.0) and compacts the
   row, so any slot the Ascend kernel leaves unfilled or out of vocabulary
   disappears from `new_token_ids` **without an error or a log line**.

## Test coverage — why this path is under-validated

- `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py:176`
  `test_rejection_sampler_block_verify_triton_kernel` compares the Ascend kernel
  against `rejection_random_sample_block_verify_pytorch` — its own relaxed twin.
- The strict test in the same file compares the Ascend **strict** kernel against
  upstream `rejection_random_sample_kernel` for `max_spec_len` 1, 2 and 3.
- No test compares block verify against strict rejection, so "n >= 3 is covered by
  tests" is true only for kernel-vs-twin equivalence, not for distribution
  equivalence. Say that with the file:line, and do not present it as proof of a
  bug either — a relaxed path is the documented intent of the option.

## The two ablations that confirm it

1. `num_speculative_tokens: 2`, everything else identical. Same `method:
   deepseek_mtp`; must behave like 1. This is the threshold test.
2. `temperature: 0` with `num_speculative_tokens: 3`. Greedy rows bypass block
   verify entirely, so the difference between this run and the failing one is the
   acceptance path and nothing else in the decode step.

Then, only if both ablations agree with the branch, `stream: false` plus
`return_token_ids: true` to show whether the *tokens* also stop short (they must,
if the mechanism is acceptance) — see `streamed-arguments-audit.md` step 3B.

## Confidence labels to use in the answer

- source-verified: the threshold, the cumulative-product acceptance, the recovered
  and bonus token slots, the greedy exemption, the `parse_output` filter, the
  missing equivalence test;
- hypothesis (unless the reporter runs them): that *this* report is block verify
  rather than a model-side early stop, a client accumulation bug, or an unrelated
  graph/async-scheduling effect;
- refuted-by-evidence: if the `n=2` and `temperature=0` ablations fail, the branch
  is not the owner — return to the audit page's layers and stop reading the
  kernel.

## What was recorded, and the limits

#7580 (vllm-ascend v0.17.0rc1, A3, 大EP, GLM-5 + `deepseek_mtp`, `n=3` works →
`n=1` 正常, `JSON Parse error: Unterminated string`): the run reached
`sample/rejection_sampler.py:114` on its tenth turn and then spent its remaining
nine turns on the Triton kernel, the pytorch twin and the test files, ending with
the answer cut inside its evidence table and no ablation plan shipped. The reporter's later reproduction then showed the assembled
arguments `{"filePath":"/…/K8sYamlAgent.java` — a missing tail, not wrong
characters — so the branch is the right *lead* but the mechanism that this paste
supports is "the call ended before the JSON closed", whose decisive instrument is
`return_token_ids`. Do not upgrade "wrongly accepted token" from hypothesis to
fact on that paste alone.

#8122 (vllm-ascend v0.14.0rc1, Qwen3-Next-80B-A3B-Instruct, `stream: true`,
`--async-scheduling`): "大概率截断，小概率格式错误，重复输出", no MTP. The pasted
97-chunk stream assembles to valid JSON, so that sample is not the failing one;
the recorded answer's ranked lead was `--async-scheduling` + streaming tool parser
on the strength of #5683 (v0.13.0rc1, Qwen3-8B: async scheduling on → only CoT, no
`tool_calls`; off → correct, reported A/B) and the CI gap
(`async_scheduling` absent from `tests/e2e/multicard/4-cards/test_qwen3_next.py`
at v0.18.0rc1). That is a reporter-confirmed precedent plus an uncovered axis, not
a source-verified mechanism — keep those labels when reusing it.
