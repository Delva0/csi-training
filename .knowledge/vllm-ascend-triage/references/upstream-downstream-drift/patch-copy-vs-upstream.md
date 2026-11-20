# A traceback frame in `vllm_ascend/patch/**`: extract, diff, then attribute

## When this page applies

The deepest `vllm_ascend` frame of a traceback is inside `vllm_ascend/patch/`
(`patch/worker/patch_model_runner.py`, `patch/platform/patch_core.py`,
`patch/platform/patch_fused_moe.py`, …). Those modules are **verbatim copies** of
upstream methods plus a small delta, installed by assignment
(`GPUModelRunner._update_states = _update_states`), and they are gated on one
upstream version, e.g. `patch/worker/patch_model_runner.py:293
if vllm_version_is('0.13.0')` with an else-branch warning "patch
GPUModelRunner._update_states failed!".

So the frame names vllm-ascend while the *statement* may be upstream text that
vllm-ascend merely carries. Attribution has to be decided by a diff, not by the
file path, and the diff is the cheapest step in the whole investigation — do it
first, not after the mechanism is derived from proxies.

## Procedure

1. Read the exact frame (`sed -n 'A,Bp' <patch file>`) and note the *two* hop
   chain: the Ascend runner's own method calling `super()` — e.g.
   `worker/model_runner_v1.py:3726 _update_states` →
   `super()._update_states(...)` resolves to the patched function, which is why
   the traceback's innermost frame is the `patch/` file.
2. Extract the upstream definition at the pinned vllm commit and diff. The patch
   file holds it at module level (4-space body), upstream holds it as a method
   (dedent by 4):

   ```bash
   sed -n '753,1018p' vllm/v1/worker/gpu_model_runner.py | sed 's/^    //' > /tmp/up.py
   sed -n '17,292p'   vllm_ascend/patch/worker/patch_model_runner.py       > /tmp/pa.py
   diff -u /tmp/up.py /tmp/pa.py
   ```

   Find the span with `grep -n 'def <name>'` in both files (the next `def` at the
   same indent ends it). A naive "read until the next `def`" extractor that
   ignores indent grabbed 4718 lines in a recorded session and cost it two turns;
   give the ranges explicitly.
3. Classify each hunk: **formatting** (black line-wrapping, `# lenth` → `# length`
   comment typo fixes) vs **an added statement**. Only the added statements can
   be a vllm-ascend defect. If the failing statement is inside a formatting-only
   region, the owner is upstream vLLM at that pin, and the answer says so — with
   the upstream `path:line` (`vllm/v1/worker/gpu_model_runner.py:888-897`), the
   patch `path:line` that carries the same text, and the version gate that binds
   the copy to one upstream release.
4. Check whether the added statement is even live for the reporter's command.
   Patch deltas are usually gated on a deployment shape (PD `is_kv_consumer`,
   `additional_config` key, `use_async_scheduling`); the reporter's serve command
   in the issue decides it in one line.
5. Only then read the `_prepare_inputs`/sampler side. The answer's first sentence
   can already be "the crash is real; the statement is upstream v0.13.0's, copied
   verbatim into `patch_model_runner.py:154`, so the mitigation is a flag, not a
   patch".

## Worked case: #8724 (vllm-ascend v0.13.0rc3, vllm v0.13.0)

Reporter: A3 16-card, DeepSeek-V4-Flash-w8a8-mtp, `--speculative-config
'{"num_speculative_tokens": 1, "method": "deepseek_mtp"}'`,
`--data-parallel-size 4 --tensor-parallel-size 4 --enable-expert-parallel
--async-scheduling --max-num-seqs 4`, `vllm bench serve --temperature 0.6`, and
three tracebacks at once:

- `patch/worker/patch_model_runner.py:154`
  `prev_req_index = self.input_batch.prev_req_id_to_index[req_id]` →
  `KeyError: 'chatcmpl-bench-…'`;
- `scheduler.py:1060` `model_runner_output.sampled_token_ids` →
  `AttributeError: 'NoneType' object has no attribute 'sampled_token_ids'`
  (the EngineCore receives no output because the workers died);
- `v1/spec_decode/metrics.py:208` `counter_spec_decode_num_accepted_tokens.inc(...)`
  → `ValueError: Counters can only be incremented by non-negative amounts`.

Diff result at the pins: `_update_states` in the patch is upstream
`gpu_model_runner.py:753-1018` re-indented, with formatting-only hunks plus **one**
added statement (`patch_model_runner.py:277-282`, `if self.is_kv_consumer and
self.speculative_config and self.use_async_scheduling:` → set
`req_state.prev_num_draft_len = len(spec_token_ids)` for newly added/resumed
requests). The reporter's command configures no KV transfer, so `is_kv_consumer`
is `False` (`model_runner_v1.py:306-309`) and that delta never ran. The failing
statement is upstream's.

The one-provenance reading — the part worth saying first, because three
tracebacks invite three hunts — is a **discarded request**:
`_prepare_inputs` computes `discard_requests_mask = seq_lens.np[:num_reqs] <
num_tokens_np` (`model_runner_v1.py:838-842`) for requests whose sampled tokens
must not be used; `prev_req_id_to_index` is built *excluding* those indices
(`model_runner_v1.py:2040-2044`), and `prepare_next_token_ids_padded` fills their
row with `-1` (`spec_decode/eagle_proposer.py:816-822`) so
`valid_sampled_tokens_count == 0`. Then, in the same step,
`num_accepted = valid_sampled_token_count[prev_req_index] - 1 == -1`
(`patch_model_runner.py:154-157`), which is exactly the negative
`num_accepted_tokens` the metrics counter rejects; and at the next step the same
request is in the batch but absent from `prev_req_id_to_index`, so the unguarded
lookup raises the `KeyError`. One provenance, three symptoms.

Why `0 < temperature < 1` and not `0`: the vllm-ascend rejection sampler takes
the all-greedy branch only when `sampling_metadata.all_greedy`
(`sample/rejection_sampler.py:119,155` — `if sampling_metadata.all_greedy:
return output_token_ids`) and otherwise falls through to the stochastic path
(`generate_uniform_probs`, `sample_recovered_tokens`, `:158` onward). So a
fractional temperature changes which branch writes the padded
`(batch, max_spec_len + 1)` output buffer, i.e. which entries survive as
`PLACEHOLDER_TOKEN_ID`. **The exact step where the row becomes discarded is not
established by this report** — label it hypothesis, and settle it by logging
`self.num_discarded_requests` per step at the reporter's own temperature.

First actions in the answer: `--async-scheduling` off (removes
`use_async_scheduling`, hence the `prev_num_draft_len` bookkeeping and the
discard-driven lookup) and `--speculative-config` off (row 1 of the MTP ladder);
then `num_speculative_tokens 1` → `2` and `temperature 0.6` with an explicit
`seed`. The `KeyError` itself is fixable upstream by using
`prev_req_id_to_index.get(req_id)` and skipping the request — that is a patch
proposal for vllm, not something to hand the reporter as their fix.

Related: `../spec-decode-mtp/async-spec-bookkeeping-desync.md` carries this
symptom as a spec-decode track with the ablation order;
`stale-module-binding.md` and `patch-import-side-effect.md` cover the other two
ways a `patch/` module disappoints (`patch/` frame present but with a different
failure shape).
