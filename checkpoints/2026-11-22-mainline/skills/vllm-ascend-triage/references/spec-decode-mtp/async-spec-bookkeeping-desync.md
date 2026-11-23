# Async scheduling + spec decode: discarded-request bookkeeping desync (#8724)

## Signature

Three errors arrive together, on a deployment that has **`--async-scheduling`
and `--speculative-config` in the same launch**:

1. worker: `KeyError: '<req id>'` at
   `vllm_ascend/patch/worker/patch_model_runner.py:154`
   (`self.input_batch.prev_req_id_to_index[req_id]`), every DP rank of the same
   data-parallel group at the same second;
2. EngineCore: `AttributeError: 'NoneType' object has no attribute
   'sampled_token_ids'` at upstream `v1/core/sched/scheduler.py` — the engine got
   no `ModelRunnerOutput` because (1) killed the workers, so it is a *symptom of
   the symptom*, never the thing to explain;
3. APIServer: `ValueError: Counters can only be incremented by non-negative
   amounts` at `v1/spec_decode/metrics.py:208`
   (`counter_spec_decode_num_accepted_tokens.inc(...)`).

The recorded trigger words are `--temperature 0.6` — and, per the branch below, *any*
temperature at or above `1e-5`, so read a reported `0.6` as one tested value rather
than as a window — `--temperature 0` runs fine (`vllm bench serve` also sends `0.0`
when the flag is absent, while a raw client's omitted field arrives as `1.0`, so ask
which client produced the working run), model DeepSeek-V4-Flash-w8a8-mtp, MTP
`num_speculative_tokens: 1`, `--data-parallel-size 4` + TP 4 + EP, 48-way concurrency
over 144 prompts.

## One provenance, not three bugs

`_prepare_inputs` marks requests whose sampled tokens must be ignored
(`discard_requests_mask = seq_lens.np[:num_reqs] < num_tokens_np`,
`model_runner_v1.py:838-842`). A discarded request

- is excluded from `prev_req_id_to_index`
  (`model_runner_v1.py:2040-2044`), and
- gets a `-1`-filled row in `prepare_next_token_ids_padded`, because
  `discard_request_indices` is index-filled with `-1` before the valid mask is
  summed (`spec_decode/eagle_proposer.py:816-822`), so
  `valid_sampled_tokens_count` is **0** for it.

Then `num_accepted = valid_sampled_token_count[prev_req_index] - 1 == -1`
(`patch_model_runner.py:155-157`) → the scheduler reports a negative
`num_accepted_tokens` → metrics `ValueError` (3). At the next step the request is
still in the batch (`req_index is not None`) with `prev_num_draft_len > 0`
(set at `patch_model_runner.py:261` whenever `use_async_scheduling` and
`num_spec_tokens > 0`) but missing from `prev_req_id_to_index` → unguarded
`KeyError` (1) → worker death → `None` output (2).

The line is upstream text: vllm v0.13.0
`vllm/v1/worker/gpu_model_runner.py:888-897` is byte-identical after dedent, and
vllm-ascend's `patch/worker/patch_model_runner.py` is a verbatim copy plus one
`is_kv_consumer`-gated hunk that a non-PD deployment never executes. So: the
crash is real, the owner is upstream vLLM's async-scheduling bookkeeping, and the
copy is why the traceback names vllm-ascend. Verify with
`../upstream-downstream-drift/patch-copy-vs-upstream.md` before saying "upstream
bug" — the diff is what earns the claim.

## Temperature: one boundary (greedy vs stochastic), not a `(0,1)` window

The stack has exactly one temperature branch, so the title's range is the range the
reporter happened to try, not a source boundary. The vllm-ascend rejection sampler
returns early on the all-greedy path (`sample/rejection_sampler.py:42,119,155`;
`all_greedy` == no request above `_SAMPLING_EPS`, i.e. every request
`temperature < 1e-5` — `vllm/v1/worker/gpu_input_batch.py:355-361,961-962`,
`vllm/sampling_params.py:21,520-525` at the pin) and otherwise
scales logits by `temperature` (`:46-54`) and falls through to
`generate_uniform_probs` / `sample_recovered_tokens` (`:158` onward), which writes
the stochastic entries of the padded `(batch, max_spec_len + 1)` buffer — and that
pattern is what `valid_sampled_tokens_count` counts. Prediction: **any
`temperature >= 1e-5` reproduces, `0` does not**, so `--temperature 1.0` is the run that
tests this reading; a clean `1.0` run means the greedy flag is not the trigger and
the discard condition is the thing to instrument.

Two controls this changes:

- **"no `--temperature`" on `vllm bench serve` is the greedy lane, not a second
  control.** The flag defaults to `None` and the body is then filled with
  `"temperature": 0.0` (`vllm/benchmarks/serve.py:1398,1412-1413` at the same pin,
  vllm v0.13.0), so it is the same launch as `--temperature 0`.
- **A raw OpenAI client that omits the field is *not* greedy**: the server falls
  back to the model's generation config, else `1.0`
  (`vllm/entrypoints/openai/protocol.py:390,406-408` at that pin). "Works without
  temperature" reported from a curl or a third-party harness therefore means "works
  at 1.0" and contradicts the reading above — ask which client produced the working
  run before quoting any range.

Still open: **the step at which a row becomes `discard`-ed in the reporter's run is
not established** — say so, and name the one instrumentation that settles it
(`self.num_discarded_requests` per step).

## Ladder

1. `--async-scheduling` off, same command: removes the `prev_num_draft_len`
   bookkeeping and the discard-driven lookup in one launch.
2. `--speculative-config` off: the MTP row, and the cheapest split between
   "sampling/verify" and "everything else" (`index.md` rule 1).
3. `--temperature 1.0`, then `0.6`, with an explicit `seed`: shows the trigger
   is any non-zero temperature, not one value (and is the row that can refute the
   greedy/stochastic reading above).
4. `num_speculative_tokens 1` → `2`: whether the lookahead budget, not the mode,
   is the trigger.
5. Only then the DP/EP axis: the same batch reaches all four DP workers, which is
   why all four frames appear at once — DP is not the cause.
6. Patch proposal for upstream: `prev_req_id_to_index.get(req_id)` with the
   request skipped (and `max(num_accepted, 0)`), not something to hand the
   reporter as a fix.

## Scope

Source-verified at vllm-ascend `v0.13.0rc3` (`bbe9bc0d`) + vllm `v0.13.0`
(`72506c98`). `patch_model_runner.py` applies only under
`vllm_version_is('0.13.0')`; on any other pairing the patched `_update_states`
is not installed and the `KeyError` would have to be re-derived from that
version's upstream body. The mechanism (discarded row → `-1` count →
`num_accepted = -1` → missing `prev_req_id_to_index` key) is what transfers, not
the line numbers.
