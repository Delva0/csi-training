# A downstream backend raises its own `… are currently not supported`

Scope: the traceback's deepest frame is a vllm-ascend backend and the exception is
a deliberate `raise` — `RuntimeError: Prefix cache and chunked prefill are currently
not supported`, next to a `# TODO: Will support … soon.` comment — not a type or
attribute error. Seen first in an RL rollout: an open-r1/`trl` GRPO trainer
constructs `vllm.LLM(...)` itself and calls `generate` on it, so the failure appears
inside the training loop. This page is the capability-gap answer: which input the
branch refuses, who enabled it, and which "fixes" cannot work. Verified at
vllm-ascend v0.7.1rc1 (`f17417f2d5b5`) + vllm v0.7.1 (`4f4d427ac2ce`); the V0-file
layout is version-specific, but the shape recurs wherever a downstream backend
enumerates the paths it implements.

## 1. The raise is unconditional for one metadata state

- `vllm_ascend/attention.py:594-598` (`AscendAttentionBackendImpl`) and `:775-779`
  (the MLA impl): the prefill branch takes `npu_selfattention` only while
  `attn_metadata.block_tables is None or numel() == 0`; every other prefill batch
  hits the `else:` that raises. So the message is about the **presence of a block
  table**, not about sampling, dtype or device.
- Two independent producers fill it in the metadata builder
  (`attention.py:369-390`): `prefix_cache_hit = any(inter_data.prefix_cache_hit …)`
  ⇒ `block_table = block_tables[seq_id]`, and `(chunked_prefill_enabled or not
  is_prompt)` with a non-`None` `block_tables`. Prefix cache **or** chunked prefill,
  either one.

## 2. Defaults would not have reached it, so the caller set a flag

- `vllm/engine/arg_utils.py:207-210`: `enable_prefix_caching` defaults to `None` and
  becomes `bool(envs.VLLM_USE_V1)`; `vllm/envs.py:479-480` defaults `VLLM_USE_V1` to
  `"0"`; `vllm/config.py:1448`: `enable_chunked_prefill: bool = False`. On V0 both
  are off unless something passed them.
- The one upstream override to rule out first: **V0 + multimodal** auto-disables
  prefix caching and logs `--enable-prefix-caching is currently not supported for
  multimodal models in v0 and has been disabled.` (`arg_utils.py:1059-1066`). If the
  paste carries that warning, the flag is not the trigger and the chunked-prefill
  leg is the suspect.
- Therefore the deliverable names the **enabler**, not the backend: in the recorded
  case the trainer's own `LLM(...)` call. GRPO duplicates each prompt
  `num_generations` times and can schedule a duplicate in a later step of the same
  `generate` call, so the first training step (`0/11208`) is enough — no warm-up
  needed. Label such a caller-side claim as recalled unless the harness source is in
  the snapshot, and give the one grep that settles it:
  `grep -n -A12 'LLM(' $(python -c "import trl,os;print(os.path.dirname(trl.__file__))")/trainer/grpo_trainer.py`.
- Also check that vllm-ascend does not force the flag itself before blaming the
  caller (`rg -n 'enable_prefix_caching|chunked_prefill' vllm_ascend/` — at this ref
  only the scheduler's read of `chunked_prefill_enabled` and the `block_size`
  rewrite appear).

## 3. What not to spend the session on

`--enforce-eager`, `ASCEND_LAUNCH_BLOCKING=1`, device health and plogs cannot move a
branch that refuses an input shape, and `VLLM_USE_V1=1` is not an escape here: at
v0.7.1rc1 the string `VLLM_USE_V1` does not occur anywhere under `vllm_ascend/`, so
the V1 engine is untested against this plugin (re-grep per ref). The capability
answer also needs the *dates*: a later release implements the path, so the honest
close is "fixed in a later rc — recheck the release notes for prefix cache /
chunked prefill" plus the probe the reporter can run on their own tree
(`sed -n '580,600p' vllm_ascend/attention.py`; a version without the `raise` is the
minimal fix).

## Answer shape for this shape

1. Direct: a hard-coded capability gap at the reported ref, not a misconfiguration
   of torch/CANN; the cheapest action is on the caller — pass
   `enable_prefix_caching=False` (and do not enable chunked prefill) in the
   trainer's `LLM(...)` construction.
2. The evidence table: the raise site, the two block-table producers, the defaults
   that prove someone enabled it, and the multimodal auto-disable row.
3. The enabler as ranked cause 1 with its one grep, chunked prefill as cause 2 with
   the engine-init line that would show it, and the in-batch duplicate as the
   reason step 0 is enough.
4. Scope: the V0 backend classes only, per-ref line numbers, plus the one-line
   probe for "is the raise still there".

The recorded #133 run is the shape to copy: 12 tool calls, one message per turn, the
answer at turn 12 — no walk through `qwen2.py`, the model file that only appears in
the stack because it calls `self.attn`.
