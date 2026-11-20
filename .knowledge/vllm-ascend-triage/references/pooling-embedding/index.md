# Pooling / embedding runners on Ascend

## Scope

A serve whose model is a **pooler**, not a decoder: `--runner pooling`,
`--task embed` / `score` / `classify` / `reward`, a request against
`/v1/embeddings`, `/v1/score` or `/v1/rerank`, a `pooler_config`
(`AllPool`, `LastPool`, `StepPool`, `tokwise`/`headwise`), or a report that
quotes a **cosine similarity** between two services instead of generated text.
The whole class is one flag inside the engine: `is_pooling_model` /
`model_runner_type == "pooling"` / `model_config.runner_type == "pooling"`.
Every branch guarded by one of those names is this topic's candidate owner, and
`vllm_ascend/worker/npu_input_batch.py` (batch slots), `_pool` in
`vllm_ascend/worker/model_runner_v1.py`, `vllm_ascend/attention/attention_mask.py`
and `vllm_ascend/attention/attention_v1.py` (attention routing, masks) are where
they live.

Two symptom classes, different owners — decide which one the report is before
reading any code:

1. **The service dies under concurrency** (vllm-bench / max-num-seqs > ~32),
   single-request serving is fine, and the engine log ends in a Python exception
   in the *batch bookkeeping* — `KeyError: <id>` / `KeyError: None` in
   `reorder_batch_to_split_decodes_and_prefills`, then `EngineDeadError` and
   `400 Bad Request` on everything after it → `batch-index-desync.md`.
2. **The vectors are wrong for long inputs** while short inputs match the
   reference to > 0.999, the corruption is deterministic inside one serve
   session, and `--no-enable-chunked-prefill` removes it →
   `chunked-prefill-vector-corruption.md`.

Not this topic: a score/MMLU gap between harnesses (`../accuracy-benchmarks/index.md`),
garbled *text* from a generation model (`../accuracy-benchmarks/`), a device dump
or aclnn error (`../custom-ops-runtime-faults/`), an OOM
(`../runtime-memory/`), 310P launch problems (`../soc-platform-paths/`).

## Rules

1. **Classify with the two-question split, then answer at G1.** (a) single request
   or only under concurrency? (b) short prompts correct or wrong at any length?
   Those two answers place the report in class 1 or 2, and each class already has
   a ranked answer — emit it before opening the attention backend. A pooling
   report is answered with an ablation ladder, not with a kernel investigation
   (`SKILL.md` rule 1 and the "a new hypothesis that needs a new code area is not
   this session's job" bullet).
2. **One grep decides whether the batch machinery is a stale downstream copy.**
   `rg -n "is_pooling_model" vllm_ascend/worker/npu_input_batch.py` plus the same
   grep in `vllm/v1/worker/gpu_input_batch.py` at the pinned vllm commit. If the
   vllm-ascend file *defines* `add_request`/`remove_request`/`condense`, it is a
   copy of upstream's `InputBatch` and every extra `is_pooling_model` branch in it
   is a candidate defect; by v0.23.0 the copy is gone
   (`class NPUInputBatch(InputBatch)`, `__init__` only) and the slot machinery is
   upstream's again. That added-branch diff is also the ownership argument when
   the report asks whether the bug is upstream's — the verbatim-copy method of
   `../upstream-downstream-drift/index.md` (`patch-copy-vs-upstream.md`).
3. **Check that the pooling accumulator is inherited, not re-implemented.**
   `rg -n "pooling_states|PoolingStates|def _pool|build_pooling_cursor" vllm_ascend/`
   — the chunked-prefill accumulation for `AllPool` lives upstream
   (`vllm/v1/pool/metadata.py:39-45`, `vllm/model_executor/layers/pooler/tokwise/methods.py:56-84`,
   `gpu_model_runner.py:_pool`). Hits only in `worker/npu_input_batch.py` mean the
   state is imported but never written; hits nowhere mean the runner calls
   upstream `_pool`, so "the accumulator is missing" is refuted and the time goes
   to rules 4-5 instead.
4. **Before any kernel theory, ask which token span was pooled.** A pooling model
   over a chunked prefill can return the vector of the *first chunk*, the *last
   chunk* or the *whole prompt* — all plausible, all deterministic. The reporter
   can settle it without touching the NPU: embed the prompt truncated at the
   chunk boundary (`max_num_batched_tokens`, printed in the serve log) and at the
   suffix. `cos ≈ 1` against one of those proves "wrong span" and points at the
   cursor/`optimistic_seq_lens_cpu`; `cos ≈ 0.2-0.6` against both proves the
   hidden states themselves are wrong and points at attention/mask/numerics.
5. **Split the determinism signature.** "Deterministic for one input inside a
   serve session, different after a restart" is a *state/reuse* signature
   (buffer or cursor carried across steps or requests) — not a fixed numeric
   defect, which reproduces identically after a restart. Say which one the report
   shows; it removes half the candidate list for free.
6. **The chunk budget is the boundary.** Any "corruption starts at N tokens"
   claim needs the effective `max_num_batched_tokens` (from the serve log, never
   from memory: the default differs between releases) compared against the
   reported N. If a sweep of that flag moves the boundary, the chunk boundary is
   the trigger and the owner is scheduling/accumulation/masking; if it does not
   move, it is length-dependent numerics.

## Pages

- `batch-index-desync.md` — class 1. Worked case vllm-ascend #5525 (v0.11.0,
  310P-class image, `vllm-bench` + `--task embed`, Qwen3-Embedding-8B,
  `--max-num-seqs 256`): the downstream `InputBatch` copy's pooling fast path
  (`add_request` → `num_reqs`, `remove_request` without `removed_append`,
  `condense` trim-only) versus upstream's slot machinery, the exact sequence that
  leaves a `None` hole and a duplicated slot, the `KeyError` at
  `vllm/v1/attention/backends/utils.py:798`, and that the copy is gone by v0.23.0.
- `chunked-prefill-vector-corruption.md` — class 2. Worked case vllm-ascend
  #17797 (v0.23.0-310p, `--runner pooling`, Qwen3-Embedding-0.6B): what the
  report proves, what is verified in the pinned tree, the three Ascend-specific
  pooling branches (encoder-attention branch, fixed-size pooling masks, 310P
  chunked-prefill dispatch), the ranked leads each with its deciding check, and
  the ablation ladder an answer can be shipped from.
