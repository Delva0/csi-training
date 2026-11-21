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

Five symptom classes, different owners — decide which one the report is before
reading any code:

1. **The service never becomes ready**: `EngineCore failed to start` at the first
   weight group, `ValueError: There is no module or parameter named
   '<name>' in <…ForCausalLM>`, `Starting to load model` printed and shards at
   0/N → the checkpoint was served as a *generation* model, not a pooler →
   `embedding-checkpoint-as-generate.md` (one call:
   `../../scripts/checkpoint-key-prefix.py <model dir>`).
2. **The service dies under concurrency** (vllm-bench / max-num-seqs > ~32),
   single-request serving is fine, and the engine log ends in a Python exception
   in the *batch bookkeeping* — `KeyError: None` (or `KeyError: <id>`) raised by
   `scheduler_output.num_scheduled_tokens[req_id]`, in
   `reorder_batch_to_split_decodes_and_prefills` *or* in
   `_prepare_inputs`' `tokens = [...]` (both are the same lookup; which frame
   names it is version luck, and a reporter who prints in the reorder helper or
   in `_update_states` legitimately sees nothing) — then `EngineDeadError` and
   `400 Bad Request` on everything after it. The `None` is a hole inside
   `InputBatch._req_ids`; reporter-confirmed shape on #5687,
   `print(input_batch)` in the second batch: `[None,None,xxx,297]`, then
   `[None,298,299,xx]` as requests are added → `batch-index-desync.md` (which also
   carries the producer-side probe recipe: wrap `add_request` /
   `remove_request` / `condense`, do not probe a consumer).
3. **The vectors are wrong for long inputs** while short inputs match the
   reference to > 0.999, the corruption is deterministic inside one serve
   session, and `--no-enable-chunked-prefill` removes it →
   `chunked-prefill-vector-corruption.md`.
4. **It never gets past engine start, and the message is about the model class or
   the HF config type** — `TypeError: Invalid type of HuggingFace config.
   Expected type: <class '...Qwen3VLConfig'>, but found type: <class '...Qwen3Config'>`
   from `qwen3_vl.py get_data_parser`, in the **APIServer**, with
   `--runner pooling --hf_overrides '{"architectures": ["<X>ForSequenceClassification"]}'`
   → the architecture string is a routing key, and the family prefix in it must
   match the checkpoint (`seq-cls-arch-routing.md`). Not a device, driver or
   `config.json` fault.
5. **It never gets past engine start, and the message is about the weight layout
   or dtype** — `call aclnnMatmulWeightNz failed`, `AclNN_Parameter_Error(EZ1001):
   Float32 weight NZ is unsupported by the current SOC version [...]` inside
   `determine_available_memory` → `profile_run` → `_dummy_pooler_run`, deepest
   `vllm_ascend` frames `ops/linear.py` → `ops/linear_op.py` → `F.linear`
   → the fp32 classifier head was cast to `FRACTAL_NZ` (`fp32-head-nz-format-crash.md`).
   This is **not** the custom-op install lane even though `EZ1001` appears in both.

Not this topic: a score/MMLU gap between harnesses (`../accuracy-benchmarks/index.md`),
garbled *text* from a generation model (`../accuracy-benchmarks/`), a device dump
or a kernel-launch aclnn error from a custom op (`../custom-ops-runtime-faults/`),
an OOM (`../runtime-memory/`), 310P launch problems (`../soc-platform-paths/`).
The split that keeps class 5 here rather than in the custom-op lane is textual:
`EZ1001 … Get regInfo failed` / `does not support opType` is the install fault
there, while `EZ1001 … <Dtype> weight NZ is unsupported … now self is DT_<x>` is a
weight-layout fault whose owner is the Python that chose the layout.

## Rules

1. **Classify with the two-question split, then answer at G1.** (a) single request
   or only under concurrency? (b) short prompts correct or wrong at any length?
   Those two answers place the report in class 2 or 3, and each class already has
   a ranked answer — emit it before opening the attention backend. A service that
   never becomes ready is a startup class (1, 4 or 5), where the launch line and
   the error text decide, not a batch or attention sweep. Either way a pooling
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
6. **A start-up failure is a launch-input question, and the traceback already
   names the input.** `AutoWeightsLoader` is prefix-strict: it groups the
   checkpoint's names by their first dotted component and raises
   `There is no module or parameter named '<prefix>' in <Class>`
   (`vllm/model_executor/models/utils.py:139/208/277`) when that component is not
   a child of the class the engine built. `<prefix>` is the *qualified* name and
   `<Class>` is the *built* class, so `'embed_tokens'` (no `model.`) plus a
   `…ForCausalLM` name says both "the tensors are root-level" and "the pooling
   wrapper was never applied" before any source read — the wrapper would have
   been renamed to `…ForEmbedding` (`adapters.py:132`). Check the launch line for
   `--task embed` / `--runner pooling` and the init line for `pooler_config=None`
   first; at vllm v0.10.0 `_ModelInfo.is_pooling_model` is `True` for every
   registered architecture, so the flag is always available.
7. **The chunk budget is the boundary.** Any "corruption starts at N tokens"
   claim needs the effective `max_num_batched_tokens` (from the serve log, never
   from memory: the default differs between releases) compared against the
   reported N. If a sweep of that flag moves the boundary, the chunk boundary is
   the trigger and the owner is scheduling/accumulation/masking; if it does not
   move, it is length-dependent numerics.

Three scripts serve this topic's startup classes directly:
`scripts/seq-cls-arch-check.sh <vllm-root> '<Arch>[,<Arch>]'` (class 4) and
`scripts/nz-format-gate-check.sh <paste|-> [vllm-ascend-root]` (class 5);
`../../scripts/checkpoint-key-prefix.py <model-dir>` answers class 1.

## Pages

- `embedding-checkpoint-as-generate.md` — class 1. Worked case vllm-ascend
  #5549 (vllm v0.10.0 + vllm-ascend v0.10.0rc1, 310P3, Qwen3-Embedding-0.6B
  launched with no `--task`): how to read the `ValueError`'s prefix and class
  name, why `--task embed` turns the root-level checkpoint keys into a loadable
  set (the wrapper's `WeightsMapper({"model.": ""})`), why the downstream class
  is not the owner (verbatim upstream loader), and the pooling-type and
  HBM-budget follow-ups.
- `batch-index-desync.md` — class 2. Worked case vllm-ascend #5525 (v0.11.0,
  310P-class image, `vllm-bench` + `--task embed`, Qwen3-Embedding-8B,
  `--max-num-seqs 256`): the downstream `InputBatch` copy's pooling fast path
  (`add_request` → `num_reqs`, `remove_request` without `removed_append`,
  `condense` trim-only) versus upstream's slot machinery, the exact sequence that
  leaves a `None` hole and a duplicated slot, the `KeyError` at
  `vllm/v1/attention/backends/utils.py:798`, and that the copy is gone by v0.23.0.
- `chunked-prefill-vector-corruption.md` — class 3. Worked case vllm-ascend
  #17797 (v0.23.0-310p, `--runner pooling`, Qwen3-Embedding-0.6B): what the
  report proves, what is verified in the pinned tree, the three Ascend-specific
  pooling branches (encoder-attention branch, fixed-size pooling masks, 310P
  chunked-prefill dispatch), the ranked leads each with its deciding check, and
  the ablation ladder an answer can be shipped from.
- `seq-cls-arch-routing.md` — class 4. Worked case vllm-ascend #14817 (v0.23.0,
  Qwen3-Reranker-0.6B, A2): how `ModelRegistry._normalize_arch` rewrites
  `*ForSequenceClassification` to a registered base class, why
  `Qwen3VLForSequenceClassification` selects `Qwen3VLForConditionalGeneration`
  (multimodal) while `Qwen3ForSequenceClassification` selects `Qwen3ForCausalLM`,
  the authoritative name from upstream `docs/models/pooling_models/scoring.md`
  and the repo's own e2e job, and
  the tutorial page that copied the VL sibling's override line.
- `fp32-head-nz-format-crash.md` — class 5. Worked case vllm-ascend #4679
  (v0.11.0rc3, Qwen3-Reranker-8B, 910B): the fp32 score head built by upstream
  `adapters.py`, the `is_enable_nz` gate that excluded only fp16/bf16, the NZ cast
  in `AscendUnquantizedLinearMethod.process_weights_after_loading`, the
  `VLLM_ASCEND_ENABLE_NZ=0` escape, the two `EZ1001` text families, and the
  v0.23.0 gate rewrite (`_should_trans_nz`, "FP32 cannot use NZ") that makes the
  verdict ref-specific.
