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

Eleven symptom classes, different owners — decide which one the report is before
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
   `remove_request` / `condense`, do not probe a consumer). The class also arrives
   under a title that names nothing but the front end — `AsyncLLM output_handler
   failed` (#4156, a Qwen2.5-VL-3B checkpoint on `/v1/embeddings`, already at
   v0.11.0rc0) with the `KeyError` readable only in the attached screenshot — and
   that same front-end line fronts a metric-arithmetic crash too
   (`../kv-pool-store/prompt-token-counter-negative.md`), so read the
   engine-core exception before choosing this lane.
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

6. **The launch line is rejected before anything is built** — `vllm serve:
   error: argument --task: invalid choice: 'rerank' (choose from 'auto',
   'classify', … 'score', …)`, or the task is accepted and the release cannot
   serve a pooler at all → the value (`score`, and `/v1/rerank` is unchanged) plus
   the **pooling release gate** (v0.9.1 = `Pooling 🟡 Planned`, "v0.9.1 doesn't
   support pooling. Use higher version instead"; V1 pooling from v0.9.2rc1,
   #1359) — `task-values-and-pooling-release-gate.md`. No worker ever started, so
   nothing in the other classes applies.
7. **The same string embedded twice in one serve session gives two different
   vectors** — call 1 matches the reference service to `cos ≈ 0.999`, call 2 drops
   to `cos ≈ 0.5-0.6`, and calls 3, 4, 5 are bit-identical to call 2 — and
   `--no-enable-prefix-caching` (not `--no-enable-chunked-prefill`) removes it, on
   a prompt short enough for one schedule → the `embed` task reads the KV prefix
   cache and the pooling attention branch attends only the scheduled tail →
   `prefix-cache-hit-embedding-drift.md`, which carries the reporter-confirmed
   workaround (#5725), the two pins' `path:line`, and the nonce-suffix check that
   confirms it without a restart. Not a numeric defect: the first call would be
   wrong too, and noise cannot freeze.

8. **It never gets past engine start, and the message is the capture guard**
   `Cannot run aclop operators during NPU graph capture. Current working aclop is
   Fill` (+ `ERR00007 PTA feature not supported`), with an FX trace above it
   showing `torch.ones_like(… dtype=torch.int32) << 30` → upstream BERT
   `_decode_token_type_ids` runs **unconditionally** in every BERT-family forward
   (`BertEmbedding.forward`) and its int32 `ones_like` is an aclop `Fill` at these
   wheels, so bge / bge-reranker and every `BertModel` checkpoint die at capture
   while a Qwen3-Embedding serve does not → `../startup-failures/aclop-op-during-graph-capture.md`.
   The `ACLgraph has insufficient available streams` block and the
   `allow_internal_format = False` advice in the same paste are both red herrings
   (the plugin assigns the option `True` at import).

9. **The request never returns to the client while the server logs `200 OK`** — a
   `/v1/embeddings` `curl`/`requests` call hangs to timeout, and the only server lines
   are the torch_npu `torch.save with "_use_new_zipfile_serialization = False" is not
   recommended for npu tensor` warning plus
   `(APIServer pid=…) INFO: … "POST /v1/embeddings HTTP/1.1" 200 OK` → the warning is
   the *success*-path notice, printed verbatim in this release's own embedding
   tutorial (`docs/source/tutorials/single_npu_qwen3_embedding.md:96` at v0.11.0), and
   the body is built in the API server before that 200 is written — so nothing here
   failed and the missing bytes are between the pod and the client →
   `client-hang-after-200-ok.md`. Not a serialization, dtype or prefix-cache lane.

10. **It dies at the first step in graph mode on a bidirectional-attention model, and
   only graph mode** — `npu_fusion_attention` / `aclnnFlashAttentionVarLenScore` +
   `AclNN_Parameter_Error(EZ1001): FlashAttentionScore launch kernel failed`, with
   `The end of actualSeqQLen & actualSeqKvLen should be 0 in EOD scenario` in the
   tiling block, while `--enforce-eager` serves
   → the pooling/encoder FA site's TND tail invariant under graph-mode token padding
   → `../custom-ops-runtime-faults/fa-tiling-argument-refusals.md` (member B). This is
   the *runtime* sibling of class 8 (which dies at capture): same `--runner pooling`
   models, different phase, and eager is the control that separates them.

11. **It serves, then every `/v1/embeddings` request is a 400 and the engine is dead** —
   deepest frame `vllm_ascend/core/scheduler.py:209 blocks =
   new_computed_blocks.blocks[0]`, `IndexError: tuple index out of range`, raised from
   `schedule()` in EngineCore's `step`, no device frame anywhere. An **encoder-only**
   checkpoint (bge-m3 / `BertModel` / `XLMRobertaModel`) has **zero** KV-cache groups,
   so `create_empty_block_list()` is `KVCacheBlocks(())` and the fork indexes it — at
   `:209` and again at `:520` — while the fork itself was force-enabled by
   `platform.py:139-150`, so no `--additional-config` was needed. A decoder-based
   embedding checkpoint (Qwen3-Embedding) has ≥1 group and never crashes, which is why
   the tutorial and the e2e job are green → `../scheduler-admission/scheduler-fork-block-index.md`
   (recorded #3462, v0.11.0rc0 + vllm v0.11.0rc3, bge-m3, k8s/`--task embed`). Not a
   batch, mask or numeric lane: one request, before any token is computed.

Not this topic: a score/MMLU gap between harnesses (`../accuracy-benchmarks/index.md`),
garbled *text* from a generation model (`../accuracy-benchmarks/`), a device dump
or a kernel-launch aclnn error from a custom op (`../custom-ops-runtime-faults/`),
an OOM (`../runtime-memory/`), 310P launch problems (`../soc-platform-paths/`),
or the **second reranker/embedding instance on one card** dying at KV sizing
(`Available memory: 0`) while the die has free HBM
(`../runtime-memory/kv-budget-per-instance-vs-device-global.md`, the recorded
`#7308` case — it came from exactly this deployment shape).
The split that keeps class 5 here rather than in the custom-op lane is textual:
`EZ1001 … Get regInfo failed` / `does not support opType` is the install fault
there, while `EZ1001 … <Dtype> weight NZ is unsupported … now self is DT_<x>` is a
weight-layout fault whose owner is the Python that chose the layout.

## Rules

1. **Classify with the three-question split, then answer at G1.** (a) single
   request or only under concurrency? (b) short prompts correct or wrong at any
   length? (c) does the report compare **repetitions of the same string** inside
   one serve session, with the first one right? — yes to (c) is class 7 and its
   workaround (`--no-enable-prefix-caching`) belongs in the first message, whereas
   class 3's is `--no-enable-chunked-prefill`; both flags are cheap, so an answer
   that lists both as one two-arm ablation wastes the reporter's restart (say which
   one the evidence favours instead). Each class already has a ranked answer — emit
   it before opening the attention backend. A service that
   never becomes ready is a startup class (1, 4, 5 or 8), where the launch line and
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
   shows; it removes half the candidate list for free. The *sharper* form is a
   **repeat inside one session**: first call right, second and later ones wrong and
   equal to each other (`cos 1.0`) is a cache/reuse signature with a known ablation
   (class 7), and it is worth one line in the first message even when the reporter
   has not asked about caching — #5725 filed it as a numerical 误差 and the cache
   was the answer.
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

8. **Ask encoder-only or decoder-based before any KV-cache arithmetic.** The split is
   one field — `architectures` in the checkpoint's `config.json` — and it moves the
   whole KV story. Encoder-only checkpoints (`BertModel`, `XLMRobertaModel`, an encoder
   `…ForSequenceClassification`) have **no** KV cache: the spec dict is empty,
   `is_kv_cache_spec_uniform({})` returns `True` under the tree's own comment
   `# Encoder-only models do not have KV cache`
   (`vllm/v1/core/kv_cache_utils.py:768-771` at vllm v0.11.0rc3), the empty-spec branch
   skips the `No available memory` `raise` (`:696-702`), and `kv_cache_groups` ends up
   **empty** (`:744-752`) so every per-group index is an empty tuple. Decoder-based
   embedding checkpoints (`Qwen/Qwen3-Embedding-*`, the only shape
   `tests/e2e/singlecard/test_embedding.py` serves at v0.11.0rc0) have ≥1 group.
   Consequences to state in an answer: a green tutorial/CI row says nothing about an
   encoder-only crash (class 11); `num_gpu_blocks is: 1` / `GPU KV cache size` lines are
   placeholders on this shape, not a pool; and any downstream code doing
   `kv_cache_groups[0]`, `blocks[0]` or prefix-cache lookup is a candidate owner before
   the attention backend is.

Three scripts serve this topic's startup classes directly:
`scripts/seq-cls-arch-check.sh <vllm-root> '<Arch>[,<Arch>]'` (class 4) and
`scripts/nz-format-gate-check.sh <paste|-> [vllm-ascend-root]` (class 5);
`../../scripts/checkpoint-key-prefix.py <model-dir>` answers class 1.

## Pages

- `embedding-checkpoint-as-generate.md` — class 1. Worked cases vllm-ascend
  #5549 (vllm v0.10.0 + vllm-ascend v0.10.0rc1, 310P3, Qwen3-Embedding-0.6B
  launched with no `--task`) and #2556 (same pair on a 910B/A2 host, screenshot-only
  paste: the cause is platform-independent): how to read the `ValueError`'s prefix and class
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
  `vllm/v1/attention/backends/utils.py:798`, that the copy is present at v0.11.0rc0
  (`00ba071`, #4156) and gone by v0.13.0rc1 — quote the recheck
  (`rg -n 'def condense|def add_request' vllm_ascend/worker/npu_input_batch.py`
  returns nothing), not a version number.
- `chunked-prefill-vector-corruption.md` — class 3. Worked case vllm-ascend
  #17797 (v0.23.0-310p, `--runner pooling`, Qwen3-Embedding-0.6B): what the
  report proves, what is verified in the pinned tree, the three Ascend-specific
  pooling branches (encoder-attention branch, fixed-size pooling masks, 310P
  chunked-prefill dispatch), the ranked leads each with its deciding check, and
  the ablation ladder an answer can be shipped from.
- `prefix-cache-hit-embedding-drift.md` — class 7. Worked case vllm-ascend #5725
  (vllm 0.13.0 + vllm-ascend v0.13.0rc1, 910B4, Qwen3-Embedding-0.6B): first call
  `cos 0.9997` vs the reference, second `0.5855`, third+ bit-identical to the
  second, `--no-enable-prefix-caching` removes it (reporter-confirmed, and still
  reproducing on v0.14.0rc1 / v0.15.0rc1). Why `embed` reads the prefix cache
  (`pooling_params.py:128-135`), why the hit is block-aligned
  (`kv_cache_manager.py:180`/`:189`), the unconditional pooling branch and its two
  `actual_seq_kvlen = actual_seq_lengths_q` sites
  (`attention_v1.py:708`/`:580`/`:598`,`:610` at v0.13.0rc1, `:937`/`:819`/`:841`
  at v0.14.0rc1) against the correct `context_lens=seq_lens` path, the three-hypothesis
  matrix, the nonce-suffix ladder, the CI gap that let it through, and why
  `usage.prompt_tokens_details.cached_tokens` is a dead check on this path.
- `task-values-and-pooling-release-gate.md` — class 6. Recorded case vllm-ascend
  #4796 (v0.9.1, `bge-reranker-v2-m3`, `--task rerank`): `rerank` is not in
  `TaskOption` (`vllm/config.py:82-83`), the endpoints already exist, and on
  vllm-ascend v0.9.1 pooling is `🟡 Planned` while the V0 engine is retired — so
  upstream's own `_raise_or_fallback` makes `--task score` either raise
  `NotImplementedError` (VLLM_USE_V1=1 exported) or fall back to an engine this
  image dropped; the model was named as working at v0.7.3rc1 (#229), which is why
  the reporter's expectation is reasonable. A 3-call answer.
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
  `VLLM_ASCEND_ENABLE_NZ=0` escape, its `EZ1001` dtype x layout text (the shape
  sibling is `../startup-failures/nz-weight-format-refusal.md`), and the
  v0.23.0 gate rewrite (`_should_trans_nz`, "FP32 cannot use NZ") that makes the
  verdict ref-specific.
- `../custom-ops-runtime-faults/fa-tiling-argument-refusals.md` — class 10, member B:
  `aclnnFlashAttentionVarLenScore` / `FlashAttentionScore launch kernel failed` +
  `… should be 0 in EOD scenario, but got (15) and (15)` under graph mode with eager
  working (#10757, bge-m3 on vllm-ascend v0.18.0 + vLLM `dev`, 910B4, CANN 9.0.0):
  the encoder FA call does not trim `query` to `actual_seq_lengths_q[-1]` the way the
  FIA sibling does, the padding hook that would reconcile it runs only under
  `cudagraph_mode: FULL`, and the `causal` gate that opens the branch is unsettable at
  the pin — so the version question is asked, not derived.
- `client-hang-after-200-ok.md` — class 9. Recorded case vllm-ascend #6424
  (v0.11.0 / vllm v0.11.0, bge-m3 on 910B, CANN 8.3.RC2, K8s): the two log lines the
  report treats as cause and evidence, the tutorial `path:line` that makes the
  torch_npu warning benign, the `serving_embedding.py:134` → `outputs.py:225-230`
  path that builds the body *before* the 200 (so a completed response rules this code
  out), the offline-only bge-m3 e2e coverage that makes "supported" a narrower claim
  than it looks, and the three-rung transport ladder with the in-pod loopback curl as
  the deciding check.
