# Speculative decoding / MTP runtime symptoms

## Scope

The deployment is up and the symptom is produced by the speculative-decoding path
(`--speculative-config` with `method: mtp`, `qwen3_5_mtp`, `ngram`, `eagle`,
`draft_model`, or an `AscendEagleProposer` / `AscendDraftModelProposer` frame in the
stack). Usually there is no traceback; one shape is a plain `AssertionError` raised
at request time. The shapes:

- the spec-decode metrics collapse and request throughput follows: `Accepted: 0`
  with `Drafted: N > 0`, `Per-position acceptance rate: 0.000` for every position,
  `Mean acceptance length` pinned at 1.00, "推理卡住" while requests stay
  `Running` — and when the report also carries a **`num_speculative_tokens`
  threshold** (a lower K works, K is above the validated 3, or the same K served on an
  older release), the K is the classification and the sweep is rung 1:
  `acceptance-zero-at-high-k.md` before `acceptance-collapse.md`'s code walk;
- the **whole service** stops with no text at all — the worker log simply ends,
  no `ERROR`, no device dump, the API server still holding the port, and the process
  only dying much later ("run 1x hour, it will be Stuck"), on a launch with an MTP
  method **and** `--async-scheduling`: the sampling/rejection-sampler lane, which has
  a reporter-confirmed three-PR fix set on the 0.18.0 line (`mtp-sampler-silent-stall.md`),
  and which competes with the device-side MC2 deadlock page for this same silence;
- a response comes back empty (`content: ""`) with `usage.completion_tokens: 1`,
  usually on the first request(s) after the server starts, and later requests are
  fine;
- the **process** dies instead: a worker `KeyError` in
  `vllm_ascend/patch/worker/patch_model_runner.py` `_update_states`, a
  `'NoneType' object has no attribute 'sampled_token_ids'` in the EngineCore and a
  negative `Counters can only be incremented by non-negative amounts`, together,
  with `--async-scheduling` and `--speculative-config` in the same launch and a
  non-zero `temperature` (`async-spec-bookkeeping-desync.md`).
- the **process** dies with a *device dump* instead of a Python error: the deepest
  frame is `counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)` in
  `_copy_valid_sampled_token_count` (`model_runner_v1.py:1393`) and the paste also
  carries `rtMemcpyAsyncEx … reason=the model stream execute failed`, an `aivec
  error … timeout or trap error` and an HCCL task line with `rankSize` /
  `numBlocks` (`drafter-count-copy-device-fault.md`);
- or it dies with a device dump whose frames are
  `execute_dummy_batch` → `_dummy_run` → `acl_graph.__call__` →
  `stream.synchronize()`, on a launch carrying MTP + `FULL_DECODE_ONLY`
  (+ `npugraph_ex`), with `fftsplus aicore/aivector … MTE DDR out of range` and a
  capture-mode `rtMemcpy … 107030` record stamped inside engine init
  (`draft-graph-capture-device-fault.md`).
- **or** the same capture-mode family on a **`suffix`/`ngram`** launch: the log keeps
  `SpecDecoding metrics: … Accepted: 0 … Drafted: 1` and a `Replaying aclgraph` line,
  then `EE9999 … reason=the current capture mode does not support this operation` /
  `synchronized memcpy failed, kind = 1` / `aivec … The MAR address of the …
  instruction is out of range`, and the deepest Python frame is
  `RejectionSampler.parse_output`'s `output_token_ids.cpu()`
  (`draft-graph-capture-device-fault.md`, section "The non-MTP variant"). At `n=3` this
  method never takes the block-verify kernel (`draft_probs is None`,
  `vllm_ascend/sample/rejection_sampler.py:114`), has no drafter graph (`dummy_run` is
  `pass`), and `suffix + FULL_DECODE_ONLY` is the combination #7301 already records as
  degraded — so the ablation is the graph mode, not `n`.
- or it dies with an **operator-level message and no Python frame**: `ERR00100 PTA
  call acl api failed` + `EZ9999 … When layout is TND, queryT(16) must be equal to
  the last element of actualSequenceLengthQ(4)` (`CheckFASeqlenDataInTND`,
  `fused_infer_attention_score.cpp`), on MTP + `FULL_DECODE_ONLY`, often with
  `--data-parallel-size > 1` and EP — or the same refusal from the other direction,
  `queryT(16)` vs `actualSequenceLengthQ(13)` behind
  `call aclnnFusedInferAttentionScoreV3 failed, error code is 561002`, on **`ngram` +
  `cudagraph_mode: FULL`** (recorded #9088, v0.19.1rc1). The two launches are
  different lanes of the same file: at a v0.19+ ref the pair alone derives the
  in-flight request count (`r = N/(1+K)`) and the branch, so read that first
  (`fia-tnd-seqlen-mismatch.md`, Lane A vs Lane B).
- or it dies at **request time** with `aclnnFusedInferAttentionScoreV3 failed,
  error code is 107030` + `the key/value's actual sequence lengths(K) is N, it should
  be in range [0, KV_S(M)]` + `FusedInferAttentionScore do tiling failed, ret is -1`,
  the stack passing `_run_merged_draft` / `acl_graph.__call__` on an eagle/eagle3
  launch, and the dumped config showing `N` one past the engine's own sequence cap:
  a host-side argument rejection at the last scheduled step before the cap, not a
  device fault (`max-model-len-boundary-fia-crash.md`). Distinct from the TND
  `queryT` refusal above by the string: that one names two host tensors that must be
  equal, this one names a value and a **capacity**.
- or it dies at **request time** on every rank in the same second with an ATB
  operator message and a *proposer* frame: `SelfAttentionOperation setup failed!`
  + `Exception raised from OperationSetup … op_plugin_atb … compiler_depend.ts`
  (dump frames `atb::OperationSetup(…)` in `torch_npu/lib/libop_plugin_atb.so`),
  then `… the current working operator name is SelfAttentionOperation` /
  `ERR00100 PTA call acl api failed`, deepest `vllm_ascend` frame
  `spec_decode/eagle_proposer.py:476 sample_hidden_states = last_hidden_states[…]
  [last_token_indices]`, one `Replaying aclgraph` line just before, on an
  eagle/eagle3 launch: an ATB **setup** failure surfacing at a join point, with
  `ASCEND_LAUNCH_BLOCKING=1` refused while graph mode is on, so the attribution
  order is `--enforce-eager` **then** blocking
  (`atb-op-setup-failure-at-proposer.md`). Two knobs it is *not*:
  `draft_tensor_parallel_size` (no consumer under `vllm/v1/**` at v0.11.0) and the
  target-family gate (`qwen3` is accepted).
- or the first request dies with a bare `RuntimeError: The expanded size of the tensor
  (H_draft) must match the existing size (H_target) at non-singleton dimension 1.
  Target sizes: […, H_draft]. Tensor sizes: […, H_target]`, deepest frame
  `eagle_proposer.py set_inputs_first_pass` at `self.hidden_states[:num_tokens] =
  target_hidden_states`, on `method: draft_model` **without** `parallel_drafting` and a
  draft whose `hidden_size` differs from the target's: the fork takes its derived flags
  from upstream `EagleProposer.__init__` (`pass_hidden_states_to_model=True`) and only
  afterwards flips the flag, so `needs_extra_input_slots` stays `False` and the EAGLE
  fast path copies target-width hidden states into a draft-width buffer. Which
  `__init__` computes the flag is the one grep that decides the ref, and at v0.17.0rc1
  the owner is downstream (`draft-model-serial-hidden-size-crash.md`).
- or the **same function's PCP branch** assigns a `None` CPU tensor:
  `TypeError: 'NoneType' object does not support item assignment` at
  `cad.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p` (`eagle_proposer.py:1158`), on a
  launch carrying `--async-scheduling` **and** `--speculative-config` with K > 0 **and**
  `--prefill-context-parallel-size` > 1, once a step mixes prefills with decodes. The
  runner deliberately passes `seq_lens_cpu=None` in async spec-decode mode and keeps the
  CPU copy only in `_seq_lens_cpu`, whose upstream lazy fallback the downstream dataclass
  field shadows; rung 1 is `--async-scheduling` off and the fix is post-pin
  (`pcp-async-spec-seq-lens-cpu-none.md`, recorded #8926, reporter-confirmed #8749).
- or the **first** request on a launch that also sets `--pipeline-parallel-size`
  > 1 dies with a plain `IndexError`/`AttributeError` (no device record), with
  `suffix`, `ngram` or `mtp` alike and startup clean: the PP × spec-decode
  contract, not the proposer — PP>1 gives the engine a batch queue of depth
  `pp_size` that no flag removes, and swapping the method is the
  proposer-innocence probe (`../pipeline-parallel/spec-decode-first-request.md`,
  #5227: `suffix` and `ngram` both fail on the same PP=2 launch).
- or it dies at **request time** on every rank in the same second with a bare
  `AssertionError` whose last frame is `attention_v1.py … assert
  attn_metadata.attn_mask is not None`, on a dense model with `method: ngram` (any
  non-`eagle` method): `AscendAttentionState.SpecDecoding` is declared in the enum
  but the dense backend has no branch for it, and `_make_attention_mask` returns
  `None` for that state (`dense-attn-state-gap.md`). The traceback may pass through
  `<eval_with_key>.N` / `unified_ascend_attention_with_output` — that is the graph
  wrapper around the same eager body, not a capture fault.
- or it dies at **request time** with a bare `AssertionError: Batch_size(N) must be
  divisible by chunk_num(M)` in `vllm_ascend/ops/linear_op.py otp_maybe_quant_comm`, the
  stack passing `eagle_proposer._run_merged_draft` → `self.model(...)` → upstream
  `llama.py o_proj`, on an eagle/eagle3 launch that exports
  `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` > 0 with `num_speculative_tokens` ≥ 2: FlashComm2's
  o_proj all-to-all reaches the **draft** model and its `pad_size` is written once per
  forward context, so the multi-step loop runs on unpadded rows. `M` is the global TP size
  for every legal flag value (the value is not the cause); `N` is rows after the pad, and
  `num_speculative_tokens: 1` is the only FC2-preserving shape
  (`../feature-enable-recipes/flashcomm2-draft-pad-token-divisibility.md`, #7809).
- or it dies at **request time on every rank in the same second** with a **Python
  `AttributeError`** on a *dense* drafter whose capture succeeded:
  `'ForwardContext' object has no attribute 'draft_attn_metadatas'`, deepest frame
  `attention_v1.py update_graph_params`, stack `model_runner_v1.propose_draft_token_ids`
  → `eagle_proposer._propose` → `_update_full_graph_params` → `acl_graph.update_full_graph_params`,
  on `cudagraph_mode: FULL_DECODE_ONLY` + `method: eagle3|eagle`. The drafter's helper
  takes the per-step metadata list and drops it while the attention impl reads it off
  `ForwardContext`, so no flag on the launch is the cause and `PIECEWISE` is the
  unblock (`draft-full-graph-metadata-unwired.md`).
- or the metrics read `Accepted: 0` / every per-position rate `0.000` / `Drafted: N > 0`
  **while the target model's own output is correct**, on a model *family* whose
  upstream spec decode has a dedicated proposer/speculator (gemma4, and any family the
  `get_spec_decode_method` dispatch does not name): the family has no downstream
  branch, so it runs on the generic single-group eagle path — `family-proposer-parity.md`.
  This is the one acceptance-collapse shape that is decided *above* the positions and
  block-table code, so the family grep comes before `acceptance-collapse.md`'s Step 3.

- or the request dies with a plain **`IndexError: list index out of range`** whose
  deepest frame is upstream `vllm/model_executor/layers/mamba/mamba_utils.py`
  `get_temporal_copy_spec` (`src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]`),
  reached through `vllm/v1/worker/mamba_utils.py` `collect_mamba_copy_meta` /
  `preprocess_mamba` from `vllm_ascend/worker/model_runner_v1.py`, on a hybrid GDN model
  carrying `--speculative-config` **and** `--enable-prefix-caching` (often
  `--async-scheduling`, `mamba_cache_mode` auto-set to `align`, and "either feature alone
  is fine"): the align-mode mamba state copy, four frames and no device code, with the
  precedent filed in the history under those **upstream** names (#9024, #10326 at
  v0.19.1rc1) — `hybrid-mtp-prefix-cache-unvalidated.md`, §*The `IndexError` form*.
- or there is **no error text at all**: the title says "同时开启 mtp+prefix cache 后，推理
  报错" on a hybrid GDN model (`qwen3_5`) and the bug body is an upload failure or one
  Chinese line, so the only evidence is the serve command
  (`--speculative-config … "method": "qwen3_5_mtp"`, `--enable-prefix-caching`,
  `--async-scheduling`, `FULL_DECODE_ONLY`). Then the pair itself is the finding: the
  tutorial documents it (`docs/source/tutorials/models/Qwen3.5-27B.md:137`, with the
  "block_size may be adjusted to 2048" caveat) while every validated row — the nightly
  yaml, the 4-card `test_qwen3_5.py`, and the dense spec-decode e2e that hard-codes
  `enable_prefix_caching=False` — runs it off, and at a v0.19+ ref the align-mode block
  arithmetic (`MambaManager`, `num_speculative_blocks`) is the crash surface. The body's
  "error" may also be a `<details>` block whose whole content is one `user-attachments`
  link (`errors.txt`, #8913) — a link is not text, so that is the same form. Verdict +
  ladder in the first message (`hybrid-mtp-prefix-cache-unvalidated.md`, worked #9003,
  #8913).

- or the **decode node's EngineCore dies at the scheduler step**, before any model
  execution, with `Warning: The token id -1 is out of range … Rejecting the token`
  (xgrammar) then `Failed to advance FSM … for tokens -1` then
  `AssertionError: (-1, '<req>', {'<req>': [-1, -1, -1]})` at
  `vllm/v1/structured_output/__init__.py:288`, on a **P/D** launch that carries
  `--speculative-config` **and** structured output (`response_format` / `guided_*`):
  the `-1`s are seeded placeholder draft slots, and their count is the decode node's
  `num_speculative_tokens`. Recorded #6421 was closed by **P/D config parity**;
  the pin also has a source-verified seeder (`RecomputeScheduler.add_request`, gated
  on `recompute_scheduler_enable` + `is_kv_consumer`). Diff both serve commands and
  run the `recompute_scheduler_enable`-off launch before ranking a code fix →
  `../pd-disaggregation/structured-output-draft-placeholder-assert.md`.

- or the **process** dies at request time with a bare `AssertionError: num_decodes: N,
  num_spec_decodes: M` (both ≥ 1) whose deepest frame is upstream
  `vllm/v1/attention/backends/gdn_attn.py` `GDNAttentionMetadataBuilder.build`, reached
  from `vllm_ascend/worker/model_runner_v1.py _build_attn_group_metadata`, on a hybrid GDN
  family (Qwen3.5/Qwen3.6, Qwen3-Next) with MTP/eagle, every rank in the same second,
  MTP-off clean for hours: one step mixed a request that had draft tokens with one that did
  not. The two counters *are* the classification — **check the version next**, because
  vllm ≥ v0.22.1 reclassifies that half as prefill instead of asserting
  (`mixed-spec-and-plain-decode-assert.md`).

- or it dies at **request time on the host, with no device frame**, in the host-side
  reshape of the Triton rope helper — `RuntimeError: shape '[6, -1]' is invalid for
  input of size 512` at `vllm_ascend/ops/triton/rope.py` `cos = cos.view(num_tokens, -1)`,
  stack `mtp_proposer.py _propose` → `self.model(...)` →
  `ops/rotary_embedding.py _rope_forward_oot` → `rope_forward_triton`, usually on a
  serve carrying `--async-scheduling` + `--speculative-config` with
  `num_speculative_tokens: 1` + an explicit `cudagraph_capture_sizes` list: the cos/sin
  slice is a **process-global** that the *target* refreshes with its
  `pad_for_cudagraph`-padded positions while the MTP proposer never refreshes it (the
  EAGLE proposer does), so the two counts differ on any step whose token count is not a
  capture size. One division (`cos.numel() / width`) and one grep
  (`rg -n 'update_cos_sin' vllm_ascend/`) name it; the model's rope width is not the
  question (`rope-cos-sin-slice-token-mismatch.md`, recorded #6449).

- or it **aborts on the host with one operator sentence and no Python frame**:
  `numAcceptedTokens[0]=4 exceeds varlen segment length=3` (or `[1]=6 … length=4`) from
  the AscendC causal_conv1d **tiling validation**
  (`csrc/moe/causal_conv1d/op_host/causal_conv1d_tiling_validation.h:358-364`), on a
  hybrid GDN + MTP `K=3` launch with `FULL_DECODE_ONLY`, after hours-to-days of healthy
  serving: the op compares **this step's** per-request segment (from the scheduler's
  `query_start_loc_cpu`) with the accepted count taken from the runner's
  **CPU/optimistic** buffer, so the pair only agrees while every spec request is
  scheduled with a full `1+K`. `--async-scheduling` off, then `--enforce-eager`, then a
  plog with **both** host tuples → `../custom-ops-runtime-faults/gdn-conv1d-accepted-tokens-varlen.md`
  (recorded #10372).

- or **nothing raises at all**: on a hybrid GDN/Mamba family (`qwen3_5`, Qwen3.5/Qwen3.6) with
  `--async-scheduling` + `--speculative-config` MTP + `--enable-prefix-caching`, a small fraction
  of requests (2/512 at 100 % KV) come back with repeated/omitted token runs, the reporter's own
  control `--async-scheduling` off is clean, and the body's diagnosis is that the previous
  iteration's accepted-token counts are gathered from mutable `InputBatch` rows that
  `add_request()`/`condense()` already rewrote — the **silent** member of the
  `num_accepted_tokens` family (`accepted-token-row-snapshot.md`).

Not this topic: a first-request fault whose only extra axis is PP > 1
(`../pipeline-parallel/spec-decode-first-request.md`), a launch/startup failure
(`../startup-failures/index.md`), a drafter that refuses to **load** (a load-phase
`AttributeError` naming a *model loader* class straight after `Loading drafter
model...`, e.g. `'DummyModelLoader' object has no attribute 'get_all_weights'` under
`--load-format dummy` — a loader-contract question whose object is the loader, so the
model, checkpoint and quant config are exonerated:
`../upstream-downstream-drift/load-format-loader-contract-drift.md`),
collective or device fault whose sync point is **not** the drafter's count copy
(`../hccl-and-hardware/index.md`; the copy-shaped variant stays here, because the
frame proves the launch shape and the ablations are spec-decode rows), wrong or alternating
answers on a **PD** pair where the decode node received the state
(`../pd-disaggregation/hybrid-state-transfer.md`), and a "does this plan fit on N
cards" question (`../deploy-sizing/index.md`). A **performance** complaint where spec decode is
switched on and the engine is healthy with acceptance visibly non-zero ("开 MTP 反而慢了",
two benchmark tables, no traceback) is the token-per-step arithmetic track:
`../performance-throughput/spec-decode-gain-accounting.md` — unless the lever that was
toggled is FlashComm1/SP, where the owner is
`../feature-enable-recipes/sequence-parallel-flashcomm.md` §*FC1 with a MoE MTP drafter*.
A **tool call** whose JSON the
client cannot parse (`Unterminated string`, 格式错误, 参数被截断) while the text
itself is fine is also not here: the open question is whether the stream or the
assembly lost the tail, so start at `../tool-call-streaming/index.md`. When its
failing `num_speculative_tokens` is >= 3 and 1 or 2 works, that topic's
`mtp-block-verify-acceptance.md` carries the `max_spec_len >= 3` branch (at newer
refs it also ANDs `rejection_sampler_config.enable_block_verify`, default `False`)
— name it with the `n=2` and `temperature: 0` ablations rather than re-deriving it
from the acceptance numbers.

## Rules

- **A per-K threshold in the report outranks the bookkeeping walk.** "Acceptance 0 at
  `num_speculative_tokens` 13, normal below 8, and v0.17.0 served 13 correctly" is a
  parameter control **plus** a version control: the failing pin holds no clamp to find
  (its only K bound is `assert decode_threshold <= 16`, three files, so 8…15 is silently
  legal) and a regression cannot be read off the failing tree, so the answer is the
  boundary sweep, the validated K=3 recipe, and a dated window — not
  `eagle_proposer.py`. Recorded #9290 spent 1.22 M prompt tokens and the whole 16 384-token
  output budget there and shipped no text (`acceptance-zero-at-high-k.md`).
- **The MTP-off ablation is row 1, not row 5.** Deleting `--speculative-config` is
  one launch and splits "the draft/verify path is wrong" from "everything else".
  Recorded #10495: the run ranked KV capacity and queue thrash first on the
  strength of 98 % cache usage, and the reporter then removed MTP and reported
  normal inference — the metric collapse tracked MTP, not the queue. Rank the
  capacity arithmetic *after* the MTP row unless MTP-off has already been run.
- **With no error text, the op-message lanes are unreachable — do not open them.** The
  `numAcceptedTokens` / conv1d-tiling, FIA `queryT` and rope-slice pages are keyed by a
  *sentence* in the paste, so ranking one of them without that sentence is a guess. Look
  for a `<details>`-wrapped `user-attachments` link first, then go to the launcher/pair
  verdict (`hybrid-mtp-prefix-cache-unvalidated.md`). Recorded #8913 read
  `../custom-ops-runtime-faults/gdn-conv1d-accepted-tokens-varlen.md` and grepped
  `num_accepted` / `conv_state` across the pinned tree a full turn before the launcher
  pair was on the table.
- **Read the metric pair, not the throughput number.** Drafts still arriving
  (`Drafted: N`, non-zero) while nothing is accepted means both models run; the
  question is whether they see the same context (positions, `num_computed_tokens`,
  accepted-token bookkeeping). A batch that is merely too large to schedule looks
  different: fewer drafts per step, `Waiting` growth, preemption lines, and
  acceptance that is *low*, not exactly 0.000 on every position.
- **Two launcher axes are outside the project's own validated matrix for hybrid
  GDN models** (Qwen3.5/Qwen3.6, DeepSeek-V4, GLM): `--async-scheduling` together
  with spec decode, and `--enable-prefix-caching` together with MTP. Both are real
  code paths with their own fix-ups, so "it is supported, therefore not the cause"
  is not available; they are ablation rows. Which page owns the second one depends on
  the **symptom**, not the SoC: corrupt text at normal acceptance on 310P is
  `../accuracy-benchmarks/mtp-prefix-cache-310p.md`, while a **request-time failure
  with little or no error text on any SoC** is `hybrid-mtp-prefix-cache-unvalidated.md`
  (coverage verdict first, align-mode `MambaManager` asserts second, ladder third).
- **A first-response-only symptom needs a restart reproduction before it gets a
  one-shot mechanism.** "The first decode after capture is wrong, the state is
  never zeroed for the first request" is a plausible story that a single report
  cannot carry: the recorded #9355 follow-up got a normal answer by changing the
  prompt and `temperature` (and posting to another port), so neither request
  content nor process identity was ever pinned. Restart, repeat the identical body
  twice, then the modified body. In the same first pass, diff the launcher against
  the family sibling tutorial in `docs/source/tutorials/models/` when the exact
  model has no page — for #9355 that sibling (`Qwen3.5-27B.md`) shipped
  `"enforce_eager": true` inside `--speculative-config` and default capture sizes,
  and those two deviations are the cheapest rows.

- **When the operator names two numbers that must be equal, the arithmetic is the
  classification — not the third source file.** `queryT(N)` vs
  `last(actualSequenceLengthQ)(M)` (or any op message of the form "A must equal
  B") is a host-side bookkeeping disagreement: `N` is the graph bucket after
  `pad_for_cudagraph` *and* the DP-wide maximum, `M` is the real per-request
  cumsum. Run `scripts/fia-tnd-seqlen-check.sh` on the pair and the launch, state
  both readings with the one log line that separates them (`Running: <n>`), and
  put `PIECEWISE`/`--enforce-eager` in the first paragraph. Recorded #6598 spent
  its whole 16 K output budget re-deriving which of two padding branches produced
  the pair and shipped no answer (`fia-tnd-seqlen-mismatch.md`). From v0.19 that
  derivation is unnecessary: one helper pads both numbers
  (`_pad_query_start_loc_for_fia`), its FULL-mode else-branch always writes
  `last = queryT`, so a refusal means the *uniform* branch ran and therefore
  `queryT == r × (1+K)` — which pins the request count and, when the reporter's own
  ablation says "each feature alone is fine", names the branch condition rather than
  the kernel. Quote the invariant from the helper's own docstring before any source
  walk. Recorded #9088 (`ngram` + FULL, v0.19.1rc1) is that shape.
  **The arithmetic only classifies while `N` exceeds the real schedule.** When
  `queryT` is *smaller* than every plausible `M` (recorded #4771: `queryT(1)` beside
  `actualSequenceLengthQ(33)` and `(2)` in one record, on an eagle3 + dense Qwen2.5
  launch at v0.11.0rc2), the padding-bucket story is excluded — at a v0.11 ref that
  branch's own `arange` ends one element short and can only ever give `N = M + 1` —
  and the pair is the *device record* instead: an AICPU task exception
  (`kernelName=ScatterElements`, `retCode=0x2a`, `task_id`/`stream_id`/`chipId`) or a
  second op's refusal in the same block. Then ask for the plog's
  `[AIC_INFO] fault kernel_name=` segment and put `--enforce-eager` first
  (`fia-tnd-seqlen-mismatch.md` §*Third reading*).

- **Three tracebacks from a spec-decode deployment are usually one provenance.**
  Read them bottom-up: the worker exception is causal, the EngineCore `None`
  output is the worker having died, and the sampler/metrics complaint is a
  *second* symptom of the same bad state, not a third bug. The recorded #8724 set
  (`KeyError` in `_update_states` + `NoneType … sampled_token_ids` + negative
  counter) all traces to one request whose sampled-token row was discarded, i.e.
  to `num_accepted = valid_sampled_token_count[...] - 1 == -1`. Say "one
  provenance, three symptoms" in the first sentence; a reporter reading three
  tracebacks otherwise reads three causes.

- **The attention backend, not the proposer, decides whether a spec method is
  supported.** One grep — `rg -n '<StateName>|<mask field>' vllm_ascend/attention/`
  — returns which backends consume the state and which merely declare it. A state
  that exists in the enum and is consumed by zero branches of the backend that is
  running is a coverage gap, not a config value and not a device fault. The recorded
  #2071 run reached that verdict only after re-reading three overlapping regions of
  `model_runner_v1.py`; the grep answers it in one call (`dense-attn-state-gap.md`).

- **The sync is not a snapshot: check what the gather *reads*, not whether the copy waited.**
  When a report says the previous iteration's counts are read out of mutable storage after the
  batch was condensed, two greps settle it — the writers in `gpu_input_batch.py` (`add_request`
  writes `1`, `condense` swaps rows) and the gather in `_prepare_inputs` — and
  `event.synchronize()` never refutes it: it orders the device-to-host copy and says nothing
  about which request owns the row afterwards. At v0.23.0rc1 that gather is copied into **three**
  sites (upstream `gpu_model_runner.py:2019-2032`, `model_runner_v1.py:1082-1098`,
  `_310p/model_runner_310p.py:283-294`), so quote the site you verified and say the fix has to
  cover the others. A defect of this shape is invisible in every "is the copy synchronized?"
  argument, which is why the reporter's own control is the one to rank on (async off → clean).
  Recorded #13863 then shipped no message text at all while re-deriving the align-mode
  auto-default and the fused kernel (`accepted-token-row-snapshot.md`).

- **On GLM-5.x the `--speculative-config` key `enforce_eager` is a recipe contract, and at
  v0.23.0 nothing restores it for you.** `glm_moe_dsa` + `method: deepseek_mtp` is the pin's
  documented pair, and every GLM-5.2 example pins `"enforce_eager": true` inside the spec
  config with the sentence "required because GLM-5.2 does not support graph-mode speculative
  decoding" (`docs/source/tutorials/models/GLM5.2.md:190`, examples `:176,258,312`). Upstream's
  automatic `enforce_eager = True` is gated on `model_type == "deepseek_v32"` only
  (`vllm/config/speculative.py:559-563`), and the v0.18-era downstream force-clear is **gone**
  at that ref — `vllm_ascend/platform.py:513-527` now only touches the xlite graph config and
  `model_config.enforce_eager` — rule 15 of `SKILL.md` carries that pin split, and its
  "dead lever" erasure stays `deepseek_v32`-only: this family at this pin does not get it.
  One grep of the reporter's launch against the tutorial decides
  whether the drafter ran in graph mode, so take it **before** ranking any graph-mode drafter
  mechanism; recorded #14686's summary line lists only `deepseek_mtp,
  num_speculative_tokens=3`, which is why the exact config is a question to ask rather than a
  fact to assume.

- **Check family parity before reading positions or block tables.** When acceptance is
  exactly 0.000 with a correct target and non-zero drafts, one grep —
  `rg -n 'use_<family>_mtp|<family>_mtp' vllm_ascend/` — decides whether the family
  even has a downstream branch; `get_spec_decode_method` (`vllm_ascend/spec_decode/__init__.py:34-47`)
  routes every `mtp` launch to `AscendEagleProposer` unless a family predicate claims it.
  A family absent there is served by machinery whose base class asserts **one** KV cache
  group for all draft layers (`vllm/v1/spec_decode/llm_base_proposer.py:1556-1574`) and
  that takes its block size from `draft_attn_groups[0]`. Rank this above
  `num_computed_tokens` drift whenever the report names a family and the version ships
  that family's upstream proposer; downgrade it when the same 0 % reproduces with a
  family that *does* have a branch (`family-proposer-parity.md`, worked #16980).

## Pages

- `mtp-sampler-silent-stall.md` — **silence**, not a traceback: a serve that stops
  dead after tens of minutes with the log just ending and the process dying hours
  later, on an MTP + `--async-scheduling` launch. The reporter-confirmed #8916 fix set
  (`0.18.0rc1` base + cherry-picks #8531 sampler `mask`/invalid `draft_token_id`,
  #7904 decoder MTP padding token id, #8764 non-blocking `num_accepted_tokens` race)
  with the attribution left open, why the installed tree is then not the tag, the pin's
  own Qwen3.5-397B tutorial as proof that "the docs config" means MTP `n=3` + async
  scheduling (`Qwen3.5-397B-A17B.md:101,121-124` at `99e1ea0`), the two candidate sites
  (`ops/triton/reject_sample.py:55,102-111`, `sample/rejection_sampler.py:116,126-129`,
  `worker/model_runner_v1.py:2046-2048,2157`), the stale-drafter-`enforce_eager` trap,
  and the separate accuracy defect that outlived the stall fix (#8932).
- `mixed-spec-and-plain-decode-assert.md` — `AssertionError: num_decodes: N,
  num_spec_decodes: M` from the GDN metadata builder: the counter derivation
  (`gdn_attn.py:179-213` at vllm v0.17.0, mask built at
  `gpu_model_runner.py:1695-1719` == `model_runner_v1.py:818-846`), the per-request
  draft-absence inventory in the scheduler, the two-pin version window (v0.17.0 asserts,
  v0.22.1 reclassifies at `:236-247`, vllm-ascend v0.22.1rc1 mirrors it in
  `ops/gdn_attn_builder.py:1010-1015`), the ladder (MTP off → async off → version) and the
  worked #7532 row (v0.17.0rc1, Qwen3.5-27B TP2, 400 pods).
- `acceptance-collapse.md` — the acceptance metric reading, the ordered ablation,
  the async-spec-decode `num_computed_tokens` drift correction, the
  `mamba_block_size` prefix-caching gate, where capacity fits, and the worked
  #10495 row (v0.20.2rc1).
- `empty-first-response.md` — `content: ""` + `completion_tokens: 1` +
  `finish_reason: "stop"` on the first request: the lane split against budget and
  reasoning routing, the restart reproduction, the sibling-tutorial diff
  (`Qwen3.5-27B.md`) and the capture-size arithmetic, reading the emitted token id,
  the two padding stories (scheduler rows refuted; the MTP proposer's padded
  drafter batch live and switchable at `eagle_proposer.py:136-142`, and why
  `--async-scheduling` makes a reporter's draft-side `enforce_eager` A/B a single
  path run twice), the
  draft-side `enforce_eager` / `disable_padded_drafter_batch` ablation rows, the
  `min_tokens` mitigation, and the worked #9355 row (v0.18.0).
- `drafter-count-copy-device-fault.md` — a device dump surfacing at the drafter's
  valid-count D2H copy: why the frame is a sync point, how the frame itself
  proves `--async-scheduling` + spec decode + padded drafter batch (event created
  only under `num_spec_tokens and use_async_scheduling`), resolving the collective
  by rank size against the DP×PCP×TP group arithmetic (`fc3_quant_x`/`mc2` are not
  DP-sized), the ordered ablations, the DeepSeek-V4 `VLLM_ASCEND_APPLY_DSV4_PATCH`
  and `deepseek_mtp` remap gates with the only in-tree validated config, and the
  worked #10546 row (v0.20.2rc1, which shipped no answer).
- `draft-graph-capture-device-fault.md` — the draft forward captured into the
  `FULL` graph under `npugraph_ex`: what the `execute_dummy_batch` frame proves
  (a DP dummy pass, not a request) and does not prove (it runs the target **and**
  the drafter), the two-timestamp reading that makes "ran fine for a while"
  compatible with a fault recorded at init, the ranked candidates with the
  recorded `aclnnAddRmsNormBias` sibling stack, and the ablation ladder with its
  measured cost (spec-scoped `enforce_eager` = stable but several times slower;
  still broken at v0.22.1rc1). Worked #10700 row.
- `fia-tnd-seqlen-mismatch.md` — the FIA TND refusal (`queryT` vs
  `last(actualSequenceLengthQ)`): reading the two numbers as bucket-vs-real, the
  five padding sites that disagree at v0.14.0rc1 (`:588-591` local
  `pad_for_cudagraph`, `:606` DP-wide max, `:856-881` FULL+`uniform_decode` gate,
  `mla_v1.py:565-605` `graph_pad_size`, `mtp_proposer.py:434-448` drafter arange),
  the `FIA_SEQ_LEN_LIMIT = 16` / RFC #3003-#4300 constraint text, the GLM-4.x
  tutorial diff that is the first half of the answer, the six-rung ladder and the
  worked #6598 row (v0.14.0rc1, DP2×TP8 A2 two-node, `num_speculative_tokens: 3`).
- `max-model-len-boundary-fia-crash.md` — the KV-side FIA refusal at the cap
  (`actual sequence lengths(N) … [0, KV_S(M)]`, `do tiling failed, ret is -1`): the
  three dump numbers that classify it (`num_computed_tokens`, `num_scheduled_tokens`,
  `scheduled_spec_decode_tokens=[-1]`), why nothing was launched so no plog/health
  check applies, the two launcher-vs-dump disagreements to surface first
  (`num_spec_tokens` 3 vs 1, a `--max-model-len` above the checkpoint's derived cap
  that vllm v0.19.1 refuses), the ranked candidates (merged-draft guard without
  upstream's `positions + total_num_drafts` lookahead; the `seq_lens_list` host list
  built in `AscendAttentionMetadataBuilder.build` vs the masked tensor; the async
  `optimistic_seq_lens_cpu`), why the co-occurring `rtMemcpy … capture mode` line is
  not the cause, the ladder and the `cann/ops-transformer` gap. Worked #9359 row
  (v0.19.1rc1, A2 2P1D, eagle3, K=3 as launched).
- `dense-attn-state-gap.md` — `assert attn_metadata.attn_mask is not None`
  (`attention_v1.py:413`) under `method: ngram`: the three state branches plus one
  catch-all `else:` that serves `ChunkedPrefill` *and* the unknown states, the
  `use_eagle`-only split at `model_runner_v1.py:977-981` that hands every non-eagle
  method `SpecDecoding`, `_make_attention_mask` returning `None` for it, the
  `spec_attn_mask` consumer table (MLA only) as the one-grep verdict, why the crash
  arrives after a healthy period (first draft-producing decode step), the
  `pytest.skip("Not current support for the test.")` at
  `tests/e2e/singlecard/spec_decode_v1/test_v1_spec_decode.py:72` as the tree's own
  statement, the ladder that excludes `--enforce-eager`, and the worked #2071 row
  (v0.9.2rc1).
- `atb-op-setup-failure-at-proposer.md` — the ATB op-plugin setup failure at a
  proposer frame: the recorded `#4323` paste line by line (all ranks in the same
  second, the `atb::OperationSetup` dump, the `eagle_proposer.py:476` join point),
  the three-layer read (frame is not the owner — history `#447`; the ATB block is
  this process's, not a replay; which op reaches ATB is not in any export, bounded by
  one `rg`), the `--enforce-eager`-then-`ASCEND_LAUNCH_BLOCKING=1` order with the
  reporter-confirmed `ValidationError`, the two candidates a grep eliminates
  (`draft_tensor_parallel_size` no-op at v0.11.0, the eagle3 target-family gate), the
  single-card-only e2e coverage, the ladder, and the delivery arithmetic.
- `../pd-disaggregation/structured-output-draft-placeholder-assert.md` — the spec
  family's **structured-output** face: `assert accepted` with `-1` scheduled draft
  ids on a P/D decode node (nothing executes on device, so no plog applies). Carries
  the P/D key diff that closed #6421 ("inconsistent configuration between prefilling
  and decoding nodes"), the seeding chain at `a17059a2`
  (`recompute_scheduler.py:101-111`/`:289-298`/`:519-532`, `rejection_sampler.py:22`
  `PLACEHOLDER_TOKEN_ID = -1`, `scheduler.py:1030-1057`,
  `structured_output/__init__.py:288`, `backend_xgrammar.py:181`), the single launch
  that separates parity from the placeholder path, and the row-alignment rule any
  patch must keep (`1 + len(spec)` bitmask rows per request).
- `async-spec-bookkeeping-desync.md` — `--async-scheduling` + MTP and the
  three-traceback signature: the discarded-request chain
  (`model_runner_v1.py:838-842` mask → `prev_req_id_to_index` built without those
  indices → `-1`-filled row → `num_accepted = -1` → unguarded lookup), why the
  statement is upstream v0.13.0's and the patch only carries it, the
  greedy-vs-stochastic branch anchor in the rejection sampler — any
  `temperature > 0`, not a `(0,1)` window: `vllm bench serve` substitutes `0.0`
  while a raw client's omitted field arrives as the generation config's value or
  `1.0` — with what remains unverified, the ladder (async off → MTP off →
  `--temperature 1.0` → token step), and the upstream fix shape (#8724).
- `accepted-token-row-snapshot.md` — the **silent** member of the `num_accepted_tokens` family:
  intermittent repeated/omitted token runs on a hybrid GDN serve with async scheduling + MTP +
  prefix caching, reporter-controlled by "async off: 0/1024". Carries the two row spaces
  (iteration-N write per row, N+1 gather through `prev_positions`), the writers that run first
  (`add_request` → `1` at `gpu_input_batch.py:466`, `_swap_states` under `condense` at `:683`),
  why `event.synchronize()` is not a snapshot, the three read sites at the pin, the consumer that
  turns the count into a state-copy offset (`mamba_utils.py:304,331`), the fix shape (a
  runner-owned buffer, the align-mode fused call's argument included), the value-level regression
  test that the pin's source-text UT cannot provide, the three redirects, and what is *not* owed
  (align auto-default, kernel internals, upstream parity). Recorded #13863 = 59 turns, zero
  message text.
- `rope-cos-sin-slice-token-mismatch.md` — `RuntimeError: shape '[N, -1]' is invalid
  for input of size M` at `ops/triton/rope.py`'s `cos = cos.view(num_tokens, -1)` under
  MTP: why `cos` is a **process-global** (`set_cos_and_sin` / `update_cos_sin` /
  `get_cos_and_sin_slice` at `ops/rotary_embedding.py:87-97,147-168`, read at `:189`,
  Triton branch `:218-232`), the division that splits `M` into a width and a row count
  (`scripts/rope-cos-sin-slice-check.sh`), the one bounded grep whose **absence** in
  `mtp_proposer.py` is the finding while `eagle_proposer.py:326/506/670` refreshes it,
  the `FULL_DECODE_ONLY` + `--async-scheduling` gate that unpads the drafter
  (`eagle_proposer.py:138-141` → `mtp_proposer.py:245-249`) against the target's
  `pad_for_cudagraph` (`model_runner_v1.py:587-591`, vllm
  `config/compilation.py:1126-1138`), the crash-vs-silent-corruption table over request
  counts, why the rope width (head size vs `partial_rotary_factor`) must not be hunted
  through a truncated `transformers` export, the 1/2/3-request confirmation and the
  ladder, and the recorded #6449 row (GLM-4.7 `mtp`, TP16, v0.14.0rc1; the recorded run
  shipped no message text).
- `draft-full-graph-metadata-unwired.md` — `AttributeError: 'ForwardContext' object
  has no attribute 'draft_attn_metadatas'` on a **dense** drafter with `FULL_DECODE_ONLY`:
  the five-site chain at v0.14.0rc1 (helper accepts the list, `acl_graph` and the
  attention signature never take it, `attention_v1.py:439` reads it off the forward
  context), why `using_paged_attention()` returning `False` under any
  `speculative_config` makes the FIA branch unconditional while `mla_v1` stays healthy,
  the tree's own **skipped** dense eagle3 e2e row (`.github/workflows/_e2e_test.yaml:128`
  wires it) as the coverage statement, the repair threaded end to end by v0.17.0rc1 with
  the two-grep version check, the ladder, and the recorded #6509 launch (Qwen3-235B w8a8
  TP8+EP, CANN 8.5.0, `n=1`).

- `draft-model-serial-hidden-size-crash.md` — `RuntimeError: The expanded size of the
  tensor … at non-singleton dimension 1` inside `set_inputs_first_pass` on
  `method: draft_model` (serial, no `parallel_drafting`) with a draft `hidden_size`
  different from the target's: where `hidden_states` comes from (draft config,
  `eagle.py:79-82,161-163`), the three-field derivation of `needs_extra_input_slots`
  and why it runs inside the *base* init, the fork-order bug at v0.17.0rc1
  (`SpecDecodeBaseProposer(EagleProposer)` + the flag reassigned at `:95`) versus
  v0.22.1rc1's `AscendSpecDecodeBaseProposer` (flag `True`, so #14453's init gates fire
  instead), why upstream is **not** the owner, the coverage fact (the only draft_model
  e2e is `parallel_drafting: True`, K=8, PARD 2048 vs 4096), and the ladder
  (`parallel_drafting: true` with K ≥ 2 — K=1 recomputes back to this crash — then an
  eagle3 speculator, then ngram, then the structural fix plus its acceptance check).
  Recorded #7365.
- `family-proposer-parity.md` — acceptance exactly 0 % with a correct target on a model
  family the downstream dispatch has no branch for: the one-grep verdict
  (`use_gemma4_mtp` absent, `AscendStep3p5MTPProposer` as the port template), the three
  family mechanisms upstream adds (per-group block tables via
  `set_per_group_block_table`, `constant_draft_positions`, draft→target KV sharing), the
  ngram row that splits draft-side from verify-side, the upstream `method` gate, and the
  support-plan paragraph (three support levels, prerequisites with anchors, the flip
  test) — worked #16980, gemma4 MTP on 910B at v0.23.0.
- `hybrid-mtp-prefix-cache-unvalidated.md` — hybrid GDN + `qwen3_5_mtp` +
  `--enable-prefix-caching` reported with **no error text**: the documented-but-never-
  tested pair (tutorial `Qwen3.5-27B.md:118,137` vs the nightly yaml / 4-card e2e /
  `enable_prefix_caching=False` in the dense spec-decode e2e), why `--mamba-cache-mode
  align` is redundant (upstream auto-default `models/config.py:436-472`), the plugin's
  copy (`patch_mamba_config.py:104-107`), the request-time
  `AssertionError` surface in upstream `MambaManager` (`single_type_kv_cache_manager.py:937-1005`,
  `:961`/`:1003`/`:1005`), the six-rung ladder, and the worked #9003 row
  (v0.19.1rc1, where 20 tool turns produced 19 characters of message text).
  Its **full-traceback** form is the same pair with the answer already written down:
  `IndexError` at upstream `get_temporal_copy_spec`, the align-mode index arithmetic
  (`prev_state_idx + num_accepted_tokens_cpu[i] - 1`, bounded by
  `1 + num_speculative_tokens`), the `finished ∪ preempted ∪ resumed` clear that already
  guards the stale-index story, the two v0.19.1rc1 precedents and the reporter-confirmed
  v0.18.0 control (#9024 / #10326), and the history search keyed on the upstream frame
  names that finds them in one call. A GLM-5.x `glm_moe_dsa` launch is **not** this pair
  (no mamba layers): `../dsa-sparse-attention/glm-moe-dsa-tag-regression-corruption.md`.
- `pcp-async-spec-seq-lens-cpu-none.md` — `TypeError: 'NoneType' object does not
  support item assignment` at `cad.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p` under
  PCP > 1 + `--async-scheduling` + MTP: the `:2466-2469` `None` decision, the
  `_seq_lens_cpu` / `seq_lens_cpu` split at `:2479-2481`, the upstream property
  (`backend.py:394-397`) shadowed by `utils.py:158`, the mixed-step trigger from
  `pcp_utils.init_batch_info`, the pin's own release-note claim (`release_notes.md:21`,
  #7924), the ladder that starts with `--async-scheduling` off, and the two coverage
  facts (the UT pins `pcp_size = 1`; every PCP e2e graph row runs
  `async_scheduling=False`). Recorded #8926, fix reporter-confirmed as #8749.
- `../accuracy-benchmarks/mtp-prefix-cache-310p.md` — when spec decode is the
  *second* feature on a **310P** hybrid model: MTP over an `align`-mode prefix cache
  (per-block SSM state, `num_speculative_blocks`, `accept_token_bias`, the fp16 state
  dtype, and the two 310P CI tests that cover each feature alone and never together).
- `../feature-enable-recipes/flashcomm2-draft-pad-token-divisibility.md` — the FC2 ×
  multi-step-draft collision: `chunk_num == global_tp_size` algebra (so the flag value is
  never the answer), the single-writing-site `pad_size`, the missing draft gate on FC2
  beside the present one on FC1, the K=1 early-exit discriminator, and the coverage fact
  that no e2e combines FC2 with any `speculative_config`.
