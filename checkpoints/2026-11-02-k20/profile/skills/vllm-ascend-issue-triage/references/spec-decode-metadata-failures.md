# Speculative-decoding metadata failures at a tag (eagle3 / deepseek_mtp, PCP, graph mode)

Scope: an eagle3 or `deepseek_mtp` draft run failing inside metadata the plugin keeps on the
runner, the PCP manager or the forward context. The two shapes are
`AttributeError: '<X>' object has no attribute '<field>'` and
`TypeError: 'NoneType' object does not support item assignment`, raised from
`spec_decode/*_proposer.py` (`_propose`, `set_inputs_first_pass`, `_update_full_graph_params`).
These are reader-versus-writer gaps in one release, not kernel, checkpoint or configuration
faults: the same serve line works once the field is populated or the guard exists. A third shape
belongs here because its parameters come from the same objects: a device-operator tiling reject
(FIA `561002` on `actualSeqLengthsKv`) raised when two metadata counts disagree (last section but
one below). A fourth is not a metadata error at all but the same objects' configuration: an
EAGLE/EAGLE3 draft model started with tensor parallelism, which the paired upstream docs forbid and
the plugin does not reject (the section after the routing checklist).

Verified by reading `/repos/vllm-project/vllm-ascend` tags `v0.11.0rc2`, `v0.14.0rc1`,
`v0.18.0rc1`, `v0.19.1rc1`, `v0.20.2rc1`, `v0.22.1rc1`, `v0.23.0rc1` and upstream `vllm`
`v0.19.0`; nothing was executed on NPU. Line numbers move between releases — re-anchor with
`git grep -n <symbol> <tag> -- 'vllm_ascend/'` before quoting.

## Route to a finding in ~four calls

1. **Reproduce the raising statement at the tag.**
   `git grep -n '<field>' <tag> -- 'vllm_ascend/'`. Both recorded PCP reports matched the tag
   line-for-line (`eagle_proposer.py:1158` at `v0.19.1rc1`, `eagle_proposer.py:606` at
   `v0.18.0rc1`), so the offending statement is normally quotable directly.
2. **Grep the tag's release notes for the subsystem before reading more code.**
   `git show <tag>:docs/source/user_guide/release_notes.md | grep -n -i 'pcp\|mtp\|eagle\|async'`.
   This either produces the documented Known Issue (v0.18.0 carries the PCP+eagle3 short-prefill
   issue) or names the feature that introduced the contract (the `v0.19.1rc1` highlights name
   zero-bubble async scheduling for spec decode, #7640, which is the fix PR's own reference).
3. **Find the writer, not only the reader.** `git grep -n '<field> =' <tag> -- 'vllm_ascend/'`.
   One hit whose enclosing `if` is a step-shape condition (`not with_prefill`,
   `use_async_spec_decode`, `is_draft_model`) explains the failure in a single read; zero hits
   mean the release never populates the field at all.
4. **Bound the fix by ancestry.** `git log --all --oneline --grep='<issue-or-keyword>'`, then
   `git merge-base --is-ancestor <sha> <reported-tag>` and `git tag --contains <sha>`. A reporter's
   "verified, it resolves my issue" is a claim; the guard in the diff plus the tag boundary is the
   evidence.

## EAGLE/EAGLE3 draft model with tensor parallelism: a documented no-TP constraint

Two recorded hangs on the 0.11 line share one configuration — an EAGLE3 draft model with TP>1.
#3398: `vllm-ascend v0.11.0rc0` + vLLM `v0.11.0`, Baichuan-M2-32B plus a sibling `draft/` checkpoint,
`tensor_parallel_size=2`, `distributed_executor_backend="mp"`; the reporter says it hangs "at the
draft model stage" with the 60-s `shm_broadcast` line and no traceback, and the thread is one
collaborator ping, `stale`, then closed — no maintainer diagnosis. #4771 (`v0.11.0rc2`, Qwen2.5-32B
with its own eagle3 draft, TP=2, `num_speculative_tokens: 5`) is the second-request form of the same
configuration (below).

- **The paired upstream docs forbid it.** The plugin tag's `Dockerfile` pins
  `ARG VLLM_TAG=v0.11.0rc3`, and that tag's `docs/features/spec_decode.md` says, under "A few
  important things to consider when using the EAGLE based draft models": "The EAGLE based draft
  models need to be run without tensor parallelism (i.e. draft_tensor_parallel_size is set to 1 in
  `speculative_config`), although it is possible to run the main model using tensor parallelism".
- **The plugin inherits the setting instead of rejecting it.** `SpeculativeConfig.__post_init__`
  is patched in `vllm_ascend/patch/platform/patch_common/patch_config.py`; it computes
  `draft_tensor_parallel_size` through upstream `SpeculativeConfig._verify_and_get_draft_tp`, which
  at `v0.11.0rc3` (`vllm/config/speculative.py`) defaults it to the **target's** TP for every method
  except `mlp_speculator`, and then builds the draft's parallel config with
  `create_draft_parallel_config`. The plugin's `EagleProposer.load_model`
  (`vllm_ascend/spec_decode/eagle_proposer.py` at `v0.11.0rc0`) loads the draft model from
  `self.vllm_config` with that TP and shares the target's `embed_tokens` whenever PP world size is
  1. So a `tensor_parallel_size=2` eagle3 run silently enters the unsupported path and hangs rather
  than raising.
- **Workaround to offer first:** `"draft_tensor_parallel_size": 1` inside `speculative_config`
  (`_verify_and_get_draft_tp` accepts only 1 or the target TP and raises `ValueError` otherwise), or
  run the whole serve at `tensor_parallel_size: 1`. Documented/source-read, not re-run.
- **Do not answer "eagle3 is unsupported at that tag".** The plugin's own `v0.11.0rc0` release-note
  section lists "Eagle3 spec decode method is back now. [#2949]" under `### Other`, and the tree
  carries `spec_decode/eagle_proposer.py` (674 lines; `_propose` asserts
  `isinstance(self.model, Eagle3LlamaForCausalLM)` and `target_hidden_states.shape[-1] ==
  self.hidden_size`). The plugin ships **no** speculative-decoding user guide of its own at that tag
  (`git ls-tree -r --name-only v0.11.0rc0 | grep docs` has no `speculative_decoding.md`), so the
  TP=1 note can only be read in the paired upstream vLLM tree.
- Boundary: the failing op in #3398 is not established — what is verified is that the report's own
  serve line violates a documented constraint and that nothing in the plugin rejects it. Ask for the
  draft-stage lines and test the two workarounds before naming a mechanism.

## `cad.seq_lens_cpu` is None under async spec decode (issue #8926, `v0.19.1rc1`)

Reported: DeepSeek-V3.1-w4a8-mtp-QuaRot, `--prefill-context-parallel-size 2 --async-scheduling
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --speculative-config
'{"num_speculative_tokens": 3, "method":"deepseek_mtp"}'`, image `v0.19.1rc1`; worker dies in
`sample_tokens -> propose_draft_token_ids -> drafter._propose -> set_inputs_first_pass` with
`TypeError: 'NoneType' object does not support item assignment` at
`vllm_ascend/spec_decode/eagle_proposer.py:1158` (`cad.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p`).

Verified at `v0.19.1rc1`:

- Line 1158 sits under `if self.pcp_size > 1:` (:1110) and `if num_prefill_reqs:` (:1152), so the
  raise needs PCP > 1 **and** a step that mixes running decodes with at least one prefill request.
- The same release deliberately blanks the field: `worker/model_runner_v1.py:2466-2468`
  `if self.use_async_spec_decode: seq_lens_cpu = None` ("GPU tensors are authoritative in async
  mode"), while `_seq_lens_cpu` stays populated (:2475-2479) as the mirror attention backends must
  read. So the proposer branch was written for the non-async contract.
- `use_async_spec_decode` is upstream's `use_async_scheduling and num_spec_tokens > 0`
  (`vllm/v1/worker/gpu_model_runner.py:582-584`, with `use_async_scheduling =
  scheduler_config.async_scheduling` at :475, upstream `v0.19.0`). It is therefore exactly
  `--async-scheduling` plus a speculative config — a feature combination, not a user error.
- Fix: `931604054` "[Performance] adapter dcp pcp full graph and async schedule spec"
  (PR #8749, merged 2026-05-08) guards both mirrors (`if cad.seq_lens_cpu is not None:`) and
  repopulates the host mirrors from `_seq_lens_cpu`. `git tag --contains 931604054` starts at
  `v0.20.2rc1`; `git merge-base --is-ancestor 931604054 v0.19.1rc1` is false. Every 0.19.x image
  still carries the defect.

Workaround candidates, each paying for the feature the reporter turned on and neither run on NPU
here: drop `--async-scheduling` (removes `use_async_spec_decode`, so the mirror stays populated), or
drop PCP (`--prefill-context-parallel-size 1`), which leaves the raising branch entirely.

## `'PCPManager' object has no attribute 'mtp_slot_pad'` (issue #8578, `v0.18.0rc1`, short inputs)

Reported: DeepSeek-V3.2-W8A8, TP=8 PCP=2 DCP=8 with EP, `deepseek_mtp`
`num_speculative_tokens: 3`, `FULL_DECODE_ONLY`, vLLM 0.18.0 / vllm-ascend 0.18.0rc1; the worker
raises at `eagle_proposer.py:606`
(`mtp_slot_mapping = self.runner.pcp_manager.mtp_slot_pad`) — an exact match at that tag.

Verified at `v0.18.0rc1`:

- The read is gated by `if self.pcp_size * self.dcp_size > 1:` and
  `if self.num_speculative_tokens > 1 and not attn_metadata_i.num_prefills:` (:592-593, :606).
- The writer exists only for a pure decode step: `worker/pcp_utils.py:584`
  `if self.decode_threshold > 2 and not with_prefill:` … :607
  `self.mtp_slot_pad = mtp_slot_pad.to(...)`. `PCPManager` has no `__init__` default for it, so any
  other step shape raises *AttributeError* (absent), not "None".
- `with_prefill` is the runner's own batch classification, `model_runner_v1.py:635`
  (`attn_state not in [DecodeOnly, SpecDecoding]`, built by `_build_attn_state` at :879) and is
  passed unchanged into `generate_pcp_mtp_input` (:654-666). `decode_threshold` is
  `1 + num_spec_tokens` (`attention/attention_v1.py:235-238`).
- The two sides therefore disagree only when the runner sees a prefill step while the attention
  metadata reports `num_prefills == 0`: `split_decodes_and_prefills`
  (`attention/utils.py:219-261`) classifies by `query_lens > decode_threshold`, so a prefill
  request whose scheduled token count is <= `1 + num_speculative_tokens` is counted as a decode.
  (Ignoring PCP the runner's `with_prefill` uses the *unsplit* scheduled tokens.) That is the
  documented Known Issue in the `v0.18.0` release notes: "PCP and eagle3 overlaying may generate
  error when a prefill req's scheduled token number is smaller than `1 + num_speculative_tokens`,
  which will make this prefill req be treated as a decode req, resulting in an error" (#8402).
  The code chain above is derived from the two classifiers, not instrumented; the report's own
  trigger statement ("as long as the input length is less than batch_size") is the reporter's
  observation.
- Lifecycle bounds the upgrade advice: the unguarded read survives into
  `spec_decode/llm_base_proposer.py:872` at `v0.22.1rc1`, so a nearby minor upgrade does not help.
  From `v0.23.0rc1` `pcp_utils.py:124` initialises
  `self.mtp_slot_pad: torch.Tensor | None = None` and the same mismatch becomes
  `assert self.mtp_slot_pad is not None` (:1341) — a different failure for the same step-shape
  disagreement. On `main` the field is `dcp_utils.py:121 mtp_slot_mapping`.
- Only the trigger condition is documented; the notes give no workaround. Report the condition
  (prefill scheduled tokens must exceed `1 + num_speculative_tokens`, so short prompts / chunk
  boundaries / a larger `num_speculative_tokens` interact) as the documented constraint, and the
  missing writer as the code-level cause.

## `'ForwardContext' object has no attribute 'draft_attn_metadatas'` (issue #6509, `v0.14.0rc1`)

Reported: eagle3 with `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` on 0.14.0RC1,
frames `sample_tokens -> propose_draft_token_ids -> drafter._propose -> _update_full_graph_params ->
update_full_graph_params -> impl_cls.update_graph_params`, failing statement
`attention_v1.py:439 attn_metadata = forward_context.draft_attn_metadatas`.

The report is self-consistent with tag `v0.14.0rc1`: `eagle_proposer.py:1181` is the
`update_full_graph_params(` call and `compilation/acl_graph.py:223` is
`impl_cls.update_graph_params(`. Verified there:

- `git grep -n 'draft_attn_metadatas' v0.14.0rc1 -- 'vllm_ascend/'` returns exactly two hits: the
  read at `attention_v1.py:439` (under `if forward_context.is_draft_model:` at :437) and
  `eagle_proposer.py:1180 def _update_full_graph_params(self, forward_context, num_tokens,
  draft_attn_metadatas=None)`, whose body (:1181-1183) never forwards that argument.
- `compilation/acl_graph.py:213-230` has no `draft_attn_metadatas` parameter, and
  `ascend_forward_context.set_ascend_forward_context` never sets the attribute (it does set
  `is_draft_model` at :129). Nothing in the tag assigns the field, so the draft-model
  graph-parameter branch cannot execute at all: a reader without a writer, in one build.
- The whole plumbing appears at `v0.15.0rc1`: `ascend_forward_context.py:47` parameter and
  `:66 forward_context.draft_attn_metadatas = draft_attn_metadatas`; `acl_graph.py:221/231`
  threads it into `update_graph_params`; `attention_v1.py:444` reads the local argument;
  `eagle_proposer.py:358/523` pass `draft_attn_metadatas=multi_steps_attn_metadata`. So
  `v0.15.0rc1` is the first tag whose branch can run — code availability, not a runtime-verified
  fix.
- The issue thread carries no maintainer answer (a single "How to solve it？" comment), so the
  release-to-release source delta above is the entire finding; do not re-read it for a cause.

## FIA `561002`: `actualSeqLengthsKv` length versus the q-row count

Reported (`v0.23.0rc1`, Qwen3-32B-w8a8c8 + eagle3 draft, TP=2, 910B4, CANN 9.0.1,
`{"cudagraph_mode": "FULL_DECODE_ONLY"}`, `{"method": "eagle3", "num_speculative_tokens": 1}`):

```text
RuntimeError: ... FusedInferAttentionScoreKernelNpuOpApi.cpp:615 NPU function error:
  call aclnnFusedInferAttentionScoreV3 failed, error code is 561002
EZ9999: the key/value's actual sequence lengths size[256] should be greater than q batch[512]
  or equal to 1.[FUNC:ProcessActualSeqLen][FILE:incre_flash_attention_tiling.cpp][LINE:977]
FusedInferAttentionScore do tiling failed, ret is -1.  Check NnopbaseExecutorDoTiling(executor) failed
```

Read the two numbers before anything else: `N` is the length of the kv-length list (one entry per
*request*), `M` is the query's batch dimension **in the layout FIA was told**, not the request
count. `561002` here is a tiling-parameter rejection, so a device fault, CANN version or
checkpoint defect is the wrong first hypothesis.

- **C8 (KV-int8) attention + speculative decoding, verified at `v0.23.0rc1`.** The C8 attention
  backend's graph path `full_graph_fia` (`vllm_ascend/attention/attention_v1.py:806`, entered from
  `forward` at `:1593-1595` when `attn_metadata.attn_state == DecodeOnly` and `_EXTRA_CTX.capturing`)
  switches layout in its quant branch at `:841-860` (gated by `self.enable_c8_quant`, read from the
  quant config's `enable_c8_quant` at `:434-435`): `# TODO: change layerout from BNSD to TND.`,
  `input_layout = "BNSD"`, `query = query.unsqueeze(2)`, `attn_mask = None`, with the int8 KV cache
  passed as a 5-D NZ view. BNSD makes the first axis the batch, so `batchSize = num_tokens`
  (`attention_v1.py:1576`), while `_get_fia_params` still returns
  `actual_seq_lengths_kv = attn_metadata.seq_lens_list` for every decode state (`:1210`, `:1220`,
  `:1231`) — one entry per request. Under spec decoding
  `decode_token_per_req = 1 + num_speculative_tokens` (`worker/model_runner_v1.py:631-636`), so the
  two counts differ by exactly that factor: 256 kv entries against 512 q rows is
  `num_reqs = 256` with `num_speculative_tokens = 1`, i.e. a 512-token capture size. Without spec
  decode the counts coincide, which is why the same report says C8 alone and eagle3 alone are fine
  and only the combination fails.
- Maintainer position in that thread: "C8 quantization with EAGLE3 is currently unsupported" and
  "vllm-ascend框架限制 … 当前暂时无法给出确切计划". The BNSD `TODO` is still present on `main`
  (`attention_v1.py:691-692`), so no release repairs it; the only workaround is to drop one half of
  the pair (spec decode, or the C8/KV-int8 quantized checkpoint).
- **The other verified cause is a padded dummy request**, and the tag's own metadata builder
  documents it (`attention_v1.py:319-343`): flashcomm1/SP or cudagraph padding inserts a dummy
  request into `query_start_loc` to satisfy the TND constraint, which raises the q-derived batchSize
  by one while `seq_lens`/`block_table` buffers stay `max_num_reqs` and are silently truncated by
  `[:num_reqs_padded]`; the comment names "FIA then fails (error 561002) checking, in order, the
  `actualSeqLengthsKv` length and then the block_table row count against batchSize". The repair
  (`num_reqs_fia = len(actual_seq_lengths_q)` and padding both lists) is `419755b4f` (#10769),
  contained from `v0.23.0` — absent at `v0.23.0rc1`.
- The recorded serve line also sets `enable_sparse_sfa_c8` / `enable_sparse_li_c8`; those gate the
  DSA sparse-C8 and LightningIndexer caches
  ([c8-dsa-deployment-flags.md](c8-dsa-deployment-flags.md)) and do **not** select this path — the
  gate is the quant config's `enable_c8_quant` (`:434-435`). Grepping the sparse flags here leads
  away from the failure.
- Next check for a new report: work out which count each number is (requests vs q rows in the
  layout in force: `hidden_states.shape[0]`/TND q rows vs the unsqueezed BNSD first axis), then
  check whether `--speculative-config` and a KV-int8 (C8) quantized checkpoint are both enabled. The
  same message with a one-row difference points at the padding class instead.

## Unresolved: second-request eagle3 failure surfacing in the sampler (`507018`, issue #4771)

`v0.11.0rc2` + Qwen2.5-32B-Instruct with its own eagle3 draft, TP=2, `num_speculative_tokens: 5`:
the first request succeeds, the second dies with `507018` reported from
`vllm_ascend/sample/sampler.py:28` (`1 <= int(k.max()) <= 1024` in `_apply_top_k_top_p`). That
`int()` on a device tensor is a host synchronisation point (verified at that tag), so it surfaces
an earlier asynchronous device fault; the device block names an AICPU failure
(`kernelName=ScatterElements`, `errcode 21008`) *and* a FlashAttention tiling reject
(`When layout is TND, queryT(1) must be equal to the last element of actualSequenceLengthQ`,
`CheckFAISeqlenDataInTND`) — two different messages, and which one is causal was not established
here. The thread ends with a maintainer "resolved in the latest version, please use the latest
release or rc" and a stale close, so it supplies no root cause. Keep it only as a signature map;
the next check is `ASCEND_LAUNCH_BLOCKING=1` (or the plog) to move the report off the sampler frame.
