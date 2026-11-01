# Full-graph reports: `FULL_DECODE_ONLY` crashes, `FULL` with PCP or silently downgraded on hybrids, legacy `full_cuda_graph` hangs

Covers reports whose serve line carries
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` and which die with
`EngineDeadError` / worker crash / graph-capture failure, plus the pre-`cudagraph_mode` switch
`--compilation-config '{"full_cuda_graph": true}'` (the `full_cuda_graph` section below) whose
failure shape is a **hang**, not a crash. Both are the same class: a mode whose model/hardware
support is scoped by the reported tag's own release notes.

Plain `cudagraph_mode: FULL` is a third, different mode: with PCP it raises in the full-graph
params update (`decode_meta` is `None` on a prefill-only step), so do not answer a `FULL` report
with the `FULL_DECODE_ONLY` signatures — see the PCP section below. One `FULL` form has no
error at all: the mode is silently downgraded for hybrid linear-attention (GDN/Mamba) models, in
the section after the PCP one. If the failure is **first-request only**
(later requests normal, or a wrong/empty first answer) go to
[first-request-only-output.md](first-request-only-output.md) first: there the graph is an amplifier
and the trigger is `enable_sleep_mode` or a `--language-model-only` warm-up mismatch, not the
capture/pad defects below. Recorded instances: two Qwen3-MoE
reports (`vllm-ascend#4326` at `v0.11.0rc1`; `#5076` at `v0.12.0rc1`, Qwen3-Coder-480B-A35B,
DP4×TP8 + EP, multi-node). Neither is a collective/deployment defect: `#5076` is
reporter-confirmed fixed by upgrading to `v0.13.0rc1`, while `#4326` closed stale with only a
generic "resolved in the latest version" comment, so there the confirmed part is the flag removal.

For a Gemma3/Gemma4 report, read [gemma-family.md](gemma-family.md) first: its Gemma4 entry
points (A5 from `v0.23.0rc1`, A2/A3 graph execution from `v0.24.0rc1`) decide whether a
`FULL_DECODE_ONLY` crash at that tag is a support gap before the mode signatures below apply.

## Fast path

1. Read the thread and the tag's own `docs/source/user_guide/release_notes.md` **Known Issues**
   before any source read: this mode's model support is release-note scoped (below), and the
   reporter often already states the working config (`after removed --compilation-config ...
   it will work`).
2. Find the **worker-side** frame. The pasted API-server stack (`serving_chat` →
   `async_llm` → `core_client.get_output_async` → `EngineDeadError`) is always secondary;
   only a worker frame names a class. If the body has no worker frame (the traceback lives in
   an unfetched attachment), the answer is the support boundary + the reporter-confirmed
   workaround, not a code pin.
3. Match the worker frame against the sections below — plain `FULL` with PCP first (a prefill-only
   step, `decode_meta` `None`), then signatures A/B/C — and pin the tag with
   `git merge-base --is-ancestor` and `git describe --contains` (not `sort -V`).
4. Answer with: mode-support status at the reported tag → the fixing release (or "drop the
   flag") → the applicability test (`--enforce-eager` disables capture; removing the flag is
   the reporter-confirmed workaround).

## `cudagraph_mode: FULL` with PCP: `decode_meta` is None (issue #9507)

`FULL` is not a synonym for `FULL_DECODE_ONLY`. At vLLM `v0.18.0`,
`vllm/config/compilation.py:51-61` defines `NONE=0`, `PIECEWISE=1`, `FULL=2` and
`FULL_DECODE_ONLY = (FULL, NONE)` — the tuple form is per routine (full on decode, plain on the
mixed/prefill routine), while plain `FULL` is one routine for every batch. So with
`--compilation-config '{"cudagraph_mode": "FULL"}'` a pure-prefill step also takes the full-graph
params update. Chain at vllm-ascend `v0.18.0` (every pasted frame matches this tag exactly; see
[revision-pinning.md](revision-pinning.md) for the pin test):

| site | statement |
| --- | --- |
| `vllm_ascend/worker/model_runner_v1.py:1788` | gate `forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not forward_context.capturing and not self.use_sparse` |
| `vllm_ascend/worker/model_runner_v1.py:1796` | `update_full_graph_params(...)` |
| `vllm_ascend/compilation/acl_graph.py:233` | `impl_cls.update_graph_params(...)` |
| `vllm_ascend/attention/context_parallel/attention_cp.py:218-225,245` | `decode_metadata` is built only `if num_decodes > 0`, otherwise `decode_meta=None` |
| `vllm_ascend/attention/context_parallel/attention_cp.py:324` | `attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]` → `AttributeError: 'NoneType' object has no attribute 'num_computed_tokens_of_pcp_dcp'` |

The raising step is therefore one with `num_decodes == 0`; a mixed batch that still decodes has
`decode_meta` and does not raise. This is a topology/mode support boundary, not a missing guard to
patch per release: the maintainer answer in the thread is "Currently, the FULL model in graph mode
does not support the PCP feature. Only the FULL_DECODE_ONLY mode is supported", and the same
unguarded dereference survives at `:361` in `v0.19.1rc1` and `:371` in `v0.20.2rc1`, so upgrading
does not remove it.

- Workaround: `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`; thread-confirmed (the
  same curl then returns a completion instead of a worker `AttributeError`).
- Second symptom after the switch: the answer degenerates into a repetition loop (the same Chinese
  sentence repeated to `max_tokens` at `temperature 0.7`, all output counted as
  `reasoning_tokens`). The maintainer's only advice was "Disable asynchronous scheduling and then
  start the service. `--no-async-scheduling`"; the reporter never confirmed it and the issue closed
  stale, so treat it as an untested candidate and route the loop to
  [sampling-and-output-quality.md](sampling-and-output-quality.md).
- Do not fold this into the `FULL_DECODE_ONLY` signatures below: same file, different mode string,
  different raising statement (a params update, not a capture or decode-pad failure).

## `FULL` accepted but never captured: hybrid linear-attention (GDN/Mamba) models

An "is `FULL` supported for this model" report has no traceback: the server starts, the flag is in
the serve line, and the reporter observes that **prefill is not in the graph** and that gaps remain
between operators. Answer it from the attention-backend capability contract, not from a crash
signature.

- The contract at vLLM `v0.21.0`: `vllm/v1/attention/backend.py:499` `class AttentionCGSupport` with
  `ALWAYS=3`, `UNIFORM_BATCH=2`, `UNIFORM_SINGLE_TOKEN_DECODE=1`, `NEVER=0` (:510);
  `docs/design/cuda_graphs.md` states that a hybrid model takes the **minimum** capability across
  its attention backends and that the requested mode is downgraded to the best fit. The same doc's
  table lists `Mamba attention` as `UNIFORM_SINGLE_TOKEN_DECODE` and "unlisted backends are `NEVER`".
- The linear-attention backend implements exactly that:
  `vllm/v1/attention/backends/linear_attn.py:50`
  `_cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` (decode-only graphs).
  Qwen3.5 is such a hybrid — `vllm/transformers_utils/configs/qwen3_5.py:88-94` builds `layer_types`
  from `full_attention_interval` (default 4): `linear_attention` except every 4th `full_attention`
  layer — so the model's minimum capability is decode-only.
- The downgrade is upstream, in `GPUModelRunner._check_and_update_cudagraph_mode`
  (`vllm/v1/worker/gpu_model_runner.py:6456` at `v0.21.0`, which takes the min and calls
  `CompilationConfig.resolve_cudagraph_mode_and_sizes`, `vllm/config/compilation.py:1301`). It logs
  and rewrites the mode:

  ```text
  CUDAGraphMode.FULL is not supported with <Backend> backend (support:
  AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=FULL_DECODE_ONLY
  ```

  (`FULL_AND_PIECEWISE` instead when `splitting_ops` contain attention), `logger.warning`.
- Do not look for a plugin-side table: `vllm_ascend/platform.py` keeps a full mode by clearing
  `splitting_ops` (`:576` `has_full_cudagraphs()` branch) and falls back to `NONE` only for modes it
  does not know (`:582` "not support on NPU. falling back to NONE"); `vllm_ascend/worker/
  model_runner_v1.py:4768` `_check_and_update_cudagraph_mode` merely calls `super()` and then seeds
  `set_graph_params(capture_sizes)`. The only plugin override that narrows the mode is
  `vllm_ascend/_310p/model_runner_310p.py:648`. Ascend's own
  `AscendAttentionMetadataBuilder.get_cudagraph_support` returns `ALWAYS`
  (`vllm_ascend/attention/attention_v1.py:260`), so the base backend is *not* what limits the model —
  the GDN/linear-attention group is.
- Answer shape: not a defect and not fixable by a flag. For this model class prefill is never
  graph-captured (the same holds upstream, including on GPU); the observed inter-operator gaps are
  the eager prefill. Confirm by grepping the served log for the warning above. If decode-only
  capture is the goal, request `FULL_DECODE_ONLY` explicitly and expect prefill to stay eager;
  `PIECEWISE` is the alternative.

**`ASCEND_LAUNCH_BLOCKING=1` cannot be combined with any graph mode.** The same platform function
raises `ValueError("ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. Please unset
ASCEND_LAUNCH_BLOCKING or set it to 0 ...")` (`vllm_ascend/platform.py:595` at `v0.21.0rc1`), so
every request for a synchronous stack (async `aclnn` faults in
[log-signatures.md](log-signatures.md)) or for a plog-based triage must also drop
`--compilation-config`/add `--enforce-eager` in the same rerun.

## Support boundary (release-note scoped)

`v0.12.0rc1` release notes, `### Known Issues`
(`docs/source/user_guide/release_notes.md` at that tag):

> DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY graph mode. We'll fix it in next
> release. [#4990]

The same file's `### Highlights`/`### Core` for that release recommends the mode anyway
("[Experimental] Full decode only graph mode is supported now ... we suggest to enable it by
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` in most case. Let us know if you
hit any error"), so a 0.12.0rc1 Qwen3/DeepSeek report can be simultaneously
"recommended-by-docs" and "unsupported". The note is carried into the `v0.13.0rc1` file only
inside the `v0.12.0rc1` section — `v0.13.0rc1`'s own Known Issues does **not** repeat it.
It holds for the `v0.13.0` final tag too: `:287` there is still inside `## v0.12.0rc1 - 2025.12.13`
(that header is `:248`), so the DeepSeek line answers a `v0.13.0` report as well — read the
enclosing `##` header rather than the line number.

Lineage of that entry, for a report that sits *before* it: it is added by `42ceaf08a` ("add release
note for 0.12.0 (#4995)"), and `grep -c "doesn't work with FULL_DECODE_ONLY"` on the tag's own file
is `0` at `v0.11.0rc1` and `v0.11.0rc2`, `1` at `v0.12.0rc1`/`v0.13.0rc1`. `v0.11.0rc2`'s file has
**no** `## v0.11.0rc2` header at all — its newest section is `## v0.11.0rc0 - 2025.09.30` — so "the
reported tag's own section" for an rc2 report is rc0's, and a missing header is not a missing file.
Third recorded instance of the class: `#5188` (`v0.11.0rc2`, A3 910, CANN 8.3.RC2, Qwen3-235B-A22B,
`-dp 4 -tp 4 --no-enable-expert-parallel`, `FULL_DECODE_ONLY` with 19 explicit capture sizes
(`{1,8,16,,{32..512..32}}` expands to a valid list — bash word-splitting drops the empty element) and
`--max-num-seqs 512`); title says "fullgraph hang", the body is the serve line only, and the thread
is one maintainer "This issue has been resolved in the latest version" plus the stale/close bots —
no mechanism. So the claimable answer there is the one-tag-ahead support note plus the code's own
workaround below.

Maintainer corroboration in `#4326` (2025-12-15, on a `v0.11.0rc1` Qwen3-30B-A3B report):
"DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY graph mode on main branch
currently. We'll fix it in next release." A later generic bot comment ("resolved in the latest
version") names no commit — do not turn it into a release boundary.

Earlier eras, for context: `v0.10.x` notes say "[Experimental] Graph mode `FULL_DECODE_ONLY` is
supported now!" (#2128), and `v0.9.1` Known Issues has "When running MoE model, Aclgraph mode
only work with tensor parallel. DP/EP doesn't work in this release." That DP/EP note is **not**
in `v0.11.0rc1`'s or `v0.12.0rc1`'s Known Issues, so do not cite it against a 0.11/0.12 report.

## Signature A: decode-pad shape mismatch (`query_start_loc`)

```text
File ".../vllm_ascend/worker/model_runner_v1.py", line 1035, in _prepare_inputs
    self.query_start_loc.gpu[num_reqs + 1:num_reqs_padded + 1] = fill_values
RuntimeError: The expanded size of the tensor (63) must match the existing size (6151)
              at non-singleton dimension 0.  Target sizes: [63].  Tensor sizes: [6151]
```

- The frame is the **uniform-decode padding block** that simulates upstream
  `GPUModelRunner`'s decode padding for full-graph replay. It exists only on the 0.12+ line:
  added by `5b179c53f` (#4706, 2025-12-10, "Implement padding logic for `query_start_loc` in
  `NPUModelRunner` to support uniform decode in full graph mode"), first tag `v0.12.0rc1`;
  `git show v0.11.0rc1:vllm_ascend/worker/model_runner_v1.py | grep -c num_reqs_padded` prints
  `0` (`0` at `v0.11.0rc2` too), so the same report at `v0.11.0rc1`/`rc2` cannot be this defect.
- Fixed on the way to `v0.13.0rc1` by two commits, both ancestors of `v0.13.0rc1`
  (`git merge-base --is-ancestor <sha> v0.13.0rc1` = yes), `git describe --contains` =
  `v0.13.0rc1~162` / `v0.13.0rc1~142`:
  - `43d974c6f` (#5134, 2025-12-17) deletes the device-side slice assignment: the pad now
    writes `self.query_start_loc.np[...]` and calls `self.query_start_loc.copy_to_gpu(num_reqs_padded + 1)`.
  - `ff3914e31` (#5164, 2025-12-18) narrows the guard from
    `decode_mode() == FULL and uniform_decode` to
    `... and self.uniform_decode_query_len <= num_input_tokens <= max_decode_tokens`
    (`max_decode_tokens = self.scheduler_config.max_num_seqs * self.uniform_decode_query_len`).
    Its PR body states why the previous bound (`num_input_tokens <= self.cudagraph_batch_sizes[-1]`)
    was invalid: that value "is actually not the max number of tokens to be padded in
    `FULL_DECODE_ONLY` mode, much larger instead".
- Post-fix shape at `v0.13.0rc1`: `model_runner_v1.py:1031-1043` (guard at `:1032`, host write +
  `copy_to_gpu` at `:1039-1043`). Pre-fix shape at `v0.12.0rc1`: guard at `:1021`, raising
  assignment at `:1035`.
- Reporter confirmation: asked to try `0.13.0rc1`, the `#5076` reporter replied
  "Great! It has been fixed."
- Why it fires only on some runs is a **candidate**, not a verified mechanism: the padding runs
  only on a uniform decode step (`max_num_scheduled_tokens == uniform_decode_query_len` and every
  request decoding), which a high-concurrency benchmark of equal-length prompts
  (`--parallel 200`, 7168-token prompts) reaches immediately after prefill. The exact
  `63` vs `6151` arithmetic is not derived from the pasted log.
- **Line drift trap:** the `#5076` maintainer wrote "error lines do not exist in the main
  branch" because the reported `:1035` is the `v0.12.0rc1` line. That is normal drift, not a
  reporter error and not evidence the report is invalid — pin the frame per tag
  (`git show <tag>:<path> | grep -n '<statement>'`).

## Signature B: graph capture fails with FlashComm1 enabled

`64669c424` (#5200, 2025-12-22, first tag `v0.13.0rc1`) adds, in
`vllm_ascend/attention/sfa_v1.py` at `v0.13.0rc1` (`AscendSFAMetadataBuilder.__init__`, `:120`,
`enable_sfa_cp` computed at `:148`, assert at `:151-155`):

> `assert not (self.enable_sfa_cp and ... cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY),
> "FlashComm1 is not compatible with FULL_DECODE_ONLY. Please set graph_mode to 'piecewise' or
> disable FlashComm1."`

The PR body is the reusable statement: "Currently, Flashcomm1 and FULL_DECODE_ONLY are
incompatible. When both features are enabled, graph capture errors occur without clear error
messages", and the adaptation was postponed, so the release only converts the silent failure
into an assert. Scope: `enable_sfa_cp` is `enable_sp() and hasattr(hf_config, "index_topk")`,
i.e. DeepSeek sparse-attention models — a Qwen3-MoE report that sets
`VLLM_ASCEND_ENABLE_FLASHCOMM=1` with `FULL_DECODE_ONLY` does **not** trip this assert and would
still fail silently. Triage action either way: ask for / test the run with one of the two
switched off; `piecewise` capture is the recommended pairing.

## Signature C: the `TransposeKvCacheByBlock` / `107030`/`587011` cluster in `FULL_DECODE_ONLY` (GLM-5 W4A8)

Two stages emit this cluster: the PD decode-rank dummy run with an MTP drafter, and an in-serving
graph replay on a single node with no speculative decoding. Match on the log cluster, not on PD or
MTP — neither is required.

Signature (`vllm-ascend 0.16.0rc1`, model `GLM-5-w4a8-mtp-QuaRot`, 1P1D with P TP8 DP2 / D TP8 DP2,
`MultiConnector` KV connector, `--max-num-batched-tokens 4096`,
`--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'`,
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`):

```text
worker.py:564 execute_dummy_batch -> model_runner_v1.py:2417 _dummy_run -> :1746 _model_forward
  -> compilation/acl_graph.py:199 torch.npu.current_stream().synchronize()
RuntimeError: synchronize ... NPU function error ... error code is 507011
```

preceded in the same log by `[InitOpInfoLib][InitOpKernel] Node[TransposeKvCacheByBlock]: failed to
initialize unknown shape dtype and format ... input_desc_info input2`, `synchronized memcpy failed,
kind = 1, runtime result = 107030` and `EE9999: rtMemcpy execution failed, reason=the current capture
mode does not support this operation`.

A second report (`vllm-ascend#7448`, `v0.17.0rc1`, GLM-5 W4A8 on 910C, single node,
`--data-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel`, **no**
`--speculative-config` and no KV connector, `--max-num-seqs 4`, capture sizes 4-32) carries the same
cluster on the in-serving **replay** path: `worker/model_runner_v1.py:1794 _model_forward` ->
`compilation/acl_graph.py:281 __call__` -> `torch.npu.current_stream().synchronize()`, with device
error `587011` (not `507011`) plus `MTE instruction ... DDR address ... out of range`. So this block
also appears without PD, without MTP and without any `execute_dummy_batch` frame.

- **Pin on the model_runner frame, not the `acl_graph.py` line.** At `v0.17.0rc1`
  `model_runner_v1.py:1794` is `hidden_states = self.model(` inside `_model_forward` (exact match),
  while that tag's only `current_stream().synchronize()` in `acl_graph.py` is `:201` (guarded by
  `runtime_mode != FULL or not is_draft_model or not use_eagle`), not the pasted `:281`. The offset
  means the installed tree is not byte-identical with the tag; one exact frame plus the statement
  text is the pin, so do not drop the report over the line number
  ([revision-pinning.md](revision-pinning.md)).

- **The `TransposeKvCacheByBlock` ERROR block is documented noise, not the cause.** The fix PR's own
  body (#7235, `4e62a2ae1`, "fix TransposeKvCacheByBlock op error report in plog") says these errors
  appear "when vllm launching, though it doesn't influence the running of vllm, but ERRORs will be
  very confused in debug". A report titled by those lines (mooncake / KV "trans..." errors) therefore
  does not point at the connector: read the worker frame and the device error instead.
- Class for the MTP-drafter form: full-decode-only padding of the **drafter's** attention
  metadata. #7290 (`a6f6e919e`,
  "[main][bugfix] Fixed the problem that eagle3 will crash in FULL_DECODE_ONLY") states the
  mechanism — in `FULL_DECODE_ONLY`, `num_tokens` must be padded to a `cudagraph_capture_sizes` value
  and `actual_seq_lengths_q` is padded, but the drafter's `seq_lens_list` keeps length `num_reqs`
  (one short, kernel exception) and the target's list is padded with corrupted tail data. Its title
  says eagle3 while this report's method is `deepseek_mtp`, so treat "the drafter's padded metadata"
  as the shared scope and confirm the list at the report's tag before quoting it as *the* fix.
- Release: `git tag --contains` on `a6f6e919e`, `cbf46fad3` and `4e62a2ae1` each starts at
  `v0.18.0rc1` (and `v0.18.0`), so `v0.16.0rc1` -> `v0.18.0rc1` is the upgrade to quote. From
  `v0.18.0rc1` the tag's own release notes carry the matching known issue: "GLM5 has an issue in the
  2-node PD mixed deployment scenario where inference may hang when concurrency exceeds 8 (fixed in
  PR #7235 #7290)" (`adc57c595`, #7436).
- **The thread's fix set is a maintainer claim, not a mechanism verified for every config.**
  `#7448` is labelled `duplicated` and closed with one comment redirecting to `#7401`, whose
  maintainer comment names `#7235`/`#7290`/`#7460`. `#7290`'s mechanism is the *drafter's* padded
  metadata and `#7448` has no speculative config, so for that report the cause stays a class
  (`FULL_DECODE_ONLY` capture/replay of a W4A8 MoE), not the named defect. Same-release candidates
  that do apply to a plain full-decode run, all first in `v0.18.0rc1` (`git tag --contains`):
  `cbf46fad3` (`#7460`) also requires `compilation_config.cudagraph_mode == FULL` before
  `_pad_query_start_loc_for_fia` collapses `num_reqs_padded` to `num_reqs`
  (`model_runner_v1.py:556-574` at `v0.18.0rc1`; PR body: the old value caused "accuracy degradation
  in Qwen3-Next"), and `67aad1fce` (`#7506`) pads GDN metadata with `0` instead of `-1`, because
  FullGraph capture froze the `ssm_state_indices >= 0` branch and the `-1` offset overwrote the SSM
  cache (NaN); `6ddfc4131` (`#7571`) covers MTP overlaid on full decode for DSV3.1 C8.
- **The reporter's own tag documents a second, quantitative workaround that is not in the fix set.**
  `docs/source/user_guide/release_notes.md` in the `## v0.16.0rc1` section's `### Known Issue` (line
  92 in that file; absent at `v0.15.0rc1`) says "MTP with large EP configurations may cause graph
  capture buffer overflow. This is a bug need to fix in vLLM" and gives the workaround
  `--compilation-config '{"max_cudagraph_capture_size": N}'` with
  `N = max_concurrency x (1 + num_speculative_tokens)`. Later files still contain the line because
  sections accumulate, so read its `##` header before treating it as current. It is a *documented
  configuration* for the MTP + large-EP + capture combination, not a proven cause of any one
  traceback: quote it as the next thing to test (size the capture list to the target concurrency
  instead of the default) alongside the first tag containing #7235/#7290/#7460, and say the
  thread repeats the failure past `v0.18.0rc1` so the upgrade alone did not close it.
- **Workaround, not thread-confirmed:** the raising synchronize exists only on the graph path, so
  `--enforce-eager` or dropping `--compilation-config` (or `piecewise`) removes this class
  (code-path inference).
- Do not merge this with the vLLM `v0.26.0` DP idle-rank `507034` timeout
  ([log-signatures.md](log-signatures.md)): that crash also passes through `execute_dummy_batch`, but
  its trigger is uneven DP traffic in the serving loop and it has no `FULL_DECODE_ONLY` capture or
  `TransposeKvCacheByBlock` context. Nor is it the steady-state `107020`/`107027` replay-time
  `OutOfMemoryError` wrapper in the same reference: Signature C's dummy-run form fails during
  start-up capture and its replay form fails on a served decode step, both inside `FULL_DECODE_ONLY`
  graph code. Read the error code and the stage, not only the frame.
- Boundary: the same thread reports the failure again on `0.18.0RC1` and `0.18.0` (same worker
  crash), so the fix set is not confirmed for this configuration — quote the first tag that contains
  the commits and the recurrence in the same breath. The `0.20.2.rc1` follow-up is a **different**
  class and is not a recurrence: 2P1D A2, DeepSeek-V4-Flash, log
  `ascend_direct_transport ... Transfer slice failed with status: 103902` ->
  `mooncake_hybrid_connector.py:551` `Mooncake transfer failed, ret: -1` (KV transfer; route to
  [pd-disaggregation.md](pd-disaggregation.md)). A third commenter on the same model in a
  single-node A3 DP2xTP8 run crashes earlier, at `gpu_model_runner.py:1050
  _get_valid_sampled_token_count` under `_update_states` — a sibling signature; ask which traceback
  the reporter holds before assuming capture/replay.

## Signature D: `107030` alone, from a custom op's host-tiling `aclrtMemcpy` (MLA/DSA models)

```text
[ERROR] RUNTIME(212914,):... [api_c.cc:1097]212914 rtMemcpy:ErrCode=107030,
        desc=[the current capture mode does not support this operation], InnerCode=0x703001c
[ERROR] RUNTIME(212914,):... [error_message_manager.cc:61]212914 FuncErrorReason:rtMemcpy execution
        failed, reason=the current capture mode does not support this operation
```

Nothing crashes here — no `EngineDeadError`, no worker exit, no `507011`/`587011` — which is what
separates this from Signature C above.

- **The reported tag's own release notes already carry this shape; read that line before any
  `csrc/` read.** `v0.23.0` `docs/source/user_guide/release_notes.md:143`, inside that tag's own
  `## v0.23.0`: "In graph mode,
  `torch.ops._C_ascend.mla_preprocess` and `torch.ops._C_ascend.batch_matmul_transpose` can emit an
  `rtMemcpy` 107030 error in plog; the reported error does not affect inference." The report body
  says the same ("does not affect actual usage"), so the answer is: documented at the tag, harmless,
  plus the mechanism and the release that removes the ops.
- Mechanism, read in that tag's `csrc/`: both are custom ops whose **host tiling** writes into a
  static device buffer with a synchronous copy, indexed by shape —
  `csrc/mla_preprocess/op_host/mla_preprocess.h:729` copies `tilingSize` bytes at
  `tilingSize * (N - 1)` with `ACL_MEMCPY_HOST_TO_DEVICE` into a buffer of
  `tilingSize * MAX_SUPPORT_TOKEN_NUMS`, and
  `csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.h:109` does the same at
  `tilingSize * (m - 1)` over `MAX_CAPTURE_NUM = 1024`. Inside `torch.npu.graph` capture the stream
  is in capture mode, a synchronous H2D is refused, and the runtime logs 107030 instead of
  performing it.
- Why inference survives is the report's statement plus an inference: the buffer is tiling metadata
  for a fixed shape, not tensor data, and the replay reuses it. Candidate, not measured — do not
  upgrade it to "the copy is skipped harmlessly" without a run.
- Call sites, to confirm the model family rather than guess: `vllm_ascend/attention/mla_v1.py`
  (`mla_preprocess` through `device/device_op.py:349`,`:419`; `batch_matmul_transpose` carries a
  weight-nz TODO at `:957`) and `vllm_ascend/attention/sfa_v1.py:1352-1357` (gated on
  `hasattr(torch.ops._C_ascend, "batch_matmul_transpose")`). Both are MLA / DSA sparse-attention
  paths.
- Removal, by `git tag --contains`: `0f3d25f9f` (#15182) migrates `mla_preprocess` to the two-stage
  ACLNN interface (`EXEC_NPU_CMD(aclnnMlaPreprocess, ...)` plus an `OpDef`, so the adapter no longer
  copies tiling by hand), first tag `v0.30.0rc1` (and `v0.30.0`); `dd24be2fe` (#17141) deletes
  `batch_matmul_transpose` and re-points `mla_v1.py`/`sfa_v1.py` at aclnn `TransposeBatchMatMul`,
  first tag `v0.29.0.rc1`. The migration PR's body states the mechanism verbatim ("the host tiling
  ... copies tiling data to device memory with synchronous `aclrtMemcpy` ... during `torch.npu.graph`
  capture, that host-to-device copy is unsupported and reports runtime error `107030`"), so quote it
  instead of re-deriving from the kernel.

## When the `107030` *does* kill the process: `v0.13.0` DeepSeek-V3.2 + MTP + `layer_sharding` (recorded `#8587`)

Signature (`v0.13.0` plugin and vLLM, DeepSeek-V3.2 w4a8 QuaRot, 2 nodes × 8 cards (910B), DP2×TP8 + EP):

```text
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'
vllm bench serve --input-len 1024 --output-len 128 --max-concurrency 4
```

All 8 workers of the DP-rank-1 node die together; concurrency 1-2 complete, and removing the
`--compilation-config` removes the crash. Worker side: `EE9999: rtMemcpy execution failed,
reason=the current capture mode does not support this operation`, `synchronized memcpy failed,
kind=1, runtime result=107030`, `The DDR address of the MTE instruction is out of range`, under
`sample_tokens -> propose_draft_token_ids -> mtp_proposer._propose -> deepseek_mtp.py forward ->
acl_graph.__call__ -> aclnnKvRmsNormRopeCache -> rtMemcpy`. Two differences from Signature D above
decide the answer: the service dies (so no "does not affect inference" reading applies), and the
MTE-DDR line means a device memory fault, not the refused host-tiling copy.

- **Do not read the named op as a plugin symbol.** `aclnnKvRmsNormRopeCache` is a torch_npu ACLNN
  op reached from `vllm_ascend/attention/mla_v1.py` (`torch_npu.npu_kv_rmsnorm_rope_cache`,
  `v0.13.0:1098`, `:1126`); the string `KvRmsNormRopeCache` has **no** hit anywhere in the
  `v0.13.0` plugin tree (it appears only in later `mla_v1.py` comments, e.g. `main:2274`). Grep
  `torch_npu`/`mla_v1.py` instead of concluding the op does not exist at the tag.
- **Maintainer (`yiz-liu`, collaborator) on the mechanism:** "I don't think the copy is the root
  cause, since the graph is static and will not add a copy out of nowhere at run-time. More likely
  there's another error before that." So the `rtMemcpy` frame is a victim surface here, and the
  deliverable is the thread's configuration workaround, not a code pin.
- **Thread workaround for this configuration:** a contributor (`Nagisa125`, 2026-05-09) names
  `layer_sharding` in a mixed configuration as the suspect and recommends "use version 0180 and
  disable layer_sharding at the same time"; if the fault persists, they read it as insufficient
  device memory and suggest lowering `--gpu-memory-utilization`. Reporter-confirmed close
  (`lingxling`, 2026-06-05): "We have switched to v0.18.0 and disabled layer_sharding ... the issue
  no longer occurs".
- **Answer shape:** the tag's own Known-Issues line (Support boundary above) + drop
  `--compilation-config` (reporter-confirmed, the mode is the feature under boundary) + the
  `v0.18.0`/`layer_sharding` move. Say the two changes were made together, so neither factor is
  isolated, and label the in-graph copy as *not* the cause per the maintainer. Do not claim a
  fixed commit: no PR is named in the thread.

## Legacy `full_cuda_graph: true` (0.9.x line): documented as unsupported, and where it hangs

Before `cudagraph_mode`, the full-graph switch was the upstream bool
`--compilation-config '{"full_cuda_graph": true}'`. Read this section instead of Signature A/B when
the report names that flag with a hang, an `aclgraph`/task-update stack, or no traceback at all.

- **Release-note status at the reported tag.** `v0.9.1`'s own
  `docs/source/user_guide/release_notes.md` carries this only in the rc sections: `:77` (inside
  `## v0.9.1rc3`) "`test_aclgraph.py` failed with `"full_cuda_graph": True` on A2 (910B1) [#2182]"
  and `:179` (inside `## v0.9.1rc2`) "Full graph mode support are not yet available for specific
  hardware types with full_cuda_graphenable. [#2182]". The final `## v0.9.1` section's Known Issues
  (`:42-47`) does **not** repeat it, so quote the rc section for a v0.9.1 report — same
  section-vs-file trap as the `FULL_DECODE_ONLY` note above.
- **`graph_task_update_end` is the plugin's replay-time parameter patch, not CANN.** One call site
  at `v0.9.1`: `vllm_ascend/compilation/piecewise_backend.py:157` inside `update_attn_params`
  (`graph_task_update_begin` `:145`, `torch_npu._npu_paged_attention` `:148`,
  `graph_task_update_end` `:157`, per-layer `ExternalEvent.record` `:159`), on the
  `self.update_stream` that exists only for this mode (`:103-105`). Capture side:
  `vllm_ascend/attention/attention_v1.py:427-439` registers each paged-attention op as a graph task
  (`graph_task_group_begin`/`graph_task_group_end`, appended to `graph_params.handles[num_tokens]`
  and `attn_params[num_tokens]`). A py-spy stack in `graph_task_update_end` therefore points at
  this pair of files; there is no deeper CANN frame to hunt.
- **The code's own workaround is the first thing to quote.** `vllm_ascend/utils.py:178-204`
  (`update_aclgraph_sizes`) prints, whenever the flag is set: "WARNING: You have enabled the *full
  graph* feature. This is an early experimental stage and may involve various unknown issues. A
  known problem is that capturing too many batch sizes can lead to OOM (Out of Memory) errors or
  inference hangs. If you encounter such issues, consider reducing `gpu_memory_utilization` or
  manually specifying a smaller batch size for graph capture." The same function truncates capture
  sizes to `max_num_seqs` (`:185-188`), so capture cost tracks `--max-num-seqs`, not
  `--max-model-len`.
- **Two silent side effects to state before accepting a "full graph is broken/slow" framing.**
  Upstream `vllm v0.9.1:vllm/config.py:4467-4473` disables cascade attention and sets
  `cache_config.enable_prefix_caching = False` whenever `full_cuda_graph` is on, so any prefix-cache
  or long-prompt measurement taken with the flag is taken without prefix caching. And
  `vllm_ascend/platform.py:239-242` adds `vllm.unified_ascend_attention_with_output` to
  `splitting_ops` only when the flag is **off**: with it on, attention sits inside the single graph
  and `piecewise_backend.py:176-179` falls back to the general compiled graph for every step whose
  `attn_state != DecodeOnly`, i.e. prefill never replays the captured graph.
- **Flag interaction.** Upstream cannot combine it with `splitting_ops`
  (`vllm/config.py:4164-4167` raises `ValueError: full_cuda_graph cannot be used together with
  splitting_ops`), and its docstring (`:3961-3965`) says it "may provide performance benefits for
  smaller models" — the opposite audience from a 32B model, which is worth saying when the report
  is a 32B deployment expecting a speedup.
- **Evidence boundary.** The recorded `#4180`-class report is a screenshot with no readable stack,
  and its thread closed stale (`not_planned`) after one contributor asking for feedback and one
  "why not use a newer image" comment: no maintainer root cause exists. Claimable answer =
  documented-known-issue at that tag + drop the flag (or shrink capture sizes / `--max-num-seqs`)
  + the graph-task update path above. A per-layer length mismatch between the captured
  `handles[runtime_shape]` list and the replay loop is a **candidate** consistent with the code
  shape, not demonstrated in any recorded run.

## Capture-time caveats that are documentation, not defects

At `v0.11.0rc1` unless the bullet names another ref (all verified in the named tag's source):

- `vllm_ascend/compilation/acl_graph.py:216-223` — comment: "When using FULL_DECODE_ONLY, there
  are some rare bugs for FULL_DECODE_ONLY mode with GQA. This is triggered by getting workspace
  for `_npu_paged_attention` in torch_npu ... we use max_model_len to calculate max workspace in
  capturing. So additional `get_workspace` is added here to avoid such bugs." A workaround
  already in the code; a GQA attention fault here is not automatically a plugin regression.
- `vllm_ascend/platform.py:219-236` — the mode logs a red warning: full graph is "an early
  experimental stage", "capturing too many batch sizes can lead to OOM ... or inference hangs",
  and suggests reducing `gpu_memory_utilization` or smaller capture sizes.
- `platform.py:201-203` — `FULL_AND_PIECEWISE` is silently downgraded to `PIECEWISE`; `:237-242`
  — any other unsupported mode logs "`%s cudagraph_mode is not support on NPU. falling back to
  NONE`". So "I set a graph mode and nothing changed" needs the startup log line quoted before
  it becomes a bug.
- **The ACL-graph stream budget does not narrow a `FULL_DECODE_ONLY` capture list at these tags,**
  so an explicit long `cudagraph_capture_sizes` is used as given and the branch's own "too many
  batch sizes" warning is the code's prescription. At `v0.11.0rc2`
  `utils.py:370 update_aclgraph_sizes` (`MAX_CAPTURE_SIZE = 1800`, AIV formula) is called only from
  the `PIECEWISE` branch (`platform.py:223`); the `FULL_DECODE_ONLY` branch (`platform.py:224-241`)
  only logs. At `v0.18.0rc1` the same budget is `utils.py:424 update_cudagraph_capture_sizes`
  (`MAX_CAPTURE_SIZE = 1800`, `HCCL_OP_EXPANSION_MODE == "AIV"` selecting the other formula at
  `:506-526`) and is called only in the sequence-parallelism branch (`platform.py:317-335`,
  `tensor_parallel_size > 1 and not enforce_eager and enable_sp(...)`), which uniform-samples the
  list down to `max_num_batch_sizes` (`:560-575`). The log lines to ask for are `Calculated maximum
  supported batch sizes for ACL graph: N`, `Adjusted ACL graph batch sizes for <arch> ... a -> b
  sizes` and `No adjustment needed ...`: absent is the expected case for a non-SP
  `FULL_DECODE_ONLY` run at either tag, and present tells you which shapes were actually captured.

## Teardown noise in these bodies

`#4326` pastes, as "one maybe usefull plog":

```text
[ERROR] TUNE(...)[CANNKB] ... AttributeError: 'ForkAwareLocal' object has no attribute 'connection'
[ERROR] TUNE(...)[CANNKB] ... tbe/common/repository_manager/interface.py ... cann_kb_finalize
[ERROR] TUNE(...)[cann_kb_api.cpp:40][CANNKB] ... "Run CannKbFinalize Error!"
[ERROR] TEFUSION(...)[python_adapter_manager.cc:235] ... Finalize call CannKbFinalize failed. res = [6].
```

This is CANN's TBE knowledge-bank manager closing during **process exit** in a forked worker:
the manager's socket is already gone, so `copy.deepcopy(self.write_kb.keys())` reconnects and
fails. It is CANN-side, names no model or plugin frame, and in that report is timestamped
`02:10:21` — about 44 minutes **before** the pasted `02:54:31` server crash — so it cannot be the
trigger. Order the timestamps and treat it as teardown/exit chatter, like the HCCL free lines.

## Evidence boundaries

- `#5076`'s worker traceback exists only in the maintainer's quotation of the (unfetched)
  `error.log` attachment; the reporter's own pasted stack stops at `EngineDeadError`. The pin is
  still solid because the quoted line matches `v0.12.0rc1` exactly.
- Which of `43d974c6f` / `ff3914e31` fixed the reporter's run is not separated — both are in
  `v0.13.0rc1`; do not attribute the crash to one of them alone.
- `#4326` produced no commit-level answer in the thread (closed stale after the bot's generic
  "resolved in the latest version"): its worker-side failure mode is unknown, so only the
  support boundary and the reporter-confirmed flag removal are claimable. Candidate fixes on the
  `v0.12.0rc1`/`v0.13.0rc1` lines that touch this surface and are **not** linked to that thread:
  `58db21f56` (#4705, DP dummy-run padding), `5b1a7514e` (#4366, force_load_balance in aclgraph,
  PR body "Temporarily fix the oom issue"), `bc5b7a5fb` (#5397, MHA aclgraph runtime error),
  `64669c424` (#5200, signature B).
