# `qwen3_5_mtp` + `--enable-prefix-caching` on a hybrid GDN model: the unvalidated pair, with or without error text

## Report shape

A hybrid linear-attention family (Qwen3.5/Qwen3.6, `model_type` `qwen3_5`) served with
`--speculative-config '{"method": "qwen3_5_mtp", …}'` **and**
`--enable-prefix-caching`, where the title says only "同时开启mtp+prefix cache后，推理报错"
(**or names neither feature** — #8913's title was just `Qwen3.5-35B-A3B启动后发请求报错`, so
the pair is visible only in the serve line: read that, not the title)
and the **body carries no error at all** (recorded #9003: the body's whole bug section was
`<!-- Failed to upload "image.png" -->`, and `/workspace/attachments` was empty) — or the
error is one line of Chinese with no traceback — **or the body carries the whole
traceback**, in which case the frames decide the page and §*The `IndexError` form* below
is the answer, not the coverage verdict. The launch usually also has
`--async-scheduling`, `cudagraph_mode: FULL_DECODE_ONLY`, an explicit
`cudagraph_capture_sizes` list, a `--quantization ascend` w8a8 checkpoint and a
`--additional-config` stack.

**A `<details>` block whose entire content is a link is the no-error-text form, not a
traceback.** Recorded #8913 (vllm 0.19.1 + vllm-ascend v0.19.1rc1, bf16
`Qwen3.5-35B-A3B`, TP2 + `--enable-expert-parallel`, K=3, `--enable-prefix-caching`,
`--async-scheduling`, `FULL_DECODE_ONLY`) streamed three chunks (~4 tokens) and then
`{"error": {"message": "EngineCore encountered an issue. See stack trace (above) for the
root cause.", … "code": 500}}`, while its `<summary>详细报错</summary>` block held exactly
one line: `[errors.txt](https://github.com/user-attachments/files/27434752/errors.txt)`.
That is a `user-attachments` URL, unreachable from the case view, and
`/workspace/attachments` was empty — nothing there can be read as a frame. The
reporter's own next comment was 详细报错log好像没传上 followed by a **second**
user-attachments id (`…/27470681/…`), so the upload failed; the log was never withheld.
Answer that form as the no-error-text form (coverage verdict + ladder + ask for the file
and its tail) and say which form the paste is in — never infer a mechanism from the
missing text.

**Not every "GLM"/DeepSeek launch in this pair's shape is this page.** GLM-5/5.1 is
`model_type: glm_moe_dsa` — DSA/MLA sparse attention, no mamba layers — so
`mamba_cache_mode`, `mamba_block_size` and `preprocess_mamba` never enter it even with
`--enable-prefix-caching` on, and a corruption report there is
`../dsa-sparse-attention/glm-moe-dsa-tag-regression-corruption.md`. The crash surface
below needs a checkpoint that really has mamba/GDN state layers — the model class exposes
`get_mamba_state_copy_func`, and the traceback reaches `mamba_utils` at all.

**The deliverable for the no-error-text form is a coverage verdict plus an ablation
plan, and it is owed in the first message.** Nothing in the pinned tree can name the
traceback line for you; what the tree *can* state is whether the pair is covered, which
knob is redundant, and which single launch splits the lanes. Reading the KV-manager
source before sending that is the recorded failure mode (see the stop rule below).

## Verified at vllm-ascend v0.19.1rc1 (`da421af`) + vllm v0.19.1 (`b1388b1`)

Coverage — the pair is documented, never tested:

| anchor | what it says |
|---|---|
| `tests/e2e/nightly/single_node/models/configs/Qwen3.5-27B-w8a8-A3.yaml` | the only nightly row for this family: `TP2 / DP1`, `--max-num-seqs 32`, `--max-model-len 133000`, `--max-num-batched-tokens 8096`, `{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}`, `FULL_DECODE_ONLY`, `--async-scheduling`, and **`--no-enable-prefix-caching`** |
| `tests/e2e/multicard/4-cards/test_qwen3_5.py:75-81` (and `:106-112`) | `qwen3_5_mtp` `n=3` + `FULL_DECODE_ONLY` + `cudagraph_capture_sizes: [4, 8, 12, 16]`, no prefix caching |
| `tests/e2e/singlecard/spec_decode/test_v1_spec_decode.py:558,680` | where the tree *does* test spec decode it **hard-codes `enable_prefix_caching=False`** (dense Qwen3-8B with `eagle3`/`ngram`/`suffix`) — so no test anywhere runs the two features together |
| `docs/source/tutorials/models/Qwen3.5-27B.md:118` | the tutorial's own serve block passes `--no-enable-prefix-caching` |
| `docs/source/tutorials/models/Qwen3.5-27B.md:137` | the same page *documents* enabling it — `--enable-prefix-caching` plus `--mamba-cache-mode align` — and adds the caveat to quote verbatim: the hybrid KV implementation "might result in a very large block_size when scheduling. For example, the block_size may be adjusted to 2048, which means that any prefix shorter than 2048 will never be cached" |

**The linked tutorial is usually the previous release's build.** Recorded #9579 quotes
`https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html`
while its env dump says `0.19.1rc1`, and the merged `Qwen3.5-27B-Qwen3.6-27B` page exists in
neither exported tree — `v0.18.0rc1` and `v0.19.1rc1` both carry
`docs/source/tutorials/models/Qwen3.5-27B.md`, which is where the two anchors above live.
Answer from the reporter's own pin and say which ref you verified; do not chase the URL
(`../tutorial-verification/docs-version-space-and-image-tag.md`).

**The variant can be off-matrix on top of the pair.** At v0.19.1rc1
`docs/source/user_guide/support_matrix/supported_models.md:82-83` lists only
`Qwen3.5-397B-A17B` and `Qwen3.5-27B`; #8913's `Qwen3.5-35B-A3B` (MoE — its serve line
passes `--enable-expert-parallel`, so `model_type` `qwen3_5_moe` and drafter arch
`Qwen3_5MoeMTP`) has no row, no tutorial page and no nightly config at that pin. State
it as a second coverage gap beside the pair, not as the cause.

Config facts that keep the answer honest:

| anchor | fact |
|---|---|
| vllm `vllm/model_executor/models/config.py:156` (`HybridAttentionMambaModelConfig`), `:436-472` | with `enable_prefix_caching` the mode goes `"none"` → `"all"` if `supports_mamba_prefix_caching` else **`"align"`**, logs `Mamba cache mode is set to '%s' … by default`, falls back `"all"` → `"align"` (`:451`), asserts chunked prefill (`:458`), and sets `mamba_block_size = block_size` (`:472`) |
| `vllm/engine/arg_utils.py:1025` | `--mamba-cache-mode` exists at this ref, so the tutorial's flag is **redundant, not missing**: a launch without it is already in `align`. "You forgot `--mamba-cache-mode align`" is not an answer |
| `vllm_ascend/patch/platform/patch_mamba_config.py:104-107` | the downstream copy of the same gate (`mamba_block_size = block_size` only when prefix caching **and** `align`; otherwise `max_model_len`) — the page-size padding at `:92-103` is the "page size" analogue |
| `vllm_ascend/utils.py:1189-1216` (`refresh_block_size`) | block_size `None` → `128` (`:1200-1201`), and a non-128 value is forced to 128 when prefix caching *or* chunked prefill is on (`:1211-1216`) — but the function **returns early for `model_config.is_hybrid`** (`:1206-1209`, "hybrid attention+mamba models rely on the model-specific sizing logic"). So the tutorial's "may be adjusted to 2048" caveat is the *hybrid* branch, this 128 rule does not apply to the reporter's model, and the only usable number is the one in their own startup log |
| `vllm_ascend/ascend_config.py` (`additional_config` is read key by key with `.get`) | a key the code never reads is silently inert — at this pin that includes `enable_weight_nz_layout`, which the project's own docs and nightly yamls still pass (`../feature-enable-recipes/weight-nz-layout-by-soc.md`, Rule 3). Do not rank it as a cause, and do not let it reassure: it is not a feature the reporter is getting either |

## The crash surface at a v0.19+ ref

When the missing detail is a request-time `AssertionError`, the align-mode
inventory is upstream and one read:

- `vllm/v1/core/single_type_kv_cache_manager.py:769` `MambaManager`, `:776`
  `num_speculative_blocks` (from the KV-cache spec).
- `:937-1005`: in `align` mode **no** blocks are allocated for lookahead
  ("current draft models don't have mamba layers"), the requirement is
  `cdiv(num_tokens, block_size) + num_speculative_blocks`, and three asserts guard it —
  `:961` `assert num_required_blocks > len(req_blocks)` whose message is
  `num_required_blocks {n} < len(req_blocks) {m}`, `:1003` `num_new_blocks <= 1` after
  the first allocation, `:1005` `num_new_blocks <= num_speculative_blocks + 1` before it.
  `:975` carries the branch this pair is about: "When a new request hits the prefix
  cache, the last block saves the hit state."
- Consequence for the answer: an `AssertionError` naming those two numbers is
  **prefix-cache-hit × speculative-block arithmetic**, not the model, the quantization
  or the card — and the cheap split is `--no-enable-prefix-caching` (hit path gone) vs
  dropping `--speculative-config` (blocks gone). Say both, and say that the mechanism is upstream
  vLLM's, so a vllm-ascend patch is not the first place to look.

## The `IndexError` form: the state-copy index at `get_temporal_copy_spec`

The same pair also arrives **with** a four-frame traceback and no device code, and then
the coverage question is already answered:

```
vllm_ascend/worker/model_runner_v1.py:1597    mamba_utils.preprocess_mamba(...)
  vllm/v1/worker/mamba_utils.py:207           collect_mamba_copy_meta(...)
    vllm/model_executor/layers/mamba/mamba_utils.py:285   get_temporal_copy_spec
        src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]
IndexError: list index out of range
```

Verified at vllm v0.19.1 (`b1388b1`): `preprocess_mamba` runs only behind
`assert cache_config.enable_prefix_caching`, and per scheduled request it derives
`num_blocks = cdiv(num_computed_tokens + num_scheduled_tokens, block_size) +
num_speculative_blocks`, `curr_state_idx = num_blocks - 1 - num_speculative_blocks`, and
passes `collect_mamba_copy_meta` the bias `num_accepted_tokens_cpu[i] - 1`; inside, the
source index is `prev_state_idx + (bias + 1) - 1`, i.e.
`prev_state_idx + num_accepted_tokens_cpu[i] - 1`, and the function's own docstring bounds
that count to `1 .. 1 + num_speculative_tokens`. The index can leave the list only when
the request's **mamba block table is shorter than `prev_state_idx + 1 +
num_speculative_blocks`** — an align-mode allocation (`MambaManager`,
`num_speculative_blocks` from the KV-cache spec) disagreeing with this step's
`prev_state_idx`. Not a card, a weight, a kernel or a quantization.

What follows from it, cheaply:

- "either feature alone is fine" becomes source-explained rather than luck: prefix
  caching off ⇒ `mamba_cache_mode="none"` ⇒ `preprocess_mamba` never runs; MTP off ⇒
  `_update_states_after_model_execute` returns early (`not self.speculative_config or not
  self.model_config.is_hybrid`) ⇒ the accepted count stays 1 ⇒ the index degenerates to
  `prev_state_idx`.
- The owner is upstream align-mode bookkeeping. The downstream contribution is only the
  gated call site (`model_runner_v1.py`, `if self.cache_config.mamba_cache_mode ==
  "align"`) plus the deferred `_update_states` correction it runs first — so do not hunt
  a `vllm_ascend` patch.
- The pin's own comment in `preprocess_mamba` already names the stale-index hazard and
  clears `mamba_state_idx` for `finished ∪ preempted ∪ resumed`; "stale state index after
  a resume" is therefore *guarded* at v0.19.1. When the paste's three scheduler sets are
  all empty, that branch simply did not run.

### Search the history by the **upstream** frame names

The deepest frame is under `vllm/`, not `vllm_ascend/`, so a history search keyed on this
project's own symbols misses the precedent. One call —
`bash <skill_root>/scripts/history-titles.sh -b 'preprocess_mamba|get_temporal_copy_spec|mamba_state_idx|mamba_cache_mode'`
— returns **#9024**, "[Bug]: IndexError when MTP speculative decoding + prefix caching
enabled simultaneously" (v0.19.1rc1-openeuler, Qwen3.5-27B-w8a8-mtp TP4): same four
frames, same "either feature alone works" control. Recorded #10326 (Qwen3.5-35B-A3B-w8a8,
TP1, 910B4) is the second report of that exact signature at the same version, and its
reporter added the window in a follow-up — **v0.19.1rc1 crashes, v0.18.0 does not** — which
puts the regression inside the v0.19 line, where the release notes list the rework:
"Zero Bubble Async Scheduling … for async scheduling and speculative decoding" (#7640).
Run that search first: the recorded #10326 run reached it at its 8th tool turn, after four
source reads the precedent would have replaced, and still ended its first turn
output-truncated at the 16 384-token completion cap.

**A 0-hit corpus is the normal result in most case views and it changes nothing.** Recorded
#9579 (v0.19.1rc1 + vllm v0.19.1, Qwen3.5-27B, "if the server config includes prefix-cache,
it will abnormal shut down" — no traceback, no serve command) got `0 hits` from the same
`-b` call: that view's corpus ends at #8849, so neither #9024 nor #10326 can appear in it.
It then spent two `github /search/issues` pages (both `total_count` 1, i.e. the issue being
read) plus greps of `enable_prefix_caching`, the two nightly yamls and
`supports_mamba_prefix_caching` to re-derive anchors already in the table above, and shipped
its ranked verdict on the 22nd of 24 assistant turns. The ids above are this page's own
record — quote them and stop. A `/search/issues` call needs a **lexical** signature (an
exception class, an op name); with no traceback in the paste there is nothing to match, and
the coverage verdict does not depend on it.

## Ladder (one axis per launch, cheapest first)

1. Delete `--enable-prefix-caching`, keep everything else: splits the pair from
   the rest of the stack. This is the row the project's own validated config already
   runs, so a clean run here is expected and is not a defect report by itself.
2. Align to the validated row: `n=3`, `--max-num-batched-tokens 8096`,
   `--max-num-seqs 32`, `--max-model-len` down to the reported input+output, default
   capture sizes. A launch at `n=5`, `max-model-len 262144` and a hand-picked
   `cudagraph_capture_sizes` list is three deviations at once. #8913's were
   `--max-num-seqs 128` / `--max-num-batched-tokens 16384` (vs `32` / `8096`) plus two
   env levers the row does not set, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` with
   `VLLM_ASCEND_ENABLE_FUSED_MC2=1` on a **MoE** drafter
   (`../feature-enable-recipes/sequence-parallel-flashcomm.md` owns the FC1 half); its
   `n=3`, `FULL_DECODE_ONLY`, `--async-scheduling` and
   `--additional-config '{"enable_cpu_binding":true}'` were **not** deviations — that key
   is live and defaults to `True` (`ascend_config.py:110`, `platform.py:951`), so do not
   rank it either way.
3. `"num_speculative_tokens": 5 → 1` with prefix caching still on: the
   speculative-block term shrinks but stays non-zero.
4. `--async-scheduling` off (the other axis outside the validated matrix for hybrid
   GDN: `index.md`, the ablation-row rule).
5. `--compilation-config '{"cudagraph_mode": "PIECEWISE"}'` or `--enforce-eager`:
   moves the failure off the graph path if it is a capture-shaped fault instead.
6. The same weights with both features **off** as the baseline, and the `block_size`
   the engine resolved from the startup log (the tutorial's 2048 caveat is only real
   for the reporter's own launch).

## What to send back

The traceback — the attachment or screenshot failed to upload, so the text is the whole
ask: name the file the body links (`errors.txt`), and ask for its last ~40 lines and the
deepest three frames rather than the whole log — plus:
whether it fails during engine init or on the first request; the startup lines
`Mamba cache mode is set to 'align' …`, `Prefix caching in Mamba cache 'align' mode …
experimental` and the `block_size` actually used; `KV cache size` / "Maximum concurrency"
if the message is capacity-shaped; and confirmation of the reporter's own control
(does it run with prefix caching off *and* with MTP off, or has neither been tried).

## Stop rule

Recorded #9003 verified every fact in the two tables above and more, but emitted
**19 characters of message text across 20 tool turns** (~29 K characters of reasoning)
and shipped the ranked answer only on its final turn, finishing at 13 393 of 16 384
completion tokens. With no error text the first message already has a complete
deliverable — "the pair is documented (tutorial `:137`) but no test on any SoC runs it;
`--mamba-cache-mode` is not missing; drop `--enable-prefix-caching` and send the
traceback" — and every further read is a turn that a truncation can take away.
The case's own `sources-summary.sh` row reads `files=1744` for vllm-ascend, i.e. `tests/`
is present — the three greps above are enough and there is no reason to check coverage a
second way.

When the paste carries the `get_temporal_copy_spec` traceback and the history search
returns #9024, the deliverable is already complete — the precedent, the two-feature split
and the version window — so send it in that turn. #10326 spent its remaining turns on the
KV manager, `config.py` defaults and the release notes after the precedent was in hand.

Not this page: the same two features on **310P with corrupt text at normal
acceptance** (`../accuracy-benchmarks/mtp-prefix-cache-310p.md`, the fp16-SSM-state and
fallback lane), and an accuracy report on any SoC where the output is wrong rather than
the request failing (`../accuracy-benchmarks/index.md`).
