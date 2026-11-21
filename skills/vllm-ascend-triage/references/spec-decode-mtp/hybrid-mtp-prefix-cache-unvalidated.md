# `qwen3_5_mtp` + `--enable-prefix-caching` on a hybrid GDN model: the unvalidated pair, often with no error text

## Report shape

A hybrid linear-attention family (Qwen3.5/Qwen3.6, DeepSeek-V4, GLM; `model_type`
`qwen3_5`) served with `--speculative-config '{"method": "qwen3_5_mtp", …}'` **and**
`--enable-prefix-caching`, where the title says only "同时开启mtp+prefix cache后，推理报错"
and the **body carries no error at all** (recorded #9003: the body's whole bug section was
`<!-- Failed to upload "image.png" -->`, and `/workspace/attachments` was empty) — or the
error is one line of Chinese with no traceback. The launch usually also has
`--async-scheduling`, `cudagraph_mode: FULL_DECODE_ONLY`, an explicit
`cudagraph_capture_sizes` list, a `--quantization ascend` w8a8 checkpoint and a
`--additional-config` stack.

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

## Ladder (one axis per launch, cheapest first)

1. Delete `--enable-prefix-caching`, keep everything else: splits the pair from
   the rest of the stack. This is the row the project's own validated config already
   runs, so a clean run here is expected and is not a defect report by itself.
2. Align to the validated row: `n=3`, `--max-num-batched-tokens 8096`,
   `--max-num-seqs 32`, `--max-model-len` down to the reported input+output, default
   capture sizes. A launch at `n=5`, `max-model-len 262144` and a hand-picked
   `cudagraph_capture_sizes` list is three deviations at once.
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

The traceback — the screenshot failed to upload, so the text is the whole ask — plus:
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

Not this page: the same two features on **310P with corrupt text at normal
acceptance** (`../accuracy-benchmarks/mtp-prefix-cache-310p.md`, the fp16-SSM-state and
fallback lane), and an accuracy report on any SoC where the output is wrong rather than
the request failing (`../accuracy-benchmarks/index.md`).
