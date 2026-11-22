# v0.7.x VL reports: the baseline is transformers, and the vision tower is unpatched

Scope: a **pre-v0.8 generation** perf report — recorded #1427: vllm-ascend `v0.7.3rc1`
(`f025df05f614`) + vllm `v0.7.3` (`ed6e9075d31e`), Qwen2.5-VL-3B served
`vllm serve … --dtype bfloat16 --max-num-seqs 16 --max_model_len 32768
--gpu_memory_utilization 0.8 --kv-cache-dtype auto --enable-prefix-caching
--multi-step-stream-outputs True -tp 2 -pp 1` on two 910B4
(`ASCEND_VISIBLE_DEVICES=3,6`), body = env dump + the client's own latency table at
client concurrency 1 and 4 + the sentence "even worse than transformers based inferring".

The pages of the v0.8.x/v0.9.x generation do not apply here at all: read
`pre-graph-mode-eager-era.md` only to see what is *absent* at an rc1 pin, and never
carry its step 3 torchair opt-in or its step 3b `VLLM_USE_V1=1` lever backwards (step 3
below). `multicard-scaling-and-vendor-baseline.md` is usable only for its step 2
arithmetic; every pinned section in it is v0.8.5rc1.

## Step 0 — what the paste already answers: send this as the first message

The comparison side ("slower than transformers") has **no numbers**, so the literal
claim is unestablished; the rows that do exist are self-consistent and worth typing:

- 200 req ÷ 135.42 s = **1.48 req/s** at client concurrency 1; 200 ÷ 74.16 s =
  **2.70 req/s** at 4. Little's law reproduces both: 1.48 × 0.677 s ≈ 1.0 in flight,
  2.70 × 1.483 s ≈ 4.0 — the engine did batch what the client offered.
- 4× concurrency bought **1.82×** throughput while per-request latency went
  677 → 1 483 ms (**×2.19**), and the K=1 p99 (862 ms) is below the K=4 mean. So the
  engine is serialising per-request work, not failing to batch.
- K=1 `max 12 780 ms` is 19× the mean: a first-request/warm-up outlier, not the
  mechanism. Average it out or discard it explicitly.
- Missing artefacts to request: the **transformers script and its numbers** (device
  count, dtype, batch size), image count/resolution per request, ISL/OSL, and the
  server's own metric block (this paste has none).

## Step 1 — two flags in that command line are inert at this pin

Check these before pricing any lever:

- **`--enable-prefix-caching` is force-disabled for a multimodal model under V0**:
  `vllm/engine/arg_utils.py:1129-1134` logs `--enable-prefix-caching is currently not
  supported for multimodal models in v0 and has been disabled.` and sets it `False`.
  This serve is V0 (step 4), so the reporter's flag never took effect and every request
  re-prefilled. Say that; do not credit the flag in either direction.
- **`--multi-step-stream-outputs True` is the default** (`arg_utils.py:162`, flag at
  `:731`) → passing it changes nothing.
- **Chunked prefill is off for multimodal in V0**: `arg_utils.py:1170-1173` sets
  `enable_chunked_prefill = bool(envs.VLLM_USE_V1)` for a multimodal model, i.e.
  `False`. One image prompt therefore occupies a whole scheduler step and the other
  three requests wait — the K=4 latency row is that shape, not a defect.

## Step 2 — the vision tower is the one component with a pinned up/downstream gap

- The plugin overrides exactly one VL class: `vllm_ascend/models/__init__.py:5-9`
  registers `Qwen2VLForConditionalGeneration` → `CustomQwen2VLForConditionalGeneration`;
  the whole `vllm_ascend/models/` directory is `__init__.py` + `qwen2_vl.py`. There is
  **no Qwen2.5-VL entry**, while `docs/source/user_guide/supported_models.md:8,11`
  marks Qwen2-VL ✅ *and* Qwen2.5-VL ✅ — the docs claim outruns the code at this ref.
- The Qwen2-VL replacement calls one Ascend fused kernel over the whole varlen batch:
  `torch_npu._npu_flash_attention_unpad` (`vllm_ascend/models/qwen2_vl.py:87-101`), and
  the release names it: `docs/source/user_guide/release_notes.md` under `## v0.7.3rc1`,
  "The performance of Qwen2-VL is improved. [#241]".
- Qwen2.5-VL runs the upstream class instead. `Qwen2_5_VisionAttention.__init__` picks
  `get_vit_attn_backend(support_fa=True)` (`vllm/model_executor/models/qwen2_5_vl.py:227`)
  and a non-CUDA platform gets `_Backend.TORCH_SDPA`
  (`vllm/model_executor/models/vision.py:87-105`), whose branch is **one
  `F.scaled_dot_product_attention` per image inside a Python loop**
  (`qwen2_5_vl.py:304-320`) — one launch per image instead of one kernel per batch.
  That loop's length is the image count, so image count is the discriminator.
- Both paths still pay a collective per vision block at TP>1:
  `tensor_model_parallel_all_gather(qkv)` (`qwen2_5_vl.py:238-239`; the same guard in
  `qwen2_vl.py:284-285`, and the Ascend override calls the same `split_qkv`). Do not
  read the gather as Qwen2.5-only — TP>1 pays it on either model, and `-tp 2` is in
  this command only because the reporter split a 3B model over two cards.
- **Is the rate abnormal?** The pin's own VL baseline is a single card and a 7B model:
  `docs/source/tutorials/single_npu_multimodal.md:112` prints `11.29s/it`,
  `output: 20.55 toks/s`, under the tutorial command `:147`
  (`--max_model_len 16384 --max-num-batched-tokens 16384`). A 3B at 0.68 s/request is
  the better number, so "abnormally slow" is not established by this paste — the
  finding is the *axis*, and cause 1 below is the missing side.

## Step 3 — what the rc1 pin cannot offer, so do not propose it

- `vllm_ascend/platform.py` at this ref is 146 lines with **no** `enforce_eager`
  override, no graph/torchair key and no `supports_v1`; `rg -n torchair` over the tree
  is empty. The `torchair_graph_config` switch later release notes call "the 0.7.3-era
  graph mode" (`upgrade-vs-older-baseline.md`'s removed-switch table) is **not** in
  v0.7.3rc1 — do not attribute eager decode to a flag here, and do not offer a graph
  opt-in.
- There is no V1 worker: `check_and_update_config` installs
  `vllm_ascend.worker.worker.NPUWorker` (or the multi-step/spec worker) unconditionally
  (`platform.py:110-117`) and `vllm_ascend/worker/` holds only the V0 runner set. At
  this pin `VLLM_USE_V1=1` is not the recommended lever it is on v0.9.x.
- Everything the 0.8/0.9 pages list as a decode lever (`max_cudagraph_capture_size`,
  ACL-graph capture, `enforce_eager` as a choice, MC2/EP flags) postdates this pin.

## Step 4 — the answer, ranked, one check each

1. **The transformers side does not exist in the paste** (cheapest, and it may close the
   report): ask for the transformers script, its numbers, the image count/resolution and
   the server's metric block; then re-run the identical 200-request set once through
   `vllm bench serve` so both sides carry a rate.
2. **TP=2 on a 3B VL model.** The ablation is one launch: same client, same request set,
   `-tp 1`. Per-request decode rate = output tok/s ÷ concurrency; if TP=1 wins, the
   owner is the per-layer TP reduce plus the per-vision-block all-gather above, not the
   model.
3. **Unpatched Qwen2.5-VL vision path.** Discriminating check: time a request with N
   images against one with a single image, on the same serve. Vision-bound (latency grows
   with image count) → `Qwen2_5_VisionAttention`'s per-image SDPA loop and the per-block
   gather; flat in image count → decode-side, go back to cause 2.
4. **The inert flag pair (step 1)** belongs in the answer as a correction to the test
   matrix, never as a cause: `--enable-prefix-caching` was already off and
   `--multi-step-stream-outputs` was already on.

Recheck conditions: a `Qwen2_5_VLForConditionalGeneration` entry appearing in
`vllm_ascend/models/__init__.py`
retires cause 3; a V1 worker or a graph key in `vllm_ascend/platform.py` retires step 3
and re-opens the newer pages.

## How the recorded run went (route, not result)

The recorded session loaded this topic's `index.md` and
`pre-graph-mode-eager-era.md` (the latter scoped to v0.8.x–v0.9.x), then read
`get_vit_attn_backend`, `Qwen2_5_VisionAttention`, `arg_utils.py`'s chunked-prefill
block and the model registry — and shipped **no message text after its second turn**:
20 assistant turns, ~875 K cumulative prompt tokens, cut at the output cap with the
ranked answer never typed. Every fact in steps 1-4 was two or three greps away at the
pin (`ls vllm_ascend/models`, `rg -n 'enable_prefix_caching|is_multimodal_model'
vllm/engine/arg_utils.py`, `sed -n '227,240p;304,321p'
vllm/model_executor/models/qwen2_5_vl.py`), and the step 0 arithmetic needed no source
at all. For this shape the first message is the arithmetic plus the two inert flags;
the vision-tower chain is the second.
