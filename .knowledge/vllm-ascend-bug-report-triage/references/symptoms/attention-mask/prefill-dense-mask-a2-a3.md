# A2/A3 prefill OOM: dense splitfuse mask sized by the prompt length

Worked case: vllm-ascend #3357 — "0.11.0rc0 qwen3-235B OOM", Qwen3-235B-A22B(-Thinking-2507),
`vllm serve` TP=16, `--max-model-len 98304`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.95`,
`--enforce-eager`, `VLLM_ASCEND_ENABLE_FLASHCOMM=1`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`,
vllm-ascend `0.11.0rc0` with vLLM `0.11.0rc3+empty`, CANN `8.2.RC1`, torch_npu
`2.7.1.dev20250919`; A2-class hardware (log: 60.97 GiB usable per device). Labels `bug`/`qwen3-moe`;
closed 2025-12-15.

## Signature

A single ~90 k-token prompt, server already serving and KV cache allocated, then in every TP rank:

```
vllm_ascend/worker/model_runner_v1.py:1381  in _prepare_inputs
vllm_ascend/worker/model_runner_v1.py:898   in _make_attention_mask
vllm_ascend/attention/attention_mask.py:101 in get_splitfuse_attn_mask
    return attn_mask.contiguous().to(device, non_blocking=True)
RuntimeError: NPU out of memory. Tried to allocate 11.50 GiB
  (NPU 15; 60.97 GiB total capacity; 55.29 GiB already allocated; 4.64 GiB free)
```

Distinguishing features: the wrapper frames are `WorkerProc hit an exception` in
`multiproc_executor.py` (not `failed to start`), the OOM happens **during a request**, and the size
requested matches `len(position) × max(seq_lens) × 2 B` — 78567² × 2 B = 11.50 GiB exactly, with
`num_scheduled_tokens={...: 78567}` in the same log (the whole prompt in one step).

## Turn the OOM line into a shape before reading any code

Two facts are in that line: which single allocation failed (`X`), and that the device was already at
`--gpu-memory-utilization` (55.29/60.97 ≈ 0.91 → 0.95 was requested), so the spike is a **transient
on top of a full KV cache**, not a sizing error. Invert a candidate shape (see the table in
`index.md`): `sqrt(11.50 · 2³⁰ / 2)` ≈ 78 570 for a dense bf16 mask. The worked report used
`--max-model-len 98304` and a ~90 k prompt, and the OOM divided out to 78 567 — i.e. **the mask is
sized by the sequence length actually scheduled, not by the configured limit**. Getting this number
first tells you whether you are looking at an `n²` mask at all, and whether `--max-model-len` can
even be the lever.

## Code at the reported ref (v0.11.0rc0, verified)

`vllm_ascend/attention/attention_mask.py`:

- `AttentionMaskBuilder.__init__` (`:38-57`) builds `self.attn_mask_cache` on **CPU** from
  `_generate_attn_mask(max_seq_len, dtype)` (`:18-33`: a bool tril, a `torch.zeros(n, n)` float32
  tensor and a `masked_fill`, i.e. tens of GB of host RAM at 78 k), and — only when
  `torch.version.cann.startswith("8.3")` (`:53`) — a fixed `2048×2048` int8 upper-triangular
  `chunked_prefill_attn_mask`.
- Two prefill entry points, **both dense** on the non-8.3 path:
  - `get_splitfuse_attn_mask()` (`:85-101`): `8.3` → return the 2048 compressed mask; otherwise
    `max_seq_len = max(seq_lens)`, `_update_attn_cache(max_seq_len, dtype)`, then
    `index_select(self.attn_mask_cache, 0, position)[:, :max_seq_len] * mask_scale_factor`
    (bf16 scale `-10000`) and `return attn_mask.contiguous().to(device)` — **line 101 is the NPU
    allocation**, exactly the frame in #3357.
  - `get_attn_mask(max_seq_len, dtype, device)` (`:72-79`, the `PrefillNoCache` path):
    `attn_mask_cache[:n, :n].to(device)` — the same quadratic shape.
- Shape semantics matter for the workaround: `index_select` selects **rows** with `position`, so
  rows = `len(position)` = `num_input_tokens` = tokens scheduled in this step
  (`worker/model_runner_v1.py:1377-1383` passes `positions_cpu[:num_input_tokens]`), while **columns**
  = `max(seq_lens)` = the longest total sequence in the batch. A 78 k prompt scheduled in one step is
  therefore `78567 × 78567 × 2 B = 11.50 GiB`; the same prompt in 8 k chunks is
  `8192 × 78567 × 2 B ≈ 1.3 GiB`.
- `_update_attn_cache` (`:103-108`) keeps the CPU cache and only rebuilds when a longer sequence
  appears, so host RAM (and the first-touch latency) still scale with the longest sequence seen.
- Construction (`worker/model_runner_v1.py:328-334`): with CANN 8.3 the builder receives
  `scheduler_config.max_num_batched_tokens`; otherwise `model_config.max_model_len`. So on 8.2 the
  process also holds a `max_model_len²` bf16 CPU cache (98304² × 2 B ≈ 19.3 GB ≈ 18 GiB, generated
  from a bool `tril` + float32 `zeros`), independent of the request.

Gating condition: **CANN**. `v0.11.0rc0`'s own release-compatibility matrix pins stable CANN
`8.2.RC1` (`docs/source/community/versioning_policy.md`, "Release Compatibility Matrix"), so the
default stack for this tag takes the dense path. Do not read this as "0.11.0rc0 lacks the fix": the
compressed-mask commit `f1f2c8f5e` (#2962, 2025-09-22) **is** an ancestor of `v0.11.0rc0`
(`git tag --contains f1f2c8f5e` lists v0.11.0rc0…v0.11.0). The gate, not the version, decides.

## Why one step carried the whole prompt (and why the flag alone is not enough)

`vllm_ascend/platform.py` (`check_and_update_config`, ≈`:150-163`) has, for non-MLA models when the
Ascend scheduler is in use and `async_scheduling` is off:

```python
if (scheduler_config.max_num_batched_tokens < scheduler_config.max_model_len
        and not chunked_prefill_enabled_in_ascend_scheduler):
    scheduler_config.max_num_batched_tokens = scheduler_config.max_model_len
```

So the requested budget is silently raised to `max_model_len` and the prompt is prefilled in one step
(that is why 78567 tokens appear as one `num_scheduled_tokens`). In the same release
`AscendSchedulerConfig.__post_init__` (`vllm_ascend/core/schedule_config.py`) *raises* when
`max_num_batched_tokens < max_model_len` and chunked prefill is off — the two behaviours are the same
switch seen from two sides.

The paired upstream tag agrees (`vllm-project/vllm` `v0.11.0rc3`, `vllm/config/scheduler.py:171-199`):
with `enable_chunked_prefill` false, `max_num_batched_tokens` defaults to
`max(max_model_len, 8192)` — 98304 here — and `_verify_args` (`:235-243`) **raises** if
`max_num_batched_tokens < max_model_len` while chunked prefill is off. Consequence: lowering the
batch budget is only effective together with chunked prefill, i.e.
`--additional-config '{"ascend_scheduler_config": {"enable_chunked_prefill": true}}'` (or leaving the
block through `--async-scheduling`); re-read the expression at the reporter's ref, it moved in later
releases.

## Why the profile run did not catch it

`profile_run` calls `_dummy_run(max_num_tokens, with_prefill=True)` with `num_reqs = num_tokens` (one
token per dummy request), and `_build_attention_metadata` only builds a mask when `force_attention` is
set — so the dummy run never allocates the quadratic mask and the KV cache is sized as if it did not
exist. The first real long prompt pays for it.

## Workaround

1. **Enable chunked prefill and cap the prefill chunk** — the documented fix path here:
   `--additional-config '{"ascend_scheduler_config": {"enable_chunked_prefill": true}}'` (equivalently
   `--enable-chunked-prefill` on releases where the CLI flag reaches the Ascend scheduler) plus
   `--max-num-batched-tokens 8192` (or 16384), keeping `--max-model-len 98304`. Rows drop to the
   chunk size while columns stay at the sequence length: 8192 × 78567 × 2 B ≈ 1.2 GiB instead of
   11.5 GiB; 16384 → ≈ 2.4 GiB. Trade-off: more prefill steps for long prompts, slower TTFT, and the
   CPU-side cache is still rebuilt at the full sequence length. Margin: this makes the mask fit, it
   does not remove the quadratic term — the columns still grow with the prompt.
2. **Move the runtime to CANN 8.3** so the compressed `2048²` int8 mask is used. The gate is
   `torch.version.cann.startswith("8.3")`, not the vllm-ascend version: `v0.11.0rc0` **already
   contains** the compressed-mask branch, but its `Dockerfile` pins
   `quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11`. `v0.11.0rc1` switches the base image to
   `8.3.rc1` **and** deletes the dense branch and the check together (`7ee0b0b5d`, the release-branch
   cherry-pick of `cc2cd42ad` #3945, subject "upgrade CANN from 8.2rc1 to 8.3rc1 and remove the CANN
   version check logic"). So on rc0 the fix is the image/CANN upgrade, and the maintainer's "fixed by
   update to CANN 8.3 / 0.11.0rc1" resolves for that reason — do not attribute it to new Python code.
   Check `torch.version.cann` in the reporter's build before promising it.
3. **Lower `--max-model-len`** (and, on 8.2, the initial CPU cache with it) so the longest servable
   prompt is one whose mask fits. This caps the worst case; it does not help a prompt that still fits.
4. **Give the spike room**: lower `--gpu-memory-utilization` (0.95 → ~0.80) so the transient mask
   allocation fits after the KV cache. Maintainer suggestion in the thread ("perhaps adjusting
   `gpu_memory_utilization` could avoid it", `reported`) at the cost of cache capacity.
5. `VLLM_USE_FLASHINFER_MOE_FP16=1` + `--gpu-memory-utilization 0.7` + torch/torch-npu 2.8 +
   BiSheng toolkit + triton-ascend is what another user reported working (#3357 comment
   `ej1-hw`), on a custom-built wheel. Treat as `reported`, and note it changes four things at once.

## Status of the worked case, and the fleet-level outcome

Maintainer (MengqingCao, 2025-10-11, `reported`): "The attn mask is created according to the
`max_seq_len` … we'll finally fix it when cann is upgraded to 8.3, by using a compressed mask …
maybe adjusting `gpu_memory_utilization` could avoid it". Closed 2025-12-15 by leo-pony: "Issue has
been fixed by update to CANN 8.3 … v0.11.0rc3".

At HEAD the CANN gate is gone: `AttentionMaskBuilder` is a `@singleton`;
`get_splitfuse_attn_mask()` takes no `seq_lens` and returns the `2048²` int8 mask unconditionally
(the dense branch exists only for non-causal/pooling cases). The change is `c33150367` (#4779,
2025-12-09), first released in `v0.12.0rc1` — so for a v0.11.x report the honest statement is
"gate on CANN 8.3, or upgrade past v0.11", not "fixed in v0.11.0rc1". The class is version-scoped to
pre-8.3 builds and closed later, but the same arithmetic applies to any `n²` allocation at any ref.

## Check before reuse

```bash
git show <tag>:vllm_ascend/attention/attention_mask.py | sed -n '30,110p'          # mask builder, gate, line 101
git show <tag>:vllm_ascend/attention/attention_mask.py | grep -n 'version.cann\|2048\|index_select'
git show <tag>:vllm_ascend/worker/model_runner_v1.py | sed -n '320,340p'            # builder argument
git show <tag>:vllm_ascend/platform.py | grep -n max_num_batched_tokens             # batch-budget rewrite
git show <tag>:Dockerfile | head -1                                                 # CANN base of the image
```

Also confirm the scheduler default at the paired vLLM tag (`vllm/config/scheduler.py`) and re-do the
arithmetic against the reported GiB figure; a mismatch means a different mask class. Line numbers
above are `v0.11.0rc0` only.
