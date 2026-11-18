# 310P long `--max-model-len`: the quadratic mask dominates memory at startup

Worked case: vllm-ascend #10872 — "310p 部署 Qwen3.6-30b-a3b 出问题", image
`quay.io/ascend/vllm-ascend:v0.21.0rc1-310p-openeuler` (Atlas 300I DUO 310P3 ×4, per-card memory
~21.5 GB), Qwen3.6-35B-A3B W8A8, TP=4, `--max-model-len 65536`,
`--max-num-batched-tokens 65536`, `--enforce-eager`. Report: the server "just hangs here".

Unlike the A2/A3 class (`prefill-dense-mask-a2-a3.md`) this one does **not** raise: the mask is built
during initialization, so the worker simply produces no further log line and the frontend repeats the
shm-broadcast warning (`../startup-and-stall.md`).

## Code path (verified at v0.21.0rc1 and unchanged at HEAD)

`vllm_ascend/_310p/attention/attention_mask.py`:

- `is_compressed_mask_supported()` is a **runtime gate**:
  `hasattr(torch_npu, "_npu_flash_attention_v3") and hasattr(torch_npu,
  "_npu_paged_attention_splitfuse_v2")`.
- `get_attention_mask` picks `COMPRESSED_MASK_SEQ_LEN = 2048` only when that gate is true; otherwise
  it uses the model's `max_seqlen`.
- `_get_causal_mask`/`gen_causal_additive_mask` then build a dense `[max_seq_len, max_seq_len]` bool
  + fp16 tensor and `npu_format_cast` it to `ACL_FORMAT_FRACTAL_NZ` via `nd_to_nz_2d`. At 65536 that
  is 65536² × 2 B ≈ 8 GiB for the fp16 mask alone, several more GiB for the triu/tril temporaries and
  the NZ copy, on cards whose per-device memory is ~21.5 GB, on top of the weight shard.

The compressed-mask feature commit `fc0b9e354` (#10181, 2026-06-09) **is** in `v0.21.0rc1`
(`git merge-base --is-ancestor fc0b9e354 v0.21.0rc1` → yes), yet the reporter hit the dense path
because the installed torch_npu lacks the two private ops. So "upgrade vllm-ascend" is not the fix:
check the runtime gate.

## Maintainer statement (`reported`)

"建议您缩小到20K先进行尝试 … 目前镜像内的 CANN9.1.0.BETA1 版本 attention算子还不支持压缩mask，64K的长度会生成大约8个G大小的
mask，然后还要进行ND2NZ格式转换，需要至少16个G多的空间" — i.e. the CANN/torch-npu in that image does
not expose the compressed-mask ops, so 64K is not viable. Treat the CANN-side claim as theirs, not as
verified code behaviour; recheck per CANN/torch_npu version. The issue was later closed as not
planned after `wait-feedback` went stale — closure there does not mean a fix landed.

## Workaround

- Drop `--max-model-len` to ~20K (and `--max-num-batched-tokens` accordingly); also lower
  `--gpu-memory-utilization`. This is the maintainer-verified path.
- Removing `--max-num-batched-tokens` (the first maintainer suggestion) only moves the allocation; it
  does not remove it.
- Verify the gate in the user's image before promising compressed-mask support:
  `python -c "import torch_npu; print(hasattr(torch_npu,'_npu_flash_attention_v3'), hasattr(torch_npu,'_npu_paged_attention_splitfuse_v2'))"`.

## Generalisation

On 310P, any dense `[max_model_len, max_model_len]` mask/workspace is the first memory suspect: size
it as `max_model_len² × 2 B × ~2` before blaming the model. Where to look next: `attention_mask.py`
at the reporter's ref, the model runner's profile run (dummy batch size = `max_num_batched_tokens`),
and `nd_to_nz_2d`/`nd_to_nz_spec` in `vllm_ascend/utils.py`.
