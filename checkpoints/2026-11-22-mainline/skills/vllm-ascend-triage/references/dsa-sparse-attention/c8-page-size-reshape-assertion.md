# Engine start dies on `assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0`

## Signature

`EngineCore` reaches `Maximum concurrency for <N> tokens per request: <x>x`, then
every worker (`Worker_TP.._DCP.._EP..`) raises a bare `AssertionError` with the
traceback

```
worker.py            initialize_from_config
model_runner_v1.py   initialize_kv_cache            -> initialize_kv_cache_tensors
model_runner_v1.py   _reshape_kv_cache_tensors
AssertionError:  assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
```

on an SFA/DSA model (`index_topk` in the HF config: GLM-5.x, DeepSeek-V3.2/V4),
usually with sparse C8 in `--additional-config` and/or `-dcp`/`-pcp` > 1, often
with MTP. Reported for GLM-5.2-w4a8 on one A3 node, TP16 + DCP16, `-dcp 16`,
`"enable_sparse_c8": true`, `"enable_dsa_cp": false`, `--max-model-len 1024000`
(#12102, vllm/vllm-ascend v0.23.0 release branch, `model_runner_v1.py:4443`).

Classify it as lane B, phase = **KV cache construction**, after weight load and
after the block count was computed. It is a *dimension-arithmetic* check, not
memory, not card health, not HCCL: the engine already printed a concurrency
number, and an arithmetic mismatch repeats at every page count the allocator can
pick (the script simulates four), so `--gpu-memory-utilization` cannot move it.

## What the assertion actually compares

`initialize_kv_cache_tensors` splits one raw int8 tensor per attention layer into
k / v / indexer-k / indexer-scale slices using
`AscendMLAAttentionSpec.sparse_kv_cache_ratio`
(`int(kv_cache_tensor.size // ratio_i)`), then `_reshape_kv_cache_tensors` sums
the slice numels and asserts the sum is a whole number of that spec's
`page_size_bytes`. Both sides are derived from the same head tuple
(`kv_lora_rank`, `qk_rope_head_dim`, `index_head_dim`), which the worker reads
from `hf_text_config` (`model_runner_v1.py:364-375`) and stores in
`sparse_head_dim`.

**The check is therefore an internal-consistency test, and the first question is
which arithmetic contract the reporter's tree has.** Two exist in this snapshot,
and the module moved between them — a report on a July release branch may predate
the move:

| pin | spec module | page_size_bytes | sparse_kv_cache_ratio | assertion |
|---|---|---|---|---|
| v0.22.1rc1 `5f6faa0` | `vllm_ascend/patch/platform/patch_kv_cache_interface.py` | real bytes `B*H*((L+R)*2 + I + 2)` (`:63-88`) | **bf16-element virtual units** `(L, R, I//factor, 1)`, `factor = 2` (`:92-160`) | `model_runner_v1.py:4288` |
| v0.23.0 `5cb98ca` | `vllm_ascend/core/kv_cache_interface.py` | byte weighted (`:67-107`) | byte weighted, same terms (`:121-190`) | `model_runner_v1.py:4547` |

Consequences, and they are the whole routing decision:

- **Legacy contract**: page and ratio are in *different units*, so the four
  `int()` floors are exact only when `page_per_head == 2*total_virtual`, i.e.
  **when `index_head_dim` is even**. An odd `index_head_dim` (or any dims where
  that identity fails) loses a byte or two per page and the assertion fires on
  every launch, at any DCP, with any flags. This is arithmetic: no ablation can
  help, and the answer is "report the dims, the spec arithmetic must pad".
- **v0.23.0 contract**: both properties are byte weighted and both carry
  `sfa_dcp_replicated_indexer_size`, so the identity is exact by construction. A
  firing assertion at that pin is therefore *not* arithmetic: the spec that sized
  the tensor is not the spec used in the reshape.
- One command tells them apart, and the reporter can run it:
  `ls vllm_ascend/patch/platform/patch_kv_cache_interface.py vllm_ascend/core/kv_cache_interface.py`
  — the first exists only ≤ v0.22.1rc1, the second only ≥ v0.23.0.

Evaluate the identity instead of deriving it by hand:

```bash
uv run --no-project python <skill_root>/scripts/kv-page-size-arithmetic.py \
    --layout v023 --mode li_c8 --soc A3 --dcp 16 \
    --k-head-dim <kv_lora_rank> --v-head-dim <qk_rope_head_dim> --index-head-dim <index_head_dim>
# same call with --layout legacy for a pre-v0.23.0 tree
```

Its `VERDICT:` line is the answer's first sentence: `does NOT hold` → arithmetic,
stop; `holds` → the spec-comparison below, then one ablation.

## When the identity holds: compare the two spec construction sites

`AscendMLAAttentionSpec` is built twice. The `KVCacheSpec` the engine sizes the
tensor with comes from `ModelRunner.get_kv_cache_spec`
(`worker/model_runner_v1.py:4907` onward; the DSA branch sets
`cache_sparse_sfa_c8` / `cache_sparse_li_c8` from `ascend_config` and
`sfa_dcp_replicated_indexer_size=self.sfa_dcp_replicated_indexer_size` at
`:4978-4984`), while the model-side constructors
(`models/layer/attention/layer.py:186`, `models/deepseek_v4.py:159`) pass neither
of those and therefore keep the defaults (`False`, `1`). The tensor size itself
is `per_layer_specs[layer].page_size_bytes * num_blocks` on the sizing side
(vllm `v1/core/kv_cache_utils.py:1286`).

So the check is: **does the spec that produced `kv_cache_tensor.size` carry the
same C8 flags and the same `sfa_dcp_replicated_indexer_size` as the spec in
`_reshape_kv_cache_tensors`?** Two `sed -n` reads at the reporter's commit
answer it. The DCP field is the one to look at first, because its gate is exactly
this deployment shape: `enable_sfa_dcp_replicated_indexer()` is
`model_uses_sfa_sparse() and decode_context_parallel_size > 1 and
prefill_context_parallel_size == 1` (`vllm_ascend/utils.py:125-136`, v0.23.0) and
it sets the multiplier to `dcp_size` (`model_runner_v1.py:455-457`). A page size
that differs between the two sites by the indexer terms —
`(dcp-1)*(index_head_dim*dtype + scale)` per head per page — cannot divide a
tensor sized by the other site. Label this **hypothesis, unverified at the
reporter's commit** (their `a218b5b` is not the snapshot's `5cb98ca`: their line
is 4443, the pin's 4547).

## Ablations, only after the two steps above

1. `"enable_sparse_sfa_c8": false` **and** `"enable_sparse_li_c8": false` in the
   same `--additional-config`, everything else identical (one launch). C8 off
   removes the indexer slices from the sum entirely; it is the one flag that
   changes the *arithmetic*, which is why it goes first.
2. Keep C8, set `-dcp 1` (one launch). This removes
   `sfa_dcp_replicated_indexer_size` from both sides and is the only check that
   separates "C8 arithmetic" from "DCP replication".
3. Keep both, drop the MTP/`--speculative-config` once. The draft layer is a
   separate spec in the same group (`cache_only_layers`/hidden-state branches in
   the same function), so it can be the layer whose spec disagrees.

Do not open the attention backend, `mla_cp.py`'s virtual block size or HCCL
first: `cp_virtual_block_size` (`attention/context_parallel/mla_cp.py:78-82`)
changes the *backend's* block granularity, and while it is worth naming as an
open question, it does not enter either side of this assertion.

## The documented pairing (say it, but do not overturn it)

At both pins every `enable_sparse_c8` occurrence in the model's own tutorial is
`false` — `docs/source/tutorials/models/GLM5.2.md` `:678,748,820,890,1053,1140`
at v0.22.1rc1, six occurrences, same value at v0.23.0 — and the DSA-CP recipes
pair `"enable_dsa_cp": true` with `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` and C8
disabled. The reporter ran the inverse (C8 on, `enable_dsa_cp` off, FLASHCOMM1=0),
which is also why they had no way to turn DSA-CP on: `enable_dsa_cp()` raises
`ValueError: DSA CP requires SP to be enabled …` without SP
(`vllm_ascend/utils.py:1345-1367` at v0.22.1rc1, `:1515-1536` at v0.23.0). The
release notes go further for exactly this pairing: GLM-5.2 DCP ×
`enable_sparse_sfa_c8` is listed as a known issue that is **not recommended**
(#14011, quoted with its line in `c8-flags-and-kv-store-path.md`). The honest
verdict is therefore "a combination the project tells you not to run, and one the
tutorial never ships", not "unsupported": the tutorial is a recipe list, not an
exclusion list, and the assertion is still a defect to report.

## What to ask back

`index_head_dim`, `kv_lora_rank`, `qk_rope_head_dim` from the checkpoint's
`config.json`; which of the two spec modules the tree has; the engine's
`Calculated ...`/`num_blocks` lines before the assertion; and whether the same
launch fails with `-dcp 1`. Those four settle the case without another launch.
