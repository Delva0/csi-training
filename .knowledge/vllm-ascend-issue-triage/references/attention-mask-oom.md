# Long-prompt OOM in the attention-mask builder (A2/A3, v0.11/v0.12 line)

The most common "OOM on a long prompt" class in the 0.11 line is not KV cache and not
fragmentation: `_make_attention_mask` materialises a **dense causal mask of
`[N, N]`** where `N` is the prompt length scheduled in that prefill step, and copies it H2D.
Memory therefore grows with the square of the prompt (or of the first prefill chunk). That is a
per-step tensor on top of an already-reserved KV cache, which fits the reported failures of
lowering `--gpu-memory-utilization` to 0.75, shrinking `cudagraph_capture_sizes` and going from
TP2 to TP4 — none of those change the size of the mask.

Verified against `/repos/vllm-project/vllm-ascend` tags `v0.11.0rc0`, `v0.11.0rc3`, `v0.11.0`,
`v0.12.0rc1`, `v0.13.0rc1`; thread statements from issues #3308, #3357, #5139 (fetched
2026-11-01). The 310P analogue (different code, `AttentionMaskBuilder310`) is in
[310p-atlas-300i-duo.md](310p-atlas-300i-duo.md). If the OOM message names neither this builder nor
a tensor, check the other two OOM classes before sizing the mask: operator workspace
([npu-workspace-memory.md](npu-workspace-memory.md)) and the replay-time `OutOfMemoryError` wrapper
([log-signatures.md](log-signatures.md)).

## Recognise it from the traceback and the allocation size

Frames, outermost to innermost:

```
vllm_ascend/worker/worker_v1.py        execute_model
vllm_ascend/worker/model_runner_v1.py  _prepare_inputs          -> self.attn_mask = self._make_attention_mask(...)
vllm_ascend/worker/model_runner_v1.py  _make_attention_mask
vllm_ascend/attention/attention_mask.py get_splitfuse_attn_mask | get_attn_mask
RuntimeError: NPU out of memory. Tried to allocate X GiB
```

`X` is the check: `X GiB = N^2 * dtype_bytes`, `dtype` = the model dtype (bf16 → 2 B).

| Report | X | Implied N | Cross-check |
|---|---|---|---|
| #3308, #3357 (v0.11.0rc0, CANN 8.2.RC1) | 11.50 GiB | 78567 | the same log's `dump_input` shows `prompt_token_ids_len=78567`, `num_scheduled_tokens={...: 78567}` |
| #5139 (v0.11.0rc3, CANN 8.3.RC2) | 7.75 GiB | ≈64500 | first prefill chunk of a 127K prompt bounded by `--max-num-batched-tokens 65536` (65536² bf16 = 8.00 GiB; the exact chunk length is not in the report) |

So read `SchedulerOutput(...)` in the worker dump before theorising: it states the scheduled
length that produced the mask. Two reporters on different plugin versions can hit the same
class through different functions, so the raising function name is the discriminator, below.

## Which branch ran, per tag — the gates decide it

`_build_attn_state` (unchanged across these tags) sets `PrefillNoCache` when
`seq_lens_np[:num_reqs] == num_scheduled_tokens`, i.e. nothing was computed before this step
(the first chunk of a fresh prompt), `ChunkedPrefill` when chunking/spec-decode applies and
chunked prefill is on, `PrefillCacheHit` otherwise; the mask builder then goes quadratic only
in the `PrefillNoCache` case.

- **v0.11.0rc0, CANN < 8.3** — `model_runner_v1.py` builds
  `AttentionMaskBuilder(self.model_config.max_model_len, self.dtype)` under
  `if torch.version.cann.startswith("8.3") ... else ...`, so *init alone* eagerly materialises
  a host `[max_model_len, max_model_len]` mask (98304² × 2 B ≈ 18 GiB, plus bool temporaries)
  and `get_splitfuse_attn_mask()` takes its dense branch:
  `index_select(attn_mask_cache, 0, position)[:, :max_seq_len]`, `*= mask_scale_factor`,
  `contiguous().to(device)`. Line 101 of that file, the `.contiguous().to(device)` return, is
  exactly the frame in #3308/#3357. Reports of both a `get_splitfuse_attn_mask:101` frame *and*
  CANN 8.2 are this branch.
- **v0.11.0rc1 … v0.11.0** — the CANN gate is gone (commit `0c6349610`, "#3048 Reduce host
  memory usage for attention mask generation", 2025-10-21, first in v0.11.0rc1):
  `_generate_attn_mask` stops building intermediate bool copies, the builder is always sized
  `scheduler_config.max_num_batched_tokens`, and `get_splitfuse_attn_mask()` always returns the
  fixed `[2048, 2048]` int8 `chunked_prefill_attn_mask`. The dense path survives only for
  `get_attn_mask(max(seq_lens))` in `PrefillNoCache` (and a fixed `128` for `PrefillCacheHit`),
  which is what #5139 hits at rc3 with a 127K prompt. At rc3 the builder construction is
  `model_runner_v1.py:323-324` (sized to `max_num_batched_tokens`) and `get_attn_mask` is
  `attention_mask.py:69-72` (`_update_attn_cache`, then
  `attn_mask_cache[:max_seq_len, :max_seq_len].contiguous().to(device, non_blocking=True)`), with
  `_make_attention_mask(seq_lens, position, attn_state)` dispatching pooling -> splitfuse -> this
  call -> `None` for decode. `0c6349610` is an ancestor of `v0.11.0rc3`
  (`git merge-base --is-ancestor` → yes), so #3048's host-memory fix is already inside the tag that
  still OOMs: never offer #3048 as the remedy for an rc3 report, and do not read its presence as
  evidence the report is stale.
- **v0.12.0rc1+** — `_make_attention_mask(attn_state)` no longer takes `seq_lens`/`position` and
  returns only fixed-size masks: compressed splitfuse `[2048, 2048]` int8, `get_mla_mask`
  `[512, 512]`, pooling `get_attn_mask(2048, torch.bool)`. No per-length prefill mask is
  transferred H2D. `v0.13.0rc1` has the identical dispatcher.

Commit message scope matters: #3048 reports itself as a **host**-memory fix ("single GPU host
memory usage exceeded hundreds of GB at 128k") and "tested: CI pass". It is *not* documented as
a device-HBM fix; the HBM relief for chunked prefill follows from the compressed mask, and the
maintainer statements below are what establish the OOM link.

## Thread statements (attribute, do not upgrade to code fact)

- #3357, maintainer reply: "The attn mask is created according to the `max_seq_len` ... the
  required memory will grow as the prompt grows. we'll finally fix it when cann is upgraded to
  8.3, by using a compressed mask ... maybe adjusting `gpu_memory_utilization` could avoid it";
  later close note: fixed by CANN 8.3, use v0.11.0rc3 or newer.
- #5139, assignee reply: "on v0.11.0rc3, since we still use `_npu_flash_attention`, the mask is
  transfered via h2d in runtime ... suggest to reduce `--max-num-batched-tokens` or
  `gpu-memory-utilization`. Or consider v0.13.0rc1, where we already unify attention with
  `npu_fused_infer_attention_score` ... fixed mask `[2048, 2048]`". The reporter then reported
  success after moving to a larger-HBM device (910B2) on the same version — a capacity change,
  not a code fix, so do not present it as evidence of a fix.

Code and thread agree on the mechanism and on the two levers; they disagree on the boundary
(maintainer names v0.13.0rc1, the source shows the per-length mask gone from v0.12.0rc1). Cite
the maintainer figure as their recommendation and v0.12.0rc1 as the code-verified boundary.

## Workarounds, strongest first

1. **Bound the first prefill chunk**: `N` is `min(prompt_tokens, --max-num-batched-tokens)`, so
   32768 → 2 GiB, 16384 → 0.5 GiB of mask instead of 8–11.5 GiB. This is the one lever that
   follows directly from the code and applies to every tag in this file.
2. **Free HBM for the transient**: lower `--gpu-memory-utilization` (maintainer and community
   advice in #3308/#5139; the reporter in #5139 had already tried 0.75 without success, so
   treat it as a degree fix, not a cure).
3. **Upgrade**: v0.11.0rc1+ removes the dense splitfuse mask (needs CANN 8.3 for the compressed
   op); v0.12.0rc1+ removes the dense prefill mask entirely.
4. Unverified community extras from #3308: `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32` /
   `expandable_segments:True`. They target fragmentation, which the numbers below rule out;
   keep them labelled as reporter-side claims.

## Pitfalls

- **The `max_split_size_mb` sentence in the OOM message is boilerplate.** It applies only when
  reserved ≫ allocated. In both reports reserved ≈ allocated (54.27 vs 54.10 GiB in #3357), so
  it is genuine demand from one tensor; saying "try max_split_size_mb" wastes the reporter's
  time.
- **A "fixed by PR #X" comment can name the wrong PRs.** The automated reply on #3308 credits
  #3305 and #3557; #3305 is "[Bugfix] Fix ascend config for qwen3-next" and #3557 is
  "[Doc] Update the multi-NPU Qwen3 Next tutorials" (both PR bodies fetched). The mask change is
  #3048 / `0c6349610`. Check the diff and `git merge-base --is-ancestor` before quoting a PR.
- **`--enforce-eager`, `cudagraph_capture_sizes` and TP size are noise here.** This mask is
  built in `_prepare_inputs` before any graph replay; the reporters' failed attempts at all
  three are consistent with that, not with a graph-mode cause.
- **Do not answer "how does a fixed `[2048, 2048]` mask serve >2048-token sequences?" from
  intuition.** The reporter asked it in #5139 and it is unanswered in the thread; treat the
  compressed-mask semantics as an open question unless the op's docs at a pinned tag are read.
- **Keep the 310P variant separate.** `AttentionMaskBuilder310` has its own gate
  (`is_compressed_mask_supported()` = torch_npu `_npu_flash_attention_v3` +
  `_npu_paged_attention_splitfuse_v2`) and an NZ conversion copy on top; a 310P OOM needs those
  sizes, not the A2/A3 ones above.

## Cheapest reading route (from the recorded passes)

Two `git show <tag>:<path> | grep -n -A 3` calls answer the dispatch question
(`AttentionMaskBuilder(`, `def _make_attention_mask`), one `git show` of
`attention/attention_mask.py` (≈100 lines) gives both mask functions, and one
`git log --oneline <t0>..<t1> -- vllm_ascend/attention/attention_mask.py` names the fix commit.
The recorded passes that instead paged `model_runner_v1.py` by line ranges spent up to 64k
characters per read — one `sed -n '/def _build_attn_state/,/return AscendAttentionState/p'`
printed to EOF because that end address never matches, and the `nl -ba` renumbering hid it —
and all four were cut off by the output budget with no answer.
