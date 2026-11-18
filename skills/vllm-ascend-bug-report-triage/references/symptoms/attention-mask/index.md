# Attention masks: the quadratic term nobody budgets for

## Scope

Reports whose failing allocation is an **attention mask**, not weights and not the KV cache. Two
observed shapes, both `O(seq_len²)`:

- `prefill-dense-mask-a2-a3.md` — A2/A3 (910B) prefill of a long prompt dies inside
  `get_splitfuse_attn_mask` / `get_attn_mask` (`vllm_ascend/attention/attention_mask.py`), e.g.
  "Tried to allocate 11.50 GiB" for a 78.5 k-token prompt. Worked case: #3357.
- `310p-dense-mask.md` — 310P startup: the mask is built at `max_seqlen` during initialization, so
  workers look *stalled* instead of raising. Worked case: #10872.

## Common rules

- **Invert the mask shape before theorising** (parent rule: a reported byte count is a shape until
  divided out). `rows × cols × itemsize`, then double for the triu/tril temporary and the
  `.contiguous()`/`.to(device)` copy. Invert a candidate shape against the reported number:

  ```bash
  uv run --no-project python -c "import math; X=11.50; print(math.sqrt(X*2**30/2))"   # densesquare fp16/bf16 -> ~78.5k
  ```

  | candidate | bytes | `n` for X = 11.50 GiB |
  | --- | --- | --- |
  | dense `n×n` fp16/bf16 mask | `n² · 2` | ≈ 78.5 k |
  | dense `n×n` fp32 mask | `n² · 4` | ≈ 55.5 k |
  | 2048² int8 compressed chunked-prefill mask | 4 MiB | — |

  A reported "Tried to allocate N GiB" usually equals one term exactly — match it and the mechanism
  is settled (78567² × 2 B = 11.50 GiB in #3357). A mismatch means a different mask class.
- **Separate rows from columns.** `cols` follows the *sequence* length (`max(seq_lens)` or
  `max_model_len`); `rows` follows how many query tokens are in this step (`position`/`position_ids`
  length). Only batching, chunking or `max_model_len` moves those two, which is why
  "just lower `--max-num-batched-tokens`" can be either the right workaround or completely inert.
- **Find the gate that selects the dense path**, then check the *installed runtime* against it —
  `torch.version.cann.startswith("8.3")` in the v0.11–v0.11.x line, `hasattr(torch_npu, …)` on
  310P. Support is decided by CANN/torch_npu at run time, not by the vllm-ascend version alone, so
  "upgrade vllm-ascend" is not automatically the fix.
- **A mask allocation is not in the KV-cache budget.** It is a transient spike on top of a cache
  sized to `--gpu-memory-utilization`; profiling may not have measured it, so the run OOMs even
  though startup succeeded. Check whether the profile/dummy run used a shorter sequence than the
  failing request (on the A2/A3 path it builds no mask at all unless `force_attention` is set).
- **A prefill mask failure is not an OOM about "not enough NPU memory"**; it is a request that must
  fail at some prompt length. Quote the length: the useful deliverable is the threshold and the flag
  that moves it.

## Routes

- Dense splitfuse/`PrefillNoCache` mask on A2/A3 at a long prompt → `prefill-dense-mask-a2-a3.md`.
- 310P, long `--max-model-len`, worker silent after weight load → `310p-dense-mask.md`.
- Mask built at *startup* for every hardware class (`AttentionMaskBuilder(...)` in the runner's
  `__init__`) → the constructor argument decides how much is allocated before any request:
  `max_model_len` on the pre-8.3 path, `max_num_batched_tokens` on the 8.3 path.
- Mask memory along with the rest of the memory budget, when the report is a plain startup OOM with
  no mask frame → `../quantization-load.md` (weights) and `../startup-and-stall.md` (stage).
