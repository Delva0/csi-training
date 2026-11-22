# A bare `setup failed!` is the caller's own pack — start at the deepest `vllm_ascend` frame

Verified at vllm-ascend `v0.7.3rc1` (`f025df05f614`) and vllm `v0.7.3`
(`ed6e9075d31e`), recorded for **#393**: `torch-npu 2.5.1.dev20250308`, an
ms-swift 3.2.1 GRPO run whose rollout engine is the **V0** vLLM engine
(`vllm/engine/llm_engine.py:1391` → `vllm/worker/worker_base.py:420`), Qwen2.5-class
model. It trains ~5 steps (`Train: 0%|… 5/8724`) and then kills the rank.

## The signature

```text
Train: 0%|… 5/8724 [02:25<…]mki_log log dir:/root/atb/log exist
[rank0]:   File ".../vllm_ascend/attention.py", line 869, in forward
[rank0]:     torch_npu._npu_flash_attention_qlens(
[rank0]: RuntimeError: setup failed!
```

No op name, no code, no device dump: the message is the ATB/op-plugin *host-side
operation setup* refusing a variant pack (shape/dtype/attribute). Nothing was
launched, so no card, HCCL, OOM or driver claim applies. The `mki_log … atb` line
immediately before it is ATB's first-use banner (string in no export — label it as
consistent-with, not proof).

## Which of the six `setup failed!` surfaces this is

| Text | Owner | Page |
|---|---|---|
| **bare `setup failed!`**, no op name, deepest `vllm_ascend` frame is a `torch_npu._npu_*`/ATB attention call | the **call site in that frame**: its own mask/size/attribute arguments | this page |
| `SelfAttentionOperation setup failed!` + `op_plugin_atb` frames at a proposer line, spec decode on | ATB at the drafter (join point; the Python line is a victim) | `../spec-decode-mtp/atb-op-setup-failure-at-proposer.md` |
| `ReshapeAndCacheOperation setup failed!` on a `-310p` image with `--kv-cache-dtype fp8`/`bfloat16` | 310P has no non-fp16 KV path | `../soc-platform-paths/310p-kv-cache-dtype.md` |
| `<Op>Operation setup failed!` beside `507018` / `RunCpuKernel` | the op ran on the CPU/AICPU fallback kernel | `../soc-platform-paths/310p-cpu-fallback-op-fault.md` |
| `SelfAttentionOperation setup failed!` with **no** `--speculative-config` at all, a dense model and one **long** prompt (12 732 tokens on `--max-num-batched-tokens 14336`) | the dense V1 one-shot prefill branch handing ATB a `[prompt, prompt]` mask — branch first, kernel contract not derivable (§ below) | this page, second section |
| `SelfAttentionOperation setup failed!` **plus** an ATB log line `tensor dimNum 0 is invalid … inTensor [N] CheckTensorShape failed` | the caller passed a `None`/empty tensor into the pack — find the never-bound attribute, do not read the op's shape contract | `atb-input-dim0-none.md` (recorded #2779) |

All of them mean the same phase. What differs is *who assembled the pack*, and the
paste answers that: read the deepest `vllm_ascend` frame, then read the size
arguments it passes.

A **`CreateOperation failed!`** message is not in this table and is not a pack question at
all: it fires before any tensor is described, so the owner is the ATB build × SoC —
`atb-create-operation-failed.md` (recorded #5170, 310P).

## The recorded mechanism at `f025df05f614` (source-verified sizes)

`vllm_ascend/attention.py:866-876` is reached only when `attn_metadata.block_tables`
is non-empty, `num_decode_tokens == 0` **and** `not chunked_prefill_enabled`
(`:866`). With chunked prefill off, the only code that gives a *prompt* a block
table is a prefix-cache hit — `vllm_ascend/worker/model_runner.py:590-645`
(`prefix_cache_hit = computed_block_nums is not None and len(...) > 0 and
self.sliding_window is None and inter_data.is_prompt`) — and upstream defaults
`enable_prefix_caching` to `bool(envs.VLLM_USE_V1)`, i.e. **False** on the V0 path
(`vllm/engine/arg_utils.py:217-220`). So the first fact to state is a derivation,
not a guess: **the rollout engine set `enable_prefix_caching=True` explicitly and
runs chunked prefill off.**

The pack's variable input is the mask, and its bound is one block:

| Evidence | Claim |
|---|---|
| `attention.py:612-617` | this branch builds `self.compress_mask = …get_attn_mask(128, dtype, device)` — a literal `128` |
| `attention.py:58-88`, `:502-505` | `get_attn_mask(n)` returns `attn_mask_cache[:n, :n]`, so `128` ⇒ a contiguous `[128, 128]` mask; the builder is a per-process singleton seeded from `128` |
| `platform.py:120-123` | `block_size` defaults to `128` and prefix caching *requires* 128 — so `128` is exactly one KV block |
| `model_runner.py:614-645` | what the same branch's metadata can actually carry: **partial** hit → `query_lens = seq_lens − prefix_cache_len`; **full** hit → `query_lens = 1`, `context_lens = seq_lens − 1`. Neither is bounded by one block |
| `attention.py:867-874` | that `[128, 128]` tensor becomes `mask=` while `seq_len=self.query_lens_tensor_cpu` and `context_lens=self.seq_lens_tensor_cpu` |

Why five steps survive: a fresh prompt has no computed block, so `block_tables` is
empty and the step takes the `_npu_flash_attention` + `max_prefill_seq_len` mask
path (`:608-612`, `:850-860`) at any length. The `qlens` branch needs a *hit*, and
it fires as soon as one arrives with a suffix (or a context) larger than a block —
a data-order trigger, not a step count.

**Confidence:** the sizes and the branch membership are source-verified at that
ref; ATB's own validation text is not in the snapshot, so "the pack is rejected
because the mask does not cover those lengths" is the hypothesis the two ablations
below settle.

## `SelfAttentionOperation` with **no** spec decode: the one-shot prefill mask

Derived at vllm-ascend `v0.9.2rc1` for #2239 (Qwen2.5-7B bf16 on a 910B4/32 GB,
`torch-npu 2.5.1.post1.dev20250619`, CANN 8.1.RC1, `--max-model-len 14336`,
`--max-num-batched-tokens 14336`, one 12 732-token prompt, cause left open). The
named op appears with no drafter anywhere, so read the **branch** before the ATB
contract — the same message has three different owners in the table above.

- Backend: V1, non-MLA, non-torchair → `AscendAttentionBackend`
  (`platform.py:228-229`) → `attention/attention_v1.py`.
- Branch: `model_runner_v1.py` selects `PrefillNoCache` when
  `seq_lens == num_scheduled_tokens`, i.e. **the whole prompt fit in one step**
  because `--max-num-batched-tokens` ≥ prompt length. That branch calls
  `torch_npu._npu_flash_attention` (`attention_v1.py:346`) with
  `mask = AttentionMaskBuilder.get_attn_mask(max(seq_lens))` — a dense
  `[12732, 12732]` bf16 tensor. A prompt that does **not** fit one step takes the
  splitfuse branch instead (`_npu_paged_attention_splitfuse`, `:420`).
- `PAGED_ATTENTION_MASK_LEN` is **not** a mask cap: the builder is seeded with
  `min(max_model_len, int(os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)))`
  (`model_runner_v1.py:213-215`) and `get_attn_mask`/`_update_attn_cache` grow the
  cache on demand (`attention/attention_mask.py:49-53`, `:98-103`). Do not report
  "masks above 10000 are unsupported" out of that default.
- The version lane was clean and proves nothing: `requirements.txt:27` pins
  `torch-npu==2.5.1.post1.dev20250619`, exactly the reporter's pta, and that
  release raises CANN to 8.1.RC1 *to enable* chunked prefill/APC — which they had.
- Discriminators, cheapest first: does a **short** prompt serve (branch vs op)?
  then `--max-num-batched-tokens 8192` (forces chunked → splitfuse) and, if it
  still dies, `VLLM_USE_V1=0` (the V0 `attention.py` path uses
  `npu_fused_infer_attention_score`). Unlike #393 above, the batched-token knob is
  the lever here, precisely because it changes the branch.
- What cannot be settled: no exported tree carries ATB's `SelfAttention` contract
  (`Ascend/op-plugin` exports no `atb/` sources; `Ascend/pytorch` and
  `Ascend/torchair` contain no `SelfAttention` string). Ship the branch, the mask
  size and the two ablations; do not derive the kernel's limit.

## Ladder

1. `enable_prefix_caching=False` on the rollout engine (ms-swift exposes the vLLM
   engine kwargs; the exact arg name is not in this snapshot — ask for the launch
   config rather than quoting one). Removes the branch entirely. **Most decisive.**
2. `enable_chunked_prefill=True` instead: the same step takes the splitfuse path
   (`:619-629` builds `chunk_mask` from `max_seq_len`; `:880-905` calls
   `_npu_paged_attention_splitfuse`), so the mask is sized by the schedule.
3. Keep both flags and still crash ⇒ out of this page: `--enforce-eager` then
   `ASCEND_LAUNCH_BLOCKING=1`, and get the plog.
4. If they must keep prefix caching without chunked prefill, the fix is the code at
   `:615-617` (build the mask from the branch's own maxima) — say so as the patch
   proposal, not as the first action.

Not useful: `--max-num-batched-tokens`, `--max-model-len`, capture sizes and
`HCCL_*` (nothing was launched); `--enforce-eager` alone (the ATB call is on the
eager path too).

## What this snapshot cannot settle

- ATB (`cann/ascend-transformer-boost`) and op-plugin sources are unavailable and
  `_npu_flash_attention_qlens` is in neither exported tree (the pytorch export is
  `1133/1134 TRUNCATED`), so the op's mask/shape contract is a hypothesis: one
  grep per tree, then stop — do not hunt the emitter.
- Whether the reporter's engine really enabled prefix caching is derived from the
  branch, not read from their config. Ask for the rollout engine's kwargs.
- Version scope: the V0 `vllm_ascend/attention.py` layout is this pin's; re-grep
  this branch on any other ref before quoting `path:line`.
- Precedent: `scripts/history-titles.sh -i 'setup failed|qlens|atb'` returns 0 hits;
  the prior-month corpus has no equivalent (it is not evidence of absence).

## Deliver

Verdict first (a real vllm-ascend prefill-path crash, host-side op setup, no
hardware involvement; the two-flag combination is what reaches it), then the one
decisive ablation, then the ranked second cause (query/context bookkeeping in
`_add_seq_group`), then what to send back: the rollout engine's full kwargs
(prefix caching / chunked prefill / `max_model_len` / `max_num_batched_tokens`),
the failing iteration's prompt lengths or `max_query_len`, and the plog window.
