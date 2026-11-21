# Long embedding inputs come back wrong: what chunked prefill does to a pooling runner

Worked case: vllm-ascend #17797 — image
`quay.io/ascend/vllm-ascend:v0.23.0-310p`, Ascend 310P, vllm/vllm-ascend
**v0.23.0** (`0fc695f` / `5cb98ca`) — i.e. the SOURCES pins match the reported
image, so the tree is citable as it stands. `Qwen/Qwen3-Embedding-0.6B`,
`--runner pooling --dtype float16`, one request at a time (`/v1/embeddings`,
batch size 1), prompts of ~4-10 K characters (~3-8 K tokens).

Reporter-confirmed evidence, all of it load-bearing:

- cosine against three references (T4+vLLM, mis-tei, the vendor API) is
  `> 0.999` up to ~4.2 K characters (the first row of the reporter's own table)
  and `~0.2-0.6` from ~4.5 K characters on;
- **within one service**, inputs differing by 2-20 characters score `0.35-0.93`
  against each other, so the vector is not self-consistent; the same input
  repeated gives the identical vector (not noise);
- the failure is deterministic inside a serve session and **changes across
  restarts**;
- `--no-enable-chunked-prefill` removes it completely.

That combination is a *state* signature, not a fixed numeric defect: a kernel
whose arithmetic is wrong for long inputs produces the same wrong vector after a
restart. Say so before choosing between the leads below.

## Verified in the pinned tree (so do not re-derive it)

- The chunked-prefill accumulator is **upstream and inherited**, not
  re-implemented: `vllm/v1/pool/metadata.py:39-45` (`PoolingStates
  .hidden_states_cache`), `vllm/model_executor/layers/pooler/tokwise/methods.py:
  35-84` (`AllPool.forward` appends each chunk's hidden states when
  `scheduler_config.enable_chunked_prefill` and concatenates them once
  `pooling_cursor.is_finished()`, `metadata.py:35-36` =
  `prompt_lens_cpu == seq_lens_cpu`), driven by `_pool`
  (`vllm/v1/worker/gpu_model_runner.py:3330-3343`, which reads
  `self.optimistic_seq_lens_cpu`). vllm-ascend's
  `worker/npu_input_batch.py` at this pin is `class NPUInputBatch(InputBatch)`
  with `__init__` only, so `add_request`/`get_pooling_states` are upstream's and
  the per-request states do get registered. "The Ascend runner never populates
  `pooling_states`, so the pooler sees one chunk" is **refuted at v0.23.0**
  (it is a live hypothesis on older copies — see `batch-index-desync.md`).
- Ascend-specific pooling branches are three, and they are where an Ascend
  divergence can live:
  1. `vllm_ascend/attention/attention_v1.py:1535-1538` — for
     `model_runner_type == "pooling"` **and** `not attn_metadata.causal` the
     layer goes to `_forward_encoder_attention` (`:1384-1406`), which passes
     `actual_seq_kvlen = actual_seq_qlen`: it attends **inside the current chunk
     only** and never reads the cached prefix. Correct for a true encoder
     (BERT-shaped) pooler, catastrophic for a decoder-derived embedding model
     whose second chunk then cannot see the first. One check decides it: on the
     reporter's ref, does `attn_metadata.causal` come out `True` for
     `--runner pooling` on a decoder model (`common_attn_metadata.causal`,
     set at `:312`/`:368`)? If it is `False`, this is the owner.
  2. The pooling mask branch: `vllm_ascend/attention/attention_mask.py:75-77`
     returns `get_attn_mask(2048, torch.bool)` for runner pooling, and
     `_310p/attention/attention_mask.py:139-145` returns
     `_get_causal_mask(max_seq_len)` / `_get_non_causal_mask(...)` with
     `max_seq_len = 2048` when the compressed-mask ops exist, else the class
     default `max_seqlen = 16384` (`:32-35`). It is a fixed-size *window* mask,
     not a per-step one.
  3. 310P dispatch: `_310p/attention/attention_v1.py:329-342` sends
     `ChunkedPrefill` / `PrefillCacheHit` / `SpecDecoding` to
     `forward_chunked_prefill_310` (`:235`) and `PrefillNoCache` to
     `forward_prefill_310` (`:205`), raising `NotImplementedError` for anything
     else.

## What the mask trail is *not*

`forward_chunked_prefill_310` does **not** use the pooling mask from branch 2: it
builds its own mask from a `16384 x 16384` additive matrix
(`get_splitfuse_mask`, `_310p/attention/attention_mask.py:78-100`) or the fixed
compressed `[2048,2048]` mask, and `_npu_paged_attention_splitfuse[_v2]` takes
`seq_len=qlens, context_lens=attn_metadata.seq_lens`. So "the pooling mask is
2048 wide while my chunk is 8192" is not the mechanism at this pin: a context or
chunk longer than the 16384 buffer would raise in `index_select`, not corrupt
silently, and the reporter's ~8 K tokens is below that. Spending turns diffing
the mask builders buys nothing here (a recorded session spent eight calls doing
exactly that).

## Ranked leads, each with its deciding check

1. **The vector is the pooling of the wrong span** (chunk 1 only, or the
   remainder only). Owner: the cursor/`optimistic_seq_lens_cpu` bookkeeping that
   `is_finished()` and `_pool`'s `finished_mask` depend on — on Ascend that value
   is written in the runner (`_310p/model_runner_310p.py:458-460, 491, 520-542`,
   `_correct_optimistic_seq_lens_cpu`) rather than upstream's path.
   **Check (no hardware):** embed the same text truncated at the chunk boundary
   (`max_num_batched_tokens`, in the serve log) and at the suffix; `cos ≈ 1`
   against one of them proves "wrong span". Confidence: hypothesis, but the
   cheapest one to kill.
2. **Chunk 2 is computed without the prefix** (branch 1 above).
   **Check:** `rg -n "model_runner_type == \"pooling\"" vllm_ascend/attention/` at
   the reporter's ref plus what sets `causal`; if the pooling branch is taken for
   a decoder model, that is the owner, and the fix is to keep the causal splitfuse
   path for it. Confidence: hypothesis.
   The same tail-only branch has a **second trigger that needs no chunked prefill**
   — a prefix-cache **hit** on a repeated prompt — and on v0.13.0rc1/v0.14.0rc1 the
   pooling branch is taken *unconditionally* (no `not causal`), with
   `--no-enable-prefix-caching` as the reporter-confirmed workaround and
   `--no-enable-chunked-prefill` doing nothing
   (`prefix-cache-hit-embedding-drift.md`, #5725). Short prompt + repeated string ⇒
   that page; long prompt + first call only ⇒ this one. Which flag moves the
   corruption is the discriminator, so ask for it before ranking either.
3. **Long-chunk numerics on 310P** (the splitfuse/`_npu_flash_attention` op or the
   fp16 path). **Check:** the chunk-budget sweep in the ladder below — if the
   corruption boundary does not move with `--max-num-batched-tokens`, this lead
   wins. Confidence: unverified, and it fights the restart-variance evidence.
4. **Scheduler-side chunking of a pooling request** (a step whose
   `num_scheduled_tokens` and `num_computed_tokens` disagree for the pooling
   request). Low: the chunking rule is upstream's `Scheduler.schedule`
   (`vllm/v1/core/sched/scheduler.py:690`, "chunked prefill has to be enabled
   explicitly to allow pooling requests to be chunked") and vllm-ascend's
   `core/scheduler_dynamic_batch.py:405` mirrors it.

## The ladder an answer can be shipped from

1. Keep `--no-enable-chunked-prefill` as the reporter's workaround (already
   reporter-confirmed) — state it in the first paragraph, before the mechanism.
2. `--max-num-batched-tokens` sweep (e.g. 2048 / 4096 / 8192 / 16384) on one
   fixed prompt: a boundary that moves ⇒ scheduling/accumulation/masking; a
   boundary that stays ⇒ length-dependent numerics. One axis, one launch each.
3. The "which span" embeddings of lead 1, compared against the failing vector.
4. `--enforce-eager` (drops ACL-graph replay, which is also a reuse site for
   buffers) — cheap, and it separates graph replay from eager math.
5. A smaller `--max-model-len` (different KV geometry) only if steps 2-4 leave
   two leads alive. Do not open a `torch_npu`/op-internals investigation for
   this report: the restart-variance signature makes it the least likely owner.
