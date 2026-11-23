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
     whose second chunk then cannot see the first. **Already checked and
     refuted on this report's path at v0.23.0 — do not re-derive it.** The v1
     runner hard-codes `causal=True` when it builds
     `AscendCommonAttentionMetadata` (`worker/model_runner_v1.py:3182`), the
     per-layer metadata's own default is `True`
     (`attention/attention_v1.py:208`) and it is copied from the common
     metadata (`attention/utils.py:280`), and 310P's impl overrides
     `forward_impl` to dispatch on `attn_state` alone
     (`_310p/attention/attention_v1.py:310-345`) — so on this runner the parent's
     `not causal` test is the only gate into `_forward_encoder_attention` and the
     hard-coded `True` never satisfies it. The only `causal=False`
     writers in `vllm_ascend/` are the spec-decode proposers
     (`spec_decode/dflash_proposer.py:146,195`) and the v2 runner's builder
     parameter (`worker/v2/attn_utils.py:118,158`, default `True`, no caller
     passes `False`). The reporter's finding that
     `--no-enable-prefix-caching` changes nothing fits the same reading, from the
     other trigger's side. Recheck is one grep,
     `rg -n 'causal=' vllm_ascend/`; it reopens only where the pooling branch
     loses its `not causal` gate (v0.13/v0.14, #5725) or the report comes from
     the v2 runner.
  2. The pooling mask branch: `vllm_ascend/attention/attention_mask.py:75-77`
     returns `get_attn_mask(2048, torch.bool)` for runner pooling, and
     `_310p/attention/attention_mask.py:139-145` returns
     `_get_causal_mask(max_seq_len)` / `_get_non_causal_mask(...)` with
     `max_seq_len = 2048` when the compressed-mask ops exist, else the class
     default `max_seqlen = 16384` (`:32-35`). It is a fixed-size *window* mask,
     not a per-step one. That branch is also why a title-only hit such as #6471
     ("310P does not support runner_type='pooling'", a VL-Embedding report) is
     not a routing result at v0.23.0: the 310P mask builder handles the pooling
     runner explicitly, `rg -n "does not support runner_type" vllm_ascend/`
     returns nothing at the pin, and the reporter's own serve runs it — so do not
     answer "pooling is unsupported on 310P" from that title.
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
   is written in the runner (`_310p/model_runner_310p.py:458-460` fills it from
   `num_computed_tokens + num_scheduled_tokens`, `:491`
   `discard_requests_mask = optimistic_seq_lens_cpu < num_tokens`, `:520-542`
   copies it into `seq_lens`) rather than on upstream's path. **Not**
   `_correct_optimistic_seq_lens_cpu` (`worker/model_runner_v1.py:1613-1641`):
   that is async *spec-decode* drift correction, guarded by
   `_needs_seq_lens_cpu_sync and use_async_spec_decode and
   valid_sampled_token_count_gpu is not None` (`_310p/model_runner_310p.py:530-542`),
   so a plain pooling serve never reaches it. Padded rows cannot leak into the
   accumulator either: upstream `_pool` slices `hidden_states[:num_scheduled_tokens]`
   before pooling and vllm-ascend overrides no `_pool`.
   **Check (no hardware):** embed the same text truncated at the chunk boundary
   (`max_num_batched_tokens`, in the serve log) and at the suffix; `cos ≈ 1`
   against one of them proves "wrong span". Confidence: hypothesis, but the
   cheapest one to kill.
2. **Long-chunk numerics on 310P** (the splitfuse/`_npu_flash_attention` op or the
   fp16 path). **Check:** the chunk-budget sweep in the ladder below — if the
   corruption boundary does not move with `--max-num-batched-tokens`, this lead
   wins. Confidence: unverified, and it fights the restart-variance evidence.
3. **Scheduler-side chunking of a pooling request** (a step whose
   `num_scheduled_tokens` and `num_computed_tokens` disagree for the pooling
   request). Low: the chunking rule is upstream's `Scheduler.schedule`
   (`vllm/v1/core/sched/scheduler.py:690`, "chunked prefill has to be enabled
   explicitly to allow pooling requests to be chunked") and vllm-ascend's
   `core/scheduler_dynamic_batch.py:405` mirrors it.

**Not a lead on this path:** "chunk 2 never sees chunk 1". It was ranked second
in earlier sessions and is refuted for the v1 runner above. It stays live in its
other setting — the same tail-only branch also fires without any chunked prefill,
on a prefix-cache **hit** for a repeated prompt, and on v0.13.0rc1/v0.14.0rc1 it
is taken *unconditionally* (no `not causal`), with `--no-enable-prefix-caching`
as the reporter-confirmed workaround and `--no-enable-chunked-prefill` doing
nothing (`prefix-cache-hit-embedding-drift.md`, #5725). Short prompt + repeated
string ⇒ that page; long prompt + first call only ⇒ this one. Which flag moves
the corruption is the discriminator, so ask for it before ranking either.

## The reported workaround is a three-flag change

The "fixed" server is not the failing server minus one flag. The failing line
carries `--compilation-config '{"cudagraph_capture_sizes": [1024, 512]}'` and
`--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'`;
the workaround line carries **neither** and no `--enforce-eager`, so the fix
ablates three flags at once. Isolation run: the failing line plus only
`--no-enable-chunked-prefill`, everything else held constant. If the corruption
survives, the accumulator is out and the owner is the capture-size/padding
geometry or the fused norm+quant numerics — `fuse_norm_quant` is the least
likely of the three on an unquantized fp16 checkpoint, so test the graph side
first with `--enforce-eager`. Hold this caution against any "flag X fixes it"
line in this class: credit the flag only after a one-flag ablation.

## Restart variance: two environment controls

Two facts in the report are cheap controls, and both beat ranking kernels:

- some broken sessions ran after a **concurrent locust load burst** on the same
  cluster ("behaved like affected by earlier concurrent load"). If the corruption
  needs a prior request, the owner is state carried **across** requests — a
  reused `PoolingStates.hidden_states_cache` or a batch slot reused before
  `remove_request` — not chunk-boundary arithmetic. Check: restart, then embed
  **one** request on an otherwise idle server; a correct first vector followed by
  a wrong second one retires lead 1 here and hands the shape to
  `batch-index-desync.md`.
- the failing line's capture sizes are non-default and the "fixed" session ran
  without them, so any session-to-session difference in the launch line also
  changes graph geometry. `--enforce-eager` keeps graph replay out of the
  comparison, and it is ladder step 4.

## The ladder an answer can be shipped from

1. Keep `--no-enable-chunked-prefill` as the reporter's own working configuration,
   in the first paragraph and before the mechanism — but mark it as a three-flag
   change and pair it with the isolation run above, rather than presenting it as
   proof of the accumulator.
2. `--max-num-batched-tokens` sweep (e.g. 2048 / 4096 / 8192 / 16384) on one
   fixed prompt, with `--compilation-config` / `--additional-config` /
   `--enforce-eager` identical across the sweep: a boundary that moves ⇒
   scheduling/accumulation/masking; a boundary that stays ⇒ length-dependent
   numerics. One axis, one launch each.
3. The "which span" embeddings of lead 1, compared against the failing vector.
4. `--enforce-eager` (drops ACL-graph replay, which is also a reuse site for
   buffers) — cheap, and it separates graph replay from eager math.
5. A smaller `--max-model-len` (different KV geometry) only if steps 2-4 leave
   two leads alive. Do not open a `torch_npu`/op-internals investigation for
   this report: the restart-variance signature makes it the least likely owner.
