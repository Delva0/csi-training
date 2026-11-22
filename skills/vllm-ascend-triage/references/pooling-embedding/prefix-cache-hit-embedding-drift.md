# The same prompt twice: a different embedding on the second call (pooling × prefix cache)

Worked case: vllm-ascend #5725 — 910B4 ×1, Qwen3-Embedding-0.6B, vllm
**0.13.0** (`72506c9834`) + vllm-ascend **v0.13.0rc1** (`1b5d5abf865f`). Reported
launch line: `python3 -m vllm.entrypoints.openai.api_server --convert embed
--trust-remote-code --port 8002 --max-model-len 32000 --max-num-seqs 20
--tensor-parallel-size 1 --gpu-memory-utilization 0.5`. `--convert embed` is a
**real** flag at that upstream ref, not a typo — `ConvertType` /
`ConvertOption` and `_RUNNER_CONVERTS["pooling"] = ["embed", "classify", "reward"]`
(`vllm/config/model.py:74-92`), with `--convert reward` deprecated in its favour
(`:815-816`) — and it resolves to the pooling runner with task `embed` (`:654-655`,
`:803`), which is what puts this request on the path below. No
`--no-enable-prefix-caching` in the line, and `enable_prefix_caching` defaults
`True` (`vllm/config/cache.py:76`). One ~1.2 K character Chinese prompt, 1024-dim
vectors.

## Reporter-confirmed evidence, all of it load-bearing

- call 1 against a standard-vLLM reference: cosine `0.9997` — the engine and the
  checkpoint are fine;
- call 2, **same string, same service**: cosine `0.5855` against call 1, and
  cosine `0.5815` against the reference — the wrong vector is not a neighbour of
  the right one;
- calls 3, 4, 5: **bit-identical to call 2** (cosine `1.0`), so the wrong value is
  stable, not noise;
- the reporter's own first reading — "似乎是命中缓存了" — was right;
- **`--no-enable-prefix-caching` removes it completely** (reporter, after a
  maintainer's hint, 2026-03-23) — this is the confirmed fact;
- the defect is **not** a single-rc regression: the reporter re-tested
  v0.14.0rc1 and v0.15.0rc1 and it persisted, so a version bump is not the fix and
  no official repair is in this snapshot.

First-call-correct / second-call-wrong / thereafter-frozen is the whole
classification: a numerical or kernel defect does not improve on the first call, and
sampling noise cannot produce bit-identical repeats. It is the *repeat* signature
(`index.md` class 7), not class 3: the prompt here is one short schedule, so
`--no-enable-chunked-prefill` is the wrong lever.

## Verified in the pinned trees (do not re-derive)

- **`embed` does read the prefix cache.** `vllm/pooling_params.py:128-135`
  (identical at vllm v0.13.0 and v0.14.1) sets `skip_reading_prefix_cache = True`
  only for `token_embed` / `token_classify`; every other pooling task, `embed`
  included, keeps `False`. The KV manager's comment explains the exception
  (`pooling_params.py:129-131`: "the output of all pooling may less than
  n_prompt_tokens").
- **The hit leaves at least one token and is block-aligned.**
  `vllm/v1/core/kv_cache_manager.py:180` (the skip condition) and `:189`
  (`max_cache_hit_length = request.num_tokens - 1`) — the same two lines at v0.13.0
  and v0.14.1 — with the comment that
  `allocate_slots()` needs `num_computed_tokens` block-aligned. So request 2 is
  scheduled as "N − k·block_size cached + a tail", never as a full recompute —
  and the same prompt always produces the same tail, which is why 2…N agree.
  The same line is why a hit *rate* reads low instead of 0: the hit is
  `B·floor((L−1)/B)`, so exact-multiple prompt lengths lose a whole block —
  `../performance-throughput/prefix-cache-hit-rate-at-block-boundary.md`.
- **Every pooling forward is routed to the encoder attention branch, which never
  sees the cached prefix.** `vllm_ascend/attention/attention_v1.py:708`
  (v0.13.0rc1) — `if attn_metadata.model_runner_type == "pooling":
  … _forward_encoder_attention(...)` with **no other condition** — and inside it
  (`:580`) both causal arms pass `actual_seq_kvlen = attn_metadata
  .actual_seq_lengths_q` (`:598`, `:610`). The non-pooling path in the same file
  is the correct contrast: `:576` `context_lens=attn_metadata.seq_lens`.
  Re-checked at v0.14.0rc1 (`52d4acfa`) / vllm v0.14.1 (`d7de043`: `:937`,
  `:819`, `:841`/`:853`) — same shape, still unconditional.
- Mechanism this predicts, matching the reporter's numbers: a cache-hit request
  computes the tail's hidden states with attention **inside the tail only**, so the
  pooled last token (Qwen3-Embedding is last-token pooling) keeps its position and
  token embedding but loses the prefix's context — a correlated-but-wrong vector
  (`cos ≈ 0.58`, not ≈ 0), identical on every later hit.
- By v0.23.0 the same branch is gated `and not attn_metadata.causal`
  (`chunked-prefill-vector-corruption.md` §lead 2), so the
  `path:line` and the guard are both ref-specific: re-grep
  `rg -n 'model_runner_type == "pooling"' vllm_ascend/attention/` plus the two
  `actual_seq_kvlen` sites before quoting either.
- **CI has no repeat.** `tests/e2e/singlecard/pooling/test_embedding.py`
  (`test_embed_models_correctness`) embeds two *different* queries once each with
  `cudagraph_capture_sizes=[4]` — the cached-prefix path is untested at these
  pins, which is why the defect survived three rcs.

The label split matters: *the workaround and the prefix-cache dependency are
reporter-confirmed*; the attention-branch frame is **source-verified at the two
pins** and remains the leading mechanism rather than a settled cause — no fix and
no diff is in the snapshot.

## Three hypotheses, one matrix (they all predict "1st right, 2nd wrong, then frozen")

| hypothesis | mechanism | what decides it |
|---|---|---|
| **A cache-hit request attends only its tail** (above) | wrong vector depends on *this* prompt's cached prefix | a nonce-suffixed prompt is right **every** time; `--no-enable-prefix-caching` removes it; `--enforce-eager` does not |
| **B graph / compile split** | call 1 eager, later calls replay an ACL graph | `--enforce-eager` removes it while the cache flag does not; the paste shows `FULL_DECODE_ONLY`/`PIECEWISE` capture |
| **C cross-request state residue** | `pooling_states`/hidden-state buffer carried between requests | the second vector changes with the *previous* request's content: `[X, X]` and `[Y, X]` differ |

Run in one process/one serve session: `X, X` → note V2a; `Y, X` → note V2b;
`X + <random nonce>, ×3`; then one restart each with `--no-enable-prefix-caching`
and with `--enforce-eager`. `V2a == V2b` and the nonce run all-correct ⇒ A.

## Ladder to ship from (cheapest first)

1. **Nonce suffix, no restart**: append a random string to the prompt and embed it
   three times — cache cannot hit, so all three must match the reference. One
   script, no configuration, and it is the fastest confirmation of A.
2. **`--no-enable-prefix-caching`** on the same command (reporter-confirmed
   workaround; keep the rest of the launch line untouched). This is rung 1 of the
   *first* message, not a later step.
3. `--enforce-eager` as the graph control (B). Cheap, and it must *not* be
   credited unless its own line appears (`Compilation disabled, using eager mode
   by default`).
4. Say plainly what this costs a batch-indexing user: any two prompts sharing a
   long prefix (same system prefix, same template, an overlapping document chunk)
   take the same tail-only path, so the repeated-string form in the report is the
   *smallest* instance of the defect.
5. Fix direction and acceptance: for a decoder-derived pooler use the causal
   path whenever `num_computed_tokens > 0` (or mirror upstream's
   `skip_reading_prefix_cache=True` semantics for `embed`), and validate with one
   real request — the same prompt three times, mutually `cos ≥ 0.9999` **and**
   `cos ≥ 0.999` against the reference implementation. A `try/except` or a
   "recompute everything" guard that leaves the numerics unvalidated is not the
   repair (`../../SKILL.md` rule 3).

## A check that looks cheap and is not

Do **not** send the reporter to `usage.prompt_tokens_details.cached_tokens` in an
embeddings response at these pins: `/v1/embeddings` builds
`UsageInfo(prompt_tokens=…, total_tokens=…)` and never fills `prompt_tokens_details`
(`vllm/entrypoints/pooling/embed/serving.py:148-160` at v0.13.0), and the batch
aggregator hardcodes `num_cached_tokens=0` (`:572-577`) — an absent or zero field
there is not evidence either way. The local evidence for "the second call hit" is
the nonce ablation, plus the engine's own interval line `Prefix cache hit rate: …`
(`vllm/v1/metrics/loggers.py:228`) or `/metrics` counters around the two calls.

## Version and hardware scope

Both pins are one-commit snapshots: v0.13.0rc1 for the reported environment,
v0.14.0rc1 for the reporter's own retest (line numbers above). The reporter also
reproduced it on v0.15.0rc1, which is **not** in the snapshot — so "is it fixed
upstream" cannot be answered from the tree; the workaround is the answer, and the
`--no-enable-prefix-caching` restart is the first thing to say. A later report of
the same shape on a newer ref should re-grep the three sites before reusing the
mechanism.
