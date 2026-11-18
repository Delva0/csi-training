# Pooling and embedding models: wrong vectors, not wrong text

Pooling runs (`--runner pooling`, `--task embed`, `/v1/embeddings`, `Qwen3-Embedding-*`, rerankers)
have their own attention route in vllm-ascend, and it is the first suspect when an embedding output
differs from a GPU reference while the service is healthy, serves fast, and returns no error anywhere.
The owning code is the attention router, not the pooler and not the `dimensions` (Matryoshka)
truncation.

## Symptom shapes that belong here

- Cosine similarity against the GPU reference is ~0.999 for some rows and 0.25–0.35 for others in the
  same request set — a content difference, not numeric noise.
- First request after start matches GPU; later requests in the same service do not (reporter's own
  follow-up in #7192).
- `--no-enable-prefix-caching` makes every request agree with GPU and with each other.
- The report may blame the output width (`dimensions=1024` differs, `4096` "no difference"). Treat
  that as a coverage artefact, not a mechanism (see "Pitfalls").

## Verified route at the reported ref

Read these call sites at the tag before anything else:

- `vllm_ascend/attention/attention_v1.py`, `AscendAttentionBackendImpl.forward` — the pooling branch:

  | ref | router line | form |
  | --- | --- | --- |
  | `v0.15.0rc1` | `:975` (`:974-978` with the surrounding `if`) | `if attn_metadata.model_runner_type == "pooling":` → `_forward_encoder_attention` |
  | `v0.16.0rc1` | `:986` | same, un-fixed |
  | `v0.17.0rc1` | `:985` | same, un-fixed |
  | `v0.18.0rc1` | `:971` | fixed: `... == "pooling" and not attn_metadata.causal` (backport `d3de7333d`) |
  | `v0.19.1rc1` | `:1132` | fixed, plus further pooling checks at `:1183/:1213/:1228` |

  No causal or cache-content test before the fix.
- `_forward_encoder_attention` (same file, ~`:855` at `v0.15.0rc1`) launches
  `torch_npu.npu_fusion_attention` with `input_layout="TND"` and
  `actual_seq_kvlen=attn_metadata.actual_seq_lengths_q` — the **current step's** lengths — so the
  operator attends over the `key/value` of this forward pass only and never reads the paged KV cache.
  `reshape_and_cache(key, value, kv_cache, attn_metadata)` still runs *before* the branch
  (`:972-973` at `v0.15.0rc1`), so the cache is written and then bypassed.
- `vllm_ascend/attention/attention_mask.py:86` — `get_attention_mask(model_config)` returns the dense
  `get_attn_mask(2048, torch.bool)` mask when `model_config.runner_type == "pooling"` instead of the
  splitfuse mask, which is consistent with an encoder-style attention and confirms the branch is
  deliberate, not dead code.

Consequence: whenever the scheduler hands the model fewer tokens than the full prompt — a prefix-cache
hit (`num_computed_tokens > 0`) — the hidden state that feeds the pooler was computed over the suffix
alone. Requests that hit the cache diverge from GPU; the first, cache-cold request agrees, which is
exactly the observed order dependence.

## Mechanism (verified by the fix, not inferred)

`npu_fusion_attention` does not support prefix caching (fix PR text, verbatim: "But
`npu_fusion_attention` does NOT support prefix caching → Result: Numerical mismatch when KV cache is
managed by prefix caching"). The fix refines the gate so only true (non-causal) encoders keep
`npu_fusion_attention`; pooling models (the family the fix tests with `Qwen3-Embedding-0.6B`) now go
through `npu_fused_infer_attention_score`, which does support the cache, and the redundant causal
branch inside `_forward_encoder_attention` was deleted.

## Version status (verified with `git tag --contains` / `merge-base --is-ancestor`)

- Fix: `bd004043d` — "[BugFix][APC] Fix prefix caching support for embedding models (#7452)",
  2026-03-28, on `main`; first mainline tags `v0.19.1rc1` … `v0.30.0rc2` contain it.
- Backport: `d3de7333d` — "[BugFix][v0.18.0][cherry-pick] Fix embedding prefix caching for APC
  (#7894)", 2026-04-01, contained in `v0.18.0rc1` and `v0.18.0`. So `v0.18.0` already has the guard
  even though the mainline commit does not reach it — check the cherry-pick before quoting
  "absent from `v0.18.0`".
- Affected: `v0.15.0rc1`, `v0.16.0rc1`, `v0.17.0rc1` (and anything older with the same router).
- Regression test added by the fix: `tests/e2e/singlecard/pooling/test_embedding.py`
  (`test_embed_models_using_prefix_caching_correctness`, `Qwen3-Embedding-0.6B`, tolerance 1e-2). Its
  absence is a hint that a build predates the fix; `tag --contains` is the authority.
- The guard survives at HEAD (`attention_v1.py:1371`), but HEAD grew three more pooling branches in
  `AscendC8AttentionBackendImpl` (`:1423`, `:1456`, `:1471`) that carry **no** causal guard. Recheck
  the exact line at the ref, and do not assume a C8/int8-KV pooling deploy inherits the #7452 fix.

## Workaround

| tier | change | cost |
| --- | --- | --- |
| feature switch off | `--no-enable-prefix-caching` (reporter-confirmed in #7192: 「关闭前缀缓存特性」) | loses APC speedup and repeated-prefix prefill latency returns; the embedding then equals the reference |
| upgrade / pin | `v0.18.0`+ (backport `d3de7333d`) or `v0.19.1rc1`+ (mainline `bd004043d`) | version jump; re-check vLLM/torch_npu pairing (`../vllm-ascend/version-pairing.md`) |

Say which one you are proposing; it is a feature-off workaround or a pin, not a fix.

## Discrimination before you assert this class

1. Send one prompt, then the same prompt again to the same server (`--enable-prefix-caching` on).
   First-vs-second divergence with the first matching the GPU reference is the APC signature; a
   uniform offset on every request is a dtype/quantisation/pooler difference instead.
2. Re-run with `--no-enable-prefix-caching`. Equal to the reference ⇒ this class; still differing ⇒
   the pooler or the embedding truncation, not the attention route.
3. Only then read the branch at the tag and check release inclusion.

## Pitfalls

- **Do not chase `dimensions`/Matryoshka.** The `dimensions` parameter slices and re-normalises the
  *pooled* vector after the forward pass; it cannot turn a wrong hidden state into a correct
  embedding, and it cannot make a 4096-wide comparison "no difference" for the same request. The
  dim-dependence in #7192 is best explained by which requests the 1024-wide table covered (six rows)
  versus the 4096 run. `[hypothesis]` Falsified by: rerun the 4096-width comparison with the cache
  warm and a repeated prefix — the same rows should diverge.
- Do not blame the pooler (`vllm/model_executor/layers/pooler.py`) or `--runner pooling` selection;
  the pooling branch is entered correctly, it just computes attention over too little of the sequence.
- A GPU-side difference between first and later requests exists too (batched-matmul
  non-determinism, high similarity); only the NPU side shows low similarity. Ask for the cosine
  numbers before agreeing that "both differ".
- The router condition tests only `model_runner_type`, so every pooling mode (`mean`, `last`, CLS) and
  every `--dimensions` value take the same branch: a difference between pooling modes is not a lead
  here.

## Check before reuse

```bash
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'model_runner_type == "pooling"' \
  <ref> -- vllm_ascend/attention/attention_v1.py
git -C /workspace/repos/vllm-project/vllm-ascend tag --contains bd004043d
git -C /workspace/repos/vllm-project/vllm-ascend merge-base --is-ancestor d3de7333d <ref>
```

Read the router line and `_forward_encoder_attention`'s `actual_seq_kvlen` argument at the reported
tag; if the router already carries `and not attn_metadata.causal`, the report needs a different
mechanism (KV-cache dtype, weight quantisation, or the pooler itself).

## Open points (do not paper over)

- #7192's reporter first framed the bug as "output dimension 1024 differs, 4096 does not". The APC
  mechanism explains *which requests* differ, not why the reported dimension mattered; the
  `dimensions` / Matryoshka truncation path (slice + renormalise) is a separate axis to check on its
  own. Say which of the two you verified.
- Chunked prefill without APC should hit the same branch (`actual_seq_kvlen` = this chunk only). That
  is an inference from the code, not a recorded case: check it by comparing a prompt longer than
  `--max-num-batched-tokens` with APC off, and label it `hypothesis` until then.
- No NPU here: nothing on this page was reproduced on hardware; the fix and its tag containment are
  read from the repository, the reporter's experiments are `reported`.

## Reported case: #7192

`[thread]` vllm-ascend `0.15.0RC1`, A2 (8×910B4), `npu-smi 25.2.3`, CANN 8.5.0, Qwen3-Embedding-8B
served with the pooling runner; a six-row cosine table against a GPU run at `dimensions=1024`
(0.9999, 0.9999, 0.2926, 0.9998, 0.2803, 0.3169). Reporter follow-up: first request vs later requests
differ on NPU with low similarity, GPU's own spread is small, GPU prefix-cache hit rate is higher, and
disabling prefix caching makes NPU match GPU. Maintainer advice in the thread: temporary
`--no-enable-prefix-caching`, long-term #7452. A later commenter asks whether #5725 is the same bug —
a cross-reference, not evidence; read #5725 before claiming it is the same defect, and the reply never
confirmed it. `[code]` fix `bd004043d` / backport `d3de7333d` as above. Issue state when read
(2026-11-17): `closed`, last thread activity 2026-03-26. No labels; the closing comment was never
checked, so the fix is established from the commit and the reporter's own confirmation, not from the
closure.
