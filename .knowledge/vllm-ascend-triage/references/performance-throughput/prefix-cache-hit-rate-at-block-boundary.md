# A hit rate that is low, or steps at prompt-length multiples, instead of being 0 %

Verified at vllm-ascend v0.18.0rc1 (`99e1ea0fe685`) + vllm v0.18.0 (`bcf2be961200`)
(recorded #7965); the upstream `max_cache_hit_length` line is unchanged back to
v0.13.0/v0.14.1, where `../pooling-embedding/prefix-cache-hit-embedding-drift.md`
records it. Re-grep the four sites below at any other pin.

## The answer first

A rate that is **nonzero but small, and that moves as the prompt length moves**, is
arithmetic, not a defect. Every lookup asks for `num_tokens − 1` tokens — the last
prompt token must be recomputed to produce logits, and `allocate_slots()` requires a
block-aligned `num_computed_tokens` — and every group rounds what it finds down to a
multiple of the coordinator's `lcm_block_size`. With block size `B` and prompt length
`L`:

```text
hit(L)  = B · floor((L − 1) / B)          rate(L) = hit(L) / L
```

| `L` | hit | rate |
|---|---|---|
| `L ≤ B` | 0 | 0 % — nothing shorter than one block is ever cached |
| `L = k·B` | `(k−1)·B` | `(k−1)/k` — the sawtooth's minimum: one whole block is lost |
| `L = k·B + r`, `1 ≤ r < B` | `k·B` | `1 − r/L` — just past a boundary, ≈ 100 % |

A workload whose prompt lengths are **exact multiples** of the block size therefore sits
on the minima, and a length one token longer jumps to the maximum. That is the whole
mechanism behind a report shaped "输入 token 数为 N 的倍数时 prefix cache 命中率骤降 / 个位数".
Reporter-confirmed on #7965 (Qwen3.5-9B, `--enable-prefix-caching`, 1024·n lengths):
「正常现象，因为 prefill 需要至少计算一个 token，最后一个 block 不会进行命中」. Nothing to
fix — not a vllm-ascend defect, not a kernel, not a config mistake.

Four "bad hit rate" readings, four owners — grade before answering:

| reading | owner |
|---|---|
| exactly `0.0 %` at every prompt length | the switch / hybrid gate — `zero-prefix-cache-hit-hybrid.md` |
| nonzero, steps with `L`, minima at multiples of one size | this page |
| `External prefix cache hit rate: 0.0%` beside a healthy local rate | the pool — `../kv-pool-store/pp-topology-pool-init-and-zero-hits.md` |
| a cache *hit* that returns the wrong numbers (embeddings) | `../pooling-embedding/prefix-cache-hit-embedding-drift.md` |

## Why `B` can be tens of times 128 (the amplifier)

One block per request is lost, so the size of the loss is `B`, and `B` is not always 128:

| path | site at v0.18.0rc1 | `B` |
|---|---|---|
| dense | `vllm_ascend/utils.py:1112-1115` coerces to 128 whenever prefix caching **or** chunked prefill is on | 128 |
| hybrid-marked (`IsHybrid`) | `vllm_ascend/utils.py:1107-1110` returns **before** that coercion, then the patched config derives it: `vllm_ascend/patch/platform/patch_mamba_config.py:58` `attn_block_size = 128 · cdiv(ssm_page, 128 · attn_token_page)`, assigned at `:66-71` with the log `Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size.` | `128·k`, often 256-2048 |

In `align` mode the same patch (`:89-90`) sets `mamba_block_size = cache_config.block_size`,
so the group specs are `[B, B]`, the coordinator's alignment is `lcm = B`
(`vllm/v1/core/kv_cache_coordinator.py:451`, `:521`) and the managers round the hit down
at `vllm/v1/core/single_type_kv_cache_manager.py:562-568`. The pin's own tutorial states
the consequence: `docs/source/tutorials/models/Qwen3.5-27B.md:137` — "the block_size may
be adjusted to 2048, which means that any prefix shorter than 2048 will never be cached."

Whether the derived branch is reachable at all is a **class-marker** fact, not a model
family: `is_hybrid` is `ClassVar[Literal[True]]` on `IsHybrid`
(`vllm/model_executor/models/interfaces.py:780-785`), read by
`vllm/config/model.py:1462-1470`, and its only caller is
`vllm/config/vllm.py:1554-1555` (`if self.model_config.is_hybrid:`). At vllm v0.18.0 the
marker is on `qwen3_5.py:659` (`Qwen3_5ForConditionalGeneration`) and **not** on the
text-only `Qwen3_5ForCausalLMBase` / `Qwen3_5ForCausalLM` (`:539`, `:634`) — a text-only
Qwen3.5 serve keeps `B = 128` and can only lose 128 tokens per request. Re-grep the
marker before reusing this at another pin.

## Check ladder (the first two rows are the whole answer)

1. **The serve log's own line fixes `B` and the mode.** `rg -n 'Setting attention block
   size to|Mamba cache mode is set to' <serve.log>` — `patch_mamba_config.py:68` and
   `vllm/model_executor/models/config.py:368-380` ("Mamba cache mode is set to 'align' …
   when prefix caching is enabled"). No such line ⇒ `B = 128`, the derived-size branch is
   out, and the loss is one 128-token block.
2. **Read one request's hit, not the interval average.** Add
   `--enable-prompt-tokens-details` (`vllm/entrypoints/openai/cli_args.py:138`; the field
   is filled at `chat_completion/serving.py:1218`) and send the same prompt twice: the
   second response's `usage.prompt_tokens_details.cached_tokens` must equal
   `B·floor((L−1)/B)`. Counters: `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`
   (`vllm/v1/metrics/loggers.py:504-520`; the interval line's `Prefix cache hit rate: %.1f%%`
   is `prefix_caching_metrics.hit_rate`, printed at `:234-245` from the observation at
   `:169`). A live hit here retires every "the cache is off" story.
3. **Length sweep, four requests in one session:** `L−1, L, L+1, 2L` over the same shared
   prefix. Rates stepping exactly as the table ⇒ this page, close the case; flat 0 at every
   length ⇒ the gate page.
4. Only if the sweep disagrees: `--no-enable-prefix-caching` as the on/off control (one
   restart) and a nonce-suffixed prompt to force a cold read.

## What to send, and what not to propose

- Quote the formula with **their** `L` and the `B` from step 1 instead of reproducing their
  percentage: the percentage depends on a reporter-side value (the derived block size) and
  an interval average mixes cold first passes in. A magnitude that does not reproduce is
  not a second mechanism.
- No flag removes the tail loss. Lowering `--block-size` only moves the dense path, and on
  a hybrid model a value below the derived `attn_block_size` is overwritten
  (`patch_mamba_config.py:66`), so block-size tuning is not a fix to recommend.
- Not a kernel / quantization / EP question: no device error, no wrong token, the text is
  correct — the cost is recomputed prefill. Do not ask for plogs or profilers.
- If the complaint is really cost, the lever is workload shape: repeat requests whose
  shared prefix spans many blocks, so `1 − B/L` is small, and set benchmark expectations
  from `B` first.

## Scope / recheck

Upstream's comment at `vllm/v1/core/kv_cache_manager.py:195-200` calls the last-token
recompute a limitation that "could slightly improve performance in the future", so this
arithmetic is ref-specific: re-read `max_cache_hit_length` before quoting it elsewhere.
The recorded report named only the family (Qwen3.5-9B) and never posted its derived block
size, so what the reporter confirmed is the mechanism — one block lost at exact-multiple
lengths — not a specific `B`. Treat an exact percentage as unestablished.
