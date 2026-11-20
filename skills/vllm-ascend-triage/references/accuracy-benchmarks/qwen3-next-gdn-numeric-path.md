# Qwen3-Next (hybrid GDN) on Ascend: digit-heavy prompts answered as soup

Worked case #4057, plus its same-version precedent #3689. Verified at vllm-ascend
v0.11.0rc0 (`00ba07102212c7c7a40de427f09848f2e203c498`) + vllm v0.11.0rc3
(`8ce5d3198d00631a76e1aa02a57947b46bc7218c`).

## Scope and symptom grading

- Model: hybrid linear-attention MoE (3 Gated-DeltaNet layers : 1 full-attention
  layer) — Qwen3-Next-80B-A3B-Instruct and its siblings.
- Symptom: a *short* prompt (tens to a few hundred tokens) comes back as a run of
  digits, binary chatter or a phrase loop; ordinary chat on the same server looks
  fine; no traceback, process alive, `finish_reason: length`.
- The recorded trigger is not length but **precise digit work**: #4057's prompt is
  a 19-token comparison ("9.11 和 9.8 哪个数更大"), #3689's is a multiple-choice
  question whose stem carries `2000`/`3000`. In #3689 the output *loses the
  prompt's own numbers* (it re-reads 2000/3000 as 1024/1010/1101) and then loops
  on a repeated sentence until `max_tokens`.
- Grade before investigating: `usage.prompt_tokens` (short ⇒ here, not the
  long-context page), `finish_reason`, and whether the degradation starts at the
  first tokens. Sampling-driven collapse on a pathological prompt is
  `short-prompt-corruption.md`; this page is the model's *numeric path*. If the
  reporter's working and failing commands differ by `--block-size`,
  `--enforce-eager`, `--enable-prefix-caching` or `--mamba-cache-mode`, that confound
  is graded first and the page is `hybrid-block-size-corruption.md`.

## What vllm-ascend replaces on this path at v0.11.0rc0

The whole linear-attention prefill kernel is rewritten downstream, so a numeric
divergence from the training-time kernel is expected by construction. This is a
mechanism map, not a proven cause.

| Site | Fact |
|---|---|
| `vllm_ascend/patch/worker/patch_common/patch_triton.py:12-16` | rebinds five upstream symbols: `causal_conv1d_fn` / `causal_conv1d_update` → `vllm_ascend/ops/casual_conv1d.py`, `fused_recurrent_gated_delta_rule_fwd_kernel` → `vllm_ascend/ops/sigmoid_gating.py`, `LayerNormFn` and **`chunk.chunk_gated_delta_rule` → `torch_chunk_gated_delta_rule`** |
| `vllm_ascend/ops/fla.py:124-218` | the replacement is a pure-torch reference: all inputs `.to(torch.float32)`, sequence padded to a multiple of `chunk_size=64`, the chunk recurrence rebuilt with an explicit `for i in range(1, chunk_size)` loop, `scale = 1/sqrt(head_dim)` computed internally |
| same, `fla.py:146-151` | `query`/`key` get `F.pad(...).repeat_interleave(2, dim=1)` — a **hard-coded ×2 head expansion**, consistent only while linear value heads = 2 × key heads (Qwen3-Next-80B: 32 vs 16; any TP keeps the 2:1 ratio, so this config is inside the assumption) |
| `vllm_ascend/models/qwen3_next.py:337-355` | the prefill branch loops over sequences and calls `chunk_gated_delta_rule(query=…, key=…, value=…, g=…, beta=…, initial_state=…, output_final_state=True, use_qk_l2norm_in_kernel=True)` — keyword names that match the replacement, and **no `cu_seqlens`**, which the replacement does not accept (upstream's varlen call passes it, `vllm/model_executor/models/qwen3_next.py:554-562`; upstream's signature also has `scale`/`head_first`, `vllm/model_executor/layers/fla/ops/chunk.py:112-121`) |
| `vllm_ascend/models/qwen3_next.py:310,383` | the **decode** branch calls `fused_recurrent_gated_delta_rule`, i.e. the patched Triton kernel — prefill and decode run different implementations of the same recurrence |
| `docs/source/tutorials/multi_npu_qwen3_next.md:4` | the project states the model "is using Triton Ascend which is currently experimental … behavioral changes around stability, accuracy and performance" |
| `docs/source/user_guide/release_notes.md:33` | at this release EP and MTP do not work with Qwen3-Next — both are out of scope for a v0.11.0rc0 report |

Binding caveat: the patch rebinds the module attribute, while the model does
`from … .chunk import chunk_gated_delta_rule` when it loads. In the recorded setup
the model module loads after patching, so the fp32 torch function is what runs; if
a stack shows the Triton `chunk_gated_delta_rule_fwd`, the ordering differed —
that is `../upstream-downstream-drift/patch-import-side-effect.md`, and it changes
which implementation produced the numbers.

## Reporter-confirmed narrowing (#4057)

The reporter later posted the deployment command publicly
(2025-11-07T07:41:01Z, same issue):

```
vllm serve /data2/model/Qwen3-Next-80B-A3B-Instruct --served-model-name "Qwen3-Next-80B-A3B-Instruct" \
  --tensor-parallel-size 4 --max-model-len 8192 --gpu-memory-utilization 0.7 --enforce-eager --port 8010
```

That single line closes four ablation rows before they are proposed: **graph mode
off** (`--enforce-eager`), **no MTP / no speculative config**, **no
`--enable-expert-parallel`**, **no quantization flag** — and, at this version, no
prefix cache to enable (see `../performance-throughput/zero-prefix-cache-hit-hybrid.md`).
`--max-model-len 8192` deviates from the tutorial's `4096` but is a documented-valid
value and irrelevant at 19 prompt tokens. What remains is the linear-attention
numeric path above, KV/state dtypes, and sampling.

## Ablation matrix (one restart or one request per row)

| Row | Change | Corrupt here means | Clean here kills |
|---|---|---|---|
| 1 | same prompt, `temperature: 0`, `max_tokens: 64` | deterministic degeneration inside the model | sampling/top-k collapse |
| 2 | same-length prose prompt, no digits | — | "short prompt" as the trigger; digit recall is |
| 3 | `return_token_ids: true` on the failing request | compare the prompt token ids the engine saw | tokenizer/template corruption |
| 4 | a non-hybrid model on the same host/flags (e.g. Qwen3-30B-A3B, TP=4) | platform-wide numerics | this model's GDN path |
| 5 | the same checkpoint on a GPU-hosted vllm, or `transformers` greedy for 32 tokens | model/weights | Ascend kernels |
| 6 | TP=2 or TP=8 (if cards allow) | per-rank shapes into the fp32 chunk rule | TP=4-specific sharding |

Rows 4-5 are the ones that decide ownership; rows 1-3 cost one request each and
should be in the first answer as the reporter's next step.

## Precedents and what is *not* established

- #3689 — same model, vLLM 0.11.1.dev0 + vllm-ascend 0.11.0rc1.dev103, TP=4,
  `enforce_eager=True`, `max_model_len=32768`, `temperature 0.7 / top_p 0.8 /
  top_k 20`: "gets many repetitive, redundant tokens", output drifting through
  digit/binary chatter into a phrase loop.
- #3571 — same tutorial, reporter-asserted that `Qwen3NextAttention.attn_output`
  is all zeros. At the pin that class matches upstream's gated-attention forward,
  so treat it as a check to run (instrument `attn_output` once) rather than as the
  cause.
- Neither precedent carries a confirmed owner, and no source in the snapshot
  settles whether the fp32 re-implementation, the Triton-Ascend decode kernel, or
  a checkpoint/sampling interaction is responsible. Say that explicitly; do not
  promote "the two kernels differ" to "the kernels are wrong".
- Recheck conditions: the patch list, the `repeat_interleave(2)` expansion and the
  per-sequence prefill loop are v0.11.0rc0 facts. Re-grep
  `patch_triton.py`, `ops/fla.py` and `models/qwen3_next.py` at any later pin — the
  same three sites moved in later releases.
