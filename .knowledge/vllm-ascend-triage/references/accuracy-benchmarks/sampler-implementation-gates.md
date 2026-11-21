# Which sampler implementation ran — the env gate moved between releases

## Scope

A healthy serve whose complaint sits at the **sampling** layer rather than in the
forward: a low rate of answers that pick the wrong script or word ("高并发场景下英文出现
概率较高（0.1% 左右）", #6338), one off-prompt token spliced into otherwise correct text,
or `-inf` printed inside the returned top-k — where the body carries `temperature > 0`
with `top_k` and/or `top_p`. It answers one question before any numeric theory:
**which top-k/top-p implementation was installed at the reported pin, and what turns it
off.**

Not this page: greedy output that differs (`repeated-run-nondeterminism.md`); output
differing only with concurrency while `VLLM_BATCH_INVARIANT=1` is exported
(`batch-invariant-coverage.md`); a whole-answer constant token
(`constant-token-degeneration.md`); the v0.18+/v0.20 `random_sample()` side-stream
family (`sampler-side-stream-corruption.md`); the mask-vs-logits reading of a `-inf`
paste (`topk-logprob-negative-inf.md`, whose v0.9.1 gate is the *first row* below).

## Rule 0 — the gate's name and default are per-release, and an `echo` of the wrong name answers nothing

| Release | Flag | Default there | What it replaces |
|---|---|---|---|
| v0.9.1 | `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | unset ⇒ native | only the patched `TopKTopPSampler.forward_native` (`patch/worker/patch_common/patch_sampler.py:104-106`) — `topk-logprob-negative-inf.md` |
| v0.9.2rc1 (introduced) | `VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION` | **`0`** — "we'll consider to enable it by default in the future" (`docs/source/user_guide/release_notes.md:374`, PR #1732) | the sampler swap below |
| v0.11.0 (`2f1aed98ccdb`) | same name | **`'1'`** | the **whole** sampler: `worker/model_runner_v1.py:270-275` imports `AscendSampler` when `envs_ascend.VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION`, else upstream `Sampler`; `envs.py:104-108` carries the comment "It's enabled by default. Please set to False if you hit any issue … once it's stable enough" |
| v0.12.0rc1 and later | same name | **removed**, feature **on by default** — so v0.11.x is the last release where the `=0` ablation exists | `../performance-throughput/upgrade-vs-older-baseline.md:60` (`release_notes.md:110`) |

So at a v0.11.0 pin `echo $VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` printing nothing does **not**
prove the upstream path is live: two similar spellings exist, and the live one is the
`_TOPK_TOPP_OPTIMIZATION` form, which is **on when unset**. Re-grep `envs.py` at the pin
before quoting any of it (SKILL.md rule 11) — a grep for `TOPK_OPTIMIZE` alone misses it,
and so does a quote of the introduction release's default. The older spelling has no
`vllm_ascend/` reader at all (`VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`'s only hit is a nightly
test's env block, `../performance-throughput/profiler-opcount-evidence.md:108-115`), so
exporting it and seeing no change proves nothing on any release.

## Rule 1 — what the swap routes (vllm-ascend v0.11.0)

`AscendSampler.__init__` installs `AscendTopKTopPSampler` as `topk_topp_sampler`
(`sample/sampler.py:11-19`). Its `_apply_top_k_top_p` (`:21-58`) returns the logits
untouched when both `k` and `p` are `None` (`:27-28`), then calls the fused CANN op
`torch_npu.npu_top_k_top_p(logits, p, k)` — note the `(logits, p, k)` order — when
`is_310p()` is false **and** `k is None or 1 <= int(k.max()) <= 1024` (`:30-34`);
otherwise it masks in place with the portable softmax/sort path (`:36-58`). Two
consequences that change the investigation:

- **A `top_p`-only batch also reaches the fused op** (`k is None` satisfies the
  condition). That is the opposite of the v0.9.1 code, where `TopKTopPSampler`'s patch
  routed to `npu_top_k_top_p` only when **both** `k` and `p` were given and a `top_k`-less
  request took the in-place sort branch (`topk-logprob-negative-inf.md`) — do not carry
  either statement across pins. "My body has no `top_k`, so the fused path cannot be mine"
  is wrong at v0.11.0.
- `int(k.max())` is a **D2H copy on the sample path** — a host sync every decoding step
  — and the recorded crash on exactly this condition is `sampler.py:28` +
  `AclrtSynchronizeStreamWithTimeout(copy_stream)` at v0.11.0rc2 (#4771; the condition
  sits at `:32` in the v0.11.0 file). It is also evaluated on the **whole batch's** `k`
  tensor, which is Rule 2.

## Rule 2 — the lane is chosen per *batch*, so the same request can run two different samplers

Upstream stores `top_k` per request: `0 < top_k < vocab_size` registers the request,
anything else is written as `vocab_size` (`vllm/v1/worker/gpu_input_batch.py:367-372` at
vllm v0.11.0 `b8b302cde434`), and `k` is passed as `None` only when **no** request in
the batch set it (`no_top_k`, `:758`, `:844-845`). Therefore, at a v0.11.0 pin:

- every row carrying `top_k=20` ⇒ `k.max() == 20` ⇒ the fused op for the batch;
- **one** row without `top_k` (any client, any prompt) ⇒ `k.max() == vocab_size`
  (> 1024) ⇒ the entire batch falls to the portable sort path;
- all rows without `top_k` ⇒ `k is None` ⇒ the fused op, with `k=None`.

A concurrency-correlated sampling difference is therefore compatible with the sampler
alone, with no kernel defect anywhere: what changed is which implementation ran. Keep it
labelled a hypothesis about your report until measured, but it is why the ablation below
must hold batch composition fixed, and why "I set the flag to 0 and nothing changed" is
not yet evidence about the fused op — with a mixed batch, the failing lane may never have
been the fused one.

## Rule 3 — rung order for a low-rate sampling-shape report

1. **The observation confound first.** 0.1% is ~1 in 1000; high concurrency is
   thousands of requests. Same body, concurrency 1 vs the reported concurrency,
   N ≈ 1000-2000 identical prompts each, same system prompt, count the rate. Equal
   rates ⇒ "under high concurrency" was sample size and the model's own tail is the
   subject, not a defect.
2. **`VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=0`**, one restart, same body, same N.
   Single variable, no code change. Unchanged rate ⇒ the fused op is not the owner
   (check Rule 2's batch condition before saying so).
3. **Greedy control** (`temperature: 0`) on the same body: still wrong ⇒ numerics
   upstream of the mask; clean ⇒ the sampling layer, which is this page.
4. **`--no-enable-prefix-caching`** plus a fixed `--max-num-batched-tokens` — only when
   a bad sample contains *another request's* text; that is KV cross-talk, a different
   owner (`short-prompt-corruption.md`,
   `../scheduler-admission/kv-block-lifetime-async-scheduling.md`).
5. GPU/vLLM control at identical params and N — the only row that attributes the rate
   to Ascend rather than to the checkpoint; a GPU run at the same rate closes this whole
   topic.

Cheapest evidence to attach to one bad sample: `logprobs: 20` with
`return_token_ids: true`, then read the id at the point the text leaves the intended
language. An id **outside** the declared `top_k=20` is a mask defect and worth filing;
an id inside it is the model's tail. (The top-k *ranks* printed by vLLM are list
positions except for the sampled token, whose entry carries the true full-vocab rank —
`topk-logprob-negative-inf.md`.)

## What this page does not establish

That the fused op is numerically wrong. No accuracy test for
`aclnnApplyTopKTopP`/`npu_top_k_top_p` is in the export (`cann/ops-*` is not part of the
snapshot), so the ablation decides the lane and the CANN side decides the cause. The
sample-path syncing in Rule 1 is a *performance* fact, not a numeric one.

## Recheck conditions

The flag's name, default and gate site (`envs.py`, `worker/model_runner_v1.py`), the
`is_310p()` / `1 <= int(k.max()) <= 1024` condition, and upstream's `top_k` filling and
`no_top_k` handling all move between releases; re-grep them at the pinned commit before
quoting. On v0.12.0rc1 and later the flag is gone and the fused path is unconditionally
on: say the ablation no longer exists there (name the version) instead of offering it, and
do not read "we upgraded and it got better" as proof that the fused op was the cause.
