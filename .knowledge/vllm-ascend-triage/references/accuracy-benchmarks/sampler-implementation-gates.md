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

It also owns one **crash** shape, which is not a numerics question: a `RuntimeError … the
current working operator name is aclnn<Op>` raised at that same `int(...)`/`int(k.max())`
condition, where the printed op name belongs to a *different* statement (Rule 1a).

## Rule 0 — the gate's name and default are per-release, and an `echo` of the wrong name answers nothing

| Release | Flag | Default there | What it replaces |
|---|---|---|---|
| v0.9.1 | `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | unset ⇒ native | only the patched `TopKTopPSampler.forward_native` (`patch/worker/patch_common/patch_sampler.py:104-106`) — `topk-logprob-negative-inf.md` |
| v0.9.2rc1 (introduced) | `VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION` | **`0`** — "we'll consider to enable it by default in the future" (`docs/source/user_guide/release_notes.md:374`, PR #1732) | the sampler swap below |
| v0.11.0 (`2f1aed98ccdb`) | same name | **`'1'`** | the **whole** sampler: the gate is `worker/model_runner_v1.py:269` (`if envs_ascend.VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION:`) and the block runs to `:277` — `:271` imports `AscendSampler`, `:273` binds it, `:275-276` the upstream `Sampler`; `envs.py:104-108` carries the comment "It's enabled by default. Please set to False if you hit any issue … once it's stable enough" |
| v0.12.0rc1 and later | same name | **removed**, feature **on by default** — so v0.11.x is the last release where the `=0` ablation exists | `../performance-throughput/upgrade-vs-older-baseline.md:60` (`release_notes.md:110`) |

So at a v0.11.0 pin `echo $VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` printing nothing does **not**
prove the upstream path is live: two similar spellings exist, and the live one is the
`_TOPK_TOPP_OPTIMIZATION` form, which is **on when unset**. Re-grep `envs.py` at the pin
before quoting any of it (SKILL.md rule 11) — a grep for `TOPK_OPTIMIZE` alone misses it,
and so does a quote of the introduction release's default. The older spelling has no
`vllm_ascend/` reader at all (`VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`'s only hit is a nightly
test's env block, `../performance-throughput/profiler-opcount-evidence.md:108-115`), so
exporting it and seeing no change proves nothing on any release.

## Rule 0b — a second gate sits upstream of the sampler: which model runner (v1 or v2)

At vllm-ascend v0.14.0rc1 (`52d4acf`) ↔ vllm v0.14.1 (`d7de043`) the sampling code is
chosen by an **upstream** env var, not by a `vllm_ascend` one, and it moves more than
the penalty kernel:

| `VLLM_USE_V2_MODEL_RUNNER` | sampler | penalty code that runs | attention backend |
|---|---|---|---|
| unset / `0` — **the default**, `vllm/envs.py:1601-1602` | `AscendSampler(Sampler)` (`vllm_ascend/sample/sampler.py:37`) overrides only the top-k/top-p sampler (`:42`) | **upstream** `Sampler.apply_penalties` (`vllm/v1/sample/sampler.py:299`, `def` at `:303`), per-request tensors from `vllm_ascend/worker/npu_input_batch.py:145-175` | `CUSTOM` / `ASCEND_MLA` / `ASCEND_SFA` (`vllm_ascend/attention/attention_v1.py:74`, `mla_v1.py:70`, `sfa_v1.py:67`) |
| `1` | `vllm_ascend/worker/v2/sample/sampler.py` `AscendSampler` | **downstream** Triton `apply_penalties_and_temperature` (`worker/v2/sample/penalties.py`, adapted from upstream `gpu/sample/penalties.py`; new in this rc — release notes v0.14.0rc1, *Others*: "model runner v2 support triton of penalty", #5854) | `FLASH_ATTN` (the same three `:74`/`:70`/`:67` branches) |

Two answers follow, and both are cheaper than a sampler theory.

- **"The penalty parameter had no effect" is not a downstream plumbing defect on the
  default path.** The penalty is upstream's `apply_penalties`; what makes a 1.1-1.3
  multiplier look inert is a logits distribution that has already collapsed, so the
  ablation that matters is the one that shows the distribution (`logprobs`), not a
  penalty bug hunt. The one path where penalty code *is* this project's is
  `VLLM_USE_V2_MODEL_RUNNER=1` — a kernel introduced in the reported rc, i.e. one
  release old. Ask for the launch environment (`env | grep VLLM_USE_V2_MODEL_RUNNER`,
  or the `vllm serve` block) before ranking it, and make the ablation "unset it,
  same request".
- **The same reader swaps the whole attention implementation**, which is a first-order
  discriminator on a numeric or accuracy report: an FP32 logits copy
  (`worker/v2/sample/sampler.py`) plus `FLASH_ATTN` instead of the Ascend `CUSTOM` /
  `ASCEND_MLA` / `ASCEND_SFA` backend is a different forward path, so a report that
  happens to export this env var is not comparing one stack. Say which side they are
  on before diffing kernels.

Re-grep `VLLM_USE_V2_MODEL_RUNNER` at the pin: the default lives in **upstream**
`vllm/envs.py`, so it moves with vllm rather than with vllm-ascend, and a later release
may flip it or drop the v1 runner.

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

## Rule 1a — the op name printed at that sync line is another statement's, so map it before ranking a mechanism

Recorded #4286 (vllm-ascend 0.11.0; 0.9.1 fine on the same script): the paste ends at
`vllm_ascend/sample/sampler.py:32  if not is_310p() and p is not None and k is not None and 1 <= int(`
with

```text
RuntimeError: The Inner error is reported as above. The process exits for this inner error,
and the current working operator name is aclnnCat.
Since the operator is called asynchronously, the stacktrace may be inaccurate. …
[ERROR] … ERR00100 PTA call acl api failed.
```

`aclnnCat` is **not** launched by `_apply_top_k_top_p` — that statement launches
`npu_top_k_top_p`/`aclnnApplyTopKTopP` — so the `int(...)` readback is reporting whatever the
stream already did. The question is *which statement on this path launches `aclnnCat`*, and at
the vllm v0.9.1 pin (`b6553be1bc75`) only two v1 sites a decode step on Ascend can reach (the
other `torch.cat` hits under `vllm/v1/attention/backends/mla/` are the Triton/ROCm reference
backends, not the Ascend path):

| site | gate that must hold for it to run |
|---|---|
| `vllm/v1/sample/sampler.py:177-178`, both `torch.cat(...)` in `Sampler.gather_logprobs` | `num_logprobs is not None` (`:61-62`) — the request asked for `logprobs` / `top_logprobs` / `prompt_logprobs` |
| `vllm/v1/worker/gpu_model_runner.py:1449`/`:1477`, `target_hidden_states = torch.cat([h[token_indices] for h in aux_hidden_states], dim=-1)` | `use_aux_hidden_state_outputs`, set only for `--speculative-config '{"method": "eagle3"}'` (`:158-163` at that pin) — not eagle/eagle2, not a plain MTP head |

Both are decided by the reporter's own request body and `--speculative-config`, not by a
kernel theory, and the grep is one call at *their* vllm ref:
`rg -n 'torch\.cat|gather_logprobs|num_logprobs' vllm/v1/sample/sampler.py` and
`rg -n 'torch\.cat' vllm/v1/worker/gpu_model_runner.py` — v0.11.0 renumbers them (the #4286
paste has `forward` at `:100`, `sample` at `:180`). Keep both readings open: a gate that holds
names the owner, and neither gate holding means a stale thread-latched code from an earlier
phase (`../hccl-and-hardware/latched-runtime-error-masking.md`), settled by
`ASCEND_LAUNCH_BLOCKING=1` plus the plog's `fault kernel_name=`/`retCode`. Also read the
missing line: `ERR00100 … PTA call acl api failed` with **no** `error code is N` means the
inner block above `The Inner error is reported as above` was cut out of the paste, and that
block is what separates an aicore exception from a parameter refusal — request it first.

**Two ranks naming different operators needs no mapping at all.** Recorded #3468
(vllm-ascend v0.11.0rc0 `00ba071` + vllm v0.11.0rc3, Atlas 800I A2, Qwen3-VL-30B-A3B,
`--tensor-parallel-size 2 --enable-expert-parallel`, one **video** request) ends at the
same `sampler.py:32` frame with `… current working operator name is aclnnSwiGlu` on
`Worker_TP1_EP1` and `aclnnMatmul` on `Worker_TP0_EP0`, one second apart in the same
step: no single statement launches both, so the pair is stream residue by construction
and neither names the statement that failed — locating `aclnnSwiGlu` in the MoE code
(or the video path) proves only that such a call site exists, which it always does.
That paste also dates the fault from its own dump — the scheduler record beside the
traceback is a `CachedRequestData` with `num_computed_tokens=[2048]` and
`num_scheduled_tokens=2048`, i.e. a continuation chunk of the request's first prefill,
so the fault belongs to ops an earlier chunk submitted, not to the sampling step — and
the two deliverables stay `ASCEND_LAUNCH_BLOCKING=1` on that same request plus the
missing first-error block. The recorded run instead spent its remaining tool turns
diffing the Ascend Qwen3-VL vision tower against upstream for a video-path shape
mismatch, which cannot outrank an error whose owning operator is still unknown.

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
