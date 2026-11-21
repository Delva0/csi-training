# The whole answer is one repeated token (`!!!!…`): id-level grading, then persistent vs per-request

Scope: a **text** request whose entire answer is one character/token repeated
(`!!!!!!!!`, `，，，`, `。。。。`, a single word), with no traceback and a healthy
process. Two onsets, and they are different lanes:

- **always** — the very first request degenerates, repeatably (this is usually a
  config × capability gap, evidence below);
- **after hours of healthy serving**, and then *every* request including brand-new
  conversations degrades until the engine is restarted (recorded #3547: GLM-4.5-Air,
  v0.11.0rc0 `00ba0710` + vllm v0.11.0rc3, 8×910B3, TP8, `--max-model-len 131072`,
  `--max-num-seqs 24`, `--tool-call-parser glm45 --reasoning-parser glm45`,
  `--additional-config '{"ascend_scheduler_config":{"enabled":true,
  "enable_chunked_prefill":true}}'`, litellm gateway + Claude Code client, no log at
  all).

Four sibling pages own their own slices; take the split from the paste, not from the
words 乱码/感叹号: an **image or video** in the request →
`vl-vision-tower-degeneracy.md` (this is the text-only twin); only the **first**
`LLM.generate()` / first serve request →
`first-generate-degenerate.md`; `temperature>0` carries the corruption while greedy
is byte-identical → `sampler-side-stream-corruption.md`; sentences/phrases repeat
inside otherwise fluent text → `long-output-repetition.md`; word salad rather than
one token → `short-prompt-corruption.md` / `garbled-long-context-output.md`.

## Step 1 — grade it at the id level, not as text

A repeated single character means the engine emitted **one token id** at every step:
greedy argmax over an all-equal or NaN logits vector (id 0 wins the tie, hence `!`
in tokenizers where id 0 is `!`), or a cached/constant token fed back. Two readings
follow from that, and both retire whole lanes cheaply:

- `usage.completion_tokens` versus the number of characters in the answer. 1:1 ⇒
  the sampler really produced N ids; far fewer ids than characters ⇒ the text is
  being repeated downstream (detokenizer, SSE assembly, gateway, client) and the
  engine may be healthy.
- one request with `"temperature": 0, "logprobs": true, "top_logprobs": 5,
  "max_tokens": 16` on a trivial unrelated prompt (and a second with `"hi"`). All
  top-5 entries equal, or `-inf`, or `finish_reason: "length"` after 16 identical
  ids ⇒ degenerate forward. A normal distribution with `!` in the text sends you to
  the text layer above instead — that single response decides the page.

`'!' == token id 0` is **recalled, not verified** for the GLM and Qwen tokenizers
(no GLM vocabulary exists in this snapshot; the Qwen note reads the same way in
`vl-vision-tower-degeneracy.md`). The inference that survives verification is
"constant id ⇒ degenerate forward"; give the reporter
`AutoTokenizer.from_pretrained(<ckpt>).convert_tokens_to_ids('!')` and label it a
hypothesis, not a cause.

## Step 2 — persistent state vs per-request, from the reporter's own words

- **"new conversations return `!` too, restart fixes it"** (recorded #3547) is a
  control of the same value as a working sibling path: a mechanism that also
  predicts a *fresh* request on the broken process would answer normally is out.
  Prompt content, sampling params, chat template, reasoning/tool parsers and the
  gateway are retired in one sentence; what remains is persistent process/device
  state — the KV pool and its prefix-cache blocks, piecewise/ACL-graph output
  buffers, or a device fault that left state dirty without an error.
- **only the long-running conversation degrades** ⇒ one sequence's KV/state, not
  the pool: re-send the same short prompt in a new request before any ablation.
- Ask for the onset in the same breath: wall-clock time, requests served, and the
  log window around the first bad answer (`Preempt`, `EngineCore`, `EZ9999`,
  `507xxx`, `Prefix cache hit rate`). And say plainly *capture while it is still
  broken* — this is the one report shape where the reporter's own recovery step
  (restart) destroys the evidence.

## Step 3 — an always-degenerate serve is a capability/config gap first

Recorded precedent family, and the reason to look at the *flags* before the model:
`#1754` — adding `--pipeline-parallel-size` to an ACLGraph serve made **all four**
prompts return `Output: '!!!!!!!!!!!!!!!!'` at vllm `0.9.2rc2.dev` + vllm-ascend
`0.9.1rc2.dev`; the owning statement is the release's own known issues line
("Pipeline parallel does not work with ray and graph mode: #1751 #1754",
`release_notes.md:359` at `048bfd5`), not any kernel. `#374` —
DeepSeek-R1-Distill-Llama-70B, one fixed query, infinite `!!!!!`, "100%
reproducible", fine on upstream vLLM: engine-side, no model defect. The V0-era
`#321` / `#445` / `#562` are the same shape at older pins. So for the *always* row:
name the mode/flag that is unsupported for that model (`../pipeline-parallel/index.md`,
`../feature-enable-recipes/index.md`, the release's known-issue list) and give one
ablation that removes it, before any numerics theory. A **quantized** serve on a
triton-ascend-backed family (Qwen3-Next and its GDN siblings) is the config row this
step means: that `!` run is the wheel build's (`qwen3-next-gdn-numeric-path.md`
§*triton-ascend build*, #5606), and its build check precedes the bf16 control.

## Step 4 — the onset-after-hours ladder (recorded #3547)

One launch per rung, same load, same request, recording minutes-to-onset; the
cheapest *evidence* rungs come before the cheapest *changes*:

1. the Step 1 body on the still-broken engine (logprobs, `completion_tokens`), plus
   a direct `curl` to the engine port bypassing litellm;
2. `--enforce-eager` — removes the piecewise/ACL-graph lane; prove it ran with
   `Compilation disabled, using eager mode by default` (`../startup-failures/
   compile-backend-banner-then-silence.md`), otherwise the row tested nothing;
3. `--no-enable-prefix-caching` — the pool/block-reuse lane;
4. the reporter's own non-default `--additional-config` removed, `--max-model-len`
   at the tutorial value — i.e. back to the configuration the tree validates for
   that model;
5. `ASCEND_LAUNCH_BLOCKING=1` + plog around the onset — the device-fault lane;
6. lower `--max-num-seqs` — preemption/recompute at high concurrency.

Read the reporter's config before ranking any of these. At `00ba0710` the pair
`ascend_scheduler_config = {enabled: true, enable_chunked_prefill: true}` means the
fork **delegates** scheduling to upstream — `AscendScheduler.schedule()` returns
`super().schedule()` as soon as `scheduler_config.chunked_prefill_enabled`
(`vllm_ascend/core/scheduler.py:65-66`), `AscendSchedulerConfig.__post_init__` sets
`chunked_prefill_enabled = self.enable_chunked_prefill`, and the platform block
skips raising `max_num_batched_tokens` to `max_model_len` for that case
(`vllm_ascend/platform.py:159-164`). So the AscendScheduler's own chunked-prefill
path is **not** in play on that serve, and "the fork's scheduling is buggy" is not
available as an answer there; also, a non-MLA model gets `enabled = True` forced
anyway (`platform.py:140-150`), so the reporter's `"enabled": true` says nothing.

## Answer shape and what stays open

Verdict first: "the engine is emitting a constant token id; that is a degenerate
forward, and the fact that it survives new conversations but not a restart makes it
persistent process/device state — not the model, not your sampling". Then the two
readings, the ladder, and the missing artefacts: one `logprobs` body captured while
broken, `usage.completion_tokens` vs the id count, the onset log window, `npu-smi
info`, the `torch_npu` line of `collect_env`, and whether a *short unrelated* prompt
also returns `!`. For #3547 the onset owner stayed unresolved from the paste alone
(no log, no request body, version `0.11.0` final not in this snapshot) — say so, and
do not dress a ranked hypothesis as a found cause.
