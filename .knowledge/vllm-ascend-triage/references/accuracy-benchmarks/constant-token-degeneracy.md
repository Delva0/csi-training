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
one token → `short-prompt-corruption.md` / `garbled-long-context-output.md`; a MoE
serve whose command carries `--enable-expert-parallel` (DP>1 especially) →
`ep-mode-reduce-double-count.md`, whose EP-on-vs-off rerun is rung 1 **and** whose Step 0
grep decides whether the pin still has a model-side second reduce at all — quote no
`tp_size`-overshoot mechanism before that grep.

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

A constant id retires *which* checkpoint is served only in the sense of the intended
weights — never the **integrity of the local copy**: a damaged, truncated or
mixed-revision model directory loads and serves and still answers `!` (recorded
#6136, Step 3b below).

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
  One thing that control does **not** retire: a *new conversation* is not a *new
  prefix*. #3547's client is Claude Code behind litellm, so every conversation opens
  with the same long system prompt and — with prefix caching on, which V1 defaults to
  (Step 4a) — reuses the same cached blocks; a corrupted block is then hit by
  "brand-new" conversations too. The control that separates pool state from
  per-request state is a prompt whose prefix no earlier request had (different or
  absent system prompt, or a random nonce on the first line): answers correctly ⇒
  pooled/prefix-block reuse stays in the lane; `!` there too ⇒ cached blocks are not
  the carrier and process state (graph buffers, a swallowed device fault) resumes.
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
`#321` / `#445` / `#562` are the same shape at older pins — and #562's
**reporter-confirmed** cause was the numeric lane, not a flag: 「将精度改为 bfloat16
就可以了」 at v0.7.3rc2/910B4 (`precision-dtype-degeneracy.md`). So for the *always*
row, name the mode/flag that is unsupported for that model
(`../pipeline-parallel/index.md`, `../feature-enable-recipes/index.md`, the release's
known-issue list) and give one ablation that removes it, before any numerics theory —
but when no flag in the command is unsupported, clear Step 3b's copy check first (one
command, no restart), then the dtype row is the next restart and the cheapest one there
is: `--dtype bfloat16` plus the log's own `dtype=` line
(`precision-dtype-degeneracy.md`). A **quantized** serve on a
triton-ascend-backed family (Qwen3-Next and its GDN siblings) is the config row this
step means: that `!` run is the wheel build's (`qwen3-next-gdn-numeric-path.md`
§*triton-ascend build*, #5606), and its build check precedes the bf16 control.

## Step 3b — the local copy of the checkpoint, not its identity (recorded #6136)

The *always* lane above assumes the weights on disk are the ones `config.json`
describes. A **damaged or mixed-revision copy of the right checkpoint** is a different
variable, and it survives every control the other rows use: the load reports success,
`dtype=` is unchanged, the flag set is the tutorial's, the card is healthy, and the
forward degenerates to one id. Recorded #6136 — Qwen3-Next-80B-A3B-Thinking on
`IMAGE=quay.io/ascend/vllm-ascend:v0.13.0rc1`, CANN 8.2, `triton-ascend==3.2.0`, TP4,
`--max-model-len 32768 --gpu-memory-utilization 0.8 --max-num-batched-tokens 4096
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`: an 11-token prompt
(`你是谁?`) answered with a sensible Chinese opening sentence and then `!` to
`completion_tokens: 512` / `finish_reason: length`. The reporter then added that the
same model answered normally on a **910B3** box and not on the **910B4-1** one, and
closed on 2026-02-26 with 「可能是模型文件有点毛病，重新下载了一遍现在可以了」 — a
re-download, not a wheel, dtype, graph-mode or card change. Three consequences:

- **A clean peer host with the same image is a control on everything except the
  files.** "Works on B3, fails on B4-1" invites a SoC/card reading; it fits two hosts
  holding different copies just as well. Before ranking hardware, ask which directory
  each serve loaded (`Starting to load model <path>...`) and whether the working host
  used *that same* directory or its own download.
- **The copy check costs one command and no restart**, so it precedes the dtype/wheel
  restarts: `uv run --no-project python <skill_root>/scripts/checkpoint-header-scan.py
  <model dir>` for the per-shard inventory (index-vs-disk names, container validity,
  declared data region vs file size), then each shard's byte size against the hub
  entry or `sha256sum` against a second copy, and the load log's
  `Loading model weights took` / `missing keys` / `unexpected keys` lines — a
  mixed-revision directory shows up in the key lists. The damage shapes are
  `../startup-failures/checkpoint-file-not-safetensors.md` §2-3, which is the same
  file damage at the point where the loader still refuses (`HeaderTooLarge`); there the
  lane closes at the file instead of at the logits.
- **A verified re-pull is the discriminator, so say it in the answer.** If the same
  directory still answers `!` after a re-download whose sizes/hashes match, the copy
  row is out and the flag/dtype/wheel rows above resume. Otherwise the reporter's own
  retry answers the issue — better as your sentence than as their later discovery.

Calibration from the recording: that answer ranked the pin's `triton-ascend` build
(`3.2.0` against the pinned `dev2025110717`) and `FULL_DECODE_ONLY` first and never
asked for a file comparison. The wheel delta is a legitimate row — it is one restart
against a zero-restart check, and it was not the cause here.

## Step 4a (pre-read, before the ladder) — resolve the *effective* config, then
read the pin's Known-issue rows

Two one-command reads that turn a rung from guess into lead; do them before ranking.
Read them here and the ladder below stays a ladder; skip them and the ladder's order
is taste.

**The reporter's command line is not the effective config.** In V1, prefix caching is
**on by default** for a non-pooling model: `_set_default_args` carries the comment
"V1 always uses chunked prefills and prefix caching for non-pooling tasks"
(`vllm/engine/arg_utils.py:1544-1547`) and sets `enable_chunked_prefill = True` then
`if self.enable_prefix_caching is None: self.enable_prefix_caching = True`
(`:1560-1561` at vllm v0.11.0rc3 `8ce5d31`). #3547 never passed
`--enable-prefix-caching` and ran with it on. Resolve it from evidence rather than
from the command: at that pin every V1 serve logs
`Engine 000: … Prefix cache hit rate: X%` (`vllm/v1/metrics/loggers.py:133`), so a
non-zero rate is the proof the cache is on **and** reusing blocks; later pins also dump
the whole `VllmConfig` at startup (`vllm/config/vllm.py`), this one does not.

**The pin's release notes may already warn about that combination.** At its pin
(vllm-ascend `00ba0710`) `docs/source/user_guide/release_notes.md:66`, inside the
`### Known issue` list at `:62` of the `## v0.10.2rc1` section (`:27`):

> Please do not enable chunked prefill with prefix cache when running with Ascend
> scheduler. The performance and accuracy is not good/correct. [#2943]

The `## v0.11.0rc0` section (`:3-24`) has no Known-issue list, so that bullet is the
tree's last statement on the subject — neither restated for rc0 nor retracted. All
three of its factors hold on #3547's serve and **none of them is visible in the
command**: chunked prefill from `--additional-config '{"ascend_scheduler_config":
{"enabled":true,"enable_chunked_prefill":true}}'`, the Ascend scheduler forced anyway
for a non-MLA model (`platform.py:140-150`, so the pasted `"enabled": true` says
nothing), prefix caching by default. That promotes the prefix-cache rung from a guess
to a lead carrying the project's own sentence — and no further: the bullet assigns the
*combination*, not a component, and `schedule()` delegating at this pin (below) leaves
the fork's `update_from_output` (`core/scheduler.py:567`) and `finish_requests` still
in the path, so the mechanism stays open.

**The onset is the discriminator and it is cheap**: this lane predicts degradation
only after prefix hits exist and a block has been reused, so "hours of healthy serving,
then every request" with a rising `Prefix cache hit rate` before the first bad answer
is consistent, while a serve that degenerates on its *first* request belongs to Step 3.
Command: `rg -n -i 'known issue' -A8 <pin>/docs/source/user_guide/release_notes.md`.

**The row dies with the fork.** The bullet names the *Ascend scheduler*, and at
vllm-ascend v0.14.0rc1 (`52d4acf`, the #6525 pin) there is no such class and no
`ascend_scheduler_config` key left (`vllm_ascend/core/` is `recompute_scheduler.py` +
`scheduler_dynamic_batch.py`; the key is gone from
`docs/source/user_guide/configuration/additional_config.md`) — a `--additional-config`
carrying it is read by nothing. On a ≥0.13-era report lead with
`--no-enable-prefix-caching` as a plain prefix-cache ablation and say the project's
sentence no longer applies to their tree, instead of quoting it as their pin's own
warning (`../upstream-downstream-drift/scheduler-fork-missing-status.md`
§*Pin boundary*).

## Step 4 — the onset-after-hours ladder (recorded #3547)

One launch per rung, same load, same request, recording minutes-to-onset (a rung whose
onset was not timed and not compared to the baseline tells you nothing); the cheapest
*evidence* rungs come before the cheapest *changes*:

1. the Step 1 body on the still-broken engine (logprobs, `completion_tokens`), plus
   a direct `curl` to the engine port bypassing litellm, plus a request whose prefix
   no earlier request had (Step 2);
2. `--no-enable-prefix-caching` — the pool/block-reuse lane, and the rung the pin's own
   Known-issue bullet names (Step 4a); keep `--additional-config` unchanged so one
   factor moves;
3. `enable_chunked_prefill: false` in the same `--additional-config` — the bullet's
   other factor, so the pair is separated rather than jointly cleared;
4. `--enforce-eager` — removes the piecewise/ACL-graph lane; prove it ran with
   `Compilation disabled, using eager mode by default` (`../startup-failures/
   compile-backend-banner-then-silence.md`), otherwise the row tested nothing;
5. the reporter's own non-default `--additional-config` removed, `--max-model-len`
   at the tutorial value — i.e. back to the configuration the tree validates for
   that model;
6. `ASCEND_LAUNCH_BLOCKING=1` + plog around the onset — the device-fault lane;
7. lower `--max-num-seqs` — preemption/recompute at high concurrency.

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
do not dress a ranked hypothesis as a found cause. The one rung that may lead is the
pin's own Known-issue sentence matching the *resolved* config (Step 4a); quote both
anchors (the bullet and the V1 default) and label it the project's documented warning,
not a measured cause — the paste records no onset, no `Prefix cache hit rate` and no
request body, and those three decide it.
