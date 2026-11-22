# Long-output repetition / decode degeneration (a few responses in a long run)

Verified at vllm-ascend `v0.23.0rc1` (`f4a08bdd`) + vllm `v0.23.0` (`0fc695fc`),
recorded for #13719: GLM-5.2 W8A8C8, 910C, 4-node PD disaggregation
(`MooncakeConnectorV1`), 64 concurrent MMLU-Pro requests, `max_tokens` 32000;
after ~2 h, **5 responses contained repeated sentences** — no crash, no OOM, no
error text, both P and D `vllm serve` commands pasted.

## This is not the prompt-length corruption pages

`garbled-long-context-output.md` and `short-prompt-corruption.md` are split by
`usage.prompt_tokens`. Neither fits a report where the *prompt* is a few hundred
tokens of benchmark question and the *completion* runs to tens of thousands of
tokens. Grade the symptom with both numbers in hand:

| Observed | Class | First owner to suspect |
|---|---|---|
| a sentence/phrase repeats, text stays fluent and on-topic | decode-time reuse or draft bookkeeping | prefix cache, preemption/recompute, MTP draft |
| 乱码 / token soup / dropped phrases | corruption | use the two prompt-length pages |
| repeat starts at a fixed generated length | length/block arithmetic | block table, preemption, KV transfer of the long sequence |
| repeat only at concurrency > 1 | scheduler/reuse interference | preemption, async scheduling, chunked prefill |
| same request repeated 3× at `--max-num-seqs 1` still loops | model/sampling, not the engine | long-CoT degeneration at 32K output; stop hunting engine causes |

That last row is the cheapest identity check and belongs in the first answer: a
long-CoT model can loop on its own, and no engine ablation outranks it until the
serial rerun is clean.

**When the launch carries `--async-scheduling` + MTP on a hybrid GDN/Mamba family, the
"concurrency > 1" row has a named owner at v0.23** and it is not sampling: the previous
iteration's accepted-token counts are gathered out of mutable `InputBatch` rows that
`add_request()`/`condense()` already rewrote, so a wrong count becomes a wrong Mamba/GDN
state-copy offset and the request repeats or omits a fixed token run. Two greps at the pin
decide it, the run's own `--async-scheduling`-off arm is the classification, and
`temperature=0` changes nothing — `../spec-decode-mtp/accepted-token-row-snapshot.md`
(recorded #13863, where the rate was 2/512 at 100 % KV: size the rerun with the N ≈ 3/p
arithmetic below before judging any ablation).

## A reported rate is the sample size, not an anecdote

Recorded #14085 (vLLM-Ascend v0.23.0rc1, GLM-5.2-w4a8c8, 8×910B PD disaggregation,
C8 sparse flags + DCP=8, called from `opencode`) said only "偶现乱码，或重复说话";
the reporter's later comment supplied the two numbers the report lacked: **one fixed
prompt** (a domain-expert question a few hundred tokens long) and **29 repeats in
193 runs (~15 %)**. Both follow-ups are worth having before the ladder runs:

- **Three reruns cannot test a 15 % effect.** At p ≈ 0.15, three clean runs happen
  with probability 0.85³ ≈ 0.61, so the identity row's "same request 3×" only
  certifies a *deterministic* loop. Size the rerun as **N ≈ 3/p** (≈20 for 15 %) and
  judge every later ablation row by the same count rather than by one clean response.
- **A clean majority of the same prompt retires the model-side row only.** 164/193
  clean runs of one prompt rules out "this prompt always degenerates", and leaves
  per-request engine state (reuse/preemption, draft bookkeeping, sparse-attention KV
  writes, graph replay) beside sampling equally open — sampler state moves with
  batching too. `temperature: 0` plus a fixed seed is what separates them, and it is
  the first launch, not the last.
- **Ask for the rate in the first answer, beside the samples**: how many runs, how
  many bad, one prompt or many, the prompt text, and `prompt_tokens`/`completion_tokens`
  of one bad response. A report that says only "偶现" is asking the reader to guess N,
  and the count is the cheapest thing the reporter can produce. The recorded run
  shipped a complete ranked answer (C8/DCP lane first, six-rung ladder) without asking,
  and the rate arrived 19 days later.

## First move: diff both commands against the model's tutorial

Same move and same justification as `garbled-long-context-output.md`, applied to
the decode side: `docs/source/tutorials/models/<model>.md` ships the validated P
and D commands per topology. Worked diff for #13719 against
`docs/source/tutorials/models/GLM5.2.md` (4-node 1M-context PD example:
"Recommended command for both prefiller nodes" `:1318`, its `vllm serve` `:1344`;
"…for both decoder nodes" `:1389`, its `vllm serve` `:1415`):

| item | doc P | reported P | doc D | reported D |
|---|---|---|---|---|
| `--max-num-seqs` | 8 | 16 | 32 | 16 |
| `--gpu-memory-utilization` | 0.75 | 0.88 | 0.93 | 0.93 |
| `--max-num-batched-tokens` | 16384 | 16384 | 128 | 96 |
| `num_speculative_tokens` | 1 | 1 | 3 | **5** |
| `--async-scheduling` | — | present | — | present |
| `--enable-prefix-caching` | — (default on) | explicit | — (default on) | explicit |
| `recompute_scheduler_enable` | true | **false** | true | true |
| topology (tp / dcp / dp) | 8 / 8 / 4 | 8 / 8 / 4 (same) | 8 / 8 / 4 | **4 / 4 / 8** |
| `--compilation-config '{"cudagraph_mode": …}'` | — | — | `FULL_DECODE_ONLY` (`:1435`, `:1308`, `:1255`) | same |

The rankable rows are the ones the tree validates differently somewhere else:

- **MTP 5 on the decode node.** Every PD example in this page uses
  `num_speculative_tokens` 1 (P) / 3 (D) (`:1366`, `:1437`); the only template that
  uses 5 is the co-located DP4·TP8 one, whose prose says so out loud:
  `:408` "Prefix caching is disabled (`--no-enable-prefix-caching`) in this
  configuration", flags at `:461` and `:521`. So "MTP 5 with prefix caching on" is
  a combination the tutorial never validates — a row, not a proven cause.
- **`--async-scheduling` on both nodes**, absent from every PD example here.
- **P node `recompute_scheduler_enable: false`** where the doc's P block sets true,
  plus `--enable-chunked-prefill` and gmu 0.88 vs 0.75 — capacity/preemption rows.
- **Decode topology is the documented 8-node A2 split, not a deviation** (tp4 / dcp4 /
  dp8 = the doc's Decode `DP8 TP4` at `:984`; the 4-node example's decoder nodes are
  tp8 / dcp8 / dp4, and the P side matches both). The reason it still matters is the
  arithmetic above: that layout's own text says both sides enable prefix caching and MTP
  with `num_speculative_tokens=3`, so **MTP 5 on this decode shape** is the untested pair,
  not the shape itself. Recorded #13719 repeats walked the topology row first; state it
  as a control, then spend the row on MTP 5.

## Four traps when reading that diff

1. **`--enable-prefix-caching` is not a deviation by itself.** At vllm v0.23.0 the
   default is on (`vllm/config/cache.py:92` `enable_prefix_caching: bool = True`),
   so a doc block that omits the flag runs with it on. Rank prefix caching from
   *which topology turns it off in the docs*, never from the flag's presence.
2. **`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` is a delayed-free timeout, not a
   generation bound.** `mooncake_connector.py:229-239` (same code in
   `mooncake_hybrid_connector.py`) drops entries from `delayed_free_requests` once
   they have been *finished* for longer than the value, logging
   `Force freed expired request: <id> … Resources have been forcibly released …`.
   The doc's own blocks set `480`, so a 480/3600 pair is not the deviation — the
   log line is: grep it, and only a hit turns "blocks freed while a reuse path
   still held them" into evidence. The same line is also the *prefiller-side* record
   of a PD request no decoder ever pulled — the producer-only send thread and the
   proxy/`--api-server-count` check are in
   `../pd-disaggregation/proxy-endpoint-vs-api-server-count.md`; read which side emitted
   it before choosing the lane.
3. **`cudagraph_mode` and the spec-config `enforce_eager` are the doc's own settings,
   so they are not rows.** `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`
   sits in the tutorial's own PD decode blocks (`:1435`, `:1308`, `:1255`, next to MTP 3) and
   every block writes `"enforce_eager": true` *inside* `--speculative-config`, so a
   reporter carrying both has matched the tutorial. The recorded #13719 repeat invented
   the graph-mode row, spent a grep on it and retired it — the graph lane is a lever only
   when the symptom is a capture/hang fault (`../hccl-and-hardware/
   aclgraph-full-decode-runtime-crash.md`), not when fluent text repeats.
4. **Check the spec-decode branch at the reporter's ref before naming it.** At
   v0.23.0rc1 `using_block_verify = max_spec_len >= 3 and
   rejection_sampler_config.enable_block_verify`
   (`vllm_ascend/sample/rejection_sampler.py:399-400`), and the config defaults to
   `False` (`vllm_ascend/ascend_config.py:766`). Without `rejection_sampler_config`
   in `--additional-config`, `num_speculative_tokens: 5` runs the strict
   per-position path — `mtp-block-verify-acceptance.md` carries the gate table.

Constraints that make ablation rows un-runnable must be stated, not discovered:
with sparse SFA models (GLM-5.x) DCP requires `dcp_size == tp_size`
(`vllm_ascend/platform.py:716-724`, `AssertionError` otherwise), so the meaningful
DCP row is off vs `dcp == tp`, not an arbitrary size.

## When the penalty itself is inert: `repetition_penalty` + `--async-scheduling`

A loop report that also says "`repetition_penalty=2` 没有效果" has one cheap
owner to check before any engine cause: output-token-based logits processors read
their history from `sampling_metadata.output_token_ids`, and on an async-scheduling
runner those are CPU-side placeholders until the runner copies the sampled ids
back. #7769's reporter diagnosed exactly that on a DeepSeek w8a8 serve:
`NPUModelRunner._sample` never called `update_async_output_token_ids()`, so every
id read `-1` and the penalty scored nothing — "it did nothing" was the runner,
not the sampler. At vllm-ascend **v0.19.1rc1** (`da421afa`) the call is present
(`vllm_ascend/worker/model_runner_v1.py:1991`, inside the `_sample` override, with
`:429-431` passing `logitsprocs_need_output_token_ids=bool(model_config.logits_processors)`),
matching upstream v0.19.1 (`vllm/v1/worker/gpu_model_runner.py:3328`, `:634`)
with the per-step CPU sync at `vllm/v1/worker/gpu_input_batch.py:850,970`.

So on a report from an image **older** than that, one grep on the reporter's tree
decides the lane — `rg -n 'update_async_output_token_ids'
$(python -c 'import vllm_ascend,os;print(os.path.dirname(vllm_ascend.__file__))')/worker/model_runner_v1.py`:
absent ⇒ the penalty is inert and the upgrade is rung 1; present ⇒ do not rank
"penalty ignored" at all, and the loop is model-side or a reuse/draft path. It
only applies when the failing request actually carries a penalty above its
default; a body without one makes the whole lane irrelevant.

**Two cheaper reasons the same wording is not a bug at a 0.13/0.14-era pin**, so read
them before ranking a sampler defect: on the default model runner the penalty is
upstream's `Sampler.apply_penalties`, and a multiplier of 1.1-1.3 over an already
collapsed distribution cannot repair it — the ablation that shows this is `logprobs`
on the looping step, not a penalty hunt; and the only path where the penalty kernel is
this project's own is `VLLM_USE_V2_MODEL_RUNNER=1` (Triton `apply_penalties_and_temperature`,
new in v0.14.0rc1), which also swaps the attention backend — one `env | grep` in the
reporter's launch decides it (`sampler-implementation-gates.md` §*Rule 0b*).

## A mid-answer word loop in a VL rollout is not the `!!!!` tower profile (recorded #3452)

The screenshot OCR of a Qwen2.5-VL answer (vllm-ascend v0.9.1 `0740d10` + vllm v0.9.1
`b6553be`, 910B, GRPO first-round rollout, `repetition_penalty` raised with no effect)
shows a correct, on-topic paragraph that names the state in the image and restates the
question, then `Court Court Court …` to the cap, with one `COur`. One observation splits
this from the vision-tower row: **the text before the loop is correct and image-derived**.
The `!!!!`/感叹号 profile (`vl-vision-tower-degeneracy.md`) is wrong from its first token,
and `constant-token-degeneracy.md`'s own routing ("image present ⇒ tower page") is the
wrong way round for a loop that starts after a good paragraph — an answer that first reads
the image correctly had an encoder that delivered, so dtype/tower/padding rows are not the
cheap owners here.

Rows on the same image+body, cheapest first:

- identity: `temperature: 0`, serial, **N ≈ 3/p** reruns (this page's identity row). A
  rollout at temperature ~1 with `n > 1` is the cheapest place to get p at all; ask for
  the count, not a retry.
- sampling: the rollout body (`temperature`/`top_p`/`top_k`, `max_tokens`, and the
  prompt's `<|image_pad|>` count) before any engine row — a word loop with casing drift
  is a sampling-side shape.
- sampler implementation: the gate is per release and **off** at old pins —
  `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` defaults to `'0'` at v0.9.1
  (`vllm_ascend/envs.py:55-56`), so the fused top-k path is not in it unless the harness
  set it (`sampler-implementation-gates.md`).
- the penalty-inert section above **cannot apply here**: vllm v0.9.1 has no
  async-scheduling flag at all (`grep -rn async_scheduling vllm/` finds only
  `disable_async_output_proc`, `vllm/engine/arg_utils.py:411`), so "the penalty did
  nothing" has no runner-bookkeeping reading at that pin — state that rather than ranking
  it.
- engine rows (reuse/preemption/MTP) only if the greedy rerun still loops; then the VL
  padding row (`USE_OPTIMIZED_MODEL=0`, live for Qwen2.5-VL at v0.9.2rc1+) needs a
  text-only control of the same question answering cleanly first, otherwise the tower is
  still not implicated.

The paste's environment block is a **confound sentence, not a rung**: `Failed to load
plugin ascend` / `module 'vllm_ascend' has no attribute 'register'` with
`No module named 'vllm_ascend._version'` and `vllm/_C.abi3.so: undefined symbol …` mean
that interpreter resolved a different `vllm_ascend` (`../image-dependency-faults/plugin-load-module-missing.md`),
while the loop was produced by the reporter's engine — so one command
(`python -c "import vllm_ascend; print(vllm_ascend.__file__)"`) and one paragraph, not a
version walk.

## The deliverable

Verdict plus one action in the first paragraph, then a ranked list where every
cause carries its own one-line check, then this matrix (each row one restart, same
failing request, `temperature=0`):

| Row | Change | Still repeats means | Clean means |
|---|---|---|---|
| identity | `--max-num-seqs 1`, same request **N ≈ 3/p** times at `temperature: 0` (3 only tests a deterministic loop) | model-side long-output looping | engine reuse/draft path owns it |
| MTP | D node back to `num_speculative_tokens 3` | not the n≥4 branch | the n=5 path |
| draft bookkeeping | D node `"enforce_eager": true` inside `--speculative-config` + `disable_padded_drafter_batch: true` | not the drafter batch | padded-drafter/graph replay |
| reuse | D node `--no-enable-prefix-caching` (and, if `--async-scheduling` is on, off) | prefix cache/async scheduling unrelated | reuse bookkeeping |
| capacity | `--max-num-seqs 8`, P `gpu-memory-utilization 0.75` | not preemption | preemption/recompute |
| longer | repeat until the repeat onsets, note the token index | — | length-tracking onset points at block/transfer arithmetic |

Ask for what the report omitted — and ask for it **inline**: #13719's later
attempt to attach the samples ("Failed to upload `repeated_responses.txt`") never
arrived, so a paste in the thread is worth more than a file request. Needed: the
headless nodes' full commands, the 5 responses themselves (`usage.prompt_tokens` /
`completion_tokens`, `finish_reason`, repeat position, and whether the repeat sits
in `reasoning` or `content` — with `--reasoning-parser` set the scoring
consequences differ), the D
node log lines `Preempt`, `Prefix cache hit rate`, `Force freed expired request`,
and how many requests the 2-hour run served (5 of what?).

## What was recorded, and the limits

#13719 shipped a complete ranked answer at its 10th assistant turn, without
truncation, and paid one useful version correction: the
`num_speculative_tokens >= 3` block-verify story from #7580 does not hold at
v0.23.0rc1, because the config gate defaults to `False`.

Labels to keep: the doc-vs-reporter deviations are source-verified (line numbers
above); the ownership of the repetition is **hypothesis** — the report carried no
logs, no response samples and no request count, so no ablation was ever run. The
report's own build was a daily "0.23.0" build, not the pinned `v0.23.0rc1`, so
every line number needs re-grepping on the reporter's tree.
