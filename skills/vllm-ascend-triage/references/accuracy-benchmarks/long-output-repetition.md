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
- **Decode topology flipped** (tp4 / dcp4 / dp8, where the doc's decoder nodes are
  tp8 / dcp8 / dp4 — the P side matches). That split does exist on the same page for
  the 8-node A2 layout (`:984`), whose text says both sides enable prefix caching
  and MTP with `num_speculative_tokens=3`, so neither topology is unsupported — but
  the tested P/D pairs differ, and MTP 5 with this decode shape is untested.

## Three traps when reading that diff

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
   still held them" into evidence.
3. **Check the spec-decode branch at the reporter's ref before naming it.** At
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

## The deliverable

Verdict plus one action in the first paragraph, then a ranked list where every
cause carries its own one-line check, then this matrix (each row one restart, same
failing request, `temperature=0`):

| Row | Change | Still repeats means | Clean means |
|---|---|---|---|
| identity | `--max-num-seqs 1`, same request 3× | model-side long-output looping | engine reuse/draft path owns it |
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
