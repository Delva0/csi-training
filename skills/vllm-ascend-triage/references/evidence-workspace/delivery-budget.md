# Output budget: recorded accounting and the delivery gates

The completion budget is 16384 tokens for the whole session — reasoning,
answer text and tool-call arguments together, never replenished. Output runs
about 3.3 characters per token, so a session is roughly **50K characters** and
a delivered answer is ~4K of them.

## Recorded accounting

| Session | Completion tokens | Turns | Reasoning chars | Answer chars | Outcome |
|---|---|---|---|---|---|
| #17357 | 15577 | — | — | — | full answer delivered, nothing spare |
| #5585 | 14542 | — | — | — | full answer delivered |
| #11863 | 13095 | 26 tool calls | 26.6K | 4.2K | full answer delivered |
| #3389 | 16384 (cap) | 25 tool calls | 44K (78 %) | 1.4K | headline + classification; closing action table lost |
| #10503 | 16384 (cap) | 30 tool calls | 36K (68 %) | — | answer began only in the final turn |
| #16022 | 16384 (cap) | 15 blocks | 37.4K (one block 7.7K) | — | verdict + causes 1-2, died mid-clause |
| #6784 | 16384 (cap) | 10 turns | 45.8K | 85 | **no answer at all** (the 85 characters are a "let me analyse" line); last two tool calls dropped as truncated |
| #11519 | 16384 (cap) | 19 turns | 43.3K | 1.0K | headline + classification only, cut mid-sentence inside the second section |
| #12339 | 16384 (cap) | 20 turns | 49.0K (largest blocks 7.2K, 6.6K) | 75 | **no answer at all** (the 75 characters are an intro line); last turn had neither text nor a tool call |
| #14506 | 14865 | 13 turns | 32.1K, of which one 12.3K block on the answering turn | 6.1K | delivered a complete answer — but half the session went into the last "one more angle" block, which ran beside it |
| #2586 | 12697 | 19 turns | 26.0K | 4.7K | complete ranked answer, gates never reached |
| #13699 | 9889 | 13 turns | 22.6K | 6.2K | complete answer; the loss was context (a 28 KB turn-1 history listing), not output |
| #10985 | 16384 (cap) | 15 turns / 28 tool calls | 49.1K (largest blocks 10.5K, 8.5K) | 70 | **no answer at all** (the 70 characters are an "I'll start by loading the skill…" line); the last tool call was dropped as truncated |
| #4812 | 10218 | 13 turns | 20.6K | 5.0K | complete ranked answer; the reporter then posted the traceback, which moved the lead from "platform unsupported" to an allocation failure at KV-cache init |
| #9335 | 11031 | 11 turns / 18 tool calls | 21.5K | 5.8K | sizing answer delivered at the last turn; the only waste was the first-turn orientation (13.7 KB bundled listing: `history` filenames + `head -100` of `SOURCES.json`) |
| #11739 | 16384 (cap) | 22 turns / 40 tool calls | 39.9K | 79 | **no report at all** (79 characters are an "I'll start by loading the skill" line); community-test task whose deliverable was a report file, and the file was never written |
| #9183 | 13219 | 10 turns / 15 tool calls | 31.9K | 5.1K | complete ranked answer, delivered on the last turn — but built on a plog block dated the day before the failure, so the ranked mechanism named the wrong owner (fixed by the timestamp gate, not by budget) |
| #6800 | 10305 | 10 turns / 17 tool calls | 19.4K | 4.6K | complete answer, correct owner; the whole cost was repo grepping that the quantization-gate route now replaces |
| #1865 | 14400 | 14 turns / 18 tool calls | 31.3K (largest blocks 6.7K, 6.1K) | 5.6K | complete ranked answer, delivered on the last turn — the last block was a 6.7K pre-write plan |
| #253 | 16384 (cap) | 12 turns / 20 tool calls | 47.6K (largest blocks 13.5K, 5.8K, 5.5K, 5.4K, 5.2K) | 72 | **no answer at all**; the mechanism was fully verified by turn 9 and the remaining three turns only re-confirmed it |
| #9583 | 16384 (cap) | 20 turns / 30 tool calls | 47.6K (largest blocks 6.2K, 6.0K, 5.5K, 4.7K, 4.6K) | 69 | **no answer at all** (the 69 characters are an "I'll start by loading the relevant skill…" line); the cap hit on a 2.6K plan block, with the route gate and the decisive `openapi.json` request both in hand since turn 5 |
| #12253 | 15164 | 14 turns / 16 tool calls | 38.0K (largest blocks 6.6K, 6.4K, 5.1K) | 5.3K | complete ranked answer delivered on the last turn; `findings.md` was written one turn earlier, so the answer survived the cap |

`#2586` and `#13699` are the calibration rows for this batch: both answered well
inside the cap because the ranked shape was written at the 8th-13th turn, and
both spent more on reasoning than on the answer (26.0K/4.7K and 22.6K/6.2K
characters). The second one shows the other kind of waste — input context, not
output: one turn-1 listing of `history/` bought nothing and cost ~28 KB of the
prompt. Cheap context is still a tax on the same session; the output gates do
not meter it. `#4812` is the other uncapped row: 20.6K of reasoning and a 5.0K
ranked answer in 13 turns, delivered before the reporter posted the traceback
that moved the lead.

`#9335` and `#11739` are the pair for the two shapes the older rows do not
cover. `#9335` asked a sizing question ("does the 2-NPU stage layout work"), and
its 5.8K answer arrived on the last turn after 21.5K characters of reasoning;
that is the shape where the answer is arithmetic, so it should be written by the
3rd-5th turn and the remaining turns spent only on numbers that change it.
`#11739` is the *document* deliverable: the run had a stale baseline by its 5th
assistant turn and two command-breaking defects (a dead-flag perf server line, a
relative path that does not exist in the image) by its 15th, then kept auditing
defaults and reading precedent bodies until the cap, and because the deliverable
was a file rather than a message it emitted nothing at all. A report-shaped task
is the worst case for the gates, so it needs the file opened early and appended —
see `../tutorial-verification/community-test-report.md`.

`#1865` and `#253` are the same case family (2025 vLLM DeepSeek deployments)
answered and not answered. `#1865` finished at 14.4K of 16.4K with a 5.6K ranked
answer: its only waste was the 6.7K block that enumerated the answer immediately
before writing it, i.e. the plan the gates already forbid. `#253` is the sharper
row: by its 9th assistant turn it had a *verified* mechanism chain (upstream
`fused_experts = None` gate, the single importer of `vllm_ascend.ops`, the single
caller of the platform hook, `fork` as the default start method) and it then spent
three more turns re-reading upstream to confirm line numbers it had already
matched to the reporter's traceback. The expensive block was not the answering
turn — it was a **13.5K-character mid-run re-derivation**, and the answer never
started. Rule for that shape: once the chain closes on `path:line` matches from the
reporter's own traceback, write the answer in that same turn and let any remaining
doubt appear in the answer's open-questions block.

`#9583` and `#12253` are the latest pair, and they split the same way as
`#10985` / `#4812`. `#9583` is the third member of the "no answer at all" family
and the second inside the observability topic: a `404` on `/start_profile`, i.e.
a route that is registered iff one config field is set. Its answer was three lines
(gate, one `curl`, ranked ways the flag can miss the process) and was available by
the 5th turn; the run instead spent eleven turns deciding whether the CLI leaves
`args.profiler_config` as a dict — a question one `sed -n '247,300p'` on
`vllm/engine/arg_utils.py` settles. Two lessons: for "an endpoint/flag did
nothing", the ownership sentence plus the gate checklist is the answer, and any
"how does argparse turn this into an object" question is a single read of
`_compute_kwargs`, not a chain of hypotheses. `#12253` is the positive row: 14
turns, findings written at turn 13, 5.3K answer on the last turn — the
`findings.md`-at-G2 habit working — with the remaining defect in *evidence
ordering* (it led with a mechanism the pinned code does not establish instead of
the shipped support matrix).

`#9183` and `#6800` then constrain the lesson from the other side. Both delivered
inside the cap (13.2K and 10.3K completion tokens, answer written by the 10th
assistant turn), so neither is a budget failure: `#9183` delivered a
*wrong-window* answer at full length because it reasoned from the plog block the
reporter quoted, and `#6800` delivered a correct one for the price of 17 greps
that the `quantization-method-gate.md` route turns into three calls. Do not read
these rows as permission to reason longer; read them as "the gate that was
missing was evidence ordering, and it costs one script call".

Four shapes recur in these runs, and all are diagnostic rather than unlucky:

- **The archaeology loop** (#6784): ten turns of 1.4-12K characters each,
  re-deciding "which upstream vllm does this image really ship", each block
  restating the previous block's open questions. The decisive observation (the
  traceback names a vllm file that does not exist in the pinned tree) had
  arrived by turn 5 and was never written as text.
- **The late plan** (#11519): the reasoning block at turn 18 already contained
  the whole answer — classification, four ranked causes, evidence table,
  actions, missing artefacts, "Keep ~4K chars" — and spent it on a `findings.md`
  heredoc plus a 4.2K-character block instead of message text. The answer that
  followed was 1.0K characters and truncated.
- **The ownership hunt** (#10985): the report was "`--enable-log-requests` /
  `--enable-log-outputs` set, no prompt or generation printed in 0.18.0" — a
  question whose owner is the upstream OpenAI frontend and a five-line gate
  checklist, reachable with four greps. The run instead walked
  `vllm/entrypoints/openai/**` for 15 turns looking for a downstream defect,
  accumulated a complete correct diagnosis in reasoning (nearly verbatim what the
  gate table now says) and never wrote any of it as message text. Whenever the
  report is "a flag/feature did nothing", write the ownership sentence plus the
  gate checklist in the same turn as the first tool call, then allow at most one
  confirming grep — that class is not a code walk, and reading more entrypoints
  cannot improve it.
- **The "one more angle" turn** (#12339, #14506): the first turns are cheap;
  the loss happens when a new hypothesis needs a new code area and gets its own
  source walk. #12339 walked four mechanisms (rank mapping, mamba mode, A5 MXFP8
  dispatch, recompute scheduler), 49K characters, and shipped nothing; #14506
  answered, but only after a 12.3K-character block on the answering turn that was
  half its budget. A new hypothesis belongs in the answer as an open question with
  the one command that decides it. When the report is an accuracy/corruption or a
  topology question, the ranked list plus the ablation matrix already **is** the
  full answer.

## The gates

Apply these as hard stops; they are the operative form of SKILL.md rule 1.

- **G1 — 3rd assistant turn.** Message text: the direct answer to the reporter's
  literal question plus the track/lane classification. A reporter traceback, a
  config value, or a registry listing is enough; do not wait for the mechanism.
- **G2 — 8th turn, or 20K cumulative output characters.** Message text: the full
  ranked answer (causes with the one check each, cheapest action, version scope,
  missing artefacts). After G2 a tool call is allowed only to confirm or correct
  a named claim, one per turn, each ≤1,500 characters of reasoning.
- **G3 — 25K characters, or the 12th turn.** Answer only: no reasoning text, no
  tool calls, no "let me check one more thing".
- **Per-turn reasoning ≤1,200 characters.** It is a scratchpad: what the last
  result proved, the single next check, go. If a block starts enumerating the
  answer's sections, or says "I have enough" / "that is the crux" / "let me now
  write the answer", that block *is* the answer — emit it as message text in that
  turn.
- **No plan-only turns.** "Now let me check…" spends budget and delivers neither
  an answer nor evidence; #16022 emitted thirteen of them.
- **`/workspace/scratch/findings.md` is the answer draft, not an extra step.**
  Write it at G2 *instead of* the reasoning block you were going to spend, and
  keep it to the answer's own bullet list. When the session is already near the
  cap, skip the file: a written finding nobody reads is worth less than the first
  two answer sections.
- **A missing or mismatched artefact is itself the finding.** An empty
  environment block, a traceback file absent from the pinned tree, a version the
  snapshot does not carry: state the gap, give the one command that closes it on
  the reporter's host, and stop. Two recorded runs spent their whole budget
  resolving exactly such a gap.
- **A new hypothesis that needs a new code area does not belong in this
  session.** State it in the answer with the one command that decides it. For an
  accuracy/corruption or topology report, the ranked list plus the ablation matrix
  is the full answer; further source walks are refinement and must not delay it.
- **Truncation eats the tail.** Put each action directly after its cause and keep
  the closing scope/open-questions block to 3-5 lines, so a capped answer still
  ends in something usable.
