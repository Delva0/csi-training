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

Three shapes recur in the capped runs, and all are diagnostic rather than
unlucky:

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
