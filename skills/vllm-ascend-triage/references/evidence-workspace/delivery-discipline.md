# Delivery discipline: answer first, and make truncation harmless

## Scope

How a triage session in this library is spent, independent of what the report is
about. It applies to every entry shape: the answer is a *message for the
reporter*, and a session that ends without one delivered nothing, however good its
leads were.

## The gates

- **G1 — 3rd assistant turn:** message text carrying the direct answer to the
  reporter's literal question plus the track/lane classification. A pasted
  traceback, one config value or one registry listing is enough; do not hold the
  verdict for the mechanism.
- **G2 — 8th assistant turn:** the full ranked answer (each cause with its one
  check, the cheapest action, version scope, missing artefacts). After that, a tool
  call is allowed only to confirm or correct an already-named claim: one per turn,
  a couple of lines of reasoning.
- **G3 — 12th assistant turn:** answer text only — no reasoning, no tool call, no
  "one more thing".
- **Reasoning is a scratchpad, a few lines per turn:** what the last result proved,
  the single next check, go. It spends the same output budget as the answer, so a
  session that reasons in 1-6K-character blocks can hit the cap with zero answer
  text. A block that starts enumerating the answer's sections,
  or says "I have enough" / "that is the crux" / "let me write the answer", *is* the
  answer — emit it as message text in that turn. A "now let me…" plan line delivers
  neither answer nor evidence: every assistant message is a tool call or text.
- **The budget is cumulative output, so the turn numbers above are not the real
  guard.** Three of the recorded sessions show the difference: #4057 was cut on its
  *14th* assistant turn with ~49 K characters of reasoning and zero message text,
  #6833 ran 19 turns and ~45 K characters the same way, #13853 burned ~11 K of a
  16 K-token budget over 22 turns and was cut inside a reasoning block. All three had
  the ranked cause in hand well before the cut (#4057 and #6833 at turn four, #13853
  at turn ten). A block of 2-16 K characters is not a scratchpad, it is the session;
  if a block is still planning when it passes a few hundred tokens, the answer is
  already owed — write it.
- **Truncation eats the tail, not the head.** Verdict and the one cheapest action in
  the first paragraph; each further action directly after its cause; the closing
  scope/open-questions block 3-5 lines. A cut answer must still be usable, and the
  evidence table goes last.
- **A new hypothesis that needs a new code area is not this session's job.** Put it
  in the answer as an open question with the one command that decides it. For an
  accuracy/corruption or topology report, a ranked list plus an ablation matrix
  already *is* the full answer.
- **An ownership question is answered by naming the owner.** "The flag did
  nothing", "the log line is missing", "the client cannot parse the tool call" are
  gate checklists on the owning component, and they are answered in 3-4 calls.
- **When the deliverable is a document, the document is the gate.** A
  community-test/verification task asks for a report file: open it on the second or
  third turn with the skeleton plus the first verified section, append one section
  per finished check group, and make the final message the condensed report (path,
  defects, open items). Never hold it for a complete audit.
- **`/workspace/scratch/findings.md` is the answer draft, not an extra step:** write
  it at G2 *in place of* the reasoning block, and skip it when the session is
  already long. Composing it late is what costs a run its ranked causes.

## Why: the recorded pattern

The difference between the runs that delivered and the runs that did not is not the
amount of evidence gathered — it is whether the message was written once a usable
lead existed. Classified by what the run already had in hand when it kept going:

| Session | In hand before it kept going | What the reporter received |
|---|---|---|
| #4296 | one pasted traceback | complete ranked answer, early |
| #2586 | a candidate mechanism named in a precedent title | complete ranked answer |
| #9183 | a quoted plog block | complete answer, but built on a block dated the day before (its failure is the timestamp gate, not delivery) |
| #3389 | the captured-size arithmetic | headline plus classification; the closing action table was lost |
| #16022 | verdict plus causes 1-2 | died mid-clause |
| #11519 | the ranked causes | headline plus classification, cut mid-sentence |
| #6784, #12339, #10985, #11739 | a usable lead in every case | **no answer at all** — only an "I'll start…" or "let me analyse" line |
| #14506 | the answer | complete answer, but half the session went into one final "one more angle" block that ran beside it |
| #9583, #10985 | the registration gate and the decisive request | nothing: the session went into the owning module instead |
| #13853 | the two-reason story (cache block never clamped to 128 on a hybrid model, `--block-size` raised by the mamba page-size constraint) by its 10th turn | **no answer at all**: 22 turns of reasoning (~11K tokens of a 16K output budget), every message a tool call, cut inside a reasoning block |
| #6833 | the metric ratios (queue prefill-bound, 0.5-1 tok/s per request during prefill) | **no answer**: fifteen further turns inside the hybrid-KV-cache and prefix-caching sources |
| #4057 | the fp32 GDN prefill replacement and the tutorial's experimental note | **no answer**: cut mid-plan on turn 14 while re-deriving the kernel algebra |

Two consequences worth restating: an "I'll start by…" line is not an answer and
does not satisfy G1, and a run whose last turn is a tool call has shipped nothing.
