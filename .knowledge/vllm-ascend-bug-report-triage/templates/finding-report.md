# Finding report — template

This is the **reply**, not a file to maintain: fill the shape and send it as the final message text
(about 2 000 characters; the compact block in `../references/workflow/answer-and-claims.md` is a good
600–1 000-character answer, and this longer skeleton is for when the deliverable is the issue text
itself or budget remains). Do not write it to `answer.md` first — the characters are charged twice and
a cut-off session delivers nothing. Delete sections that do not apply; keep the evidence/limits split.
Put the workaround first: it is what the reporter can act on now, then the root cause, then the
status/limits line and the falsifier, and **last** the per-claim evidence. In that order a reply clipped
by the output cap still carries the lever and the mechanism (the recorded clip lost exactly the sections
that came after `root cause`).

## Symptom (as reported)
- Version: <product + vLLM + torch_npu + CANN> (resolved ref: `<tag/branch>`)
- Hardware: <device class, count, per-card memory>
- Command / config: <the exact flags that matter>
- Observed: <error line, stall point, score delta> — from <issue body / attachment name>

## Workaround (apply now)
- Change: <exact flag or version change>
- Expected effect: <what stops happening>
- Cost: <capability or performance given up>
- Applies to: <version range, node role (P/decode), model family>
- Evidence: <maintainer comment / code path / release note that predicts this>

## Root cause
- Owning path at `<ref>`: `<file:line>` — <one sentence on what it does>
- Gating condition: <what made the bad path reachable at that ref>
- Mechanism: <what value/shape/ordering is wrong, and how it reaches the symptom>
- Why this symptom and not another: <link mechanism to the observed log line or accuracy delta>
- Confidence: <observed-in-code / stated-by-maintainer / hypothesis>
- Falsifier: <the check that would disprove it, if unverified>

## Upstream state / applicability
- Fixed in: <release/PR, or "not fixed">; scope: <which versions/hardware/configurations>
- Recheck when: <version bump, hardware change, flag rename>

## Not verified
- <what you could not check locally and why; stale/partial knowledge about the stack>
