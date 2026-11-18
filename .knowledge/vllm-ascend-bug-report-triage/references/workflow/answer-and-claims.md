# Answer and claims: the deliverable, its confidence, and drafting it early

## Required shape

The task asks for at least one effective workaround and at least one root cause concrete enough to
file as an issue. Deliver both explicitly, then the evidence, then the open questions. Workaround
first: it is what the user can act on now. `../../templates/finding-report.md` is the skeleton.

1. **Verdict line** — symptom → mechanism, with the version scope
   (`at vllm-ascend v0.11.0 ...; changed at ...`).
2. **Workaround** — the exact command/flag/config/env delta, where it applies (P node only, decode
   only, one model family), why it works, what it trades away, and whether the reporter already
   confirmed it (`关掉 C8 就可以正常推理` in the report) or you are proposing it.
3. **Root cause** — repository, `path:line` at the reported ref, the gating condition that made the
   bad path reachable, the data flow in one paragraph (what value/shape/ordering is wrong and how
   it propagates), and what a correct fix changes. If you cannot localise a line, say which module
   and which invariant instead of inventing a line number.
4. **Why this symptom and not another** — link the mechanism to the observed log line, stall point
   or accuracy delta. "Somewhere in the MoE path" is not a root cause, and a mechanism that
   predicts an exception while the report shows a five-minute stall is not yet the cause.
5. **Evidence and confidence** — one label per claim, the tags/paths/commits/docs that carry it, and
   the check that would falsify it.
6. **Status and limits** — fixed-in-later-release / open / maintainer-declared known issue, which
   refs are affected, plus what you did not verify.
7. **If the deliverable is issue text** — minimal repro (start command, model, versions), expected
   vs actual, the failing frame, and the proposed fix or workaround.

## When the paste stops before the exception line

A traceback that ends above the failing call, or a log that shows only the wrapper
(`WorkerProc hit an exception`, `EngineCore failed to start`), does not license an enumeration of what
could have raised: the list costs 2–4 K characters and is not evidence. Do this instead.

1. One search for the failing function/attribute name across issues and PRs (exact identifier, not the
   symptom word); take the closest sibling report. First establish *which* wrapper you are looking at: a
   child that raised has already logged `WorkerProc failed to start.` with its traceback, and its absence
   means the child was killed (`../symptoms/startup-and-stall.md` §8).
2. Read the named module's own guard at the reported ref — the condition that makes the failing branch
   reachable — and state the answer for that condition or rank.
3. Label it `hypothesis`, name the falsifier (the print, the line that would appear in a full log, the
   ancestry check) and stop. A labelled partial answer is useful; a truncated turn delivers nothing.

## Workaround tiers, strongest first

| tier | shape | when it is honest |
| --- | --- | --- |
| feature switch off | remove/disable the specific optimisation key (`"c8_enable_reshape_optim": false`, `--no-enable-prefix-caching`) | the code path is gated by that key at the reporter's tag |
| feature class off | disable the whole feature (sparse C8, MTP/speculation, KV pooling) | the fine-grained switch does not exist at that tag |
| environment | HDK/CANN/driver version constraint; env var toggling an alternate path | the failure is documented as a platform interaction |
| upgrade / pin | move past the commit that removes or fixes the path (`>= v0.30.0` after #18128 removed StoreKVBlock) | you have checked the target version actually contains the change (`tag --contains`) |

State a workaround's cost: disabling an optimisation is a performance regression, disabling C8
changes KV-cache precision and accuracy baselines, disabling speculation removes the speedup and is
a bisect tool rather than a fix. Do not present a diagnostic-only change as a workaround.

## Evidence ladder

`verified` — any of:

- the consuming line read at the reported ref (`git show <ref>:<path>`, `git grep -n <ref>`),
  showing how the value is used and what failure it produces;
- an ancestry result: `git merge-base --is-ancestor <fix> <reported-tag>`;
- a unit test that pins the contract — e.g. `tests/ut/attention/a2/test_sfa_v1.py` asserting that
  `c8_enable_reshape_optim` requires `enable_sparse_li_c8`;
- a commit/PR message describing the mechanism, ideally with the failing scenario;
- the release notes or docs shipped at that ref stating the supported combination.

`reported` — issue text, maintainer comment, the reporter's log. Quotable, not proof. A bug closed
as stale with a one-line answer ("you downloaded the wrong weights, use <link>") gives a workaround
to relay, not a root cause to assert.

`hypothesis` — anything inferred from a similar issue, model behaviour or plausibility. State the
discriminating check that would confirm it.

## Wording that keeps you honest

- "At `v0.23.0rc1`, `vllm_ascend/ascend_config.py:282` computes
  `c8_enable_reshape_optim = enable_sparse_li_c8 and additional_config.get(...)`, so the docs
  config sets it to true" — verified.
- "The optimisation launches a custom AICPU operator; #15896 documents that pooling + custom AICPU
  tiling operators hang on that HDK generation" — verified as a linked issue, mechanism link to
  this report inferred.
- "Candidate fix #10259 is absent from `v0.20.2rc1` (`merge-base --is-ancestor` returns 1) and its
  subject matches the reported sampler behaviour; I could not run it on NPU, so this is the
  strongest lead, not a confirmed cause" — the correct form for an unverified fix.
- "The clone has no tag for the release the reporter names, so statements about it come from the
  release branch tip" — the correct form for a substituted ref.

Never: "this is fixed in main so your version is fine", "the issue is closed, so it is resolved",
"it works locally, so the report is environment noise", or a UT-level check presented as hardware
validation. Do not restate a maintainer claim as verified; do not restate the maintainer comment as
your own analysis — your value is the code-level *why*, or the honest statement that the thread is
the only evidence available.

## Verification you can hand the reporter

The reporter has the hardware. Give them a command that splits the hypotheses.

- Quantised checkpoint suspected incomplete:

  ```bash
  uv run --no-project python - <<'PY'
  import json
  d = json.load(open('/root/models/<model>/quant_model_description.json'))
  print('embed_tokens:', d.get('model.embed_tokens.weight'))
  print('entries:', len(d))
  PY
  grep -n 'quantization_config' /root/models/<model>/config.json
  ```

  A missing `embed_tokens` entry vs `"FLOAT"`, and absence of `quantization_config` in
  `config.json`, are different faults with different fixes.
- Runtime gate suspected (for example the 310P compressed-mask ops):

  ```bash
  python -c "import torch_npu; print(hasattr(torch_npu,'_npu_flash_attention_v3'), hasattr(torch_npu,'_npu_paged_attention_splitfuse_v2'))"
  ```
- Runtime flag suspected: re-run with the flag off/on and compare; a knob that flips the symptom
  localises the branch but does not by itself name the defect.
- Version suspected: report the vllm-ascend tag, the `vllm` version from the log banner and the
  image tag together; a mixed install invalidates conclusions drawn from the stated version.

You cannot raise the evidence ceiling by reading more issues. Stop when the mechanism is confirmed
at the ref and one independent signal (later fix, unit test, related issue, doc line) agrees.

## Deliver early: the reply, not the search, is the deliverable

Exploration of this stack is effectively unbounded; the run is not. Hidden reasoning, every tool
call's arguments and the final reply draw on the same allowance, and an answer that never arrives
delivers nothing however good the reading was. The recurring failure on this task class is not wrong
analysis but **no deliverable at all**: runs end mid-read, inside a long reasoning block or inside a
tool call the framework never executes — and they had usually already reached the mechanism (an
allocation that divided out to the scheduled sequence length, a pooling branch that bypasses the KV
cache, a maintainer sentence naming the unsupported model family, a DP all-reduce that was skipped).
What was missing was the write-up, not the insight.

`SKILL.md` ("The write gate") states the gate; this section is how to execute it.

- The reply begins as `content` in the message where the mechanism settles, and that message carries
  **no reasoning**: `reasoning_content` is generated before `content`, so a closer that drafts the
  answer spends the reply on text that is never delivered.
- Keep reasoning tight in every message — a few sentences: the next call and why. Bullets and
  symbols (`suspect`, `gating line`, `next check`) beat re-planned prose, and do not restate tool
  output.
- Two or three independent calls per message, each with one question; read windows, not files
  (`sed -n '1500,1560p'`, `git grep -n -A15 '<symbol>'`).
- If the next action must be either a tool call or the reply, choose the reply: an answer carrying
  `hypothesis` labels is an answer; a truncated message is nothing.

### Stop triggers (verbatim)

These openings, taken from recorded overruns, mean the message ends *now* — the mechanism is settled
and the next token should be the reply (or one bare, small tool call that is the last thing in the
message).

> "Let me structure the reply", "Now write reply (~N chars)", "Now I have enough", "Key facts:",
> "Key facts now:", "Facts settled:", "Now I have the mechanism confirmed:", "Confirmed everything:",
> "Excellent.", "Very promising:", "Interesting:", "Key finds in <range>:", "The key:", "I have the
> decisive fact. Key findings:", "Let me consider another angle", "Another idea grounded in the
> skill", "New idea: …", "Hmm. What explains …", "Better: …", "Wait — but the commenter's matrix",
> "Actually — note …", "Hmm, actual code:", "my count is off by N", "Key: no <symbol> at <path>",
> "Key suspects: 1. … 2. …", "One more check is worth it", "One more small/tiny batch", "Let me do
> one more focused check", "let me just confirm the fix PR", "check whether that branch is already
> there", the budget self-audits — "Budget check: I'm around maybe …", "Let me count: skill_view
> 1 (small), msg2 …", "Careful budget. One call.", "I'm at moderate output. Let me do one small
> call" — and the verdicts-in-waiting: "The class page settles it: workaround = …, status open", "to
> nail the `path:line`", "one quick confirmation read", "given the budget, I have enough", "the page
> is authoritative", the rule-debate shapes — "the gate is a guideline", "per the gate the message
> after the third is content. Hmm", "the batch table says reply but", "I've used N tool messages" —
> "Honestly, <the keys> could be explained if …", "Let me be efficient: one bash
> with a couple of greps", "Possible known classes: …", "What typically produces X".

The shapes behind them:

- **The trigger fires when the phrase is written, not when the paragraph ends.** Four recorded reports in
  one group each opened their last message with one of these ("Key facts:", "Confirmed everything:", "Let me
  structure the reply", "the gate is a guideline; but I need a lever") and then went on for 4 K–12 K
  characters, so the run ended with no `content` at all. The opener stops the *whole message*: there is no
  "but first", because the reply-slot message has no tool to call.
- **Noticing the rule, stating the rule or promising to comply is output.** One recorded closer wrote
  "Now write reply … uh oh, my thinking is long. Guidance says keep the closing message's reasoning
  short … I should now write content directly" and then spent thousands of characters on reasoning
  before a reply that was cut inside the workaround line.
- **A sentence carrying the workaround *and* the status is the reply, not a note on the way to one.**
  Written into `reasoning_content` it is followed by a "confirmation read" of the very page it came
  from, and the report is lost; the next field is `content`.
- **A recap opener is the reply's derivation, not the reply.** "Now I have enough. Key facts: …"
  restates evidence already in context; when it opens the closing message the run believes it has
  started answering and is cut before the workaround line. A report with nothing to find produces
  exactly this shape, with invented facts.
- **A second candidate mechanism for the same evidence** means the mechanism is not readable in the
  clones: the reply goes out with the lever plus the candidate as `hypothesis` and its falsifier
  (`SKILL.md`, "a mechanism that no clone can show").
- **Post-answer exploration is the same mistake.** After the tool result names the lever, a fix's
  first tag, `merge-base --is-ancestor` over candidate shas, `git show --stat` of a PR the thread
  named and "does the file exist at HEAD" are `status and limits` lines; a release-branch NO is not
  absence, and explaining it costs another paragraph.
- **A reply that starts and then re-decides** — a verdict and a workaround followed by a long "which
  one do I recommend" — is the same failure one field later: the first two lines are the verdict and
  the workaround, the rest is the emitted report, not a deliberation.
- **A reasoning-only last message, or a plan for the next batch, is the same loss with no artifact at
  all.** If the analysis needs more than a few lines, it belongs in the answer text, where it is at
  least the deliverable.

### The reply is the only artifact

`scripts/open_answer.sh "<repo>#<N> — <symptom>" <ref>` in batch 1 prints the `ref_snapshot.sh`
facts and drops a one-screen skeleton in `/workspace/scratch/answer.md`; that is its whole job —
keep it as the record of which ref you resolved and what you had verified when you stopped, but
**do not rewrite it, and never re-emit the report as a `cat > answer.md <<'EOF'` heredoc**: the
arguments are charged twice and the run can be cut off inside the heredoc, leaving neither a file
nor a reply.

Write the answer as the message, in one block, once:

```text
<repo>#<issue> at <tag> — <symptom> → <mechanism in one line>          [verified|reported|hypothesis]
workaround: <exact flag/env/version delta>; costs <what it gives up>; <reporter-confirmed | proposed>
root cause: <file:line at the ref>; gated by <condition>; <what is wrong and where it propagates>
why this symptom: <the log line / stall point / accuracy delta the mechanism produces>
status: <fixed-in / open / maintainer-declared known issue>; not verified: <the gap>
falsifier: <the one check that would kill the mechanism>
evidence: <one line per claim, labelled> — keep last and short; a cut here costs little
```

Aim ~1.2–2 K characters (a CJK reply costs more of the allowance per character). If the last claim is
still a hypothesis, say so in one clause and ship the workaround anyway.

### Keep the report in reply shape while you read

1. Keep the report in its **final shape** from the first time you state it: symptom→mechanism,
   workaround, root cause, why-this-symptom, evidence labels, status. Shortening a shape you already
   have is cheap; assembling one from scattered findings at the end is what runs out.
2. Label each claim `verified` / `reported` / `hypothesis` as you state it (the template shows the
   shape), and prefer a narrow verified claim to a broad hedged one.
3. A thread statement of the form "not supported for this model", "fixed in vX" or "disable Y and it
   works" settles the workaround and the status where you read it; you still owe a mechanism, but at
   most one more batch (grep the gating symbol, read one window) and then the reply.
4. Never end a message on a long speculative block or on an unexecuted tool call. If the remaining
   allowance is unclear, the reply goes out now; a named gap can be a follow-up.
5. Once you have (a) the gating condition at the tag and (b) one commit, issue or doc line that
   explains the symptom, stop reading. Prefer `grep -n <token>` over 100+ line ranges of a
   3000-line document.
6. Answer inside-out when the allowance is tight: resolved ref/version, workaround, best current
   mechanism — then deepen. An unstated root cause is worth nothing; a stated one with a named
   falsifying check is useful even when partly wrong, and a reply that names its open point is not a
   failure.

## Deferred work

If a downstream artifact (upstream vLLM commit link, CANN operator, HDK release note, vendor driver
documentation) is referenced but not inspectable from the local clones or the online tools, cite it
as unattributed context and say it was not checked. The credentials available to the online tools
cover the target repositories, not vendor operating-system or driver documentation.
