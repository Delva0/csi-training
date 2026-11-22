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
- **A follow-up turn is a delta turn.** When the reporter replies and the harness sends
  the reply as another user turn, the earlier answer is already in the thread and
  the second cap is often smaller (recorded #8294: 16 384 then **8 192**). Write
  only what the reply changes — the one fact it closes, the one it opens, the
  rung it makes cheapest — and point back at the shipped ranking instead of
  restating it. Restating the whole answer is how the tail block ("open items",
  "what to send back") is the part that gets cut, in a turn that had already
  delivered. **The same delta rule binds a correction you make to your own shipped
  answer.** #11923 delivered a complete ranked answer, then read two more doc blocks
  and reopened with "Final answer — one correction that sharpens the cheapest
  action": a fresh full answer, cut mid-sentence inside its first paragraph, so the
  correction (the A2 `device_sdma`-in-the-sample vs `device_rdma` row) never reached
  the reporter and the delivered answer stayed uncorrected. A correction is 2-4
  lines appended to the ranking already in the thread — the one fact and the action
  it changes — never a re-issue of the answer. Recorded #5227 is the same miss one
  batch later: it delivered a ranked answer, the reporter then replied that the same
  crash happens with `method: "ngram"` too, and the delta turn re-walked the executor
  copy, the proposer and the PP broadcast path before reaching the one fact that
  mattered (`max_concurrent_batches` = `pipeline_parallel_size`, so PP>1 pipelines
  batches whether or not `--async-scheduling` is set) — the 8 192-token cap landed
  inside the revised answer's reasoning, so the improved ranking never shipped. The
  delta was two sentences: "the ngram run retires the suffix parameters; the shared
  axis is PP's batch queue, and PP=1 is the only ablation that removes it"
  (`../pipeline-parallel/spec-decode-first-request.md`).
- **The index is the weak part of the gates: your own count drifts low.** Both runs of
  one recorded batch reported a turn index 2-3x under the real one — #6529 wrote
  "budget: I'm at turn 7. G2 is turn 8" on its **15th** assistant turn (and "Turn 3 now"
  on its 5th), #6243 wrote "I'm at 6 assistant turns" on its **14th**. So a gate that had
  visibly passed was still not applied, and both runs died at the cap with the ranked
  answer already named. Two consequences: (a) the operative trigger is content, not the
  number — the turn you can state a cause with its one check is the turn the answer is
  owed; (b) if you do count, count the `tool` results in the transcript you can see (each
  is a turn already spent, and it is enumerable while memory is not), and treat a
  from-memory estimate as an underestimate by a factor of two.
- **The budget arithmetic, so "a third of it is gone" is measurable:** the cap in these
  sessions is 16 384 output tokens, and mixed Chinese/English ran at ~2.7-3.0 characters
  per token — #6529 spent 44.4 K characters (40.8 K reasoning + 3.6 K text) and #6243
  50.7 K (50.6 K reasoning + **67 characters** of text), both reaching exactly 16 384. So
  a session is ~45-50 K characters, a 5 K-character reasoning block is a tenth of it, and
  the recorded cuts landed at 35-51 K characters of reasoning. A block that is still
  planning past a few hundred tokens is spending the answer's budget. The budget belongs to
  the whole session, not to any one block: #8944's forward wrote ten blocks of 300-2 000
  characters and six of 3.7-8 K, reaching 44.9 K characters of reasoning against 2.7 K of
  text over **17** assistant turns before the cap cut its ranked answer at 1 195 characters
  — the floor is the turn count times ~1.5 K characters, so a run that keeps taking one more
  short look arrives at the same place as one long sweep. Its cheapest lost turns were the
  ones re-deriving an op-map page that was already in context (see step 0 of
  `../custom-ops-runtime-faults/aclnn-error-dump-reading.md`): a page at the reported pin is
  evidence, not something to re-verify in source before writing the answer.
- **Reasoning is a scratchpad, a few lines per turn:** what the last result proved,
  the single next check, go. It spends the same output budget as the answer, so a
  session that reasons in 1-6K-character blocks can hit the cap with zero answer
  text. A block that starts enumerating the answer's sections,
  counts its own turn number against these gates, or says "I have enough" / "that
  is the crux" / "let me write the answer", *is* the answer — emit it as message
  text in that turn; a turn-counting block measures the budget instead of spending
  it on the deliverable (recorded #16365: the verdict was pinned by turn six and
  twenty turns later the answer existed only as reasoning). A "now let me…" plan line delivers
  neither answer nor evidence: every assistant message is a tool call or text. Gate
  *recognition* is not delivery either: #6243's 14th turn reasoned "G1 requires answer
  text by turn 3 (already passed …) I need to send the deliverable", then sent two tool
  calls with empty text; its last turn was 14.4 K characters of reasoning holding the
  whole ranked answer and the repro script, cut inside it, 67 characters shipped.
- **The mechanism is the last thing owed; a source the snapshot does not carry is not
  owed at all.** #1579 had the full traceback, owning frame included
  (`torch_npu/distributed/distributed_c10d.py` `_gather` → `dist.broadcast_object_list`),
  by its **5th** turn, then spent the remaining nine deciding *why* the size payload
  arrived empty — inside `torch/distributed/distributed_c10d.py`, a file that case's
  snapshot does not contain. On its 13th turn it wrote "Time check: this is turn 5
  already … I must deliver the answer now" (its own count was 8 short: the existing
  drift rule, live), then reasoned to the cap instead: 14 assistant turns, **zero answer
  text**, 16 384 completion tokens all inside reasoning, `output_truncated`. The ranked
  answer it needed was already in its own draft — "what the paste does pin is the
  component and phase … the exact reason cannot be pinned" — and never reached the
  reporter. Redirect: once the next check can only confirm a mechanism whose owner is
  outside every local tree, emit the ranking with the mechanism labelled open
  (`../startup-failures/environment-correspondence.md` records the same run as a
  missing-source hunt).
- **A contradiction the snapshot cannot settle — and that cannot change the verdict —
  is a gap, not a lane.** The case to recognise is a paste whose own numbers contradict
  the pinned tree: #119's `Unknown quantization method: ascend. Must be one of [... 22
  names ...]` while that tag's `register()` does import the plugin's config module, so
  the readings are "their wheel differs from the tag" and "an import-order effect".
  Both readings leave the deliverable identical (the release ships the name without the
  implementation), which is the test: resolve only what changes the answer. Write the
  version verdict, list the mismatch as a gap, give the one command that separates the
  readings on the reporter's own box (here:
  `python -c "import vllm_ascend.quantization.quant_config; from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS as m; print(m)"`),
  and stop. That run spent its whole 16 384-token budget on the contradiction and
  shipped no text, while the pinned release note and `AscendQuantizer.get_quantizer`'s
  `NotImplementedError` had answered it two calls in
  (`../startup-failures/quantization-method-gate.md` §E-i).
- **A recomputed estimate is not new evidence: when it keeps agreeing, it retires
  that lane.** A second pass over the same inputs cannot change the verdict — it must
  *become* a message ("the budget fits, so the owner is not the budget") or change
  lane. #2069 priced a DP8 layout three times (~37 of 64 GiB each) while hunting a
  memory cause behind a hang; #1617 re-derived engine defaults while the owner was the
  profile batch — both cut at the cap with no answer text, the *agreement* pointing at
  the real owner
  (`../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`,
  `../deploy-sizing/vl-profile-run-multimodal-budget.md`). A second sizing pass means
  the ranking is owed **now**.
  **A recomputation that *disagrees* with the first is the stronger case, not a
  detour: it changes the verdict, so it is owed a message in the same turn.**
  #3112 derived 2 in-flight sequences from the median ITL, then 26 from the mean
  three turns later — the correction invalidated the KV-capacity lane it had
  already opened — and still shipped no verdict (13 assistant turns, 763
  characters, cut at 16 384 with the ranking in hand:
  `../performance-throughput/prefill-queue-and-throughput.md`, second half).
- **Every assistant turn carries message text, not only a tool call.** A gate only
  binds if the text exists to be cut. #5295 is the clean case: 16 assistant turns,
  every one of them a bare tool call, ~46 K characters of reasoning (≈12 K of a 16 K
  output budget), three explicit notes that it "must produce the answer NOW", and a
  cut inside a reasoning block while opening the one file that would have refuted
  its own last hypothesis. Carry one or two sentences of the current answer draft in
  the same message as every tool call — verdict line first, then what is being
  checked — so any truncation leaves a usable answer.
  #11936 is the same shape at the cap: 28 assistant turns, **zero** message text,
  16 384 completion tokens all inside reasoning blocks, cut mid-thought while
  reading the 310P attention backend — although it had named a ranked candidate
  (the 310P rope branches) by its middle turns, which the platform
  topic now reduces to one probe
  (`../soc-platform-paths/310p-vl-ocr-garbage-output.md`).
  #6429 is the same arithmetic on a **new lane**: 15 assistant turns, 69
  characters of message text (none at all on 14 of them), 45 K characters of
  reasoning, cut at the cap inside a block that was still choosing between
  mechanisms — after it had already found both `acl_base.h` copies and their shared
  include guard, which is the whole diagnosis. That lane page now opens with the
  one command and a stop rule (`../image-dependency-faults/host-cxx-compile-header-conflict.md`).
  #4178 is the same arithmetic with the *mechanism* trap attached: an rc-only
  `FULL_DECODE_ONLY` hang on a VL model, 22 assistant turns, 48 243 characters of
  reasoning and 55 characters of message text (one status line), cut at the 16 384
  token cap with **no answer** — although the attachment's lane-separating lines
  (mode banner, per-rank `Graph capturing finished`, the crash dump's
  `scheduled_new_reqs` step) were in hand by its fifth turn and the missing coverage
  check was three greps. An rc-only failure is a version verdict, not a mechanism:
  `../hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`, SKILL.md rule 16.
  The status-note form is the same waste: #17797 put one short note on each of its
  first 17 turns ("the 310P path has its own model runner — let me check X") and
  #5525 emitted zero message text on 11 of 20 turns, neither ever stating a verdict;
  #8590 put a single 21 K-character block of hypotheses it had already rejected in
  front of the two-line verdict the paste already supported, was cut, and was cut
  again after the reporter's ping.
- **The budget is cumulative output, so the turn numbers above are not the real
  guard.** Three of the recorded sessions show the difference: #4057 was cut on its
  *14th* assistant turn with ~49 K characters of reasoning and zero message text,
  #6833's first run ran 19 turns and ~45 K characters the same way, #13853 burned
  ~11 K of a 16 K-token budget over 22 turns and was cut inside a reasoning block.
  All three had the ranked cause in hand well before the cut (#4057 and #6833 at
  turn four, #13853 at turn ten). #4474 is the extreme of the other shape: only
  **8** assistant turns, but 47 K characters of reasoning against 2.1 K of message
  text, so the cap arrived around turn eight while the last block was still
  re-reconciling two OCR'd screenshot numbers — turn count is no protection when one
  block is the session (`../deploy-sizing/tp-scaling-oom-at-startup.md` step 2 now
  names those two numbers as a legitimate pair). A block of 2-16 K characters is not a
  scratchpad, it is the session; if a block is still planning when it passes a few
  hundred tokens, the answer is already owed — write it. #5750 (10 turns, ~40 K of
  reasoning, 3.8 K of message text, no ranked answer ever sent) and #4357 (14 turns,
  ~38 K, cut inside the answer it had just started) are the same arithmetic with the
  per-turn verdict lines already in place: those lines are hypotheses, and the
  ranked deliverable is still owed as one message.
- **Truncation eats the tail, not the head.** Verdict and the one cheapest action in
  the first paragraph; each further action directly after its cause; the closing
  scope/open-questions block 3-5 lines. A cut answer must still be usable, and the
  evidence table goes last.
- **A new hypothesis that needs a new code area is not this session's job.** Put it
  in the answer as an open question with the one command that decides it. For an
  accuracy/corruption or topology report, a ranked list plus an ablation matrix
  already *is* the full answer. The same holds when two code paths produce the
  identical observation: write both readings and the single check that separates
  them, instead of reading the third file to pick one. Recorded #6598 had the two
  padding candidates for its operator message by its 8th turn and spent the rest of
  the budget choosing between them. The same loop appears when the open question
  lives in a component the snapshot does not export at all (ATB / op-plugin-atb,
  CANN GE, the ops-* repos): no grep can close it, so it is an open question in the
  message from the turn it is noticed — #4323 spent its last third hunting which
  `torch_npu` op reaches ATB, with zero message text emitted.
- **A log line whose fields another component fills is a mapping question, not a
  hypothesis.** #12015 spent roughly half its budget on an echo mismatch: the paste
  read `Enabled custom fusions: norm_quant, act_quant` although the reporter had set
  `fuse_norm_quant: false`. The line is upstream's
  (`vllm/config/compilation.py:313`) and reports `compilation_config.pass_config`,
  which upstream's optimization level fills, while the reporter's key went into
  `additional_config.ascend_compilation_config` — and on 310P that key is gated off
  outright (`compilation/graph_fusion_pass_manager.py:54`). One question ("which
  component fills this field?") settles it; the answer belongs in the reply as
  "expected, not the cause", with one `path:line`, not in a walk through two config
  layers.
- **A documented mechanism that is absent or fixed at the reporter's ref is the
  answer's first sentence, not a detour.** Reading the branch a page names and
  finding it changed (`_is_string_type` now type-aware, the frontend
  `remaining_args` reconciliation deleted, a constant moved) dates the page and
  retires a cause — that verdict ships *before* the next owner is explored,
  because "your ref is not this bug, here are the two it can be with one check
  each" is a complete reply. #14656 is the recorded cost: the gate was shown fixed
  at its 7th turn, then three further owners were opened one at a time and the
  session ended with two empty-text turns and no answer.
- **An ownership question is answered by naming the owner.** "The flag did
  nothing", "the log line is missing", "the client cannot parse the tool call" are
  gate checklists on the owning component, and they are answered in 3-4 calls.
- **When the deliverable is a document, the document is the gate.** A
  community-test/verification task asks for a report file: open it on the second or
  third turn with the skeleton plus the first verified section, append one section
  per finished check group, and make the final message the condensed report (path,
  defects, open items). Never hold it for a complete audit. **The unit of writing
  is one 2-3 K-character append, not one heredoc containing the report**: #11759
  had a finished script and a drafted report in the last two turns and lost both to
  the cap, because a full report body is 5-6 K characters on its own. If the next
  message would be a tool call that carries more than ~2 K characters of text, it
  is already too late to be safe — write the smaller section now. **The writes are
  spent from the answer's budget, and the delivery is what gets cut**: tool-call
  arguments are output tokens. #11751 got the file written (a 4.3 K skeleton plus
  two appends, 15.4 K characters) and then had its *message* cut inside its defect
  table; the session's 16 384 output tokens were 24.0 K characters of reasoning,
  16.1 K of tool-call arguments (the three writes — 36 % of the cap) and 3.8 K of
  text. So cap the file body at ~8 K characters and the message at ~2.5 K, and
  order the message so the acceptance items (the three scores, the report path, the
  verdict) precede any script or table the file already holds.
- **Verifying what a page already verified is not evidence, it is spent budget.** A
  reference page records the pins it was checked at. When `SOURCES.json` carries
  those same refs, cite its `path:line` and re-grep only the claims it marks as
  version-specific. #6833's replay re-walked the whole prefix-cache gate chain at
  identical pins (config hook, `MambaModelConfig`, the model's assert, the
  downstream patch, the loggers line) — ten turns that bought nothing, and its
  action ladder was the part that got cut. The mirror case costs the same when the
  page's pins are **older** than `SOURCES.json`: its `path:line` may no longer exist,
  and a grep that answers `No such file or directory` is not a lane change. #5835
  ended exactly there — 15 assistant turns, 27 tool calls, 50.3 K characters of
  reasoning against **74** characters of message text (the opener), the ranked
  two-lane answer already drafted in its reasoning, and the cut landed while checking
  `vllm/entrypoints/openai/processor.py`, a path a page verified at vllm v0.7.3
  against a v0.13.0rc1 snapshot (the fact it wanted — `guided_decoding_backend` is
  gone, the backend is server-resolved — was one whole-tree `rg` away and is now in
  `../accuracy-benchmarks/structured-output-garble.md`).
- **A reasoning block that drafts the ranked answer charges it twice, and the second
  copy is what gets cut.** The #14731/#12102 pair is the measured case: 17 assistant
  turns each, 42.9 K and 39.4 K characters of reasoning against **1.3 K and 2.7 K
  characters of delivered answer**, both ending at the 16 384-token cap with the
  ranked list half-written. #14731 spent one mid-session block of 11.8 K characters
  deriving the GLM parser (a quarter of the budget in one turn), then opened its
  final turn with a 6.2 K block that listed the verdict, the five ranked causes, the
  evidence table and the gaps *before* typing any of it; #12102's final 5.0 K block
  did the same and explicitly noted "this is turn 12 = G3: answer text only" while
  writing the answer in reasoning. Two habits fix it: the ranked list is written
  **once, in message text**, and a block that has started numbering causes ends
  there — the next characters are the message. The per-turn habit was the other half
  of the loss: #14731 emitted message text on only 3 of its 17 turns (77, 444 and
  1 309 characters), so its ranked answer only ever existed in the turn that was cut,
  while #12102 carried a one-line verdict on 14 turns and still kept the ranked list
  for the last message.
- **`/workspace/scratch/findings.md` is the answer draft, not an extra step:** write
  it at G2 *in place of* the reasoning block, and skip it when the session is
  already long. Composing it late is what costs a run its ranked causes.
- **The ranked answer has a size, not only a deadline.** #16563's two runs on the
  same container-memory report are the pair: one emitted no answer text on 20
  turns (a 69-character opener, then bare tool calls) and was cut inside a
  reasoning block; the other kept a verdict line on most turns and wrote
  `/workspace/scratch/findings.md`, and was *still* cut inside its last section,
  because the answer ran to five causes with a sub-paragraph each, a precedent
  table and a six-rung ladder. For a report with no log at all, the body is
  **3-4 causes, one ladder of ≤6 lines, precedents as one line of `#id: fact`,
  3-5 lines of scope** (~3-4 K characters). The block that gets cut is the
  closing scope/"what to send back" list, which is exactly what the reporter acts
  on next — so it stays 2-3 lines and the mechanism detail goes to the cause it
  belongs to.

**A block that has already started the answer is not closer to delivery.** #16111
wrote `Verified facts (pinned …)` and #8788 wrote `**Answer**: Yes, real.` inside
`reasoning_content`; both were cut there, so the reporter received neither. Copy the
sentence out into the message in the turn it is written — a verdict line, a numbered
fix list or an evidence table inside reasoning is the same spend with the same risk as
a plan.

**A table-shaped answer is emitted row by row.** When the deliverable is a
comparison or evidence table (verdict per model, cause per check, defect per
tier), the rows do not have to ship together: write the verdict sentence plus the
first two rows on the turn they become true and append the rest, because a wide
markdown table composed once, late, is the single largest message the session ever
writes — and #1684's cap landed inside row 1 of it, in a session whose every text
turn had been a status line. A status line per turn is not delivery either; a row
with its anchor is.

**Copy the sentence, not the plan.** #17991 is the clearest recorded case: the mechanism was settled in the reasoning block and the message text said it was about to check two more facts. The remedy is mechanical — when the reasoning block reaches a sentence that names the owner and the line, that sentence is the message text of *this* turn; the tool call rides along after it. A block ending in a plan ("let me pin two last facts") ships nothing, and the next call may never run.

## Why: the recorded pattern

The difference between the runs that delivered and the runs that did not is not the
amount of evidence gathered — it is whether the message was written once a usable
lead existed, classified by what the run already had in hand when it kept going.

The per-run table — what each session already had in hand and what the reporter
received, annotated with `no answer shipped` / `cut mid-answer` — is
`delivery-recorded-sessions.md`; open it only when a delivery decision is
contested or a run is about to spend turns after a usable lead.

Two consequences worth restating: an "I'll start by…" line is not an answer and
does not satisfy G1, and a run whose last turn is a tool call has shipped nothing.

**#6529 and #6243 are the batch's natural experiment.** Same skill read, same report
family (DeepSeek-V3.2 on A2), near-identical spend (44.4 K vs 50.7 K characters) and the
same mis-tracked turn count — the difference is that most turns in #6529's loop carried
the current verdict while #6243's fifteen carried none. Everything else that separated
them is downstream of that habit.

**#12261 and #2015 are the pair that separates "carries message text" from
"carries the answer".** Both kept a one- or two-sentence finding on nearly every
turn — the habit #5750 and #4357 lacked — and both still shipped nothing, because
what the reporter needed was never written: the ranked causes, each with its one
check, and the action they own (declare the argument `type: "string"`; pass
`quantization="ascend"`). The mechanical test before sending any turn's text: would
the reporter be able to act on this message alone, with the session ending here? A
finding without its action is a status line, and six of them are still not an
answer. Both sessions also show what the last-third of the budget goes to when the
deliverable is already owed: re-deriving arithmetic the paste plus two functions
already settle.

**Reasoning is not a draft, and the tell is the commit phrase.** Two runs of one
minibatch (#2446 perf, #2537 accuracy — the same GLM-4.5 / 2×8 910B / DP2×TP8+EP
stack) each classified correctly inside three turns, then composed the ranked table
*inside `reasoning_content`* across 6-10 K-character blocks. #2446 ended two message
drafts mid-sentence and shipped 159 characters when the 8 192-token item cap cut it;
#2537 wrote "Now write the answer in Chinese, ranked…" as its final reasoning block
and shipped 53 characters in 26 turns. So: a reasoning block that already holds the
ranked causes, or that ends "现在写答案 / now write the final answer", has delivered
nothing — that turn's **message** is the answer, and the next call is a correction to
it, not a prerequisite.

A third, from #13002 and #6833 together: **a long source file is not a lead.** Both
runs chose one big module (a 2600-line KV connector; the hybrid-KV/prefix-cache
sources) and read it in slices — the least actionable way to spend an output budget. When the next step would be "read
more of this file", write the answer with the open question and the one command that
settles it — the config diff, the ablation row, the `nm` line. In #13002's first run
the model tutorial's validated commands were read at turn 17 of 21 and the ranked
causes were in hand from that read onward, while the replay that had the worked page
never opened the connector at all. #12432 repeats it one batch later: 23 turns spent
in `mooncake_connector.py` and the recompute scheduler, a "draft verdict" cut
mid-sentence, no answer — while the decisive comparison (the tree's recommended proxy
command versus the DP/`--api-server-count` layout of both serve commands) was in the
tutorial the reporter had linked. #3162: the whole budget went to `pyhccl.py` and
recalled trl internals with no verdict, although #942's broadcast-returning-zeros
precedent plus one question (which patch supplies the communicator) was the deliverable
from turn four. The paste and the tree's own recipe decide these reports;
reading further into a module does not.

**A `csrc/` `TORCH_CHECK` is the answer, not a lead.** #14728 had its line by the third
turn (`csrc/torch_binding.cpp:476`, `x.size(1) <= slice_size`) and then spent the rest of
the budget inside that kernel and upstream `vllm/lora/layers/` asking *why* the guard
exists, while the deliverable was the two operands — the engine's `--max-lora-rank`
against the packed part's width — plus the rung that changes one of them. When an
assert's message is a comparison whose two quantities you can already name, the guard's
existence *is* the finding: the kernel body cannot move the verdict, and its tiling
arithmetic only softens the wording of the code-change rung. That is #6082's op hunt and
#4323's ATB emitter one layer up, and it is the same rule: stop when the next read can
only add detail to a sentence you could already send.
