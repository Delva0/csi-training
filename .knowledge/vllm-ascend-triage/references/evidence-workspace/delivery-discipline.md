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
  planning past a few hundred tokens is spending the answer's budget.
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
  action ladder was the part that got cut.
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

**Copy the sentence, not the plan.** #17991 is the clearest recorded case: the mechanism was settled in the reasoning block and the message text said it was about to check two more facts. The remedy is mechanical — when the reasoning block reaches a sentence that names the owner and the line, that sentence is the message text of *this* turn; the tool call rides along after it. A block ending in a plan ("let me pin two last facts") ships nothing, and the next call may never run.

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
| #6833 | the metric ratios (queue prefill-bound, 0.5-1 tok/s per request during prefill) | run 1: **no answer**, fifteen further turns inside the hybrid-KV-cache and prefix-caching sources. Replay (same paste, after the worked page existed): verdict, arithmetic and the gate chain were written by turn 19 — then the session was cut mid-sentence in the action ladder, because it opened only the cache page and re-derived the queue arithmetic from source |
| #4057 | the fp32 GDN prefill replacement and the tutorial's experimental note | **no answer**: cut mid-plan on turn 14 while re-deriving the kernel algebra |
| #13002 | at turn ~10: the model's own tutorial carried the validated P and D commands, i.e. the config diff that *is* this report's answer | **no answer**: 21 turns and the full 16K output budget, ~8 of the turns inside one 2600-line `mooncake_connector.py`, cut inside a reasoning block after noting "I'm now at a good point to deliver" |
| #13002 / replay | the page's own P/D diff table and its two report-side traps, available from the first skill read | complete ranked answer at its 11th assistant turn (17 tool calls): the three symptom shapes separated, the five-row flag diff, the ladder, and the connector opened once for one `dp_size`/`tp_size` grep — the same paste the first run spent 21 turns and its whole budget failing on |
| #11454 | at turn 2: the literal question (`what does hccl_result = 4 mean?`) is the `HcclResult` enum, and the two red herrings were already refutable from the paste — `error_codes=[-1]` is not the out-of-space code `-200`, and `"Check memory and store capacity"` is canned text at `mooncake_backend.py:194` | **no answer**: 22 turns, every message a bare tool call, the full 16K output budget inside reasoning blocks, three or four whole-`/workspace/repos` greps hunting an emitter that is in no export, cut inside a reasoning block (`../kv-pool-store/hccl-onesided-store-put.md`) |
| #5295 | by turn 10: the selector predicate (`world_size_across_dp/pp >= 16` on A2), the correction that MC2 inherits the All2All `finalize`, and the capacity arithmetic from `cudagraph_capture_sizes: [4]` | **no answer**: 16 turns, every message a tool call, ~46 K characters of reasoning, cut inside a block that was opening `ascend_forward_context.py` to check `padded_num_tokens` — the check that refutes the theory it was about to state (`../feature-enable-recipes/moe-expert-parallel-with-dp.md`) |
| #15815 | the whole chain (stale factory alias, `OOT, None`, masked property) by turn 12 and the traceback lines matching the pin by turn 3 | complete ranked answer — but only in its **15th** turn, after 50 K characters of output of which 31 K were reasoning (14,553 of a 16,384-token cap). It shipped because 1.8 K tokens were left; nothing about the session was shaped to survive a cut |
| #8724 | the frame line and the three-traceback signature by turn 3; the decisive patch-vs-upstream diff was the one step that never ran | **no answer**: 29 turns, every message a tool call, ~48.5 K characters of reasoning against 73 characters of text, cut inside a reasoning block on the turn it finally started the diff (`../upstream-downstream-drift/patch-copy-vs-upstream.md`) |
| #17991 | the full source-verified diagnosis (`fa3_v1.py:132-136`, request-indexed `block_tables` sliced by a token count) written in the reasoning at its 17th turn | **no answer**: ~51 K characters of reasoning against 255 characters of message text; the 186-character message on that turn was `Confirmed the fault mechanism. Let me pin two last facts`, and the cap landed on the next tool call (`../mixed-batch-splits/fa3-mixed-batch-split.md`) |
| #6598 | the two numbers from the screenshot OCR (a padded graph bucket vs the real `actualSequenceLengthQ`) plus both candidate padding owners and the tutorial diff, by turn ~8 | **no answer**: 11 turns, ~35 K characters of reasoning against 1.2 K of status-line text, cut at the 16 K output cap inside a block that had just written "I must emit the answer NOW" (`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`) |
| #3098 | the model-specific split (the same ray cluster serves the sibling) and the release note's first-support statement | a ranked answer did reach the message, but the cap landed inside its own ladder (rung 3 of 6): the closing "what to send back" list never shipped, which is the block the answer-size rule above says keeps 2-3 lines |
| #3114 (v0.10.2rc1, 2×8 910B3, ray TP8×PP2, "启动 vllm serve 依然报找不到NPU") | by turn 2: the paste's own `ray status` lists `0.0/16.0 GPU` and no `NPU` line while vllm-ascend asks for the `NPU` resource key (`platform.py:52`), so the answer is one `ray status` check plus `ray start … --resources '{"NPU": 8}'` — and the tutorial the reporter followed is absent from the pinned tree (a doc gap, state it) | **no answer at all**: 14 assistant turns / 22 tool calls, ~49 K characters of reasoning against **81** characters of greeting, cut at the 16 384-token cap; the walk continued past the discriminator into ray's accelerator docs, upstream `ray_utils`, release notes and the precedence corpus (`../startup-failures/ray-npu-resource-label.md`) |
| #11759 | each section as it became true: the absent benchmark section (turn 6), the ignored `max_completion_tokens` (turn 8), the one-card sizing arithmetic (turn 11), the `transformers==4.51.0` pin conflict (turn 14) | **no file, no message text**: 16 turns, 46.5 K characters (29.8 K reasoning, 69 characters of message text), the report heredoc written on the last turn and discarded — a 5.7 K-character composition after a 3.5 K-character script the turn before |
| #11751 | the page's defect set pinned by its 10th assistant turn (the 32B tab's `max_completion_tokens` TypeError, the `v0.11.0rc3-a3` baseline, the missing performance section, the doc's 0.5389 against the project's 0.55 e2e gate) and written into a 15.4 K-character report file | **the file shipped, the message was cut**: 34 messages, the full 16 384-token cap split into 24.0 K characters of reasoning / 16.1 K of tool-call arguments (the three appends) / 3.8 K of text; the cut landed in row 5 of the 6-row defect table, so the last row, the three 完整性/易用性/正确性 scores and the FAQ — three 交付件 items — never reached the reporter (`../tutorial-verification/community-test-report.md`) |
| #16111 | by turn ~12: the lazy init is documented (#9975), the config-table diff (`dram.size` per die, the non-existent `standalone.dram.size`) and the #14106 swallowed-save → collective-desync family | **no answer at all**: 36 turns, every message a bare tool call, cut inside a reasoning block that was composing the deliverable ("Verified facts (pinned …)") — the answer existed only as reasoning |
| #8788 | by turn ~18: the invocation chain and the four `csrc/third_party` fetches plus the catlass submodule | **no answer at all**: 30 turns, every message a bare tool call, cut inside a block that had already begun "**Answer**: Yes, real…" |
| #16563 / A | by turn ~14: the counter is cgroup host memory, the +80 GB/7 d arithmetic (3.5-7 KB per request, <16 B per decode step, so per-step host buffers are priced out), the closest precedents (#2650 host memory 1-2 GB/h, #12345 DeepSeek-V4 leak, #5847 CPU-memory leak) and the A2 recipe diff (`--no-enable-prefix-caching`, `additional_config.enable_flashcomm1`) | **no answer at all**: 20 turns, a 69-character opener then bare tool calls, ~48 K characters of reasoning, cut inside a reasoning block (`../runtime-memory/container-working-set-growth.md`) |
| #12261 | by turn 11: the parser's `_is_string_type` gate on a nullable `anyOf: [string, null]` argument, the raw branch, the length-prefix delta and the arithmetic that reproduces the paste | **no answer at all**: 15 turns / 23 tool calls, ~35 K characters of reasoning against 3.5 K of message text, a one-line verdict on most turns, cut on a tool call while trying to `import vllm` (no torch in the sandbox) to simulate two functions it had already read |
| #14656 | by turn 7: the page's schema-type gate is **fixed** at v0.23.0 (`_is_string_type` now `extract_types_from_schema`), so the GLM-5.2 paste needed another owner, and by turn 12 two verified candidates plus the downstream patches excluded | **no answer at all**: 15 assistant turns / ~16 K completion tokens of reasoning (blocks of 5-10 K characters) against ~2.1 K of message text, two final turns with empty text, cut inside a reasoning block; the streamed-text owner (`resetting decode stream`) was named as a hypothesis and never grepped (`../tool-call-streaming/detokenizer-text-delta.md`) |
| #7682 | by turn 3: the assembled tool-call arguments are the *last two characters* of the non-streaming string (non-stream complete, stream `"}`), i.e. only a suffix was ever emitted — one subtraction, no source read | **no answer at all**: 16 turns / 16 384 output tokens, ~9 K-char reasoning blocks, ~630 characters of message text in total; the turns went into the missing `glm47` registry in a vllm v0.13.0 export that predates the runtime, then `serving_chat`'s `remaining_call`, then the sampler, and the cap landed on a block deciding which file to open next (`../tool-call-streaming/suffix-only-arguments.md`) |
| #2015 | by turn 11: the pinned `quantization.md:37/:5/:94` contract (the W8A8 export carries no `quantization_config`, so the flag is mandatory) and the `enforce_eager` exclusion of the ACL-graph cause | **no answer at all**: 17 turns / 25 tool calls, ~45 K characters of reasoning against 4.0 K of text, cut on a tool call |
| #1796 | by turn ~6: the wheel's `Requires-Dist` pin is `torch-npu==2.5.1.post1.dev20250619` while `setup.py:341` drops the `--extra-index-url`/`--pre` lines from the metadata, i.e. a mirror-only build made unfindable — and `torchvision<0.21.0` puts the same requirement set on the 2.5.x torch line against vllm 0.9.2's `torch==2.7.0` | **no answer at all**: 39 tool calls, ~16 K completion tokens of reasoning, every assistant message a bare tool call; the budget went to whether that dev wheel was rebased onto torch 2.7.1, a fact no local file can settle (`../image-dependency-faults/pip-index-visibility.md`) |
| #3107 | by turn 3: the paste names the two objects whose roles differ (`attn_metadata.max_tokens_across_dp` → `get_forward_context().max_tokens_across_dp`), and by turn 5-6 the pin's writer (`ascend_forward_context.py:91-98`), the three consumers and the `dp_size × max_tokens = 544` arithmetic against the pasted `[544, 5120]` | **no answer**: 12 turns, 7 with no message text, ~49.6 K characters of reasoning against 661 characters of text, 22 tool calls, cut on the last turn's reasoning block; the `pangu_moe.py`-absent-from-the-pin finding (turn 4) was the one sentence the reporter needed (`../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`, `../startup-failures/environment-correspondence.md`) |
| #8294 | first turn: the `profile_run` extra-`_dummy_run` lane read off the traceback line, the capacity/`global_bs` derivation from *both* nodes' capture-size lists, and the refuted-rows list — a complete ranked answer in 11 turns | second turn, after the reporter's "single node works" reply, under a **8 192**-token cap: it re-derived the whole ranking instead of the delta and was cut inside its closing "still unproven" section. The delta was two sentences (at world 8 the single node never selects MC2; the two nodes' `cudagraph_capture_sizes` disagree, 64 vs 512) |
| #698 | by turn 3: the phase after `Loading model weights took` (`worker.py:240-256` → `profile_run` → `npu.synchronize`), and by turn 6 the V0/V1 fingerprint from the log's own `model_runner.py:904` | **no answer**: 14 turns, 11 of them bare tool calls, ~47.6 K characters of reasoning against 322 characters of text; the answer's first paragraph was the artefact request (the full `vllm serve` line — the maintainer's own first reply asked for exactly that, and the reporter supplied it the next day) (`../startup-failures/silent-stall-after-weight-load.md`) |
| #2754 | by turn ~6: the failing phase (torchair graph load), the all-peer `DestLink` block refuting "one bad edge", and the KV-before-capture ordering | **no answer at all** — 12 turns, 7 with no message text, the whole 16 K output budget in reasoning, cut inside the answer's opening sentence; the budget went to the `EI0007` QP-reservation mechanism, whose sources (`cann/hccl`, `cann/hcomm`, `cann/ge`) the case's own `SOURCES.json` lists as `unavailable`, so no local file could settle it and only the budget ladder was owed (`../hccl-and-hardware/qp-resource-exhausted-at-graph-load.md`) |
| #16600 | by turn 9: the model's support row sits only in the `Atlas 300I DUO` tab (so "only 310P has quantized weights" is the project's SoC scope, not a gap), the same-SoC e2e recipe and the two `--additional-config` keys no code reads — a complete capability verdict for a "速度很慢 + 能否提供量化版本" report | **no answer**: the verdict sat in a mid-session message, then ~15 turns went into `platform.py` capture-mode gates, the hybrid-KV gate and issue bodies; the cap landed inside the block that had just started the ranked list (`#1 没开 MTP…`). A perf claim with no metric block is answered by the capability verdict plus the request for the metrics, not by a gate chain (`../feature-enable-recipes/per-soc-recipe-and-quantized-weights.md`) |
| #16563 / B | by turn ~9: the same class and arithmetic, the env-flag liveness findings (no consumer at the pin for `VLLM_ASCEND_ENABLE_FLASHCOMM` / `PYTHON_NPU_ALLOC_CONF`), the MTP `pin_memory()` asymmetry and a `findings.md` draft | verdict, five causes, four precedents and the six-rung ladder did reach the message, but the cap landed inside the closing version-scope section: five causes with sub-paragraphs plus a precedent table plus a ladder is more answer than the budget holds |
| #894 | by turn 2: a 4-token paste is a degeneration report with no serve command, no log and no env, so a ranked matrix *is* the answer — and the pin's forced `enforce_eager = True` (`platform.py:124`) plus the `-Chat`-only validated row were one grep each | **no answer at all**: 20 turns / 35 tool calls, 69 characters of message text against ~43 K characters of reasoning, the full 16 K output budget spent on `deepseek_v2.py` / `fused_moe.py` / `platform.py` (would this branch break V2-Lite?), cut inside a reasoning block (`../accuracy-benchmarks/deepseek-v2-lite-mla-degeneracy.md`) |
| #4323 | by turn ~12: the ATB block is a real `atb::OperationSetup` failure inside `libop_plugin_atb.so`, so the pasted `eagle_proposer.py:476` frame is a join point (precedent #447 carries the same wording over `_npu_reshape_and_cache`); `draft_tensor_parallel_size` has no consumer under `vllm/v1/**` at the pin, and the eagle3 e2e there is single-card Llama-3.1 drafts only | **no answer**: 26 turns, **zero** message text on every one of them, the 16 384-token cap reached inside a reasoning block that already held the ranked causes and the ladder; the closing turns asked which `torch_npu` op reaches ATB — a fact no export in the case contains (`../spec-decode-mtp/atb-op-setup-failure-at-proposer.md`) |
| #11095 | the phase (frozen `Loading safetensors checkpoint shards: 0/11` inside `vllm_ascend/worker/model_runner_v1.py:3635` `get_model`), and a *self-derived* "HBM 26618 MB ≈ 51.75 GiB / TP2 ⇒ the weights are fully loaded, so the stall is after the copies" | **no answer at all**: 11 assistant turns / 606 K prompt tokens, the whole 16 K output budget spent in reasoning on the post-load paths that inference pointed at (Qwen3.5 registration, GDN Triton warmup, NZ/quant cast, cpu binding). Resident HBM is the *pre-load* state — the loader builds the model on the device before reading (`base_loader.py:53-64`) — and the reporter's own resolution (a fresh copy of the checkpoint directory plus a fresh bind mount, serve command unchanged) was the one hypothesis that arithmetic had excluded (`../startup-failures/weight-load-stall-in-shard-loop.md`) |
| #3062 (v0.9.1, `llmdatadist` PD) | by turn 9: the 120 s warning's emitter, the three port formulas (the advertised port carries no `tp_rank`), `ThreadPoolExecutor(1)` and the `query_register_mem_status` loop — i.e. the source-verified statement that the pasted line is P-side bookkeeping, plus both TTFT discriminators | **no answer at all**: 12 turns / 22 tool calls, ~51 K characters of reasoning against 1.2 K of text, and that text was status notes ("Now the decisive part: …", "Two last code facts to pin: …"); cut inside the block composing the ranked answer (`../pd-disaggregation/llmdatadist-done-signal-and-ttft.md`) |
| #6529 (v0.14.0rc1, DS-V3.2 4-machine 1P1D, D node dies on the first curl) | by turn 5: the traceback shape (`_generate_dummy_run_hidden_states` → `acl_graph.__call__` → `npu.synchronize` → `507015` + the capture-mode `rtMemcpy` record) matches the MTP + captured-decode-graph family, and by turn 8 the pinned tutorial's D-node flag set (`FULL_DECODE_ONLY` + `deepseek_mtp` + `--async-scheduling`, and P nodes are the ones with `--enforce-eager`) | a ranked answer from turn 6, and answer-draft text on 11 of its 15 messages — the **only** run of its batch that shipped anything, precisely because of that habit; the cap still landed inside the turn-15 answer's evidence table, so the ladder and the scope block never shipped, and its own count read "turn 7" on that turn |
| #13199 (v0.23.0rc1 `-310p` x86, Qwen3-VL-8B, `module 'triton' has no attribute 'language'` at `import torch`) | by turn ~5: the image facts (**`Dockerfile.310p:45-63` uninstalls `triton` and `triton-ascend` on purpose**, `docs/.../hardwares/310p.md:7` "does not support `triton` or `triton-ascend`") plus the corpus's 310P-triton family (#7421, #7423) — i.e. a resolved `triton` on that box is uninvited, and the repair is removal | **no answer at all**: 12 assistant turns, message text ~1.9 K characters against four reasoning blocks of 9.1-15.3 K each (the 15.3 K block derives *why* a `triton` module can lack `language` — circular import, namespace package, partial uninstall — a property of the reporter's site-packages that no local file carries), last turn 106 characters of reasoning, cut at the 16 384-token cap. The answer owed was the verdict plus one `python -c` printing `triton.__file__`/`__path__`/`__version__` (`../image-dependency-faults/triton-module-residue.md`) |
| #6243 (v0.12.0rc1, DS-V3.2 A2 双机, `npu_kv_rmsnorm_rope_cache` `cos or sin shape is invalid`) | by turn ~12: the paste's own `[FUNC:GetShapeAttrsInfo]` puts the failure in attribute validation, so the title's "tiling 切分有误" is the wrong frame and the owner is the caller's rows; plus the pinned call sites, the MLAPO 1024-token branch and the DSA-CP slice | **no answer at all**: 15 turns, 50.7 K characters of reasoning against **67** characters of message text, the last turn a 14.4 K block holding the ranked answer and the repro script, cut inside it; the reporter's later comment (operator side returns it as a "调度问题") went unanswered (`../custom-ops-runtime-faults/tiling-attribute-rejection.md`) |
| #2807 (v0.10.1rc1 image, Qwen3-30B-A3B-W8A8, `-tp 4 --enable-expert-parallel`) | by turn ~12: the paste's `ops/common_fused_moe.py:99` matches the pin line-for-line (so the rejected call is gmm1, not the scale operand), `output_dtype=torch.int32 if use_int8_w8a8` is `:106`, `ep_size < 16` forces `allgathercommimpl` (`:487-490`) whose `permute` ignores `apply_a8_quantization`, and the only W8A8 e2e case is the eager TP2 one — a complete ranked answer with `--enforce-eager` as rung 1 | **no answer at all**: 16 assistant turns / 28 tool calls, **zero** characters of message text on every one of them, the whole 16 K output budget spent deciding what `quant_mode=-1` *means* in `npu_moe_init_routing_v2` — an op attribute no export in the case view carries (`rg -ln 'moe_init_routing' Ascend/` is empty), i.e. the #2754 ending again (`../custom-ops-runtime-faults/quant-scale-dtype-gate.md` rule 5) |
| #15394 (image `v0.23.0-a3`, A3 双机 PD, Qwen3.5-122B-A10B-w8a8-mtp, TP16/16) | by turn ~8: `:1627` is the **non-Mamba** `_get_kv_split_metadata` call (the MambaSpec sibling is `:1623-1625`), nine throw sites with their exception texts, and the `build_connector_meta` DEBUG line that prints exactly `local_block_ids` / `remote_block_ids` / `chunk_finish` — i.e. the whole deliverable except the paste's missing exception line | **one 509-character status note** (560 characters of text in all): 12 assistant turns / 16 tool calls, ~51 K characters of reasoning, the remainder of the budget spent deriving the `chunk_finish`/`to_trans_idx` arithmetic that the DEBUG line would have printed, cut before the ranked answer (`../pd-disaggregation/layerwise-split-metadata-crash.md`) |
| #15927 (v0.26.0rc1, GLM-5.3-w8a8c8 C8 PD) | by turn 8: the direct precedents (#12533 "关掉C8就可以正常推理", #12905 first-curl garbled) and the flag gating that makes `c8_enable_reshape_optim` the one-variable ablation; the pinned GLM-5.2 recipe diff by turn 13 | a ranked answer did reach the message, but only on the **17th** turn, and the cap landed inside its action ladder (rung 2 of 4) after ~39 K characters of reasoning against 3.8 K of text; turns 3-16 carried lead lines ("The key hit: … Let me read the gating code"), not the verdict (`../dsa-sparse-attention/c8-flags-and-kv-store-path.md`) |
| #15925 (v0.26.0rc1, GLM-5.3-w8a8c8 vs GLM-5.2-w4a8c8, ifbench-style strict/loose, `reasoning_effort: max`) | by turn 4: the reporter's own same-model row (GLM-5.2-w8a8 73.67/76.67 vs w4a8c8 73.67/76.5) is the control that retires the quant-width hypothesis, and the loose−strict spread tripling (9.66/8.83 vs 3.33/2.83) is the truncation-hypothesis sentence; the pinned recipe diff (`GLM5.2.md:211,261` vs `:505,579`), the 2-3 point run-to-run floor (#14378) and "GLM-5.3 is in no file at the pin" were one page away the whole time | the ranked answer stopped at **cause 1**: the cap landed mid-sentence (3 975 characters of text, 786 K cumulative prompt tokens) because the C8 page carrying those facts was opened around the tenth tool block, after the 38 KB accuracy index, two history searches, the release notes, the tutorial and the nightly e2e read. The route now starts from `../accuracy-benchmarks/quantized-sibling-score-gap.md` |
| #9278 (v0.19.1rc1, GLM-5-w4a8 DSA, `suffix` n=3 + `FULL_DECODE_ONLY` + `npugraph_ex`) | by turn 6: the frame `RejectionSampler.parse_output` `output_token_ids.cpu()` is a D2H **sync point**, the paste carries `reason=the current capture mode does not support this operation`, and the method is `suffix` at `n=3` — so the two checks owed were the sampler branch (`draft_probs is None` → no block verify, `vllm_ascend/sample/rejection_sampler.py:114`) and #7301's known `suffix` + `full_decode_only` conflict | **no answer at all**: 11 tool blocks, the full 16 384-token cap in reasoning, blocks spent counting turns against its own gates ("I'm at turn ~12 … I must write the answer now") and a long detour into `_build_attn_state`'s `SpecDecoding`→`ChunkedPrefill` split, which the MLA/DSA consumers at that ref treat alike; the titles-only history search that returns the real precedents was never re-run after the 207-hit body search (`../spec-decode-mtp/draft-graph-capture-device-fault.md`) |
| #6266 (v0.12.0rc1, GLM-4.7 1P1D) | by turn 6: the assert is upstream `_update_from_kv_xfer_finished` freeing blocks on a `finished_sending` id, and the pair that matters — `step_with_batch_queue` in the stack proves `--async-scheduling`, and the connector's `request_finished` delays the free only for `FINISHED_LENGTH_CAPPED` (`mooncake_connector.py:910-919`) | **no answer at all**: 11 assistant turns, the 16 384-token cap spent in 2-8 K-character reasoning blocks, a draft verdict line on most turns, then turns 6-11 spent enumerating eight further failure modes ("Failure mode A…H", a duplicate report, a pull-model send, a DP-shared tracker) instead of emitting the two-fact ranked answer; the cap landed inside the reasoning of the final turn, whose message text was empty (`../pd-disaggregation/prefill-scheduler-kv-xfer-assert.md`) |
| #5571 | by turn 6: the upstream `__init__` binds `mm_budget` on every path (`gpu_model_runner.py:565-573`) and the Ascend subclass does call `super().__init__()` (`model_runner_v1.py:191-193`) — the two facts that make "upstream renamed it" impossible | **no answer at all**: 12 turns / 21 tool calls, 226 characters of message text against ~48 K characters of reasoning, and the later turns re-measured those same two facts instead of reading the third line of the call site (`with _torch_cuda_wrapper():`, whose `except Exception` under the `yield` swallows the constructor's exception). A contradiction between two verified facts is a missing code path — and naming it is the verdict sentence (`../upstream-downstream-drift/swallowed-init-exception.md`) |
| #10522 (v0.20.2rc1, GLM-5.1 PD `PP4 TP8 EP32`, P-node `fault kernel_name=token_bin_counts_and_mask_kernel_0`) | by the 12th tool result: one `rg` over both trees resolves the kernel to `vllm_ascend/ops/triton/bincount.py:32`, which runs only when the request sets a penalty (`sample/sampler.py:55-56` returns early otherwise) — so the ownership verdict and the penalties-at-defaults ablation were both in hand | **no answer at all**: 38 messages / 16 384 output tokens, three message texts totalling 230 characters; the rest went upstream into `prompt_token_ids` construction and into decoding the plog's `args(0 to 9)` blob, which cannot name the operand. The reporter then confirmed a post-pin commit fixes it, so the mechanism was never in this tree — the deliverable was the ownership line plus the ask (`../custom-ops-runtime-faults/triton-jit-kernel-device-fault.md`) |
| #1138 (v0.8.5rc1, dense 32B on 4×910B, "AICore < 15 %") | by turn 3: the OCR'd metrics line is the **V0** logger (`Running: 6 reqs, Swapped: 0, Pending: 0, GPU KV 0.2 %, CPU KV 0.0 %`), so 57 tok/s ÷ 6 = 9.5 tok/s per request and nothing was queued — the reporter's own "what caps concurrency at 6" was already answered; by turn 5 the pin's `enforce_eager = True` (`platform.py:124`) with the torchair opt-in and its two silent gates | **no answer at all**: 12 assistant turns, ~45 K characters of reasoning (blocks of 5.4-8.5 K) against three status notes (79/212/217 characters), the full 16 384-token budget gone, cut inside a block re-deriving V0-vs-V1 selection (`../performance-throughput/pre-graph-mode-eager-era.md`) |
| #4948 (v0.11.0rc3, DeepSeek-R1-0528-BF16, TP8 PP4) | by turn 10: `dispose_tensor` → `x.set_` (`utils.py:504-505`) reached from `torchair_deepseek_v2.py:985-986` inside a `fullgraph=True` torchair compile (`torchair_model_runner.py:423,446`), the same signature in precedent #1615, the `PP + ray + graph` known-issue line, and `check_ascend_config`'s refusal to combine `--enforce-eager` with graph mode | **no answer at all**: 17 assistant turns, ~45 K characters of reasoning (one block 10.7 K) against a single 270-character verdict paragraph at its 14th turn; the cap landed while it planned `findings.md`. Never sent: the per-rank KV budget on the paste's own `Available memory: 5.08e9, total memory: 6.55e10` line, or the BF16-vs-W8A8 recipe question the reporter settled himself a week later (`../startup-failures/torchair-dynamo-compile-unsupported.md`) |
| #6578 (v0.14.0rc1, Qwen3-32B on 910B, `no attribute qkv_rmsnorm_rope`) | by turn 5: the op is registered only under `if HAS_TRITON:` (`ops/__init__.py:25-26`) while `fuse_qknorm_rope` defaults on with no such gate, and the pass's only gates are dtype + `head_size == 128` | the ranked answer did reach the message — but on its **11th** turn, as one 4.2 K-character block, and the cap landed inside its closing scope paragraph. Text on 9 of 11 turns (5.1 K against 19 tool calls); turns 6-10 pinned the doc line and the precedent titles, which was worth keeping, but the answer itself could have opened at turn 5 (`../startup-failures/missing-triton-op-in-compile-pass.md`) |
| #2315 (v0.9.2rc1-310p, 2×310P3, Qwen3-32B TP2) | by turn 14: the failure is inside torch_npu's ACL/GE bring-up (`SetPrecisionMode … ACL_PRECISION_MODE … 500001`, `PlatformInfoInit … SocVersion`), no `vllm_ascend` frame, so the driver/firmware × image-CANN pairing owns it | **no answer at all**: 18 assistant turns / 26 tool calls, **232 characters** of message text in total (an opener and one status note on turn 17), an empty final message; the budget went into `Ascend/pytorch` `LazyInitAclops.cpp` + the `SocVersion` enum and vllm-ascend's `is_310p()`. The mechanism frames are two reads, and the answer was the probe plus four version files (`../startup-failures/acl-init-500001-and-soc-version.md`) |
| #527 (v0.7.3rc1, `vllm serve` from a conda env, `cannot import name 'PoolingParams' from 'vllm' (unknown location)`) | at turn 2: the OCR'd traceback resolves `vllm` from `$HOME/hzy/vllm/vllm/…`, **not** `site-packages`, and the message's `(unknown location)` tail says a bare `vllm` directory was bound — one `python -c` prints the binding | **no answer at all**: 15 turns, ~70 characters of text on turn 1 and bare tool calls after it, the whole 16 384-token budget spent on the `+empty` version marker (`setup.py:382,506-507`), the docs' source-build block and `patch_module.py`'s `sys.modules` splice; the reporter deleted the `vllm` folder himself the next day (`../image-dependency-faults/wrong-vllm-module-shadowed.md`) |
| #10910 (v0.19.1rc1 + vllm v0.19.1, 2×16 910C, Qwen3-235B-A22B) | turn 1: the body is a tested-configuration matrix (`TP=32` works, single-node `PP=2,TP=8` works, `PP=4,TP=8` fails on `layers.47`/`layers.0`, `DP=2` fails on `NPU_group_*`), so two owners and one check each; by turn ~12 the PP fingerprint (`scripts/pp-layer-partition.sh` → stage starts 0/23/47/71) had confirmed the kv-config index | **no answer at all**: 45 messages, 966 K cumulative prompt tokens, the full 16 384-token cap in reasoning, cut inside the block that had just located `adjust_rank` — a second symptom that needed one log-line grep (`Creating placement groups for data parallel`) never got it either |
| #10370 (v0.20.2rc1, 2×910B, GLM-5.1-w4a8, 图模式/单算子) | by turn ~10: single-op working on the same cluster exonerates ranktable/HCCL/NICs, and the A2 selector arithmetic says the MC2 op is in the graph | **cut inside its own answer** at 14 043 output tokens: the verdict and the plog ask made it, the ranked list and the ladder did not. The later reporter comment (compile-cache ablation **没有效果**) shows the answer's direction was right and its length was the defect |
| #6082 (v0.10.0rc1-310p, 300I Duo, InternVL3-8B: `call aclnnIsNegInf failed, error code is 561000` + `EZ9999 … Op IsNegInf does not has any binary`, text prompts fine, image requests die) | by turn 8: the wording is a pre-launch **op-package** gap for the SoC (not a custom op, not a tiling fact), the multimodal path is the only code an image request adds (`intern_vit.py:275` SDPA, no mask), the image's CANN base is `Dockerfile.310p:18`, and no `isneginf` call site exists in any pinned tree | **no answer at all** — cut inside a 10th-turn tool call after ~5 K characters of reasoning per turn hunting the *caller* of `isneginf` across vllm / vllm-ascend / transformers / pytorch / op-plugin, i.e. an emitter in no export, twice (the same trap as #11454 and #4323). The deliverable was the standalone op probe plus `ASCEND_LAUNCH_BLOCKING=1` (`../soc-platform-paths/kernel-package-vs-card.md`) |
| #9324 (v0.19.1rc1, QwQ-32B TP4 on A3, `精度下降` 0.7500 vs 0.7843, `FULL_DECODE_ONLY` + `--reasoning-parser deepseek_r1`) | by turn ~6: `FULL_DECODE_ONLY` on the newer image also turns Npugraph_ex and three fusion passes on (`graph_mode.md:115`; `ascend_config.py:350-353`), so "same deploy script" is not the same numeric path, and the repeat-run noise floor is one launch | an answer did reach the message (~7 K characters over four text turns), **cut inside its rung 5** («不是 block-size»): 18 assistant turns, ~33 K characters of reasoning, turns 9-15 (two 2.3-2.8 K blocks among them) inside `AscendYaRNRotaryEmbedding`, whose one divergence the run itself then judged sub-token; the ladder, the version scope and the "what to send back" list never shipped (`../accuracy-benchmarks/harness-vs-engine-attribution.md`) |
| #11771 (v0.18.0 image, community-test row 15: Qwen3.5-27B, 8×A2, dp2 tp4, AIME + 2048/2048 bench) | by turn ~12: the page the 任务书 names does not exist at the pin, the page's serve line is DP1/TP2, the verification curl's `max_completion_tokens` is dropped, and every baseline is `v0.17.0rc1` gsm8k | **no message text at all** (409 characters across 17 turns): the report file got the deploy script and the curl/AISBench sections (6.3 K) and the cap landed on the turn that was about to write the 资料验证 defect table, so the table, the three scores and the FAQ — the graded 交付件 — never existed; every section had first been composed in reasoning (`../tutorial-verification/community-test-report.md`) |
| #5227 (v0.12.0rc1 image, Qwen3-32B, PP=2, `method: suffix`) | by turn 6 the answer had its decisive pair: the OCR carries no `vllm*/` frame (a `tensor_new.cpp:1578` tail proves the raising frame was lost), and the sketch was already "PP × spec-decode, not the suffix tree" | **one answer shipped, its correction did not**: the turn-13 answer reached the message, then the reporter's follow-up (the same crash with `ngram`) made a *new* deliverable owing — and the delta turn re-walked the executor copy, the proposer and the broadcast path until the 8 192-token cap; the revised ranking (PP>1's batch queue, `PP=1` as the only queue-removing ablation) existed only as reasoning (`../pipeline-parallel/spec-decode-first-request.md`) |
| #7947 (v0.18.0rc1, Qwen3.5-122B-A10B, TP8, `--language-model-only`) | the stack is recorded verbatim in this library — `call_size` → `qwen3_5.py` → `qwen3_next.py:1358` → `_model_forward`, with `--enforce-eager` clean — i.e. `first-generate-degenerate.md`'s Step 2 with the fix attribution already collected | **no answer at all**: ~16 K completion tokens over 15 turns, the whole budget spent re-deriving the mechanism from `patch_qwen3_5.py`, the 122B/A10B coverage gap and the ACL-graph stream budget; the routing grep that would have hit the recorded stack (`rg -n 'call_size' references/`) was never run, and the history-title searches it *did* run (`call_size`, `nonetype.*size`) return 0 because that corpus is titles-only (`../accuracy-benchmarks/first-generate-degenerate.md`) |
| #8370 (v0.18.0rc1 `99e1ea0f` + vllm v0.18.0, Gemma4 hybrid sliding+full, TP4, `FULL_DECODE_ONLY`: capture `AclmdlRICaptureEnd ... 107033 / task group status error` at `[1,2,4,8]`, replay `ValueError: not enough values to unpack (expected 13, got 9)` at `attention_v1.update_graph_params`) | by turn 31: capture appends a 9-field PA tuple (`attention_v1.py:657-670`) while replay selects the branch on `using_paged_attention` without the `sliding_window is None` term (`:397` vs `:920-926`), `pa_shape_list` default `[]`; by turn 37 the pinned vLLM registry has no `Gemma4ForConditionalGeneration` (`model_executor/models/` stops at `gemma3n`) | **no answer text at all**: 14 assistant turns, ~49.5 K characters of reasoning, and the only message text was next-action narration ("Now the decisive line...", "That `-r` cluster corrupted my grep...", "Decisive coverage gap found...") - the ranked answer and its ladder existed only as reasoning, and the page it now sits on (`../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md` section 9) was one write away |
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

A third, from #13002 and #6833 together: **a long source file is not a lead.** Both
runs chose one big module (a 2600-line KV connector; the hybrid-KV/prefix-cache
sources) and read it in slices, which is the most expensive way to spend an output
budget and the least likely to produce a sentence the reporter can act on. When the
next step would be "read more of this file", instead write the answer with the
open question and the one command that settles it — the config diff, the ablation
row, the `nm` line. Before that, notice when the evidence you already hold is the
deliverable: in #13002's first run the model tutorial's validated commands were read
at turn 17 of 21 and the ranked causes were in hand from that read onward, while the
replay that had the worked page never opened the connector for a walk at all.
#12432 is that failure again, one batch later and in the same file: 23 assistant
turns, the budget spent in slices of `mooncake_connector.py` and the recompute
scheduler, a "draft verdict" written into a message and cut mid-sentence, and no
answer ever shipped — while the decisive comparison (the tree's own recommended
proxy command versus the DP/`--api-server-count` layout of both serve commands) was
in the same tutorial the reporter had linked. #3162 is the same arithmetic in
another lane: its whole output budget went to `pyhccl.py` and recalled trl
internals without a stated verdict, although one precedent (#942's broadcast that
returned zeros) plus one question (which patch supplies the communicator) was the
deliverable from the fourth turn. Two sessions, one conclusion: the paste and the
tree's own recipe decide these reports, and reading further into a module does not.
