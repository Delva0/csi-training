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
- **Every assistant turn carries message text, not only a tool call.** A gate only
  binds if the text exists to be cut. #5295 is the clean case: 16 assistant turns,
  every one of them a bare tool call, ~46 K characters of reasoning (≈12 K of a 16 K
  output budget), three explicit notes that it "must produce the answer NOW", and a
  cut inside a reasoning block while opening the one file that would have refuted
  its own last hypothesis. Carry one or two sentences of the current answer draft in
  the same message as every tool call — verdict line first, then what is being
  checked — so any truncation leaves a usable answer.
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
  turn four, #13853 at turn ten). A block of 2-16 K characters is not a
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
  already *is* the full answer.
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
  is already too late to be safe — write the smaller section now.
- **Verifying what a page already verified is not evidence, it is spent budget.** A
  reference page records the pins it was checked at. When `SOURCES.json` carries
  those same refs, cite its `path:line` and re-grep only the claims it marks as
  version-specific. #6833's replay re-walked the whole prefix-cache gate chain at
  identical pins (config hook, `MambaModelConfig`, the model's assert, the
  downstream patch, the loggers line) — ten turns that bought nothing, and its
  action ladder was the part that got cut.
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
| #11454 | at turn 2: the literal question (`what does hccl_result = 4 mean?`) is the `HcclResult` enum, and the two red herrings were already refutable from the paste — `error_codes=[-1]` is not the out-of-space code `-200`, and `"Check memory and store capacity"` is canned text at `mooncake_backend.py:194` | **no answer**: 22 turns, every message a bare tool call, the full 16K output budget inside reasoning blocks, three or four whole-`/workspace/repos` greps hunting an emitter that is in no export, cut inside a reasoning block (`../kv-pool-store/hccl-onesided-store-put.md`) |
| #5295 | by turn 10: the selector predicate (`world_size_across_dp/pp >= 16` on A2), the correction that MC2 inherits the All2All `finalize`, and the capacity arithmetic from `cudagraph_capture_sizes: [4]` | **no answer**: 16 turns, every message a tool call, ~46 K characters of reasoning, cut inside a block that was opening `ascend_forward_context.py` to check `padded_num_tokens` — the check that refutes the theory it was about to state (`../feature-enable-recipes/moe-expert-parallel-with-dp.md`) |
| #15815 | the whole chain (stale factory alias, `OOT, None`, masked property) by turn 12 and the traceback lines matching the pin by turn 3 | complete ranked answer — but only in its **15th** turn, after 50 K characters of output of which 31 K were reasoning (14,553 of a 16,384-token cap). It shipped because 1.8 K tokens were left; nothing about the session was shaped to survive a cut |
| #8724 | the frame line and the three-traceback signature by turn 3; the decisive patch-vs-upstream diff was the one step that never ran | **no answer**: 29 turns, every message a tool call, ~48.5 K characters of reasoning against 73 characters of text, cut inside a reasoning block on the turn it finally started the diff (`../upstream-downstream-drift/patch-copy-vs-upstream.md`) |
| #17991 | the full source-verified diagnosis (`fa3_v1.py:132-136`, request-indexed `block_tables` sliced by a token count) written in the reasoning at its 17th turn | **no answer**: ~51 K characters of reasoning against 255 characters of message text; the 186-character message on that turn was `Confirmed the fault mechanism. Let me pin two last facts`, and the cap landed on the next tool call (`../mixed-batch-splits/fa3-mixed-batch-split.md`) |
| #11759 | each section as it became true: the absent benchmark section (turn 6), the ignored `max_completion_tokens` (turn 8), the one-card sizing arithmetic (turn 11), the `transformers==4.51.0` pin conflict (turn 14) | **no file, no message text**: 16 turns, 46.5 K characters (29.8 K reasoning, 69 characters of message text), the report heredoc written on the last turn and discarded — a 5.7 K-character composition after a 3.5 K-character script the turn before |
| #16111 | by turn ~12: the lazy init is documented (#9975), the config-table diff (`dram.size` per die, the non-existent `standalone.dram.size`) and the #14106 swallowed-save → collective-desync family | **no answer at all**: 36 turns, every message a bare tool call, cut inside a reasoning block that was composing the deliverable ("Verified facts (pinned …)") — the answer existed only as reasoning |
| #8788 | by turn ~18: the invocation chain and the four `csrc/third_party` fetches plus the catlass submodule | **no answer at all**: 30 turns, every message a bare tool call, cut inside a block that had already begun "**Answer**: Yes, real…" |
| #16563 / A | by turn ~14: the counter is cgroup host memory, the +80 GB/7 d arithmetic (3.5-7 KB per request, <16 B per decode step, so per-step host buffers are priced out), the closest precedents (#2650 host memory 1-2 GB/h, #12345 DeepSeek-V4 leak, #5847 CPU-memory leak) and the A2 recipe diff (`--no-enable-prefix-caching`, `additional_config.enable_flashcomm1`) | **no answer at all**: 20 turns, a 69-character opener then bare tool calls, ~48 K characters of reasoning, cut inside a reasoning block (`../runtime-memory/container-working-set-growth.md`) |
| #12261 | by turn 11: the parser's `_is_string_type` gate on a nullable `anyOf: [string, null]` argument, the raw branch, the length-prefix delta and the arithmetic that reproduces the paste | **no answer at all**: 15 turns / 23 tool calls, ~35 K characters of reasoning against 3.5 K of message text, a one-line verdict on most turns, cut on a tool call while trying to `import vllm` (no torch in the sandbox) to simulate two functions it had already read |
| #2015 | by turn 11: the pinned `quantization.md:37/:5/:94` contract (the W8A8 export carries no `quantization_config`, so the flag is mandatory) and the `enforce_eager` exclusion of the ACL-graph cause | **no answer at all**: 17 turns / 25 tool calls, ~45 K characters of reasoning against 4.0 K of text, cut on a tool call |
| #16563 / B | by turn ~9: the same class and arithmetic, the env-flag liveness findings (no consumer at the pin for `VLLM_ASCEND_ENABLE_FLASHCOMM` / `PYTHON_NPU_ALLOC_CONF`), the MTP `pin_memory()` asymmetry and a `findings.md` draft | verdict, five causes, four precedents and the six-rung ladder did reach the message, but the cap landed inside the closing version-scope section: five causes with sub-paragraphs plus a precedent table plus a ladder is more answer than the budget holds |

Two consequences worth restating: an "I'll start by…" line is not an answer and
does not satisfy G1, and a run whose last turn is a tool call has shipped nothing.

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
deliverable: at #13002 the model tutorial's validated commands were read at turn 17
of 21, and the ranked causes were in hand from that read onward.
