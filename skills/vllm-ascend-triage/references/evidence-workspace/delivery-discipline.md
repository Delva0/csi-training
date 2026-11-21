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
  #11936 is the same shape at the cap: 28 assistant turns, **zero** message text,
  16 384 completion tokens all inside reasoning blocks, cut mid-thought while
  reading the 310P attention backend — although it had named a ranked candidate
  (the 310P rope branches) by its middle turns, which the platform
  topic now reduces to one probe
  (`../soc-platform-paths/310p-vl-ocr-garbage-output.md`).
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
| #13002 / replay | the page's own P/D diff table and its two report-side traps, available from the first skill read | complete ranked answer at its 11th assistant turn (17 tool calls): the three symptom shapes separated, the five-row flag diff, the ladder, and the connector opened once for one `dp_size`/`tp_size` grep — the same paste the first run spent 21 turns and its whole budget failing on |
| #11454 | at turn 2: the literal question (`what does hccl_result = 4 mean?`) is the `HcclResult` enum, and the two red herrings were already refutable from the paste — `error_codes=[-1]` is not the out-of-space code `-200`, and `"Check memory and store capacity"` is canned text at `mooncake_backend.py:194` | **no answer**: 22 turns, every message a bare tool call, the full 16K output budget inside reasoning blocks, three or four whole-`/workspace/repos` greps hunting an emitter that is in no export, cut inside a reasoning block (`../kv-pool-store/hccl-onesided-store-put.md`) |
| #5295 | by turn 10: the selector predicate (`world_size_across_dp/pp >= 16` on A2), the correction that MC2 inherits the All2All `finalize`, and the capacity arithmetic from `cudagraph_capture_sizes: [4]` | **no answer**: 16 turns, every message a tool call, ~46 K characters of reasoning, cut inside a block that was opening `ascend_forward_context.py` to check `padded_num_tokens` — the check that refutes the theory it was about to state (`../feature-enable-recipes/moe-expert-parallel-with-dp.md`) |
| #15815 | the whole chain (stale factory alias, `OOT, None`, masked property) by turn 12 and the traceback lines matching the pin by turn 3 | complete ranked answer — but only in its **15th** turn, after 50 K characters of output of which 31 K were reasoning (14,553 of a 16,384-token cap). It shipped because 1.8 K tokens were left; nothing about the session was shaped to survive a cut |
| #8724 | the frame line and the three-traceback signature by turn 3; the decisive patch-vs-upstream diff was the one step that never ran | **no answer**: 29 turns, every message a tool call, ~48.5 K characters of reasoning against 73 characters of text, cut inside a reasoning block on the turn it finally started the diff (`../upstream-downstream-drift/patch-copy-vs-upstream.md`) |
| #17991 | the full source-verified diagnosis (`fa3_v1.py:132-136`, request-indexed `block_tables` sliced by a token count) written in the reasoning at its 17th turn | **no answer**: ~51 K characters of reasoning against 255 characters of message text; the 186-character message on that turn was `Confirmed the fault mechanism. Let me pin two last facts`, and the cap landed on the next tool call (`../mixed-batch-splits/fa3-mixed-batch-split.md`) |
| #6598 | the two numbers from the screenshot OCR (a padded graph bucket vs the real `actualSequenceLengthQ`) plus both candidate padding owners and the tutorial diff, by turn ~8 | **no answer**: 11 turns, ~35 K characters of reasoning against 1.2 K of status-line text, cut at the 16 K output cap inside a block that had just written "I must emit the answer NOW" (`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`) |
| #3098 | the model-specific split (the same ray cluster serves the sibling) and the release note's first-support statement | a ranked answer did reach the message, but the cap landed inside its own ladder (rung 3 of 6): the closing "what to send back" list never shipped, which is the block the answer-size rule above says keeps 2-3 lines |
| #11759 | each section as it became true: the absent benchmark section (turn 6), the ignored `max_completion_tokens` (turn 8), the one-card sizing arithmetic (turn 11), the `transformers==4.51.0` pin conflict (turn 14) | **no file, no message text**: 16 turns, 46.5 K characters (29.8 K reasoning, 69 characters of message text), the report heredoc written on the last turn and discarded — a 5.7 K-character composition after a 3.5 K-character script the turn before |
| #16111 | by turn ~12: the lazy init is documented (#9975), the config-table diff (`dram.size` per die, the non-existent `standalone.dram.size`) and the #14106 swallowed-save → collective-desync family | **no answer at all**: 36 turns, every message a bare tool call, cut inside a reasoning block that was composing the deliverable ("Verified facts (pinned …)") — the answer existed only as reasoning |
| #8788 | by turn ~18: the invocation chain and the four `csrc/third_party` fetches plus the catlass submodule | **no answer at all**: 30 turns, every message a bare tool call, cut inside a block that had already begun "**Answer**: Yes, real…" |
| #16563 / A | by turn ~14: the counter is cgroup host memory, the +80 GB/7 d arithmetic (3.5-7 KB per request, <16 B per decode step, so per-step host buffers are priced out), the closest precedents (#2650 host memory 1-2 GB/h, #12345 DeepSeek-V4 leak, #5847 CPU-memory leak) and the A2 recipe diff (`--no-enable-prefix-caching`, `additional_config.enable_flashcomm1`) | **no answer at all**: 20 turns, a 69-character opener then bare tool calls, ~48 K characters of reasoning, cut inside a reasoning block (`../runtime-memory/container-working-set-growth.md`) |
| #12261 | by turn 11: the parser's `_is_string_type` gate on a nullable `anyOf: [string, null]` argument, the raw branch, the length-prefix delta and the arithmetic that reproduces the paste | **no answer at all**: 15 turns / 23 tool calls, ~35 K characters of reasoning against 3.5 K of message text, a one-line verdict on most turns, cut on a tool call while trying to `import vllm` (no torch in the sandbox) to simulate two functions it had already read |
| #14656 | by turn 7: the page's schema-type gate is **fixed** at v0.23.0 (`_is_string_type` now `extract_types_from_schema`), so the GLM-5.2 paste needed another owner, and by turn 12 two verified candidates plus the downstream patches excluded | **no answer at all**: 15 assistant turns / ~16 K completion tokens of reasoning (blocks of 5-10 K characters) against ~2.1 K of message text, two final turns with empty text, cut inside a reasoning block; the streamed-text owner (`resetting decode stream`) was named as a hypothesis and never grepped (`../tool-call-streaming/detokenizer-text-delta.md`) |
| #2015 | by turn 11: the pinned `quantization.md:37/:5/:94` contract (the W8A8 export carries no `quantization_config`, so the flag is mandatory) and the `enforce_eager` exclusion of the ACL-graph cause | **no answer at all**: 17 turns / 25 tool calls, ~45 K characters of reasoning against 4.0 K of text, cut on a tool call |
| #1796 | by turn ~6: the wheel's `Requires-Dist` pin is `torch-npu==2.5.1.post1.dev20250619` while `setup.py:341` drops the `--extra-index-url`/`--pre` lines from the metadata, i.e. a mirror-only build made unfindable — and `torchvision<0.21.0` puts the same requirement set on the 2.5.x torch line against vllm 0.9.2's `torch==2.7.0` | **no answer at all**: 39 tool calls, ~16 K completion tokens of reasoning, every assistant message a bare tool call; the budget went to whether that dev wheel was rebased onto torch 2.7.1, a fact no local file can settle (`../image-dependency-faults/pip-index-visibility.md`) |
| #3107 | by turn 3: the paste names the two objects whose roles differ (`attn_metadata.max_tokens_across_dp` → `get_forward_context().max_tokens_across_dp`), and by turn 5-6 the pin's writer (`ascend_forward_context.py:91-98`), the three consumers and the `dp_size × max_tokens = 544` arithmetic against the pasted `[544, 5120]` | **no answer**: 12 turns, 7 with no message text, ~49.6 K characters of reasoning against 661 characters of text, 22 tool calls, cut on the last turn's reasoning block; the `pangu_moe.py`-absent-from-the-pin finding (turn 4) was the one sentence the reporter needed (`../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`, `../startup-failures/environment-correspondence.md`) |
| #698 | by turn 3: the phase after `Loading model weights took` (`worker.py:240-256` → `profile_run` → `npu.synchronize`), and by turn 6 the V0/V1 fingerprint from the log's own `model_runner.py:904` | **no answer**: 14 turns, 11 of them bare tool calls, ~47.6 K characters of reasoning against 322 characters of text; the answer's first paragraph was the artefact request (the full `vllm serve` line — the maintainer's own first reply asked for exactly that, and the reporter supplied it the next day) (`../startup-failures/silent-stall-after-weight-load.md`) |
| #2754 | by turn ~6: the failing phase (torchair graph load), the all-peer `DestLink` block refuting "one bad edge", and the KV-before-capture ordering | **no answer at all** — 12 turns, 7 with no message text, the whole 16 K output budget in reasoning, cut inside the answer's opening sentence; the budget went to the `EI0007` QP-reservation mechanism, whose sources (`cann/hccl`, `cann/hcomm`, `cann/ge`) the case's own `SOURCES.json` lists as `unavailable`, so no local file could settle it and only the budget ladder was owed (`../hccl-and-hardware/qp-resource-exhausted-at-graph-load.md`) |
| #16600 | by turn 9: the model's support row sits only in the `Atlas 300I DUO` tab (so "only 310P has quantized weights" is the project's SoC scope, not a gap), the same-SoC e2e recipe and the two `--additional-config` keys no code reads — a complete capability verdict for a "速度很慢 + 能否提供量化版本" report | **no answer**: the verdict sat in a mid-session message, then ~15 turns went into `platform.py` capture-mode gates, the hybrid-KV gate and issue bodies; the cap landed inside the block that had just started the ranked list (`#1 没开 MTP…`). A perf claim with no metric block is answered by the capability verdict plus the request for the metrics, not by a gate chain (`../feature-enable-recipes/per-soc-recipe-and-quantized-weights.md`) |
| #16563 / B | by turn ~9: the same class and arithmetic, the env-flag liveness findings (no consumer at the pin for `VLLM_ASCEND_ENABLE_FLASHCOMM` / `PYTHON_NPU_ALLOC_CONF`), the MTP `pin_memory()` asymmetry and a `findings.md` draft | verdict, five causes, four precedents and the six-rung ladder did reach the message, but the cap landed inside the closing version-scope section: five causes with sub-paragraphs plus a precedent table plus a ladder is more answer than the budget holds |
| #894 | by turn 2: a 4-token paste is a degeneration report with no serve command, no log and no env, so a ranked matrix *is* the answer — and the pin's forced `enforce_eager = True` (`platform.py:124`) plus the `-Chat`-only validated row were one grep each | **no answer at all**: 20 turns / 35 tool calls, 69 characters of message text against ~43 K characters of reasoning, the full 16 K output budget spent on `deepseek_v2.py` / `fused_moe.py` / `platform.py` (would this branch break V2-Lite?), cut inside a reasoning block (`../accuracy-benchmarks/deepseek-v2-lite-mla-degeneracy.md`) |
| #4323 | by turn ~12: the ATB block is a real `atb::OperationSetup` failure inside `libop_plugin_atb.so`, so the pasted `eagle_proposer.py:476` frame is a join point (precedent #447 carries the same wording over `_npu_reshape_and_cache`); `draft_tensor_parallel_size` has no consumer under `vllm/v1/**` at the pin, and the eagle3 e2e there is single-card Llama-3.1 drafts only | **no answer**: 26 turns, **zero** message text on every one of them, the 16 384-token cap reached inside a reasoning block that already held the ranked causes and the ladder; the closing turns asked which `torch_npu` op reaches ATB — a fact no export in the case contains (`../spec-decode-mtp/atb-op-setup-failure-at-proposer.md`) |

| #3062 (v0.9.1, `llmdatadist` PD) | by turn 9: the 120 s warning's emitter, the three port formulas (the advertised port carries no `tp_rank`), `ThreadPoolExecutor(1)` and the `query_register_mem_status` loop — i.e. the source-verified statement that the pasted line is P-side bookkeeping, plus both TTFT discriminators | **no answer at all**: 12 turns / 22 tool calls, ~51 K characters of reasoning against 1.2 K of text, and that text was status notes ("Now the decisive part: …", "Two last code facts to pin: …"); cut inside the block composing the ranked answer (`../pd-disaggregation/llmdatadist-done-signal-and-ttft.md`) |
| #15927 (v0.26.0rc1, GLM-5.3-w8a8c8 C8 PD) | by turn 8: the direct precedents (#12533 "关掉C8就可以正常推理", #12905 first-curl garbled) and the flag gating that makes `c8_enable_reshape_optim` the one-variable ablation; the pinned GLM-5.2 recipe diff by turn 13 | a ranked answer did reach the message, but only on the **17th** turn, and the cap landed inside its action ladder (rung 2 of 4) after ~39 K characters of reasoning against 3.8 K of text; turns 3-16 carried lead lines ("The key hit: … Let me read the gating code"), not the verdict (`../dsa-sparse-attention/c8-flags-and-kv-store-path.md`) |

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
