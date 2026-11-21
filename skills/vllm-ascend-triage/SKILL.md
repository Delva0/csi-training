---
name: vllm-ascend-triage
description: 'Triage vLLM-Ascend issue reports; load alone first'
---

# vLLM-Ascend issue, deployment and support triage

## Before the first shell call

**Your first tool block holds exactly one call: `skill_view` on this skill, by name**
(`skill_view(name='vllm-ascend-triage')`; a **filtered** `skills_list` returns `[]` even
though the skill is installed, while the unfiltered call lists it — so an empty list is
a tool quirk, not "no skill for this task"). Anything batched beside that call — `ls /workspace`, `ls
/workspace/history`, `cat SOURCES.json` — executes before this body is in context, so no
rule here can stop it, and its output is re-sent on every later turn: five recorded runs
paid 39 KB / 14 KB (both forwards of one minibatch, #16365 among them), 54 KB / 28 KB
(#14731, #12102), and 37 KB of directory names from `ls /workspace/history` batched beside
this call in #16581 — which ended output-truncated at 1.1 M cumulative prompt tokens. That
is why the frontmatter description says "load alone first": the description is the only
text the agent reads *before* this body. If something has to share it, let it be
`bash <skill_root>/scripts/sources-summary.sh` (~10 lines). Then orient with
`search_files (target='files')` on one level of
`/workspace` plus that same summary and nothing
else: never `cat /workspace/repos/SOURCES.json`, never enumerate a history directory,
never `cat /workspace/history/*.md`. Search by token (`rg -l`,
`scripts/history-titles.sh`), never by listing. **Attachments are the one exception**:
open the specific files the issue links (`ATTACHMENTS.txt` lists names and SHA256s
only, so a grep of it is not an existence test) — rule 7. Costs and traps: rule 7.

## Problem class this covers

Ten entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter. The last shapes deliver a *decision* (sizing),
a *document review*, a *capability verdict* or a *change plan* instead of a root
cause, but the same rules apply — answer first, evidence pinned to the reported
version, gaps stated.

- **Issue report** (template-shaped: environment dump + logs + narrative) about
  something that went wrong on an Ascend NPU host — a device-side operator crash, an
  accuracy/score gap, garbled or truncated output, a long-run OOM, a container metric
  that climbs for days, a performance claim, a flag or endpoint that does nothing, or
  a `python -c "import <pkg>"` that aborts in a shipped library. When the task is only
  "explain/answer this issue", the answer is still evidentiary, not a summary.
- **Install / pip-resolution report**: no engine, no worker, no NPU traceback — two
  `pip install` lines and what they printed (`Could not find a version that
  satisfies the requirement …`, `No matching distribution found for <pkg>`,
  "unmatched torch version between vllm=X and vllm-ascend=Y", an `ImportError:
  Please install vllm[bench] …` inside a shipped image), or an install that finished
  and only **warned** (`requires torch==2.10.0`). The deliverable is *which* artefact
  pip could not see — a mirror-only pin the metadata cannot carry, an unsatisfiable
  torch/torch-npu/torchvision set, an air-gapped container with no reachable index —
  never a release survey — `references/image-dependency-faults/pip-index-visibility.md`
  and `pip-resolver-warning-block.md`.
- **Deployment failure**: engine start or a running worker fails — `EngineCore failed
  to start`, `WorkerProc initialization failed`, silent worker death, allreduce /
  `ERR01100` / `ERR00100`, OOM, "the model failed to load", an `AttributeError` /
  `TypeError` from a patched forward or a KV connector, or a `ValueError` from
  upstream `arg_utils` rejecting the parallel/DP flags.
- **Support / version question**: "does vllm-ascend <version> deploy <model>?" — a
  version gate with a one-line answer and a registry-level proof, not a bug hunt.
  When the follow-up is "支持计划是什么样的 / when will F work for M on my SoC", answer
  in three separate levels — the base model family at that SoC (release-note line,
  support-matrix row, the SoC fallback `TODO` that bounds it), whether the *feature's*
  code branch exists (one grep), and whether anything covers the combination (tests,
  tutorials) — then the prerequisites and the grep whose result would change the answer,
  and no date (`references/feature-enable-recipes/index.md` item 9).
- **Feature-enable / capability question**: "如何正确开启 <feature>", "F 单独开均有不同的
  报错，能否给出能跑通的示例", "does F work on model M on this SoC", usually with no log
  and two errors in prose. The deliverable is the validation verdict at the reported
  ref plus the constraints the code enforces, not a launcher assembled from them —
  `references/feature-enable-recipes/index.md`.
- **PD / EPD disaggregation**: the deployment is up but P and D cannot move KV
  (`HIXL Connect failed | 103900`, HCCL `RanktableCheck` / `'device_ip' is not set
  correctly`, `wait_for_save failed`, `Failed to put N keys out of N`, a recipe that
  needs `hostNetwork: true`), or KV moves and the decode node emits wrong/alternating
  output on a hybrid linear-attention model (GDN/Mamba — but a **DSA/MLA** model such
  as GLM-5.2 that repeats one phrase to the token cap is a recipe-contract question
  first: `references/pd-disaggregation/recipe-contract-diff.md`), or a KV pool is attached and the *first*
  transfer fails (`503900` / `HcclCommPrepare ret: 0x13`, the pool's
  `global_segment_size` the cause), or the put never lands (`HcclBatchPut,
  hccl_result = 4`, `PutEnd` 0), or the decoder never receives the request
  (`proxy通信超时`) — an endpoint-count question before any connector read; transport
  identity and rank tables are a topology question, a pool deployment a config
  question — `references/pd-disaggregation/index.md`.
  If instead the engines start and the **first request** kills one side's EngineCore
  behind a bare `EngineDeadError: … See stack trace (above)` on a deployment setting
  `--decode-context-parallel-size` > 1: DCP is a *matched* parameter (a 1M recipe's
  `8` reduced on one side fails the connector's divisibility assertion), so compare
  both serve commands before any source walk —
  `references/feature-enable-recipes/context-parallel-pcp-dcp.md` §3.
  If instead requests *are* served and TTFT is slow while the prefiller prints
  `Some requests in prefill node fail to receive KV Cache transfer done signal in 120s`
  (`VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`, default 120): that line is expiry bookkeeping
  emitted by design on the non-zero P `tp_rank`s — the advertised RPC port carries no
  `tp_rank` — so it is not the slow-TTFT mechanism. Settle TTFT with the concurrency and
  first-request discriminators before any connector or rank-table read —
  `references/pd-disaggregation/llmdatadist-done-signal-and-ttft.md`.
  If instead the **decode node logs `Error in KVCacheTransferThread. error=unhashable
  type: 'list'.`** and the deployment keeps serving: that line is the recv thread's catch-all
  printed *in place of* the real failure — downstream `_mark_failed_recv_request` feeds
  `set[int].update(<list[list[int]]>)` a grouped `BlockIds` before the `logger.exception` that
  would have named it — so the paste carries no transport information; unmask (flatten, or the
  layerwise connector, which already flattens) before any HCCL, plog or rank-table read
  (`references/pd-disaggregation/kv-transfer-thread-masked-error.md`).
  If instead **one side's EngineCore dies on its own bookkeeping** and the other keeps
  serving, name the side and the owning structure from the last three frames before any
  transport read: a prefiller bare `AssertionError` in upstream
  `v1/core/sched/scheduler.py` `_update_from_kv_xfer_finished`
  (`assert req_id in self.requests`) is a send-completion for a request the scheduler
  already freed, and `step_with_batch_queue` in the same stack proves
  `--async-scheduling` (or PP>1) — one launch-flag ablation, no connector walk
  (`references/pd-disaggregation/prefill-scheduler-kv-xfer-assert.md`); a decode
  `assert … chosen_rank_list(…) does not match num_group_pulls(…)` on P TP≠D TP with an
  MLA/DSA model is one pull-count derivation skipping the MLA single-pull rule
  (`references/pd-disaggregation/asymmetric-tp-pull-count.md`).
- **Deployment sizing / stage layout**: the question is a plan — "does this model
  fit on 2×64 GB", "the GPU docs do a 2-device stage deploy and mine does not come up
  on 2 NPUs", a bundled `platforms.<vendor>` yaml with more stages than the reporter
  has cards, "how do I pin stage 1 to one card" — **or** two serve commands that
  differ by one resource knob (`--tensor-parallel-size`, `--max-model-len`,
  `--gpu-memory-utilization`) where the smaller one comes up and the larger one OOMs
  at engine start. Answer with per-card arithmetic plus the device-visibility
  variable — `references/deploy-sizing/index.md`.
- **Community-test / doc-verification task** (`[Community Test]`, a 众测任务书 row
  with 蓝区镜像 / 权重链接 / 卡数 / 交付件 测试报告, or a `[Doc]` report that a
  tutorial step does not work as written): the deliverable is a *report about a
  document*, graded on completeness/usability/correctness without the reporter's
  hardware. The evidence is the pinned tree — which command no longer parses, which
  path is absent from the image, which baseline predates the image, whether the card
  count matches the weights — `references/tutorial-verification/index.md`.
  When the reported defect is an **absence** instead ("the Quick Start should
  mention a mirror / link the FAQ", "the contributor docs never state a
  prerequisite"), three greps decide it — does the content exist, does any page
  reach it, do the sibling pages already carry it — and the reporter's own URL
  anchor is checked against the pin before it is quoted
  (`references/tutorial-verification/missing-guidance-and-contributor-docs.md`).
- **Release known-issues bulletin**: a maintainer's notice for one release —
  `[Bug][vX.Y.Zrc1]: Known Issues for <model A>/<model B> service hang on A3`, an
  impact table (version / models / hardware / scenario), one or two tracked issue
  ids with error codes, "reply in the comments instead of opening a new issue" —
  or a user quoting that notice and asking whether their run is the same thing.
  No log, and none is owed: the deliverable is a **claim-check at the pinned
  release** — which signatures the notice pins and what they localize, what they
  cannot settle (mechanism, event ordering), whether each code is even
  model-specific, whether the reported config is inside the release's validated
  envelope — plus one ranked list per tracked issue and the reply template
  (plog `[AIC_INFO] fault kernel_name=` block, both serve commands).
  `references/release-known-issues/index.md`.
- **Code-change / cleanup request**: the body asks for an *edit*, not a diagnosis —
  "remove `VLLM_ASCEND_ENABLE_<X>` before <release> final release", "delete the
  temporary switch added in PR #N", "deprecate this flag", often with nothing but two
  links and no log. First question: does the named identifier exist at the pinned ref
  (absence dates the snapshot against the PR and is itself the finding); deliverable:
  the removal sweep with `path:line` anchors plus the behaviour change a default flip
  implies — `references/code-change-requests/index.md`, then `env-flag-retirement.md`.
  The delivery gates apply here too: no crash to report, so the absence verdict goes in
  the first message (recorded #1262 replay: first message text on turn 15).

Recorded cases, per-issue index and how each session ended:
`references/evidence-workspace/recorded-cases.md`; the load-bearing
delivery pattern is rule 1.

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and obey the delivery gates.** A long
   session can be cut off mid-sentence, so the deliverable has to survive
   truncation: what decides the outcome is not how much evidence was gathered but
   whether the message exists. The runs that shipped **no answer at all** each had
   a usable lead in hand (#6784, #13853, #11454, #11095, #17991, #3107, #6266, #5571, #527,
   #587, #11680, #1579, #11771, #7947, #10522 — the last with the plog naming the kernel
   and the owner one grep away; #2807 and #15394 held every verified fact of a complete
   ranked answer and spent the remainder on a host detail no local file settles, i.e. the
   `quant_mode`/`chunk_finish` arithmetic their own pages now replace with one print and one
   DEBUG line; per-run table `delivery-discipline.md`), while #4296
   answered in ten turns from one traceback.
   Hard stops:
   - **G1 — 3rd assistant turn:** message text with the direct answer to the
     reporter's literal question plus the track/lane classification. A pasted
     traceback, one config value or one registry listing is enough; do not hold
     the answer for the mechanism.
   - **G2 — 8th assistant turn:** the full ranked answer (each cause with its one
     check, cheapest action, version scope, missing artefacts). Afterwards a tool
     call is allowed only to confirm or correct a named claim: one per turn, a
     couple of lines of reasoning.
   - **G3 — 12th assistant turn:** answer text only; no reasoning, no tool call,
     no "one more thing".
   - **The gates cap cumulative output (~45-50 K characters of reasoning+text), and
     the turn index is the weak part: your own count drifts 2-3x low.** #6529 wrote
     "I'm at turn 7" on its 15th turn, #6243 "I'm at 6" on its 14th, so G1-G3 never
     fired as written. Trigger on evidence instead of the number: the turn you can
     state a cause with its one check is the turn the answer is owed; if you do
     count, count the `tool` results in the transcript, not from memory. A verdict
     line is a *hypothesis*: the ranked deliverable is still owed as one message, and
     a block that restates the verdict, numbers the fixes or lays out the evidence
     table *is* this turn's message: send it, then one tool call
     (`references/evidence-workspace/delivery-discipline.md`).
   - **Reasoning is a scratchpad, a few lines per turn:** what the last result
     proved, the single next check, go. It spends the **same output budget** as the
     answer, before any of it becomes text (ids in `delivery-discipline.md`). If a
     block enumerates the answer's sections, counts turns against G1/G2 ("this is
     turn 5, G2 is turn 8"), or says "I have enough" / "let me write the answer",
     that block *is* the answer — and a turn that reasons "the gate has passed, I
     must send the deliverable" while sending empty text and another tool call is
     the recorded way to ship nothing (#6243: 15 turns, 67 characters of text, cut
     inside a 14 K character block that held the whole ranked answer). The
     #14731/#12102 pair shows the arithmetic: 42.9 K and 39.4 K characters of
     reasoning against 1.3 K and 2.7 K characters of delivered answer, both cut at
     the 16 384-token cap — the bigger run spent an 11.8 K character block deriving
     a parser mid-session and then re-wrote the whole ranked answer in a 6.2 K block
     before typing it.
   - **Budget in characters per block, not turns** — the count-free form of the rule
     above. #13527 hit the cap with no draft at all: 16 turns, 26 `tool` results,
     **43.8 K characters of reasoning against 2.2 K of message text**, nine of its turns
     opening with a self-dated count of its own turns ("I'm at turn 5", "turn 8",
     "turn 10") that was 2-3 low and closing with two more tool calls. #587 spent 52.6 K
     characters in **7** assistant turns and shipped 800 characters of text, #11680
     51.2 K in 10 turns and 2.8 K: G1/G2 never fired because the turn index was low
     while single blocks ran 8-12 K characters. So test the characters, not the count:
     **message text ≥ a third of the reasoning you have written so far**, and keep each
     reasoning block short (~1.5-4 K characters) — what the last result proved, the one
     next check, and this turn's answer section typed rather than "one more check".
     That last ending is recorded twice more: #11680 held its decisive datum on turn 8
     (the pinned tree's own `tests/e2e/weekly/single_node/configs/GLM-5_1-W8A8_A3_weekly.yaml`
     matching the reporter's launch on every flag except `TP16/DP1` + `--async-scheduling`)
     and wrote "One last check on the replay-ordering guard" instead of the answer; #587
     had the owning line (`glm4.py:203` assign-a-tuple) on turn 4 and spent three more
     turns re-deriving the corrected math. Once a message can name the discriminator, the
     **next** message is the ranked answer built on it — a further read is a turn whose
     text, if there is one, cannot carry the answer after a truncation.
   - **The paste can already contain the whole answer surface.** An argparse error
     prints the complete valid choice list; a log line can name its own thread and
     owner. When the evidence is one such line, the verdict is owed at G1 with no
     source read: #4796 had `--task rerank` vs the printed `score` by its 4th turn,
     spent 32 turns and 1.77 M cumulative prompt tokens before the cap, and emitted
     **no message text at all**; the same issue's pooling release gate (v0.9.1 🟡
     Planned, v0.9.2rc1 #1359) was two greps from the same paste. A traceback can
     also name its own owner and rule: `'DummyModelLoader' object has no attribute
     'get_all_weights'` (#2346) names the loader class *and* the method that only
     `DefaultModelLoader` has, and the MLA `queries per kv … only support {32, 64,
     128}` assert (#1451) states the rule and tells the reporter to check the ratio
     after the TP split — two greps plus arithmetic close both. Its minibatch
     sibling #10962 reasoned the whole source-verified answer and typed it on the
     last turn, cut mid-sentence. Test at every turn: could I write the first
     paragraph now? — if yes, the tool call goes in a message that already carries it.
   - **A new hypothesis that needs a new code area is not this session's job.**
     Put it in the answer as an open question with the one command that decides
     it; #14506 delivered only after a final block that took half the session. For
     an accuracy/corruption or topology report, a ranked list plus an ablation
     matrix already *is* the full answer. The same applies to a **mechanism the
     paste cannot carry**: *which* scheduler step mixed decode with prefill, or
     *which* phase refused a communication resource, is a fact about the reporter's
     run that local source reading cannot produce. Both forwards of one recorded
     minibatch spent their entire budget there after the answer existed — #1369
     held the pinned tree's own `only work without chunked-prefill` doc and CI
     comments while it derived mixed-batch reachability (46 messages, 13 K- and
     19 K-character blocks), #5947's paste carried no Python frame and no
     `LoadGraph failed` block at all while it re-derived the MC2 capacity — and
     neither shipped message text.
   - **A paste that stops in silence is answered, not investigated.** When the log
     simply ends (no traceback, last line inside init), the cause is decided by
     artefacts on the reporter's machine — process count, per-rank logs, device
     mapping — never by more source reading, so the ranked list plus "send these
     three" *is* the message and is owed on the turn the shape is classified.
     #1496 spent 12 turns and ~49 K characters of reasoning across four mechanisms
     (HCCL env, the gloo `cpu_group`, a `all_reduce`, "all ranks picked npu:0") and
     shipped **zero** characters before the cap;
     `references/startup-failures/multiprocess-launch-hang.md` holds the cheap
     discriminators that replace that walk.
   - **An ownership question is answered by naming the owner.** "The flag did
     nothing" and "the log line is missing" are gate checklists on the owning
     component — usually the upstream vLLM OpenAI frontend, not vllm-ascend — and
     take 3-4 calls; #10985 and #9583 each spent 20-30 calls inside the owning
     module with the decisive check in hand.
   - **When the deliverable is a document, the document is the gate.** Open the
     report file on the 2nd-3rd turn with the skeleton plus the first verified
     section, then append **2-3 K characters per write** and make the final
     message the condensed report (path, defects, open items). Never compose the
     whole report in one heredoc: #11739 wrote no file, #11759 lost a finished 5.7 K
     report to the cap, #11742 opened its heredoc on turn 9 of 10 and shipped **no
     text** (`community-test-report.md`). The file body and the message share one
     budget — heredoc characters are tool-call arguments, i.e. output tokens: #11751
     wrote a 15.4 K-character report in three appends (16.1 K characters of tool
     arguments, 36 % of the cap) and then had its **message** cut inside the defect
     table, losing the three required 完整性/易用性/正确性 scores and the FAQ. Keep
     the file ≤ ~8 K characters and lead the message with the 交付件 items (verdict,
     report path, the three scores, one line per defect) instead of re-pasting the
     deploy script or the tables.
   - **`/workspace/scratch/findings.md` is the answer draft, not an extra step:**
     write it at G2 *in place of* the reasoning block; once it exists the next
     turn's text is the answer (`delivery-discipline.md` §draft).
   - **A missing or mismatched artefact is itself the finding** — an empty
     environment block, a file absent from the pinned tree, a version the snapshot
     does not carry. State the gap, give the one command that closes it, stop.
   - **A byte count that "matches" is not the answer.** Re-deriving a mechanism from
     an allocation size is the most expensive way to spend the budget, because the
     decode needs inputs the report does not carry (a `config.json`, a group size,
     an expert count). #7123 had 30 tool results, four assistant turns with **no
     message text at all**, and died at the 16 384-token cap while re-deriving
     upstream DP/EP group algebra to explain a 674 MiB request — the two log lines
     that settle it (`[EP Rank r/N] … Local/global number of experts`,
     `rank … in world size … assigned as … EP rank …`) were never asked for, and what
     the reporter later published as the resolution was a serve line (the cross-node
     `--data-parallel-*` flags + `gpu_memory_utilization`), not a mechanism. Ask for
     the line; send the ranked answer
     (`references/deploy-sizing/moe-load-oom-expert-shard.md`).
   - **Truncation eats the tail, not the head — and the body has a size too.**
     Each action directly after its cause, the closing scope block 3-5 lines. With
     no log pasted: 3-4 causes with one check each, one ladder of ≤6 lines,
     precedents as one line of `#id: fact`, 3-5 lines of scope (≈3-4 K characters);
     a fifth cause is an open question with its one command.
   - **Every assistant turn carries message text, not only a tool call.** A turn
     whose whole output is a tool call leaves nothing for a truncation to keep.
     #5295 emitted *zero* message text on all 16 turns and #16563 on all 20; #17797
     and #5525 emitted only status notes and never a verdict. Both forwards of one
     recorded minibatch repeated the shape: #2349 reasoned the entire ranked answer
     into a 6.3 K-character block and shipped 386 characters before the cap, and
     #6018 emitted nothing on its first 13 turns and wrote the answer on its last
     one. Carry one or two sentences of the
     current answer draft in the same message as every tool call — verdict line
     first, then the evidence being checked — so G1/G2 need no special turn. A
     *status note* is not a draft: the sentence must state the current verdict even
     when the next call may overturn it. #8370 is the pure form: 14 turns, ~49.5 K
     characters of reasoning, zero verdict text - every message a next-action
     narration ("Now orienting with…", "Let me pin both sides", "Two source-verified
     facts now…") while its last turn held the answer. The mechanical test at any
     turn: no message text yet names a cause plus a `path:line` or `#id`, so nothing
     has been drafted.
2. **Classify before investigating.** For a startup log, place it in lane A/B/C;
   for a runtime report, decide whether it is a device-op fault, an accuracy/score
   gap, a corruption report or a performance claim. The signature index
   (`references/first-checks/signature-routes/index.md`) routes to the page holding the
   row for a specific signature; the track-level routes are in *Support entry points*. The
   classification decides which code and which evidence even matter. Do not open
   a deep source walk when a config value, one re-run flag or a health tool can
   settle the question.
3. **The deliverable is a ranked diagnosis the reporter can act on**, not a code
   walk: per hypothesis, the evidence you have, the one command or log line that
   confirms or refutes it, the cheapest fix to try, and what to send back.
4. **The reporter's machine is the only environment here.** You cannot inspect
   it, and local NPU state (healthy, absent, a different card) is not evidence
   about it. Prefer checks the reporter can run — config value, env var, health
   tool, one re-run with a flag — and when the report is hardware-shaped, name
   the artefact that settles it (plog, tiling log, raw request/response dumps,
   `npu-smi info`).
5. **Pin evidence to the reported version.** Read `/workspace/repos/SOURCES.json`
   before citing code: it records repo, ref, commit, basis and coverage cutoffs.
   Every code claim gets `path:line` plus the commit/ref it came from. State
   recheck conditions when a constant, table or log line is version-specific. A
   report citing `docs.vllm.ai/.../latest/…#<section>` is quoting a doc that may
   postdate the pinned release: check that section exists in the pinned tree,
   answer inside the pinned version, and if it is absent say so and ask for the
   text rather than assuming what it says. **When the title names one release and
   the body's launch log another** (recorded #11231: title v0.21.0rc1, body
   v0.22.1rc1; #18315: title v0.27.1RC1, env v0.26.0rc1), say which one you
   verified and give the one command the reporter runs on *their* tree
   (`sed -n 'A,Bp' <file>` around the cited line) instead of asking for another
   launch — and when a **newer pin in the snapshot already carries the repair**,
   quote it: that settles "is it fixed upstream" without PR or comment access.
   A reference page states the pins it was
   verified at: when those equal the refs in `SOURCES.json`, its `path:line` are
   citable as they stand — re-grep only what the page itself marks as
   version-specific, and spend the saved turns on the delivery.
   **A dependency's patch release is part of the version contract.** `torch_npu`
   (pta) `2.8.0` leaks memory and `2.8.0.post1`/`.post2` fix it — the pinned tree
   says so itself (`docs/source/user_guide/release_notes.md:197` at v0.13.0rc1,
   under `## v0.11.0rc1`: "The memory may be leaked … This is a bug from torch-npu"),
   reporter-confirmed on #6299 (`--no-enable-prefix-cache` changed nothing, the
   `.post2` install plus a vllm-ascend rebuild ended it) — while v0.13.0rc1's **own**
   `requirements.txt:31` pins `torch-npu==2.8.0`, so an install that matches the
   release table exactly can still carry the defect. Read the env dump's `torch_npu==` against the patch
   releases before attributing a symptom to vllm-ascend code, and keep the mechanism
   open when only the fix is confirmed
   (`references/runtime-memory/host-ram-and-per-rank-rss.md` §Step 0).
6. **Label the strength of every claim**: reporter-confirmed > verified in the
   pinned source (`path:line` at commit X) > precedent in the issue history
   (quote the issue id) > unverified hypothesis / recalled tool default. Never
   present a recalled default (sampling params, tool flags, CANN behaviour) or a
   matching-looking number (tensor count, hostname, line number) as proof. A
   maintainer's clean run on different hardware is a data point, not a
   refutation of a reporter-confirmed fix on theirs; say what was verified on
   which hardware.
7. **Keep reads small and exact.** Use `rg -n -C2 <symbol>` and
   `sed -n 'A,Bp' file`; reserve whole-file reads for files under ~150 lines. A
   20-40 KB full-file read of an AscendC kernel header buys one fact and costs
   the budget for several conclusions. Two traps that each cost a recorded run
   whole steps:
   - `rg -r` is `--replace`, not "recursive": in any short cluster whose first
     letter is `r`, the rest of the cluster is the replacement string, so `-rn`,
     `-rl` **and `-rln`** rewrite every match to `n` / `l` / `ln`. Output then
     looks like corrupted source (`self.n`, `decoder_layer.l`) or, worse, like a
     real upstream defect: #14423's `rg -rn "daocloud|nju.edu.cn"` turned
     `m.daocloud.io` into `m.n.io`, and #17718's `rg -rln 'set_env.sh'` turned
     `source …/ascend-toolkit/set_env.sh` into `source …/ascend-toolkit/ln` in
     `faqs.md` — which it nearly published as a doc typo and dropped only by
     luck, blaming a "stale .po". **The tell is one short token spliced
     uniformly into every match across unrelated files**; re-run as
     `rg -n '<pat>' <dir>` before quoting any matched text. Recursion is implicit
     on a directory: use `rg -n "pat" .` or `grep -rn`.
     An **empty pattern matches every line of every file**: a probe typed while
     unsure how to recurse — `rg -rn ""`, `rg -n "" .`, `grep -rn "" .` — dumped
     65 KB into one recorded session (#6240, mid-investigation, buying nothing).
     To list files use `rg -ln '<pat>' <dir>`; with no pattern in mind use
     `rg --files -g '<glob>' <dir>` (or `find <dir> -maxdepth 1 -name '<glob>'`) —
     never an empty pattern, which is still what #549 typed while looking for an
     upstream selector and cost itself 65 KB in one result.
   - **A pasted message is usually split across source lines.** C++ `OP_LOGE` and
     Python messages are emitted as adjacent literals (`"… default chat template is no
     longer "` / `"allowed …"`), so `grep -rn "<whole sentence>"` returns nothing and reads
     as "not in this tree" — grep the shortest distinctive clause, or add `-A1`. #6345
     spent three calls that way, on the exact sentence it was looking for.
   - **A `--glob '!<dir>/**'` exclusion is matched against the path you passed,
     not against the search root.** `rg -l 'x' docs/source --glob '!locale/**'`
     still searches `docs/source/locale/**` (verified at v0.22.1rc1: one .po hit
     survives), while `--glob '!**/locale/**'` or `cd docs/source` first excludes
     it. A failed exclusion floods the result with gettext `msgid`/`msgstr`
     lines that read exactly like doc prose — and `.po` files are machine
     translations whose `#: ../../source/x.md:N` references lag the source, so
     neither their text nor their line numbers are citable.
   - **Scope a grep to one repo.** A case view holds 20-40 exported trees under
     `/workspace/repos/<issue>/`; an unscoped `rg -n '<string>' .` there takes
     seconds and returns hits from unrelated products (torchair, metadef,
     ascend-transformer-boost). #11454 ran three of them looking for one log
     string, and the finding was that the string is in **no** export — state that
     as the gap (rule 10) instead of widening the pattern: `rg -n '<pat>'
     /workspace/repos/<issue>/<owner>/<repo>` per candidate, then stop.
   - **Never list, never dump** — the recipe at the top of this file is the rule; this
     is what breaking it costs. A listing was ~24-28 KB once (#13699: 2857 entries) and
     13-104 KB per later session (#9335, #9183, #11739, #8984, #11454 — the last in the
     same block as `skill_view`), `cat SOURCES.json` likewise, `cat
     /workspace/history/*.md` 7 KB for one issue because those files are flat whole
     bodies (#3107). A missing `/workspace/attachments` also makes a bundled listing
     exit 2, which reads like a failed orientation. One recorded two-line
     orientation block (`ls /workspace`, `ls scratch`, `cat SOURCES.json`,
     `ls history`) came back as a single **62 K-character** tool result (#15985:
     `SOURCES.json` alone is 25 KB at v0.26.0rc1, the history listing ~800 names)
     and that run ended at 1.1 M cumulative prompt tokens, output-truncated; #5687's
     same block cost 3 KB only because `SOURCES.json` was piped through `head -50`.
   - **Attachments are the exception: a manifest grep is not an existence test.**
     `ATTACHMENTS.txt` lists names and SHA256s only, so a grep for the issue's
     tokens proves nothing (#12658 read that as "no attachments", skipped
     `raw_completion.json`, and burned the session in the parser). Search the
     attachments view by the file names the issue links (`search_files pattern
     '*.json'`) and open them — they are often machine-readable dumps.
8. **The prior-month corpus is not evidence of absence, and a lexical accident is
   not "no precedent".** Search titles with **one** distinctive token — an op name,
   an error string, a numeric code, a fault kernel, a model id — never the failure
   category and never a word the environment dump already carries (`suffix` ←
   `ATB_SHARE_MEMORY_NAME_SUFFIX=`, `mtp`, `ngram`, `spec`, `npu`, `acl`) or a
   mechanism phrase (`graph capture|capture size|图捕获`). Two recorded forms: #9278
   searched `-i 'suffix|507035|aivec|MPU address'`, got "207 hits / 207 distinct
   issue ids", read the whole result as env-var noise and stopped — while #488 in
   that same output carries the reporter's own `error code:507035`, and the
   titles-only search returns eight real precedents (`suffix` +
   `full_decode_only` among them); #16581's mechanism words returned 442 hits of
   file names, while the paste's own code tokens (`507015|MTE|DDR address`)
   returned 91 and named #3263/#2424. When a pattern hits ~every file, split the
   alternation instead of discarding the survivors.
   `history/*.md` and
   lexical issue search cover only earlier issues; a current issue is never in
   them. Make one grep for the distinctive token (op name, error string, dataset
   name), read 2-3 hits, then stop: precedent supports a hypothesis, it does not
   replace reading the source at the pinned commit. Reach
   for the `github` issue search only when the corpus token has **no** hit: it returns
   whole issue bodies
   paginated (17 K characters in #9986 with `display_complete: false`), so read the
   first page and stop. **Pick the bodies by whether
   the title names a mechanism, not by whether it repeats your symptom.** A title
   like "多卡ViT qkv部分权重为0导致精度异常" carries the file, the check and often the
   version flip ("rollback to 0.7.3rc2 works"); four titles that only say
   "<model>-7B 的问题" are four empty shells. Recorded #2586 read the empty shells
   and cited #976 from its title line alone — the one body that named the exact
   `qwen2_5_vl.py` weight-padding mechanism the answer was guessing at.
9. **Closed components exist, but the split is not where it used to be.** The
   HCCL collectives, opbase and pipeline sources are exported under **`cann/hcomm`**
   (e.g. `src/framework/op_base/src/`, `src/algorithm/impl/operator/`,
   `src/platform/common/externalinput.cc` holds the env parsing), so grep there
   before calling a behaviour undocumented; `cann/hccl` itself is mostly headers,
   examples and the API surface. The genuinely absent pieces are named in
   `SOURCES.json` under `unavailable` (commonly `cann/hixl`, `cann/opbase`,
   `cann/ge`, the ops-* repos). Even when a tree is exported, `.cc` files that
   emit a given error string may be missing (the rank-table checkers under
   `cann/hcomm/src/framework/common/src/topo/` exist only as headers). A tree
   `sources-summary.sh` prints as `files=1478/4849 TRUNCATED` can also lack the whole
   module you came for (a `transformers` export with no `tokenization_utils_base.py`,
   exit 2 on the grep): check the path exists before reading its absence as "the string
   is not there", and prefer the caller's tree — vLLM re-raises its dependency's message
   verbatim. One grep pass per tree, then use the plog, the version correspondence and
   the hardware path.
10. **Report gaps instead of filling them.** Missing shapes, dtype, batch
    regime, server flags, harness config and filtered log lines are the usual
    reason a report is unanswerable or a lane misclassified; naming them is part
    of the answer, and the task environment asks you to flag stale/incomplete
    knowledge. Missing lines are a data gap, not a fact.
11. **A log sentence is not a diagnosis: some are fixed wrapper text.** vllm-ascend
    logs canned advice inside `except` blocks and re-raises, so the sentence names a
    cause the code never checked — the `ACLgraph has insufficient available streams
    … Recommended solutions` block prints for *any* capture exception
    (`model_runner_v1.py:3553-3565` at v0.11.0rc0, gone by v0.20.2rc1: re-grep per
    version). Before adopting a log's own explanation, `rg -n '<fragment>'
    vllm_ascend/` and read the emitting site: an `except Exception` that appends
    `{str(e)}` and re-raises carries the inner error as evidence, not the wrapper.
    Same family as procedure step 3's sync-point trap: where an error surfaced is
    not necessarily its owner.
12. **A crash inside a native module the image installed is an environment
    question, and it is answerable without that module's source.** Three forms — a
    `python -c "import <pkg>"` glibc allocator abort, `std::logic_error` /
    `basic_string::_S_construct null not valid` at engine start, and
    `ImportError: …/_ops*.so: undefined symbol: …Ss` from a prebuilt wheel
    (`std::string` ABI split, decided by the env dump's `cxx_abi_{0,1}` root,
    reporter-confirmed on #1144) — all route from
    `references/image-dependency-faults/index.md`; do not walk the module's source.
    Cap ~6 calls (#5609, #10943 burned the session there). The fourth form is an
    **ACL/GE bring-up failure** at device open — `SetPrecisionMode … AclSetCompileopt(
    aclCompileOpt::ACL_PRECISION_MODE …) error code is 500001` +
    `PlatformInfoInit … SocVersion <name>` + `GEInitialize failed`, ending
    `ERR99999` with no `vllm_ascend` frame: the driver/firmware × image-CANN pairing
    owns it, the printed SoC name is read against `npu-smi`, and no precision/dtype
    flag is the cause → `references/startup-failures/acl-init-500001-and-soc-version.md`.
13. **A pasted error block is not necessarily this failure.** Ascend re-prints a
    recorded device/runtime error block inside a fresh traceback and keeps that
    block's original date, so the title's sentence can be a day older than the
    incident (#9183 named the wrong owner that way) — while a record from *this*
    process's own `init engine (...)`/capture/warmup window is this incident
    reported late (#10700). One call decides both:
    `bash <skill_root>/scripts/plog-timeline.sh --since <report date> <paste>`; a
    **constant whole-hour** lag is a timezone, not a replay, so re-run at the cluster
    nearest the reported time before declaring the fault absent. Formats, the
    `W<MMDD>` blind spot and both worked cases:
    `references/hccl-and-hardware/error-report-timestamps.md`.
14. **A "how do I enable F" question is a capability question, and the verdict comes
    first.** When the report asks how to turn a feature on for a model/hardware pair
    and pastes no log, the reporter is asking whether it is *known to work*: give
    yes / no / partially in the first sentence, with the evidence class — documented
    as supported at the pinned ref, measured on that hardware, covered by
    `tests/e2e/**` (and the nightly yaml's `npu_per_node`), or only by an older
    precedent issue. "No validated recipe for this combination" is an answer;
    a launcher synthesized from the constraints the code enforces is not, and it
    provokes the follow-up "但有人在 A2 上成功跑通过吗?" (#8984). Then map each
    enforced constraint to the error text it raises, so the reporter can place their
    own two errors, and stage the experiment one axis per launch. Three turns,
    `references/feature-enable-recipes/index.md`. Two traps: a pinned doc's narrower
    hardware scope is the project's position for that version (quote it, do not
    overturn it with an older issue), and a comment saying a sub-feature is unsupported
    is often a branch the code already takes (`sfa_cp.py` vs the table's MLAPO ✅).
15. **A working sibling path in the reporter's own evidence constrains the cause.**
    When the same box fails on one entry point and works on another — offline
    `LLM()` vs `vllm serve`, one flag, one model, one card — a mechanism that also
    predicts the working path fails is not yet the answer: name the difference
    first and rank the mechanism below it, with the log line or ablation that
    separates them. #1534 ranked "mindie_turbo replaces RMSNorm" above the working
    `vllm serve` path and left the follow-up ("是因为在线推理默认没有开启 acl graph 吗?")
    unanswered. Two log lines settle which path compiled and which process loaded
    the plugin — `PIECEWISE compilation enabled on NPU` / `Compilation disabled,
    using eager mode by default`, plus the wheel's INFO line
    (`references/image-dependency-faults/mindie-turbo-graph-break.md`);
    `enforce_eager=True` on the failing path is the one-line ablation.

    **Your own ablations obey the same rule: an ablation only rules out a phase it
    actually removes, and its own log line has to be in the paste.** `--enforce-eager`
    is neither universal nor free: at vllm-ascend v0.22.1rc1 it removes torch.compile,
    the ACL graph and even the `Using OOT custom backend for compilation` banner
    (`vllm/config/vllm.py:1032-1038` → `decorators.py:390` → `platform.py:590-592`),
    while the weight load runs identically in both lanes — so it cannot test a
    load-phase stall, and a reporter's "tried it, still stuck" eliminates nothing
    unless the paste carries that flag's own lines (`Compilation disabled, using
    eager mode by default`, `Cudagraph is disabled under eager mode`). Pick the
    ablation that removes the suspected phase, name the line that proves it ran
    (`references/startup-failures/compile-backend-banner-then-silence.md`).

    **The strongest form of this evidence is a tested-configuration matrix in the body**
    (`DP=1,PP=1,TP=32 → works` / `PP=2,TP=8 → works` / `PP=4,TP=8 → fails` / `DP=2 → fails`):
    each failing row then has a named owner and one check, so the matrix answer — one
    verdict line per row — *is* the G1 message, and the rows that work are the controls
    that close the transport, capacity and labelling lanes. #10910 walked 45 turns and
    966 K prompt tokens past that matrix and shipped **no** answer text at all.

    The same test applies when the reporter's *control* is another platform: "nv 卡多次
    一致，npu 多次不一致" makes sampling randomness (`没有 seed`, `temperature > 0`) a
    refuted cause, because it would randomise NV too — so it can be a one-line
    confirmation but not the leading rank. #7180 led with it and the reporter's next
    comment pasted `temperature: 0, top_p: 1, top_k: 1` — reproducibility between
    repeated runs, not accuracy (`references/accuracy-benchmarks/repeated-run-nondeterminism.md`).
16. **A release candidate's experimental mode is a version question before it is a
    mechanism question.** When the mode printed its own banner
    (`FULL_DECODE_ONLY compilation enabled on NPU`, `PIECEWISE …`), every rank finished
    capture, and the *first* request then stalls into the 300 s watchdog, the paste
    has already closed the capture lane and the "setting never applied" chain; what
    remains is model×mode support at that ref or a defect fixed after it. Three
    checks, then deliver: (a) which models the pinned tree validates for that mode —
    `rg -n 'cudagraph_mode.*FULL_DECODE_ONLY' <ascend>/tests/`, then confirm a
    `.github/workflows` file actually runs that test (at v0.11.0rc1 the single
    FULL-mode e2e file was not wired anywhere while the wired ACL-graph runs were
    PIECEWISE at `max_model_len=1024` on Qwen3-0.6B / DeepSeek-V2-Lite / Qwen3-30B-A3B,
    so no VL, 72B or 16 K case was covered); (b) whether the reporter's own control
    works — a default-mode success on the same command makes the *mode* the lever;
    (c) whether the tree carries any mode-specific handling for their family (absence
    is quotable as coverage, never as the cause). A fix that lands in the next rc **is
    not in the snapshot**, so a source walk cannot find it: #4178 walked 22 turns and
    48 K characters of reasoning on that path, emitted zero message text, and was cut
    before answering, while the coverage verdict was three greps and its reporter
    later reported the bug fixed on `0.11.0rc2`
    (`references/hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`).

## Classify the evidence

Two passes. First the **track**, from the topic-entry table under *Support entry
points* below; for a startup log that means lane A (worker died without a per-rank
traceback), lane B (worker raised a Python error) or lane C (collective or device
fault), and for a report with no symptom that means one of the D tracks (sizing,
document, capability, code-change). Then, when the signature is specific — a log
fragment, an error string, a flag that did nothing, a title that names a mechanism —
the exact row in `references/first-checks/signature-routes/index.md` (five pages split by
the shape's home track): each row carries the lane or track it
belongs to and the page that owns it. Read the row's middle cell before opening the
page; it carries the decisive local rule. A token you cannot place: one
`rg -n '<fragment>' <skill_root>/references/` searches every row **and** every page
body, which is the version that pays. **A traceback whose deepest frame is in
`torch/`, `torch_npu/` or `vllm/` rather than in `vllm_ascend/` is a routing
signature: search the helper's name or the exception text in the same call as the
history search, before any source read.** Those trees are frequently absent from the
export (this snapshot's pytorch is `files=1733/19378`, no `torch/_dynamo/`), so no
source walk can confirm such a frame and the library is the only local evidence
there is — recorded #7947 spent its entire 16 384-token budget re-deriving
`AttributeError: 'NoneType' object has no attribute 'size'` from `qwen3_5.py` →
`qwen3_next.py` → `torch/_dynamo/utils.py::call_size` while
`references/accuracy-benchmarks/first-generate-degenerate.md` already carried that
stack, its ruling-out and its fix attribution.

## Procedure

1. One level of `/workspace` (`search_files (target='files')`) — that level only,
   never the history or attachments directory (rule 7) — then run
   `bash <skill_root>/scripts/sources-summary.sh` for the `SOURCES.json` version
   contract; note which repos/refs are present and which are partial exports. The
   layout differs per case (flat `/workspace/history/<issue>.md` vs
   `/workspace/history/<issue>/` + `/workspace/repos/<issue>/`); adapt after
   looking, do not assume.
2. Read the pasted log and place it in a track/lane. Note what is *missing* (no
   worker prefixes, no `Starting to load model`, no version block) — reporters
   paste filtered logs, and missing lines are a data gap. When the body carries no
   log at all, the attachments under `/workspace/attachments/<issue>/` **are**
   the evidence: an image's `.ocr.txt` is the log paste, a referenced `*.json` is
   usually a machine-readable dump (request body, rendered prompt, tokenized
   request, completion) whose fields decide the case before any source read
   (`references/evidence-workspace/index.md`). Not only logs: a screenshot of a **vendor
   docs page** is the vendor's own table — a capability/image question ("可以用
   `quay.io/ascend/vllm-omni:vX` 部署 <model> 么") is often settled by its OCR's
   `Hardware` columns in one read, while reconstructing tag conventions from other
   issues costs a dozen calls and yields weaker evidence (#10303 never opened it). When
   the reported title is the end of a
   chain (`Failed to infer device type`, `Failed to load plugin ascend`), the owning
   line is two or three above it —
   `references/image-dependency-faults/plugin-load-module-missing.md`.

3. Take the track's cheap checks before source archaeology:
   - Lane A/B: effective `max_model_len`, then the diff between the reporter's
     command/env and the matching tutorial or CI test
     (`references/startup-failures/max-model-len-budget.md`).
   - Lane C: re-run with `--enforce-eager`, then `ASCEND_LAUNCH_BLOCKING=1` to
     restore error attribution; read the plog; only then judge whether the
     Python frame named in the traceback is the failing op
     (`references/hccl-and-hardware/collective-error-chain.md`). If the
     collective failure survives re-runs, check health per device and correlate
     the failing device id (`references/hccl-and-hardware/device-health-checks.md`).
     An `npu_*` op on a 2×4/2×64 tensor inside `torch.compile` is a sync point by
     construction: go to the plog and the per-rank `EZ9999`/`chipId` line.
   - Any count or list derived in the log (capture sizes, block counts, chunk
     sizes) that does not match the upstream default: vllm-ascend is probably
     rewriting it. `rg -n '<config field>' vllm_ascend/` (platform.py,
     ascend_config.py, utils.py), read the log's own `Calculated ...` / `N → M`
     lines, and reproduce the number with
     `bash <skill_root>/scripts/aclgraph-size-budget.sh` instead of reading the
     formula by eye. The same holds for **behaviour switches**: downstream
     `check_and_update_config` may turn a feature on that the reporter never
     requested (at v0.11.0rc0 `platform.py:139-150` force-enables the
     `AscendScheduler` fork for a default dense serve and overwrites an explicit
     `enabled: false`), so grep the config field in `vllm_ascend/platform.py`
     before asking which flags they passed (recorded #3979).
   - Then the per-track checks: `references/first-checks/index.md` routes to
     `cheap-checks-by-track.md`, the full list. Open the entry that matches your
     classification instead of reading the whole list, and take its named script
     or check before any source walk.
4. **Write the answer now, then keep going** (rule 1). With the lane/track
   classified and the owning path or decisive check named, emit the ranked
   diagnosis in the message. When the report lacks what the track needs (the two
   serve commands of a PD deployment, prompt length, per-rank log), that first
   answer is the ablation plan plus the list of missing artefacts — already a
   complete deliverable. Continue into source work afterwards as corrections.
5. Fix the version correspondence: image tag → vllm/vllm-ascend commits →
   torch_npu/CANN/driver
   (`references/startup-failures/environment-correspondence.md`). Read the
   release's own `tests/vllm_ascend_v<ref>_test_conclusion.md` when the tree has
   one: it names the models that release validated and its unresolved known
   issues with issue ids. Claim "model unsupported" only after checking
   `vllm_ascend/models/__init__.py` registration and the upstream model file, and
   claim a *method or feature* unsupported only after the consumer grep plus the
   tree's own test for it (`rg -n '<feature>' tests/`) — a
   `pytest.skip("Not current support for the test.")` on exactly that feature is
   the project's own statement at that ref and citable as such (#2071). For the
   recognition-error form, answer from
   `references/startup-failures/model-type-not-recognized.md` before any deeper
   walk.
6. One grep of `history/` for the distinctive symptom token, 2-3 hits, then stop
   (rule 8); `bash <skill_root>/scripts/history-titles.sh [-i] '<regex>'` gives the
   titles of all hits in one call without opening the bodies (`-i` when the
   corpus capitalises the token, e.g. `reranker` vs `Reranker`; bodies are searched
   only with `-b`, because `suffix`/`mtp`/`ngram`/`spec` occur in every pasted
   `collect_env.py` and would return the whole corpus). **A token that is a code
   identifier or an exception string is the exception — run it with `-b` in the
   same call**, because such tokens live in tracebacks and launch scripts, never in
   the env dump: `-b 'TRITON_ALL_BLOCKS|putenv|marked as skipped'` returned 8 hits
   where the title search returned 1 (#8224) and named the `export
   TRITON_ALL_BLOCKS_PARALLEL=1` line five other reporters already carry in their
   launch block. A title-only search that comes back empty is **not** "no
   precedent" until the distinctive identifier has been body-searched once. Then
   read those 2-3 bodies with `bash <skill_root>/scripts/history-body.sh <id> [<id> …]`, not a shell
   prefix read of their first lines: the template puts a 100-250 line
   `collect_env.py` paste before the symptom section (#3963's bug heading is line
   148), and the script collapses that dump to a marker (`--env` restores it,
   `-g REGEX` pulls just the traceback).
7. Locate the owning code path at the pinned commit and walk it from the entry
   point (kernel launch site, harness config, model/feature flag) to the line
   that produces the symptom. Prefer computing the decision (a tiling branch, an
   aggregate metric) with a small `uv run --no-project python` snippet over
   reading code by eye.
8. Cross-check upstream vs downstream: vllm-ascend frequently re-implements or
   bypasses upstream vLLM (custom AscendC op vs `torch_npu` fallback, upstream
   reasoning parsers vs downstream template kwargs, docs baselines vs
   third-party harness, flat `hf_config` reads vs `model_config.hf_text_config`
   accessors, a copied forward body vs upstream `super().forward()`). Record
   which layer owns the behaviour and for which version, plus what would
   invalidate the claim. One level down when the frame is in an SoC subpackage
   (`vllm_ascend/_310p/`, a later `_<soc>/`): diff the override against the
   generic `vllm_ascend` implementation and expect a **missing propagation** of a
   fix the generic path already carries, not a new defect
   (`references/soc-platform-paths/index.md` rule 4).
9. Re-emit the answer in the shape below with the corrections, then stop.

## Answer shape

- The direct answer first: yes/no for the reporter's literal question, the
  version that works or the version where it breaks, and whether the reported
  crash is real. One sentence, before any analysis, in the language the report was
  filed in. **The cheapest next action belongs in that same first paragraph**, not
  below the evidence table: a session can be cut wherever it happens to be, and
  #7580 was cut inside its table, after the verdict and before its ablation plan.
- Then the one-line classification: lane A/B/C and the phase that failed (load,
  memory profiling, graph capture, first collective); or, for a runtime report,
  the primary cause plus its track.
- Ranked causes, each with the evidence you have, the exact check that confirms
  or refutes it, and its confidence label (reporter-confirmed / source-verified
  at commit X / hypothesis).
- An evidence table (`path:line`) for code claims, with precedent issue ids
  quoted as `#<id>: <one-line fact>`.
- The action ladder, cheapest first: an explicit `--max-model-len`, a re-run with `--enforce-eager` /
  `ASCEND_LAUNCH_BLOCKING=1`, `return_token_ids: true` on the same request body,
  one `num_speculative_tokens` step, `asys health -d=$i`, a version correction —
  before any patch proposal.
- Version and hardware scope of the conclusion, plus what to send back (full
  per-rank log, plog, tiling I-line, raw request/response dumps, `npu-smi info`,
  `asys health` output, `dmesg` OOM lines, both PD serve commands).
- What you could not verify locally and which questions remain open, stated as
  such.

## Support entry points

Open **one** topic entry: every `references/<topic>/index.md` states
its scope, inherits these rules and routes to its own pages, so page-level triggers
live where they are used. Two procedure helpers come first:
`references/first-checks/index.md` (the per-track cheap checks of step 3,
`cheap-checks-by-track.md`) and `references/evidence-workspace/index.md` (case
layout, `SOURCES.json`, history search, attachment/OCR caveat;
`delivery-discipline.md` holds rule 1's G1/G2/G3 gates and the reasoning patterns
that break them, `recorded-cases.md` the per-issue index).

| Classified shape | Topic entry |
|---|---|
| score/accuracy disagreement (**including the same command line on two images**: the newer image's default-on graph-fusion stack is part of its serve flags — `harness-vs-engine-attribution.md`, "Two images"), 乱码/重复 on short or long prompts, truncated output, **the same prompt run repeatedly giving different lengths/contents**, **or the same body differing only with concurrency while `VLLM_BATCH_INVARIANT=1` is set** (batch-invariance coverage, not run-to-run noise), reasoning routing (including a thinking stream that never emits `delta.content` and burns the whole completion budget), **or fluent-but-wrong answers on a MoE (A3B) serve — mixed-language CoT, a final answer that breaks the prompt's own rule, normal tokens/s, where no parser/`enable_thinking` flag can help because neither moves logits** (`references/accuracy-benchmarks/moe-gating-topk-precision.md`), VL degeneration **or a VL/OCR answer that is merely wrong/coarse on multi-card while the single-card run is correct** (a TP-axis ablation, not a sampler question), **audio/Omni requests** over `/v1/chat/completions` that loop or return garbage, a score comparison **across two quantized formats** of one model family (`w8a8c8` vs `w4a8c8`: recipe and node count differ with the width), quantized-checkpoint conversion, **or a first request that dies with `AttributeError: 'NoneType' object has no attribute 'size'` while the deepest frame is `torch/_dynamo/utils.py::call_size` / `compilation/acl_graph.py` on a VL-capable checkpoint served text-only (`--language-model-only`)** | `references/accuracy-benchmarks/index.md` |
| the asking is for an edit: remove/deprecate an env flag or a temporary switch | `references/code-change-requests/index.md` |
| a custom op crashes the device (incl. a `507015` "MPU address access is invalid" in the chunked FLA/GDN `chunk_fwd_o` path, or a nightly op test that passes while the device faults), **or the plog's `fault kernel_name=` resolves to a downstream `@triton.jit` kernel** (`token_bin_counts_and_mask_kernel_0` — the sampler/penalty family, owner and ablation in `references/custom-ops-runtime-faults/triton-jit-kernel-device-fault.md`), or an aclnn SoC **registration**/`EZ1001` error (the NZ weight-layout ones — a dtype family on a pooling head and a **shape** family, `Not support mat2 n = 1 or k = 1 when format is FRACTAL_NZ`, on any narrow linear such as a MoE shared-expert gate — are `references/startup-failures/nz-weight-format-refusal.md` and the pooling lane), a `Tensor scale not implemented for DT_FLOAT16` / `161002` on `aclnnQuantMatmul*`/`aclnnGroupedMatmul*` (a quant scale dtype gate whose sibling methods must be swept too), **an `EZ9999 … Cannot find bin of op <Op>, integral key <dtype list>`** beside `Kernel Run failed. opType: N` (no kernel binary for that dtype key — the model dtype is the lever, `--dtype bfloat16` the one-command test, and the image only the second suspect), a **bare `RuntimeError: setup failed!`** or `<Op>Operation setup failed!` with no device code (nothing launched: the deepest `vllm_ascend` frame's own mask/size argument owns it), **or the same `161002` naming `y` instead of a scale** — `GMM:y dtype does not match with required dtype[DT_BFLOAT16]` with `found DT_INT32` on `aclnnGroupedMatmulV5`: the output dtype the launch asked for against the activation the MoE permute actually produced (`ep_size < 16` forces the permute that ignores `apply_a8_quantization`; the eager W8A8 test is the control — rule 5 of the page), a `<tensor> shape is invalid` / `do tiling failed, ret is -1` out of `<op>_tiling.cpp` `GetShapeAttrsInfo` (an argument rejection, nothing launched), or a MoE init-routing tiling code, or an op the operator side returns as "非cann算子" (a framework-owned `…Custom` op — the dispatch-site/ownership answer), **or** garbled MoE output / an invalid-GM-address fault under `enable_fused_mc2=1`, where the W8A8/W4A8 `dispatch_ffn_combine` variant sizes its aliased GMM1 workspace and row stride by `K` instead of `N = 2×moe_intermediate` (`mc2-variant-workspace-and-strides.md`) | `references/custom-ops-runtime-faults/index.md` |
| a `call aclnn<Op> failed, error code is <6 digits>` on a MoE deployment (`aclnnMoeInitRoutingV3 … 561103`), often with "DP2 TP4 有没有限定 / 能否启动时校验拦截" attached | `references/custom-ops-runtime-faults/moe-init-routing-tiling.md` — the code is a failed **tiling check on one input**, never a card/HCCL claim, and the named op is not a grep key |
| **"MP or Ray, and what DP/TP/PP for my N nodes"** on a non-quantized model, **or a pasted multi-node line that cannot execute as written** (`--headless` with no `--data-parallel-*`, TP/PP beyond the cards in that command without `--nnodes`/Ray, a misspelled long flag), capacity or stage-layout questions, two serve commands that differ by one resource knob where the larger OOMs at start (128K-256K contexts **and** a short 4K one), or a load-phase `NPU out of memory` inside `create_weights`/`get_weight` on a 384-512-expert MoE with `--enable-expert-parallel` + `--data-parallel-size` | `references/deploy-sizing/index.md` — the TP4-up/TP8-OOM shape is its `tp-scaling-oom-at-startup.md` page, which separates a post-sizing reserve shortfall (frame: capture/workspace) from the KV pool's own allocation loop (frame: `initialize_kv_cache_tensors`); the EP+DP load OOM (experts-per-card, the `[EP Rank r/N]` greps) is `moe-load-oom-expert-shard.md` |
| DSA / DeepSeek-V3.2 sparse attention (indexer scatter, MLAPO token gate, concurrency crash with no log), an SFA/LI **C8** deployment whose prefill node stops answering or garbles the first request with `enable_sparse_sfa_c8` / `enable_sparse_li_c8` / `c8_enable_reshape_optim` in `--additional-config` (or that only loses accuracy against a quantized sibling — `references/accuracy-benchmarks/quantized-sibling-score-gap.md`), or an engine-start `assert sum_page_size_bytes % … page_size_bytes == 0` in `_reshape_kv_cache_tensors` | `references/dsa-sparse-attention/index.md` — the flag algebra and the `store_kv_block*` ops are `c8-flags-and-kv-store-path.md`, the KV-page assertion is `c8-page-size-reshape-assertion.md` |
| version contract, snapshot coverage, precedent search, delivery gates | `references/evidence-workspace/index.md` |
| a maintainer's release known-issues notice for one model/hardware pair, or "does this notice cover my run" | `references/release-known-issues/index.md` |
| "how do I turn feature F on" / capability verdict (PCP, DCP, EP+DP), **torchair graph mode on a DeepSeek/MLA model** (incl. its first request dying with `UnboundLocalError … 'decode_hs_or_q_c'` in `mla_v1.py`, or `decode_k_pe` / `TypeError … 'NoneType'` after a hand patch, **or dying at layer construction on the `queries per kv … only support {32, 64, 128}` gate, which is heads-per-rank arithmetic — TP ≤ 4 for a 128-head model**), or only another SoC has a recipe/quantized weights for it | `references/feature-enable-recipes/index.md` |
| the first cheap checks after classification | `references/first-checks/index.md` |
| a two-node start whose visible error is a **collective or device op** while the plog's first block is the HCCL *socket link* (`[Wait][LinksEstablishCompleted] is failed. ret[9]`, a `LINK_ERROR_INFO` table, `[Create][InterLinks] socket manager create connections failed`, `[Create][CommPlane] failed`) — including `aclnnMoeDistributeDispatch… 561000` / `HcclAllocComResourceByTiling … ret = 4` where `4` wraps a `9`/TIMEOUT and the reporter's only check is `ping` — a collective/device error (`HcclAllreduce`, `ERR00100`, `507035`, `EZ9999`), a **communicator-creation** failure (`HCCL function error: hcclCommInitRootInfoConfig(…), error code is N` out of `create_config`, `HCCLUtils.cpp:140`), device health, port band, MC2 profile-run hang, RPC watchdog, or a graph mode that printed its banner and finished capturing and then hangs the first request **or** dies under concurrency with a `507011` / aivector / `Illegal instruction … unaligned UUB` device fault, **or** a no-log "图模式算子通信报错, 只有开启单算子才能规避" on a two-node/DP MoE serve (a mode split to ladder, not a component: `graph-vs-single-op-comm-fault.md`) | `references/hccl-and-hardware/index.md` |
| the interpreter/image owns it: native aborts, a platform plugin that cannot import (`Failed to infer device type`), missing `libatb.so`/`ascend_transport.so`, ABI split, triton-ascend pair, **a `triton` that imports but has no `language` at `import torch`** (`AttributeError: module 'triton' has no attribute 'language'` from `torch/_dynamo/utils.py`, no `vllm_ascend` frame, usually a `-310p` box — the `-310p` Dockerfile removes triton on purpose, so the resolved module is uninstall residue or a shadow directory: `triton-module-residue.md`), **a host C++ compile of a generated TU failing inside a `torch_npu`/CANN header** (`undeclared identifier 'aclOpExecutor'` at `AclOpCompileInterface.h`, `launcher_cxx11abi1.cxx`, `'cstdint' file not found`), offline source build, pip index visibility, an accelerator wheel breaking `torch.compile`, **a Triton-Ascend extension op that runs and returns wrong values** (`extract_slice` of a bf16 sub-tile at an unaligned start — owner is the wheel, the page carries the doc-vs-UT coverage gap and the dtype-vs-offset discriminator), no image/wheel for that arch or SoC | `references/image-dependency-faults/index.md` |
| a KV pool (AscendStore + MemCache/Mooncake) is attached and the request path breaks — **or** nothing breaks and only the cold/hit split looks wrong ("初始 TTFT 慢 6-7 倍，命中后很快") | `references/kv-pool-store/index.md` (the speed-only shape: `pool-attached-cold-request-cost.md`) |
| a LoRA-enabled serve is far slower than its base model, or dies when a graph is captured on a request | `references/lora-adapters/index.md` |
| a device fault or corruption whose deepest frame is an attention backend in a mixed decode+prefill batch; **or** a host-side `could not broadcast input array` / `np.repeat` length error in `worker/pcp_utils.py` with spec decode on and PCP > 1 | `references/mixed-batch-splits/index.md` (the PCP schedule variant is `pcp-schedule-index-units.md`) |
| a logging/profiling flag appears to do nothing, `/start_profile` 404, unknown-env warning | `references/observability-and-logs/index.md` |
| P/D or EPD starts but cannot move KV, moves it and the output alternates, or the decode node **repeats one token/phrase to the model cap** on a DSA/MLA model (GLM-5/5.2, DSv3.x: `1 + 1 = 2.` ×2048) or dies as a bare `ACL stream synchronize failed` only when the batch/sequence/capture-size knobs are raised — diff both serve commands against that model's own tutorial recipe (connector class, `--speculative-config` on **both** sides, the HMA flag) before any `kv_p2p/` read, `recipe-contract-diff.md`; or the client times out while D logs no request and only P prints `Force freed expired request … (480 seconds)`; **or** requests are served but TTFT is slow while P prints `fail to receive KV Cache transfer done signal in 120s` (`llmdatadist-done-signal-and-ttft.md`); **or the prefiller's worker dies inside `start_load_kv` with the traceback cut at the layerwise connector's `_get_kv_split_metadata(...)` call** (symmetric TP/DP, hybrid attn+mamba + MTP under chunked prefill — the paste carries no exception class, so deliver the throw-site inventory, the `VLLM_LOGGING_LEVEL=DEBUG` ask and the three single-axis ablations instead of deriving the exception, `layerwise-split-metadata-crash.md`); **or** the transport *plane* is asked about before any run — A3 双机, 参数面 RoCE only, no 灵衢 switch, no log → `plane-selection-a3-parameter-plane.md` | `references/pd-disaggregation/index.md` |
| throughput, queueing, prefix-cache hit rate on a healthy engine; **or the whole claim is a comparison against an older release** ("相同条件下性能测试不如0.7.3版本", no numbers on either side, `upgrade-vs-older-baseline.md`); a TPOT/inter-token **step** at one concurrency number (its `GPU KV cache usage` at that step decides KV pool vs graph ceiling); **or the evidence is a profiler table** (`<Op> 被调用了 N 次`, "NPU 空闲时间 95%", an NPU-vs-GPU operator-count compare), **or one row of such a table is `N/A`** (`Input Shapes: N/A`, `Block Dim: 0` for a custom op while its neighbours carry shapes — a launch-lane question, `profiler-shape-columns-custom-op-launch.md`); **or a quantized (`w8a8`) artifact that is no faster than its BF16 sibling while the dump shows a whole layer class (e.g. the full-attention `self_attn`) without a quant kernel**, **or a design-intent question on a hybrid linear-attention (GDN) family** ("为什么 linear attention 在 Ascend 上不能像 CUDA 一样融合成一个更大的算子" — answer per prefill/decode lane, and label the fusion roadmap as a maintainer statement: the announced op covers the chunk/prefill lane while decode keeps Triton, `gdn-fusion-lanes-and-roadmap.md`, **or two runs compared by one number each — two platforms, two images or two releases ("12s vs 4s", "吞吐高了一倍"), where the evidence is an EvalScope/AISBench table or a screenshot of one**, **or a DP×TP MoE serve whose own command has `--enable-expert-parallel` missing or on one node only, typically because EP garbled the output so the reporter dropped it** | `references/performance-throughput/index.md` (profiler tables: `profiler-opcount-evidence.md` first, missing columns `profiler-shape-columns-custom-op-launch.md`; a layer-*coverage* claim is not a count question: `quantized-artifact-layer-coverage.md`; two-run ratios are `cross-run-ratio-normalization.md` + `scripts/benchmark-table-normalize.py`, the EP-off MoE serve is `moe-comm-path-when-ep-off.md`; GDN fusion lanes and roadmap: `gdn-fusion-lanes-and-roadmap.md`) |
| the engine stops answering after one oversized request (`Running` 0, nothing scheduled, restart the only recovery), or requests wait forever on an idle pool | `references/scheduler-admission/index.md` — the admission gate (`full_sequence_must_fit`) and the FIFO `break` are `waiting-queue-wedge.md`; a queue that drains is *not* this topic |
| an embedding/pooling service never starts (checkpoint served as generate, score/rerank arch-name routing, fp32 head weight format), dies under concurrency, returns wrong vectors, **or the launch line is rejected outright** — `vllm serve: error: argument --task: invalid choice: 'rerank'`, where the value is `score` *and* the reported release may not carry pooling at all (v0.9.1: 🟡 Planned, V0 retired), so give both verdicts in one message | `references/pooling-embedding/index.md` |
| OOM after a healthy period, HBM not released, host RAM, container working set, **a "memory leak" title whose `collect_env` block carries `torch_npu==2.8.0`** (known pta leak: `2.8.0.post1`/`.post2` + rebuild, reporter-confirmed #6299 — read that line before the RSS arithmetic), KV-init allocation failure, a start-up **KV budget near 0 or negative** on a device with free HBM (`Available KV cache memory: … (halved for workspace)` on 310P, `Available memory: 0` on the second engine of one card), or an `expandable_segments` unmap crash at `empty_cache` | `references/runtime-memory/index.md` |
| traceback frames are only `vllm/**` + `site-packages/**` (tokenizer `Already borrowed`, frontend preprocessing), or every chat request fails with `ValueError: As of transformers v4.44, default chat template is no longer allowed …` (a chat-template/mode answer, not a checkpoint one — `chat-template-missing-and-tokenizer-mode.md`) | `references/serving-frontend-faults/index.md` |
| a SoC-conditional path: 310P/300I (including a **VL/OCR** model there that serves and answers garbage), a **newer model family on 300I Duo whose only doc mention is a "preview … follow #NNNN" closing note** (a capability verdict owed in the first message, not a mechanism hunt — the pinned 310P tutorial's own table and closing note are the answer), **an omni/TTS model there with a `vllm-omni` image tag in the question** (no `-310p` omni image exists; the route is `SOC_VERSION=ascend310p1` source build + a per-stage fp16 `--deploy-config`), a first-forward `AttributeError: '<X>310' object has no attribute '_…'` (a `_310p/` copy that lost its A2 sibling's helper), a 310P first-request fault that names a **plain tensor op** rather than a kernel (`507018` "aicpu execution is abnormal" / `funcName=RunCpuKernel` / `<Op>Operation setup failed!` — the statement or the un-gated flag owns it, not the driver), a 310P serve line carrying a KV dtype the doc excludes (`--kv-cache-dtype fp8`), **a 310P `FULL_DECODE_ONLY` + MTP start that dies inside capture with `aclrtMemcpy … 107030` at `get_splitfuse_mask` while eager works** (the fallback mask reads back on the host; the capture-safe branch is gated on two torch_npu `_npu_*` bindings, so the CANN-upgrade advice moves nothing), a 310P **W8A8 start-up that dies in an int8 matmul kernel** (`507015` + `L0C read/write conflict` + `QuantBatchMatmulV3_NZ_NZ_…` during `profile_run`: image CANN/op-binary vs runner V1/V2, and `VLLM_USE_V2_MODEL_RUNNER` is a version-gated no-op whose proof is the `NPUModelRunner310V2` banner), A5 vs A2 wheel, kernel package for the running SoC | `references/soc-platform-paths/index.md` (`310p-mtp-graph-splitfuse-mask.md` for that capture crash) |
| a PP>1 deployment (`--pipeline-parallel-size`) — including **PP together with `--speculative-config`** (`suffix`/`ngram`/`mtp` alike dying on the first request with a bare `IndexError`) — or "does PP work on this model/hardware" | `references/pipeline-parallel/index.md` |
| spec-decode/MTP symptoms: acceptance collapse, empty first response, async bookkeeping, drafter count copy, a TND `queryT`/`actualSequenceLengthQ` refusal, a dense-backend `attn_mask` `AssertionError`, **or a dense drafter whose first request dies with `AttributeError … 'draft_attn_metadatas'` on `FULL_DECODE_ONLY`**, **or acceptance exactly 0.000 with a correct target on a model family the tree has no spec-decode branch for** (upstream ships a dedicated proposer, `vllm_ascend/spec_decode/__init__.py` names no such family — e.g. gemma4 MTP; the family grep decides it before any block-table read) | `references/spec-decode-mtp/index.md` (`family-proposer-parity.md`) |
| engine/worker never becomes ready (both lanes), model-type/quantization gates, arg validation, a multi-process launch (`external_launcher`/torchrun SPMD, TP>1) that hangs at init with no traceback and no `Starting to load model`, a `--distributed-executor-backend ray` start whose paste carries `ValueError: Current node has no NPU available … 'GPU': 8.0` after `Connected to Ray cluster.` (the raylets advertise `GPU` where the platform asks for `NPU` — one `ray status` settles it), or the same message with `'NPU': 0` beside `NPU_group_<i>_<pgid>` labels and `--data-parallel-size` > 1 (the check reads the *free* view, so a live placement group holds the node: `ray-dp-placement-group-npu-exhausted.md`), a parallel-size `reshape` failure, metadata/capture phase (including a capture-time `Not allow to synchronize captured-stream` / `107027` — a host readback inside the captured region, not a fabric fault: `references/startup-failures/captured-stream-host-sync.md` — where a bare `107027` with no `copy_stream` and no CANN sentence is a different lane, the blocking-env lever of its section 6), a graph setting that never took effect (a torchair DeepSeek/MLA serve whose capture **succeeded** and whose first request then raises `UnboundLocalError … 'decode_hs_or_q_c'` is the chunked-prefill contract, not a startup fault — `references/feature-enable-recipes/torchair-mla-graph-and-chunked-prefill.md`), a **VL/audio model whose graph-mode start dies at config time** with an `AttributeError` on the composite config and no `Starting to load model` (`references/startup-failures/vl-layer-count-attribute.md`), a **torchair** capture failing with `torch._dynamo.exc.Unsupported … DelayGraphBreakVariable` (`fullgraph=True`), **or the vLLM `torch.compile` lane failing the same way** with `Attempted to call function marked as skipped` / `posix.putenv` / `Developer debug context: module: posix, qualname: putenv` while the `from user code` block ends in a downstream `os.environ[...] = …` (the launch-side fix is to pre-set that variable; `env-write-in-compiled-forward.md`), an inductor fusion pass calling an op that was never registered (`'_OpNamespace' … has no attribute 'qkv_rmsnorm_rope'`), a `torch_npu` ACL/GE init failure at device open (`SetPrecisionMode … 500001`, `PlatformInfoInit … SocVersion`, `ERR99999`), an offloading-spec `KeyError: '<layer>.self_attn.attn'`, first-weight-group `ValueError` (embedding checkpoint served as generate), a shard that is not a safetensors file (`HeaderTooLarge`), a load-phase `KeyError: '<param>.weight_scale'` whose key set is a few sub-projections (checkpoint variant vs quant config, not a per-layer bug), a `RuntimeError: Promotion for Float8 Types is not supported … Float8_e4m3fn and Bool` on the first MoE forward (W4A8-MXFP activations × flashcomm1 × EP all-gather mask), **or** a memory-profiling window where every rank but the driver dies in the same second with `EOFError: Ran out of input` from `torch_npu/…/distributed_c10d.py:_gather` → `dist.broadcast_object_list` (the torch_npu gather emulation's size handshake, not the fabric; the driver's `HcclAllgather` `EI0006`/`DestLink` block ~120 s later is the cascade, and the log's `npu is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.` is an opt-in that `VLLM_USE_V1=1` clears — `references/startup-failures/profile-run-logits-gather-eof.md`) | `references/startup-failures/index.md` |
| a served VL/omni request dies **before any kernel** with `Attempted to assign N = N multimodal tokens to M placeholders` / a `shape mismatch … cannot be broadcast to indexing result` out of `_merge_multimodal_embeddings`, or an image arrives with no `<|image_pad|>` run — two counts from two implementations (prompt builder vs vision tower + merge), a host `ValueError` whose deepest frame is upstream `models/<model>.py::embed_input_ids`, so no Ascend lane applies | `references/multimodal-inputs/index.md` — `placeholder-count-mismatch.md` carries both formulas, the `mm_processor_kwargs` asymmetry and `scripts/mm-placeholder-count.py` (recorded #14014, #3286, #6934, #7023) |
| the symptom is a tool call (`JSON Parse error` in `arguments`, no `tool_calls` while streaming, or a clean `name` with `arguments` exactly `{}` on a GLM serve) — **or both fields empty at once** (`content` null *and* `tool_calls: []`, often with `finish_reason: "tool_calls"`, on a serve that already passes `--tool-call-parser deepseek_v3`) | `references/tool-call-streaming/index.md` — the `{}` shape is `empty-arguments-object.md`; the empty/empty shape is the parser's silent drop, `deepseek-parser-silent-drop.md` |
| the deliverable is a report about a document (community test, `[Doc]`, tutorial commands), **or a `[Doc]` whose defect is an absence** — a page that should link a section existing elsewhere, a contributor path missing a prerequisite | `references/tutorial-verification/index.md` (absences: `missing-guidance-and-contributor-docs.md`) |
| a long-running serve (hours) dies on a bare `AssertionError` in **upstream** `vllm/v1/core` KV bookkeeping — `prev_free_block of fake_free_list_tail should always exist` at `kv_cache_utils.append_n`, reached through `allocate_slots` → `remove_skipped_blocks` → `free_blocks`, one instance of many, PD-mix/prefix-cache/hybrid-KV deployment | Upstream core owns the invariant and the linked-list mechanism cannot be derived from the paste; a `patch_balance_schedule.py` frame is a pass-through unless `enable_balance_scheduling` is set, and the real variable is the KV-group set (compressed/SWA) plus external pool hits — deliver the ownership verdict with the per-trigger ablations: `references/upstream-downstream-drift/kv-block-free-list-assert.md` |
| `AttributeError`/`TypeError` in a downstream patch, **or the missing attribute belongs to a library object** — `'Tensor' object has no attribute 'mask_fill_'` in a `vllm_ascend/` frame is a downstream **typo**, not drift and not a version pairing: one grep, one rename, plus the branch gate that makes it prompt-length dependent, `references/upstream-downstream-drift/library-method-typo.md` (a config-object `AttributeError` raised at engine config time on a VL model is not this row — see `references/startup-failures/vl-layer-count-attribute.md`), an upstream refactor that kept upstream code, an attribute the upstream `__init__` assigns unconditionally (a swallowed constructor exception), **a load-phase `AttributeError` whose object is a *model loader* (`'DummyModelLoader' object has no attribute 'get_all_weights'` under `--load-format dummy`)**, a `KeyError` from upstream's `Attention`-only layer filter under an offloading spec, or a **legacy wrapper** whose first frame is `TypeError: get_manager_for_kv_cache_spec() missing 1 required positional argument: 'enable_caching'` (or `'list' object has no attribute 'ref_cnt'` at `block_pool.touch`, or `Can't get local object 'get_request_block_hasher.<locals>.request_block_hasher'` in the connector's `pickle.dumps`) — count the **four contracts**, not the first `TypeError`: a matching signature is not a matching block-container shape, and the fix also needs the reachability and replacement-import checks, `legacy-wrapper-contract-audit.md` | `references/upstream-downstream-drift/index.md` |

Scripts are re-runnable; `<skill_root>` is the path `skill_view` reports and each
header carries its flags. Their arguments, output and originating issue are in
`references/first-checks/scripts-catalogue.md`; open it rather than guessing a name,
and the topical pages name the ones they use. `templates/` holds starter files to copy
and edit rather than retell — currently `templates/omni-310p-deploy-config.yaml` (a
two-stage fp16 `--deploy-config` for one 310P die, cited by
`references/soc-platform-paths/310p-omni-model-deploy.md`). The per-signature
classification index (log fragment → track → owning page) is
`references/first-checks/signature-routes/index.md`.

