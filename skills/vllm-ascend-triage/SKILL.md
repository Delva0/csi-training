---
name: vllm-ascend-triage
description: 'vLLM-Ascend issue triage; load alone, no shell beside'
---

# vLLM-Ascend issue, deployment and support triage

## Before the first shell call

**Your first tool block holds exactly one call: `skill_view` on this skill, by name**
(`skill_view(name='vllm-ascend-triage')`; a **filtered** `skills_list` returns `[]` even
though the skill is installed, while the unfiltered call lists it — so an empty list is
a tool quirk, not "no skill for this task", and this name is already in
`<available_skills>`, so do not spend a `skills_list` call at all).
Anything batched beside that call — `ls
/workspace`, `cat SOURCES.json` — executes before this body is in context, so no rule
here can stop it, and its output is re-sent on every later turn: six recorded runs paid
14-54 KB that way, one ending output-truncated at 1.1 M cumulative prompt tokens, which
is why the frontmatter description names the prohibition — "load alone, no shell
beside" — instead of only the reading order. If something has to share it, let it be
`bash <skill_root>/scripts/sources-summary.sh` (~10 lines) — that summary plus the
reporter's own paste **is** the orientation. Never `cat /workspace/repos/SOURCES.json`,
never enumerate a history directory, never `cat /workspace/history/*.md`, and do not
"list one level" with `search_files (pattern='*', target='files', path='/workspace')`:
recorded in both runs of one minibatch, it returns ~100 recursive paths truncated at 50
(~2.8-3 K characters re-sent on every later turn, `/workspace/history/*.md` among them)
and neither run used a line of it (#12278, #11761); `ls -d /workspace/repos/*/` is the
12-line substitute when the repo taxonomy is actually needed. Any `search_files` call
keeps `target='files'`: `pattern='*'` **without** it is compiled as a regex and fails
(`regex parse error … repetition operator missing expression`), one recorded call paid
for that (#15389).

Search by the **most specific** token first (`rg -l`, `scripts/history-titles.sh`),
never by listing — a recursive `search_files` glob is a listing too: `pattern='*Qwen3*'`
over `/workspace/repos` returned 51 paths, nearly all vllm `tests/evals/gsm8k` configs,
while the wanted page was one `rg --files -g '*Qwen3.5*'` inside
`docs/source/tutorials/models`. A model **family** name is a listing token, not an
identifier: `rg -l 'Qwen3.5'` and `'Qwen3.6'` over `references/` return ~20 unrelated
pages each and neither returned the owning page (the five-token loop cost 9.3 K
characters of transcript, re-sent every later turn), while the task's own
phrase token (`'dp2 tp4'`, 5 paths) and a numeric/error token (`'248044'` → the page
that owns it) land on the first call. **Attachments are the one
exception**: open the specific files the issue links (`ATTACHMENTS.txt` lists names and
SHA256s only, so a grep of it is not an existence test) — rule 7. Costs and traps:
rule 7.

## Problem class this covers

Ten entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter. The last shapes deliver a *decision* (sizing),
a *document review*, a *capability verdict* or a *change plan* instead of a root
cause, but the same rules apply — answer first, evidence pinned to the reported
version, gaps stated.

**Issue report** (template-shaped: environment dump + logs + narrative) ·
**Install / pip-resolution report** · **Deployment failure** · **Support / version
question** · **Feature-enable / capability question** · **PD / EPD disaggregation** ·
**Deployment sizing / stage layout** · **Community-test / doc-verification task** ·
**Release known-issues bulletin** · **Code-change / cleanup request**.

Their applicability conditions, the exclusions that keep a neighbouring shape out
(a `-310p` image, a *decision* task, a maintainer bulletin) and each shape's route
into a topic index are `references/dispatch/problem-classes.md` — read that entry on
the turn the shape is classified, not a second copy of it here.

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and obey the delivery gates.** A long
   session can be cut off mid-sentence, so the deliverable has to survive
   truncation: what decides the outcome is not how much evidence was gathered but
   whether the message exists. The runs that shipped **no answer at all** each had a
   usable lead in hand — #1255/#5772 held the owning line, #2807/#15394 every fact of
   a ranked answer, #4296 answered from one traceback in ten turns — and the thinnest
   wrote 0-2 234 characters against 43-48 K of reasoning (#8301 **nothing**, #11171
   54, #11080 115, #2069 217, #3706 743, #1617 2 234, #6488, #7996), the newest
   **#18691 nothing** (21 turns / 34 tool results) with the decisive correction — a
   `/health` 200 is not liveness — already on its tenth turn. Each ended on a
   reasoning block with no message and no tool call: the cap landed on the answer's
   shape. Per-run ids, counts and cut points: `delivery-discipline.md`,
   `delivery-recorded-sessions.md`.
   Two triggers would have saved them. (a) A **second** pass over the reporter's own
   numbers that reaches the same estimate is the turn the ranking is owed
   (`delivery-discipline.md`, *a recomputed estimate is not new evidence*). (b) The
   library already holding the case: **#16079 was cut twice against its own owning
   page**, the second time re-reading three facts that page already stated. When a
   `recorded-cases-*` entry or an owning page matches the report's versions, model
   and stack, **that entry's ladder is this turn's message**: type it, and let
   further checks confirm or correct a named claim — or drop them. The recorded
   undoing is *verification as a precondition*: **#5605 was cut twice**, the second
   run holding its ladder by turn six before source-verifying citations from another
   pin in 8-12 K-character blocks and shipping no text. A page whose Scope names
   another ref is still the case: send the ladder with its confidence labels, then
   append each recheck as a two-line correction.
   Hard stops:
   - **G1 — 3rd assistant turn:** message text with the direct answer to the
     reporter's literal question plus the track/lane classification; a pasted
     traceback, one config value or one registry listing is enough, so do not hold
     the answer for the mechanism.
   - **G2 — 8th assistant turn:** the full ranked answer (each cause with its one
     check, cheapest action, version scope, missing artefacts). Afterwards a tool
     call only confirms or corrects a named claim: one per turn, a couple of lines of
     reasoning. Text that names a decisive fact and closes with "one more confirmation
     read" is **not** G2: a recorded run wrote text on all 13 turns that way (43 K of
     reasoning against 2.9 K of text) and ended on an empty final message.
   - **G3 — 12th assistant turn:** answer text only; no reasoning, no tool call,
     no "one more thing".
   - **A new user turn restarts the gates.** The reporter's own follow-up — a retry
     on another version, a second pod's log, a different error — makes turn 1 of that
     segment the **delta** answer: what the new paste changes about the lane, per
     artefact, before any fresh source work, with `rg -n '<the new token>'` against
     `references/` as the one call that finds the page already holding it. Neither the
     earlier answer nor the new paste substitutes for it (recorded #7352: 18 tool
     calls re-deriving a chain two pages already carried, no message text;
     `delivery-discipline.md` §*follow-up turn*).
   - **The gates cap cumulative output (~45-50 K characters of reasoning+text), and
     the turn index is the weak part: your own count drifts 2-3x low.** #6529 wrote
     "I'm at turn 7" on its 15th turn, #6243 "I'm at 6" on its 14th, so G1-G3 never
     fired as written. Trigger on evidence instead of the number: the turn you can
     state a cause with its one check is the turn the answer is owed; if you do
     count, count the `tool` results in the transcript, not from memory
     (`delivery-discipline.md`, budget arithmetic).
   - **When the discriminator lives only on the reporter's machine, the ranking *is*
     the deliverable.** If the check that separates your top two causes needs their
     plog, their two serve commands, their `pip show`/`env`, or a re-run, nothing you
     can read locally closes it — so the turn where you notice that is the turn to
     send: verdict, ranked causes each with its one check, and the request list. A
     page that answers your paste and then lists reporter-side artefacts has already
     handed you the message: type it, do not go execute its investigation (#13175
     read both routing pages on turn 3, spent 18 more turns on a program only the
     reporter could run, and shipped **zero** text; `delivery-recorded-sessions.md`).
   - **Reasoning is a scratchpad, a few lines per turn:** what the last result
     proved, the single next check, go. It spends the **same output budget** as the
     answer. A block that enumerates the answer's sections, counts turns against
     G1/G2, or says "I have enough" / "let me write the answer" *is* the answer — and
     the recorded way to ship nothing is to reason "the gate has passed, I must send
     the deliverable" and then send empty text plus another tool call (#6243: 15
     turns, 67 characters of text, cut inside the 14 K block holding the ranked
     answer). Its second form is the **hypothesis sweep** — three or more
     "alternatively / another candidate" branches in one breath: two candidates is
     already the deliverable's shape, so send the pair and run the one check that
     separates them (`delivery-discipline.md`, hypothesis-sweep rows).
   - **Budget in characters per block, not turns** — the count-free form of the rule
     above: test **message text ≥ a third of the reasoning written so far**, keep each
     reasoning block ~1.5-4 K characters, and once a message can name the
     discriminator the **next** message is the ranked answer built on it (#13527 hit
     the cap with no draft at 43.8 K reasoning / 2.2 K text; #587 and #11680 each held
     the decisive datum by turn 8 and kept reading; `delivery-discipline.md`).
   - **The paste can already contain the whole answer surface.** An argparse error
     prints the complete valid choice list, a log line can name its own thread and
     owner, and a traceback can name the class *and* the rule
     (`'DummyModelLoader' object has no attribute 'get_all_weights'` names the method
     only `DefaultModelLoader` has; the MLA `queries per kv … only support {32, 64,
     128}` assert states the rule and leaves only the TP-split arithmetic). Then the
     verdict is owed at G1 with no source read: #4796 had its answer by turn 4 and
     spent 32 turns to the cap emitting **no message text**; #10962 reasoned the whole
     answer and typed it on the last turn, cut mid-sentence. Test at every turn: could
     I write the first paragraph now? — if yes, the tool call goes in a message that
     already carries it.
   - **A new hypothesis that needs a new code area is not this session's job.**
     Put it in the answer as an open question with the one command that decides it.
     For an accuracy/corruption or topology report a ranked list plus an ablation
     matrix already *is* the full answer, and a **mechanism the paste cannot carry**
     (*which* scheduler step mixed decode with prefill, *which* phase refused a comm
     resource) is a fact about the reporter's run that no local source read produces:
     #14506, #2934, #1369, #5947 and #8350 spent the rest of their budget there after
     the answer existed and shipped nothing or 185 characters, and #7774 derived *how*
     an MTE "DDR address out of range" arises inside a CANN `NonZero` listed as
     `unavailable` (45 K of reasoning against 39 characters of text) instead of
     shipping its pin's version lane
     (`references/custom-ops-runtime-faults/hybrid-ssm-mask-nonzero-fault.md`).
   - **A paste that stops in silence is answered, not investigated.** When the log
     simply ends (no traceback, last line inside init), the cause is decided by
     artefacts on the reporter's machine — process count, per-rank logs, device
     mapping — never by more source reading, so the ranked list plus "send these
     three" *is* the message and is owed on the turn the shape is classified (#1496
     spent 12 turns and ~49 K of reasoning on four mechanisms and shipped nothing;
     `references/startup-failures/multiprocess-launch-hang.md`).
   - **An ownership question is answered by naming the owner.** "The flag did
     nothing" and "the log line is missing" are gate checklists on the owning
     component — usually the upstream vLLM OpenAI frontend, not vllm-ascend — and
     take 3-4 calls; #10985 and #9583 each spent 20-30 calls inside the owning
     module with the decisive check in hand.
   - **When the deliverable is a document, the document is the gate.** Open the
     report file on the 2nd-3rd turn with the skeleton plus the first verified
     section, append **2-3 K characters per write**, keep the file ≤ ~8 K characters,
     and lead the final message with the 交付件 items (verdict, path, the three
     完整性/易用性/正确性 scores, one line per defect) — never one heredoc, never a
     re-paste of the deploy script, whose characters are output tokens from the same
     budget as the message (#11759, #11751, #11739/#11742;
     `community-test-report.md`, `delivery-discipline.md`).
   - **`/workspace/scratch/findings.md` is the answer draft, not an extra step:**
     write it at G2 *in place of* the reasoning block; once it exists the next
     turn's text is the answer (`delivery-discipline.md` §draft).
   - **A missing or mismatched artefact is itself the finding** — an empty
     environment block, a file absent from the pinned tree, a version the snapshot
     does not carry. State the gap, give the one command that closes it, stop.
   - **A byte count that "matches" is not the answer.** Re-deriving a mechanism from
     an allocation size needs inputs the report does not carry (a `config.json`, a
     group size, an expert count): ask for the line that settles it and send the
     ranked answer (#7123 died at the cap deriving DP/EP algebra for a 674 MiB
     request whose two `[EP Rank r/N] … experts` lines were never asked for,
     `references/deploy-sizing/moe-load-oom-expert-shard.md`). **The same holds for a
     tensor shape quoted from a debugger** (`变量 x.shape [5040, 1, 1152]`): it is a
     gap in the paste, not the failing allocation. #7419 spent 19 tool turns locating
     the producing `.unsqueeze(1)` and shipped **zero** text, while the two answer
     facts came from two greps (`references/lora-adapters/multimodal-vit-lora-ignored.md`).
   - **A refusal that quotes an argument value is answered by the contract, not by the
     producing line.** When the op error names the value it rejected — a mask shape, a
     seq-len list, a dtype, an out-of-range index — the deliverable is the contract,
     the one call site (or the two candidates) that feeds it, the single print or repro
     that discriminates, and the workaround; *which* line computed the value is a
     confirmation. #6573 walked the attention dispatch before finding the mask in the
     model file, #10757 spent most of its budget on whether the encoder gate could ever
     be true, and #10372 shipped no revision while choosing between two producers of a
     `numAcceptedTokens` overflow — all three held the contract by their 3rd–5th turn
     and then shipped **zero** text
     (`references/custom-ops-runtime-faults/fa-tiling-argument-refusals.md`,
     `gdn-conv1d-accepted-tokens-varlen.md`). **Dead-branch variant:** a gate in the
     snapshot that nothing sets is a version-skew fact to *ask* for — the reporter's
     build differs from the pin — not a mechanism to derive.
   - **Truncation eats the tail, not the head — and the body has a size too.** One
     minibatch showed both halves (#1255 typed no message text across 24 tool turns
     and shipped nothing; #3170 kept its verdict and ranked causes but lost the
     action ladder's last commands), so type the ladder before the scope block:
     each action directly after its cause, the closing scope block 3-5 lines. With
     no log pasted: 3-4 causes with one check each, one ladder of ≤6 lines,
     precedents as one line of `#id: fact`, 3-5 lines of scope (≈3-4 K characters);
     a fifth cause is an open question with its one command.
   - **Every assistant turn carries message text, not only a tool call.** A turn
     whose whole output is a tool call leaves nothing for a truncation to keep;
     #5295 emitted *zero* text on all 16 turns, #16563 on all 20, #8370 narrated
     next actions for 14 turns while its last turn held the answer. Carry one or
     two sentences of the current answer draft in the same message as every tool
     call — verdict line first, then the evidence being checked — so G1/G2 need no
     special turn. A *status note* is not a draft: the sentence must state the
     current verdict even when the next call may overturn it. The mechanical test
     at any turn: no message text yet names a cause plus a `path:line` or `#id`,
     so nothing has been drafted (`delivery-discipline.md` gates).
2. **Classify before investigating.** For a startup log, place it in lane A/B/C;
   for a runtime report, decide whether it is a device-op fault, an accuracy/score
   gap, a corruption report or a performance claim. The signature index
   (`references/first-checks/signature-routes/index.md`) routes to the page holding the
   row for a specific signature; the track-level routes are the table in
   `references/dispatch/classified-shapes.md` (*Support entry points*). The
   classification decides which code and which evidence even matter. Do not open
   a deep source walk when a config value, one re-run flag or a health tool can
   settle the question.
3. **The deliverable is a ranked diagnosis the reporter can act on**, not a code
   walk: per hypothesis, the evidence you have, the one command or log line that
   confirms or refutes it, the cheapest fix to try, and what to send back.
   **A repair that only removes an exception is not a fix.** For a downstream copy of a
   library body, a `getattr`/`try`/`except` guard starts the service and leaves the
   numerics unvalidated (reporter-confirmed #11991: the `attention_type` guard made
   `vllm serve` come up and broke image recognition), so name the structural repair and
   state the acceptance check — one real request whose output is compared with a working
   run — rather than offering the guard
   (`references/upstream-downstream-drift/patched-forward-api-drift.md`, rule 6 for the
   claim label).
4. **The reporter's machine is the only environment here.** You cannot inspect
   it, and local NPU state (healthy, absent, a different card) is not evidence
   about it. Prefer checks the reporter can run — config value, env var, health
   tool, one re-run with a flag — and when the report is hardware-shaped, name
   the artefact that settles it (plog, tiling log, raw request/response dumps,
   `npu-smi info`). When that artefact **is** a process on a card ("an extra
   worker process from another card appears on a certain NPU"), it is the PID's own
   `cmdline` and `/proc/<pid>/environ`: a stray PID is attributed before it is
   explained, and an engine that keeps serving is exonerated by the attribution
   rather than by an argument (`references/deploy-sizing/node-processes-and-device-attribution.md`,
   reporter-confirmed #12338).
5. **Pin evidence to the reported version.** Read `/workspace/repos/SOURCES.json`
   before citing code: it records repo, ref, commit, basis and coverage cutoffs.
   Every code claim gets `path:line` plus the commit/ref it came from. State
   recheck conditions when a constant, table or log line is version-specific. A
   report citing `docs.vllm.ai/.../latest/…#<section>` is quoting a doc that may
   postdate the pinned release: check that section exists in the pinned tree,
   answer inside the pinned version, and if it is absent say so and ask for the
   text rather than assuming what it says (`latest` is main's build; the tag a page
   prints is substituted at build time —
   `references/tutorial-verification/docs-version-space-and-image-tag.md`, #16954).
   When the title names one release and the body's launch log another, say which one
   you verified and give the one command the reporter runs on *their* tree; when a
   newer pin in the snapshot already carries the repair, quote it. A reference page
   states the pins it was verified at: when those equal the refs in `SOURCES.json`,
   its `path:line` are citable as they stand — re-grep only what the page marks as
   version-specific. **The contract covers the whole dependency and image stack, and
   it decides which rung comes first**: a dependency's patch release (a release can
   pin the *broken* one), an image recipe that installs its own pairing, a mixed
   self-built trio, an old tag and the direction of the comparison, and a capability
   that worked on an older release (a regression, not a mechanism) — one grep and one
   rung each, all in `references/evidence-workspace/version-contract.md`. Open that
   page when the env dump's `torch_npu==`/`transformers==` line, the image tag, or
   the reporter's own older-release control is part of the ranking.
6. **Label the strength of every claim**: reporter-confirmed > verified in the
   pinned source (`path:line` at commit X) > precedent in the issue history
   (quote the issue id) > unverified hypothesis / recalled tool default. Never
   present a recalled default (sampling params, tool flags, CANN behaviour) or a
   matching-looking number (tensor count, hostname, line number) as proof. A
   maintainer's clean run on different hardware is a data point, not a
   refutation of a reporter-confirmed fix on theirs; say what was verified on
   which hardware. **A `path:line` at commit X is a claim about X, and a *defect*
   claim expires as readily as an absence one**: a page can prove a branch or an
   interface missing at an older pin that has since gained it, so re-run the page's own
   grep at the tree in your hand before ranking that defect. Recorded #12339 was run
   with "the generic Mooncake connector cannot describe a hybrid page" — true at
   v0.18.0, `rg -c Mamba` empty — while its pin (v0.22.1rc1) carries a full `MambaSpec`
   branch (`is_mamba_group`, `_append_mamba_transfer_meta`), so the mechanism had to be
   demoted inside the answer; presence claims and absence claims both belong to the pin
   they were seen at (`references/pd-disaggregation/hybrid-state-transfer.md`).
   **A page's *rank order* can rest on an arithmetic, and that arithmetic is pin-bound
   too**: before quoting a ladder that turns on a division, a chunk split or a token
   gate, recompute it from the pin's own source — one read, and it is the step that
   rescues a wrong ordering. Recorded #6240: its page argued the two reported load
   cases died "on different code paths" because only the 1017-token request fits the
   MLAPO `<= 1024`-token branch (`sfa_v1.py:763`); but that gate tests the **step's**
   token count (`total_num_scheduled_tokens`, `model_runner_v1.py:615`), so the
   8270-token prompt at `--max-num-batched-tokens 8192` is chunks 8192 + 78 and its
   78-token tail takes the same branch — the shared mechanism *raises* that candidate
   and reorders its A/B, where the page had used the split to demote it
   (`references/dsa-sparse-attention/concurrency-crash-with-no-log.md`).
7. **Keep reads small and exact.** Use `rg -n -C2 <symbol>` and
   `sed -n 'A,Bp' file`; reserve whole-file reads for files under ~150 lines — the
   exception is the page that **is** the protocol for the deliverable in hand
   (`community-test-report.md` 386 lines, `delivery-discipline.md` 446,
   `tutorial-command-audit.md` 513, a topic `index.md` whose conditional routes sit
   after its rules): its rules, routes and worked tables sit past the first window, so
   read it whole or list its `^## ` headings first. `tutorial-verification/index.md`
   is the exception now that its *Route table* opens the file, so `sed -n '1,60p'`
   there reaches every route; a window like `sed -n '1,80p'` of the 386-line protocol
   or `'1,120p'` of that index is how the #11759 rerun walked past the worked table
   for its own row. A
   20-40 KB full-file read of an AscendC kernel header buys one fact and costs
   the budget for several conclusions. Listing, dumping and a handful of
   search-command traps each cost a recorded run whole steps; the full text is in
   `references/evidence-workspace/case-layout.md` (§*Search-command traps*), and the
   one-line forms follow because they recur on every case:
   `skill_view` with `file_path` also returns the whole link inventory — read pages
   from the skill root with `sed -n 'A,Bp'`; **type a grep as `rg -n '<pat>' <one dir>` and
   never add an `r` flag** — `rg -r` is `--replace`, not "recursive" (`-rn`, `-rln`, `-r ''`
   rewrite or dump whole lines; an empty pattern matches every line), so a result whose matched
   text is missing or replaced by `n` is that accident, not a finding: re-run without the flag
   before quoting it, and never read a `self.…n == "align"` line as source (recorded #13863
   reached for it **four times in one session** with this warning already in context, and
   `rg -n '_update_states_after_model_execute' -r ''` printed a `self.(sampler_output…)` line); a pasted message is
   often split across adjacent literals, so grep the shortest clause (or `-A1`); a
   `--glob '!<dir>/**'` exclusion is matched against the path you passed; a piped
   `sed -n 'A,Bp' | rg -n` numbers the stream (`A + n - 1`); scope every grep to one
   repo; never list a directory or dump a whole file, with **one** exception: a
   single `ls -la /workspace/attachments/<issue>/` (or `cat ATTACHMENTS.txt`) as soon
   as the report names a log you have not located. Attachment dumps carry no settled
   extension — a recorded segfault case kept its scripts as `.txt` and **both node
   logs as `.log`**, so a `*.txt` glob showed 3 of 5 files and the session asked the
   reporter for logs the case view already held — and a one-level
   `search_files (target='files')` is capped (returned `total_count: 102` twice), so
   neither a glob nor that listing is an existence test
   (`references/evidence-workspace/case-layout.md` §*Attached log dumps*). Then open
   the files the issue links; a manifest grep is not an existence test either.
   **A `| head -N` on a reference grep can cut the page that owns the case.** These
   pages sit in large topic directories, so `rg -n -i '<feature token>'
   references/<topic>/ references/<other>/ | head -60` can spend all sixty lines in
   the first file and drop the hit that matters: recorded #10961 ran
   `rg -i 'mtp' pd-disaggregation/ accuracy-benchmarks/ spec-decode-mtp/index.md |
   head -60`, got sixty lines of `spec-decode-mtp/index.md`, and lost
   `pd-disaggregation/hybrid-state-transfer.md:165` — the page holding that exact
   case's refuted top rank — then re-derived the recipe from the tutorial and
   re-shipped the refuted claim. `rg -l` over `references/` first, pick the page by
   its heading's issue id, its title and its stack, then read it; raise the cap only
   after reading the file list.
8. **The prior-month corpus is not evidence of absence, and a lexical accident is
   not "no precedent".** Search titles with **one** distinctive token — an op name,
   an error string, a numeric code, a fault kernel, a model id — never the failure
   category and never a word the environment dump already carries (`suffix` ←
   `ATB_SHARE_MEMORY_NAME_SUFFIX=`, `mtp`, `ngram`, `spec`, `npu`, `acl`, and any version
   placeholder such as `0\.0\.0`, whose `-b` search returns every `0.0.0.0` bind address in
   the corpus — #16559 spent a turn on 573 hits that way) or a
   mechanism phrase (`graph capture|capture size|图捕获`). Two recorded forms: #9278
   searched `-i 'suffix|507035|aivec|MPU address'`, got "207 hits / 207 distinct
   issue ids", read the whole result as env-var noise and stopped — while #488 in
   that same output carries the reporter's own `error code:507035`, and the
   titles-only search returns eight real precedents (`suffix` +
   `full_decode_only` among them); #16581's mechanism words returned 442 hits of
   file names, while the paste's own code tokens (`507015|MTE|DDR address`)
   returned 91 and named #3263/#2424. When a pattern hits ~every file, split the
   alternation instead of discarding the survivors. **A `-l` list cannot be graded
   from its output** — it prints paths and no matched text, so "my alternation was
   a bag" is a guess, not a reading: split it per token (`search_files` on one
   token under `<skill_root>/references/`, or `rg -l '<one token>'
   <skill_root>/references/`) before writing it off. Recorded #3543 did exactly
   that to a `head -40`-capped `-l` list over `references/`, with
   `startup-failures/silent-worker-exit.md` and
   `startup-failures/max-model-len-budget.md` — the two pages owning its case — on
   it; the recorded-cases detour it took instead cost six calls. Inside
   `references/` the **model id is a listing token, not a distinctive one**: every
   case line carries its model, so `-i 'qwen3-vl|qwen3_vl|qwen3vl'` over
   `recorded-cases*` returns lines from every family page of the index (accuracy,
   startup, install, device, pd-kv, memory), all of them other symptoms' ladders.
   Grep the wrapper sentence, the op name or the code.
   `history/*.md` and
   lexical issue search cover only earlier issues; a current issue is never in
   them, and the ids this skill's own pages cite as recorded cases are usually
   newer than the case's corpus (a #6533 view ends at 6436 while the pages cite
   8590, 7689, 13591 …). So a `not in this corpus` from `history-body.sh` weakens
   nothing: quote the page's pin and facts instead of re-opening the id, and never
   drop the script's only message with `2>/dev/null | head` — #6533 read the empty
   stdout as an empty body, then spent a GitHub call on a 404 and disowned the
   same-pin #8590 precedent it had. A raw `head -1` of
   `/workspace/history/<id>.md` with stderr dropped is the same trap: absent and
   empty print identically, and this is the one id that is reliably absent (the
   page's own case is newer than the corpus) — #10495 spent a call there after its
   owning page had already matched its body line for line. Make one grep for the
   distinctive token (op name, error string, dataset
   name), read 2-3 hits, then stop: **for a corruption report the distinctive token
   is the pasted output itself** — the repeated character run (`-b '!!!'`) or the
   exact answer text — not the category word. #3547 searched titles for
   `乱码|garbled|重复|感叹号|!!!` and got three V0-era hits; `-b '!!!'` also returns
   #1754, whose **title names the mechanism** ("The ACLGraph graph model does not
   support pipeline parallelism") while its body is `Output: '!!!!!!!!!!!!!!!!'` for
   every prompt — the neighbour a symptom-word title search can never reach. Precedent supports a hypothesis, it does not
   replace reading the source at the pinned commit. Reach
   for the `github` issue search only when the corpus token has **no** hit: it returns
   whole issue bodies
   paginated (17 K characters in #9986 with `display_complete: false`), so read the
   first page and stop — and skip it entirely once the corpus search returned hits or
   the pinned source already answers the question (#10074 spent one 17 K-character
   `/search/issues` page after its source evidence was complete, then discarded it). **Pick the bodies by whether
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
   examples and the API surface. One more exported tree settles device-open
   pastes: **`cann/runtime`** — `src/platform/platform_info.cpp` is the FE code that prints
   `RealPath:Path '…/data/platform_config' does not exist.` and holds
   `PLATFORM_RELATIVE_PATH = "../data/platform_config"` relative to the loading `.so`'s own
   directory, so a `Get ACL JitCompile default value failed` / `File path[] is not valid`
   paste is two reads there, not a `cann/ge` hunt. The genuinely absent pieces are named in
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
11. **A log sentence — or a `/health` 200 — is not a diagnosis: some of them are
    fixed wrapper text.** vllm-ascend
    logs canned advice inside `except` blocks and re-raises, so the sentence names a
    cause the code never checked — the `ACLgraph has insufficient available streams
    … Recommended solutions` block prints for *any* capture exception
    (`model_runner_v1.py:3553-3565` at v0.11.0rc0, `:3143-3152` retCode-gated at
    v0.12.0rc1, absent at v0.18.0rc1, back `207008`-gated by v0.21+: re-grep per
    version) — so **classify the appended error, not the wrapper**: Step 2 of
    `references/startup-failures/acl-graph-capture.md` sends a collective/transport code
    (`EI0006 … Remote Rank did not send the data in time` inside an `HcclAllreduce`, the
    `[Create][DestLink]Create Dest error!` beside it), a plain Python exception, a
    synchronize refusal (`107027`), a capture-mode copy (`107030`) and an aclop refusal
    each to its own page, while `capture-stream-table-exhaustion.md`'s stream arithmetic
    applies only when that inner line is the capture-begin resource refusal (`207005` /
    `Alloc sq cq fail` / `stream_id=…`). Its "Recommended solution 2" is a lever a
    pooling serve refuses, so price a logged recommendation against the engine's own
    gates before repeating it
    (`references/startup-failures/capture-stream-table-exhaustion.md` §4). Before
    adopting a log's own explanation, `rg -n '<fragment>'
    vllm_ascend/` — and in the upstream `vllm/` tree when that comes back empty, since
    a message naming triton/torch internals is usually upstream's: `Failed to import
    Triton kernels … please make sure your triton version is compatible` is vllm's
    `logger.error` around an optional `triton_kernels` import, so its advice names the
    action the reporter already tried and the crash is elsewhere
    (`references/image-dependency-faults/triton-kernels-import-error.md`) — and read
    the emitting site: an `except Exception` that appends
    `{str(e)}` and re-raises carries the inner error as evidence, not the wrapper.
    Same family as procedure step 3's sync-point trap: where an error surfaced is
    not necessarily its owner. **One member of the family is a warning that names a
    harmless feature**: `Failed to import vllm_ascend_C:<tail>. Sleep mode will be
    disabled.` reads as a sleep-mode notice, but an extension that never imported
    registers **no** `_C_ascend` op, so the `AttributeError: '_OpNamespace'
    '_C_ascend' object has no attribute '<op>'` that follows — at a request *or* in
    the memory-profiling dummy run — names whichever op was reached first, never an
    unsupported one (#3342/#3090 `weak_ref_tensor`, #6527 `moe_gating_top_k`). Read
    that warning's **tail** before ranking the op: `No module named …` = no `.so` in
    the image, `: undefined symbol: …` = present and refused by a SoC source-list
    gate (`references/image-dependency-faults/custom-op-extension-missing-from-image.md`,
    Step 0). CANN prints the same kind of wrapper under the code:
    `[Error]: Failed to apply for memory.` / `Check the remaining storage space in the
    hardware environment` + `EL0006 … resources are insufficient` is the canned text of
    a *resource-alloc* refusal (`207005`), so read the `[SqCqManage]Alloc sq cq fail …`
    line under it before treating such a paste as an HBM, disk or container-size report
    (`references/startup-failures/capture-stream-table-exhaustion.md`).
    The rule covers *advice* too: `Cannot run aclop operators during NPU graph capture …
    please try to set torch.npu.config.allow_internal_format = False` is torch_npu's
    capture guard quoting a suggestion, and vllm-ascend assigns that option `True` at
    import (`worker/model_runner_v1.py:138` at v0.11.0rc0, `:154` at v0.18.0rc1), so it
    cannot be applied from the launch shell; the op named in the message is the subject
    and the FX trace beside it names the owner
    (`references/startup-failures/aclop-op-during-graph-capture.md`).
    The same rule covers the multiproc wrapper, in the one direction that changes what
    you ask for: `WorkerProc failed to start.` is `logger.exception`
    (`vllm/v1/executor/multiproc_executor.py:749` at vllm v0.14.1), so the rank's whole
    stack is printed **below** it, every line under that same
    `[multiproc_executor.py:749]` prefix — ask for the block below the line, not above
    it. When the paste cannot carry that block at all (a screenshot whose `*.ocr.txt`
    is noise, a title-only report), the wrapper is all the evidence there is: classify
    from the serve path's own tokens — a quant variant in the model directory
    (`…-W8A8`, `…-w8a8_dynamic`) decides the lane — and put the traceback request in
    the answer. Two quant lanes start there, and the launch line separates them before
    any page read: **no** `--quantization ascend` / `quantization="ascend"` anywhere in
    the command is the gate that never fires (the msmodelslim export's `config.json`
    carries no `quantization_config` on purpose, so upstream's auto-detect cannot fire
    and a dense engine meets int8 tensors — the quant gate's §B
    (`references/startup-failures/quantization-method-gate.md`), recorded #2015, where
    the pin's `DeepSeek R1 ✅` row says nothing about it), while the aux-key family
    (`…weight_scale`, `w2_weight_offset` —
    `references/startup-failures/weight-scale-keyerror-quant-variant.md`) needs a
    traceback that names such a key.
    Reading `supported_models.md` first is the recorded detour: the flag, not the model
    row, is the discriminator. **The oddest flag in the command is not a
    substitute**: `--mm_processor_cache_type shm` ranks only with its own
    `Data for address:id …` string. Recorded #7744 ranked that flag first on a
    load-phase `KeyError: 'layers.37.mlp.experts.w2_weight_offset'` and its resolution
    was the newer tag (reporter-confirmed `v0.17.0rc1`) —
    `references/startup-failures/silent-worker-exit.md`. **Check a suspect env var
    against the pin's own default before calling it the reporter's knob** —
    `rg -n '<ENV>' vllm_ascend/envs.py` — because a launch line that restates the
    default carries no information about the fault (`VLLM_ASCEND_ENABLE_MLAPO=1` is the
    default at v0.14.0rc1 `envs.py:95-99` and v0.18.0rc1 `:82-86`, and the tutorials
    export the same), while `=0` on that key is still a legitimate one-launch ablation;
    recorded #7015 shipped "你自己加的非默认开关" for exactly that line
    (`references/hccl-and-hardware/mc2-dispatch-profile-run-hang.md`, refuted rows).
    **A 200 from `/health` is the same kind of wrapper in the one place it decides a
    verdict**: every "it hangs while health stays 200" report rests on it, and at vllm
    v0.23.0 the route only reads two frontend flags — `AsyncLLM.check_health` raises
    `dead_error` when `errored`, and `errored` is
    `engine_core.resources.engine_dead or not is_running(output_handler)`
    (`vllm/v1/engine/async_llm.py:900-903,1045-1055` behind
    `entrypoints/serve/instrumentator/health.py:30-34`) — while the executor's real
    probe, `check_health` → `collective_rpc("check_health", timeout=10)`
    (`multiproc_executor.py:491-493`), has **no caller** (`rg -n 'check_health' vllm/`
    returns the definition, the `AsyncLLM` method and the route) and could not be
    answered anyway: `worker_busy_loop` (`:969-997`) serves one RPC at a time. So a
    worker blocked mid-step looks healthy for as long as the RPC deadline, and the 2 ms
    200 is not evidence about the engine. One grep settles it per pin — the route has
    drifted (vllm v0.18.0 carries the same `errored` pair at `async_llm.py:874-877`;
    an older release does reach a 10 s `check_health`,
    `references/hccl-and-hardware/rpc-watchdog-timeouts.md`) — and then the answer says
    "health is not a liveness probe here" once, with the lines, instead of ranking a
    health defect or asking for another health poll:
    `references/scheduler-admission/background-helper-death-wedge.md` (recorded #18691,
    v0.23.0, a dead Dynamic-EPLB planner subprocess; it also carries the
    `platform.py:1275-1282` floor that makes a launch-side
    `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=8` inert).
12. **A crash inside a native module the image installed is an environment
    question, and it is answerable without that module's source.** Three forms — a
    `python -c "import <pkg>"` glibc allocator abort, `std::logic_error` /
    `basic_string::_S_construct null not valid` at engine start, and
    `ImportError: …/_ops*.so: undefined symbol: …Ss` from a prebuilt wheel
    (`std::string` ABI split, decided by the env dump's `cxx_abi_{0,1}` root,
    reporter-confirmed on #1144) — all route from
    `references/image-dependency-faults/index.md`; do not walk the module's source.
    Cap ~6 calls (#5609, #10943 burned the session there). **The fourth import-time
    form is `Illegal instruction (core dumped)` at `import torch` on an aarch64 host,
    and it is the CPU core's ISA level rather than the module**: an ARMv8.0 core
    (Phytium D2000 / FT-2000-class FTC663) against a wheel built on an ARMv8.2
    baseline, while Kunpeng 920 / D3000 are ARMv8.2 and import the same image fine —
    reporter-confirmed on #13936 (`v0.23.0rc1-310p-openeuler`, `torch==2.10.0` /
    `torch-npu==2.10.0.post2` at `requirements.txt:16-17`; 「d2000只支持armv8 …
    torch.2.10默认使用armv8.2的指令集编译的」), where the reporter's own control is
    `torch` 2.9 importing on the same board. One host-side round trip decides it —
    `dmesg -T | grep -i 'trap invalid opcode'` names the `.so`, `/proc/cpuinfo`
    `Features` / `LD_SHOW_AUXV=1` names the missing extension — and the rung is the
    last image line before the 2.10 bump (`release_notes.md:316`, v0.20.2rc1 at that
    pin), **never** `pip install torch==2.9.0` under the newer `torch_npu`. Do not
    confuse it with the aicore's `Illegal instruction, which is usually caused by
    unaligned UUB addresses` (`references/hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`)
    — same three words, runtime device fault, no import →
    `references/image-dependency-faults/host-cpu-isa-sigill-at-import.md`. The fifth
    form is an **ACL/GE bring-up failure** at device open — `SetPrecisionMode … AclSetCompileopt(
    aclCompileOpt::ACL_PRECISION_MODE …) error code is 500001` +
    `PlatformInfoInit … SocVersion <name>` + `GEInitialize failed`, ending
    `ERR99999` with no `vllm_ascend` frame. Owner, in this order: a **containerized**
    run whose `docker run` already carries the documented `--device`/mount list takes
    that same command plus `--privileged=true` as its first re-run (reporter-confirmed
    #302: 「已解决，加个--privileged=true即可」, driver unchanged, one variable); then the
    driver/firmware × image-CANN pairing, with the printed SoC name read against
    `npu-smi`. No precision/dtype flag is ever the cause →
    `references/startup-failures/acl-init-500001-and-soc-version.md`. The **sixth** form is an
    **ATB op-create refusal**: `RuntimeError: LinearOperation CreateOperation failed!` (also
    `GroupTopkOperation …`) out of a `torch_npu._npu_*` call — at a 310P engine start it is the
    unconditional `_warm_up_atb` (`worker_v1.py:332,337-341` at v0.11.0) meeting an ATB/NNAL build
    that cannot create the op, so the caller's shapes, the KV flags and `--enforce-eager` are all
    spent levers and the wording exists in no export; three checks (one-op repro outside
    vllm-ascend, the `nnal/atb` identity, the SoC stamp) plus the pin's own `-310p` recipe answer
    it — `references/custom-ops-runtime-faults/atb-create-operation-failed.md` (#5170). **The same phase *without* `500001` is a file
    question, not a privilege or pairing one**: `RealPath:Path '<…>/<arch>-linux/lib64/
    ../data/platform_config' does not exist.` + `InitializePlatformInfo:File path[] is not
    valid.` ending `RuntimeError: Get ACL JitCompile default value failed.` names the FE `.so`'s
    own directory plus a sibling `data/platform_config`
    (`cann/runtime/src/platform/platform_info.cpp:27,955-981`), and `RealPath` prints that
    sentence for *any* `realpath()` failure including EACCES — so check the directory as the
    job's UID inside the worker container, and do not carry `--privileged`, the driver pairing
    or a `lib64/data` symlink into it
    (`references/startup-failures/platform-config-file-missing.md`). **The seventh form is a
    tiling line whose verb is `not support`** — `E89999 … When layout is TND, not support Page
    Attention[FUNC:RunBigKernelTilingWithParams][FILE:prompt_flash_attention_tiling.cpp]` under
    `call aclnnFusedInferAttentionScoreV3 failed, error code is 561002`: the sentence names a
    mode × cache-kind combination the installed toolkit's tiling has no path for, so the reporter's
    `ascend-toolkit/latest/version.cfg` against the pin's own documented CANN floor
    (`docs/source/installation.md:15`) is rung 1 and the caller's arguments are not — reading the
    launch pair instead costs the fix (reporter-confirmed #4373, CANN 8.3;
    `references/custom-ops-runtime-faults/tiling-capability-refusal-cann-release.md`). **The same
    op and code with dims printed** — `queryD(256), keyD(256) and valueD(256) must be same equal
    192/128/64 … [FUNC:CheckInputShapeWhenLayoutIsTND]` — is not that form: it compares the
    model's `head_dim` (Qwen3-Next is 256) with the tiling's set, so the owner is which batch
    state reached the paged-TND call (`ChunkedPrefill`; the paste's own `Dumping scheduler output`
    line proves it) and the upgrade rung is inert for an image run whose `Dockerfile` base already
    is the floor — `references/custom-ops-runtime-faults/fia-tnd-head-dim-refusal.md` (#4135).
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
    **The third mechanism is in the emitting macro, and it decides the code you would
    otherwise rank on.** For `RuntimeError: <op>:<file> NPU function error: call aclnnX
    failed, error code is N`, `N` is not necessarily this call's status: PTA's
    `NPU_CHECK_ERROR` overrides the printed code with `AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)`
    (peeked, never cleared) and `getDeviceErrorMessage` supplies the text — so a
    capture-window `107030` can be printed two hours later by an unrelated aclnn op, with
    that op's own words around it, and the op's real status is masked (the macro only fires
    when the call itself returned non-zero). Test the number against what the named op can
    mint (`107030` = `ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT`) rather than against the
    timestamp gap, and deliver the plog at the failure second, not the capture block —
    `references/hccl-and-hardware/latched-runtime-error-masking.md`.
14. **A "how do I enable F" question is a capability question, and the verdict comes
    first.** When the report asks how to turn a feature on for a model/hardware pair
    and pastes no log, the reporter is asking whether it is *known to work*: give
    yes / no / partially in the first sentence, with the evidence class — documented
    as supported at the pinned ref, measured on that hardware, covered by
    `tests/e2e/**` (and the nightly yaml's `npu_per_node`), or only by an older
    precedent issue. "No validated recipe for this combination" is an answer;
    a launcher synthesized from the constraints the code enforces is not, and it
    provokes the follow-up "但有人在 A2 上成功跑通过吗?" (#8984). **Answer that
    follow-up in the same message, and run its search while establishing the
    verdict**: `scripts/history-titles.sh -i '<feature>|<alias>|...'` enumerates the
    precedent family, so report what the hits are — at #8984, 10 issues, 3 RFCs and
    7 failure reports, no success report for the reporter's weight set on any SoC —
    with the snapshot's limit (titles/bodies only, no comments or PRs), i.e. "nobody
    in the searchable corpus reports success", not "impossible". The recorded
    #8984 re-ask came a day *after* a verdict-first answer that ran that search only
    at turn 40 of 45, so late that it could not shape the answer. Any launcher you
    still hand back must also fit the reporter's cards, by the model tutorial's own
    node count for that weight set, **and its `--gpu-memory-utilization` with the
    HBM the paste's own `npu-smi` rows already show held by other processes** — the
    flag is a fraction of *total* device HBM, so on an occupied device a *raised*
    ratio buys a `No available memory for the cache blocks` stop instead of a fix:
    hand back the reporter's own value and put the free-chip step
    (`ASCEND_RT_VISIBLE_DEVICES=<idle id>`, or stop the other process) in the same
    paragraph (`references/pooling-embedding/embedding-checkpoint-as-generate.md`
    step 1, recorded #5549 replay). Then map each
    enforced constraint to the error text it raises, so the reporter can place their
    own two errors, and stage the experiment one axis per launch. Three turns,
    `references/feature-enable-recipes/index.md`. Two traps: a pinned doc's narrower
    hardware scope is the project's position for that version (quote it, do not
    overturn it with an older issue), and a comment saying a sub-feature is unsupported
    is often a branch the code already takes (`sfa_cp.py` vs the table's MLAPO ✅).
    A **weight-layout capability** report is the same shape with the premise inverted:
    "910B3 是否支持权重 nd 到 nz 的转换，提高 matmul 的性能？0.11.0 的代码好像只支持 310P"
    — `is_310p()` guards read both ways in this tree (`if not is_310p():` runs the NZ
    cast on 910 and skips 310P), so classify the polarity per site and give the
    per-path verdict before any patch, and grep `envs.py`/`ascend_config.py` at the
    reported ref because `weight_nz_mode` does not exist before ~v0.20
    (`references/feature-enable-recipes/weight-nz-layout-by-soc.md`).
    The same table settles the **"<model> 离线推理报错 / 跑不通"** shape when the
    release is named and the only evidence is screenshots: one `rg -n -i '<family>'
    docs/source/user_guide` (the matrix path moved — `support_matrix/supported_models.md`
    at v0.9.2rc1, a flat `supported_models.md` at older refs), and a `❌` row with a
    tracking issue *is* the verdict, while `| Need test |` is a coverage gap, not a
    refusal. Say the op fault is the manifestation and that the reporter's next
    attempt will surface *different* errors (recorded #6257: four more screenshots
    titled 其他错误 after the ❌-row answer), so the ladder must not read "then it
    will work" — `references/feature-enable-recipes/unsupported-model-row.md`.
    **A deployment-recipe question ("现在支持 k8s ray 插件做分布式推理么") is answered
    from the docs tree at the report's ref, and an absence borrowed from another pin is
    not a finding.** One `rg --files docs/source/tutorials | rg -i '<feature>'` decides
    recipe / no recipe, and the page it finds carries the support statement in its own
    contents (docker + manual `ray start`, no KubeRay chapter) — quote that with the
    pin. `tutorials/multi_node_ray.md` is **absent** at v0.10.2rc1 (`048bfd5`) and
    **present** at v0.11.0rc1 (`c5fe179`, 182 lines), so recorded #8156 shipped "this
    tag has no Ray deployment tutorial" in its first message and retracted it a turn
    later; every absence recorded in this library is that pin's
    (`references/startup-failures/ray-npu-resource-label.md`,
    `references/tutorial-verification/missing-guidance-and-contributor-docs.md`).
15. **A working sibling path in the reporter's own evidence constrains the cause.**
    When the same box fails on one entry point and works on another — offline
    `LLM()` vs `vllm serve`, one flag, one model, one card — a mechanism that also
    predicts the working path failing is not yet the answer: name the difference
    first and rank the mechanism below it, with the log line or ablation that
    separates them (#1534 ranked a plugin fix above the working `vllm serve` path and
    left the follow-up unanswered; which path compiled and which process loaded the
    plugin: `references/image-dependency-faults/mindie-turbo-graph-break.md`).
    **The same test governs your own ablations and every control the report offers**:
    an ablation only rules out a phase it actually removes and its own log line has to
    be in the paste (`--enforce-eager` is neither universal nor free, and
    `--async-scheduling` with `method: mtp` makes an "eager" rerun the same path
    twice); a workaround the report's own prose names is rung 1, but only after
    diffing it against the failing launch line — one that also drops other flags is
    a multi-flag change, not an ablation (#17797's "fixed" line moved three); a
    tested-configuration matrix in the body is one verdict line per row, with the rows
    that work as the controls; a **ranked diagnosis the report pastes** (a
    问题分析报告 / 根因分析 block with a 建议排查 list) is a hypothesis table: grade each
    named lever against the pin (`scripts/env-knob-liveness.py`; `ABSENT` ends it) and
    each cited log line's emit site first (#10700: a name no tree defines, and a
    `warning_once` on the default config); and a control the reporter only *asserts* —
    another platform, a healthy node, a direct `curl`, a repro over lower-level primitives —
    is a premise, not a finding, so build the ranking so it holds if the premise is
    false. `references/dispatch/controls-and-ablation.md` carries the seven forms, the
    pin splits that make an ablation inert, and the recorded cost of each (#2763,
    #13964, #10910, #9447, #15527, #13688, #10700).
16. **A release candidate's experimental mode is a version question before it is a
    mechanism question.** When the mode printed its own banner
    (`FULL_DECODE_ONLY compilation enabled on NPU`, `PIECEWISE …`), every rank finished
    capture, and the *first* request then stalls into the 300 s watchdog, the paste
    has already closed the capture lane and the "setting never applied" chain; what
    remains is model×mode support at that ref or a defect fixed after it. Three
    checks, then deliver: (a) which models the pinned tree validates for that mode —
    `rg -n 'cudagraph_mode.*FULL_DECODE_ONLY' <ascend>/tests/`, then confirm a
    `.github/workflows` file runs that test **and sets the mode it names**, and read
    the markers above the case: a wired file is not coverage when the case is
    `@pytest.mark.skip`-ped (v0.11.0rc1 and v0.11.0rc2 `a2e4c3f`:
    `tests/e2e/singlecard/test_aclgraph.py:103-108` skips "it fails intermittently
    and needs investigation" on the file's only `FULL_DECODE_ONLY` case, so that
    line's full-graph coverage was a multicard file no workflow referenced) —
    v0.11.0rc1 also has wired `test_mtp*_correctness_full_graph` cases that pass
    `FULL`, which `platform.py:237-242` sends to `NONE`; §3 and the #4326 section of
    `hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`); (b) whether the
    reporter's own control
    works — a default-mode success on the same command makes the *mode* the lever;
    (c) whether the tree carries any mode-specific handling for their family (absence
    is quotable as coverage, never as the cause). A fix that lands in the next rc **is
    not in the snapshot**, so a source walk cannot find it: #4178 walked 22 turns and
    48 K characters of reasoning on that path, emitted zero message text, and was cut
    before answering, while the coverage verdict was three greps and its reporter
    later reported the bug fixed on `0.11.0rc2`
    (`references/hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`).

## Classify the evidence

Two passes. First the **track**, from the dispatch table under *Support entry points*
below (`references/dispatch/classified-shapes.md`); for a startup log that means lane
A (worker died without a per-rank traceback), lane B (worker raised a Python error) or
lane C (collective or device fault), and for a report with no symptom that means one of
the D tracks (sizing, document, capability, code-change). **An `[Installation]` / 安装
report that fails before any engine exists** — a compiler or linker log, a pip
resolution error, a missing library at import — is its own track: the paste's last
frames name the failing *stage* and the stage names the owner
(vllm-ascend's `CMakeLists.txt`/`setup.py` vs the CANN toolkit vs the image), the
reporter's CANN/torch/torch-npu trio is checked against the pin's own requirements
table before any version advice, and the cheapest working install is rung 1. The
**no-frame** variants of that track — a pairing story, the `+cpu`/build-isolation
conflict, a Python-floor annotation error, a dangling build path, the pybind11
`--cmake` rename — are written up in
`references/dispatch/no-frame-install-shapes.md`, each with its rung and the recorded
run that spent its budget there. Open that page when no failing frame names a stage. A paste that **does** carry compiler/linker frames (`multiple
definition`, `ccec_compiler`/`ascendc_pack_kernel`, `recompile_binary.py`, `collect2:
error: ld returned 1`, `setup.py:<n>` `cmake --build` error 2) goes straight to
`references/image-dependency-faults/custom-op-build-stage.md` (Step 1 stage → Step 2
two-prefix relink → Step 3 rung 1); the no-frame page holds only a one-line route
(#1774 read it whole).
Then, when the signature is specific — a log fragment, an error string, a flag that did
nothing, a title that names a mechanism — the exact row in
`references/first-checks/signature-routes/index.md` (six pages split by the shape's home
track): each row carries the lane or track it belongs to and the page that owns it. Read the row's middle cell before opening the
page; it carries the decisive local rule. A token you cannot place: one
`rg -n '<fragment>' <skill_root>/references/` searches every row **and** every page
body, which is the version that pays (one distinctive fragment — an alternation of
family names returns a path listing, procedure step 6). **Run it before any
`/workspace/repos` grep**:
several recorded signatures are emitted by build tooling that ships in no pinned tree
(the `of bin package … run.sh` SoC text, an op-store refusal), so the library — not the
source — is where they are written down (#5599 spent its 4th tool call on an empty
`repos/` grep for one of them). **Grep that text rather than picking the page
by the operator name in its filename**: an op-named page is usually the kernel/tiling
map for a *launch* fault, so an `EZ1001 Get regInfo failed` / `does not support
opType` / `not in libopapi.so` / `undefined symbol` paste is the install lane however
prominent the operator is in the title — recorded #11519 read
`custom-ops-runtime-faults/add-rms-norm-bias-op-map.md` first for a registry text
(that map is verified at another pin and answers a different surface) before the
signature row sent it to `op-not-registered-for-soc.md`.
**A traceback whose deepest frame is in
`torch/`, `torch_npu/` or `vllm/` rather than in `vllm_ascend/` is a routing
signature: search the helper's name or the exception text in the same call as the
history search, before any source read.** A **returned `None`** is the same shape
with the owner one layer up: `AttributeError: 'NoneType' object has no attribute
'sampled_token_ids'` at `update_from_output`, with no `--async-scheduling` and no
spec config, means the worker handed back `None` — the downstream `rank == 0` gate,
which equals upstream's `is_driver_worker` in every multiproc executor and diverges
**only** under `distributed_executor_backend="external_launcher"`
(`vllm/executor/uniproc_executor.py:68-120`: `rank = os.environ["RANK"]`,
`is_driver_worker = True` for all processes) — the backend torchrun/SPMD trainers
select, so only `RANK=0` returns an output: print `RANK`, and TP=1 does not test it
(`references/upstream-downstream-drift/execute-model-none-external-launcher.md`,
#960). Its **import-time form** is the same
signature and is decided arithmetically: an `ImportError` / `cannot import name`
naming a `vllm.attention*` symbol from a `vllm_ascend` module, enclosing line
`if vllm_version_is("X.Y.Z"): … else: …`, on an install whose vllm is an rc/dev/post
of that release line — the gate is exact PEP 440 equality, so the pre-release takes
the *other* generation's branch. One equality probe settles it, and rung 1 is
`VLLM_VERSION=X.Y.Z` before launch (or the paired final), not a patch
(`references/upstream-downstream-drift/version-gate-branch-selection.md`).
Those trees are frequently absent from the
export (this snapshot's pytorch is `files=1733/19378`, no `torch/_dynamo/`), so no
source walk can confirm such a frame and the library is the only local evidence
there is — recorded #7947 spent its entire 16 384-token budget re-deriving
`AttributeError: 'NoneType' object has no attribute 'size'` from `qwen3_5.py` →
`qwen3_next.py` → `torch/_dynamo/utils.py::call_size` while
`references/accuracy-benchmarks/first-generate-degenerate.md` already carried that
stack, its ruling-out and its fix attribution. The same applies one level deeper: an
`IndexError: list index out of range` whose deepest frame is
`vllm/model_executor/models/<model>.py::get_mrope_input_positions`
(`image_grid_thw[image_index]`, called from `_init_mrope_positions` in
`v1/worker/gpu_model_runner.py`) is a **model-directory** question — the image count
is read off the checkpoint's `vision_start_token_id` while the features come from
`mm_features` — so the decisive lines are the server's own `Detected the chat template
content format to be '…'` and the ids in `config.json`, not a walk through
`paddleocr_vl.py` and the mm caches: recorded #5772 spent 20 turns and 50 K
characters of reasoning there and shipped nothing, while the reporter's own later
reply (a working run of the same tag and image, fix = re-download the model files)
was the answer (`references/multimodal-inputs/mrope-image-index-error.md`).

## Procedure

1. One level of `/workspace` (`search_files (target='files')`) — that level only,
   never `history/` and never a sweep of an attachments directory (the one allowed
   read is the single `ls -la /workspace/attachments/<issue>/` when the paste names a
   log, rule 7) — then run
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
   (`references/evidence-workspace/index.md`). **Read that `.ocr.txt` whole or grep it
   for the error line, never a prefix**: its line order is scrambled and the
   `raise …`/`XxxError: <message>` pair can sit 200+ lines below the frames — #3816
   read a 200-line prefix of a 466-line OCR, stopped on the last frame, and asked for
   a traceback whose `RuntimeError: Invalid request status: WAITING_FOR_FSM` was
   already there *and* already a route row in this skill (grep the exception text,
   not the mechanism word). Not only logs: a screenshot of a **vendor
   docs page** is the vendor's own table — a capability/image question ("可以用
   `quay.io/ascend/vllm-omni:vX` 部署 <model> 么") is often settled by its OCR's
   `Hardware` columns in one read, while reconstructing tag conventions from other
   issues costs a dozen calls and yields weaker evidence (#10303 never opened it).
   **A bare CANN/HCCL code in a paste (`EJ0001`, `EI0006`, `EL0004`, `507xxx`, a numeric
   `HcclResult`) is one call, not a source walk**: `bash <skill_root>/scripts/cann-error-lookup.sh <code>`
   prints the vendor's own `errTitle`, message and `Possible Cause`/`Solution` from
   `cann/metadef/error_manager/error_code.json` / `cann/hcomm/…/error_code.cc`, or the enum
   name from any exported `hccl_types.h` — and, for a numeric `507xxx`/`107xxx` **runtime**
   code that no `HcclResult` enum holds, its `ACL_ERROR_RT_*` name with the header's own
   comment and the vendor's action
   (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h`,
   `cann/runtime/docs/api_docs/aclError.md`). **Two families the same call now resolves are outside
   every ACL table**, and a plog paste using them reads "not found" if you only search those:
   a hex `retCode=0x2a` beside `[aicpu exception]` is `TS_ERROR_AICPU_EXCEPTION` in the driver's
   `tag_ts_error` enum (`cann/driver/…/tsch/task_scheduler_error.h`, whose comment *is* the
   bracketed text), and the `errcode:N, msg:execute kernel param invalid` field of an AICPU record
   is `enum rtAICPUErrorType` (`cann/runtime/…/device_error_info.hpp`: `1` =
   `KERNEL_EXECUTE_PARAM_INVLID`, `21001` = the aicpu scheduler layer, i.e. a different owner) —
   `references/hccl-and-hardware/aicpu-task-exception-empty-kernel.md`. That name is the lane: `507010` =
   `LOST_HEARTBEAT` "ts lost heartbeat", `507014/507015/507016` = aicore timeout/exception/
   trap, so such a paste has no vllm-ascend cause to find
   (`references/hccl-and-hardware/device-health-checks.md`). Two of its answers change the
   lane outright:
   `7 = HCCL_E_UNAVAIL` beside `EJ0001` means **the previous process is still alive**
   (sleep mode, an engine that never exited), not a parameter or NIC fault; and `EI0006`'s
   solution text names `HCCL_CONNECT_TIMEOUT` before any link theory. When
   the reported title is the end of a
   chain (`Failed to infer device type`, `Failed to load plugin ascend`), the owning
   line is two or three above it —
   `references/image-dependency-faults/plugin-load-module-missing.md`. Split the two
   shapes from the paste's own lines: `plugin ascend loaded.` present **and** neither
   `Failed to load plugin ascend` nor `Platform plugin ascend is activated` present
   means the module imported and the hook `register()` raised into vLLM's `except Exception:
   pass` — the traceback must be recovered with
   `python -c "import vllm_ascend; print(vllm_ascend.register())"`, and the pin's
   `release_notes.md` known-issue bullet that calls `No platform detected` /
   `vllm._C` harmless is the pre-emption for the follow-up, not the verdict —
   `references/image-dependency-faults/platform-plugin-register-raised.md`.

   **2b — one locate call on your own library, before any repo read.** Take the paste's
   two or three most distinctive strings (the error sentence, the op or kernel name,
   the numeric code) and grade them with `scripts/reference-locate.py`, next
   paragraph: the library is keyed on the reporter's own wording, so this is the call
   that finds the page — or the case — already holding it, and a plain `rg -l` loop is
   its fallback rather than the first move. **Put the task's own issue id in the same
   tool block — when the body carries one.** A paste that is only the issue text has
   no cheap id route: listing `history/` (flat `<id>.md`) returns every other case
   (#6972 got 20 unrelated names), and the id appears only if the layout exposes
   `/workspace/repos/<id>/` or `/workspace/attachments/<id>/`. Skip the id grep then:
   the string locate reaches the page (that is how #6972 reached
   `startup-failures/graph-mode-not-applied.md`, whose body reads `recorded #6972`).
   With the id in hand: a page that records a case carries the id in its heading (`# Device fault
   under MTP + a captured decode graph (#10700)`), so `rg -n '#<id>'
   <skill_root>/references/` answers "has this very issue been worked, and which page
   owns it" in one call — open the page whose **heading** carries the id and start
   from that section's ranked candidates and its ablation order. A section headed by
   its *mechanism* rather than by the id still owns the case when its own opening
   lines say `recorded #<id>` / `Derived … for #<id>` (`setup-failed-call-site.md`'s
   `## SelfAttentionOperation with no spec decode` opens "Derived at vllm-ascend
   `v0.9.2rc1` for #2239"), so read the hit line itself rather than the heading
   level; the id in a table row or link list pointing at another page is a citation,
   not ownership, and no hit is the useful "this case is new"
   (the id is never in `history/`, rule 8). Recorded #10700: the token grep had
   already listed `spec-decode-mtp/draft-graph-capture-device-fault.md` — heading
   `(#10700)` — fifth of thirty, but the session opened
   `hccl-and-hardware/aclgraph-full-decode-runtime-crash.md` and led its ladder with
   "drop `--async-scheduling`", the CI-config delta that the owning page's own Scope
   disowns, while that page's first ablation (spec-scoped `enforce_eager`) was the
   lever the reporter had already confirmed.

   **Grade the library in one call — `scripts/reference-locate.py` — and read the top
   page**: `uv run --no-project python <skill_root>/scripts/reference-locate.py '<op>'
   '<env var>' '<numeric code>' '<symptom phrase>'` prints each token's file count,
   then every page carrying **all** tokens with that page's own `# heading` beside the
   path, then the near-miss grades. **Read that all-token list to its last line**: the
   all-token grade prints every member when it has at most 25, while near-miss grades
   cap at 8, and `[case]` aggregates sort first, so the owner is often the tail (#6972:
   13 members; the owning page was in the five a capped list hid). A flag is a token as
   written (`'--compilation-config'`). The page on every list is the
   owning page; one list is periphery (membership in all = the case, rule 8), and the
   heading chooses between four candidates without opening them. Two hand-run
   sessions paid 3.4-19.8 KB for what the script prints in ~1-3 KB (#6509, #9003);
   it counts the file **name** as a hit and orders `[case]` first, `[index]` last.
   A `[case]` line ends with the entry sharing most tokens — `→ #12985:488 (3/3)`:
   `sed -n '488,528p'` on that path is the case history, and its ladder is the
   message. #12985's second session saw the `[case]` file first but opened no entry,
   skipped the `#<id>` grep, and spent 28 calls deciding from upstream source whether
   the checkpoint was multimodal — the entry had already named it a 310P
   platform-limit case — shipping no text. Recorded #18315 — `MoeDistributeDispatch`,
   `HCCL_OP_EXPANSION_MODE`, `PYTORCH_NPU_ALLOC_CONF`, `107020`, lists of 13-20 pages
   each — left `hccl-and-hardware/aiv-mc2-env-ablation.md` as the only page on all four:
   owning page read on turn 3, answer delivered on turn 7. Recorded #13699
   (`AscendMMEncoderAttention|_run_vit_fia|npu_fused_infer_attention_score` = 21 pages,
   `sparse_mode` = 8) had `custom-ops-runtime-faults/attention-op-numeric-errors.md` on
   both. Fall back to the bare loop (`for t in '<op>' '<code>'; do echo "== $t"; rg -l
   "$t" <skill_root>/references/; done`) when no token is distinctive — all lists
   longer than ~20 pages means the tokens are family words, so shorten them to the
   attribute, flag or code the paste names; the script now prints a family-word
   guard and names the two moves that work. **A paste with no distinctive token has
   no grade to run** (two tables, two screenshots, "X is slower than Y"): take the
   shape's row in `references/dispatch/classified-shapes.md` and open the `[case]`
   file by the report's own wording (#5605's five family tokens graded `4-of-5: none`,
   widest 183 pages, and never listed the owner). An
   **empty** grade is not "no page": one token is spelled differently in the
   library — an op-name suffix (a paste's `aclnnMoeDistributeDispatchV4` against the
   pin's `npu_moe_distribute_dispatch_v2`) or a Chinese phrasing — so read the
   shortest list by hand. **A hit in
   `references/evidence-workspace/recorded-cases*` is opened first** — but that glob
   spans **two or more files per family**, `recorded-cases/<family>.md` (the condensed
   index line) and its `recorded-cases-<family>.md` detail page(s) — the startup
   family splits into `recorded-cases-startup.md` and
   `recorded-cases-install-and-environment.md`, and the device family into
   `recorded-cases-device-and-communication.md` and its `hccl-and-hardware`-owned half
   `recorded-cases-hccl-and-watchdog.md` — so one case carries different line
   numbers per file: re-read the exact `file:line` `rg` printed and nothing else. A
   number borrowed from the sibling file lands on another case — recorded #3543 ran
   `sed -n '80,105p'` of the detail page for a hit in the index, read
   #8022's entry, and spent three more greps on the phantom. Read
   `sed -n '<hit-30>,<hit+40>p'` on that same path before any topical page or repo
   grep: an entry matching the report's versions, model and
   stack *is* the case history, and its ladder is the message. Both halves were paid
   for in one minibatch: #6275's second session had
   `recorded-cases-hccl-and-watchdog.md`, carrying #6275's own entry, in its
   turn-3 hit list and read two route pages instead (16 turns, 20 tool calls, cut
   inside ranked cause 1); #11863 skipped this grep, went straight to the two repos,
   and re-derived what `startup-failures/arg-validation-errors.md:8` already held
   (21 turns, cut mid-sentence). Redirect: the moment a hit's title or index line
   names the paste's symptom, the next call opens that file; a repo grep for the
   raising line is step 3's move, after that read.

3. Take the track's cheap checks before source archaeology:
   - **The recipe diff that pays is by key name, not by flag position.** A reporter's
     `--additional-config` / `--hf-overrides` object carries keys the pin's tutorial for
     that model never shows, so a side-by-side command diff reads as "the documented
     recipe" while the extra key is the whole difference. Grep each key by name:
     `rg -n '<key>' docs/` (a pin with no hit means undocumented, hence outside every
     validated recipe), then `rg -n '<key>' vllm_ascend/` for the consumer
     (`getattr(…hf_config, "<key>", …)`), which turns the key into a code path with a
     gate — and one launch with it set to `false` into a real ablation. Recorded #11080:
     the GLM5 DSA-CP recipe (`docs/source/tutorials/models/GLM5.md:819`) and the
     reporter's script differ by exactly `--hf-overrides '{"use_index_cache": true,
     "index_topk_freq": 4}'`, a key in no doc at that pin
     (`references/dsa-sparse-attention/sfa-sparse-indices-row-mismatch.md`).
   - Lane A/B: effective `max_model_len`, then the diff between the reporter's
     command/env and the matching tutorial or CI test
     (`references/startup-failures/max-model-len-budget.md`; an explicit
     `No available memory for the cache blocks` with `Available memory: 0` is that
     page's profiling-peak section — raising the ratio alone is not the lever, and a
     stated long-context requirement forbids the shrink ladder).
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
   - **A host-side `RuntimeError: shape '[N, -1]' is invalid for input of size M`
     is a division, then one grep — never a model-constant hunt.** `M` splits into a
     row width and a row count: divide it by the paste's own widths and read the
     result as *padded* or *real*. A quotient equal to a **parallel size** is the
     third reading, and it stops a token-count hunt: the paste's own rank lines carry
     the number (`EP rank k` in world size N), so a MoE layer whose output is
     `ep_size` rows too wide is a combine with no dispatch, not a graph pad
     (`references/soc-platform-paths/310p-moe-expert-parallel-shape.md`). If the
     tensor is a **process-global** rather than
     a per-call one, the count belongs to whichever forward wrote it last, so
     `rg -n '<writer>' vllm_ascend/` over the whole package (not one file) and treat
     an **absence** as the finding: the rope cos/sin slice
     (`ops/rotary_embedding.py` `_cos`/`_cos_slice`, `update_cos_sin`) is refreshed by
     the target runner and by the EAGLE proposer but **not** by `mtp_proposer.py` at
     v0.14.0rc1, so `cos.view(num_tokens, -1)` in `ops/triton/rope.py` raises exactly
     on the steps whose token count sits in a capture-size gap under
     `--async-scheduling` (`references/spec-decode-mtp/rope-cos-sin-slice-token-mismatch.md`,
     #6449; `scripts/rope-cos-sin-slice-check.sh` prints the division). The model's
     width (`head_size` vs `partial_rotary_factor`) decides only whether the division
     *raises* or silently strides over the wrong rows, so it changes no rung of the
     ladder — do not spend turns deriving it from an export that may not carry the
     model's config at all, and send the verdict as soon as the division separates the
     two counts (the recorded #6449 run held that answer for twelve turns and shipped
     no message text). The **other writer family** names itself in the shape: a 4-D
     reshape whose last two dims are a vision tower's `origin_hidden_size_per_attention_head`
     and `hidden_size` (`'[-1, 3, 80, 1280]'`) is the qkv **weight pad**, and the size that
     fails it is the already-padded parameter of a **second** `load_weights` — an RL
     rollout's per-step weight sync, never a request, which is why the report says "step 2"
     and why plain inference cannot reproduce it (`references/code-change-requests/vl-weight-pad-not-idempotent.md`, #809).
     The **load-phase mirror** is the same division one layer down:
     `RuntimeError: The expanded size of the tensor (P) must match the existing size (C)`
     out of a quant `weight_loader`'s `copy_` means `P = C // N` for a parallel size `N`,
     i.e. the engine allocated the parameter *with* the divisor and the loader branch it
     took never shards it — the branch is the param's own `quant_method` tag, so read the
     allocation site and the tag together, and one read of the checkpoint's
     `quant_model_description.json` (`version`, `group_size`) decides whether that
     conditional applies before any loader source
     (`references/startup-failures/quant-aux-param-tp-shape-mismatch.md`, #7088).
   - Any track, when the paste is a **serve command on a 2×N 910B box at v0.10.x** (MoE
     model, perf *or* accuracy symptom): classify the **graph lane** from the command before
     ranking mechanisms — `--enforce-eager` present ⇒ `NO_COMPILATION`
     (`platform.py:143-145`, log `Compilation disabled, using eager mode by default`), absent
     with a `--data-parallel-size` + `--enable-expert-parallel` pair ⇒ `PIECEWISE` ACL graph,
     which that release's Known Issues call broken for DP+EP (`release_notes.md:53`, #2229).
     The same flag is a **cost** on a throttled decode (remove it) and a **cure** on a DP+EP
     corruption report (add it) — one grep decides the direction, one launch confirms it
     (`references/performance-throughput/graph-mode-from-the-serve-line.md`).
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
   issues with issue ids. **Then grep the same pin's `docs/source/user_guide/
   release_notes.md` for the feature the report names, before any source walk**:
   its `### Known Issues` rows are the project's own admission and outrank a
   derivation — the #8600 CPU-binding overlap is stated twice (`:547` under
   `## v0.11.0`, `:650` under `## v0.11.0rc2`, "doesn't work for multi instance
   case … We'll fix it in the next release") while the recorded run re-derived it
   from `cpu_binding.py` and never opened the file
   (`references/performance-throughput/cpu-binding-multi-instance-overlap.md`).
   Quote the row with its own `## vX` header, and check whether the later fix the
   same file lists actually covers the reporter's path — there the non-overlap fix
   (`global_slice`, #6945) is A3-only, so the row stays accurate on A2.
   Claim "model unsupported" only after checking
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
   `collect_env.py` and would return the whole corpus). **A bag of product-family
   words is a listing, not a token**: `-i 'GLM5|GLM-5|kimi|MC2|1P1D|挂死'` came back
   with 60 titles, needed a second call just to page the tail, and named no body
   worth reading (recorded #10963) — and the same bag handed to
   `rg -l '<a|b|c>' <skill_root>/references/` prints ~40 page paths with no rule in
   them. Take the dispatch row for the shape, or one symptom string, error code or
   op name. **A token that is a code
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
   `-g REGEX` pulls just the traceback). **Its output then stops at 80 lines unless
   you pass `-n 0`, and the numbers `-g` prints are raw-file numbers while the
   printed stream is collapsed** — `history-body.sh <id> | sed -n '190,320p'`
   therefore returns nothing and reads as an empty body (recorded #4766 lost two
   calls to exactly that on #3513).
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
  **The *order* of the remaining rungs goes in the same paragraph when each ranked
  cause already carries its one check** — compress the ladder to a single ordered line
  there instead of leaving it to a `Ladder` section lower down, which is the block
  truncation eats: recorded #10856 shipped the verdict plus four ranked causes with
  their checks and was cut exactly before its ladder, evidence table and scope, so the
  reporter could not see which of the four to run first.
- **A decisive fact already in hand is the answer.** Once one grep has settled the
  mechanism — the pin's own class carries the attribute the paste calls missing, the
  owning branch or call site is named, the one config line that differs is identified
  — send the verdict in the next message and spend nothing more on confirmation. Two
  recorded runs (#10398, #2239) each held exactly that fact and kept searching; neither
  shipped an answer at all, so the reporter got nothing from work that had already
  succeeded.
- Then the one-line classification: lane A/B/C and the phase that failed (load,
  memory profiling, graph capture, first collective, the profiler's start/stop
  export); or, for a runtime report,
  the primary cause plus its track.
- Ranked causes, each with the evidence you have, the exact check that confirms
  or refutes it, and its confidence label (reporter-confirmed / source-verified
  at commit X / hypothesis).
- The action ladder, cheapest first: **on a reasoning checkpoint, the request's own
  thinking mode and effort value** — a score measured against an official/vendor number
  is compared across two harnesses, and the official number is a thinking-mode number, so
  `chat_template_kwargs={"thinking": false}` (or a body that omits the key, which is the
  same non-thinking run) and an effort value below the baseline's are the first two rungs
  and are dataset-calibrated, not monotonic (reporter-confirmed #10413: MMLU-Pro closed at
  `max`, GPQA only at `high`; `references/accuracy-benchmarks/
  harness-vs-engine-attribution.md` §*thinking checkpoint*), then **the request's own sampling value**
  (`temperature`/`top_p` — on a reasoning-distill checkpoint a ≤0.1 temperature is
  itself a degeneration, so the rung *raises* it; `references/accuracy-benchmarks/
  short-prompt-corruption.md`), `--dtype bfloat16` when the answer is one repeated
  token and the reporter's own controls are clean (transformers and a GPU box
  correct — reporter-confirmed #562, `references/accuracy-benchmarks/
  precision-dtype-degeneracy.md`), an explicit `--max-model-len`, a re-run with
  `--enforce-eager` / `ASCEND_LAUNCH_BLOCKING=1`, `return_token_ids: true` on the
  same request body, one `num_speculative_tokens` step, `asys health -d=$i`, a
  version correction — before any patch proposal.
- The evidence table (`path:line` for code claims, precedent ids as
  `#<id>: <one-line fact>`) comes **after** the ladder, never between the verdict and
  the actions — the table is the most compressible block and truncation eats the tail
  (`references/evidence-workspace/delivery-discipline.md`, *Truncation eats the tail*).
  #8926 shipped the verdict and rung 1 in its first paragraph, spent the rest of the
  16 384-token output budget in the table, and lost every later rung; keep the table to
  the rows the verdict needs and move the rest into "what to send back". #3468 lost the
  same cap with its ranked causes as a two-column table and truncation inside row 2:
  write causes as a numbered list, one line and one check each, and let only the
  `path:line`/precedent table follow the ladder.
- Version and hardware scope of the conclusion, plus what to send back (full
  per-rank log, plog, tiling I-line, raw request/response dumps, `npu-smi info`,
  `asys health` output, `dmesg` OOM lines, both PD serve commands). **The ask list is
  the complement of the paste *and of this skill's own record of that issue*, not a
  repetition of either.** A fact a reopen of a recorded case already holds — a serve
  command that arrived as a later comment, the failure timestamp a previous session
  pinned — is used, not requested: cite it and add "if you now run something else, say
  so". The #4057 replay read the page's *Reporter-confirmed narrowing* section, quoted
  the reporter's TP=4/`--enforce-eager` command as its mitigation, and still listed "the
  actual `vllm serve` command line" under *send this back* in the same message, so the
  one ask that could not move the case was the one it already had. The template's `<details>`
  *Your current environment* block is a `collect_env.py` dump and already prints
  `vLLM Version:`, `vLLM Ascend Version:`, the torch/torch_npu/CANN versions, the
  launch env vars and a full `npu-smi` table, so "which vllm / vllm-ascend version"
  or "run collect_env.py" is answered before you ask: on #9183 that request drew
  「vllm-ascend和vllm版本 这些The output of `python collect_env.py`输出的信息里不都有吗？」
  (2026-05-26), while the two asks that moved the case were the plog at the failure
  second and the full first-error block. The same block's `npu-smi` is a *startup*
  table, so `No running processes found` / `Health OK` on every chip says nothing
  about the failure minute — ask for health **at** that minute
  (`npu-smi info -t health -i <id>`), not for the table again.
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
that break them, `recorded-cases/index.md` the per-issue index).

**The recorded-shape → topic-entry table** — one row per routing decision, written in
the reporters' own symptom wording, each row carrying the owner and the page to open — is
`references/dispatch/classified-shapes.md`; `references/dispatch/index.md` states how it
is used (classify first, match wording not title words, open one topic entry). Take the
classification from *Classify the evidence* above, then find the row. The coarse families
it covers, so a first guess needs no read: **accuracy and output**
(`accuracy-benchmarks`, `multimodal-inputs`, `serving-frontend-faults`,
`tool-call-streaming`, `pooling-embedding`); **startup and load** (`startup-failures`,
`image-dependency-faults`, `soc-platform-paths`); **runtime device, memory and custom
ops** (`hccl-and-hardware`, `runtime-memory`, `custom-ops-runtime-faults`,
`mixed-batch-splits`, `spec-decode-mtp`, `pipeline-parallel`, `lora-adapters`,
`observability-and-logs`); **PD, KV pool and sparse attention** (`pd-disaggregation`,
`kv-pool-store`, `dsa-sparse-attention`); **capability, sizing, perf and docs**
(`feature-enable-recipes`, `deploy-sizing`, `performance-throughput`,
`scheduler-admission`, `release-known-issues`, `code-change-requests`,
`upstream-downstream-drift`, `tutorial-verification`, `evidence-workspace`,
`first-checks`). A row that says *not this topic* is a route away from it — use it.

Scripts are re-runnable; `<skill_root>` is the path `skill_view` reports and each
header carries its flags. Run `.sh` with `bash` and `.py` with
`uv run --no-project python` — the executable bit is not durable, and
`uv run --no-project python <script>.sh` is a SyntaxError on line 1 (a bash wrapper
around an embedded program); when a page names a script without flags,
`bash <script> -h` is the one call. Their arguments, output and originating issue are in
`references/first-checks/scripts-catalogue.md`; open it rather than guessing a name,
and the topical pages name the ones they use. `templates/` holds starter files to copy
and edit rather than retell — currently `templates/omni-310p-deploy-config.yaml` (a
two-stage fp16 `--deploy-config` for one 310P die, cited by
`references/soc-platform-paths/310p-omni-model-deploy.md`) and
`templates/profiler-patch-bisect.py` (the vLLM-free `torch_npu.profiler`
start/stop probe that bisects a `/stop_profile` segfault across `.postN` wheels,
cited by `references/observability-and-logs/online-profiler-stop-segfault.md`). The per-signature
classification index (log fragment → track → owning page) is
`references/first-checks/signature-routes/index.md`.

