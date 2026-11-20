---
name: vllm-ascend-triage
description: 'pip/install, deploy, HCCL/PD, hangs, OOM, accuracy, LoRA'
---

# vLLM-Ascend issue, deployment and support triage

## Problem class this covers

Nine entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter. The last shapes deliver a *decision*
(sizing), a *document review*, a *capability verdict* or a *change plan* instead
of a root cause, but the same rules apply — answer first, evidence pinned to the
reported version, gaps stated.

- **Issue report** (usually Chinese or English, template-shaped: environment dump
  + logs + narrative) about something that went wrong on an Ascend NPU host: a
  device-side operator crash, an accuracy/score disagreement, a garbled long-input
  output, a long-run OOM / "VRAM not released" report, a container memory metric
  that climbs for days without a crash, a performance claim, a
  logging flag that appears to print nothing, an endpoint that answers `404`
  although the reporter configured it (`/start_profile`), output that stops after
  a handful of tokens while the text itself is sane, an operator that runs and
  returns the wrong numbers, or a bare `python -c "import <pkg>"` that aborts
  inside a library the image shipped.
  When the task is just "explain/answer this issue", the answer is still expected
  to be evidentiary, not a summary of the report.
- **Install / pip-resolution report**: no engine, no worker, no traceback from the
  NPU stack — the body is two `pip install` commands and the error line they produced
  (`Could not find a version that satisfies the requirement torch-npu==<dev>`,
  `No matching distribution found for <pkg>`, "unmatched torch version between vllm=X
  and vllm-ascend=Y", or an `ImportError: Please install vllm[bench] …` inside a shipped
  image), or the install finished and pip only **warned** (`requires torch==2.10.0`).
  The deliverable is *which* artefact pip could not see — a mirror-only wheel pin the
  metadata cannot carry, an unsatisfiable torch/torch-npu/torchvision set, an
  air-gapped container with no reachable index — or which file supplied each warning
  line; never a release survey — `references/image-dependency-faults/pip-index-visibility.md`
  and the warned shape `pip-resolver-warning-block.md`.
- **Deployment failure**: a vLLM / vLLM-Ascend deployment fails to come up or
  fails during engine start — `EngineCore failed to start`, `WorkerProc
  initialization failed`, silent worker death, `HcclAllreduce` / allreduce
  errors, `ERR01100`/`ERR00100`, OOM, "the model failed to load", a Python
  `AttributeError`/`TypeError` raised from a vllm-ascend patched forward or a KV
  connector, a `ValueError` from upstream `arg_utils` rejecting the parallel/DP
  flags, or a deployment question filed with only a trimmed log.
- **Support / version question**: "does vllm-ascend <version> deploy <model>?",
  `… has model type '<mt>' but Transformers does not recognize this
  architecture`. This is not a bug hunt; it is a version gate with a one-line
  answer and a registry-level proof.
- **Feature-enable / capability question**: "v<ver> 上如何正确开启 <feature>",
  "PCP 和 DCP 单独开均有不同的报错，能不能给出能跑通的示例", "does F work on model M
  on this SoC", usually with no log pasted and two errors described in prose. The
  deliverable is the validation verdict at the reported ref plus the constraints the
  code enforces, not a launcher script assembled from them —
  `references/feature-enable-recipes/index.md`.
- **PD / EPD disaggregation**: the deployment is up but the P and D sides cannot
  move KV (`HIXL Connect failed | HIXL Error Code: 103900`, HCCL
  `RanktableCheck` / `'device_ip' is not set correctly`, `wait_for_save failed
  ... skipping save`, a recipe that only works with `hostNetwork: true`), or the
  transfer succeeds and the decode node emits wrong or alternating output on a
  hybrid linear-attention model, or a KV pool is attached and the *first*
  transfer fails with `503900` / `HcclCommPrepare ret: 0x13` while the pool's
  `global_segment_size` is the cause, **or** the pool is attached, the request
  succeeds, and the put never lands (`Failed to invoke HcclBatchPut,
  hccl_result = 4`, `Failed to put N keys out of N`, master `PutEnd` 0), **or** the
  deployment runs and the decoder never receives the request (`proxy通信超时`) — an
  endpoint-count question before any connector read.
  Transport identity and rank tables are a topology question, not a connector
  bug, and a pool deployment is a config question before either —
  `references/pd-disaggregation/index.md`. If instead the engines start and the
  **first request** kills one side's EngineCore behind a bare
  `EngineDeadError: EngineCore encountered an issue. See stack trace (above)`, on a
  deployment that sets `--decode-context-parallel-size` > 1: DCP is a *matched*
  launch parameter between the two sides (a differing dcp — a 1M recipe's `8`
  reduced on one side — fails the connector's divisibility assertion on the first
  transfer), so get both serve commands and compare before any source walk —
  `references/feature-enable-recipes/context-parallel-pcp-dcp.md` §3.
- **Deployment sizing / stage layout**: the question is a plan — "does this model
  fit on 2×64 GB", "the GPU docs do a 2-device stage deploy and mine does not
  come up on 2 NPUs", a bundled `platforms.<vendor>` yaml with more stages than
  the reporter has cards, "how do I pin stage 1 to one card" — **or** it is two
  serve commands that differ by one resource knob (`--tensor-parallel-size`,
  `--max-model-len`, `--gpu-memory-utilization`) where the smaller one comes up
  and the larger one OOMs at engine start. Answer with per-card arithmetic plus
  the device-visibility variable, not with a source walk —
  `references/deploy-sizing/index.md`.
- **Community-test / doc-verification task** (`[Community Test]`, a 众测任务书 row
  with 蓝区镜像 / 权重链接 / 卡数 / 交付件 测试报告, or a `[Doc]` report that a
  tutorial step does not work as written): the deliverable is a *report about a
  document*, graded on completeness/usability/correctness and produced without
  the reporter's hardware. The evidence is the pinned tree — which command no
  longer parses, which path is absent from the image, which baseline predates the
  image, whether the stated card count matches the weights —
  `references/tutorial-verification/index.md`.
- **Code-change / cleanup request**: the body asks for an *edit*, not a
  diagnosis — "remove `VLLM_ASCEND_ENABLE_<X>` before <release> final release",
  "delete the temporary switch added in PR #N", "deprecate this flag", often with
  nothing but two links and no log. The first question is whether the named
  identifier exists at the pinned ref; absence dates the snapshot against the PR
  and is itself the finding, and the deliverable is the removal sweep with
  `path:line` anchors plus the behaviour change a default flip implies —
  `references/code-change-requests/index.md`, then `env-flag-retirement.md`.
  The delivery gates apply to this shape too: with no crash to report, the
  absence verdict goes in the first message (the recorded #1262 replay put its
  first message text on its 15th turn).

Recorded cases, per-issue index and how each session ended:
`references/evidence-workspace/recorded-cases.md`; the load-bearing
delivery pattern is rule 1.

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and obey the delivery gates.** A long
   session can be cut off mid-sentence, so the deliverable has to survive
   truncation: what decides the outcome is not how much evidence was gathered but
   whether the message exists. The runs that shipped **no answer at all** each had
   a usable lead in hand (#6784, #13853, #11454, #17991, #3107 are the clean cases;
   the per-run table is `delivery-discipline.md`); #4296 answered completely in ten
   turns from one pasted traceback. Hard stops:
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
   - **The gates cap cumulative output, not only turns.** Runs #4057, #6833,
     #13853, #5750, #4357, #8345, #16563, #16111 and #8788 were cut after 11-49 K
     characters of reasoning with the ranked cause in hand, two of them inside the
     block that was composing the answer (arithmetic in
     `references/evidence-workspace/delivery-discipline.md`). A verdict line is a
     *hypothesis*: the ranked deliverable is still owed as one message. A block
     that restates the verdict, numbers the fixes or lays out the evidence table
     *is* this turn's message: send it, then one tool call.
   - **Reasoning is a scratchpad, a few lines per turn:** what the last result
     proved, the single next check, go. It spends the **same output budget** as the
     answer: #15815, #11759, #8590, #16563, #3107 and #698 all reached the cap with
     most of it inside a reasoning block. If a block enumerates the answer's
     sections, or says "I have enough" / "let me write the answer", that block *is*
     the answer.
   - **A new hypothesis that needs a new code area is not this session's job.**
     Put it in the answer as an open question with the one command that decides
     it; #14506 delivered only after a final block that took half the session. For
     an accuracy/corruption or topology report, a ranked list plus an ablation
     matrix already *is* the full answer.
   - **An ownership question is answered by naming the owner.** "The flag did
     nothing" and "the log line is missing" are gate checklists on the owning
     component — usually the upstream vLLM OpenAI frontend, not vllm-ascend — and
     take 3-4 calls; #10985 and #9583 (a `404` on `/start_profile`) each spent
     20-30 calls inside the owning module with the decisive check in hand.
   - **When the deliverable is a document, the document is the gate.** Open the
     report file on the 2nd-3rd turn with the skeleton plus the first verified
     section, then append **2-3 K characters per write** and make the final
     message the condensed report (path, defects, open items). Never compose the
     whole report in one heredoc: #11739 made 40 tool calls and wrote no file,
     #11759 lost a finished 5.7 K report to the cap, and #11742 (`[Community
     Test]`) opened its heredoc on turn 9 of 10 and shipped **no text**.
   - **`/workspace/scratch/findings.md` is the answer draft, not an extra step:**
     write it at G2 *in place of* the reasoning block; once it exists the next
     turn's text is the answer (`delivery-discipline.md` §draft).
   - **A missing or mismatched artefact is itself the finding** — an empty
     environment block, a traceback file absent from the pinned tree, a version
     the snapshot does not carry. State the gap, give the one command that closes
     it, stop investigating.
   - **Truncation eats the tail, not the head — and the body has a size too.**
     Each action directly after its cause, the closing scope block 3-5 lines. With
     no log pasted: 3-4 causes with one check each, one ladder of ≤6 lines,
     precedents as one line of `#id: fact`, 3-5 lines of scope (≈3-4 K characters);
     a fifth cause is an open question with its one command.
   - **Every assistant turn carries message text, not only a tool call.** A turn
     whose whole output is a tool call leaves nothing for a truncation to keep.
     #5295 emitted *zero* message text on all 16 turns (~46 K characters of
     reasoning) and #16563's text-free run on all 20 turns; #17797 and #5525
     emitted only status notes ("the 310P path has its own model runner — let me
     check X") and never stated a verdict. Carry one or two sentences of the
     current answer draft in the same message as every tool call — verdict line
     first, then the evidence being checked — so G1/G2 need no special turn. A
     *status note* is not a draft: the sentence must state the current verdict even
     when the next call may overturn it.
2. **Classify before investigating.** For a startup log, place it in lane A/B/C
   (table below); for a runtime report, decide whether it is a device-op fault,
   an accuracy/score gap, a corruption report or a performance claim. The
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
   text rather than assuming what it says. A reference page states the pins it was
   verified at: when those equal the refs in `SOURCES.json`, its `path:line` are
   citable as they stand — re-grep only what the page itself marks as
   version-specific, and spend the saved turns on the delivery.
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
   - `rg -r` is `--replace`, not "recursive". `rg -rn "pat" .` rewrites every
     match to `n` and `rg -rl` to `l`, so output looks like corrupted source
     (`self.n`, `decoder_layer.l`) and invites a false theory about the code.
     Recursion is implicit on a directory: use `rg -n "pat" .` or `grep -rn`.
     An **empty pattern matches every line of every file**: a probe typed while
     unsure how to recurse — `rg -rn ""`, `rg -n "" .`, `grep -rn "" .` — dumped
     65 KB into one recorded session (#6240, mid-investigation, buying nothing).
     To list files use `rg -ln '<pat>' <dir>`; never pass an empty pattern.
   - **Scope a grep to one repo.** A case view holds 20-40 exported trees under
     `/workspace/repos/<issue>/`; an unscoped `rg -n '<string>' .` there takes
     seconds and returns hits from unrelated products (torchair, metadef,
     ascend-transformer-boost). #11454 ran three of them looking for one log
     string, and the finding was that the string is in **no** export — state that
     as the gap (rule 10) instead of widening the pattern: `rg -n '<pat>'
     /workspace/repos/<issue>/<owner>/<repo>` per candidate, then stop.
   - Never list a history or attachments directory, never dump `SOURCES.json`,
     never `cat /workspace/history/*.md` (flat whole issue bodies, so even a
     truncated read is a listing — #3107 pulled 7 KB that way), and
     never bundle a listing with the `skill_view` call (it lands before this rule is
     read): a listing cost ~24-28 KB (#13699: 2857 entries) and later batches still
     paid 13-104 KB per session (#9335, #9183, #11739, #8984, #12487, #698, #5750,
     #11454, the last inside the same block as `skill_view`). Load the skill alone,
     then make the first shell call
     `ls /workspace`
     plus `bash <skill_root>/scripts/sources-summary.sh`, nothing else — it prints
     the version contract and `unavailable` repos in ~10 lines, not 20-54 KB of
     `cat SOURCES.json`. Orientation is one level of `/workspace`
     (`search_files (target='files')`) plus that script; a missing
     `/workspace/attachments` makes a bundled listing exit 2, which reads like a
     failed orientation. Search by token (`rg -l`, `scripts/history-titles.sh`),
     never by listing a directory.
8. **The prior-month corpus is not evidence of absence.** `history/*.md` and
   lexical issue search cover only earlier issues; a current issue is never in
   them. Make one grep for the distinctive token (op name, error string, dataset
   name), read 2-3 hits, then stop: precedent supports a hypothesis, it does not
   replace reading the source at the pinned commit. Reach for the `github` issue
   search only when the corpus token has **no** hit: it returns whole issue bodies
   paginated (one page measured 17 K characters with `display_complete: false` in
   #9986, and the decisive precedent still came from `history-titles.sh`), so read
   the first page and stop instead of paging. **Pick the bodies by whether
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
   `cann/hcomm/src/framework/common/src/topo/` exist only as headers). One grep
   pass per tree, then use the plog, the version correspondence and the hardware
   path.
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
    Cap ~6 calls (#5609, #10943 burned the session there).
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
    `tests/e2e/**` (and the nightly yaml's `npu_per_node`), or shown only by an
    older precedent issue. "No validated recipe for this combination" is an answer;
    a launcher synthesized from the constraints the code enforces is not, and it
    provokes the follow-up "但有人在 A2 上成功跑通过吗?" (#8984). Then map each
    enforced constraint to the error text it raises, so the reporter can place their
    own two errors, and stage the experiment one axis per launch. Three turns,
    `references/feature-enable-recipes/index.md`. Two traps in this track: a pinned
    doc sentence stating a narrower hardware scope is the project's position for that
    version (quote it, do not overturn it with an older issue), and a code comment
    saying a sub-feature is unsupported is often a branch the code already takes,
    not a launch-flag prohibition (`sfa_cp.py` vs the feature table's MLAPO ✅).
15. **A working sibling path in the reporter's own evidence constrains the cause.**
    When the same box fails on one entry point and works on another — offline
    `LLM()` vs `vllm serve`, one flag, one model, one card — a mechanism that also
    predicts the working path fails is not yet the answer: name the difference
    first and rank the mechanism below it, with the log line or ablation that
    separates them. Recorded #1534: "mindie_turbo replaces RMSNorm and `fullgraph`
    makes a graph break fatal" was answered while the same install served
    `Qwen/Qwen2-7B-Instruct` through `vllm serve`, and the reporter's follow-up
    ("是因为在线推理默认没有开启 acl graph 吗?") got no answer. Two log lines settle
    which path compiled and which process loaded the plugin — `PIECEWISE
    compilation enabled on NPU` / `Compilation disabled, using eager mode by
    default`, plus the wheel's own INFO line
    (`references/image-dependency-faults/mindie-turbo-graph-break.md`);
    `enforce_eager=True` on the failing path is the one-line ablation.

## Classify the evidence

| Signature in the report | Track | Go to |
|---|---|---|
| Startup log: generic `WorkerProc initialization failed due to an exception in a background process`, **no** per-rank worker traceback; optional `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!`; long silence before failure | Lane A: worker died without reporting | `references/startup-failures/silent-worker-exit.md`, then `references/startup-failures/max-model-len-budget.md` |
| Startup log **ends** at `Loading model weights took X.XXXX GB` (nothing raised, port `connection refused`), often as the whole paste | Lane A: silent stall in the memory-profiling forward, or a killed process — bound the phase from the pinned worker, then ask for the `vllm serve` line | `references/startup-failures/silent-stall-after-weight-load.md` |
| A `vllm_ascend/**` `file:line` in the paste names a module the pinned tree does not contain (`ls vllm_ascend/models/` has no such file) | Site-vendored file, not a snapshot gap: cite the framework calls it makes, not its arithmetic | `references/startup-failures/environment-correspondence.md` (vendored-file section) |
| Startup log: per-rank `WorkerProc failed to start.` / `WorkerProc hit an exception.` plus a traceback with rank prefix | Lane B: worker raised a Python error | `references/startup-failures/environment-correspondence.md` (diff against the known-good tutorial/CI config) |
| …and the traceback is `OSError: libatb.so: cannot open shared object file` → `Could not load this library: .../torch_npu/lib/libop_plugin_atb.so` → `_register_atb_extensions()` at worker init, with an outer `OSError: [Errno None]` and no `Starting to load model` line | Lane B, environment-owned: NNAL/ATB component missing, or installed and never sourced — CANN's `set_env.sh` does not export the ATB paths | `references/image-dependency-faults/nnal-atb-missing-library.md` |
| …or the per-rank/plain traceback is `ImportError: <name>.so: cannot open shared object file` — e.g. `ascend_transport.so` from the connector's own `from mooncake.engine import TransferEngine` (`kv_p2p/mooncake_connector.py:24`) — in a deployment started by a launcher's `args:`/`command:` block, while the pasted `collect_env.py`/`/etc/profile` dump shows the directory on `LD_LIBRARY_PATH` | Lane B, environment-owned: the environment that decides the load is the launched process's, not the shell's — the export belongs in the launch command; image content only if the file is absent from disk | `references/image-dependency-faults/launch-env-and-library-path.md` |
| The traceback ends in `AttributeError: '<X>Config' object has no attribute '<attr>'` with a frame in `vllm_ascend/distributed/*connector*.py`, reached from `ensure_kv_transfer_initialized` or `register_kv_caches` — a VL/MoE model under PD disaggregation, **or** a flat config whose scalar is derived rather than stored (Qwen2 `head_dim`), where `hf_text_config` was already in use | Lane B: connector reads a config attribute that *nesting* or *absence* leaves undefined — the two are different findings and give different fixes | `references/startup-failures/connector-config-attributes.md` |
| P/D will not connect and the recipe contains a **KV pool**: `AscendStoreConnector` alone or inside a `MultiConnector`, `MOONCAKE_CONFIG_PATH`, a `pool_worker.py` / `pool_scheduler.py` line; the first transfer fails with `Failed to connect to target … 503900` / `E19999` / `HcclCommPrepare … ret: 0x13`, then the decode node answers with garbage while every other metric looks healthy | KV-pool store segment, not the transport: the pool's `global_segment_size` is `aclrtMallocHost`-ed and registered into the same ADXL engine the p2p connector dials, so an oversized segment fails whichever connector dials first (#13439: 40 GB failed, 2 GB worked, HDK 25.2.3 needs < 20 GB) | `references/pd-disaggregation/kv-pool-store-segments.md` |
| A KV pool is attached, the request **succeeds**, and nothing is stored: prefill logs `Failed to invoke HcclBatchPut, hccl_result = 4` (or `HcclBatchGet`) then `Failed to put N keys out of N. error_codes=[-1]. Check memory and store capacity.`, while `mooncake_master` shows `PutStart`/`PutRevoke` > 0, `PutEnd` 0 and unused memory | Store **put data plane** in Mooncake's Ascend build, not capacity and not the connector: `4` = `HCCL_E_INTERNAL` (generic, not PARA/MEMORY/TIMEOUT), `-1` = the store's generic `INTERNAL_ERROR` (an out-of-space put returns `-200`), `PutStart`>0/`PutEnd`0 = control plane allocated and the write never landed, and the capacity sentence is canned text (`mooncake_backend.py:194`); the emitting code is in the `mooncake-transfer-engine-npu` wheel, absent from every export | `references/kv-pool-store/hccl-onesided-store-put.md` |
| P/D will not connect although both engines started and there is **no** KV pool: `HIXL Connect failed \| HIXL Error Code: 103900` (or `503900`), older `LLM_LINK_FAILED` from `llmdatadist_c_mgr_connector`, `wait_for_save failed ... skipping save`, HCCL `RanktableCheck` / `'device_ip' is not set correctly, must be set when multi Server`, `get dev list error:serverId[...]`, a recipe that only works with `hostNetwork: true` / `--net=host` | PD transport topology: two addresses on one node became two HCCL servers (the engine id *is* the rank-table `server_id`); look at the rank table and the plog, not the Python traceback | `references/pd-disaggregation/transport-topology.md` |
| PD deployment that connects and then corrupts on a **hybrid** model (Qwen3.5/Qwen3.6 GDN, DeepSeek-V4, GLM): alternating good/bad answers, empty or wrong first decoded token, works with MTP but not without | PD hybrid-state transfer: connector choice plus the `mamba_cache_mode='align'` gate only the KV-store connectors satisfy | `references/pd-disaggregation/hybrid-state-transfer.md` |
| `ValueError: Non-MoE models do not support external data parallel mode`, or any `ValueError`/`AssertionError` raised from `vllm/engine/arg_utils.py` that names a parallel mode or flag (`data_parallel_*`, `nnodes`, `node_rank`), with no `WorkerProc` line in the log | Startup argument validation in upstream vLLM — launch-command fix, not a defect | `references/startup-failures/arg-validation-errors.md` |
| `KV transfer 'prefill' config has a conflicting data parallel size. Expected 1, but got 2.` (or `tensor parallel size`, or `'decode'`), on a PD deployment whose `--kv-transfer-config` sets `kv_connector_extra_config.prefill/decode.{dp_size,tp_size}`; DP=2, DP=4 … all fail the same way, no worker/device text, and model-load lines appear *before* the raise | The connector block is validated against `parallel_config.data_parallel_size` at engine start — and upstream has already collapsed a **dense** model's per-engine DP to 1, so `dp_size: 2` can never validate; the tutorial's `dp_size × tp_size = cards` table is a MoE recipe. Fix the block's number (1), do not delete the keys (the connector asserts them) | `references/startup-failures/arg-validation-errors.md` (#8345) |
| pydantic `ValidationError`: `Unexpected keyword argument [type=unexpected_keyword_argument, input_value=True, input_type=bool]` + an `errors.pydantic.dev` URL, naming a key from `--additional-config`, often with `ERR99999`, no `WorkerProc` line | `additional_config` **schema** rejection: `AscendConfig` is `extra="forbid"`, so the key is nested-only (e.g. `ascend_compilation_config.fuse_muls_add`) or removed, and a tutorial serve command usually carries the wrong nesting | `references/startup-failures/additional-config-schema.md` |
| The report asks whether a plan fits rather than what broke: "can this model run on N cards", a multi-stage orchestrator (`--stage-id`, `--headless`, `platforms.<vendor>` yaml) that works on GPU, stages that must share a card, `CUDA_VISIBLE_DEVICES` in a command aimed at an NPU host | Deployment sizing / stage layout: memory arithmetic plus the device-visibility variable, not a fault | `references/deploy-sizing/index.md` |
| Two serve commands differing only by `--tensor-parallel-size` (usually 4 vs 8) on one node: the smaller serves, the larger dies at engine start with `NPU out of memory` / `Worker failed with error '... out of memory ...'`, often on a hybrid linear-attention model (Qwen3-Next / Coder-Next) with a 128K-256K `--max-model-len`; sometimes only the two commands are pasted | TP-scaling **budget**: the tutorial's validated serve line, then the OOM line read as a budget (`already allocated` vs `gmu × total`), then the per-rank terms that do not shrink with TP. Not a TP8 defect and not an allocator-mechanism hunt | `references/deploy-sizing/tp-scaling-oom-at-startup.md` (one call: `scripts/oom-budget-split.sh`) |
| `[Community Test]` / `[Doc]` / 资料验证 / 测试报告, or "the tutorial's command does not work as written" | Tutorial-document verification: grade the page against the pinned tree; the report is the deliverable | `references/tutorial-verification/index.md` |
| `value error, the checkpoint you are trying to load has model type '<mt>' but Transformers does not recognize this architecture`; `Model architectures [...] are not supported`; "can version V deploy model M?" | Model-support / version gate | `references/startup-failures/model-type-not-recognized.md` |
| Python `AttributeError`/`TypeError` in a frame under `vllm_ascend/{ops,patch,models}` where the class subclasses or re-implements an upstream vLLM/transformers class (a missing *quantization tensor* such as `aclnn_input_scale` is the DSA row below, not drift) — e.g. `'AscendQwen2DecoderLayer' object has no attribute 'attention_type'`, Python `TypeError: … got an unexpected keyword argument` (the pydantic form is the schema row above), `cannot import name` | Downstream patch drift | `references/upstream-downstream-drift/index.md` |
| `RuntimeError: Invalid request status: <STATUS>` (e.g. `WAITING_FOR_FSM`) raised from `vllm_ascend/core/scheduler.py` when the request carries `guided_*` / `response_format` / `structured_outputs`, while plain chat works; EngineCore dead, client 500, `nanobind: leaked … GrammarMatcher` at shutdown | Downstream **scheduler-fork drift**: `AscendScheduler` is a fork of upstream's v1 scheduler and never learned the request state upstream sets for grammars — and its `SchedulerOutput` hardwires the bitmask empty, so "stop the raise" alone degrades to silent non-enforcement. First move is one grep on `platform.py`: at rc0 the fork is **force-enabled** for a default dense serve (an explicit `enabled: false` is overwritten), later builds are opt-in — so the escape is a gate input (`--structured-outputs-config '{"backend": "xgrammar"}'`) or dropping the key, not reading xgrammar | `references/upstream-downstream-drift/scheduler-fork-missing-status.md` |
| `AttributeError: 'AscendRowParallelLinear' object has no attribute 'aclnn_input_scale'` (or `aclnn_input_scale_reciprocal` / `aclnn_input_offset`), deepest frame `vllm_ascend/attention/sfa_v1.py:707` in `_init_o_proj_tp_full_params` reached from `:544` in `process_weights_after_loading`, on a DSA model (GLM-5/5.1, DeepSeek-V3.2 — `index_topk` in `hf_text_config`) with `VLLM_ASCEND_ENABLE_FLASHCOMM1` / `VLLM_ASCEND_ENABLE_FLASHCOMM` / shared-expert-DP on, and a **W4A8 or BF16** checkpoint | DSA-CP o_proj TP/Full weight switching exists only for a W8A8-static o_proj: the three tensors are created by the static-w8a8 method alone, so this is an unsupported feature×quant combination, not install or patch drift — FLASHCOMM1 off, or the tutorial's W8A8 weights | `references/dsa-sparse-attention/o-proj-quant-attribute-crash.md` |
| The same `AttributeError` raised **inside an upstream vllm file** on a class vllm-ascend is meant to replace — `'ZeroExpertFusedMoE' object has no attribute 'select_experts'` — or the reporter asks "did it ever reach vllm-ascend's ops?" | Downstream registry drift: the Ascend replacement is keyed by class *name* (`CustomOp.register_oot`), so an upstream refactor that adds a class silently keeps upstream code | `references/upstream-downstream-drift/oot-class-registration-drift.md` |
| An `AttributeError` inside an upstream vllm file whose message contradicts the code — `'<QuantMethod>' object has no attribute '<property>'. Did you mean: '<similarly named method>'?` — reached from an upstream model's `forward` at startup (`profile_run`), where the property exists and calling it directly gives `'NoneType' object has no attribute '<property>'` | Downstream patch **incomplete**, not absent: a factory/property rebound only in its defining modules while the model module cached the old binding at import. Compare the three attributes, do not read more of the model | `references/upstream-downstream-drift/stale-module-binding.md` |
| A bare `TimeoutError` — `RPC call to execute_model timed out.`, or frames in `multiproc_executor.py` (`collective_rpc` / `get_response`) and `shm_broadcast.py` (`acquire_read` / `acquire_write` / `recv`) — reported as "多进程卡间通讯超时", with no rank prefix and no error code | vLLM's own RPC watchdog (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, 300 s), not an HCCL/NIC timeout: one worker stopped answering, and the paste does not contain its fault | `references/hccl-and-hardware/rpc-watchdog-timeouts.md` (one call: `scripts/executor-timeout-check.sh`) |
| `!!!!!!! Segfault encountered !!!!!!!` in a server log, **either phase**: mid-run (an evalscope/vllm-bench client dies with `ConnectionRefused [Errno 111]`, `Worker proc … died unexpectedly (exit code: None)`, dump top frames `rtEventCreateExWithFlag` / `NPUEvent::record` / `ProcessGroupHCCL::_allgather_base`, no `vllm_ascend/*.py` frames) **or** at engine start (`Failed core proc(s): {'EngineCore': -11}`, no `Starting to load model`, dump top frame a **trivial CPU** `torch.zeros` / `fill_kernel` / `VectorizedLoop2d`, `[ERROR] TBE Subprocess[task_distribute] … main process disappeared` after it) | A signal kill, not an OOM and not a Python exception — `-11` **is** SIGSEGV; client and TBE lines are messengers, and even a trivial CPU top frame is only where damage surfaced. Name candidates with one check each, ask for the plog, add the `--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1` rerun | `references/hccl-and-hardware/worker-segfault-no-python-frames.md` (both phases; startup cross-ref `references/startup-failures/silent-worker-exit.md`) |
| `TypeError: 'NoneType' object is not callable` from an upstream module-level name that is `None` on non-CUDA platforms (`fused_experts = None`), with the `forward_oot` frame in `vllm/model_executor/custom_op.py` and `forward_native = forward_cuda` next in the trace; often `--distributed_executor_backend ray` or a `spawn` worker | The downstream op patch was never installed **in that process**: it is imported as a side effect of a driver-only hook (`check_and_update_config`), so fresh-interpreter workers keep upstream code | `references/upstream-downstream-drift/patch-import-side-effect.md` |
| The first collective after the weights load dies as `Transport_Init_Error(EI0009)` / `[Create][DestLink]Create Dest error!` / `createLink para:rank[i]…dst_rank[j]`, with `halShmemOpenHandle … drvRetCode=17` + `rtsIpcMemImportByKey … 507899` beside it | HCCL transport **link creation**: `507899` is the generic driver internal error and equal `localIpAddr`/`remote_ip_addr` text is not a duplicate `device_ip`; the failing edge set (same-card vs cross-card, from the `npu-smi` Bus-Ids) puts it on the host driver/CANN pairing or the device passthrough, not the model | `references/hccl-and-hardware/transport-destlink-ipc-import.md` (one call: `scripts/hccl-link-pair-audit.sh`) |
| `HcclAllreduce`, `hcclRet -> 4`, `ERR00100`/`ERR01100`, `acl api failed`, `AllReduceOpbasePipeline`, `RuntimeError: SUSPECT REMOTE ERROR` / `507057`, `EZ9999` device dump, hang in `determine_available_memory` / `profile_run`, a long-run `fftsplus task timeout … timeout=<N>s` hang | Lane C: collective / device error (a `profile_run` frame ending in an `npu_*` op on a 2×4 tensor is a compile-phase sync point, not an op bug) | `references/hccl-and-hardware/index.md` |
| Per-rank traceback whose frames are `torch/_subclasses/fake_tensor.py` and `torch/_ops.py`, ending `NotImplementedError: <ns>::<op>: attempted to run this operator with Meta tensors` (e.g. `_C::rotary_embedding`), after the weights loaded and before capture | Compile-phase fake-tensor trace: a custom op registered for `PrivateUse1` only. The namespace dates the tree (`_C` pre-rename vs `_C_ascend` + `meta_registration.py`), the selecting predicate is a dtype/shape test, and `--enforce-eager` is rung 1 | `references/startup-failures/meta-kernel-missing.md` |
| Model construction (after `Starting to load model`, **no** `vllm_ascend` frame below the loader) ends in `AssertionError: Torch not compiled with CUDA enabled`, from an upstream `__init__` such as a `get_rope` branch — a checkpoint `config.json` key (`dual_chunk_attention_config`) selecting an upstream class that hardcodes `torch.cuda`; the env dump's `torch==2.7.1+cpu` is normal on NPU | Checkpoint-config → upstream CUDA-only path: an unimplemented NPU feature, not a torch install, CANN, driver or flag problem. Three checks (device-agnostic class, OOT registry, consumer of the emitted tensor) decide whether a device patch is even credible; the workaround is stripping the key from a copy of `config.json`, with the feature's accuracy caveat | `references/startup-failures/checkpoint-config-cuda-only-path.md` (one call: `scripts/cuda-only-path-audit.sh`) |
| Startup log reached graph capture: `Starting to capture ACL graphs for cases: [...]`, `Capturing ACL graphs (...): 0/N` frozen, `ACLgraph sizes capture fail`, `NPUGraph.cpp` "Waiting for pending NCCL work" | Lane C during capture — or whatever the appended inner error says; the `insufficient available streams` advice is canned text | `references/startup-failures/acl-graph-capture.md` |
| A *graph/compilation setting* "未生效" — `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` or a stage yaml, nothing crashed, and the `FULL_DECODE_ONLY compilation enabled on NPU` banner never appears | One gate chain mutates `cudagraph_mode` and **each gate prints one line**, so answer with the chain plus that grep; `level: 1` means `STOCK_TORCH_COMPILE` and is refused | `references/startup-failures/graph-mode-not-applied.md` (#6972) |
| `VLLM_ASCEND_ENABLE_FLASHCOMM(1)` / `pass_config.enable_sp` / `shared_expert_dp` in play: the engine refuses a VL/Omni model up front, or `profile_run` dies with fake-tensor sizes `((sN//tp), H)` vs `(sM, H)` at an add | Sequence parallelism, not install/card: `enable_sp()` has four doors, and SP leaves the row-parallel output reduce-scattered while a per-layer full-length embedding does not shrink | `references/feature-enable-recipes/sequence-parallel-flashcomm.md` (#6261) |
| Startup was clean and **one request** kills the workers: `RuntimeError: Worker failed with error 'CUDA graph capturing detected at an inappropriate time. This operation is currently disabled.'`, logged from `patch_core.py` as "EngineCore encountered a fatal error"; typically `--enable-lora` with graph mode, where the **base-model** request dies while adapter requests work (or the reverse) | A graph was captured **at request time**: upstream `vllm/compilation/monitor.py` arms that guard `False` only after startup capture, and both capture wrappers (upstream `compilation/cuda_graph.py`, downstream `compilation/acl_graph.py`) capture lazily — so the dispatcher advertised a `BatchDescriptor` key (`num_tokens`, `num_reqs`, `uniform`, `has_lora`) that startup never built. Diff the advertised key set against the captured one; do not read the line as a device refusal | `references/lora-adapters/graph-capture-descriptor-mismatch.md` (one call: `scripts/lora-graph-descriptor-audit.sh`) |
| Device dump naming an operator: `call aclnnX failed`, `AclNN_Runtime_Error`, fftsplus aivector error, MTE/DDR "out of range"; several ranks reporting the same `pc start` in the same second — or the title alone says `aclnn<Op> operator execution error` with no log at all | Custom-op runtime fault, after the timestamp gate — a shared PC across chips is software address computation, not several bad cards. When `ls csrc` at the reporter's ref has no kernel with that name (a **built-in** CANN op reached through `torch_npu`), the host driver/CANN pairing is the first ranked cause: #5236 was reporter-confirmed solved by updating the NPU driver | `references/hccl-and-hardware/error-report-timestamps.md`, then `references/custom-ops-runtime-faults/index.md` |
| `507035` + `ERR00100 PTA call acl api failed` + `EZ9999 … aivec error` + `The DDR address of the MTE instruction is out of range`, one `pc start` repeated on every core, and the deepest frame is `torch_npu.npu_scatter_nd_update_` in `vllm_ascend/attention/sfa_v1.py` with fault kernel `ScatterNdUpdate_…`; the crash follows concurrency (mixed prefill+decode batch) while single requests are fine | DSA indexer-cache write with the wrong index tensor: a host-side slicing bug, not a card — compare the index tensor's row count with the source's before any hardware theory | `references/dsa-sparse-attention/indexer-cache-scatter-fault.md` |
| `rtMemcpy:ErrCode=107030, desc=[the current capture mode does not support this operation]` / `MemCopySync ... kind = 1` on a small `cnt` (72/456 B), often with `deepseek_mtp` + `FULL_DECODE_ONLY` | Capture/replay-time host→device sync copy — a real family (#6002, #4208, #8502) **only** when the record belongs to this process: an earlier date can be this log's own init/capture window (#10700: record 27 min before the traceback, still this process) as easily as a replay, so place it against `init engine (...)` before discarding it (rule 13) | `references/startup-failures/acl-graph-capture.md`; with MTP + `FULL_DECODE_ONLY` (+ `npugraph_ex`) also `references/spec-decode-mtp/draft-graph-capture-device-fault.md` |
| `<method> quantization is currently not supported in npu` inside a pydantic `ValidationError` from `ModelConfig` (e.g. `--quantization awq`), or `Unknown quantization method: <x>`; no `WorkerProc` line | Platform quantization gate before any worker: the NPU platform declares only `ascend` and `compressed-tensors` — re-quantize, not a defect | `references/startup-failures/quantization-method-gate.md` |
| An msmodelslim-converted W8A8 checkpoint (DeepSeek-R1-w8a8 and friends) dies at start behind the bare `RuntimeError: Engine core initialization failed` with **no worker traceback**, launched without `quantization="ascend"` / `--quantization ascend` | The gate that never fires: the documented export has **no** `quantization_config` in `config.json`, so upstream's auto-detect has nothing to read and the int8 tensors are loaded against a dense model — the flag is mandatory, not a preference (#2015) | `references/startup-failures/quantization-method-gate.md` (section B), then `silent-worker-exit.md` |
| An attention op **runs**, returns a correctly shaped tensor, and the numbers are wrong (uniform average, softmax apparently dropped, error only vs an fp32 reference); often a framework-free `torch_npu` repro, never a dump | Silent operator numerics — below the framework | `references/custom-ops-runtime-faults/attention-op-numeric-errors.md` |
| `AclNN_Parameter_Error(EZ1001)` with `Get regInfo failed` / `binary_info_config.json of socVersion [...] does not support opType [...]` — aclnn cannot resolve the op for the SoC before any launch (the *registry* family; read the text after the code, since an `EZ1001` whose text is `<Dtype> weight NZ is unsupported` is the next row) | Custom-op **build/install** fault: package built for another SoC, stale vendor package, wrong image tag | `references/custom-ops-runtime-faults/op-not-registered-for-soc.md` |
| `AclNN_Parameter_Error(EZ1001): <Dtype> weight NZ is unsupported by the current SOC version [...]` / `now self is DT_FLOAT, mat2 is DT_FLOAT`, raised at engine start from `determine_available_memory` → `profile_run` → `_dummy_pooler_run`, deepest `vllm_ascend` frames `ops/linear.py` → `ops/linear_op.py` → `F.linear`, on a `--runner pooling` / `--task score\|classify` serve | Weight-**layout** fault, the other `EZ1001` family: an fp32 pooling head cast to `FRACTAL_NZ` by the NZ gate while bf16 weights stay ND. Owner is the gate, not the custom-op package — grep `is_enable_nz` / `_should_trans_nz` at the reporter's ref, and `VLLM_ASCEND_ENABLE_NZ=0` is rung 1 (#4679; the gate was rewritten by v0.23.0, so the verdict is ref-specific) | `references/pooling-embedding/fp32-head-nz-format-crash.md` |
| `TypeError: Invalid type of HuggingFace config. Expected type: <class '…Qwen3VLConfig'>, but found type: <class '…Qwen3Config'>` from `qwen3_vl.py get_data_parser` / `get_hf_config`, in the **APIServer**, `ERR99999` after it, on a `--runner pooling --hf_overrides '{"architectures": ["<X>ForSequenceClassification"]}'` serve; a text checkpoint whose `config.json` says `model_type: qwen3` | Architecture-name **routing**: `registry._normalize_arch` rewrites the `*ForSequenceClassification` suffix to a registered base class, so a VL prefix selects the multimodal model (and its `Qwen3VLConfig` assert) on a text checkpoint. Change one string to the family-matching name (`Qwen3ForSequenceClassification`); the `classifier_from_token` / `is_original_qwen3_reranker` keys and `config.json` are innocent (#14817) | `references/pooling-embedding/seq-cls-arch-routing.md` |
| `call aclnnArange failed … EZ9999: Inner Error!` + `Parse dynamic kernel config fail` + `AclOpKernelInit failed opType` + `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed` on a **built-in** op at `Starting to load model`, often asked as "这个镜像支持 910C 吗" (a card the docs exclude) | Image-vs-card CANN **kernel package**: the built-in op has no kernel config for the running SoC, so nothing launched — not the custom-op row above, not a flag, and the container uses the image's CANN, not the host's | `references/soc-platform-paths/kernel-package-vs-card.md` (one standalone `torch.arange(8, device="npu:0")` probe decides it) |
| Output is fluent but stops after tens of tokens ("输出被截断", "只输出几十个 tokens") and a large `max_tokens` appears to fix it; `usage.prompt_tokens` is small; a `pass_config` / SP / FLASHCOMM / quantization flag changed at the same time | Early stop: `finish_reason` decides budget vs early stop, an integer `stop_reason` names the declared stop id that ended it (never the cause), then every reader of the toggled key plus the shipped support matrix (accuracy track, not the length budget) | `references/accuracy-benchmarks/early-eos-short-output.md` |
| Spec-decode metrics collapse: `Accepted: 0` with `Drafted: N > 0`, `Per-position acceptance rate: 0.000` at every position, `Mean acceptance length: 1.00`, generation throughput ~0 while requests stay `Running`; often `--async-scheduling` and/or `--enable-prefix-caching` on a hybrid GDN model (Qwen3.5/Qwen3.6, DeepSeek-V4, GLM) | Speculative-decoding bookkeeping (draft and target no longer see the same context), not capacity: the MTP-off ablation is row 1 | `references/spec-decode-mtp/acceptance-collapse.md` |
| First request after start answers `content: ""` with `usage.completion_tokens: 1` and `finish_reason: "stop"`, later requests fine; MTP plus a full-graph decode on a hybrid GDN model | Empty first response: one model-issued blank/stop token — neither the length budget nor reasoning routing — so reproduce by restart before naming a one-shot mechanism, and diff the spec/capture config against the family sibling tutorial (`"enforce_eager": true` inside `--speculative-config`, default capture sizes) | `references/spec-decode-mtp/empty-first-response.md` |
| The **first** offline `LLM.generate()` (or first serve request) is degenerate while call 2 is correct — a constant token id repeated to `finish_reason: "length"`, an immediate stop after `<think>`, or `AttributeError: 'NoneType' object has no attribute 'size'` at `torch/_dynamo/utils.py::call_size` via `compilation/acl_graph.py` — on a hybrid GDN checkpoint of a VL-capable architecture, TP≥2, often `enable_sleep_mode=True` | Two owners, one field apart: a constant-tie logits vector retires the numerics lanes, so read the response first — the crash is the text-only `is_multimodal_model` vs `supports_mm_inputs` drift in `_dummy_run` (#14638, fix on main, not backported); the garbage is first-request state that sleep mode exposes (reporter: "give the `LLM` some time"), with `expandable_segments` the confound inside the `sleep=False` row | `references/accuracy-benchmarks/first-generate-degenerate.md` |
| "score with tool A is N points below tool B / another image / the vendor number", accuracy regression after a change | Accuracy/score discrepancy | `references/accuracy-benchmarks/index.md` |
| An **embedding / pooling** checkpoint never starts: `EngineCore failed to start` at `Loading safetensors checkpoint shards: 0% … 0/1`, deepest frame `vllm_ascend/models/<x>.py load_weights` → `vllm/model_executor/models/utils.py` `raise ValueError`, message `There is no module or parameter named 'embed_tokens' in Custom…ForCausalLM`, and the init line says `pooler_config=None` | An embedding checkpoint served as a **generation** service: the `ValueError`'s *prefix* (no `model.`) and *class name* (a `ForCausalLM`, where a pooler build would read `…ForEmbedding`) both say so before any source read, and `--task embed` plus the wrapper's `WeightsMapper({"model.": ""})` is the whole fix — the downstream loader is a verbatim upstream copy, not the owner | `references/pooling-embedding/embedding-checkpoint-as-generate.md` (one call: `scripts/checkpoint-key-prefix.py <model dir>`) |
| An **embedding / pooling** service (`--runner pooling`, `--task embed\|score\|classify`, `/v1/embeddings`) dies **under concurrency** (vllm-bench, `--max-num-seqs` in the tens-hundreds) while single-request serving is fine: engine log ends in `KeyError` inside `reorder_batch_to_split_decodes_and_prefills`, then `EngineDeadError` / `400 Bad Request` | Pooling batch bookkeeping: a downstream `InputBatch` copy's `is_pooling_model` slot fast path, not the attention backend | `references/pooling-embedding/batch-index-desync.md` |
| An **embedding** service returns **wrong vectors** for long inputs (cosine vs a reference service collapses from `>0.999` to `0.2-0.6`) while short inputs are correct, text tasks are fine, and `--no-enable-chunked-prefill` removes it | Pooling + chunked prefill: which token span got pooled, then the Ascend pooling attention/mask branches | `references/pooling-embedding/index.md` |
| Per-layer `MSE loss` / epoch table printed while converting a checkpoint with msmodelslim (W4A4/W8A8), one layer far above its neighbours, or "the converted checkpoint answers garbage" with no engine-side failure | Quantization-conversion phase: converter-owned, before the engine starts | `references/accuracy-benchmarks/quantized-checkpoint-conversion.md` |
| A long prompt (100K+) answered as 乱码 / garbage / repeated / empty text while short prompts are fine, with sparse c8 attention, DCP/PCP, PD disaggregation or KV offload enabled | Generation corruption (accuracy track, different evidence) | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| A **few** responses in a long accuracy run repeat sentences ("5 条响应的句子重复") on a documented model with PD + MTP + prefix caching, prompt short but `max_tokens` in the tens of thousands, no crash and no error text | Decode-time repetition, not prompt-length corruption: the owners are reuse and draft bookkeeping (prefix cache, preemption, MTP), so the answer is the tutorial P/D diff plus the `--max-num-seqs 1` identity row — long-CoT models loop on their own | `references/accuracy-benchmarks/long-output-repetition.md` |
| 乱码 / token soup on a few-thousand-token prompt, process alive, no traceback, `finish_reason: length`, often MoE + EP across nodes or a quantised weight set | Generation corruption, short-prompt profile | `references/accuracy-benchmarks/short-prompt-corruption.md` |
| Garbage/repetition that the reporter ties to **token content** — "only when the prompt contains these special tokens (id N)", identical on every release they tried, one model only, "works on GPU" | Refute the id framing first: identical ids on both input routes, no id-keyed branch in `vllm_ascend/`, TP=1 embedding has no mask to zero with. Then `Using max model len N` (the checkpoint's own `config.json`, not the launch command) and one CPU run of the same directory | `references/accuracy-benchmarks/short-prompt-corruption.md` (step 0) |
| 乱码 on a **short** prompt of a hybrid linear-attention model (Qwen3.5/Qwen3.6 GDN and siblings), where the "正常" command differs from the failing one by a scheduling knob — `--block-size` (e.g. 512 → 2048) and/or `--enforce-eager` — and nothing else | Hybrid KV-cache geometry: the cache block is never clamped to 128 on these models and is only raised by the mamba/attention page-size constraint, so a bigger block re-slices block tables, slot mapping and the mamba state block at once | `references/accuracy-benchmarks/hybrid-block-size-corruption.md` |
| A **VL image/video** request answered as a long run of one token (`!!!!`, 感叹号, `，，，`), usually with "特别耗时 / very slow"; text-only prompts may answer fine | Vision-tower degeneration (accuracy track), not sampling | `references/accuracy-benchmarks/vl-vision-tower-degeneracy.md` |
| The server served traffic then died on memory: "运行一段时间后显存不释放最后崩溃", `NPU out of memory` / `NPUWorkspaceAllocator` / `Memory_Allocation_Failure(EL0004)`, `npu-smi` pinned at 100 %, a P node holding KV the D node never takes | Runtime memory after a healthy period: capacity vs growth vs a config key that never engaged (contrast the Lane C row, where the OOM happens during `profile_run` / `determine_available_memory` before the server is ready) | `references/runtime-memory/index.md` |
| "The machine crashed" / the host became unusable / the reporter pastes `top` with every python3 rank at tens of GiB RES on one host, while the serve log has no device, collective or Python error | Host RAM (and host CPU), not the card: the ranks' resident footprints share one RAM budget, and `npu-smi` cannot see it | `references/runtime-memory/host-ram-and-per-rank-rss.md` + `scripts/host-ram-check.sh` |
| …or the box died **during engine start** and there is no `top` to sample: ssh lost, the log's last lines are the per-rank `rank N in world size M is assigned ...` set with **no** `Starting to load model`, and `--max-model-len` is 128K+ (256K = `Using max model len 262144`) | One eager host allocation, not a workload footprint: every worker builds the `max_model_len²` attention mask in `NPUModelRunner.__init__` — 9·N² bytes **peak** per rank (576 GiB at 262144, ×TP on one host) — before a weight is loaded, so `npu-smi` shows cards nearly empty and the reporter's "313T"/HCCL framing is a red herring (#2873 reporter-confirmed, 0.10.1rc1 mask refactor #2442; 0.10.0rc1 works; #3002) | `references/runtime-memory/attention-mask-host-allocation.md` + `scripts/attn-mask-budget.sh` |
| A container-scoped counter climbs for days while the service stays healthy — `container_memory_working_set_bytes` "持续增加，7 天增加 80G", `container_memory_rss` / `container_memory_cache` beside it, `npu-smi` flat, no traceback and no OOM | Container **host** memory, not HBM: the counter sums every process in the container plus active page cache, so split rss from cache and convert the slope to bytes/request before any code read — the device-side knobs in the report cannot move it | `references/runtime-memory/container-working-set-growth.md` (one call: `scripts/working-set-arithmetic.sh`) |
| Startup/KV-init failure whose deepest frame is `_convert_torch_format` / `torch_npu.npu_format_cast` with `Memory_Allocation_Failure(EL0004)`, `alloc device memory failed … 207001`, `507013`, SDMA/SMMU text | Allocation failure at KV-cache init — the cast is the allocation site, not an operator bug | `references/runtime-memory/kv-init-and-format-cast-alloc.md` |
| Engine start dies on **every** rank in `torch.npu.empty_cache()`: `RuntimeError: unmapHandles:…NPUCachingAllocator.cpp:595 … AclrtUnmapMem((char *)ptr_ + segment_size_ * i, getHcclComm()), error code is 507899` with `ERR00100` / `EL0003`, deepest frame `patch_torch_accelerator.py` inside the `memory_profiling` window right after graph compile; the issue title is often a different line (`unable to save AOT compiled function`), and the reporter's own launch script exports `expandable_segments:True` (`unset` does not help — see the other column) | The allocator **unmap** under `expandable_segments:True`, not an OOM and not a collective: vllm-ascend injects that flag itself, so `unset` is not off. Reporter-confirmed on 310P: `expandable_segments:False` starts it | `references/runtime-memory/allocator-expandable-segments-unmap-crash.md` (one grep: `Set PYTORCH_NPU_ALLOC_CONF`) |
| `python -c "import <pkg>"` (or a first real use) raises `ImportError:` naming that package's own extension — `<pkg>/_ops.cpython-310-…so: undefined symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs` — and the ENV dump selects `atb/cxx_abi_1` (or `cxx_abi_0`) | C++ `std::string`-ABI split between a prebuilt wheel and the selected NNAL/ATB root, pre-engine and not vllm-ascend code: `Ss` in the mangled name = old ABI, `NSt7__cxx11` = new | `references/image-dependency-faults/abi-undefined-symbol.md` (one call: `scripts/abi-symbol-check.sh <paste>`; fix reporter-confirmed on #1144, the defining ATB file still the reporter's `nm` step) |
| `python -c "import <pkg>"` or the first use of an image-installed native module aborts the interpreter: `corrupted size vs. prev_size`, `free(): invalid pointer`, `double free or corruption`, `munmap_chunk(): invalid pointer`, `Aborted (core dumped)`, no Python frame, no NPU text | Image / bundled native module (pre-engine, not vllm-ascend code) | `references/image-dependency-faults/index.md` |
| Engine init ends in a C++ exception with **no** Python frame: `terminate called after throwing an instance of 'std::logic_error'` / `what(): basic_string::_S_construct null not valid`, then `RuntimeError: Engine core initialization failed … Failed core proc(s): {}`; the last application line is plugin activation or `Initializing a V1 LLM engine` | Native null-string abort in a bundled `.so` during core/worker creation — an image/host question, not the checkpoint's config, processor or architecture (the same two lines appear for two unrelated models) | `references/image-dependency-faults/native-string-abort.md` |
| `POST /start_profile` (or `/stop_profile`) returns `404 Not Found` although the server was started with `--profiler-config '{...}'`; or the same route answers 500 naming `Profiling is not enabled` | Observability/route registration: the endpoint is attached by upstream `serve/profile/api_router.py` `attach_router` only when the serving process's `args.profiler_config.profiler` is set — not a model/device question, and a down-stream worker miss is a 500, never a 404 | `references/observability-and-logs/online-profiling-endpoints.md` (one `openapi.json` request) |
| `RuntimeError: Already borrowed` raised inside a HuggingFace fast tokenizer (`set_truncation_and_padding` → `self._tokenizer.no_truncation()`), typically under `_preprocess_chat` while a tool parser's `__init__` encodes its special tokens, wrapped by `serving.py … Error in preprocessing prompt inputs`; more generally any traceback whose frames are only `vllm/**` + `site-packages/**` with no `vllm_ascend` frame and no device text | Upstream serving-frontend / Python-dependency fault, not the NPU stack: one tokenizer object driven from the event loop and from a worker thread, and vLLM's own retry for this exception covers only the multimodal processor. Mechanism source-verified at vllm v0.16.0; the trigger (client concurrency) is unverified and the serial rerun settles it | `references/serving-frontend-faults/index.md`, then `tokenizer-already-borrowed.md` |
| Report is about log text rather than a crash: `--enable-log-requests` / `--enable-log-outputs` set but no prompt or generation printed, `VLLM_LOGGING_LEVEL=DEBUG` seemingly ignored, `--max-log-len`, a stage of the startup log missing, or a proxy that swallows output | Observability/flag question — **upstream frontend config, not a vllm-ascend defect**; answer with a gate checklist and two greps, cap it at ~4 tool calls | `references/observability-and-logs/index.md`, then `request-output-logging-flags.md` |
| Startup `WARNING … [envs.py:1717] Unknown vLLM environment variable detected: VLLM_ASCEND_<NAME>` (one line per variable, engine then starts; reporter calls it 误报 and says the export is commented out) | Upstream env-registry check, **warn-only** and cosmetic: vllm-ascend keeps its own `vllm_ascend/envs.py` registry and never merges it into `vllm.envs.environment_variables`, so any `VLLM_ASCEND_*` in that process's environment warns. Owner is the missing merge in `AscendPlatform.pre_register_and_update`, not FlashComm or the card — one `env \| grep -i vllm_ascend` | `references/observability-and-logs/env-var-validation-warning.md` |
| A **tool call** is the symptom, not the prose: `JSON Parse error: Unterminated string` / `Expecting ',' delimiter` / `Unterminated string starting at: line 1 column N` while the client assembles `tool_calls[0].function.arguments`, arguments reported truncated / 格式错误 / 重复输出, `JSON.parse` failing on `{"hvd": Pura900"}` (balanced braces, one quote short, `stream: false` clean), or `stream: true` returns only CoT and no `tool_calls`; often with `--speculative-config '{"num_speculative_tokens": 3, ...}'` where 1 正常, or with `--async-scheduling` | Tool-call output integrity: upstream vLLM's delta assembly vs the vllm-ascend sampler vs the model stopping early — decided by concatenating the argument fragments, then by one `return_token_ids: true` request, not by reading a parser or a kernel; a *string* value streamed with no opening quote is the parser's delta keyed on the argument's declared JSON-Schema type | `references/tool-call-streaming/index.md`, then `streamed-arguments-audit.md` (`scripts/sse-toolcall-audit.sh <raw-sse>`); the missing-opening-quote shape `arg-type-gate-unquoted-value.md` (`scripts/tool-schema-type-audit.py <request-body.json>`) |
| A constant in the code equals the number in the report — `num_speculative_tokens: 3` fails while `1` (and `2`) work — and the wrong output is exact-form: tool-call JSON, ids, code | Named code branch, not an investigation: `using_block_verify = max_spec_len >= 3` is the only spec-decode branch keyed on that value, **but newer refs also require `rejection_sampler_config.enable_block_verify`, default `False` (v0.23.0rc1) — read the line first**; then state it with its two ablations (`n=2`, `temperature: 0`) before its kernel is read | `references/tool-call-streaming/mtp-block-verify-acceptance.md` |
| Report pastes only a response body: the CoT/thinking text sits in `content`, `message.reasoning` / `reasoning_content` never appears, `completion_tokens_details.reasoning_tokens` is `0`, thinking language mixed, or the complaint is only "回答中英混杂 / the answer is in English" while `content` reads `null` | Output routing — **upstream OpenAI frontend, not a vllm-ascend defect**: `reasoning_parser` defaults to `""`, so `--reasoning-parser <name>` (e.g. `qwen3`) is the whole gate; invariant to image tag and MTP | `references/accuracy-benchmarks/reasoning-output-routing.md` |
| Startup fails with `KeyError: "Reasoning parser '<name>' not found. Available parsers: …"` (or `invalid reasoning parser: <name> (chose from {…})`), usually right after the reporter tries a name taken from their `--tool-call-parser` line (`qwen3_coder`), or asks "qwen3_coder可以吗?" | Two separate registries: `--reasoning-parser` reads `vllm/reasoning/__init__.py`, `--tool-call-parser` reads `vllm/tool_parsers/__init__.py` (v0.18.0), so the flag is the whole gate and the printed list also fingerprints the release. A Coder / non-thinking checkpoint needs **no** reasoning parser: `qwen3` with no `</think>` routes the whole answer to `reasoning` and leaves `content` empty | `references/accuracy-benchmarks/reasoning-output-routing.md` (name-space section) |
| Host is **310P** (Atlas 300I Duo, `Ascend310P3`, image `-310P`), or the traceback sits on an `is_310p()` branch (NZ/`FRACTAL_NZ` format casts, no Ascend custom ops, `patch_mamba_config_310`) | SoC-conditional platform path: experimental per the pinned docs, NZ casts are real conversions, fp16/eager are mandatory | `references/soc-platform-paths/atlas-300i-310p.md` |
| `EngineCore failed to start` whose deepest frame is `worker.py` → `check_ascend_device_type()` with **no** `Starting to load model` line: `AssertionError: Current device type: AscendDeviceType.A5 does not match the installed version's device type: AscendDeviceType.A2, please check your installation package.` (or `RuntimeError: Can not support soc_version: <n>`), typically on a 950PR/A5 host that installed a mirror wheel | SoC/package correspondence: the device type is **baked into the package at build time** (`_build_info.__device_type__` from `SOC_VERSION`), the release pipeline ships a2/a3/310p variants only while `Dockerfile.a5` exists in-tree, so an A5 host needs a source build on the node — and the guard must not be patched | `references/soc-platform-paths/device-type-build-vs-runtime.md` (two one-liners decide it) |
| The report asks how to enable a feature rather than what broke: "如何正确开启 PCP/DCP", "F alone and G alone each报错，能否给个能跑通的示例", no log pasted, or two errors described in one sentence | Feature-enable / capability question: the validation verdict (docs statement at the pinned ref, whose hardware the measured table used, `tests/e2e/**` coverage, precedent versions) before any constraint list | `references/feature-enable-recipes/index.md`, then `context-parallel-pcp-dcp.md` |
| `Communication_Error_Bind_IP_Port(EI0019)` / `Failed to enable listening for the host network adapter socket` / `port <n> have already been bound` where n is above 65535, often in the issue title | HCCL control-plane port band, not an OS socket conflict — and only if the record's date is inside the window (rule 13) | `references/hccl-and-hardware/control-plane-port-binding.md` |
| Output degenerates (`</think>` leaking into `content`, repetition collapse, 乱码) at long-ish context on a **documented** model with PD / spec-decode / prefix-cache flags, with **both** `vllm serve` commands pasted and no ablation; a concurrency claim is usually attached but rarely matches the reporter's own table | Config-diff track: the tree's own `docs/source/tutorials/models/<model>.md` ships the validated P and D commands and a `Recommended Configurations` table, so the first answer is that diff plus the ablation matrix — never a walk through the KV connector; #13002's five deviations are ranked ablation rows, not established causes | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| "服务崩溃" / 崩溃 under concurrency or high load with **no** error text, no traceback and no plog pasted, while single-request serving is correct (often DeepSeek-V3.2 + MTP + DP/EP) | Report-gap track: nothing in the paste can confirm a mechanism, so the answer is the config diff against the project's own validated command at the pinned commit plus 1-2 single-variable A/Bs | `references/dsa-sparse-attention/concurrency-crash-with-no-log.md`; `references/runtime-memory/index.md` for the OOM row |
| "多并发时请求排队那么久" / "生成吞吐只有个位数" / `Prefix cache hit rate: 0.0%`, with a `loggers.py` metrics block, a healthy engine and no error text | Throughput/queueing on a healthy engine: TTFT is prefill work ÷ prefill throughput, the interference is `gen tok/s ÷ Running`, and the cache line is a capability switch before it is a tuning knob | `references/performance-throughput/index.md` (both pages in one turn when the paste carries both complaints — the arithmetic page first) |
| Throughput falls **in one step** as one concurrency number is crossed — recorded 256 running requests at 3048 tok/s, 258 at 1182, 264 at 1100, with `Waiting` 0, KV 81-84 %, no error or warning, AICore 32 % → 8 % and ~3 host cores busy (tp=8, MTP, `FULL_DECODE_ONLY`, DeepSeek-V4-Flash-w8a8) | Graph-dispatch ceiling: a decode step is `running_seqs × (1 + num_speculative_tokens)` tokens against `max_cudagraph_capture_size`, which Ascend defaults to `min(max_num_seqs × decode_query_len, 512)`; above it `cudagraph_dispatcher` returns `NONE` with no log and every step is eager. One call — `scripts/seq-cliff-capture-budget.sh` — gives the threshold from the serve command alone | `references/performance-throughput/sequence-count-cliff-capture-size.md` |
| `Running: 1`, `Waiting: 0`, KV usage below 1 %, `Avg generation throughput` 4-6 tok/s on a hybrid GDN MoE (Qwen3-Next-80B-A3B) at TP=4, when the project's own tutorial command at that ref already carries `--enforce-eager` | Single-stream decode on a release that cannot capture a full graph: ~190 ms/step against a ~1 ms per-card HBM floor is dispatch-bound, so the lever is the release (newer `FULL_DECODE_ONLY` tutorial command), not a flag diff — and the flag flip is a hypothesis until the `PIECEWISE compilation enabled on NPU` line is confirmed | `references/performance-throughput/qwen3-next-decode-throughput.md` |
| Base model reaches N tok/s and the same model under `--enable-lora` a fraction of it (recorded 9 → 2 tok/s), no error, no OOM, single stream, often on 310P where `--enforce-eager` is mandatory | LoRA decode **overhead**, not arithmetic: which punica backend the ref binds (`vllm.lora.ops.torch_ops` per-slice gather+cast+einsum vs the AscendC `vllm_ascend/lora/lora_ops.py`, plus the 310P / `max_lora_rank >= 128` predicate that forces torch), then whether that SoC can amortise launches at all (300I is eager-only) | `references/lora-adapters/decode-overhead-vs-base-model.md` |
| A 19-token to few-hundred-token prompt on a hybrid GDN model (Qwen3-Next) answered as a digit run or a repeated phrase while ordinary chat is fine, `finish_reason: length`, no traceback | Hybrid linear-attention numeric path (accuracy track): the prefill GDN kernel is a downstream fp32 re-implementation while decode keeps a patched Triton kernel | `references/accuracy-benchmarks/qwen3-next-gdn-numeric-path.md` |
| "去掉 `HCCL_OP_EXPANSION_MODE`(AIV) / `PYTORCH_NPU_ALLOC_CONF` 就能拉起", first in-window error `aclnnMoeDistributeDispatchV4` / `…CombineV4 … 107020`, `event wait timeout … task_timeout` | The env knob changed the **selected MoE path**, not the hardware: AIV × MC2 dispatch/combine is a defect family (#14928, #13688, #15195), rank arithmetic decides whether MC2 was eligible, and `expandable_segments` is innocent apart from the CaMem pool assert | `references/hccl-and-hardware/aiv-mc2-env-ablation.md` |
| The reporter wants `--enable-expert-parallel` **and** `--data-parallel-size` > 1 together ("同时开启 EP + DP", "去掉 EP 或 DP 设成 1 就不报错"), or engine init dies at `vllm_ascend/ops/fused_moe/prepare_finalize.py` `dist.all_gather` behind `No available shared memory broadcast block found in 60 second(s)` on 910B | MoE communication-method selection, not a flag bug: A2 picks `MC2` only when `num_tokens <= mc2_tokens_capacity` **and** `world_size_across_dp/pp >= 16`, so DP2×TP8 = 16 is the boundary while DP=1 and EP-off fall to `ALLGATHER` by arithmetic; `PrepareAndFinalizeWithMC2` inherits the All2All `finalize`, so that all_gather **is** the MC2 path | `references/feature-enable-recipes/moe-expert-parallel-with-dp.md`; a real-request hang or garbage instead → `dp-moe-padding-and-owner-slice.md` |
| Startup dies in `profile_run` / `determine_available_memory` with `507014` "The aicore execution times out", deepest Python frame `aclnnInplaceCopy` or an `ERR02005` HCCL-watchdog abort, repeated `No available shared memory broadcast block found in 60 seconds`, and the `[AIC_INFO]` line names a MoE all-to-all (`fault kernel_name=MoeDistributeDispatchV2_…`) | The profiling batch is `mc2_tokens_capacity`, not `--max-num-batched-tokens`: `profile_run` runs an extra capacity-sized dummy batch when EP is on, `world_size_across_dp/pp >= 16` and capacity < max tokens, so the first collective of the process is an MC2 dispatch. The `HCCL_INTRA_PCIE_ENABLE=1/ROCE=0` pair behind the hierarchical variant is the *documented* A2 setting | `references/hccl-and-hardware/mc2-dispatch-profile-run-hang.md` |
| The body asks for a source change with no symptom — "Remove `<ENV_VAR>` before the final release", "delete the switch added in PR #N", "deprecate this flag", usually only PR/review-comment links, no env dump and no log | Code-change / flag-retirement request: grep the identifier at the pinned ref first; absence means the snapshot predates the PR and the answer is the removal sweep plus the default-flip note | `references/code-change-requests/index.md`, then `env-flag-retirement.md` |
| Deepest `vllm_ascend` frame is in `vllm_ascend/patch/**` (`patch_model_runner.py`, `patch_core.py`, …) | Patch **copy**: each `patch/` module assigns a verbatim upstream method under a `vllm_version_is('<ver>')` gate, so the failing statement may be upstream text. Diff the copy against the pinned upstream definition before deriving anything | `references/upstream-downstream-drift/patch-copy-vs-upstream.md` (2-3 calls) |
| `--async-scheduling` **and** `--speculative-config` with fractional `temperature` (`0` is fine): worker `KeyError: '<id>'` in `_update_states`, EngineCore `'NoneType' … sampled_token_ids`, APIServer `Counters can only be incremented by non-negative amounts`, same second, every DP rank | One provenance, three symptoms: a discarded request's valid count is 0 → `num_accepted = -1`, and `prev_req_id_to_index` omits exactly those requests; the unguarded lookup is upstream vLLM's, carried by the patch | `references/spec-decode-mtp/async-spec-bookkeeping-desync.md`; row 1 is `--async-scheduling` off |
| Deepest `vllm_ascend` frame is `counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)` in `_copy_valid_sampled_token_count` (`model_runner_v1.py:1393`), batched with `Execution_Error_SDMA` / `rtMemcpyAsyncEx … reason=the model stream execute failed`, an `aivec … timeout or trap error` and an HCCL `AllGather`/`AllReduce` line carrying `rankSize` / `numBlocks`; DP or PD deployment, under concurrency | Device fault surfacing at the drafter's valid-count D2H copy (Lane C sync point) — and the frame is also free evidence: it only runs with `--async-scheduling` **and** spec decode **and** the padded drafter batch, so the paste proves the launch shape. Resolve the collective by rank size against the DP×PCP×TP group arithmetic (a 2-rank group is not `fc3_quant_x` on DP2×TP4), then `--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1` for attribution | `references/spec-decode-mtp/drafter-count-copy-device-fault.md` |
| `ACL stream synchronize failed, error code:507015` / `EE9999 … aicore exception` whose deepest frame is an attention backend while the step held **both** decodes and prefills; often "completes with spec decode disabled" | Mixed-batch split **units**: `num_decodes` counts requests while `num_decode_tokens` counts tokens (× `1+num_spec`), so a request-indexed `block_tables`/`seq_lens` sliced by the token count, or an unrebased `query_start_loc` passed as `cu_seqlens`, reads out of its own storage; the reported line is often the decode half's `.npu()` | `references/mixed-batch-splits/index.md`, then `fa3-mixed-batch-split.md` (one call: `scripts/batch-split-audit.sh <root>`) |
| KV pool / MemCache on **800I/T A3** with a DeepSeek-V4 (`compress_ratios`) model: first request fine, the **second** kills the workers (`HCCL watchdog … ERR02005`), or the `host_shm` variant dies at memcache init with a "device id 转换异常"; reporter says "懒加载" | KV-pool store lane: lazy init is the documented A3+DSV4+`device_sdma` behaviour (#9975) and a rank-local save failure is swallowed before it desyncs collectives (#14106) — diff the resolved `mmc-local.conf` against the tree's tables first | `references/kv-pool-store/memcache-a3-dsv4-init.md` |
| `ERR00100 PTA call acl api failed` + `EZ9999 … When layout is TND, queryT(16) must be equal to the last element of actualSequenceLengthQ(4)` (`CheckFASeqlenDataInTND`, `fused_infer_attention_score.cpp`), no Python frame, real requests after a healthy start, on `FULL_DECODE_ONLY` + MTP (`num_speculative_tokens` recorded as 3), often `--data-parallel-size` > 1 with EP, on an MLA model (GLM-4.x, DeepSeek-V3.x) | The operator's own TND invariant refused two padding bookkeepings that disagree: `queryT` is the graph bucket (local `pad_for_cudagraph`, then the DP-wide max) while `last(actualSequenceLengthQ)` is the real per-request cumsum, padded only on the FULL + `uniform_decode` gate (drafter: only when `graph_pad_size > num_reqs`) — one call, `scripts/fia-tnd-seqlen-check.sh`, splits bucket-vs-real; `PIECEWISE` is rung 1, and the tree's GLM-4.x tutorial pins `num_speculative_tokens: 1` + explicit capture sizes + single node | `references/spec-decode-mtp/fia-tnd-seqlen-mismatch.md` |
| `triton.compiler.errors.MLIRCompilationError` + `[ConvertLinalgRToBinary] encounters error: … operation's operand is unlinked` + `bishengir-compile … returned non-zero exit status 1`; engine started, the first real request dies; deepest `vllm_ascend` frames in `vllm_ascend/ops/triton/fla/**` under `triton/runtime/autotuner.py`; the recipe has a `pip install triton-ascend` or a standalone BiSheng/AscendNPU-IR `.run` over the image | Triton-Ascend kernel **compile** pair installed into the image — not the AscendC/aclnn custom-op lane and not the kernel's shapes: the wheel and `bishengir-compile` come from different builds | `references/image-dependency-faults/triton-ascend-pair.md` (remove the install, then move the release — maintainer-confirmed on #5750) |
| The report **is a pip resolution report** and nothing has crashed: `ERROR: Could not find a version that satisfies the requirement torch-npu==2.5.1.post1.dev20250619`, `No matching distribution found for <pkg>`, or a title like "unmatched torch version between vllm=X and vllm-ascend=Y"; also `pip install pandas` inside the shipped image ending in `No matching distribution found` after an `ImportError: Please install vllm[bench] …`; or the install **finished** and pip only warned (`vllm 0.18.0 requires torch==2.10.0, but you have torch 2.9.0`) | Index visibility, not compatibility: the tree's `--extra-index-url`/`--pre` option lines never reach the wheel's metadata (`setup.py` skips `--` lines), the same requirement set can straddle two torch lines, and an air-gapped image's `global.index-url` makes every lookup unreachable — one call (`scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]`) prints both sides' `torch`/`torch-npu`/`torchvision` pins | `references/image-dependency-faults/pip-index-visibility.md` (#1796, #4737); warned-not-failed shape: `pip-resolver-warning-block.md` (#8286) |
| "找不到支持 linux on arm64 的 <soc> 镜像" / the quay tag page lists an arm64 tag but the pull lands on amd64 (`WARNING: The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8) …`) / "有没有可 pip 装的 <soc> 轮子" | Published-artefact availability, not a vllm-ascend defect: tag suffix → SoC variant (unsuffixed tags and `latest` are the generic A2 line) and multi-arch only where a build may publish (the same `push:` expression), so the single-arch fallbacks cover check builds and a published tag resolving to one arch means the arm64 leg failed on the `FROM` CANN base or a mirror flattened the list — then one `docker buildx imagetools inspect`; a mirror tag page and the client warning are not evidence | `references/image-dependency-faults/image-arch-and-tag-availability.md` (one call: `scripts/image-arch-matrix.sh <vllm-ascend-root> --soc 310p`) |
| Compiler or missing-header error while installing/building — or `pip install` from source in a pod with **no public network** (custom-ops step pulls from `gitcode.com`, `fetch failed`) | Build-time dependency closure, else the general sequence below | `references/image-dependency-faults/offline-source-build.md`, else skip the topical routes |
| `Unsupported: Graph break due to unsupported builtin mindie_turbo._ops.PyCapsule.rms_norm` (or any accelerator wheel's symbol) while the model forward is traced, frames `vllm/compilation/decorators.py` `patched_inline_call` → `torch/_dynamo/**`; often "mindie_turbo 算子不支持图模式?" beside "但是 vllm serve 正常运行" | An **op-substituting plugin × `fullgraph` compile**: both worker classes import such a wheel unconditionally with no flag, the pybind capsule is untraceable, and `VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE` defaults to 1, so a graph break is fatal instead of a fallback; the wheel's INFO line names which process had it, its compat matrix answers the pairing | `references/image-dependency-faults/mindie-turbo-graph-break.md` |
| Nothing crashed and the complaint is a **version string**: `pip show vllm-ascend` reports an older release than the reporter installed ("版本号显示不对", "0.9.0rc2 instead of 0.9.1dev"), or they ask whether a `-dev`/branch name is a valid version | Build-metadata question: the number is `setuptools_scm`'s description of the build tree's tags (`setup.py:305-311`), `vllm_ascend/` has no version constant and no runtime reader, and the pairing gates read `vllm.__version__` — decide it with the import path plus `git describe`, not a version survey | `references/image-dependency-faults/installed-version-vs-checkout.md` |

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
   log at all but references an image, the OCR text under
   `/workspace/attachments/<issue>/` **is** the paste: read `ATTACHMENTS.txt` then
   the `.ocr.txt` (a few hundred bytes) before theorising
   (`references/evidence-workspace/index.md`).
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
     requested — at v0.11.0rc0 `platform.py:139-150` force-enables the
     `AscendScheduler` fork for a default dense serve and overwrites an explicit
     `enabled: false` — so grep the config field in `vllm_ascend/platform.py`
     before asking which flags they passed, and before believing "you must have
     enabled it" (recorded #3979).
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
   `vllm_ascend/models/__init__.py` registration and the upstream model file —
   for the recognition-error form, answer from
   `references/startup-failures/model-type-not-recognized.md` before any deeper
   walk.
6. One grep of `history/` for the distinctive symptom token, 2-3 hits, then stop
   (rule 8); `bash <skill_root>/scripts/history-titles.sh [-i] '<regex>'` gives the
   titles of all hits in one call without opening the bodies (`-i` when the
   corpus capitalises the token, e.g. `reranker` vs `Reranker`).
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
   invalidate the claim.
9. Re-emit the answer in the shape below with the corrections, then stop.

## Answer shape

- The direct answer first: yes/no for the reporter's literal question, the
  version that works or the version where it breaks, and whether the reported
  crash is real. One sentence, before any analysis. Write it in the language the
  report was filed in (the recorded Chinese-issue answers were Chinese and still
  carried the `path:line` table). **The cheapest next action belongs in that same
  first paragraph**, not below the evidence table: a session can be cut wherever
  it happens to be, and #7580 was cut inside its evidence table, after the verdict and
  before the ablation plan. Verdict plus one action first; table and
  mechanism after.
- Then the one-line classification: lane A/B/C and the phase that failed (load,
  memory profiling, graph capture, first collective); or, for a runtime report,
  the primary cause plus its track.
- Ranked causes, each with the evidence you have, the exact check that confirms
  or refutes it, and its confidence label (reporter-confirmed / source-verified
  at commit X / hypothesis).
- An evidence table (`path:line`) for code claims, with precedent issue ids
  quoted as `#<id>: <one-line fact>`.
- The action ladder, cheapest first (the first rung is already in the opening
  paragraph): an explicit `--max-model-len`, a re-run with `--enforce-eager` /
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
| score/accuracy disagreement, 乱码/重复 on short or long prompts, truncated output, reasoning routing, VL degeneration, quantized-checkpoint conversion | `references/accuracy-benchmarks/index.md` |
| the asking is for an edit: remove/deprecate an env flag or a temporary switch | `references/code-change-requests/index.md` |
| a custom op crashes the device, or an aclnn SoC **registration**/`EZ1001` error (the NZ weight-layout one is the pooling lane), or a MoE init-routing tiling code | `references/custom-ops-runtime-faults/index.md` |
| a `call aclnn<Op> failed, error code is <6 digits>` on a MoE deployment (`aclnnMoeInitRoutingV3 … 561103`), often with "DP2 TP4 有没有限定 / 能否启动时校验拦截" attached | `references/custom-ops-runtime-faults/moe-init-routing-tiling.md` — the code is a failed **tiling check on one input**, never a card/HCCL/"layout unsupported" claim, and the named op is not a grep key |
| capacity or stage-layout questions, and two serve commands that differ by one resource knob where the larger OOMs at start | `references/deploy-sizing/index.md` — the TP4-up/TP8-OOM shape is its `tp-scaling-oom-at-startup.md` page |
| DSA / DeepSeek-V3.2 sparse attention (indexer scatter, MLAPO token gate, concurrency crash with no log) | `references/dsa-sparse-attention/index.md` |
| version contract, snapshot coverage, precedent search, delivery gates | `references/evidence-workspace/index.md` |
| "how do I turn feature F on" / capability verdict (PCP, DCP, EP+DP) | `references/feature-enable-recipes/index.md` |
| the first cheap checks after classification | `references/first-checks/index.md` |
| a collective/device error (`HcclAllreduce`, `ERR00100`, `507035`, `EZ9999`), device health, port band, MC2 profile-run hang, RPC watchdog | `references/hccl-and-hardware/index.md` |
| the interpreter/image owns it: native aborts, missing `libatb.so`/`ascend_transport.so`, ABI split, triton-ascend pair, offline source build, pip index visibility or a warned-not-failed install, an accelerator wheel's symbol breaking `torch.compile`, no image/wheel for that arch or SoC variant | `references/image-dependency-faults/index.md` |
| a KV pool (AscendStore + MemCache/Mooncake) is attached and the request path breaks | `references/kv-pool-store/index.md` |
| a LoRA-enabled serve is far slower than its base model, or dies when a graph is captured on a request | `references/lora-adapters/index.md` |
| a device fault or corruption whose deepest frame is an attention backend in a mixed decode+prefill batch | `references/mixed-batch-splits/index.md` |
| a logging/profiling flag appears to do nothing, `/start_profile` 404, unknown-env warning | `references/observability-and-logs/index.md` |
| P/D or EPD starts but cannot move KV, moves it and the output alternates, or the client times out while D logs no request and only P prints `Force freed expired request … (480 seconds)`; **or** the transport *plane* is asked about before any run — A3 双机, 参数面 RoCE only, no 灵衢 switch, no log → `plane-selection-a3-parameter-plane.md` | `references/pd-disaggregation/index.md` |
| throughput, queueing, prefix-cache hit rate on a healthy engine | `references/performance-throughput/index.md` |
| an embedding/pooling service never starts (checkpoint served as generate, score/rerank arch-name routing, fp32 head weight format), dies under concurrency, or returns wrong vectors | `references/pooling-embedding/index.md` |
| OOM after a healthy period, HBM not released, host RAM, container working set, KV-init allocation failure, or an `expandable_segments` unmap crash at `empty_cache` | `references/runtime-memory/index.md` |
| traceback frames are only `vllm/**` + `site-packages/**` (tokenizer `Already borrowed`, frontend preprocessing) | `references/serving-frontend-faults/index.md` |
| a SoC-conditional path: 310P/300I, A5 vs A2 wheel, kernel package for the running SoC | `references/soc-platform-paths/index.md` |
| spec-decode/MTP symptoms: acceptance collapse, empty first response, async bookkeeping, drafter count copy, a TND `queryT`/`actualSequenceLengthQ` refusal | `references/spec-decode-mtp/index.md` |
| engine/worker never becomes ready (both lanes), model-type/quantization gates, arg validation, metadata/capture phase, a graph setting that never took effect, an offloading-spec `KeyError: '<layer>.self_attn.attn'`, first-weight-group `ValueError` (embedding checkpoint served as generate) | `references/startup-failures/index.md` |
| the symptom is a tool call (`JSON Parse error` in `arguments`, no `tool_calls` while streaming) | `references/tool-call-streaming/index.md` |
| the deliverable is a report about a document (community test, `[Doc]`, tutorial commands) | `references/tutorial-verification/index.md` |
| `AttributeError`/`TypeError` in a downstream patch, an upstream refactor that kept upstream code, or a `KeyError` from upstream's `Attention`-only layer filter under an offloading spec | `references/upstream-downstream-drift/index.md` |

Scripts are re-runnable; `<skill_root>` is the path `skill_view` reports and each
header carries its flags. The 28 scripts — arguments, output and originating issue
— are listed in `references/first-checks/scripts-catalogue.md` (history/timeline,
budget arithmetic, rank table and HCCL link audits, batch-split, KV-pool, pip and
image pairing, checkpoint/tool-schema audits); open it rather than guessing a
name, and the topical pages name the ones they use.

