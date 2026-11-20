---
name: vllm-ascend-triage
description: 'Triage vllm-ascend: deploy, HCCL/PD, hangs, OOM, accuracy'
---

# vLLM-Ascend issue, deployment and support triage

## Problem class this covers

Seven entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter. The last shapes deliver a *decision*
(sizing), a *document review* or a *capability verdict* instead of a root cause,
but the same rules apply — answer first, evidence pinned to the reported version,
gaps stated.

- **Issue report** (usually Chinese or English, template-shaped: environment dump
  + logs + narrative) about something that went wrong on an Ascend NPU host: a
  device-side operator crash, an accuracy/score disagreement, a garbled long-input
  output, a long-run OOM / "VRAM not released" report, a performance claim, a
  logging flag that appears to print nothing, an endpoint that answers `404`
  although the reporter configured it (`/start_profile`), output that stops after
  a handful of tokens while the text itself is sane, an operator that runs and
  returns the wrong numbers, or a bare `python -c "import <pkg>"` that aborts
  inside a library the image shipped.
  When the task is just "explain/answer this issue", the answer is still expected
  to be evidentiary, not a summary of the report.
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
  `global_segment_size` is the cause. Transport identity and rank tables are a
  topology question, not a connector bug, and a pool deployment is a config
  question before either — `references/pd-disaggregation/index.md`.
- **Deployment sizing / stage layout**: nothing has crashed yet and the question
  is a plan — "does this model fit on 2×64 GB", "the GPU docs do a 2-device
  stage deploy and mine does not come up on 2 NPUs", a bundled
  `platforms.<vendor>` yaml with more stages than the reporter has cards, "how do
  I pin stage 1 to one card". Answer with per-card arithmetic plus the
  device-visibility variable, not with a source walk —
  `references/deploy-sizing/index.md`.
- **Community-test / doc-verification task** (`[Community Test]`, a 众测任务书 row
  with 蓝区镜像 / 权重链接 / 卡数 / 交付件 测试报告, or a `[Doc]` report that a
  tutorial step does not work as written): the deliverable is a *report about a
  document*, graded on completeness/usability/correctness and produced without
  the reporter's hardware. The evidence is the pinned tree — which command no
  longer parses, which path is absent from the image, which baseline predates the
  image, whether the stated card count matches the weights —
  `references/tutorial-verification/index.md`.

Recorded uses and how each session ended — the full per-issue index (version,
symptom, track, and what the run got right or wrong) is
`references/evidence-workspace/recorded-cases.md`. Read it when placing a new report
beside a precedent; the load-bearing pattern is in rule 1: the sessions that
delivered said the answer early (#4296 in ten turns, #2586, #9183, #8122), and the
ones that shipped **nothing** (#6784, #12339, #10985, #11739, #11759, #5295,
#4057, #13853, and #6833's first run) all had a usable lead in hand when they kept
going.

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and obey the delivery gates.** A long
   session can be cut off mid-sentence, so the deliverable has to survive
   truncation: what decides the outcome is not how much evidence was gathered but
   whether the message exists. The runs that shipped **no answer at all** (#6784,
   #12339, #10985, #11739, #11759, #5295) each had a usable lead; the run that
   answered completely inside ten turns (#4296) had one pasted traceback and said
   so immediately. Hard stops:
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
   - **The gates cap cumulative output, not only turns.** The recorded failures
     died of volume first: #4057 cut on turn 14 after ~49 K characters of
     reasoning, #6833 after ~45 K over 19 turns, #13853 at ~11 K of a 16 K-token
     budget over 22 turns — each with the ranked cause in hand by turn four or ten
     and no message text; #6833's replay wrote its verdict, then re-derived it from
     source for ten turns and was cut inside its action ladder. A reasoning block
     that is still a plan past a few hundred tokens *is* the answer: emit it.
     Count characters, not turns.
   - **Reasoning is a scratchpad, a few lines per turn:** what the last result
     proved, the single next check, go. It spends the **same output budget** as the
     answer: #15815 and #11759 each spent ~30 K characters of reasoning against
     ~9 K of answer, and #11759 hit the cap with 69 characters of message text. If
     a block starts enumerating the answer's sections, or says "I have enough" /
     "let me write the answer", that block *is* the answer. A "now let me…" line
     delivers neither answer nor evidence.
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
     whole report in one heredoc: #11739 made 40 tool calls and wrote no file, and
     #11759 spent 16 turns with 69 characters of message text, then was cut while
     composing a 5.7 K-character report heredoc after 30 K characters of reasoning,
     so no file was ever written.
   - **`/workspace/scratch/findings.md` is the answer draft, not an extra step:**
     write it at G2 *in place of* the reasoning block, and skip it when the
     session is already long (#11519 composed it late and lost its ranked causes).
   - **A missing or mismatched artefact is itself the finding** — an empty
     environment block, a traceback file absent from the pinned tree, a version
     the snapshot does not carry. State the gap, give the one command that closes
     it on the reporter's host, stop investigating.
   - **Truncation eats the tail, not the head.** Each action directly after its
     cause; the closing scope/open-questions block 3-5 lines, so a capped answer
     still ends in something usable.
   - **Every assistant turn carries message text, not only a tool call.** A turn
     whose whole output is a tool call leaves nothing for a truncation to keep.
     #5295 emitted *zero* message text on all 16 turns: ~46 K characters of
     reasoning (≈12 K of the 16 K output budget), three notes that it "must produce
     the answer NOW", and a cut inside a reasoning block while opening the one file
     that would have refuted its own last hypothesis. Carry one or two sentences of
     the current answer draft in the same message as every tool call — verdict line
     first, then the evidence being checked — and G1/G2 need no special turn. The
     recorded counter-example is #15815, which did ship a complete answer, but only
     on its 15th turn with 31 K of its 50 K output characters spent on reasoning.
   - The reasoning patterns behind the gates:
     `references/evidence-workspace/delivery-discipline.md`.
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
   - Never list a history or attachments directory: doing so on one recorded
     case returned ~3k filenames (~24 KB) for no information, and on the
     v0.23.0rc1 case (#13699) a first-turn listing of `/workspace/history`
     printed 2857 entries (~28 KB). The trap is bundling that listing with the
     `skill_view` call, so it lands before this rule is read — and all four runs
     of the last two batches repeated it anyway: a first-turn `ls /workspace`,
     `cat SOURCES.json` (at #9335/#9183 capped at 100 lines) and an
     `ls /workspace/history` beside `skill_view` returned 13.7 KB of stdout
     twice (#9335, #9183), 43.2 KB when the line cap was dropped (#11739), and
     3.7 KB when the history listing was left out (#6800), then 39 KB and 48 KB in
     the two most recent runs, which both bundled `skill_view` with
     `cat SOURCES.json` and a directory listing (#8984, #12487); the 100-line cap
     stops mid-repo anyway. That is
     context tax rather than answer budget, but it is paid in the same session.
     Load the skill alone, then make the first shell call `ls /workspace` plus
     `bash <skill_root>/scripts/sources-summary.sh`, nothing else. Orientation
     is one level of `/workspace` (`search_files (target='files')`) plus
     `sources-summary.sh`, and a missing `/workspace/attachments` also makes a
     bundled shell listing exit 2, which reads like a failed orientation. Search
     by token (`grep -rl`, `rg -l`) or with `scripts/history-titles.sh`, and read
     `SOURCES.json` through `scripts/sources-summary.sh` instead of dumping the
     whole file. `cat SOURCES.json` is 20-54 KB and the runs that dumped it paid
     it (#12339 34 KB, #14506 54 KB, #11739 43 KB in its first turn), although
     the script prints the same version contract plus the `unavailable` repos in
     ~10 lines. It is the right first shell call; `history-titles.sh` prints the
     matching body line when the regex hit the body rather than the title, so one
     call separates a real family from a lexical accident.
8. **The prior-month corpus is not evidence of absence.** `history/*.md` and
   lexical issue search cover only earlier issues; a current issue is never in
   them. Make one grep for the distinctive token (op name, error string, dataset
   name), read 2-3 hits, then stop: precedent supports a hypothesis, it does not
   replace reading the source at the pinned commit. **Pick the bodies by whether
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
    logs canned advice inside `except` blocks and re-raises, so the sentence names
    a cause the code never checked — `ACLgraph has insufficient available streams
    to capture the configured number of sizes ... Recommended solutions: ...` is
    printed for *any* capture exception (`vllm_ascend/worker/model_runner_v1.py:
    3553-3565` at v0.11.0rc0; the string is gone by v0.20.2rc1, so re-grep per
    version). Before adopting a log's own explanation, `rg -n '<a distinctive
    fragment>' vllm_ascend/` and read the emitting site: if it is an
    `except Exception` that appends `{str(e)}` and re-raises, the appended inner
    error is the evidence and the wrapper is not. Say which text you trust and why.
    Same family as the sync-point trap in procedure step 3: the line where
    an error surfaced is not necessarily the owner of the defect.
12. **A crash inside a native module the image installed is an environment
    question, and it is answerable without that module's source.** If the
    repro is one `python -c "import <pkg>"` line and the process aborts with
    glibc allocator text (`corrupted size vs. prev_size`, `free(): invalid
    pointer`, `double free or corruption`, `Aborted (core dumped)`) and no
    vllm-ascend frame, then: identify which file is actually imported (Ascend
    images carry a second copy under `<toolkit>/python/site-packages/`),
    interpose one allocator (`apt install libjemalloc2` +
    `LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2` on Ubuntu,
    `yum install jemalloc` + `/usr/lib64/libjemalloc.so.2` on openEuler), and
    diff the Dockerfile's *copied* installer against the component's own
    dependency script. The project's `docs/source/tutorials/*.md` startup blocks
    are the list of ingredients the image is expected to provide. Cap the source
    work at ~6 calls and emit the answer; #5609's recorded triage spent both
    session on ABI/RPATH/`glog` archaeology and delivered nothing, having
    already found the jemalloc omission and ranked it low. Route:
    `references/image-dependency-faults/index.md`, then
    `references/image-dependency-faults/native-import-abort.md`.
    The C++-exception form of an in-image abort follows the same rule:
    `terminate called after throwing an instance of 'std::logic_error'` with
    `what(): basic_string::_S_construct null not valid` at engine start is
    `std::string(nullptr)` inside a native library (libstdc++
    `basic_string::_S_construct`), so it is answered by the stock-model
    discriminator, three `python -c` native probes and one `gdb -ex 'break
    std::__throw_logic_error'` backtrace that names the `.so` — never by the
    checkpoint's config and never by grepping the exported CANN/torch_npu/triton
    C++ trees. That is what cost #10943 the whole session (six calls inside
    `cann/`, one line of answer). Route:
    `references/image-dependency-faults/native-string-abort.md`.
    The third form is a **failed symbol resolution**, not an abort:
    `ImportError: …/<pkg>/_ops.cpython-310-aarch64-linux-gnu.so: undefined symbol:
    _ZNK6AsdOps3Ops18GetOperationByNameERKSs` from a prebuilt wheel. Demangle it —
    `Ss` is the pre-C++11 `std::string` (`_GLIBCXX_USE_CXX11_ABI=0`), the cxx11 ABI
    spells `NSt7__cxx1112basic_string…` — and compare with the env dump's
    `ATB_HOME_PATH` / `…/atb/cxx_abi_{0,1}/lib` entry: NNAL ships both roots and the
    wheel needed the other one. Answer with the three `nm` commands and the
    matching-root export; **reporter-confirmed** on #1144 ("the env setting works
    for me"), and the two consumers of that wheel degrade differently (a worker
    `warning_once`, or `NotImplementedError: There is no available ascend quantizer`
    on the quantised path). Route:
    `references/image-dependency-faults/abi-undefined-symbol.md`.
13. **A pasted error block is not necessarily this failure.** Ascend re-prints a
    recorded device/runtime error block inside a fresh traceback and keeps that
    block's original date, so the sentence in the issue title can be a day older
    than the incident. Recorded #9183: the quoted `rtMemcpy ... 107030 the current
    capture mode does not support this operation` lines were all dated the previous
    day, and the in-window first error was an `EZ9999` fftsplus-aivector /
    MTE-DDR exception — the recorded answer built a capture-window mechanism out of
    the stale block and named the wrong owner. One call decides it:
    `bash <skill_root>/scripts/plog-timeline.sh --since <report date> <paste>`;
    families it flags as older are history, and if none is in-window the paste does
    not contain the fault at all — say that and ask for the plog covering the
    failure minute. Same trap as rule 11 one level up: not just the sentence but
    the whole block can be the wrong one.
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

## Classify the evidence

| Signature in the report | Track | Go to |
|---|---|---|
| Startup log: generic `WorkerProc initialization failed due to an exception in a background process`, **no** per-rank worker traceback; optional `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!`; long silence before failure | Lane A: worker died without reporting | `references/startup-failures/silent-worker-exit.md`, then `references/startup-failures/max-model-len-budget.md` |
| Startup log: per-rank `WorkerProc failed to start.` / `WorkerProc hit an exception.` plus a traceback with rank prefix | Lane B: worker raised a Python error | `references/startup-failures/environment-correspondence.md` (diff against the known-good tutorial/CI config) |
| …and the traceback is `OSError: libatb.so: cannot open shared object file` → `Could not load this library: .../torch_npu/lib/libop_plugin_atb.so` → `_register_atb_extensions()` at worker init, with an outer `OSError: [Errno None]` and no `Starting to load model` line | Lane B, environment-owned: NNAL/ATB component missing, or installed and never sourced — CANN's `set_env.sh` does not export the ATB paths | `references/image-dependency-faults/nnal-atb-missing-library.md` |
| …or the per-rank/plain traceback is `ImportError: <name>.so: cannot open shared object file` — e.g. `ascend_transport.so` from the connector's own `from mooncake.engine import TransferEngine` (`kv_p2p/mooncake_connector.py:24`) — in a deployment started by a launcher's `args:`/`command:` block, while the pasted `collect_env.py`/`/etc/profile` dump shows the directory on `LD_LIBRARY_PATH` | Lane B, environment-owned: the environment that decides the load is the launched process's, not the shell's — the export belongs in the launch command; image content only if the file is absent from disk | `references/image-dependency-faults/launch-env-and-library-path.md` |
| The traceback ends in `AttributeError: '<X>Config' object has no attribute '<attr>'` with a frame in `vllm_ascend/distributed/*connector*.py`, reached from `ensure_kv_transfer_initialized` or `register_kv_caches` — a VL/MoE model under PD disaggregation, **or** a flat config whose scalar is derived rather than stored (Qwen2 `head_dim`), where `hf_text_config` was already in use | Lane B: connector reads a config attribute that *nesting* or *absence* leaves undefined — the two are different findings and give different fixes | `references/startup-failures/connector-config-attributes.md` |
| P/D will not connect and the recipe contains a **KV pool**: `AscendStoreConnector` alone or inside a `MultiConnector`, `MOONCAKE_CONFIG_PATH`, a `pool_worker.py` / `pool_scheduler.py` line; the first transfer fails with `Failed to connect to target … 503900` / `E19999` / `HcclCommPrepare … ret: 0x13`, then the decode node answers with garbage while every other metric looks healthy | KV-pool store segment, not the transport: the pool's `global_segment_size` is `aclrtMallocHost`-ed and registered into the same ADXL engine the p2p connector dials, so an oversized segment fails whichever connector dials first (#13439: 40 GB failed, 2 GB worked, HDK 25.2.3 needs < 20 GB) | `references/pd-disaggregation/kv-pool-store-segments.md` |
| P/D will not connect although both engines started and there is **no** KV pool: `HIXL Connect failed \| HIXL Error Code: 103900` (or `503900`), older `LLM_LINK_FAILED` from `llmdatadist_c_mgr_connector`, `wait_for_save failed ... skipping save`, HCCL `RanktableCheck` / `'device_ip' is not set correctly, must be set when multi Server`, `get dev list error:serverId[...]`, a recipe that only works with `hostNetwork: true` / `--net=host` | PD transport topology: two addresses on one node became two HCCL servers (the engine id *is* the rank-table `server_id`); look at the rank table and the plog, not the Python traceback | `references/pd-disaggregation/transport-topology.md` |
| PD deployment that connects and then corrupts on a **hybrid** model (Qwen3.5/Qwen3.6 GDN, DeepSeek-V4, GLM): alternating good/bad answers, empty or wrong first decoded token, works with MTP but not without | PD hybrid-state transfer: connector choice plus the `mamba_cache_mode='align'` gate only the KV-store connectors satisfy | `references/pd-disaggregation/hybrid-state-transfer.md` |
| `ValueError: Non-MoE models do not support external data parallel mode`, or any `ValueError`/`AssertionError` raised from `vllm/engine/arg_utils.py` that names a parallel mode or flag (`data_parallel_*`, `nnodes`, `node_rank`), with no `WorkerProc` line in the log | Startup argument validation in upstream vLLM — launch-command fix, not a defect | `references/startup-failures/arg-validation-errors.md` |
| The report asks whether a plan fits rather than what broke: "can this model run on N cards", a multi-stage orchestrator (`--stage-id`, `--headless`, `platforms.<vendor>` yaml) that works on GPU, stages that must share a card, `CUDA_VISIBLE_DEVICES` in a command aimed at an NPU host | Deployment sizing / stage layout: memory arithmetic plus the device-visibility variable, not a fault | `references/deploy-sizing/index.md` |
| `[Community Test]` / `[Doc]` / 资料验证 / 测试报告, or "the tutorial's command does not work as written" | Tutorial-document verification: grade the page against the pinned tree; the report is the deliverable | `references/tutorial-verification/index.md` |
| `value error, the checkpoint you are trying to load has model type '<mt>' but Transformers does not recognize this architecture`; `Model architectures [...] are not supported`; "can version V deploy model M?" | Model-support / version gate | `references/startup-failures/model-type-not-recognized.md` |
| Python `AttributeError`/`TypeError` in a frame under `vllm_ascend/{ops,patch,models}` where the class subclasses or re-implements an upstream vLLM/transformers class — e.g. `'AscendQwen2DecoderLayer' object has no attribute 'attention_type'`, `unexpected keyword argument`, `cannot import name` | Downstream patch drift | `references/upstream-downstream-drift/index.md` |
| The same `AttributeError` raised **inside an upstream vllm file** on a class vllm-ascend is meant to replace — `'ZeroExpertFusedMoE' object has no attribute 'select_experts'` — or the reporter asks "did it ever reach vllm-ascend's ops?" | Downstream registry drift: the Ascend replacement is keyed by class *name* (`CustomOp.register_oot`), so an upstream refactor that adds a class silently keeps upstream code | `references/upstream-downstream-drift/oot-class-registration-drift.md` |
| An `AttributeError` inside an upstream vllm file whose message contradicts the code — `'<QuantMethod>' object has no attribute '<property>'. Did you mean: '<similarly named method>'?` — reached from an upstream model's `forward` at startup (`profile_run`), where the property exists and calling it directly gives `'NoneType' object has no attribute '<property>'` | Downstream patch **incomplete**, not absent: a factory/property rebound only in its defining modules while the model module cached the old binding at import. Compare the three attributes, do not read more of the model | `references/upstream-downstream-drift/stale-module-binding.md` |
| A bare `TimeoutError` — `RPC call to execute_model timed out.`, or frames in `multiproc_executor.py` (`collective_rpc` / `get_response`) and `shm_broadcast.py` (`acquire_read` / `acquire_write` / `recv`) — reported as "多进程卡间通讯超时", with no rank prefix and no error code | vLLM's own RPC watchdog (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, 300 s), not an HCCL/NIC timeout: one worker stopped answering, and the paste does not contain its fault | `references/hccl-and-hardware/rpc-watchdog-timeouts.md` (one call: `scripts/executor-timeout-check.sh`) |
| `TypeError: 'NoneType' object is not callable` from an upstream module-level name that is `None` on non-CUDA platforms (`fused_experts = None`), with the `forward_oot` frame in `vllm/model_executor/custom_op.py` and `forward_native = forward_cuda` next in the trace; often `--distributed_executor_backend ray` or a `spawn` worker | The downstream op patch was never installed **in that process**: it is imported as a side effect of a driver-only hook (`check_and_update_config`), so fresh-interpreter workers keep upstream code | `references/upstream-downstream-drift/patch-import-side-effect.md` |
| `HcclAllreduce`, `hcclRet -> 4`, `ERR00100`/`ERR01100`, `acl api failed`, `AllReduceOpbasePipeline`, `RuntimeError: SUSPECT REMOTE ERROR` / `507057`, `EZ9999` device dump, hang in `determine_available_memory` / `profile_run` | Lane C: collective / device error (a `profile_run` frame ending in an `npu_*` op on a 2×4 tensor is a compile-phase sync point, not an op bug) | `references/hccl-and-hardware/index.md` |
| Startup log reached graph capture: `Starting to capture ACL graphs for cases: [...]`, `Capturing ACL graphs (...): 0/N` frozen, `ACLgraph sizes capture fail`, `NPUGraph.cpp` "Waiting for pending NCCL work" | Lane C during capture — or whatever the appended inner error says; the `insufficient available streams` advice is canned text | `references/startup-failures/acl-graph-capture.md` |
| Device dump naming an operator: `call aclnnX failed`, `AclNN_Runtime_Error`, fftsplus aivector error, MTE/DDR "out of range"; several ranks reporting the same `pc start` in the same second | Custom-op runtime fault, after the timestamp gate — a shared PC across chips is software address computation, not several bad cards | `references/hccl-and-hardware/error-report-timestamps.md`, then `references/custom-ops-runtime-faults/index.md` |
| `507035` + `ERR00100 PTA call acl api failed` + `EZ9999 … aivec error` + `The DDR address of the MTE instruction is out of range`, one `pc start` repeated on every core, and the deepest frame is `torch_npu.npu_scatter_nd_update_` in `vllm_ascend/attention/sfa_v1.py` with fault kernel `ScatterNdUpdate_…`; the crash follows concurrency (mixed prefill+decode batch) while single requests are fine | DSA indexer-cache write with the wrong index tensor: a host-side slicing bug, not a card — compare the index tensor's row count with the source's before any hardware theory | `references/dsa-sparse-attention/indexer-cache-scatter-fault.md` |
| `rtMemcpy:ErrCode=107030, desc=[the current capture mode does not support this operation]` / `MemCopySync ... kind = 1` on a small `cnt` (72/456 B), often with `deepseek_mtp` + `FULL_DECODE_ONLY` | Capture/replay-time host→device sync copy — a real family (#6002, #4208, #8502) **only** when it is the in-window first error; check its record date first (rule 13) | `references/startup-failures/acl-graph-capture.md` |
| `<method> quantization is currently not supported in npu` inside a pydantic `ValidationError` from `ModelConfig` (e.g. `--quantization awq`), or `Unknown quantization method: <x>`; no `WorkerProc` line | Platform quantization gate before any worker: the NPU platform declares only `ascend` and `compressed-tensors` — re-quantize, not a defect | `references/startup-failures/quantization-method-gate.md` |
| An attention op **runs**, returns a correctly shaped tensor, and the numbers are wrong (uniform average, softmax apparently dropped, error only vs an fp32 reference); often a framework-free `torch_npu` repro, never a dump | Silent operator numerics — below the framework | `references/custom-ops-runtime-faults/attention-op-numeric-errors.md` |
| `AclNN_Parameter_Error(EZ1001)`, `Get regInfo failed`, `binary_info_config.json of socVersion [...] does not support opType [...]` — aclnn cannot resolve the op for the SoC before any launch | Custom-op **build/install** fault: package built for another SoC, stale vendor package, wrong image tag | `references/custom-ops-runtime-faults/op-not-registered-for-soc.md` |
| Output is fluent but stops after tens of tokens ("输出被截断", "只输出几十个 tokens") and a large `max_tokens` appears to fix it; `usage.prompt_tokens` is small; a `pass_config` / SP / FLASHCOMM / quantization flag changed at the same time | Early EOS: `finish_reason` decides budget vs model-issued EOS, then every reader of the toggled key plus the shipped support matrix (accuracy track, not the length budget) | `references/accuracy-benchmarks/early-eos-short-output.md` |
| Spec-decode metrics collapse: `Accepted: 0` with `Drafted: N > 0`, `Per-position acceptance rate: 0.000` at every position, `Mean acceptance length: 1.00`, generation throughput ~0 while requests stay `Running`; often `--async-scheduling` and/or `--enable-prefix-caching` on a hybrid GDN model (Qwen3.5/Qwen3.6, DeepSeek-V4, GLM) | Speculative-decoding bookkeeping (draft and target no longer see the same context), not capacity: the MTP-off ablation is row 1 | `references/spec-decode-mtp/acceptance-collapse.md` |
| First request after start answers `content: ""` with `usage.completion_tokens: 1` and `finish_reason: "stop"`, later requests fine; MTP plus a full-graph decode on a hybrid GDN model | Empty first response: one model-issued blank/stop token — neither the length budget nor reasoning routing — so reproduce by restart before naming a one-shot mechanism, and diff the spec/capture config against the family sibling tutorial (`"enforce_eager": true` inside `--speculative-config`, default capture sizes) | `references/spec-decode-mtp/empty-first-response.md` |
| "score with tool A is N points below tool B / another image / the vendor number", accuracy regression after a change | Accuracy/score discrepancy | `references/accuracy-benchmarks/index.md` |
| Per-layer `MSE loss` / epoch table printed while converting a checkpoint with msmodelslim (W4A4/W8A8), one layer far above its neighbours, or "the converted checkpoint answers garbage" with no engine-side failure | Quantization-conversion phase: converter-owned, before the engine starts | `references/accuracy-benchmarks/quantized-checkpoint-conversion.md` |
| A long prompt (100K+) answered as 乱码 / garbage / repeated / empty text while short prompts are fine, with sparse c8 attention, DCP/PCP, PD disaggregation or KV offload enabled | Generation corruption (accuracy track, different evidence) | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| 乱码 / token soup on a few-thousand-token prompt, process alive, no traceback, `finish_reason: length`, often MoE + EP across nodes or a quantised weight set | Generation corruption, short-prompt profile | `references/accuracy-benchmarks/short-prompt-corruption.md` |
| 乱码 on a **short** prompt of a hybrid linear-attention model (Qwen3.5/Qwen3.6 GDN and siblings), where the "正常" command differs from the failing one by a scheduling knob — `--block-size` (e.g. 512 → 2048) and/or `--enforce-eager` — and nothing else | Hybrid KV-cache geometry: the cache block is never clamped to 128 on these models and is only raised by the mamba/attention page-size constraint, so a bigger block re-slices block tables, slot mapping and the mamba state block at once | `references/accuracy-benchmarks/hybrid-block-size-corruption.md` |
| A **VL image/video** request answered as a long run of one token (`!!!!`, 感叹号, `，，，`), usually with "特别耗时 / very slow"; text-only prompts may answer fine | Vision-tower degeneration (accuracy track), not sampling | `references/accuracy-benchmarks/vl-vision-tower-degeneracy.md` |
| The server served traffic then died on memory: "运行一段时间后显存不释放最后崩溃", `NPU out of memory` / `NPUWorkspaceAllocator` / `Memory_Allocation_Failure(EL0004)`, `npu-smi` pinned at 100 %, a P node holding KV the D node never takes | Runtime memory after a healthy period: capacity vs growth vs a config key that never engaged (contrast the Lane C row, where the OOM happens during `profile_run` / `determine_available_memory` before the server is ready) | `references/runtime-memory/index.md` |
| "The machine crashed" / the host became unusable / the reporter pastes `top` with every python3 rank at tens of GiB RES on one host, while the serve log has no device, collective or Python error | Host RAM (and host CPU), not the card: the ranks' resident footprints share one RAM budget, and `npu-smi` cannot see it | `references/runtime-memory/host-ram-and-per-rank-rss.md` + `scripts/host-ram-check.sh` |
| Startup/KV-init failure whose deepest frame is `_convert_torch_format` / `torch_npu.npu_format_cast` with `Memory_Allocation_Failure(EL0004)`, `alloc device memory failed … 207001`, `507013`, SDMA/SMMU text | Allocation failure at KV-cache init — the cast is the allocation site, not an operator bug | `references/runtime-memory/kv-init-and-format-cast-alloc.md` |
| `python -c "import <pkg>"` (or a first real use) raises `ImportError:` naming that package's own extension — `<pkg>/_ops.cpython-310-…so: undefined symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs` — and the ENV dump selects `atb/cxx_abi_1` (or `cxx_abi_0`) | C++ `std::string`-ABI split between a prebuilt wheel and the selected NNAL/ATB root, pre-engine and not vllm-ascend code: `Ss` in the mangled name = old ABI, `NSt7__cxx11` = new | `references/image-dependency-faults/abi-undefined-symbol.md` (one call: `scripts/abi-symbol-check.sh <paste>`; the fix is reporter-confirmed on #1144, but which ATB file defines the symbol is still the reporter's `nm` step) |
| `python -c "import <pkg>"` or the first use of an image-installed native module aborts the interpreter: `corrupted size vs. prev_size`, `free(): invalid pointer`, `double free or corruption`, `munmap_chunk(): invalid pointer`, `Aborted (core dumped)`, no Python frame, no NPU text | Image / bundled native module (pre-engine, not vllm-ascend code) | `references/image-dependency-faults/index.md` |
| Engine init ends in a C++ exception with **no** Python frame: `terminate called after throwing an instance of 'std::logic_error'` / `what(): basic_string::_S_construct null not valid`, then `RuntimeError: Engine core initialization failed … Failed core proc(s): {}`; the last application line is plugin activation or `Initializing a V1 LLM engine` | Native null-string abort in a bundled `.so` during core/worker creation — an image/host question, not the checkpoint's config, processor or architecture (the same two lines appear for two unrelated models) | `references/image-dependency-faults/native-string-abort.md` |
| `POST /start_profile` (or `/stop_profile`) returns `404 Not Found` although the server was started with `--profiler-config '{...}'`; or the same route answers 500 naming `Profiling is not enabled` | Observability/route registration: the endpoint is attached by upstream `serve/profile/api_router.py` `attach_router` only when the serving process's `args.profiler_config.profiler` is set — not a model/device question, and a down-stream worker miss is a 500, never a 404 | `references/observability-and-logs/online-profiling-endpoints.md` (one `openapi.json` request) |
| `RuntimeError: Already borrowed` raised inside a HuggingFace fast tokenizer (`set_truncation_and_padding` → `self._tokenizer.no_truncation()`), typically under `_preprocess_chat` while a tool parser's `__init__` encodes its special tokens, wrapped by `serving.py … Error in preprocessing prompt inputs`; more generally any traceback whose frames are only `vllm/**` + `site-packages/**` with no `vllm_ascend` frame and no device text | Upstream serving-frontend / Python-dependency fault, not the NPU stack: one tokenizer object driven from the event loop and from a worker thread, and vLLM's own retry for this exception covers only the multimodal processor. Mechanism source-verified at vllm v0.16.0; the trigger (client concurrency) is unverified and the serial rerun settles it | `references/serving-frontend-faults/index.md`, then `tokenizer-already-borrowed.md` |
| Report is about log text rather than a crash: `--enable-log-requests` / `--enable-log-outputs` set but no prompt or generation printed, `VLLM_LOGGING_LEVEL=DEBUG` seemingly ignored, `--max-log-len`, a stage of the startup log missing, or a proxy that swallows output | Observability/flag question — **upstream frontend config, not a vllm-ascend defect**; answer with a gate checklist and two greps, cap it at ~4 tool calls | `references/observability-and-logs/index.md`, then `request-output-logging-flags.md` |
| A **tool call** is the symptom, not the prose: `JSON Parse error: Unterminated string` / `Expecting ',' delimiter` / `Unterminated string starting at: line 1 column N` while the client assembles `tool_calls[0].function.arguments`, arguments reported truncated / 格式错误 / 重复输出, or `stream: true` returns only CoT and no `tool_calls`; often with `--speculative-config '{"num_speculative_tokens": 3, ...}'` where 1 正常, or with `--async-scheduling` | Tool-call output integrity: upstream vLLM's delta assembly vs the vllm-ascend sampler vs the model stopping early — decided by concatenating the argument fragments, then by one `return_token_ids: true` request, not by reading a parser or a kernel | `references/tool-call-streaming/index.md`, then `streamed-arguments-audit.md` (one call: `scripts/sse-toolcall-audit.sh <raw-sse>`) |
| A constant in the code equals the number in the report — `num_speculative_tokens: 3` fails while `1` (and `2`) work — and the wrong output is exact-form: tool-call JSON, ids, code | Named code branch, not an investigation: `using_block_verify = max_spec_len >= 3` is the only spec-decode branch keyed on that value, so state it with its two ablations (`n=2`, `temperature: 0`) before its kernel is read | `references/tool-call-streaming/mtp-block-verify-acceptance.md` |
| Report pastes only a response body: the CoT/thinking text sits in `content`, `message.reasoning` / `reasoning_content` never appears, `completion_tokens_details.reasoning_tokens` is `0`, thinking language mixed | Output routing — **upstream OpenAI frontend, not a vllm-ascend defect**: `reasoning_parser` defaults to `""`, so `--reasoning-parser <name>` (e.g. `qwen3`) is the whole gate; invariant to image tag and MTP | `references/accuracy-benchmarks/reasoning-output-routing.md` |
| Host is **310P** (Atlas 300I Duo, `Ascend310P3`, image `-310P`), or the traceback sits on an `is_310p()` branch (NZ/`FRACTAL_NZ` format casts, no Ascend custom ops, `patch_mamba_config_310`) | SoC-conditional platform path: experimental per the pinned docs, NZ casts are real conversions, fp16/eager are mandatory | `references/soc-platform-paths/atlas-300i-310p.md` |
| `EngineCore failed to start` whose deepest frame is `worker.py` → `check_ascend_device_type()` with **no** `Starting to load model` line: `AssertionError: Current device type: AscendDeviceType.A5 does not match the installed version's device type: AscendDeviceType.A2, please check your installation package.` (or `RuntimeError: Can not support soc_version: <n>`), typically on a 950PR/A5 host that installed a mirror wheel | SoC/package correspondence: the device type is **baked into the package at build time** (`_build_info.__device_type__` from `SOC_VERSION`), the release pipeline ships a2/a3/310p variants only while `Dockerfile.a5` exists in-tree, so an A5 host needs a source build on the node — and the guard must not be patched | `references/soc-platform-paths/device-type-build-vs-runtime.md` (two one-liners decide it) |
| The report asks how to enable a feature rather than what broke: "如何正确开启 PCP/DCP", "F alone and G alone each报错，能否给个能跑通的示例", no log pasted, or two errors described in one sentence | Feature-enable / capability question: the validation verdict (docs statement at the pinned ref, whose hardware the measured table used, `tests/e2e/**` coverage, precedent versions) before any constraint list | `references/feature-enable-recipes/index.md`, then `context-parallel-pcp-dcp.md` |
| `Communication_Error_Bind_IP_Port(EI0019)` / `Failed to enable listening for the host network adapter socket` / `port <n> have already been bound` where n is above 65535, often in the issue title | HCCL control-plane port band, not an OS socket conflict — and only if the record's date is inside the window (rule 13) | `references/hccl-and-hardware/control-plane-port-binding.md` |
| Output degenerates (`</think>` leaking into `content`, repetition collapse, 乱码) at long-ish context **and** concurrency on a **documented** model with PD / spec-decode / prefix-cache flags, with **both** `vllm serve` commands pasted and no ablation | Config-diff track: the tree's own `docs/source/tutorials/models/<model>.md` ships the validated P and D commands and a `Recommended Configurations` table, so the first answer is that diff plus the ablation matrix — never a walk through the KV connector; #13002's five deviations are ranked ablation rows, not established causes | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| "服务崩溃" / 崩溃 under concurrency or high load with **no** error text, no traceback and no plog pasted, while single-request serving is correct (often DeepSeek-V3.2 + MTP + DP/EP) | Report-gap track: nothing in the paste can confirm a mechanism, so the answer is the config diff against the project's own validated command at the pinned commit plus 1-2 single-variable A/Bs | `references/dsa-sparse-attention/concurrency-crash-with-no-log.md`; `references/runtime-memory/index.md` for the OOM row |
| "多并发时请求排队那么久" / "生成吞吐只有个位数" / `Prefix cache hit rate: 0.0%`, with a `loggers.py` metrics block, a healthy engine and no error text | Throughput/queueing on a healthy engine: TTFT is prefill work ÷ prefill throughput, the interference is `gen tok/s ÷ Running`, and the cache line is a capability switch before it is a tuning knob | `references/performance-throughput/index.md` (both pages in one turn when the paste carries both complaints — the arithmetic page first) |
| A 19-token to few-hundred-token prompt on a hybrid GDN model (Qwen3-Next) answered as a digit run or a repeated phrase while ordinary chat is fine, `finish_reason: length`, no traceback | Hybrid linear-attention numeric path (accuracy track): the prefill GDN kernel is a downstream fp32 re-implementation while decode keeps a patched Triton kernel | `references/accuracy-benchmarks/qwen3-next-gdn-numeric-path.md` |
| The reporter wants `--enable-expert-parallel` **and** `--data-parallel-size` > 1 together ("同时开启 EP + DP", "去掉 EP 或 DP 设成 1 就不报错"), or engine init dies at `vllm_ascend/ops/fused_moe/prepare_finalize.py` `dist.all_gather` behind `No available shared memory broadcast block found in 60 second(s)` on 910B | MoE communication-method selection, not a flag bug: A2 picks `MC2` only when `num_tokens <= mc2_tokens_capacity` **and** `world_size_across_dp/pp >= 16`, so DP2×TP8 = 16 is the boundary while DP=1 and EP-off fall to `ALLGATHER` by arithmetic; `PrepareAndFinalizeWithMC2` inherits the All2All `finalize`, so that all_gather **is** the MC2 path | `references/feature-enable-recipes/moe-expert-parallel-with-dp.md` |
| Compiler or missing-header error while installing/building, or any report with no matching track | General sequence below | skip the topical routes |

## Procedure

1. One level of `/workspace` (`search_files (target='files')`) — that level only,
   never the history or attachments directory (rule 7) — then run
   `bash <skill_root>/scripts/sources-summary.sh` for the `SOURCES.json` version
   contract (`skill_root` is the path `skill_view` reports); note which
   repos/refs are present and which are partial exports. The
   layout differs per case (flat `/workspace/history/<issue>.md` vs
   `/workspace/history/<issue>/` + `/workspace/repos/<issue>/`); adapt after
   looking, do not assume.
2. Read the pasted log and place it in a track/lane. Note what is *missing* (no
   worker prefixes, no `Starting to load model`, no version block) — reporters
   paste filtered logs, and missing lines are a data gap.
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
     When the deepest frame is an `npu_*` op on a 2×4/2×64 tensor inside
     `torch.compile`, it is a fusion-pattern sample kernel and therefore a sync
     point by construction: go to the plog and the per-rank `EZ9999`/`chipId` line,
     not to the operator's shape code.
   - Any count or list derived in the log (capture sizes, block counts, chunk
     sizes) that does not match the upstream default: vllm-ascend is probably
     rewriting it. `rg -n '<config field>' vllm_ascend/` (platform.py,
     ascend_config.py, utils.py), read the log's own `Calculated ...` / `N → M`
     lines, and reproduce the number with
     `bash <skill_root>/scripts/aclgraph-size-budget.sh` instead of reading the
     formula by eye.
   - Image / bundled-native-module abort (no vllm-ascend frame, glibc allocator
     text): which copy is imported, then allocator interposition, then the
     installer-vs-upstream dependency diff — steps 1-3 of
     `references/image-dependency-faults/native-import-abort.md`. Skip the
     module's C++.
   - Worker init dying on `libatb.so` / `libop_plugin_atb.so`: list
     `<ASCEND_HOME>/nnal/atb/set_env.sh`, then run the worker's own
     `_register_atb_extensions()` probe (step 4 of
     `references/image-dependency-faults/nnal-atb-missing-library.md`). Two calls
     split "not installed" from "installed, not sourced", and the NNAL
     `set_env.sh` is a *separate* file each install root must source itself. Do
     not read `torch_npu`'s loader or hand-copy the `.so`.
   - `ImportError: <name>.so: cannot open shared object file` raised while the
     connector module itself imports (`ascend_transport.so` from
     `from mooncake.engine import TransferEngine`), in a deployment a launcher
     started (`args:`/`command:`, compose `entrypoint`, an operator CR): ask for
     the pod spec of **both** roles and `cat /proc/1/environ` inside the running
     container before any image theory. The ENV dump describes a shell, not the
     launched process; the export goes in the launch command (steps 1-2 of
     `references/image-dependency-faults/launch-env-and-library-path.md`).
   - Report pasting only a response body with thinking in `content` and
     `reasoning_tokens: 0`: the serve command's `--reasoning-parser` is the
     answer before any model/device theory, and the chat field is `reasoning`,
     not `reasoning_content` (checked at vllm v0.19.1 and v0.22.1 — name the
     field the reporter's version emits)
     (`references/accuracy-benchmarks/reasoning-output-routing.md`).
   - Native null-string abort at engine init (`std::logic_error` /
     `basic_string::_S_construct null not valid`, no Python frame): stock-model
     run on the same host/image, then the three `python -c` probes, then
     `gdb --batch -ex 'break std::__throw_logic_error' -ex run -ex bt`. Steps 1-3
     of `references/image-dependency-faults/native-string-abort.md` are the whole
     answer; the checkpoint's own config was already proven innocent.
   - `ImportError: …/_ops*.so: undefined symbol: <mangled>` from a prebuilt wheel:
     run `bash <skill_root>/scripts/abi-symbol-check.sh <paste>` first — it demangles,
     says old (`Ss`) or new (`NSt7__cxx11`) `std::string` ABI, and prints the three
     `nm` commands plus the matching `cxx_abi_0`/`cxx_abi_1` export. Reporter-confirmed
     on #1144 ("the env setting works for me"), so this is a three-command lane, not
     a rebuild (`references/image-dependency-faults/abi-undefined-symbol.md`).
   - Custom op that aclnn refuses to *resolve* (`EZ1001`, `Get regInfo failed`,
     `does not support opType`, `binary_info_config.json of socVersion`): registry
     and install, before launch. Do not open `csrc/<op>/`: check the tree's
     per-SoC build branch and `def.cpp` `AddConfig` once to rule the source out,
     then `SOC_VERSION`, `_cann_ops_custom`, `ASCEND_CUSTOM_OPP_PATH` and the
     image tag (`references/custom-ops-runtime-faults/op-not-registered-for-soc.md`).
   - Python attribute error whose frame is an **upstream** vllm file (or the
     reporter asks whether vllm-ascend's ops were ever reached): the Ascend
     replacement is name-keyed, so compare the class name against
     `REGISTERED_ASCEND_OPS` / `CustomOp.op_registry_oot` before theorising about
     upstream (`references/upstream-downstream-drift/oot-class-registration-drift.md`).
     If the class *is* registered, or the missing name is a factory/property the
     patch rebinds, check for a pre-patch alias in the consumer module —
     `consumer.<name> is defining_mod.<name>` settles it in one line
     (`references/upstream-downstream-drift/stale-module-binding.md`, #15815).
   - PD / EPD transport complaint: **first ask whether a KV pool is configured**
     (`AscendStoreConnector` in a `MultiConnector`, `MOONCAKE_CONFIG_PATH`). If it
     is, the pool decides before the transport: run
     `bash <skill_root>/scripts/kvpool-segment-check.sh <mooncake.json|log>` — the
     worker log's `register mem … mem type:host, length:N` line carries the
     per-rank `global_segment_size`, and ≥ 20 GB on HDK 25.2.3 is the recorded
     cause of `503900` + 乱码 (#13439), fixed at 2 GB
     (`references/pd-disaggregation/kv-pool-store-segments.md`). Only without a
     pool, or once the segment is small, read the rank table with
     `bash <skill_root>/scripts/ranktable-check.sh <table-or-log>` — a declared
     `server_count > 1` (or more than one distinct `server_id`) decides the rest
     (multi-server view, `device_ip` + RoCE required). Then the plog grep set in
     `references/pd-disaggregation/transport-topology.md`.
   - Report is "the flag did nothing" (logging/verbosity, output printing, a
     missing log stage): name the owner first — upstream vLLM OpenAI frontend,
     not vllm-ascend — then walk the gate checklist in
     `references/observability-and-logs/index.md`: both flags on the serving
     command, `--enable-log-outputs` requires `--enable-log-requests`,
     INFO-vs-DEBUG split, `--max-log-len`, a log-config/`VLLM_CONFIGURE_LOGGING`
     override, and an endpoint that cannot log outputs at all. Grep the two
     warning strings, then answer; do not read `vllm/entrypoints/**` for an
     Ascend-specific cause.
   - An endpoint that is missing rather than wrong (`POST /start_profile` →
     `404`, or a 500 `Profiling is not enabled`): a route-registration question
     in the serving process. Name `attach_router` (`path:line`) and its condition,
     then ask the live app: `curl -s <host>:<port>/openapi.json` filtered for
     `profile`; check the startup log for the gate's `warning_once` line, and only
     then wonder where the flag went (`references/observability-and-logs/online-profiling-endpoints.md`).
     Model class, quantization and NPU health are irrelevant to a 404 — say so.
   - Output that is fluent but stops after tens of tokens ("被截断") with a feature
     flag toggled nearby: `finish_reason` first (`length` = budget, `stop` = early
     EOS), then `rg -n "<key>"` every reader of that key and read the shipped
     support matrix before any mechanism
     (`references/accuracy-benchmarks/early-eos-short-output.md`).
   - A spec-decode symptom (acceptance collapse, a stall while drafts keep being
     produced, "works if I remove MTP"): order the ablations MTP off →
     `--async-scheduling` off → `--no-enable-prefix-caching` → `num_speculative_tokens 1`
     → concurrency/`--max-model-len`, and do not lead with the KV arithmetic:
     acceptance dropping to exactly 0.000 everywhere while drafts continue is a
     draft/target context question, and the recorded reporter confirmed the owner by
     removing MTP (`references/spec-decode-mtp/acceptance-collapse.md`).
   - A **tool-call** report (client JSON error, "参数被截断", 格式错误, 重复输出, or
     only CoT with `stream: true`): locate the pasted `data:` chunks or ask for one
     `curl -N` capture, then run
     `bash <skill_root>/scripts/sse-toolcall-audit.sh <file|->` **before** any parser
     or sampler theory. Its outcome names the layer: assembly parses = client,
     tail missing = the stream ended early (split with `return_token_ids: true`),
     unbalanced middle = generation. Say which, with the assembled tail, in the
     first answer (`references/tool-call-streaming/streamed-arguments-audit.md`).
   - If that report's failing `num_speculative_tokens` is >= 3 while 1 or 2 works,
     name `vllm_ascend/sample/rejection_sampler.py:114 using_block_verify =
     max_spec_len >= 3` (the line is pin-dependent: `:116` at v0.18.0) in the same
     answer with the `n=2` and `temperature: 0`
     ablations; walk the kernel afterwards, never instead
     (`references/tool-call-streaming/mtp-block-verify-acceptance.md`).
   - A **first** response with `content: ""`, `completion_tokens: 1` and
     `finish_reason: "stop"`: that is one model-issued blank/stop token. Ask for the
     emitted token id (`return_token_ids: true`), a restart with the identical body
     twice, and a warm-up request row before attributing it to one-shot server state;
     offer `min_tokens: 1` for the case where the id turns out to be EOS. Diff the
     launcher against the family sibling tutorial first (`docs/source/tutorials/models/`),
     and do not re-derive the scheduler padded-rows/null-block theory, it is already
     refuted. The MTP proposer's *padded drafter batch* is a different story and still
     live: `disable_padded_drafter_batch: false` (the default) is what keeps the draft
     model replaying aclgraph, so `"enforce_eager": true` inside `--speculative-config`
     and `"disable_padded_drafter_batch": true` are the two draft-side-only rows
     (`references/spec-decode-mtp/empty-first-response.md`).
   - 310P / Atlas 300I report: take the SoC from
     `_build_info.__soc_version__` rather than the image tag, quote the
     documented status (experimental, stable = v0.10.0rc1 at v0.11.0rc3,
     `faqs.md:18`, `single_node_300i.md:3-7`) as the first sentence, and then
     apply the 300I environment (`--enforce-eager`, `--dtype float16`,
     `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`) and a smaller
     `--max-model-len` before blaming the platform
     (`references/soc-platform-paths/atlas-300i-310p.md`).
   - Device allocation failure before the engine is ready
     (`Memory_Allocation_Failure(EL0004)`, `runtime result = 207001`, `507013`
     plus SDMA/SMMU text) at a KV-cache or weight `npu_format_cast`: read
     timestamps, treat the cast frame as the allocation site, and shrink the pool
     (`--max-model-len`, then `--gpu-memory-utilization`) before any
     driver/hardware theory
     (`references/runtime-memory/kv-init-and-format-cast-alloc.md`).
   - Runtime memory after a healthy period: take the two-sample split (capacity
     vs growth) and `grep -m1 'Set PYTORCH_NPU_ALLOC_CONF' <serve log>` first, and
     check `additional_config` key liveness (`rg -n '<key>' vllm_ascend/`) before
     any leak theory (`references/runtime-memory/hbm-growth-and-oom.md`).
   - "The machine crashed" / the host dropped / a `top` paste whose python3 ranks
     each hold tens of GiB RES: run
     `bash <skill_root>/scripts/host-ram-check.sh <paste|->` and report the ratio
     before any NPU lane, quoting RES and never the `8326g` VIRT
     (`references/runtime-memory/host-ram-and-per-rank-rss.md`).
   - Any pasted plog / RUNTIME excerpt that carries more than one error family: run
     `bash <skill_root>/scripts/plog-timeline.sh --since <report date> <log>` before
     naming a cause. Blocks it flags as older are re-printed records (rule 13); the
     in-window first error is the one to classify
     (`references/hccl-and-hardware/error-report-timestamps.md`).
   - A device fault whose deepest frame is a `torch_npu` scatter/index write in
     `vllm_ascend/attention/sfa_v1.py` (`507035`, `ERR00100`, `EZ9999 … aivec error`,
     `The DDR address of the MTE instruction is out of range`, fault kernel
     `ScatterNdUpdate_…`, one `pc start` on every core): compare the index tensor's row
     count with the source tensor's before anything else. In #4296
     `attn_metadata.slot_mapping` carried the whole batch while `indexer_select` was
     called once per decode/prefill half, so a mixed batch read past `k` — and the
     deployment already ran `--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1`, which only
     made the attribution exact. The decisive check is an A/B on request concurrency
     (`--max-num-seqs 1` versus the failing value), not a plog
     (`references/dsa-sparse-attention/indexer-cache-scatter-fault.md`).
   - Degenerate output (`</think>` in `content`, repetition collapse, 乱码) at long
     context **and** concurrency on a model the tree documents, with both `vllm serve`
     commands pasted: open `docs/source/tutorials/models/<model>.md` at the pinned
     commit and diff the reporter's P and D commands against the validated ones plus
     the `Recommended Configurations` tables **before** reading any connector — that
     diff, with the ablation matrix, is the first answer
     (`references/accuracy-benchmarks/garbled-long-context-output.md`). Recorded
     #13002 spent 21 turns and its whole output budget inside a 2600-line
     `mooncake_connector.py` and shipped nothing, though the tutorial's validated
     commands deviated on five flags and were reachable in two calls.
   - A crash under concurrency with **no** error text pasted: do not hunt the mechanism.
     The answer is the gap, the config diff against the tree's own validated command
     (`tests/e2e/**`, `docs/source/tutorials/<model>.md`) and ranked candidates with one
     check each, emitted at the normal gate
     (`references/dsa-sparse-attention/concurrency-crash-with-no-log.md`).
   - A "communication timeout" whose only text is a bare `TimeoutError` with
     `multiproc_executor.py` / `shm_broadcast.py` frames: run
     `bash <skill_root>/scripts/executor-timeout-check.sh <paste|->` before any
     HCCL reading. It is the framework's own RPC watchdog (300 s) for a worker that
     stopped answering, so the deliverable is the lane classification plus the list
     of per-rank artefacts — which rank, which op and OOM-vs-hang are *not* in that
     paste (`references/hccl-and-hardware/rpc-watchdog-timeouts.md`, #1865).
   - `TypeError: 'NoneType' object is not callable` where the `None` is an upstream
     module-level name (`fused_experts = None` on a non-CUDA platform): read the
     **file** of the `forward_oot` frame. `vllm/model_executor/custom_op.py` there
     means the Ascend replacement was never imported in that process (Ray/`spawn`
     workers; the hook runs only where `VllmConfig` is built) — two greps decide it,
     do not theorise about the model or the checkpoint
     (`references/upstream-downstream-drift/patch-import-side-effect.md`, #253).
   - Quantization method rejected before any worker (`<method> quantization is
     currently not supported in npu`, `--quantization awq`): answer from
     `references/startup-failures/quantization-method-gate.md` — three calls, and
     the flag is not the only trigger, since the checkpoint's own `quant_method`
     reaches the same gate.
   - Deployment sizing / stage layout (does it fit on N cards, why a stage
     layout fails): run `bash <skill_root>/scripts/card-budget.sh` with the
     checkpoint's parameter count, the TP and the per-stage
     `gpu_memory_utilization`/device lists, and report the two numbers it prints
     (weights per card vs card budget; per-device gmu sum). Check the device
     variable first: `ASCEND_RT_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`
     (`vllm_ascend/platform.py:101` at v0.19.1rc1;
     `references/deploy-sizing/card-budget-and-device-visibility.md`). A sibling
     project's yaml/CLI absent from `SOURCES.json` (vllm-omni) is a stated gap,
     not something to infer.
   - A parallel-topology question about EP **with** DP, or an engine-init crash whose
     deepest frame is the MoE `prepare/finalize` `dist.all_gather` on a 910B DP×TP
     deployment: read `_select_moe_comm_method`
     (`worker/model_runner_v1.py:1373-1432`) and `set_mc2_tokens_capacity`
     (`ascend_forward_context.py:169-183`) first, then answer with that SoC's row,
     the reporter's `world_size_across_dp` arithmetic and the two ablations
     (`references/feature-enable-recipes/moe-expert-parallel-with-dp.md`); say that
     MC2 shares the All2All `finalize`, so the pasted line is not proof MC2 was
     skipped.
   - Feature-enable / capability question ("how do I correctly enable F on H", no
     log): answer in this order — the validation verdict at the pinned ref (docs
     sentence, the hardware the docs measured, `rg -l
     "prefill_context_parallel_size|prefill-context-parallel-size" tests/` for e2e
     coverage, precedent issues with their versions), then each enforced constraint
     with the error text it raises, then one-axis-at-a-time staging. Grep the flag in
     `vllm_ascend/` for hard errors (`AssertionError`, `NotImplementedError`) versus
     silent overrides; do not read the kernel/code path beyond that
     (`references/feature-enable-recipes/index.md`).
   - Community-test / doc-verification task: run the nine checks of
     `references/tutorial-verification/tutorial-command-audit.md` in order
     (flag existence at the pinned vllm, bench argument names, image WORKDIR vs
     the doc's relative paths, relied-on defaults, card-count arithmetic vs
     sibling docs and the CI config, baseline release, one precedent pass,
     package version the doc installs vs the image's own pins, support matrix and
     release notes) and write the report file as sections become
     true (`references/tutorial-verification/community-test-report.md`). Stop
     auditing after 3-5 tiered defects; never present a doc baseline as a
     measurement.
   - Accuracy/corruption: establish whether both numbers came from the same
     server, weights and serve flags before choosing a culprit
     (`references/accuracy-benchmarks/index.md`); for a garbled prompt, get the
     ablation matrix (`references/accuracy-benchmarks/garbled-long-context-output.md`).
   - A **VL image/video** request that degenerates into one repeated token
     (`!!!!`, 感叹号) while text-only prompts answer: first row of the ablation is
     `USE_OPTIMIZED_MODEL=0` on the same request, then text-only, TP=1 and an
     official checkpoint (`references/accuracy-benchmarks/vl-vision-tower-degeneracy.md`).
   - An operator that **runs** and returns wrong numbers with a framework-free
     `torch_npu` repro (no dump): find the one caller holding that kwargs
     combination before reading the kernel; `cann/ops-nn` is usually absent, so
     the ownership split is the answer
     (`references/custom-ops-runtime-faults/attention-op-numeric-errors.md`).
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
   (rule 8); `bash <skill_root>/scripts/history-titles.sh '<regex>'` gives the
   titles of all hits in one call without opening the bodies.
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
  it happens to be, and #7580 was cut inside its evidence table — after the verdict, before the ablation plan — so the reporter
  received a diagnosis and no next step. Verdict plus one action first; table and
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

- `references/evidence-workspace/index.md` — read first in an unfamiliar case:
  case layout, `SOURCES.json` fields and limits, history search, OCR/attachment
  caveat, what the snapshot does and does not contain.
  `references/evidence-workspace/delivery-discipline.md` holds the G1/G2/G3
  delivery gates of rule 1 and the reasoning patterns that break them.
  `references/evidence-workspace/recorded-cases.md` is the per-issue index behind
  rule 1 (version, symptom, track, and how each recorded session ended).
- `scripts/history-titles.sh '<regex>' [root] [limit]` — precedent index: one
  `<file> :: <title>` line per matching history issue, so a whole issue family is
  triaged without reading bodies. Run it as
  `bash <skill_root>/scripts/history-titles.sh '<regex>'`, where `skill_root` is
  the path `skill_view` reports.
- `scripts/sources-summary.sh [SOURCES.json ...]` — the version contract, one
  line per repo with coverage and `runtime_verified` flags, plus the
  `unavailable` repos, so "is `cann/hixl` in this snapshot?" is answered without
  dumping 20-54 KB of JSON.
- `scripts/device-type-check.sh <paste|serve-log|-> | --soc-version N [--stamp X]` —
  decodes the `check_ascend_device_type()` failure: it parses the two-sided
  assertion text and says which side is the **card** and which is the **wheel**
  (a class of reading error that flips the diagnosis), classifies a raw
  `soc_version` against the `220-225`/`250-255`/`200-205`/`260` map, flags the
  unmapped-code `RuntimeError` branch, and labels the co-occurring
  `liboptiling.so` line as non-fatal. Verified on the #12141 paste
  (`A5` card, `A2` wheel).
- `scripts/kvpool-segment-check.sh <mooncake.json|serve-log|-> [...]` — the KV-pool
  store segment gate: prints `global_segment_size` / `local_buffer_size` per rank
  from the store config (`MOONCAKE_CONFIG_PATH` json, `MOONCAKE_GLOBAL_SEGMENT_SIZE`
  env) or, when only a log is available, from the worker's
  `TransferExecutor register mem addr:…, length:N, mem type:host` lines, plus the
  `npu-smi` HDK version and the pool/channel markers, and gives the verdict bands
  (≥ 20 GiB → #13439's fault, fix at `"2GB"`; 1 GiB multiple required). Verified on
  the #13439 D-node paste: it recovers 40.00 GiB from
  `length:42949672960` and flags the ceiling.
- `scripts/ranktable-check.sh <rank_table.json|log.txt> [...] | -` — parses a
  pasted (or plog-embedded) HCCL rank table and prints the effective
  `server_count`, per-server `server_id` / `device_id` / `device_ip`, duplicate or
  non-contiguous ids, and the verdict that follows: a declared `server_count > 1`
  or more than one distinct `server_id` means a multi-server view where
  `device_ip` is mandatory and the data path is the NIC; `server_count == 1` with
  one `server_id` means HCCS between local cards is selectable. Use it before
  reading any KV-connector code (verified against the #14506 and #2399 rank
  tables).
- `scripts/aclgraph-size-budget.sh --layers N [--tp --dp --pcp --dcp --ep --moe
  --aiv --max-num-seqs --sizes]` — reproduces `update_aclgraph_sizes`: the
  maximum captureable batch sizes for a model/parallel configuration and the
  sampled subset, so a startup log's size list can be checked arithmetically
  (verified: 94 layers, TP=16 gives 6 sizes = `[512,408,304,192,88,1]`, #3389).
- `scripts/plog-timeline.sh [--since YYYY-MM-DD] <log|-> [more logs]` — per-error-family
  timeline of a pasted plog / serve log: the span of the embedded device records vs
  the log-line dates, which families are re-printed from an older date, and the
  first error inside the incident window. Verified on #9183: without `--since` it
  uses the newest date in the paste; with `--since 2026-05-13` it flags the
  `107030` block as a 2026-05-12 replay and names the `EZ9999` aivector line as the
  first in-window error.
- `scripts/executor-timeout-check.sh <log|->` — one-call split between vLLM's own
  RPC/ringbuffer watchdog and a real collective timeout: it reports the `RPC call
  to <method> timed out.` text, which `shm_broadcast` frame is present (read side
  = caller waiting for a reply, write side = readers not consuming), any
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` bound in the paste, ringbuffer starvation
  lines and collective-side `EI…`/`Execute timeout` text, then the verdict and the
  next artefacts. Tolerates screenshot-OCR garble (`sh]n broadcast`), so it works
  on the #1865 paste where line numbers are unreadable.
- `scripts/host-ram-check.sh [--total-gib N] <paste|-> [...]` — host-side memory
  from whatever the reporter pasted (`top`/`free`/`/proc/meminfo` plus `top` or
  `ps` process rows): host RAM total, the worker RSS sum and mean per rank, the
  percentage held and the verdict band, plus the reminder that VIRT is
  torch_npu's reservation and a first-sample `%CPU` is a since-boot average.
  Verified on the #3902 `top` paste: 16 ranks, 1472 of 2014 GiB (73 %), swap 0.
- `references/deploy-sizing/index.md` — deployment/capacity questions with no
  crash: per-card weight footprint vs `gpu_memory_utilization`, the sum of gmu
  over processes sharing a device, `ASCEND_RT_VISIBLE_DEVICES` vs the inert
  `CUDA_VISIBLE_DEVICES`, and how to answer when the orchestrator (vllm-omni) is
  not in the snapshot. `card-budget-and-device-visibility.md` is the worked #9335
  case (30B-A3B Omni on 2×64 GB: TP=2 owns both cards, the 2-NPU three-stage
  squeeze sums gmu to 1.8). Reproduce the numbers with
  `scripts/card-budget.sh --params P [--dtype] [--tp] [--gmu] [--hbm] [--cards]
  [--stage-gmu] [--stage-devices]`.
- `references/feature-enable-recipes/moe-expert-parallel-with-dp.md` — EP together
  with DP, and any engine-init failure whose deepest frame is the MoE
  `prepare_finalize.py:191 dist.all_gather`: the `MC2`/`ALLTOALL`/`ALLGATHER`
  selector table for A2 and A3, the derived capacity, why one server runs MC2 in
  decode and ALLGATHER in prefill, the MC2-inherits-All2All correction, the refuted
  uneven-split theory (`padded_num_tokens` is always a TP multiple), and the only
  EP+DP recipes the tree ships (#5295, v0.12.0rc1).
- `references/feature-enable-recipes/index.md` — "how do I turn F on" questions:
  the capability verdict first (docs statement at the pinned ref, whose hardware the
  measured numbers used, `tests/e2e/**` + nightly `npu_per_node` coverage, precedent
  issue versions), then each enforced constraint with its error text, then a
  one-axis-per-launch experiment. `context-parallel-pcp-dcp.md` is the PCP/DCP worked
  case (#8984, v0.19.1rc1, DeepSeek-V3.2 QuaRot on A2): `world = dp × pp × pcp × tp`
  (`distributed/parallel_state.py:41-52`, DCP rides inside TP), DCP divisibility,
  the `cp_kv_cache_interleave_size == block_size` `AssertionError` under PD
  (`platform.py:513-522`) versus the SFA silent override, the
  `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1` gate in the MoE path
  (`prepare_finalize.py:401,473`), the v2-runner `NotImplementedError`, why the
  pinned tree validates A3 only, and the #8578/#8579/#8580/#7375 family.
- `references/tutorial-verification/index.md` — the document is the deliverable:
  community-test/众测 reports and `[Doc]` reports that a tutorial step fails.
  `community-test-report.md` gives the task shape, the report skeleton and the
  staged-emission protocol (recorded #11739 shipped no report at all);
  `tutorial-command-audit.md` is the eight-check audit with the verified vllm
  v0.18.0 / vllm-ascend v0.18.0 and v0.19.1rc1 examples (dead `--swap-space` and
  `--disable-log-request`, `--random-input` vs `--random-input-len` and why the
  abbreviation still parses, `WORKDIR /workspace` vs `/vllm-workspace/`, default
  `limit_per_prompt` 999 and open media URLs, the boilerplate TP sentence that
  contradicts the sibling docs, the `0.13.0rc1` baseline).
- `references/image-dependency-faults/index.md` — faults the image or its build
  recipe owns, before any vllm-ascend code: an interpreter abort inside a bundled
  native module, a missing library the tutorials prescribe, a stale copied
  installer. `references/image-dependency-faults/native-import-abort.md` is the
  worked #5609 case (mooncake `corrupted size vs. prev_size`; toolkit-copy
  shadowing, jemalloc preload, version that works).
  `references/image-dependency-faults/nnal-atb-missing-library.md` is the
  hand-built-environment case (#10525, v0.20.2rc1): `libatb.so` at worker init
  means the NNAL/ATB component is missing or its own `set_env.sh` was never
  sourced — every official Dockerfile sources it, a manual build must too — plus
  the reporter-confirmed `CANN: None` `collect_env.py` blind spot and the
  non-fatal `triton_kernels` log noise.
  `native-string-abort.md` takes the other abort text — `std::logic_error` /
  `basic_string::_S_construct null not valid` at engine start: what the message
  proves (a null `char*` in native code, thrown outside any bound-call frame),
  the stock-model discriminator, the three native probes, the `gdb` backtrace
  that names the `.so`, and why the model's config is not the owner (#8745,
  #10943). `launch-env-and-library-path.md` takes
  `ImportError: <name>.so: cannot open shared object file` raised while a
  connector module is imported in a launcher-managed deployment (`args:` /
  `command:`, compose `entrypoint`, operator CR) while the pasted ENV dump shows
  the directory on `LD_LIBRARY_PATH`: the launch command's effective environment
  is the one that loads, the tutorials' exports belong in it, and image content
  is the answer only when the file is absent from disk (#7352, reporter-confirmed).
- `references/performance-throughput/index.md` — speed and cache questions on a
  healthy engine ("排队那么久", "缓存命中为 0", single-digit generation
  throughput): read the metrics block arithmetically first —
  `Waiting`/prompt-throughput for TTFT, `gen tok/s ÷ Running` for prefill/decode
  interference, `KV cache usage` plus the `Preemptions` line for capacity, and the
  prefix-cache line as an unconditional print, never as evidence.
  `prefill-queue-and-throughput.md` is the worked #6833 block with its arithmetic
  and lever table; `zero-prefix-cache-hit-hybrid.md` is the Qwen3-Next `0.0%` case
  — `SupportsMambaPrefixCaching` → `MambaModelConfig` disables prefix caching at
  vllm v0.14.1 (`patch_mamba_config.py` inherits it), the model's own `assert not
  cache_config.enable_prefix_caching`, the one log line that confirms it on the
  reporter's host, and the later-version recheck. A paste that carries both
  complaints needs **both** pages in one `skill_view` turn: the queue page is the
  arithmetic, the cache page is the gate, and the recorded #6833 replay opened
  only the cache page, re-derived the queue arithmetic from `scheduler.py`, and
  was cut mid-answer.
- `references/startup-failures/index.md` — engine/worker never becomes ready (lanes
  A/B). Pages: `silent-worker-exit.md`, `max-model-len-budget.md` (#3543,
  reporter-confirmed; the per-card HBM decides whether the default fits),
  `environment-correspondence.md`, `model-type-not-recognized.md` (the three registry
  gates, first-supporting-release precedent), `connector-config-attributes.md`
  (`AttributeError` on a config scalar in a PD KV connector: nested multimodal config
  #5572, or a flat config missing Qwen2 `head_dim` #7352),
  `arg-validation-errors.md` (a `ValueError` naming a flag combination, #11863),
  `acl-graph-capture.md` (capture-phase stall with canned advice, #3389),
  `quantization-method-gate.md` (`<method> quantization is currently not supported in
  npu`, #6800).
- `references/upstream-downstream-drift/index.md` — Python attribute/signature
  errors raised when a vllm-ascend module subclasses or re-implements an upstream
  vLLM/transformers class and the installed library moved on, **including when the
  failing frame is an upstream vllm file** because the Ascend replacement is
  keyed by class name. Scope: the attribute exists upstream; not a device-op or
  collective fault. Worked cases: #11661 `attention_type`
  (`patched-forward-api-drift.md`); #6784 `ZeroExpertFusedMoE` with no
  `select_experts` (`oot-class-registration-drift.md`, which also carries where
  the real vllm↔vllm-ascend pairing is recorded);
  `patch-import-side-effect.md` for the **other** way the Ascend op goes missing —
  a class monkeypatch installed as an import side effect of a hook that only the
  arg-parsing process runs, so Ray/`spawn` workers keep upstream's `forward_oot`
  and hit `fused_experts = None` (#253, v0.7.1rc1). `stale-module-binding.md` is
  the **third** scheme: the patch ran and rebound the defining modules, but the
  model module had cached the factory at import, so a masked
  `AttributeError` (`. Did you mean: 'apply_monolithic'?`) hides a `None`
  sub-object (#15815, v0.26.0rc1).
- `references/pd-disaggregation/index.md` — PD/EPD with the engine up but KV not
  moving, or moved and wrong. Pages: `transport-topology.md` (`HIXL Connect failed |
  103900`, `RanktableCheck`, `device_ip`, non-hostNetwork Pods, `HCCL_INTRA_ROCE_ENABLE`
  / `HCCL_INTRA_PCIE_ENABLE`, port bands; owner is LMCache-Ascend or `cann/hixl`, not
  the shim; #14506); `hybrid-state-transfer.md` (alternating corruption on GDN/Mamba
  models, the connector matrix, the `mamba_cache_mode='align'` gate, and the MTP
  `"mtp" in layer_name` sites; #12339); `kv-pool-store-segments.md` (the pool
  deployment: `global_segment_size` vs the HDK ceiling, reporter-confirmed < 20 GB on
  25.2.3, #13439).
- `references/dsa-sparse-attention/index.md` — DeepSeek-V3.2-Exp / DSA (SFA) reports.
  `indexer-cache-scatter-fault.md`: a device fault raised at the indexer's
  `npu_scatter_nd_update_` (MTE DDR out of range, one `pc start`, mixed-batch trigger) is
  a host slicing bug — the verified v0.11.0rc1 `c5fe179` slice mismatch, the concurrency
  A/B, the fix, and the v0.13.0rc2 re-derivation where the same asymmetry moves to the
  DSA-CP path (#4296). `concurrency-crash-with-no-log.md`: "多并发时崩溃" with nothing
  pasted — the config diff against `tests/e2e/**` and the tutorial, the phase-blind
  `MLAPO_MAX_SUPPORTED_TOKENS = 1024` gate, the maintainer-filed #4996 high-load family,
  and the `--max-num-seqs 1` + MTP arithmetic (#6240).
- `references/hccl-and-hardware/index.md` — HCCL, ACL/PTA and NPU device faults
  (lane C): the error-report timestamp gate first (`error-report-timestamps.md`,
  #9183), then `collective-error-chain.md`, then `device-health-checks.md` (the
  recorded answer was a physical HCCS link fault on 6 of 8 cards).
  `rpc-watchdog-timeouts.md` — a bare `TimeoutError` from `multiproc_executor.py` /
  `shm_broadcast.py`: vLLM's own RPC watchdog, not a collective (#1865).
  `control-plane-port-binding.md` — `Communication_Error_Bind_IP_Port (EI0019)` /
  `port <n> have already been bound` above 65535: HCCL's control-plane band, a
  finding only when the record date is in-window (#12487).
- `references/runtime-memory/index.md` — memory, both phases: healthy service that
  later died on HBM ("显存不释放", `NPU out of memory`), or an allocation that failed
  during KV-cache/weight placement. The capacity/growth/dead-key split, the
  `PYTORCH_NPU_ALLOC_CONF` injection gate, why `torchair_graph_config` /
  `ascend_scheduler_config` are accepted and ignored, and the OOM/leak precedent
  family (#12692, #13185, #13248, #13240, #3198, #2650; worked for #16022).
  `kv-init-and-format-cast-alloc.md` — the `EL0004`/`207001`/`507013`
  `npu_format_cast` signature (#4812). `host-ram-and-per-rank-rss.md` — "the machine
  crashed" with a `top`/`ps` paste and no NPU error (#3902: 16 ranks, 1472 of 2014
  GiB), plus `scripts/host-ram-check.sh`.
- `references/soc-platform-paths/index.md` — SoC-conditional paths, 310P first:
  `is_310p()` gating, the documented 300I/310P status, the seven platform gates,
  the NZ/`FRACTAL_NZ` conversions, and the 300I launch environment
  (`atlas-300i-310p.md`, verified at vllm-ascend v0.11.0rc3 + vllm v0.11.0).
  `device-type-build-vs-runtime.md` takes engine start dying at
  `check_ascend_device_type()`: the build stamp vs the live `soc_version`, the
  a2/a3/310p-only wheel matrix against the in-tree `Dockerfile.a5`, the
  don't-patch-the-guard argument, and the A2/A3-only support matrix (#12141,
  v0.21.0rc1).
- `references/observability-and-logs/index.md` — reports about log text rather
  than a crash ("the flag prints nothing", a missing startup stage).
  `request-output-logging-flags.md` is the verified gate table for
  `--enable-log-requests` / `--enable-log-outputs` / `VLLM_LOGGING_LEVEL` /
  `--max-log-len` at vllm v0.18.0 + vllm-ascend v0.18.0 (#10985), including the
  endpoint that can never log outputs and the `inspect.getsourcefile` check for
  which patched serving generator is live. `online-profiling-endpoints.md` takes
  the *missing route*: `POST /start_profile` → `404` after `--profiler-config`,
  attached only when the serving process's own `args.profiler_config.profiler` is
  set (`vllm/entrypoints/serve/profile/api_router.py` `attach_router`), so the
  answer is that gate plus one `openapi.json` request — and a worker-side miss is
  a 500, never a 404 (worked for #9583 at vllm v0.18.0 + vllm-ascend v0.18.0rc1).
- `references/serving-frontend-faults/index.md` — a Python exception that
  surfaces in the OpenAI server process while the report is filed as an Ascend
  problem, and whose traceback has **no** `vllm_ascend` frame: the owner is
  upstream vLLM or an installed Python dependency, so the answer is an ownership
  verdict plus a client-side A/B, and the artefacts to ask for are the serve
  command, the request body and the client concurrency — never a plog or
  `npu-smi`. `tokenizer-already-borrowed.md` is the worked #7197 case:
  `RuntimeError: Already borrowed` at `Hermes2ProToolParser.__init__` inside
  `_preprocess_chat` (vllm v0.16.0), the event-loop thread plus the
  `AsyncMicrobatchTokenizer` worker thread sharing one HF fast tokenizer, the
  single retry upstream already ships for the same exception, and the
  serialise/retry ladder.
- `references/custom-ops-runtime-faults/index.md` — aclnn/AscendC custom-op
  device faults: reading the device dump, mapping the operator to `csrc/<op>/`,
  tiling-mode tables. The dump names the operator and error class, not the
  shape; custom ops are often behind a runtime switch.
  `references/custom-ops-runtime-faults/attention-op-numeric-errors.md` is the
  silent-numerics half: an attention op that runs and returns a wrong tensor
  (uniform average / softmax dropped) with a framework-free repro — which caller
  holds the uncovered kwargs combination, the counter-example already in the
  tree, the zero-stub unit test that hid it, the three fix sites, and the cheap
  discriminators (worked for #13699 at v0.23.0rc1).
  `references/custom-ops-runtime-faults/op-not-registered-for-soc.md` is the
  other half: `EZ1001 Get regInfo failed` / `does not support opType` is a
  build/install SoC fault raised before any launch, and that page shows where the
  SOC_VERSION, `_cann_ops_custom` and image-tag decisions are made (#11519,
  v0.18.0).
- `references/accuracy-benchmarks/index.md` — score/accuracy disagreements between
  harnesses, images or runs; the decisive harness-vs-engine experiments. Pages, each
  with its own pinned refs and ablations: `reasoning-output-routing.md` (thinking
  inside `content`, `reasoning_tokens: 0`), `early-eos-short-output.md` (fluent stop
  after tens of tokens), `garbled-long-context-output.md`, `short-prompt-corruption.md`,
  `hybrid-block-size-corruption.md` (a `--block-size` diff on a hybrid GDN model),
  `qwen3-next-gdn-numeric-path.md`, `vl-vision-tower-degeneracy.md` (image answered as
  `!!!!` runs), `quantized-checkpoint-conversion.md`,
  `harness-vs-engine-attribution.md`. When a page's pins equal `SOURCES.json`'s refs,
  cite its `path:line` as they stand.
- `references/tool-call-streaming/index.md` — the report is about a **tool call**
  rather than prose: a client-side `Unterminated string` / `Expecting ',' delimiter`
  while it assembles `function.arguments`, arguments "被截断"/格式错误/重复输出, or
  `stream: true` returning only CoT. The three layers (upstream frontend delta
  assembly, the vllm-ascend sampler, the model stopping early) and the one request
  that separates them. `streamed-arguments-audit.md` is the procedure: obtain the
  raw SSE, run `scripts/sse-toolcall-audit.sh <raw-sse|->`, route on its three
  outcomes (parses = client; tail missing = stream ended early, split with
  `return_token_ids: true`; unbalanced middle = generation), then the ordered
  ablations and the ownership table. Calibrated on #7580 (reassembles to
  `{"filePath":"/…/K8sYamlAgent.java`, `json.loads` reproduces the reporter's own
  error) and #8122 (97 chunks assemble to valid JSON, so the posted sample is not
  the failing one). `mtp-block-verify-acceptance.md` carries the
  `using_block_verify = max_spec_len >= 3` branch
  (`vllm_ascend/sample/rejection_sampler.py:114`, comment `# MagicMTP: Improving
  acceptance rate with Block Verify.`), the cumulative-product acceptance and its
  recovered/bonus slots, the greedy exemption, the `parse_output` placeholder
  filter that removes slots silently, the missing strict-equivalence test, and the
  `n=2` / `temperature: 0` ablations.
- `scripts/abi-symbol-check.sh <paste|-> [...]` — classifies a pasted native
  `undefined symbol` against the two ATB/NNAL builds: demangles every mangled name
  (`c++filt` when present) and labels the `std::string` ABI (`Ss` old vs
  `NSt7__cxx11` new), notes line-wrapped fragments, reports which `cxx_abi_*` root
  the env dump selects, warns when the same paste is really the missing-library or
  old-`GLIBCXX` class, then prints the verdict, the three decisive `nm` commands and
  the reporter-confirmed `cxx_abi_0` export. Verified on the #1144 paste (old-ABI
  `AsdOps::Ops::GetOperationByName`, `ATB_HOME_PATH=…/cxx_abi_1`).
- `scripts/sse-toolcall-audit.sh <capture|-> [more files]` — concatenates every
  `choices[i].delta.tool_calls[j].function.arguments` fragment in arrival order and
  prints per index the fragment count, the empty fragments, the `{ } [ ] "`
  balance, the assembled string's head/tail and `json.loads`, plus the
  `stop_reason` / `finish_reason` chunks and the last non-empty fragment, then a
  verdict (client / tail lost / middle corrupted). Tolerates HTML-Excel wrappers,
  console prefixes and `[REDACTED]` placeholders. Verified on the #7580 paste
  (tail lost, `Unterminated string starting at: line 1 column 13`) and the #8122
  paste (balanced, valid JSON).
- `references/spec-decode-mtp/index.md` — symptoms the speculative-decoding path
  owns while the engine is otherwise healthy. `acceptance-collapse.md` — `Accepted: 0`
  with `Drafted: N`, ordered MTP-off ablation, async-spec-decode
  `num_computed_tokens` drift, the `mamba_block_size` prefix-caching gate (#10495,
  where the run led with capacity and the reporter showed MTP was the owner).
  `empty-first-response.md` — `content: ""` / `completion_tokens: 1` /
  `finish_reason: "stop"` on the first request, the restart reproduction, the sibling
  tutorial diff and capture-size arithmetic, the refuted scheduler padding/null-block
  theory next to the live MTP padded-drafter gate, the draft-side ablation rows and
  `min_tokens` (#9355).