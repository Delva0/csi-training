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

Recorded cases, per-issue index and how each session ended:
`references/evidence-workspace/recorded-cases.md`; the load-bearing
delivery pattern is rule 1.

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and obey the delivery gates.** A long
   session can be cut off mid-sentence, so the deliverable has to survive
   truncation: what decides the outcome is not how much evidence was gathered but
   whether the message exists. The runs that shipped **no answer at all** (#6784,
   #12339, #10985, #11739, #11759, #5295, #8724, #17991, #16111, #8788, #10546)
   each had a usable lead; #4296 answered completely in ten turns from one pasted
   traceback. Hard stops:
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
   - **The gates cap cumulative output, not only turns.** #4057, #6833, #13853,
     #16111 and #8788 were each cut after 11-49 K characters of reasoning with the
     ranked cause in hand and no message text; #6833's replay re-derived its verdict
     for ten turns and was cut inside its action ladder, and #16111 and #8788
     composed the whole answer inside the reasoning block the cap landed on. A block
     that restates the verdict, numbers the fixes or lays out the evidence table *is*
     this turn's message: send it, then one tool call. Count characters, not turns.
   - **Reasoning is a scratchpad, a few lines per turn:** what the last result
     proved, the single next check, go. It spends the **same output budget** as the
     answer: #15815 and #11759 each spent ~30 K characters of reasoning against
     ~9 K of answer, and #11759 hit the cap with 69 characters of text. If a block
     enumerates the answer's sections, or says "I have enough" / "let me write the
     answer", that block *is* the answer.
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
     whole report in one heredoc: #11739 made 40 tool calls and wrote no file;
     #11759 spent 16 turns, 69 characters of message text and 30 K of reasoning,
     then was cut composing a 5.7 K report, so no file was ever written.
   - **`/workspace/scratch/findings.md` is the answer draft, not an extra step:**
     write it at G2 *in place of* the reasoning block, and skip it when the
     session is already long (#11519 composed it late and lost its ranked causes).
     The file is not the message: once it exists, the next turn's text is the
     answer, not a re-derivation of it (#10546 wrote it on its 12th of 14 turns,
     then spent the last turn restating the same verdict and was cut with zero
     text).
   - **A missing or mismatched artefact is itself the finding** — an empty
     environment block, a traceback file absent from the pinned tree, a version
     the snapshot does not carry. State the gap, give the one command that closes
     it, stop investigating.
   - **Truncation eats the tail, not the head.** Each action directly after its
     cause; the closing scope/open-questions block 3-5 lines, so a capped answer
     still ends in something usable.
   - **Every assistant turn carries message text, not only a tool call.** A turn
     whose whole output is a tool call leaves nothing for a truncation to keep.
     #5295 emitted *zero* message text on all 16 turns: ~46 K characters of
     reasoning (≈12 K of the 16 K output budget), three "must answer NOW" notes,
     and a cut inside a reasoning block while opening the file that would have
     refuted its last hypothesis. Carry one or two sentences of the
     current answer draft in the same message as every tool call — verdict line
     first, then the evidence being checked — and G1/G2 need no special turn.
     #15815 shipped only on turn 15, after 31 K of its 50 K output characters
     spent on reasoning.
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
   - Never list a history or attachments directory, never dump `SOURCES.json`, and
     never bundle a listing with the `skill_view` call (it lands before this rule is
     read): a listing cost ~24-28 KB (#13699: 2857 entries), and recent batches
     still paid 13-54 KB per session (#9335, #9183, #11739, #8984, #12487). Load
     the skill alone, then make the first shell call `ls /workspace`
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
    question, and it is answerable without that module's source.** Three forms,
    one cap of ~6 calls before you emit: a `python -c "import <pkg>"` abort with
    glibc allocator text and no vllm-ascend frame (which copy is imported, one
    `LD_PRELOAD` allocator, the installer-vs-dependency diff →
    `native-import-abort.md`); `'std::logic_error'` /
    `basic_string::_S_construct null not valid` at engine start (`std::string(nullptr)`
    in a `.so` → `native-string-abort.md`); and `ImportError: …/_ops*.so: undefined
    symbol: …Ss` from a prebuilt wheel (`std::string` ABI split, decided by the env
    dump's `cxx_abi_{0,1}` root, reporter-confirmed on #1144 →
    `abi-undefined-symbol.md`). Start at
    `references/image-dependency-faults/index.md`; #5609 and #10943 each burned the
    session in ABI/RPATH or `cann/` archaeology after the answer was in hand. The
    three pages above live under `references/image-dependency-faults/`.
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
    failure minute. The script needs a timestamped record — long-form
    `YYYY-MM-DD-HH:MM:SS` device records (the usual `[PID: n]` headers) or
    `MM-DD HH:MM:SS` line prefixes; a paste of ordinary `W<MMDD>`/`E<MMDD>` lines
    yields its "no timestamped error line found" verdict, so do not rebuild it into
    a file first (#18315). Same trap as rule 11 one level up: not just the
    sentence but the whole block can be the wrong one.
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
| pydantic `ValidationError`: `Unexpected keyword argument [type=unexpected_keyword_argument, input_value=True, input_type=bool]` + an `errors.pydantic.dev` URL, naming a key from `--additional-config`, often with `ERR99999`, no `WorkerProc` line | `additional_config` **schema** rejection: `AscendConfig` is `extra="forbid"`, so the key is nested-only (e.g. `ascend_compilation_config.fuse_muls_add`) or removed, and a tutorial serve command usually carries the wrong nesting | `references/startup-failures/additional-config-schema.md` |
| The report asks whether a plan fits rather than what broke: "can this model run on N cards", a multi-stage orchestrator (`--stage-id`, `--headless`, `platforms.<vendor>` yaml) that works on GPU, stages that must share a card, `CUDA_VISIBLE_DEVICES` in a command aimed at an NPU host | Deployment sizing / stage layout: memory arithmetic plus the device-visibility variable, not a fault | `references/deploy-sizing/index.md` |
| `[Community Test]` / `[Doc]` / 资料验证 / 测试报告, or "the tutorial's command does not work as written" | Tutorial-document verification: grade the page against the pinned tree; the report is the deliverable | `references/tutorial-verification/index.md` |
| `value error, the checkpoint you are trying to load has model type '<mt>' but Transformers does not recognize this architecture`; `Model architectures [...] are not supported`; "can version V deploy model M?" | Model-support / version gate | `references/startup-failures/model-type-not-recognized.md` |
| Python `AttributeError`/`TypeError` in a frame under `vllm_ascend/{ops,patch,models}` where the class subclasses or re-implements an upstream vLLM/transformers class — e.g. `'AscendQwen2DecoderLayer' object has no attribute 'attention_type'`, Python `TypeError: … got an unexpected keyword argument` (the pydantic form is the schema row above), `cannot import name` | Downstream patch drift | `references/upstream-downstream-drift/index.md` |
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
| `call aclnnArange failed … EZ9999: Inner Error!` + `Parse dynamic kernel config fail` + `AclOpKernelInit failed opType` + `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE failed` on a **built-in** op at `Starting to load model`, often asked as "这个镜像支持 910C 吗" (a card the docs exclude) | Image-vs-card CANN **kernel package**: the built-in op has no kernel config for the running SoC, so nothing launched — not the custom-op row above, not a flag, and the container uses the image's CANN, not the host's | `references/soc-platform-paths/kernel-package-vs-card.md` (one standalone `torch.arange(8, device="npu:0")` probe decides it) |
| Output is fluent but stops after tens of tokens ("输出被截断", "只输出几十个 tokens") and a large `max_tokens` appears to fix it; `usage.prompt_tokens` is small; a `pass_config` / SP / FLASHCOMM / quantization flag changed at the same time | Early EOS: `finish_reason` decides budget vs model-issued EOS, then every reader of the toggled key plus the shipped support matrix (accuracy track, not the length budget) | `references/accuracy-benchmarks/early-eos-short-output.md` |
| Spec-decode metrics collapse: `Accepted: 0` with `Drafted: N > 0`, `Per-position acceptance rate: 0.000` at every position, `Mean acceptance length: 1.00`, generation throughput ~0 while requests stay `Running`; often `--async-scheduling` and/or `--enable-prefix-caching` on a hybrid GDN model (Qwen3.5/Qwen3.6, DeepSeek-V4, GLM) | Speculative-decoding bookkeeping (draft and target no longer see the same context), not capacity: the MTP-off ablation is row 1 | `references/spec-decode-mtp/acceptance-collapse.md` |
| First request after start answers `content: ""` with `usage.completion_tokens: 1` and `finish_reason: "stop"`, later requests fine; MTP plus a full-graph decode on a hybrid GDN model | Empty first response: one model-issued blank/stop token — neither the length budget nor reasoning routing — so reproduce by restart before naming a one-shot mechanism, and diff the spec/capture config against the family sibling tutorial (`"enforce_eager": true` inside `--speculative-config`, default capture sizes) | `references/spec-decode-mtp/empty-first-response.md` |
| "score with tool A is N points below tool B / another image / the vendor number", accuracy regression after a change | Accuracy/score discrepancy | `references/accuracy-benchmarks/index.md` |
| Per-layer `MSE loss` / epoch table printed while converting a checkpoint with msmodelslim (W4A4/W8A8), one layer far above its neighbours, or "the converted checkpoint answers garbage" with no engine-side failure | Quantization-conversion phase: converter-owned, before the engine starts | `references/accuracy-benchmarks/quantized-checkpoint-conversion.md` |
| A long prompt (100K+) answered as 乱码 / garbage / repeated / empty text while short prompts are fine, with sparse c8 attention, DCP/PCP, PD disaggregation or KV offload enabled | Generation corruption (accuracy track, different evidence) | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| A **few** responses in a long accuracy run repeat sentences ("5 条响应的句子重复") on a documented model with PD + MTP + prefix caching, prompt short but `max_tokens` in the tens of thousands, no crash and no error text | Decode-time repetition, not prompt-length corruption: the owners are reuse and draft bookkeeping (prefix cache, preemption, MTP), so the answer is the tutorial P/D diff plus the `--max-num-seqs 1` identity row — long-CoT models loop on their own | `references/accuracy-benchmarks/long-output-repetition.md` |
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
| A constant in the code equals the number in the report — `num_speculative_tokens: 3` fails while `1` (and `2`) work — and the wrong output is exact-form: tool-call JSON, ids, code | Named code branch, not an investigation: `using_block_verify = max_spec_len >= 3` is the only spec-decode branch keyed on that value, **but newer refs also require `rejection_sampler_config.enable_block_verify`, default `False` (v0.23.0rc1) — read the line first**; then state it with its two ablations (`n=2`, `temperature: 0`) before its kernel is read | `references/tool-call-streaming/mtp-block-verify-acceptance.md` |
| Report pastes only a response body: the CoT/thinking text sits in `content`, `message.reasoning` / `reasoning_content` never appears, `completion_tokens_details.reasoning_tokens` is `0`, thinking language mixed, or the complaint is only "回答中英混杂 / the answer is in English" while `content` reads `null` | Output routing — **upstream OpenAI frontend, not a vllm-ascend defect**: `reasoning_parser` defaults to `""`, so `--reasoning-parser <name>` (e.g. `qwen3`) is the whole gate; invariant to image tag and MTP | `references/accuracy-benchmarks/reasoning-output-routing.md` |
| Host is **310P** (Atlas 300I Duo, `Ascend310P3`, image `-310P`), or the traceback sits on an `is_310p()` branch (NZ/`FRACTAL_NZ` format casts, no Ascend custom ops, `patch_mamba_config_310`) | SoC-conditional platform path: experimental per the pinned docs, NZ casts are real conversions, fp16/eager are mandatory | `references/soc-platform-paths/atlas-300i-310p.md` |
| `EngineCore failed to start` whose deepest frame is `worker.py` → `check_ascend_device_type()` with **no** `Starting to load model` line: `AssertionError: Current device type: AscendDeviceType.A5 does not match the installed version's device type: AscendDeviceType.A2, please check your installation package.` (or `RuntimeError: Can not support soc_version: <n>`), typically on a 950PR/A5 host that installed a mirror wheel | SoC/package correspondence: the device type is **baked into the package at build time** (`_build_info.__device_type__` from `SOC_VERSION`), the release pipeline ships a2/a3/310p variants only while `Dockerfile.a5` exists in-tree, so an A5 host needs a source build on the node — and the guard must not be patched | `references/soc-platform-paths/device-type-build-vs-runtime.md` (two one-liners decide it) |
| The report asks how to enable a feature rather than what broke: "如何正确开启 PCP/DCP", "F alone and G alone each报错，能否给个能跑通的示例", no log pasted, or two errors described in one sentence | Feature-enable / capability question: the validation verdict (docs statement at the pinned ref, whose hardware the measured table used, `tests/e2e/**` coverage, precedent versions) before any constraint list | `references/feature-enable-recipes/index.md`, then `context-parallel-pcp-dcp.md` |
| `Communication_Error_Bind_IP_Port(EI0019)` / `Failed to enable listening for the host network adapter socket` / `port <n> have already been bound` where n is above 65535, often in the issue title | HCCL control-plane port band, not an OS socket conflict — and only if the record's date is inside the window (rule 13) | `references/hccl-and-hardware/control-plane-port-binding.md` |
| Output degenerates (`</think>` leaking into `content`, repetition collapse, 乱码) at long-ish context **and** concurrency on a **documented** model with PD / spec-decode / prefix-cache flags, with **both** `vllm serve` commands pasted and no ablation | Config-diff track: the tree's own `docs/source/tutorials/models/<model>.md` ships the validated P and D commands and a `Recommended Configurations` table, so the first answer is that diff plus the ablation matrix — never a walk through the KV connector; #13002's five deviations are ranked ablation rows, not established causes | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| "服务崩溃" / 崩溃 under concurrency or high load with **no** error text, no traceback and no plog pasted, while single-request serving is correct (often DeepSeek-V3.2 + MTP + DP/EP) | Report-gap track: nothing in the paste can confirm a mechanism, so the answer is the config diff against the project's own validated command at the pinned commit plus 1-2 single-variable A/Bs | `references/dsa-sparse-attention/concurrency-crash-with-no-log.md`; `references/runtime-memory/index.md` for the OOM row |
| "多并发时请求排队那么久" / "生成吞吐只有个位数" / `Prefix cache hit rate: 0.0%`, with a `loggers.py` metrics block, a healthy engine and no error text | Throughput/queueing on a healthy engine: TTFT is prefill work ÷ prefill throughput, the interference is `gen tok/s ÷ Running`, and the cache line is a capability switch before it is a tuning knob | `references/performance-throughput/index.md` (both pages in one turn when the paste carries both complaints — the arithmetic page first) |
| A 19-token to few-hundred-token prompt on a hybrid GDN model (Qwen3-Next) answered as a digit run or a repeated phrase while ordinary chat is fine, `finish_reason: length`, no traceback | Hybrid linear-attention numeric path (accuracy track): the prefill GDN kernel is a downstream fp32 re-implementation while decode keeps a patched Triton kernel | `references/accuracy-benchmarks/qwen3-next-gdn-numeric-path.md` |
| "去掉 `HCCL_OP_EXPANSION_MODE`(AIV) / `PYTORCH_NPU_ALLOC_CONF` 就能拉起", first in-window error `aclnnMoeDistributeDispatchV4` / `…CombineV4 … 107020`, `event wait timeout … task_timeout` | The env knob changed the **selected MoE path**, not the hardware: AIV × MC2 dispatch/combine is a defect family (#14928, #13688, #15195), rank arithmetic decides whether MC2 was eligible, and `expandable_segments` is innocent apart from the CaMem pool assert | `references/hccl-and-hardware/aiv-mc2-env-ablation.md` |
| The reporter wants `--enable-expert-parallel` **and** `--data-parallel-size` > 1 together ("同时开启 EP + DP", "去掉 EP 或 DP 设成 1 就不报错"), or engine init dies at `vllm_ascend/ops/fused_moe/prepare_finalize.py` `dist.all_gather` behind `No available shared memory broadcast block found in 60 second(s)` on 910B | MoE communication-method selection, not a flag bug: A2 picks `MC2` only when `num_tokens <= mc2_tokens_capacity` **and** `world_size_across_dp/pp >= 16`, so DP2×TP8 = 16 is the boundary while DP=1 and EP-off fall to `ALLGATHER` by arithmetic; `PrepareAndFinalizeWithMC2` inherits the All2All `finalize`, so that all_gather **is** the MC2 path | `references/feature-enable-recipes/moe-expert-parallel-with-dp.md` |
| Deepest `vllm_ascend` frame is in `vllm_ascend/patch/**` (`patch_model_runner.py`, `patch_core.py`, …) | Patch **copy**: each `patch/` module assigns a verbatim upstream method under a `vllm_version_is('<ver>')` gate, so the failing statement may be upstream text. Diff the copy against the pinned upstream definition before deriving anything | `references/upstream-downstream-drift/patch-copy-vs-upstream.md` (2-3 calls) |
| `--async-scheduling` **and** `--speculative-config` with fractional `temperature` (`0` is fine): worker `KeyError: '<id>'` in `_update_states`, EngineCore `'NoneType' … sampled_token_ids`, APIServer `Counters can only be incremented by non-negative amounts`, same second, every DP rank | One provenance, three symptoms: a discarded request's valid count is 0 → `num_accepted = -1`, and `prev_req_id_to_index` omits exactly those requests; the unguarded lookup is upstream vLLM's, carried by the patch | `references/spec-decode-mtp/async-spec-bookkeeping-desync.md`; row 1 is `--async-scheduling` off |
| Deepest `vllm_ascend` frame is `counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)` in `_copy_valid_sampled_token_count` (`model_runner_v1.py:1393`), batched with `Execution_Error_SDMA` / `rtMemcpyAsyncEx … reason=the model stream execute failed`, an `aivec … timeout or trap error` and an HCCL `AllGather`/`AllReduce` line carrying `rankSize` / `numBlocks`; DP or PD deployment, under concurrency | Device fault surfacing at the drafter's valid-count D2H copy (Lane C sync point) — and the frame is also free evidence: it only runs with `--async-scheduling` **and** spec decode **and** the padded drafter batch, so the paste proves the launch shape. Resolve the collective by rank size against the DP×PCP×TP group arithmetic (a 2-rank group is not `fc3_quant_x` on DP2×TP4), then `--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1` for attribution | `references/spec-decode-mtp/drafter-count-copy-device-fault.md` |
| `ACL stream synchronize failed, error code:507015` / `EE9999 … aicore exception` whose deepest frame is an attention backend while the step held **both** decodes and prefills; often "completes with spec decode disabled" | Mixed-batch split **units**: `num_decodes` counts requests while `num_decode_tokens` counts tokens (× `1+num_spec`), so a request-indexed `block_tables`/`seq_lens` sliced by the token count, or an unrebased `query_start_loc` passed as `cu_seqlens`, reads out of its own storage; the reported line is often the decode half's `.npu()` | `references/mixed-batch-splits/index.md`, then `fa3-mixed-batch-split.md` (one call: `scripts/batch-split-audit.sh <root>`) |
| KV pool / MemCache on **800I/T A3** with a DeepSeek-V4 (`compress_ratios`) model: first request fine, the **second** kills the workers (`HCCL watchdog … ERR02005`), or the `host_shm` variant dies at memcache init with a "device id 转换异常"; reporter says "懒加载" | KV-pool store lane: lazy init is the documented A3+DSV4+`device_sdma` behaviour (#9975) and a rank-local save failure is swallowed before it desyncs collectives (#14106) — diff the resolved `mmc-local.conf` against the tree's tables first | `references/kv-pool-store/memcache-a3-dsv4-init.md` |
| Compiler or missing-header error while installing/building — or `pip install` from source in a pod with **no public network** (custom-ops step pulls from `gitcode.com`, `fetch failed`) | Build-time dependency closure, else the general sequence below | `references/image-dependency-faults/offline-source-build.md`, else skip the topical routes |

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
   - Then the per-track checks: `references/first-checks/index.md` routes to
     `cheap-checks-by-track.md`, the full list (image/native aborts, source
     builds, connector and config-attribute lanes, PD transport, single-node
     KV-pool/MemCache, spec-decode and tool-call rows, memory, sizing, EP×DP, SoC
     paths, `patch/**` frames). Open the entry that matches your classification
     instead of reading the whole list, and take its named script or check before
     any source walk.
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

- `references/first-checks/index.md` — the per-track cheap checks of procedure
  step 3 (`cheap-checks-by-track.md`): one entry per lane/track, each ending at the
  owning topic page, so a classified report needs one entry rather than the whole
  list.
- `references/evidence-workspace/index.md` — read first in an unfamiliar case:
  case layout, `SOURCES.json` fields/limits, history search, OCR/attachment
  caveat, snapshot coverage.
  `references/evidence-workspace/delivery-discipline.md` holds the G1/G2/G3
  delivery gates of rule 1 and the reasoning patterns that break them.
  `references/evidence-workspace/recorded-cases.md` is the per-issue index behind
  rule 1 (version, symptom, track, outcome).
- `scripts/history-titles.sh '<regex>' [root] [limit]` — precedent index: one
  `<file> :: <title>` line per matching history issue, so a family is triaged
  without reading bodies. Run as `bash <skill_root>/scripts/history-titles.sh
  '<regex>'`, where `skill_root` is the path `skill_view` reports.
- `scripts/sources-summary.sh [SOURCES.json ...]` — the version contract, one
  line per repo with coverage and `runtime_verified` flags, plus the
  `unavailable` repos, so "is `cann/hixl` in this snapshot?" needs no JSON dump.
- `scripts/device-type-check.sh <paste|serve-log|-> | --soc-version N [--stamp X]` —
  decodes the `check_ascend_device_type()` failure: which side is the **card** and
  which is the **wheel** (the reading error that flips this diagnosis), the raw
  `soc_version` map (`220-225` A2, `250-255` A3, `200-205` 310P, `260` A5), the
  unmapped-code `RuntimeError` branch, and the non-fatal `liboptiling.so` line.
  Verified on #12141 (`A5` card, `A2` wheel).
- `scripts/kvpool-segment-check.sh <mooncake.json|serve-log|-> [...]` — the KV-pool
  store segment gate: per-rank `global_segment_size` / `local_buffer_size` from
  `MOONCAKE_CONFIG_PATH` / `MOONCAKE_GLOBAL_SEGMENT_SIZE` or, from a log, the
  worker's `register mem addr:…, length:N, mem type:host` lines, plus the
  `npu-smi` HDK version and pool/channel markers, and the verdict bands (≥ 20 GiB
  → #13439's fault, fix at `"2GB"`; 1 GiB multiple). Verified on #13439: recovers
  40.00 GiB from `length:42949672960`.
- `scripts/ranktable-check.sh <rank_table.json|log.txt> [...] | -` — parses a pasted
  or plog-embedded HCCL rank table: effective `server_count`, per-server
  `server_id`/`device_id`/`device_ip`, duplicate or non-contiguous ids, and the
  verdict — `server_count > 1` (or two `server_id`s) means a multi-server view, so
  `device_ip` + RoCE is mandatory; `server_count == 1`, one `server_id` means local
  HCCS is selectable. Use before any KV-connector code (#14506, #2399).
- `scripts/aclgraph-size-budget.sh --layers N [--tp --dp --pcp --dcp --ep --moe
  --aiv --max-num-seqs --sizes]` — reproduces `update_aclgraph_sizes`: the
  maximum captureable batch sizes for a model/parallel configuration and the
  sampled subset, so a startup log's size list can be checked arithmetically
  (verified: 94 layers, TP=16 gives 6 sizes = `[512,408,304,192,88,1]`, #3389).
- `scripts/plog-timeline.sh [--since YYYY-MM-DD] <log|-> [more logs]` — per-error-family
  timeline of a pasted plog / serve log: the span of the embedded device records vs
  the log-line dates, which families are re-printed from an older date, and the
  first error inside the incident window (default window = the newest date in the
  paste). Verified on #9183: the `107030` block as a previous-day replay, the
  `EZ9999` aivector line as the first in-window error.
- `scripts/executor-timeout-check.sh <log|->` — one-call split between vLLM's own
  RPC/ringbuffer watchdog and a real collective timeout: the `RPC call to <method>
  timed out.` text, which `shm_broadcast` frame is present (read = caller waiting,
  write = readers not consuming), any `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` in the
  paste, ringbuffer starvation and collective-side `EI…`/`Execute timeout` text,
  then the verdict and next artefacts. Tolerates screenshot-OCR garble
  (`sh]n broadcast`), so it works on #1865 where line numbers are unreadable.
- `scripts/host-ram-check.sh [--total-gib N] <paste|-> [...]` — host-side memory
  from a `top`/`free`/`/proc/meminfo` paste with `top` or `ps` rows: RAM total,
  worker RSS sum and mean per rank, percentage held, verdict band, plus the
  reminders that VIRT is torch_npu's reservation and a first-sample `%CPU` is a
  since-boot average. Verified on #3902 (16 ranks, 1472 of 2014 GiB, 73 %).
- `references/deploy-sizing/index.md` — deployment/capacity questions with no
  crash: per-card weight footprint vs `gpu_memory_utilization`, the gmu sum over
  processes sharing a device, `ASCEND_RT_VISIBLE_DEVICES` vs the inert
  `CUDA_VISIBLE_DEVICES`, and the answer when the orchestrator (vllm-omni) is not
  in the snapshot. `card-budget-and-device-visibility.md`: #9335's 30B-A3B Omni on
  2×64 GB, where TP=2 owns both cards and the 2-NPU three-stage squeeze sums gmu to
  1.8. Reproduce with `scripts/card-budget.sh --params P [--dtype] [--tp] [--gmu]
  [--hbm] [--cards] [--stage-gmu] [--stage-devices]`.
- `references/feature-enable-recipes/moe-expert-parallel-with-dp.md` — EP together
  with DP, and any engine-init failure whose deepest frame is the MoE
  `prepare_finalize.py:191 dist.all_gather`: the `MC2`/`ALLTOALL`/`ALLGATHER`
  selector table for A2 and A3, the derived capacity, why one server runs MC2 in
  decode and ALLGATHER in prefill, the MC2-inherits-All2All correction, the refuted
  uneven-split theory (`padded_num_tokens` is always a TP multiple), and the only
  EP+DP recipes the tree ships (#5295, v0.12.0rc1).
- `references/feature-enable-recipes/index.md` — "how do I turn F on" questions:
  the capability verdict first (docs statement at the pinned ref, the hardware the
  measured numbers used, `tests/e2e/**` + nightly `npu_per_node` coverage, precedent
  versions), then each enforced constraint with its error text, then a one-axis
  experiment. `context-parallel-pcp-dcp.md` is the PCP/DCP worked case (#8984,
  v0.19.1rc1): `world = dp × pp × pcp × tp` (`parallel_state.py:41-52`), DCP
  divisibility, the `cp_kv_cache_interleave_size == block_size` `AssertionError`
  under PD (`platform.py:513-522`), the `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1` MoE
  gate, the v2-runner `NotImplementedError`, and the #8578-#8580/#7375 family.
- `references/tutorial-verification/index.md` — the document is the deliverable:
  community-test/众测 and `[Doc]` reports that a tutorial step fails.
  `community-test-report.md`: task shape, report skeleton, staged-emission
  protocol (#11739 shipped no report). `tutorial-command-audit.md`: the
  eight-check audit verified at vllm v0.18.0 / vllm-ascend v0.18.0 and v0.19.1rc1
  (dead `--swap-space`/`--disable-log-request`, `--random-input` vs
  `--random-input-len`, `WORKDIR` vs `/vllm-workspace/`, default
  `limit_per_prompt` 999 and open media URLs, the boilerplate TP sentence, the
  `0.13.0rc1` baseline).
- `references/image-dependency-faults/index.md` — faults the image or its build
  recipe owns, before any vllm-ascend code: an interpreter abort inside a bundled
  native module, a missing library the tutorials prescribe, a stale copied
  installer (`native-import-abort.md` is the worked #5609 mooncake case).
  `nnal-atb-missing-library.md` is the hand-built-environment case (#10525,
  v0.20.2rc1): `libatb.so` at worker init means NNAL/ATB is missing or its
  `set_env.sh` was never sourced — every official Dockerfile sources it — plus the
  `CANN: None` `collect_env.py` blind spot and non-fatal `triton_kernels` noise.
  `native-string-abort.md` takes `std::logic_error` /
  `basic_string::_S_construct null not valid` at engine start: the null `char*`
  outside any bound-call frame, the stock-model discriminator, three native probes
  and the `gdb` backtrace naming the `.so` (#8745, #10943).
  `launch-env-and-library-path.md` takes `ImportError: <name>.so: cannot open
  shared object file` in a launcher-managed deployment while the ENV dump shows
  the directory on `LD_LIBRARY_PATH`: the launched process's environment loads, so
  the export belongs in the launch command; image content only if the file is
  absent (#7352, reporter-confirmed).
  `offline-source-build.md` is the no-public-network *source build* (#8788,
  v0.13.0rc3): the `setup.py build_aclnn` → `csrc/build_aclnn.sh` →
  `csrc/build.sh` → `csrc/CMakeLists.txt` chain, the `catlass` submodule (always)
  plus the four `gitcode.com` fetches with the existence gate each one honours,
  and the two knobs that change nothing (`CANN_PKG_SERVER`, an exported
  `CANN_3RD_LIB_PATH`).
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
  reporter's host, and the later-version recheck. A paste carrying both complaints
  needs **both** pages in one `skill_view` turn: #6833's replay opened only the
  cache page, re-derived the queue arithmetic and was cut mid-answer.
- `references/startup-failures/index.md` — engine/worker never becomes ready (lanes
  A/B): `silent-worker-exit.md`, `max-model-len-budget.md` (#3543, per-card HBM vs
  the default), `environment-correspondence.md`, `model-type-not-recognized.md`
  (three registry gates, first-supporting-release precedent),
  `connector-config-attributes.md` (`AttributeError` on a PD KV-connector scalar:
  nested multimodal config #5572, flat config missing Qwen2 `head_dim` #7352),
  `arg-validation-errors.md` (`ValueError` naming a flag combination, #11863),
  `acl-graph-capture.md` (capture-phase stall, #3389), `quantization-method-gate.md`
  (`<method> quantization is currently not supported in npu`, #6800).
- `references/upstream-downstream-drift/index.md` — Python attribute/signature errors
  from a vllm-ascend module subclassing/re-implementing an upstream class, **including
  when the failing frame is an upstream file** (the replacement is keyed by class
  name). Worked: #11661 `attention_type` (`patched-forward-api-drift.md`); #6784
  `ZeroExpertFusedMoE` (`oot-class-registration-drift.md`, also where the real
  vllm↔vllm-ascend pairing is recorded); `patch-import-side-effect.md` (the patch is an
  import side effect only the arg-parsing process runs, so Ray/`spawn` workers keep
  upstream's `forward_oot`, #253); `stale-module-binding.md` (patch ran, the consumer
  module had cached the factory, #15815); `patch-copy-vs-upstream.md` (the `patch/**`
  frame: extract-and-diff, formatting vs added statement, #8724).
- `references/pd-disaggregation/index.md` — PD/EPD with the engine up but KV not
  moving, or moved and wrong. `transport-topology.md`: `HIXL Connect failed |
  103900`, `RanktableCheck`, `device_ip`, non-hostNetwork Pods,
  `HCCL_INTRA_ROCE_ENABLE`/`HCCL_INTRA_PCIE_ENABLE`, port bands; owner is
  LMCache-Ascend or `cann/hixl`, not the shim (#14506). `hybrid-state-transfer.md`:
  alternating corruption on GDN/Mamba, the connector matrix, the
  `mamba_cache_mode='align'` gate, MTP `"mtp" in layer_name` sites (#12339).
  `kv-pool-store-segments.md`: `global_segment_size` vs the HDK ceiling,
  reporter-confirmed < 20 GB on 25.2.3 (#13439).
- `references/kv-pool-store/index.md` — a KV pool is attached (AscendStore +
  Memcache/Mooncake) and the request path breaks, **also with no PD**;
  `memcache-a3-dsv4-init.md` is the #16111 lane: the deliberate lazy init
  (#9975), the two inits (scheduler client vs worker; `mmc_init` first-wins),
  the per-die `dram.size` and separated-deployment key gates, the
  `standalone.dram.size` no-op trap, the version pins, and the swallowed-save
  → collective-desync family (#14106).
- `references/dsa-sparse-attention/index.md` — DeepSeek-V3.2-Exp / DSA (SFA) reports.
  `indexer-cache-scatter-fault.md`: a device fault at the indexer's
  `npu_scatter_nd_update_` (MTE DDR out of range, one `pc start`, mixed-batch
  trigger) is a host slicing bug — the verified v0.11.0rc1 `c5fe179` slice mismatch,
  the concurrency A/B, the fix, and the v0.13.0rc2 DSA-CP re-derivation (#4296).
  `concurrency-crash-with-no-log.md`: "多并发时崩溃" with nothing pasted — the config
  diff against `tests/e2e/**` and the tutorial, the phase-blind
  `MLAPO_MAX_SUPPORTED_TOKENS = 1024` gate, the maintainer-filed #4996 family and the
  `--max-num-seqs 1` + MTP arithmetic (#6240).
- `references/hccl-and-hardware/index.md` — HCCL, ACL/PTA and NPU device faults
  (lane C): the error-report timestamp gate first (`error-report-timestamps.md`,
  #9183), then `collective-error-chain.md`, then `device-health-checks.md` (the
  recorded answer was a physical HCCS link fault on 6 of 8 cards).
  `rpc-watchdog-timeouts.md`: a bare `TimeoutError` from `multiproc_executor.py`
  / `shm_broadcast.py` is vLLM's own RPC watchdog, not a collective (#1865).
  `control-plane-port-binding.md`: `Communication_Error_Bind_IP_Port (EI0019)` /
  `port <n> have already been bound` above 65535 — HCCL's control-plane band, a
  finding only in-window (#12487).
- `references/runtime-memory/index.md` — memory, both phases: healthy service that
  later died on HBM ("显存不释放", `NPU out of memory`), or an allocation that failed
  during KV-cache/weight placement. The capacity/growth/dead-key split, the
  `PYTORCH_NPU_ALLOC_CONF` injection gate, why `torchair_graph_config` /
  `ascend_scheduler_config` are accepted and ignored, and the OOM/leak precedent
  family (#12692, #13185, #13248, #13240, #3198, #2650; worked for #16022).
  `kv-init-and-format-cast-alloc.md` — the `EL0004`/`207001`/`507013`
  `npu_format_cast` signature (#4812). `host-ram-and-per-rank-rss.md` — "the machine
  crashed" with a `top`/`ps` paste and no NPU error (#3902), plus
  `scripts/host-ram-check.sh`.
- `references/soc-platform-paths/index.md` — SoC-conditional paths, 310P first:
  `is_310p()` gating, the documented 300I/310P status, the seven platform gates,
  the NZ/`FRACTAL_NZ` conversions and the 300I launch environment
  (`atlas-300i-310p.md`, vllm-ascend v0.11.0rc3 + vllm v0.11.0).
  `device-type-build-vs-runtime.md` takes engine start dying at
  `check_ascend_device_type()`: build stamp vs live `soc_version`, the
  a2/a3/310p-only wheel matrix against the in-tree `Dockerfile.a5`, the
  don't-patch-the-guard argument, the A2/A3-only support matrix (#12141,
  v0.21.0rc1). `kernel-package-vs-card.md` takes the built-in-op `EZ9999 Parse
  dynamic kernel config fail` class ("does image <tag> support <card>?"): the
  docs' supported list, the image's CANN base and one standalone op probe (#1696,
  v0.7.3).
- `references/observability-and-logs/index.md` — reports about log text rather
  than a crash ("the flag prints nothing", a missing startup stage).
  `request-output-logging-flags.md` is the verified gate table for
  `--enable-log-requests` / `--enable-log-outputs` / `VLLM_LOGGING_LEVEL` /
  `--max-log-len` at vllm v0.18.0 + vllm-ascend v0.18.0 (#10985), including the
  endpoint that can never log outputs and the `inspect.getsourcefile` check for
  the live patched serving generator. `online-profiling-endpoints.md` takes the
  *missing route*: `POST /start_profile` → `404` after `--profiler-config`, attached
  only when the serving process's `args.profiler_config.profiler` is set
  (`serve/profile/api_router.py` `attach_router`), so the answer is that gate plus
  one `openapi.json` request; a worker-side miss is a 500, never a 404 (#9583).
- `references/serving-frontend-faults/index.md` — a Python exception that
  surfaces in the OpenAI server process while the report is filed as an Ascend
  problem, and whose traceback has **no** `vllm_ascend` frame: the owner is
  upstream vLLM or an installed Python dependency, so the answer is an ownership
  verdict plus a client-side A/B, and the artefacts to ask for are the serve
  command, the request body and the client concurrency — never a plog or
  `npu-smi`. `tokenizer-already-borrowed.md` is the worked #7197 case:
  `RuntimeError: Already borrowed` at `Hermes2ProToolParser.__init__` inside
  `_preprocess_chat` (vllm v0.16.0) — one HF fast tokenizer shared by the event
  loop and the `AsyncMicrobatchTokenizer` worker thread, upstream's single retry
  for the same exception, and the serialise/retry ladder.
- `references/custom-ops-runtime-faults/index.md` — aclnn/AscendC custom-op
  device faults: reading the device dump, mapping the operator to `csrc/<op>/`,
  tiling-mode tables; the dump names operator and error class, not shape.
  `attention-op-numeric-errors.md` is the silent-numerics half: an op that runs
  and returns a wrong tensor (uniform average / softmax dropped) with a
  framework-free repro — the caller holding the uncovered kwargs combination, the
  counter-example in-tree, the zero-stub unit test, the three fix sites and the
  cheap discriminators (#13699, v0.23.0rc1). `op-not-registered-for-soc.md` is
  the other half: `EZ1001 Get regInfo failed` / `does not support opType` is a
  build/install SoC fault before any launch — the SOC_VERSION, `_cann_ops_custom`
  and image-tag decisions (#11519, v0.18.0) — while a **built-in** op takes the
  kernel-package route above.
- `references/accuracy-benchmarks/index.md` — score/accuracy disagreements between
  harnesses, images or runs; the decisive harness-vs-engine experiments. Pages, each
  with pinned refs and ablations: `reasoning-output-routing.md` (thinking inside
  `content`, `reasoning_tokens: 0`), `early-eos-short-output.md`,
  `garbled-long-context-output.md`, `short-prompt-corruption.md`,
  `hybrid-block-size-corruption.md` (a `--block-size` diff on a hybrid GDN model),
  `qwen3-next-gdn-numeric-path.md`, `vl-vision-tower-degeneracy.md` (`!!!!` runs),
  `long-output-repetition.md` (long completions that repeat),
  `quantized-checkpoint-conversion.md`, `harness-vs-engine-attribution.md`. When a
  page's pins equal `SOURCES.json`'s refs, cite its `path:line` as they stand.
- `references/mixed-batch-splits/index.md` — a batch holding decodes **and** prefills
  that faults inside an attention backend: the two counting units, the unit of each
  tensor (`query`/`output` tokens; `block_tables`/`seq_lens` requests;
  `query_start_loc` cumulative tokens, rebased to 0), why spec decode is the
  multiplier, and why the reported line is often the decode half's `.npu()`.
  `fa3-mixed-batch-split.md` is the worked #17991 case (v0.23.0rc1 + FA3 NPU
  v0.4.2): `fa3_v1.py:132-136`, the corrected lines, the sanctioned in-tree pattern
  (`attention_v1.py:1843-1845,1864`, `gdn_attn_builder.py:652-653`), the two
  secondary defects. Audit a tree with `scripts/batch-split-audit.sh` (2 hits at
  #17991, clean at #8724).
- `references/tool-call-streaming/index.md` — the report is about a **tool call**
  rather than prose: a client-side `Unterminated string` / `Expecting ',' delimiter`
  while it assembles `function.arguments`, arguments "被截断"/格式错误/重复输出, or
  `stream: true` returning only CoT. The three layers (upstream frontend delta
  assembly, the vllm-ascend sampler, the model stopping early) and the one request
  that separates them. `streamed-arguments-audit.md` is the procedure: obtain the
  raw SSE, run `scripts/sse-toolcall-audit.sh <raw-sse|->`, route on its three
  outcomes (parses = client; tail missing = stream ended early, split with
  `return_token_ids: true`; unbalanced middle = generation), then the ordered
  ablations and the ownership table (#7580's tail lost, #8122's sample valid).
  `mtp-block-verify-acceptance.md` carries the `using_block_verify` branch with
  its version gate, cumulative-product acceptance, greedy exemption, the
  `parse_output` placeholder filter and the `n=2` / `temperature: 0` ablations.
- `scripts/abi-symbol-check.sh <paste|-> [...]` — classifies a pasted native
  `undefined symbol` against the two ATB/NNAL builds: demangles every name
  (`c++filt`), labels the `std::string` ABI (`Ss` old vs `NSt7__cxx11` new), notes
  line-wrapped fragments, reports which `cxx_abi_*` root the env dump selects,
  warns when it is really the missing-library or old-`GLIBCXX` class, then prints
  the verdict, the three decisive `nm` commands and the confirmed `cxx_abi_0`
  export. Verified on #1144 (`AsdOps::Ops::GetOperationByName`,
  `ATB_HOME_PATH=…/cxx_abi_1`).
- `scripts/sse-toolcall-audit.sh <capture|-> [more files]` — concatenates every
  `choices[i].delta.tool_calls[j].function.arguments` fragment in arrival order and
  prints per index the fragment count, empty fragments, `{ } [ ] "` balance, the
  assembled head/tail and `json.loads`, the `finish_reason` chunks, then a verdict
  (client / tail lost / middle corrupted). Tolerates HTML-Excel wrappers, console
  prefixes and `[REDACTED]` placeholders. Verified on #7580 and #8122.
- `references/spec-decode-mtp/index.md` — symptoms the speculative-decoding path
  owns while the engine is otherwise healthy. `acceptance-collapse.md` — `Accepted:
  0` with `Drafted: N`, the ordered MTP-off ablation, async-spec-decode
  `num_computed_tokens` drift, the `mamba_block_size` prefix-caching gate (#10495).
  `empty-first-response.md` — `content: ""` / `completion_tokens: 1` /
  `finish_reason: "stop"` on the first request: restart reproduction, sibling
  tutorial diff, the refuted scheduler-padding theory next to the live MTP
  padded-drafter gate, draft-side ablations and `min_tokens` (#9355).
  `drafter-count-copy-device-fault.md` — a device dump surfacing at the drafter's
  valid-count D2H copy (`model_runner_v1.py:1393`): the frame as sync point *and*
  as proof of `--async-scheduling` + spec decode + padded drafter batch, the
  rank-size-versus-group arithmetic, the ordered ablations, and the DeepSeek-V4
  `VLLM_ASCEND_APPLY_DSV4_PATCH` / `deepseek_mtp` remap gates (#10546).