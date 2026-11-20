---
name: vllm-ascend-triage
description: 'Triage vllm-ascend deploy, HCCL/PD/EPD, image, OOM, accuracy'
---

# vLLM-Ascend issue, deployment and support triage

## Problem class this covers

Four entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter.

- **Issue report** (usually Chinese or English, template-shaped: environment dump
  + logs + narrative) about something that went wrong on an Ascend NPU host: a
  device-side operator crash, an accuracy/score disagreement, a garbled long-input
  output, a long-run OOM / "VRAM not released" report, a performance claim, or a
  bare `python -c "import <pkg>"` that aborts inside a library the image shipped.
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
- **PD / EPD disaggregation**: the deployment is up but the P and D sides cannot
  move KV (`HIXL Connect failed | HIXL Error Code: 103900`, HCCL
  `RanktableCheck` / `'device_ip' is not set correctly`, `wait_for_save failed
  ... skipping save`, a recipe that only works with `hostNetwork: true`), or the
  transfer succeeds and the decode node emits wrong or alternating output on a
  hybrid linear-attention model. Transport identity and rank tables are a
  topology question, not a connector bug — `references/pd-disaggregation/index.md`.

Recorded uses (issue: version, symptom, track): #8944 v0.19.1rc1
`aclnnAddRmsNormBias` device fault; #10068 evalscope vs ais-bench MMLU-Pro gap;
#3543 v0.11.0rc0 silent worker death; #7477 v0.17.0rc1 HCCL allreduce ending in
an HCCS link fault; #11661 v0.22.1rc1 missing `attention_type` on a patched
decoder layer; #8409 v0.13.0 `model_type` not recognised (version gate); #16015
v0.26.0rc1 GLM-5.3 w8a8c8, PD disaggregation + sparse sfa c8 + DCP=16, 400K+
prompt → 乱码; #5572 v0.12.0rc1 Qwen3-VL-30B-A3B PD disaggregation, Mooncake
connector `AttributeError` on a nested config; #3389 v0.11.0rc0 Qwen3-VL-235B
TP=16 whose posted cause ("insufficient available streams") is canned text around
a capture-phase HCCL timeout; #10503 v0.20.2rc1 Kimi-K2.5-w4a8 on 2×A2 with
tp8/dp2/EP, 乱码 on a 3.6K-token prompt; #17357 v0.26.0rc1-a3 DeepSeek-V4-Pro
w4a8 `SUSPECT REMOTE ERROR`/507057 surfacing at a compile-phase pattern-registration
sample op (lane C, reporting rank unrelated to the faulting chip); #5585 msmodelslim
W4A4 Qwen3-32B with one blown-up decoder layer (converter-owned, conversion phase);
#5609 v0.12.0rc1 `corrupted size vs. prev_size` abort on
`from mooncake.store import MooncakeDistributedStore` in the official image (image-owned
native module, before any vllm-ascend code); #11863 v0.21.0rc1 dense Qwen3.6-27B
launched from the 27B tutorial's multi-node PD recipe → upstream non-MoE external-DP
gate (usage error plus a real doc bug); #16022 v0.23.0rc1 4×910B4 Qwen3.6-27B
long-run HBM growth then crash, with `torchair_graph_config`/`ascend_scheduler_config`
keys that never engaged; #6784 v0.14.0rc1-a3-openeuler LongCat-Flash
`AttributeError: 'ZeroExpertFusedMoE' object has no attribute 'select_experts'`
(an upstream MoE refactor outside vllm-ascend's name-keyed OOT map, with the
failing file absent from the pinned upstream tree); #11519 v0.18.0 Step-3.5 on
A3 `AclNN_Parameter_Error(EZ1001) Get regInfo failed … socVersion
[ascend910_93] does not support opType [AddRmsNormBias]` (custom-op package
built/installed for another SoC — same op as #8944, opposite failure class);
#14506 v0.18.0rc1 single node, 8×910B3 with every RoCE link DOWN, 1P1D in two
non-hostNetwork Pods → `HIXL Connect failed | 103900` because two Pod IPs became
two HCCL servers (transport topology, not a connector bug); #12339 v0.22.1rc1
Qwen3.5-397B-W8A8-MXFP8 on Ascend950, `MooncakeConnectorV1` PD without MTP,
alternating good/bad answers on a hybrid GDN model (the run spent its whole
output budget on source walks and delivered no answer).

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and obey the delivery gates.** The
   completion budget is 16384 tokens for the whole session — reasoning, answer
   text and tool-call arguments together, never replenished — and the mechanism
   of loss is per-turn reasoning volume (~50K characters per session, ~4K for the
   answer). The newest runs confirm it: #6784 spent 45.8K characters of reasoning
   over 10 turns and emitted **no answer at all**; #11519 spent 43.3K and
   delivered 1.0K characters, cut mid-sentence; #12339 spent 49K over 20 turns on
   four competing mechanisms and shipped nothing. So these are hard stops:
   - **G1 — 3rd assistant turn:** message text with the direct answer to the
     reporter's literal question plus the track/lane classification. A pasted
     traceback, one config value or one registry listing is enough; do not hold
     the answer for the mechanism.
   - **G2 — 8th turn or 20K cumulative output characters:** the full ranked
     answer (each cause with its one check, cheapest action, version scope,
     missing artefacts). Afterwards a tool call is allowed only to confirm or
     correct a named claim: one per turn, ≤1,500 characters of reasoning.
   - **G3 — 25K characters or the 12th turn:** answer text only; no reasoning,
     no tool call, no "one more thing".
   - **Reasoning is a scratchpad, ≤1,200 characters per turn:** what the last
     result proved, the single next check, go. If a block starts enumerating the
     answer's sections, or says "I have enough" / "that is the crux" / "let me
     write the answer", that block is the answer — emit it as message text in
     that turn. A "now let me…" plan line delivers neither answer nor evidence;
     every assistant message is a tool call or answer text.
   - **A new hypothesis that needs a new code area is not this session's job.**
     The expensive turn is the "one more angle" one, not the first ones: #14506
     delivered its 6K-character answer only after a final reasoning block of
     12.3K characters that was half the session. Put the hypothesis in the answer
     as an open question with the one command that decides it. For an
     accuracy/corruption or topology report, a ranked list plus an ablation matrix
     already *is* the full answer — the two newest runs were not missing facts,
     only the message.
   - **`/workspace/scratch/findings.md` is the answer draft, not an extra step:**
     write it at G2 *in place of* the reasoning block, and skip it entirely when
     the session is near the cap. Composing it late is what cost #11519 its
     ranked causes.
   - **A missing or mismatched artefact is itself the finding** — an empty
     environment block, a traceback file absent from the pinned tree, a version
     the snapshot does not carry. State the gap, give the one command that closes
     it on the reporter's host, stop investigating. Both recorded runs above
     burned their whole budget resolving such a gap.
   - **Truncation eats the tail, not the head.** Put each action directly after
     its cause and keep the closing scope / open-questions block to 3-5 lines, so
     a capped answer still ends in something usable.
   - Recorded per-run accounting and the reasoning patterns behind the caps:
     `references/evidence-workspace/delivery-budget.md`.
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
   text rather than assuming what it says.
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
   - Never list a history or attachments directory: doing so on one recorded
     case returned ~3k filenames (~24 KB) for no information. Search by token
     (`grep -rl`, `rg -l`) or with `scripts/history-titles.sh`, and read
     `SOURCES.json` through `scripts/sources-summary.sh` instead of dumping the
     whole file. `cat SOURCES.json` is 20-54 KB and both newest runs paid it
     (#12339 34 KB, #14506 54 KB) although the script prints the same version
     contract plus the `unavailable` repos in ~10 lines. It is the right first
     shell call; `history-titles.sh` prints the matching body line when the regex
     hit the body rather than the title, so one call separates a real family from
     a lexical accident.
8. **The prior-month corpus is not evidence of absence.** `history/*.md` and
   lexical issue search cover only earlier issues; a current issue is never in
   them. Make one grep for the distinctive token (op name, error string, dataset
   name), read 2-3 hits, then stop: precedent supports a hypothesis, it does not
   replace reading the source at the pinned commit.
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
    output budgets on ABI/RPATH/`glog` archaeology and delivered nothing, having
    already found the jemalloc omission and ranked it low. Route:
    `references/image-dependency-faults/index.md`, then
    `references/image-dependency-faults/native-import-abort.md`.

## Classify the evidence

| Signature in the report | Track | Go to |
|---|---|---|
| Startup log: generic `WorkerProc initialization failed due to an exception in a background process`, **no** per-rank worker traceback; optional `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!`; long silence before failure | Lane A: worker died without reporting | `references/startup-failures/silent-worker-exit.md`, then `references/startup-failures/max-model-len-budget.md` |
| Startup log: per-rank `WorkerProc failed to start.` / `WorkerProc hit an exception.` plus a traceback with rank prefix | Lane B: worker raised a Python error | `references/startup-failures/environment-correspondence.md` (diff against the known-good tutorial/CI config) |
| The traceback ends in `AttributeError: '<X>Config' object has no attribute '<attr>'` with a frame in `vllm_ascend/distributed/*connector*.py`, reached from `ensure_kv_transfer_initialized` (usually a VL/MoE model under PD disaggregation) | Lane B: connector reads a nested config attribute | `references/startup-failures/connector-config-attributes.md` |
| P/D will not connect although both engines started: `HIXL Connect failed | HIXL Error Code: 103900` (or `503900`), older `LLM_LINK_FAILED` from `llmdatadist_c_mgr_connector`, `wait_for_save failed ... skipping save`, HCCL `RanktableCheck` / `'device_ip' is not set correctly, must be set when multi Server`, `get dev list error:serverId[...]`, a recipe that only works with `hostNetwork: true` / `--net=host` | PD transport topology: two addresses on one node became two HCCL servers (the engine id *is* the rank-table `server_id`); look at the rank table and the plog, not the Python traceback | `references/pd-disaggregation/transport-topology.md` |
| PD deployment that connects and then corrupts on a **hybrid** model (Qwen3.5/Qwen3.6 GDN, DeepSeek-V4, GLM): alternating good/bad answers, empty or wrong first decoded token, works with MTP but not without | PD hybrid-state transfer: connector choice plus the `mamba_cache_mode='align'` gate only the KV-store connectors satisfy | `references/pd-disaggregation/hybrid-state-transfer.md` |
| `ValueError: Non-MoE models do not support external data parallel mode`, or any `ValueError`/`AssertionError` raised from `vllm/engine/arg_utils.py` that names a parallel mode or flag (`data_parallel_*`, `nnodes`, `node_rank`), with no `WorkerProc` line in the log | Startup argument validation in upstream vLLM — launch-command fix, not a defect | `references/startup-failures/arg-validation-errors.md` |
| `value error, the checkpoint you are trying to load has model type '<mt>' but Transformers does not recognize this architecture`; `Model architectures [...] are not supported`; "can version V deploy model M?" | Model-support / version gate | `references/startup-failures/model-type-not-recognized.md` |
| Python `AttributeError`/`TypeError` in a frame under `vllm_ascend/{ops,patch,models}` where the class subclasses or re-implements an upstream vLLM/transformers class — e.g. `'AscendQwen2DecoderLayer' object has no attribute 'attention_type'`, `unexpected keyword argument`, `cannot import name` | Downstream patch drift | `references/upstream-downstream-drift/index.md` |
| The same `AttributeError` raised **inside an upstream vllm file** on a class vllm-ascend is meant to replace — `'ZeroExpertFusedMoE' object has no attribute 'select_experts'` — or the reporter asks "did it ever reach vllm-ascend's ops?" | Downstream registry drift: the Ascend replacement is keyed by class *name* (`CustomOp.register_oot`), so an upstream refactor that adds a class silently keeps upstream code | `references/upstream-downstream-drift/oot-class-registration-drift.md` |
| `HcclAllreduce`, `hcclRet -> 4`, `ERR00100`/`ERR01100`, `acl api failed`, `AllReduceOpbasePipeline`, `RuntimeError: SUSPECT REMOTE ERROR` / `507057`, `EZ9999` device dump, hang in `determine_available_memory` / `profile_run` | Lane C: collective / device error (a `profile_run` frame ending in an `npu_*` op on a 2×4 tensor is a compile-phase sync point, not an op bug) | `references/hccl-and-hardware/index.md` |
| Startup log reached graph capture: `Starting to capture ACL graphs for cases: [...]`, `Capturing ACL graphs (...): 0/N` frozen, `ACLgraph sizes capture fail`, `NPUGraph.cpp` "Waiting for pending NCCL work" | Lane C during capture — or whatever the appended inner error says; the `insufficient available streams` advice is canned text | `references/startup-failures/acl-graph-capture.md` |
| Device dump naming an operator: `call aclnnX failed`, `AclNN_Runtime_Error`, fftsplus aivector error, MTE/DDR "out of range" | Custom-op runtime fault | `references/custom-ops-runtime-faults/index.md` |
| `AclNN_Parameter_Error(EZ1001)`, `Get regInfo failed`, `binary_info_config.json of socVersion [...] does not support opType [...]` — aclnn cannot resolve the op for the SoC before any launch | Custom-op **build/install** fault: package built for another SoC, stale vendor package, wrong image tag | `references/custom-ops-runtime-faults/op-not-registered-for-soc.md` |
| "score with tool A is N points below tool B / another image / the vendor number", accuracy regression after a change | Accuracy/score discrepancy | `references/accuracy-benchmarks/index.md` |
| Per-layer `MSE loss` / epoch table printed while converting a checkpoint with msmodelslim (W4A4/W8A8), one layer far above its neighbours, or "the converted checkpoint answers garbage" with no engine-side failure | Quantization-conversion phase: converter-owned, before the engine starts | `references/accuracy-benchmarks/quantized-checkpoint-conversion.md` |
| A long prompt (100K+) answered as 乱码 / garbage / repeated / empty text while short prompts are fine, with sparse c8 attention, DCP/PCP, PD disaggregation or KV offload enabled | Generation corruption (accuracy track, different evidence) | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| 乱码 / token soup on a few-thousand-token prompt, process alive, no traceback, `finish_reason: length`, often MoE + EP across nodes or a quantised weight set | Generation corruption, short-prompt profile | `references/accuracy-benchmarks/short-prompt-corruption.md` |
| The server served traffic then died on memory: "运行一段时间后显存不释放最后崩溃", `NPU out of memory` / `NPUWorkspaceAllocator` / `Memory_Allocation_Failure(EL0004)`, `npu-smi` pinned at 100 %, a P node holding KV the D node never takes | Runtime memory after a healthy period: capacity vs growth vs a config key that never engaged (contrast the Lane C row, where the OOM happens during `profile_run` / `determine_available_memory` before the server is ready) | `references/runtime-memory/index.md` |
| `python -c "import <pkg>"` or the first use of an image-installed native module aborts the interpreter: `corrupted size vs. prev_size`, `free(): invalid pointer`, `double free or corruption`, `munmap_chunk(): invalid pointer`, `Aborted (core dumped)`, no Python frame, no NPU text | Image / bundled native module (pre-engine, not vllm-ascend code) | `references/image-dependency-faults/index.md` |
| Compiler or missing-header error while installing/building, a performance claim, or any report with no matching track | General sequence below | skip the topical routes |

## Procedure

1. List `/workspace`, then run `bash <skill_root>/scripts/sources-summary.sh` for
   the `SOURCES.json` version contract (`skill_root` is the path `skill_view`
   reports); note which repos/refs are present and which are partial exports. The
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
   - PD / EPD transport complaint: read the rank table with
     `bash <skill_root>/scripts/ranktable-check.sh <table-or-log>` before any
     connector reading — a declared `server_count > 1` (or more than one distinct
     `server_id`) decides the whole report (multi-server view, `device_ip` + RoCE
     required). Then the plog grep set in
     `references/pd-disaggregation/transport-topology.md`.
   - Runtime memory after a healthy period: take the two-sample split (capacity
     vs growth) and `grep -m1 'Set PYTORCH_NPU_ALLOC_CONF' <serve log>` first, and
     check `additional_config` key liveness (`rg -n '<key>' vllm_ascend/`) before
     any leak theory (`references/runtime-memory/hbm-growth-and-oom.md`).
   - Accuracy/corruption: establish whether both numbers came from the same
     server, weights and serve flags before choosing a culprit
     (`references/accuracy-benchmarks/index.md`); for a garbled prompt, get the
     ablation matrix (`references/accuracy-benchmarks/garbled-long-context-output.md`).
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
  crash is real. One sentence, before any analysis.
- Then the one-line classification: lane A/B/C and the phase that failed (load,
  memory profiling, graph capture, first collective); or, for a runtime report,
  the primary cause plus its track.
- Ranked causes, each with the evidence you have, the exact check that confirms
  or refutes it, and its confidence label (reporter-confirmed / source-verified
  at commit X / hypothesis).
- An evidence table (`path:line`) for code claims, with precedent issue ids
  quoted as `#<id>: <one-line fact>`.
- The cheapest action first: an explicit `--max-model-len`, a re-run with
  `--enforce-eager` / `ASCEND_LAUNCH_BLOCKING=1`, `asys health -d=$i`, a version
  correction — before any patch proposal.
- Version and hardware scope of the conclusion, plus what to send back (full
  per-rank log, plog, tiling I-line, raw request/response dumps, `npu-smi info`,
  `asys health` output, `dmesg` OOM lines, both PD serve commands).
- What you could not verify locally and which questions remain open, stated as
  such.

## Support entry points

- `references/evidence-workspace/index.md` — read first in an unfamiliar case:
  case layout, `SOURCES.json` fields and limits, history search, OCR/attachment
  caveat, what the snapshot does and does not contain.
  `references/evidence-workspace/delivery-budget.md` holds the recorded
  per-session token accounting and the G1/G2/G3 delivery gates of rule 1.
- `scripts/history-titles.sh '<regex>' [root] [limit]` — precedent index: one
  `<file> :: <title>` line per matching history issue, so a whole issue family is
  triaged without reading bodies. Run it as
  `bash <skill_root>/scripts/history-titles.sh '<regex>'`, where `skill_root` is
  the path `skill_view` reports.
- `scripts/sources-summary.sh [SOURCES.json ...]` — the version contract, one
  line per repo with coverage and `runtime_verified` flags, plus the
  `unavailable` repos, so "is `cann/hixl` in this snapshot?" is answered without
  dumping 20-54 KB of JSON.
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
- `references/image-dependency-faults/index.md` — faults the image or its build
  recipe owns, before any vllm-ascend code: an interpreter abort inside a bundled
  native module, a missing library the tutorials prescribe, a stale copied
  installer. `references/image-dependency-faults/native-import-abort.md` is the
  worked #5609 case (mooncake `corrupted size vs. prev_size`; toolkit-copy
  shadowing, jemalloc preload, version that works).
- `references/startup-failures/index.md` — engine/worker never becomes ready
  (lanes A/B): silent worker exit; `max_model_len` and memory budget
  (reporter-confirmed for #3543, but per-card HBM decides whether the default
  fits); environment/version correspondence and known-good command diffing;
  `references/startup-failures/model-type-not-recognized.md` for "recognised
  architecture" / support-matrix questions (three registry gates,
  first-supporting-release precedent);
  `references/startup-failures/connector-config-attributes.md` for the
  `AttributeError` on a nested multimodal config inside a PD KV connector
  (verified for #5572, v0.12.0rc1);
  `references/startup-failures/arg-validation-errors.md` for a `ValueError` that
  names a flag combination instead of a device fault — the non-MoE external-DP
  gate reached by following a MoE-shaped DP tutorial with a dense model
  (verified for #11863, v0.21.0rc1 + vllm v0.21.0/v0.24.0);
  `references/startup-failures/acl-graph-capture.md` for a failure or stall
  inside ACL graph capture, where the logged advice is canned and the
  capture-size set is usually already the computed maximum (#3389, v0.11.0rc0).
- `references/upstream-downstream-drift/index.md` — Python attribute/signature
  errors raised when a vllm-ascend module subclasses or re-implements an upstream
  vLLM/transformers class and the installed library moved on, **including when the
  failing frame is an upstream vllm file** because the Ascend replacement is
  keyed by class name. Scope: the attribute exists upstream; not a device-op or
  collective fault. Worked cases: #11661 `attention_type`
  (`patched-forward-api-drift.md`); #6784 `ZeroExpertFusedMoE` with no
  `select_experts` (`oot-class-registration-drift.md`, which also carries where
  the real vllm↔vllm-ascend pairing is recorded).
- `references/pd-disaggregation/index.md` — PD/EPD with the engine up but KV not
  moving, or moved and wrong: `transport-topology.md` for `HIXL Connect failed |
  103900` / `RanktableCheck` / `device_ip` / non-hostNetwork Pods (engine id =
  HCCL `server_id`, plog grep set, `HCCL_INTRA_ROCE_ENABLE` /
  `HCCL_INTRA_PCIE_ENABLE` pairing, port bands, ownership: LMCache-Ascend or
  `cann/hixl`, not the vllm-ascend shim; #14506); `hybrid-state-transfer.md` for
  alternating corruption on GDN/Mamba models — connector matrix
  (`MooncakeConnectorV1` vs `MooncakeHybridConnector` vs
  `MooncakeLayerwiseConnector` vs `AscendStoreConnector`), the
  `mamba_cache_mode='align'` gate in `patch_mamba_config.py` and the runner paths
  it skips, and the ablation matrix (#12339).
- `references/hccl-and-hardware/index.md` — HCCL, ACL/PTA and NPU device faults
  (lane C): collective error chain then per-device health checks. Scope:
  failures whose text names a collective or the runtime stack, not a
  Python-level cause; the recorded answer was a physical HCCS link fault on 6 of
  8 cards.
- `references/runtime-memory/index.md` — the service was healthy and later died on
  HBM ("显存不释放", `NPU out of memory`): the capacity/growth/dead-key split, the
  `PYTORCH_NPU_ALLOC_CONF` injection gate that `max_split_size_mb` silently
  disables, the KV-pool and profile-peak arithmetic against the tutorial's
  hardware floor, why `torchair_graph_config` / `ascend_scheduler_config` are
  accepted and ignored, and the recorded OOM/leak precedent family (#12692,
  #13185, #13248, #13240, #3198, #2650; worked for #16022 at v0.23.0rc1).
- `references/custom-ops-runtime-faults/index.md` — aclnn/AscendC custom-op
  device faults: reading the device dump, mapping the operator to `csrc/<op>/`,
  tiling-mode tables. The dump names the operator and error class, not the
  shape; custom ops are often behind a runtime switch.
  `references/custom-ops-runtime-faults/op-not-registered-for-soc.md` is the
  other half: `EZ1001 Get regInfo failed` / `does not support opType` is a
  build/install SoC fault raised before any launch, and that page shows where the
  SOC_VERSION, `_cann_ops_custom` and image-tag decisions are made (#11519,
  v0.18.0).
- `references/accuracy-benchmarks/index.md` — score/accuracy disagreements
  between harnesses, images or runs: thinking/reasoning output routing, where
  the project's own baselines live, and the decisive harness-vs-engine
  experiments. When the project has no baseline for the reported dataset, the
  reporter's harness configs are the primary evidence.
  `references/accuracy-benchmarks/quantized-checkpoint-conversion.md` covers the
  conversion phase: per-layer loss tables printed by msmodelslim, the two documented
  gates (engine version and the msit branch, which differ between doc pages), the
  published-checkpoint split test, and why exactly one bad layer means converter
  numerics rather than checkpoint layout.
  `references/accuracy-benchmarks/garbled-long-context-output.md` covers the
  corruption end: long-input 乱码 reports under sparse c8 attention + DCP + PD
  disaggregation, the feature-ablation matrix that decides ownership, and the
  tested envelope (#16015, #12905, #14320).
  `references/accuracy-benchmarks/short-prompt-corruption.md` covers corruption
  on a few-thousand-token prompt: test the prompt shape first, then the
  EP/FLASHCOMM1/prefix-cache/graph-mode/weight-identity rows (#10503,
  v0.20.2rc1).
