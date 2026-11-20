---
name: vllm-ascend-triage
description: 'Triage vllm-ascend deploy, HCCL, support, accuracy faults'
---

# vLLM-Ascend issue, deployment and support triage

## Problem class this covers

Three entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter.

- **Issue report** (usually Chinese or English, template-shaped: environment dump
  + logs + narrative) about something that went wrong on an Ascend NPU host: a
  device-side operator crash, an accuracy/score disagreement, a garbled long-input
  output, or a performance claim. When the task is just "explain/answer this
  issue", the answer is still expected to be evidentiary, not a summary of the
  report.
- **Deployment failure**: a vLLM / vLLM-Ascend deployment fails to come up or
  fails during engine start — `EngineCore failed to start`, `WorkerProc
  initialization failed`, silent worker death, `HcclAllreduce` / allreduce
  errors, `ERR01100`/`ERR00100`, OOM, "the model failed to load", a Python
  `AttributeError`/`TypeError` raised from a vllm-ascend patched forward or a KV
  connector, or a deployment question filed with only a trimmed log.
- **Support / version question**: "does vllm-ascend <version> deploy <model>?",
  `… has model type '<mt>' but Transformers does not recognize this
  architecture`. This is not a bug hunt; it is a version gate with a one-line
  answer and a registry-level proof.

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
W4A4 Qwen3-32B with one blown-up decoder layer (converter-owned, conversion phase).

## Rules that apply whenever this skill is used

1. **Say the answer in a message, early, and budget for it.** The budget is
   16384 completion tokens for the whole session — reasoning, answer text and
   tool-call arguments together — and it is not replenished per turn. Before this
   step every recorded run ended with no reply: the final message was reasoning
   text with no answer and no tool call, the run stopped on `output_truncated`
   (`stop_reason: output_tokens`), one earlier run ended in a refusal, and the
   diagnoses reached in reasoning were sound. In this step #17357 and #5585
   delivered on the pattern below and still spent 15577 and 14542 tokens — nothing
   spare — while #3389 and #10503 hit the cap with the answer beginning only in
   the final turn: 44K and 36K characters of reasoning (78% and 68% of all
   output, single blocks of 6-9K) over 25 and 30 tool calls, so one headline plus
   classification (1.4K characters) was delivered and one closing action table
   (4.4K characters) was lost. The mechanism is per-turn reasoning volume. So:
   - **Reasoning is a scratchpad, not the deliverable.** One short paragraph per
     turn: what the last result proved, the single next check, go. Keep a block
     under ~1,200 characters; a 5-9K block costs a third of the session and
     usually restates hypotheses that belong in the answer.
   - **"I have enough", "that is the crux", "let me now write the answer" is the
     trigger to emit answer text in that same turn.** Recorded runs reached that
     state around call 15-20 and answered only at call 25-30.
   - Answer the reporter's literal question as soon as one artefact settles it
     ("does this crash exist on that image?" — their own traceback; "can 0.13.0
     deploy Qwen3.5?" — the support matrix or model registry at the pinned
     commit; "is the cause the log states real?" — the emitting code line). Write
     it as message text; a reasoning-only turn delivers nothing.
   - Emit in instalments: headline answer + lane/track classification + ranked
     causes with their confirming checks by ~10 tool calls, then re-emit the
     corrected version at the end. An earlier answer stays valid, a missing one
     does not, and everything after the first answer is refinement that must not
     delay it.
   - Every assistant message should be a tool call or answer text. A "now let
     me…" plan line spends budget and delivers neither.
   - Do not carry a running hypothesis list in reasoning — that list belongs in
     the answer, where it is delivered. Keep confirmed findings in
     `/workspace/scratch/findings.md` so a late turn composes the answer without
     re-reading sources; both successful runs wrote it and the answer as their
     last two calls, so write it at the checkpoint, not at the end.
   - Version archaeology ("which release dropped attribute X", "when did this
     model land") is optional depth: state it as an open question plus the one
     command the reporter can run, instead of resolving it first. Do not open a
     second line of inquiry once a complete answer exists.
   - Deployment failure modes are narrow: in the two older deployment cases the
     decisive evidence was a single omitted flag and one hardware health code,
     not a source-level defect. Cutting exploration is always cheaper than cutting
     the answer.
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
   recheck conditions when a constant, table or log line is version-specific.
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
     whole file.
8. **The prior-month corpus is not evidence of absence.** `history/*.md` and
   lexical issue search cover only earlier issues; a current issue is never in
   them. Make one grep for the distinctive token (op name, error string, dataset
   name), read 2-3 hits, then stop: precedent supports a hypothesis, it does not
   replace reading the source at the pinned commit.
9. **Closed components exist.** CANN/HCCL/PTA have closed implementations: the
   public exports (e.g. `cann/hccl`) do not contain the opbase pipeline sources.
   If a string from an HCCL/PTA error is not found in one pass over the owning
   public export, stop searching and use the plog, the version correspondence
   and the hardware path instead.
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

## Classify the evidence

| Signature in the report | Track | Go to |
|---|---|---|
| Startup log: generic `WorkerProc initialization failed due to an exception in a background process`, **no** per-rank worker traceback; optional `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!`; long silence before failure | Lane A: worker died without reporting | `references/startup-failures/silent-worker-exit.md`, then `references/startup-failures/max-model-len-budget.md` |
| Startup log: per-rank `WorkerProc failed to start.` / `WorkerProc hit an exception.` plus a traceback with rank prefix | Lane B: worker raised a Python error | `references/startup-failures/environment-correspondence.md` (diff against the known-good tutorial/CI config) |
| The traceback ends in `AttributeError: '<X>Config' object has no attribute '<attr>'` with a frame in `vllm_ascend/distributed/*connector*.py`, reached from `ensure_kv_transfer_initialized` (usually a VL/MoE model under PD disaggregation) | Lane B: connector reads a nested config attribute | `references/startup-failures/connector-config-attributes.md` |
| `value error, the checkpoint you are trying to load has model type '<mt>' but Transformers does not recognize this architecture`; `Model architectures [...] are not supported`; "can version V deploy model M?" | Model-support / version gate | `references/startup-failures/model-type-not-recognized.md` |
| Python `AttributeError`/`TypeError` in a frame under `vllm_ascend/{ops,patch,models}` where the class subclasses or re-implements an upstream vLLM/transformers class — e.g. `'AscendQwen2DecoderLayer' object has no attribute 'attention_type'`, `unexpected keyword argument`, `cannot import name` | Downstream patch drift | `references/upstream-downstream-drift/index.md` |
| `HcclAllreduce`, `hcclRet -> 4`, `ERR00100`/`ERR01100`, `acl api failed`, `AllReduceOpbasePipeline`, `RuntimeError: SUSPECT REMOTE ERROR` / `507057`, `EZ9999` device dump, hang in `determine_available_memory` / `profile_run` | Lane C: collective / device error (a `profile_run` frame ending in an `npu_*` op on a 2×4 tensor is a compile-phase sync point, not an op bug) | `references/hccl-and-hardware/index.md` |
| Startup log reached graph capture: `Starting to capture ACL graphs for cases: [...]`, `Capturing ACL graphs (...): 0/N` frozen, `ACLgraph sizes capture fail`, `NPUGraph.cpp` "Waiting for pending NCCL work" | Lane C during capture — or whatever the appended inner error says; the `insufficient available streams` advice is canned text | `references/startup-failures/acl-graph-capture.md` |
| Device dump naming an operator: `call aclnnX failed`, `AclNN_Runtime_Error`, fftsplus aivector error, MTE/DDR "out of range" | Custom-op runtime fault | `references/custom-ops-runtime-faults/index.md` |
| "score with tool A is N points below tool B / another image / the vendor number", accuracy regression after a change | Accuracy/score discrepancy | `references/accuracy-benchmarks/index.md` |
| Per-layer `MSE loss` / epoch table printed while converting a checkpoint with msmodelslim (W4A4/W8A8), one layer far above its neighbours, or "the converted checkpoint answers garbage" with no engine-side failure | Quantization-conversion phase: converter-owned, before the engine starts | `references/accuracy-benchmarks/quantized-checkpoint-conversion.md` |
| A long prompt (100K+) answered as 乱码 / garbage / repeated / empty text while short prompts are fine, with sparse c8 attention, DCP/PCP, PD disaggregation or KV offload enabled | Generation corruption (accuracy track, different evidence) | `references/accuracy-benchmarks/garbled-long-context-output.md` |
| 乱码 / token soup on a few-thousand-token prompt, process alive, no traceback, `finish_reason: length`, often MoE + EP across nodes or a quantised weight set | Generation corruption, short-prompt profile | `references/accuracy-benchmarks/short-prompt-corruption.md` |
| Install, build, import, performance claim, or any report with no matching track | General sequence below | skip the topical routes |

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
- `scripts/history-titles.sh '<regex>' [root] [limit]` — precedent index: one
  `<file> :: <title>` line per matching history issue, so a whole issue family is
  triaged without reading bodies. Run it as
  `bash <skill_root>/scripts/history-titles.sh '<regex>'`, where `skill_root` is
  the path `skill_view` reports.
- `scripts/sources-summary.sh [SOURCES.json ...]` — the version contract, one
  line per repo with coverage and `runtime_verified` flags.
- `scripts/aclgraph-size-budget.sh --layers N [--tp --dp --pcp --dcp --ep --moe
  --aiv --max-num-seqs --sizes]` — reproduces `update_aclgraph_sizes`: the
  maximum captureable batch sizes for a model/parallel configuration and the
  sampled subset, so a startup log's size list can be checked arithmetically
  (verified: 94 layers, TP=16 gives 6 sizes = `[512,408,304,192,88,1]`, #3389).
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
  `references/startup-failures/acl-graph-capture.md` for a failure or stall
  inside ACL graph capture, where the logged advice is canned and the
  capture-size set is usually already the computed maximum (#3389, v0.11.0rc0).
- `references/upstream-downstream-drift/index.md` — Python attribute/signature
  errors raised when a vllm-ascend module subclasses or re-implements an upstream
  vLLM/transformers class and the installed library moved on. Scope: the frame is
  in `vllm_ascend/{ops,patch,models}` and the attribute exists upstream; not a
  device-op or collective fault. Worked case: #11661 `attention_type`.
- `references/hccl-and-hardware/index.md` — HCCL, ACL/PTA and NPU device faults
  (lane C): collective error chain then per-device health checks. Scope:
  failures whose text names a collective or the runtime stack, not a
  Python-level cause; the recorded answer was a physical HCCS link fault on 6 of
  8 cards.
- `references/custom-ops-runtime-faults/index.md` — aclnn/AscendC custom-op
  device faults: reading the device dump, mapping the operator to `csrc/<op>/`,
  tiling-mode tables. The dump names the operator and error class, not the
  shape; custom ops are often behind a runtime switch.
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
