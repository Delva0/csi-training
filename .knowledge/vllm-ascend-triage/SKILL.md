---
name: vllm-ascend-triage
description: 'Triage vllm-ascend bugs, accuracy gaps, startup/HCCL faults'
---

# vLLM-Ascend issue and deployment triage

## Problem class this covers

Two entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter.

- **Issue report** (usually Chinese or English, template-shaped: environment dump
  + logs + narrative) about something that went wrong on an Ascend NPU host: a
  device-side operator crash, an accuracy/score disagreement, or a performance
  claim. When the task is just "explain/answer this issue", the answer is still
  expected to be evidentiary, not a summary of the report.
- **Deployment failure**: a vLLM / vLLM-Ascend deployment fails to come up or
  fails during engine start — `EngineCore failed to start`, `WorkerProc
  initialization failed`, silent worker death, `HcclAllreduce` / allreduce
  errors, `ERR01100`/`ERR00100`, OOM, "the model failed to load", or a
  deployment question filed with only a trimmed log.

Recorded uses: vllm-ascend #8944 (v0.19.1rc1, GLM-5.1, `aclnnAddRmsNormBias`
device fault), #10068 (evalscope vs ais-bench MMLU-Pro gap), #3543 (v0.11.0rc0 +
vllm v0.11.0rc3, 910B3, TP=4, Qwen3-VL-30B-A3B, silent worker death), #7477
(v0.17.0rc1 + vllm v0.17.0, driver 25.2.0, CANN 8.5.1, torch_npu 2.9.0, TP=4,
Qwen3.5-27B, HCCL allreduce ending in an HCCS link fault).

## Rules that apply whenever this skill is used

1. **Say the answer before exhausting the investigation.** Recorded runs of both
   classes failed by spending their whole output budget on exploration and then
   producing no answer at all, or by ending in a refusal. Deployment failure
   modes are narrow: in the two recorded deployment cases the decisive evidence
   was a single omitted flag and one hardware health code, not a source-level
   defect. Keep a running scratch note of confirmed findings. Deliver a first ranked answer once you can
   classify the failure and name the confirming check — for a pure issue report,
   once you have the owning code path — and keep refining afterwards. List open
   questions instead of omitting the answer; cutting exploration is always
   cheaper than cutting the answer.
2. **Classify before investigating.** For a startup log, place it in lane A/B/C
   (table below); for a runtime report, decide whether it is a device-op fault,
   an accuracy/score gap or a performance claim. The classification decides
   which code and which evidence even matter. Do not open a deep source walk
   when a config value, one re-run flag or a health tool can settle the question.
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
7. **Never dump whole files into context.** Use `rg -n -C2 <symbol>` and
   `sed -n 'A,Bp' file`. A 20-40 KB full-file read of an AscendC kernel header
   buys one fact and costs the budget for several conclusions. Reserve
   whole-file reads for files under ~150 lines.
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

## Classify the evidence

| Signature in the report | Track | Go to |
|---|---|---|
| Startup log: generic `WorkerProc initialization failed due to an exception in a background process`, **no** per-rank worker traceback; optional `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!`; long silence before failure | Lane A: worker died without reporting | `references/startup-failures/silent-worker-exit.md`, then `references/startup-failures/max-model-len-budget.md` |
| Startup log: per-rank `WorkerProc failed to start.` / `WorkerProc hit an exception.` plus a traceback with rank prefix | Lane B: worker raised a Python error | `references/startup-failures/environment-correspondence.md` (diff against the known-good tutorial/CI config) |
| `HcclAllreduce`, `hcclRet -> 4`, `ERR00100`/`ERR01100`, `acl api failed`, `AllReduceOpbasePipeline`, hang in `determine_available_memory` / `profile_run` | Lane C: collective / device error | `references/hccl-and-hardware/index.md` |
| Device dump naming an operator: `call aclnnX failed`, `AclNN_Runtime_Error`, fftsplus aivector error, MTE/DDR "out of range" | Custom-op runtime fault | `references/custom-ops-runtime-faults/index.md` |
| "score with tool A is N points below tool B / another image / the vendor number", accuracy regression after a change | Accuracy/score discrepancy | `references/accuracy-benchmarks/index.md` |
| Install, build, import, performance claim, or any report with no matching track | General sequence below | skip the topical routes |

## Procedure

1. `ls /workspace` and read `SOURCES.json`; note which repos/refs are present
   and which are partial exports. The layout differs per case (flat
   `/workspace/history/<issue>.md` vs `/workspace/history/<issue>/` +
   `/workspace/repos/<issue>/`); adapt after looking, do not assume.
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
   - Accuracy: establish whether both numbers came from the same server, weights
     and serve flags before choosing a culprit
     (`references/accuracy-benchmarks/index.md`).
4. Fix the version correspondence: image tag → vllm/vllm-ascend commits →
   torch_npu/CANN/driver
   (`references/startup-failures/environment-correspondence.md`). Claim "model
   unsupported" only after checking `vllm_ascend/models/__init__.py`
   registration and the upstream model file.
5. One grep of `history/` for the distinctive symptom token, 2-3 hits, then stop
   (rule 8).
6. Locate the owning code path at the pinned commit and walk it from the entry
   point (kernel launch site, harness config, model/feature flag) to the line
   that produces the symptom. Prefer computing the decision (a tiling branch, an
   aggregate metric) with a small `uv run --no-project python` snippet over
   reading code by eye.
7. Cross-check upstream vs downstream: vllm-ascend frequently re-implements or
   bypasses upstream vLLM (custom AscendC op vs `torch_npu` fallback, upstream
   reasoning parsers vs downstream template kwargs, docs baselines vs
   third-party harness). Record which layer owns the behaviour and for which
   version, plus what would invalidate the claim.
8. Write the answer in the shape below, then stop.

## Answer shape

- One-line classification: lane A/B/C and the phase that failed (load, memory
  profiling, graph capture, first collective); or, for a runtime report, the
  primary cause plus its track.
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
  `asys health` output, `dmesg` OOM lines).
- What you could not verify locally and which questions remain open, stated as
  such.

## Support entry points

- `references/evidence-workspace/index.md` — read first in an unfamiliar case:
  case layout, `SOURCES.json` fields and limits, history search, OCR/attachment
  caveat, what the snapshot does and does not contain.
- `references/startup-failures/index.md` — engine/worker never becomes ready
  (lanes A/B): silent worker exit; `max_model_len` and memory budget
  (reporter-confirmed for #3543, but per-card HBM decides whether the default
  fits); environment/version correspondence and known-good command diffing.
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
