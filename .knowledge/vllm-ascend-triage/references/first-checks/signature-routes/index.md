# Signature routes: log fragment → track → owning page

## Scope

The classification index: one row per report signature seen in this issue family.
It answers *where does this paste belong, and which page owns it* — the
identification step of `SKILL.md` procedure step 2, not the diagnosis. It used to
sit in `SKILL.md`; it lives here because the always-on rules and the routing
tables have to stay readable, while these rows are read per case, once.

Reach for it when the report's signature is **specific** — a log fragment, an
error string, a flag that did nothing, a title that names a mechanism. When the
shape is already clear at track level, the dispatch table in
`../../dispatch/classified-shapes.md` (reached from *Support entry points*) routes
straight to the topic, and the topic's own `index.md` routes to the page; that path
needs no row here. It is the expanded form of that same table: finer signatures,
the same topic pages. Rows describe **shapes seen in
recorded reports**, not causes; the owning page carries the mechanism, the pinned
`path:line` and the version scope.

The rows are split by the *shape's* home track so that a page stays scannable;
the split is an access path, not a second classification — a paste that is a
startup log in one rank and a serving error in another belongs to two pages, and
the most specific row wins.

## Common rules

- Inherit every rule in `SKILL.md`.
- Match the **left** cell against what the reporter pasted. Several rows can
  match one report: the most specific row wins, and a row that names an exact
  error string beats one that names a family.
- Read the **middle** cell before following the link. It carries the decisive
  local rule, the owner and often the ablation that separates the candidate
  causes — the link is where the detail lives, not the answer by itself.
- Where a row quotes a `path:line` or a release number it is version-scoped:
  re-grep at the pin in `SOURCES.json` (`SKILL.md` rule 5). Line numbers move
  between releases; the row's shape does not.
- No row matching a distinctive token is a data gap, not a fact about the code
  (`SKILL.md` rule 10): say which token you searched for.
- Rows are in no particular order. Scan the left column, do not read a table
  front to back. **Unsure which page?** One grep over this directory searches all
  every row at once: `rg -n '<distinctive fragment>' references/first-checks/signature-routes/`
  — a token, not a category (SKILL.md rule 8).

## Tracks

- **A** — the engine/worker never became ready (startup): no per-rank traceback, or
  the log simply stops. Lane page: `../../startup-failures/index.md`.
- **B** — a worker raised: a per-rank/plain traceback, a Python error with a
  `vllm_ascend` or `site-packages` frame.
- **C** — the engine ran: the fault is in a request, a device op, a metric or an
  answer.
- **D** — the deliverable is not a root cause: a sizing plan, a document report, a
  capability verdict, a code-change sweep.

## Routes

- Every row's third cell is the owning page; the pages in turn open
  `cheap-checks-by-track.md` for the first moves and `scripts-catalogue.md` for a
  re-runnable check.
- One topic entry per row: `../<topic>/index.md` states its scope, inherits the
  `SKILL.md` rules and routes to its own pages, so follow the row's link rather
  than reading the whole topic.

## The index

| The paste is… | Rows live in |
|---|---|
| a startup/load/capture log, a SoC- or image-conditional gate, a model/arg gate, a Ray or multi-process launch, or a sizing question | `startup-and-load.md` |
| an **artifact** stops it rather than a phase: a quantization method or a checkpoint's weight/aux keys, a pip resolution, an image tag or arch, an install/build step, a version string, two files disagreeing about a pin | `checkpoint-quant-and-install.md` |
| a device-op failure: an aclnn/custom-op error, a tiling rejection, a device dump, an `EZ1001` family, an HCCL collective or an MC2 dispatch | `device-ops-and-collectives.md` |
| a served request or an answer: chat template/tokenizer mode, reasoning routing, tool calls, truncation, a score gap, garbled or repeated text, a logging flag | `serving-output-and-toolcalls.md` |
| a parallel or stateful deployment: PD/KV transfer and pools, pooling services, spec decode/MTP, PP, mixed-batch attention, LoRA, host/device memory, a downstream patch | `parallel-state-and-kv.md` |
| track D — a capability verdict, a per-SoC recipe question, a document report, a flag-retirement sweep — or a perf claim backed by a profiler table | `capability-docs-and-throughput.md` |
