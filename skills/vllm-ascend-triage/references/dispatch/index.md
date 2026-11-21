# Dispatch: from a classified shape to the topic entry

Scope: the routing decision that follows *Classify the evidence* in SKILL.md — given
the pasted symptom, which `references/<topic>/index.md` owns it. The one content page,
`classified-shapes.md`, is the table: one row per routing decision (a topic entry,
or one page inside it), written in the reporter's own symptom wording, each carrying
the owner and the page to open. Inherit every rule in SKILL.md; these three decide
how the table is used.

1. **Classify first, then match the wording.** The table is a matcher, not a
   classifier: a row is selected by what the paste *says* (an error string, a flag
   combination, a request/response body), never by the reporter's title word. Several
   rows share a word (`乱码`, `OOM`, `卡死`, `精度`); the distinguishing clause is
   inside each row, and a row that carries `not this topic` is there to stop the wrong
   turn — take that exclusion and continue in the row it names.
2. **Open one row's topic entry, not the table's neighbours.** The routing answer is
   one `index.md`; its own rules then route to a page. Reading two or three topic
   indexes "to compare" is what a 16 384-token budget buys instead of an answer.
3. **The row's conditions travel with the route.** When a row names a version window,
   a reporter-confirmed resolution or a `cause left open`, carry that into the short
   answer rather than upgrading a hypothesis to a cause.

## Routes

- `classified-shapes.md` — the table. Read the row whose wording matches the paste,
  then open the topic entry its last cell names. Rows are grouped by topic (accuracy
  and output, startup and load, runtime device/memory/custom-op, HCCL/cluster, PD and
  KV pool, spec decode and parallelism, SoC paths, capability/perf, docs, evidence
  workspace), not by frequency.
- `../first-checks/signature-routes/index.md` — the same problem from the other end:
  a single log fragment or traceback line → the track it belongs to. Use it when the
  paste's wording is a fragment rather than a whole shape, or when two dispatch rows
  are still tied after both were read.
- `../first-checks/index.md` — once the topic is known, the per-track cheap checks
  (step 3 of the procedure) are there, not in this table.
