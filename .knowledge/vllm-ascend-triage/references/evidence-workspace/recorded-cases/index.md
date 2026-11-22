# Recorded cases: the per-issue index

## Scope

The per-issue index behind rule 1 of `SKILL.md`. One entry per issue this library
has triaged: the version, the symptom as filed, the track it landed in, and — where
it is instructive — what the recorded run got right or wrong. Read it to place a
new report beside a precedent, or to check what a delivery failure looked like
before deciding a session is going well; do not read it end to end. The
signature index (`../../first-checks/signature-routes/index.md`) routes by signature,
topical pages by track, and `scripts/history-titles.sh '<regex>'` by earlier-issue
title.

The entries are split by track family into the six pages below, one index line per
case: the version, the symptom, and — where the case has a topical page — a
`../../<topic>/<page>.md` pointer. A line that ends in `…` is a condensed index
line; its mechanism and checks are in the family page named on the page's first
line, so read that page before quoting the case. A line that stops at a full stop is
complete as written. Where the recorded session's ending is the point, the line
keeps it in a `(run: …)` marker — `no answer shipped`, `cut mid-answer` — because
that is the delivery axis rule 1 turns on; the turn counts and sizes are in the
family page and in `../delivery-recorded-sessions.md`. Open the page whose track matches the
classified shape, or `rg -n '#8826'` across this directory. A new report's issue id is
unknown, so search it with the **paste's own** token (the error sentence, the op or
kernel name, the numeric code) before the first source read: a line that matches on
versions and model means this index already carries that case's ladder (`SKILL.md`
procedure step 2b).

## Rules for using it

- A precedent supports a hypothesis; it does not replace reading the source at the
  pinned commit (`SKILL.md` rule 8).
- Pick precedents by whether the title names a *mechanism*, not by symptom match.
- An entry that ends "shipped no answer" or "cut mid-table" is a delivery lesson
  about that session, not evidence about the fault.
- The per-entry session annotations (turn counts, reasoning/delivery sizes,
  "shipped no answer") describe the environment each case was **recorded** in.
  They are not properties of the reporter's machine, of the pinned release, or of
  the session reading them — never size your own plan, or a claim about the
  reporter, from those numbers. What transfers is the corrective: which check was
  skipped, which artefact was never requested, which lever was never ablated.
- A new case costs one line here plus its family page; keep each page under the
  per-file size cap and split by the same family boundaries again rather than
  condensing entries when one reaches it.

## Routes by track family

- `startup-and-install.md` — engine start, load, capture, device open, plus every
  install, image, SoC, feature-enable, drift, tutorial and release-notes case. The
  case goes to the engine-start family page when its owning page is
  `../../startup-failures/` (engine start, load, capture, device open), plus the two
  cases whose answer starts at startup (`../../hccl-and-hardware/` at capture); every
  other owner puts it in the install family page. Detail: `../recorded-cases-startup.md`
  and `../recorded-cases-install-and-environment.md`.
- `device-and-communication.md` — device, HCCL/cluster, graph-capture and
  multi-instance faults. Two detail pages: `../recorded-cases-device-and-communication.md`
  (device, op-store and capture faults, mixed-batch) and
  `../recorded-cases-hccl-and-watchdog.md` (every case whose owning page is
  `../../hccl-and-hardware/`: collectives, communicator creation, transport/link,
  MC2/AIV dispatch, RPC watchdog, executor teardown, multi-instance).
- `accuracy-and-output.md` — accuracy gaps, degeneration, garbled/truncated output,
  logprobs and sampler faults. Detail: `../recorded-cases-accuracy-and-output.md`.
- `memory-and-throughput.md` — OOM, KV budget, sizing, performance and scheduling
  cases. Detail: `../recorded-cases-memory-and-throughput.md`.
- `pd-kv-and-features.md` — PD/EPD, KV pool and connector, speculative-decode and
  feature-flag cases that served. Detail: `../recorded-cases-pd-kv-and-features.md`.
- `brief-mentions.md` — the lines that list several issues at once, each with the
  one fact it establishes. Detail: `../recorded-cases-brief-mentions.md`.
