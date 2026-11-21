# Case layout: concrete commands

Orientation (cheap, one call):

```bash
ls /workspace
bash <skill_root>/scripts/sources-summary.sh   # version contract, 1 line per repo
```

The repo root is not the package root: grep paths start at `vllm/` and
`vllm_ascend/` inside `repos/<owner>/<repo>/`, so `rg model_executor/...` from the
repo root returns `No such file or directory` (and one recorded run spent two
calls on it before prefixing). The workspace root also differs per case:
`/workspace/repos/<owner>/<repo>` here, `/workspace/repos/<issue>/<owner>/<repo>`
when several cases share the snapshot — `ls` once, then use the absolute path.

One file in this tree is a documentation registry, not code:
`vllm_ascend/patch/__init__.py` was 1473 lines at v0.26.0rc1 and lists every patch
with its rationale. Never `cat` it — `rg -n '<patch_module>'` it, and read the set
that actually runs from `vllm_ascend/patch/platform/__init__.py` (59 lines) or
`patch/worker/__init__.py` (75) instead.

Source reading at the pinned commit (when the repo is a one-commit git):

```bash
git -C /workspace/repos/<...>/vllm-ascend log --oneline -3     # confirm you are at ref
rg -n "aclnnAddRmsNormBias" /workspace/repos/<...>/vllm-ascend -g '!*.po' | head
sed -n '240,300p' path/to/file.cpp
```

`git log --oneline -3` on these exports usually shows a single commit (the release); there
is no divergence history, so "when was this introduced" cannot be answered from the export.
Say that instead of guessing.

The release's own validation record (vllm-ascend ships one per release when present):

```bash
ls /workspace/repos/<...>/vllm-ascend/tests/vllm_ascend_v*_test_conclusion.md
sed -n '1,20p' .../tests/vllm_ascend_v0.26.0rc1_test_conclusion.md   # validated models
sed -n '50,76p' .../tests/vllm_ascend_v0.26.0rc1_test_conclusion.md  # known issues + ids
```

At v0.26.0rc1 §1 names "the following 4 models, which are the only models
supported in this release" and §3 lists the unresolved known issues with issue
ids. That separates "inside the tested envelope" from "untested combination"
before you build a theory, and the issue ids are ready-made precedent.

History precedent (never `ls` the history directory — thousands of entries):

```bash
S=<skill_root>/scripts        # skill_root is reported by skill_view
bash $S/history-titles.sh 'aclnnAddRmsNormBias|DDR address of the MTE'
bash $S/history-titles.sh 'qwen3\.5|qwen3_5'
bash $S/history-body.sh 3963 4663                     # symptom section, env dump collapsed
bash $S/history-body.sh 3963 -g 'HcclAllreduce|ERR00100|107027'   # only the traceback lines
bash $S/history-body.sh 3963 --env -n 40              # the env block, when the pin is the artefact
```

Do **not** `sed -n '1,60p'` a body (an earlier version of this page said to): the bug
template puts `### Your current environment` plus a 100-250 line `collect_env.py` paste
first, so a prefix read returns the title and the environment and no symptom — #3963's
`### 🐛 Describe the bug` is at line 148 and #4663's at 168, and one recorded run paid
two calls for one body (`sed -n '1,60p'`, then an inline `awk '/Describe the bug/{f=1} f'`).
`history-body.sh` takes several ids per call, prints `<id> :: <title>` plus the body with
the fenced dump replaced by a marker, and drops `<details>`/`<summary>` furniture so `-n`
counts report text. Keep the dump reachable with `--env`: it is the artefact when the
question is the torch_npu/CANN pin, the card count or host RAM, and reporters also type
free prose into that field (#4554's whole symptom sits there). Flat layout:
`/workspace/history/<issue>.md`.

With a per-case history root the same issue id can appear once per case directory
(`history/<issue>/<id>.md`); `history-titles.sh` now reports `hits / distinct issue
ids` and prints one line per id, so a `limit` is not spent on copies. Read one copy
and cite the id.

`grep -rn` / `rg -n` are safe; `rg -rn` and `rg -rl` are not (`-r` is `--replace`, so the
matches are rewritten to `n`/`l` and the output looks like corrupted source). `grep -rn`
over the same tree is the tie-breaker when output looks mangled.

A bare 2-4 letter token is not a search key: every pasted `collect_env` dump in
the corpus carries the CPU-flags line (`fp asimd evtstrm … fcma dcpop asimddp
asimdfhm …`), so `dcp`, `dp`, `ep`, `mtp` or `hccl` match hundreds of unrelated
issues — recorded #12677 spent its precedent call on `dcp|decode.context.parallel`
and got "382 hits / 382 distinct issue ids", the first 40 of them `dcpop`
matches. Search a flag- or message-shaped pattern instead
(`--decode-context-parallel-size|decode.context.parallel|dcp==`, and `-i` only
when the corpus capitalises the token), and read the script's
`hits / distinct ids` line before opening a body.

Quote precedent as `#<id>: <one-line fact>` (e.g. "upstream parser fix tracked in vllm PR
34779, referenced by #7061/#7094"). Distinguish "the same symptom was reported" from "the
same cause was confirmed".

## Attached log dumps (multi-MB, several nodes, several incidents)

A report can carry its whole evidence as attached dumps instead of a paste
(#16652: four files, 0.5-5.5 MB, 62 K lines together, two separate crashes plus
one hang). Read them by counting, never by printing:

```bash
cd /workspace/attachments/<issue> && cat ATTACHMENTS.txt          # 5-10 lines
wc -l *.txt                                                       # size per file
rg -c 'fftsplus|Heartbeat Lost|ERR02005|error cqe' *.txt          # tokens per file
rg -o 'timeout=[0-9]+s' *.txt | sort | uniq -c                    # how many incidents
```

Three properties of the recorded format decide every later grep:

- **Three time bases on one line.**
  `[2026-09-06 21:43:12.113][sv-<host>-1] EE9999[PID: 3992] 2026-09-06-13:43:12.110.307 …`
  — the collector's bracket prefix is **local** time, while the logger's own
  `ERROR MM-DD HH:MM:SS` field and every embedded Ascend record
  (`YYYY-MM-DD-HH:MM:SS.mmm.uuu`) are **UTC** (8 h behind on the recorded UTC+8
  hosts). A constant whole-hour lag is a timezone, not a replayed record;
  `bash <skill_root>/scripts/plog-timeline.sh <file>` detects and normalises it
  and names the in-window family.
- **Greps must use the field you mean.** A grep on the collector's local hour
  (`'^\[2026-09-06 21:1'`) and one on the UTC hour inside the line (`'13:1'`)
  select different windows; state which one you used before ordering events.
- **Line order is not time order.** The recorded dumps print the newest line
  first, so `head` and `tail` are the two ends of a 10-14 h span and a lower line
  number can be a *later* instant. Locate a window by its timestamp string; do not
  interpolate line numbers.

One file per (node, engine restart) means a dump set can hold two incidents plus
the restarts between them: anchor each incident on its own window (see
`../hccl-and-hardware/error-report-timestamps.md` for the `timeout=` arithmetic
that back-dates a stall) and say which of the reporter's "N times crashed" the
attachments cover. For the second incident in a set, remember the UTC offset
crosses local midnight — the record dates then read one day *earlier* than the
log lines they belong to.

Checklist for the final answer: repo+ref+commit cited once; every code claim has
`path:line`; precedent ids quoted; missing artefacts listed; OCR/snapshot limits stated.
