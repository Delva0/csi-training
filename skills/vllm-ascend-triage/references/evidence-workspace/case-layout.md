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

**The `vllm` pin is not automatically the version the vllm-ascend pin pairs with.** Check the
pairing before quoting upstream code as the failing version:
`rg -n "vllm_version|pip_vllm_version|ci_vllm_version" <root>/vllm-ascend/docs/source/conf.py` and
`requirements.txt`. Recorded #2975: `SOURCES.json` pins vllm-ascend `v0.10.2rc1` (`048bfd5`) with
upstream `vllm v0.9.1` (`b6553be`) while `conf.py:68` says `'vllm_version': 'v0.10.2'` — the
upstream tree is the older *working* side (the reporter's own 「0.9.1没有问题」), so its code cannot
show the regression and `rg` for a 0.10-only string there returns nothing. In that situation the
failing pin's evidence is the **downstream copy** of the shared machinery
(`vllm_ascend/worker/model_runner_v1.py`, `vllm_ascend/core/scheduler.py`, `vllm_ascend/patch/**`),
with the upstream tree as the comparison side; state the absent version as a gap instead of
reconstructing it from memory (`../upstream-downstream-drift/patch-copy-vs-upstream.md`,
`../multimodal-inputs/empty-message-500-encoder-cache.md`).

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
counts report text. Do **not** then slice the script's own output with
`sed -n 'A,Bp'`: the default cap is 80 lines, so `history-body.sh <id> | sed -n '120,260p'`
prints nothing at all and reads as an empty body (two such calls in the #10532 run). Pass
`-n 0`, `-n <lines>` or `-g REGEX` instead, and read the `... (N more lines; raise -n or use
-g REGEX)` marker the script prints. Keep the dump reachable with `--env`: it is the
artefact when the question is the torch_npu/CANN pin, the card count or host RAM, and
reporters also type free prose into that field (#4554's whole symptom sits there). Flat
layout: `/workspace/history/<issue>.md`.

**A `not found` id is usually newer than the corpus, not a missing precedent.** Each
case view's corpus stops at the highest `<id>.md` it carries (at the #6533 view:
6436), while this skill's own pages and the recorded-cases index cite ids past that
(8590, 7689, 13591 …) from earlier sessions. `history-body.sh` reports the miss on
stdout *and* stderr for exactly this reason: the recorded #6533 run typed
`history-body.sh 8590 2>/dev/null | head -120`, saw an empty result, read it as an
empty body, spent the next call on `/repos/…/issues/8590` (404 in this dated replay)
and then treated the page's same-pin #8590 precedent as unusable. The page's pin,
status lines and verdict are the evidence of record — quote them, and drop the id
from the search instead of retrying it through another lens.

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

## Search-command traps (SKILL.md rule 7 detail)

These each cost a recorded run whole steps. `rg -n -C2 <symbol>` and
`sed -n 'A,Bp' file` are the default reads; reserve whole-file reads for files under
~150 lines.

- **`skill_view` with `file_path` returns the page *and the whole link
  inventory*** (every reference, script and template path): 34.7 KB and 43.6 KB
  for two page reads in one recorded run, re-sent on every later turn. Read pages
  from the skill root instead —
  `sed -n 'A,Bp' <skill_root>/references/<topic>/<page>.md`.
- `rg -r` is `--replace`, not "recursive": in any short cluster whose first
  letter is `r`, the rest of the cluster is the replacement string, so `-rn`,
  `-rl` **and `-rln`** rewrite every match to `n` / `l` / `ln`. Output then
  looks like corrupted source (`self.n`, `decoder_layer.l`) or, worse, like a
  real upstream defect: #14423's `rg -rn "daocloud|nju.edu.cn"` turned
  `m.daocloud.io` into `m.n.io`, and #17718's `rg -rln 'set_env.sh'` turned
  `source …/ascend-toolkit/set_env.sh` into `source …/ascend-toolkit/ln` in
  `faqs.md` — which it nearly published as a doc typo and dropped only by
  luck, blaming a "stale .po". **The tell is one short token spliced
  uniformly into every match across unrelated files**; re-run as
  `rg -n '<pat>' <dir>` before quoting any matched text, and use `grep -rn`
  as the tie-breaker when output looks mangled. Recursion is implicit on a
  directory: `rg -n "pat" .` or `grep -rn`. **An empty replacement is
  the same flag with a worse tell:** a replace flag prints whole matching *lines*,
  so `rg -n "30566" -r '' .` returned **63 KB** — one minified
  `tests/models/fixtures/*.json` line, matched by accident inside a number — and
  that run (#8036) ended output-truncated with no answer typed. A "hit" whose
  result is a single gigantic line is this trap, not a finding.
  An **empty pattern matches every line of every file**: a probe typed while unsure
  how to recurse — `rg -rn ""`, `rg -n "" .`, `grep -rn "" .` — dumped 65 KB in one
  result (#6240; #549 while looking for an upstream selector). To list files use
  `rg -ln '<pat>' <dir>`; with no pattern in mind use `rg --files -g '<glob>' <dir>`
  (or `find <dir> -maxdepth 1 -name '<glob>'`), and quote only the `path:line`,
  with `-l` or `--no-messages -c` when existence is all you need. Detail:
  `references/evidence-workspace/case-layout.md`.
- **A pasted message is usually split across source lines.** C++ `OP_LOGE` and
  Python messages are emitted as adjacent literals (`"… default chat template is no
  longer "` / `"allowed …"`), so `grep -rn "<whole sentence>"` returns nothing and reads
  as "not in this tree" — grep the shortest distinctive clause, or add `-A1`. #6345
  spent three calls that way, on the exact sentence it was looking for.
- **A `--glob '!<dir>/**'` exclusion is matched against the path you passed,
  not against the search root.** `rg -l 'x' docs/source --glob '!locale/**'`
  still searches `docs/source/locale/**` (verified at v0.22.1rc1: one .po hit
  survives), while `--glob '!**/locale/**'` or `cd docs/source` first excludes
  it. A failed exclusion floods the result with gettext `msgid`/`msgstr`
  lines that read exactly like doc prose — and `.po` files are machine
  translations whose `#: ../../source/x.md:N` references lag the source, so
  neither their text nor their line numbers are citable.
- **A piped `sed -n 'A,Bp' file | rg -n '<pat>'` numbers the stream, not the file.**
  `rg` counts its stdin from 1, so every match line is `A + n - 1` — an off-by-one
  that lands as a wrong `path:line`. Cross-check one match against an absolute
  `rg -n '<pat>' file`, or keep the file's own numbers:
  `rg -n '<pat>' file | awk -F: '$1>=A && $1<=B'`. Recorded #12905 re-derived a
  tutorial block's anchors twice before one absolute grep confirmed them. The same
  off-by-one arrives with no pipe at all: a line number counted by eye inside a
  window. A doc-verification anchor is the first thing a maintainer re-checks —
  recorded #11738 shipped `:113` for the page's `--served-model-name`, which is `:114`.
- **Scope a grep to one repo.** A case view holds 20-40 exported trees under
  `/workspace/repos/<issue>/`; an unscoped `rg -n '<string>' .` there takes
  seconds and returns hits from unrelated products (torchair, metadef,
  ascend-transformer-boost). #11454 ran three of them looking for one log
  string, and the finding was that the string is in **no** export — state that
  as the gap (rule 10) instead of widening the pattern: `rg -n '<pat>'
  /workspace/repos/<issue>/<owner>/<repo>` per candidate, then stop.
- **Never list, never dump** — the recipe at the top of this file is the rule; this is
  what breaking it costs. A listing of `/workspace` was 24-28 KB once (#13699) and
  13-104 KB per later session (#9335, #9183, #11739, #8984, #11454), `cat
  SOURCES.json` or `cat /workspace/history/*.md` likewise (#3107), and one recorded
  orientation block (`ls /workspace`, `ls scratch`, `cat SOURCES.json`, `ls history`)
  returned **62 K characters** (#15985) and ended output-truncated at 1.1 M cumulative
  prompt tokens — #5687's same block cost 3 KB only because `SOURCES.json` was piped
  through `head -50`. A missing `/workspace/attachments` also makes a bundled listing
  exit 2, which reads like a failed orientation.
- **Attachments are the exception: a manifest grep is not an existence test.**
  `ATTACHMENTS.txt` lists names and SHA256s only, so a grep for the issue's
  tokens proves nothing (#12658 read that as "no attachments", skipped
  `raw_completion.json`, and burned the session in the parser). Search the
  attachments view by the file names the issue links (`search_files pattern
  '*.json'`) and open them — they are often machine-readable dumps.
