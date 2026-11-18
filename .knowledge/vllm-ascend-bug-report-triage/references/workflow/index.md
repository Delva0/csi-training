# Workflow: from a filed report to a workaround and a root cause

## Scope

The end-to-end procedure for one task class: a bug report (issue body, start command, logs,
`collect_env` dump) against a released tag of a fast-moving LLM-serving stack on Ascend NPU
(`vllm-project/vllm-ascend` and the `vllm-project/vllm`, torch_npu, CANN layers it sits on) must
become (a) at least one effective workaround and (b) a root cause concrete enough to file —
`path:line` at the reporter's ref plus the mechanism. The deliverable is prose for the reporter,
not a patch, a reproduction, or a performance study.

This topic covers the method only. Stack facts live in `../vllm-ascend/index.md`, failure
families and worked cases in `../symptoms/index.md`.

## Common rules

- **The reported version is the only authority for what the reporter ran.** Read code at that
  tag or release branch; label anything else "later history" or "HEAD-only". A claim that is true
  at `HEAD` and false at the reporter's tag is worse than no claim.
- **A fix existing on main is not a fix in the reporter's build.** A merged PR, a closed issue or
  a release-note bullet does not prove release inclusion. Prove membership with
  `git merge-base --is-ancestor <sha> <tag>` or `git tag --contains <sha>`.
- **The issue thread is not the answer.** Maintainer comments are often only "Pls take a look"; an
  issue can be closed by a stale bot with no fix; closure is not resolution. A code comment
  referencing another issue is frequently the real explanation. Thread statements are
  *attributed* evidence (who said it, when), not verified fact.
- **Both deliverables, always.** A workaround that follows from the mechanism (the exact
  flag/config/env or version change, not just "upgrade") **and** a root cause with repo + path +
  symbol/line at the reported ref. An investigation that never states them fails regardless of how
  much was read.
- **Keep three evidence classes apart** — *verified* (code read at the ref, ancestry result,
  linked patch, unit test, doc line), *reported* (issue text, maintainer comment, reporter log),
  *hypothesis* (inference from a similar issue or plausibility). Never let a hypothesis inherit
  the confidence of the other two.
- **No hardware.** You cannot reproduce on NPU; local hardware state says nothing about the
  reporter's machine nor about their card count, memory or CANN build. Verify by code reading,
  unit tests under `tests/ut/`, ancestry checks and docs, and state what would settle the rest.
- **Name the ref in every code claim**, next to the `file:line`. Say whether the finding holds at
  the reported ref, on a release branch, or only at HEAD.
- **Declare gaps.** If the clone lacks the reported ref or the vendored dependency version, say so
  instead of quietly substituting current code. Report stale or incomplete knowledge when the code
  disagrees with this skill's notes.
- **Bound every read and answer inside-out.** Whole-file reads, repo-wide greps and top-level
  directory tours burn the generation budget the write-up needs; `answer-and-claims.md` carries the
  schedule and the draft-first rule that keep the reply inside it.

## Classify before reading code

Five facts decide the owning module; write them down before the first read.

1. **Release** — vllm-ascend version → tag or release branch, plus the pinned vLLM, torch_npu, CANN and
   driver versions. Reconcile the report's prose against the log banner; when they disagree, fingerprint
   the built ref from the traceback's `path:line` or from the `g<sha>` suffix of a dev string
   (`…dev849+g05e095a20` → `05e095a20`). An image tag hides its CANN base — read the `Dockerfile*` at the
   tag (`../vllm-ascend/version-pairing.md`).
2. **Hardware class** — 310P/Atlas 300I DUO vs A2/A3/A5/950. 310P has its own implementation family under
   `vllm_ascend/_310p/` and must be checked against the 310P override first.
3. **Parallelism axes** — TP, DP, EP on/off, PP, SP, PCP, DCP, DSA-CP. Many bugs are combinations (PCP
   without EP; PP with one list of layer names per stage).
4. **Quantisation** — W8A8 / W4A8 / W4A4, MXFP8, and KV-cache quantisation ("C8" = fp8/int8 KV cache,
   sparse vs dense).
5. **Feature switches** — the `--additional-config` JSON verbatim and any `VLLM_ASCEND_*` env vars; a key
   that does not exist at the tag changes the answer (`../vllm-ascend/config-flag-lifetimes.md`).

## Order of work

Eight steps, an order rather than eight turns, inside the write gate (`SKILL.md`: `skill_view` **inside**
batch 1, then batch 2, an optional third read, then the reply — the reply is `content` in the message where
the mechanism settles, with no reasoning in that message; a message carrying only the skill load still
spends one of the three tool-calling slots): batch 1 is steps 1–2, one symptom-scoped batch 2
covers whichever of steps 3–6 the symptom needs, and steps 7–8 are the reply. A symptom that already matches a
route-table row, a class page or a ledger entry may be answered right after batch 1; stop once the
mechanism is confirmed at the ref and one independent signal agrees.

1. **Read the report, then only the missing thread parts** — the prompt carries body, logs and
   `collect_env`; one `github` call gives the maintainer advice (`evidence-sources.md`). In the same
   `bash` call, grep the ledger for the issue number: a row means the thread, the frames and the
   ruling-out are already written down (`../symptoms/issue-ledger/index.md`).
2. **Pin the ref** — `scripts/resolve_ref.sh` / `scripts/ref_snapshot.sh <ref>`, both accepting the
   reported spelling; fingerprint a conflicting build with `scripts/ref_line_of.sh` (`version-pinning.md`).
3. **Harvest the decisive log** — `scripts/fetch_issue_evidence.sh <outdir> <url>...`.
4. **Read the failing path at that ref, not from memory** — `git ls-tree`, `git grep -n -F '<symbol>'
   <ref>`, `git show <ref>:<path> | sed -n 'A,Bp'` (`../vllm-ascend/repo-map.md`,
   `../vllm-ascend/code-layout.md`).
5. **Check release notes and docs at the ref** for known issues, renames and the tutorial config the
   reporter copied (`../vllm-ascend/config-flag-lifetimes.md`).
6. **Find the fix or regression after the report**, then test inclusion with
   `scripts/ref_ancestry.sh <repo> <ref> <commit>...` (`fix-discovery.md`).
7. **Match the symptom to a failure class**, re-read its code at the ref, confirm with a second
   independent signal (`../symptoms/index.md`).
8. **Answer** — workaround first, then root cause, evidence labels, scope and the falsifier
   (`answer-and-claims.md`, `../../templates/finding-report.md`).

## Routes

- `version-pinning.md` — reported version string → ref, conflicting version signals (traceback
  line fingerprint) and release-branch/ancestry mechanics. Read first, always.
- `evidence-sources.md` — the issue thread, attachments and the issue corpus through the `github`
  tool, a third-party file (an eval harness's meta JSON, a model directory's `config.json`) read with
  one contents-API call instead of a web fetch, plus the shell/job mechanics that decide how many turns
  an investigation costs, and the shipped nightly/e2e configs as the deployment the project actually
  tests.
- `log-signatures.md` — the routing table from a verbatim log line to the file that logged it
  (`Added request`/`Aborted request` → `multiprocessing/engine.py` vs `async_llm_engine.py`, the LoRA
  `TORCH_CHECK` messages, `not found in this op store`, the `*failed to start` wrappers). One
  `git grep -n '<message>' <ref>` replaces a written-out candidate list.
- `fix-discovery.md` — find the fix or regression after the report: history searches, release
  notes and fragments, tutorial docs, tests, how to verify a fix by its hunk rather than its subject,
  how to read a same-line `rc` → final regression (the range, the reverts/re-defaults and the ancestry
  of the PRs the thread names, then the reply), and how to avoid citing a reverted fix.
- `answer-and-claims.md` — the shape of the deliverable, workaround tiers and costs, confidence
  wording, the checks that keep a lead from being reported as a cause, and delivery discipline: the
  verbatim stop triggers and the shapes behind them (a verdict written into reasoning, a recap opener,
  post-answer exploration, a reply that starts and re-decides), plus why `answer.md` is turn-1 scratch
  and the heredoc transcription is a double charge. The reply is the only artifact.

## Support

- `../vllm-ascend/index.md` — repository and release facts for this stack.
- `../symptoms/index.md` — symptom → suspect routing and the verified failure classes.
- `../../scripts/` — one read-only probe per step; run shell with `bash`, Python with
  `uv run --no-project python`, all write only under `/workspace/scratch`. Every `[repo]` below takes a
  clone **path or bare name** (`vllm-ascend`, `vllm`, `Mooncake`) through `repo_root.sh`, which exits 3
  when the argument names no clone — that exit is not evidence that a tag or path is absent:
  - `repo_root.sh [repo-path-or-name]` — the clone root behind a name/path; exit 3 for a name that is
    no clone (prints the clones that exist). Every repo-taking script calls it.
  - `open_answer.sh "<repo>#<N> — <symptom>" [version-or-ref] [repo]` — batch-1 scratch: resolves the
    reported version to a tag (`0.18.0`, `0.23.0rc1`, `0.26.rc1`, `0.21.0rc1-310p`, a `+g<sha>` dev
    string), writes `/workspace/scratch/answer.md` with the skeleton plus the `ref_snapshot.sh` facts
    and prints the file whole; exit 2 with the candidate tags **and release branches** (`origin/releases/
    vX.Y` — where an rc/dev build lives before a tag exists) when the ref does not resolve, plus the sha
    extracted from a `+g<sha>` string to re-run with, **3** when the repo argument names no clone. A
    `X.Y.ZrcN.devK+g<sha>` version is a branch build: read the tree at that sha, not at the nearest tag,
    because the two differ (a path "absent at the tag" says nothing — `../vllm-ascend/install-and-image.md`
    §5). The file is not the deliverable.
  - `resolve_ref.sh <version-or-ref> [repo]` — one line, the ref a reported version means, or exit 1
    with the near tags; `open_answer.sh` and `ref_snapshot.sh` call it, so both accept the version as
    written.
  - `version_ref.sh <version> [repo]` — version string → exact tag, near tags, release branches,
    default-branch HEAD.
  - `version_archaeology.py --repo <repo> --version <v> --keyword <k>...` — version → refs, ancestry gap
    to main, per-keyword hits at the tag and keyword history.
  - `ref_snapshot.sh <ref> [repo]` — paired vLLM tag, runtime pins, package layout, moving-path
    existence, the `vllm_ascend/core` and `patch/platform` file names (a downstream patch named after
    the symptom is a first-class suspect), whether the 310P model-runner V2 exists, the NZ layout policy
    and commits since, in one call; an unresolvable ref is reported as a version-string problem with
    near tags, not a git fatal.
  - `ref_grep.sh <repo> <pattern> [ref...]` — grep a symbol across refs (fallback for pickaxe).
  - `ref_line_of.sh <repo> <path> <pattern> [ref...]` — fingerprint the built ref from a traceback line.
  - `ref_ancestry.sh <repo> <ref> <commit>...` — is a fix inside the reported release?
  - `export_ref.sh <repo> <ref> [dest]` — materialise a ref under `/workspace/scratch` for whole-tree
    reads.
  - `fetch_issue_evidence.sh <outdir> <url>...` — download attachments, print the exception/stall digest
    and file tails.
  - `flag_lifecycle.sh <flag> [repo]` — per-tag presence of a flag, release-note mentions, pickaxe
    commits, definition sites.
  - `merge_tool_deltas.py <sse-or-json-file>` — merge a raw `curl -sN` tool-call stream by `index` and
    print the shape (id never streamed, empty terminal arguments with `finish_reason`, argument bytes in
    the finish chunk, a `tool_calls` finish with no delta) — `../symptoms/tool-call-streaming.md`.
- `../../templates/finding-report.md` — the answer skeleton.
