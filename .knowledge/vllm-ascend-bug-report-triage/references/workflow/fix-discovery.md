# Fix discovery: what changed after the report

Goal: a short list of commits whose stated repro conditions match the report, ordered from most to
least plausible, plus the release-note or doc sentence that names the user-visible effect.

## Order of operations

1. **Exact tokens at the reported tag.** Every key from the reporter's `--additional-config`,
   `--speculative-config`, env exports, and the error strings in their logs:

   ```bash
   git -C "$REPO" grep -n -F '<token>' "$TAG" -- '*.py' | head -20
   ```

   Two outcomes matter:
   - the token is read at that tag, with a gating condition you can quote;
   - the token is *not* read at that tag (renamed, split, or removed) — the reporter's setting is
     then inert, and "disabling it changed nothing / fixed it" needs re-reading. A silently
     ignored key is a common source of misattributed reports; see
     `../vllm-ascend/config-flag-lifetimes.md`.

   Model ids are loose in reports (`-073` for a `-0731` checkpoint, `0.26.rc1` for `v0.26.0rc1`);
   match them fuzzily against `docs/source/tutorials/models/` and `recipes/overlays/`.

2. **Commits after the tag, filtered by subsystem words** (cheap, high yield):

   ```bash
   git -C "$REPO" log --oneline "$TAG"..origin/main | grep -iE 'dspark|prefix|accuracy|hang|reshape|c8|mtp'
   git -C "$REPO" log --oneline "$TAG"..HEAD -i --grep='<symptom words>'
   git -C "$REPO" log --oneline -- <path>        # when a feature appeared or changed in one file
   ```

   For a 1900-commit gap this returns a few dozen lines. Skim subjects, then open bodies.

3. **Open the candidate bodies.** In vllm-ascend, commit bodies have stable sections:
   `### What this PR does / why we need it?`, `### Does this PR introduce _any_ user-facing
   change?`, `### How was this patch tested?`. The second is the release-note sentence; the third
   names the regression test, which is effectively the reproduction recipe.
   `git -C "$REPO" log -1 --format=%B <sha>`

4. **Check the diff, not just the message**, and bound the window so the message cannot eat it:

   ```bash
   git show --format= <sha> -- <path>            # diff only, no message/stat preamble
   git diff <sha>^ <sha> -- <path>               # same, explicit
   git diff <old-tag> <new-tag> -- <path>        # when the thread names the fixing release
   ```

   `git show <sha> -- <path> | head -40` prints the commit message and no `+`/`-` lines; a recorded
   run then spent three further messages re-reading the same hunk. `git log -S'<expression>'` matches
   nothing when the source wraps that expression across lines — an empty result there is not evidence,
   and a diff between the two tags the thread names is the cheaper question.

5. **Confirm the user-facing statement** in the aggregated release notes or a fragment (below), so
   the answer can cite documentation rather than only a code diff.

6. **Check what the reported version's own docs shipped**, because the reporter copied them
   (below).

7. **Confirm release inclusion** with `git merge-base --is-ancestor <sha> "$TAG"` or
   `git tag --contains <sha>` (`version-pinning.md`). A fix that post-dates the tag is a strong
   root-cause candidate; a fix that predates it means the reporter is on a stale build or the fix
   is incomplete. When a thread lists numbers, check each one before quoting the list: `/pulls/N`
   `merged` + `base/ref`, then ancestry of its merge/cherry-pick commit. A closed **unmerged** backport
   (`merged: false`, base `releases/vX.Y.Z`) is not a fix at all — #13349 is that shape, while
   #12762/#13028/#12643 are ancestors of the `v0.23.0` tag, so the "merge these PRs" comment on #13418
   names nothing a reporter on that release lacks (`../symptoms/kv-pool-store-hang.md` §E2).

## Release notes and fragments as an index

- `docs/source/user_guide/release_notes.md` — one aggregated file (~300 KB, thousands of lines).
  Grep it, never print it:

  ```bash
  git -C "$REPO" show "$TAG":docs/source/user_guide/release_notes.md | \
    grep -n 'c8_enable_reshape_optim\|StoreKVBlock'
  ```

  It carries `Removed ...`, `Breaking changes`, dependency pins and per-release PR links, which is
  exactly the "workaround = upgrade" evidence, plus a per-release `Known Issues` section that is
  the single fastest answer when the symptom is listed. Check which `## vX.Y.Z` heading a line sits
  under: an entry from a later release section may not describe the reporter's build, and release
  notes can run ahead of a release branch's code.
- `release_notes/fragments/<PR>.yaml` — one file per user-visible PR:

  ```yaml
  pr: 20321
  entries:
    - category: bugfix
      text: Fixed AscendStore bulk and layerwise worker initialization errors caused by stale Memcache barrier arguments.
  ```

  Grep all fragments for a symptom word: `grep -ril 'hang\|accuracy\|garbled' release_notes/fragments/`.
  `release_notes/archive/` is empty in this clone, so old fragments are only visible through the
  aggregated file.
- `release_notes/README.md` states the inclusion rule ("can the user of the released artifact
  notice the change?"), a useful filter when deciding whether a fix is worth citing.
- `docs/source/user_guide/configuration/additional_config.md` — the flag table with defaults;
  flags are deleted from it when removed.

## Tutorial docs are the reporter's source of truth

Reports say "using the official configuration". `docs/source/tutorials/models/<Model>.md` holds
those command lines (2–5 k lines each: prerequisites, single-node, PD separation, KV pooling,
accuracy eval, FAQ, per-hardware sections). Grep, do not read:

```bash
git -C "$REPO" grep -n '"c8_enable_reshape_optim"' "$TAG" -- docs/source/tutorials/models/GLM5.2.md
```

Verified example: `v0.23.0rc1:GLM5.2.md:683` and `:752` ship
`"enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "c8_enable_reshape_optim": true`,
`v0.26.0rc2:GLM5.2.md:505` the same, and `main` has no such key at all. That is how a reporter ends
up enabling an optimisation nobody meant to turn on, and how you show the docs, not the user, were
the trigger. Check the same way for whatever switch replaced a removed key at HEAD
(`../vllm-ascend/config-flag-lifetimes.md` lists the current equivalents).

**Then check the block against the hardware row it sits under.** One page carries several per-hardware
tabs, and a value copied across them can exceed the smaller platform's capacity: at vllm-ascend
`v0.30.0rc2` the DeepSeek-V4.1-Flash A2 tab pins `--max-model-len 1048576` (`:691`) for servers the
same page describes as **64 GB per NPU** (`§3.1`), while `§11` says "The A2 configuration is retained
unchanged and is not revalidated by this update" — so a reporter on 64 GB A2 cards gets an engine that
refuses to start, and at `HEAD` the 4-node 1P1D A2 tabs carry the same `1048576`
(`:786`, `:896`, `:1696`). Grep every tab and read the capacity/limitations sections before declaring
the doc correct:

```bash
git -C "$REPO" show <tag>:docs/source/tutorials/models/<Model>.md \
  | grep -n -E '^#{2,3} |=== "|max-model-len|gpu-memory-utilization|max-num-batched-tokens'
```

The rejecting code is usually upstream and its message carries the corrected value
(`kv_cache_utils._check_enough_kv_cache_memory`, `../symptoms/startup-and-stall.md` §3b, #18158); the
deliverable is that doc line plus the block that fits. Docs are **not** backported to every tag —
`git show v0.18.0:docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` is legitimately absent — so pin
the tag the reporter's image names (`export IMAGE=…` inside the page) before diffing two versions.

Also useful in the tutorials: `FAQ` / `Common Issues` sections and the `--no-enable-prefix-caching`
baseline most PD examples use.

**A feature that is absent, not broken.** For "this input/flag is simply not supported on my tag", the
introducing commit *is* the fix. `git log --oneline --reverse -S'<feature>' --all -- <the fork's copy
of the file>` names it, then `git tag --contains <sha> | sort -V | head -3` gives the first release, and
`merge-base --is-ancestor <sha> <reported-tag>` proves absence. Reading `HEAD`'s code answers neither
half; `../symptoms/missing-downstream-feature.md` (prompt embeds → `v0.12.0rc1`) is the worked case.

## Tests as specification

`tests/ut/...` names encode the contract and often the bug:
`tests/ut/attention/a2/test_sfa_v1.py::test_li_c8_reshape_optim_requires_layer_li_c8`,
`tests/ut/kv_offload/test_dspark_pd_context.py`, `test_dspark_transfer_status.py`. A test that
exists at the tag but not on main tells you which behaviour was later removed; a test added by the
fix names the exact regression case. For a launch-time contract, unit tests also carry the
behavioural invariants by directory: `tests/ut/sample/`, `tests/ut/patch/worker/`, `tests/ut/core/`,
`tests/ut/quantization/`, `tests/ut/worker/test_pcp_manager.py`.

## Issue threads

The `github` tool reads issues and PRs (REST paths under `/repos/OWNER/REPO/...`); request shapes
and pagination are in `evidence-sources.md`. Read state before content: a closed issue may have
been closed by a stale bot with no fix (seen: #12533), in which case the thread is evidence of *no*
resolution. Comments are often only "Pls take a look" (seen: #15927). Prioritise issues that a code
comment references (`# See https://github.com/<org>/<repo>/issues/<N>`): that is a maintainer
stating the cause.

Do not infer release inclusion from a merged PR or a mention; use `merge-base --is-ancestor` or
`tag --contains` — and read the tag's tree as well, because a false ancestry result is not absence
(below).

## When the fixing repo is upstream `vllm` (or torch_npu/CANN)

If the deepest frame is in `vllm/`, the fix may never appear in the vllm-ascend clone. Run the same
history search in `/workspace/repos/vllm-project/vllm` at the paired tag, and test inclusion with the
*vLLM* tags, not the plugin's:

```bash
git -C /workspace/repos/vllm-project/vllm log --oneline --all \
  -S'<symbol from the traceback>' -- vllm/<subsystem>/<file>.py
git -C /workspace/repos/vllm-project/vllm tag --contains <fix-sha> --list 'v0.*' | sort -V | head -3
```

The `--list 'v0.*'` is not cosmetic: that clone also carries `proto-v0.1.0 … proto-v0.5.0`, which sort
before every release tag, so the bare form reports `proto-v0.1.0` as the first tag containing the fix
(a bare form reads `proto-v0.1.0` for a fix whose first release tag is `v0.15.0`).

Verified example (#16237): `get_open_port()` probes a port and closes the socket before
`MessageQueue` binds it, so concurrent workers can collide. The fix `f7ef489e93`
("[Bugfix] Fix ZMQ port TOCTOU race in shm_broadcast MessageQueue (#50960)") replaces the probe with
`bind("tcp://ip:0")` + `zmq.LAST_ENDPOINT`, and `git tag --contains f7ef489e93` starts at
`v0.28.0rc1` — so a report on vLLM `v0.23.0` pre-dates it, and the plugin's release notes never
mention it. That case also carried a reporter-written "preliminary diagnosis" quoting a symbol that
exists in no ref; treat hand-written snippets as `reported` evidence and verify them at the tag
(`version-pinning.md`, "Reporters' code quotations").

The same applies to torch_npu/CANN-side gates: when the fix needs a newer runtime, the deliverable is
"upgrade X to version Y", proven by the gate the code actually tests.

## Verify a cited fix by its hunk, not its subject

A maintainer's "fixed by PR #NNNN" or a bot's "see #MMMM" is a claim about a diff, and titles mislead
in both directions. **Verified example (#8664):** a maintainer named vLLM PR #37160 as the fix for
`ValueError: Counters can only be incremented by non-negative amounts`. The PR is titled *"[Feat][v1]
Simple yet General CPU KV Cache Offloading"*, which looks unrelated, and a recorded run concluded from
the title that the maintainer had mis-numbered it and named the later metrics rework #37460 instead —
wrong. `git show 91e4521f9f -- vllm/v1/metrics/stats.py` shows #37160 adds the `max(0, …)` clamp
itself, with the comment that `local_cache_hit` "can go negative after preemption" because
`num_cached_tokens` is a one-time snapshot never reset on preemption while
`num_external_computed_tokens` is overwritten on re-scheduling. #37460 (`d3af8c1831`) only replaced
that arithmetic with `PrefillStats` + `assert num_cached_tokens <= num_prompt_tokens` two weeks later.

```bash
git -C "$REPO" show <sha> -- <the failing file>       # the hunk, not the subject
git -C "$REPO" log --all --oneline -S'<expression from the buggy line>' -- <path>
git -C "$REPO" tag --contains <sha> --list 'v0.*' | sort -V | head -3  # inclusion, in the fixing repo's tags
```

`-S` on the *expression* also finds the lineage the subject hides: for the counter case it returns the
feature PR that added the clamp and the later accounting rewrite that made it unnecessary.

**A second verified case, in the other direction (#10466).** The double TP all-reduce that garbles
MiniMax-M2 was removed by `57c582440`, whose subject is *"[BugFix][v0.23.0] Avoid partial MiniMax
parameter arguments (#11505)"* — a streaming tool-call fix; its body says in one closing sentence that it
"also removes the obsolete MiniMax worker-side manual TP all-reduce call … leaving it in the release branch
blocks MiniMax TP8+EP startup". `git show --format= 57c582440 -- vllm_ascend/patch/worker/patch_minimax_m2.py`
shows the deleted `if self.tp_size > 1: … maybe_all_reduce_tensor_model_parallel(…)` pair, and
`git tag --contains` puts the first fixed tag at **`v0.23.0rc1`**. Neither the subject nor the issue thread
would have led here: the reporter's own guard PR #10443 stayed open and was superseded, so "the cited PR is
unmerged" is not "unfixed" — `git log -S'<the removed expression>' <old-tag>..<new-tag> -- <path>` is one
call and answers it.

**A release-branch cherry-pick carries a different PR number — search the subject, not the number.**
Verified on #15985: the thread named three mainline PRs (#16712/#16713/#16714), and the commits that
actually reach the fixed tag are `155f68974 [v0.26.0rc2][BugFix][SpecDecode] Handle empty DSpark context
(#16712)` and `69e54c8b7 [v0.26.0rc2][bugfix] Validate stop and allowed token ids (#16714)` — same PR
numbers here, but the same PR also appears on other release lines under *other* numbers (`1653ec76b`
#16719, `6f9e86a71` #16720, `2ed6cbbaf` #16539, `7719bd3a1` #16161), so `git log --grep='#16712'` alone
finds one lane only. The durable query is on the **PR title** across all refs, then ancestry on the tag
the thread recommends:

```bash
git -C "$REPO" log --all --oneline -i --grep='handle empty dspark context'
git -C "$REPO" tag --contains 155f68974 | sort -V | head -2      # -> v0.26.0rc2
```

**`git log --grep='#NNNN'` also matches commit *bodies*, so a release-notes commit shadows every PR it
cites.** Verified on the #7401/#7448 class: `git log --all --format='%h %s' --grep='#7235' -1` returns
`adc57c595 [release] Add GLM5 known issue for 2-node PD mixed deployment (#7436)`, whose body lists
"fixed in PR: - #7235 - #7290", and the same commit is returned for `#7290` — so a run that trusted `-1`
cited both PRs as fixes without ever reading their hunks. Anchor the number to the subject's closing paren
instead, and keep the whole list in view:

```bash
git -C "$REPO" log --all --format='%h %s' --grep='(#7235)$'   # -> 4e62a2ae1 [Bugfix] fix TransposeKvCacheByBlock op error report in plog (#7235)
```

The same anchoring matters when the number is *only* in a body legitimately: a fix listed as "also fixes
#N" then shows up as a release-notes or feature commit, and the hunk (`git show <sha> -- <path>`) decides
whether it is the fix or a citation of it.

## Ancestry says no, the tag's tree says yes

`git merge-base --is-ancestor <mainline-sha> <tag>` answers "is this exact commit in the tag", not
"does the tag contain this change". Release branches carry cherry-picked copies under new shas, so a
false ancestry result is not absence. **Verified example (#8664):** `91e4521f9f` is not an ancestor of
vLLM `v0.19.0`, yet `git show v0.19.0:vllm/v1/metrics/stats.py` contains the clamp, because the branch
carries `1dbbafd3f3`, its cherry-pick of the same PR
(`git log -S'local_cache_hit += max' origin/releases/v0.19.0 -- vllm/v1/metrics/stats.py`).

```bash
git -C "$REPO" show <tag>:<path> | grep -n '<the changed expression>'
git -C "$REPO" log --oneline -S'<changed expression>' origin/releases/<line> -- <path>
```

Both directions matter for the deliverable: ancestry alone can declare a fix absent from the version a
maintainer recommended (when it is present), just as a tag's tree can show a fix present while the
reporter's own build predates it. Read the reported line and the recommended version's line.

## Thread says 「已解决」 and names no fix: use the reporter's own build sha

A closed issue whose only resolution comment is 「已解决」/"resolved" gives no lever, but the
`### Your current environment` block usually does: a release-branch build is pasted as
`vLLM Ascend releases/v0.23.0 6cee2e16308ae34aedb7c3b7ed2df0d42284713e`. That sha is a **needle**, and it
turns an open-ended search into one ancestry query.

1. **Find candidate fixes by path, not by symptom words.** One path-scoped log over the window after the
   reporter's build: `git log --oneline <their-sha>..origin/main -- <the module the symptom lives in>`.
   For #12339 that is `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`, and the title
   that matched the symptom was `[BugFix]Fix precision issues caused by incorrect reordering timing
   leading to TP inequality`.
2. **Expect one change shipped several times.** `git log -S'<a symbol the fix introduces>' --oneline --all
   -- <path>` lists the mainline commit *and* every branch cherry-pick. For #12339 that returned three:
   `86217325b` (#12371, `[BugFix][v0.23.0]`, "Backport of #12359"), `b6826ec33` (#12382,
   `releases/v0.24.0rc`) and `1db626510` (#12359, mainline). **The branch copy is the one that matters to
   the reporter**, and `git tag --contains <sha> --list 'v0.2*' | sort -V` names it.
3. **Compare with their sha, twice.**
   `git merge-base --is-ancestor 86217325b <their-sha>` → not an ancestor, while the same sha *is* an
   ancestor of tag `v0.23.0` ⇒ their build predates the branch fix and any rebuild of that branch after
   the fix date carries it. Also run `git log -1 --format='%h %ad %s' <their-sha>`: the build date next to
   the fix date is what makes "their build is 8 days older" a fact rather than a guess.
4. **Then reply.** The deliverable is "the fix exists in your branch as `<sha>`/PR #N, first tag `vX.Y.Z`,
   your build `<sha>` is 8 days older" plus the mechanism from the fix's own PR body — that body is
   usually the only place the mechanism is stated, and it is one `git show <sha>` away. Do **not** quote
   the mainline first tag alone: `v0.27.1rc1` for a 0.23 reporter reads as "wait for 0.27" and is the
   wrong answer to a question their own branch already fixed.

## When the report is a same-line regression (`rc` → final)

Some reports have no traceback and no bad log line: "`v0.18.0` scores 3.5 points lower than `0.18.0rc1` on
AIME" (#8932), or a feature the rc had and the final lost, or a throughput/hit-rate delta between two
releases. There is no code window to open; the deliverable is the range, the direction and the lever. The
offending change is **inside** the range, and every cherry-picked fix in it is a fix the bad version *has*,
so a cited fix can never be the regression. One listing call, then the reply.

1. **List the range once.** `git log --oneline <rc>..<final> | wc -l` (108 for `v0.18.0rc1..v0.18.0`, ~25
   of them non-doc/CI) and `git log --oneline <rc>..<final> --no-merges -- vllm_ascend`
   (`--invert-grep --grep='^\[Doc\|^\[CI'` is a cheap filter). Re-filtering the same range with different
   `--grep` words is the recorded waste (four `git log` calls over one range in the batch-11 run); if the
   first listing did not settle it, another filter over the same commits will not either.
2. **A cherry-picked fix cannot be the suspect by itself.** Most of the range is fixes, which move the
   final *up*. The commits that can lower a score are the ones that *remove* or *re-default* something:
   `git log --oneline <rc>..<final> -i --grep='revert\|rollback'` — in `v0.18.0rc1..v0.18.0` that returns
   `6bdc72949` (`Revert "[v0.18.0][BugFix] Fix dimension mismatch error when SP padding causes
   num_tokens_padded != num_tokens_unpadded"` #8413, whose original fix `f2956ce94` #8133 is in the same
   range), `c124e8df0` (revert of a `dispatch_ffn_combine` rollback) and `fcf4d477a` (rollback of three
   `dispatch_ffn_combine` optimisations, #8534). Read each body before naming it: a rollback is usually
   itself an accuracy fix.
3. **Defaults/pins, not only code.** A release can change a default middleware flag, a dependency pin or a
   capture size between rc and final: `git diff <rc> <final> -- vllm_ascend/envs.py
   vllm_ascend/ascend_config.py requirements.txt pyproject.toml` is one call and covers the common ones.
4. **The thread's named PRs are the status, not the suspect.** "Resolved on the latest main, PR #8560 /
   #8764" gives the workaround as soon as each PR's ancestry is known:
   `git merge-base --is-ancestor <sha> <reported-tag>`. Verified for #8932: `a5622f56d` (#8560, xmask for
   `dispatch_ffn_combine`, w8a8 branch) is **not** in `v0.18.0`, while #8764's cherry-pick `9a0b786f2`
   ("Fix race in non-blocking num_accepted_tokens", which appears in the `rc..final` log) **is**. That
   asymmetry is the answer — the reporter's build lacks the change the maintainer pointed at, and the
   symptom (a quantised-MoE accuracy delta) is exactly what that operator controls. Resolve the sha from
   the log rather than from a PR field — a `fields` projection that asks for `merge_commit_sha` can come
   back as `missing_projection_field`, and an unverified sha makes the ancestry result meaningless.
5. **Stop there.** Which in-range commit moved the score is not answerable from the clones; the reporter
   owns the benchmark. Ship the range, the named fixes with their ancestry, the lever ("a build containing
   #8560 — i.e. after `v0.18.0` — or stay on `0.18.0rc1`") and the artefact ranking (quantised vs bf16,
   sampling params, dataset revision) as `hypothesis` with the falsifier being a re-run on the newer build.

Do not scan subjects by symptom word across the whole gap (`HEAD` may be hundreds of releases ahead): the
rc..final range is the only place the regression can be. If the range yields no plausible offending change,
the honest answer is the maintainer's lever plus "not bisected in this range", not a ranked list.

## A path-scoped range log is the highest-yield single query for a silent failure

When the symptom carries no exception and therefore no module name — a worker that dies silently, a
transfer that stalls, a count that stops advancing — the owning directory is the only search key you
have, and its commit subjects after the report already name the mechanism in plain language:

```bash
git -C "$REPO" log --oneline <tag>..HEAD -- vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/
```

One call over the pool package returned 25 subjects for a `v0.23.0rc1` report, several of them directly
on-symptom ("Fix segmented bulk KV transfers", "Isolate Memcache transfer failures", "Invalidate stale
load plans and bound worker transfers"). This is the *first* query for that class, before any file read —
and it is not the `--grep='<symptom word>'` over thousands of commits that `answer-and-claims.md`
forbids: the path bounds it to a subsystem and the count to a screen. Read the diff of at most one of them.

## Do not cite a reverted fix

Reverts happen: `ba945e40f Revert "[BugFix] Solve the d2h free_bubble in glm5.2 dspark" (#17184)`
(#17525), and `6f8c631cc` (#17480) was later reverted by `42ef1c363` (#18029). Before using a commit
as the answer, check

```bash
git -C "$REPO" log --oneline --all --grep='Revert' | grep -i '<keyword>'
```

and whether the code change is still present at `origin/main` or at the fix's release branch. A
reverted fix is evidence of an area in flux, not a resolution.
