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
sed -n '1,60p' /workspace/history/<issue>/<id>.md    # flat layout: /workspace/history/<issue>.md
```

With a per-case history root the same issue id can appear once per case directory
(`history/<issue>/<id>.md`); `history-titles.sh` now reports `hits / distinct issue
ids` and prints one line per id, so a `limit` is not spent on copies. Read one copy
and cite the id.

`grep -rn` / `rg -n` are safe; `rg -rn` and `rg -rl` are not (`-r` is `--replace`, so the
matches are rewritten to `n`/`l` and the output looks like corrupted source). `grep -rn`
over the same tree is the tie-breaker when output looks mangled.

Quote precedent as `#<id>: <one-line fact>` (e.g. "upstream parser fix tracked in vllm PR
34779, referenced by #7061/#7094"). Distinguish "the same symptom was reported" from "the
same cause was confirmed".

Checklist for the final answer: repo+ref+commit cited once; every code claim has
`path:line`; precedent ids quoted; missing artefacts listed; OCR/snapshot limits stated.
