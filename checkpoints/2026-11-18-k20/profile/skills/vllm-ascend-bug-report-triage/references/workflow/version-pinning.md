# Version pinning: from the report's version string to the ref it ran

## Inputs

Extract from the report's title, body and environment block — not from the task prompt: product
name and version string(s), model + quantisation, hardware class (A2/A3/A5/950, 310P), deployment
mode (single node, PD disaggregation, PP/TP/DP degrees), the failing stage, and the exact
`--additional-config` / `--speculative-config` JSON. Config keys are search tokens; the version is
the ref.

Signals to collect and reconcile:

- the `collect_env.py` block (vllm-ascend, vllm, torch-npu, CANN);
- the log banner, which can disagree with the prose: one report claims `vllm-ascend 0.11.0` while
  its log prints `vLLM API server version 0.10.2`. Record both;
- the docker image tag / wheel in the start command.

Gate: the reported ref must exist in the clone. If it does not, say so and analyse the nearest
available ref, explicitly labelled as a substitute.

## Reporters' version strings are not tag names

| report says | tag |
| --- | --- |
| `vllm-ascend:0.26.rc1` | `v0.26.0rc1` |
| `0.23.0rc1` | `v0.23.0rc1` |
| `0.20.2rc1` | `v0.20.2rc1` |
| `0.21.0rc1-310p` (image suffix) | `v0.21.0rc1` |
| `0.9.2RC1` (`collect_env` prints the suffix in upper case) | `v0.9.2rc1` |
| `0.23.0/0.26.0rc1` (two versions in one title) | triage the newer, then check the older still has the defect |

Naming anomalies exist: `v0.28.0.rc1` and `v0.29.0.rc1` carry a double dot while `v0.30.0rc1`
does not, so an exact-match lookup for `0.28.0rc1` fails. Normalise by removing non-alphanumerics
apart from dots and `rc`/`post` before matching: `scripts/version_archaeology.py` does this, and the
dependency-free `scripts/version_ref.sh`/`resolve_ref.sh` normalise the same way, **lower-casing the
suffix first** (`0.26.rc1` → `v0.26.0rc1`, `0.9.2RC1` → `v0.9.2rc1` — the old form resolved nothing and a
recorded run then treated its own unresolved ref as a dead end),
`0.7.3.post1` → `v0.7.3.post1`, dotted tags such as `v0.29.0.rc1` accepted as a fallback) while also
printing near tags, release branches and the default-branch HEAD for contrast.

A **development build** carries its own ref: `vLLM Ascend Version: 0.19.1rc2.dev849+g05e095a20
(git sha: 05e095a20)` is a setuptools-scm string, and the `g<sha>` suffix is the commit that was built.
`git cat-file -t 05e095a20` in the clone answers whether you can analyse it exactly; that commit turned
out to be `[CI] main2mian v0.24.0 (#11227)`, an ancestor of `v0.23.0rc1`+, so its `Dockerfile.a3`,
`ascend_config.py` and attention code are all readable at the built state (#12059). Use the sha before
falling back to "nearest tag, labelled as a substitute", and when an image tag and a `collect_env`
version disagree, say so and fingerprint the commit — that is the conflict this rule exists for.

A **spoken minor** is not a ref either. Threads say 「v0.20 解决了」 or "on 0.24"; the clone may carry no
`v0.20.0` at all (it has `v0.19.1rc1` and `v0.20.2rc1`), so `git log v0.18.0..v0.20.0` answers
`fatal: ambiguous argument 'v0.18.0..v0.20.0': unknown revision or path not in the working tree` — a
*tag-name* failure, not evidence that no fix exists — a run that treats it as one spends two batches on a
version that was never a tag. Enumerate
first, then build the range from tags that exist, and name the enumerated tag in the reply ("the `v0.20`
line here means `v0.20.2rc1`").

Enumerate instead of guessing:

```bash
git -C "$REPO" tag --list | grep -E '0\.(20|23|26|30)'
git -C "$REPO" tag | sort -V
git -C "$REPO" branch -r | grep -E 'releases/|rfc/'
```

## Ref families

- `vX.Y.ZrcN` / `vX.Y.Z` — release tags. The tag commit is the rc cut or the release commit.
- `origin/releases/vX.Y.Zrc` — the release branch. Commits here with `[v0.23.0] ...`,
  `[BugFix][v0.20.2rc] ...`, `[Cherry-pick][releases/v0.30.0] ...` subjects are backports: the
  maintainers judged them important for that release. A branch can also run ahead of its last tag
  (`releases/v0.31.0rc` has no tag), so for an rc report check `git log -1 origin/releases/<branch>`
  before assuming the tag is the newest state of that line.
- `origin/rfc/<version>-bugfix` (seen: `rfc/0.26.0rc1-bugfix`) — ad-hoc collection of candidate
  fixes for an rc. Worth one `git log --oneline origin/rfc/<version>-bugfix | head`.
- **Release lines are not linearly ordered.** A maintenance line keeps shipping after the next minor
  branches, so tag names and dates do not order two releases: `git merge-base --is-ancestor v0.9.2rc1
  v0.9.1rc3` returns 1, `git merge-base v0.9.1rc3 v0.9.2rc1` is `706de023`, `v0.9.2rc1` is dated
  2025-07-11 (`vLLM v0.9.2`, image CANN `8.1.rc1`) and `v0.9.1rc3` 2025-08-22 (`vLLM v0.9.1`, CANN
  `8.2.rc1`), with 167 commits in `v0.9.1rc3..v0.9.2rc1`. For "why is version A faster / fixed in
  version B" questions, get the merge-base first and diff per line; a conclusion built on "the newer
  version" collapses when the two tags are on different branches with different vLLM and CANN
  pairings (`../vllm-ascend/version-pairing.md`).
- `main` — always ahead. Measure the gap so you do not quote HEAD by accident:
  `git -C "$REPO" rev-list --count "$TAG"..origin/main` (1902 for `v0.26.0rc1` and 2342 for
  `v0.23.0rc1` in the clone these notes were written against).

Pair the release with upstream vLLM from the release notes' dependency section, then use that tag
in the `vllm` clone when a frame lands in `vllm/...`
(`git show v0.26.0rc1:docs/source/user_guide/release_notes.md | grep -n -A8 '### Dependencies'`).
The full pairing table and runtime pins are in `../vllm-ascend/version-pairing.md`.

## Fingerprint the ref from the traceback line

A traceback carries `path` + `line`, and that line number decides which tree was installed when
prose and banner disagree. Print the failing statement's line at every tag and pick the match:

```bash
bash scripts/ref_line_of.sh /workspace/repos/vllm-project/vllm-ascend \
  vllm_ascend/quantization/quant_config.py "quant_description\[prefix \+ '\.weight'\]"
```

Verified example, issue #3494: prose says `vllm-ascend 0.11.0`, the log banner says
`vLLM API server version 0.10.2`, and the traceback names
`quant_config.py:147 in is_layer_skipped_ascend`. That statement sits at line 147 only in
`v0.10.2rc1` (141 at `v0.10.0`/`v0.10.1`, 152 at `v0.11.0rc0`, 161 at `v0.11.0rc1`–`rc3`, 170 at
`v0.11.0`), so the reporter ran a 0.10.2-era build. Analysing `v0.11.0` still reached the right
statement there only because the code had not changed — the line numbers had.

Caveats: equal line numbers occur in unrelated tags (`v0.9.1` also gives 142 for a nearby
statement), so combine the fingerprint with the log's upstream vLLM version and a symbol grep.
After a refactor the statement may not exist at HEAD at all, which is itself a finding about when
the code moved.

The **line number, not the spelling**, is the fingerprint — and it must be checked, not assumed:

- A misspelled message is usually a paste artefact. #12891's body reads `Loading draftter model..` at
  `model_runner_v1.py:3686`; `git grep -n "Loading draft" v0.23.0rc1 -- vllm_ascend` prints
  `model_runner_v1.py:3686: logger.info("Loading drafter model...")` — same file, same line, one doubled
  letter. A run that greps the misspelling in both clones finds nothing and reasons from "this string
  exists in no release"; grep the stable fragment and compare the number instead.
- A line number that disagrees with the tag is the real signal. #6765 quotes
  `glm4_moe_tool_parser.py:104`; the statement is at **`:104` in vLLM `v0.13.0`** and at `:118` in
  `v0.14.0rc1`/`v0.14.0`/`v0.14.1` (the `_is_string_type` helper sits above it), while the image tag
  `v0.14.0rc1`'s own `Dockerfile:51` says `ARG VLLM_TAG=v0.14.1` — so the file the reporter pasted is a
  0.13.0-era one (an image built before that `ARG` moved, or a different tag than the prose), and the guard
  has to be read at `v0.13.0`; the fix's absence from every `v0.14.x` is what the answer rests on.

## A path the tag does not contain: a newer build, or a name that never existed

A quoted `vllm_ascend/<path>` that is missing at the reported tag has two readings, and one call each
decides which. Do not fill the gap from HEAD.

- **The tag predates the file, so the image carries newer (vendor or private) code.** Ask the history:
  `git log --oneline --diff-filter=A -- <path>` gives the commit that added it and `git tag --contains <sha>`
  the first tag that ships it. Worked case #16365: the report (GLM-5.3-Flash, image `glm-5.3-flash`, tag
  `v0.23.0`) quotes `vllm_ascend/models/glm5_next.py`; the directory `vllm_ascend/models/glm5next/` is added
  by `78f3a90f2` (#15127, 2026-09-03) and first appears in **`v0.28.0.rc1`** — five minor releases after the
  tag. The tag-side answer is then "this path is not in your release; it ships from `v0.28.0.rc1`", plus the
  nearest release class labelled `hypothesis` — never a transplanted `path:line`.
- **The clone never carried that name at any tag, so the mechanism must be read on an analogue.** Reports
  routinely name a module or model no clone has — a private build's `models/glm5_next.py`, a `Qwen3.6-…`
  checkpoint directory — and `git ls-tree` returns nothing. That empty listing is not evidence of a missing
  feature (the same trap as an empty grep for the wrong word): the mechanism is usually readable on a
  sibling model file, and the report may describe a build that is not public at all. Verified example: at
  `v0.23.0` `vllm_ascend/models/` holds only `deepseek_v4.py`, `deepseek_v4_mtp.py`, `layer/`,
  `llama_eagle3_vwn.py`, so a report's GLM state-cache class has to be read through `deepseek_v4.py`'s
  `AscendCompressorStateCache` — same spec/manager pattern, different model name. Say which analogue you
  substituted and keep the reporter's own name in the claim, so a later reader can tell `verified` from
  `mapped`.

One `git ls-tree -r --name-only <ref>:vllm_ascend/<pkg>` answers the layout question first; an empty
`git grep` for a *word* and an empty `ls-tree` for a *name* are both about the spelling before they are
about the feature.

## Reporters' code quotations are claims, not the ref

A traceback's line numbers can be fingerprinted (above); a reporter's *hand-written* snippet cannot.
Treat it as `reported` evidence: read the file at their ref before repeating or building on it.

- Worked case #16237: the 「初步定位」 section quotes `get_open_port()` with a process-local
  `_used_ports` de-duplication set. That symbol exists in **no** vllm ref checked (`v0.22.1` …
  `v0.30.0`, `main`) and nowhere in the vllm-ascend clone; the real `v0.23.0` body just calls
  `_get_open_port()` and honours `VLLM_DP_MASTER_PORT`. The mechanism the reporter describes (probe,
  close, re-bind later) is still correct — the code quotation is not from their tree.
- Consequence for the write-up: confirm the *call chain* they quote at the tag (in that case all four
  `file:line` frames matched `vllm` `v0.23.0` exactly), then correct the snippet explicitly. Saying
  "your quotation of `_used_ports` does not exist at 0.23.0; the race is real without it" is a
  finding; silently reusing their snippet imports a wrong claim into your answer.
- A symbol that exists at no ref in the clones is also a version signal: the reporter may be reading a
  PR, a fork, an internal patch, or an earlier mis-diagnosis. Name it, and do not let it choose the ref
  when the traceback lines disagree.
- **A quoted `vllm_ascend/<path>` that the tag does not contain** is a date or a substitution, never a
  gap to fill from HEAD — the two branches and their one-call tests are above ("A path the tag does not
  contain").

## Reading code at the tag

The clones under `/workspace/repos/<owner>/<repo>` are read-only; do not check out and do not
create worktrees inside them.

```bash
# one file, bounded window
git -C "$REPO" show "$TAG":vllm_ascend/attention/sfa_v1.py | sed -n '1980,2010p'
# in-place reads and searches, no extraction
git -C "$REPO" grep -n -F '<symbol>' "$TAG" -- '*.py'
git -C "$REPO" ls-tree -r --name-only "$TAG":vllm_ascend/worker
# whole tree, greppable
mkdir -p /workspace/scratch/$TAG && git -C "$REPO" archive "$TAG" | tar -x -C /workspace/scratch/$TAG
```

`scripts/export_ref.sh <repo> <ref> [dest]` materialises a tree under `/workspace/scratch/refs/`.
`git archive` respects `.gitattributes` export-ignore, so a snapshot can miss files the tag
tracks — fall back to `git show "$TAG":<path>` when something looks absent. `git ls-tree` first is
the cure for guessed paths.

### When the frame is in `vllm/` — read that clone, never recall it

A good share of tracebacks on this stack end in upstream vLLM (`vllm/config/vllm.py`,
`vllm/v1/worker/…`, `vllm/distributed/kv_transfer/…`), and the reporter's installed vLLM is a *separate*
version from `vllm-ascend`. Read it from `/workspace/repos/vllm-project/vllm` at the tag that matches
the report:

```bash
git -C /workspace/repos/vllm-project/vllm show <vllm-ref>:vllm/config/vllm.py | sed -n '/def get_layers_from_vllm_config/,+24p'
git -C /workspace/repos/vllm-project/vllm rev-parse --verify <vllm-ref>   # in the same call
```

Choose `<vllm-ref>` in this order: the tag `ref_snapshot.sh` prints (`.github/vllm-release-tag.commit`
in the vllm-ascend tree), else the version in the report's `collect_env` / log banner
(`vLLM API server version 0.14.1`), else the `site-packages` path line in the traceback. The two
adjacent line numbers in the traceback are a cheap fingerprint: for `get_layers_from_vllm_config`,
`return {` at `:1483` and the `isinstance(...)` line at `:1486` mean the `v0.14.0`/`v0.14.1` body, so a
report naming both `vllm-ascend 0.14.0rc1` and vLLM `0.14.1` is consistent and the clone at `v0.14.1`
is the one that ran.

Why this is a rule and not advice: these functions are rewritten across releases, so a remembered body
is both expensive and wrong. A recorded run spent most of its allowance on remembered
variants of `get_layers_from_vllm_config` and `static_forward_context` — `no_compile_layers` or
`compilation_config.static_forward_context`? `vllm/config/vllm.py` or `forward_context.py`? — then ended
with zero answer characters. The single `git show` above answers it in one line of output.

For one ref's facts in one call, `scripts/ref_snapshot.sh <ref> [repo-path-or-name]` prints the paired vLLM
tag, the torch/torch-npu/triton-ascend pins, the package layout, whether the fast-moving paths
exist, whether the 310P model-runner V2 (`_310p/worker/v2`) exists and which runner the 310P worker
instantiates, the NZ layout policy sites, and the commits since.
`scripts/version_archaeology.py` combines version → refs, the ancestry gap to main, per-keyword hits
at the tag, and keyword history:

```bash
uv run --no-project python scripts/version_archaeology.py \
  --repo /workspace/repos/vllm-project/vllm-ascend --version 0.26.rc1 \
  --keyword enable_sparse_li_c8 --keyword dspark
```

## Prove membership, not chronology

Commit dates lie (cherry-picks and backports are dated later than the branch point). Use:

```bash
git -C "$REPO" merge-base --is-ancestor <sha> "$TAG"; echo $?   # 0 = contained in the tag
git -C "$REPO" tag --contains <sha> --list 'v0.*' | sort -V     # which release tags contain it
git -C "$REPO" log -1 --format='%h %ci %s' "$TAG"              # when the tag was cut
```

Keep the `--list 'v0.*'` in the **vllm** clone: it also carries six `proto-v0.1.0 … proto-v0.5.0` tags,
which sort before every release tag, so a bare `… | sort -V | head -5` reports `proto-v0.1.0` as "the
first tag" (the `--list` form answers `v0.15.0`).

Worked example: for a report against `v0.20.2rc1`, `merge-base --is-ancestor` showed the
release-branch backports `7cb5d6180` (#10259) and `91e6b539d` (#10308) were **not** in the tag
while `367b8e62d` (#9946) was — and their mainline originals `a3b10cb6f` (#10043) and `1a44dc5f2`
(#10307) are not in any 0.20 release either. So an rc tag can miss backports that the release
branch received days later; "the branch has the fix" does not help the reporter. That single check
converted "maybe fixed already" into a candidate root cause.

`scripts/ref_ancestry.sh <repo> <ref> <commit>...` prints IN/OUT per commit with its date and
subject.

## Time-stamp the report

Compare the issue's `created_at` with the tag date and with candidate fix dates. A report created
before a fix cannot be explained by that fix; a report created after a fix that is not in the tag
*can* be explained by it, provided the repro conditions match.

## Read-only clone traps

- `git checkout <branch>` and `git worktree add` fail, sometimes silently. A swallowed failure
  (`2>/dev/null`) leaves the tree on `main` while later reads look like "the code": one
  investigation did exactly this (`git checkout -q origin/releases/v0.21.0rc 2>/dev/null`) and
  trusted the result for two turns. If you attempted a checkout, verify with
  `git rev-parse --abbrev-ref HEAD`, and prefer ref-only reads.
- A bare `sed -n 'A,Bp' <path>`, `grep -rn …`, `wc -l <path>` or `cat <path>` inside the clone reads the
  **working tree** (`main`), not the tag, so line numbers from a traceback will not line up and the quoted
  code belongs to another release. Recorded runs made this read and lost a turn to the offset mismatch; the
  same read on `.github/vllm-release-tag.commit` answers `v0.32.0` (main) instead of the tag's `v0.23.0` and
  sent one run's whole `vllm/` evidence to the wrong ref (`../vllm-ascend/version-pairing.md`). Use
  `git show <ref>:<path> | sed -n …`, `git grep -n <pattern> <ref>`, or a tree from
  `scripts/export_ref.sh`.
- `git grep <pattern> <ref>` on a ref that does not exist prints `fatal: unable to resolve revision`
  on stderr and **nothing** on stdout, indistinguishable from "symbol absent". This clone has no
  `v0.19.0rc1` (the tags jump `v0.18.0` → `v0.19.1rc1`); confirm with `git rev-parse --verify <ref>`
  in the same call before concluding absence, and read stderr rather than only the text block.
- A **capped listing is not a tree.** `git ls-tree -r --name-only <ref> | grep -i minimax` sorts `docs/` <
  `tests/` < `vllm_ascend/`, so a ten-line limit keeps docs and tests and drops every package hit — one
  delivered answer concluded `vllm_ascend/patch/worker/patch_minimax_m2.py` was "gone from main" when
  `git cat-file -e HEAD:<path>` succeeds there and the file merely no longer carries the MiniMax MoE hook
  (`../symptoms/flashcomm1-and-moe-reduce.md` §B). Ask existence directly
  (`git cat-file -e <ref>:<path> && echo EXISTS || echo ABSENT`), never through a capped pipeline, and
  distinguish "file absent" from "symbol absent in a file that exists".
- `git describe --tags` does not identify the tree: in the vllm-ascend clone it prints
  `v0.19.1rc1-3204-g<sha>` for a v0.30-era tree whose paired vLLM is v0.32.0, and
  `git merge-base --is-ancestor v0.30.0 main` says no — release tags there are not ancestors of
  `main`. Never infer the version from `describe`.
