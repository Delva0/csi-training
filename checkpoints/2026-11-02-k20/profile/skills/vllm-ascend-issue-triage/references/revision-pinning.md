# Pinning a reported release to a source revision

Verified against the read-only clones at `/repos/vllm-project/vllm-ascend` (HEAD `1bd23a0a9`,
2026-10-30) and `/repos/vllm-project/vllm` (HEAD `5423fb8b93`) on 2026-11-01. The clones are
read-only and sit on `main`; the report's version is a tag (or a `releases/*` branch) that is
usually not merged into `main`. Read the reported code only from the tag; never infer it from
the working tree.

## Prove the tag by line number

The traceback in the report contains file:line:statement for every frame. Reproduce it:

```sh
cd /repos/vllm-project/vllm
git show v0.21.0:vllm/v1/core/kv_cache_manager.py | sed -n '355,372p' | nl -ba -v355
```

Measured against issue vllm-project/vllm-ascend#10724 (vLLM 0.21.0 report):

| frame | tag `v0.21.0` content at that line | verdict |
| --- | --- | --- |
| `vllm/v1/core/kv_cache_manager.py:362` | `self.coordinator.remove_skipped_blocks(` | exact match |
| `vllm/v1/core/single_type_kv_cache_manager.py:426` | `self.block_pool.free_blocks(removed_blocks)` | exact match |
| `vllm/v1/core/kv_cache_utils.py:337` | `assert last_block is not None` | exact match |

`main` does not match (same call at `kv_cache_manager.py:563`), so line numbers alone
discriminate tags. This takes two commands and is stronger evidence than the version string or
`collect_env.py` alone. Do it before reading any logic.

Two ways a line match can still mislead, both worth one line in the answer:

- **The matched tag can be older than the image tag's own upstream pin.** For
  vllm-ascend#6765 the three frames (`glm4_moe_tool_parser.py:104` `findall`, `:122`
  `logger.exception`, `:171` "Failed to extract any tool calls") reproduce upstream vLLM
  `v0.13.0` exactly, while the reported image's `v0.14.0rc1:Dockerfile` is
  `ARG VLLM_TAG=v0.14.1` and that tag puts the same three statements at `:118`, `:137`, `:186`.
  Say both: what the paste's layout matches and what the image tag claims to bundle. Do not
  silently re-pin the report to the older tree, and do not discard the image tag either — the
  mismatch is itself a finding (stale image, local rebuild, or a paraphrase).
- **The exception text has to be producible by the statement.** `re.findall(None)` on Python 3.11
  raises `TypeError: expected string or bytes-like object, got 'NoneType'`, so a paste saying
  `expected string or buffer` did not come from stdlib `re` there — that is the `regex` C module's
  wording (`regex` 2.5.148 raises exactly it, and vLLM's tool parsers import that module as `re`;
  a 2026 `regex` raises `a bytes-like object is required`). Wording like this identifies the
  *implementation or interpreter*, not the release: check it against the statement before using the
  line number. The parser defect this example comes from is in
  [serving-toolcall-reasoning-streaming.md](serving-toolcall-reasoning-streaming.md).

When the baked version string and the traceback disagree, the line match wins and both belong in
the answer. Issue #9507's `collect_env.py` printed `vLLM Ascend 0.19.1rc1`, but six frames
(`worker/worker.py:395`, `worker/model_runner_v1.py:1359`, `:1796`, `:1836`,
`compilation/acl_graph.py:233`, `attention/context_parallel/attention_cp.py:324`) matched `v0.18.0`
exactly and shifted at `v0.19.1rc1`; the reporter's plain "0.18_release" was the accurate statement
and the printed rc came from the installed distribution metadata, not from the tree that raised.

For vllm-ascend patches, match the same way on the release branch, e.g.
`git show origin/releases/v0.21.0rc:vllm_ascend/patch/platform/patch_balance_schedule.py`.

**Read the frame's full path before pinning: basenames collide.** A frame under
`/usr/local/python3.*/site-packages/torch_npu/...` belongs to the torch_npu wheel, which is not one
of these clones, so it cannot be line-pinned here — say that and pin a plugin frame instead. In
#12989 the raise was `torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/acl_graph.py:823/:894`, while
`vllm_ascend/compilation/acl_graph.py:192/:194` at `v0.23.0rc1` is `with torch.npu.graph(...)` /
`output = self.runnable(...)` — plausible enough that a recorded step declared "matches the traceback
exactly" for the wrong file. `acl_graph.py` exists at least in `vllm_ascend/compilation/` and in
torch_npu's `npugraph_ex` (the latter is not in `/repos/Ascend/pytorch` either), so a basename plus a
matching line number is not a pin.

Two plugin statements pin better than one, and a mismatch is itself evidence:

```sh
F=vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
for t in $(git tag --list | sort -V); do
  echo "$t $(git show $t:$F | grep -n 'Error in KVCacheTransferThread' | cut -d: -f1)"           "$(git show $t:$F | grep -n 'Mooncake transfer failed for request' | cut -d: -f1)"
done
```

A report titled `0.23.0rc1` whose pasted frames are `mooncake_connector.py:565` and `:781` matches
`v0.21.0rc1` (563/780) and sits 61/159 lines off `v0.23.0rc1` (626/940): the plugin frames are a
0.21-line build, while the curl's `system_fingerprint: vllm-0.23.0-...` names the upstream vLLM
version. Report the two separately. A **1-2 line residual** means a near-identical tree (say "matches
to within two lines"); tens of lines means a different release line. Table and interpretation:
[pd-disaggregation.md](pd-disaggregation.md).

A report whose only evidence is a single log line can still be pinned when the line is
plugin-specific. `[model_runner.py:904] Loading model weights took 3.3461GB` matches
`vllm_ascend/worker/model_runner.py:904` (`logger.info("Loading model weights took %.4f GB", ...)`)
at tag `v0.8.4rc1`; the same statement sits at `:1000` at `v0.9.0rc1` and `v0.9.1`, and the V1
runner's copy at `model_runner_v1.py:781` (`:1544` at v0.9.0rc1, `:1912` at v0.9.1). Two tells keep
this honest: upstream vLLM v0.8.4 prints `GiB` and only in the xpu/hpu runners (no such line in the
GPU model runner), and the `_v1` file name says which engine ran. Pin the file+line, then say which
engine and which tag window you established.

A `...dev<N>+g<sha>` build string carries its commit directly:
`git rev-parse --verify -q <sha>^{commit}` resolves it whenever the clone has the object
(`0.13.0rc2.dev15+g45c3c279e` → `45c3c279e2b31…`, subject `moe_gating_top_k (#5271)`), so that
commit is the pin even with no matching tag; use `git tag --contains <sha>` to state which release
first shipped it and read code with `git show <sha>:<path>`. The version half of the string can
name a release that does not exist as a tag: `0.20.2rc2.dev0+g367b8e62d` is commit `367b8e62d`,
and `git describe --tags 367b8e62d` prints `v0.20.2rc1` with no `-N-g` suffix — the commit *is*
that tag, and versioningit only derived the next dev version, while `git tag --list | grep -i
'0.20.2'` shows no `v0.20.2rc2` at all. So resolve `rc<N+1>.dev0` to the `rc<N>` tag instead of
searching for the tag the string implies.

## Version, image tag and upstream counterpart

| reported version / image tag | vllm-ascend tag | tag commit | tag date |
| --- | --- | --- | --- |
| `v0.8.5rc1` report | `v0.8.5rc1` | `ec27af346` | 2025-05-06 |
| | `v0.9.1rc3` | `763ed69c1` | 2025-08-22 |
| | `v0.9.2rc1` | `b5b7e0ecc` | 2025-07-11 |
| `v0.11.0rc0` report | `v0.11.0rc0` | `00ba07102` | 2025-09-30 |
| | `v0.11.0rc1` | `c5fe179ce` | 2025-11-10 |
| | `v0.11.0` | `2f1aed98c` | 2025-12-16 |
| | `v0.18.0` | `e18643f8a` | 2026-04-30 |
| | `v0.18.0rc1` | `99e1ea0fe` | 2026-04-01 |
| `...:v0.20.2rc1[-310p...]` | `v0.20.2rc1` | `367b8e62d` | 2026-06-03 |
| `...:v0.21.0rc1[-310p...]` | `v0.21.0rc1` | `80610e443` | 2026-06-12 |
| `...:v0.22.1rc1[-310p-openeuler]` | `v0.22.1rc1` | `5f6faa0cb` | 2026-06-30 |
| | `v0.23.0rc1` | | 2026-07-19 |
| `...:v0.23.0` | `v0.23.0` | `5cb98caaa` | 2026-08-17 |
| | `v0.24.0rc1` | | 2026-08-04 |
| | `v0.25.1rc1` | | 2026-08-19 |
| | `v0.26.0rc1` | | 2026-09-03 |
| | `v0.26.0rc2` | | 2026-09-16 |
| | `v0.27.1rc1` | | 2026-09-25 |
| | `v0.28.0.rc1` | | 2026-09-30 |
| | `v0.30.0rc1` | | 2026-10-12 |
| | `v0.30.0` | | 2026-10-29 |

The upstream vLLM side of the same pairing, verified in `/repos/vllm-project/vllm`: `v0.23.0` is
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (2026-06-14), and the plugin's v0.23.0 release note
states it is "aligned with upstream vLLM v0.23.0". Tag spelling is not regular: both
`v0.23.0.post1` and dotted `v0.28.0.rc1`/`v0.29.0.rc1` exist beside `v0.23.0rc1`, so match with
`git tag --list | grep -i '<version>'` instead of constructing the name.

`-310p`, `-310p-openeuler`, `-a3` and `-a5` are image variants; drop them for the git ref.
`git tag | sort -V` lists `v0.11.0rc2` … `v0.30.0` (tags exist up to `v0.30.0rc2`), so the
mapping holds for later reports too — resolve each one with the tag list rather than trusting
the pattern blindly. Dates and plain `git describe` do not order tags; read them per tag with
`git log -1 --format='%ad %h' --date=short <tag>`. Version order is not date order
(`v0.23.0` was tagged after `v0.24.0rc1`), so never assume adjacency.

The same applies to when a feature or model first became available: release branches are cut and
backported out of order, so "absent in v0.18.0 and v0.19.1rc1, present in v0.20.2rc1" does not
mean "first supported in v0.20.2rc1" unless every tag was checked. DeepSeek-V4 is the recorded
example — main-line `7bce23cc0` (#9270) is contained from `v0.20.2rc1`, while a separate initial
support commit `bbe9bc0d9` (#8648) is contained only in `v0.13.0rc3` (2026-04-24, later in date
than `v0.18.0`). Resolve first availability with
`git log --all --oneline --diff-filter=A -- '<path>'` followed by `git tag --contains <sha>` and
the per-tag dates, and when the reported tree still lacks the symbol, state that the revision is
a preview/floating image rather than that the feature did not exist.

`git describe --tags` on `main` prints `v0.19.1rc1-3204-g1bd23a0a9` because the newer release
tags are on unmerged release branches. Treat that output as noise; the vllm clone behaves the
same way (`v0.33.0rc0-381-...` on a clone whose report is v0.21.0).

The upstream vLLM counterpart carries the same number and is recorded in the tag's own
`docs/source/user_guide/release_notes.md`: the v0.22.1rc1 entry says it is "building on
v0.21.0rc1 and aligning the plugin with upstream vLLM v0.22.1", and the v0.20.2rc1
"Dependencies" section says "Upgraded the matched upstream vLLM baseline to v0.20.2" plus
CANN 9.0.0, triton-ascend 3.2.1 and torch-npu 2.10.0. The note section for a tag can lag the
tag: at tag `v0.21.0rc1` the newest section is still `v0.20.2rc1`. Confirm the pairing through
the next release's notes, a `vllm_version_is(...)` guard in the plugin, or the
build/requirements pins instead of assuming a missing section means no release. These are
intended pins; installed runtime versions need in-container collection (see the
`quay-image-versions` skill for registry-side declarations).

## A build with no tag: daily / nightly / "X月X日日构建"

Reports sometimes name a date-stamped daily build instead of a release ("0.23.0 8月10日日构建
版本"). There is no tag for it; pin it to the release branch by date and state the residual
range:

```sh
for t in v0.23.0rc1 v0.23.0; do echo "$t $(git log -1 --format='%ci %h' $t)"; done
git log --oneline --since=2026-08-05 --until=2026-08-12 origin/releases/v0.23.0 \
  --format='%h %ci %s'
```

`v0.23.0rc1` is 2026-07-19 and `v0.23.0` 2026-08-17, so the build sits between them on
`origin/releases/v0.23.0`, and a fix that landed before the build date (for example
`19c2fcf8d`, 2026-08-09) can be assumed present while anything after it cannot. Say that the
pin is a date window on a branch, not a commit, and that the tags at either end bound it.

Three nearby traps, all cheap to retire:

- **A `system_fingerprint` suffix is not a commit.** `system_fingerprint:
  vllm-0.23.0-tp8-ep-26bc8ca3` carries the upstream vLLM version, the deployment shape and a build tag;
  `26bc8ca3`, `7f569517` and the vllm-ascend string `0.1.dev50+gdb701c1fd` all fail
  `git cat-file -t` in *both* `/repos/vllm-project/vllm-ascend` and `/repos/vllm-project/vllm`. Probe
  once, then stop resolving the hex and date the build against tag dates
  ([pd-disaggregation.md](pd-disaggregation.md) reads the same fingerprints as upstream-version
  evidence).
- **A report naming both a release and a commit is pinned by the commit, which can be a branch tip
  from before the fix.** `vllm-ascend#13501` titles itself "v0.23.0" yet gives
  "releases/v0.23.0, commit `390a07b97`" (2026-08-04); tag `v0.23.0` is `5cb98caaa` (2026-08-17) and
  the fix `42a70b7ba` merged in between. `git merge-base --is-ancestor 390a07b97 v0.23.0` succeeds (the
  sha is *in* the tag), while the fix artifact is absent at the sha
  (`git ls-tree -r --name-only 390a07b97 | grep async_swa` prints nothing). Ancestry alone then reads
  as "the release has the fix" and the missing artifact as "the release lacks it" — both wrong for
  that build. Compare dates (`git log -1 --format='%ci' <sha>`, `git log -1 <fix>`) and answer with
  the upgrade when the tag or a post-tag release carries it. The defect in that example is in
  [pd-disaggregation.md](pd-disaggregation.md).
- **Model code that exists in no tag the report could be running.** Before pinning, check
  `git ls-tree -r --name-only <tag> -- vllm_ascend/models | grep -i '<family>'` and
  `git log --all --diff-filter=A --format='%h %ad %s' --date=short -- <path>`. GLM-5.3-Flash
  (`vllm_ascend/models/glm5next/`) first appears in `v0.28.0.rc1` and is absent from
  `v0.23.0`-`v0.27.1rc1`; its landing commits are dated 2026-09-12 to 2026-09-22, after a 2026-09-11
  report on a 2026-09-03 image. That report ran a dev/branch build: answer from the earliest tag that
  carries the feature, say the reported version pairs nothing in the tree, and keep the model's pool
  numbers from [c8-dsa-deployment-flags.md](c8-dsa-deployment-flags.md).

## Pairing matrix, and comparing two releases for a "why is B faster" report

For pre-0.10 reports the cheapest pairing source is the tag's own compatibility matrix:
`docs/source/community/versioning_policy.md` ("Release Compatibility Matrix": vLLM Ascend ↔
upstream vLLM ↔ Python ↔ stable CANN ↔ torch-npu), present from at least `v0.8.5rc1` through the
0.9.x line. Read it at the tag (`git show <tag>:docs/source/community/versioning_policy.md`), e.g.
`v0.9.2rc1` → vLLM `v0.9.2`, `v0.9.1rc3` → vLLM `v0.9.1`, `v0.8.5rc1` → vLLM `v0.8.5.post1`.
Later tags record the same pairing in `docs/source/user_guide/release_notes.md` "Dependencies".

A report shaped "[Misc]/benchmark: why is version B faster than version A" is a comparison of two
releases, not a defect. Answer it in this order and stop:

1. `git merge-base --is-ancestor <A> <B>; echo $?` for both directions. Release-candidate tags are
   tips of separate release branches, so neither contains the other and tag dates invert:
   `v0.9.1rc3` (2025-08-22) is dated after `v0.9.2rc1` (2025-07-11), and their merge base is
   `706de0231` (2025-06-10). `git log A..B` therefore lists "what B has that A lacks", which is
   the right set for the question — do not read it as a chronological changelog.
2. Matrix delta: if the upstream vLLM version moved between the two releases (0.9.1 → 0.9.2), the
   engine bump is the first explanatory factor and belongs first in the answer.
3. Plugin perf commits only: `git log --oneline A..B --grep='perf\|Performance\|[0-9]ms'` and the
   release notes of B. In the 0.9.2rc1 line that surfaces items such as #1732 (apply_top_k_top_p
   patch), #1355 (`npu_moe_gating_top_k_softmax`), #1407 (router-logit removal, "3ms"), #1395
   (shared-experts all-reduce, "5ms"), #1353 (MLA multistream). Report those, not the whole range
   (167 commits here).
4. The benchmark delta itself (serving-test.json, output length, dataset) is a confounder worth one
   sentence; the maintainers' own answer to such a report was simply "the version was upgraded with
   inference-speed optimizations".

## Version guards mark upstream layout moves

`vllm_ascend/utils.py: vllm_version_is(...)` compares the installed upstream version, so guarded
imports state exactly where upstream moved a module. `vllm_ascend/_310p/ops/fla/gdn_310.py`
(present at v0.21.0rc1 and v0.22.1rc1):

```python
if vllm_version_is("0.21.0"):
    from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
else:
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
```

Verified upstream: `vllm/model_executor/layers/mamba/gdn/base.py` is absent at `v0.21.0` and
present from `v0.22.0`. A traceback naming `gdn_linear_attn.py` therefore pins the report to the
0.21.x layout (or to the subclass that still lives in that file).

## Known-issues index (cheapest documented answers)

`docs/source/user_guide/release_notes.md` at the tag has `### Known Issues` and
`### Dependencies` per release. Example already documented before the reports arrived:
"Qwen3.6-35B-A3B may shut down when MTP/speculative decoding is enabled, with
`numAcceptedTokens[0]=4 exceeds varlen segment length=3` reported during shape/dtype processing"
(issue #9956), listed in the v0.20.2rc1 section, which is present at tag `v0.21.0rc1` onward.
A second worked entry: at `v0.23.0` the notes list "GLM-5.2-W4A8C8 P/D-disaggregated
serving with `enable_sparse_sfa_c8` can show GPQA accuracy fluctuations of approximately 2-3
percentage points" (#14378) — the documented answer for that whole flag report; details in
[c8-dsa-deployment-flags.md](c8-dsa-deployment-flags.md). Read this list before searching
issues: a lexical `search/issues` for the flag plus "accuracy" returned `total_count: 0`
while the release note and issue carried both words, so a zero-hit issue search is not evidence
that nothing is known.

Those notes can also mislead in the other direction: a `Highlights`/`Features` bullet can outlive
its implementation. `v0.17.0rc1` advertises W8A8C8 for DeepSeek-V3.2/GLM5 in PD-mix (#7029) while
the revert #7288 (`0c299f79b`) sits inside that very tag and the restore #7294 (`4d443b922`) first
appears in `v0.18.0rc1` ([c8-dsa-deployment-flags.md](c8-dsa-deployment-flags.md)). Before quoting
a feature bullet as availability, grep one of its symbols at the tag
(`git grep -n '<key>' <tag> -- vllm_ascend/ docs/`, `git ls-tree -r --name-only <tag> | grep -c
<op-dir>`) and, when a revert is plausible, `git log -1 <sha>` plus
`git merge-base --is-ancestor <revert> <tag>`.

The `v0.18.0` section indexes that release's accuracy class, and none of its entries appear in the
`v0.18.0rc1` section of the same file: "Qwen3.x now has accuracy issue with PD disaggregation
case" (#8421), GLM-5/5.1 PD with a TP16/DP2 D node missing the GPQA standard (#8844), GLM5 and
DeepSeek-V3.2 PD "probabilistic empty output or garbled characters" (#8853), and
`VLLM_ASCEND_ENABLE_FUSED_MC2` being "not recommended for multi-DP and large number of tokens
case (`kv_producer` or `kv_both`)" because the tokens padded across DP get routed to particular
experts. Diffing the two tags' sections is therefore itself the first check for a "release
regressed against its rc" report.

A release file accumulates every earlier section, so a tag past `v0.21.0rc1` also carries the
crash-side entries for the same PD deployments: GLM-5.1 P-node AICore-timeout hangs (#9958) and
GLM5.1 1P1D four-machine hangs with CANN 9.0.0 (#10963) — cascade and pairing checks in
[log-signatures.md](log-signatures.md) — plus, from `v0.22.1rc1`, "Qwen3.x with PD disaggregation
plus MTP can still show precision issues because former KVCache blocks may remain dirty" (#10961).
When the report's own version section does not repeat them, quote these as the *nearest documented
class* and state the model/version delta (GLM-5.2 vs 5.1, node count, CANN) instead of presenting
them as that report's diagnosis.

Per-version hardware notes live in `docs/source/tutorials/hardwares/310p.md` (exists at
v0.20.2rc1 and later).

## Candidate fixes on main only

`git log --oneline v0.22.1rc1..main --grep=<topic>` plus `git tag --contains <sha>`; as of the
clone snapshot none of these is in any tag, so no released image contains them:

| commit | date | subject |
|---|---|---|
| `87f2b060a` | 2026-10-22 | [BugFix][Ops] Fix GDN recurrent state indices exceeding the token count (#19378) |
| `ccc6564de` | 2026-10-26 | [Refactor][310P] Avoid bool-mask IndexPut in GDN prefill state init (#19295) |
| `52ba694bb` | 2026-10-15 | [Bugfix] Gate fused GDN prefill paths on bfloat16 to fix fp16 models crash (#18522) |
| `465e0aa92` | 2026-10-30 | [Performance] Fuse GDN metadata construction for mixed batches (#19954) |
| `a458ef194` | 2026-09-30 | [Refactor][Ops][FLA] Replace causal conv and GDN state ops with fla_npu (#17727) — contained in `v0.30.0*` |

Read them as candidates for the failure class, and report them as "fixed on main only", never as
an upgrade path for a v0.20.2rc1/v0.21.0rc1/v0.22.1rc1 report. Ask both questions of any fix:
`git merge-base --is-ancestor <sha> <reported-ref>` (in the build?) and `git tag --contains <sha>`
(in any release?).

## A release that regressed against its own rc

"AIME score dropped N points from `v0.18.0rc1` to `v0.18.0`" is settled by direction, not by the
PR numbers a maintainer cites. The two tags are separate tips: on the 0.18 line the rc is
`99e1ea0fe` (2026-04-01) and the release `e18643f8a` (2026-04-30), and
`git log --oneline v0.18.0rc1..v0.18.0 | wc -l` is 108 commits, every one of them contained in the
release. Ask the ancestry question of each cited fix against both tags:

```sh
for ref in v0.18.0rc1 v0.18.0; do
  git merge-base --is-ancestor <sha> $ref && echo "$sha in $ref" || echo "$sha NOT in $ref"
done
```

Worked example (vllm-ascend#8932, Qwen3.5-MoE PD accuracy on AIME, closed stale): the maintainer
answered "resolved by PR #8560 and #8764, verified on latest main". `9a0b786f2` (#8764, base
`releases/v0.18.0`) is in `v0.18.0` and **not** in `v0.18.0rc1`, and `a5622f56d` (#8560, base
`main`, a w8a8 `dispatch_ffn_combine` xmask feature) is in neither tag — `git describe --contains
a5622f56d` gives `v0.19.1rc1~32`, so the first released image that carries it is `v0.19.1rc1`, not
`main`. A fix only the release contains cannot explain the release scoring lower than its rc, and
the honest upgrade statement for the reporter is ">= v0.19.1rc1", not "verify on main" — so the
cited PRs do not close the accuracy report even though the thread says so.

Read the linked duplicate before believing the redirect as well: the issue #8932 was called a
duplicate of #8916, which is a *different symptom* on the same 0.18.0rc1 build — "Qwen3.5-moe
disaggregation Server Run 1x hour, it will be Stuck". There the reporter's own comment confirms the
hang stopped after cherry-picking #8531/#7904/#8764 and says in the same breath that "the accuracy
bug still exists", linking back to #8932. So those PRs closed the hang, not the AIME drop: a
cross-issue "resolved by #N" must be matched to the symptom it actually closed.

The usable answer is the release note's own known issue plus the eval configuration (dataset size,
sampling params, seed, MTP/Eagle3 and PD connector flags) that produced the 3.5-point delta. Read it
as locators before any code or PR archaeology: at `v0.18.0`,
`docs/source/user_guide/release_notes.md:112` opens that release's own `### Known Issue` block and
`:118` says "Qwen3.x now has accuracy issue with PD disaggregation case. [#8421]" — the reported
model family and topology verbatim. The same block carries the two config-level causes worth ruling
out first: `:114` "`VLLM_ASCEND_ENABLE_FUSED_MC2` is not recommended for multi-DP and large number of
tokens case (`kv_producer` or `kv_both`), this case may create large number of padded tokens across
dp, which will be routed to certain experts ... resulting accuracy and performance issues" (#8320),
and `:116` the PCP+eagle3 classification hazard (#8402). The rc tag cannot carry the entry: at
`v0.18.0rc1` the same file's newest section is `## v0.17.0rc1` (`:3`), so an rc→final report has to
be answered from the *later* tag's notes.

Two traps in that lookup. (1) The file is a multi-release accumulation and its order is not reliably
newest-first: at `v0.20.2rc1` the `v0.18.0` section is again the top block (`:3`, same entries at
`:114-118`) with `## v0.19.1rc1` *below* it at `:124`, and `v0.19.1rc1`'s own file has dropped the
`## v0.18.0` section entirely while keeping `## v0.18.0rc1` — so grep the tag being answered, and
attribute a hit by the section that contains it, not by its line number. (2) Dates do not decide
ancestry: `v0.18.0`'s release-notes header says `2026.04.22` while its newest commit is dated
`2026-04-30`, and `#8764` (merged 2026-04-27) *is* inside the tag — ask `git merge-base --is-ancestor`
/ `git tag --contains`, never "merged after the release date, therefore absent".

## First release containing a fix

Resolve a PR number to a commit the clone actually has before asking ancestry questions: the
`head_sha`/`merge_commit_sha` projected out of `/pulls/N` can name a fork-only or squash-merge object
that the read-only clone lacks, and `git tag --contains <sha>` then answers `error: no such commit`
(recorded on a PR whose number a maintainer had cited). `git log --all --oneline --grep='#<PR>'` — or
`-S'<symbol the PR adds>'` — yields the commit that landed, and only that one is worth running the
commands below on.

A PR number can also match **no** commit message in the clone, and that is not proof the change is
absent: recorded on the three cherry-picks a reporter credited with stopping a hang — #8764 resolves
(`9a0b786f2 [bugfix][0.18.0] Fix race in non-blocking num_accepted_tokens (#8764)`) and #8531
resolves (`696dcc926`), while `git log --all --oneline --grep='#7904'` returns nothing. Squash titles
are edited and release-branch backports often restate the title, so when the number misses, search
the branch by subsystem words (`git log --oneline origin/releases/v0.18.0 | grep -i 'mtp\|padding'`)
or grep the changed line per tag (below); never trade the miss into "the fix does not exist", and
never quote a sha you could not resolve.

When a fix commit is found, two commands turn it into an applicability statement:

```sh
git merge-base --is-ancestor <sha> <reported-ref> && echo IN_BUILD || echo NOT_IN_BUILD
git tag --contains <sha> --sort=creatordate | grep '^v' | head -3
git describe --contains <sha>          # e.g. v0.13.0rc1~142 = earliest containing tag
```

Do not order with `sort -V`: it places a final release before its own rc tags (`v0.13.0` before
`v0.13.0rc1`), so a fix that arrived in the rc reads as first-shipped in the final release. That
matters for rc-era reports, where the whole question is whether the tested rc already carries it.

`git tag --contains <sha>` and `git describe --contains <sha>` only see that exact commit, so a fix
that was cherry-picked to a release branch is invisible: they name where the original landed, not
every line that carries the change. When the question is "is my release fixed", grep the changed line
in the file at each candidate tag as well —
`for t in v0.25.1rc1 v0.26.0rc1 v0.27.1rc1; do printf "%s: " $t; git show $t:<path> | grep -c <line>; done`.
Recorded case: `q.record_stream(...)` in `vllm_ascend/sample/sampler.py` — `git describe
--contains` on the original commit says `v0.27.1rc1`, while the line is already present from
`v0.26.0rc1` through a backport. Quoting the ancestry answer would tell a `v0.26.0rc1` user that
their build still has the defect. The per-tag table and commit pairs are in
[sampling-and-output-quality.md](sampling-and-output-quality.md).

Worked example: `f7ef489e93` (2026-08-08) fixes the ZMQ probe-then-bind race in
`shm_broadcast.py`. It is not an ancestor of `v0.23.0`, and the earliest `v*` tag containing it is
`v0.28.0rc1`, so no 0.23.x–0.27.x image carries it. `git tag --contains` also returns `proto-*`
tags; filter to `^v` before reading the "first release" off the list.

## Read-only clone commands that work

```sh
# single file at a tag
git show <tag>:path/to/file.py

# one function with real line numbers
git show <tag>:path | sed -n 'A,Bp' | nl -ba -vA

# search across a tag without checkout
git grep -n "<pattern>" <tag> -- 'vllm_ascend/*' 'docs/*'

# unpack a whole tag into scratch for multi-file greps / ctags
mkdir -p "$TMPDIR/tag" && git archive <tag> | tar -x -C "$TMPDIR/tag"

# what changed after the report on a release branch
git log --oneline <tag>..origin/releases/<ver> -- <paths>
git log --all --oneline --grep="<flag-or-issue-number>"
```

`git checkout <branch>` fails in the read-only clone and leaves you on `main`; the follow-up
`git rev-parse` check is the way to notice. `git worktree` has the same problem. Extract snippets
into `$TMPDIR`/scratch — `/tmp` is not writable.

Tool notes:

- `git grep -n '<pat>' <tag> -- <pathspec>` searches a revision without checking it out; it is
  the fastest way to answer "does this symbol exist at the reported version". Options go before
  the pattern and `--include` is not supported (use the pathspec form).
- `git ls-tree -r --name-only <tag> | grep -i <name>` enumerates a tree (useful for `_310p/`,
  docs, Dockerfiles, tests).
- `search_files` is ripgrep-backed and scoped to `/repos`: use it for content search across the
  clones. A recursive `grep -rn /repos` runs long enough to be backgrounded with no usable
  output.

## Check flag presence per tag, not per clone

Presence of a config switch is non-monotonic. One cheap sweep decides the whole question:

```sh
for t in v0.23.0rc1 v0.26.0rc1 v0.28.0.rc1 v0.30.0; do
  printf "%-12s " "$t"; git grep -c "<flag>" "$t" -- 'vllm_ascend/*' | wc -l
done
```

Empty output for a tag means the code at that revision does not know the flag, so the reported
config could have been ignored or rejected there. Do not conclude "removed" from `main`; the
worked example is the `c8_enable_reshape_optim` lifecycle in
[c8-dsa-deployment-flags.md](c8-dsa-deployment-flags.md).

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **A tag loop can manufacture an absence.** `for t in ...; do git show $t:setup.py | grep -c X;
  done` prints `0` for a tag the clone lacks — `git show` writes `fatal: invalid object name` to
  stderr, nothing to stdout, and the pipeline hides the exit status. Gate each iteration with
  `git rev-parse --verify -q "$t^{commit}"`. The same silence covers a moved path:
  `git show main:vllm_ascend/quantization/methods/w4a8.py | wc -l` prints `0` for a file that now
  lives at `methods/w4a8/w4a8.py`, which reads as "421 lines of MoE only" instead of "wrong path".

- **`git tag --contains <sha> | head -20` hides the earliest release, because refnames sort
  lexicographically (`v0.10.x` before `v0.9.x`); `sort -V` fails the same way in the other
  direction, printing a final release before its own rc tags (`v0.13.0` before `v0.13.0rc1`).** A
  fix that first shipped in `v0.9.0rc1` then looks first-shipped in `v0.10.0rc1`, and one that
  arrived in `v0.13.0rc1` looks first-shipped in `v0.13.0` — so both the upgrade advice and an
  rc-era report's fix status come out wrong. Use `git describe --contains <sha>` (prints
  `v0.13.0rc1~142`, the earliest containing tag) or
  `git tag --contains <sha> --sort=creatordate | head -3`, filter `^v` to drop `proto-*` tags, and
  compare the answer against the thread's recommendation instead of assuming the window you saw
  was complete.

- **`git log -S'<string>'` finds adds *and* removes, so it does not answer "which commit introduced
  this Dockerfile line".** A recorded `git log -S'libjemalloc2' --all -- 'Dockerfile*'` returned
  four hits whose subjects named a docs/perf commit, an A5-Dockerfile commit and two "Delete the
  usage of jemalloc" commits — only the diffs distinguished the commit that added the package from
  the one that removed it. Use `-S` to collect candidates, a tag range plus the single path to bound
  them (`git log --oneline v0.12.0rc1..v0.13.0rc1 -- Dockerfile.a3`), and `git show <sha> -- <path>`
  to read the `+`/`-` lines before quoting a "first release".

- **`git grep` options go before the pattern, and `--include` is not a git-grep option
  here.** `git grep -rn PATTERN --include=*.py -l` dies with
  `fatal: option '--include=*.py' must come before non-option arguments`, and piping the
  command can turn that failure into an apparently empty search. Context flags behave the
  same way: `git grep -n PATTERN -A 12 <ref> -- <path>` dies with
  `fatal: unable to resolve revision: -A` because everything after the pattern is read as a
  revision. Use the pathspec form `git grep -n PATTERN <ref> -- '*.py'`, then a short
  `sed -n` window on that hit. A `-- 'docs/*'` search also matches the translated
  `docs/source/locale/**` copies, so every hit appears twice and the useful English line is buried;
  append `| grep -v locale`.

- **A pathspec needs a space after `--`.** `git log --all --oneline
  --vllm_ascend/patch/...` dies with `fatal: unrecognized argument:
  --vllm_ascend/patch/...` because the token is parsed as an unknown option, and the round
  returns nothing; write `git log --all --oneline -- '<path>'`. The same shape bites
  `git show <ref> --<path>`.

- **`git log --grep` uses basic regex.** `--grep='a|b'` searches for a literal pipe and
  silently matches nothing, which looks like an absence of fixes; write `a\|b` or pass
  `-E`, and note that repeating `--grep` ORs the patterns (for example
  `-i --grep='accuracy\|precision\|wrong\|garbage\|MTP\|prefix.cach'`).

- **Do not carry `main` findings back to an older report.** Paths are renamed, split and
  removed between releases (`quantization/quant_config.py` → `quantization/modelslim_config.py`
  → `quantization/configs/modelslim_config.py`), so a `main` file may not exist at the
  reported ref, and the reverse. Reading `main` first and pinning later wastes the pass.

- **A name can appear twice in one file; match the definition by its signature.** At
  `v0.23.0rc1`–`v0.24.0rc1` `mooncake_layerwise_connector.py` has `start_load_kv(self,
  forward_context, **kwargs)` at `:766` and `start_load_kv(self, metadata:
  MooncakeLayerwiseConnectorMetadata)` at `:1603`; the reported frame `:1627 -> _get_kv_split_metadata`
  is inside the second. Grepping the def name finds the first and turns a correct pin into a false
  "line out of range" mismatch. Read the reported function's argument list before choosing.
