# Missing-guidance and contributor-path doc reviews: the defect is an absence

Scope: a `[Doc]` / `[Usage]` issue that does **not** claim a command fails. Two
shapes land here:

- **discoverability** — "the Quick Start should mention X, or link to the FAQ":
  the guidance usually already exists somewhere in the tree, just not where the
  entry page needs it;
- **contributor path** — "the docs are written for users; a new contributor
  cannot get from a clean clone to a passing UT run".

Both are graded on the *pinned tree*, not on a reproduction: the evidence is two
or three greps (does it exist, does anything reach it, do the sibling pages
already carry it) plus the tree's own build metadata. Tier as in
`tutorial-command-audit.md` — **breaks the command / stale / cosmetic** — and
note that a missing cross-link is normally *usability*, unless the absent step
is one that breaks the command (the first two rows of the contributor table
below are that case).

Verified drivers: **#14423** (vllm-ascend v0.22.1rc1 `5f6faa0cb883`, vllm
v0.22.1) and **#17718** (v0.23.0 `5cb98caaadef`, vllm v0.23.0 `0fc695fc6d1d`).

## Checks, in order

1. **Does the content already exist?** One `rg -n '<distinctive token>'
   docs/source` decides whether the report is "add this" or "link this". At
   v0.22.1rc1 the China mirror methods are already in `faqs.md:31-44` (§2
   `### 2. How to get our docker containers?`): `m.daocloud.io/quay.io/ascend/
   vllm-ascend:$TAG` (`:40`), `quay.nju.edu.cn/ascend/vllm-ascend:$TAG` (`:42`),
   plus the offline `docker save`/`load` subsection (`:45-72`). So #14423 is a
   discoverability defect, and the answer says so before any patch.
2. **Does anything reach it?** `rg -n 'faqs|troubleshooting|FAQ' installation.md
   quick_start.md` returned **0 hits** at that pin, in both entry pages. Quote
   the zero as the finding — it is a measurement, not an inference from
   headings.
3. **Parity: do the sibling pages already do it?** `rg -l 'daocloud|nju\.edu\.cn'
   docs/source --glob '!**/locale/**'` returns 17 files: `faqs.md` plus 16 other
   pages that use the mirror prefix as their default (`tutorials/models/DeepSeek-R1.md`,
   `tutorials/features/ray.md`, …) while the two entry pages use bare
   `quay.io`. That converts "please add guidance" into "make the entry pages
   consistent with 16 pages that already do it", and it is the strongest line in
   the report.
4. **Resolve the reporter's own URL and anchor at the pin.** `rg -n '^#+ '
   docs/source/faqs.md`: #14423's cited `faqs.html#1` is §1 *What devices are
   currently supported?* (`:13`), not the container section (`:31`). The
   reporter's anchor is wrong at this pin — say it, because the fix has to name
   the right target.
5. **In a translated tree the anchor is locale-specific.** `zh-cn` pages are a
   gettext build of the English source: `locale/zh_CN/LC_MESSAGES/faqs.po:125-126`
   translates the heading to `2.如何获取我们的 Docker 容器？`, and
   `conf.py:103` `myst_heading_anchors = 5` slugifies the *rendered* heading, so
   the English `#2-how-to-get-our-docker-containers` is not what the zh page
   carries. For a cross-language fix recommend an explicit MyST label
   (`(faq-docker-containers)=` above the heading, as
   `developer_guide/performance_and_debug/service_profiling_guide.md:196,232`
   already does) or a link to the English page. Label this **source-derived,
   build not run**, and give the command that settles it (`python -m sphinx -b
   html docs/source …` and read the built zh page's anchor).
6. **Is the section the report wants to link to itself stale?** The mirror
   snippet hardcodes `TAG=v0.9.1` (`faqs.md:39`) while the same page's offline
   block substitutes `|vllm_ascend_version|` (`conf.py:81` = v0.22.1rc1). One
   extra defect line, same section.

## Contributor-path docs: run what the page names

Do not read the contributor prose, run the script or image step it points at.
Items that paid off at v0.23.0:

| Claim in the report | What the tree says | Tier |
|---|---|---|
| "`bash format.sh ci` is not the test suite" | `format.sh:43` = `pre-commit run --all-files --hook-stage manual`; `docs/source/developer_guide/contribution/index.md:35` still titles that section "Run CI locally", command at `:59` | misleading |
| "mounting a checkout hides the image's `vllm-ascend`" | `setup.py:189` writes `vllm_ascend/_build_info.py` **into the source tree** (called from `build` `:204` / `build_py` `:210`), and `vllm_ascend/utils.py:836` imports it — a fresh clone does not carry it, so a bind-mount over `/vllm-workspace/vllm-ascend` reproduces `ImportError: cannot import name '_build_info'` until `pip install -e .` | breaks the command |
| "the UT commands never `source set_env.sh`" | the image sources both `set_env.sh` **inside `RUN` only** (`Dockerfile:30`, `:67-68`); what persists is `LD_PRELOAD`/`LD_LIBRARY_PATH` in `~/.bashrc` (`:75-76`). But `ENV SOC_VERSION` *is* baked (`:57-60`) and the CPU tab already sets it (`testing.md:59-61`) — so split the item: right for CANN/NNAL, refuted for `SOC_VERSION` | breaks the command / refuted |
| "the image already contains vLLM" | `Dockerfile:42` `ARG VLLM_TAG=v0.23.0` + `:49` clone it, while `contribution/index.md` tells the reader to build vLLM separately | inconsistent |
| "the page says `nightly-main`" | `testing.md:77`,`:116` say `quay.io/ascend/vllm-ascend:main` — a literal tag where every other page substitutes `|vllm_ascend_version|`; the reporter's tag is main-branch drift | version drift |
| "`COMPILE_CUSTOM_KERNELS=0` is the UT-on-CPU escape hatch" | `vllm_ascend/envs.py:43` still reads it (default 1) and `Dockerfile:58` still passes it, while `release_notes.md:1408` (under `## v0.12.0rc1`, `:1395`) says the env was *removed* | stale release note |

The `_build_info` row is the reusable mechanism: when a doc says "clone/mount
your tree and run tests", the failure mode is a **build-generated file**, so
grep `setup.py` for the generator and the package for the importer before
theorising about `PYTHONPATH`.

## The `.po` tree is not the document

`locale/**/LC_MESSAGES/*.po` is machine-generated (translation workflow under
`.github/workflows/`), and two of its properties cause false findings:

- its `#: ../../source/<page>.md:<N>` line references lag the source
  (`installation.po` points `## Set up using Docker` at `installation.md:236`
  while the heading is at `:249`), so they are never citable line numbers;
- its `msgstr` is a translation of an older msgid, so a path or command that
  looks wrong there may be right in the English source — check the source first.

#17718 nearly published `source …/nnal/atb/ln` as a doc typo; that string was
its own `rg -r ln` replacement (SKILL.md rule 7), not repo content.

## Version scope to state in the report

A `[Doc]` reporter works from `main`; the snapshot is a release tag. Expect
three drift classes and label each instead of repeating the reporter's
number: image tag (`nightly-main` vs `:main`), line offsets
(`envs.py:51` vs `:43`), and build args (`VLLM_TAG=v0.30.0` vs `v0.23.0` at
`Dockerfile:42`). Grade the *claim*; the offset is not a defect.

## Answer shape that worked (#17718, 12 assistant turns)

Verdict paragraph first — which of the reporter's items hold, which are refuted,
which they missed — then the defect table with `path:line` and tier, then the
minimal patch per page, then version scope and open items (a registry 403 or a
mirror's reachability cannot be checked from the tree). For #14423 the verdict
was "issue valid, but both anchors the report cites are wrong", which is the
whole deliverable and needed no source walk.
