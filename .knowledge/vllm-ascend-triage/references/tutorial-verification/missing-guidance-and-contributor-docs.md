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
v0.22.1), **#17718** (v0.23.0 `5cb98caaadef`, vllm v0.23.0 `0fc695fc6d1d`),
**#5158** (promised install step that exists nowhere, v0.12.0rc1 `42ceaf08a1ef`,
vllm v0.12.0 `4fd9d6a85c00`) and **#8151** (v0.18.0rc1 `99e1ea0`: the same
mirror-discoverability shape as #14423, reporter-confirmed).

## Checks, in order

1. **Does the content already exist?** One `rg -n '<distinctive token>'
   docs/source` decides whether the report is "add this" or "link this". At
   v0.22.1rc1 the China mirror methods are already in `faqs.md:31-44` (§2
   `### 2. How to get our docker containers?`): `m.daocloud.io/quay.io/ascend/
   vllm-ascend:$TAG` (`:40`), `quay.nju.edu.cn/ascend/vllm-ascend:$TAG` (`:42`),
   plus the offline `docker save`/`load` subsection (`:45-72`). So #14423 is a
   discoverability defect, and the answer says so before any patch. At v0.18.0rc1
   (`99e1ea0`, #8151) the same section is `faqs.md:28-40` — heading `:28`, prose `:32`,
   `TAG=v0.9.1` hardcoded `:36`, `m.daocloud.io` `:37`, `quay.nju.edu.cn` `:39` — with
   the offline `docker save`/`load` block at `:42-68`; the line numbers move with the
   pin, the section does not.

   **The registry half of a mirror report is answered from the workflows, not from
   the docs.** "镜像地址只有 quay 的吗？昇腾镜像仓库/华为云镜像仓库也没有" is two
   questions, and both are settled in the pinned tree: the release build logs in to
   `registry: quay.io` only and pushes `name=quay.io/ascend/vllm-ascend`
   (`_schedule_image_build.yaml:69,90,149,169,178`), so **no release tag is published
   anywhere else**; Huawei Cloud SWR receives only the **moving nightly** images
   `…/base_image/ascend-ci/vllm-ascend:nightly-a2|a3|cpu`, pushed with
   `HW_USERNAME`/`HW_TOKEN` secrets (`_nightly_image_build.yaml:42-49,56,68-71`,
   consumed at `schedule_nightly_test_a3.yaml:126,226`), so a `v0.18.0rc1`-style tag is
   not there and anonymous pull is unverified; and the project's own CI pulls quay
   through `m.daocloud.io/quay.io/ascend/cann:…` (`_e2e_test.yaml:481`), which is what
   makes the FAQ's proxy line citable rather than folklore. Verdict order: "the release
   is quay-only, the China path is the FAQ's proxy plus `docker save`, and the defect is
   that the entry page never links it" — grade it **usability**, and quote the
   reporter's own follow-up as the confirmation: #8151 found the snippet only after
   being pointed at §2 and answered 实在难找 ("why is it not in the registry the project
   maintains"), which is the cross-link finding in the reporter's words.
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

## A promised install step that exists nowhere (recorded #5158)

The shape: a tutorial sentence tells the reader to install or compile something —
`docs/source/tutorials/DeepSeek-V3.2.md:32`, "You can using our official docker image
**and install extra operator** for supporting `DeepSeek-V3.2`" — the reporter cannot
find the *how*, and asks whether a neighbouring issue's manual `deepseek_v3_ops` build
is what they are expected to do (#5158; image `quay.io/ascend/vllm-ascend:v0.12.0rc1`,
vllm-ascend v0.12.0rc1 `42ceaf08a1ef` + vllm v0.12.0 `4fd9d6a85c00`). Four reads, in
this order:

1. **Does the step exist anywhere?** `rg -n 'extra operator' docs/source --glob
   '!**/locale/**'` → 2 hits, both tutorial prose (`DeepSeek-V3.2.md:32`,
   `Qwen2.5-7B.md:27`): no script, `pip` line, URL or section. `installation.md:165-170`
   documents building custom operators *from source*, not a post-image step, so the
   promise has no target even by cross-reference.
2. **Sibling parity.** `DeepSeek-V3.1.md:39` and `DeepSeek-R1.md:28` say "… to run
   `<model>` **directly**", and the V3.2 sentence is the same template line as
   `Qwen2.5-7B.md:27` (identical text, same "You can using"). One page inherited
   boilerplate; the siblings state the correct behaviour.
3. **Does the same release's own release notes contradict it?** The strongest line,
   because both statements ship in one tag: `release_notes.md:8` under
   `## v0.12.0rc1 - 2025.12.13` — "DeepSeek 3.2 is stable … **In this release, you don't
   need to install any other packages now.**"; `:15` — custom ops became mandatory and
   `COMPILE_CUSTOM_KERNELS` was removed; and the image builds the tree at build time
   (`Dockerfile:62`, `Dockerfile.a3:61`), so there is nothing to install inside the
   container. The only real optional extra at that tag is the **triton-ascend** wheel
   (`release_notes.md:14`; aarch64, x86 builds from source) — most plausibly what the
   sentence gestures at, and the page's own note at `:35` recommends it with no URL,
   which *is* the defect worth filing. Tier: **stale/misleading**, not cosmetic; the fix
   is the sibling wording plus a triton note carrying that URL.
4. **Do not reconstruct the identifier the reporter cites.** `rg -n 'deepseek_v3_ops'`
   over the whole tree returns 0 hits and #3278 is outside the snapshot (404): say both,
   and do not derive a procedure from an issue number.

**Then close the loop: a cleared step is not a working deployment.** This rung must not
read "skip it and proceed". #5158's public follow-up (2025-12-19) is "I switch to latest
main branch and hit another error" — the reporter, told the documented step does not
exist, moved off the tag. So name the pair the verdict holds for (tag `v0.12.0rc1` ↔
vllm 0.12.0 ↔ that release's own CANN/torch-npu row), say that a `main` checkout
re-pairs vllm and the custom-op build and is a different report, and ask for the runtime
error if deployment still fails — rule 14 of `SKILL.md`: the next attempt surfaces
*different* errors.

## Contributor-path docs: run what the page names

Do not read the contributor prose, run the script or image step it points at.
`testing.md` is `docs/source/developer_guide/contribution/testing.md` at v0.23.0
(374 lines) — it is under `contribution/`, and `docs/source/developer_guide/testing.md`
does not exist, so locate it with `rg --files docs/source | rg 'testing'` rather
than guessing the directory. Items that paid off at v0.23.0:

| Claim in the report | What the tree says | Tier |
|---|---|---|
| "`bash format.sh ci` is not the test suite" | `format.sh:43` = `pre-commit run --all-files --hook-stage manual`; `docs/source/developer_guide/contribution/index.md:35` still titles that section "Run CI locally", command at `:59` | misleading |
| "mounting a checkout hides the image's `vllm-ascend`" | `setup.py:189` writes `vllm_ascend/_build_info.py` **into the source tree** (called from `build` `:204` / `build_py` `:210`), and `vllm_ascend/utils.py:836` imports it — a fresh clone does not carry it, so a bind-mount over `/vllm-workspace/vllm-ascend` reproduces `ImportError: cannot import name '_build_info'` until `pip install -e .` | breaks the command |
| "the page says run UT but never how to test **my** checkout" | the Single-card tab's `docker run` (`testing.md:76-92`) mounts only `dcmi` / `npu-smi` / `driver` / `.cache` and then `cd`s into the image's own `/vllm-workspace/vllm-ascend` (`:101-102`), so a contributor has to add `-v <checkout>:/vllm-workspace/vllm-ascend` — and that one edit is what trips the `_build_info` row above, so name both edits and `pip install -e . --no-build-isolation` | breaks the command |
| "`pip install` dies on `HTTP 403` / the recommended pip mirror does not serve the package" | one page, two mirrors: `testing.md:99` sets `mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` while the same file's own container block (`:33-34`) uses `mirrors.huaweicloud.com` for apt **and** pypi. The inconsistency is the reportable defect — point at `:33-34`; the 403 itself is the reporter's network and is never checkable from the tree | misleading |
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
whole deliverable and needed no source walk. A later run of #17718 on the same
pin re-verified every row of the table above and added two defects the reporter
had not listed — the page's two mirrors (`testing.md:99` vs `:33-34`) and the
Single-card tab's missing source mount — so treat the table as the floor and
spend the remaining budget on what it does not carry.
