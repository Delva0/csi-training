# "There is no image/wheel for my architecture or my SoC"

Verified at vllm-ascend **v0.9.2rc1** (`b5b7e0ec`) for #2178 — 310P on an
`linux/arm64` host, "找不到支持 linux on arm64 的 310p 的镜像" — plus the same
facts' new home at **v0.23.0rc1** (`f4a08bdd`), because the CI layout was
rewritten between them. Re-grep per version; the shape (suffix → SoC variant,
arch matrix, `FROM` base) does not move.

## Classify before searching

Two shapes, neither owned by vllm-ascend code:

- **No image for this CPU architecture.** The paste carries
  `WARNING: The requested image's platform (linux/amd64) does not match the
  detected host platform (linux/arm64/v8) and no specific platform was
  requested`, plus one or two tags the reporter tried. That sentence is printed
  by the *client* when the reference resolves to a single-arch manifest: it is a
  symptom, it names no vllm-ascend file, and it never means "the project dropped
  arm64".
- **No wheel for this SoC.** A wheel is built for one SoC: `setup.py:85-88`
  requires `SOC_VERSION` and writes `__soc_version__` into
  `vllm_ascend/_build_info.py` (`:95-98`), `:193` passes it to cmake — so "which
  wheel" and "which image tag" are the same question with different artefacts.

Three items in a paste are **not** evidence: a third-party mirror's tag page
(`docker.aityp.com` and friends republish single-arch content and lag), a
`…?tab=tags&tag=latest` URL (the bare `latest` belongs to the generic A2
pipeline — see the suffix table), and the client warning above.

**"镜像只有 quay 吗 / 华为云镜像仓库也没有" is not this page** — nothing is missing in
the artefact, the guidance is just not where the reporter looked. Release tags are
quay-only and the tree's own China path (`m.daocloud.io/quay.io/ascend/…`,
`quay.nju.edu.cn/ascend/…`, `docker save`/`load`) sits in `docs/source/faqs.md` §2
while `quick_start.md`/`installation.md` never link it; Huawei Cloud SWR carries only
`nightly-a2|a3|cpu`. Route it to
`../tutorial-verification/missing-guidance-and-contributor-docs.md` check 1 (recorded
#14423, #8151) instead of hunting arch manifests.

## Step 1 — the artefact itself (one command, needs network)

```bash
docker buildx imagetools inspect quay.io/ascend/vllm-ascend:<ver>-<soc>   # Platform: list
docker buildx imagetools inspect quay.io/ascend/cann:<tag from the FROM>  # the base image
docker pull --platform linux/arm64 quay.io/ascend/vllm-ascend:<ver>-<soc> # explicit: no fallback
```

`Platform: linux/arm64` present = it exists and the reporter's tag/mirror was
wrong. Absent = the tag is genuinely that arch (step 2 says why), and the
fallbacks below apply. Ask for the `imagetools` output verbatim, plus
`docker info | grep -A4 'Registry Mirrors'` (a proxy/mirror can flatten a
manifest list) and `uname -m`.

## Step 2 — what the tree intends to publish (two greps)

Run `bash <skill_root>/scripts/image-arch-matrix.sh <vllm-ascend-root> --soc 310p`
first: one call prints the tag suffixes, the `platforms:`/`arch:` matrix, the runner
legs, every `FROM` base, the wheel-release matrix and the per-SoC variant list —
the four hand `rg`/`sed` reads below only re-derive what it already prints (a
recorded #2178 run spent four calls on exactly those greps after planning to run
this script). Read it as three facts:

1. **Tag suffix ⇒ SoC variant.** At v0.9.2rc1 the `docker/metadata-action` tag
   lists are the table: `-310p` (ubuntu,
   `.github/workflows/image_310p_ubuntu.yml:69-71`), `-310p-openeuler`
   (`image_310p_openeuler.yml:73-75`), `-openeuler` (`image_openeuler.yml:73-75`)
   and the **unsuffixed** tags (`image_ubuntu.yml:69-71`), which are the generic
   A2 image; `latest` itself is only ever produced there, and a SoC pipeline's
   `latest` equivalent would carry its suffix (`latest-310p`) and only for a
   non-prerelease pep440 tag — an `rc` tag gets no `latest` at all (the block
   comment at `image_310p_ubuntu.yml:63-67` states the rule). At v0.23.0rc1 the
   same table is one list, `schedule_image_build_and_push.yaml:75-90`
   (`''`, `openeuler`, `a3`, `a3-openeuler`, `310p`, `310p-openeuler`, each with
   its `dockerfile:`). So "the 310P image" is never a bare tag.
2. **The arch matrix and the publish permission are the same condition.** At
   v0.9.2rc1 both 310P workflows request `'linux/amd64,linux/arm64'` only when
   `github.event_name == 'push' && github.repository_owner == 'vllm-project'`,
   else one arch — and `push:` on the build step is that identical expression
   (`image_310p_ubuntu.yml:96-100` with `:104`, `image_310p_openeuler.yml:100-104`).
   So every build allowed to publish is also requested multi-arch, and the
   single-arch fallbacks (differing by workflow: `'linux/amd64'` for the ubuntu
   one, `'linux/arm64'` for the openEuler one, whose runner flips to
   `ubuntu-24.04-arm` off the push event, `:43-47`) only describe *check* builds
   that publish nothing. The **generic** pipelines carry the same expression and
   the same two fallbacks (`image_ubuntu.yml:96-101`, `image_openeuler.yml:100-105`),
   so an unsuffixed tag is not "amd64-only by design" either — the single-arch
   reading always needs step 1's manifest check, never the workflow alone.
   Consequence: a published `-310p*` tag that resolves to
   amd64 alone is **not** explained by the fallback — either the arm64 leg failed
   on the base image (fact 3), or the client/mirror flattened the list (step 1).
   At
   v0.23.0rc1 the matrix is explicit and merged: `_schedule_image_build.yaml:83-91`
   (`linux/amd64` → `ubuntu-latest`, `linux/arm64` → `ubuntu-22.04-arm`),
   `:241` `platforms: ${{ matrix.arch }}`, with a `merge-image` job (`:296-299`)
   publishing the manifest list — and `push_build_csrc_cache.yaml:18` states the
   premise: "3 chip_type variants (a2 / a3 / 310p) — all ARM64 + Ubuntu".
3. **An arm64 leg needs an arm64 base.** `Dockerfile.310p:18` is
   `FROM quay.io/ascend/cann:8.1.rc1-310p-ubuntu22.04-py3.10`;
   `Dockerfile.310p.openEuler:18` the openEuler tag. If that base is amd64-only,
   the arm64 leg cannot be built for **any** 310P tag, which explains the report
   completely, and nothing in vllm-ascend can change it. `Dockerfile.310p:52`
   (`…/latest/\`uname -i\`-linux/devlib`) shows arm64 is a design target, and
   `:53` `SOC_VERSION=ASCEND310P3` shows the SoC is fixed at build time — which is
   also why an A2 wheel cannot be installed on a 310P host
   (`../soc-platform-paths/device-type-build-vs-runtime.md`).

### `quay.io/ascend/vllm-omni`: a second registry with a shorter matrix

The suffix table above is vllm-ascend's. **vllm-omni publishes A2 and A3 only** — its own
install page's table is `name | image-tag | image-tag-a3 | Hardware Atlas A2 | Atlas A3 |
OS Ubuntu` ("verified on 4 x NPUs"), i.e. no `-310p` variant exists to look for, and the
unsuffixed `image-tag` is the A2 build. The page is often reachable only as the post's
screenshot: read `/workspace/attachments/<issue>/<img>.ocr.txt` before reconstructing
conventions from another issue (`#8201`'s `镜像：vllm-omni:v0.18.0（from
quay.io/ascend/vllm-ascend:v0.18.0rc1）` says the same thing one hop weaker). Consequence
for a 310P report: the answer is not a different tag, it is a source build on a
`-310p` CANN base with `SOC_VERSION=ascend310p1` →
`../soc-platform-paths/310p-omni-model-deploy.md`.

## Step 3 — the wheel path, when no image fits

- `release_whl.yml:46-47` builds on `os: [ubuntu-24.04, ubuntu-24.04-arm]` and
  uploads with `twine` (`:118`), so an **aarch64 wheel does exist** on PyPI.
- But `.github/Dockerfile.buildwheel:18` is the **910b** manylinux base
  (`quay.io/ascend/manylinux:8.0.0-910b-manylinux_2_28-py<ver>`), so that wheel is
  an A2 build; `setup.py:90-92` also refuses a 310 build without
  `COMPILE_CUSTOM_KERNELS=1`. The matching 310P wheel had to be attached to the
  GitHub release by hand — #1742's v0.9.2rc1 checklist still has
  `- [ ] Upload 310p wheel to Github release page` unchecked.
- The split is already there at **v0.18.0**, three wheel jobs, not one:
  `.github/workflows/dockerfiles/Dockerfile.buildwheel.a2:20` (`ARG
  SOC_VERSION="ascend910b1"`), `.a3:20` (`ascend910_9391`), `.310p:20`
  (`ascend310p1`). It did **not** change what PyPI serves: in
  `.github/workflows/schedule_release_code_and_wheel.yml` only the A2 job
  twine-uploads `dist/*.whl` (`:192-196`; the `:84-89` Release uploads the sdist),
  while the A3/310P packages are emitted as
  **variant** wheels (`-l a3` / `-l 310p`, labels in
  `.github/workflows/scripts/wheel/config.json`) by the
  `generate_and_upload_variant_index` job and land on the OBS variant index. So
  `pip install vllm-ascend==<tag>` on an A3 / 910_93 host still installs A2
  custom-op binaries — the wheel-side twin of the `-a3` image question
  (`../custom-ops-runtime-faults/op-not-registered-for-soc.md`,
  `../custom-ops-runtime-faults/opapi-symbol-absent.md` §*The published wheel is
  built for one SoC: A2*). Earlier notes dated this split to v0.23.0rc1; the
  three Dockerfiles are already present at v0.18.0, so treat the split, not a
  single-wheel build, as the shape from that release on.

## Answer shape

Verdict first — *which artefact exists for this arch + SoC at this release*, or
"cannot be decided offline, here is the one command" — then the ladder, cheapest
first: correct the tag suffix (`<ver>-310p` / `<ver>-310p-openeuler`, never
`latest`); `docker pull --platform linux/arm64`; check the `FROM` base's arch;
then the local build on the arm host (`docker build -f Dockerfile.310p -t … .`,
slow because `COMPILE_CUSTOM_KERNELS=1` compiles the AscendC ops); then the
CANN-base + source/wheel install with the SoC caveat. Close with what to send
back: the two `imagetools` outputs, `docker info` mirrors, `uname -m`.

## Limits

Registry manifests, quay's tag pages and the CANN base images are not in the
snapshot: whether a specific historical tag carries arm64 is **not** locally
verifiable, so state it as an open item rather than as a fact. What *is*
source-verified here is only what the pipeline requests, which is weaker than what
it produced — a failed or skipped workflow run leaves the published tag
single-arch without any trace in the tree.
