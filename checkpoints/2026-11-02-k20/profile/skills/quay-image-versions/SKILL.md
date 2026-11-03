---
name: quay-image-versions
description: Query Quay vLLM Ascend image versions and source revisions.
---
# vLLM Ascend Quay image versions

Inspect or compare the version metadata published with `quay.io/ascend/vllm-ascend`
images when a report names an image or asks about its framework, CANN or dependency
versions. Registry metadata identifies manifests and configs by digest and exposes version
declarations and build instructions; installed versions and official compatibility
require separate checks. It does not diagnose accuracy or serving defects, so a runtime bug
report needs source analysis instead.

This skill only interprets registry-published metadata. Explaining a reported runtime
failure, or reading the code that a reported version ran, belongs to the
`vllm-ascend-issue-triage` skill; this skill supplies the revision label that identifies
which source to read.

## Query published images

Run [scripts/query.py](scripts/query.py) using the `skill_root` returned by `skill_view`;
loading the script's source is unnecessary for ordinary queries. It needs Python's
standard library, `curl` and public HTTPS access to Quay. It retrieves manifests and
config blobs without Docker or image filesystem layers.

Pass complete tags or observed `sha256:...` digests, without the registry/repository
prefix. Preserve hardware and OS suffixes from the reported tag. Select the report's
OS/CPU platform with `--platform os/architecture[/variant]`; the default is `linux/amd64`:

```sh
uv run --no-project python <skill_root>/scripts/query.py \
  --platform linux/amd64 --output "$TMPDIR/image-versions" v0.30.0rc1 v0.30.0rc2
```

Stdout is a JSON comparison with `images` and `errors`. Each image records the requested
reference, platform, index/root digest, selected manifest digest, config digest and source
URLs. `observed_at` records the query time. Tags can move; use the observed index or
platform manifest digest to repeat a query against the same content. Referenced content
digests and the selected config's platform are validated.

`--output` also saves `result.json` and digest-named raw responses. Choose a directory
without an existing `result.json`; the program refuses to overwrite that result. Omit
`--output` when stdout alone is sufficient. Up to four images are queried concurrently,
without a cache. An unavailable image, missing or ambiguous platform, or digest mismatch
produces an error and nonzero exit while preserving successful results for other images.

## Interpret version metadata

- `config_environment`, `cann_paths` and `labels` are image config declarations. Check
  the meaning of a label before treating it as a framework version. `ASCEND_TOOLKIT_HOME` /
  `cann_paths` give the declared CANN of the image and are worth cross-checking against the tag's
  `Dockerfile.<variant>` (`FROM quay.io/ascend/cann:<ver>-<variant>-...`): both agreed for
  `v0.23.0rc1-310p` (`cann-9.1.0-beta.1`) and `v0.23.0.post1-310p` (`cann-9.1.0`).
- `build_arguments`, `recorded_package_pins` and `build_steps` describe recorded build
  instructions. Arguments may be unused, conditional or overridden later. Inspect the
  relevant steps and scripts at the build's verified source revision before interpreting
  them. An empty commit argument or `COPY .` does not identify the copied source
  revision; take that from the revision label described below.
- `runtime_versions` remains unknown. The query does not inspect installed packages,
  editable source trees or container startup changes. To establish those versions,
  collect package versions, source paths, Git SHAs and dirty status inside the actual
  container when runtime inspection is part of the task.

Keep unknowns explicit rather than completing a version table from tag names. Report
the image digest and platform with each comparison so declarations remain tied to the
observed image.

## Map an image to a source revision

`labels["org.opencontainers.image.revision"]` is the vllm-ascend commit baked into the
image, and `labels["org.opencontainers.image.version"]` repeats the release tag. For
v0.7.3.post1, v0.9.0rc2 and v0.11.0 the label equals the corresponding tag's commit in
`vllm-project/vllm-ascend`, so `git show <revision>:<path>` in a clone reads the code the
image actually contains; prefer it over reading `main` when a report names an image.
`build_arguments.VLLM_TAG` is the upstream vLLM release tag, visible in the build step
`git clone --depth 1 $VLLM_REPO --branch $VLLM_TAG ...`.

A missing or mismatched revision label, or a commit absent from the clone, still leaves
the source revision unknown; do not complete it from the tag name or a `COPY .` step. The
label equality above was checked against tag commits, not against image rebuilds.

The label names what the image was built from, not what runs in the reporter's container: a
`pip install -U vllm vllm-ascend`, an editable mount or a nightly wheel makes the installed
tree newer than the tag. Test that before treating the tag as the source. A reported model
or feature that did not exist at that revision is decisive — upstream vLLM adds
`vllm/model_executor/models/gemma4*.py` only in `v0.19.0` (#38826, `8adcf8c40a`), so a
`v0.18.0rc1` / "vLLM 0.18.0" report that serves `Gemma4ForConditionalGeneration` is running
a newer install, and a traceback whose frames match no tag confirms it. Report the tag as
the reporter's environment claim and pin the code by symbol instead.

Hardware-variant images do not all carry the revision label: `v0.23.0rc1-310p` and
`v0.23.0.post1-310p` publish only `org.opencontainers.image.version: "22.04"` (the base OS) and no
`org.opencontainers.image.revision`. For these, map the image to source through the tag itself
(`<tag>-310p` -> vllm-ascend tag) and cross-check `build_arguments.VLLM_TAG`; do not read `22.04` as
a framework version.

Match the platform before comparing anything to a reported value. One tag resolves to a separate
manifest and a different config digest per platform — `v0.23.0rc1-310p` has
`linux/amd64` config `sha256:86385e48…` and `linux/arm64` config `sha256:c91472dc…` — so a
reporter's `sha256:...` image ID is the config digest of *their* host architecture (Ascend servers
are usually `linux/arm64`) and only matches when `--platform` matches. The same tag also works on
the default `linux/amd64` without error, which makes a wrong-platform comparison look valid.

## Check compatibility or a reported fix

For official compatibility, consult the matching version's
[installation guide](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)
and [source](https://github.com/vllm-project/vllm-ascend), pin the revision and resolve
documentation variables/includes. `main` and `latest` can describe a different release.
Repository requirements express intended dependencies, not installed versions.

A repository tag containing a fix or a merged PR establishes source availability only.
To establish that the reported image contains the fix, identify its actual source
revision and check that revision (`git tag --contains <sha>`; a commit reachable only
from `main` is in no release). `git tag --contains` names only the original commit, so a
fix that was cherry-picked to a release branch can already be in the image while the
command says otherwise: grep the changed line at the image's revision as well. To
establish that it fixes the reported failure, verify the applicable runtime path.

For the intended dependency set of a release, read
`docs/source/user_guide/release_notes.md` at the matching tag before parsing registry
configs: its per-release "Dependencies" section names the matched upstream vLLM baseline
and the CANN, torch-npu and triton-ascend versions, and its "Known Issues" section lists
failures already acknowledged for that release. Grep the tag's own file for the issue first: a
tag's own section may already carry it, and later tags accumulate earlier sections. Read the
enclosing `## vX.Y.Z` header of every hit, though — the accumulation means a model/flag grep often
lands in a section several releases back (in the `v0.20.2rc1` file a Kimi hit belongs to
`## v0.15.0rc1` and names upstream vLLM 0.15.0), so attribute a hit only to its own section and check
any upstream version it names against the tag's `Dockerfile` `ARG VLLM_TAG`. A section
can still be added after the tag exists, so when the tag's own file greps clean, fall back to
the next release's notes or the plugin's `vllm_version_is(...)` guards. These are intended
pins, not installed runtime versions.
For diagnosing an issue report against a pinned tag, see the `vllm-ascend-issue-triage`
skill.

Registry protocol and config semantics are documented by
[OCI distribution](https://github.com/opencontainers/distribution-spec/blob/v1.1.1/spec.md)
and [OCI image config](https://github.com/opencontainers/image-spec/blob/v1.1.1/config.md).
The bundled query has been validated for metadata retrieval and version interpretation
boundaries; it does not validate NPU execution or certify compatibility.
