# The page's printed image tag, the docs build, and `latest` vs `/en/vX.Y.Z/`

Verified at vllm-ascend v0.26.0rc1 (`f2f74a16c3c5`) + vllm v0.26.0
(`568afb3a1380`); recorded for #16954, whose resolution came from the maintainer
and the reporter after the fact.

Use this page when a `[Doc]` report's defect is a **version disagreement**: the
page says `vllm-ascend 0.X.Y`, the pasted command fails on exactly that image, or
the reporter is reading a `…/en/latest/…` URL and asks which release it belongs
to. The finding is then about the docs build, not a code path.

## 1. What the URL means

The docs are built on Read the Docs: `mkdocs.yml:157` declares
`rtd_version: !ENV [READTHEDOCS_VERSION, latest]`, and `tools/rtd_build.sh:39`
sets `DOCS_IS_RELEASE=true` only when `READTHEDOCS_VERSION_TYPE=tag`
(`mkdocs.yml:154 extra.is_release`, consumed by `docs/overrides/main.html:24`).
So `/en/latest/` is the build of **main**, and every release is a separate
`/en/vX.Y.Z/` tree. Maintainer's own statement on #16954:

> Documentation for a specific branch should correspond to that branch. The
> `latest` docs only align with the latest code on the `main` branch. For the
> `0.23.0` release branch, please refer to …/en/v0.23.0/

A command that only works on main is therefore *correct text in the wrong
place* when it sits on `latest` beside an older image tag.

## 2. Where the image tag in a tutorial comes from — one grep

Tutorials do not spell the tag out. `docs/source/tutorials/models/GLM5.md:48`
reads `export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3`
(`:92` the same without `-a3`); the value is substituted at build time through
the macros plugin (`docs/hooks/macros_main.py:47-51`, whose fallbacks are old
values — `v0.22.1rc1` — so a substitution that was never set silently prints an
ancient tag) from `mkdocs.yml:159-163 extra:`:

```text
vllm_version: v0.26.0          vllm_ascend_version: v0.26.0rc1
pip_vllm_ascend_version: v0.26.0rc1   pip_vllm_version: v0.26.0
```

The tag a page prints is thus a property of the **built revision**, not of the
page text: `rg -n 'vllm_ascend_version|pip_vllm_ascend_version' mkdocs.yml`.
Compare it with the tag the reporter pasted. Equal → the mismatch is elsewhere
(a branch URL, or a flag that no release carries). Different → the built
revision's substitution is stale, which is the defect to report.

## 3. The #16954 disposition, step by step

Report: `latest` GLM-5 §5.3 A3 PD prints `--kv-cache-dtype int8` +
`--attention_config.indexer_kv_dtype int8`; the 0.23.0 image dies at argparse
with `argument --kv-cache-dtype: invalid choice: 'int8'`.

1. Printed tag vs substitution — the page printed `0.23.0`, the pin's
   `mkdocs.yml:159-163` says `v0.26.0rc1`: the tag is stale for the code that
   carries the flags.
2. Is the flag pair in the pinned tree at all? `GLM5.md` has no
   `kv-cache-dtype` and `rg -n 'indexer_kv_dtype' -g '!*.po' .` returns 0 hits,
   so this is main-only content and cannot run on any *released* image the
   snapshot carries (`rg '^## v0' docs/source/user_guide/release_notes.md`:
   `v0.26.0rc1 - 2026.09.03`, `v0.23.0 - 2026.08.16`).
3. Owner of the accepted values, **corroboration only**: upstream
   `CacheDType` (`vllm/config/cache.py:19-36`, no `int8`) and `IndexerKVDType`
   (`vllm/config/attention.py:14` = `Literal["bf16","fp8","mxfp4","nvfp4"]`,
   no `int8`), surfaced by `vllm/engine/arg_utils.py:1168` and `:1562`. The
   underscore spelling `--attention_config.…` is legal —
   `FlexibleArgumentParser` "allows both underscore and dash in names"
   (`vllm/utils/argparse_utils.py:113-114`) — so never report it as a typo.
4. Deliverable: the version gate the page needs (which release first carries
   the flag, or an explicit "main/nightly only"), the `/en/vX.Y.Z/` URL for the
   branch the reporter runs, and the correction site
   (`mkdocs.yml extra.vllm_ascend_version` + a rebuild of `latest`). The
   reporter's own follow-up on #16954 was exactly that: "Please adjust the
   description of the image version in the latest version. Currently, the image
   version in the latest version is still 0.23.0."

## 4. What the recorded run paid by skipping step 1

#16954's run grepped correctly (flags absent at the pin, `indexer_kv_dtype`
absent repo-wide) and then spent the rest of its 16 384-token budget deciding
whether an upstream `Literal` *could* accept `int8` and which build had carried
which content; it ended `output_truncated` with no answer. Both decisive facts
are two lines (`mkdocs.yml:159-163`, the `## v0` list in the release notes).
Read those first and ship; the upstream literals are corroboration, never the
finding.

## 5. The page may already carry a version sentence — grade against it

- `GLM5.md:9`: "The `GLM-5` model is first supported in `vllm-ascend:v0.17.0rc1`
  … To use the latest features (e.g., PD separation, MTP), it is recommended to
  use the latest release candidate or official version."
- `GLM5.3.md:15`: "All the scripts below is based on **v0.23.0**, so some params
  are not supported in main code" (quoted in
  `../startup-failures/additional-config-schema.md`).

That is the sibling convention. A page that prints an image tag but never says
which release its flags belong to — or says it only in prose far from the
command block — is the defect to state, with `path:line` for both the tag line
and the sentence that fails to gate it.
