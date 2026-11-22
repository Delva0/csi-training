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
Compare it with the tag the reporter pasted. Equal → the mismatch is elsewhere — a
branch URL, a flag that no release carries, or a **second, literal tag on the same
page** (§6). Different → the built revision's substitution is stale, which is the
defect to report, with §3's correction site.

**The substitution has two homes, and which one exists is a property of the tree.**
Older releases build the same docs with **Sphinx + myst**: there is no
`mkdocs.yml` at all, the keys live in `docs/source/conf.py` → `myst_substitutions`,
and a page writes the pipe form `export IMAGE=…/vllm-ascend:|vllm_ascend_version|`
(verified at v0.11.0rc0 where the value is `v0.11.0rc0`, and at v0.18.0). At v0.18.0
the dict carries six keys (`vllm_version` `:68`, `vllm_ascend_version` `:72`,
`pip_vllm_ascend_version` `:75`, `pip_vllm_version` `:77`, `cann_image_tag` `:79`,
`ci_vllm_version` `:80`); the `{{ … }}` form and `mkdocs.yml extra` belong to the
mkdocs era (§2's v0.26/v0.27 pins), and neither syntax renders in the other build. So name both
files in the one grep — `rg -n 'vllm_ascend_version|pip_vllm_ascend_version'
mkdocs.yml docs/source/conf.py` — and treat an empty result as "this tree uses the
other build", not as "the page prints a literal".

At v0.18.0 (`e18643f8a4d5`) that grep is itself a finding: `conf.py:72`
`vllm_ascend_version` and `:75` `pip_vllm_ascend_version` are both **`v0.17.0rc1`**,
`:68` `vllm_version` is `v0.17.0`, while `:80` `ci_vllm_version` is `v0.18.0` — so
the docs built from the v0.18.0 tag tell a reader to pull the previous release's
image and `pip install` its pair. Grade it as a version/正确性 defect (the command
runs, on the wrong release), quote both keys, and note that the release-cut
comment above the dict (`# Change this when cut down release`, `:63`) is the
correction site. Recorded #11750, whose Qwen3-Omni page prints that pull at `:37`.

**`extra` carries three version keys that differ by several rcs — quote the right one.**
At v0.27.1rc1 (`3b31886`) `mkdocs.yml:167-173` holds `vllm_version: v0.27.1`,
`vllm_ascend_version: v0.27.1rc1`, `stable_vllm_ascend_version: v0.23.0`,
`release_vllm_ascend_version: 0.27.1rc1`. The `stable_*` key is consumed only by the
developer-preview banner (`docs/hooks/macros_main.py:57`), never by a tutorial body — so
a `[Doc]` report quoting "`extra.vllm_ascend_version: v0.23.0`" (recorded #17951) has
quoted the *stable* key or a `…/en/stable/` build, and the first paragraph owes that
distinction before it ranks a stale-substitution defect: their build's macro may be
`v0.27.1rc1` while the page they read is another branch. Both facts are one
`rg -n 'vllm_ascend_version|stable_vllm_ascend_version' mkdocs.yml` apart; state which
key the quoted value matches and send that command.

**A proposed suffix needs a census, not an impression.** When the report claims a tag
family (`-a5`, `-950dt`, …) has no precedent, count it in one call:
`rg -o --no-filename 'vllm-ascend:[A-Za-z0-9._{} -]+' docs/source/tutorials/models/*.md
| sed 's/.*://' | sort | uniq -c | sort -rn`. At the same pin the released-stack
families are bare `{{ vllm_ascend_version }}` ×46, `-a3` ×33, `-310p` ×14, `-950DT` ×3,
while `-a5` appears only on a *model-branded* tag (`Qwen3.8-27B.md:47` `qwen3.8-a5`) —
so "`{{ vllm_ascend_version }}-a5` invents a family" is a source-verified sentence and
"the `-a5` image does not exist" is not (check the registry, §6 item 4).

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

## 6. When the substitution is right: the literal tag printed beside it

Verified at vllm-ascend **v0.27.1rc1** (`3b31886237ed`) + vllm **v0.27.1**
(`6e448d0e`); recorded for #18041.

Report: `/zh-cn/latest/tutorials/models/DeepSeek-V4.1-Flash.html` says to pull a
`deepseek-v4.1-flash-a3` image; on A3 the 1P1D deployment only starts after the
reporter switches to `v0.27.1rc1-a3`. That "only after" is the finding: the tag
that works *is* the build-time substitution (`mkdocs.yml:167-168` →
`v0.27.1rc1-a3`), so unlike §3 the build is not stale — do not edit `extra`, and do
not ask for a `latest` rebuild.

1. **One grep lists every image reference on the page.**
   `rg -n 'quay.io/ascend/vllm-ascend:' docs/source/tutorials/models/<Page>.md`.
   At this pin `DeepSeek-V4-Flash.md:41` (prose under §4.1's "Attention") names
   `quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731-a3` for A3, while the A3 tab
   at `:85` runs `{{ vllm_ascend_version }}-a3` (`:48` `-950dt` and `:130`
   `nightly-main` are the other tabs). Two A3 instructions, one of them literal:
   that is the defect, and `:41` is the correction site.
2. **A literal tag is not itself wrong at this pin.** Model-branded all-in-one
   images are a page-level convention here (`Kimi-K3.md:61,99` `kimi-k3-a3`,
   `Qwen3.8-27B.md:84` `qwen3.8-a3`, `Dots3-Note.md:110-113`
   `dots3-note-prev-a3`), so grade the *pairing*, not the spelling: a branded tag
   is a frozen vllm/vllm-ascend pair, and the section's command block is written
   for the release line. Precedent for that failure mode — #9092:
   `quay.io/ascend/vllm-ascend:deepseekv4` serving `DeepSeek-V4-Flash-w8a8-mtp`
   died at `EngineCore failed to start` with `AttributeError: 'list' object has no
   attribute 'merge'` raised in upstream `kv_cache_utils.py::is_kv_cache_spec_uniform`,
   i.e. in-image vllm/vllm-ascend KV-spec API drift. Label it precedent plus
   inference: no snapshot carries the image's contents, so ask for
   `pip list | grep -iE 'vllm|torch-npu'` inside that image before claiming a
   mechanism.
3. **The release claim is the scope gate.** `docs/source/user_guide/release_notes.md:5`
   (… "DeepSeek-V4-Flash support on 950DT Products … Models and hardware
   combinations outside this declared scope are not part of the v0.27.1rc1 release
   claim") and that release's own `### Hardware and Operator Support` row (`:26`:
   `DeepSeek-V4-Flash | 950DT Products`) put the page's **A3** §5.2.1 outside the
   validated matrix. A section describing a model × hardware pair the release does
   not claim, and never saying which release its commands belong to, is the doc-side
   root cause: report that missing scope line, not only the tag.
4. **The CI tree is the coverage counter-evidence, and it uses the other image.**
   `tests/e2e/nightly/multi_node/external_dp/config/DeepSeek-V4-flash-w8a8-PD-prefix.yaml:2-25`
   (2 nodes, `npu_per_node: 16`, prefill dp4/tp4, decode dp16/tp1) runs this A3 1P1D
   shape from the repo/release image — which is why "only the branded image fails"
   is consistent rather than contradictory.
5. **A `/zh-cn/` URL puts the `.po` in the fix list.** The Chinese page is the
   gettext build: `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/models/DeepSeek-V4-Flash.po:433-441`
   carries the same sentence in `msgid`/`msgstr`, so (a) while the msgstr matches
   the current msgid it may be quoted as what that page prints, and (b) an
   English-only edit leaves the Chinese page printing the old tag. The `#:` line
   refs still lag and are never citable
   (`missing-guidance-and-contributor-docs.md`).
6. **The linked page may not exist at the pin.** `DeepSeek-V4.1-Flash` does not
   here (`rg --files docs/ | rg -i 'deepseek-v4'` → `DeepSeek-V4-Flash.md`,
   `DeepSeek-V4-Pro.md`; nav `mkdocs.yml:232-233`). Say it in the first message,
   grade the nearest page *as* the nearest page (`index.md` rule 3), and ask for the
   linked page's §4.1 text before grading its wording.
