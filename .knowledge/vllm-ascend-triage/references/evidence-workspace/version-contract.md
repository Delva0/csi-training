# The version contract: the pin, the image recipe, and the dependency line

## Scope

Rule 5 of `SKILL.md` in full. The rule itself — read `SOURCES.json`, give `path:line`
plus the commit, state recheck conditions, answer inside the pinned release — is
stated there. This page carries the *rest of the contract*: the dependency stack,
the image recipe that can override the pin, the tag rung and its direction, and the
reports where the reporter's own older release is the control. Open it when any of
these is part of the ranking:

- the env dump's `torch_npu==` / `torch==` / `transformers==` line;
- the image tag, or a `docker run` whose recipe installs its own dependency set;
- a title/body release disagreement, or a `…/en/latest/…` doc URL;
- the reporter says an older release worked (symptom or capability);
- the trio matches but the pin is an rc and the trigger is a feature that crosses it.

Report the pin you verified, quote what the pinned tree *does* carry (the
registration, the dead lever, the coverage absence), and say plainly when "is it
fixed" cannot be answered from the snapshot.

## The doc URL: `latest` is main, not the pin

A report citing `docs.vllm.ai/.../latest/…#<section>` is quoting a doc that may
postdate the pinned release: check that section exists in the pinned tree, answer
inside the pinned version, and if it is absent say so and ask for the text rather
than assuming what it says. `latest` is the Read the Docs build of main
(`READTHEDOCS_VERSION`) and each release is `/en/vX.Y.Z/`, while the tag a page
prints is substituted at build time from `mkdocs.yml extra.*` — so a tag/flag
disagreement is graded against the docs build with those two greps, not by hunting
the upstream `Literal` that owns the value
(`../tutorial-verification/docs-version-space-and-image-tag.md`, #16954).

## Title one release, body another

When the title names one release and the body's launch log another (recorded
#11231: title v0.21.0rc1, body v0.22.1rc1; #18315: title v0.27.1RC1, env
v0.26.0rc1), say which one you verified and give the one command the reporter runs
on *their* tree (`sed -n 'A,Bp' <file>` around the cited line) instead of asking for
another launch — and when a **newer pin in the snapshot already carries the repair**,
quote it: that settles "is it fixed upstream" without PR or comment access.

A reference page states the pins it was verified at: when those equal the refs in
`SOURCES.json`, its `path:line` are citable as they stand — re-grep only what the
page itself marks as version-specific, and spend the saved turns on the delivery.

## A dependency's patch release is part of the contract

`torch_npu` (pta) `2.8.0` leaks memory and `2.8.0.post1`/`.post2` fix it — the
pinned tree says so itself (`docs/source/user_guide/release_notes.md:197` at
v0.13.0rc1, under `## v0.11.0rc1`: "The memory may be leaked … This is a bug from
torch-npu"), reporter-confirmed on #6299 (`--no-enable-prefix-cache` changed
nothing, the `.post2` install plus a vllm-ascend rebuild ended it) — while
v0.13.0rc1's **own** `requirements.txt:31` pins `torch-npu==2.8.0`, so an install
that matches the release table exactly can still carry the defect. Read the env
dump's `torch_npu==` against the patch releases before attributing a symptom to
vllm-ascend code, and keep the mechanism open when only the fix is confirmed
(`../runtime-memory/host-ram-and-per-rank-rss.md` §Step 0).

**And a release can pin the *broken* patch.** vllm-ascend v0.26.0rc1's own
`requirements.txt:17` and `pyproject.toml:21` are `torch-npu==2.10.0.post4` — the
one patch release on which `/stop_profile` segfaults while `.post2`/`.post6` serve
(recorded #15983, later confirmed in-thread as a `torch_npu` profiler-backend
compatibility issue). There "install what the release pins" is the failing side, so
the patch change *is* rung 1 and the pin is the finding
(`../observability-and-logs/online-profiler-stop-segfault.md`).

**The image recipe can override that pin, so an off-pin `torch_npu==` is not by
itself a mismatch.** At v0.13.0rc2 `requirements.txt:31` says `torch-npu==2.8.0`
while every recipe installs `'torch-npu==2.8.0.post1'` on a `cann:8.5.0-*` base
(`Dockerfile:76`, `Dockerfile.openEuler:75`, `Dockerfile.310p:61`,
`Dockerfile.a3:75`): a paste from that tag showing post1 is the image's own pairing.
One grep — `rg -n 'FROM |torch-npu' <repo>/Dockerfile*`, or
`scripts/image-recipe-inventory.sh <root> --soc <variant>` §1b — separates "the tag
ships this" from "the reporter changed the env", and the first reading rules out both
the hand-modified-env diagnosis and the "use the release image" rung (#6271, image
tag named by the reporter afterwards; the collision
`../image-dependency-faults/host-cxx-compile-header-conflict.md` describes ships in
that tag).

**Any other dependency line in the env dump is checked the same way, from the pin's
own requirement files, before the consuming code is read.** The two trees can
disagree by shape: vllm v0.17.0 `requirements/common.txt:10` is
`transformers >= 4.56.0, < 5` while vllm-ascend v0.17.0rc1 `requirements.txt:37` is
`transformers>=4.57.4` — floor only — so a `transformers==5.5.0.dev0` install
satisfies the plugin and violates upstream, and the tree's per-model
`tests/models/registry.py` `min_transformers_version` (`:40`, GLM-5 `5.0.1` at
`:283`, Kimi K2.5 none at `:845`) says whether the family *needs* the newer line,
with `release_notes.md:52` for the documented exception. Recorded #7876 spent 25
turns and 1.2 M prompt tokens inside the model's processor for an init-time
tokenizer assertion whose answer was that range plus the paste's own two
`incorrect regex pattern … fix_mistral_regex=True` warnings, and shipped no message
text. One call:
`bash <skill_root>/scripts/dep-range-check.sh <pkg> <ver> <vllm-root> <ascend-root>`.

## A mixed trio is itself the finding

**A self-built vllm beside an rc vllm-ascend is a version-contract defect even when
the torch line matches.** #488 pasted `vllm: 0.7.2.dev` + `vllm_ascend: 0.7.1rc1`
(from an older page's manual-compile recipe) while the rc's own pair is vllm
`v0.7.1` (`docs/source/conf.py:65`); that answer excluded the version lane and
ranked a device fault first, and the issue closed on the maintainer's "install
0.7.3rc2" plus the reporter's "tried latest … the problem solved". With a mixed or
several-rcs-old trio, make the install-and-re-run **rung 1-2** and keep the
mechanism below it — a snapshot pinning one commit cannot hold a later fix
(`../startup-failures/environment-correspondence.md` §*A self-built vllm beside an
rc vllm-ascend*).

## The old-tag rung, and its direction

**An old image tag with no mismatch in it is still a rung.** When the tag in the
report is two or more rcs (or ~a month) older than the analysis date, the snapshot
predates any fix by construction, so for a **garbage/degenerate output or a crash on
a recently registered model** the tag is rung 1: one line in the first message —
"re-run the same body on the current image; `<tag>` predates the fix" — and the
ranked mechanism below it for the case where the re-run still fails. #3806 pinned
v0.11.0rc0 while rc2/rc4 were current; the same `curl` body on
`quay.io/ascend/vllm-ascend:latest` returned the correct table (reporter-confirmed),
and the recorded session never offered the upgrade
(`../accuracy-benchmarks/vl-vision-tower-degeneracy.md` §*Fourth pin*). Do not read
this as licence to demote a real mechanism: quote what the pinned tree *does* carry
and say that "is it fixed" cannot be answered from the snapshot.

**And when the paste or a later comment already shows the reporter rebuilt from
source and it still failed, the tag rung is spent** — recorded #5484 (reporter tried
`vllm-ascend==0.11.0`, then `0.11.0-dev` `pip install -e`, same GE `E89999`
infershape error): rank the checkpoint/flag axis and say the upgrade was already
tried, instead of repeating it as rung 1.

**Read the direction of the comparison first — the tag rung above is
one-directional.** It applies when the **older** tag is the broken one. When the
reporter's own control runs the other way (old tag structures the document
correctly, new tag garbles it, same weights/card/command — #4730: `v0.11.0rc0`
good, `v0.11.0rc1` "html结构都错乱"), upgrading cannot help: they are already on the
newer side. #4730's public follow-up (2025-12-05) also has rc2 **and** rc3 failing,
so "the next rc fixes it" is not a rung either. Then the first message owes three
things instead: the zero-restart request arithmetic (`finish_reason` /
`usage.prompt_tokens` against the tutorial's image-token envelope), the one-command
stack dump taken **inside each image** (a tag moves vllm / torch_npu / CANN /
transformers together, so the delta is not yet a vllm-ascend claim), and the
within-image one-restart rows (`USE_OPTIMIZED_MODEL=0`, `--enforce-eager`, a
text-only request, the checkpoint's dtype against the engine's) —
`../accuracy-benchmarks/harness-vs-engine-attribution.md` §*When the older image is
the good one*, `../accuracy-benchmarks/vl-vision-tower-degeneracy.md` §*Fifth pin*.

## A capability that worked on an older release is a regression before it is a mechanism

The direction rule above is about a *symptom*; this is its twin for a *deliverable* —
a profiling collection that produced no table, a mode that now refuses to start, a
flag that no longer reaches its consumer — where the body says the same combination
worked before, or works on the sibling node. Then the ranked answer is "regression
between `<works>` and `<fails>`, here is the one command that dates it", and every
vendor-stack limitation (`msprof` cannot record graph-replay tasks, CANN skips a
table under graph mode) sits *below* it as a labelled hypothesis, because the older
release is the control that refutes it. Ask which release did it before believing
the premise: a release that never collected is coverage, not a regression. Recorded
#12241 (`releases/v0.23.0`, `enable_npugraph_ex` + MTP, no `kernel_details.csv`) had
a `msprof`/graph-replay limitation ranked first when the reporter wrote "这个组合旧版本
一直支持"; the maintainer restored it and the reporter confirmed "现在可以采集了" —
`../performance-throughput/profiler-no-kernel-details-graph-mtp.md`.

## A matching trio does not close the lane when the pin is an rc

**A matching trio does not close the version lane when the pin is an rc and the
trigger is a feature crossing that pin never tests.** An accuracy/numeric report
whose own controls are "feature A off: fine, feature B off: fine" points at the
intersection, and a fix that lands after the pin is invisible in the snapshot — so
the *next rc/final* is rung 1, not the mechanism. Recorded #12060 (`v0.20.2rc1`
image, PP2+TP2 + chunked prefill, 首token wrong) closed on the reporter's "tried the
latest version, it works fine", while the recorded run spent its whole budget
deriving the PP token-id path (`../pipeline-parallel/pp-chunked-prefill-numeric.md`).
