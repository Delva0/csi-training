# Version / implementation correspondence

Purpose: decide whether a surprising startup failure is a genuine defect, a
missing/extra feature registration, or a configuration that differs from a
known-good path — before writing a theory about model code.

The same reasoning applies one level up on a multi-node deployment: when the pin's own
class **has** the attribute the paste says is missing, the question is which node's
build lacked it, not which reader is wrong —
`../pd-disaggregation/node-build-skew-attribute-error.md` (#10398, reporter-confirmed).

## Build the table first

1. From the evidence capsule: `/workspace/repos/SOURCES.json` lists
   `repo`/`ref`/`commit`/`basis` (and coverage limits per repo) for the versions
   frozen at issue opening. These are the only versions you may cite; do not
   substitute current knowledge for a `null` runtime verification.
2. From the reporter: image tag, `vllm` and `vllm-ascend` versions, `torch` /
   `torch_npu`, `ascend-toolkit`, `nnal`, `ASCEND driver`, plus the full command.
   Where the pair is actually recorded (two files, not `requirements.txt`):
   `docs/source/community/versioning_policy.md` release compatibility matrix, and
   the `.github/workflows/pr_test_*.yaml` `vllm_version` matrix. At v0.14.0rc1 the
   matrix row is `v0.14.1rc1 | v0.14.0/v0.14.1`
   (`docs/source/community/versioning_policy.md:26`) and CI pins
   `[d68209402ddab3f54a09bc1f4de9a9495a283b60, v0.14.1]`
   (`.github/workflows/pr_test_full.yaml:78`). `requirements.txt` does not pin
   vllm at that commit (its only `vllm` mention is a comment at `:19`), so a grep
   for `vllm==` there returns nothing.
   **On an official image the tag is readable in the pinned tree, and it can be newer
   than the snapshot's export.** `ARG VLLM_TAG=` in the image's own Dockerfile names the
   upstream vllm branch it clones: at vllm-ascend `v0.21.0rc1` (`80610e44`) both
   `Dockerfile.310p:36` and `Dockerfile.310p.openEuler:35` read `VLLM_TAG=v0.21.0`, while
   this snapshot exports vllm `v0.13.0`. So a `-310p` (or any official-tag) paste whose
   vllm `file:line` values match neither the export nor its neighbours is **expected**,
   not evidence of a hand-built install — one `rg -n 'ARG VLLM_TAG' <ascend>/Dockerfile*`
   separates the two lanes before any mismatch advice, and the frames are then mapped by
   function name (`gpu_model_runner.py` `profile_run`), never by line number. Recorded
   #10872 spent four turns concluding "the reporter's vllm is not v0.13.0" (both of its
   `file:line` pairs moved in opposite directions) when the Dockerfile tag was the
   citation.
   **The snapshot's vllm pin can be older than the image's.** Recorded #6784:
   `SOURCES.json` pinned vllm `v0.13.0` while the image is vllm-ascend
   `v0.14.0rc1`, which pairs with vllm `v0.14.0/v0.14.1`. A traceback frame
   naming a vllm file absent from the pinned tree is therefore a
   snapshot-coverage gap, not evidence about upstream — say so, ask for
   `pip show vllm`, and do not hunt for the missing line.
   **Three gaps make the row lookup come back empty, and none is a missing matrix.**
   The tag's table can stop at the last rc: at vllm-ascend `v0.11.0` (`2f1aed98`)
   `docs/source/community/versioning_policy.md:23-43` runs `v0.12.0rc1 … v0.7.3` with
   **no** `v0.11.0` row, so read the nearest rc row (`v0.11.0rc3 | v0.11.0 | … |
   8.3.RC2 | 2.7.1 / 2.7.1.post1`) and confirm it against `requirements.txt`'s
   `torch==2.7.1` / `torch-npu==2.7.1` (`:13,28`).
   **The table can instead stop at the previous release *line*, and then the nearest
   row is not usable.** At vllm-ascend `v0.20.2rc1` (`367b8e62da79`, #10525) the top
   row is `v0.19.1rc1 | v0.19.1 | … | 8.5.1 | 2.9.0 / 2.9.0 | 3.2.0`
   (`docs/source/community/versioning_policy.md:26`), `docs/source/conf.py:71-79` still
   substitutes `vllm_version = v0.19.1` / `pip_vllm_version = 0.19.1`, and
   `docs/source/user_guide/release_notes.md` starts at `## v0.18.0` — so quoting that
   nearest row would declare the reporter's CANN 9.0.0 / `torch` 2.10.0 stack **out of
   matrix** when it is exactly the tag's own pair. Read the pair off the tag's component
   table and pins instead: `installation.md:12-18` (CANN `== 9.0.0`, `torch-npu ==
   2.10.0`, `torch == 2.10.0`, NNAL `== 9.0.0`) and `requirements.txt:17,20,38`
   (`torch-npu==2.10.0`, `triton-ascend==3.2.1`, `transformers==5.5.3`). A version table
   the reporter **screenshots** (that issue's `v0.20.2rc1 / v0.20.2 / CANN 9.0.0 /
   2.10.0 / 2.10.0` row) is then the GitHub release page, consistent with those in-tree
   lines — not a hand-edited table, and not something to re-derive from the lagging
   docs. When `SOURCES.json` marks the
   vllm entry `version_role: "dated_reference_only"` (with `reported_ref_status:
   "image_variant_recipe_unverified"`), that pin is a dated reference, not the
   reporter's upstream: the matrix row plus the requirements pins are the pair to
   cite, and a pasted `arg_utils.py:N` matching neither is the image's own build (the
   `ARG VLLM_TAG` paragraph above), not a "nearby build" to hedge about. Recorded
   #6931 cited the snapshot's vllm `v0.13.0` for a vllm-ascend `v0.11.0` image and
   then spent four calls reconciling line numbers the matrix row had already answered.
   The same rule covers every library the snapshot does not carry **at all**: `torch`
   (only the `Ascend/pytorch` `content_export` is present — it holds `torch_npu/` but no
   `torch/`), CANN, the driver. When a traceback's deepest frames are inside one of
   those, the emulation or wrapper line that names the component is already enough for
   the deliverable, and the missing upstream body cannot change the ablation ladder —
   ship the answer and label the mechanism open. Recorded cost of ignoring it: on #1579
   the one snapshot-absent file (`torch/distributed/distributed_c10d.py`) was hunted for
   ~10 K characters of reasoning after the owning `torch_npu` frame was already read, and
   the session ended at the output cap with **no answer**.
3. Recorded pairs:
   - vllm-ascend `v0.11.0rc0` (`00ba07102212c7c7a40de427f09848f2e203c498`) with
     vllm `v0.11.0rc3` (`8ce5d3198d00631a76e1aa02a57947b46bc7218c`), image
     `quay.io/ascend/vllm-ascend:v0.11.0rc0`.
   - vllm-ascend `v0.17.0rc1` (`e20f0b1a...`) with vllm `0.17.0+empty`,
     `torch`/`torch_npu` 2.9.0, driver 25.2.0, `ascend-toolkit`/`nnal` 8.5.1.
   - vllm-ascend `v0.13.0rc1` (`1b5d5abf`) with vllm `0.13.0`, `torch==2.8.0+cpu`,
     **`torch-npu==2.8.0` (`requirements.txt:31`)**, CANN 8.3.RC2 (#6299).

   **The `torch_npu` line has its own history and is not cleared by the release
   matrix.** A pta patch release can carry a fix the base version lacks: `2.8.0`
   leaks memory and `2.8.0.post1`/`.post2` fix it (reporter-confirmed on #6299,
   where the reporter's `torch_npu==2.8.0` was exactly the release's own pin and a
   `.post2` install + vllm-ascend rebuild ended the growth). Read the paste's pta
   line against the known-good patch releases for that pta minor, and treat
   "everything matches the release table" as no evidence in either direction —
   a source-built install inherits the pin, including its defects. The rebuild is
   part of the recipe because the extension is compiled against the installed pta
   (`setup.py:324-333`, `-DTORCH_NPU_PATH` from `pip show torch-npu`). Route:
   `../runtime-memory/host-ram-and-per-rank-rss.md` §Step 0.

## "The docs say it is supported, but my tag cannot do it" — a release-**line** gate

Trigger: the reporter is pinned to one release ("因为客观因素必须用 0.7.3.post1"), a docs
page they link claims the combination (`/en/stable/tutorials/…`), and the paste is a
load-phase failure. Two reads of the pinned tree's own
`docs/source/user_guide/release_notes.md` decide it:

1. `rg -n -i '<feature>' docs/source/user_guide/release_notes.md`, then date each hit
   by the nearest `## vX.Y.Z — date` header **above** it. The file is cumulative and
   its sections are not in version order, so position is not evidence.
2. Compare **release lines (minors)**, not version numbers and not dates. A
   maintenance `.postN` on an older minor is published *after* the minor that
   introduced the feature: at v0.10.0rc1 the file lists `## v0.7.3.post1 — 2025.05.29`
   and `## v0.8.4rc2 — 2025.04.29`, while W8A8 quantization and EP/DP support entered
   with v0.8.4rc2 (`release_notes.md:376`, `:380` "Ascend W8A8 quantization method is
   supported now" #580; EP/DP still experimental in the same paragraph). So
   "0.7.3.post1 is newer than 0.8.4rc2" is true by date and **irrelevant** — features
   do not flow back into a maintenance line. The sentence to ship is "the 0.8.4 line
   and later carry `--quantization ascend` with DP/EP; the 0.7.3 line never did,
   whatever its date", and a newer ref's release notes still carry both headers, so
   this is re-verifiable at any pin.

Recorded #2378: 双机 A2 16 卡, `deepseek-v3-w8a8`, reporter on `v0.7.3.post1`, linking
`/en/stable/tutorials/multi_node.html` (at the pin that page is the 2×A2(64G×8)
DP4×TP4 + `--enable-expert-parallel` + `--quantization ascend` recipe,
`docs/source/tutorials/multi_node.md:60,116-133`), OOM at load. The quantization term
is decisive because it is arithmetic and not a flag: a 671B checkpoint at TP16 is
~42 GiB/card at 1 byte/param (int8) and ~84 GiB/card at 2 bytes/param (bf16), so a
wall at `60.30 GiB already allocated` of `60.96 GiB` fits only the second — the
quant method was not in effect (or the shard width was < 16;
`../deploy-sizing/moe-load-oom-expert-shard.md` Step 0/Step 3 separates the two).

Then read the whole paste instead of the flags the reporter says they used: that
traceback runs through `vllm/engine/llm_engine.py` +
`vllm/executor/ray_distributed_executor.py` + `vllm/worker/worker_base.py`, i.e. the
**V0** engine with Ray TP, which cannot run the doc recipe's `--data-parallel-size`
(DP is a V1 feature) — so "部署过程与官网一致" is false in a way the paste proves and
the reporter cannot see. Deliverable in this order: the release-line gate, the
`/en/v<tag>/` docs branch for their line
(`../tutorial-verification/docs-version-space-and-image-tag.md`), the two per-card
byte numbers, then the two artefacts that settle the residual — the serve line actually
used on each node, and `pip show vllm vllm-ascend` (a `mindie_turbo` frame in the load
path dates the stack by itself: the Turbo integration is a v0.8.5rc1 release note,
`release_notes.md:336-352`, so a stack carrying it is not the 0.7.3 line it claims).
Answering "官网骗人" or "去问维护者" is wrong: both facts are in the pinned docs.

## A self-built vllm beside an rc vllm-ascend is not the release's set

Trigger: the paste's banner or `collect_env` block shows vllm-ascend at an **rc**,
and the `vllm` line is not the branch that rc installs — `0.7.2.dev`, a
`+empty`/`+cu`-style local suffix, a self-chosen branch, or a version two releases
away. Read the pair from the rc's own docs, not from `requirements.txt`:
`docs/source/conf.py` holds `vllm_version`, and the install page consumes it.

| line | at v0.7.1rc1 |
|---|---|
| the pair | `docs/source/conf.py:65` `'vllm_version': 'v0.7.1'` |
| the recipe | `docs/source/installation.md:120` `git clone --branch v0.7.1 …`; `:149` clones `--branch` the `vllm_version` substitution |
| what the release itself spells out | `installation.md:141,171` the manual `torch_npu-2.5.1.dev20250218` wheel — the one line a reporter is most likely to match |

**A matching `torch-npu` line excludes nothing.** It is one of three lines, and it
is the one the install page always prints, so it says the reporter followed the
pta step — not that the trio is the release's. Recorded **#488** (2025-04, 910B,
cards 4+5, `deepseek-r1-distill-qwen-32B`, `--tensor-parallel-size 2
--gpu-memory-utilization 0.8 --max-model-len 20480`, ~100 short requests) pasted
`vllm: 0.7.2.dev`, `vllm_ascend: 0.7.1rc1`, `torch-npu: 2.5.1.dev20250218`,
`transformers: 4.49.0`, and said in the same block that it was installed with the
**older 0.7.2 page's** manual-compile recipe (which is how a `0.7.2.dev` build
appears beside a `0.7.1rc1` wheel). The recorded answer wrote "版本不匹配：排除" from
the matching `torch_npu`/`requirements.txt` line and shipped a device-fault
ranking with the version lane last. The maintainer's reply named the versions —
"you are using a old version vllm / vllm-ascend / torch_npu" — and asked for vLLM
Ascend `0.7.3rc2`; the reporter's next comment is the resolution: "i tried latest
version (vLLM Ascend 0.7.3rc2) and the problem solved."

So when the trio is mixed, or the rc is several releases old, **rung 1-2 of the
action ladder is the install-and-re-run**: the same installation page, the current
rc, the whole stack, then the identical request loop. Two reasons it outranks the
plog here: a snapshot that pins one commit cannot contain a fix that landed after
it (SKILL.md rule 16's constraint, not only for experimental modes), and the
reporter can run it in one command. Keep the mechanism answer ranked below the
upgrade and say which lane would change if the upgrade changes nothing — that is
what makes the pair a diagnosis instead of a shrug.

Recheck at the pin: `sed -n '60,70p' docs/source/conf.py`,
`rg -n 'branch v0\.|vllm_version' docs/source/installation.md`. On a modern tree
the pair and the compatibility row are in
`docs/source/community/versioning_policy.md` plus the `.github/workflows/pr_test_*.yaml`
`vllm_version` matrix (step 2 above); the v0.7.x-era tree has no
`docs/source/community/` directory at all, which is itself the answer to "where is
the matrix".

## A `vllm_ascend/**` frame the pin does not contain: vendored file, or a relocated module

The mirror of the rule above. There the missing file is a **snapshot** limitation
and the finding is a data gap; here the missing file is **the reporter's own
code**, and the finding is that the release does not own the failing path.

- Trigger: the paste's `file:line`, or a traceback frame, points inside
  `vllm_ascend/models/<x>.py`, `vllm_ascend/ops/<x>.py` etc. and
  `ls vllm_ascend/models/` at the pin has no such file. One call each:
  `ls <repo>/vllm_ascend/models/`, `rg -ln -i '<model-or-symbol>' vllm_ascend/`,
  `rg -n '<Class>' vllm_ascend/models/__init__.py`.
- Discrimination: an in-tree file whose *line numbers* shifted is version drift
  (re-grep, cite the pin); a file that does not exist at the pin while the same
  module is mentioned only in a comment or a config docstring is a **site-vendored
  file** — after the relocation check at the end of this section has come back negative. Recorded #3107 at vllm-ascend v0.9.1rc2 (`b9f715d`): the log names
  `vllm_ascend/models/pangu_moe.py` at lines 526-877, the pin's
  `vllm_ascend/models/` has no `pangu_moe.py`, `models/__init__.py` registers no
  Pangu class, and the only in-tree mention of the family is the torchair comment
  at `vllm_ascend/ascend_config.py:78`. By v0.11.0rc0 `PanguProMoEForCausalLM`
  **is** registered (see the section below), so the out-of-tree file is a
  pre-upstream port, and "drop it and use the release's class" is a real rung.
- What the answer may still claim: the framework functions that file calls
  (`set_ascend_forward_context`, `ops/fused_moe.py`, the attention metadata
  builder) are in-tree, so their behaviour, their fallbacks and their
  `path:line` are citable — say explicitly that the model arithmetic is not.
- What to ask for: the file itself with its line numbers, its registration patch
  (`ModelRegistry.register_model` / `register_oot`), and whether the same failure
  reproduces with the release's own class. Do not reconstruct their module from
  the log's debug prints; the printed shapes and values are their instrumentation,
  not the framework's.

**The other way a downstream file can be absent: the release moved it.** Before calling an
absent `vllm_ascend/**` file a vendored port, ask whether a refactor removed or relocated
it — the two answers send the reporter in opposite directions.

- Two calls: `rg --files -g '*<file-stem>*' vllm_ascend/` (and `ls` the containing
  directory) shows the relocation inside the tree; a **history body** shows the deletion.
  Some issue bodies paste a factual `git diff --stat` table between two refs and name a
  removal verbatim, which no source read can show once the only pin is the newer tree:
  `scripts/history-titles.sh -i '<stem>'`, then
  `scripts/history-body.sh <id> -g '<stem>|deleted|removed'`.
- Recorded #18518 at v0.26.0rc1 (`f2f74a16`): the paste cites
  `vllm_ascend/worker/pcp_utils.py:811` / `:958` and `model_runner_v1.py:2106`, the pin has
  no `pcp_utils.py` at all (`rg --files -g '*pcp*' vllm_ascend/` → only
  `worker/v2/pcp_manager.py`), and the corpus carries both halves of the proof — #13973's
  table row `worker/pcp_utils.py | −2387 (deleted)` beside `worker/v2/pcp_manager.py | +107
  (new)`, and #14320's sentence "the v0.26 PCP-removal refactor (#12592 'remove PCP from
  MRV1')". So PCP was **upstreamed** (`vllm/v1/worker/gpu/pcp_manager.py`), and the
  reporter's file belongs to ≤ v0.23.x — either an older tree or a `collect_env` that does
  not describe `site-packages` (one existence check plus `pip show` decides, and it costs
  the reporter nothing).
- Absence is not yet the answer. The deliverable is the **current owner** and whether the
  reported number still occurs there: at the #18518 pin the padding became
  `max(per_rank_tokens) * pcp`, which reproduces the pasted `index value 8194 exceeds
  bounds 8192` from the same `[6661, 1531]` schedule — so "0.26 moved it, so 0.26 is fine"
  would have been wrong (`../mixed-batch-splits/pcp-padding-vs-token-budget.md`). Re-derive
  the arithmetic at the new site before writing a verdict, and label it arithmetic, not a run.

## A `SyntaxError` the pin's own file cannot produce: file integrity, not a parser question

- Trigger: the run dies during `adapt_patch(is_global_patch=True)` (or any import of
  `vllm_ascend`) with `File "…/vllm_ascend/….py", line N` / the quoted source line /
  `SyntaxError: invalid syntax` — or `IndentationError: expected an indented block
  after 'try'` — before any device open, weight load or graph capture. A source
  checkout in the image (`/vllm-workspace/vllm-ascend`) is what makes this possible;
  a wheel install cannot have it.
- One call decides whether the tag can raise it: parse the **pinned** copy and print
  the reported range: `python3 -c "import ast; ast.parse(open('<file>').read())"`,
  `wc -l <file>`, `sed -n '18,32p' <file>`. Recorded #3920 at vllm-ascend
  v0.11.0rc0 `00ba07102212`:
  the reported `except RuntimeError:` at `patch_mamba_config.py:28` does not exist
  (`ast.parse` exit 0, 100 lines, no `try`/`except` anywhere in the file) while the
  other four frames of the same traceback match the pin line for line
  (`platform.py:67` `adapt_patch`, `utils.py:246`, `patch_common/__init__.py:20`) —
  the mismatch is that one file on the reporter's disk, and the four matching frames
  are the evidence that the rest of the image is the tag they named.
- Verdict and ladder in the first message, as a file-integrity answer:
  `python -c "import vllm_ascend.patch.platform.patch_common.patch_mamba_config"`
  reproduces in a second with no NPU; then
  `git -C /vllm-workspace/vllm-ascend rev-parse --short HEAD; git status --short`;
  `git checkout -- <file>` (or the tag) restores it. Nothing about the model,
  CANN or the device is involved, and the reported `file:line` is the only artefact
  needed to say so.
- Do not spend turns on a synthetic parser-case matrix. The recorded run built one
  (`orphan except` → `SyntaxError` line 3, empty `try` body → `IndentationError`) to
  confirm the corruption shape after the verdict was already established, and
  finished at 86 % of the output cap (14 088 / 16 384 completion tokens) — the
  `sed`/`ast.parse` output had already answered it.

## The version string names the nearest tag; the tree can be a later line

A report that prints `vllm-ascend: v0.9.2rc1(e7d32ed3f18d…)` gives the *nearest
reachable tag* plus a commit (derivation: `../image-dependency-faults/installed-version-vs-checkout.md`).
It does **not** say the tree is that release: a main-branch build keeps the previous
tag's number. Decide it from the reporter's own `file:line`s against the pinned trees
— one grep per candidate ref, and the direction matters more than the label:

- #2064 printed `v0.9.2rc1(e7d32ed…)` and its traceback carried
  `vllm_ascend/utils.py:321 num_hidden_layers = get_max_hidden_layers(hf_config)` and
  `vllm_ascend/worker/model_runner_v1.py:71 from vllm.v1.worker.utils import
  (bind_kv_cache, gather_mm_placeholders, …)`. At the case's pin (v0.9.2rc1
  `b5b7e0e`) there is **no** `get_max_hidden_layers`, and `model_runner_v1.py:66`
  imports `bind_kv_cache` from `vllm.v1.utils`. The next release whose tree this
  library holds, v0.10.0rc1 (`4604882`), has `get_max_hidden_layers` at
  `utils.py:291-292`, the call at `utils.py:321`, and the same import block at
  `model_runner_v1.py:70-71`. So that tree is a **0.10-dev build**, and its
  counterpart is vllm 0.10.0 — not the vllm 0.9.2 that was installed.
- The consequence is a **pairing** answer, not a patch. `bind_kv_cache` is at
  `vllm/v1/utils.py:278` in vllm v0.9.2 and at `vllm/v1/worker/utils.py:120` in vllm
  v0.10.0, so `ImportError: cannot import name 'bind_kv_cache' from
  'vllm.v1.worker.utils' (<a real path>)` says the plugin is written for the newer
  vLLM while the installed vLLM is the older one. Pair them; do not edit the import,
  and do not call the build a stale nightly without checking the newer tree.
- Read the **direction** before naming it: the same message is produced by a plugin
  *newer* than vLLM (symbol not there yet) and by one *older* than vLLM (symbol moved
  away). The reporter's own import line compared against both candidate vLLM trees is
  what separates them; "the pin does not export it" cannot, because the pin may be
  older than the reporter's build.
- One reply settles it: `pip show vllm vllm-ascend`, `python -c "import vllm,
  vllm_ascend; print(vllm.__file__, vllm_ascend.__file__)"` (in #2064 `vllm` resolved
  to a source tree at a CI workspace path, `/__w/…/vllm-empty/vllm/`, not
  site-packages), `git -C <checkout> log -1 --format=%H`, and the two source lines
  their traceback printed.
- If the snapshot holds no release near enough to match their lines, say the tree is
  **later than the pin** and label the exact release as inferred — then still give the
  pairing verdict, which does not depend on the label.

## A `cannot import name` at `from vllm.utils import …`: count the importers, then read the newer pin

The `patch/**` variant of the section above. Reported #4863: image
`quay.io/ascend/vllm-ascend:v0.11.0rc3-openeuler`, `Qwen3-Reranker-8B`, `--task score`,
`ImportError: cannot import name 'STR_DTYPE_TO_TORCH_DTYPE' from 'vllm.utils'` raised
inside `make_arg_parser` → `pre_register_and_update` →
`patch/platform/patch_mamba_config.py:6`. vllm-ascend `v0.11.0rc3` (`b6d63bbd5294`).

1. **The reported line is one of three, so a one-line edit is not a repair.** At that
   pin the symbol is imported unguarded at `patch/platform/patch_mamba_config.py:6`,
   `worker/worker_v1.py:38` (used `:134`) and `worker/model_runner_v1.py:72` (used
   `:301`), and only two files in the package use `vllm_version_is` at all
   (`utils.py`, `models/layers/mla.py`). Fixing line 6 therefore relocates the identical
   `ImportError` to worker start. One `rg -n '<symbol>' <vllm-ascend>/` before any
   advice, and quote the count.
2. **The module *shape* in the traceback names the upstream generation.** `vllm/utils.py`
   is the monolith (vllm v0.9.1 `b6553be1bc75`: the dict at `utils.py:174`), while
   `vllm/utils/__init__.py` is the split package — and in the package generation the
   top-level name is **gone**: at vllm v0.18.0 (`bcf2be96`) the dict lives at
   `vllm/utils/torch_utils.py:32` and `vllm/utils/__init__.py` has zero occurrences. So
   "the import is stale against a *newer* vllm" and "the pair itself is inconsistent"
   are the two live lanes, and the monolith/package path is the tell.
3. **Read a newer vllm-ascend pin: it names the new home and often shows downstream has
   already migrated.** At `v0.18.0rc1` the same file is
   `patch/platform/patch_mamba_config.py:9 from vllm.utils.torch_utils import
   STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size` and `patch_mamba_config_310.py:11` /
   `worker/worker.py:41` import it the same way. That converts "which import do I patch"
   into a cite-able answer plus a bounded site list — and it is a check on the *pin the
   case already holds*, not a search.
4. **Check whether the patched body is even live for the reporter's model.** This file
   rewrites `MambaModelConfig.verify_and_update_config`; a reranker/embedding head has no
   mamba layers, so only its import line is on the failing path. Say that, then pair the
   versions (`docs/source/community/versioning_policy.md:25` at that pin pairs
   `v0.11.0rc0 | v0.11.0rc3`); do not ship a behaviour theory for a body that never runs.
5. Conclusion state: no reporter follow-up on #4863 in this snapshot — items 1-4 are
   source facts read at the two pins, and the pairing verdict stays labelled as inferred
   where the snapshot holds no vllm of the reported generation.

## Check "is model X supported here" in the source, once

- Registration: `vllm_ascend/models/__init__.py` `register_model()` is the
  authoritative list of Ascend overrides. In v0.11.0rc0 it registers
  `Qwen3VLMoeForConditionalGeneration` and `Qwen3VLForConditionalGeneration`
  (both to `vllm_ascend.models.qwen2_5_vl_without_padding`), `Qwen2VL...`,
  `Qwen2_5_VL...`, DeepSeek V2/V3/V3.2, `Qwen3MoeForCausalLM`,
  `Qwen3NextForCausalLM`, `PanguProMoEForCausalLM`. A VL key registered **twice**
  under an env gate (Qwen2.5-VL: `USE_OPTIMIZED_MODEL` picks `qwen2_5_vl` vs
  `qwen2_5_vl_without_padding` at v0.9.2rc1) means the flag is itself a
  known-good-baseline row — that is the first ablation for a VL accuracy report,
  see `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`.
- Upstream files: vllm v0.11.0rc3 ships `vllm/model_executor/models/qwen3_vl.py`
  and `qwen3_vl_moe.py`. So "Qwen3-VL unsupported in this image" is **false**
  for that pair — verify before suggesting it.
- Release notes and `docs/source/user_guide/support_matrix/supported_models.md`
  state what the release claims (v0.11.0rc0 claims Qwen3-VL support, PR #3103).
- A model name newer than the image (e.g. a `Qwen3.5` path against a snapshot
  whose vllm has `qwen3_5.py`) means: check both sides' model/config files exist
  at the pinned commits, then rely on CI tests
  (`tests/e2e/multicard/4-cards/test_qwen3_5.py` for TP=4 Qwen3.5-27B in
  v0.17.0rc1) as the known-good baseline.

## Read the environment dump's gaps before its contents

An empty `CANN:` field (or a missing component version) in the issue's
`collect_env.py` block is usually the *collector*, not the install:
`collect_env.py` reads hard-coded default roots (`collect_env.py:300` at
v0.20.2rc1 reads
`/usr/local/Ascend/ascend-toolkit/latest/<arch>-linux/ascend_toolkit_install.info`),
so an install under `/usr/local/Ascend/cann-9.0.0` or a conda prefix reports
`None`. Reporter-confirmed on #10525: the path existed, "但这个检查脚本确实没检查到".
Cross-check `ASCEND_HOME_PATH`, `python3 -c "import acl; print(acl.get_soc_name())"`
and `which npu-smi`, and ask for the actual install root before building a
thesis on a missing component. The same rule applies to components the image
installs but a hand-built environment must add itself (NNAL/ATB:
`../image-dependency-faults/nnal-atb-missing-library.md`).

**The `ENV Variables:` block has its own filter — read what it cannot contain.**
`get_env_vars()` (`vllm-ascend/collect_env.py:304-316` at v0.19.1rc1 and v0.21.0rc1)
prints only keys registered in upstream `vllm.envs.environment_variables` plus keys
starting with `TORCH`, `PYTORCH`, `ASCEND_` or `ATB_`. `HCCL_*`,
`GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `MOONCAKE_*` and `OMP_*` therefore **never
appear**, however faithfully the reporter copied the tutorial's env block. Do not rank
"the reporter skipped `HCCL_BUFFSIZE`/`HCCL_INTRA_*`" on that absence: #10724's launcher
exports `HCCL_IF_IP`, `HCCL_SOCKET_IFNAME`, `HCCL_BUFFSIZE`,
`HCCL_OP_EXPANSION_MODE`, `HCCL_INTRA_ROCE_ENABLE`, `HCCL_CONNECT_TIMEOUT`,
`OMP_NUM_THREADS` and `MOONCAKE_CONFIG_PATH`, and its ENV block (20 keys, all
`ASCEND_*`/`ATB_*`/`TORCH*`/`PYTORCH*`) carries none of them; the recorded #9872
triage came within one paragraph of shipping that inference as its leading cause.
Ask for `env | grep -E 'HCCL|SOCKET_IFNAME|MOONCAKE'` on each node.

## The launch environment is one environment per worker process

On `--distributed_executor_backend ray` (and any multi-node launch) every worker is
a separate interpreter on its own node, started by that node's raylet: a `source
set_env.sh` or `export …` in the driver's shell does **not** reach it. Requirements
that are per-process therefore have to be checked inside each node's worker, not in
the shell that submitted the job:

```bash
pip show vllm vllm-ascend | grep -E 'Name|Version'
python -c "import triton; print(triton.__version__, triton.__file__)"
which bishengir-compile
```

Run it once per node (or carry the values through ray `runtime_env` `env_vars` /
`worker_process_setup_hook`), and ask the reporter for that output rather than for
"the versions".

Before the per-worker environment, check the cluster's resource *view*: a paste that
carries `Current node has no NPU available … 'GPU': 8.0` (or the preceding `The
number of required NPUs exceeds …` warning) never reached a worker at all — the
nodes advertise `GPU` where the platform asks for `NPU`, and one `ray status` per
node settles it: `ray-npu-resource-label.md`.

Recorded #3098 (vllm-ascend v0.10.2rc1, ray with 2 nodes x 8 910B4,
Qwen3-Next-80B-A3B): that model imports Triton at module import
(`vllm_ascend/ops/fla.py:11-12`, `ops/casual_conv1d.py:14-15`), and the release's
own tutorial makes triton-ascend plus the BiSheng toolkit a manual, **aarch64-only**
install (`docs/source/tutorials/multi_npu_qwen3_next.md:41-68`, with
`source /usr/local/Ascend/8.3.RC1/bisheng_toolkit/set_env.sh` at `:54` and repeated
at `:75-78` as a precondition for inference; the x86_64 tab is "Coming soon" at
`:67-68`). In a multi-node launch that install and that `source` have to exist in
every node's worker, and a node that skipped either fails alone — which is exactly
the "model X deploys here, model Y 报错" split of that report, and why the pair of
models is not the first evidence. The same release note
(`docs/source/user_guide/release_notes.md:9`) says EP and MTP "doesn't work with
this release" (Qwen3 Next, PR #2917), so those flags are ablation rows there, and
its `:89` command (`--tensor-parallel-size 4 --max-model-len 4096
--gpu-memory-utilization 0.7 --enforce-eager`) is the only validated shape — a
cross-node TP16 launch has no published recipe at that ref. For the wheel/compiler
pairing itself, `../image-dependency-faults/triton-ascend-pair.md`.

## Diff the reporter's configuration against a known-good baseline

CI e2e tests and the pinned vllm-ascend `docs/source/tutorials/*.md` are the shipped known-good command
lines; they carry env vars (`HCCL_BUFFSIZE`, `TASK_QUEUE_ENABLE`,
`PYTORCH_NPU_ALLOC_CONF`, `ASCEND_RT_VISIBLE_DEVICES`) and flags
(`--max-model-len`, `--compilation-config`, `--async-scheduling`,
`--additional-config`) the reporter may have added or dropped. Enumerate the
differences, then rank them by whether they can plausibly produce the observed
error class. In the recorded #7477 case the config matched the CI shape closely
and the failure turned out to be hardware; in #3543 the single missing flag was
decisive.

## Reporting a non-reproduction

Say exactly what was compared: same vllm/vllm-ascend pair, NPU model and per-card
HBM, driver/CANN, command line. A clean run on different hardware is a useful
data point, not a refutation of a reporter-confirmed fix on theirs.
