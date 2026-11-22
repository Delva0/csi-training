# Tutorial / documentation verification: the report is the deliverable

Scope: the task is to *judge a document*, not to explain a crash. Entry shapes:

- a community-test row (`[Community Test]`, 众测任务书: 蓝区镜像 / 权重链接 /
  配套 HDK / 卡数 / 交付件 = 测试报告, "资料验证 + 部署测试"), where the answer is
  a report covering deploy script, curl and `vllm bench` commands, the doc's
  完整性/易用性/正确性 scores and a FAQ;
- a `[Doc]` / `[Usage]` issue reporting that a tutorial step, command or number
  does not work as written — including a command from the repo's own
  `examples/*.py` whose docstring/argparse block is the defect;
- a `[Doc]` issue whose defect is an **absence**: the Quick Start should mention a
  mirror/flag/section that already exists elsewhere, or the contributor docs
  never state a prerequisite the image or `setup.py` demands — then the three
  deciding greps are "does it exist", "does anything link to it", "do the sibling
  pages already carry it" (`missing-guidance-and-contributor-docs.md`); or the page
  *promises* a step that exists nowhere ("install extra operator", "am I expected to
  compile `X`?"), where a fourth read decides the report — the same release's own
  `release_notes.md` (same page, §#5158);
- any "verify this recipe" request where the reporter's hardware is not
  available to you and the evidence is the **pinned tree**, not a run.

## Route table — read the row, then the detail below

The detailed *Conditional routes* sit ~90 lines below, after the rules, so a short
window on this file used to return the rules and one route header. This is the same
set by shape; match the **reporter's own wording**, not the title's category word.

| the report is … | page |
|---|---|
| a 众测任务书 row (蓝区镜像 / 权重链接 / 配套 HDK / 卡数 / 交付件 = 测试报告) | `community-test-report.md` |
| a sample request or `vllm bench serve` answering 404, or a first start that "加载 3 小时 / 高并发读取" | `serve-contract-and-weight-predownload.md` |
| asking for *added* guidance, a cross-link, the contributor path, or a mirror/registry | `missing-guidance-and-contributor-docs.md` |
| a flag, path, count, install pin or printed request body that is wrong as written | `tutorial-command-audit.md` (its worked pages first) |
| a printed image tag, a `…/en/latest/…` URL, any version disagreement, or a hardware tab printing another generation's tag (「此处镜像提供错误」 on the `Atlas 800I A3` tab) | `docs-version-space-and-image-tag.md` (§7 for the tab/suffix shape) |
| the doc's log/flag claims against the code | `../observability-and-logs/request-output-logging-flags.md` |
| a doc accuracy/performance number that looks anomalous | `../accuracy-benchmarks/index.md` |
| an actual failure log whose owner may be the reporter's box | leave the topic — the closing paragraph of *First call* ("Leave this topic when …") names the index to open |

## First call: does a worked page already carry this row?

`rg -n '<model family or row id>' <skill_root>/references/tutorial-verification/` —
the worked-page tables in `tutorial-command-audit.md` record verified defect sets by
page name (`Qwen3-32B-W4A4`, `Qwen3.5-27B`, `Qwen3-Omni-30B-A3B`,
`Qwen3-VL-30B-A3B`, `Qwen-VL-Dense` — row `hjh2`, the Qwen3-VL-8B row —
`InternVL3.5`, `Qwen3.5-397B-A17B` §5.6, `Qwen3-Coder-30B-A3B`), reached from the
`tutorial-command-audit.md` bullet under *Conditional routes* below. A hit means the
report's `资料验证` §1 is writable on turn 2 from the table; spend the rest of the
budget on what the table does not have. The table is not exhaustive — it is the floor,
not the ceiling.

Cost of skipping it (recorded #11759 rerun: v0.18.0 image, the same row as the second
worked page, 15 assistant turns, `completed`): it read this index to `:120` and
`community-test-report.md` to `:80` of 352 — both cuts before the section that mattered —
never opened `tutorial-command-audit.md`, re-derived the benchmark-section,
`max_completion_tokens`, card-count and sizing rows over ten turns, **dropped the one row it
explicitly considered** (`pip install transformers==4.51.0` at page `:51` set aside as
"a separate env", while its own deploy script installs benchmarks inside the serving
image, whose pins are `transformers >= 4.56.0, < 5` at
`vllm/requirements/common.txt:10` and `>=4.57.4` at `vllm-ascend/requirements.txt:42`),
and re-derived as new the `conf.py` substitution lag that
`docs-version-space-and-image-tag.md:62-70` already carried. A window read of a protocol
page is not a read.

Leave this topic when the report contains an actual failure log **and** the
reporter's own environment is in question (a device/collective lane:
`../hccl-and-hardware/index.md`, `../runtime-memory/index.md`, or a startup fault
whose owner is neither the command nor the doc,
`../startup-failures/index.md`) — then the task is a fault triage and the doc is
only context. A `[Doc]` report whose pasted traceback *is* the documented command
failing on argument arithmetic stays here: there is nothing wrong with the
reporter's box, the finding is the command block plus the invariant it omits, and
`../startup-failures/arg-validation-errors.md` (third raise site, #2973) supplies
the mechanism to cite. A documented serve block that dies at *request* time on its own
flag combination is the same shape one layer down: the v0.14.0rc1 GLM-4.x block
(`docs/source/tutorials/GLM4.x.md:81-102` — `--async-scheduling` + mtp `K=1` +
`cudagraph_capture_sizes [1,2,4,8,16,32]`) crashes in the MTP drafter's
`cos.view(num_tokens, -1)` at every concurrency except `1, 2, 4, 8, 16` and above 16, so
the finding is the capture list and the pin's own recipe
(`../spec-decode-mtp/rope-cos-sin-slice-token-mismatch.md` carries the division and the
corrected list). Grade the doc even when the paste looks like a lane-B
startup failure.

Inherit the rules in SKILL.md, plus these:

1. **The document is graded, so every claimed defect needs `path:line` at the
   pinned commit plus the parity grep that shows it is wrong.** A finding of the
   form "this flag no longer parses at vllm vX" is one `rg -n` in the pinned
   tree; a finding of the form "the number is stale" is one comparison of the
   doc's stated release against the task's image tag. Say which pinned release
   you checked, because a doc page and an image tag drift independently.
2. **Never invent a measured number.** You cannot run the reporter's cards.
   Produce the command set, the doc-derived baseline *labelled as such*, and an
   explicit slot plus the one command that fills it on the host. A report that
   presents a doc baseline as this environment's measurement is a fabricated
   result; a report with empty labelled slots is a usable template.
3. **Audit in a fixed, finite order and stop.** The checklist has twelve items
   (`tutorial-command-audit.md`); after 3-5 concrete defects plus the corrected
   deploy command and a three-item FAQ you have the deliverable. #11739's
   recorded run confirmed several real doc bugs and then kept auditing until the
   session ended, shipping nothing (`community-test-report.md`), and #11759
   repeated it a batch later with the defects in hand by turn 7, #11771 with the
   graded defect table still unwritten at the cap, #16954 spent the whole
   budget re-deriving an upstream `Literal` for a version-line finding it had
   already located (`docs-version-space-and-image-tag.md` §4), and #11753 landed
   the graded §1 (4.4 K characters) and then died inside one oversized §2-4
   heredoc — the cap cut the command mid-argument, so §2-4 *and* the message never
   existed. If the 任务书
   names a section of the page, resolve that section name against the **whole**
   docs tree before grading it missing — the two cases grade differently. The same
   resolution applies to the **page name itself**: a path that does not exist at the
   pin (`Qwen3.5-27B-Qwen3.6-27B.md` at v0.18.0, where the page is
   `Qwen3.5-27B.md`) is a finding to state in the first message, and the nearest
   page is then graded *as* the nearest page.
4. **Separate severity in the report**: breaks the command (dead flag, missing
   path, wrong device env var), stale baseline (older release, numbers
   predating the image), cosmetic (wording, a flag abbreviation that still
   parses). Grading a doc 2/5 needs at least one "breaks the command" item.
5. **Deployment-sizing claims belong to the sizing route.** Whether the stated
   card count can hold the model is arithmetic, not a doc-reading exercise:
   `../deploy-sizing/index.md` and `scripts/card-budget.sh`.
6. **The report file and the message are opened before the audit finishes, and the
   reasoning channel is not where the report is drafted.** One output budget (16 384
   tokens) holds the reasoning, every heredoc and the message, so a run that reasons
   each finding out at length and writes at the end ships nothing. Concretely: on the
   turn the first defect is pinned (turn 2-3) write `资料验证` §1 to
   `/workspace/scratch/<row>-report.md` — defect table with `path:line`, tier and
   corrected line, then the three 完整性/易用性/正确性 scores — and send the verdict plus
   the severe defects as **message text in that same turn**; after that keep 2-4 lines
   of reasoning per turn and one artefact per `bash` call. #11750 is the seventh run
   of this shape and the first to ship neither a file nor a message: 40.8 K characters
   of reasoning (~80 % of its budget) over 14 turns, then the report drafted inside a
   reasoning block and a single heredoc for the deploy script, cut mid-argument — with
   eight verified findings already in hand, now tabulated in
   `tutorial-command-audit.md` ("Sixth worked page") so the next run on that row can
   open §1 on turn 2. For scale, the same shape at its best: #11741 used 15 546 of
   16 384 tokens with the file opened early; #11751 spent 24 K characters of reasoning
   plus 16.1 K of tool arguments on a 15.4 K report and still lost the message.

Conditional routes:

- The task shape, the deliverable skeleton and the staged-emission protocol for
  a community-test report, the recurring 已知问题 "容器内无法使用 postman"
  (host-side `--net=host` postman, python `requests` as the guaranteed in-image
  client, `curl` absent from the Dockerfile), and the 已知问题/任务问题 field that
  names a **version gate or a stale tag** — compare its tag with the row's own
  蓝区镜像 (and with the page's `first supported in` line) before grading
  anything, and open the deploy script from `docker load` when the row ships a
  `.tar.gz` → `community-test-report.md`.
- The report's own failing item is a **sample request** — `vllm bench serve` or a
  curl body answering 404 (`The model X does not exist`) while the service is
  healthy — or its slow item is a first start that "loads for hours": the
  served-name contract (bench `--model` vs the page's `--served-model-name`, and
  the `--model`-omitted form that adopts the server's own name) plus the weight
  pre-download the page omits → `serve-contract-and-weight-predownload.md`
  (recorded #11738, and the standalone `curl /v1/models` form).
- The report asks for *added* guidance or a cross-link, or reviews the
  contributor/developer path (mount, `set_env.sh`, `format.sh ci`, image tag):
  existence + reachability + parity greps, the `.po`-is-not-the-document rule,
  and the verified #14423 / #17718 / #8151 defect tables →
  `missing-guidance-and-contributor-docs.md`. A **mirror/registry** report
  ("镜像地址只有 quay 的吗？华为云镜像仓库也没有") belongs there too: its registry half
  is answered from the pinned workflows (release tags quay-only, Huawei SWR carries
  only `nightly-*`), its China path is `faqs.md` §2, and the defect is the missing
  cross-link — not an architecture question, so do not open
  `../image-dependency-faults/image-arch-and-tag-availability.md` first.
  The same page's #5158 section owns the sibling shape — a tutorial sentence that
  promises an **install/compile step which exists nowhere** ("install extra
  operator", "am I expected to compile `X`?"): existence grep, sibling parity, then
  the same release's own `release_notes.md` (which usually contradicts the page),
  with the answer naming the image↔vllm pair and asking for the runtime error
  instead of ending at "skip it".
- The twelve-item audit checklist with the verified vllm v0.18.0 / vllm-ascend
  v0.18.0 and v0.19.1rc1 examples (dead `--swap-space` / `--disable-log-request`,
  `--random-input` vs `--random-input-len`, **which parses as an unambiguous
  argparse prefix and is tier cosmetic — check 2's six-line probe settles it once,
  it is not re-opened** (recorded #11753 spent six calls re-deriving this);
  image WORKDIR vs doc relative path,
  `limit_mm_per_prompt` and remote-media defaults, boilerplate TP sentence vs
  sibling docs, stale baseline release, CI config as the known-good command,
  `pip install <pkg>==<ver>` against the image's own requirements pins) plus the
  ready-made Qwen3-32B-W4A4, Qwen3.5-397B-A17B-§5.6, Qwen3-Omni-30B-A3B-Thinking
  (row `hjh1`: the `:177` "TP at least 1" sentence against its own two-card serve,
  the dead `--swap-space`/`--disable-log-request` on `:288`, the `:185` serve line
  that omits `VLLM_USE_MODELSCOPE` the page's own `:286` exports, the `:255`/`:296`
  `v0.13.0rc1` baseline, and the `--random-input` cosmetic), Qwen3.5-27B (community
  row 15: page-name absence, DP1/TP2-only serve line, the dropped
  `max_completion_tokens`, the nightly yaml as the in-tree gate; plus the same
  page graded on a 4-card row — which findings transfer, the `--device` comment
  that re-tiers the device-list defect, the TP2-vs-TP4 choice, the unmounted
  weights cache), InternVL3.5 (row `hjh4`: the w8a8 checkpoint the page never
  serves — no `--quantization ascend` on its `:123-127` serve block — the missing
  2-card container recipe, `require 1 Atlas 800I A2 (64G × 8) node` against its
  own TP2 serve, an empty CI list) and
  Qwen3-VL-30B-A3B (row 6: the `requires 1 Atlas … node` sentence against the
  page's own two-device map, and a sample body carrying a request-side
  `reasoning_content`) defect tables, check 12 (the printed sample body against
  the pinned protocol models), the
  `--hf_overrides` architecture string check added after #14817, and check 11
  (a tab-set's block against its own prose **and** its own `kv_connector` — the
  swapped-proxy defect of #15389; a missing `bash proxy.sh` is check 3) →
  `tutorial-command-audit.md`; the mechanism behind a wrong architecture prefix
  is `../pooling-embedding/seq-cls-arch-routing.md`.
- The question is **why one page shows the same command twice** ("文档中为什么创建两次容器",
  "why two `docker run`s", "do I need both"): separate boilerplate from the page's own
  example before grading anything. The `## Environment Preparation` → `### Installation`
  block is the shared image/install container (`docker pull` + a `--net=host`, no `-p`,
  `--name vllm-ascend-env` run), while the `## Deployment` → `### Run docker container`
  block is the page's serving container (`--privileged=true`, `-p <port>`, fewer
  `--device` lines matching its TP and serve port). One container is needed, not two;
  the reportable defects are that neither block says so, the install block's
  `In the [Run docker container] …` sentence points at the *other* container instead
  of `installation.md`, and the deployment block's
  `export IMAGE=…|vllm_ascend_version|` drops the `-a3` suffix its own Atlas A3 example
  needs (recorded #9988 at v0.18.0: `Qwen3-Dense.md:65-96` vs `:126-146`, note `:117`).
  Parity control: the A2/A3 tab pairs (`GLM5.md:53/96`, `Qwen2.5-7B.md:41/69`,
  `DeepSeek-V3.2.md:44/87`) are parallel options, not a second container — 9 of 33
  tutorial pages carry two `docker run` blocks.
- The report is a `[Doc]: Feedback for` `…/en/latest/tutorials/models/<Page>.html`
  and that page **does not exist at the snapshot pin** — `/en/latest` is the build of
  main while the pin is an older release (`docs-version-space-and-image-tag.md`). Say
  so in the first message, then treat the corpus as the only local witness of what the
  page says: `scripts/history-titles.sh -i '<the page slug>'` lists the earlier
  `[Doc]: Feedback for …/<Page>.html` issues on that same page, and their bodies carry
  both the page's own serve line and the reporter's fix. Worked case: the
  DeepSeek-V4.1-Flash thread #17236, whose fix now lives in
  `../feature-enable-recipes/engram-offload.md` row 1. Grade only the defect you can
  verify at the pin (an absent page, an incomplete serve line, a missing envelope row)
  — never a reconstruction of prose you cannot read.
- The defect is a **version disagreement** — the page prints an image tag the
  pasted command fails on, the report quotes a `…/en/latest/…` URL, or the doc
  names an older release than the flag it shows: `latest` is the build of main
  and each release lives at `/en/vX.Y.Z/`, the printed tag is a build-time
  substitution from `mkdocs.yml extra.*`, and the answer is the version gate
  plus the correction site — `docs-version-space-and-image-tag.md` (recorded
  #16954, main-only `--kv-cache-dtype int8` pair beside a stale `0.23.0` tag).
  When the substitution **matches** the tag that works and the page also prints a
  literal model-branded tag in prose, the substitution is not the defect and
  `extra.*` must not be touched: §6 of that page has the discriminating grep, the
  frozen-image precedent and the release-claim scope gate (recorded #18041).
  When the tag is the wrong **generation** rather than stale — a tutorial's
  `Atlas 800I A3` tab printing the bare A2 tag, reported as 「此处镜像提供错误」
  with no log at all — §7 has the one-call per-tab read
  (`scripts/recipe-tab-diff.py --page <page>`, identical `IMAGE=` lines in the A3
  and A2 tabs), the `installation.md` image table as the suffix authority, and the
  registry split (pullable `-a3` ⇒ text fix; `manifest unknown` ⇒ release gap).
  Recorded #11238; the dropped suffix in #9988's page (the two-container row above) is
  that §7 defect, not a second-container one.
- The doc's log/flag claims override the doc rather than the code:
  `../observability-and-logs/request-output-logging-flags.md`.
- The doc's accuracy or performance numbers look anomalous: check the precedent
  family before calling it a doc error → `../accuracy-benchmarks/index.md`.
