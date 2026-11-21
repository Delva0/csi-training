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
  pages already carry it" (`missing-guidance-and-contributor-docs.md`);
- any "verify this recipe" request where the reporter's hardware is not
  available to you and the evidence is the **pinned tree**, not a run.

Leave this topic when the report contains an actual failure log **and** the
reporter's own environment is in question (a device/collective lane:
`../hccl-and-hardware/index.md`, `../runtime-memory/index.md`, or a startup fault
whose owner is neither the command nor the doc,
`../startup-failures/index.md`) — then the task is a fault triage and the doc is
only context. A `[Doc]` report whose pasted traceback *is* the documented command
failing on argument arithmetic stays here: there is nothing wrong with the
reporter's box, the finding is the command block plus the invariant it omits, and
`../startup-failures/arg-validation-errors.md` (third raise site, #2973) supplies
the mechanism to cite. Grade the doc even when the paste looks like a lane-B
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

Conditional routes:

- The task shape, the deliverable skeleton and the staged-emission protocol for
  a community-test report, and the recurring 已知问题 "容器内无法使用 postman"
  (host-side `--net=host` postman, python `requests` as the guaranteed in-image
  client, `curl` absent from the Dockerfile) → `community-test-report.md`.
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
  and the verified #14423 / #17718 defect tables →
  `missing-guidance-and-contributor-docs.md`.
- The twelve-item audit checklist with the verified vllm v0.18.0 / vllm-ascend
  v0.18.0 and v0.19.1rc1 examples (dead `--swap-space` / `--disable-log-request`,
  `--random-input` vs `--random-input-len`, **which parses as an unambiguous
  argparse prefix and is tier cosmetic — check 2's six-line probe settles it once,
  it is not re-opened** (recorded #11753 spent six calls re-deriving this);
  image WORKDIR vs doc relative path,
  `limit_mm_per_prompt` and remote-media defaults, boilerplate TP sentence vs
  sibling docs, stale baseline release, CI config as the known-good command,
  `pip install <pkg>==<ver>` against the image's own requirements pins) plus the
  ready-made Qwen3-32B-W4A4, Qwen3.5-397B-A17B-§5.6, Qwen3.5-27B (community
  row 15: page-name absence, DP1/TP2-only serve line, the dropped
  `max_completion_tokens`, the nightly yaml as the in-tree gate; plus the same
  page graded on a 4-card row — which findings transfer, the `--device` comment
  that re-tiers the device-list defect, the TP2-vs-TP4 choice, the unmounted
  weights cache) and
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
- The doc's log/flag claims override the doc rather than the code:
  `../observability-and-logs/request-output-logging-flags.md`.
- The doc's accuracy or performance numbers look anomalous: check the precedent
  family before calling it a doc error → `../accuracy-benchmarks/index.md`.
