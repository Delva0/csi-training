# Tutorial / documentation verification: the report is the deliverable

Scope: the task is to *judge a document*, not to explain a crash. Entry shapes:

- a community-test row (`[Community Test]`, 众测任务书: 蓝区镜像 / 权重链接 /
  配套 HDK / 卡数 / 交付件 = 测试报告, "资料验证 + 部署测试"), where the answer is
  a report covering deploy script, curl and `vllm bench` commands, the doc's
  完整性/易用性/正确性 scores and a FAQ;
- a `[Doc]` / `[Usage]` issue reporting that a tutorial step, command or number
  does not work as written;
- any "verify this recipe" request where the reporter's hardware is not
  available to you and the evidence is the **pinned tree**, not a run.

Leave this topic when the report contains an actual failure log with a lane
(`../startup-failures/index.md`, `../hccl-and-hardware/index.md`,
`../runtime-memory/index.md`) — then the task is a fault triage and the doc is
only context.

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
3. **Audit in a fixed, finite order and stop.** The checklist has ten items
   (`tutorial-command-audit.md`); after 3-5 concrete defects plus the corrected
   deploy command and a three-item FAQ you have the deliverable. #11739's
   recorded run confirmed several real doc bugs and then kept auditing until the
   session ended, shipping nothing (`community-test-report.md`), and #11759
   repeated it a batch later with the defects in hand by turn 7. If the 任务书
   names a section of the page, resolve that section name against the **whole**
   docs tree before grading it missing — the two cases grade differently.
4. **Separate severity in the report**: breaks the command (dead flag, missing
   path, wrong device env var), stale baseline (older release, numbers
   predating the image), cosmetic (wording, a flag abbreviation that still
   parses). Grading a doc 2/5 needs at least one "breaks the command" item.
5. **Deployment-sizing claims belong to the sizing route.** Whether the stated
   card count can hold the model is arithmetic, not a doc-reading exercise:
   `../deploy-sizing/index.md` and `scripts/card-budget.sh`.

Conditional routes:

- The task shape, the deliverable skeleton and the staged-emission protocol for
  a community-test report → `community-test-report.md`.
- The ten-item audit checklist with the verified vllm v0.18.0 / vllm-ascend
  v0.18.0 and v0.19.1rc1 examples (dead `--swap-space` / `--disable-log-request`,
  `--random-input` vs `--random-input-len`, image WORKDIR vs doc relative path,
  `limit_mm_per_prompt` and remote-media defaults, boilerplate TP sentence vs
  sibling docs, stale baseline release, CI config as the known-good command,
  `pip install <pkg>==<ver>` against the image's own requirements pins) plus the
  ready-made Qwen3-32B-W4A4 defect table, and the `--hf_overrides` architecture
  string check added after #14817 → `tutorial-command-audit.md`; the mechanism
  behind a wrong architecture prefix is `../pooling-embedding/seq-cls-arch-routing.md`.
- The doc's log/flag claims override the doc rather than the code:
  `../observability-and-logs/request-output-logging-flags.md`.
- The doc's accuracy or performance numbers look anomalous: check the precedent
  family before calling it a doc error → `../accuracy-benchmarks/index.md`.
