# The community-test report: shape, staged emission, and what went wrong in three runs

## The task shape (recorded #11739, vllm-ascend v0.18.0 image `v0.18.0`)

The issue body is a 任务书 row, not a bug report. Fields that matter and what
each one forces:

| Field | What it settles |
|---|---|
| 蓝区镜像 `quay.io/ascend/vllm-ascend:v0.18.0` | the pinned vllm/vllm-ascend pair to audit against — check it with `scripts/sources-summary.sh` before citing any code |
| 权重链接 (ModelScope / HF) | the checkpoint identity; the repo snapshot rarely carries the HF config, so context length stays a "verify with one command" item |
| 配套信息 `Ascend HDK 25.5.0.B078`, 卡数 `2`, 内存 `1T` | the hardware envelope the report may assume, and the card count for the sizing arithmetic |
| 核心要求 / 测试方法 | the doc page under test is named there — this is the artefact you actually grade |
| 交付件 `测试报告` + required sections | the report's table of contents: 使用权重, 部署脚本, 测试配置 (curl + `vllm bench`), 测试结果, 资料验证 (完整性/易用性/正确性 各 5 分), 问题记录/FAQ |

The graded artefact is the **document**: does every command it prints still
parse and resolve at the pinned versions, does the stated card count match the
model's memory footprint, is the reference number from the image the reporter
was told to use, and is anything a reporter needs silently missing.

## What the recorded run did, and the cost

#11739: 22 assistant turns, 40 tool calls, **no report file and message text that
never got past an "I'll start" line**. The findings were real and arrived early:

- **5th assistant turn** (reasoning only): the doc's reference results are
  labelled `vllm-ascend:0.13.0rc1` while the task image is v0.18.0 — a
  stale-baseline finding, complete as it stands;
- **11th**: the perf server line (`:288` at v0.18.0, `:294` at v0.19.1rc1)
  passes `--swap-space 16` and `--disable-log-request`, neither of which exists
  at vllm v0.18.0 → the documented benchmark server cannot start;
- **15th**: the image's `WORKDIR` is `/workspace` and the source is copied to
  `/vllm-workspace/vllm-ascend/` (`Dockerfile:30,32`), so the doc's relative
  `pip install -r vllm-ascend/benchmarks/requirements-bench.txt` does not
  resolve from the container's default cwd;
- turns 16-22 re-verified defaults that were already innocuous
  (`limit_mm_per_prompt` 999, remote media URLs allowed) and read six history
  bodies for FAQ colour — none of it reached a file.

That is a report lost to sequencing, not to missing evidence: two
command-breaking defects and one stale baseline were in hand before the halfway
point of the session.

## The 任务书 names a section: find it in the tree before grading it missing

A community-test row's 测试方法 step usually reads "参考资料社区文档 <tutorial URL>
的 <section> 章节". Two different findings hide there, and they grade differently:

- the section is on **another page of the same docs tree** — recorded #11759: the
  Qwen3-32B-W4A4 row points at the model tutorial for "Run basic benchmark", that
  page (143 lines) has no performance section, but `## 3. Run basic benchmarks`
  (`:47`) and `### 3.2 Run basic benchmark` (`:93`) live in
  `docs/source/developer_guide/performance_and_debug/performance_benchmark.md`,
  which also carries the `pip install -r benchmarks/requirements-bench.txt`
  prerequisite (`:43`). Grade the tutorial page **incomplete relative to its
  siblings** (`Qwen3-Dense.md:265-311` has Accuracy and Performance sections) and
  hand the tester the real page plus its command — never "the section does not
  exist".
- the section is nowhere in the tree: then the task book is unexecutable as
  written and *that* is the headline defect.

One call decides it: `rg -n '<section title>' docs/` before any grading.

## Staged emission protocol

The report is a file, so write it as the sections become true, and keep each
write small. Suggested split (three writes, ~2-2.5 K characters each):

1. **Turn 2-3, one message + one file write: the graded section first.** 资料验证 —
   which page (and whether the task book's own path resolves at the pin), the defect
   table with `path:line`, tier and corrected line, then the three
   完整性/易用性/正确性 scores with the defects that justify each. This is the 交付件
   the reviewer scores, and it is complete as soon as one page read and one flag grep
   are in hand. #11771 wrote 6.3 K characters of deploy/curl/accuracy sections and
   died before this table, so the graded part never existed anywhere (see below).
2. **After the flag/path/defaults audit (checklist items 1-4): the deploy script.**
   Container run line, env block, the corrected `vllm serve` line, plus the audit
   verdict on each flag in it. Bulky and the most derivable section — write it second.
3. **After one `scripts/history-titles.sh '<Model>'` pass (item 7).** 测试报告
   剩余部分: curl + `vllm bench` commands, the doc baseline labelled as a
   baseline, the empty result slots with the exact command that fills each, and
   the FAQ from 2-3 precedents whose titles name a mechanism.

**The file body and the final message draw on one budget, and a heredoc's
characters are output tokens** — they travel as tool-call arguments, charged to
the same 16 384-token cap as the reasoning and the answer. Measured on #11751
(third run of this shape, 34 messages, cap reached): 24.0 K characters of
reasoning (55 %), **16.1 K characters of tool-call arguments — the three writes
of a 15.4 K-character report (36 %)** and 3.8 K of message text (9 %). Nothing
failed loudly: the file was complete and the *message* was cut inside row 5 of
its 6-row defect table. So budget it explicitly:

- file body **≤ ~8 K characters** in total, 2-2.5 K per write. A 15 K-character
  report is a "how to write it" document, not a "what I found" one — the defect
  set is 5-6 rows;
- a few lines of reasoning per turn once a finding is already in the file; do
  not re-audit it there;
- the last ~2.5 K characters of the session are the message, written in cut
  order: (1) verdict + report path + the three 完整性/易用性/正确性 scores, one
  line each (they are 交付件 items, not appendix material); (2) one line per
  defect, `path:line` → fix, most severe first; (3) what the tester must supply
  or measure. **Do not re-paste the docker block or the tables** — the file
  holds them and the message names it; repeat only the one corrected
  `vllm serve` line.

The message's first block — verdict, report path, the scores — belongs on the
turn the first defect is pinned, not on a turn the audit "finishes"; it never
does.

## #11759, the second run of this exact shape

v0.18.0 image, `yxr2_qwen3-32B` (Qwen3-32B-W4A4, 1×Atlas 800I A2, HDK
25.5.0.B078). Sixteen assistant turns, **69 characters of message text, no report
file**: the deploy script was written on turn 15 and the report heredoc on turn 16,
and the output cap ate the write, so the framework never executed it. Every section of
the answer existed by turn 14: the absent benchmark section (turn 6), the curl's
ignored `max_completion_tokens` (turn 8), the one-card sizing pass (turn 11) and
the `transformers==4.51.0` install against the image pins (turn 14). The page's verified defect set is tabulated in
`tutorial-command-audit.md` ("Second worked page"), so a future run on this row can
write section 1 at turn 2 and skip the audit that killed this one.

## #11751, the third run: the file fit, the message did not

Same shape (v0.18.0 image, `hjh2-Qwen3-VL-8B-Instruct`, 1×Atlas 800I A2, HDK
25.5.0.B078, page `Qwen-VL-Dense.md`). It fixed what killed #11739/#11759: the
report file was opened at the 10th assistant turn with a 4.3 K-character
skeleton plus two appends (15.4 K characters by the end), and every finding was
pinned to the tree. It still lost the delivery to the same cap from the other
side — the final message was a *second copy* of the report (the docker block
plus the whole defect table) and the cut landed inside that table. Lost: the
last defect row, the three sub-scores and the FAQ. Kept: verdict, the deploy
script, five of six defect rows (the 32B tab's `SamplingParams(max_completion_tokens=512)`
→ `TypeError` at vllm v0.18.0, the `v0.11.0rc3-a3` stale baseline, the missing
performance result, the doc's 0.5389 reference below the project's own 0.55 e2e
gate, and `--random-input` for `--random-input-len`). The audit was not the
problem; the write and message sizes above were.

## #11771, the fourth run: the graded table was never written

Same shape, row 15 of the same 任务书 (image `quay.io/ascend/vllm-ascend:v0.18.0`,
8×Atlas 800I A2, weights `Eco-Tech/Qwen3.5-27B-w8a8-mtp`, page `Qwen3.5-27B.md`; its
verified defect set is tabulated in `tutorial-command-audit.md`, "Fourth worked
page", so a future run on this row can write section 1 on turn 2). Seventeen
assistant turns, **409 characters of message text in total** (41 on turn 1, 368 on
turn 8), the 16 384-token cap consumed as ~24 K characters of reasoning plus 6.3 K of
heredoc: the file got §1 deploy script (3.3 K) and §2-3 curl + AISBench (3.0 K), then
the cap landed on a block that was about to write §4 — the defect table. Missing from
both the file and the transcript: the 资料验证 table, the three scores and the FAQ,
i.e. three 交付件 items, and any concluding message.

Two mechanisms, both already stated above and both repeated here: every section was
composed **twice** (an 8.0 K-character block drafted §1-2 and was then re-typed into
the heredoc; a 2.8 K block did the same for §2-3), and the write order put the graded
section last. Convert the draft block into the heredoc in the turn that produces it,
and let the table go first.

## What the report must not do

- Present doc numbers as measurements of this environment, or fill a slot with
  a plausible value. Leave the slot empty and name the command.
- Assert a defect without the pinned-tree grep behind it ("this flag was
  removed" needs the `rg -n` at *that* version; see `tutorial-command-audit.md`).
- Grade 完整性/易用性/正确性 without citing the defects behind the score.
- Blame the reporter's environment for a doc line that cannot parse.
