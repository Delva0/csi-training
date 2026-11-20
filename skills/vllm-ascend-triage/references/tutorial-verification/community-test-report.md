# The community-test report: shape, staged emission, and what went wrong twice

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
write small. Suggested split (three writes, ~2-4K characters each):

1. **Turn 2-3, one message + one file write.** Which doc page, which image, and
   the corrected deploy script: container run line, env block, `vllm serve`
   line, plus the audit verdict on each flag in it. This is the section a human
   reads first, and it is already a deliverable on its own.
2. **After the flag/path/defaults audit (checklist items 1-4).** 资料验证: the
   defect table with `path:line`, pinned commit, severity, corrected line; then
   the three sub-scores with the defects that justify each one.
3. **After one `scripts/history-titles.sh '<Model>'` pass (item 7).** 测试报告
   剩余部分: curl + `vllm bench` commands, the doc baseline labelled as a
   baseline, the empty result slots with the exact command that fills each, and
   the FAQ from 2-3 precedents whose titles name a mechanism.

Each write is an **append of 2-3 K characters**; a full report body is 5-6 K and
does not fit in one heredoc once the session has spent its reasoning
(`delivery-discipline.md`).

Then the final message is the condensed report: verdict on the doc (works /
works with the listed corrections), the corrected deploy line, the 3-5 defects
with severity, the report path, and the items that need the reporter's hardware.
Do not hold that message until the audit "finishes" — it never does.

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

## What the report must not do

- Present doc numbers as measurements of this environment, or fill a slot with
  a plausible value. Leave the slot empty and name the command.
- Assert a defect without the pinned-tree grep behind it ("this flag was
  removed" needs the `rg -n` at *that* version; see `tutorial-command-audit.md`).
- Grade 完整性/易用性/正确性 without citing the defects behind the score.
- Blame the reporter's environment for a doc line that cannot parse.
