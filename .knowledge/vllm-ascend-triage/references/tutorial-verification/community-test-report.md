# The community-test report: shape, staged emission, and what went wrong in three runs

## The task shape (recorded #11739, vllm-ascend v0.18.0 image `v0.18.0`)

The issue body is a 任务书 row, not a bug report. Fields that matter and what
each one forces:

| Field | What it settles |
|---|---|
| 蓝区镜像 `quay.io/ascend/vllm-ascend:v0.18.0` | the pinned vllm/vllm-ascend pair to audit against — check it with `scripts/sources-summary.sh` before citing any code |
| 权重链接 (ModelScope / HF) | the checkpoint identity; the repo snapshot rarely carries the HF config, so context length stays a "verify with one command" item |
| 配套信息 `Ascend HDK 25.5.0.B078`, 卡数 `2`, 内存 `1T` | the hardware envelope the report may assume, and the card count for two decisions: the sizing arithmetic, and **which of the page's routes is executable at all** — a conversion/quantisation step that asks for more cards than the row grants is dead, so the pre-quantised-weight route is the one to document, and it then needs a download command *and* a `docker run -v` mount of that directory (`tutorial-command-audit.md` "Second worked page", the 1-卡 row from #11769) |
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

## The 已知问题 field: "容器内无法使用 postman 工具发送推理请求"

Both Qwen3.5-27B rows carry it verbatim (`任务问题和需求` #11770,
`已知问题和补充说明` #11760). It is a **document gap, not an image fault** — so it
becomes the FAQ's first item and at most costs 易用性; do not grade the image for
it, and do not treat it as the task's root cause.

- postman is a desktop GUI client; a headless serving container can neither run
  nor install it. State that, then give the path that works.
- **host-side postman, no container change**: the page's `docker run` already
  passes `--net=host` (`Qwen3.5-27B.md:52`), so port 8000 sits on the node's
  network interface and postman on the tester's PC/宿主机 reaches
  `http://<node-ip>:8000`. This is the only reading under which the 任务书's
  "postman or requests" is executable as written.
- **in-container client**: `requests` (`vllm/requirements/common.txt:6`,
  `requests >= 2.26.0`) and `openai` (`:15`) are hard dependencies of vllm, so a
  `python -c` client is always importable — the guaranteed substitute.
- **`curl` is not guaranteed**: the page's only request example is one `curl`
  (`:148`), while vllm-ascend's `Dockerfile:36` installs `git vim wget net-tools
  gcc g++ cmake numactl libnuma-dev libjemalloc2` and nothing else, and no
  `curl` appears anywhere in that Dockerfile. The base (`Dockerfile:18`, a
  `quay.io/ascend/cann:8.5.1-…-ubuntu22.04-py3.11` image) may still carry it, so
  phrase it as an open item with the one-line check (`command -v curl` in the
  container) — never "the image lacks curl".
- **parity for grading the page**: five pages in the same tree do print a python
  client (`MiniMax-M2.md`, `PaddleOCR-VL.md`, `Qwen3_reranker.md`,
  `Qwen3-VL-Reranker.md`, `user_guide/feature_guide/batch_invariance.md`), so
  "this page has no in-container python example" is a completeness defect with an
  in-tree sibling behind it.

### When the 已知问题 field is a version gate, not a doc gap

Two rows carry the sentence 「官方文档没有 vllm-ascend:v0.18.0 部署指导」, and
comparing it with the row's **own** 蓝区镜像 decides whether anything is gradeable:

- `#11753`'s 蓝区镜像 *is* `quay.io/ascend/vllm-ascend:v0.18.0`, and the page's
  Introduction answers it: "The `InternVL3.5` model is first supported in
  `vllm-ascend:v0.20.2`" (`InternVL3.5.md:7` at v0.22.1rc1). An image older than
  the page's first-supported release cannot deploy the model, and no missing
  section would fix that — so this is **not** a document defect and must not be
  graded as one (完整性 unchanged). The headline is the task book's 蓝区镜像 being
  below the gate: state the minimum tag, audit the page at the pin, and ask the
  task book to be corrected.
- `#11742` — the **same row `hjh4`** — repeats the sentence verbatim while its
  蓝区镜像 is
  `vllm-ascend:dev-26.1.0.B050-cann9.0.0-800I-A2-py311-Ubuntu24.04-lts-aarch64`,
  an internal dev build *above* that gate and neither the tag the field names nor
  the tag the page prints. Nothing here is a doc defect either: the page's gate
  plus the build-time substitution answer the question, and carrying the "蓝区镜像
  below the gate" headline onto this row would be a scoring error. Read the two
  tags side by side before choosing a headline — the field is routinely stale
  relative to the row it sits in.

One grep decides the page side in every variant —
`rg -n 'first supported in' docs/source/tutorials/models/<page>.md` — and the
version-space page supplies the drift rules when the page's own tag is the defect
(`docs-version-space-and-image-tag.md` §5).

**The delivered tag is a version the snapshot cannot pin, so the audit answers for
the page's substituted tag, not the tester's build.** At v0.22.1rc1 the
substitution is `docs/source/conf.py:81` `vllm_ascend_version: v0.22.1rc1`
(`cann_image_tag` `9.0.0-910b-ubuntu22.04-py3.12` at `:87`; `mkdocs.yml extra.*`
on later trees — `docs-version-space-and-image-tag.md` §2). Write "checked at
`<pin>`" beside every dead flag, and hand the tester the one command that closes
the gap on their own image — `vllm --version` and `pip show vllm-ascend` inside
it — instead of declaring a flag broken on a build nobody verified.

**A `.tar.gz` 蓝区 image is a deploy-script defect of its own.** When the 任务书
reads 「把 vllm-ascend:…-aarch64.tar.gz 传到蓝区」, the tester's image arrives as a
local tarball (`docker load -i <file>.tar.gz`), while the page offers only
`export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|` followed by
`docker run … $IMAGE bash` (`InternVL3.5.md:41`, `:69`) — an implicit pull of the
public `quay.io` tag, which is not the route the row hands them. Grade it
完整性, and open the deploy script with the `docker load` line plus the tag it
loads.

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

- file body **≤ ~8 K characters** in total, 2-2.5 K per write, **plus the script**.
  A 15 K-character report is a "how to write it" document, not a "what I found" one —
  the defect set is 5-6 rows. It is a target for the *message*, not a cap, and the pair
  to measure is (report + deploy script): the eighth run (#11759) wrote 9 675 and still
  shipped its closing message at 24 K characters of reasoning, while the ninth (#11769,
  same row, this protocol followed to the letter — §1 in one 2.9 K write, script + §2
  next at 3.1 K, one 2.8 K append for §3-6, both files landing) spent 27.8 K characters of
  reasoning (61 %), 8.8 K of heredoc arguments (13.7 K for the whole tool-call channel,
  30 %) and 4.0 K of message text (9 %) — 45 417 characters, exactly the 16 384-token cap
  at 2.77 characters per token — and had its closing message cut inside defect row 4 of
  6. So: **when the row's verified rows are already tabulated** (`tutorial-command-audit.md`
  "Second worked page" for this row), write §1 as the three scores + **only** the rows that
  changed since the recorded pin + a pointer to the table; new rows still get written, the
  re-typed seven cost ~2 K characters of the message's tail. Cap the message's own defect
  list at 4 rows — the file holds the rest — and keep the reasoning trail under ~25 K
  characters if the closing message is to survive;
- **size each write against what is *left*, not against what the section needs.**
  #11753 wrote the graded §1 as one 4.4 K block early (it landed) and then put
  §2 deploy + §3 curl + §4 bench into a single append: the cap cut that command
  mid-argument, the tool never executed, and the run shipped no closing message.
  A section that does not fit is a section to shorten, not to finish;
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

## #11759 again, the eighth run: the table unread, the delivery made

Same row and image as the second run, in a fresh session that had this page but read
only `:1-80` of its 352 lines (`tutorial-verification/index.md` only to `:120`, so the
route to `tutorial-command-audit.md`'s worked table further down that page was never
seen either).
Fifteen assistant turns, `completed`, 16 165 of 16 384 completion tokens: 24.1 K
characters of reasoning, 14.4 K of tool-call arguments (three heredocs → a 9 675-byte
report file, above the ≤8 K target) and **5.3 K of message text, including a complete
3 249-character closing message** — verdict, the three 完整性/易用性/正确性 scores,
three defects with `path:line`, the corrected one-card deploy script, the corrected curl
(`max_tokens`, not `max_completion_tokens`) and the bench line whose `--model` is the
served name. So the emission protocol held from the other side: it is reachable from a
late start, and what this run paid was audit, not delivery.

Avoidable, each with one call: three rows it re-derived are in the table (absent
benchmark section, ignored `max_completion_tokens`, the two-card conversion block), and
two more it found are worth keeping (the missing weight-download command,
`--max-model-len 4096` as cosmetic); the `conf.py` substitution lag (`v0.17.0rc1` at
the v0.18.0 tag) is `docs-version-space-and-image-tag.md:62-70`; and the row it
dropped —
`pip install transformers==4.51.0` (page `:51`) against `transformers >= 4.56.0, < 5`
(`vllm/requirements/common.txt:10`) and `>=4.57.4`
(`vllm-ascend/requirements.txt:42`), a downgrade below both pins of the image the deploy
script is installing into — was in the table too, and was set aside on reasoning
("a separate env") without the two greps.

Nothing in it is new knowledge either: the two items it presented as its own findings —
the page's `IMAGE=…|vllm_ascend_version|` rendering the previous release at this tag, and
the conversion step's `ASCEND_RT_VISIBLE_DEVICES=0,1` on a one-card row — are already on
the version-space page and in the table (conversion-only break).

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
performance result, the doc's 0.5389 reference against the in-tree 0.55 e2e gate,
and `--random-input` for `--random-input-len`). The audit was not the
problem; the write and message sizes above were.

One correction and one shortcut for a later run, both re-checked at the pin. The 0.5389 row is
**not** "below the gate": a config's `metrics.value` is an
`np.isclose(ground_truth, measured, rtol=0.05)` target
(`tests/e2e/models/test_lm_eval_correctness.py:10,138-140`), so 0.55 accepts
0.5225-0.5775 and 0.5389 sits inside it — grade it stale baseline and quote the band
(`accuracy-benchmarks/harness-vs-engine-attribution.md:13`). And the row's **full**
verified defect set at v0.18.0, with `path:line`, is now the eighth worked page in
`tutorial-command-audit.md`, so a future run on row `hjh2` writes 资料验证 §1 from that
table instead of auditing the page again (#11740 did, and reproduced this list).

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

## #11741, the fifth run: the protocol held, and the verified defect set

Same shape and same image (row 6, `hjh3-Qwen3-VL-30B-A3B-Instruct`, 2×Atlas 800I
A2, HDK 25.5.0.B078, page `docs/source/tutorials/models/Qwen3-VL-30B-A3B-Instruct.md`,
207 lines). This is the run that followed the protocol: one page read, then 19
shell calls in 14 assistant turns (headings, the e2e yaml, the flag greps, the
model's identity in the chat protocol, history titles), the report file in two
appends (3.9 K then 5.3 K characters), and a final message carrying verdict,
path, three scores, five defects and the slots the tester must fill — 15 546 of
16 384 completion tokens, `completed`. Two of its
five findings are new species, now generalised as checklist items 5 and 12:
the `requires 1 Atlas 800I A2 (64G × 8) node` sentence (`:18`) contradicted by
the page's own two-device map (`:42-43`), its `--tensor-parallel-size 2` (`:87`)
and the in-tree gate `tests/e2e/models/configs/Qwen3-VL-30B-A3B-Instruct.yaml`
(A2, TP2, EP, `max_model_len: 128000`, `gpu_memory_utilization: 0.7`,
mmmu_val `0.58`), and the sample body's `"reasoning_content": null` (`:130`,
`:192`), which vllm v0.18.0's `ChatMessage` cannot emit.

| Tier | Finding | Evidence |
|---|---|---|
| wrong requirement | the node sentence (above) — 2 cards is the validated recipe | page `:18` vs `:42-43`, `:87`; e2e yaml `:2,10` |
| completeness | no Accuracy / Performance section and no CANN/HDK requirement, while the deliverable demands aisbench config + 性能截图 and the sibling page has both (`Qwen-VL-Dense.md:468`, `:543`) | page headings end at `:205 ### Offline Inference`; bench route `developer_guide/performance_and_debug/performance_benchmark.md` §3.2.4 `:168` |
| cosmetic (sample) | `reasoning_content` in the printed body is a request-side key at this pin | `vllm/entrypoints/openai/chat_completion/protocol.py:54-64`; `entrypoints/chat_utils.py:329` |
| cosmetic (link) | the Offline Inference section links an absolute `…/en/latest/…` URL (`:207`) instead of the relative page; the anchor itself resolves (`Qwen-VL-Dense.md:117`) | page `:207` |
| cross-page | a tester who goes on to benchmark hits `performance_benchmark.md:42` `cd /workspace/vllm-ascend`, but the image's `WORKDIR` is `/workspace` and the source is at `/vllm-workspace/vllm-ascend` | `Dockerfile:30,32` |
| pass (say so) | `max_completion_tokens` on the **chat** endpoint is honoured (`protocol.py:164`, `entrypoints/openai/engine/serving.py:758-761`), `--limit-mm-per-prompt.video 0`, `--enable-expert-parallel` and `--allowed-local-media-path` all resolve; only `modelscope` is installed, so no dependency-pin conflict | page `:87-90`, `:123` |

Scores it gave: 完整性 3/5 · 易用性 4/5 · 正确性 3/5, each tied to a row above.
The reusable part is the ordering, not the count: page + headings, then the
device map and the e2e yaml (which pin the card count without arithmetic), then
the flag greps, then history titles for the FAQ — and the file opened before the
fifth defect was checked.

## #11753, the sixth run: the graded table landed, the rest did not

Row `hjh4-InternVL3.5-38B` (2×Atlas 800I A2, HDK 25.5.0.B078, weights
`Eco-Tech/InternVL3_5-38B-w8a8`, page `docs/source/tutorials/models/InternVL3.5.md`,
288 lines; pinned vllm-ascend v0.22.1rc1 `5f6faa0c` + vllm v0.22.1 `0decac0d`).
It followed the front of the protocol — skill alone, one-level orient plus
`sources-summary.sh`, the page outline, then the page — and put a 4.4 K-character
资料验证 §1 into `/workspace/scratch/internvl35-report.md`; its verified defect set
is tabulated in `tutorial-command-audit.md`, "Fifth worked page", so a future run
on this row writes §1 on turn 2. Then the cap (16 384 completion tokens,
1 024 044 cumulative prompt tokens, `output_truncated`) cut the single §2-4 append
mid-argument: no deploy script, no curl/bench, no FAQ, and 443 characters of
message text in total.

Two bought-and-paid-for lessons:

- **A settled cosmetic is not re-opened.** Six of its 24 tool calls carried the
  `--random-input 200` question (`:285`): grepping `--random-input-len`'s definition, then
  re-reading `FlexibleArgumentParser.__init__`, after a three-line argparse probe
  had already returned `Namespace(random_input_len=200)`. The tier was recorded in
  `tutorial-command-audit.md` check 2 and reachable from `index.md`; the second
  pass cost exactly the margin §2 needed.
- **The partial "Verdict forming" block was the right instinct, in the wrong
  place.** Its 10th assistant turn emitted a 443-character content block naming the version gate
  and the TP2-vs-8-card contradiction while still verifying. That block *is* the
  message's first section — ship it as message text on the turn it becomes true,
  then keep auditing for the file.

Its new reusable species, beyond the version gate above: the page's own
"Supported Features" link resolves to a matrix that has **no InternVL3.5 row** and
marks the family row ❌ (`supported_models.md:116`), while the tutorial exists and
the 8B sibling is CI-gated — check 8's failing form, one `rg -n -i '<family>'`.

## #11750, the seventh run: neither the file nor the message

Row `hjh1-Qwen3-Omni-30B-A3B-Thinking` (v0.18.0 image, 2×Atlas 800I A2, 1 T RAM, HDK
25.5.0.B078, page `docs/source/tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md`,
324 lines). Fourteen assistant turns, `output_truncated`, **no report file and no
closing message** — the first run of this shape to deliver neither. Its verified
defect set is tabulated in `tutorial-command-audit.md` ("Sixth worked page"), so a
future run on this row writes §1 on turn 2.

What the budget went to: **40.8 K characters of reasoning (~80 % of the 16 384-token
cap)**, 8.2 K of tool-call arguments, 3.3 K of message text. The audit itself was
sound — the defect set was complete by turn 13 of 14 — but the
end sequence was the failure: the full report was composed *inside a reasoning
block*, and the next turn carried a registry re-grep plus one `cat > … <<'EOF'` for
the deploy script. The cap cut that heredoc mid-argument, the framework did not run
it, and the session ended with an "I'll write the deliverables now" line.

A later replay of this same row (fresh session, the worked-page table and this
section already in the library) **completed**: 18 assistant turns, a
10.7 K-character report file with the deploy script, the curl and bench lines,
the result slots and the FAQ, and a complete closing message. What it still did
wrong is the measurable part: it re-derived the defect set instead of reading
`tutorial-command-audit.md`'s "Sixth worked page" table at turn 2, and it
re-opened the one settled cosmetic — `--random-input 200` was written into its
message as 基准无法运行 instead of check 2's cosmetic tier, which is exactly the
severity slip #12562 paid for. Read the table, then spend the turns on what it
does not have.

Two lessons, both already stated above and now gated in `index.md` rule 6:

- **Reasoning is the largest single charge and it is charged twice when you draft
  there.** The report text drafted in a reasoning block is paid for again when the
  heredoc repeats it; #11771 was the first recorded instance and this run repeats it
  at roughly double the size. Convert a draft into the file in the turn that produces
  it, and keep 2-4 lines per turn once a finding is in the file.
- **This page was read to line 110 of 279** — `sed -n '1,110p'` stopped on the
  `## Staged emission protocol` heading, so the budget arithmetic (§ above) and the
  "last ~2.5 K is the message" rule were never seen; the file write was attempted
  once, on the last turn, and never executed. When a page is the protocol you are about to follow, read it
  whole; its heading is not its rule.

## #11737, the Qwen3-Coder-30B-A3B row: both files landed, the tail did not

Row `cjw1` (4×Atlas 800I A2, image `v0.18.0`, page `Qwen3-Coder-30B-A3B.md`, 105 lines,
任务问题和需求 「测试通过」). Thirteen assistant turns, `output_truncated`. It is the
first run of this shape to put the graded §1 **and** the deploy script into files
before the cap — 3.4 K characters of 资料验证 (defect table + the three scores), then
2.4 K of `deploy_*.sh` that it syntax-checked with `bash -n` — so "graded section
first" now has a recorded success, and its verified defect set is
`tutorial-command-audit.md`'s *Seventh worked page*: a next run on this row writes §1
from that table on turn 2.

What it lost is #11753's tail: §2-§5 and the closing message never existed. Budget:
**33.7 K characters of reasoning (~70 %)**, 10.2 K of tool-call arguments (the two
heredocs), 4.4 K of message text. Two of its thirteen turns were a flag re-lookup —
`rg -n 'add_argument("--model|…)'` matched only `--port`, because vLLM writes the
option name on the line *after* `add_argument(` — and the next turn was the one
heredoc carrying every remaining section.

So the rule for the next run is not "write §1 first", it is **after the deploy script
lands, the next turn is §3 plus the closing message**: §3's bench line is
`tutorial-command-audit.md` check 2's two greps (one tool block), §4/§5 are empty
result slots with their filling command and 2-3 precedent lines, and nothing in them
needs an audit. This run also re-verified that `--swap-space`/`--disable-log-request`
are absent at v0.18.0 — settled here since #11739; a settled fact is not worth the
turn that the message gets written in.

## What the report must not do

- Present doc numbers as measurements of this environment, or fill a slot with
  a plausible value. Leave the slot empty and name the command.
- Assert a defect without the pinned-tree grep behind it ("this flag was
  removed" needs the `rg -n` at *that* version; see `tutorial-command-audit.md`).
- Grade 完整性/易用性/正确性 without citing the defects behind the score.
- Blame the reporter's environment for a doc line that cannot parse.
- Cite this library, an earlier session of the same row, another row's run, or your own
  token/character budget — in the report **or** in the closing message. Everything above
  about *runs* is written for you; the tester reads a deliverable. A recorded replay of
  #11750 spent part of its closing message on 「本 row 此前一次运行（#11750）在无文件、无消息的情况下被截断」 —
  provenance the reporter cannot act on, and the last thing the cap cuts.
