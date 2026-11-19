# `[Community Test]` task sheets: a deployment question with no failure to diagnose

## Scope

The issue title is `[Community Test] <model> <task kind>` and the body opens with
`> Source: 推理众测任务书-vllm.xlsx row N`, then a task-sheet table (分类 / 任务名称 / 任务类别 /
蓝区镜像 / 权重链接 / 配套信息 / 卡数（Atlas 800I A2）/ 内存 / 硬盘), 核心要求 pointing at a **versioned**
tutorial URL, a 交付件 + 验收标准 block (部署文档, 部署脚本, 测试结果, 问题记录) and a
`### 任务问题和需求` section that often reads only 「测试通过」. There is no start command, no log and no
traceback: the sheet asks "can this model be deployed with this image, on this hardware, following this
page?".

Verified family shape (GitHub search, 2026-11-17): **14** such issues in `vllm-project/vllm-ascend`, all
created 2026-07-09, all `open`, **all with zero comments** (#11737–#11747, #11769–#11771). So the thread
is empty by construction — `…/issues/N/comments` returning `data: []` is the whole truth, and there is no
twin to read, no maintainer lever in the thread, and nothing to gain from an issue search.

The deliverable is therefore a **deployment recipe plus the page lines that contradict it**, exactly the
shape of `doc-guide-vs-capability.md`, except the governed artifact is the sheet's 蓝区镜像 tag rather than
"the latest guide". Not this page: anything that carries a traceback (route by its deepest frame).

## Triage order — this fits in batch 1 plus one code window

1. Take the sheet's 蓝区镜像 tag (`v0.18.0`, `v0.21.0rc1-310p`, …) as the ref, not `main` and not the
   reporter's prose. One call gets both artifacts at that tag:

   ```bash
   cd /workspace/repos/vllm-project/vllm-ascend
   git show <tag>:docs/source/tutorials/models/<Model>.md | cat -n
   git show <tag>:docs/source/user_guide/support_matrix/supported_models.md | grep -n -i '<family>'
   ```

   A missing page at the tag is a finding (the sheet cites a page the release does not have); a page that
   exists only at `HEAD` is a newer guide, not their state.
2. Compare four things **inside that page**: the image pin in the install block, the hardware statement,
   the serve block (TP/EP and the flags it does *not* pass), and the intro's capability claim. The support
   matrix's per-model row is the independent second signal (feature columns, then `Supported Hardware`).
3. Reply with: the sheet's image + the recommended serve command, the page lines that disagree (quote
   `file:line`), the one capacity check that matters on the sheet's card count, and what you did not run.

## Worked case #11737 (Qwen3-Coder-30B-A3B, 4×A2, sheet image `v0.18.0`)

- Sheet: 蓝区镜像 `quay.io/ascend/vllm-ascend:v0.18.0`, HDK 25.5.0.B078, 4 cards (Atlas 800I A2), 1 T RAM;
  the task is 资料验证和部署测试 against
  `docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3-Coder-30B-A3B.html`; 任务问题和需求
  「测试通过」; zero comments.
- [verified] at `v0.18.0` the page still tells the reader to **install a seven-release-old image**:
  `:29-32` is a ` {code-block} bash` with `:substitutions:` whose body hardcodes
  `export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc1`. The substitution directive does not rewrite a
  literal, so the rendered `en/v0.18.0` page ships `v0.11.0rc1` too — and the sheet mandates `v0.18.0`.
  `:25` "first supported in `vllm-ascend:v0.10.0rc1`" is history, not a pin.
- [verified] the page contradicts its own hardware statement: `:19` says the BF16 checkpoint "requires … 1
  Atlas 800 A2 node (with 8x 64G/32G NPUs)", while `:63` says TP ≥ 4 on 32 GB cards and `:69` serves with
  `--tensor-parallel-size 4 --enable_expert_parallel` — i.e. the command needs 4 cards, a quarter of what
  `:19` demands, and 4 is exactly what the sheet provides. Quote both lines; do not call 4 cards
  insufficient.
- [verified] the same tag's support-matrix row `Qwen3-Coder` (`supported_models.md:25`) marks
  `Supported Hardware` A2/A3 with Tensor/Data/**Expert** Parallel ✅, so `--enable_expert_parallel` at TP4 is
  a supported configuration and not the defect. The row's `max-model-len` column is empty.
- [verified] the serve block passes **no** `--max-model-len`, so the engine takes the checkpoint's
  `max_position_embeddings`; the page's `:5` "extended context support of up to 1M tokens" is a family
  claim the block never configures. On 4×32 GB this is the number to check against the KV budget — the
  refusal and its estimate are `startup-and-stall.md` §3b. Do not derive the KV size from remembered model
  geometry: the checkpoint's `config.json` is the only source, and it is not in the clones.
- Recipe that answers the sheet: image `quay.io/ascend/vllm-ascend:v0.18.0` (the sheet's), the page's
  `:69` serve line (`--tensor-parallel-size 4 --enable_expert_parallel`), `--max-model-len` chosen from the
  checkpoint's value and the KV estimate, plus the page's `:77-86` `curl` check. Whether v0.18.0 starts
  inside 4×32 GB with the default context is `hypothesis` — no hardware here.

## Worked case #11770 (Qwen3.5-27B-w8a8-mtp, 4×A2, sheet image `v0.18.0`) — a tooling blocker, not a deployment defect

- Sheet: 蓝区镜像 `quay.io/ascend/vllm-ascend:v0.18.0`, HDK 25.5.0.B078, 4 cards (Atlas 800I A2), 1 T RAM,
  weights `Eco-Tech/Qwen3.5-27B-w8a8-mtp`; 核心要求 points at
  `…/blob/releases/v0.18.0/docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md`; the 任务问题和需求 line
  reads **「容器内无法使用postman工具发送推理请求」**; zero comments — no start command, no log, nothing else.
- [verified] **The cited tutorial path does not exist at the sheet's tag.** `v0.18.0` carries
  `docs/source/tutorials/models/Qwen3.5-27B.md` (`git cat-file -e
  v0.18.0:docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md` fails), and the merged Qwen3.5/Qwen3.6
  name first exists at `v0.22.1rc1` (`acff4ed2d`, #11008). Locate the page before reading it —
  `git ls-tree -r --name-only <tag> docs/source/tutorials/models | grep -i qwen` — because a failing
  `git show <tag>:<cited path>` here is a **sheet** defect, not a missing guide, and the working tree's own
  filename is not evidence for the sheet's tag. Quote the real path.
- [verified] **`postman` is not a plugin defect, and it is the whole 问题记录 deliverable.** The image is a
  headless container (no X server) and Postman Desktop is an Electron GUI, but the page's own block needs
  neither: `:52` is `docker run --rm … --net=host`, and `:106-107` serve `--host 0.0.0.0 --port 8000`, so
  requests go from the **host** (Postman, `curl`) at `http://<node-ip>:8000/v1/completions` — `:148-155` is
  the page's own `curl` — or from inside the container with `curl` / `python3 -c "import requests"`. A
  bridge-network container needs `-p 8000:8000` (`:49` says to expose the ports). Answer that, not the
  model's accuracy.
- [verified] **The page's numbers do not fit the sheet's 4 cards.** `:21-22` says both the BF16 and the w8a8
  checkpoint "requires 1 Atlas 800 A3 (64G × 16) node or 1 Atlas 800 A2 (64G × 8) node", while its serve
  block is `--data-parallel-size 1 --tensor-parallel-size 2` (`:108-109`) — 2 of the 4 cards, the other two
  idle. On a 4-card sheet serve TP4 or DP2×TP2 and keep `--max-model-len 133000` (`:114`) inside the KV
  budget (`startup-and-stall.md` §3b). `:119-121`'s MTP block, `FULL_DECODE_ONLY` and `enable_cpu_binding`
  are part of the page's measured command, not decoration; do not open with "4 cards is not enough".

## Traps

- **A `任务问题和需求` line is not always a deployment failure, and one that names a *tool* is not a code
  report.** #11770's is a GUI-tool complaint (「容器内无法使用postman」 inside a headless container) and a
  sibling sheet's may be 「测试通过」; both are answered from the page (headless image, `--net=host`,
  `--host 0.0.0.0`, the port line) — the host-side request path plus the page's own contradictions —
  rather than by diagnosing the engine, and neither is a reproduce-the-environment task.
- **A sheet's 参考链接 404s more often than it should.** Take the tag from 蓝区镜像, list
  `docs/source/tutorials/models/` at that tag, and match by model family; the URL may name a page that was
  merged or renamed later (Qwen3.5-27B → Qwen3.5-27B-Qwen3.6-27B at `v0.22.1rc1`).
- **Model tutorials are rewritten and eventually deprecated, so `git diff <tag> HEAD -- <page>` is not a
  fix-finding tool.** For this page HEAD carries a 3-line stub — "This model has been discontinued since
  vLLM Ascend v0.30.0 … see the historical version documentation" — written by `3185b7b0a`
  (`[Doc][Misc] Deprecate discontinued models and update support matrix`, #19818, 2026-10-26). The diff
  therefore prints the whole page as deletions and `git show HEAD:<page>` as a stub; that is main-only
  state and says nothing about the sheet's tag. A page that was *deleted* likewise means the model dropped
  out of support after their release, not that the sheet is wrong.
- **An empty comment list is the norm here (14/14), not a fetcher bug.** Do not re-fetch with `paginate`,
  and do not go looking for the neighbouring issue number.
- **The 交付件 list is not a request for files.** 部署文档/部署脚本/测试结果 are what the community tester
  submits to the task owner; the reply is the recipe, its caveats and the doc lines to file.
- Do not answer with the model's weight repository, the `transformers` support floor, or a kernel
  hypothesis: nothing failed, so the only ownable defect is the published block that disagrees with itself.

## Recheck conditions

Task sheets are copied from an `.xlsx` per campaign and the tutorials are rewritten each release (the
Qwen3-Coder page was rewritten twice: #10148, then #11239 tabs). Re-read the page at the sheet's tag before
reusing any line number, and re-run the `[Community Test]` search if the family's comment behaviour matters
to the answer — if a sheet ever gains a thread, that thread outranks this page.
