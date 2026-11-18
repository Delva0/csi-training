# `[Doc]` reports: the published guide's serve block exceeds the hardware it names

## Scope

The report is filed as a documentation issue ("the latest official guide causes OOM", "the tutorial's
command cannot start"), or its root cause is a doc/config block rather than a code path: a deployment
that **copied a documented `vllm serve` block** (model tutorial, release note, blog) and dies at startup —
OOM during the profile run or KV-cache init, a `ValueError` from the KV-budget check, or an unrecoverable
restart afterwards — with no `vllm_ascend/` frame in the traceback and no code change on the reporter's
side. The decided question is *which value in a published serve block exceeds what the page itself says
the hardware or release supports*: context length (`--max-model-len`), `--gpu-memory-utilization`,
`--max-num-batched-tokens`, or the per-card memory the layout assumes. Worked report: #18158 (A2, 1M
context, DeepSeek-V4.1-Flash; zero comments). Two members have **no symptom at all** — a `[Community
Test]` tracking row whose body ends `测试通过` with zero comments, and a `[Misc]` question about a page's
own structure (duplicated `docker run` steps, a stale hardcoded `export IMAGE=…:v0.1x.0rcN` inside a
`|vllm_ascend_version|` fence): see "Nothing failed" below. A task sheet is the same shape with its own
page, and its 蓝区镜像 tag is the governed ref (`community-test-task-sheets.md`). The other doc classes are
elsewhere: proxy↔connector pairing (`pd-proxy-examples.md`), a dependency floor
(`model-type-support-floor.md`), an image-only env var (`../vllm-ascend/dockerfile-env.md`), and a
**contributor's dev/test-env** walk-through ("A contributor dev-env report is a build-artefact gap" below).

Not this page: an OOM whose traceback names the mask (`attention-mask/`), the weight loader
(`quantization-load.md`) or the KV-cache tensors (`startup-and-stall.md` §3) — those pages own the
mechanism; this one owns the *documented configuration* half, which raises nothing in Python. The
rejecting check and its `ValueError` are `startup-and-stall.md` §3b.

## Which ref is the governed one

- "The **latest** guide" (`docs.vllm.ai/projects/ascend/en/latest/…`) is the **default branch's docs
  tree**, not the reporter's release tag. Read `docs/source/…` from the clone's working tree, and say
  that is what you read.
- A **versioned** docs URL (`…/en/v0.7.3.post1/installation.html`, `…/en/v0.23.0/…`) is that tag's tree:
  `git show v0.7.3.post1:docs/source/faqs.md`.
- Model tutorials are **not** backported. Check whether the page existed at the release the reporter
  names: `git ls-tree -r --name-only <tag> docs/source/tutorials/models | grep -i <model>`. #18158 names
  "v0.18.0 FAQ, September 2026", but `v0.18.0` (tagged 2026-04-30) has **no DeepSeek-V4 page at all**
  (only R1/V3.1/V3.2/OCR2) and the page was added by `7a5b24735` (#16294, 2026-09-11). So the report's
  version statement cannot be reconciled: quote the mismatch, target the docs tree, and do not pretend
  the tag's docs said it. A `latest`-only page means the guide post-dates their build.
- A page that is a **deprecation stub** at `HEAD` is main-only state, not the reporter's: `[Doc][Misc]
  Deprecate discontinued models and update support matrix` (`3185b7b0a`, #19818, 2026-10-26) left
  `Qwen3-Coder-30B-A3B.md` with three lines pointing at the historical docs. So
  `git diff <tag> HEAD -- <page>` prints the whole page as deletions and answers nothing about
  "was this fixed?"; read the tag, and only use HEAD for a report that quotes `latest`.

## A contributor dev-env report is a build-artefact gap, not a serve block

Worked report: **#17718** — a new contributor followed `testing.md` into the `nightly-main` container,
mounted their checkout to edit code, and lost the session to `ImportError: cannot import name
'_build_info'`, missing CANN/NNAL `set_env.sh`, an unset `SOC_VERSION`, `--no-build-isolation`, a 403 on
the documented Tsinghua mirror, and `bash format.sh ci` not running any test. It is a `[Doc]` report, but
the root cause is one **generated file**, not the prose: the page's container path assumes "the code is
the image's own copy" while a contributor must mount theirs (mechanism, verified lines and the fix
sequence: `../vllm-ascend/install-and-image.md` §4).

What the pages say at the default branch (quote these, they are the deliverable):

| claim in the report | verified state |
| --- | --- |
| the test guide never mounts a checkout, yet mounting is required to edit code | `docs/source/developer_guide/contribution/testing.md` Single-card docker run `:66-87`, Multi-cards `:104-125` — no `-v <src>:/vllm-workspace/vllm-ascend`; the install blocks (`:91-100`, `:129-137`) is only `pip config set …tuna` + `pip install -r requirements-dev.txt` |
| `set_env.sh`, `COMPILE_CUSTOM_KERNELS`, `--no-build-isolation`, `SOC_VERSION` are missing there | true of both container tabs; `SOC_VERSION` appears only in the Local (CPU) tab `:57-59`, and the `COMPILE_CUSTOM_KERNELS` comment lives in `vllm_ascend/envs.py:39-44` |
| "Run CI locally" does not run tests | `contribution/index.md:35-60` ends in `bash format.sh ci`, and `format.sh:40-44` is `pre-commit run --all-files --hook-stage manual` (style only) |
| the stable install page and the nightly dev image give conflicting signals | `installation.md` routes its image tags through `quick_start/ascend_image/*.inc.md` (`quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-…`) while the test guide pins `nightly-main`; no page cross-references the other |

Answer shape: one line per missing item with the page `file:line`, the mechanism row (the editable
install + `_build_info`) as the concrete root cause with `setup.py:157`/`:161`/`:182` and
`Dockerfile:65`, the failure-free container sequence as the workaround, and `status: doc defect, issue
open, no PR merged at the default branch`. Nothing here needs a `vllm_ascend/` serving path; do **not**
read the model runner, and do not chase the mirror 403 (it is external, `reported`).

## Nothing failed: `[Community Test]` rows, `[Misc]` page questions and stale pins

Some doc reports contain no symptom at all. Two recorded shapes:

- a vllm-ascend **`[Community Test]`** tracking row copied out of `推理众测任务书-vllm.xlsx`
  (`分类 / 任务名称 / 蓝区镜像 / 卡数 / 配套信息`, `### 任务问题和需求: 测试通过`, state `open`, **zero
  comments**) — #11737, Qwen3-Coder-30B-A3B on 4×Atlas 800I A2 with image `v0.18.0`. This one has its own
  page: `community-test-task-sheets.md`, which owns the sheet's 蓝区镜像 tag, the page-vs-sheet comparison
  and the recipe answer. The family's thread is empty by construction (14/14 open, zero comments).
- a **`[Misc]`** question about the page itself — #9988, "why does the Qwen3-Dense doc create two
  containers?" → "The page ships the same step twice" below.

There is no failing path to localise, so **do not hunt for what "might" break**: one recorded run spent
~10 K characters ranking candidate failures (EP on 4 cards, `max_model_len` versus the KV budget, MoE
comm) and shipped nothing. The deliverable is the page read at the ref the report names, plus the values
that decide the task's own question.

**The stale image pin is the one fileable defect in a no-symptom task sheet.** At `v0.18.0` several model
pages hardcode an older tag *inside a `:substitutions:` fence*: `Qwen3-Coder-30B-A3B.md:32` `v0.11.0rc1`
(seven releases behind the sheet, and the directive does not rewrite a literal), `PaddleOCR-VL.md:31`
`v0.13.0rc1`, `Qwen3-235B-A22B.md:35` and `Qwen3-VL-235B-A22B-Instruct.md:34` `v0.11.0rc2`/`-a3`,
`Qwen3.5-27B.md:35` `v0.17.0rc1` — while sibling pages substitute `|vllm_ascend_version|` (Qwen3-Dense
`:62-64`). One grep lists the family:
`git -C .../vllm-ascend grep -n 'quay.io/ascend/vllm-ascend:v' <tag> -- docs/source/tutorials/models`.
A task whose 蓝区镜像 is `v0.18.0` and a page that renders `v0.11.0rc1` is a real doc defect with a
one-line workaround (`export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0`, `-a3` on A3).

Answer shape for a task sheet or a stale pin: `no failure reported` + the page ref + the pin/TP facts +
the stale-pin workaround, and `status: doc defect (stale image tag); the reporter's task itself passed`.
Two calls, then the reply. A duplicated step has its own answer shape below.

## Compare inside the page: tabs are per hardware class

At HEAD, `docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` documents A2 as 8 NPUs of 64 GB (`:74`)
and then its **A2 tab** (`:643` onward, the four-node 1P1D layout — the reporter's layout) uses:

| role | flag | line (HEAD) |
| --- | --- | --- |
| A2 prefill | `--max-model-len 1048576`, `--gpu-memory-utilization 0.9` | `:786`, `:797` |
| A2 decode | `--max-model-len 1048576`, `--gpu-memory-utilization 0.95` | `:896`, `:908` |

i.e. the same 1M context and the same utilizations as the A3 tabs (`:407`/`:496`). Any reader who picks
the A2 tab on 64 GB cards inherits the A3 envelope. Read the tab, not the page title, and quote the line
numbers in the reply. `git log --format='%h %ad %s' --date=short --
docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` then dates the block: A2 updates `0b60ebe4d`
(#19586, 2026-10-25) and `8a94735e1` (#20292, 2026-10-30).

## The four lines that decide a guide OOM (#18158, verified on the default branch, 2026-11-17)

1. **The serve block's own value** (the table above), beside §3.1's "**A2 series**: … 8 NPUs with 64GB
   memory per NPU".
2. **The page's own limitation section.** §11 `:1856`: "The A2 configuration is retained unchanged and is
   not revalidated by this update. Its revised configuration will be documented separately."
3. **The sibling page's warning.** `DeepSeek-V4-Flash.md:1698`: "The 1M cases require a separately
   validated long-context deployment, not a larger `--max-model-len` in these scripts." (its A2 tabs pin
   1048576 too — the contradiction is internal to the docs tree, which is what makes it a doc defect).
4. **The release-notes known issue.** `user_guide/release_notes.md:488`: "GLM-5.2 1M-context deployments
   were validated only on Atlas 800 A3; the A2 series was not validated for 1M context." And `:774`:
   DeepSeek-V4 KV Pool needs `--no-disable-hybrid-kv-cache-manager` or the service OOMs at startup, with
   "a sequence of 1M tokens takes approximately 300 GB" (#9975) — the scale of the object.

The envelope is also tabulated per hardware: `user_guide/support_matrix/supported_models.md:54` carries
`GLM-5.3` = `135k (A2) / 1M (A3)` while the DeepSeek V4-Flash/V4-Pro rows list plain `1M`
(`:29-30`, `:40-41`), so a 1M context is model- *and* hardware-dependent — read the row for the reported
model, do not generalise either way.

## Deliverable shape

The defect is the **doc line plus the number that has to change**, not a code path:

- quote the block (page at ref, tab, line) and the hardware row or limitation it contradicts;
- name the validated counterpart (the A3 tab, the matrix's A2 column value, the earlier guide that
  worked) and what the reporter should set instead: a context the matrix validates, plus
  `--gpu-memory-utilization` below the tab's value and a `--max-num-batched-tokens` from the validated
  config;
- state the status honestly — this is a documentation/validation-scope issue, so the fix is a doc PR or a
  matrix row, and no `vllm_ascend/` line raises it. Cite the four lines, not a frame.

**Workaround** for #18158: stay on the reporter's own previously-working window (128K/512K) — do **not**
raise `--max-model-len` to "use the guide's value"; long context on A2 is a separately validated
deployment, and A3 is where 1M was validated. Cost: the context length the newer guide advertises.
**Status**: #18158 is an open `[Doc]` report with no comments; the doc defect above is verified, the OOM
arithmetic is not (the reporter pasted no log).

## The "irrecoverable after OOM" half is operator-owned

A restart that fails with conservative settings is a **residual-state** question, not a code path.
The runner's own guard names it: `Error in memory profiling. … This happens when the NPU memory was not
properly cleaned up before initializing the vLLM instance.` (`vllm_ascend/worker/worker.py:729` at HEAD,
`_310p/worker_310p.py:120`). Residual NPU processes, stale shared memory, a half-initialised multi-node
HCCL group and partially written compile caches are **runtime/driver state**, not repo code. Ask for that
line, `npu-smi info`'s per-process table (a surviving pid holding device memory — also the falsifier:
"no process holds the card"), `ps -ef | grep -E 'vllm|EngineCore'`, the `/dev/shm` entries for the
engine's broadcast queues (the guide's own `--shm-size=512g`), the HCCL/CANN plog for the failed group
teardown, and whether the ranks were `SIGKILL`ed. The reporter's list — memory fragmentation, HCCL group
corruption, CANN compile caches outside `/root/.cache` — is their `hypothesis`; do not adopt it as the
mechanism without one of those checks. "Needs `pkill -9` + shm cleanup + re-init" is a workaround, not a
root cause.

## The page ships the same step twice (a `[Misc]` question, not an OOM)

The other doc shape: a model tutorial whose *Installation* chapter creates the container and whose
*Deployment* chapter creates it again, so a reader cannot tell whether both must run, which one to use,
or why they differ. Nothing crashes; the report is a question ("why two containers?"), and the deliverable
is the difference between the blocks plus the page's own cross-reference.

Worked report #9988 (`Qwen3-Dense`, `zh-cn/v0.18.0`, filed 2026-06-04, closed): at `v0.18.0`
`docs/source/tutorials/models/Qwen3-Dense.md` has

- `:53` `#### Docker run` under `### Installation` — `--name vllm-ascend-env`, `--net=host`, all eight
  `/dev/davinci*`, the extra `hccn_tool` mount, **no** `-p`, **no** `--privileged`;
- `:92` its prose links to the section anchor `#run-docker-container` — "In the [Run docker container],
  detailed explanations are provided through specific examples" — so installation redirects the reader
  into the second block;
- `:112` `### Run docker container` under `## Deployment` — `--name vllm-ascend`, `--privileged=true`,
  only `davinci0-3`, `-p 8113:8113`, and neither `--net=host` nor the `hccn_tool` mount, i.e. the block a
  multi-node/HCCL deployment cannot use;
- and the deployment note still says 「v0.11.0rc2-a3 is image tag」 while its code block substitutes
  `|vllm_ascend_version|` — the tell that this chapter is an older standalone guide pasted in whole.

Read those four lines and the answer follows: **use the first container and run the serve command inside
it** (`--net=host` makes the second block's `-p` unnecessary, and `--privileged` buys nothing once the
devices are mapped), which is also the maintainer's answer in the thread (`earthmanylf`: 「就本issue来看，采用
第一个即可，两者差别仅有部分docker启动参数不同…Qwen3-Dense的文档会在近期进行更新」). Report that as
`reported`, not as a code fact.

**Fixed later, on the docs tree.** The default branch now has one container block per hardware tab
(`§4.1 Docker Image Installation`, `:76` A3 / `:120` A2 / `:152` 300I DUO) and `§5 Online Service
Deployment` (`:219`) reuses that container, with the stale image-tag note gone — the rewrite is
`908915982` (#10728, 2026-06-29, 635 changed lines in that page), first tag containing it **`v0.23.0`**;
the `v0.18.0`/`zh-cn/v0.18.0` page the reporter read still has the duplicate. So this class is
"fixed on the default branch, still visible on the versioned docs URL a user lands on" — say which one
you read, because a doc fix on `main` does not change a pinned release page.

Answer shape for this class (three lines, no code window): the two headings with `path:line`; the
parameter deltas that matter (device list, `--net=host`, mounts, `-p`, `--privileged`); one sentence
saying no second container is needed and what to run instead; then the fix commit/tag or "open". A
`docker run` line is not evidence a container was started, and the blocks' parameter drift is not
itself a bug — the defect is the duplicated step plus the internal cross-reference.

## Falsifier and recheck conditions

- Falsifier: the same image and layout with the context reduced to a matrix-validated value (and
  utilizations lowered) starts and serves — then the block was the defect. If it still OOMs at a
  validated context, route back to `startup-and-stall.md` §3 / `quantization-load.md` and read the real
  frame; if the log shows an allocator OOM (`NPU out of memory`, a failed `torch.zeros`) instead of the
  KV-budget `ValueError`, the lever is weight/HCCL/activation memory, not `max_model_len`.
- Recheck whenever the tutorial, matrix or release notes change: the A2 tabs and the §11 sentence are
  moving targets (`#19586`, `#20292` both rewrote A2 deployment in six days) and the model tutorial may
  not exist at the older tag at all. If the A2 block's `--max-model-len` ever drops to the validated
  window, this report's class is gone.
