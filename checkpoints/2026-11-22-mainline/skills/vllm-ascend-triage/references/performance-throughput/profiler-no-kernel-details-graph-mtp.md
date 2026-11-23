# `kernel_details.csv` absent: graph mode + MTP collection is a capability/regression report

## Scope

The service is healthy and answers requests; the complaint is that the Ascend PyTorch
Profiler collection produced **no usable kernel data** — "混部脚本采集profiling，解析后无
`kernel_details.csv`" — and the launch carried graph mode and speculative decoding. The
same recipe on the PD **prefiller** alone collects fine, and turning the features off
(`--enforce-eager`, no `--speculative-config`) collects fine.

Deliverable: whether the collection is even a valid one for that window, which single axis
(graph, MTP, or their intersection) is implicated, and — when the reporter says the
combination used to collect — the **regression** statement with a version bisect. It is not
a model, driver or card fault, and it is not the "counts look wrong" question of
`profiler-opcount-evidence.md`.

Recorded #12241 (branch `releases/v0.23.0` `cd46f390f8`, GLM-5.2, TP16/DCP16 + DSA sparse
C8, `--profiler-config '{"profiler": "torch", "torch_profiler_dir": …,
"torch_profiler_with_stack": true}'`). **The pasted serve script is the working control,
not the failing run**: it carries `--enforce-eager` and both `--compilation-config` and
`--speculative-config` are commented out (`# … "method": "deepseek_mtp", "enforce_eager":
true`), while the failure was reported for the 混部 script with them on. Ask for the failing
launch before attributing anything. The reporter later stated the combination had always
worked in older versions and asked for it to be restored, then confirmed collection works
again — i.e. the disposition was a downstream regression, not a vendor limitation.
As filed (2026-07-17) the report carried none of that: the restore request came on
2026-07-20 — "@saisharan0103 Could you please help restore this? This combination has
always been supported in older versions, and clearly users will need it" — and
"现在可以采集了" only on 2026-08-25, five weeks on. Two consequences for the first
message: the old release **is** the control the reporter will produce on request, so ask
for its tag rather than waiting for it; and the workaround (eager, no MTP) is a
collection-only fallback, never the resolution — this pair is a supported combination
somebody dropped, so say that it should be restored, and do not publish "graph mode +
MTP cannot be profiled" as a limitation.

## Rules

1. **Decide file-existence semantics before any cause.** `kernel_details.csv` is written
   into `<profiler_dir>/<host>_<pid>_<stamp>_ascend_pt/ASCEND_PROFILER_OUTPUT/` only after
   the raw directory is **analysed** (`torch_npu.profiler.profiler.analyse(dir)`,
   `docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:86-105` at
   vllm-ascend v0.22.1rc1, which lists `kernel_details.csv` as "Kernel-level related
   data"). torch_npu's own test uses the file as the "device collection happened" predicate:
   `test/profiler/test_npu_profiler.py:75` names it, `:139-145` (CPU activity only) asserts
   it is **absent**, `:152-158` (NPU activity only) asserts it is **present** (Ascend/pytorch
   `v26.0.1-pytorch2.10.0` `8751b36`). So: (a) which directory was analysed, (b) does
   `ASCEND_PROFILER_OUTPUT` exist at all, (c) are `op_statistic.csv`/`operator_details.csv`
   there while only the kernel table is missing. That one listing separates "collected
   nothing" from "analysed the wrong dir" from "device rows missing".

2. **The profiler flags are not the reporter's to set — that is why this is a version
   question.** vllm-ascend builds the torch_npu profiler itself:
   `vllm_ascend/profiler/torch_npu_profiler.py:38-73` at v0.22.1rc1 always requests
   `ProfilerActivity.CPU + NPU` with `export_type=Text`, `profiler_level=Level1`,
   `data_simplification=True`, and `tests/ut/profiler/test_torch_npu_profiler.py:92-104`
   pins exactly that set. Nothing in `--profiler-config` reaches those values, so "switch
   the level / disable simplification" is a **code change**, and the question becomes which
   release made graph/MTP collection stop. Do not derive a CANN story instead.

3. **A combination the reporter says collected on an older release is a regression: the
   older release is the control that refutes vendor-limitation theories.** Rank
   "`msprof`/CANN cannot record graph-replay tasks under `npugraph_ex`", "graph capture
   precedes the collection window", "the daemon and the torch profiler exclude each other"
   as *labelled hypotheses* below it, and make rung 1 a one-command bisect: collect the
   same fixed request window on the last release that worked and on the reported one, and
   say which release that is (`git log`/release-notes line, not a guessed tag). Recorded
   #12241's own follow-ups are the evidence for the ordering: the reporter's "这个组合在旧版本
   一直支持" was followed by a maintainer restore and "现在可以采集了", so an answer built on
   "CANN does not support this" would have been wrong; the recorded run had already put
   that hypothesis first. The **mechanism stays open** — the follow-ups name no fix, so say
   the combination was restored and keep the cause unverified rather than inventing one.

4. **Read which launch the paste is before ranking anything, then count its variables.**
   A profiler report pastes the serve script that *collected*, which is usually the
   reporter's own ablation — #12241's block has `--enforce-eager` present and the two
   feature lines commented out. Remove three variables at once and the graph one is not the
   flag they think: `--enforce-eager` on v0.22.1rc1 drops torch.compile **and** the
   `npugraph_ex` backend **and** the ACL graph while the MTP row is toggled in the same
   launch. The chain is two reads, and the first one is the easy miss — the tokens sit six
   lines apart, so `rg 'enforce_eager' vllm/config/*.py | rg cudagraph` returns **nothing**
   (recorded cost: three calls). Read it as
   `rg -n -A6 'if self.model_config is not None and self.model_config.enforce_eager'
   vllm/config/vllm.py`: vllm v0.22.1 `0decac0` `:1032-1038` sets `mode = CompilationMode.NONE`
   and `cudagraph_mode = CUDAGraphMode.NONE` (`:1233-1238` also empties the capture sizes),
   then vllm-ascend v0.22.1rc1 `5f6faa0` `vllm_ascend/platform.py:590-592` writes
   `enable_npugraph_ex = False` in the `cudagraph_mode == NONE` branch — so a
   `"enable_npugraph_ex": true` left in `--additional-config` is **inert**, and an argument
   that treats the paste as "graphs on" is wrong. Line numbers move between pins (same
   branch at v0.23.0rc1 `platform.py:595/622`); the per-pin gate table is
   `../startup-failures/graph-mode-not-applied.md`. Their "图模式" is not
   `--compilation-config`: it is `ascend_compilation_config.enable_npugraph_ex`
   (`docs/source/user_guide/feature_guide/graph_mode.md:140-166`) whose compile path is
   `compilation/compiler_interface.py:283-296` → `npugraph_ex_compile` (`:149-164`, torchair
   fallback) — in their launch the `--compilation-config` line is commented out. So the
   first message owes the one-axis matrix (graph on/MTP off, graph off/MTP on, both, neither
   — decide graph with `enable_npugraph_ex`, not with `--compilation-config`) plus, when no
   restart is acceptable, the zero-restart split: profile a **prefill-only** window (long
   prompt, `max_tokens=1`) and a **decode-only** window (short prompt, long output). On a PD
   deployment the prefiller is eager and carries no MTP drafts, so "P node alone collects
   fine" is equally consistent with prefill-vs-decode — say that instead of crediting PD.

5. **Ask for the artefacts the collection is decided on, and treat the empty environment
   block as a gap.** One `ls` of the profiler dir (every `*_ascend_pt`, per rank — a
   TP16/DCP16 run writes one per worker, so a file missing on all ranks is the window and
   on one rank is the worker), the `ASCEND_PROFILER_OUTPUT` listing, the `torch_npu`/CANN
   versions from `collect_env.py` (recorded #12241 shipped the template's placeholder
   instead), and the `--profiler-config` line as launched. With DP/TP fan-out, name
   `worker_name` in the answer's request list so the reporter compares like with like.

## Not this page

- The table exists but shapes read `N/A`: `profiler-shape-columns-custom-op-launch.md`.
- The table's counts/idle time disagree with wall time or with an unchanged TPOT:
  `profiler-opcount-evidence.md` (collection health, then per-step normalization).
- `/start_profile` returns 404/500 although `--profiler-config` is set:
  `../observability-and-logs/online-profiling-endpoints.md`.
