# Problem classes this skill covers: ten entry shapes and their routes

Moved verbatim out of `SKILL.md` (§*Problem class this covers*) so the always-on
rules stay readable and the body stays inside its write limit. It is the *coverage
statement and per-shape route* of the skill — read the entry for the shape you are
handling on the turn you classify it, then follow its topic index. Inherit every
rule in `SKILL.md`; this page adds applicability, exclusions and route only.

Ten entry shapes, one deliverable: which component and line owns the symptom (or
which failure lane and phase), what the reported evidence proves and does not
prove, ranked candidate causes each with the cheapest confirming check, and the
concrete next step for the reporter. The last shapes deliver a *decision* (sizing),
a *document review*, a *capability verdict* or a *change plan* instead of a root
cause, but the same rules apply — answer first, evidence pinned to the reported
version, gaps stated.

- **Issue report** (template-shaped: environment dump + logs + narrative) about
  something that went wrong on an Ascend NPU host — a device-side operator crash, an
  accuracy/score gap, garbled or truncated output, a long-run OOM, a container metric
  that climbs for days, a performance claim, a flag or endpoint that does nothing, or
  a `python -c "import <pkg>"` that aborts in a shipped library. When the task is only
  "explain/answer this issue", the answer is still evidentiary, not a summary.
- **Install / pip-resolution report**: no engine, no worker, no NPU traceback — two
  `pip install` lines and what they printed (`Could not find a version that
  satisfies the requirement …`, `No matching distribution found for <pkg>`,
  "unmatched torch version between vllm=X and vllm-ascend=Y", an `ImportError:
  Please install vllm[bench] …` inside a shipped image), or an install that finished
  and only **warned** (`requires torch==2.10.0`). The deliverable is *which* artefact
  pip could not see — a mirror-only pin the metadata cannot carry, an unsatisfiable
  torch/torch-npu/torchvision set, an air-gapped container with no reachable index —
  never a release survey — `../image-dependency-faults/pip-index-visibility.md`
  and `pip-resolver-warning-block.md`.
- **Deployment failure**: engine start or a running worker fails — `EngineCore failed
  to start`, `WorkerProc initialization failed`, silent worker death, allreduce /
  `ERR01100` / `ERR00100`, OOM, "the model failed to load", an `AttributeError` /
  `TypeError` from a patched forward or a KV connector, or a `ValueError` from
  upstream `arg_utils` rejecting the parallel/DP flags.
- **Support / version question**: "does vllm-ascend <version> deploy <model>?" — a
  version gate with a one-line answer and a registry-level proof, not a bug hunt.
  When the literal ask is 「是否有模型对应版本表」, there are **two** tables and no
  model × version matrix: the release pairing (`docs/source/community/
  versioning_policy.md` §*Release compatibility matrix* — vLLM Ascend | vLLM | Python |
  Stable CANN | PyTorch/torch_npu, which is also the citation for "your two versions
  are the released pair") and the family list (`user_guide/support_matrix/
  supported_models.md`, ✅/❌ only, `:3` pointing at issue #1608); per-release
  *validation* lives in `release_notes.md` + `developer_guide/evaluation/
  accuracy_report/` + `tests/e2e`. The worked answer, with the coverage gap stated, is
  rule 7 of `../performance-throughput/quantized-dense-no-gain.md`.
  A **format suffix in the model name** (`-GPTQ-Int8`, `-AWQ`, `-FP8`, `-w4a8`) turns the
  same 是否支持 question into a quantization-format verdict that no family row can
  answer: route to `../startup-failures/quantization-method-gate.md` §C and answer from
  the pin's platform list plus the checkpoint's own `quantization_config` (recorded
  #2960: `Qwen2.5-32B-Instruct-GPTQ-Int8` on v0.9.0rc2 — the base family is ✅, the GPTQ
  variant has no NPU implementation there, and startup dies as a `KeyError` in the
  quant-method lookup, so "the model is supported" alone is the wrong answer).
  When the follow-up is "支持计划是什么样的 / when will F work for M on my SoC", answer
  in three separate levels — the base model family at that SoC (release-note line,
  support-matrix row, the SoC fallback `TODO` that bounds it), whether the *feature's*
  code branch exists (one grep), and whether anything covers the combination (tests,
  tutorials) — then the prerequisites and the grep whose result would change the answer,
  and no date (`../feature-enable-recipes/index.md` item 9).
- **Feature-enable / capability question**: "如何正确开启 <feature>", "F 单独开均有不同的
  报错，能否给出能跑通的示例", "does F work on model M on this SoC", usually with no log
  and two errors in prose. The deliverable is the validation verdict at the reported
  ref plus the constraints the code enforces, not a launcher assembled from them —
  `../feature-enable-recipes/index.md`.
- **PD / EPD disaggregation**: the deployment is up but P and D cannot move KV
  (`HIXL Connect failed | 103900`, HCCL `RanktableCheck` / `'device_ip' is not set
  correctly`, `wait_for_save failed`, `Failed to put N keys out of N`, a recipe that
  needs `hostNetwork: true`), or KV moves and the decode node emits wrong/alternating
  output on a hybrid linear-attention model (GDN/Mamba — but a **DSA/MLA** model such
  as GLM-5.2 that repeats one phrase to the token cap is a recipe-contract question
  first: `../pd-disaggregation/recipe-contract-diff.md`), or a KV pool is attached and
  the *first* transfer fails (`503900` / `HcclCommPrepare ret: 0x13`, the pool's
  `global_segment_size` the cause), or the pool serves for minutes and then one ADXL
  peer times out (`Transfer timeout to:`, `TRANSFER_FAIL (-800)` on the keys,
  `is not connected` for peers that never failed, optional `processSliceList`
  segfault — `../kv-pool-store/store-put-transfer-timeout-and-adxl-segfault.md`,
  and note the recipe's `ASCEND_TRANSFER_TIMEOUT=10000` is the code default), or the
  put never lands (`HcclBatchPut, hccl_result = 4`, `PutEnd` 0), or the decoder never
  receives the request
  (`proxy通信超时`) — an endpoint-count question before any connector read; transport
  identity and rank tables are a topology question, a pool deployment a config
  question — `../pd-disaggregation/index.md`.
  If instead a `LLMDataDistCMgrConnector` deployment **starts and then loops forever** on
  `INFO … Checking query_register_mem_status again` once a second (the decode worker has
  already printed `try link with remote`, the rank table, `comm name` and `cluster rank
  info`, so the handshake and the local memory registration worked): that wait is the
  connector's own `while True` with no timeout, and the field to read in the printed table
  is `server_count` — `"2"` for two engines on **one** machine in two container-network
  namespaces, because their `get_ip()`s differ, so the datapath required is cross-server
  RoCE over `device_ip` from inside a container; **relaunch both containers with
  `--net=host` (`hostNetwork: true`) and `server_count` becomes `"1"`** — reporter-confirmed
  #2498, a CANN limitation. Both sides' `ranktable` / `cluster_rank_info` / `comm name` must
  first be printed and compared (any difference is a config cause); cross-machine with
  matching values is the NPU NIC / after-sales rung —
  `../pd-disaggregation/llmdatadist-register-mem-loop.md`.
  When the transport lines are Mooncake's (`ascend_direct_transport.cpp:820 Failed to
  connect to target … 103900`, `mooncake_connector.py:410 Mooncake transfer failed …
  ret: -1`) rather than HIXL's, the decoder's plog is the first request, not a
  port/`hostNetwork` test: cross-node P/D is already multi-server, so the requirement is
  a `device_ip`, and `use_ascend_direct` (default `True` at v0.11.0rc2, engine-name
  only) is not the fix — `../pd-disaggregation/mooncake-adxl-connect-103900.md`.
  If instead the engines start and the **first request** kills one side's EngineCore
  behind a bare `EngineDeadError: … See stack trace (above)` on a deployment setting
  `--decode-context-parallel-size` > 1: DCP is a *matched* parameter (a 1M recipe's
  `8` reduced on one side fails the connector's divisibility assertion), so compare
  both serve commands before any source walk —
  `../feature-enable-recipes/context-parallel-pcp-dcp.md` §3.
  If instead requests *are* served and TTFT is slow while the prefiller prints
  `Some requests in prefill node fail to receive KV Cache transfer done signal in 120s`
  (`VLLM_LLMDD_ABORT_REQUEST_TIMEOUT`, default 120): that line is expiry bookkeeping
  emitted by design on the non-zero P `tp_rank`s — the advertised RPC port carries no
  `tp_rank` — so it is not the slow-TTFT mechanism. Settle TTFT with the concurrency and
  first-request discriminators before any connector or rank-table read —
  `../pd-disaggregation/llmdatadist-done-signal-and-ttft.md`.
  If instead the **decode node logs `Error in KVCacheTransferThread. error=unhashable
  type: 'list'.`** and the deployment keeps serving: that line is the recv thread's catch-all
  printed *in place of* the real failure — downstream `_mark_failed_recv_request` feeds
  `set[int].update(<list[list[int]]>)` a grouped `BlockIds` before the `logger.exception` that
  would have named it — so the paste carries no transport information; unmask (flatten, or the
  layerwise connector, which already flattens) before any HCCL, plog or rank-table read
  (`../pd-disaggregation/kv-transfer-thread-masked-error.md`).
  If instead the same catch-all carries a **bare** `AttributeError` on a downstream
  metadata field — `Error in KVCacheTransferThread: 'ReqMeta' object has no attribute
  'block_ids'`, with no environment block in the issue — the pin's own class **has** the
  field (a compat property over the renamed `block_ids_by_group`), so the line reports
  the reporter's *build*, not a reader bug: diff `vllm_ascend`'s install path and that
  one class's source on P and D before any source walk (reporter-confirmed #10398,
  「P和D节点没注意统一代码版本」) —
  `../pd-disaggregation/node-build-skew-attribute-error.md`.
  If instead **both engines run on one machine** and the second one to start never gets
  past worker init — `Bind_Failed(EJ0003) … port have been bound already` +
  `hcclCommInitRootInfoConfig … error code is 7` out of `w8a8_dynamic.py`'s
  `get_hccl_comm_name`, plog `[HcclSocket][Listen] … port[16666] fail`, "start P first,
  D crashes; start D first, P crashes" — the port is real and the collision is a shared
  **card**: intersect the two launches' device sets (the external-DP launcher numbers
  cards from 0 every invocation) before any `HCCL_IF_BASE_PORT` / control-plane
  arithmetic — `../pd-disaggregation/same-host-pd-device-partition.md`.
  If instead **one side's EngineCore dies on its own bookkeeping** and the other keeps
  serving, name the side and the owning structure from the last three frames before any
  transport read: a prefiller bare `AssertionError` in upstream
  `v1/core/sched/scheduler.py` `_update_from_kv_xfer_finished`
  (`assert req_id in self.requests`) is a send-completion for a request the scheduler
  already freed, and `step_with_batch_queue` in the same stack proves
  `--async-scheduling` (or PP>1) — one launch-flag ablation, no connector walk
  (`../pd-disaggregation/prefill-scheduler-kv-xfer-assert.md`); a decode
  `assert … chosen_rank_list(…) does not match num_group_pulls(…)` on P TP≠D TP with an
  MLA/DSA model is one pull-count derivation skipping the MLA single-pull rule
  (`../pd-disaggregation/asymmetric-tp-pull-count.md`); and a decode
  `IndexError: list index out of range` inside `_get_kernel_block_ids`
  (`meta.remote_block_ids[kv_cache_group_id]`, `_get_kv_split_metadata` above it) is
  a KV-cache-**group-count** mismatch — the decode side indexes the prefiller's list
  with its own group ids — so diff the two `--speculative-config` blocks (a draft
  model on one side only adds a group) before any connector or transport read
  (`../pd-disaggregation/kv-group-count-mismatch.md`, reporter-confirmed #16177).
  The same one-sided-config class has a second decode-side face that never reaches the
  connector: spec decoding **and** structured output on the same launch, the decode
  EngineCore dying at the scheduler step on `AssertionError: (-1, '<req>', {'<req>':
  [-1, -1, -1]})` at `vllm/v1/structured_output/__init__.py` right after xgrammar's
  `Failed to advance FSM … for tokens -1`. The ids are seeded **placeholder** draft
  slots (count = the decode node's `num_speculative_tokens`), so the parity diff is
  row 1 *and* one launch rides with it — `recompute_scheduler_enable` off on both
  nodes separates the recorded #6421 resolution (P/D parity) from the source-verified
  seeder (`vllm_ascend/core/recompute_scheduler.py:107-111`) —
  `../pd-disaggregation/structured-output-draft-placeholder-assert.md`.
  If instead the pair **serves** and then individual *decode* ranks keep their
  requests `Running` while generation and completion counters stop — health 200, no
  traceback, one rank taking all the new work, affected ranks recovering hours
  later — the KV blocks are being freed and reused under an in-flight step on an
  `--async-scheduling` + hybrid-KV (`--no-disable-hybrid-kv-cache-manager`) serve:
  `../scheduler-admission/kv-block-lifetime-async-scheduling.md`
  (#13964 → vllm-ascend #13518), not a collective and not the throughput topic.
- **Deployment sizing / stage layout**: the question is a plan — "does this model
  fit on 2×64 GB", "the GPU docs do a 2-device stage deploy and mine does not come up
  on 2 NPUs", a bundled `platforms.<vendor>` yaml with more stages than the reporter
  has cards, "how do I pin stage 1 to one card" — **or** two serve commands that
  differ by one resource knob (`--tensor-parallel-size`, `--max-model-len`,
  `--gpu-memory-utilization`) where the smaller one comes up and the larger one OOMs
  at engine start — **or** a KV number against the reporter's own arithmetic ("27
  KB/token/card, I compute ~7.4 KB/token from the fp8 KV layout") and a context
  target ("a path to ≥512K on A2?"). Answer with per-card arithmetic plus the
  device-visibility variable, and for the KV half reconcile the cache-kind term list
  and quote the tutorial's per-SoC envelope before accepting a dtype-promotion
  hypothesis — `../deploy-sizing/index.md`, `../deploy-sizing/dsv4-kv-footprint-terms.md`.
  A report that the **whole model is loaded on every card instead of being split** is
  the same topic and is *not* answered by `npu-smi`: the per-rank weight-footprint line
  against `params × bytes ÷ TP` decides it —
  `../deploy-sizing/sharding-not-applied-full-model-per-card.md`.
- **Community-test / doc-verification task** (`[Community Test]`, a 众测任务书 row
  with 蓝区镜像 / 权重链接 / 卡数 / 交付件 测试报告, or a `[Doc]` report that a
  tutorial step does not work as written — including the `[Doc]: Feedback for <url>`
  form quoting an `…/en/latest/…` page, which may not exist at the snapshot pin at all):
  the deliverable is a *report about a document*, graded on
  completeness/usability/correctness without the reporter's hardware. The evidence is
  the pinned tree — which command no longer parses, which path is absent from the
  image, which baseline predates the image, whether the card count matches the weights
  — `../tutorial-verification/index.md`.
  When the reported defect is an **absence** instead ("the Quick Start should
  mention a mirror / link the FAQ", "the contributor docs never state a
  prerequisite"), three greps decide it — does the content exist, does any page
  reach it, do the sibling pages already carry it — and the reporter's own URL
  anchor is checked against the pin before it is quoted
  (`../tutorial-verification/missing-guidance-and-contributor-docs.md`).
- **Release known-issues bulletin**: a maintainer's notice for one release —
  `[Bug][vX.Y.Zrc1]: Known Issues for <model A>/<model B> service hang on A3`, an
  impact table (version / models / hardware / scenario), one or two tracked issue
  ids with error codes, "reply in the comments instead of opening a new issue" —
  or a user quoting that notice and asking whether their run is the same thing.
  No log, and none is owed: the deliverable is a **claim-check at the pinned
  release** — which signatures the notice pins and what they localize, what they
  cannot settle (mechanism, event ordering), whether each code is even
  model-specific, whether the reported config is inside the release's validated
  envelope — plus one ranked list per tracked issue and the reply template
  (plog `[AIC_INFO] fault kernel_name=` block, both serve commands).
  `../release-known-issues/index.md`.
- **Code-change / cleanup request**: the body asks for an *edit*, not a diagnosis —
  "remove `VLLM_ASCEND_ENABLE_<X>` before <release> final release", "delete the
  temporary switch added in PR #N", "deprecate this flag", often with nothing but two
  links and no log. First question: does the named identifier exist at the pinned ref
  (absence dates the snapshot against the PR and is itself the finding); deliverable:
  the removal sweep with `path:line` anchors plus the behaviour change a default flip
  implies — `../code-change-requests/index.md`, then `env-flag-retirement.md`.
  The delivery gates apply here too: no crash to report, so the absence verdict goes in
  the first message (recorded #1262 replay: first message text on turn 15).
  The variant that ships a **review** instead of a plan links a main-only fix and its
  backport PR ("问题在主线已修复 … 未回合进<branch>分支, pr已提，请review"): the first move is
  the predicate at both pins — the fixed form on main versus what the branch still has,
  one grep in the image's tree — then the sibling call sites and the CI test that a
  file-local change does not select. Worked checklist, anchors and the CI gap:
  `../accuracy-benchmarks/first-generate-degenerate.md`, "When the ask is to
  review the backport".

Recorded cases, per-issue index and how each session ended:
`../evidence-workspace/recorded-cases/index.md`; the load-bearing
delivery pattern is rule 1.
