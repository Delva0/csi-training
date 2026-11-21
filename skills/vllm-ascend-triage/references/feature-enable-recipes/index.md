# Feature-enable / capability questions

Scope: the report asks *how to turn a feature on* instead of what broke —
"v<ver> 上如何正确开启 F", "单独 PCP 和 DCP 均有不同的报错，能不能给出示例", "does
vllm-ascend <ver> support F for model M on this SoC". The reporter often pastes no
log at all, or describes two different errors in one sentence. Deliver a verdict
plus a recipe, not a source walk. Not this topic: a model type Transformers does
not recognise (`../startup-failures/model-type-not-recognized.md`), a capacity plan
(`../deploy-sizing/index.md`), a document audit
(`../tutorial-verification/index.md`), or corruption *after* the feature is on
(`../accuracy-benchmarks/garbled-long-context-output.md`).

Inherit the rules in SKILL.md. Three parts, in this order; part 1 is the answer.

1. **The validation verdict in the first sentence.** Is F×M×H validated at the
   reported ref — yes / no / partially, with the evidence class: documented as
   supported, measured on this hardware, covered by an e2e test, or only shown to
   work in a precedent issue at another version. "Nobody has a validated recipe
   for this combination" is a complete answer and the one the reporter is really
   asking for; a launcher script synthesized from the constraints the code
   enforces is not, and it invites the follow-up "has anyone ever actually run
   this?" (recorded #8984: the answer ranked config causes, the reporter's next
   public comment was exactly that question).
2. **Each enforced constraint with the error text it produces**, so the reporter
   maps their own two errors onto the list instead of testing the recipe blind.
   Prefer constraints the code enforces (`rg -n` the flag in `vllm_ascend/`) over
   doc prose, and say which are hard errors (an `AssertionError`, a
   `NotImplementedError`) versus a silent override.
3. **The staged experiment** — one feature axis at a time, cheapest first, each
   step with what to send back when it fails.

Establishing the verdict takes four reads, all cheap and none of them a source
walk:

- **The docs' own support statement, at the pinned ref.** Grep the exact sentence
  (`rg -n "only supported on Atlas|experimental|not supported" docs/`) — a
  statement in the pinned tree is the project's position *for that version*, so
  do not overturn it with an older issue; quote it and date it.
- **Whose hardware the docs measured.** A "Supported Scenarios" table is a claim;
  the experimental-results table below it names the cards actually measured. "All
  numbers are A3 64-card" means A2 support is untested, not rejected.
- **The e2e/test tree, which is the strongest validation signal.** `tests/e2e/**`
  and `tests/e2e/nightly/**/config/*.yaml` carry the model, the parallel flags and
  (in the nightly yaml) `npu_per_node`/`num_nodes`, which size the SoC. Grep the
  flag names, not the feature name: `rg -l
  "prefill_context_parallel_size|prefill-context-parallel-size" tests/`. A feature
  with unit tests only, or whose e2e case uses a different model family and a 1K
  context, is not validated for the reporter's model. **Search the paths as well as
  the contents** — a lane can carry its SoC only in a directory name
  (`rg --files tests/ | rg -i 310`), so a content grep for `310p|_310P` reports "no
  e2e case" while `tests/e2e/310p/test_offline_inference_310p.py` exists (recorded
  #5075 concluded exactly that, then had to be corrected). Then read the workflow
  that runs the file: at v0.12.0rc1 `.github/workflows/vllm_ascend_test_310p.yaml`
  is label-gated rather than per-PR, every case in its one existing test file is
  `enforce_eager=True` + fp16, and its 4-card leg invokes a
  `test_offline_inference_parallel_310p.py` that is absent from the commit — so the
  real coverage is single-card eager and graph mode has none.
- **Precedent issues, each with its version and its exact failure point.** A
  precedent showing a partial run ("works to N tokens, then dies") proves the
  feature comes up on that hardware at that version — it does not make it
  supported, and it does not transfer to another weight set (QuaRot/MTP variants
  included).

Rules for this topic:

1. Name the hardware from the report, never from local state: `A2` = 910B
   (8-card node), `A3` = 910_93 (16-die node). A nightly yaml's `npu_per_node: 16`
   is A3, so its pass says nothing about A2.
2. A code comment is not a launch prohibition. Before repeating "turn X off when
   F is on", check whether the comment describes a branch the code already takes
   (that is what `sfa_cp.py`'s "pcp prefill does not support mlapo" is) and
   whether the pinned feature table claims support for the pair.
3. Give the reporter the smallest discriminating run, not the full production
   command: one axis per launch, short input, `--enforce-eager` first, then graph
   mode, then speculative decoding.
4. Cap it: verdict + constraints + staged run is the deliverable inside 3 turns;
   the recorded #8984 run spent 19 turns
   and still left the capability question open.
5. A **transport-plane** question ("A3 双机 PD 分离，只能走参数面 RoCE，超平面不通、没有
   灵衢交换机，是否支持") is a capability verdict whose deciding code is the P/D rank
   table, not a feature flag: answer from
   `../pd-disaggregation/plane-selection-a3-parameter-plane.md`, and label the plane
   HCCL then selects as open (its sources are not in the snapshot). Do not read the
   connector matrix in `../pd-disaggregation/index.md` — that lane starts from a
   deployment that runs.
6. A **parallel-topology** question ("how do I enable EP together with DP",
   "why does removing `--enable-expert-parallel` make it work") is answered from
   the tree's own selector, not from the flags' prose: quote the predicate that
   picks `MC2`/`ALLTOALL`/`ALLGATHER` for the reported SoC, the arithmetic of the
   reporter's `world_size_across_dp`, and the shipped recipe that comes closest —
   `moe-expert-parallel-with-dp.md`. When the symptom is a **hang or garbage on a
   real request under DP>1** (dummy/profile run fine) rather than engine init,
   the two numbers to separate are the DP-wide pad target and the per-rank owner
   slice — `dp-moe-padding-and-owner-slice.md`.

7. A **sequence-parallel / FlashComm** question ("能不能开 `VLLM_ASCEND_ENABLE_FLASHCOMM1`",
   or the export is already in the failing command) is answered from the predicate
   that turns SP on — four doors, including a legacy env name and upstream's
   async-TP pass — plus the platform's VL/Omni guard, before any shape walk:
   `sequence-parallel-flashcomm.md`.

8. An **other-SoC** question — the model runs, and the report is "速度很慢" plus "能否提供
   量化版本 / 社区只看到 310P 的" where the model's tutorial, support-matrix row and
   published quantized exports all belong to a different SoC than the reporter's — is a
   capability verdict read off the support matrix's **tabs** (a row present only in
   `Atlas 300I DUO` and absent from `A2/A3` is the statement for that version), not a
   tuning ladder: quote the same-SoC e2e config instead of the other SoC's tutorial, and
   prune the `--additional-config` keys no code reads before ranking a lever →
   `per-soc-recipe-and-quantized-weights.md` (recorded #16600).

9. A **schedule / roadmap** question — "310系列什么时候支持图模式和量化", "when will
   F land for M on H", often with a performance complaint attached — has no answer in
   the code, and the project publishes no dates. Give the *staging evidence* instead,
   in this order: (a) the RFC or roadmap issue that owns the area, found by token in
   the history corpus (`scripts/history-titles.sh -i '300I|310P'`; #1309
   `[RFC]: Support Altlas 300I series` holds an empty `Stage2: Performance
   improvement` heading, which is the citable "no committed scope"), (b) the earliest
   release whose docs or code actually carry the capability — anchor it with
   `path:line`, and if that is a later release than the reporter's, say the interval
   between two pinned releases is unanswerable on a `one_commit_git` snapshot, (c) what
   is available *today* at their ref, which is the only actionable half. Worked for
   #5075 (300I Pro, v0.10.0rc1): quantization answered "already, with a 310-targeted
   checkpoint", graph mode "not supported at the pinned ref and no committed date" —
   `../soc-platform-paths/atlas-300i-310p.md`.

Pages:

- `per-soc-recipe-and-quantized-weights.md` — the model's only validated recipe and
  published weights are on another SoC (#16600: 910B2 + Qwen3.5-9B, "能否提供量化版本"):
  reading `support_matrix/supported_models.md` by tab (`A2/A3` at `:142` vs `Atlas 300I
  DUO` at `:156`, the `Qwen3.5-Dense` row at `:161`), the tutorial's own hardware sentence,
  the same-SoC e2e launch config (`test_qwen3_5_0_8b.py:45,55-60` bf16 + `method: mtp`)
  against the 310P one (`float16` + `mamba_ssm_cache_dtype` + `qwen3_5_mtp`), the
  test-path→SoC routing, the stale `enable_weight_nz_layout` key (code reads
  `weight_nz_mode`), and why a `-310P` W8A8 export is not portable.
 — FlashComm1/SP: the `enable_sp()` predicate and its
  `tp_size > 1` / EP asserts, the `ValueError` guard for VL and Omni-thinker models
  (`platform.py:394-403`, `is_vl_model` counting `thinker_config`), and why the
  shape mismatch `(sN//tp)` vs `sM` inside `profile_run` is arithmetic: SP keeps the
  row-parallel output reduce-scattered (`ops/linear_op.py:386-396`) while a per-layer
  deepstack/full-length auxiliary tensor does not shrink. Worked for #6261.
- `dp-moe-padding-and-owner-slice.md` — DP>1 MoE arithmetic behind a hang or a
  garbage tensor after an all-gather: `set_ascend_forward_context`'s pad target
  (`ascend_forward_context.py:91-98`, DP-wide only while `dp_metadata` is present,
  phase-dependent because prefill and decode carry different vectors), the three
  consumers and the `reduce_scatter_tensor` owner-slice convention
  (`ops/fused_moe.py:1382-1395,1446-1452`), the `get_fused_moe_state` table that
  decides whether the DP `all_gather` runs at all (EP at `ep_size < 16` is
  `All2All`, not `AllGather`), and the rank-parity checks. Worked for #3107
  (v0.9.1rc2, Pangu Pro MoE, DP8×TP1 with EP).
- `moe-expert-parallel-with-dp.md` — EP **with** DP: `_select_moe_comm_method`'s
  per-batch table for A2 (`MC2` only when `num_tokens <= capacity` **and**
  `world_size_across_dp/pp >= 16`) and A3, `set_mc2_tokens_capacity`'s arithmetic
  (why one server runs MC2 in decode and ALLGATHER in prefill), the correction that
  `PrepareAndFinalizeWithMC2` inherits the All2All `finalize` whose `dist.all_gather`
  is the pasted crash site, the refuted uneven-`tensor_split` theory, and the only
  EP+DP recipes the tree ships. Worked for #5295 (v0.12.0rc1, GLM-4.5-W8A8,
  DP2×TP8 on one 910B node), whose recorded session shipped no answer.
- `context-parallel-pcp-dcp.md` — PCP/DCP (context parallel), worked for #8984
  (DeepSeek-V3.2 QuaRot on A2, v0.19.1rc1): card arithmetic, the DCP divisibility
  and `cp_kv_cache_interleave_size` gates, the `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`
  requirement of the MoE path, what the pinned tree actually validates for A2, and
  the PCP-defect precedent family (#8578/#8579/#8580/#7375).
