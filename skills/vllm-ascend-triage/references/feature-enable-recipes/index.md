# Feature-enable / capability questions

Scope: the report asks *how to turn a feature on* instead of what broke —
"v<ver> 上如何正确开启 F", "单独 PCP 和 DCP 均有不同的报错，能不能给出示例", "does
vllm-ascend <ver> support F for model M on this SoC". The reporter often pastes no
log at all, or describes two different errors in one sentence. Deliver a verdict
plus a recipe, not a source walk. **Also this topic:** "<model> 离线推理报错 / 跑不通" on a named release with only
screenshots — the release's own claim table decides it in one grep, and a `❌` row
with a tracking issue *is* the verdict (`unsupported-model-row.md`: the matrix path
moved between refs, and the reporter will come back with a second batch of
*different* errors).

Not this topic: a model type Transformers does
not recognise (`../startup-failures/model-type-not-recognized.md`), a capacity plan
(`../deploy-sizing/index.md`), a document audit
(`../tutorial-verification/index.md`), or corruption *after* the feature is on
(`../accuracy-benchmarks/garbled-long-context-output.md`).

**Also this topic, and it is a crash rather than a capability question:** the feature
being switched on is the **legacy MC2 env knob** — the launch exports
`VLLM_ENABLE_MC2=1` and the worker dies with a bare
`TypeError: object of type 'NoneType' has no len()` at
`vllm_ascend/quantization/w8a8_dynamic.py:156` in `fused_experts_with_mc2`, on the
first **decode** step of a DeepSeek-V2/R1 W8A8 serve whose expert-parallel group is 1
(no `--enable-expert-parallel`). That is a two-rung answer — unset the env var, then
EP ≥ 2 — and the page also dates the install: the knob has zero hits in the tree by
v0.20.2rc1, so the selector-era MC2 pages (AIV ablation, `mc2_tokens_capacity`, the
EP-off perf page) must not be quoted at such a report:
`mc2-env-knob-without-expert-parallel.md`.

Inherit the rules in SKILL.md. Three parts, in this order; part 1 is the answer.

1. **The validation verdict in the first sentence.** Is F×M×H validated at the
   reported ref — yes / no / partially, with the evidence class: documented as
   supported, measured on this hardware, covered by an e2e test, or only shown to
   work in a precedent issue at another version. "Nobody has a validated recipe
   for this combination" is a complete answer and the one the reporter is really
   asking for; a launcher script synthesized from the constraints the code
   enforces is not, and it invites the follow-up "has anyone ever actually run
   this?" (recorded #8984: the answer ranked config causes, the reporter's next
   public comment was exactly that question — and a day after a later,
   verdict-first answer the same reporter asked 「有人在 A2 机器上成功跑通过
   DeepSeek-V3.2-w8a8-mtp-QuaRot 开启 PCP 特性吗?」). So the verdict part also
   carries the **precedent census**, produced in the same step by
   `scripts/history-titles.sh -i '<feature>|<alias>|...'`: how many issues mention
   the feature, how they split (RFC vs failure), and whether any reports success
   for *this* weight set and SoC — plus the coverage limit of the snapshot
   (titles/bodies, no comments or PRs), so the claim reads "nobody in the
   searchable corpus reports success" rather than "impossible". Read `-n` bodies
   only for the hits you quote. Any launcher you still hand back must then fit the
   reporter's cards per the model tutorial's own node requirement for that weight
   set (`context-parallel-pcp-dcp.md` §5 is the worked case: the community w8a8 set
   needs 2×A2, so a "single A2 node" rung is only real for the w4a8 set).
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
- **A "the code only supports <other SoC>" premise is a guard-polarity question, not a
  hardware one.** `is_310p()` appears both ways in this tree — `if not is_310p():` runs
  the cast on 910 and skips 310P — so the reporter's own reading can be inverted, and the
  per-path inventory (quantized Linear/MoE, unquantized MoE, unquantized dense Linear)
  plus the versioned switch decide it. Weight ND→NZ layout is the worked case:
  `weight-nz-layout-by-soc.md` (#3859 at v0.11.0rc0, where no `weight_nz_mode` exists and
  only the dense unquantized cast is 310P-gated).
- **For "F is simply absent from the plugin", the roadmap question is answered by
  the status ledger, not by release notes.** `docs/source/user_guide/support_matrix/
  supported_features.md` is the project's own feature table: its legend (`:30-34`
  at v0.11.0rc2/rc3) defines 🟢 Functional / 🔵 Experimental / 🚧 WIP /
  🟡 Planned / 🔴 NO plan, and the `Next Step` column carries the tracking issue or
  PR (LoRA at `:11` → vllm-ascend#396, #893). Quote the row for the feature, or
  state that there is **no row** — a missing row is untracked in both directions,
  so no plan and no refusal exists, and a dated commitment needs a maintainer or a
  new tracking issue. Say that instead of "search the release notes of a later
  version": recorded #4505 answered `prompt_embeds` correctly as absent at the pin
  (0 hits in `vllm_ascend/`) with a port plan, and the reporter's public follow-up
  was 「这个我们后续有计划支持吗」 — the table, not a future tree, is the answer to
  the follow-up, and it is one read.
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
   slice — `dp-moe-padding-and-owner-slice.md`. When the same DP>1 MoE report has
   **no traceback at all** (a hang plus a plog OOM) and the launch carries `etp` with
   EP off, that page's *DP output reduction* section is the lane: `ep_size` is 1 and
   the reduction's call form is the owner, not the memory budget.

7. A **sequence-parallel / FlashComm** question ("能不能开 `VLLM_ASCEND_ENABLE_FLASHCOMM1`",
   or the export is already in the failing command) is answered from the predicate
   that turns SP on — four doors, including a legacy env name and upstream's
   async-TP pass — plus the platform's VL/Omni guard **and the remedy that guard's own
   message names at the reporter's ref** (unset-only on the v0.14 family; the Inductor SP
   pass on the v0.15 family, whose support matrix makes it the VL+Dense route), before any
   shape walk: `sequence-parallel-flashcomm.md`. When **nothing crashes** and the
   complaint is 「开启后无效」 (the profiler still shows allreduce, #6862), that page's
   §*other entry shape* is the whole answer: the gate is per forward (`num_tokens > 1000`
   for a dense target), so a decode-window profile is allreduce by construction and the
   deciding launch is a profile of one >1000-token schedule — with the competing
   explanation (`_get_row_parallel_op` returns before `enable_sp()` for finegrained-TP,
   matmul-allreduce or FlashComm2) stated in the same reply. When nothing crashes and the
   complaint is an **accepted-token collapse** after the env is enabled (TTFT improves as
   promised, acceptance halves, no error, MTP drafter), that page's §*FC1 with a MoE MTP
   drafter* refutes the usual `pad_size` / double-shard story from four reads at v0.18.0 and
   asks for the per-position rate line. On the v0.19.x hybrid families
   (Qwen3.5/Qwen3-Next, `expected src to have a size equal to the slice of self`) that page's
   §*v0.19.x hybrid layers* separates layer 0's SP-sized buffer from its two writers and its two
   enable gates — use it instead of grepping `copy_`.
   **FlashComm2 is a different switch with a different failure**: when
   `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` > 0 is combined with
   `"layer_sharding": ["o_proj"]` and a worker dies while loading weights with
   `AssertionError: layer_idx must be consecutive` (`ops/layer_shard_linear.py`), one
   `o_proj` linear has been registered into the same shard *series* twice — once by the
   Flashcomm2 row-parallel op, once by the attention impl's `layer_sharding` kwargs — and
   the two one-line ablations plus the value/`recompute_scheduler_enable` answers are
   `flashcomm2-o-shard-layer-sharding.md`. Do not read it as a bad
   `FLASHCOMM2_PARALLEL_SIZE` value or a PD/transport fault.
   **FlashComm2's other failure arrives at request time, not while loading**: a worker
   dying with `AssertionError: Batch_size(N) must be divisible by chunk_num(M)` at
   `ops/linear_op.py otp_maybe_quant_comm` on an eagle/eagle3 launch is the same o_proj
   all-to-all applied to the **draft** model, whose row count is normalised only by a
   `pad_size` written once per forward context — never by the flag's value, which cannot
   move `chunk_num` off the global TP size (`flashcomm2-draft-pad-token-divisibility.md`,
   recorded #7809, reporter-confirmed).

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
   The same question usually arrives as a **follow-up to a root-cause answer**
   ("感谢回复，请问 <family> MTP 在 <SoC> 上的支持计划是什么样的？"), so put its paragraph
   in the first answer and keep the three levels apart: the base model family at that
   SoC (the release-note line that introduced it as *experimental*, the support-matrix
   row, and the SoC fallback `TODO` that still bounds it), whether the **feature's**
   branch exists at all (one grep, e.g. `use_gemma4_mtp`), and whether anything covers
   the combination (tests, tutorials). Then the prerequisites in port order and the
   grep whose result would change the verdict — never a date. A family documented as
   experimentally supported at level 1 with no level-2 branch is the commonest real
   answer; writing "the family is unsupported" collapses the two and is wrong
   (`../spec-decode-mtp/family-proposer-parity.md`, worked #16980 for gemma4 MTP on A2).

10. A **graph-mode enable** question or crash on a DeepSeek/MLA model — "如何开启 torchair 图模式",
   `additional_config={"torchair_graph_config": …}` present, and the first request dying with
   `UnboundLocalError: local variable 'decode_hs_or_q_c'` in `vllm_ascend/attention/mla_v1.py`
   (or, after a hand patch, `decode_k_pe` / `TypeError: 'NoneType' object is not subscriptable`) —
   is answered from the pinned tree's own two statements before any scheduler reasoning: torchair
   does not work with chunked prefill at that ref, and `ascend_scheduler_config.enabled: true`
   (exact spelling; unknown `additional_config` keys are silently ignored) is the half that turns
   chunked prefill off. Do not re-derive the mixing step from prompt sizes — the paste has no
   per-step batch composition → `torchair-mla-graph-and-chunked-prefill.md`.

11. A graph-mode serve that dies **at layer construction** on
   `AssertionError: The allowed number of queries per kv … only support {32, 64, 128}`
   (or the kernel's `EZ9999 … numHeads / numKvHeads = N, MLA only support {32, 64, 128}`)
   is arithmetic, and the message already states the rule: MLA hardcodes `num_kv_heads=1`,
   so the ratio is heads **per rank**, i.e. `num_heads // tp_size`. Answer with the
   TP that passes for that head count (128-head DS-V3/R1: TP ∈ {1,2,4}), the
   `--enforce-eager` + delete-the-key lever, and the ACL-graph-is-closed fact —
   never with a kernel or checkpoint theory, and never believing the title over the
   serve command → `torchair-mla-head-count-tp-gate.md`.

12. A **DS-V4.x Engram / host-offload** question or failure — "如何开启 engram", an
   `enable_engram` / `engram_storage` / `enable_engram_ple_offload` key in
   `--additional-config`, the host-memory pair `dp_shared_memory` +
   `enable_cpu_binding`, a load-time `ValueError: layers.N.engram.embed.weight:
   expected BF16 source for bf16`, an A3 start that hangs **after** warm-up/graph
   capture, or `507011` / invalid-GM at concurrency only while PLE offload is on — is
   answered from four corpus rows, not from a source walk (no local pin carries the
   engram code): the reporter-confirmed missing-key fix, the NUMA/`migratepages`
   startup block, the PLE-offload A/B with its measured memory gain, and the field
   launcher for 4×8 A2 colocated at 262144 → `engram-offload.md`. Two hand-offs: its
   row 1 is a `[Doc]` defect on the DS-V4.1-Flash tutorial page, and the row-2 log
   line is *also* `../startup-failures/mm-encoder-profile-stall.md`'s signature, so
   discriminate on the launch (engram keys, phase past graph capture), not the line.

Pages:

- `weight-nz-layout-by-soc.md` — "910B3 是否支持权重 ND→NZ 转换提升 matmul 性能？0.11.0
  好像只支持 310P" (#3859): the `is_310p()` polarity rule, the v0.11.0rc0 per-path
  inventory (`w8a8.py:139`, `w8a8_dynamic.py:104`, `w4a8_dynamic.py:396`,
  `common_fused_moe.py:128`/`fused_moe.py:81` `not is_310p()`, and the single
  `is_310p()` dense cast at `model_runner_v1.py:2633-2643` with `ACL_FORMAT` NZ only on
  310P; torchair's `converting_weight_acl_format` is 310P-only, so graph mode converts
  nothing on 910), the version boundary for `weight_nz_mode`/`VLLM_ASCEND_ENABLE_NZ` —
  and the **rc** as its own boundary, since v0.11.0rc0 has no gate function and no dense
  hook while `is_enable_nz` exists by v0.11.0rc3 — plus the five-line
  `npu_format_cast(w, 29)` probe before any patch.
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
  deepstack/full-length auxiliary tensor does not shrink. Worked for #6261; §*v0.19.x hybrid
  layers* for #8990 (Qwen3.5-27B-w8a8 + `qwen3_5_mtp` + FC1 at TP2): layer 0's `ceil(tokens/tp)`
  buffer (`patch_qwen3_5.py:91-96`), its two writers (`:74` `o_proj` vs `ops/gdn.py:136` `out_proj`,
  whose `[:num_tokens]` clamps so the paste shows the buffer's rows), the build-time
  `sp_row_prefixes` decision, the `num_tokens > 1000` dense gate against the always-on MoE branch,
  the already-sharded drafter inputs, and the vllm-pin coverage gap (`v0.13.0` vs the paired
  `v0.19.1`).
- `dp-moe-padding-and-owner-slice.md` — DP>1 MoE arithmetic behind a hang or a
  garbage tensor after an all-gather: `set_ascend_forward_context`'s pad target
  (`ascend_forward_context.py:91-98`, DP-wide only while `dp_metadata` is present,
  phase-dependent because prefill and decode carry different vectors), the three
  consumers and the `reduce_scatter_tensor` owner-slice convention
  (`ops/fused_moe.py:1382-1395,1446-1452`), the `get_fused_moe_state` table that
  decides whether the DP `all_gather` runs at all (EP at `ep_size < 16` is
  `All2All`, not `AllGather`), and the rank-parity checks. Worked for #3107
  (v0.9.1rc2, Pangu Pro MoE, DP8×TP1 with EP). **§*The DP output reduction* is the
  no-traceback form of the same lane**: `ep_size == 1` with DP > 1 (`etp` in
  `additional_config`, `--enable-expert-parallel` absent) reported as
  「推理卡死 + plog out of memory」 — the exit call's *form* is the defect
  (`dist._functional_collectives.reduce_scatter_tensor` over a `dispose_tensor`-freed
  input versus `get_dp_group().reduce_scatter`, reporter-confirmed #2069), so read it
  before any per-card memory arithmetic.
- `moe-expert-parallel-with-dp.md` — EP **with** DP: `_select_moe_comm_method`'s
  per-batch table for A2 (`MC2` only when `num_tokens <= capacity` **and**
  `world_size_across_dp/pp >= 16`) and A3, `set_mc2_tokens_capacity`'s arithmetic
  (why one server runs MC2 in decode and ALLGATHER in prefill), the correction that
  `PrepareAndFinalizeWithMC2` inherits the All2All `finalize` whose `dist.all_gather`
  is the pasted crash site, the refuted uneven-`tensor_split` theory, and the only
  EP+DP recipes the tree ships. Worked for #5295 (v0.12.0rc1, GLM-4.5-W8A8,
  DP2×TP8 on one 910B node), whose recorded session shipped no answer. Two A2
  dispatch **tiling refusals** hang off it: §Step 2b the per-card expert cap
  (`moeExpertNum … no more than 24`) and §Step 2c the token dim (`batchsize is
  invalid` with `CheckShapeAndSetTiling`, #2858), where bs is derived
  (`padded_num_tokens / tp`) rather than the reported batch.
- `context-parallel-pcp-dcp.md` — PCP/DCP (context parallel), worked for #8984
  (DeepSeek-V3.2 QuaRot on A2, v0.19.1rc1): card arithmetic, the DCP divisibility
  and `cp_kv_cache_interleave_size` gates, the `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL`
  requirement of the MoE path, what the pinned tree actually validates for A2, and
  the PCP-defect precedent family (#8578/#8579/#8580/#7375), and **§4b** for the
  accuracy-degradation shape: PCP on with `--no-enable-expert-parallel`, where the
  answer is a coverage statement (every CP accuracy test in the tree is EP-on) plus
  one flag, not a MoE source walk.
- `torchair-mla-graph-and-chunked-prefill.md` — torchair graph mode on DeepSeek/MLA
  (v0.9.1rc1 `c30ddb8`), worked for #1369, **plus §*chance-level accuracy variant* for #1553
  (the same three-flag cell with no traceback and an MMLU at 4-way chance, where the pinned
  #1350 prefix-cache row is retired by the reporter's own APC-off re-run: rank graph capture,
  then the scheduler fork, then the `-Pruning` checkpoint/harness)**: the `decode_hs_or_q_c` UnboundLocalError at
  `mla_v1.py:1099` explained by the `running_in_graph` binder structure (`:1040,1065-1075`),
  the doc/CI comment "only work without chunked-prefill" plus the `ascend_scheduler_config`
  pair (`graph_mode.md:50-51`, `test_torchair_graph_mode.py:46-60`,
  `schedule_config.py:26,43`), the no-startup-guard fact (`ascend_config.py:131-147`), the
  silent-typo / cached-config check (`ascend_config.py:34-37,92-101`) and the guard must key
  on `running_in_graph`, not `torchair_graph_enabled`.
- `torchair-mla-head-count-tp-gate.md` — the `queries per kv … only support
  {32, 64, 128}` gate at layer construction, worked for #1451 (v0.9.1rc1,
  DeepSeek-V3 TP8×PP4 ray + torchair, title naming V2-Lite) and for #1458's
  `ACL Graph does not support deepseek` refusal at TP16, where the refusal's own
  torchair advice is already closed by the same division (128/16 = 8): the
  heads-per-rank arithmetic (16 at TP8), the three sites at that pin, the eager×graph mutex and the
  ACL-graph-deepseek refusal, the single graph e2e row (TP=4, mp, no PP), the
  FAQ-only documentation of the rule, and the fact that the Python assert is
  **deleted by v0.10.0rc1** while the FAQ and the kernel check remain.
- `engram-offload.md` — DS-V4.x Engram host offload as four recorded rows: #17236's
  load-time `layers.N.engram.embed.weight: expected BF16 source for bf16`, fixed
  reporter-confirmed by adding `{"enable_engram":true,"engram_storage":"int8"}` to a
  tutorial serve line that omits it (a `[Doc]` defect, no environment block in the
  issue); #17286's A3 block after warm-up and graph capture, `migratepages` helpers
  alive >15 min on a ~206 GiB HOST_UVA table registered under `dp_shared_memory` +
  `enable_cpu_binding` (workaround: the key off); #16828's controlled A/B for
  `enable_engram_ple_offload=true` — ≈12.87 GiB/rank less weight memory and ≈1.97× KV,
  then `507011` at concurrency 24-32, with the PR-force-push caveat; and #16877's
  4×8 A2 colocated launcher (`--max-model-len 262144`, dspark 5,
  `tokenizer-mode`/`reasoning-parser deepseek_v41`) whose own defect is the output
  shape, not the offload.
