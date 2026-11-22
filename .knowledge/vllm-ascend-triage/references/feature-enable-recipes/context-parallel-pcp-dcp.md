# PCP / DCP (context parallel): enabling it, and what is actually validated

Worked for #8984 — vLLM-Ascend v0.19.1rc1 @ `da421af`, vLLM v0.19.1 @ `b1388b1`,
report: "在 DS-V3.2 量化版本上（w4a8-mtp-QuaRot 或 w8a8-mtp-QuaRot），单独 PCP 和
DCP 均有不同的报错，能不能给出 A2 上能跑通的示例". No log was pasted, A2 = 910B.

## 1. The verdict this case needs (state it first)

At v0.19.1rc1 **no A2 recipe for DeepSeek-V3.2 + PCP is validated**:

- The pinned tutorial states it outright:
  `docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:6`
  "Context parallel feature currently is only supported on Atlas A3 device, and
  will be supported on Atlas A2 in the future." Quote it, do not overturn it.
- Every measured number in the feature guide is A3:
  `docs/source/user_guide/feature_guide/context_parallel.md:86-96` (DeepSeek-R1-W8A8
  and Qwen3-235B on 64×A3-64G). No A2 row, and no DeepSeek-V3.2 row at all.
- No e2e test exercises context parallel on DeepSeek-V3.2. The CP e2e set is
  `tests/e2e/multicard/4-cards/long_sequence/{test_basic,test_accuracy,test_mtp,
  test_prefix_caching_cp,test_chunked_prefill_cp}.py`, all TP2/PCP2/DCP2 with
  `deepseek-ai/DeepSeek-V2-Lite-Chat` and `Qwen3-30B-A3B-W8A8` at
  `max_model_len=1024`; the long-seq CP job that does run long context is
  `tests/e2e/nightly/multi_node/config/DeepSeek-R1-W8A8-longseq.yaml` and
  `Qwen3-235B-W8A8-longseq.yaml`, whose `npu_per_node: 16, num_nodes: 2` is an A3
  (910_93) pair.
- The nearest human evidence is precedent-only, partial, and SoC-specific — spell the
  SoC out, because it is the whole answer:
  - #8580 (v0.18.0rc1, **2×A2 / 910B2** per its own dump, TP8 / DP1 / PCP2 / DCP8,
    `--no-enable-expert-parallel --cp-kv-cache-interleave-size 128`,
    FULL_DECODE_ONLY) **serves short prompts and dies around 32K input tokens on
    A2** (~64K on A3) with `AclrtSynchronizeStreamWithTimeout ... 507015` at
    `sfa_cp.py` `_execute_indexer_select`. Plain `DeepSeek-V3.2-W8A8` weights.
  - #8579 (v0.18.0rc1, **2×A2 / 910B2** in its `npu-smi` block, TP8 / DP1 / PCP2 /
    DCP1, `--no-enable-expert-parallel`) comes up and **serves with degraded
    accuracy**: an A2 PCP2 run that starts is still not usable.
  - #8578 (same A2 shape, PCP2 / DCP8 with EP on and MTP) dies on *short* inputs with
    `AttributeError: 'PCPManager' object has no attribute 'mtp_slot_pad'`.
  - #7375 (v0.13.0, **A3**, `DeepSeek-V3.1-Terminus-w4a8-mtp-QuaRot`, TP8 / DP1 /
    PCP2) fails with a tensor mismatch. It is the corpus's **only**
    `*-mtp-QuaRot` × PCP report, it is on the *supported* SoC, and it is a failure.
  - #7305 (v0.13.0, **A3**, Qwen3-235B, PCP2 / DCP2) is the corpus's only *working*
    CP datapoint, and it is filed as a performance regression.

  So: no `*-mtp-QuaRot` weight set has a success report with PCP on any SoC, and on
  A2 the plain W8A8 set comes up under PCP2 only with the defect each report carries.

The honest answer is therefore three sentences: what the docs claim (A3-only), what
the precedents show (A2 runs plain W8A8 under PCP2 and then fails by length, accuracy
or `mtp_slot_pad`, v0.18.0rc1, none of them a QuaRot set), and that DCP alone has no
separate A2 validation either. Then the constraints and the staged experiment.

## 1b. The follow-up the verdict must pre-empt

Recorded #8984, second round: the answer **did** open with "no validated A2 recipe"
and then handed two launchers; one day later the same reporter's next public comment
was 「有人在 A2 机器上成功跑通过 DeepSeek-V3.2-w8a8-mtp-QuaRot 开启 PCP 特性吗?」
(`#issuecomment-4412075282`, 2026-05-09). "Not validated" does not answer "has anyone
ever run it", so the first message must carry the **corpus sentence** too, and the
search that produces it belongs with the verdict (the recorded run ran it at turn 40
of 45, after the source walk, so it could no longer shape the answer):

```bash
bash <skill_root>/scripts/history-titles.sh -i 'PCP|DCP|context.parallel|长序列'
```

> 10 hits at this snapshot: 3 RFCs (#4310, #4312, #6764) and 7 failure reports. No
> success report for any `*-mtp-QuaRot` weight set on any SoC; the A2 reports are all
> plain `DeepSeek-V3.2-W8A8` with a defect (#8578 / #8579 / #8580); the only working
> CP datapoint is A3 Qwen3-235B (#7305). Comments and PRs are outside this snapshot,
> so "nobody in the searchable corpus reports success" is the strongest claim the
> tooling supports — say it in that form, not "impossible".

Three failure modes to avoid when writing that sentence: do not promote a *different*
weight set's A2 datapoint to the reporter's (`#8580` is `DeepSeek-V3.2-W8A8`, never
`-mtp-QuaRot`); do not let a launcher you still hand back contradict your own
card arithmetic or the SFA `dcp == tp` rule (§2, §5 rung 1); and **phrase it as the
answer, not as a defence of the answer**. Recorded #8984's replay did carry the corpus
sentence in its first message and still shipped it as 「语料检索（这一步必须和结论一起给，
否则你会紧接着问"那有人在 A2 上跑通过吗"）」 — this page's own instruction plus a
prediction of the reporter's next question, inside the text he reads. Ship the finding
in his terms (「没有任何人在 A2 上成功跑通这个权重集；A2 上最近的三条都带缺陷 …」) and
keep the clause about *why* it belongs in the first message for the analyst
(`../evidence-workspace/delivery-discipline.md`, the gates).

## 2. Constraints the code enforces (map each to its error text)

Card arithmetic — PCP costs cards, DCP does not:
`vllm_ascend/distributed/parallel_state.py:41-52` lays ranks out as
`torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)`, so
`world = external_dp × dp × pp × pcp × tp`; DCP never appears there because it
splits KV inside the TP group. `docs/source/user_guide/feature_guide/
context_parallel.md:73` says the same for the simple case ("total world size is
`tensor_parallel_size` * `prefill_context_parallel_size`"). A reporter who keeps
the tutorial's `--data-parallel-size 2 --tensor-parallel-size 8` and adds
`--prefill-context-parallel-size 2` has asked for 32 cards on a 16-card pair — the
most common "PCP alone errors" shape, and it fails as a device-count/world-size
mismatch, not as a PCP defect.

DCP divisibility (`context_parallel.md:70-78`): MLA models (DeepSeek) need
`tp >= dcp` and `tp % dcp == 0`; GQA models need the same on
`tp // num_key_value_heads`.

An **SFA** weight set (DeepSeek-V3.2, GLM-5.x — `hf_text_config.index_topk` present)
carries a second, tighter DCP rule: `dcp == tp`, not merely `tp % dcp == 0`. At a pin
that has the replicated-indexer flag (v0.26.0rc1 `f2f74a1`,
`vllm_ascend/utils.py:122-129` and `vllm_ascend/platform.py:723-731`)
`enable_sfa_dcp_replicated_indexer` is on as soon as
`decode_context_parallel_size > 1`, and the platform then refuses anything else with
`AssertionError("DCP for SFA is only supported when dcp_size(...) == tp_size(...)")`;
rule, its pinned rows and the A5 refusal are
`../dsa-sparse-attention/dcp-c8-accuracy-garble.md` Rule 1. **A reported pin that
lacks that assert is not permission**: at v0.19.1rc1
`rg -n 'replicated_indexer|sfa_dcp' vllm_ascend/` returns nothing, so
`--decode-context-parallel-size 2 --tensor-parallel-size 8` is *accepted* and runs an
unvalidated split whose family failure mode is silent (garbled tokens / accuracy
loss, no exception). Hand back `dcp == tp` instead — for the tutorial's A2 row
(DP2 / TP8) that is `--decode-context-parallel-size 8`, which is also the corpus's
own A2 value (#8580 is TP8 / PCP2 / **DCP8**). A reporter already running `dcp != tp`
gets one rung: the same command with `--decode-context-parallel-size 1`, outputs
compared, before any fix is believed.

KV transfer + CP (`vllm_ascend/platform.py:513-522`): when
`kv_transfer_config is not None` (PD disaggregation, KV pool) and
`dcp*pcp > 1`, `cp_kv_cache_interleave_size` **must equal** `block_size` or the
platform raises `AssertionError: cp_kv_cache_interleave_size(...) and
block_size(...) needs to be equal if use pcp or dcp > 1 in P/D disaggregate and
kv pool scenario.` Launch both sides with `--cp-kv-cache-interleave-size 128
--block-size 128`.

Sparse attention (DeepSeek-V3.2 is SFA: `hf_text_config.index_topk` exists) —
`platform.py:525-537` **silently overrides** `cp_kv_cache_interleave_size` to
`block_size` with a `warning_once`. The override sits *after* the PD assertion, so
in a PD launch the assertion still fires first; the override only helps a
co-located run. The SFA CP backend itself exists:
`vllm_ascend/attention/context_parallel/sfa_cp.py`.

`VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1` — required by the MoE path, not by
attention: `vllm_ascend/ops/fused_moe/prepare_finalize.py:401` and `:473` both gate
on `prefill_context_parallel_enable() and pcp_size > 1`, so without the env var the
attention is PCP-sharded while MoE all-gathers/reduce-scatters over the wrong group
(garbage or a hang, not a clean error). The documented multi-node example exports
it; the in-tree example does too
(`examples/offline_inference_npu_long_seq.py:8`), while the feature guide's online
example omits it. On a co-located (non-PD) single-node run the bf16 path works
without it.

MLAPO — **do not** turn the code comment into "unset `VLLM_ASCEND_ENABLE_MLAPO`".
`sfa_cp.py:37-38` says "In sfa, pcp prefill does not support mlapo" but the flag is
then used as an internal branch (`:136` decode slot-mapping, `:576`
`if self.pcp_size == 1 or self.enable_mlapo: return k`), and the pinned feature
guide's supported-features table lists MLAPO ✅ for both PCP and DCP
(`context_parallel.md:22-27`). Unsetting it is an experiment row, not a rule.

The v2 model runner has no CP at all: `vllm_ascend/worker/v2/model_runner.py:68-70`
raises `NotImplementedError("Context parallelism is not supported by Ascend NPU
model runner v2.")` when either size is > 1 — a one-line, unambiguous error to
match first.

FlashComm1 + PCP: `vllm_ascend/ascend_config.py:91-104` rounds
`max_num_batched_tokens` up to a multiple of `tp × pcp` and logs "When using
FLASHCOMM1, the max_num_batched_tokens should be divisible by tp_size * pcp_size
(%s). It has been adjusted to %s." — expected, not an error.

## 3. PD disaggregation: DCP is a *matched* launch parameter, not per-node tuning

Reporter-confirmed on **#12677** (v0.23.0rc1, GLM-5.2 W8A8C8, A2, PD
disaggregation, MemCache ("mmc") KV pool, `tp=2`/`dcp=2`, "推理时 P 实例报错";
the paste held only the APIServer's `EngineDeadError: EngineCore encountered an
issue. See stack trace (above)`). The reporter's own later comment on that issue:
"配置存在问题，使用 1M 配置 dcp=8，PD 需要同并行 dcp，可以 mmc 池化使用" — i.e. the
deployment was misconfigured, the 1M recipe's `dcp=8` is the value, **both sides
must launch the same dcp**, and the MemCache pool works with that config. The
engine had reached `Application startup complete`, so no startup check was
involved and the source was never the owner of the symptom.

Why only the two serve commands could have settled it (source-verified at
v0.23.0rc1 `f4a08bd`, in case the same shape arrives without the reporter's
answer):

- The prefill side advertises its own size to the decode side in the handshake
  (`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1863`
  `remote_dcp_size=self.dcp_size`); the decode side stores it per request
  (`:1440`).
- The first request that needs a transfer builds the block/port mapping and
  asserts divisibility there:
  `assert (meta.remote_pcp_size * meta.remote_dcp_size) % (self.pcp_size *
  self.dcp_size) == 0` (`mooncake_connector.py:2663-2664`, called from `:2843`;
  the module-level twin is `distributed/kv_transfer/utils/utils.py:88-104`). So a
  P/D pair with dcp 2 on one side and 8 on the other is `2 % 8 != 0` and raises
  inside EngineCore **on the first transfer** — after serving started, which is
  the reported shape. There is no launch-time check for this pair. The failure
  text is not confirmed for #12677 (no traceback was pasted): the site and the
  config verdict are, the exact message is a hypothesis.
- Two related gates raise at *startup* instead, so a deployment that reached
  serving already satisfies them: sparse-attention models need `dcp == tp`
  (`vllm_ascend/platform.py:716-724`, predicate
  `vllm_ascend/utils.py:114-136` — `index_topk` present, `compress_ratios`
  absent), and KV transfer with `pcp*dcp > 1` needs
  `cp_kv_cache_interleave_size == block_size` (`platform.py:731-740`).

The tree's own 1M table is symmetric and says so
(`docs/source/tutorials/models/GLM5.2.md:1218-1220`): single-node `DCP16`,
dual-node `DCP8`, 1P1D PD `Prefill DP4 TP8 PCP1 DCP8, Decode DP4 TP8 PCP1 DCP8`,
all with `--cp-kv-cache-interleave-size 128` (`:1306`, `:1362`, `:1433`).

So for a PD report with DCP on and the first request killing an EngineCore:
**put the two serve commands side by side before any source walk** — compare
`--tensor-parallel-size`, `--decode-context-parallel-size`,
`--prefill-context-parallel-size`, `--cp-kv-cache-interleave-size`,
`--block-size`. A dcp that differs between the sides, or a recipe value edited on
one side only (1M `dcp=8` → `2`), *is* the answer; the KV pool is not the
suspect, and `dcp` equal to the recipe on **both** sides is rung 1 of the
experiment ladder. Later rungs: `--max-model-len` down to the value the reporter
needs, then `--decode-context-parallel-size 1` on both sides, then the
connector/pool pages (`../pd-disaggregation/index.md`).

## 4. Precedent family for DeepSeek-V3.2 + PCP (quote with versions)

- #8580 v0.18.0rc1 2×A2 TP8/PCP2/DCP8 — serves short inputs, fails ~32K (A2) /
  ~64K (A3) with `507015` at the SFA indexer select; the closest thing to an A2
  datapoint and it is a failure report.
- #8579 v0.18.0rc1 — accuracy degradation with PCP enabled and
  `--no-enable-expert-parallel`: the coverage verdict, the one flag that swaps the whole
  MoE path, and the ladder are §4b, not a source walk.
- #8578 v0.18.0rc1 — `AttributeError: 'PCPManager' object has no attribute
  'mtp_slot_pad'` for PCP + MTP on short inputs; in the pinned tree
  `mtp_slot_pad` is still assigned only inside a `decode_threshold > 2 and not
  with_prefill` branch of `vllm_ascend/worker/pcp_utils.py`, so verify before
  claiming it is fixed.
- #7375 v0.13.0 A3 — MTP + PCP tensor mismatch (same w4a8-mtp-QuaRot family).
- #11231 v0.21.0rc1 — PCP2 + MTP `num_speculative_tokens: 2` on Qwen3.5-27B:
  serves, then all workers die on a mixed step with `ValueError: could not
  broadcast input array from shape (29,) into shape (31,)` at
  `pcp_utils.py:667`. Root cause, arithmetic and the fix as it reads in the
  v0.23.0 pin: `../mixed-batch-splits/pcp-schedule-index-units.md`. **The
  reporter stated on the same issue that MTP alone had been tested clean before
  the combined launch**, which is what this defect predicts
  (`num_decode_tokens == num_decode_reqs` without spec decode) — so when a
  PCP+MTP report carries that sentence, do not spend a launch on the MTP-only
  row; ask for the four source lines and the mixed-step numbers instead.

## 4b. Accuracy degradation with PCP on and EP off (no crash)

Worked for #8579 — DeepSeek-V3.2-W8A8, `--nnodes 2 --tensor-parallel-size 8
--data-parallel-size 1 --prefill-context-parallel-size 2 --decode-context-parallel-size 1
--no-enable-expert-parallel --cp-kv-cache-interleave-size 128 --max-num-seqs 4
--no-enable-prefix-caching --max-model-len 64k --max-num-batched-tokens 8192`,
FULL_DECODE_ONLY, v0.18.0rc1 (`99e1ea0f`). The deliverable is a **coverage** statement plus
one flag. The recorded replay spent 26 assistant turns and 2.0 M prompt tokens inside
`prepare_finalize.py` / `moe_comm_method.py` / `fused_moe.py` and emitted **no answer text
at all** — the turn it matched this page's precedent was the turn the verdict was owed.

- **One flag swaps the MoE communication strategy, so name which phase differs.**
  `vllm_ascend/ascend_forward_context.py:237-238`:
  `if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
  moe_comm_type = MoECommType.ALLGATHER`, and only then do the SoC branches (A2, A3, A5,
  310P) run. On A2 with EP on the choice is `MC2` when `num_experts // ep_world_size <= 24
  and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity` (`:239-248`, condition at
  `:245`), else `ALLGATHER`. For #8579's TP8 × 2 nodes and a 256-expert config (verify
  `num_experts` in the reporter's `config.json`), ep_world_size is 16, so that means **decode took
  `PrepareAndFinalizeWithMC2` with EP on and `PrepareAndFinalizeWithAllGather` with EP
  off**, while prefill was AllGather either way — so the phase to compare is decode, and a
  defect that also predicts a prefill difference is not it.
- **PCP is handled in only one of the two AllGather halves.** `prepare_finalize.py:397-413`
  pads to `_EXTRA_CTX.max_tokens_across_pcp` and `get_pcp_group().all_gather(...)`s both
  tensors; `:469-470` reduce-scatters back. Both sit in
  `_prepare_with_dp_group`/`_finalize_with_dp_group`, taken when `enable_sp()` /
  `enable_sp_by_pass()` are false — SP is `VLLM_ASCEND_ENABLE_FLASHCOMM1` /
  `VLLM_ASCEND_ENABLE_FLASHCOMM` (`utils.py:772-784`), **not** EP, so a launch that sets
  neither keeps the PCP-aware branch. The sibling `_prepare_with_ep_group` has no PCP
  gather at all.
- **One code asymmetry is worth one check, labelled a hypothesis.** The PCP branch records
  `self.num_tokens_pcp` (`:399`) and never slices it back, while the DP branch slices
  `[: self.num_tokens]` on return (`:436-437`): with differing per-rank PCP token counts
  the pad rows stay in the reduced output. Ask for `max_tokens_across_pcp` beside the local
  count, or run one launch with `--max-num-batched-tokens` a multiple of `pcp`. It predicts
  a *batch-shape-dependent* degradation, so a metric that is stable at one shape and worse
  at another supports it and a uniform shift does not.
- **Coverage, quotable, never the cause:** every variant of the tree's own CP-vs-TP
  accuracy test sets `enable_expert_parallel: True`
  (`tests/e2e/multicard/4-cards/long_sequence/test_accuracy.py:53,59,113,123,152,159,188,195,234,241`),
  and the multi-node CP tutorial says CP is A3-only at this pin
  (`docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md:6`). If the
  box is A2 while the reporter's EP-on run is clean, the reporter's own control outranks the
  doc's scope line.
- Ladder: `--enable-expert-parallel` alone (their clean row, as the expected control) →
  `--prefill-context-parallel-size 1` on the same command (isolates PCP) →
  `--decode-context-parallel-size 1` as well (serial baseline) → the token-count check
  above. Ask for the metric **and its harness**: `--max-num-seqs 4`, no prefix caching and a
  64K window make a batch-shape or long-context effect read as a numeric one.

## 5. Staged experiment to hand back (one axis per launch)

When the report compares EP-on with EP-off, rung 0 is `--enable-expert-parallel` with
nothing else changed (§4b).

1. DCP only, co-located, short input: one field added to the tutorial's own A2 row —
   `--data-parallel-size 2 --tensor-parallel-size 8 --decode-context-parallel-size 8
   --max-model-len 8192 --enforce-eager` (world = 2 × 8 = 16, and `dcp` = `tp` = 8
   because the set is SFA, §2) — no PD assertion, no CP env var. The
   `--decode-context-parallel-size 2` this page used to recommend is **not** an SFA
   row, and `tp=8 / dp=1` is 8 cards, which cannot hold the set at all. **Check the
   weight set fits one node first**: at this pin
   `docs/source/tutorials/models/DeepSeek-V3.2.md:20` puts
   `DeepSeek-V3.2-w8a8` on 1×A3 (64G×16) or **2×A2 (64G×8)**, so a single 8-card A2
   node cannot hold that set and every rung stays a two-node launch (world 16; use
   `--nnodes 2 --node-rank N` for the non-DP rungs and the `--data-parallel-*` block
   for a DP one); only a set that actually fits one node does
   (compare the checkpoint's size with `cards × card HBM × --gpu-memory-utilization`,
   and note the tutorial lists no w4a8 row at all, so its A2 node count is the only
   published one). The recorded #8984 answer quoted the 16-card requirement
   and then gave "A2 单机 8 卡, DCP=2" in the same message — keep the rung consistent
   with the tutorial's own node count for the reporter's weight set.
2. Add `--prefill-context-parallel-size 2` only after setting
   `--data-parallel-size 1` (world = tp × pcp) and
   `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1`.
3. Grow `--max-model-len` 8K → 32K → 64K; the input length where it dies **is**
   the #8580 signature and is the finding to report back.
4. Graph mode, then `--speculative-config '{"method":"deepseek_mtp",...}'` last;
   each on its own step so an `mtp_slot_pad`-class error is not confused with a
   CP error.

Ask for: card count and both serve commands, the first error line with its rank
prefix, `npu-smi info`, the image tag, and the log's `world_size=` line.
