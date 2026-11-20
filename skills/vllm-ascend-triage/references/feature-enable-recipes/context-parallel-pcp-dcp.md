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
- The nearest human evidence is precedent-only and partial: #8580 (v0.18.0rc1,
  2×A2, TP8 / PCP2 / DCP8, `--cp-kv-cache-interleave-size 128`) reports that the
  deployment **serves short prompts and dies around 32K input tokens on A2**
  (~64K on A3) with `AclrtSynchronizeStreamWithTimeout ... 507015` at
  `sfa_cp.py` `_execute_indexer_select`. So PCP does come up on A2 for V3.2 at
  0.18.0rc1 — it is not usable long context, and nothing in that report covers a
  w8a8/w4a8 **mtp QuaRot** weight set.

The honest answer is therefore three sentences: what the docs claim (A3-only), what
the only precedent shows (A2 runs to ~32K then fails, v0.18.0rc1, different
weights), and that DCP alone has no separate A2 validation either. Then the
constraints and the staged experiment.

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
  `--no-enable-expert-parallel`; treat `--enable-expert-parallel` as part of the
  recipe for MoE models.
- #8578 v0.18.0rc1 — `AttributeError: 'PCPManager' object has no attribute
  'mtp_slot_pad'` for PCP + MTP on short inputs; in the pinned tree
  `mtp_slot_pad` is still assigned only inside a `decode_threshold > 2 and not
  with_prefill` branch of `vllm_ascend/worker/pcp_utils.py`, so verify before
  claiming it is fixed.
- #7375 v0.13.0 A3 — MTP + PCP tensor mismatch (same w4a8-mtp-QuaRot family).

## 5. Staged experiment to hand back (one axis per launch)

1. DCP only, co-located, short input: `--tensor-parallel-size 8
   --decode-context-parallel-size 2 --max-model-len 8192 --enforce-eager` — no
   extra cards, no PD assertion, no env var.
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
