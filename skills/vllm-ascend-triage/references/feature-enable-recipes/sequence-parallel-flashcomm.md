# FlashComm1 / sequence parallelism: its gates, and the models it breaks

Scope: `VLLM_ASCEND_ENABLE_FLASHCOMM=1` / `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` appears in
the report and the engine either refuses the combination up front, dies in the
memory-profiling forward with a **shape** error, **or does nothing at all** — the serve
is healthy and the reporter's profiler still shows allreduce (that lane is the §*other
entry shape* below, and its answer is a per-forward token gate, not a bug), **or** the
serve is fine and the *accepted-token* metrics fall when the env is turned on (§*FC1
with a MoE MTP drafter*). Verified at vllm-ascend v0.14.0rc1
(`52d4acfa`) + vllm v0.14.1 (`d7de043`), the #6261 pins, and the VL guard re-verified at
v0.15.0rc1 (`3d43ed99`) + vllm v0.15.0 (`f176443`) for #6950: the guard, the env name, the
predicate and the assertion texts are all ref-specific — re-grep at the reporter's ref, and
read the guard's *own wording* before deciding what to recommend.

Verdict shape for the recorded case (Qwen3-Omni-30B-A3B + `VLLM_ASCEND_ENABLE_FLASHCOMM=1`,
TP2 + EP, crash at `profile_run`):

> FlashComm1 (= sequence parallelism) is not a supported combination with this model.
> Unset the variable (or set it to 0) and keep the rest of the command; the crash is
> `hidden_states` arriving sequence-sharded while the per-layer deepstack visual
> embedding is full length.

## The other entry shape: env set, nothing crashes, profiler still shows allreduce

Recorded #6862 (Qwen3-32B-W8A8 *dense*, 910B3 x8 idle, CANN 8.5.0, torch_npu 2.9.0,
vllm 0.14.1 + vllm-ascend v0.14.0rc1): "根据Qwen3-Dense文档，开发FLASHCOMM1后采集prof，
显示的仍然是allreduce算子" — no traceback, no launch line, no profile table. This is a
capability report about a **switch that appears to do nothing**, and the answer is not
the VL guard above: SP is gated **per forward**, so a low-token forward is allreduce by
construction.

- The dense trunk needs `num_tokens > 1000` **in the forward being profiled**:
  `ascend_forward_context.py:88-95` at v0.14.0rc1 has three branches — MoE context model
  (`enable_sp(...) and num_tokens is not None`), drafter (`False`), else the dense trunk
  (`... and num_tokens > 1000`) — and `platform.py:605-610` carries the same two-branch
  copy without the drafter case. The pinned doc says it in words:
  `docs/source/tutorials/Qwen3-Dense.md:335` "FlashComm_v1 is only enabled if the actual
  token count for each inference schedule exceeds the threshold" (`:343` repeats it for
  the matmul+reduce-scatter fusion).
- SP off means literally allreduce: `ops/linear_op.py:572-576`
  `if not sp_enabled: ... return tensor_model_parallel_all_reduce(output_parallel)`.
- Decode `num_tokens` = the sequences in the batch (bounded by `--max-num-seqs`), so a
  dense serve at `--max-num-seqs 20` never crosses the threshold and a decode-window
  profile cannot show reduce-scatter. A non-SP dense forward at TP>1 issues **2 allreduce
  per layer** (`o_proj`, `down_proj`), i.e. ~128 for a 64-layer model — a count in that
  band confirms "gate closed", and a count far below it reopens the branch question below.
- The decisive experiment, one launch: profile a **single schedule above 1000 tokens**
  (two ~3.5 K-token prompts together, `--max-num-batched-tokens` ≥ 8192, `--enforce-eager`
  so the rows are per-op). The allreduce pair must vanish and
  `npu_mm_reduce_scatter_base` appear (`ops/linear_op.py:596`, and the W8A8 arm at
  `:608-633`; with `mmrs_fusion` false the generic `tensor_model_parallel_reduce_scatter`
  at `:640`) —
  **and no all-gather**, which is the whole point of SP.
- The env did reach the worker if the serve started at all with `tp>1`: `enable_sp()`
  asserts `tensor_parallel_size > 1` (`utils.py:833-836`), so no need to re-ask for a
  `collect_env` to prove the variable was read.
- **The threshold is dense-only, and that decides two questions.** The MoE context
  branch enumerated above is `enable_sp(...) and num_tokens is not None` — no threshold —
  so on a MoE trunk (Qwen3-MoE, the Omni thinker, Qwen3-Next) SP is engaged in **every**
  forward, including the `profile_run`/`_dummy_run` pass used for memory profiling. That
  is why a MoE × SP shape error is a *startup* failure in `determine_available_memory`
  rather than a first-request failure, and why no batch size or
  `--max-num-batched-tokens` value avoids it (recorded #6261: the reporter's 4096 was
  never the variable). `--enforce-eager` does not avoid it either — it only drops the
  dynamo/fake-tensor wrapper, so the same `add` fails on real tensors; offer that one
  launch as the confirmation that the shape, not the graph path, is wrong. Conversely,
  on a MoE model an allreduce-only profile **cannot** be this gate: go to the
  earlier-branch question below with the launch line in hand.

### When the big-schedule profile still shows allreduce: an earlier branch claimed the layer

`_get_row_parallel_op` (`ops/linear_op.py:728-757` at v0.14.0rc1) returns in order:
`enable_dsa_cp_with_layer_shard()` (`o_proj`) → `mlp_tp_enable()` (`down_proj`) →
`oproj_tp_enable()` (`o_proj`) → `matmul_allreduce_enable()` → `flashcomm2_enable()`
(`o_proj`/`out_proj`) → `enable_sp()`. Whichever fires first owns the layer, and SP never
builds `SequenceRowParallelOp` at all. `mlp_tp`/`oproj` come from
`finegrained_tp_config.{mlp,oproj}_tensor_parallel_size > 0` (`utils.py:800-805`, i.e. in
`--additional-config`), matmul-allreduce from `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`
(`utils.py:808-809`; default `0`, `envs.py:74`), FlashComm2 from
`VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE > 0` (`envs.py:82`). The `mlp_tp` branch is also a
**memory** switch, not only an op-selection one: at v0.11.0rc0 the same knob is the env var
`VLLM_ASCEND_ENABLE_MLP_OPTIMIZE` and `_MLP_TP` is sized by `data_parallel_size`
(`distributed/parallel_state.py:65-81`), so on a TP-only serve the chosen op shards nothing,
replicates the MLP on every card and turns its own collectives into no-ops —
`../deploy-sizing/mlp-tp-group-oom.md`. The matmul-allreduce branch is
the deceptive one — it yields a **fused MatmulAllReduce**, which reads exactly like
"allreduce is still there" in a kernel-name table while SP is engaged nowhere. So read the
launch line for those three knobs *before* crediting the token gate, and say which of the
two the evidence supports instead of offering both.
- `TP > 8` cannot produce this symptom: `mmrs_fusion = tp_world_size <= 8`
  (`ascend_forward_context.py:76`) only removes the fused kernel; the fallback is
  reduce-scatter (`:640`), never allreduce.
- Ask back for: the profiler rows (op name **and** call count), the full serve command,
  and **which window** was captured (prefill or decode, concurrency) — a decode-window
  profile is this section's answer with no further reading.

## SP is on through four doors, not one

`enable_sp()` (`vllm_ascend/utils.py:812-841`) ORs
`compilation_config.pass_config.enable_sp` **or** `VLLM_ASCEND_ENABLE_FLASHCOMM1` **or**
the legacy env `VLLM_ASCEND_ENABLE_FLASHCOMM` (the comment says backward
compatibility), plus `enable_shared_expert_dp` when that argument is set. Upstream
also flips `pass_config.enable_sp` on by itself for async TP
(`pass_config.fuse_gemm_comms`, `vllm/config/vllm.py:726-728`; the field is
`PassConfig.enable_sp`, `vllm/config/compilation.py:125`). So "I never set FLASHCOMM"
is not a refutation — read the predicate, then the report's `additional_config`.

Those four doors are the *v0.14.0rc1* predicate. At v0.17.0rc1 / v0.22.1rc1 the same
function is driven by `additional_config["enable_flashcomm1"]` /
`ascend_config.enable_flashcomm1` / `VLLM_ASCEND_ENABLE_FLASHCOMM1`, with
`pass_config.enable_sp` a separate compile pass (reader table in
`../accuracy-benchmarks/early-eos-short-output.md` §Step 2) — re-grep `enable_sp()` at
the reporter's ref before quoting the doors, and do not carry this page's VL guard onto
a ref whose support matrix lists VL + SP.

Hard asserts that fire before any shape error: `tp_size > 1`
(`utils.py:833-836`, "Flash Comm v1 (Sequence Parallelism) is only supported when
tp_size > 1") and, for a MoE model, `enable_expert_parallel=True`
(`utils.py:838-841`). Both are recorded as satisfied in #6261, which is why the run
reached the forward.

## The VL / Omni guard: two different guards, one per generation

Read the reporter's ref before quoting either. Same symptom string family, different predicate
and a different recommendation inside the message.

| ref | guard | what counts as VL | what it tells the reporter |
|---|---|---|---|
| v0.14.0rc1 (`52d4acfa`) | `ValueError: Currently, VL models doesn't support FLASHCOMM in vllm-ascend. We will fix this in the future. Please set VLLM_ASCEND_ENABLE_FLASHCOMM1=0.` at `platform.py:394-403` | `thinker_config` in `hf_config` — the comment names "Qwen-Omni-thinker models" (`utils.py:905-915`) | unset the env |
| v0.15.0rc1 (`3d43ed99`, #6950) | `assert not is_vl_model(vllm_config), "Flash Comm V1 is not supported for VL models. Please disable it by setting VLLM_ASCEND_ENABLE_FLASHCOMM1=0. For optimal performance with VL models, we recommend enabling Sequence Parallelism via --compilation-config '{"pass_config": {"enable_sp": true}}'."` at `platform.py:399-403`, inside `if enable_flash_comm_v1():` | `thinker_config` **or** `vision_config` in `hf_config.to_dict()` (`utils.py:816-827`) — every ordinary VLM, not just Omni | **switch to the Inductor SP pass**, not just unset |

Unchanged across both: this fires **fast at engine start** (`check_and_update_config`), so a report
whose crash is later (inside `profile_run`) means the guard did not cover that model at the
reporter's version.

**At v0.15.0rc1 the pass route is the documented replacement, not a way around the check.**
`docs/source/user_guide/feature_guide/sequence_parallelism.md:41-57` is a support matrix, not
prose: without quantization SP × VL+Dense = `graph`, SP × VL+MoE = `x`, FC1 × VL = `x`; with
quantization every cell is `x`. The page states SP for VL-class models is implemented on the
Inductor pass, and the pass is graph-only — `enable_sp_by_pass` is
`not enforce_eager and pass_config.enable_sp` (`utils.py:736-738`), `sp_threshold` defaults 1000
(`compilation/passes/sequence_parallelism.py:16-24`). The tree also tests it:
`tests/e2e/multicard/2-cards/test_sp_pass.py:9-50` serves `Qwen/Qwen3-VL-2B-Instruct` at TP2 with
`enable_sp` false/true and compares outputs. That is in-tree coverage at this ref, **not** wired in
`.github/workflows`, so call it "the tree's own e2e test", never "CI-validated".

What the older reading still owns: the assert removes only the *check*, and the machinery behind
both doors is the same `sp_enabled` path, so a model whose forward adds a **full-length** auxiliary
tensor (deepstack / per-layer injected features — next section) still breaks, and VL+MoE and every
quantized checkpoint stay out of the matrix. Procedure therefore: verify the model's forward first
(§*The mechanism*), then route VL+Dense to the pass and keep "unset the env" as the answer for
everything the matrix marks `x`.

**The guard predates the reporter's "works on v0.13.0" — expect the predicate, not the guard, to be
what changed.** `release_notes.md:296` at this ref, under `## v0.11.0rc2` (`:272`; the same bullet sits at `:498`
on the v0.15.0rc1 pin, which is what `../evidence-workspace/recorded-cases-pd-kv-and-features.md`
quotes — cite the line for the ref you are on), records "Add a check that to ensure
FLASHCOMM feature doesn't work with vl model. It'll be supported in 2025 Q4 [#4222]". So when a
reporter says an older release worked, the likely difference is the VL predicate (thinker-only →
`vision_config`) or the error class (ValueError → assert), not the guard's existence. A one-commit
snapshot cannot diff their release: state that, and give them the one command —
`sed -n '/def is_vl_model/,+12p' vllm_ascend/utils.py` — on their own tree.

## The mechanism, so the shape error reads as arithmetic

With SP enabled the row-parallel output projection reduce-scatters and then **skips**
the all-gather: `vllm_ascend/ops/linear_op.py:386-396`
(`reduce_scatter(output_parallel, dim=0)` then `if not forward_context.sp_enabled:
all_gather(...)`). The layer output therefore stays `(L/tp, H)`. Any model that adds a
tensor of *full* sequence length to it at the same point breaks, and the fake-tensor
message says so: `size=((s59//2), 2048)` against `size=(s68, 2048)` for TP=2.

The Qwen3-Omni thinker does exactly that: `Qwen3MoeLLMModel.forward` adds
`deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]` after every decoder
layer (`vllm/model_executor/models/qwen3_omni_moe_thinker.py:644-650` in upstream
v0.14.1), and the deepstack tensors are built at full prompt length from the vision
tower (`:540-568`). `grep -rln deepstack vllm_ascend/` is **empty** at this ref, and
`grep -rln -i omni` returns only the `is_vl_model` comment in `utils.py:905-915` — no
model file and no sharding code: nothing downstream shards them, so this is a model ×
feature gap, not a flag the reporter got wrong. Expect that second grep to look
non-empty and read what it matched before concluding otherwise.

Generalise it as a pre-check before promising SP for any model: does the model's
`forward` combine the decoder output with an auxiliary tensor of full sequence length
(deepstack / multimodal side embeddings / per-layer injected features)? Grep the
upstream model file for the add, then grep `vllm_ascend/` for the key — empty means no
shard handling and the same failure.

## v0.19.x hybrid layers

One SP-sized buffer at layer 0, two writers into it, two enable gates — the Qwen3.5 / Qwen3-Next
(GatedDeltaNet) family.

Recorded #8990 (image `v0.19.1rc1-openeuler`, Qwen3.5-27B-w8a8-mtp, 910B3 x8, TP2, `qwen3_5_mtp`
num_spec 3, `FULL_DECODE_ONLY` capture `[1,2,4,8,16,32,64]`, `--async-scheduling`,
`--quantization ascend`, `ASCEND_RT_VISIBLE_DEVICES=0,1`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`):
`compile_or_warm_up_model` → `NPUModelRunner init failed … expected src to have a size equal to the
slice of self. src size = [64, 5120], slice size = [32, 5120]`. Verified at v0.19.1rc1
(`da421afad719`). The same model family x feature pair at v0.18.0rc1 is `#7996` (Qwen3.5-35B-A3B,
MTP 3 + FLASHCOMM1, dies in `drafter.dummy_run` at the MoE combine add, `qwen3_next.py:358`, 4096 vs
8192), with `#4011` ("flashcomm1 report shape error in DP>1") and `#6283` (PP with flashcomm1/sp)
beside it — the family is older than the v0.19 regression reading.

- **The MoE + shared-expert `add` is a row-count readout, and the operand order names
  the side.** #7996's message is not a `copy_` check but upstream's
  `final_hidden_states[0] + final_hidden_states[1]` (`qwen3_next.py:358` at vllm v0.18.0);
  upstream's tuple is `(shared_out, fused_out)` (`fused_moe/shared_fused_moe.py:58`) and the
  downstream override keeps that order (`vllm_ascend/ops/fused_moe/fused_moe.py:752` at
  v0.18.0rc1 `99e1ea0f`). `shared_out` is computed from the MoE block's **input** rows
  (`:743`; the `multistream_overlap_gate` branch does the same *before* `prepare` at `:491`),
  and `shared_expert*` prefixes are excluded from both SP op selections (`linear_op.py:638`,
  `:681`), so **`a` in the paste is the block's input row count and `b` is the routed
  count after prepare/finalize**. #7996's `a = 4096`, `b = 8192` therefore says: the input
  arrived TP-sharded, and the routed side gained rows in prepare without losing them in
  finalize. Two prints, no gate walk: the row count entering the block, and which
  `finalize()` branch ran (`prepare_finalize.py:436-437`).
- **`use_sequence_parallel_moe` needs DP > 1, so the model's own chunk is not the
  sharder in a single-DP command.** `is_sequence_parallel` is
  `parallel_config.use_sequence_parallel_moe` (`qwen3_next.py:268`), which requires an
  `all2all_backend` in its list **and** `enable_expert_parallel` **and** TP > 1 **and**
  `data_parallel_size > 1` (`vllm/config/parallel.py:586-600`). In #7996 (DP1 x TP2 x EP)
  the `sequence_parallel_chunk` at `qwen3_next.py:342-343` never ran, so the sharding came
  from the plugin/eagle side (`eagle_proposer.maybe_pad_and_reduce`, `:1593-1607`) —
  check that flag before attributing rows to the model.
- **`_EXTRA_CTX` is not a second source of truth.** It is a proxy onto the current vLLM
  forward context (`ascend_forward_context.py:285-310`), so the gather
  (`register_custom_ops.py:46`) and the scatter (`:79`) read the same
  `flash_comm_v1_enabled`. The asymmetries that can actually disagree are (i) the static
  strategy dispatch `enable_sp() or enable_sp_by_pass()` (`prepare_finalize.py:327`,
  `:436`) versus each tensor op's dynamic flag, and (ii) the scatter's extra exclusion
  `forward_context.is_draft_model and is_vl_model()` (`:83`), which the gather does not
  carry. #7996 spent most of its budget re-deriving the two gates from that false premise.
- **This family's answer is six lines**: the direction sentence (`a` = input rows, `b` =
  routed rows), the two `path:line`s that can disagree at the reporter's ref, the
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=0` ablation with the rest of the command unchanged, and
  the ask for the full per-rank traceback. What sits *below* the flag is an open question
  with one print — #7996 (11 turns, 1 540 characters of text against 48.9 K of reasoning)
  shipped no answer while walking it.

- **Read the direction out of the message before reading any source.** It is a `Tensor.copy_` /
  `__setitem__` size check: dst rows are the *buffer's* rows, src rows the *producer's*.
  `src = L`, `slice = ceil(L/tp)` means the buffer was SP-sized and the producer returned full
  length; the reverse (`src = L/tp`, `slice = L`) means the buffer was full and the producer
  reduce-scattered. The #8990 run reversed the two twice and lost turns to it.
- **The SP-sized buffer exists at layer 0 only.** `patch/worker/patch_qwen3_5.py:91-96`, under
  `if self.layer_idx == 0 and _EXTRA_CTX.flash_comm_v1_enabled`, allocates `(ceil(tokens/tp),
  hidden)`; every other layer uses `empty_like`. So the failing layer is layer 0 of whichever model
  is running — the target, or the MTP drafter's own single layer, which is the same patched class.
  The capture dump proves which layer type that is (`l_self_modules_layers_modules_0_modules_linear_attn_…`
  = GatedDeltaNet at this model).
- **Two writers, told apart by the printed dst.** `output[:], _ = self.o_proj(attn_output)`
  (`patch_qwen3_5.py:74`, `full_attention`) vs `output[:num_tokens], _ = self.out_proj(core_attn_out)`
  (`ops/gdn.py:136`, `linear_attention`). In the GDN one Python **clamps** `[:num_tokens]` to the
  buffer, so the pasted `slice size` is the buffer's rows and never `num_tokens` — that is exactly why
  a full-length producer prints as `src > dst`.
- **Whether the producer shards is a build-time decision, not a runtime one.** `ops/linear_op.py`
  `_get_row_parallel_op` returns `SequenceRowParallelOp` only for prefixes in `sp_row_prefixes`
  (`o_proj`, `out_proj`, `down_proj`, `attention.dense`, `:679-687`; the query side is
  `sp_column_prefixes` = `gate_up_proj`, `in_proj`, `qkv_proj`, `conv1d`, `query_key_value`). Its
  `matmul_and_reduce` (`:500-576`) then picks `all_reduce` (SP off, **full rows**), `npu_mm_reduce_scatter_base`
  (unquant + `mmrs_fusion`, `:530`; w8a8 `:559`, needs `tp<=8`), or `apply` +
  `tensor_model_parallel_reduce_scatter` (`:574`). A module whose name is outside those lists keeps
  full rows: grep the prefix list before grepping `copy_`, and check a new model revision's projection
  names against it.
- **The gate is token-count dependent and is written twice, differently.** `ascend_forward_context.py:112-129`
  sets `flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None` for a MoE context
  model, `False` for a **dense drafter** (comment: "for dense drafter, sp is redundant and is not
  compatible with dp and graph"), and requires `num_tokens > 1000` for a dense target;
  `platform.py:743-748` carries the same two-branch version **without** the drafter case and sets
  `mmrs_fusion` unconditionally. So at 64 decode tokens the dense target is below the threshold while
  the MoE 35B variant is always on — say which function built the context before claiming a mode, and
  do not carry the threshold onto the MoE branch.
- **The drafter shards its own inputs.** `eagle_proposer.py:1767-1777` `maybe_pad_and_reduce`s the
  drafter's `hidden_states` and `positions` when `flash_comm_v1_enabled`, and the patched layer 0 then
  re-derives `ceil(rows/tp)` from what it receives: print the row count each level sees, because a
  double shard and a missing shard give opposite message directions.
- **Coverage at this ref**: `tests/e2e/multicard/4-cards/test_qwen3_5.py:87-100` is the only case
  with `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` + `qwen3_5_mtp` + num_spec 3 + `FULL_DECODE_ONLY [4,8,12,16]`,
  and it is Qwen3.5-35B-A3B at DP2 x TP2 x EP — the MoE branch. A dense w8a8 27B command with
  `--async-scheduling` is outside it; say "in-tree tested for the MoE variant, not this one".
- **Coverage gap to state once**: this snapshot pins vllm `v0.13.0` while the paired upstream is
  `v0.19.1` (`docs/source/conf.py` `myst_substitutions.vllm_version`), so
  `vllm/model_executor/models/qwen3_5.py` does not exist in it and the upstream layer forward is
  unreadable — work from the downstream patch plus the capture graph's own module names instead of
  re-grepping the upstream file.
- A reporter's "works on v0.18.0rc1" is consistent with this layer-0 allocation being new or changed,
  and a `one_commit_git` snapshot cannot diff their release: give them the read,
  `sed -n '80,110p' vllm_ascend/patch/worker/patch_qwen3_5.py`, on their own tree.

## FC1 with a MoE MTP drafter: acceptance drops, nothing crashes

Recorded #10504 (vllm-ascend v0.18.0 `e18643f8` + vllm v0.18.0 `bcf2be96`, 2×A2 910B2,
DeepSeek-V3.2 **w4a8**, `method: deepseek_mtp`, `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`):
TTFT improves as the feature promises (16.86→6.05 s at concurrency 1) while the
accepted-token rate falls 62.5 %→30.8 %, 48.8 %→17.9 %, 57.8 %→14.4 % at
concurrency 1/4/16 and output TPS drops at 1 and 4. No traceback, no device record,
both models keep executing. The issue's own root-cause section blames
`is_drafter_moe_model` letting the MTP drafter into the FC1 branch and a
drafter-vs-main `pad_size` mismatch. The gate is real; the stated mechanism is not, and
four reads at the pin settle it:

- `ascend_forward_context.py:97-106`: the MoE-context branch is `enable_sp(vllm_config)
  and num_tokens is not None` with **no** token threshold, and only a *dense* drafter is
  forced to `False` (`:101-105`, comment "sp is redundant and is not compatible with dp
  and graph"). `is_drafter_moe_model` (`utils.py:813-819`) is `_is_contain_expert` on the
  draft's own `hf_text_config` with **no method special-case**, so "eagle3 is already
  special-cased, MTP is not" is wrong at this ref — an eagle3 draft config that carries
  expert keys takes the same branch, and a dense one takes the other. Read the draft
  config, never the method string.
- So the fix the issue proposes is a behaviour change to a path the tree exercises:
  the DSV3.2 configs ship `VLLM_ASCEND_ENABLE_FLASHCOMM1: 1` **together with**
  `method: deepseek_mtp` — `tests/e2e/nightly/single_node/models/configs/DeepSeek-V3.2-W8A8.yaml:15,40`,
  `tests/e2e/nightly/multi_node/config/DeepSeek-V3_2-W8A8-EP.yaml:33,44`,
  `.../DeepSeek-V3_2-W8A8-A3-dual-nodes.yaml:15,40` — though none of them carries an
  acceptance threshold. The honest width is "in-tree nightly combination, never
  acceptance-validated": neither "unsupported" nor "validated".
- `pad_size` is taken from the same count as the rows being sharded, so the
  drafter-vs-main mismatch is not available as stated: the drafter's context receives
  `num_tokens=num_input_tokens`, the graph-padded value from
  `cudagraph_dispatcher._bs_to_padded_graph_size` (`eagle_proposer.py:534`; context at
  `:704-712` with `num_actual_tokens=num_tokens`), and
  `pad_size = (tp_world_size - num_tokens % tp_world_size) % tp_world_size` uses that
  same value (`ascend_forward_context.py:114-117`). **Scope that refutation to
  `dp_world_size == 1`** — see the next bullet.
- **`dp_world_size > 1` re-opens the reporter's pad intuition, as one check, not a
  verdict.** `ascend_forward_context.py:135-142` then overrides both fields from a
  different source: `padded_length = ceil(max_tokens_across_dp_cpu / tp) * tp`,
  `pad_size = padded_length - num_tokens`. The pad rows no longer follow from the count
  being sharded, so a target-vs-drafter offset is available again without any
  graph-padding story. The issue carries no serve line, so dp is unknown: the cheap
  discriminator is `--data-parallel-size 1` with tp/ep unchanged (acceptance recovers
  ⇒ this branch owns it) or one print of `padded_length` / `max_tokens_across_dp` in
  the target's and the drafter's context.
- The scatter/gather pair is symmetric for `method == "mtp"`: the scatter gates on
  `_EXTRA_CTX.flash_comm_v1_enabled` (`eagle_proposer.py:1624-1626`), the gather on
  `self.enable_shared_expert_dp` (`:1641-1646`), which is `shared_expert_dp_enabled()` =
  `enable_shared_expert_dp or enable_sp() or enable_sp_by_pass()` (`:102`,
  `utils.py:796-797`) — FC1 on implies gather on. And the target's hidden states are
  all-gathered to full length before the handoff (`model_runner_v1.py:1841`,
  `_all_gather_hidden_states_and_aux` `:1758-1779`), so no double shard from that side.

Attribution comes before the shape question, and it costs one launch: send the *same*
greedy request (`temperature: 0`, fixed `seed`, `return_token_ids: true` — a request
field at this pin, `vllm/entrypoints/openai/completion/protocol.py:134` and
`.../chat_completion/protocol.py:311`) with FC1 off and on, and compare the **target's**
own token ids. Divergence beyond a token or two there makes the drafter a bystander and
moves the owner to target-side SP numerics on a w4a8 checkpoint — the reading that also
fits output TPS *rising* at concurrency 16 while acceptance falls. Only a stable target
leaves the drafter as the suspect.

What remains is then a per-draft-step row-shape question, and the reporter's own metrics
hold the discriminator: ask for the `Per-position acceptance rate:` line with FC1 on
**and** off. Read the shape before any source walk: a drop that appears only from
position 2 onward is K-dependent (the later draft steps), a uniform drop is not. Send it with
`--enforce-eager` (removes graph padding, so `num_input_tokens == num_tokens`) and an
FC1 on/off pair at `num_speculative_tokens: 1`. Which of the two this report is stays
open — the reporter's later comment (2026-06-23) carries only `collect_env` output
(v0.18.0, 910B2, CANN 8.5.1) and no resolution.

Cost note: the answer is those four refutations plus the attribution pair and the
`--enforce-eager` row, and it is deliverable before any MLA/proposer line walk. An
earlier #10504 replay spent its entire 16 384-token output budget re-deriving
`eagle_proposer.py` / `mla_v1.py` offsets over 11 turns and shipped no answer text
before the cap; a later replay that led with the four reads above shipped the refutations
and the asks in 8 assistant turns / 9.6 K output tokens, and the dp rung above is what
that replay added to this section.

## Procedure

1. Answer first, then the replacement. On a ref whose guard says only "disable it" (v0.14.0rc1
   family): unset the env (or set 0), keep TP/EP/graph flags, re-run. On a ref whose guard names
   SP-by-pass (v0.15.0rc1 family): unset the env **and** add
   `--compilation-config '{"pass_config": {"enable_sp": true}}'` when the model is VL+Dense
   unquantized — that is the same message's own instruction and it is in the matrix.
2. Cite the guard `path:line` for the reporter's ref family and the mechanism for the
   shape arithmetic; label the guard's absence at their version as the open item.
3. Send back / ask for: the full per-rank traceback (the `from user code:` frame names
   the model line), `npu-smi info`, and whether `shared_expert_dp` or `fuse_gemm_comms`
   is in the command — either keeps SP on after the env is removed.
4. Do not propose sharding the auxiliary tensor yourself: the guard's text
   ("we will fix this in the future") is the project's position for that version.

Related SP side effects worth one line each: with SP on, the ACL-graph capture-size
list must keep multiples of `tp_size` or `platform.py:250-267` asserts, and spec-decode
sizing uses `max(query_len, tp_size)` (`vllm/config/compilation.py:1080-1090`); on DSA
models the same env family selects an o_proj TP path that needs W8A8 weights
(`../dsa-sparse-attention/o-proj-quant-attribute-crash.md`). FlashComm**2** is a separate
switch: its `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` > 0 plus `layer_sharding: ["o_proj"]`
registers one `o_proj` twice and asserts `layer_idx must be consecutive` while loading
weights — `flashcomm2-o-shard-layer-sharding.md`, not the SP shape walk above.
