# FlashComm1 / sequence parallelism: its gates, and the models it breaks

Scope: `VLLM_ASCEND_ENABLE_FLASHCOMM=1` / `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` appears in
the report and the engine either refuses the combination up front or dies in the
memory-profiling forward with a **shape** error. Verified at vllm-ascend v0.14.0rc1
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
what changed.** `release_notes.md:498`, under `## v0.11.0rc2`, records "Add a check that to ensure
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
tower (`:540-568`). `grep -rn deepstack vllm_ascend/` and `grep -rln omni
vllm_ascend/` are both **empty** at this ref: there is no downstream patch sharding
them, so this is a model × feature gap, not a flag the reporter got wrong.

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
