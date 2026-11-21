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
(`../dsa-sparse-attention/o-proj-quant-attribute-crash.md`).
