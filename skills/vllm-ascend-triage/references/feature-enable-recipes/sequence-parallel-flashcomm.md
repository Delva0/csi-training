# FlashComm1 / sequence parallelism: its gates, and the models it breaks

Scope: `VLLM_ASCEND_ENABLE_FLASHCOMM=1` / `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` appears in
the report and the engine either refuses the combination up front or dies in the
memory-profiling forward with a **shape** error. Verified at vllm-ascend v0.14.0rc1
(`52d4acfa`) + vllm v0.14.1 (`d7de043`), the #6261 pins; the guard, the env name and
the assertion texts are ref-specific — re-grep at the reporter's ref.

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

## The VL / Omni guard, and what it does *not* cover

`platform.py:394-403` raises
`ValueError: Currently, VL models doesn't support FLASHCOMM in vllm-ascend. We will fix
this in the future. Please set VLLM_ASCEND_ENABLE_FLASHCOMM1=0.` when `is_vl_model()`
is true and either env is set, and `is_vl_model()` counts a checkpoint whose
`hf_config` carries `thinker_config` as VL — the comment names "Qwen-Omni-thinker
models" (`utils.py:905-915`). Consequences:

- On a ref that has this branch the same export fails **fast at engine start**; a
  report showing the crash later (inside `profile_run`) means the guard did not cover
  that model at the reporter's version. A one-commit snapshot cannot diff another
  release: state that as the open comparison and ask them to re-run the same command
  on the newer ref.
- The guard is the project's own statement of support. Do not route around it with
  `pass_config.enable_sp` — that keeps the unsupported path and skips only the check.

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

1. Answer first: unset the env (or set 0), keep TP/EP/graph flags, re-run. That is the
   rung-1 fix and it is what the guard itself asks for.
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
