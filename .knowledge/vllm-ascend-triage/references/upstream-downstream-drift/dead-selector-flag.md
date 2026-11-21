# An upstream kernel-selector flag that cannot fire downstream, plus a quoted "error" nobody emits

## Scope

The report blames a **backend-selection flag** for a layout/dtype failure on NPU —
"`--<x>-backend auto` 选了 flashinfer，推理时报张量布局错误", "`--enforce-eager`
没有关掉它", "只加 `--<x>-backend triton` 只修了 prefill，decode 还在用 flashinfer" —
and the pasted evidence is one line that is not a traceback. Recorded #15206
(Qwen3.5-9B, 910B/A2, vllm-ascend v0.23.0 `5cb98caa` + vllm v0.23.0 `0fc695fc`);
re-grep the sites at another pin, the shape recurs for every upstream kernel selector.

## Step 1 — attribute the quoted line before ranking any mechanism

1. Grep the quoted fragment **verbatim in both trees**
   (`rg -n 'expected BTH|got BHT' <vllm>/vllm <vllm-ascend>/vllm_ascend`). At #15206
   it matched nothing: the title's error string exists nowhere, so it is either the
   reporter's paraphrase or a *different* component's text.
2. Then grep the fragment that does exist (`chunk_gated_delta_rule`, the warning's
   first words) and read its raise/warn site. Here it is **downstream**:
   `vllm_ascend/ops/triton/fla/chunk.py:355-363`
   `if not head_first and q.shape[1] < q.shape[2]: warnings.warn("chunk_gated_delta_rule:
   Input tensor shape suggests potential format mismatch: seq_len (…) < num_heads (…)")`
   — a heuristic copied from FLA that fires whenever the packed prefill token count is
   below the key-head count (`4 < 8`), on the *correct* `[1, T, H, D]` layout. A
   `UserWarning` from the path the reporter believed was the fix, not an exception.
3. Only then decide whether a real failure exists. If the reporter cannot paste a
   per-rank traceback, the honest verdict is "the quoted line is benign; runtime is
   unproven" — do not construct a mechanism for a string that no code emits.

## Step 2 — why the flag is dead, in the downstream patch (not in the flag)

| Claim | Evidence (vllm v0.23.0 / vllm-ascend v0.23.0) |
|---|---|
| The flag parses and lands in `additional_config` | `vllm/engine/arg_utils.py:715` (field), `:1534` (`--gdn-prefill-backend`), `:2226-2227` (`additional_config["gdn_prefill_backend"] = …`) |
| Non-CUDA platforms never select flashinfer for GDN prefill | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` `_resolve_gdn_prefill_backend`: `if not current_platform.is_cuda(): return backend, "triton"`; flashinfer only when `supports_flashinfer`; an explicit `flashinfer` falls back with `warning_once("… cannot use this kernel on the current platform. Falling back to Triton/FLA.")` |
| The flashinfer import lives only in that branch | same file, `from flashinfer.gdn_prefill import …` inside the flashinfer forward |
| NPU does not consult the flag at all, because the patch replaces the methods that read it | `vllm_ascend/patch/worker/patch_qwen3_5.py:197-209` rebinds `forward`, `_forward_core`, `get_attn_backend`, `_warmup_prefill_kernels` on the upstream GDN class |
| Prefill and decode kernels on NPU | prefill → `vllm_ascend/ops/gdn.py:402-413` `vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule(..., head_first=False)`; decode → `:423-425` `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` |

So "only prefill was fixed, decode still uses flashinfer" is refutable from code: the
option is named *prefill* and there is no flashinfer decode branch to reach. The
stable lesson is the split — **upstream decides the flag, downstream decides the
kernel** — so a flag-level answer is only correct when the downstream patch leaves
that method alone; check `patch/**` for the class before quoting upstream's guard.

## Decisive checks for the reporter (each one line)

- `grep "GDN prefill kernel" <serve.log>` → `Using Triton/FLA GDN prefill kernel
  (requested=auto, head_k_dim=…)`, printed once per process by
  `_log_gdn_backend_decision` at model build ⇒ flashinfer never ran. A line saying
  `FlashInfer GDN prefill` instead means their vllm is not the pinned release
  (the guard is unconditional off CUDA) — ask for `pip show vllm vllm-ascend`.
- Re-send a prompt longer than the key-head count: the warning disappears, which
  proves it tracks `T < H` and nothing else.
- `--enforce-eager` disables graph capture only; it never selected a kernel, so the
  report's inference "eager should have disabled flashinfer" is correct in effect and
  wrong in mechanism — say which.
- Documentation/scope statement worth quoting: at v0.23.0 the support matrix lists
  `Qwen3.5-Dense` (2B/4B/9B) **only** for Atlas 300I DUO
  (`docs/source/user_guide/support_matrix/supported_models.md`, A2/A3 tab has only
  Qwen3.5-397B-A17B / Qwen3.5-27B), and the tutorial's `--dtype float16` +
  `--mamba-ssm-cache-dtype float16` notes are the 310P contract. A 9B dense on A2 is
  an undocumented shape: report that as scope, not as a defect.

## Limits

The kill-switch and the warning text are version-scoped (FLA upstream syncs the
heuristic, and GDN backend resolution was rewritten across releases). The
patch-replaces-the-method part is the durable claim; the line numbers are not.
