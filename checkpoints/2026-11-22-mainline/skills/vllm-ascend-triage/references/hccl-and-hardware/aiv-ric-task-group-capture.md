# `507009` in a graph-task-group (RIC) capture: AIV expansions × NPU graph

## Scope

A **graph-capture** paste whose first error is inside the capture window, not a
timeout after a healthy serve:

```text
[rank0]: EE9999: Failed to submit capture task, retCode=0x7100003.[FUNC:CaptureEventProcess][FILE:event.cc]
[rank0]:         Capture event record failed.[FUNC:EventRecord][FILE:api_impl.cc]
[rank0]:         rtEventRecord execution failed, reason=task not supported[FUNC:FuncErrorReason]
[rank0]:         record event failed, runtime result = 507009[FUNC:ReportCallError]
RuntimeError: capture_end:../torch_npu/csrc/core/npu/NPUGraph.cpp:247 NPU function error:
    c10_npu::acl::AclmdlRICaptureEnd(capture_stream_, &model_ri), error code is 507903
[rank0]:         rtStreamEndCapture execution failed, reason: an error has invalidated the stream capture sequence
```

The title usually names the RIC/task-group API itself (`graph_task_group (RIC) +
AIV causes NPUGraph capture failure`), the `HCCL_OP_EXPANSION_MODE=AIV` export is
in the launch block, and the same body reports a **replay** deadlock: the first
request sits at `Replaying aclgraph` with `AIVector` 8–16 % / `AICore` 0 %, then the
300 s `shm_broadcast` watchdog kills it. Recorded #13688 (v0.22.1rc1 image on
910C ×8, GLM-4.6V TP8+EP8).

Not this page: the `107020` `event wait timeout` at `profile_run` with
`aclnnMoeDistribute{Dispatch,Combine}V4` (`aiv-mc2-env-ablation.md`), a
`graph_task_update_end`/`AclmdlRICaptureTaskUpdateEnd` failure
(`aclgraph-full-decode-runtime-crash.md`), or an `EI0007 …[qp]` refusal at capture
(`qp-resource-exhausted-at-graph-load.md`).

## The two codes, one line each

| code | name | anchor |
|---|---|---|
| `507009` | `ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT` — the TS refused a **task type** inside the capture window | `cann/runtime/include/external/acl/error_codes/rt_error_codes.h:91` |
| `507903` | `ACL_ERROR_RT_STREAM_CAPTURE_INVALIDATED` — the stream was already poisoned, so `EndCapture` fails | `…/rt_error_codes.h:144` |

So `507903` is the consequence; the cause text is `rtEventRecord … task not
supported` + `507009`. Both are runtime verdicts, so the owner of the *capture*
failure is torch_npu/CANN, not vllm-ascend — and "detect AIV and raise clearly"
is new code, see step 3.

## Step 1 — inventory the tree's task groups before crediting the reporter's repro

The RIC task group exists for one purpose: rebinding replay-time **host metadata**
from the update stream for a single dispatched op
(`vllm_ascend/ops/mm_encoder_attention.py:20-22`, "…so replay-time host metadata
can be rebound from the update stream, matching the LLM full-graph pattern"). The
capture skeleton is `torch_npu/npu/graphs.py:196-222` — `ExternalEvent()` →
`event.wait/reset` → `graph_task_group_begin` → op → `graph_task_group_end` →
dispatch record (`graph_task_group_begin/end` themselves at `:81`, `:85`) —
and it only fires for ops registered in `_NPU_GRAPH_OP_HANDLERS`.

One call enumerates every region vllm-ascend builds
(`rg -n 'graph_task_group_begin' vllm_ascend/`). At v0.22.1rc1 (`5f6faa0`) the
ten regions all wrap **one compute op**, never a collective:

| region | op inside |
|---|---|
| `attention/attention_v1.py:800`, `:897` | `npu_fused_infer_attention_score{,_v2}.out` |
| `attention/attention_v1.py:968` | `torch_npu._npu_paged_attention` |
| `attention/mla_v1.py:1582` | `npu_fused_infer_attention_score_v2.out` |
| `attention/context_parallel/attention_cp.py:667`, `mla_cp.py:735` | FIA `.out` |
| `ops/mm_encoder_attention.py:316` | ViT `npu_fused_infer_attention_score` |
| `ops/gdn.py:476`, `:592` | `torch.ops._C_ascend.npu_causal_conv1d_custom` |

**The rule that decides the answer:** a hand-built matrix over
`graph_task_group_begin` + `dist.all_reduce` (recorded #13688's four rows:
baseline PASS, `tg_no_hccl` PASS, `tg_allreduce` FAIL, `allreduce_tg` PASS)
proves that torch_npu/CANN cannot put an AIV collective in a task group. It does
**not** prove the served failure travels through that pair, because the serving
stack never builds it. Deliver them as two findings with two owners — the
torch_npu/CANN `507009` for the API pair, and the serving lane below for the
deadlock — and say plainly that the minimal repro is not the serving path rather
than letting the tidy matrix close the serving lane.

## Step 2 — the serving deadlock is the MC2 × AIV family

On an A3/910C MoE (or MoE-in-VL) serve the batch selector decides the comm op per
step, `vllm_ascend/ascend_forward_context.py:233` (`select_moe_comm_method`); the
A3 branch at `:283-305` takes FUSED_MC2/MC2 whenever `num_tokens <=
mc2_tokens_capacity`, else ALLTOALL — so a decode-heavy GLM/DeepSeek serve with
capture sizes ≤ 16 is on MC2 by construction. AIV switching collectives to
AI-vector scheduling inside that kernel is the recorded hang family
(`aiv-mc2-env-ablation.md`, mechanism stated on #3881: AIV comm ops contend with
the other parallel AIV ops for the vector cores). The paste's `AIVector` 8–16 % /
`AICore` 0 % with no error is consistent with that spinning kernel and not with
the RIC pair.

**There is no force-ALLTOALL switch at this pin**, so do not plan a diagnostic
that flips MC2 off by env: the only MC2-variant knob is
`VLLM_ASCEND_ENABLE_FUSED_MC2` (`vllm_ascend/envs.py:102`, 0/1/2, described at
`:95-101`). Selection is capacity arithmetic; the honest ablation is AIV unset,
then `--enforce-eager` (which removes capture *and* replay, so a serve that then
works is replay-side).

## Step 3 — if the body blames a tree auto-enable, read its gate first

vllm-ascend does set the variable, but only for one model family:
`vllm_ascend/patch/platform/patch_minimax_m2_config.py:110-127` runs inside
`_patched_verify_cuda_graph` and sets `HCCL_OP_EXPANSION_MODE=AIV` when unset
**and** `current_platform.device_name == "npu"` **and** `_get_model_type(self) ==
"minimax_m2"` (`:38-52`) **and** `not enforce_eager`. A GLM/Qwen/DeepSeek
reporter's AIV therefore came from their own `export`, and "the patch should be
reverted" changes nothing for their run — correct that in the answer, keep the
concern for MiniMax-M2 (which also selects MC2), and propose a conditional/warned
variant rather than removal.

The same read settles the "why does the tree recommend AIV with graph mode" half:
`tests/e2e/pull_request/two_card/aclgraph/test_single_request_aclgraph.py:46` and
`tests/e2e/pull_request/two_card/test_qwen3_performance.py:60` set it, while
`test_shared_expert_dp.py:21-22` deletes it ("FlashComm v1 / shared-expert-DP
require it to be unset") — so the pairing is model/path-specific in the tree's own
tests, not a blanket recommendation.

## Step 4 — torch_npu cannot detect AIV, so that ask is a feature request

`rg -n HCCL_OP_EXPANSION_MODE` over the whole `Ascend/pytorch` export
(v26.0.1-pytorch2.10.0, `8751b36`) returns **zero** hits in Python and in C++ alike
— the env name is unknown to torch_npu. The mode lives in HCCL's communicator
config, visible in that export only through the vendored headers — `HCCL_COMM_CONFIG_OP_EXPANSION_MODE = 3`
(`third_party/hccl/inc/hccl/hccl_types.h:156`), default
`HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0` (`:125`), the only writer being
`config->hcclOpExpansionMode = HCCL_COMM_DEFAULT_OP_EXPANSION_MODE`
(`third_party/hccl/inc/hccl/hccl.h:438`). torch_npu neither reads the env nor
queries the communicator, so a capture-time AIV rejection needs new API in
torch_npu/CANN; a vllm-ascend-side warning can at most mention the variable when
the selected MoE path is MC2.

## Ladder (cheapest first)

1. Keep `HCCL_OP_EXPANSION_MODE` unset — the reporter's own working row; treat the
   replay deadlock as the priority, not the capture error.
2. Same command with `--enforce-eager` to place the fault: a serve that works
   removes both capture and replay, i.e. the replay-side lane (step 2); a capture
   failure that survives eager is a different report.
3. File the task-group item with torch_npu/CANN using the matrix, with the
   `allreduce_tg` PASS row as the proof of the *wrapping*, and ask for a
   capture-time rejection or an AIV-compatible task type.
4. Optional vllm-ascend guard: warn when `HCCL_OP_EXPANSION_MODE=AIV` and the
   selected path is MC2, in the same env-aware hook as the MiniMax gate. A guard
   that only silences the exception is not a fix.

## Gaps to state, not fill

- `torch_npu/csrc/core/npu/NPUGraph.cpp` — the file the paste's `capture_end`
  frame names — is **not in the export**; only `NPUGraph.h` and the pybind layer
  (`torch_npu/csrc/npu/Graph.cpp:136-142`, `_graph_task_group_begin/end`) are, so
  no local read can show what `AclmdlRICaptureEnd` did.
- `cann/runtime` supplies the code names only; the refusal is minted in the closed
  TS/driver.
- Ask for the plog at the replay second (AIV/TS task lines) and `npu-smi info`
  during the hang; the `AIVector`/`AICore` percentages come from the reporter's
  screenshot, not from a log line.
- Whether a later release fixes the AIV × task-group pair is invisible from the
  snapshot: state that as the version gap instead of implying the next rc does.
