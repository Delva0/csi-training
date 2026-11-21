# `… NPU function error: call aclnnX failed, error code is N` can be a stale thread-latched code

Scope: a serve that ran healthily for minutes or hours dies in an ordinary `torch_npu`
op, the traceback ends in the PTA macro format

```text
RuntimeError: npu_quant_matmul:../third_party/op-plugin/op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp:297
NPU function error: call aclnnQuantMatmulV5 failed, error code is 107030
```

and the **same code appears in a plog record from an earlier phase** — classically the
startup graph-capture window (`Not_Supported(EE1016): Synchronous memory copy failed …
the current thread … is in the capture state … capture mode is GLOBAL(0)`, "nearly two
hours earlier"). Recorded shape: vllm-ascend v0.23.0 (`5cb98caa`) + vllm v0.23.0
(`0fc695fc`), FULL_DECODE_ONLY + `enable_npugraph_ex`, GLM-5.3 w8a8 SFA decode node
under PD, `deepseek_mtp`; all four workers of a DP rank die together.

Not this page: a device dump / `chipId` / `fault kernel_name=` (topic `index.md`); a
pre-launch argument refusal (`../custom-ops-runtime-faults/index.md`, `EZ1001` family);
a host-scalar readback frame (`LocalScalarDenseNpu`, `aclgraph-full-decode-runtime-crash.md`
§7 — there the code reports the model stream's death); a failure *during* capture
(`../startup-failures/acl-graph-capture.md`).

## 1. Read the macro before the op: the printed code may not be this call's status

Verified in the PTA export at `Ascend/pytorch` v26.0.1-pytorch2.10.0 (`8751b36d`),
`torch_npu/csrc/core/npu/NPUException.h:121-200` — `CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(err_code)`
and `NPU_CHECK_ERROR_CHECK_UCE(err_code, check_uce=true, …)`:

1. `int error_code = err_code;` — the expression's own return.
2. `error_code_peek = c10_npu::acl::AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);` and **if it
   is non-zero it replaces `error_code`** (`:124-127`, `:149-152`).
3. `device_error_msg = c10_npu::getDeviceErrorMessage(error_code)` (`:128`, `:153`); when that
   is empty the text falls back to the static `#err_code` string (`:166`, `:194`) — which is why
   the message reads "call aclnnQuantMatmulV5 failed" (this call's own words) next to a code
   that this call did not produce.

`NPU_CHECK_ERROR(err_code, …)` is the `check_uce = true` wrapper (`:212`);
`NPU_CHECK_ERROR_WITHOUT_UCE` (`:211`) skips the peek, and `OPS_CHECK_ERROR` (`:214+`)
guards `CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE` with an outer non-zero test. The
**same `NPU function error: … , error code is N` wording comes out of all of them**, so the
message alone does not say whether the code was overridden — decide from the code's own
phase validity (§2), never from the wording.

Vendor semantics, in the exported CANN docs (`cann/runtime` `0b16a5e9`):

| API | Fact |
|---|---|
| `aclrtPeekAtLastError` (`cann/runtime/docs/api_docs/aclrtPeekAtLastError.md`) | "获取当前线程的Runtime错误码，**仅获取但不清空**错误码"; thread level only |
| `aclGetRecentErrMsg` (`…/aclGetRecentErrMsg.md`) | returns **and clears** the failure description; per-process or per-thread by `aclInit`'s `err_msg_mode` (**default thread**, `aclInit.md:280`); several messages accumulate, newest first; the doc warns that not reading it after each failure lets messages "堆积、丢失" |

So one thread keeps a latched runtime code plus a description until something reads or
clears them, and **any later failure on that thread prints the latched code instead of its
own**. A capture-window refusal is the extreme case because the thread that captured is
usually the thread that later steps the engine.

## 2. Three consequences that change the answer

- **The code is not the op's status; judge it by whether this op could mint it.**
  `107030` is `ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT`
  (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:51`) — mintable *only*
  while a stream is in capture. On a serve that has been decoding for two hours it cannot be
  this step's error, so the number is stale even though the timestamp gap proves nothing by
  itself (rule 13). Cross-check the family with
  `bash <skill_root>/scripts/cann-error-lookup.sh <code>` before ranking anything on it.
- **The op did fail, and the status that would name its cause is gone.** The wrapper enters
  only when the expression returned non-zero (`NPUException.h:146`), so the reporter's
  "a stashed record was popped later" story is half right: there *is* a real failure in the
  aclnn call, but the printed code and text were overwritten by the latch. Say this
  explicitly — it inverts the priority, because the incident is then the plog block at the
  failure timestamp, not the capture record the report is built around.
- **There is no supported flush.** `clear_device_error_info()` is called only from
  `THNPModule_restart_device_wrap` (`torch_npu/csrc/npu/Module.cpp:846`, call at `:861`,
  binding at `:2339`), i.e. `torch_npu.npu.restart_device` →
  `_npu_restart_device` (`torch_npu/npu/_recovery.py:57-63`) — a device-recovery API, and the
  call is conditional on `ShouldAppendDeviceErrorVerbose()`. A request of the form "clear the
  deferred error queue after graph capture" has no runnable rung on this pin; the honest
  answer is that the lever is *attribution* (blocking mode, plog, error-reporting mode), and
  the flush is a PTA-side change request.

## 3. Why the capture window mints `107030` at all — and why mode ablations miss it

The custom-op host adapters write their per-batch tiling table with a **blocking H2D copy on
every call**: `static global_tiling_data = at::empty({tilingSize * MAX_…})` and then
`aclrtMemcpy(…, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE)`, guarded
only by an index range — `csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.h:105-112`,
`csrc/mla_preprocess/op_host/mla_preprocess.h:723-730` (v0.23.0). Inside a capture region the
sync copy is refused, so every captured execution of either op mints one `107030`. The SFA
decode path reaches `batch_matmul_transpose` once per layer per step while
`num_input_tokens <= BMM_TRANS_MAX_SUPPORTED_TOKENS` (`vllm_ascend/attention/sfa_v1.py:81`
(1024), `:1352-1357`), which is exactly why the pinned tree's own known-issue bullet names
these two ops (`docs/source/user_guide/release_notes.md:143` at v0.23.0, "#14347 … the
reported error does not affect inference").

Two operational readings:

- **The record follows the op, not the mode.** Any mode that captures a step containing the
  op mints it, so `cudagraph_mode` / `enable_npugraph_ex` ablation is only a discriminator if
  the paste shows the record **count** before and after; the one run that cannot mint it is
  `--enforce-eager` (with its own line proving capture was skipped,
  `../startup-failures/compile-backend-banner-then-silence.md`).
- **The record count names the op when the plog block is not at hand**: once per bucket ⇒ a
  per-forward host adapter (`mla_preprocess`); once per layer per bucket ⇒
  `batch_matmul_transpose` in `_v_up_proj`.

## 4. Deliverable and asks

First paragraph: which code owns the printed number (the thread latch), that the fatal
failure's own status is masked, and the one artefact that answers it — the failing rank's
plog at the failure second (first `[ERROR]` / `[E…]` line in that second, `±50` lines),
not the capture block. Then:

1. `ASCEND_LAUNCH_BLOCKING=1` where the release allows it (refused while a graph mode is on:
   `ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1`,
   `platform.py:457-469` at v0.20.2rc1) to re-attribute the real failure to its op.
2. The same request body on the pinned launch shape (`rg -ln -i '<model>' tests/e2e/*/*/configs/`)
   before any mechanism theory about the masked call.
3. Whether the error coincides with a re-capture, a bucket change or a spec-decode step count
   — the latch also fires on a *fresh* capture-window refusal, so a first-request death after
   a recompile is a different reading (rule 13's "this process's own window").

Label the claim honestly: the macro and the two API semantics are source/documented-verified;
which op's failure was masked is **not** recoverable from the paste, and `107030`'s
"harmless" status in the release notes is consistent with this mechanism (it is a
diagnosability defect, not the corruption the reporter suspects).

## 5. Gaps at this snapshot

- `getDeviceErrorMessage` (the per-code cache that lets a description keep its original
  timestamp) is declared at `NPUException.h:333` but its `.cpp` is not in the export.
- `Ascend/op-plugin` exports 107 files with **no** `ops/opapi/` and no `op_api_common.h`
  (`SOURCES.json` `files: 107`), so the op-plugin copy of this macro inside the reporter's
  `QuantMatmulKernelNpuOpApi.cpp:297` cannot be read: do not spend calls hunting it, quote
  PTA's identical message shape and say which copy you verified.
- Line numbers above are the v0.23.0 / PTA v26.0.1-pytorch2.10.0 pins; re-grep
  `AclrtPeekAtLastError` in `NPUException.h` on a newer tag before reusing the anchors — the
  `check_uce` argument and the branch layout move.
