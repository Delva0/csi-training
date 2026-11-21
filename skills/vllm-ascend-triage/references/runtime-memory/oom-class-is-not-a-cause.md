# `torch.OutOfMemoryError` whose text is a sync/stream/timeout code

## Report shape

The traceback class is `torch.OutOfMemoryError`, but the message is not an
allocation:

```text
torch.OutOfMemoryError: synchronize:.../torch_npu.dir/compiler_depend.ts:47 NPU function error:
  c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream), error code is 107020
```

often beside `EngineDeadError` on the request path and an `event wait timeout during
task execution` / `task_id=… event_id=… timeout=1866s` block from the driver. The
report usually argues the *classification* is wrong ("this is not an OOM, it is a
stream-state error") and sometimes ships a patch proposal on top of that argument;
recorded #13218 (vllm-ascend v0.23.0rc1 `f4a08bdd`, vllm v0.23.0 `0fc695fc`) proposed
a replay-path twin of `_is_stream_resource_capture_error` in
`vllm_ascend/compilation/acl_graph.py`.

## Where the class comes from (verified, Ascend/pytorch v26.0.1-pytorch2.10.0)

`torch_npu/csrc/core/npu/NPUException.h:192-204`, inside `NPU_CHECK_ERROR`'s generic
`else` branch:

```c
auto retmsg = std::string(__func__) + ":" + __FILE__ + ":" + std::to_string(__LINE__) +
    " NPU function error: " + (device_error_msg.empty() ? getErrorFunction(#err_code, …) : device_error_msg) +
    ", error code is " + std::to_string(error_code) + PTA_ERROR(ErrCode::ACL) + … + "\n" +
    c10_npu::c10_npu_get_error_message();          // :196-198  ← ACL's RECENT-error buffer
if (c10_npu::isCannOOM(retmsg)) {                  // :199      ← the only class switch
    ASCEND_LOGE("%s", retmsg.c_str());
    TORCH_CHECK_WITH(OutOfMemoryError, false, retmsg.c_str());
}
TORCH_CHECK(false, retmsg);                        // :206      ← otherwise a plain RuntimeError
```

Consequences, in the order they change an investigation:

1. **The class is a text predicate over a string that includes stale text.**
   `c10_npu_get_error_message()` is ACL's *most recent* error message (declared
   `NPUException.h:307`; `isCannOOM` is declared at `:337`), not this call's error. An earlier
   `Memory_Allocation_Failure(EL0004)` / `alloc device memory failed, runtime result =
   207001` still sitting in that buffer rides into a later, unrelated failure's retmsg and
   satisfies `isCannOOM` — which is why the misclassification appears *after* an
   allocation failure and not on a clean run. `isCannOOM`'s own body is not in the
   content export (`torch_npu/csrc/core/*.cpp` absent), so the exact substring is a
   hypothesis: `strings $(python -c 'import torch_npu,os;print(os.path.dirname(torch_npu.__file__))')/_C*.so
   | grep -i 'out of memory\|alloc device memory'` is the one check.
2. **vLLM does not mint it and does not branch on it.** vllm v0.23.0 mentions
   `OutOfMemoryError` in exactly two files (`v1/worker/gpu_model_runner.py`, a p2p NCCL
   engine); nothing in `v1/engine/core.py`, `core_client.py` or the multiproc executor
   keys on the class, and vllm-ascend never mentions it. So "vLLM's OOM path logs
   `--gpu-memory-utilization` hints" is unsupported at the pin — there is no such path.
   Re-raised as plain `RuntimeError` the message is *better*, but no upstream behaviour
   changes in either direction (nothing to regress, nothing to fix).
3. **`107020` is a task timeout, not a synchronize timeout.** From
   `third_party/acl/inc/acl/error_codes/rt_error_codes.h`: `107019`
   `ACL_ERROR_RT_WAIT_TIMEOUT`, `107020` `ACL_ERROR_RT_TASK_TIMEOUT`, `107027`
   `ACL_ERROR_RT_STREAM_CAPTURED`, `207001` `ACL_ERROR_RT_MEMORY_ALLOCATION`. The
   `AclrtSynchronizeStreamWithTimeout(stream)` in the message is merely the API whose
   return carried the code. So correct the report's wording before quoting it: a 107020
   paste shows the stream was still waiting on a *task*, and the `event wait timeout …
   timeout=1866s` line is the driver naming it.
4. **It is fatal by design on the torch_npu side.** `torch_npu/utils/_error_code.py:104`
   (`npu_timeout_exception = "error code is 107020"`, sleep `3` at `:105`) makes the
   process linger briefly so the elastic agent can reap it. Do not propose a
   catch-and-retry design for 107020; "restart vLLM / reset the card after aborted
   requests" is the honest mitigation.

## Ranked owners

1. **Something earlier killed the stream and the sync is the symptom.** The skill's
   `../startup-failures/captured-stream-host-sync.md` §6 already rules this for the bare
   `107027`: `ASCEND_LAUNCH_BLOCKING=1` makes `OpCommand.cpp:241-242` sync after *every*
   op, so any op failure surfaces as a sync error carrying whatever code killed the
   stream. Read *up* the log for `207001` / `Memory_Allocation_Failure` / a plog
   `[AIC_INFO] fault kernel_name=` line: a real device fault names its kernel, an idle
   stream does not.
2. **A genuinely pending task that timed out** (a hung collective, an MC2 dispatch that
   never completed). Same evidence direction: the plog timeline
   (`scripts/plog-timeline.sh`) and the `task_id`/`event_id` ids from the driver block.
3. **The wrapper's own hash/ordering barrier** when the deployment is async-scheduling
   or multi-threaded: `acl_graph.py`'s replay path syncs before replay by comment
   ("so that `update_attn_params` only executes after the previous graph replay has fully
   completed"), gated on `runtime_mode == FULL`, `not is_draft_eagle` and
   `not self.enable_enpu`. That gate tells you whether the sync even ran on this launch —
   check it before blaming the call site.

## Do not fold the codes together

A proposed detector of the form "code ∈ {107020, 107027} **and** `synchron` in the
message" over-matches by construction: (a) `synchron` is in every
`AclrtSynchronizeStreamWithTimeout` message, i.e. in any op failure under a blocking
launch; (b) `107027` `STREAM_CAPTURED` is the *capture-phase* lane whose owner is a host
readback inside the captured region (`.item()` / `.cpu()` / `bincount`), a different fix
from a timeout. Key the predicate on `107020` alone and treat 107027 as its own branch;
also mirror the loosest part of the existing helper with care — `:42` returns true on the
marker alone, without the code.

## Also, the gap the patch proposal should close

`rg -ln 'stream.resource' tests/` at v0.23.0rc1 is **empty**: the existing 207008 capture
helper has no test, and the docs site that carries its guidance is
`docs/source/user_guide/feature_guide/graph_mode.md:274` (plus
`developer_guide/Design_Documents/ACL_Graph.md` and `faqs.md:212`). The helpers are pure
string predicates, so a unit test needs no NPU — that is the cheapest contribution to
name.

## Ask for

The full worker log tail from `Starting to load model` to the failure (not just the
traceback), the plog slice with `[AIC_INFO] fault kernel_name=` around the same second,
the serve command verbatim, and `python -c "import vllm, vllm_ascend; print(vllm.__version__,
vllm_ascend.__version__)"` — a version block that names an older vLLM than the
vllm-ascend release pairs with invalidates every line number in the reporter's claim.
