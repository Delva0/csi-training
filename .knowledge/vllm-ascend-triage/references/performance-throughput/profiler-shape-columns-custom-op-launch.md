# kernel_details.csv with `N/A` shapes for one custom op: the launch lane, not `SetCustomHandler`

## Scope

The evidence is a profiler table whose *metadata* is missing for one row while the
neighbours are complete: `kernel_details.csv` prints `Input Shapes: N/A`,
`OP State: N/A`, `Block Dim: 0` for one op (e.g. `mla_preprocess` /
`mla_preprocess_0_mix_aic`, 2028 calls, ~103 µs each, cube utilisation normal)
between an `RmsNorm` row with `102,6144;6144` and a `MatMulCommon` row with
`102,6144;128,6144`. It arrives almost always as a *self-diagnosed* claim: the op
uses `OpCommand::SetCustomHandler`, therefore it bypasses `aclopCompileAndExecute`,
therefore GE registers no tensor desc and msprof has nothing to print — plus a
proposed patch that reports tensor info around the lambda with `aclprofRangePushEx`
and asks the project to adopt it.

Deliverable: whether that causal link holds at the pin, which launch layer actually
carries the metadata, whether the proposed API exists in the reporter's CANN, and the
ranked cheap actions. Not a kernel or throughput walk: nothing here claims the op is
slow or numerically wrong.

Siblings: `profiler-opcount-evidence.md` is the *count* question (collection health,
window, step normalisation, NPU-vs-GPU buckets) and its rule 1 still applies to any
collection that printed an error; this page is the *missing column* question. An op
whose table is fine but whose numbers disagree with the service is the parent
`index.md`.

## Rules

1. **`SetCustomHandler` is not the discriminator — read the macro before accepting
   the chain.** At vllm-ascend v0.18.0 (`e18643f8a4d5`) the project's own gated ops
   reach `OpCommand` through `EXEC_NPU_CMD`, and that macro ends in
   `cmd.SetCustomHandler(acl_call); cmd.Run();`
   (`csrc/aclnn_torch_adapter/op_api_common.h:523-600`, the site at `:584`). So
   "customHandler ⇒ bypasses GE ⇒ no shape" is refuted by the tree the reporter is
   quoting: the ops with shapes take the same branch. What differs is what the
   lambda *invokes* — an aclnn op API (`aclnnXxxGetWorkspaceSize` + `aclnnXxx`,
   which builds an `aclOpExecutor` and registers the tensors) versus a bare AscendC
   kernel launch (`mla_preprocess_impl(...)` at
   `csrc/mla_preprocess/mla_preprocess_torch_adpt.h:124-135`; its host header keeps
   the commented-out `// #include "aclrtlaunch_mla_preprocess.h"`,
   `op_host/mla_preprocess.h:24`). The two-branch `OpCommand` snippet the reporter
   pastes is *real and used* — `OpParamMaker.cpp:199-200` (`params.customHandler()`)
   returns before `:227` (`aclopCompileAndExecute`) in
   `Ascend/pytorch/torch_npu/csrc/framework/` at the torch_npu pin — but both ops in
   question take the first branch, so it does not decide the case. Say exactly that:
   correct snippet, wrong middle link.

2. **One script decides the lane.** `bash <skill_root>/scripts/op-shape-lane.sh
   <vllm-ascend-root> <op> [<op> …]` prints the `CUSTOM_OPS`/`CUSTOM_OPS_ARRAY`
   build lists from `csrc/build_aclnn.sh` (`:28` A2 and `:45-64` A3 at v0.18.0) and,
   per op, whether the op's own adapter calls `EXEC_NPU_CMD` or only
   `SetCustomHandler`. An op in a build list has an `aclnn<Op>` entry point built by
   `bash build.sh -n "$CUSTOM_OPS"` and is reached as `EXEC_NPU_CMD(aclnn<Op>, …)` in
   `csrc/torch_binding.cpp` — `aclnnAddRmsNormBias`
   (`add_rms_norm_bias_torch_adpt.h:49`), `aclnnGemmaRmsNorm` (`torch_binding.cpp:581`),
   `aclnnTransposeKvCacheByBlock` (`:596`), `aclnnCausalConv1d` (`:650`) — and those
   rows carry shapes. `mla_preprocess` is in **no** build list at v0.18.0, so there is
   no `aclnnMlaPreprocess` to call: "just use the aclnn entry point" is a build-list
   change plus an op_api registration, not a call-site change. Quote the script's
   verdict as the check — its `raw-kernel-lane` line names exactly this lane, and read
   its discriminator clause (the adapter never calls `EXEC_NPU_CMD`, so the lambda
   launches the kernel itself) rather than its `SetCustomHandler` mention: quoting
   "customHandler ⇒ no shape" back at the reporter re-installs the middle link rule 1
   refutes on the same box (the recorded #9026 replay had to reconcile the two in-flight
   and said so in its reasoning before shipping rule 1's framing).

3. **The other customHandler sites show the same gap; fix it once.** At v0.18.0:
   `csrc/torch_binding.cpp:192,247,293,333,380` (LoRA/SD ops) and
   `csrc/batch_matmul_transpose/batch_matmul_transpose_torch_adpt.h:43`. Ask which
   rows of the reporter's table are `N/A` before offering a per-op patch; a shared
   helper is the answer if several are.

4. **The CANN tensor-info record is small and name-keyed, and that bounds what the
   fix can promise.** In the exported runtime
   (`cann/runtime/pkg_inc/profiling/aprof_pub.h`, export `0b16a5e`):
   `MSPROF_GE_TENSOR_DATA_NUM 5` and `MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8` (`:34-35`),
   `struct MsprofTensorInfo { uint64_t opName; uint32_t tensorNum;
   MsrofTensorData tensorData[5]; }` (`:322-326`), node type
   `MSPROF_REPORT_NODE_TENSOR_INFO_TYPE 1` (`:82`), public entry `MsprofReportApi`
   (`:508`). Two consequences, both to be *tested*, not assumed: (a) `opName` is a
   hash, so the row/record join is plausibly by op name — one report could cover all
   2028 rows; (b) a 26-tensor op cannot fit one five-slot record, so the first
   experiment reports two to four shape-bearing tensors (`hidden_states`, one weight,
   `kv_cache`) and reads the table, instead of pasting the reporter's full macro and
   debugging a mismatch.

5. **Whether the aclprof tensor-info API exists is a CANN-version fact and is not
   answerable from a truncated export.** Verified here: `cann/runtime`'s
   `include/external/acl/acl_prof.h` ends at `aclprofMarkEx` (`:532`) — no
   `aclprofRangePushEx`, no `aclprofEventAttributes`, no `aclprofTensorInfo` — and
   those symbols have **zero** hits in any repo of the snapshot, while
   `cann/asc-devkit`'s export contains **no `.asc` files at all** (`files=1484/4419`,
   `content_export`). State the coverage and never "the API does not exist"; hand the
   reporter two greps on their own box:
   `grep -n "RangePush\|TensorInfo\|EventAttributes" $ASCEND_TOOLKIT_HOME/include/acl/acl_prof.h`
   and the same under the `cann-<ver>/include/external/acl/` copy. Their report names
   CANN 8.5.0; the export is untagged.

6. **`Block Dim: 0` and `OP State: N/A` are different records.** The same header
   carries `node_basic_info`, `op_info` and `launch` types; do not promise that a
   tensor-info report fills them. Check the table after the experiment and say which
   column moved.

7. **Graph capture changes when the lambda runs.** If decode runs under an ACL graph
   (`FULL_DECODE_ONLY`/`PIECEWISE`), the customHandler executes at capture, so a
   per-launch report lands once — unless the join is by op-name hash (rule 4a). The
   first thing to read in the post-fix table is therefore *how many rows* gained
   shapes.

## Answer shape

Ranked, cheapest first:

1. State the premise gap (rule 1, with `path:line`) and settle the API surface on the
   reporter's own box (rule 5) *before* any patch; running the CANN `add_custom`
   sample under `msprof` on that host answers whether the aclprof tensor-info route
   reaches `kernel_details.csv` at all.
2. Report a few tensors (rule 4) from one shared helper gated on "profiling active"
   (the op is on the decode hot path: 2028 calls, ~103 µs), then read rule 7.
3. If the reporting route is unavailable or does not fill the column, the build-list
   change of rule 2 makes the op identical to the rows that already have shapes —
   price the host-side `GetWorkspaceSize` cost on that hot path explicitly.
4. Zero-risk fallback: the shapes are deterministic from the model config
   (`hidden_dim`, tp, decode batch), so a name→shape map in the consumer
   (msmodeling-side) needs no CANN API and survives upgrades.

Ask with the answer: the profiler command and CANN version, which other rows are
`N/A` (rule 3), and whether the collection ran under graph capture.

## Worked case

**#9026** — vllm-ascend v0.18.0 (`e18643f8a4d5`), CANN 8.5.0, Atlas A3,
GLM-5 W8A8 (`hidden_dim=6144`), PD + MTP (`num_speculative_tokens=2`),
`VLLM_ASCEND_ENABLE_MLAPO=1`. The recorded run verified the adapter site and both
`OpCommand` branches, then found the reporter's premise refuted at
`op_api_common.h:584`, `mla_preprocess` absent from every `CUSTOM_OPS` list, the
5 × 8 tensor-info limits, and `acl_prof.h` stopping at `aclprofMarkEx` — and then
ended `status=output_truncated` (`output_tokens`) while still drafting the ranked
answer. The investigation had the facts and no delivery: put the verdict sentence in
the first message and keep the enumeration for later turns
(`../evidence-workspace/delivery-discipline.md`).

Route: parent `index.md`; count/window questions `profiler-opcount-evidence.md`; the
lane script's flags `../first-checks/scripts-catalogue.md`.
