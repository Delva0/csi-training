# HCCL / ACL / NPU device faults

Scope: failures whose text names a collective, a device error category or the
runtime stack rather than a Python-level cause — `HcclAllreduce`, `hcclRet -> N`,
`AllReduceOpbasePipeline`, `ERR00100 PTA call acl api failed`, `ERR01100 OPS call
acl api failed`, `the current working operator name is HcclAllreduce`,
`RuntimeError: SUSPECT REMOTE ERROR` / `error code is 507057`, `EZ9999` device
dumps, collective hangs/timeouts during `profile_run` or
`determine_available_memory`, and processes killed by a device fault.

Inherit the rules in SKILL.md. **`hcclRet -> N` / `hccl_result = N` is the
`hcclResult_t` enum**: `0` SUCCESS, `1` PARA, `2` PTR, `3` MEMORY, **`4`
INTERNAL**, `5` NOT_SUPPORT, `6` NOT_FOUND, `7` UNAVAIL, `8` SYSCALL, `9`
TIMEOUT, `19` NETWORK, `20` AGAIN, `21` REMOTE
(`cann/hccl/docs/api/context/HcclResult.md:10-35`), so quote the enum name, not
the number. A code that is neither `1`, `3` nor `9` is a
generic rejection: the surrounding message text names the precondition, the code
does not. The one-sided `HcclBatchPut` / store-put variant of that argument is
`../kv-pool-store/hccl-onesided-store-put.md` §1.

Order matters more than depth here:

**First separate the watchdog from the collective.** If the only failure text is a
bare `TimeoutError` whose frames are `multiproc_executor.py` (`collective_rpc` /
`get_response`) and `shm_broadcast.py` (`acquire_read` / `acquire_write` /
`recv`), it is not a collective error at all: it is vLLM's own RPC watchdog
(`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, 300 s) reporting a worker that stopped
answering. `rpc-watchdog-timeouts.md` classifies the direction, the configurations
(DP>1 + EP) that turn one stalled rank into a whole-group stall, and the per-rank
artefacts to request. One call decides it:
`bash <skill_root>/scripts/executor-timeout-check.sh <paste|->`. Recorded #1865 was
titled 卡间通讯超时 ("inter-card communication timeout") for exactly this text.

**Then separate a documented env-knob ablation from the mechanism.** If the title
or narrative is "removing `HCCL_OP_EXPANSION_MODE` / `PYTORCH_NPU_ALLOC_CONF` lets
it start" and the first in-window error is `aclnnMoeDistributeDispatchV4` /
`...CombineV4 ... 107020` with `event wait timeout ... task_timeout`, the enabled
knob changed the selected MoE communication path (AIV × MC2) — a known defect
family, not a cabling fault: `aiv-mc2-env-ablation.md` has the single-variable
ablation matrix, the rank arithmetic that says whether MC2 was even selected, and
the four precedents (#14928, #13688, #9447, #15195).

**Then separate an MC2 timeout the reporter did *not* ablate.** If the paste dies
inside `profile_run` / `determine_available_memory` with `507014` "The aicore
execution times out", the deepest Python frame is `aclnnInplaceCopy` or an
`ERR02005` HCCL-watchdog abort, and the `[AIC_INFO] fault kernel_name=` line names
a **MoE all-to-all** (`MoeDistributeDispatchV2_…`), the dispatch is the first
collective of the process and never returned. The trap is the arithmetic: the
batch that runs there is `mc2_tokens_capacity`, not `--max-num-batched-tokens`,
because `profile_run` runs an extra `_dummy_run(capacity, with_prefill=True)`
(`worker/model_runner_v1.py:2202-2208` at v0.13.0rc1). `mc2-dispatch-profile-run-hang.md`
has the capacity/`global_bs` arithmetic, the `HCCL_INTRA_PCIE_ENABLE=1 /
HCCL_INTRA_ROCE_ENABLE=0` → hierarchical-dispatch gate that is the *documented*
A2 setting (do not rank it as the misconfiguration), the tutorial diff that closes
the flag lane, and the ladder.

**Then separate a bind/port line from the incident.** If the title or the paste
carries `Communication_Error_Bind_IP_Port(EI0019)` / `port <n> have already been
bound`, check the record's date first (rule 13) and the number second: a port above
65535 cannot be an OS socket at all, so the "already been bound" reason is HCCL's
own control-plane arithmetic — base `HOST_CONTROL_BASE_PORT = 60000`,
`HCCL_IF_BASE_PORT` accepted `[1024, 65520]`
(`cann/hcomm/src/pub_inc/hccl_common.h:170`,
`src/platform/common/externalinput.cc:213,828-853`). `control-plane-port-binding.md`
has the levers (`HCCL_IF_BASE_PORT` + `ip_local_reserved_ports`, stale processes) and
the rule that it is a finding only when it is in-window — in #12487 the title named
that line while it was a three-day-old re-print and the in-window error was an AICPU
`reduce_scatter` param-invalid on a chip the reporting rank did not own.

**Then separate a signal death from a collective error.** `!!!!!!! Segfault
encountered !!!!!!!` plus `Worker proc VllmWorker-<n> died unexpectedly (exit
code: None)` is a rank killed by signal: no error code, no Python exception, and
the peer ranks' gloo `all_reduce` in `_sync_metadata_across_dp` (`Connection
closed by peer`) is only the messenger. A client-side `ConnectionRefused` from
the benchmark harness is secondary as well. Read the dump's direction — top
frames are the last native calls (event create inside
`ProcessGroupHCCL::_allgather_base` in the recorded case), bottom frames are
CPython's own entry — and count the `vllm_ascend/*.py` frames: zero means the
call site is a data gap, so ask for the plog and an
`--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1` rerun instead of reading an
attention backend. `worker-segfault-no-python-frames.md` has the candidate table
and the precedent families behind that marker (#16079, and #10136 / #10522 /
#10532 / #14283 / #14860 for the other mechanisms that share it).

1. `error-report-timestamps.md` — when the paste carries several error families at
   once (`EE9999` runtime records, `EZ9999` chip dumps, `507011`, `ERR00100`,
   `ERR02005`), decide *which block happened when the reporter says it failed*:
   Ascend re-prints a recorded error with its original date, so the sentence in the
   title can be a day older than the incident (#9183). One call:
   `bash <skill_root>/scripts/plog-timeline.sh --since <report date> <log>`.
2. `collective-error-chain.md` — restore error attribution (the Python frame in
   the traceback is usually a later synchronization point), read the plog, and
   stop attributing the defect to whichever op the traceback names. That page
   also covers the compile-phase case: fusion-pattern registration runs sample
   kernels (`torch.randn(2, 4, device="npu")`), so a `profile_run` traceback
   ending in an `npu_*` op on a 2×4 tensor is a sync point, and it explains what
   `SUSPECT REMOTE ERROR` / 507057 is and why the reporting rank need not be the
   faulting chip.
   Exception to that attribution rule: when the frames from `vllm_ascend/` inward are all
   Python and the last one is a single `torch_npu` index/scatter write, and the failure
   tracks batch composition instead of hardware, the named op *is* the owner's evidence —
   the caller's index tensor. See `../dsa-sparse-attention/indexer-cache-scatter-fault.md`
   (#4296) before asking for device health.
3. `device-health-checks.md` — check every device's health, correlate the
   failing device id, then decide hardware vs software. In the recorded case the
   answer was a physical HCCS link fault on 6 of 8 cards.

If the text is an allocation failure that merely names a collective or an
operator (`HcclAllGather` / `current working operator name is …`) after a server
has been running for a while, the collective is the allocation site, not the
owner: use `../runtime-memory/index.md` for the capacity/growth split instead of
reading that op's shape code.

If the text is `507013` + `sdma copy error` + `The SMMU returns a Terminate error
during page table translation` **together with** `Memory_Allocation_Failure(EL0004)`
or `alloc device memory failed, runtime result = 207001`, the synchronizing copy is
reporting a failure that happened while KV cache or weights were being placed, not
inside a collective: read the timestamps, then use
`../runtime-memory/kv-init-and-format-cast-alloc.md`. Contrast `507057`
(`SUSPECT REMOTE ERROR`), which is a device-error category and stays in this topic.

If the collective failed while the engine was still capturing ACL graphs, take
`../startup-failures/acl-graph-capture.md` as well: the advice printed around it
there is canned wrapper text, and the capture-size set needs the arithmetic check
before any "insufficient streams" reading.

If the text is a P/D transport connect failure rather than a training-style
collective (`HIXL Connect failed | 103900`, `RanktableCheck`,
`'device_ip' is not set correctly, must be set when multi Server`,
`get dev list error:serverId[...]`), the collective is only the messenger: the
report is decided by the rank table's `server_count`, which is a topology input.
Leave this topic for `../pd-disaggregation/transport-topology.md` and run
`<skill_root>/scripts/ranktable-check.sh` first. If the recipe carries a KV pool
(`AscendStoreConnector` in a `MultiConnector`, `MOONCAKE_CONFIG_PATH`), the
`503900` is the pool's `global_segment_size` before it is the rank table: take
`../pd-disaggregation/kv-pool-store-segments.md` and
`<skill_root>/scripts/kvpool-segment-check.sh` first (#13439), and come back to
the rank table only once the segment is small or the failure survives without the
pool.

Common rules for this topic:

1. Reproduce with `--enforce-eager` and `ASCEND_LAUNCH_BLOCKING=1` before
   interpreting any Python frame; unset blocking mode afterwards (it degrades
   performance). `--enforce-eager` is not optional: vllm-ascend refuses
   `ASCEND_LAUNCH_BLOCKING=1` while `cudagraph_mode != NONE` — the gate raises
   `ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1` from
   `AscendPlatform.check_and_update_config` (`vllm_ascend/platform.py:457-469` at
   v0.20.2rc1; the line moves between releases, so grep the message).
2. Treat the plog (`~/ascend/log/debug/plog/plog-<pid>_*.log`) as the primary
   source for HCCL errors, and the vLLM log as the secondary one.
3. Do not derive the root cause from tensor-size arithmetic: in the recorded
   case the reported `count[75497472]` matched both a `[8192, 9216]` language
   model allreduce and a `[65536, 1152]` vision allreduce exactly, and neither
   was the defect.
4. The HCCL implementation sources are usually exported as **`cann/hcomm`**, not
   `cann/hccl`: `src/framework/op_base/src/`, `src/algorithm/impl/operator/`,
   `src/algorithm/base/alg_template/` (that is where `streamNotifyMain_` lives) and
   `src/platform/common/externalinput.cc` (env parsing). Grep `cann/hcomm` before
   saying a behaviour is undocumented, then check `SOURCES.json: unavailable` for
   the genuinely absent repos (`cann/hixl`, `cann/opbase`, `cann/ge`, `ops-*`).
   One pass per tree; after that, use the plog, not more source.
5. Before blaming the hardware, exclude the cheap software knobs
   (`HCCL_BUFFSIZE`, `HCCL_DETERMINISTIC`, `HCCL_OP_EXPANSION_MODE`,
   `PYTORCH_NPU_ALLOC_CONF`), the device visibility set, rank-dependent collective
   buffer sizing, and on a multi-node DP/PD start the address variables
   (`node0_ip` must be an IP, not a NIC name) — but when a per-card error code is
   present, hardware is the answer regardless of those knobs.
