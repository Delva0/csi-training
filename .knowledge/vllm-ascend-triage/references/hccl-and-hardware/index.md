# HCCL / ACL / NPU device faults

Scope: failures whose text names a collective, a device error category or the
runtime stack rather than a Python-level cause — `HcclAllreduce`, `hcclRet -> N`,
`AllReduceOpbasePipeline`, `ERR00100 PTA call acl api failed`, `ERR01100 OPS call
acl api failed`, `the current working operator name is HcclAllreduce`,
`RuntimeError: SUSPECT REMOTE ERROR` / `error code is 507057`, `EZ9999` device
dumps, collective hangs/timeouts during `profile_run` or
`determine_available_memory`, and processes killed by a device fault.

Inherit the rules in SKILL.md. Order matters more than depth here:

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
   `ASCEND_LAUNCH_BLOCKING=1` while `cudagraph_mode != NONE`
   (`vllm_ascend/platform.py:601-613`).
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
