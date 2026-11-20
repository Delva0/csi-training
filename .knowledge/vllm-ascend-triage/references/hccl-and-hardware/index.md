# HCCL / ACL / NPU device faults

Scope: failures whose text names a collective, a device error category or the
runtime stack rather than a Python-level cause — `HcclAllreduce`, `hcclRet -> N`,
`AllReduceOpbasePipeline`, `ERR00100 PTA call acl api failed`, `ERR01100 OPS call
acl api failed`, `the current working operator name is HcclAllreduce`,
`RuntimeError: SUSPECT REMOTE ERROR` / `error code is 507057`, `EZ9999` device
dumps, collective hangs/timeouts during `profile_run` or
`determine_available_memory`, and processes killed by a device fault.

Inherit the rules in SKILL.md. Order matters more than depth here:

1. `collective-error-chain.md` — restore error attribution (the Python frame in
   the traceback is usually a later synchronization point), read the plog, and
   stop attributing the defect to whichever op the traceback names. That page
   also covers the compile-phase case: fusion-pattern registration runs sample
   kernels (`torch.randn(2, 4, device="npu")`), so a `profile_run` traceback
   ending in an `npu_*` op on a 2×4 tensor is a sync point, and it explains what
   `SUSPECT REMOTE ERROR` / 507057 is and why the reporting rank need not be the
   faulting chip.
2. `device-health-checks.md` — check every device's health, correlate the
   failing device id, then decide hardware vs software. In the recorded case the
   answer was a physical HCCS link fault on 6 of 8 cards.

If the text is an allocation failure that merely names a collective or an
operator (`HcclAllGather` / `current working operator name is …`) after a server
has been running for a while, the collective is the allocation site, not the
owner: use `../runtime-memory/index.md` for the capacity/growth split instead of
reading that op's shape code.

If the collective failed while the engine was still capturing ACL graphs, take
`../startup-failures/acl-graph-capture.md` as well: the advice printed around it
there is canned wrapper text, and the capture-size set needs the arithmetic check
before any "insufficient streams" reading.

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
4. HCCL's pipeline/opbase internals are closed; the public `cann/hccl` export has
   no `streamNotifyMain_`/opbase sources. One grep pass, then move on.
5. Before blaming the hardware, exclude the cheap software knobs
   (`HCCL_BUFFSIZE`, `HCCL_DETERMINISTIC`, `HCCL_OP_EXPANSION_MODE`,
   `PYTORCH_NPU_ALLOC_CONF`), the device visibility set, rank-dependent collective
   buffer sizing, and on a multi-node DP/PD start the address variables
   (`node0_ip` must be an IP, not a NIC name) — but when a per-card error code is
   present, hardware is the answer regardless of those knobs.
