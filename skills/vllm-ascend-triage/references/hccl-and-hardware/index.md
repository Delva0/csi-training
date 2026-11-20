# HCCL / ACL / NPU device faults

Scope: failures whose text names a collective or the runtime stack rather than a
Python-level cause — `HcclAllreduce`, `hcclRet -> N`, `AllReduceOpbasePipeline`,
`ERR00100 PTA call acl api failed`, `ERR01100 OPS call acl api failed`,
`the current working operator name is HcclAllreduce`, collective hangs/timeouts
during `profile_run` or `determine_available_memory`, and processes killed by a
device fault.

Inherit the rules in SKILL.md. Order matters more than depth here:

1. `collective-error-chain.md` — restore error attribution (the Python frame in
   the traceback is usually a later synchronization point), read the plog, and
   stop attributing the defect to whichever op the traceback names.
2. `device-health-checks.md` — check every device's health, correlate the
   failing device id, then decide hardware vs software. In the recorded case the
   answer was a physical HCCS link fault on 6 of 8 cards.

Common rules for this topic:

1. Reproduce with `--enforce-eager` and `ASCEND_LAUNCH_BLOCKING=1` before
   interpreting any Python frame; unset blocking mode afterwards (it degrades
   performance).
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
   `PYTORCH_NPU_ALLOC_CONF`) and the device visibility set — but when a per-card
   error code is present, hardware is the answer regardless of those knobs.
