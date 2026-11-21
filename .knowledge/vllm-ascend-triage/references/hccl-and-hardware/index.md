# HCCL / ACL / NPU device faults

Scope: failures whose text names a collective, a device error category or the
runtime stack rather than a Python-level cause — `HcclAllreduce`, `hcclRet -> N`,
`AllReduceOpbasePipeline`, `ERR00100 PTA call acl api failed`, `ERR01100 OPS call
acl api failed`, `the current working operator name is HcclAllreduce`,
`RuntimeError: SUSPECT REMOTE ERROR` / `error code is 507057`, `EZ9999` device
dumps, collective hangs/timeouts during `profile_run` or
`determine_available_memory`, a **long-run** hang reported as
`hccl fftsplus task timeout ... stuck notify num:1, timeout=<N>s` with
`[TaskExecStage][HeartbeatAbnormal] ... Heartbeat Lost Occurred`, and processes
killed by a device fault, or the DP group's gloo `cpu_group` reporting
`pair.cc:534 Connection closed by peer`.

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

**Then separate a *shapeless* hang from a collective error.** A DP>1 MoE run
whose dummy/`profile_run` batch passes and whose first **real request** stops a
layer or two in (no error text, peers' logs also stop) is not a device fault: the
per-rank contribution to the DP all-gather disagreed, and the pad target that must
match is `forward_context.max_tokens_across_dp` — the arithmetic, the owner-slice
convention and the rank-parity checks are in
`../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`. Ask for
`dp_rank`/pad-target/`num_tokens` printed on **every** rank before touching the
device lanes.

If instead the **traffic shape** is the trigger — one request at a time hangs at
graph replay and ends in the bare `TimeoutError: RPC call to execute_model timed
out`, while several concurrent requests are served normally, on `-dp N -tp M` with
ACL graph on — the idle ranks are on `execute_dummy_batch() -> _dummy_run(1)` and
the pad target is *not* the suspect (the dummy path joins the same DP all_reduce
first). `dp-aclgraph-single-request-hang.md` has the verified split, the pad map
that says which bucket a single request replays, and the four-row ablation; one
call: `bash <skill_root>/scripts/dp-single-request-bucket.sh --capture-sizes 8,1
--dp 8 --tp 2`.

**But when the graph mode printed its banner and finished capturing, the mode's own
request path owns the stall.** `FULL_DECODE_ONLY compilation enabled on NPU` (or
`PIECEWISE`) plus a per-rank `Graph capturing finished in N secs`, then the bare
`TimeoutError: RPC call to execute_model timed out` on the *first* request at DP=1 —
that is neither the capture lane nor a dropped setting, and on a release candidate it
is usually a version answer before it is a mechanism one: which models the pinned
tree validates for that mode, whether the reporter has a working control, and the
mode-specific handling (or its absence) for their family. `aclgraph-full-decode-runtime-hang.md`
has the four paste lines that settle the lanes, the two checks to run first, and the
recorded #4178 case.

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

**Then separate a link-creation failure from the collective it is reported as.** If the
first collective after the weights load dies with `Transport_Init_Error(EI0009)` /
`[Create][DestLink]Create Dest error!` / `createLink para:rank[i]…dst_rank[j]` plus
`halShmemOpenHandle … drvRetCode=17, device_id=N` / `rtsIpcMemImportByKey … 507899`,
nothing launched: HCCL could not build one edge of the transport. Two traps decide it —
`507899` is the generic `[driver error:internal error]` (the verb above it names the
operation), and the identical `localIpAddr` / `remote_ip_addr` text is *not* a duplicate
`device_ip` (the field is printed from the local rank on both sides; the rank table is
`scripts/ranktable-check.sh`'s job). The failing **edge set** is the finding:
`bash <skill_root>/scripts/hccl-link-pair-audit.sh <log>` classifies each failed pair
same-card vs cross-card against the `npu-smi` Bus-Ids, and all-cross-card with every
card-local edge up puts it on the host driver/CANN pairing and the container's device
passthrough, not on the model. `transport-destlink-ipc-import.md` (#4148, #486, and the
#5236 driver-update precedent).

**Then separate a *refused resource allocation* from that link-creation failure.**
The two share the `[Create][DestLink]Create Dest error!` sentence, so read the
**first** code and its reason. `EI0007 … Failed to allocate resource[qp] … Reason:
Memory resources are exhausted` (plus `Possible Cause: …Notify register…`) is a
local device comm-resource refusal: the DestLink block then prints **every** peer
in the same millisecond, and the owning phase is graph capture
(`compile_or_warm_up_model` → `capture_model` → `_compile_torchair_graph` →
`LoadGraph failed, error code:1343225857` / `ERR03005 GRAPH internal error`). That
is the OOM-class budget lane — KV cache is already resident because
`initialize_from_config` allocates it before `compile_or_warm_up_model` — not an
edge, a GE build or a card fault: `qp-resource-exhausted-at-graph-load.md`
(#2754, maintainer-classified OOM; the `send_cq_depth` field is a request, not a
knob).

**Then separate a cascade from the primary, and count the ranks.** When a paste
carries two error families and one of them comes from exactly **one** process, that
process owns the fault. The recorded shape is an RL/GRPO job whose rollout is a
separate `vllm serve`: rank 0 dies inside the weight update
(`pyhccl.py … hcclBroadcast`, `HCCL error: not found specific resource`) while ranks
1-6 report `HcclAllgather … error code is 6` / `EI0009 [Create][DestLink] … dst_rank[0]`
— the link failures exist *because* rank 0 never joined, so
`transport-destlink-ipc-import.md`'s hardware ladder is the wrong lane here.
`weight-sync-hccl-cascade.md` has the pin's `PyHcclCommunicator` facts (the tree never
aliases it over upstream's, so a trl/ms-swift patch supplies it), the init sequence
that rules out a "peer never joined" reading, and the value-comparing broadcast that
is rung 1 (#459, #942 — the recorded broadcast returned silent zeros).

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
#10532 / #14283 / #14860 for the other mechanisms that share it). The same marker
**at engine start** — before `Starting to load model`, parent side `Failed core
proc(s): {'EngineCore': -11}`, the `TBE Subprocess … main process disappeared`
block after it — is the startup section of that page: `-11` is SIGSEGV, a trivial
CPU `torch.zeros` at the top of the dump is where the damage surfaced, and the
answer is the image/CANN pairing plus the release's own validated command.

**Then separate a peer that *finished* from a peer that died.** `Connection closed
by peer` also appears when the DP group's gloo `cpu_group` loses a rank that exited
**normally**: the offline multiprocess DP demo (`VLLM_DP_*` env vars plus
`LLM(device="npu:N")` in one `multiprocessing.Process` per rank) gives each rank its
own `max_tokens`, the first rank returns from `generate()`, sleeps and exits, and its
peer dies at `set_forward_context` → `dist.all_reduce(num_tokens_tensor,
group=get_dp_group().cpu_group)`. The same report usually needs the engine-mode answer
first: external multiprocess DP is a **V1** feature (#541 reporter-confirmed), and at
vllm v0.7.3 + vllm-ascend v0.7.3rc1/rc2 there is no `get_dp_group` anywhere, so that
traceback cannot come from the pair the report names. `dp-group-peer-close.md` has
both steps, the peer-log discriminator (printed `Processed prompts: 100%` = finished
launch shape, `exit code: None`/sigquit dump = killed rank) and the "works after
debugging" caveat.

1. `error-report-timestamps.md` — when the paste carries several error families at
   once (`EE9999` runtime records, `EZ9999` chip dumps, `507011`, `ERR00100`,
   `ERR02005`, or a `timeout=<N>s` fftsplus expiry), decide *which block happened
   when the reporter says it failed*:
   Ascend re-prints a recorded error with its original date, so the sentence in the
   title can be older than the incident — a previous run (#9183, 36 h) or **this
   process's own init/capture window reported late** (#10700, 27 min); pid plus log
   position decides, not the gap. One call:
   `bash <skill_root>/scripts/plog-timeline.sh --since <report date> <log>`.
2. `transport-destlink-ipc-import.md` — `DestLink` / IPC-import failure at link
   creation: the edge-set read, the generic `507899`, why equal `localIpAddr` text is not
   a duplicate `device_ip`, and the driver-pairing ladder.
3. `qp-resource-exhausted-at-graph-load.md` — `EI0007` +
   `Failed to allocate resource[qp]` at graph load: why a DestLink line for every
   peer means the *local* resource pool refused, the GE cascade that carries it, the
   `[1, max_num_seqs]` graph-count arithmetic, and the budget ladder (gmu,
   `HCCL_BUFFSIZE`, graph mode off) before any driver theory.
4. `weight-sync-hccl-cascade.md` — RL weight-sync channel failures across a trainer and
   a `vllm serve` rollout process: which rank owns the primary (the singleton), what
   `pyhccl.py`'s init already proves, why "not found specific resource" is CANN's own
   text for code 6, the standalone two-process broadcast with value comparison, and the
   release-scope caveat (no aliasing, no validated RL sync path at v0.10.2rc1).
5. `collective-error-chain.md` — restore error attribution (the Python frame in
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
6. `device-health-checks.md` — check every device's health, correlate the
   failing device id, then decide hardware vs software. In the recorded case the
   answer was a physical HCCS link fault on 6 of 8 cards.
7. `aclgraph-full-decode-runtime-hang.md` — the graph mode's banner printed, its
   capture finished on every rank, and the **first request** then stalls into the
   watchdog: which two lanes the paste already closes, the two coverage checks to run
   before any source read, and why an rc-only hang is a version verdict (recorded
   #4178; 310P variant #12985).

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
