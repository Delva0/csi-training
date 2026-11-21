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
does not.

**`… NPU function error: …, error code is N` from a `torch_npu` op is a different
namespace**, and `N` may not be that op's status at all: PTA's `NPU_CHECK_ERROR`
overrides the printed code with the thread's latched runtime error
(`AclrtPeekAtLastError`, "仅获取但不清空"), so a capture-window `107030` can be printed by
an unrelated op hours later and mask the real failure —
`latched-runtime-error-masking.md`. The one-sided `HcclBatchPut` / store-put variant of
that argument is
`../kv-pool-store/hccl-onesided-store-put.md` §1.

**First decide whether a communicator can be created at all.** If the text is
`HCCL function error: hcclCommInitRootInfoConfig(…), error code is N` out of
`…/torch_npu/csrc/distributed/HCCLUtils.cpp:140` (`create_config`), no collective
was launched: a **group** creation was refused. `1` there is not a network code —
`HcclCommInitRootInfoConfigInner` folds `InitExternalInput()`, `InitEnvConfig()` and
`commConfig.Load()` failures into `HCCL_E_PARA` and prints the sub-check
(`init external input error` / `init environment config error` / `load comm config
failed` / `nRanks…`), so read the HCCL `ERROR` line, then diff the two nodes' serve
scripts, and only then ask whether the recipe exists at the pin for that model and
node count. `comm-create-errors.md`. At the same entry code `15` (`HCCL_E_RUNTIME`) is the
**opposite** lane: comm setup asked the driver for the link type of a *physical device pair*
(`hrtGetPairDeviceLinkType` over every same-server rank pair) and the driver declined, so the
evidence is the container's device set (`ls /dev/davinci*`, `ASCEND_RT_VISIBLE_DEVICES`,
`npu-smi info -t topo`) plus a minimal `torch.distributed` HCCL `all_gather` on the same devices —
never the launch script, and never more `cann/hcomm` source: the failing return code comes from the
closed driver. `comm-create-errors.md` §*Code 15*.
**The no-config sibling `create … HcclCommInitRootInfo(…)` with code `2` is a different lane**:
the group belongs to whichever caller the traceback names, often a third-party trainer
(DeepSpeed `_broadcast_model`, TRL `prepare_deepspeed`) colocated in the engine's process, with no
`vllm_ascend/` frame at all; `2` is `HCCL_E_PTR` and names no precondition, and on a one-rank group
the store cannot be what failed. Get the trainer's launch line (it *is* `numRanks`) and compare a
multi-rank run of the same recipe before any torch_npu read — `comm-create-errors.md`
§*The no-config `create` sibling*. **Code `7` (`HCCL_E_UNAVAIL`) beside an `EJ0001`
line is not a script problem**: the vendor's table names it
`HCCP_Process_Initialization_Failure` and its fix is "wait for 10s after killing the
last training process", i.e. a **previous process of the same box is still alive** —
sleep mode, an RL engine that never exited, a crashed leader whose HCCP stays up
(`../runtime-memory/sleep-mode-multi-instance.md`). Any bare `<E-code>` in a paste is
one call: `bash <skill_root>/scripts/cann-error-lookup.sh <code>`.

Order matters more than depth here:

**Then separate the watchdog from the collective.** If the only failure text is a
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
the precedents (#14928, #13688, #15195). The same op name returning an HCCL code
(`hccl_ret 9`) instead of running into a device timeout is the **transport/comm-plane**
sibling, not this family: #9447 was still
unresolved on 2026-06-22 with the failure on **two** fabrics, and its passing
`hccl_test` cleared nothing — `mc2-dispatch-profile-run-hang.md` §Step 3c.

**If the report is a mode split instead of an error — "图模式算子通信报错, 只有开启单算子
才能规避", no log, the attachment failed** — that is an execution-mode classification
before it is a kernel question: `单算子` is GE's single-op/eager mode, not a key in the
tree, the reporter's working single-op run on the same cluster exonerates the rank
table, the HCCL links and the NICs, and the deliverable is the knob ladder
(`cudagraph_mode` NONE → PIECEWISE vs `FULL_DECODE_ONLY` → `HCCL_OP_EXPANSION_MODE`)
plus the two-node recipe diff → `graph-vs-single-op-comm-fault.md`. Two traps it
carries: `ASCEND_LAUNCH_BLOCKING=1` is refused outright while a graph mode is on
(`platform.py:457-467`, v0.20.2rc1), and the `saved AOT compiled function …
torch_compile_cache` line is a save message whose cache ablation was already refuted
by the reporter on #10370.

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
the flag lane, and the ladder. **On a multi-node start, diff the two nodes' own
commands first**: capacity and `global_bs` are derived per rank from that node's
`--compilation-config`, so an asymmetric capture-size list makes the halves of the EP
group declare different sizes to the same all-to-all — one call:
`bash <skill_root>/scripts/mc2-capacity-asymmetry.sh --capture-a … --capture-b … --tp
8 --dp 2` (recorded #8294). **A symmetric pair of commands does not close the lane:**
the same derived capacity is compared per rank against that rank's own batch at run
time, so a serve whose `--max-num-batched-tokens` (or `max_num_seqs ×
(1 + num_spec)`) exceeds `capacity` can have some DP ranks on MC2 and the others on
the fallback dispatcher in one step — same script, "branch-split" section (recorded
#9887). If the reporter has already run the env matrix, treat it as closed
(`aiv-mc2-env-ablation.md`, "When the whole matrix has already been run").

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

**Changed shape — the peers keep serving.** If the router's own counts show the
*other* DP ranks finishing requests and their KV transfers completing while the
affected ranks hold `Running` > 0 with a flat generation counter, no collective can
be waiting on them (a pending all-to-all would block the peers too) and the paste
has no RPC-watchdog or plog `STUCK` block: classify it as a scheduler/KV-state
failure and go to `../scheduler-admission/kv-block-lifetime-async-scheduling.md`
(#13964: SWA/chunked-local blocks reused under async scheduling, vllm-ascend #13518).
The recorded run of that case spent its budget on the DP cudagraph-mode sync and the
skipped DP all-reduce instead — both are real mechanisms, and both are refuted by
that one sentence in the report.

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

**Then separate "no kernel is running" from "the host is stuck".** `AICore 0 %` on
every chip next to a listening API server and a silent `curl` is *not* evidence of a
host-side stall: a rank that entered an all-to-all and waits for a peer that never
arrived also executes no AI-Core work, and `AIVector` is no test either (8-16 % in
#13688's MC2 replay deadlock, so neither a non-zero nor a zero reading picks a lane).
A multi-node or P/D MoE serve that hangs **whole** after minutes of healthy load with
no traceback and no device dump belongs here, before any `py-spy`: get the plog from
every node for the hang window and run
`bash <skill_root>/scripts/hccl-stuck-timeline.sh <plogs…>` —
`[Load] comm config of [group_name_N]` (loaded `bufferSize` is **bytes**,
`execTimeOut`), `[BuildRemoteResByTag] newtag[…mc2RunAlltoAllDirectFullmesh_device]`
(the op family), and `count of currently executed operators` (frozen on the same
number at the same second across nodes = one collective instance waiting on the
missing ranks; still rising = that rank is fine, whatever its `STUCK` lines say).
Recorded #8975 ranked a host-side DP `cpu_group` all_reduce first and was answered by
a 32-rank MC2 all-to-all frozen at AICore 0 %:
`multinode-mc2-alltoall-silent-deadlock.md` has the four reads, the MC2 selector
arithmetic, the rule that the `mc2` group takes its comm space only from
`HCCL_BUFFSIZE` (`utils.py:961-965`), and the 1836 s expiry that explains the silence.

**If the same mode instead dies *under load* with a device fault, start from the
replay's arguments.** `507011` (`ACL_ERROR_RT_MODEL_EXECUTE` — a wrapper that names
no op) plus `fftsplus aivector error`, `Illegal instruction … unaligned UUB
addresses`, `mte error` or a `notify wait … sq sw status error` on another stream,
appearing only once concurrency rises, is the crash variant of the paragraph above:
`aclgraph-full-decode-runtime-crash.md` has the two config discriminators
(`cudagraph_capture_sizes` vs the failing concurrency; `pa_shape_list`, which gates
the legacy paged-attention patch path and defaults to empty), the `origin_op_name` /
`fault kernel_name` ask, the ladder and the recorded #6018 case.

**Then separate an HCCL-owned AIV collective kernel from a vllm-ascend op.** When a
plog line reads `fault kernel_name=aiv_reduce_scatter_bfloat16_t` /
`aiv_all_gather_*` / `aiv_all_reduce_*` beside `The DDR address of the MTE instruction
is out of range` + `isSmmuFault is 1` + `[vector core exception]`, the name is HCCL's
own opbase AIV template — `cann/hcomm/src/algorithm/base/alg_aiv_template/
aiv_communication.h:227-258` generates it with `##type`, so the custom-op lane's
`ls csrc` test is a dead end by construction. Read that line before any
`reduce_scatter` call-site walk, then compare the `opData information is …` line
across every attached plog (identical `index`/`count`/`src`/`dst` refutes "one rank
passed a different size"), count the ranks by `device_id=` rather than by filename,
and remember that `HCCL_OP_EXPANSION_MODE=AIV` is what `vllm_ascend/utils.py:506,554`
recommends at v0.18.0 — unsetting it is the discriminator, not the fix:
`aiv-collective-kernel-fault.md` (recorded #13527). The env-variable correlation for
this same `AIV` family — which launch/flag combination makes it fire at all — stays
`aiv-mc2-env-ablation.md`; this paragraph owns only the ownership split. **The
`retCode=0x30` / `timeout or trap error` variant has no bad address at all**: one
shared `pc start` on every core plus a fault print minutes after the `opData`
`timeStamp` means a collective some peer never issued, and when the op is a small
`int32` `aiv_all_gather_*` whose `count` equals the checkpoint's `num_experts` and
whose `rankSize` equals the EP world size, it is the MoE dispatcher's own preprocess
histogram gather — suspect a per-rank branch split first (some DP ranks past
`mc2_tokens_capacity` leaving MC2), not a hardware or transport fault:
`aiv-collective-kernel-fault.md` § "The other retCode", then
`../feature-enable-recipes/moe-expert-parallel-with-dp.md` Step 2 (recorded #9887).

**`507018` is not this topic's.** "The aicpu execution is abnormal" beside a
`kernelName=<Op>, soName=libpt_kernels.so, funcName=RunCpuKernel` block is one
operator that ran on the CPU/AICPU fallback because no AI-Core kernel matched the
SoC/dtype/layout; the code is what the copy-stream sync reports afterwards, and the
`devmm … should call halMemMap` dmesg line follows the fault rather than causing it.
Route by the op name — on 310P `../soc-platform-paths/310p-cpu-fallback-op-fault.md`
— instead of looking for a collective or a card here.

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

`Bind_Failed(EJ0003)` on the *normal* port `16666` is the other family: 16666 is
`HETEROG_CCL_PORT`, the one default comm port, so the endpoint is decided by the local
device IP alone and a collision means **two live processes on one device** — count
engines against cards before touching any port variable. Two engines started side by
side on one host (a P/D pair) are the usual pair, and the external-DP launcher hands
every invocation cards from 0: `comm-create-errors.md` §code 7, then
`../pd-disaggregation/same-host-pd-device-partition.md` (#4725,
reporter-confirmed).

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

**Then separate a socket link that never came up from the resource refusal it is
reported as.** If the plog carries `[Wait][LinksEstablishCompleted] is failed. ret[9]`
/ `[Create][Sockets] … local role is client. ret[9]` / a `LINK_ERROR_INFO` table
(`dest_ip(user_rank) | dest_port | src_ip(user_rank) | src_port | MyRole | Status |
TlsStatus`) / `[Create][InterLinks] socket manager create connections failed, ret[9]`
/ `[Create][CommPlane] failed, tag[…MC2MultiServer]` **before** the
`HcclAllocComResource … ret = 4` line, no communicator was built: `4` is
HCCL_E_INTERNAL wrapping the earlier `9`/TIMEOUT, so a title of
`aclnnMoeDistributeDispatch… 561000` is **not** evidence for the QP/budget lane and
`HCCL_BUFFSIZE`/gmu cannot fix it. Ask for the plog from **both** nodes and **every
card**, then read the table wholesale-vs-subset. `link-establish-timeout.md`
(recorded #9872, #1955; also the `[HcclSocket][Listen] … port[16666] fail`
server half, and why `ping` between the nodes does not clear it).

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
knob). **Read the phase off the frames before choosing a lever:** the same refusal
delivered by an MC2 MoE dispatch (`aclnnMoeDistributeDispatchV3 … 561000`,
`HcclAllocComResourceByTiling … ret = 4`, no `LoadGraph failed`/`ERR03005` block) is
the first MC2 collective, which on v0.13.0rc1 runs in `profile_run`'s
`mc2_tokens_capacity` dummy run *before* the KV pool — there the KV budget is not the
lever and the capacity/`global_bs` arithmetic of
`mc2-dispatch-profile-run-hang.md` §Step 1 is (recorded #5947; under
`HCCL_INTRA_PCIE_ENABLE=1` only the cross-node RoCE peers print `DestLink`).

**A one-box paste has no cross-node leg, so the branch question comes first.**
Read the SoC term of `HcclAllocComResourceByTiling` (`op_base_mc2.cc:120-121`: every
`devType != DEV_TYPE_910_93` returns through `HcclCreateComResourceByComm`) and the
tutorial's own topology row before any transport lane: on a single 16-die 910B box
the A2 `_MC2MultiServer` resource path runs with no remote peers, and the one-node
recipe a reporter copies for these DeepSeek models is A3, i.e. the other branch —
`mc2-dispatch-profile-run-hang.md` §Step 3b2.

**Then separate a device *event-table* refusal from a memory leak.** If the paste
reads `[Error]: Event resources are insufficient.` beside
`EE9999 … Event id alloc error, deviceId=N, tsId=0, retCode=0x7020019`,
`rtEventRecord … no event resource` and `record event failed, runtime result =
207007` — typically a serve that ran fine and then died under load, with
`current working operator name is aclnn<Op>` — the code is
`ACL_ERROR_RT_NO_EVENT_RESOURCE` and nothing ran out of memory: read the last
`GPU KV cache usage` line, then treat the named op as the last queued kernel and
the Python frame as one sync point off the `record`. Split the pool's two consumers
(graph capture via `--enforce-eager`) and measure it (`aclrtGetEventAvailNum`, the
`npu-smi` process table for that `deviceId`) before any leak theory —
`acl-event-resource-exhaustion.md` (#13007; capture-phase precedent #10521).

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

**Then separate a death reported as a plain Python traceback from a queue bug.**
`RuntimeError: cancelled` raised at `vllm_ascend/patch/platform/patch_shm_broadcast.py:70`
(or `shm_broadcast.py:701`), reached from `multiproc_executor.py:390`
(`mq.dequeue`) under `core.py:693 model_output = future.result()`, is the same
family as the segfault marker with the marker absent: a worker **process** died,
`monitor_workers` logged `Worker proc … died unexpectedly (exit code: N)` and shut
the response queue down, and the parked reader re-raised. The
`Dumping input data / scheduler output / scheduler stats` block above it is that
step's in-flight `SchedulerOutput` (one call site: `core.py:486-496` around
`future.result()`), not the fault. A worker *Python* exception never produces this
text — `worker_busy_loop` catches it and returns a `FAILURE` response, so the
EngineCore re-raises the worker's own exception instead. Ask for the dying rank's
log and the plog, decode the exit code, and rank the driver/firmware × CANN
pairing while waiting: on #16023 that pairing was the reporter-confirmed cause
("CANN 与 HDK 存在冲突 … firmware 与 driver 不匹配 … 升级到 26.1.1 后解决").
`cancelled-executor-teardown.md`.

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
3. `acl-event-resource-exhaustion.md` — `Event resources are insufficient` /
   `207007` / `0x7020019` after a healthy serve: the enum name, why the named op and
   the `aclnn…Div` frame are victims (`wait_stream` allocates a fresh event per
   decode step; capture holds its own share of a documented per-Device table), the
   `--enforce-eager` split, the `aclrtGetEventAvailNum` sample, the 310P
   `VLLM_BATCH_INVARIANT=1` localization run, and what the driver's `no event
   resource` means that the ACL doc's "waits for release" does not (#13007).
4. `qp-resource-exhausted-at-graph-load.md` — `EI0007` +
   `Failed to allocate resource[qp]` at graph load: why a DestLink line for every
   peer means the *local* resource pool refused, the GE cascade that carries it, the
   `[1, max_num_seqs]` graph-count arithmetic, and the budget ladder (gmu,
   `HCCL_BUFFSIZE`, graph mode off) before any driver theory.
5. `weight-sync-hccl-cascade.md` — RL weight-sync channel failures across a trainer and
   a `vllm serve` rollout process: which rank owns the primary (the singleton), what
   `pyhccl.py`'s init already proves, why "not found specific resource" is CANN's own
   text for code 6, the standalone two-process broadcast with value comparison, and the
   release-scope caveat (no aliasing, no validated RL sync path at v0.10.2rc1).
6. `collective-error-chain.md` — restore error attribution (the Python frame in
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
7. `device-health-checks.md` — check every device's health, correlate the
   failing device id, then decide hardware vs software. In the recorded case the
   answer was a physical HCCS link fault on 6 of 8 cards.
8. `aclgraph-full-decode-runtime-hang.md` — the graph mode's banner printed, its
   capture finished on every rank, and the **first request** then stalls into the
   watchdog: which two lanes the paste already closes, the two coverage checks to run
   before any source read, and why an rc-only hang is a version verdict (recorded
   #4178; 310P variant #12985).
9. `aclgraph-full-decode-runtime-crash.md` — the same mode captured and then dies
   **later, under concurrency**, with `507011` / aivector / MTE / illegal-instruction
   text instead of a hang: `507011` is `ACL_ERROR_RT_MODEL_EXECUTE` and names nothing,
   the notify-wait and second-stream lines are victims, and the per-replay argument
   patch (`acl_graph.py:198-304`) is why the bucket and the `pa_shape_list` gate decide
   the ranking (recorded #6018).
10. `comm-create-errors.md` — `hcclCommInitRootInfoConfig … error code is N` at
   communicator creation (`create_config`, `HCCLUtils.cpp:140`), usually inside
   `get_hccl_comm_name` for a MoE all-to-all group: the `HcclResult` codes that
   outrank it, the six `HCCL_E_PARA` sub-checks at `op_base.cc:1377-1408`, the
   launch-script values that reach them (`HCCL_BUFFSIZE` ≥ 1 MB, config magic word),
   the tutorial-recipe check, and the rank-asymmetry reading when a config change
   turns the error into a hang.
11. `aiv-collective-kernel-fault.md` — the device fault is an **HCCL AIV collective
   kernel** (`fault kernel_name=aiv_reduce_scatter_bfloat16_t`, `Task AIV` /
   `tag[ReduceScatter_group_name_N]`, MTE `DDR address … out of range`,
   `isSmmuFault is 1`, `retCode=0x31`): the macro that generates the name, the
   `opData` comparison that refutes a per-rank size mismatch in one `rg`, the
   `size == count × sizeof(dtype)` identity and the `len` the plog never prints,
   why AIV is the configured path rather than a misconfiguration, the victim
   `InplacePartialRotaryMul` / `Remote rank: [0]` line, the `stackcore_tracer` files
   that are *not* a Python traceback, and the two one-axis reruns (recorded #13527).
   Its §0 covers the paste that stops **before** the name (a `device_error_core_proc.cc`
   dump, no `fault kernel_name=`, client `code: 599`): rank by deployment shape — the
   pinned MoE comm-path selector's decode branch (`ep_size ≥ 16` → MC2, prefill →
   All2All), the AIV lane only after `env | grep -i hccl`, then a long-VL
   partial-prefill index fault when `max-num-batched-tokens ≪ max-model-len` (#3911).
12. `link-establish-timeout.md` — `[Wait][LinksEstablishCompleted] is failed. ret[9]`
   and the `LINK_ERROR_INFO` table on a multi-node start: the chain that ends as
   `HcclAllocComResource … ret = 4` / `aclnn…Dispatch 561000`, the `src_port 0` /
   `connecting` reading (wholesale vs subset), the `Listen … port[16666] fail` half
   that lives in the *peer's* plog, and the device-plane checks (`hccn_tool -ip/-optical/
   -lldp`, address variables per node) that replace `ping`.
13. `multinode-mc2-alltoall-silent-deadlock.md` — a **silent** whole-service hang after
   minutes of healthy load on a multi-node/PD MoE serve (`AICore 0 %`, port listening,
   `curl` silent, no error): why the utilisation column does not separate a host stall
   from a rank waiting inside an all-to-all, the four plog reads (comm config,
   `newtag[…]` op family, heartbeat `STUCK`, `count of currently executed operators`),
   the MC2 selector arithmetic that says whether the all-to-all was even the expected
   path, `HCCL_BUFFSIZE` as the `mc2` group's only comm-space source (MB × 1024², while
   `bufferSize` is printed in bytes), and the follow-up-turn delta (#8975).
14. `cancelled-executor-teardown.md` — a runtime worker death surfacing as
   `RuntimeError: cancelled` in the EngineCore: the verified
   monitor→shutdown→`acquire_read` chain, why a Python worker exception cannot
   produce it, the one-grep `died unexpectedly (exit code: N)` decode, the
   in-flight-`SchedulerOutput` dump, the candidate table (driver/firmware × CANN
   first, reporter-confirmed on #16023), and the deliver-first budget rule.
15. `latched-runtime-error-masking.md` — a `torch_npu` op failure whose
   `error code is N` also appears in a capture-window plog record: PTA's macro replaces the
   printed code with `AclrtPeekAtLastError(ACL_RT_THREAD_LEVEL)` (peek, never cleared) and
   `getDeviceErrorMessage` supplies the text, so the code is stale and the op's own status is
   masked; the two exported CANN doc semantics that make it possible, the blocking H2D tiling
   `aclrtMemcpy` in the custom-op host adapters that mints `107030` at capture, why no flush
   exists (`clear_device_error_info` only via `torch_npu.npu.restart_device`), and the
   deliverable (plog at the failure second, not the capture block).

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
   `src/algorithm/base/alg_template/` (that is where `streamNotifyMain_` lives),
   `src/algorithm/base/alg_aiv_template/` (the `aiv_*` opbase kernels the plog's
   `fault kernel_name=` can name) and
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
