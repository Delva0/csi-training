# Log signatures in vLLM Ascend hang/crash reports

Use to locate where a reported failure is exposed; none of these signatures alone names the
root cause. Order the timestamps first — teardown noise follows the first fatal error.
A crash that happens only on the **first** request (later requests normal) routes elsewhere
first: `AttributeError: 'NoneType' object has no attribute 'size'` at
`torch/_dynamo/utils.py call_size` is the `--language-model-only` warm-up mismatch in
[first-request-only-output.md](first-request-only-output.md), not an ACL-graph or teardown
signature below.

## EngineCore worker stall / hang

```
(EngineCore_DP0 ...) INFO [shm_broadcast.py:705] No available shared memory broadcast block
  found in 60 seconds. ...
TimeoutError: RPC call to sample_tokens timed out.
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

- The `shm_broadcast` warning says a worker has not published for 60 s; the `TimeoutError` is
  the EngineCore RPC deadline; `EngineDeadError` is always secondary (the API server wrapping
  the dead core).
- It localises the stall to the model-forward/worker side, not to the scheduler or API layer.
- Next evidence needed is worker-side stderr/plog and the flags in force. vllm-ascend patch
  frames in the stack (`patch_minimax_usage_accounting.py`, the `serving.py` wrappers) are
  wrapper frames through which the engine exception propagates; the originating failure is the
  EngineCore stack printed above them, ended by
  `core_client.get_output_async ... raise self._format_exception(outputs)`.
- In multi-node PD reports, establish which role's log produced the timeout before drawing
  conclusions: the pasted blocks interleave several nodes' EngineCore/APIServer lines (different
  IPs and ports), and the failing role is usually the one whose local IP answers the curl.

## `RuntimeError: cancelled` — engine teardown after a worker died

```text
(EngineCore_DP2 pid=57922) ERROR ... [core.py:1332] EngineCore encountered a fatal error.
  File ".../vllm/v1/executor/multiproc_executor.py", line 390, in get_response
    status, result = mq.dequeue(timeout=dequeue_timeout)
  File ".../vllm/v1/engine/core.py", line 693, in step_with_batch_queue
    model_output = future.result()
  File ".../vllm_ascend/patch/platform/patch_shm_broadcast.py", line 70, in acquire_read
    raise RuntimeError("cancelled")
RuntimeError: cancelled
```

- `acquire_read` raises this only when `self.shutting_down` is already set, and only
  `MessageQueue.shutdown()` sets it (`vllm/distributed/device_communicators/shm_broadcast.py:541-544`
  at vLLM `v0.26.0`; upstream's own `acquire_read` raises the same at `:700-701`). The vllm-ascend
  copy (`patch_shm_broadcast.py:35-70` at `v0.26.0rc1`) re-implements the ring-buffer read with a
  5 s recheck; the cancellation check is upstream's.
- The queue is normally shut down by the executor's worker monitor: when a worker process dies,
  `MultiprocWorkerMonitor` logs
  `Worker proc <name> died unexpectedly (exit code: <N>), shutting down executor.` and calls
  `self.shutdown()`, which shuts `rpc_broadcast_mq` and `worker_response_mq`
  (`multiproc_executor.py:273-292` and `:459-492`). An EngineCore blocked in `get_response` then
  gets `cancelled`. Plain process teardown at exit reaches the same raise.
- So this is a teardown echo of a dead worker, not a prompt-size, kernel, sampler or
  "request cancelled by client" fault. Look for the `Worker proc ... died unexpectedly` line and
  its exit code in the same log, then that rank's stderr/plog. Do not report `shm_broadcast.py` or
  the patch as the origin.
- The `dump_input.py` blocks immediately above it (engine config, `SchedulerOutput`, scheduler
  stats) are printed *because* the engine is dying. They describe the step in flight, not the
  cause. Recorded instance #16023 (`v0.26.0rc1`, A3, Kimi-K3 + DSpark spec decode, DP4 with P/D,
  short curls fine, ~100k-token prefill dies): the dump shows `num_scheduled_tokens=8064`,
  `num_computed_tokens=8064`, `queries=30782, hits=0`, `kv_cache_usage=0.06` — a chunked prefill
  was running. The reporter's own closing note on #16023 (not a maintainer statement) says core
  dumps and traffic tests traced it to a bottom-layer operator/CANN-vs-HDK incompatibility with
  firmware and driver mismatched, and that upgrading to `26.1.1` fixed it — so treat a long-prefill
  `cancelled` in a fresh 0.26.x environment as "worker died at the device layer" first.
- The maintainer's evidence request for that shape is the right follow-up set: the `Worker_*` log
  segment immediately before the crash (look for `5070xx` / `EZ9999` / a fault `kernel_name`), the
  CANN version from `collect_env.py`, and whether the same 100k-token request reproduces with the
  speculative-decoding config removed.

## `WorkerProc initialization failed ... See stack trace for root cause` — a worker died before reporting READY

```text
[EngineCore_DP0 pid=40852] WARNING [multiproc_executor.py:438] [shutdown] Executor: workers still
  running after grace period; sending SIGTERM count=7
[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!
[EngineCore_DP0 pid=40852] WARNING [multiproc_executor.py:448] [shutdown] ... sending SIGKILL count=7
[EngineCore_DP0 pid=40852] ERROR [core.py:1195] EngineCore failed to start.
  File ".../vllm/v1/executor/multiproc_executor.py", line 762, in wait_for_ready
    raise e from None
Exception: WorkerProc initialization failed due to an exception in a background process.
  See stack trace for root cause.
```

- Mechanism, verified at vLLM `v0.23.0`: `WorkerProc.wait_for_ready`
  (`vllm/v1/executor/multiproc_executor.py:733-762`) builds a fixed `Exception("WorkerProc
  initialization failed due to an exception in a background process. See stack trace for root
  cause.")` before the wait loop and raises it `from None` on `except EOFError` — i.e. a worker's
  ready pipe closed without a response, so the child died before reporting. The engine therefore
  never holds that worker's traceback, and the SIGTERM/SIGKILL warnings above it come from the
  executor's terminate helper (`:438-449`, the two `logger.warning` calls whose numbers the paste shows)
  — both are teardown, emitted *after* the first worker was already gone.
- `[ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared!` is a CANN TBE
  host-compile child noticing its parent went away, not a plugin frame; its position among the
  shutdown lines is what makes it downstream. Do not derive a cause from it — no file:line in the
  plugin or upstream vLLM emits that text.
- So the answer is the class plus the artifact: "a worker process exited before `READY`, the
  engine discarded its traceback; the first worker-side error is needed" — the `Worker_*`/`VPP`
  log segment, the plog, and the device fault code (`5070xx`/`EZ9999`), exactly what the
  maintainers asked for in the recorded thread. For a worker that dies with no traceback at all,
  continue with the device/teardown sections above rather than with model code.
- Recorded instance #12891 (`vllm-ascend 0.23.0rc1`, `GLM-5.2-w8a8`, 2x Atlas 800 A2, DP2/TP8/EP,
  MTP `deepseek_mtp` k=5, `FULL_DECODE_ONLY`): the launcher is byte-identical to that tag's own
  `docs/source/tutorials/models/GLM5.2.md` A2 recipe (`:317-345`), so diffing the command against
  the docs yields nothing, and the thread closed `wait-feedback`/stale with the maintainer asking
  for the complete initial stack trace — no root cause exists. The `wait_for_ready` line number
  (`:762`) does pin the bundled upstream to vLLM `v0.23.0`, matching `ARG VLLM_TAG=v0.23.0` in the
  tag's Dockerfile, even when the pasted paths are mangled by copy-paste/OCR (`:438`, `:448` and `:762` all match
  `v0.23.0`).

## Silent hang: port listening, AICore 0 %, no error in the vLLM log

```text
[INFO] HCCL(pid,):[heartbeat.cc:1314] [TaskExecStage][HeartbeatAbnormal]local rank [10.119.14.222/0]:
       crimer rank [10.119.14.172/1] status[STUCK] by informer rank [10.119.14.172/1]
[INFO] HCCL(pid,):[heartbeat.cc:1587] [HCCL_TRACE]rank:10.119.14.222/0,
       count of currently executed operators:2706
[INFO] HCCL ... [BuildRemoteResByTag] newtag[group_name_2338_mc2RunAlltoAllDirectFullmesh_device]
```

- This is the plog-only signature of a whole job blocking inside one collective: the service keeps
  listening, `curl` gets no reply, AICore stays at 0, and the vLLM log stops without an ERROR.
  Nothing here is thrown by Python, so a report built on stdout alone looks empty.
- Read it as: the heartbeat names which rank is `STUCK`, the operator count freezes, and the same
  count on every rank of one group means they are waiting inside a single collective rather than
  running slow kernels. The tag/group name carries the operator family
  (`mc2RunAlltoAllDirectFullmesh_device` = MC2 AlltoAll fullmesh), which maps back to the
  per-rank MoE comm-type selection in [pd-disaggregation.md](pd-disaggregation.md).
- A later "another node's rank also STUCK" line is downstream: the other role blocks on its KV
  handshake or its own collective once the first role stops. Order timestamps before citing it.
- Request the plog and per-rank stderr for these reports; without them the only verifiable claim
  is that a collective deadlocked.

## Start-up hang before any model log: fewer DP ranks than `--data-parallel-size`

```text
INFO ... [__init__.py:217] Platform plugin ascend is activated
INFO ... [__init__.py:217] Platform plugin ascend is activated
<no further lines, no ERROR, no "All engine subscriptions received by DP coordinator">
```

- At vLLM `v0.13.0`, `vllm/v1/engine/coordinator.py:180-187` waits for exactly
  `engine_count = parallel_config.data_parallel_size` subscriptions (`:82`, `:117`) by calling
  `publish_back.recv()` once per engine in a blocking loop with no timeout; only after all of
  them arrive does it send `READY` and log `All engine subscriptions received by DP coordinator`
  (`:191`). The wait is unchanged on `main` (2026-10-30, `:245-256`). The launching front-end blocks
  in the same window: `wait_for_engine_startup` (`vllm/v1/engine/utils.py:924-927`) sets
  `local_count = data_parallel_size_local`, `remote_count = len(core_engines) - local_count` and
  spins on `conn_pending`/`start_pending` for cores that were never launched. A DP rank that never
  starts therefore hangs the job silently right after the per-process plugin banner.
- Arithmetic to check before reading any code: the sum over nodes of
  `--data-parallel-size-local` must equal `--data-parallel-size`, and each headless node's
  `--data-parallel-start-rank` must be the first rank it owns (local size × TP should fill that
  node's cards). vLLM `v0.13.0` `vllm/engine/arg_utils.py:1464-1487` derives rank and
  hybrid/external LB from these values, `:1457` requires `data_parallel_size_local` to be 1 or
  `None` when `data_parallel_rank` is set, and `:1471-1479` warns and switches to external LB when
  hybrid LB has local size 1 — nothing validates the sum against `data_parallel_size`.
- Reporter-confirmed fix (DP=4/TP=4 on two 8-card nodes; DP=2/TP=8 worked because one local rank
  × TP=8 filled a node): node0 `--data-parallel-size-local 2`, node1
  `--data-parallel-size-local 2 --data-parallel-start-rank 2`. The same thread reports
  `--gpu-memory-utilization` needing to exceed 0.95 on that hardware.
- Boundary: the rank-count explanation is the reporter's own conclusion plus this code reading.
  It explains why no error is printed, not every start-up silence. Confirm the worker count first
  (`ps`/`npu-smi`: EngineCore processes missing vs `data_parallel_size`), and distinguish this
  from the collective `status[STUCK]` signature above, where all ranks did start. Do not adopt a
  maintainer duplicate-closure that names a different error either: the recorded thread was closed
  against a `bIndex is out of range` report fixed by another PR, which this log-free body never
  shows, while the reporter's own DP-flag fix is what reproduced as working.

## DP all-reduce `Connection closed by peer` (Gloo, not HCCL)

```text
File ".../vllm/forward_context.py", line 92, in set_forward_context
    dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534]
  Connection closed by peer [127.0.0.1]:21428
```

- `get_dp_group().cpu_group` is created with `backend="gloo"`
  (`vllm/distributed/parallel_state.py:180-190` at vLLM `v0.8.0`), so a log line
  `backend=hccl` names the model/TP group, not this call. A report titled "hccl backend" can be
  wrong about which group failed.
- The DP forward path all-reduces the per-rank token count every step
  (`num_tokens_tensor`, `vllm/forward_context.py:85-90` at `v0.8.0`). Every DP rank must execute
  the same number of these calls; if one rank's prompts finish earlier and its process/engine
  exits, the surviving rank's next all_reduce sees the Gloo peer close. Check the per-rank
  `max_tokens` / prompt counts before blaming HCCL: the recorded #541 example gives the two ranks
  `max_tokens` 16 and 20, and the shorter rank finishes and exits first.
- The `ProcessGroupGloo.cpp:715 Warning: Unable to resolve hostname ... loopback` line above it is
  a fallback warning, not the failure; both ranks run on localhost.
- Distinguish this from the `status[STUCK]` / frozen-operator-count hang: there all ranks are
  alive and choose different collectives; here one rank has exited and the survivor gets a socket
  close. The comm-type map is in [pd-disaggregation.md](pd-disaggregation.md).
- Revision pin: `num_tokens_tensor` is absent at vLLM `v0.7.3` and first appears in `v0.8.0`
  (`3e472d882a`, #13591, first tag `v0.8.0rc1`). A `collect_env` line saying vLLM 0.7.3 with this
  frame is a newer source tree (or a backport), not the 0.7.3 release — see
  [revision-pinning.md](revision-pinning.md).

## HCCL link-init failure while pushing weights (pyhccl / trl)

```text
RuntimeError: Multiple errors: [RuntimeError('HCCL error: memory error')]      # training side
RuntimeError: InnerRunOpApi ... OPS function error: HcclAllgather, error code is 6
EI0009: [PID: ...] Transport init error. Reason: [Create][DestLink]Create Dest error!
        createLink para:rank[6]-localUserrank[6]-localIpAddr[172.16.0.114],
        dst_rank[0]-remoteUserrank[0]-remote_ip_addr[172.16.0.114]
RuntimeError: HCCL error: not found specific resource                          # rollout side
```

- `EI0009 ... [Create][DestLink]Create Dest error` is rank-to-rank link creation, not an
  out-of-memory or a too-large tensor. Identical local and remote IPs put both ranks on one host,
  so the link that fails is intra-node; check the device each communicator was bound to and
  whether the peer process reached the same communicator before drawing conclusions.
- The rollout stack is `trl/scripts/vllm_serve.py update_named_param -> vllm_ascend/.../pyhccl.py
  broadcast -> pyhccl_wrapper.py hcclBroadcast/HCCL_CHECK`. At tag `v0.10.2rc1` the raising
  statement is `pyhccl.py:163-165`; `hcclBroadcast` sits at `:163` at `v0.10.2rc1`, `v0.11.0rc0`,
  `v0.11.0rc1` and `v0.11.0` alike, so a `:166` frame is a build between tags — say which one you
  established instead of treating the offset as version evidence.
- **Why the traceback names `pyhccl.py` at all: trl's import shadow, not a plugin monkeypatch.**
  trl `v0.19.0` `trl/scripts/vllm_serve.py:50-59` imports upstream `PyNcclCommunicator` and then,
  under `if is_vllm_ascend_available():`, rebinds the name to
  `vllm_ascend.distributed.device_communicators.pyhccl.PyHcclCommunicator`; `:108` constructs it,
  `:129` allocates the weight tensor, `:132` calls `broadcast`. The plugin never patches upstream
  `pynccl` (`git grep -n 'pynccl' v0.10.2rc1 -- '*.py'` is empty; `patch/worker/patch_common/
  patch_distributed.py` only swaps `GroupCoordinator`), so do not spend calls looking for one.
- **Thread boundary for this class (checked 2026-11):** reporters add 0.11.0rc0 and 0.11.0rc1 to the
  affected set while stating that the 0.8.5 and 0.9.2 official images work; the issue was closed
  stale with no maintainer root cause. So "fails on the 0.10.2rc1/0.11.0rc* line, a 0.9.x image
  avoids it" is reporter experience, and the mechanism (rollout process limited by
  `ASCEND_RT_VISIBLE_DEVICES=7` vs an 8-rank trainer, device binding inside the communicator) stays
  an open candidate.
- Two silent paths to rule out first: `PyHcclCommunicator.__init__` disables itself when
  `world_size == 1` (`:66-69`) or when the HCCL library is missing, and `broadcast` then returns
  without doing anything — a no-op weight update, no error. `broadcast`/`all_reduce` also assert
  the tensor sits on the communicator's device (`:154-156`, still present on `main`).
- A reporter patch that hardcodes `device=0` inside trl's helper is an unvalidated experiment and
  a confounder (`ASCEND_RT_VISIBLE_DEVICES` remapping makes 0 mean different physical cards per
  process); treat it as one variable to revert, not as a fix.

## Free-list corruption assertion

```
assert last_block is not None,
AssertionError: prev_free_block of fake_free_list_tail should always exist
  vllm/v1/core/kv_cache_utils.py:337 FreeKVCacheBlockQueue.append_n
  <- block_pool.free_blocks
  <- single_type_kv_cache_manager.py:426 remove_skipped_blocks
  <- kv_cache_coordinator.py:251 remove_skipped_blocks
  <- kv_cache_manager.py:362 allocate_slots
```

- Means the doubly-linked free list lost the tail's `prev_free_block` before `append_n`; the
  call chain is "free skipped blocks for a request, then return them to the free queue".
- Surrounding features in the observed report (#10724): hybrid KV cache manager, prefix caching,
  `AscendStoreConnector` (KV pool), MTP spec decode, `--async-scheduling`, `block-size 16`.
  Candidate area, not a confirmed cause; the issue closed as non-reproducible.
- Reproduce by reading `popleft_n` / `remove` / `append_n` in `kv_cache_utils.py` at the pinned
  tag, then diff against later fix commits on those files.

## Device-op abort during start-up: `aclnn... failed` with a Python frame at a sync point

```
RuntimeError: Worker failed with error '... the current working operator name is aclnnMoeDistributeDispatchV3
  ... set the environment variable ASCEND_LAUNCH_BLOCKING=1'
[rank2]:[E107 ... OpParamMaker.cpp:444] NPU function error: call aclnnMoeDistributeDispatchV3 failed, error code is 561000
EZ9999: Nnopbase fails to invoke the HcclAllocComResourceByTiling function of the hccl module. ret = 4, comm = ...
  Check NnopbaseExecutorGetMc2Num / NnopbaseExecutorPrepareParamsExt / NnopbaseRunWithWorkspace failed
```

- `ret = 4` is `HCCL_E_INTERNAL`; `error code is 561000` is the aclnn layer. The Python frames
  above it (`multiproc_executor` `collective_rpc`, and any plugin patch such as
  `patch_compile_backend.py`) are where the async device fault surfaces — they are not the bug.
- This is the MC2 (communication-fused) MoE dispatch, so the flags to check are
  `--enable-expert-parallel` and the topology, not the compile backend. Operator-name-to-source
  map, the A2/A3 `select_moe_comm_method` rule and the route that avoids the op are in
  [context-parallel-moe-paths.md](context-parallel-moe-paths.md).
- `ASCEND_LAUNCH_BLOCKING=1` is the next evidence to ask for; the report itself is a start-up
  abort, not a steady-state accuracy problem. Any graph mode must be dropped in that same rerun,
  because the platform raises `ValueError("ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1
  ...")` ([full-decode-only-graph-mode.md](full-decode-only-graph-mode.md)).
- An `aclnn...` error whose Python frame is a *later, unrelated* op (a sampler, a reshape) is the
  same async shape: the operator name is the fault, the frame is where the stream drained.

## Start-up `GuardOnDataDependentSymNode` in the LoRA graph path (0.9.x)

```text
INFO [platform.py:177] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
...
File ".../vllm/lora/ops/torch_ops/lora_ops.py", line 78, in bgmv_shrink
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not guard on
  data-dependent expression Ne(u0, 1) (unhinted: Ne(u0, 1)).  (Size-like symbols: u0)
RuntimeError: Engine core initialization failed. ...
```

- This is fake-tensor/torch.compile tracing of the PyTorch-native LoRA ops during V1 graph
  capture, not an NPU kernel or a bad adapter. At `v0.9.1rc1`
  `vllm_ascend/lora/punica_wrapper/punica_npu.py:6-8` imports `vllm.lora.ops.torch_ops`;
  `_shrink_prefill` -> `sgmv_shrink` -> `bgmv_shrink` reaches `torch.einsum`, and the dynamic
  token dimension (`u0`) cannot be guarded at trace time. The line is emitted only when
  `use_aclgraph` is true (`model_runner_v1.py:325-327`, no LoRA guard at that tag).
- `Engine core initialization failed` is the executor wrapper; classify from the `einsum` /
  `bgmv_shrink` frame. `--enforce-eager` is the maintainer workaround; `capture_model()` then logs
  `Skipping NPU graph capture for eager mode.` and the fake-tensor pass is skipped.
- LoRA + aclgraph first works in `v0.10.1rc1` after #2113 (`c0f0b7081`, the Meta registration in
  the `_C::<op> ... Meta` section below) and #2325 (`3648d18e6`, AscendC kernels); the 2-3 tok/s
  path and the per-tag import split are in [lora-adapters.md](lora-adapters.md).

## HCCL / driver teardown lines (usually downstream)

```
[ERROR] HCCL ... [one_sided_service_adapt.cc:484] The parameter memHandle[...] is invalid.
[ERROR] HCCL ... HcclDeregisterGlobalMem call trace: hcclRet -> 1
[ERROR] GE ... HcclOneSidedCommDestroy / Deinit one sided service failed
[ERROR] RUNTIME ... HostMemFree: halMemFree failed ... rtFreeHost: ErrCode=507899
```

- These appear while the process is already dying (in #10724 they start ~10 s after the
  assertion). They describe unregister/free of host and device memory registered for one-sided
  HCCL communication, typical with KV-pool/AscendStore or mooncake host segments.
- Do not promote them to root cause without evidence that they precede the first fatal error.

## `aicore execution times out` → HCCL watchdog abort → worker death

```text
[Error]: The aicore execution times out.
EE9999[PID: xxxx] rtEventQueryStatus execution failed, reason=aicore timeout
[rank26]:[E910 ... ProcessGroupHCCL.cpp:1869] [Rank 2] HCCL watchdog thread terminated with exception
ERROR ... Worker proc VllmWorker-0 died unexpectedly, shutting down executor.
(DP2) TimeoutError: RPC call to sample_tokens timed out.
```

- The AICore timeout is the primary error and the HCCL watchdog abort is its consequence (the
  recorded report counted 8 `SIGABRT` cores across the 4 DP ranks plus decode). `Worker proc ... died unexpectedly`,
  `Executor failed`, the DP ranks going fatal and `sample_tokens timed out` are the cascade — do not
  report the `sample_tokens` timeout or a `RuntimeError: cancelled` further down as the origin.
- The nearest documented class is in the `v0.21.0rc1` / `v0.20.2rc1` release-notes sections of
  `docs/source/user_guide/release_notes.md` (so every later tag's file carries it): GLM-5.1 hanging
  on the P node in 200K long-sequence 1P1D agent workloads after long-running service, with
  `MoeDistributeDispatchV2`/`aclnnMoeDistributeDispatchV4` reporting an AICore timeout (#9958), and
  GLM5.1 1P1D four-machine hangs in 140K-context tests with CANN 9.0.0 (#10963). Cite them with the
  model/node/CANN delta stated, not as the report's own diagnosis; the known-issues index and its
  accumulation rule are in [revision-pinning.md](revision-pinning.md).
- Before blaming the model code, pair the box's numbers the way the maintainers' evidence requests
  do: the CANN/firmware/driver versions (`collect_env.py`, `npu-smi`), whether host memory, kernel
  OOM and container restarts are clean, and whether the same load reproduces with the speculative
  decoding / sparse flags removed. A host CANN on a different numbering line from the image's
  `FROM quay.io/ascend/cann:<ver>` base means the toolkit was mounted or upgraded outside the image,
  which is itself a delta to state.
- Anchor the first-error timestamp before ordering anything: in multi-node PD pastes the DP-rank
  fatals and the driver queue dumps (`trs_drv` "Logic cq stat show") can sit minutes after the
  AICore timeout that started it.
- **A `107020` returned by an aclnn op belongs to this class when the same paste carries the
  timeout lines**, e.g. `aclnnMoeDistributeDispatchV4 failed, error code is 107020` together with
  `event wait timeout occurred during task execution ... timeout=230s`, `Notify record failed ...
  ret_code=0x150059`, `rtNotifyRecord execution failed, reason=task_timeout` and
  `No available shared memory broadcast block found in 60 seconds` from another engine. The op
  returns the code after its stream task timed out, so the hung collective is the cause; this is
  not the steady-state `107020`/`107027` row above, whose frame is `torch.OutOfMemoryError` from
  `acl_graph.py`'s own synchronize. Recorded: GLM-5.3-w8a8c8 on 4x A2 — go to the MC2 dispatch
  section in [context-parallel-moe-paths.md](context-parallel-moe-paths.md) rather than the
  allocator.

## KV-pool lookup stall: repeated `pool_scheduler.py` lines, then silence

```text
(EngineCore_DP0 pid=...) INFO [pool_scheduler.py:85] Reqid: embd-bench-...-382-0,
  Total tokens 1182, kvpool hit tokens: 1024, need to load: 1024
(EngineCore_DP0 pid=...) INFO [pool_scheduler.py:85] Reqid: ...-383-0, ... need to load: 0
```

- `pool_scheduler.py:85` is `KVPoolScheduler.get_num_new_matched_tokens` of the KV-pool connector
  (`vllm_ascend/distributed/kvpool/` at v0.13.0rc1, `.../kv_transfer/kv_pool/ascend_store/` from
  v0.14.0rc1). One line per waiting request per step; the signal is that the stream stops after
  such a line with no ERROR and a frozen client progress bar.
- Read it as a blocking lookup RPC, not slow inference: the scheduler-side `LookupKeyClient` uses
  `zmq.REQ` with a bare `socket.recv()` (no timeout) and the worker-side `LookupKeyServer` REP
  thread has no exception handling, so a dead lookup thread or a stalled mooncake
  `batch_is_exist` freezes EngineCore in place.
- Requires the KV-pool connector in the serve line; the recorded thread's discriminator was that
  removing `--kv-transfer-config` let the whole benchmark finish. Topology, error-handler defect,
  release window and next checks:
  [kv-pool-ascend-store.md](kv-pool-ascend-store.md).
- A run that keeps serving while `need to load: 0` recurs is **not** this stall: when external hits
  overlap the local prefix cache, the lookup is still paid and the store loads 0 tokens while
  throughput sits below the Mooncake-only baseline — a throughput class, not a freeze
  ([kv-pool-ascend-store.md](kv-pool-ascend-store.md)).

## `KeyError: None` in worker `_prepare_inputs` (stale `input_batch.req_ids`)

```text
vllm_ascend/worker/model_runner_v1.py:1260, in _prepare_inputs
    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
KeyError: None
```

- `req_ids` here is `self.input_batch.req_ids`, i.e. upstream vLLM's
  `vllm/v1/worker/gpu_input_batch.py::InputBatch._req_ids` list. `None` is the placeholder
  `remove_request` writes into it (`:463` at v0.11.0rc3); `condense()` is what slides requests down
  and trims (`del self._req_ids[num_reqs:]`, `:690`). A `None` reaching `_prepare_inputs` means the
  list was not repaired before the forward.
- Two verified traps in that machinery: `condense()` returns early when
  `batch_update_builder.removed` is empty (`:592-595`) — `add_request` pops those indices itself
  (`_register_add_request`, `:295-297`) — and `num_reqs` is `len(req_id_to_index)` (`:829`), not
  the length of `_req_ids`. Either can leave gaps or a too-long list with no exception raised;
  which one produced a given report is unverified.
- Debug it with the raw list, not the property view: print
  `len(input_batch._req_ids)`, `input_batch.num_reqs`, `input_batch._req_ids` and
  `input_batch.batch_update_builder._removed` at the *end* of `_update_states`. The invariant is
  "`_req_ids[:num_reqs]` all non-None and `_req_ids[num_reqs:]` empty". In the recorded report the
  prints inside `_update_states` never showed a `None`, so do not accept a clean print as settling
  it: that run was ray/TP and each worker process keeps its own `InputBatch`, and the gap can
  appear between the print and `_prepare_inputs`.
- Recorded report: `https://github.com/vllm-project/vllm-ascend/issues/5687`, v0.11.0rc3,
  Qwen3-Embedding-8B, `--max-num-seqs 256` (at 8 it disappears), ray executor; a contributor
  suspected the condense removal logic and the issue is still open with no fix or workaround
  beyond lowering `--max-num-seqs`. The report's line numbers sit ~18 lines off the tag
  (`:1260` vs `:1242`), i.e. a slightly different build — say which one you established.

## Start-up duration: `init engine (profile, create kv cache, warmup model) took N seconds`

- The line and its timer are **upstream** code, not plugin code:
  `vllm/v1/engine/core.py::EngineCore._initialize_kv_caches` takes `start = time.time()` before
  `model_executor.get_kv_cache_specs()` and prints `elapsed` right after
  `model_executor.initialize_from_config(kv_cache_configs)` (`v0.9.1:171` and `v0.10.2:218` — the
  same block, so numbers from those two tags are comparable). It therefore sums the worker's
  memory-profiling dummy forward (`determine_available_memory` -> `profile_run`), KV-cache tensor
  creation, and the plugin's `compile_or_warm_up_model` (ACL-graph capture on NPU). Weight loading
  happens *before* this timer and prints its own line, so a large `init engine` value is not a
  checkpoint, disk or download question.
- V1 logs no per-phase duration of its own at these tags (at `v0.10.2`
  `v1/worker/gpu_worker.py:222` `determine_available_memory` calls `profile_run()` and prints
  nothing; the
  `Capturing CUDA graphs (...)` progress line exists in that tag's `v1/worker/gpu_model_runner.py:3134`
  but not at `v0.9.1`). Decompose with a rerun instead of guessing: `--enforce-eager`
  removes capture/warm-up, so `default - eager` is the capture share; the dummy forward is sized by
  `--max-num-batched-tokens` and capture by `cudagraph_capture_sizes` /
  `--compilation-config`, so collect both with the serve line.
- Comparing `main` against a release moves **both** trees. Pin the `main` commit
  (`git rev-list -1 --until=<report timestamp> main`) and read its `ARG VLLM_TAG`: `v0.9.1` pins
  `v0.9.1`, and the plugin `main` of 2025-09-19 pins `v0.10.2`, so the bundled upstream release
  moved with the plugin. Name both pins in the answer; a plugin-only delta is not established.
- Do not walk the release range for a "timing" commit. Recorded forwards spent their whole budget on
  `git log` commit lists between the two tags, a search for issues containing "init engine took", and
  a `VLLM_TAG` diff, and delivered nothing — the composition above plus one `--enforce-eager` rerun is
  what separates a dummy-run regression from a capture regression. If the body carries only the two
  numbers and a link in place of an environment dump, state which sub-phase is unmeasured and ask
  for that rerun; do not infer a mechanism.

## Silence after `Loading model weights took`

```text
Loading safetensors checkpoint shards : 100% completed
[model_runner.py:904] Loading model weights took 3.3461GB
no more logs, http port connection refused
```

- This line is **vllm-ascend's own runner**, not upstream's: at `v0.8.4rc1`
  `vllm_ascend/worker/model_runner.py:904` is
  `logger.info("Loading model weights took %.4f GB", ...)` and the V1 copy sits at
  `model_runner_v1.py:781`; upstream vLLM v0.8.4 logs `GiB` and has no such line in the GPU
  model runner. So the "GB" spelling plus `:904` pins both the revision and the V0 engine — a
  useful pin when the report is a single log line, and a reason to ask whether `VLLM_USE_V1=1`
  was set.
- The line you should expect next is upstream's memory-profiling summary (`vllm/worker/worker.py`
  at vLLM v0.8.4 logs `Memory profiling takes ... seconds` after the dummy run). vllm-ascend's own
  `worker/worker.py` adds no INFO around `determine_num_available_blocks` (`:240`, which runs the
  dummy `profile_run`), `initialize_cache` (`:284`), `_init_cache_engine` (`:305`) or
  `_warm_up_model` (`:325`), and its V0 `model_runner.py` logs only `Starting to load model` and
  the weight line above. A silent stall with a refused HTTP port therefore sits in the dummy
  `profile_run` forward or the cache/graph warm-up, not in weight loading.
- A `wait-feedback`/`stale` closed issue of this shape has no root cause: say where the stall is
  bounded, and ask for the full serve line, the engine (V0/V1), NPU memory at the stall and
  worker stderr instead of guessing a cause.

## Invalid kernel launch: `coreDim` exceeds UINT16_MAX

```text
(Worker_TP1 ...) EE1001: [PID: 38630] The argument is invalid.Reason:
    coreDim=68888 can't be greater than UINT16_MAX.
(Worker_TP1 ...)         The argument is invalid.Reason: rtKernelLaunch execute failed, reason=[invalid value]
(Worker_TP2 ...) RuntimeError: The Inner error is reported as above. The process exits for this inner error,
    and the current working operator name is layer_norm_fwd_kernel.
(Worker_TP2 ...) Since the operator is called asynchronously, the stacktrace may be inaccurate.
    If you want to get the accurate stacktrace, please set the environment variable ASCEND_LAUNCH_BLOCKING=1.
```

- **The Python frames above this line are not the failing kernel.** Under async dispatch they show
  whatever was enqueued (`npu_moe_gating_top_k_softmax`, `npu_grouped_matmul`, `qwen3_next.py`
  MoE frames in #3971) while the `current working operator name is ...` line names the kernel that
  actually failed the launch. Read that line first; only then locate the kernel.
- `layer_norm_fwd_kernel` is the Triton gated RMSNorm copied from FLA. Upstream body:
  `vllm/model_executor/layers/fla/ops/layernorm_guard.py` (`grid = (M, ngroups)`). The plugin ships
  its own copy at `vllm_ascend/ops/fla.py` (v0.10.x–v0.12.x, installed by
  `vllm_ascend/patch/worker/patch_common/patch_triton.py`, which imports the plugin's `LayerNormFn`
  and reassigns `vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn`), moved to
  `vllm_ascend/ops/triton/fla/layernorm_guard.py` from `v0.13.0rc1`. So the failing launch is plugin
  code whose kernel body is upstream. The patch is applied whenever `HAS_TRITON` is true
  (`patch_common/__init__.py`), so no runtime flag avoids it at those tags.
- `M` is the row count after `x.reshape(-1, head_v_dim)`, i.e. **tokens × value-heads-per-rank**
  (Qwen3-Next `core_attn_out` is `[tokens, num_v_heads_local, head_v_dim]`). The NPU launch caps a
  grid dimension at 65535, so the limit is reached by token count: 8192 tokens × 8 local heads
  ≈ 65536 sits exactly on the boundary, which is why an 8192-token request fails while 6144
  (≈49k) works at TP=4 on Qwen3-Next. The reported 68888 implies a few more rows than 8192×8
  (padding or draft tokens); that composition was not established. Divide the reported `coreDim` by
  the per-rank value-head count to get the row count the launch actually saw (68888 / 8 = 8611, not
  8192) instead of assuming it equals the reported input length.
- Fix: #3549 stops passing `M` as a grid dimension and loops row blocks inside the kernel —
  `MAX_CORES = 65535`, the kernel takes an `N_CORES: tl.constexpr`,
  `BLOCK_ROWS = M if M < N_CORES else N_CORES`, `n_iters = M // BLOCK_ROWS`, `remain = M % BLOCK_ROWS`,
  and the call site's `grid = (M, ngroups)` line is deleted — commit `534f32d27` (2025-10-21,
  "[BugFix][mian] Fixed a triton kernel bug of layer_norm_fwd_kernel for Qwen3-next"). First
  release: `v0.11.0rc1` (2025-11-10); **not** in `v0.11.0rc0` (2025-09-30, the #3971 report). Duplicate issue #3548 (same signature at v0.10.2rc1) was closed
  with "solved in #3549"; #3971 itself closed stale with only "use the latest release".
- Workaround candidate at an affected tag: bound rows per forward so a prefill chunk stays under
  the cap (`--max-num-batched-tokens` around the failing input length or below). Costs throughput
  and assumes the failing call's M tracks the scheduled token count — label it a candidate.
- The same arithmetic applies to any Triton kernel whose grid is derived from token count: compare
  the reported input length × per-token expansion with 65535 before reading kernel math.
- **Second recorded instance of this class, and it does *not* print a launch error: the sampler's
  penalty kernel.** `fault kernel_name=token_bin_counts_and_mask_kernel_0`
  (`vllm_ascend/ops/triton/bincount.py`, used from `ops/triton/penalty.py` for
  repetition/presence/frequency penalty) at `v0.20.2rc1` launched a 2D grid
  `(progs, triton.cdiv(total_blocks, progs))` with `progs = min(core_num, total_blocks)` and
  `total_blocks = n_rows * triton.cdiv(n_cols, 256)`, so the launch crossed the same 65535 cap as
  soon as `num_seqs × ceil(max_prompt_len / 256)` grew (long prompt in a padded batch, or many
  sequences). Fix `#10138` (`f95a31ec2`, 2026-06-09) replaces it with a 1D grid plus a
  `tl.range(pid, total_blocks, num_progs)` stride loop and states the trigger as
  `core_num x ceil((num_seqs x ceil(max_prompt_len / 256)) / core_num) > 65535`; first tag
  `v0.21.0rc1`, **absent** from `v0.20.2rc1` (the #10522 report: GLM-5.1, PD P node, ~202k
  `max_model_len`, over-long requests landing in the same batch). The thread's only substantive
  comment is a verification rather than a guess — "已验证该commit可修复" (contributor
  `MarinaMiao`), pointing at `f95a31ec2`. On that build the same overflow surfaces as a **device** fault instead of a launch
  rejection: plog `aivec error`, `The DDR address of the MTE instruction is out of range`,
  `IDEDD ... argAddr[...] has invalid attribute`, then `rtEventSynchronize ... 507035` /
  `aclrtSynchronizeEventImpl ... 507035`. So do not require a `coreDim`/`grid should be less than
  65536` string to route this class — match the `fault kernel_name` and check the tag's grid
  arithmetic. Boundary: one recorded plog line for that launch also prints `blockDim=2` and
  `program id=430`, which does not fit a large grid; the exact launch shape was not re-derived, so
  cite the maintainer's verification (upgrade to `v0.21.0rc1` or a build with `f95a31ec2`) rather
  than a reconstructed mechanism.

## Start-up crash during graph capture: host synchronisation inside the captured stream (`107027`)

```text
(Worker_TP2 pid=102322) File ".../vllm_ascend/quantization/w8a8_dynamic.py", line 384, in init_routing_quant
(Worker_TP2 pid=102322)     global_expert_tokens = torch.bincount(expanded_expert_idx,
(Worker_TP2 pid=102322) RuntimeError: NPU function error:
    c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027
(Worker_TP2 pid=102322) EE9999: [PID: 102322] Not allow to synchronize captured-stream, stream_id=9.[FUNC:StreamSynchronize]
(Worker_TP2 pid=102322)        rtStreamSynchronizeWithTimeout execute failed, reason=[stream is captured]
```

- `107027` on its own is a generic stream-synchronise failure; the `Not allow to synchronize
  captured-stream` / `reason=[stream is captured]` reason lines are what identify this class. Read
  the stack bottom-up: `worker_v1.py::compile_or_warm_up_model -> capture_model -> _capture_aclgraphs
  -> _dummy_run` above a plugin frame means the crash is at **start-up graph capture**, before any
  request is served, not steady state and not a device fault.
- **Here the innermost Python frame is the failing statement** — it is the call that asks for a
  host round-trip, so unlike the `coreDim` async-launch case above the frames are not stale. Any op
  whose result is needed on the host cannot be captured: `torch.bincount` (output length depends on
  the data), `.item()`, `.to("cpu").numpy()`, `nonzero`, a dynamic-shape `empty()`. Whichever comes
  first in the forward is the one that raises.
- Locators at `v0.10.0rc1` (the #2567 report, Qwen3-30B-A3B-W8A8, TP4 + `--enable-expert-parallel`):
  `vllm_ascend/quantization/w8a8_dynamic.py::init_routing_quant:384`
  (`torch.bincount(expanded_expert_idx, minlength=global_num_experts)`), reached from
  `fused_experts_with_all2all` (`:448` `token_counts_combined.to("cpu", non_blocking=True).numpy()`
  and `all_tokens.item()` for the packed output shape). The all2all EP path is host-synchronous by
  design, so it was never capturable; which line raises depends on the CANN/torch_npu build — the
  fused branch at `:420` runs only `if hasattr(torch_npu, "npu_moe_init_routing_quant")`, else the
  `init_routing_quant` fallback with the bincount. The same report on a newer torch_npu aborts at
  `.item()` instead of the bincount.
- State gate at that tag: `vllm_ascend/ascend_forward_context.py::_get_fused_moe_state` with
  `ep_size = get_ep_group().world_size if enable_expert_parallel else 1` — `ep_size == 1` →
  AllGather (decode) / NaiveMulticast (prefill); `ep_size < 16 or with_prefill` → All2All; `>= 16`
  decode → MC2. The all2all frames therefore pin the `--enable-expert-parallel` command: when a
  report pastes a plain-TP command and an EP command with one traceback, only the EP run is
  explained by this stack — do not attribute both, and ask for the non-EP run's own error.
- The reported tag's own docs already carry it, which is cheaper than reading code: `v0.10.0rc1`
  release-notes Known Issues — "Aclgraph could not work with DP + EP currently ... #2229" — and in
  the same file's v0.9.1rc2 section, "Qwen3 MoE aclgraph mode with tp failed when enable ep due to
  bincount error #2226". #2226 recorded the *other* expression of the same call under torch.compile
  (`dynamic shape operator: aten.bincount.default; Operator does not have a meta kernel that
  supports dynamic output shapes`) and was closed as fixed by #2183 on main (present from
  `v0.10.0rc1`) and #2478 (`v0.9.1rc3`); that compile-side fix does not make the call capturable,
  which is why the same call still aborts at `v0.10.0rc1`.
- Release boundary: `init_routing_quant` and `fused_experts_with_all2all` are gone from
  `w8a8_dynamic.py` at `v0.10.1rc1` — removed by `320edde2d` (#2570, refactor onto token
  dispatchers), **not** by #2469 (MC2 method), which is the PR the reporter credited; both shipped
  in `v0.10.1rc1`, so the version advice is unchanged while the mechanism should name the refactor.
  At `v0.10.1rc1` `ops/common_fused_moe.py::forward_impl` also forces `allgathercommimpl` when
  `ep_size < 16`, and that impl has no host round-trip, so the TP4+EP captured path no longer needs
  one. That tag's known issues still say "Aclgraph mode only work with tensor parallel. DP/EP
  doesn't work in this release", so promise that this crash is gone, not that EP + graph mode is
  supported.
- Workaround at an affected tag: `--enforce-eager`. `worker/worker_v1.py::compile_or_warm_up_model`
  calls `capture_model()` only `if not self.model_config.enforce_eager`, and
  `model_runner_v1.py:335` clears `use_aclgraph` for the same reason, so the blocking sync runs on an
  ordinary stream. Code-path verified, not run on hardware here. Capture-size knobs do not help: the
  reporter's `--compilation_config '{"cudagraph_capture_sizes": [16, 32, 64, 128]}'` still failed,
  because every captured size re-runs the same dummy forward through the same host-sync op.
- At another tag, locate the call before suspecting the runtime:
  `git grep -n 'bincount\|\.item()\|to("cpu")' <ref> -- vllm_ascend/` over the fused-MoE and
  quantization paths, then check whether the reported config selects it (EP size, `use_aclgraph`,
  `torchair_graph_config.enabled`).

### Replay-time variant: `107020`/`107027` from the `acl_graph.py` synchronize, reported as OOM

Same error family, different stage and a different triage consequence. Here the service has been
serving, then one request hangs for many minutes and the traceback ends in
`torch.OutOfMemoryError: synchronize ... error code is 107020` (or `107027`), usually followed by
`EngineDeadError` and an HTTP 500.

- Call site at the reported `v0.23.0rc1`: `vllm_ascend/compilation/acl_graph.py:256`
  `torch.npu.current_stream().synchronize()`, inside `ACLGraphWrapper.__call__` on the replay path,
  with no `try`/`except`. It is guarded, so first check the config actually reaches it:
  `is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle`,
  `need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle`, then
  `if not self.enable_enpu and need_sync`. On `main` the same call sits at `:296`.
- Only the **capture** path is diagnosed there: `_STREAM_RESOURCE_ERROR_CODE = "207008"` with markers
  `insufficient_stream_resources` / "stream resources are insufficient" feeding
  `_is_stream_resource_capture_error` -> `_raise_stream_resource_capture_error`. At `v0.23.0rc1`
  nothing in `vllm_ascend/` matches `107020`/`107027` (`git grep -n '107020\|107027' v0.23.0rc1 --
  '*.py'` hits only a `_310p` GDN comment and a unit-test docstring), so a replay-time timeout keeps
  whatever type and text the runtime gave it.
- **Do not let the `OutOfMemoryError` wrapper become the finding.** Treat the synchronize timeout as
  the last symptom and read upward for the first device-side failure (`207001`
  `Memory_Allocation_Failure` / `alloc device memory failed`, `507011`, a `507035` aicore
  exception). Two independent supports: engine-wide documentation in
  `vllm_ascend/attention/sparse_flash_mla.py` (`_ensure_sinks` docstring) — "Device errors are
  reported asynchronously, so the failure surfaces at an unrelated synchronize() rather than at the
  operator that caused it, which makes it very hard to attribute", written for the case where a
  capture-time-allocated sinks tensor is freed and replay traps with `507011`; and the fact that
  neither tree contains code that turns a message into `OutOfMemoryError` — grep finds only
  `except torch.cuda.OutOfMemoryError` with the `--gpu-memory-utilization` hint at
  `vllm/v1/worker/gpu_model_runner.py:5077`, i.e. a handler, not a classifier. Which layer raised
  it in a given report stays unverified until the worker's own traceback is in hand.
- The report in this shape (vllm `0.11.x` + `vllm-ascend 0.23.0rc1`, Qwen3-VL-8B,
  `FULL_AND_PIECEWISE`, capture sizes to 256, `--max-model-len 65535`) got one substantive
  collaborator reply — "vllm-ascend v0.23.0rc1 is aligned with upstream vLLM v0.23.0, not vLLM
  0.11.x ... please provide the unmodified logs" — then `wait-feedback` -> `stale` -> bot closure
  with no root cause. Answer class: version pairing first, then the upward read;
  workarounds are the allocation-pressure knobs (lower `--max-model-len`, trim
  `cudagraph_capture_sizes`, `--enforce-eager`) and an engine restart after aborted requests, since
  the stream state that hangs is not recoverable in-process.
- A message that names `NPUWorkspaceAllocator` / `Memory_Allocation_Failure(EL0004)` is the other
  OOM shape and not this one: it is operator workspace with its own per-stream pool, so the upward
  read above and `--gpu-memory-utilization` do not apply
  ([npu-workspace-memory.md](npu-workspace-memory.md)).

## `execute_dummy_batch` in the traceback: the idle-DP dummy pass, not a request

```text
(Worker_DP6_EP6 pid=29379) ERROR 09-05 08:04:13 [...] File ".../vllm_ascend/worker/worker.py", line 1082, in execute_dummy_batch
(Worker_DP6_EP6 pid=29379) ERROR [...]     self.model_runner._dummy_run(
(Worker_DP6_EP6 pid=29379) ERROR [...]   File ".../vllm_ascend/worker/model_runner_v1.py", line 3525, in _dummy_run
(Worker_DP6_EP6 pid=29379) ERROR [...]   File ".../vllm_ascend/worker/model_runner_v1.py", line 2712, in _model_forward
(Worker_DP6_EP6 pid=29379) ERROR [...]   File ".../vllm_ascend/compilation/acl_graph.py", line 262, in __call__
(Worker_DP6_EP6 pid=29379) ERROR [...]     torch.npu.current_stream().synchronize()
RuntimeError: synchronize:../torch_npu/csrc/core/npu/NPUStream.cpp:414 NPU function error:
c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream()), error code is 507034
[Error]: Vector core execution timed out.
```

- Recorded reports: `vllm-ascend#15892` (DSV4-Flash) and #15930 (DSV4-Pro, four-host A3 PD,
  TP16/DP2 per role), both indexed as known issues of `v0.26.0rc1` in #15985.
- The frame set is the finding. `execute_dummy_batch` at `vllm_ascend/worker/worker.py:1079-1087`
  (`v0.26.0rc1`) takes `num_tokens = getattr(model_runner, "uniform_decode_query_len", 1)` and calls
  `_dummy_run(num_tokens, uniform_decode=True, skip_gdn_state_update=True)`, so the failing kernel ran
  on a **synthetic uniform-decode batch**, not on the request being served. `vllm_ascend/` defines
  `execute_dummy_batch` and never calls it; the caller is upstream.
- Upstream trigger at vLLM `v0.26.0`: `vllm/v1/engine/core.py:2055` calls it inside the step loop when
  `_process_engine_step()` executed nothing while the local engine or another DP rank still has
  unfinished requests; `llm_engine.py:297` does the same when `has_unfinished_requests_dp` sees this
  engine idle but the aggregate not idle (`:202` sets `should_execute_dummy_batch`). Precondition:
  `--data-parallel-size` > 1 with uneven per-rank traffic — which is why these reports say "after
  running for a while under sustained load". Every rank writes to the same log, so read the
  `Worker_DP<l>_...` prefix of the failing lines for the rank that actually trapped (recorded:
  `Worker_DP6_EP6`) instead of inferring the DP size from the report's topology table.
- Keep this apart from the start-up `_dummy_run` crash in the `107027` capture section above
  (profile/graph capture, before serving) and from the GLM `_dummy_run` slot-invalidation
  defect ([glm-moe-accuracy.md](glm-moe-accuracy.md)), whose symptom is garbled DSA output
  rather than a `507034` timeout.
- `uniform_decode_query_len = 1 + num_spec_tokens` (`vllm/v1/worker/gpu_model_runner.py:858` at
  `v0.26.0`), so speculative decoding widens the dummy batch (DSpark k=7 -> 8 tokens); a plain run
  sends one token per idle rank. Under `--enable-expert-parallel` that rank still joins the MoE
  all2all, so dummy-batch and real-batch shapes must agree across the EP group.
- Triage consequence: `AclrtSynchronizeStreamWithTimeout` / `507034` is the last symptom — the dummy
  pass replays through the same unguarded `acl_graph.py:262` synchronize as real decode. Read upward
  for the first device error: the paired report on the same build has
  `IndexCheck_int64_high_performance_0` rejecting very large indices (`507035`) on one decode
  instance before the other times out in `MoeDistributeDispatchV2` (`507034`). Do not attribute the
  crash to the request length, the KV transfer or the model's own inputs: none of them are in the
  dummy batch.
- Levers: there is no disable knob at vLLM `v0.26.0` (`vllm/envs.py` carries only
  `VLLM_RANDOMIZE_DP_DUMMY_INPUTS`, default `0`). `--enforce-eager` removes the aclgraph replay but
  not the dummy forward, so it moves where the timeout surfaces rather than preventing it. Reduce
  per-DP-rank imbalance (load-balancing proxy, one DP engine per instance) or move past the release
  that documents the fix. Candidate, not runtime-verified.
- Boundary: frames and trigger read from the `v0.26.0rc1` plugin tree and vLLM `v0.26.0`, plus the
  reported traceback; nothing was re-run. Whether the dummy batch itself exposes an uninitialised
  index, or the trailing `507034` is only a downstream casualty of the other instance's `507035`,
  stays open.

## Decode-time hang under DP + EP with aclgraph: two chunks, then no traceback

```text
(APIServer pid=9447) INFO:     51.62.10.94:46968 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(Worker_DP0_TP0_EP0 pid=10278) INFO 10-29 06:29:53 [acl_graph.py:187] Replaying aclgraph
(Worker_DP0_TP1_EP1 pid=10281) INFO 10-29 06:29:53 [acl_graph.py:187] Replaying aclgraph
(APIServer ...) [loggers.py:127] ... Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%
(EngineCore_DP2 pid=9588) INFO 10-29 06:30:53 [shm_broadcast.py:466] No available shared memory
  broadcast block found in 60 seconds. ...
(EngineCore_DP0 pid=9586) ERROR 10-29 06:34:54 [core.py:710] EngineCore encountered a fatal error.
(EngineCore_DP0 pid=9586) ERROR ... TimeoutError: RPC call to execute_model timed out.
client: data: {"error": {"message": "EngineCore encountered an issue. See stack trace (above) ...",
  "type": "BadRequestError", "param": null, "code": 400}}
```

- The request is accepted, ~2 SSE chunks arrive from the prefill, then the first *decode* step never
  returns: no worker traceback at all, no `107027`, no `stream is captured`, KV usage stays 0 %.
  Every DP EngineCore then hits the 60-s `shm_broadcast` warning and ~4 min later the
  `RPC call to execute_model timed out` deadline (contrast `sample_tokens` in the first section
  above). Report it as a collective/step-shape deadlock in the worker, not as a slow kernel.
- Only the ranks that own the request log `Replaying aclgraph` (here `DP0_TP0`/`DP0_TP1`); under
  `--enable-expert-parallel` the idle DP ranks must still join the all2all, so a rank set that
  enters the captured decode while the others do not blocks the whole EP group.
- **`--no-enable-expert-parallel` does not remove the cross-DP coupling.** The plugin's own runner
  packs `num_tokens` per DP rank plus `with_prefill`/`enable_dbo` into one `dp_size + 2` int32
  tensor and calls `dist.all_reduce(..., group=get_dp_group().device_group)` at the top of every
  step (`_sync_metadata_across_dp`, `vllm_ascend/worker/model_runner_v1.py:696-738` at
  `v0.10.2rc1`; torchair variant `torchair/torchair_model_runner.py:79-99`), and its own comment
  says the sync "CANNOT skip ... even if we are running in eager mode". Every DP rank must
  therefore enter the same step, so a **single** request on one DP rank depends on the DP
  coordinator's dummy wave for the other seven, while N concurrent requests keep every rank busy.
  The recorded #3128 (0.10.2rc1, Qwen3-235B, `-tp 2 -dp 8`, `--no-enable-expert-parallel`,
  `cudagraph_capture_sizes: [8,1]`, 66560 context: one request hangs "on in-flight replay", then
  RPC timeout, concurrency fine) fits this coupling and not an EP all2all. Candidate, not
  runtime-confirmed; first cuts are `--enforce-eager` (removes the captured path) and DP1.
- That report's only comment (association `NONE`; issue `closed`/`completed` by the reporter
  himself) names "PR #7259" and `_validate_cudagraph_config_for_dp()`. Neither resolves in the
  clones: `git log --all --oneline -S'_validate_cudagraph_config_for_dp'` has no commit at any tag,
  and `--grep='(#7259)'` matches nothing in `vllm-ascend` or upstream `vllm`. The sibling symbol it
  also names, `_sync_metadata_across_dp`, does exist. Verify each identifier a comment supplies —
  one real symbol beside one invented one is normal for generated analyses — and keep the real
  half as a lead only; a PR-number miss is not proof either ([revision-pinning.md](revision-pinning.md)).
- Class evidence, not a statement about the reported build: the same release-notes file at
  `v0.11.0rc0` still carries, in older sections, "When running MoE model, Aclgraph mode only work
  with tensor parallel. DP/EP doesn't work in this release" (v0.9.1) and "Aclgraph could not work
  with DP + EP currently … #2229" (v0.10.0rc1). Cite them as the recurring gap behind the 107027
  class above, and take the tag's own section for what it promises.
- Applicability tests and workaround: `--enforce-eager` skips the captured path
  (`worker/worker_v1.py:306-307` calls `capture_model()` only `if not
  self.model_config.enforce_eager`, and `model_runner_v1.py:551` clears aclgraph for the same
  reason, both at `v0.11.0rc0`); dropping either `--data-parallel-size` or
  `--enable-expert-parallel` removes the EP coupling. Code-verified paths, not run on hardware
  here. Otherwise request per-rank worker stderr/plog — the vLLM log alone does not say which
  collective stalled.
- Candidate fixes stay candidates. `v0.11.0rc0..v0.11.0` contains three aclgraph/MoE/DP commits,
  all contained from `v0.11.0rc2` or `v0.11.0rc3`: `277670730` "failed to update graph task" (#4282),
  `9c6d0b422` Qwen3-MoE full-dp default capture size (#4205), and `a3164ac37` "enable
  force_load_balance in aclgraph" (#4367) — whose own message says it solves OOM, so it cannot be
  cited as the hang fix. The reporter's only follow-up is "v0.11.0-dev 版本已解决" with no commit,
  so the release that fixes this case is unconfirmed.

## Start-up: graph capture at 0 % with 60-s `shm_broadcast` repeats

```text
(Worker_TP0_EP0 pid=48) INFO [backends.py:278] Compiling a graph for compile range (1, 4096) takes 55.98 s
(EngineCore_DP0 pid=38) INFO [kv_cache_utils.py:1291] GPU KV cache size: 521,088 tokens
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|          | 0/6 [00:00<?, ?it/s]
(EngineCore_DP0 pid=38) INFO [shm_broadcast.py:542] No available shared memory broadcast block
  found in 60 seconds. This typically happens when some processes are hanging or doing some
  time-consuming work (e.g. compilation, weight/kv cache quantization).
... the same line every 60 s, no ERROR, no sample_tokens timeout ...
```

- This is a **start-up capture/compile stall**, not the steady-state engine hang above: the KV cache
  is already sized, the API server is not answering requests yet, and `TimeoutError: RPC call to
  sample_tokens timed out` / `EngineDeadError` never appear. The `shm_broadcast` line repeats because
  EngineCore waits on a worker publication that a stuck capture never produces — the same message,
  opposite meaning from the steady-state case.
- Rank CPU is the discriminator: every `VLLM::Worker_TP` pinned at 67-100 % (with EngineCore near
  100 %) says the ranks are spinning inside capture/compile, unlike the plog
  `status[STUCK]`/AICore-0 collective signature. `Compiling a graph ... takes 55.98 s` shows
  per-range capture is already expensive at this model size, so "slow" and "stuck" need a
  per-rank progress line to separate.
- Recorded report: issue #5876, vllm-ascend `0.13.0rc2.dev15+g45c3c279e`, two nodes × 8×910B3 (Kunpeng,
  CANN 8.3.RC2), Qwen3-235B with **TP16 + EP across two nodes**. Maintainer answer: "how you run
  with tp16? with ray? and we suggest to deploy with DP2+TP8"; the reporter replied "可以的" (that
  works) and the thread then only answered why DP2 lowers per-card memory ("the MoE layers in the
  MoE model in vLLM are not replicated across dp"). The topology change is reporter-acknowledged;
  the capture stall itself has no root cause in the thread.
- Cheap next checks: rerun with `--enforce-eager` (skips graph capture) to prove the stall is
  capture-side; `py-spy dump` one `VLLM::Worker_TP`; ask for per-rank stderr and the last
  `Compiling a graph` line each rank printed.
- **Before calling it a capture bug, check which custom op the model is routed to.** A body that
  names one op (`dispatch_ffn_combine_bf16`) and admits the reporter "modified the related files"
  is a local hack on an unsupported path, not a config: at `v0.15.0rc1` the A3 selector needs
  `quant_type == "w8a8_dynamic"`, so no flag routes a BF16 MoE there. Gate, the reporter-verified
  M <= 2048 boundary, and the op's later commits:
  [env-vars-and-additional-config.md](env-vars-and-additional-config.md) (#7154).

## `attempted to run this operator with Meta` during graph warm-up

```text
(VllmWorker rank=0 ...) ERROR [multiproc_executor.py:522]   File ".../torch/_subclasses/fake_tensor.py", line ...
(VllmWorker rank=0 ...) ERROR [multiproc_executor.py:522]   File ".../torch/_ops.py", line ...
NotImplementedError: _C::rotary_embedding: attempted to run this operator with Meta You may have run
into this message while using an operator with PT2 compilation APIs (torch.compile/torch.export); ...
```

- Raised from `torch/_subclasses/fake_tensor.py` -> `torch/_ops.py` seconds after
  `Loading model weights took ... GB`, while the runner captures graphs: the operator has no Meta
  (fake-tensor) kernel, so shape/dtype propagation under `torch.compile`/aclgraph cannot run it.
  No device kernel is involved, so device logs, OOM hints and `ASCEND_LAUNCH_BLOCKING` are irrelevant.
- The name before the colon is the vllm-ascend C++ extension (`_C`), so look in `csrc/`, not at the
  Python caller. At `v0.9.2rc1` `csrc/torch_binding.cpp:302` is
  `ops.impl("rotary_embedding", torch::kPrivateUse1, &vllm_ascend::rotary_embedding)` with no Meta
  entry, and no `csrc/torch_binding_meta.cpp` exists at `v0.9.2rc1`, `v0.10.0rc1` or any v0.9.x tag.
  The same shape applies to every custom op registered only for `kPrivateUse1`
  (`get_masked_input_and_mask`, `bgmv_expand`, `sgmv_expand`, ...).
- Fix: PR #2113 (`c0f0b7081`, "[core] Support capture custom ops into aclgraph") adds
  `csrc/torch_binding_meta.cpp` with `TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, ops)` and meta bodies
  returning empty tensors. First tag containing it: `v0.10.1rc1`; absent from `v0.10.0rc1`, and the
  v0.9.x line never gets it. `v0.9.1` is dated after the merge but sits on a branch without the
  commit, so decide by `git tag --contains c0f0b7081`, never by tag dates.
- Thread route: #1895's maintainer comment names the missing `_meta_registration`, advises keeping
  `_C.rotary_embedding` out of `torch.compile` (for rotary that means
  `custom_rotary_embedding_enabled()` false: fp16 query, neox style, `head_size % 32 == 0`,
  `enable_custom_op()`), and a reporter asked "which tag" without an answer; the release boundary
  above is from `git tag --contains`, not from the thread.
- Bound the dtype workaround before recommending it: vLLM may cast the model down anyway. The
  report's `WARNING [config.py:3371] Casting torch.bfloat16 to torch.float16.` is upstream
  `vllm/config.py:3371 logger.warning("Casting %s to %s.", ...)` at `v0.9.2`, which re-enables
  the fp16 custom path. A run that logs that cast is still exposed. Candidate, not
  thread-verified: disabling graph mode (`--enforce-eager`, or a compilation level below PIECEWISE)
  skips the fake-tensor pass altogether and is the broader workaround when the build cannot move.

## `Aborted request <id>`: the caller went away, not the engine

```text
INFO 04-01 21:24:35 engine.py:280] Added request chatcmpl-93fa13be3529493c887ad55270047cb9.
INFO 04-01 21:25:30 engine.py:298] Aborted request chatcmpl-0a5743529b054b00b676646fa092a677.
```

- At vLLM `v0.7.3` that exact pair is `vllm/engine/multiprocessing/engine.py:280`
  (`logger.info("Added request %s.", ...)`) and `:295-298` (`_handle_abort_request` →
  `logger.info("Aborted request %s.", ...)`). Both print only under `log_requests`, and the two
  line numbers together pin the MQLLMEngine/V0 serving path. The `async_llm_engine.py:223` copy
  logs the same text in the non-multiprocessing engine; `vllm/v1/engine/async_llm.py:300` in V1.
- The abort is client-initiated. The engine client aborts in a `finally:` when the output stream
  ends before `finished` — `# Request was canceled by the client.` /
  `if not finished and not self.errored: await self.abort(request_id)`
  (`vllm/engine/multiprocessing/client.py:663-666`), which reaches the log through an
  `RPCAbortRequest`. Upstream of that, `with_cancellation` (`vllm/entrypoints/utils.py:17-59`)
  cancels the route handler as soon as Starlette delivers `http.disconnect`, and
  `serving_chat.py:688-689` turns the cancellation into "Client disconnected".
- So the line says the request's consumer left mid-generation; it does not say the scheduler, the
  model or the plugin failed, and `POST /v1/chat/completions 200 OK` lines from the requests that
  did finish sit next to it. Match the timestamp deltas against the repro client's own timeout.
- Recorded shape (vllm-ascend#457, vLLM 0.7.3 / vllm-ascend 0.7.3rc2, Qwen2.5-VL-7B, TP=4): the
  repro posts 64 concurrent image requests with `requests.post(..., timeout=60)`; in the pasted
  log `Added request` at 21:24:35 is followed by `Aborted request` at 21:25:30-31 (≈55 s) with
  `Running: 64 reqs`, and the reporter's own benchmark reports Mean TTFT 26060 ms and P99 TTFT
  63250 ms over 1000 requests. A 60 s client timeout below the tail TTFT explains the pairing;
  the thread itself never produced a maintainer cause (9 comments, closed the next day, and a
  later commenter reported the same abort).
- Workaround to report: raise the client timeout above the measured P99 TTFT and bound
  concurrency (`--max-num-seqs` server-side, the client's connection pool client-side) so queueing
  cannot push TTFT past it. A streaming client that stops reading ends in the same abort, so ask
  whether the client timed out, cancelled or closed the socket.
- What made TTFT exceed the timeout is a separate question (VL image prefill, `--max-model-len`
  32768 on one node, chunking). Do not answer it from the abort line, and do not present the
  timeout correlation as a vllm-ascend defect: as recorded, the mechanism is verified and the
  slow-prefill explanation is not.

## Start-up `KeyError: 'model.layers.<N>.self_attn.attn'` while initialising attention backends

```text
vllm_ascend/worker/model_runner_v1.py:2701 in initialize_attn_backend
vllm_ascend/worker/model_runner_v1.py:2649 in get_attn_backends_for_group
vllm/worker/worker.py:451 in initialize_from_config
KeyError: 'model.layers.0.self_attn.attn'
```

- Report shape (`https://github.com/vllm-project/vllm-ascend/issues/6678`, vLLM 0.14.1 +
  vllm-ascend 0.14.0rc1, 4 nodes × 8×910B, GLM-4.7 92 layers, `--pipeline-parallel-size 4`): the
  reporter also lists `model.layers.23/46/69.self_attn.attn` — the first layer of each PP stage.
  The thread is closed `stale`/`wait-feedback` with four comments and no root cause; one commenter
  reports the same shape for GLM-5 TP8/PP4 on four nodes.
- The raising statement is a direct dict index, not a plugin MLA bug: at upstream vLLM `v0.14.1`,
  `vllm/config/vllm.py::get_layers_from_vllm_config(vllm_config, layer_type, layer_names)` returns
  `{name: forward_context[name] for name in layer_names ...}`, and `v0.14.0rc1`
  `model_runner_v1.py:2645-2651` feeds it `kv_cache_group_spec.layer_names`. So the name exists in
  the rank's KV-cache group spec but not in `compilation_config.static_forward_context` at that
  moment.
- Do **not** explain it as "the config lists all 92 layers while the rank holds only its stage":
  at v0.14.1 `get_kv_cache_configs` already filters each worker's groups by that worker's own spec
  keys (`vllm/v1/core/kv_cache_utils.py:1531-1534`, with an assert that every spec key landed in a
  group), and `gpu_model_runner.get_kv_cache_spec` (`:5716-5746`) builds those keys from
  `static_forward_context` itself. The two reads of the same dict disagree, so the mismatch is a
  registration/rename between spec collection and attention-backend init — for example a module
  re-registered under a `unique_prefix` (`vllm_ascend/ops/linear.py:244-248` does exactly that when
  `enable_sp()` is on and the prefix already exists), or a PP-specific model-construction path.
  Which one is unresolved in the recorded pass.
- Next checks: at the failing rank, compare `set(static_forward_context)` against the group's
  `layer_names` and print both at `initialize_attn_backend`; check whether `enable_sp()` /
  `--enable-expert-parallel` is on (that is what makes the rename path live); and retest with
  `--pipeline-parallel-size 1` to confirm PP is a necessary condition. This is a different caller
  from the offloading-connector `self_attn.attn` KeyError in `kv-pool-ascend-store.md`
  (`Attention`-only filter over an MLA model) — do not merge the two answers.

## Device-side bound assertion inside a CANN built-in kernel (`/opp/.../ascendc/<op>_base.h`)

```text
[ERROR] ... [dump_printf.cpp:598][tid:...] [deprecated] NOTICE:assert has been deprecated, please use ascendc_assert instead!
[ASSERT] /home/slave1/Ascend/8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/../ascendc/gather_v3/gather_v3_base.h:137: Assertion `(0 <= val && val < this->gxSize_)'
[ERROR] ... dex 65536 out of range[0 65536)
```

- The assert lives in the **CANN-shipped** implementation of a built-in operator
  (`/usr/local/Ascend/.../opp/built-in/op_impl/ai_core/tbe/impl/.../ascendc/<op>/`), not in the
  plugin's `csrc/` custom ops and not in Python. No plugin tag pin or model patch moves it; what
  triage can establish is which caller feeds the out-of-range index.
- The pair `dex <i> out of range[0 N)` is the whole diagnostic: `<i>` is the offending index,
  `N` (`gxSize_`) the axis size the **caller** allocated. Treat `N` as a caller-side buffer or
  table size and match it against the reported flags and shapes (`--max-model-len`,
  `--max-num-batched-tokens`, mask/position/routing-table sizes) before reading model code; a
  round `N` (65536 here) normally comes from an allocation, not from the prompt length.
- `[deprecated] NOTICE:assert has been deprecated ...` is CANN's banner about the assert macro
  itself, not a statement about the model or the plugin.
- Neighbours to keep apart: `coreDim ... can't be greater than UINT16_MAX` /
  `layer_norm_fwd_kernel` is the Triton grid cap (section above and
  [torch-compile-graph-breaks.md](torch-compile-graph-breaks.md)); `aclnn<op> failed` with a
  Python frame at a sync point is the device-op abort section above; `aicore execution times out`
  is the HCCL watchdog; `torch._dynamo.exc.Unsupported` on the same >65535-token premise is a
  host-side compile fault.
- Recorded case (#6395: vLLM `0.14.1` + vllm-ascend `0.14.0rc1`, Qwen3-Next-80B-A3B, 8x910B2,
  `--max-model-len 131072 --max-num-batched-tokens 131072`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
  failure at >65535 total tokens, assert only in the plog with no Python traceback). That run read
  `patch/worker/patch_qwen3_next.py`, the plugin's `torch.gather`/`index_select` sites and
  `ops/rotary_embedding.py` without converging; a `65536` grep over `vllm_ascend/` at the tag
  finds only FLA Triton `MAX_FUSED_SIZE` limits, `ops/fused_moe/moe_comm_method.py:327
  max_output_size=65536` (the `VLLM_ASCEND_ENABLE_FUSED_MC2 == 1` `dispatch_ffn_combine` call) and
  `patch_qwen3_next.py:67 divide_grid < 65536` (the FLA grid cap). Candidate next checks, none
  verified: locate the caller that allocates a 65536-length axis (advanced indexing such as
  `table[idx]`, `torch.gather`/`index_select`, or the MC2 dispatch `max_output_size`), and bisect
  `--max-num-batched-tokens` below/above 65536 to separate a per-chunk axis from the prompt length.

## Signature-vs-layer map

| signature | layer exposed | flags/paths to inspect first |
| --- | --- | --- |
| `sample_tokens timed out`, shm_broadcast 60 s | worker forward | C8/DSA-CP flags, HCCL collectives, KV-transfer path |
| `Aborted request <id>` after `Added request` | caller / serving layer | client timeout, cancelled stream, `with_cancellation` disconnect; not a worker fault |
| free-list assertion in `append_n` | KV cache manager / block pool | hybrid KV cache manager, prefix caching, KV connector, preemption |
| `assert slot_mapping...numel() == hidden_states.shape[0]` / `slot_mapping_cp.shape[0]` | sparse attention metadata build | `enable_dsa_cp`, SFA C8 paths, graph padding |
| HCCL memHandle / rtFreeHost errors | process teardown | connector teardown, KV pool registration |
| heartbeat `status[STUCK]`, frozen operator count, AICore 0 | HCCL collective, whole job | per-rank comm-type / DP all-reduce decisions (`pd-disaggregation.md`) |
| `aclnnMoeDistributeDispatchV3 failed`, `HcclAllocComResourceByTiling ... ret = 4` | device MC2 MoE dispatch op | `--enable-expert-parallel` + EP/world-size rule in `context-parallel-moe-paths.md` |
| `EI0009 ... Create DestLink`, `not found specific resource`, `HCCL error: memory error` | weight sync communicator | `pyhccl.py` device binding, peer comm init, trl `vllm_serve` patch |
| repeated `pool_scheduler.py:85` lookup lines, then no log | KV-pool connector lookup RPC | `--kv-transfer-config` (AscendStoreConnector), mooncake/memcache backend, `lookup_rpc_port` (`kv-pool-ascend-store.md`) |
| `KeyError: None` in `_prepare_inputs` | worker `input_batch` state | `remove_request`/`condense`/`add_request` interplay, `--max-num-seqs`, ray TP workers |
| `Capturing CUDA graphs ... 0%` + 60-s `shm_broadcast` repeats at start-up | worker graph capture/compile | TP16+EP across nodes, capture sizes, `--enforce-eager`, per-rank stderr |
| `KeyError: 'model.layers.<N>.self_attn.attn'` in `initialize_attn_backend` | `static_forward_context` vs KV-cache group names | PP split, `enable_sp()` prefix renaming, per-rank `static_forward_context` dump |
| `Error in KVCacheTransferThread. error=unhashable type: 'list'` | KV-transfer worker failure path (secondary) | the line before it (`Mooncake transfer failed ... ret<0`, `ascend_direct_transport ... status: 103905`) — `pd-disaggregation.md` |
| `RuntimeError: cancelled` from `patch_shm_broadcast.py:70` / `shm_broadcast.py:700` | engine/executor teardown | worker death: `Worker proc ... died unexpectedly (exit code: N)` plus that rank's stderr/plog; not the shm_broadcast patch |
| `Mooncake transfer failed for request ... ret=-1` | KV-transfer worker | peer handshake port/`kv_port`, transport `status`, plugin build pin (`pd-disaggregation.md`) |
| last log `Loading model weights took ... GB`, port refused | startup, post-load profiling | V0 vs V1 engine, dummy `profile_run`, KV-cache init, graph warm-up |
| `coreDim=... can't be greater than UINT16_MAX`, `current working operator name is layer_norm_fwd_kernel` | Triton launch (NPU grid cap 65535) | tokens × value-heads-per-rank in the FLA gated RMSNorm; fix #3549 from `v0.11.0rc1` |
| `[ASSERT] .../opp/built-in/op_impl/.../ascendc/<op>_base.h:NNN` plus `dex <i> out of range[0 N)` in the plog (recorded: `gather_v3`, `N = 65536`) | CANN built-in device kernel bound check | the caller that allocated the `N`-sized axis: advanced indexing/`gather`/`index_select`, a mask or routing table, MC2 `dispatch_ffn_combine` `max_output_size`; not the plugin `csrc/`, not Dynamo (section above) |
| `AclrtSynchronizeStreamWithTimeout(copy_stream)`, `107027`, `Not allow to synchronize captured-stream`, `reason=[stream is captured]` | host round-trip inside ACL-graph capture (start-up) | `.item()`/`.to("cpu")`/`torch.bincount` in the captured forward; EP all2all MoE path at `v0.10.0rc1`; `--enforce-eager` or a release without the host-sync path |
| `torch.OutOfMemoryError: synchronize ... error code is 107020` (or `107027`) at steady state | the replay-path `torch.npu.current_stream().synchronize()` in `compilation/acl_graph.py` | not an OOM: read up for `207001`/`507011`/`507035`; check the `need_sync` guard and version pairing; restart the engine (section above) |
| `AclrtSynchronizeStreamWithTimeout ... 507034` / `Vector core execution timed out` whose frames pass through `execute_dummy_batch` -> `_dummy_run` -> `acl_graph.py:262` | DP idle-rank dummy pass: a synthetic uniform-decode batch, not the served request | DP>1 with uneven per-rank traffic; read up for the first device error (`507035` `IndexCheck_int64_high_performance_0`, `MoeDistributeDispatchV2`); no disable env at vLLM `v0.26.0` (section above) |
| `_C::<op>: attempted to run this operator with Meta` just after weights load | custom-op ACL-graph capture | Meta registration in `csrc/torch_binding_meta.cpp`; fix #2113 (`c0f0b7081`) from `v0.10.1rc1`; keep the op out of `torch.compile` (section above) |
| `No quantization signature detected ... loaded as float` at start-up | checkpoint load on 310P/A2 with an unquantized HF checkpoint | the documented 310P route is 310P-converted weights with `--quantization ascend` + `--load_format sharded_state` + `--dtype float16` + `--no-enable-prefix-caching`; this line means the run is off that path, not that a kernel is broken (310p-atlas-300i-duo.md) |
| `GuardOnDataDependentSymNode ... Ne(u0, 1)` during PIECEWISE start-up with `--enable-lora` | LoRA graph capture (0.9.x) | `vllm.lora.ops.torch_ops` `bgmv_shrink`/`einsum`; `--enforce-eager`; #2113 + #2325 from `v0.10.1rc1` (`lora-adapters.md`) |
| `Connection closed by peer` from Gloo in `set_forward_context` DP all-reduce | DP CPU (Gloo) group | per-rank step counts / unequal `max_tokens`; `num_tokens_tensor` first in vLLM `v0.8.0`; not HCCL |
| `rejection_sampler.py:278` `repeat_indices_cpu.to(...)` + `SUSPECT REMOTE ERROR` | ACL `507057` (`ACL_ERROR_RT_SUSPECT_REMOTE_ERROR`): a peer device's memory or the inter-device SDMA/HCCL link, surfaced at the local copy — not the sampler | read the *peer* rank's plog and `current working operator name` (`SUSPECT MEM ERROR`/`LINK ERROR`/`HCCL OP RETRY FAILED`), rerun with `ASCEND_LAUNCH_BLOCKING=1` to move the surface; semantics in `pd-disaggregation.md` |
| `Aicore kernel execute failed ... fault kernel_name=reshape_and_cache_200000000` + `The DDR address of the MTE instruction is out of range` + sync `507057` | `torch_npu._npu_reshape_and_cache(..., slot_indices=...)` in the plugin attention impl (recorded at `v0.10.2rc1`, `attention/attention_v1.py:543`); 507057 stays the victim-surface code of the row above, and this run is single-node TP4, so the peer is a sibling rank | read up for the first device error; the recorded Qwen3-Next report has no maintainer root cause, and the untruncated `attn_metadata.slot_mapping` there is a candidate, not a cause (`qwen3-next-family.md`) |
| `aclnnUniqueConsecutive ... 507018 (aicpu exception)` or `aclnnInplaceCopy ... 507057` during `profile_run -> _dummy_run -> maybe_dummy_run_with_lora` | LoRA adapter metadata/activation at start-up, in upstream plain-torch ops (`vllm/lora/punica_wrapper/utils.py compute_meta`, `vllm/lora/layers/base_linear.py set_lora`) — not the plugin AscendC kernels | read the named op and statement; multimodal tower/connector flags change the stacked shapes (`--enable-tower-connector-lora`, `--fully-sharded-loras`); `lora-adapters.md` |
| `The DDR address of the MTE instruction is out of range` with no matching `fault kernel_name` (e.g. a title-level "aivec error") | device memory fault shared by unrelated surfaces | `search/issues` on the exact phrase returns dozens of reports spanning `reshape_and_cache`, ACL-graph capture `507015`, HCCL AIV AllGather, sparse attention; classify by the named kernel/operator plus code, never by the phrase. "aivec" reads as AIV (vector core): an HCCL AIV AllGather fault fits a multi-node DP/TP report as a candidate only (`qwen3-next-family.md`, `pd-disaggregation.md`). Recorded instance #3911 (Qwen3-VL-235B-A22B, 2x A2 DP2xTP8, `v0.11.0rc0`, MMMU_Pro_V): tracebacks are images, the thread is closed `not_planned` with only "use the latest release", so ask for the plog text instead of re-reading it |

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **Teardown logs are not root causes.** HCCL `DeregisterGlobalMem`, `memHandle ... is
  invalid`, `aclrtFreeHost failed`/`rtFreeHost` and process-exit chatter arrive after the
  first fatal error (in #10724 about 10 s later); order the timestamps before citing them. The
  same holds for CANN's `CANNKB` chain (`ForkAwareLocal object has no attribute 'connection'` ->
  `Run CannKbFinalize Error!` -> `Finalize call CannKbFinalize failed. res = [6].`), which is the
  TBE knowledge-bank manager closing inside an exiting forked worker: in the `#4326` body it is
  stamped ~44 min *before* the server crash it is offered as evidence for.
  `RuntimeError: cancelled` belongs to the same family: it is raised from `MessageQueue.acquire_read`
  after the executor shut the queue down, i.e. some worker already died (section above).
  See [full-decode-only-graph-mode.md](full-decode-only-graph-mode.md) for the exit-ordering side.

- **`KeyError: None` in the worker's `_prepare_inputs` is stale `input_batch` state, not a
  scheduler bug.** Its `None` placeholders live in `_req_ids` until `condense()` repairs them, and
  `condense()` early-returns once `add_request` consumed the removed indices; instrument
  `len(_req_ids)` against `num_reqs` in each worker.
