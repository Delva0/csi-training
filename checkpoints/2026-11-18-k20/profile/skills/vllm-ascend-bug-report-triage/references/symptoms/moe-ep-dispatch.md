# MoE expert-parallel dispatch: comm-resource exhaustion and capture-time host sync

## Scope

Reports whose decisive frame is the MoE all-to-all path — `aclnnMoeDistributeDispatchV4` /
`MoeDistributeCombine`, `torch_npu.npu_moe_distribute_dispatch(_v2)`, or the pre-dispatcher wrappers
`fused_experts_with_all2all` / `init_routing_quant` — with `--enable-expert-parallel` on. Four shapes
live here: the operator cannot get **communication resources** at run time (§A, `561000` / `EI0007`);
the same path is **captured into an ACL graph** while an op needs a host-side scalar (§B, `107027`,
"Not allow to synchronize captured-stream"); the collective **disagrees or loses its shared buffer** —
one rank group runs `MC2` while others run `All2AllV`/`ALLGATHER`, or the all-gather cannot get shared
memory (`dist.all_gather` in `PrepareAndFinalizeWithAll2All.finalize`, ledger #5295 — unresolved), and
when the same collectives instead *stall* silently the scheduler-side divergence is
`pd-collective-hang.md`; and a **hang with no traceback at all**, `VLLM_ASCEND_ENABLE_FUSED_MC2=1`
forced onto a BF16 checkpoint — whose gate admits `w8a8_dynamic` only — which hangs on the first forward
once the `dispatch_ffn_combine` op is reached with only upstream's 60 s `shm_broadcast` warning to show
for it (§C, #7154). A fifth symptom, silent garbage from a doubled MoE all-reduce, is
`flashcomm1-and-moe-reduce.md` §B.

## Common rules

- **The Python picks the dispatcher and the communicator; CANN allocates the QPs.** A resource error
  carries HCCP/HCCL wording, so the answer has two halves: the vllm-ascend decision that made that op
  run (EP on, which dispatcher, which group the comm came from), and the CANN component that owns the
  pool being exhausted. There is no `vllm_ascend/` line that raises it — do not spend turns looking
  for one.
- **`--enable-expert-parallel` off is the confirmed lever for the whole family**, and it costs more
  than it looks: on a MoE model FlashComm1/SP *requires* EP (`platform.py:691-694` at `v0.21.0rc1`
  raises `"Flash Comm v1 requires enable_expert_parallel=True for MoE models."`), so FC1 has to go
  with it, and `VLLM_ASCEND_ENABLE_FLASHCOMM1` is DEPRECATED there (`envs.py:74-75`; the live key is
  `additional_config.enable_flashcomm1`, `ascend_config.py:83-87`). Give both halves of that trade in
  the workaround, and check the tag you are answering for — `enable_flashcomm1` as an
  `additional_config` key first appears at `v0.20.2rc1`
  (`../vllm-ascend/config-flag-lifetimes.md`).
- **Name the communicator and its size.** The MC2 dispatcher takes its HCCL comm from the MC2 group:
  `get_mc2_group()` (`distributed/parallel_state.py:233`) →
  `ops/fused_moe/token_dispatcher.py:101-110` (`device_group._get_backend(...).get_hccl_comm_name(
  local_rank)`, `ep_rank_id`, `ep_world_size`) at `v0.21.0rc1`. QP demand scales with that group,
  multiplied by the number of MC2 ops/streams the step launches.
- **Keep the reporter's op-plugin frame as `reported`.** `third_party/op-plugin/.../MoeDistributeDispatchV2KernelOpApi.cpp:184`
  is their installed wheel; op-plugin is not vendored in the clones, so that path is a pointer, not a
  verifiable line.

## Entry: which collective did the run choose

Read the gate at the reported ref (`vllm_ascend/ascend_forward_context.py`, `select_moe_comm_method`)
— verified at `v0.21.0rc1` (`:262-319`) — before reading any other line:

```bash
cd /workspace/repos/vllm-project/vllm-ascend
git show v0.21.0rc1:vllm_ascend/ascend_forward_context.py | sed -n '262,320p'
```

- `not enable_expert_parallel or get_ep_group().world_size == 1` → `ALLGATHER`
  (`TokenDispatcherWithAllGather`, `ops/fused_moe/token_dispatcher.py:343`).
- A2 (`AscendDeviceType.A2`): `MC2` only when `num_experts // ep_world_size <= 24 and ep_world_size >= 16
  and num_tokens <= mc2_tokens_capacity`; otherwise `ALLGATHER`. A 16-card A2 node with EP is exactly
  the shape that lands on MC2; a smaller EP world or more experts per device does not.
- A3: `FUSED_MC2` (when `enable_fused_mc2` and the guards hold) else `MC2` for small steps, `ALLTOALL`
  for prefill-sized steps; A5: `MC2`/`ALLGATHER`/`ALLTOALL` by world size vs `num_experts_per_tok`.

`MC2`/`FUSED_MC2` use `TokenDispatcherWithMC2`: it takes
`moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)` from `get_mc2_group()`
(`distributed/parallel_state.py:233`) and calls
`torch_npu.npu_moe_distribute_dispatch_v2` / `npu_moe_distribute_dispatch` with `group_ep`,
`ep_world_size`, `moe_expert_num` (`ops/fused_moe/token_dispatcher.py:101-110`, call sites `:232`/`:234`,
`:189`). `aclnnMoeDistributeDispatchV4` in the traceback is that op, on that comm. **Consequence:**
dropping `--enable-expert-parallel` selects `ALLGATHER` and the failing operator is never created —
which is why the flag works, and why the crash says nothing about the model, the W8A8 checkpoint or the
attention backend.

## §A. `call aclnnMoeDistributeDispatchV4 failed, error code is 561000` — QP resources exhausted

### Signature (verbatim, worth grepping)

```text
[E625 ... OpParamMaker.cpp:454] operator():.../MoeDistributeDispatchV2KernelOpApi.cpp:184
  NPU function error: call aclnnMoeDistributeDispatchV4 failed, error code is 561000
Resource_Error(EI0007): Failed to allocate resource qp with info aicpu rank:1,localUserrank:1,
  localIpAddr: <ip>,deviceLogicId:1,sendCqDepth:32768. Reason: Resources are exhausted.
Nnopbase fails to invoke the HcclAllocComResourceByTiling function of the hccl module. ret = 4
Check NnopbaseGetHcomResource(executor, stream) failed
```

`EI0007` + "Resources are exhausted." is HCCP transport allocation, and the reporter's own wording
proves which site ran: `hcomm/src/legacy/ascend910/common/error_code/error_code.cc` defines the template
(errClass `HCCL Errors`, errTitle `Resource_Error`, `ErrMessage` "Failed to allocate resource %s with
info %s. Reason: Resources are exhausted."), while `,sendCqDepth:` is appended to the resource info by
the QP creation path — `platform/resource/transport/host/transport_ibverbs.cc:703-712`
(`hrtRaQpCreateWithAttrs` → `RPT_ENV_ERR(..., "EI0007", {qp, CreateOneQp})`) and the plain
`dlRaTypicalQpCreate` variant in `platform/common/adapter/adapter_hccp.cc:165-172`
(`{qp, CreateQp}`). The `NnopbaseGetHcomResource` / `HcclAllocComResourceByTiling` chain means the
*operator launch* asked HCCL for its comm resources and the pool refused, i.e. too many QPs already
live on that AI CPU rank. Verified by reading `cann/hcomm` locally; the pool size itself is a
CANN/driver build property, not a vllm-ascend setting.

### Worked case: vllm-ascend #10944 (2026-06-25, closed `stale` + `wait-feedback`)

- Config: image `v0.21.0rc1`, GLM-5.1-W8A8, one node 16×910B2C (A2), CANN `9.0.0`, driver
  `npu-smi 24.1.rc2.8`, TP16 + DP1 + `--enable-expert-parallel`, MTP `deepseek_mtp` 3 tokens,
  `{"cudagraph_mode":"FULL_DECODE_ONLY"}`, `multistream_overlap_shared_expert`, `fuse_muls_add`,
  `enable_npugraph_ex`, `HCCL_OP_EXPANSION_MODE=AIV`, `HCCL_BUFFSIZE=200`,
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`.
- Thread: the reporter's own workaround — 「亲测关掉专家并行就可以了」 (drop EP) — confirmed by the
  reporter next day; `MengqingCao`: "there is an functional issue with ep on GLM series";
  `Nagisa125` confirms and routes the GLM-EP fix to the model owner; `wangzhishenghw` points at
  #11394; the thread then goes stale. **The QP mechanism is never stated in the thread.**
- Follow-up #11394 carries the next wall: adding `--additional-config '{"enable_flashcomm1": true,
  "enable_mlapo": true}'` fails with `Flash Comm v1 requires enable_expert_parallel=True for MoE
  models.`, and the same user reports 乱码 with FC1 off and DP2×TP16 without EP. So on this model
  family, "EP off" and "FC1 on" are mutually exclusive at these tags.
- Answer shape: workaround = drop `--enable-expert-parallel` (reporter-confirmed) and drop FC1 with
  it, or pin a build whose GLM EP path is fixed; root cause = the MC2/EP all-to-all op's QP demand
  against the per-rank HCCP pool (`EI0007`), owned by CANN, with the vllm-ascend half being *which*
  communicator the dispatcher took and that EP was on; status = GLM-series EP is a maintained
  functional gap, tracked in #11394, no fix commit identified; limit = no per-rank QP count was
  printed, so "how many MC2 streams were live" is not established.
- Falsifier: reproduce on the same tag with EP off (service starts) and with EP on at a smaller EP
  world size (`--enable-expert-parallel` with TP8/EP8) — if 8 ranks also exhaust, the demand is per-op
  rather than per-rank.

### What not to do

Do not attribute it to `HCCL_BUFFSIZE`, `HCCL_OP_EXPANSION_MODE`, `multistream_overlap_shared_expert`
or MTP without a per-flag experiment: the recorded run cycled through exactly those candidates for
three messages, ran out of output budget, and shipped nothing. The reporter-confirmed lever was in the
thread from the first read.

**The exception is the reporter's own bisect.** When the thread says "removing `PYTORCH_NPU_ALLOC_CONF`
and `HCCL_OP_EXPANSION_MODE` lets it start", the A/B has already been run and the env var *is* the
lever; the mechanism is then the vendor's support matrix for that value, not a code hunt
(`../hccl/env-support-matrix.md`): `AIV` on A2/A3 is symmetric-inference-only, forbids parallel
communicators, and in graph mode restricts AlltoAll/AllGather/ReduceScatter to a single node. Cite the
tutorial block that told them to set it (`GLM5.3.md:158,169` at `v0.27.1rc1`) and the doc's own
tested-image line, then say what the flag costs.

**The number after `error code is` is the op's own ACL code, not a verdict on the class.** `561000` is
HCCP QP exhaustion (§A); a `107020` in the same position (`aclnnMoeDistributeDispatchV4 failed, error
code is 107020`, `v0.26.0rc1`, GLM5.3 4-node EP, preceded by a host `event wait timeout occurred
during task execution … timeout=230s` and `Notify record failed`) is that op's failure under an
execution timeout — do **not** route a bare `107020` here to the ACL-graph replay-sync class
(`graph-capture-and-custom-ops.md` §C2) unless a `capture`/`synchronize` frame is in the traceback, and
do not read it as an OOM. The owning diagnosis for the timeout is EI0002 in
`../hccl/init-failures.md` (find the comm domain, full-cluster vs partial, per-rank `timeStamp`,
`HCCL_ENTRY_LOG_ENABLE=1`).

### Answer shape for this member

```text
symptom -> mechanism: MoE MC2 dispatch at first warm-up dies allocating HCCL comm resources  [verified: A2+EP gate]
workaround: drop --enable-expert-parallel (and FlashComm1 with it) -> ALLGATHER, the MC2 dispatch is
            never built; cost: no MC2 A2A on the MoE path; reporter-confirmed
root cause: the failing allocator is CANN's (nnopbase HcclAllocComResourceByTiling / AICPU QP pool,
            sendCqDepth 32768); the vllm-ascend contribution is only that
            ascend_forward_context.py:262-281 selects MC2 for this deployment
status: GLM-family EP is a maintained functional gap (#10944 -> #11394, stale-closed, no fix commit)
limit: which of experts/streams/MTP inflates the QP request is not verified
falsifier: run with EP off on the same node - the aclnnMoe... line must disappear; if 8 EP ranks also
            exhaust, the demand is per-op rather than per-rank
```

| workaround | evidence | cost |
| --- | --- | --- |
| drop `--enable-expert-parallel` (use `ALLGATHER`) | `Bu1bul` tested it (「亲测关掉专家并行就可以了」), the author confirmed (「可以了，感谢~」), maintainer `Nagisa125` confirmed (「确认关闭 EP 可以规避」) | MC2 A2A throughput on the MoE path, and FC1 has to go too (§Common rules) |
| fewer MC2 comm resources (no `multistream_overlap_shared_expert`, no MTP) | `hypothesis` — untested here, follows from "number of comms/streams" | MTP speedup, shared-expert overlap |
| raise/lower HCCL tunables, newer CANN/driver | `hypothesis`; the reporter's CANN 9.0.0 + `npu-smi 24.1.rc2.8` mismatch is worth quoting, not asserting | none if it fails |

A second, independent defect usually sits underneath and is worth one sentence: #10944's thread records
that even with EP working, GLM-5.x EP is reported to produce wrong output (#11394,
maintainer-attributed), so "EP is usable once the resource fault is fixed" is not established.

## §B. `107027` inside ACL-graph capture: an op that needs a host scalar

### Signature

```text
NPU function error: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027
EE9999: ... Not allow to synchronize captured-stream, stream_id=9.[FUNC:StreamSynchronize]
       rtStreamSynchronizeWithTimeout execute failed, reason=[stream is captured]
```

The Python frame that actually ran is the line *above* the error in the traceback — in #2567 it was
`global_expert_tokens = torch.bincount(expanded_expert_idx, minlength=global_num_experts)`.

### Mechanism at the reported ref (verified, `v0.10.0rc1`)

- `quantization/w8a8_dynamic.py:368` `init_routing_quant` computes the per-expert token counts with
  `torch.bincount` (`:384`); `bincount`'s output length depends on the data (max index / `minlength`),
  so the call needs a host-visible scalar and synchronises a copy stream — legal in eager, illegal
  inside a captured stream.
- The fused replacement is preferred when it exists: inside the `expert_map is not None` (EP) branch,
  `:420` `if hasattr(torch_npu, "npu_moe_init_routing_quant")` returns the counts from the device op
  and never calls `bincount`; the `else` branch is `init_routing_quant`.
  **A `hasattr` gate on the installed torch_npu is the real condition**, so the same tag can work on
  one user's torch_npu and fail on another's.
- Which wrapper runs is chosen in `ascend_forward_context.py:24-46` `_get_fused_moe_state`:
  `ep_size == 1` → AllGather/NaiveMulticast; with `envs.VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ`
  (default `0` at this tag, `envs.py:160-161`) → `All2AllSeq if (ep_size < 16 or with_prefill) else MC2`;
  otherwise `All2All if (ep_size < 16 or with_prefill) else MC2`. A TP2/EP2 or TP4/EP4 W8A8 MoE
  deployment therefore takes `FusedMoEState.All2All` → `w8a8_dynamic.py:1001` `fused_experts_with_all2all`
  → `init_routing_quant` → the `bincount` under capture.

### Worked case: vllm-ascend #2567 (2025-08-27)

Reporter on vllm `0.10.0` + vllm-ascend `v0.10.0rc1`, Qwen3-30B-A3B-W8A8, `--tensor_parallel_size 4
--enable-expert-parallel`, dies in `compile_or_warm_up_model` → `_capture_aclgraphs` → `_dummy_run` →
`fused_experts_with_all2all`. Thread: one maintainer comment, "The PR #2469 fixed it".

- **Ancestry verified:** #2469 = `a6bb502e7` ("[2/N][Feat] Add MC2 communication method for MoE
  layers", merged 2025-08-26) is **not** an ancestor of `v0.10.0rc1` (tag 2025-08-07); `git tag
  --contains` starts at `v0.10.1rc1`.
- **The hunk matches the symptom:** from `v0.10.1rc1` the W8A8 EP path is rebuilt on token dispatchers
  (`ops/moe_dispatcher/token_dispatcher.py` `TokenDispatcherWithAll2AllV`, `distributed/moe_comm_method.py`)
  and `quantization/w8a8_dynamic.py` no longer contains `init_routing_quant`/`torch.bincount` at all —
  the counts come from the dispatched op. So "fixed by #2469" survives the check the skill asks for.
- Workarounds: upgrade to ≥ `v0.10.1rc1` (proposed); or avoid the capture at the reported tag
  (`--enforce-eager` / `-O0`) — proposed, not run: the failing call is inside `_capture_aclgraphs`, so
  no capture means no capture-time sync. Note the reporter's own attempt (setting
  `--compilation_config '{"cudagraph_capture_sizes": …}'`) does not disable capture.

### Generalisation worth reusing

Inside ACL-graph capture, any op whose result shape depends on data (a `bincount`, `.item()`,
`int(x.max())`, a non-native op with a CPU fallback) breaks with `107027` / "stream is captured", and
the traceback points at the *caller* line, not the device op. Ask "which line above this needs a host
scalar", then whether the tag has a fused replacement gated by `hasattr(torch_npu, ...)`.

## §C. `VLLM_ASCEND_ENABLE_FUSED_MC2=1` on a **BF16** MoE: the gate is the quant type, and the fused op stalls above M≈2048 (#7154)

Signature: the engine loads weights and finishes graph compilation, then **hangs** on the first forward;
the only error-looking line anywhere is upstream vLLM's ring-buffer warning
`[shm_broadcast.py:542] No available shared memory broadcast block found in 60 seconds`, and the reporter
says they edited files to reach `dispatch_ffn_combine` at all ("I modified the related files to enable
calling dispatch_ffn_combine op for bf16 models"). Report **#7154** (`v0.15.0rc1`, BF16
Qwen3-235B-A22B MoE, CANN `8.3.RC2`, A3; still open).

Two facts settle it, and both are read from the ref rather than from the log:

1. **The env var cannot select `FUSED_MC2` for a BF16 checkpoint.** At `v0.15.0rc1`,
   `vllm_ascend/ascend_forward_context.py:252` is
   `fused_mc2_enable = envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 and quant_type == "w8a8_dynamic"`
   (`quant_type` read from `hf_text_config.moe_quantize`/`quantize` at `:230-234`; `:253` then adds the
   `dispatch_ffn_combine` guards — EP world ≤ 32, not a draft model, not dynamic EPLB; `:254-267` turn
   that into `FUSED_MC2`/`MC2` for small steps and `FUSED_MC2`/`ALLTOALL` for prefill-sized ones). A BF16
   `quant_type` therefore falls back to `MC2`/`ALLTOALL` whatever the env says — the reporter's hang needs
   their local gate edit, so "set/unset the variable" is not a lever for the code path they are running.
   Read the gate at the *reported* tag: the same term is still there at `v0.16.0rc1:247`, it is **gone**
   by `v0.18.0rc1:253` (the A3 fused branch also moved — `ops/fused_moe/moe_comm_method.py` is the later
   home), and from `v0.20.2rc1` the switch is `additional_config.enable_fused_mc2`
   (`ascend_config.py:152-154`; `v0.23.0:173-175`). The line numbers drift, so read the ref.
2. **The fused op has its own M-dimension limit, and the reporter verified it.** Their follow-up
   (`ShawnHSH`, 2026-03-16): *"when M-dimension of the dispatch_ffn_combine_bf16 op is larger than 2048,
   i.e. the input prompt length is larger than 2048 with 1-batch (or 2048/n with n-batch), an OOM problem
   would occur in the op. We have verified that, the stuck issue would disappear if the input length is
   set less than 2048."* That is a workspace/UB limit inside the operator, so "OOM in the op" surfaces as
   a stuck device task, not as a Python exception — and the hung worker never drains the ring buffer,
   which is all the `shm_broadcast.py:542` line reports. The 2048 bound is `reported`, not in any Python
   gate.

- **The tag's own tutorial already calls the combination unsupported**, on the report's own model:
  `docs/source/tutorials/models/Qwen3-235B-A22B.md:392` at `v0.15.0rc1` — "This is an experimental feature
  which only supports W8A8 quantization on Atlas A3 servers now. If you encounter any problems when using
  this feature, you can disable it by setting `export VLLM_ASCEND_ENABLE_FUSED_MC2=0`". That line is both
  the support statement and the zero-cost workaround (the gate makes it a no-op for BF16 anyway).
- **The 60 s line is a symptom of another process being stuck.** `v0.15.0rc1`'s
  `vllm/distributed/device_communicators/shm_broadcast.py:542` is `logger.info(long_wait_time_msg(…))`
  inside the reader's spin loop, and the message text itself says *"This typically happens when some
  processes are hanging or doing some time-consuming work"* (`:81-87`); its users at that tag are
  `vllm/v1/executor/multiproc_executor.py` and `vllm/distributed/parallel_state.py`. Never answer with
  the broadcast or its timeout; the owning process is the NPU worker in the op, so do not route this
  report to the DP-rank-shortfall or KV-transfer pages on that line alone
  (`../workflow/log-signatures.md`).
- **Levers, in order of what they cost:** stay on the stock gate (drop the local BF16 edit and the env
  var — back to `MC2`/`ALLTOALL`, cost = the fused MoE throughput); if the fused BF16 path is required,
  keep the op's M ≤ 2048 — reporter-verified with 1-batch prompts (`--max-num-batched-tokens 2048` plus
  prompts under the limit; with n-batch the per-request share shrinks accordingly, cost = more prefill
  steps); a build whose gate admits BF16 *and* whose op raises/falls back instead of hanging is a code
  change, `proposed` only.
- **Falsifier:** with `--max-num-batched-tokens 2048` and a short prompt the first forward completes
  (reporter-confirmed) — if it still hangs at M ≤ 2048, this is not the op's workspace limit and the
  search moves to the collective itself (`§A`'s `EI0007`, `pd-collective-hang.md`).
- **What to ask for:** the exact edited files (the reporter links an internal Huawei wiki such as
  "dispatchffn大融合算子接入记录", which is not fetchable and cannot be re-derived), and the plog of the
  stuck forward — the CANN side is where a stuck device task shows (`device-kernel-faults.md`).
