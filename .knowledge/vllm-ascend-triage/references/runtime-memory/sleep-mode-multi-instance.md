# `sleep()` then a second engine on the same cards: the HCCP/HCCL resource refusal

Scope: reports that use **sleep mode as a model switch** — `enable_sleep_mode=True`,
`llm.sleep(level=…)`, "多卡环境下 sleep 掉模型 A，再加载一个运行模型 B 的新实例" — and the
RL sleep/wake runtime around them. The second engine does **not** die of OOM: it dies
creating a communicator. Worked case #3172 (vllm-ascend v0.9.1 `0740d10`, vllm v0.9.1
`b6553be`, 8×910B3, torch_npu 2.5.1.post1, CANN 8.2.RC1, `VLLM_USE_V1=1`, TP8 across 8 cards; the only evidence was a screenshot, OCR'd under
`/workspace/attachments/<issue>/`). The call site is
`../hccl-and-hardware/comm-create-errors.md`; this page owns "what sleep does not
release".

## Verdict first

`sleep()` is a **memory** API, not a teardown. At v0.9.1 it returns the model's
weights and the KV pool to the allocator and nothing else: the worker process stays
alive, its HCCL communicators stay created, and HCCL's host-side **HCCP** process
(RDMA/HDC agent) for those devices stays running. A new engine on the same cards
therefore fails at its first communicator creation — the vendor's own error table
states both the cause and the fix (`EJ0001`: *"Maybe the last training process is
running"* … **Solution: "Wait for 10s after killing the last training process and try
again."**). So the supported shape is **one engine per card set**: sleep/wake the same
engine, or exit the process before starting the next one. Two live engines on one card
set is a capability gap at this ref, not a missing cleanup call (`platform.py:58-59`
`is_sleep_mode_available → True` and `docs/source/user_guide/support_matrix/supported_features.md:28`
Sleep Mode 🟢 Functional are both about the single-engine feature).

## One call, before any source walk

`bash <skill_root>/scripts/cann-error-lookup.sh EJ0001` prints the CANN entry
(errClass `HCCP Errors`, errTitle `HCCP_Process_Initialization_Failure`, message,
Possible Cause/Solution) from whichever export in the snapshot carries the table:
`cann/metadef/error_manager/error_code.json` and
`cann/hcomm/src/common/error_code/error_code.cc:186-197` (verified in a later
snapshot; the v0.9.1-era `cann/` export holds only `catlass`, so there the code is
quoted from the paste and the enum from `Ascend/pytorch`). The message is **HCCP**,
not HCCL — the pasted/OCR'd text reads `Failed to initialize the HCCP process.`, so
grep `Failed to initialize the HCC`, never `HCCL`.

`error code is 7` beside it is `7 = HCCL_E_UNAVAIL` "resource unavailable"
(`Ascend/pytorch/third_party/hccl/inc/hccl/hccl_types.h:38` at the v0.9.1 pin). Say
the enum name and let the HCCP line name the resource; do not translate `7` into a
parameter or NIC claim (`../hccl-and-hardware/comm-create-errors.md` is explicit that `1` is the folded
parameter code).

## What `sleep()` releases, verified at v0.9.1

| Fact | Evidence |
|---|---|
| worker `sleep(level)` is `allocator.sleep(offload_tags=("weights",) if level == 1 else ())` plus one log line (`Sleep mode freed X GiB memory, Y GiB memory is still in use.`) | `vllm_ascend/worker/worker_v1.py:114-125` |
| the allocator's `sleep` iterates **only** `pointer_to_data` (allocations made inside `use_memory_pool`), copies offloaded tags to a pinned CPU tensor, `unmap_and_release`s each handle | `vllm_ascend/device_allocator/camem.py:175-209` |
| only `load_model` (tag `weights`) and `initialize_from_config` (tag `kv_cache`) allocate inside that pool — everything else (comm buffers, workspace, ACL-graph memory, streams/events, the device context) is outside it by construction | `vllm_ascend/worker/worker_v1.py:227-236`, `:262-268` |
| nothing in the package tears a comm down at sleep: `rg -n 'destroy_process_group|destroyHcclComm' vllm_ascend/` → no hits | v0.9.1 tree |
| each communicator registers its destructor with `NpuSysCtrl::RegisterReleaseFn(…, ReleasePriority::PriorityMiddle)` at creation, i.e. release at **process exit** | `Ascend/pytorch/torch_npu/csrc/distributed/HCCLUtils.cpp:141-151` |
| the project states the single-owner assumption itself | `vllm_ascend/worker/worker_v1.py:230` `assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."` |
| both doc examples set `VLLM_WORKER_MULTIPROC_METHOD=spawn` (offline and online) | `docs/source/user_guide/feature_guide/sleep_mode.md:44`, `:79` |

Stop there. Do **not** walk `ProcessGroupHCCL::getHCCLComm` / `createHCCLComm`
internals to "find the leak": the recorded run spent ~5 calls in that C++ and every
fact in the answer was already in the paste plus the four greps above. Re-grep the
three paths per ref — the allocator was rewritten later, and a ref that *does* add a
teardown hook changes the answer to "upgrade".

## Ranked causes, each with its one check

1. **The sleeping engine's process still owns the devices' HCCP/HCCL resources.**
   Check, in one message to the reporter: `npu-smi info` taken *while A is asleep* —
   A's worker processes are still listed — and the `Y GiB memory is still in use`
   number from A's own sleep line. Confidence: source-verified for what sleep does,
   vendor-table-verified for the EJ0001 cause and fix; that the *sleeping* process is
   the "last training process" is inference.
2. **The second engine was built inside the first one's interpreter** (the usual RL
   script: `LLM(A)` → `llm.sleep()` → `LLM(B)`): with the default
   `VLLM_WORKER_MULTIPROC_METHOD=fork` the new workers inherit A's initialised
   ACL/allocator state; the doc's examples set `spawn`, and the per-process assert
   above is the same assumption. Check the launch block for `spawn` and for whether A
   and B are one script.
3. **B's device set overlaps A's** — `ASCEND_RT_VISIBLE_DEVICES` is the platform's
   `device_control_env_var` (`vllm_ascend/platform.py:53`). With A on all 8 cards
   there is no disjoint set left, and that *is* the answer.

**The discriminator between 1 and 3 is one launch**: B on a disjoint card pair while
A sleeps. It starts → the block is the sleeping process's device-side resources, and
per-card co-tenancy has no supported path at this ref. It still fails → per-process or
box-wide HCCP/RA state, and the vendor's 10 s rule (below) is the whole remedy. Never
answer this shape with a synthesized two-instance launcher; the honest deliverable is
the verdict plus these two ablations.

## Action ladder

1. Point B at **disjoint** cards (`ASCEND_RT_VISIBLE_DEVICES`), or
2. let A **exit** (`pkill`/kill the worker processes), wait ≥10 s (vendor's own
   solution text), then start B;
3. if the goal is model switching in one process: one engine, `sleep(level=2)` +
   `wake_up()` re-allocates, but the checkpoint is fixed at construction — a different
   model needs a re-load path (`reload_weights`-style), not a second engine. Label
   that as the design direction, not a tested recipe here;
4. true co-tenancy of two engines on one card set: say it is unsupported at this ref
   and that the question belongs upstream.

## What not to claim

- That the text refers to the reporter's *training* job — it is HCCL's canned wording
  for "another process of mine is already on this device".
- That an `HCCL_*` env var, `empty_cache`, `destroy_process_group` from user code, or a
  `--enforce-eager` re-run fixes it: the first is untested for this code, the second
  frees only allocator blocks, and the third never reaches comm creation.
- That this is an OOM: an OOM in the same window prints `NPU out of memory` /
  `Memory_Allocation_Failure(EL0004)`; here the new instance dies **creating** a
  communicator, before any weight is loaded.
