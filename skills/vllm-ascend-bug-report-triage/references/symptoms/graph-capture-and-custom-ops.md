# Graph capture and custom ops: a missing meta kernel, an op the SoC lacks, or a host copy inside a captured stream

## Scope

A run dies while a graph is being built, traced, captured or replayed — ACL graph (`PIECEWISE` with
`use_inductor=False`, the only NPU compile mode), `torch.compile`, torchair — and the owning artifact
is a **custom operator or the capture itself**, not model code. Four verified shapes:

- **A. No meta/fake-tensor implementation**: `NotImplementedError: _C::<op>: attempted to run this
  operator with Meta`. The kernel exists for real execution, so eager works and only the trace fails
  (`rotary_embedding`, #1895).
- **B. The op exists in Python, the device does not have it**: `EZ3002 Optype [TransData] of Ops
  kernel [AIcoreEngine] is unsupported ... not found in this op store`, after a
  `torch_npu/op_plugin/atb/_atb_ops.py` frame (`_npu_rotary_embedding` on 310P3, #1383).
- **C. The runtime refuses an operation because the stream is capturing**: `rtMemcpy execution failed,
  reason=the current capture mode does not support this operation`, usually beside
  `AclrtSynchronizeDeviceWithTimeout … 507015` / "The aicore execution is abnormal", with a frame on
  the ACL-graph wrapper's `synchronize()` (#6529).
- **C2. The replay barrier has no handler**, so any failure parked on the captured stream surfaces there raw
  (`107020` relabelled as `OutOfMemoryError` after an earlier `207001` allocation failure, #13218).

Not this page: a DFX kernel name/hash plus `runtime error 507015` *during real execution*
(`device-kernel-faults.md`), a Triton launch-geometry ceiling (`kernel-launch-limits.md`), a startup
stall with no traceback (`startup-and-stall.md`), a **written** `{"cudagraph_mode":
"FULL_DECODE_ONLY"}` on a model the same tag cannot serve — that is a mode-support gap, not a capture
bug: `aclgraph-mode-and-capture.md` §A — or a `torch._dynamo` error whose `Explanation:` names a
CPython builtin (`posix.putenv`, `builtins.open`): that is a graph break from a side effect in the
traced Python, `dynamo-graph-break.md`, and it never reaches the device.

## Common rules

- **`attempted to run this operator with Meta` is about *tracing*, not hardware, dtype support or the
  missing `vllm._C` warning.** The framework is building a fake-tensor graph and asks the operator for
  a shape-only (Meta) implementation; a custom op registered only for its device backend has none, so
  tracing raises instead of falling back to eager. The Python frames above it are torch internals —
  `torch/_subclasses/fake_tensor.py` (`r = func(*args, **kwargs)`) and `torch/_ops.py` (`return
  self._op(*args, **kwargs)`) — and no `vllm_ascend/` frame appears at all.
- **vLLM V1 puts custom ops into the graph on purpose.** At `v0.9.2rc1`, `vllm_ascend/platform.py`
  `check_and_update_config` sets `compilation_config.custom_ops = ["all"]` for V1 on anything but 310P,
  so `torch.ops._C.<op>` is traced, not executed eagerly. That is why an op without a Meta impl fails
  in *any* default V1 config.
- **`enable_custom_op()` is not a switch.** At `v0.9.2rc1` (`utils.py:217-236`) it imports
  `vllm_ascend.vllm_ascend_C` once, memoises `True`, and returns `False` only if the extension failed
  to import. There is no env var that turns it off at that ref; the usable levers are the op's own
  predicate and the compilation level.
- **`--enforce-eager` / `-O0` is a real lever, verified in the same hook**: `if enforce_eager or
  compilation_config.level == CompilationLevel.NO_COMPILATION: level = NO_COMPILATION`, otherwise a
  non-PIECEWISE level is downgraded to `NO_COMPILATION` with a warning. Eager mode removes the tracing
  path entirely; the cost is ACL-graph capture and every compile-time optimisation.
- **A `Casting torch.bfloat16 to torch.float16` line is evidence of the runtime dtype**, not of the
  platform refusing bf16: it is `vllm/config.py::_get_and_verify_dtype` (`:3371` at `v0.9.2`) printing
  the configured/resolved pair. Check the reporter's command for `--dtype`; at `vLLM 0.9.2` the base
  `Platform.supported_dtypes` is `[bfloat16, float16, float32]` and vllm-ascend `0.9.2rc1` does not
  override it, so `--dtype auto` keeps the checkpoint's bf16 and only an explicit fp16 pin reaches an
  fp16-only gate.

## Read the deepest frame, then the op's registration

1. Keep the last three frames, not the message. Both shapes are decided by *which* op the Python
   selected and *where* it is dispatched: `vllm_ascend/ops/<x>.py` → `torch.ops._C...` or
   `torch_npu.<op>` → `torch_npu/op_plugin/atb/_atb_ops.py` (ATB) or an aclnn wrapper.
2. `Failed to import from vllm._C with ModuleNotFoundError` in the same log (the reporter's own build)
   means **every** `_C.*` custom op is unavailable; a patched `forward_oot` that gates on
   `enable_custom_op()` then silently takes the `torch_npu.*` fallback. Quote that line: it retires half
   the candidate paths in one read — and it is not itself the cause.
3. `current working operator name is Identity` and "Since the operator is called asynchronously, the
   stacktrace may be inaccurate" mean the emitted op name is a neighbour; the reliable trace needs
   `ASCEND_LAUNCH_BLOCKING=1` (reporters rarely have it set — say so rather than trusting the name).
4. The registration half is in `csrc/`: `torch_binding.cpp` (real kernel) versus
   `torch_binding_meta.cpp` (meta/fake kernel). `git ls-tree --name-only <ref> -- csrc | grep meta`
   answers "could this op be traced at all on that build?" in one call (`../vllm-ascend/repo-map.md`,
   `csrc/` row).

## A. `NotImplementedError: _C::<op>` under ACL-graph capture (#1895)

Verified at `v0.9.2rc1` (tag `b5b7e0e`, 2025-07-11; reporter's vLLM `0.9.2`,
`torch-npu 2.5.1.post1.dev20250619`, Qwen2.5-32B, TP4, no `--dtype`, `--enforce-eager` off,
`PIECEWISE compilation enabled on NPU` in the log). The weights load
(`Loading model weights took 15.3924 GB`) and ~2 s later every rank dies in `WorkerProc hit an
exception` with the `Meta` line — which is why the report looks like a weight-load or memory bug.

Verified chain at `v0.9.2rc1`:

1. `custom_rotary_embedding_enabled(query, neox_style, head_size)`
   (`vllm_ascend/ops/rotary_embedding.py:29`) is `query.dtype == torch.float16 and neox_style and
   head_size % 32 == 0 and enable_custom_op()`; Qwen2.5-32B (head_size 128, neox style, fp16 runtime)
   satisfies all four.
2. `rope_forward_oot` (`:60-69`, call at `:62`) then calls `torch.ops._C.rotary_embedding(...)` when
   that predicate holds **and** `not is_310p()`; every other tolerance falls through to
   `torch_npu._npu_rotary_embedding` (`:80-87`), a torch_npu op that does have a Meta impl.
3. No Meta implementation exists for `_C::rotary_embedding` at that ref
   (`vllm_ascend/meta_registration.py` and `csrc/torch_binding_meta.cpp` do not exist before the fix
   below), so tracing fails.
4. The failure surfaces in the profile/dummy run right after weight load. The log's
   `Casting torch.bfloat16 to torch.float16` line shows the resolved runtime dtype was FP16, which is
   what selects this gate; the recorded thread does not settle whether that came from an explicit
   `--dtype float16` or from a checkpoint/config default (see the common rules — `--dtype auto` keeps a
   bf16 checkpoint), so read the reporter's own command before asserting either.

- **Workaround, in order of preference.** (a) Do not force FP16: `--dtype bfloat16` (or drop
  `--dtype float16`) makes the gate false and rope runs through `torch_npu._npu_rotary_embedding`; cost:
  bf16 numerics differ from the fp16 baseline the reporter compared against. (b) `--enforce-eager` (or
  `-O0` / `--compilation-config '{"level":0}'`) keeps fp16 and skips compilation; cost: no ACL graph, so
  decode throughput drops. (c) upgrade past `v0.10.1rc1`. Thread support: maintainer `yiz-liu`
  (2025-07-21) gives exactly the (a)-style advice — "due to the absence of `_meta_registration`, the
  custom operation `_C.rotary_embedding` is not compatible with `torch.compile`" — and `ganyi1996ppo`
  names PR #2113; `Yikun` closes with "Resolved by" it.
- **Fix and inclusion (verified).** #2113 = `c0f0b7081`, "[core] Support capture custom ops into
  aclgraph" (2025-08-11), adds `csrc/torch_binding_meta.cpp` (Meta impls registered with
  `TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, ops)` for `rotary_embedding` and `get_masked_input_and_mask`)
  and `vllm_ascend/meta_registration.py` (Python route, `register_meta_if_necessary`, also covering
  `bgmv_expand` / `sgmv_expand`). `git tag --contains c0f0b7081` starts at **`v0.10.1rc1`**; the whole
  `v0.9.x` line and `v0.10.0rc1` lack it, so "it is fixed upstream" is wrong for a `0.9.2rc1` report.
- **Falsifying checks.** TP=1 or `--enforce-eager` still reaching the same line rules the op gate out;
  the error naming an op whose `torch_binding_meta.cpp` entry already existed at the ref means the ref is
  wrong or the failure is at another stage; a crash *after* the first request (not during warm-up) is
  another class.

## B. ATB rope on 310P: the op store has no `TransData` (#1383)

Verified at `v0.9.1rc1` (tag `c30ddb8`, 2025-06-22), Atlas 300I Duo / 310P3, Qwen2.5-0.5B, V0 engine,
`Triton not installed`, built from source on the mainline pins.

- Frames, in order: `vllm_ascend/ops/rotary_embedding.py:69 torch_npu._npu_rotary_embedding(...)` →
  `torch_npu/op_plugin/atb/_atb_ops.py:72/:80` → `torch.ops.atb.<op>` → `ERR00100 PTA call acl api
  failed` / `EZ3002 Optype [TransData] ... [tbe-custom]:op type TransData is not found in this op
  store` → `aicpu_ascend_kernel ... Transdata op, groups should be greater than 1, but now is 1` →
  GE `SelectEngine ... engine_place.cc:148` / `build op model failed, result = 500002`.
- So the failing artifact is the **ATB** rope graph, not an aclnn kernel and not vllm-ascend
  arithmetic: that build's `_npu_rotary_embedding` is an ATB op, and 310P's op information library does
  not carry the format conversion the graph needs. The dispatch at that ref has exactly two paths —
  custom `_C.rotary_embedding` (FP16-only, skipped on 310P by `not is_310p()`, and unavailable anyway
  because `vllm._C` failed to import) and `_npu_rotary_embedding` — so rope has no third option.
- Repo-side state: `vllm_ascend/_310p/` does **not** exist before `v0.14.0rc1` (`1ffca8673`, #5776,
  2026-01-17, "Add basic 310p support. Only dense models work with eager mode now": absent at
  `v0.12.0rc1`/`v0.13.0rc1`, 12 files at `v0.14.0rc1`). At `v0.9.1rc1` the 310P deliverable is the
  **separate image** `quay.io/ascend/vllm-ascend:<version>-310p` documented in
  `docs/source/tutorials/single_node_300i.md` at that tag, which is why the reporter's own docs mandate
  it; a mainline wheel built from source carries the mainline ATB/CANN generation instead.
- Workaround: run a `-310p` image of that release line or a later one (confirm the exact
  `vX.Y.Z-310p` tag exists, since rc tags need not be published) instead of a mainline rc wheel built
  from source. Getting Triton imported (which makes rope take the Triton path) only becomes a lever at
  refs whose `ops/rotary_embedding.py` has a Triton branch; `v0.9.1rc1` has none (`git show
  v0.9.1rc1:vllm_ascend/ops/rotary_embedding.py | grep -n triton` → nothing), while `v0.18.0` prefers
  `rope_forward_triton` when `HAS_TRITON`.
- **Not established**: that any single vllm-ascend commit fixes this rope call. The base
  `rope_forward_oot` still calls `torch_npu._npu_rotary_embedding` at `v0.14.0rc1` and `v0.18.0`, so the
  change that makes 310P work is in the platform/image pins (or in the Triton branch), not provably in
  this file. Do not post "fixed by #5776" for a **model-layer** op like this — but do post it for the
  worker-level warm-up call that the `_310p` worker overrides (§B2, #5170).

Falsifier for "ATB op missing on this SoC": the deepest frame is an aclnn wrapper or a
`torch.ops._C*` op instead of `op_plugin/atb/*`; or the same error appears on A2/A3 hardware.

### B2. ATB `LinearOperation` in the startup warm-up on 310P: `RuntimeError: LinearOperation CreateOperation failed!` (#5170)

The sibling of §B with the opposite answer: **here #5776's `_310p` tree *is* the fix**, because the 310P
worker overrides the very method that fails. Verified at `v0.11.0` (reported), against the clone.

- Frames, bottom-up: `patch/platform/patch_core.py:49 run_engine_core` → `vllm/v1/engine/core.py:207
  _initialize_kv_caches` → `collective_rpc("compile_or_warm_up_model")` → `vllm_ascend/worker/worker_v1.py:332
  self._warm_up_atb()` → `:341 torch_npu._npu_matmul_add_fp32(x, weight, c)` → `torch/_ops.py:1158` →
  `RuntimeError: LinearOperation CreateOperation failed!`, then `Engine core initialization failed`. So the
  engine dies *after* KV-cache sizing and *before* any request, with a `vllm_ascend` frame present.
- The call is unconditional on every device class: `_warm_up_atb` builds three random tensors and calls
  `torch_npu._npu_matmul_add_fp32`, which is an **ATB `LinearOperation`**, and the in-code comment says it
  exists only to avoid a first-op slowdown (`ReshapeAndCache`) — it is a *performance* warm-up, never a
  correctness step. torch_npu's own maintainers answer the API question in
  `Ascend/pytorch#94`: `_npu_matmul_add_fp32` is a **sample** API, not intended for external use and not
  supported on every device model (relayed by a reporter in the thread, `reported`).
- The string most reports paste is the executor's wrapper:
  `RuntimeError: Worker failed with error 'LinearOperation CreateOperation failed!', please check the stack
  trace above for the root cause` — read past it to the `_warm_up_atb` frame before classifying.
- The method's home moves, the call does not: `worker/worker_v1.py` at `v0.11.0`, `worker/worker.py` from
  `v0.13.0` on (a `git show <later-tag>:worker_v1.py` returning nothing is a rename, not a removal). The
  base `worker.py:399` at `v0.14.0rc1` also gates the call on
  `get_ascend_device_type() != AscendDeviceType.A5`, so A5 never reaches it either.
- Repo state: `vllm_ascend/_310p/` does not exist before `v0.14.0rc1` (#5776, `1ffca8673`, 2026-01-17), and
  from `v0.14.0rc1` `vllm_ascend/_310p/worker_310p.py:35-37` overrides it: `def _warm_up_atb(self):` then
  `# 310p device do not support torch_npu._npu_matmul_add_fp32 atb ops`, then
  `logger.info("Skip warm-up atb ops for 310P device")` and no call at all. `git tag --contains 1ffca8673` starts
  at `v0.14.0rc1`. The base path is **unchanged**: `vllm_ascend/worker/worker.py:1102-1106` on `main` still
  calls `_npu_matmul_add_fp32`, so "the warm-up was removed" is wrong — only 310P stops calling it.
- Workaround, ordered. (a) Run a 310P build at `v0.14.0rc1` or later (`_310p` tree present; dense models,
  `--enforce-eager` only, per the PR's own scope) — the image must be the `-310p` one for that release line.
  (b) On the reporter's own `0.11.0`/`0.11.0rc3`, make `_warm_up_atb` a no-op — delete the
  `self._warm_up_atb()` call at `worker_v1.py:332`, or `return` at the top of `:337` — cost: the first
  `ReshapeAndCache` is not pre-warmed (a first-request latency effect only); `proposed`, not
  reporter-confirmed. (c) Do **not** offer "use the `-310p` image of the same release": a 310P image at
  `v0.11.0rc3` hits the identical traceback (a second reporter in the thread, `reported`), because the
  override does not exist yet at that tag. `v0.12.0rc1` still carries the unconditional call, so the failure
  covers every 310P start on `≤ v0.12.0rc1` and nothing earlier in the `0.13`/`0.14` rc line is implied.
- Why the 310P/other split (`hypothesis`, built from the override's own comment and §B's op-store class):
  the op reaches ATB's fp32 `LinearOperation` kernel, which the 310P build does not carry, while A2/A3
  builds run it unchanged — the same family as §B's `TransData` gap, a different op. What is `verified` is
  that A2/A3 code paths still call it on `main` and that 310P stopped. The thread's closing maintainer line — "resolved in the latest version,
  please use the latest release or rc version" — is correct only from `v0.14.0rc1`, and only via (a).
- Falsifier: the same `LinearOperation CreateOperation failed!` on A2/A3 hardware (then the op, not the SoC,
  is the defect); or no `_npu_matmul_add_fp32`/`_warm_up_atb` frame under
  `compile_or_warm_up_model` (then it is not this row — a fatal `LinearOperation` raised anywhere *else*
  during warm-up, e.g. from a matmul in the model forward, is a plain ATB/op-store fault).
- **Generalise §B's warning, do not repeat it verbatim**: for a *model-layer* op (`rope_forward_oot`) #5776
  is not the fix; for a *worker-level* call the `_310p` worker overrides, it is. The discrimination is one
  `git grep -n '<the symbol>' <tag> -- vllm_ascend/_310p`, not the PR title.

## C. A synchronous operation inside the capture window (`107041` / `EE1016`), often beside `507015` (#6529)

Signature: `rtMemcpy execution failed, reason=the current capture mode does not support this operation`,
`operation not permitted when a stream is capturing and the specified capture mode is not relaxed,
ret=107041`, or `MemCopySync failed. Reason: The current thread … is in the capture state … This
operation is supported only in the RELAXED mode`. Often printed beside a device-side error such as
`507015 The aicore execution is abnormal` in the same run, which makes one fault look like two.

Worked case: vllm-ascend #6529 — DeepSeek-V3.2, four-node `1P1D`, image `v0.14.0rc1` with CANN 8.5.0
(`0.13.0rc2` reports the same), where the first `curl` against the load-balancer takes both decode nodes
down. The pasted log carries both halves:

```text
RuntimeError: npuSynchronizeDevice: ... AclrtSynchronizeDeviceWithTimeout, error code is 507015
[Error]: The aicore execution is abnormal.
EE9999: rtMemcpy execution failed, reason=the current capture mode does not support this operation
model_runner_v1.py:1905 (_generate_dummy_run_hidden_states) -> model(...) -> acl_graph.py:199 (__call__)
  -> torch.npu.synchronize()
```

The capture-window string is **not in any clone** — `grep -rn "capture mode does not support"
/workspace/repos` is empty — because it is the CANN runtime's capture-window check, documented in the
CANN clone:

| fact | where (`cann/runtime`) |
| --- | --- |
| `107041` = `ACL_ERROR_RT_CAPTURE_MODE_BLOCK_ASYNC`, "async oper convert to sync oper, stream is captured, not support current oper" | `include/external/acl/error_codes/rt_error_codes.h:62`; `docs/zh/api_ref/25-01_aclError.md:149` |
| `EE1016` "Not_Supported": `MemCopySync failed. Reason: The current thread … is in the capture state … only in the RELAXED mode` | `docs/en/error_code_ref/RTS-Errors/EE1016-Not_Supported.md` |
| between `aclmdlRICaptureBegin` and `aclmdlRICaptureEnd`, a sync/query on stream/event/device/context fails; in `ACL_MODEL_RI_CAPTURE_MODE_GLOBAL` a synchronous `aclrtMemcpy`/`aclrtMemset` also fails. Remedies: move the sync out of the window, or capture in `ACL_MODEL_RI_CAPTURE_MODE_RELAXED` | `docs/zh/FAQ/ACLGraph_restrictions.md` |

What the frames mean (verified at `v0.14.0rc1`):

- `vllm_ascend/compilation/acl_graph.py:185` logs `Replaying aclgraph`; `:198-200` is the
  `if self.runtime_mode != CUDAGraphMode.FULL …` guard, `torch.npu.synchronize()` at `:199`, then
  `entry.aclgraph.replay()` at `:200` — the **replay** branch, entered only when
  `self.runtime_mode != CUDAGraphMode.FULL or not forward_context.is_draft_model or not use_eagle`, and
  after the `is_debugging_mode` address assert at `:176`. A traceback naming `:199` (as the worked
  report does) is therefore *replay*: the exception belongs to an earlier asynchronous operation, so do
  not place a capture-time refusal on that line.
- **The same line moves, and its condition flips.** At `v0.23.0rc1` the replay sync is
  `compilation/acl_graph.py:256` with
  `need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle` (`:254`) and
  `if not self.enable_enpu and need_sync:` (`:255`) — the *opposite* sense of the `!= FULL` guard at
  `:199` on `v0.14.0rc1`, and two new skip conditions (`--enable-enpu`, merged EAGLE draft replay). Read
  the condition at the reporter's ref before saying which phase a traceback names; the polarity changed
  between `v0.14` and `v0.23`.
- `_generate_dummy_run_hidden_states` is `vllm_ascend/worker/model_runner_v1.py:1902`, called from
  `_dummy_run` at `:2118` inside `set_ascend_forward_context(..., aclgraph_runtime_mode=
  cudagraph_runtime_mode)`: the dummy run executes under the *same* runtime mode the graph will replay
  under, so it captures a size the first time and replays it afterwards — one report can contain a
  capture-window refusal from one step and a replay-time `synchronize()` from another.
- Inside it, `:1911-1915` refreshes full-graph parameters only when `cudagraph_runtime_mode == FULL and
  not capturing and not self.use_sparse`, where `use_sparse` is a model-config `hasattr` at `:260` and
  is true for DeepSeek-V3.2's sparse attention. The update is **deliberately skipped** on this model.
  What runs instead lives in the sparse backend and was **not** established — do not present the skip as
  the cause; keep it as the first place to look if a captured graph might hold stale metadata.
- `507015` is the device-side AICore fault the next `synchronize()` surfaces
  (`device-kernel-faults.md`); the capture-window refusal is a host-side copy or sync the graph must not
  contain. The sibling class with a `107027` / `Not allow to synchronize captured-stream` wording is a
  data-dependent op inside capture, worked through in `moe-ep-dispatch.md` §B (`torch.bincount`).

### C2. The replay sync carries no handler at all (`107020`, #13218)

A sibling of the above on a *later* tag: the **replay** barrier itself is unguarded, so any ACL failure
parked on the captured stream surfaces there as a raw `RuntimeError`/`OutOfMemoryError` naming only an
error code. Verified at `v0.23.0rc1`: `vllm_ascend/compilation/acl_graph.py:254-256`
(`need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle` → a bare
`torch.npu.current_stream().synchronize()`), while the only coded signature is the **capture** path
(`_STREAM_RESOURCE_ERROR_CODE = "207008"`, `_is_stream_resource_capture_error` `:42-52`, applied at
`:209-211`) — the replay branch has no `except`. `git grep -n '107020\|107027' HEAD --
vllm_ascend/compilation/` is empty, so nothing has landed since.

Reported shape (#13218, Qwen3-VL-8B, `FULL_AND_PIECEWISE`, `cudagraph_capture_sizes` out to 256,
`--max-model-len 65535`): `Memory_Allocation_Failure(EL0004)` / `alloc device memory failed, runtime
result = 207001` first, then a plog `event wait timeout` (`stream_id:44`, `timeout=1866s`), then
`torch.OutOfMemoryError: synchronize … error code is 107020` and `EngineDeadError`. The OOM label is
**cosmetic** — the engine is dead from the earlier allocation failure — and the 1866 s is the driver's
event-wait timeout, not a slow kernel. So the workaround targets the `207001` (lower `--max-model-len`,
trim the capture-size list, or `--enforce-eager`, which skips the replay path), never the sync line.

**Why the label says OOM: a substring test.** Verified in the torch_npu tree (`Ascend/pytorch`, `main`
`988eaa3220`): `torch_npu/csrc/core/npu/NPUException.cpp:426-436` `c10_npu::isCannOOM` returns true when
the whole message contains `EL0004` **or** `Failed to allocate`, and `NPUException.h:190-204` appends the
sticky device text `c10_npu::c10_npu_get_error_message()` to the ACL error code before choosing
`TORCH_CHECK_WITH(OutOfMemoryError, …)`; the async queue path repeats the test at `NPUQueue.cpp:605`. So a
`synchronize` failure inherits the class from the earlier allocation failure still sitting in the device
error buffer: **read it as a pointer to the root cause, not as memory exhaustion, and ignore the HBM-sizing
hints that follow.**

Triage lever: that report's body said "vllm 0.11.x" while `v0.23.0rc1` pairs with vLLM `v0.23.0`; the
maintainer asked for the real versions and unmodified worker log and said a mismatch "will not be fixed",
then the issue was stale-closed. Read the pairing (`../vllm-ascend/version-pairing.md`) before accepting a
reported upstream version, and ship the mechanism labelled `reported`/`hypothesis` when the log the
report gives is fragments rather than a worker log.

Reading rules:

1. **A synchronous copy inside the window is host-side, like the Triton ceiling**: it names no kernel.
   Find the copy or the `.cpu()`/`.item()`/host sync that runs while the graph is capturing, at the
   reported ref, and the capture mode in force (`torch.npu.graph(...)`, `capture_error_mode`) — do not
   route it to `device-kernel-faults.md`.
2. **Split the two messages, then answer in halves.** `507015` is CANN's kernel fault
   (`device-kernel-faults.md`); the capture-window refusal is a host sync. Do not merge them into "the
   graph is corrupted".
3. **The official config can be the trigger.** In the DSv3.2 tutorial at `v0.14.0rc1` the **prefill**
   nodes run `--enforce-eager` (`:550`, `:626`) while the **decode** nodes run
   `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,9,12]}'`
   (`:699`, `:776`; deployment line `:886` = P DP2+TP16, D DP8+TP4) — `FULL_DECODE_ONLY` alone also at
   `:159`, `:219`, `:268`. A decode-node crash inside graph capture/replay is therefore the *documented*
   configuration.

- **Workaround (proposed).** Run the decode nodes eager — `--enforce-eager`, or
  `--compilation-config '{"cudagraph_mode": "NONE"}'` — to turn capture off on the failing role only;
  cost: decode throughput and the compile-time fusion the graph gives, and it cannot remove a device
  `507015`. Second lever, if the crash depends on which sizes were captured: use the tutorial's
  `cudagraph_capture_sizes` list instead of the default sweep.
- **Status for #6529.** `reported`: maintainer `LoganJane` (2026-06-01) "resolved in the latest version.
  Please use the latest release or rc version" — no code change named; the thread was then stale-closed.
  The class above is the mechanism, not a confirmed cause for that deployment. The only later ACL-graph
  guidance commit found, `a2e91398a` (#10555, "Improve ACL graph stream resource error guidance"), only
  adds error text for stream-resource failures and an HDK 25.5.1+ hint for the `alloc sq cq fail`
  signature — it does not change this mechanism, so do not cite it as the fix.
- **Screenshot-only report of the same class (#6257, gemma-3-1b-it at `v0.9.2rc1`).** The body carries
  three images and no text, so the error is only visible in a screenshot — but a maintainer's
  verification comment (2026-03-12) types the failing strings while testing the same script on
  `v0.15.0rc1`: "未出现原 issue 中的 `AclrtSynchronizeStreamWithTimeout` 或 `captured-stream` 报错". That is
  `AclrtSynchronize*WithTimeout` beside a word the reporter's own log called "captured-stream", i.e. §C,
  and the same comment is the status: offline `LLM(...)` inference completed on `v0.15.0rc1`
  (`vllm 0.16`, `torch/torch_npu 2.9.0`). Route from a *typed* paraphrase in the thread tail rather than
  guessing at the image (`../workflow/evidence-sources.md`), and answer with the class + `--enforce-eager`
  / `{"cudagraph_mode": "NONE"}` lever + "no longer reproduces on `v0.15.0rc1`". Do not spend the batch
  hunting a gemma3-specific line: `git grep -n -i gemma <ref> -- vllm_ascend` is empty at `v0.9.2rc1`
  (`vllm_ascend/models/` holds deepseek/qwen files only), so the owning path is the capture window, not a
  model patch.
- **Falsifier.** Re-run the same 1P1D config with capture off on the decode nodes: if the pair
  disappears, capture was the selector and this page owns it; if `507015` persists, it is a device
  kernel fault and the answer needs the DFX/plog line (`device-kernel-faults.md`), not a flag.

## The same class on a later release

The registration surface moves, so grep both places at the reported ref before claiming a gap:

```bash
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'register_meta_if_necessary' <ref>
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'TORCH_LIBRARY_IMPL' <ref> -- csrc
```

At `main` the namespace is `_C_ascend` (not `_C`): `csrc/torch_binding_meta.cpp` carries the Meta impls
for the kernels (`rotary_embedding`, `npu_gemma_rms_norm`, `dequant_situ_quant`, the gated-delta-rule /
KDA kernels, …) and `vllm_ascend/meta_registration.py` only the Python LoRA ops (`bgmv_expand`,
`sgmv_expand`), imported from `vllm_ascend/utils.py:571/:581`. A newly added custom op missing from both
files reproduces this class on any later tag; the fix is a Meta impl, not a retry.

## Checklist before reusing either half

```bash
# A: could this ref trace custom ops at all?
git -C /workspace/repos/vllm-project/vllm-ascend ls-tree --name-only <ref> -- csrc | grep -c meta
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'torch.ops._C\|enable_custom_op' <ref> -- vllm_ascend/ops
# B: what is the deepest op frame in the report, and does the ref even have a 310P tree?
git -C /workspace/repos/vllm-project/vllm-ascend ls-tree -r --name-only <ref> | grep -c '^vllm_ascend/_310p/'
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:vllm_ascend/ops/rotary_embedding.py | grep -n 'atb\|_npu_rotary\|triton'
git -C /workspace/repos/vllm-project/vllm-ascend show <ref>:Dockerfile.310p | grep -n '^FROM'
# B2: does the tag have the 310P worker, and does it skip the warm-up?
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'def _warm_up_atb' <ref> -- vllm_ascend | head
```

The dispatch bodies move (rope became a free function with per-feature branches by `v0.18.0`, and later
releases replace `_npu_rotary_embedding` with the public `npu_mrope`/`npu_rotary_mul` APIs), so re-grep
the call site at the ref instead of quoting the line numbers above.

## Triage order

1. Read the leaf line: an `attempted to run this operator with Meta` wording means tracing and shape A;
   a `capture mode does not support` / `107041` / `EE1016` / `Not allow to synchronize captured-stream`
   wording means a sync inside the capture window and a capture-path answer (§C); a `runtime error
   <code>` / DFX kernel name means `device-kernel-faults.md`; a `not found in this op store` wording
   means shape B and an image/CANN answer; a `CreateOperation failed` worded op (`LinearOperation`,
   `_npu_matmul_add_fp32`, `_warm_up_atb` under `compile_or_warm_up_model`) is §B2 — a worker-level warm-up
   the 310P tree skips from `v0.14.0rc1`, not a model-layer op; a `marked as skipped` / `posix.putenv` wording is not a
   capture bug at all — it is a host write inside the traced Python, `dynamo-graph-break.md`. **No leaf
   line at all**, only `EngineCore encountered an
   issue` plus a `cudagraph_mode` in the command, means a mode-support gap, not a capture bug: check
   the model family's support at the tag before reading any capture code —
   `aclgraph-mode-and-capture.md` §A.
2. Find the predicate that selects the op at the ref, and the compilation level the report ran
   (`--enforce-eager`? `--compilation-config`? torchair on? — the hook at `platform.py` decides).
3. Check the Meta surface at the ref (both greps above); if the op is absent, the class is confirmed and
   the fix commit is known.
4. State the workaround the predicate allows (dtype/style/head-size change, or eager/`-O0`, or a
   vendor-matched image) with its cost, and the first tag that carries the fix.

## Recheck conditions

The custom-op namespace (`_C` → `_C_ascend`), the `custom_ops = ["all"]` default, the predicate bodies
and the meta-registration files change between releases; on 310P the V1 custom-op default is skipped
altogether. Re-read `vllm_ascend/platform.py` `check_and_update_config`, the op's predicate and the two
registration files at the reported tag. Nothing here was reproduced on hardware.
