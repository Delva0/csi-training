# Memory profiling dies with `EOFError: Ran out of input` — the V0 TP logits gather emulation

Scope: engine start, **memory-profiling window**, every rank but the driver dies in
the same second with `EOFError: Ran out of input`, and the traceback's bottom frame is
`torch_npu/distributed/distributed_c10d.py` `_gather` → `dist.broadcast_object_list`.
TP > 1, V0 engine, dense model. The driver then dies ~2 min later on a collective the
peers can no longer join.

**Not this page when the raising line carries a device code** — `RuntimeError: ACL stream
synchronize failed, error code:507018` (or any `507xxx`) at the same
`hidden_states[logit_indices]` line, with `retCode=0x2a` / `[aicpu exception]` beside it or
`(function copy_between_host_and_device_opapi)` under the raise: that is the
memory-profiling **device fault** lane, where this frame is a sync point, not the owner —
`../hccl-and-hardware/collective-error-chain.md` §Step 1a's `opapi` bullet (recorded
#2105, cause open) and, for the code decodes, the topic entry
`../hccl-and-hardware/index.md`. This page owns the `EOFError` shape only: no device code,
`dist.broadcast_object_list` at the bottom frame. **Not this page either when the frames
are `_v1` ones** (`worker_v1.py` / `model_runner_v1.py`) and the raise is a device/HCCL
one — a V1 rerun of this same setup died at communicator creation instead
(`../hccl-and-hardware/comm-create-errors.md` §*Code 7 with the `tsd … check hdc service`
line*), so read which engine the paste came from before reading the frame.

Recorded case **#1579**: vllm-ascend `v0.9.1rc1` (`c30ddb8`), vllm `v0.9.1`
(`b6553be`), torch_npu `v7.1.RC1.alpha002` (`d33df55`) / torch 2.7.1, image
`quay.io/ascend/vllm-ascend:v0.9.1rc1-310p`, one 310P node × 8, Qwen3-32B dense,
`vllm serve … --tensor-parallel-size 8 --enforce-eager --dtype float16`, no
`--max-model-len` (the log's own warning shows the default 40960). The paste opens with
a passing `hccl_test` `all_gather_test -b 8K -e 64M`.

## 1. Name the engine from its own lines — the V0 notice is an opt-in, not a verdict

The paste carries `INFO … [arg_utils.py:1653] npu is experimental on VLLM_USE_V1=1.
Falling back to V0 Engine.` and then `[llm_engine.py:230] Initializing a V0 LLM engine`.
That message is upstream vLLM, not vllm-ascend: `_warn_or_fallback()` prints it from the
`else` of `if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1`
(`vllm/engine/arg_utils.py:1645-1657`), reached from `_is_v1_supported_oracle`'s
`v0_hardware` branch — "not CUDA, not TPU, not x86 CPU" (`:1444-1450`) — with
`current_platform.device_name` as the feature name. The line number matches the pin
exactly, so the pin's `arg_utils.py` is the reporter's file.

What it does **not** say: that V1 cannot run on this SoC. At that ref the decision is
`try_v1 = envs.VLLM_USE_V1 or not envs.is_set("VLLM_USE_V1")` (`:1026`) and the default
value is `1` (`vllm/envs.py:616-617`), so an **unset** `VLLM_USE_V1` "tries V1" and the
non-CUDA hardware row then declines it. Exporting `VLLM_USE_V1=1` takes the
`logger.warning("Detected VLLM_USE_V1=1 with %s …")` branch and the oracle returns True.
Downstream the plugin accepts it: `AscendPlatform.supports_v1()` returns `True`
unconditionally (`vllm_ascend/platform.py:254-258` at v0.9.1rc1), and
`check_and_update_config` binds `vllm_ascend.worker.worker_v1.NPUWorker`
(`platform.py:187`) — on 310P it only skips `compilation_config.custom_ops = ["all"]`
(`:210-211`, the `if not is_310p()` guard). So **the engine vintage is a live lever here**, and any failure whose
frames are `worker/worker.py` + `worker/model_runner.py` (no `_v1`) is a V0-only code
path by construction.

## 2. The failing step is the emulation's size handshake, not the collective

HCCL has no `gather`; torch_npu emulates it (`Ascend/pytorch`
`torch_npu/distributed/distributed_c10d.py:62-142`, docstring: "NPU doesn't support
gather currently, replaced with all_gather"). With an explicit group it:

1. builds `recv_size_list` — `[tensor.size() for tensor in gather_list]` on the
   destination, `[None] * group_size` everywhere else (`:98-99`);
2. `dist.broadcast_object_list(recv_size_list, dst, group)` (`:126` — the frame in
   every worker's traceback);
3. allocates `gather_list = [torch.empty(tensor_size, …).npu() for tensor_size in
   recv_size_list]` (`:127-128`) and runs one `allgather` (`:132`).

`EOFError: Ran out of input` is raised by torch's `_tensor_to_object`
(`torch/distributed/distributed_c10d.py:2649`, `_unpickler(io.BytesIO(buf)).load()`,
frame printed above it at `:3154`): the receiving rank unpickled an **empty** buffer, so
the per-object size it read was 0 — i.e. step 2 delivered zeros to the non-destination
ranks while the destination, which is the broadcast source, kept its own list and never
noticed. A `None`/short `recv_size_list` would instead raise `TypeError` at `:127`, and a
plain allgather failure would raise a device error — so this shape localizes to the
**object broadcast inside the gather emulation**, not to HCCL transport and not to the
logits tensor's size.

What that does not settle, and what to stop looking for: *why* the receiver sees zeros
(object-collective device resolved by `_get_object_coll_device`, an HCCL broadcast of a
small CPU/host tensor, 310P behaviour). torch's own `distributed_c10d.py` is **not** in
this case's snapshot: `Ascend/pytorch` `v7.1.RC1.alpha002-pytorch2.7.1` is a
`content_export` (1 190 of 1 240 regular files, `complete_regular_files: false`) that
carries `torch_npu/` and no `torch/` directory at all, and upstream `pytorch/pytorch`
`v2.7.1` is truncated to 1 781 of 16 542 files with no `torch/distributed/`. So the
mechanism cannot be closed locally — say so and ship the ladder. The recorded run spent
its whole 16 384-token budget re-deriving torch's pickle/zero-fill semantics here and
delivered **no answer at all**.

Recheck condition for a wider deployment: `my_rank = get_rank()` in that `_gather` is
the **global** rank while `dst` is a **group** rank (`:97-99`). With DP > 1, PP > 1 or a
second engine on the same cluster the TP group's ranks are no longer 0..7, so the rank
that fills its list with real sizes need not be the rank the broadcast treats as source
— check `get_rank()` vs `get_group_rank(tp_group, 0)` before reusing this page for such
a launch.

## 3. The driver's later error is the victim line

In #1579 ranks 1-7 die at `05:46:34`; the EngineCore process (rank 0) dies at
`05:48:33` — 119 s later — with `ERR00100 … The process exits for this inner error, and
the current working operator name is HcclAllgather`, `EI0006 … Getting socket times out.
Reason: Remote Rank did not send the data in time`, `[Create][DestLink]Create Dest
error! … dst_rank[7]/dst_rank[1]`, all under
`collective_rpc("determine_num_available_blocks")`. That is the ~120 s HCCL timeout of
the collective rank 0 was left waiting in once its peers had exited: **a cascade, not a
fabric fault.** Do not route it to
`../hccl-and-hardware/transport-destlink-ipc-import.md`, `device-health-checks.md` or
`soc-platform-paths/kernel-package-vs-card.md`; the peers' `EOFError` is 2 minutes older
in the same file. `ASCEND_LAUNCH_BLOCKING=1` is offered by the runtime text and buys
nothing here — the Python frames are already exact.

Two more lines in that paste are noise: `Warning: [Check][offset] Check input
storage_offset[%ld] = 0 failed, result is untrustworthy151935` (`151935 = vocab - 1` for
Qwen3's 151936 — an aclnn warning about the logits slice) and the trailing
`disabled custom all-reduce kernel`. Neither is the fault.

## 4. What their `hccl_test` proves

`all_gather_test -b 8K -e 64M -d fp16` exercises `HcclAllgather` only. The primitive
that fails here is `dist.broadcast_object_list` (a `broadcast` of a tiny host/device
tensor plus local pickling) and the *pairing* of it with the follow-up allgather — a
passing all-gather sweep says nothing about either. Say that explicitly, because "HCCL
works, I tested it" is the reporter's next message.

## 5. Ladder

1. `export VLLM_USE_V1=1` and repeat the identical command. Cheapest, and §1 shows the
   whole failing stack is V0-only code. A third party on the reported issue said the same
   setup started for them with it (peer report, their environment unverified); the
   reporter's own later update says it was **not** resolved for them, so treat it as an
   ablation, not a fix. **The rerun's frame is itself the result.** A second 310P ×8 paste
   in that thread (2025-07-17) carries `model_runner_v1.py` frames and no `EOFError` at
   all: the first TP all_reduce of the profiling dummy run died at *communicator creation*
   (`vocab_parallel_embedding.py:63` → `dist.all_reduce` →
   `hcclCommInitRootInfoConfig … error code is 7`, `… check hdc service`,
   `driver and firmware packages do not match` at startup). So a V1 run that dies at
   **another** frame closes this page's question (the `gather` emulation is V0-only) and
   opens `../hccl-and-hardware/comm-create-errors.md` §*Code 7 with the `tsd … check hdc
   service` line* — do not read the second paste as the same failure.
2. `--tensor-parallel-size 2` (or 4): does the handshake fail at world_size 8 only? One
   run bounds "world-size dependent" vs "every NPU gather".
3. Reproduce the primitive alone, same host, same torch_npu, no vLLM in the way:
   `torchrun --nproc_per_node=2 ../../scripts/npu-gather-probe.py` and again with
   `--group` (the `:126` explicit-group branch the traceback names), then
   `--nproc_per_node=8`. Same `EOFError` → the image's torch_npu/CANN object collective
   owns it, file there with the `_gather` line number; clean → the logits tensor's
   shape/dtype is back in play and the `storage_offset` warning deserves the next look.
4. `--max-model-len 4096` only *after* (1)-(3): the default 40960 the log warns about
   changes the dummy batch, not the sizes payload, so it is not a mechanism candidate.
5. If V1 also dies at the same `_gather`/`broadcast_object_list` frame, quote the two
   ablated runs and hand it to torch_npu with the `_gather` line number and the `hccl_test`
   result — no vllm-ascend flag remains. If V1 dies *elsewhere*, say what moved and put the
   thread on that page instead: on 310P the recorded thread reached the first collective
   and never got past it, and the reporter's working stack there was MindIE on a 310P T18
   image (step 6).

## 6. What to ask the reporter for

The exact `torch`/`torch_npu` pair inside the image (`pip show torch torch_npu`, or
`python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"`),
the `VLLM_USE_V1=1` rerun's first 60 lines, and — if they moved to another stack instead
— which one. When the rerun's first error is a `hcclCommInitRootInfoConfig … error code is
7`, add the host's `cat /usr/local/Ascend/driver/version.info` and the `npu-smi info`
version block: that frame is a pairing question, not an HCCL knob. #1579's reporter ended
on MindIE on a 310P T18 image (「这个镜像可能要联系一下华为官方」), so "V0+V1 both fail on
310P" stayed open for vllm-ascend; do not report this shape as solved, and do not offer the
MindIE image as a vllm-ascend fix.
