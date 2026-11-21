# Start hangs in silence on a multi-process launch (`external_launcher` / torchrun SPMD, TP>1)

Verified at vllm-ascend **v0.8.5rc1** (`ec27af3`) with vllm **v0.8.5.post1**
(`3015d563`); recorded for **#1496** (Qwen2.5-Math-1.5B, `tensor_parallel_size=4`,
`distributed_executor_backend="external_launcher"`, torch 2.5.1 / torch_npu 2.5.1 /
CANN 8.1.rc1), reproduced by both an `LLM()` example and an `AsyncLLM` script.
Re-grep the line numbers per release; the ordering below does not move.

**Sibling shape, different backend:** the same silence after the worker-constructor
warning with `(RayWorkerWrapper pid=…)` prefixes and no `vllm serve` line is a
*programmatic* Ray start (`AsyncLLM.from_engine_args` in a trainer), where the rank
arity is Ray's business and the waiting frame is upstream's startup poll —
`ray-programmatic-init-hang.md`. Read that page instead when
`distributed_executor_backend="ray"`; the `RANK`/`WORLD_SIZE` checks here do not apply.

## The shape

The title says "process deadlock during driver initialization … works when TP is 1,
deadlock when TP > 1"; the paste ends **after** the worker's own device line with no
traceback, no `Starting to load model`, and nothing after it. Both entry points hang,
which retires every model/accuracy lane at once.

## Three facts decide the lane, one call each — do these before any HCCL theory

1. **`external_launcher` means one process per rank, launched by torchrun.**
   `vllm/executor/uniproc_executor.py:68-83` (docstring: one worker per executor,
   torchrun-compatible launcher, `examples/offline_inference/torchrun_example.py`),
   `:104-111` (`RANK`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` required,
   `distributed_init_method="env://"`, `rank = int(os.environ["RANK"])`).
   Consequence worth saying out loud: a bare `python script.py` with TP=4 **cannot**
   hang there — it raises `KeyError: 'RANK'` at `:110` before a worker exists. The
   paste does contain worker-side lines (the
   `Methods … not implemented in <… NPUWorker object>` warning from the wrapper's
   `run_method`, `vllm/utils.py:2522` in the paste), so the env *was* exported and
   "you forgot torchrun" is refuted. What remains is arity or rendezvous.
2. **The world size is the requested TP, not the launcher's process count.**
   `vllm_ascend/worker/worker.py:208-222` (`torch.device(f"npu:{local_rank}")` →
   `NPUPlatform.set_device`) calls `:526-541` `_init_worker_distributed_environment(...,
   backend="hccl")`, which passes `parallel_config.world_size` into upstream
   `vllm/distributed/parallel_state.py:886-895`:
   `if not torch.distributed.is_initialized(): init_process_group(backend="hccl",
   init_method="env://", world_size=parallel_config.world_size, rank=rank)`. The
   `world_size` argument is **explicit**, so `WORLD_SIZE` from torchrun only enters as
   `rank`. Launch `--nproc_per_node` < TP (a wrapper script, a stale torchrun, TP edited
   in code, four processes under two different `torchrun`s) and the TCP-store rendezvous
   waits for the missing ranks forever. This is why the failure is a hang and not the
   `reshape ... invalid for input of size W` error of `arg-validation-errors.md`: that
   raise is inside `initialize_model_parallel`, which is only reached after the group
   exists. `:906-909` then builds the world group, whose `GroupCoordinator` also creates
   a **gloo** `cpu_group` (`parallel_state.py:225`), i.e. a second rendezvous.
   Check: `ps -ef | grep -c python` on the node, and
   `tr '\0' '\n' < /proc/<pid>/environ | grep -E 'RANK|LOCAL_RANK|WORLD_SIZE|MASTER_ADDR|MASTER_PORT'`
   per process — all processes must agree on `MASTER_ADDR`/`MASTER_PORT`.
3. **The mode is already printed.** `vllm/config.py:1725-1729` sets
   `VLLM_ENABLE_V1_MULTIPROCESSING=0` and logs `Disabling V1 multiprocessing for
   external launcher.` (that line is in the paste), and
   `config.py:923-928` asserts `seed is not None` for this backend. So the launcher
   path *is* the one that ran; nothing else needs establishing.

## Ranked candidates, each with its one check

1. **Process count < TP, or a split store.** First, because TP=1 works and the log
   carries no rank-side error at all. Check: the launch command plus the per-process
   env dump above; fix by launching exactly `torchrun --nproc_per_node=<TP> …`.
2. **Rank → device collision.** `init_device` builds `npu:{local_rank}` and sets it;
   if every process sees the same device set (or `ASCEND_RT_VISIBLE_DEVICES` pins each
   process to one device while `LOCAL_RANK` still runs 0..3), HCCL's intra-node setup
   blocks. Check: `npu-smi info` **during** the hang (one process per device?) and print
   `rank, local_rank, torch.npu.current_device()` at the top of the script.
   The `npu:0` in the paste is rank 0's local device; it is **not** evidence that all
   ranks chose 0 — a one-process paste says nothing about the other ranks.
3. **HCCL transport/interfaces** (multi-NIC host, RoCE choice, unreachable
   `MASTER_ADDR`). Bound the wait before theorising: `HCCL_CONNECT_TIMEOUT=120`,
   `ASCEND_LAUNCH_BLOCKING=1`, and `py-spy dump --pid <pid>` on each rank to see whether
   they sit inside `init_process_group` at all. Do not rank it above 1-2; those are
   cheaper and match "TP=1 fine".

## The support answer for this ref

At `ec27af3`, `rg -n -i 'external_launcher' vllm-ascend/` returns **nothing** and
`examples/` holds only `dp_offline/`, `offline_distributed_inference_npu.py`,
`offline_inference_npu{,_v1}.py`, `offline_inference_audio_language.py`, so there is no
vllm-ascend recipe or e2e test for this backend at the reported version: what runs is
the upstream mechanism. Say that (it is the answer to "is external_launcher supported
here?"), point at upstream's `torchrun_example.py`, and do not synthesize a launcher
from the constraints. DP's gloo→hccl stateless init
(`vllm_ascend/patch/platform/patch_common/patch_distributed.py:104+`) is a **different**
path and does not apply here.

## Gap to report, not chase

`>>>>>>>>>>>>>>>device<<<<<<<<<<<<<<<<<<<<<<< npu:0` exists in **neither** pinned tree
(`rg` at `ec27af3` and at v0.9.1rc1 `c30ddb8`, both 0 hits), so the running module is
not the tag as exported — a local patch or another build. One call locates it on their
box (`python -c "import vllm_ascend; print(vllm_ascend.__file__)"`, then grep that
tree) and then the paste's line numbers are unciteable: report the divergence and ask
for `pip show vllm-ascend`. Two recorded turns went into "which repo prints this
string" before it was written down as a gap.

## Delivery

This shape is **answered, not investigated**: a paste that stops in silence on a TP>1
launch is already classified, and the deliverable is the ranked launch/rendezvous list
plus three artefacts (launch command, per-rank logs, rank→device mapping) — not a
mechanism. Recorded #1496 produced no message text at all: 12 assistant turns, ~49 K
characters of reasoning across four competing mechanisms (HCCL env, the gloo
`cpu_group`, the `all_reduce` in `determine_num_available_blocks`, "all ranks picked
npu:0"), and the 16 384-token cap hit with zero characters delivered. None of those
four was needed to write the answer, and the `npu:0` theory was decidable from "the
paste is one process".
