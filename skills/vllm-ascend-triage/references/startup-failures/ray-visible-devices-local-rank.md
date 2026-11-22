# `107001` / `Set visible device failed, invalid device=N, input visible devices:1` in a Ray worker: the raylet narrowed `ASCEND_RT_VISIBLE_DEVICES`

## Scope

A `--distributed-executor-backend ray` serve whose engine never becomes ready, every
line carrying a `(RayWorkerWrapper pid=…)` prefix, no `Starting to load model`, and one
`Error executing method 'init_device'. This might cause deadlock in distributed
execution.` per failing actor. Deepest frames (V0 engine path):

```text
vllm/worker/worker_base.py:620     execute_method → run_method
vllm/worker/worker_base.py:604     self.worker.init_device()
vllm_ascend/worker/worker.py:185   NPUPlatform.set_device(self.device)
vllm_ascend/platform.py:99         torch.npu.set_device(device)
torch_npu/npu/utils.py:83          torch_npu._C._npu_setDevice(device_id)
RuntimeError: … c10_npu::SetDevice(device_id_), error code is 107001
ERR00100 PTA call acl api failed / [Error]: Invalid device ID. Check whether the device ID is valid.
EE1001: … The argument is invalid.Reason: Set visible device failed, invalid device=1, input visible devices:1
rtSetDevice execute failed, reason=[device id error] … open device 1 failed, runtime result = 107001
New device failed … open device 0 failed, runtime result = 507033 … ctx is NULL!
```

Recorded **#727**: vllm-ascend `v0.8.4rc2` (`1fce70a2fb26`) with vllm `v0.8.4`
(`dc1b4a6f1300`), two nodes × 8 NPU, Qwen3-235B-A22B,
`vllm serve … -tp 16 --distributed-executor-backend ray`, V0 engine (`npu is
experimental on VLLM_USE_V1=1. Falling back to V0 Engine.`). The reporter's own framing
is "单节点可以用 vllm 拉起，双节点报错" — which is the fact that decides the ranking
below, not a multi-node defect by itself.

**The line that decides this page is not in the traceback.** In the same paste, above it:

```text
(RayWorkerWrapper pid=26536, ip=…) WARNING [utils.py:634] Overwriting environment
  variable ASCEND_RT_VISIBLE_DEVICES from '0' to '0,1,2,3,4,5,6,7'
(RayWorkerWrapper pid=204576) … Overwriting environment variable
  ASCEND_RT_VISIBLE_DEVICES from '7' to '0,1,2,3,4,5,6,7'   [repeated 14x across cluster]
```

Read the **`from`** half: each actor was started with a **single** card id. That is Ray's
accelerator manager assigning one device per actor; vLLM's write comes afterwards.

## The arithmetic: `local_rank` indexes a list that Ray already narrowed

1. `ray_distributed_executor.py:331-333` (vllm v0.8.4) builds the per-worker env from
   `node_gpus[node_id]` — the **union** of accelerator ids granted to that node's workers,
   i.e. `0,1,2,3,4,5,6,7` — and `:383` computes `local_rank =
   node_workers[node_id].index(rank)`, the node-local index `0..7`.
2. `worker.py:184-185` then asks for `torch.device(f"npu:{local_rank}")` and calls
   `set_device`. The index is **logical, relative to the process's visible list**.
3. If that process's device set is still the one-entry set Ray gave it, only index `0`
   exists: `local_rank=1` is out of range → `invalid device=1, input visible devices:1`,
   and `107001 = ACL_ERROR_RT_INVALID_DEVICEID` (`rt_error_codes.h:22`). The trailing
   `open device 0 failed … 507033` / `ctx is NULL` lines are the cascade of the failed
   open, not a second fault.
4. The `Overwriting` line is vLLM's own helper (`vllm/utils.py:631-639`, `os.environ[k] =
   v` plus a warning whenever the old value differs), executed **inside the worker**
   (`worker_base.py:533-541`). So its `from` value is the actor's pre-existing
   environment — the raylet's, never vLLM's.

Rewriting `os.environ` after the fact does not re-resolve the set: torch_npu/CANN bind
the visible set at the first device init, and in the paste's own timeline each actor
imports the plugin (`Platform plugin ascend is activated`) before vLLM's write at
`13:51:24`. Treat "the write should have fixed it" as a hypothesis with one probe (check
4 below), not as a refutation of the narrowing.

## Why `platform.py:52` does not cover it

`os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"`
(`vllm_ascend/platform.py:52`) runs only in the process that imports `vllm_ascend` — the
driver/engine — and Ray actors inherit the **raylet's** environment. vLLM cannot carry it
either: the carry list is filtered to names its own `envs.environment_variables` dict
knows (`vllm/envs.py:137` → `ray_distributed_executor.py:337-341`, applied `:346-350`),
and no `RAY_EXPERIMENTAL_NOSET_*` name is in that dict at v0.8.4. The paste proves it in
its own words:

```text
INFO [ray_distributed_executor.py:354] Copying the following environment variables to workers:
  ['LD_LIBRARY_PATH', 'VLLM_WORKER_MULTIPROC_METHOD', 'VLLM_USE_V1']
```

`WORKER_SPECIFIC_ENV_VARS` (`:56-58`) is not what excluded it — the name was never a
candidate.

## The fix, from the pin's own tutorial

`docs/source/tutorials/multi_node.md` states it at `:92` ("For Ray versions above 2.1,
also set the `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` variable to avoid device
recognition issues") and ships both node blocks (`:107-110`, `:122-125`). On **each**
node, in the shell that starts that node's raylet:

```bash
export HCCL_IF_IP=…; export GLOO_SOCKET_IFNAME=…; export TP_SOCKET_IFNAME=…
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --head --num-gpus=8                 # worker nodes: --address=<head>:<port> --node-ip-address=<local_ip>
```

The page's own notes add the ordering rule: the variables must be set **before** the
cluster starts, and updating them requires restarting it (`ray stop` then start on both
nodes). Verification is the same log line: the per-worker `Overwriting … from
'0,1,2,3,4,5,6,7' to '0,1,2,3,4,5,6,7'` — or no such line at all.

## Checks, cheapest first

1. **Count the failing ranks in the paste.** One `ERROR … init_device` block per node
   (the actors whose `local_rank` ≥ 1) is this page: with a one-entry visible list only
   index 0 can open. **Every** rank failing, or a `--device /dev/davinciN` docker line
   beside the exported ids, is the container/device-set lane instead —
   `device-id-invalid-at-worker-init.md`.
2. Grep the paste for the two tokens that place it: `Overwriting environment variable
   ASCEND_RT_VISIBLE_DEVICES from '` and `(RayWorkerWrapper pid=`. Neither present, and
   the `107001` came from `check_ascend_device_type()` instead of `init_device` → the
   row in `device-id-invalid-at-worker-init.md`.
3. Ask for the two `ray start` lines (both nodes) and `ray status`. The value each
   raylet was started with is the only thing that decides whether the narrowing is
   still there.
4. Inside a failing actor, one line separates the two readings: print
   `os.environ["ASCEND_RT_VISIBLE_DEVICES"]` and `torch_npu.npu.device_count()` — a count
   of 1 after vLLM's overwrite means the set was bound before the write.
5. Only if the narrowing is gone and `107001` persists, go to the container page's
   `ls /dev/davinci*` / `npu-smi info` list.

## Not the causes, and say so

- **The model, the dtype, the quant config, `--block-size`/`--swap-space`/`--gpu-memory-utilization`:**
  nothing above `init_device` in the worker ran. There is no `Starting to load model`.
- **The wheel's SoC stamp / a 310P-vs-A2 package:** that is the `check_ascend_device_type()`
  frame, one call earlier in `Worker.__init__`, and it is absent from this paste.
- **A missing card or a busy card:** `npu-smi` shows the node's devices and Ray granted
  all eight (`from '0'` … `from '7'` across the 15 repeats).
- **Cross-node TP=16.** The `ray_utils.py:199` warning (`tensor_parallel_size=16 is bigger
  than a reserved number of NPUs (8 NPUs)`) is informational, and the pin's tutorial
  gives `--tensor-parallel-size 16` as a supported option for 16 NPUs across two nodes
  (`multi_node.md:157-165`, the alternative to TP=8 + PP=2 at `:140-155`). Do not tell a
  reporter their topology is wrong on the strength of that warning (the recorded #727 run
  did).
- **A single-node ray run working:** the narrowing is per raylet, so one node behaves the
  same. Ask whether the working single-node run used the default `mp` executor
  (`--distributed-executor-backend` absent) — that, not the node count, is the control.

## Limits

- Ray's `RAY_EXPERIMENTAL_NOSET_*` branch lives in `ray/_private/accelerators/`, **absent
  from this snapshot's ray export** (only the env-var names survive, in
  `ray/_private/ray_constants.py:428-438`, where `NPU_RT_VISIBLE_DEVICES_ENV_VAR =
  "ASCEND_RT_VISIBLE_DEVICES"`). So the suppression itself is not readable here; the
  *narrowing* is proven by the paste's own `from '<one id>'`, and the fix is the pinned
  tutorial's, not a source reading.
- The "only `local_rank` ≥ 1 fails" arithmetic is inferred from the two error codes; the
  failing-rank count in check 1 confirms or refutes it. No hardware ran here.
- Recorded run #727 shipped **no answer text at all**: 16 tool results over 32 messages,
  cut at the 16 384-token output cap while it walked vLLM's `local_rank` plumbing and
  torch_npu's visible-device parsing. The tutorial line above was already read on turn
  12, and the reply owed at that point was one paragraph: "your raylets were started
  without `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`, so Ray hands each actor a
  single card while vLLM indexes the node's full list by `local_rank`; export it on both
  nodes before `ray start`, restart the cluster, re-check the `Overwriting … from` line".
