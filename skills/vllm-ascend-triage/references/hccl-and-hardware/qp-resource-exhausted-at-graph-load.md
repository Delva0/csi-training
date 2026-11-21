# `EI0007 … Failed to allocate resource[qp] … Memory resources are exhausted` at graph load

## Scope

Weights are loaded, then the **graph capture** phase dies, usually on a multi-node
DP/TP MoE serve with graph mode on (`torchair_graph_config`, the era before
`cudagraph_mode`):

```text
RuntimeError: EI0007: [PID: 2322] 2025-09-04-09:44:05.909.517 Failed to allocate resource[qp] with info
  [aicpu rank:7,localUserrank:7,localIpAddr:<addr>,deviceLogicId:3,send_cq_depth:32768]. Reason: Memory resources are exhausted.
        Possible Cause: Failed to allocate memory or the Notify register due to resource insufficiency.
        Transport init error. Reason: [Create][DestLink]Create Dest error! creakLink para:rank[7]-localUserrank[7]-localIpAddr[<addr>], dst_rank[j]-remoteUserrank[j]-remote_ip_addr[<addr>]
        Assert ((ArgsFormatUtils::GetHcomHiddenInputs(op_desc_, *davinci_model_, context_addrs)) == ge::SUCCESS) failed[FUNC:SetNotifyHandleByEngine][FILE:notify_wait_task_info.cc][LINE:27]
        Failed to init task index 207, related node op_index_-1_node[FUNC:InitTaskInfoV2][FILE:model_args_manager.cc][LINE:626]
        Load graph failed, error code:1343225857, session_id:1, graph_id:0, isEnableSliceSchedule:0.[FUNC:LoadGraph][FILE:ge_api.cc][LINE:1034]
[ERROR] 2025-09-04-09:44:06 (PID:2322, Device:3, RankID:7) ERR03005 GRAPH internal error
```

reached from `worker_v1.py` `compile_or_warm_up_model` → `model_runner_v1.py`
`capture_model` → `torchair/torchair_model_runner.py` `_capture_model` →
`_compile_torchair_graph` → `_dummy_run` → `compiled_model(...)`.

Not this lane:

- the same `[Create][DestLink]Create Dest error!` sentence **with**
  `halShmemOpenHandle … drvRetCode=17` / `rtsIpcMemImportByKey … 507899` and a
  *subset* of failing edges → `transport-destlink-ipc-import.md` (HCCL could not
  build one edge; host driver/CANN pairing or device passthrough);
- app-allocator text (`Memory_Allocation_Failure(EL0004)`, `alloc device memory
  failed … 207001`, `NPU out of memory`) → `../runtime-memory/index.md`. Same
  budget question, different reporting layer; the ladder below is still the
  budget one.

## Read it in this order

1. **The first line, not the DestLink block.** `EI0007` +
   `Failed to allocate resource[qp]` + `Reason: Memory resources are exhausted` is
   a **refused allocation of communication resources (QP / send-CQ / notify
   register) for the local device** — nothing launched and no edge was attempted
   far enough to fail on its own.
2. **A DestLink line for *every* peer is the signature of that refusal.** In
   #2754 rank 7 printed all 15 `dst_rank` lines (same- and cross-node) within the
   same millisecond, and no `halShmemOpenHandle` / `507899` line appeared. That is
   a wholesale abort of the local transport init, not an edge-set finding:
   `scripts/hccl-link-pair-audit.sh` on this paste prints "all edges", which
   refutes "one bad pair" by construction. A subset of edges is the *other* page.
3. **Fields.** `aicpu rank:N` / `localUserrank:N` name the process that asked,
   `deviceLogicId:n` its local NPU (with DP4×TP4 that is a per-engine 0-3 slice,
   not a physical card id), and `send_cq_depth:32768` is the **depth the refused
   request asked for** — a request field, not a knob the reporter set. Do not go
   looking for an `HCCL_*CQ_DEPTH` variable: none appears in the trees exported
   for these cases and the HCCL env parser (`cann/hcomm
   src/platform/common/externalinput.cc`) is `unavailable` at the pin.
4. **The GE block is the reporting parent.** `GetHcomHiddenInputs` /
   `Failed to init task index N` / `Load graph failed, error code:1343225857` /
   `ERR03005 GRAPH internal error` say the graph loader could not bind one hcom
   task because its hidden (notify/QP) inputs were never allocated. The decimal
   code is not a diagnosis, and this is not a GE build or torchair bug.
5. **Class: resource budget (OOM-class), arriving as HCCL text.** On #2754 the
   maintainer read this exact paste as OOM before any source discussion and asked
   for the card capacity first; the reporter answered 64 GB × 8, i.e. capacity
   matched and the *budget configuration* was the remaining axis.

## Why graph load is where it lands

In V1 the KV cache is allocated **before** capture, inside one executor call:
`Executor.initialize_from_config` runs `collective_rpc("initialize_from_config")`
(the worker's `initialize_kv_cache`) and then
`collective_rpc("compile_or_warm_up_model")` — `vllm/v1/executor/abstract.py:60-68`
at vllm v0.10.1.1, called from `vllm/v1/engine/core.py:211`. So at the moment the
torchair graphs ask for their QP/CQ/notify resources, the process already holds
weights + a KV cache sized by `--gpu-memory-utilization` + one `HCCL_BUFFSIZE`
reservation per comm. A weight load that "succeeded" and a capture that fails is
therefore the expected order when the budget is tight, and it is why the levers
are gmu / `HCCL_BUFFSIZE` / graph mode rather than the checkpoint.

## Checks, cheapest first

1. **Say "budget", then ask for the two numbers.** `npu-smi info` captured in the
   failing minute (all 8 device rows, `HBM-Usage` per device) plus the card model
   (`64G` vs `32G`): the recipe assumes two Atlas 800 A2 (64G×8) nodes
   (`docs/source/tutorials/multi_node.md:60`), and the tutorial's own knobs are
   `HCCL_BUFFSIZE=1024`, `--gpu-memory-utilization 0.9` (prefill) / `0.92`
   (decode) with `--max-model-len 32768` (`:111-139`). A larger
   `--max-model-len`, a bigger `HCCL_BUFFSIZE`, or another process on the same
   cards is the first thing that moves.
2. **One-flag ablation, everything else identical:** graph mode off —
   `--additional-config '… "torchair_graph_config":{"enabled":false}'` (on this
   ref `model_config.enforce_eager` skips `capture_model()` outright,
   `worker_v1.py:255-256`, so `--enforce-eager` is the same ablation). Starts ⇒
   the graph-load QP request is what broke the budget; dies at the same place or
   earlier ⇒ the base reserve is already over budget, so lower gmu and
   `HCCL_BUFFSIZE` before touching the driver.
3. **Count the graphs before blaming their number.** With `enabled: true` and no
   `graph_batch_sizes`, the ref derives `[1, max_num_seqs]` (×
   `decode_token_per_req`), pads each to a `tensor_parallel_size` multiple when EP
   is on, and drops anything above `--max-num-batched-tokens`
   (`torchair/torchair_model_runner.py:400-437`) — the tutorial command
   (DP4×TP4, `--max-num-seqs 16`) gives **two** graphs, `[4, 16]`. So "too many
   graphs" is only a lever when the report carries an explicit long
   `graph_batch_sizes` list. `Batchsize %d is compiled successfully: %d/%d`
   (`:186-188`) is printed per shape: its absence in the paste means the **first**
   (largest) shape failed.
4. **Which ranks failed.** Two ranks on different devices of the same node failing
   in the same second (as in #2754: rank 0 / `deviceLogicId:0`, rank 7 /
   `deviceLogicId:3`) is per-device local resource pressure, not one bad chip;
   per-device `asys health -d=$i` is only needed when a device fault code
   accompanies it (`device-health-checks.md`).
5. **Leftover QP holders.** A crashed earlier attempt whose processes survive
   keeps its comm resources: `npu-smi info` process table and
   `ps -ef | grep -e EngineCore -e vllm` on **both** nodes before a re-run.
6. **Driver/CANN pairing last**, with the rung-1 evidence of
   `transport-destlink-ipc-import.md`; it cannot explain a `Memory resources are
   exhausted` reason once the budget is demonstrably free.

## Fingerprint the ref from the pasted line numbers

Cheap and decisive: the pasted frames must land on plausible statements. For
vllm-ascend v0.10.1rc1 (`7e16b4a`) they match exactly —
`torchair/torchair_model_runner.py:159` `hidden_states = compiled_model(`,
`:185` `self._dummy_run(num_tokens, is_torch…compile=True)`, `:224`
`self._compile_torchair_graph(torchair_graph_batch_sizes)`,
`worker_v1.py:256` `self.model_runner.capture_model()`,
`model_runner_v1.py:2583` `self._capture_model()`, `:2133`
`_generate_dummy_run_hidden_states(`. A mismatch means the report is another
release and the batch-size/graph arithmetic above must be re-derived.

## Precedents and limits

- **#2754** (vllm-ascend v0.10.1rc1, `deepseek-v3.1-w8a8`, two Atlas 800 A2
  64G×8 nodes, tutorial DP4 / DP-local 2 / TP4 / EP, torchair + ascend scheduler
  on): the maintainer classified the `EI0007` paste as OOM and asked for the card
  capacity; the reporter confirmed 64G×8. Which adjustment actually started the
  serve was never recorded — report the ladder, not a chosen fix.
- `cann/hccl`, `cann/hcomm` and `cann/ge` are `unavailable` in that case's
  `SOURCES.json`, so the QP/CQ reservation arithmetic is not source-verifiable
  here. One `rg` in an exported tree at most, then work from `npu-smi`, the
  ablation and the plog (topic rule 4 in `index.md`).
- The paste may carry no version block and no `Starting to load model`; the frame
  fingerprint above plus the tutorial diff is the substitute.
