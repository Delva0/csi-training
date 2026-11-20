# ACL graph capture failures at startup

Symptom family: startup reaches graph capture and then fails or stalls —
`Starting to capture ACL graphs for cases: [512, 408, 304, 192, 88, 1], mode:
PIECEWISE`, `Capturing ACL graphs (mixed prefill-decode, PIECEWISE): 0/6` that
never advances, `ACLgraph sizes capture fail`, `NPUGraph.cpp:... Waiting for
pending NCCL work to finish before starting graph capture`, then a worker
traceback that ends in `torch_npu/npu/graphs.py capture_end()`. The model is
loaded and memory profiling is done; this is not a load, profile or request-path
failure.

## Step 1 — separate the printed advice from the raised exception

v0.11.0rc0 (`00ba071`): `vllm_ascend/worker/model_runner_v1.py:3530-3565` wraps
`self._capture_aclgraphs(...)` in `try/except Exception`, logs

```
ACLgraph sizes capture fail: {type(e).__name__}:
ACLgraph has insufficient available streams to capture the configured number of sizes. ...
Recommended solutions:
1. ... '{"cudagraph_capture_sizes":[size1, size2, ...]}'.
2. Utilize ACLgraph's full graph mode as an alternative to the piece-wise approach.
```

and then `raise`. The sentence is emitted for **every** capture exception — a
collective timeout, an OOM, a Python error — because the `except` never inspects
the exception. The deciding text is what follows it (`{str(e)}`). So do not
answer "insufficient streams" or "graph mode unsupported" from that line; quote
the appended inner error first.

Version scope: this block exists at v0.11.0rc0. `rg -n "insufficient available
streams" vllm_ascend/` finds nothing at v0.20.2rc1 (`367b8e6`) — re-run that grep
before quoting the string on a newer image.

## Step 2 — classify the appended inner error

- `EI0006 ... Getting socket times out. Reason: Remote Rank did not send the data
  in time` + `ERR00100 PTA call acl api failed` + `The current working operator
  name is HcclAllreduce`, raised from `capture_end()` → a collective never
  completed during capture: lane C. Continue in
  `../hccl-and-hardware/collective-error-chain.md` and
  `../hccl-and-hardware/device-health-checks.md`.
- A Python-shaped exception (`AttributeError`, `TypeError`, allocation failure)
  that merely surfaced here → classify on the exception, not on the phase.

Timeline reading (recorded #3389, v0.11.0rc0, TP=16): 12 of the 16 ranks print
`NPUGraph.cpp:149` within one second (the excerpt is filtered — the ranks that
report the transport failure are not among them, so ask whether they ever
entered capture); `shm_broadcast.py:466 "No available shared memory broadcast
block found in 60 seconds"` at +60 s and +120 s is the EngineCore waiting on its
workers, not a separate fault; the inner `EI0006` is timestamped 11:32:29,
exactly 120 s after capture start, which is the connect-timeout value the message
itself quotes — read it as a timeout inside the capture window, with a genuinely
stuck peer still possible. `Transport init error ... rank[L]-... dst_rank[D]`
names the peers that never answered, so read that list rather than the reporting
rank.

## Step 3 — check whether the capture-size set is already the computed maximum

Upstream first: vllm v0.11.0rc3 (`8ce5d31`) `vllm/config/scheduler.py:222-227`
defaults `cuda_graph_sizes=[min(max_num_seqs * 2, 512)]`;
`vllm/config/__init__.py:551 _set_cudagraph_sizes` expands
`[1, 2, 4] + range(8, N + 1, 8)`, filters by `max_num_batched_tokens`
(:606-609) and calls `init_with_cudagraph_sizes` (:612), which sorts the list
descending. With `max_num_seqs=256` that is 67 sizes, `[512, 504, ..., 8, 4, 2, 1]`.

vllm-ascend then rewrites that list: `vllm_ascend/utils.py:286-390
update_aclgraph_sizes` (called from `platform.py:237`, PIECEWISE branch only).
`MAX_CAPTURE_SIZE = 1800` (:294; noted as bounded by 2048 streams with 248 kept
as buffer), `resources_per_graph = num_hidden_layers + 1` (:310),
`num_comm_groups = count(DP > 1, TP > 1)` (:316-319), then

- `HCCL_OP_EXPANSION_MODE=AIV` branch (:321-339): `parallel_factor = 1 + ncg + EP
  + multistream_overlap_shared_expert` (plus `DP > 1` for MoE), `max =
  floor(1800 / resources_per_graph / parallel_factor)`;
- default FFTS+ branch (:345-356): `max = floor((1800 - ncg*40) /
  resources_per_graph / (1 + 2*ncg))`.

It then samples `max` sizes uniformly from the original list, keeping the first
and last (:366-376), and logs
`Adjusted ACL graph batch sizes for <arch> model (layers: L): 67 → 6 sizes`
(:380).

Reproduce it instead of reading the formula by eye:

```
bash <skill_root>/scripts/aclgraph-size-budget.sh --layers 94 --tp 16 --dp 1 --era 0.11
```

gives `6` and `[512, 408, 304, 192, 88, 1]`, which is exactly the recorded
reporter's list — that is how the layer count and the parallel factor are
confirmed from the log alone.

Consequence for the answer: whether to advise "reduce `cudagraph_capture_sizes`"
is a computed question. Find `Calculated maximum supported batch sizes for ACL
graph: N` and `Adjusted ACL graph batch sizes ... 67 → 6 sizes` in the full log.
If the pasted count already equals the computed maximum, the sizes were
auto-reduced; the lever the code itself recommends for a higher ceiling is
`HCCL_OP_EXPANSION_MODE=AIV` (:358-364), and clearly it is not always higher —
run the script with and without `--aiv` for the reporter's layer and parallel
numbers before advising it. Setting fewer sizes than the maximum remains a
workaround for a long/stuck capture window, not a fix for stream exhaustion. If the count is far
below the maximum, a `--compilation-config` override is in play.

v0.20.2rc1 (`367b8e6`): same function at `vllm_ascend/utils.py:567`, called from
`platform.py:425`, same AIV/FFTS+ split, plus `CP_ADDITIONAL_STREAM_NUM = 100` (:578)
subtracted once per `pcp > 1` / `dcp > 1` (:641-643), MTP draft layers added to
`resources_per_graph` (:602-606) and `hf_text_config` instead of `hf_config`
(:592). At this commit `FULL_DECODE_ONLY` is a first-class ACL-graph mode whose
own startup warning names the observed failure shape — "capturing too many batch
sizes can lead to OOM (Out of Memory) errors or inference hangs ... manually
specifying a smaller batch size for graph capture" (`platform.py:432-447`) —
whereas v0.11.0rc0 coerced `FULL_DECODE_ONLY` to `PIECEWISE` for MLA models
(`platform.py:223-231`), so on 0.11 an MLA model's capture sizes are the
downsized ones even when the reporter asked for full decode graphs.

## Step 4 — cheapest discriminators before any defect claim

1. `--enforce-eager` skips capture entirely (`_use_aclgraph`,
   `vllm_ascend/worker/model_runner_v1.py:550-551` at v0.11.0rc0). If the server
   then starts and serves, the fault is in the capture phase, not the fabric or
   the request path.
2. Fewer capture sizes, or `HCCL_CONNECT_TIMEOUT` raised, separates "peers are slow
   during a long capture" from "a peer never answers".
3. Per-device health for the `dst_rank` ids from the `Create Dest` lines
   (`../hccl-and-hardware/device-health-checks.md`) plus `npu-smi info`.
4. Check the visibility map before calling a card faulty: #3389 reported
   `ASCEND_VISIBLE_DEVICES=10,7,6,14,4,9,1,2,13,12,0,3,15,5,11,8`, so the device
   ids in the log are not card indices. Ask for the card↔rank mapping.

## What the pasted log usually lacks

Reporters paste the traceback and the warnings, not the lines in between. Ask for
the log from `Starting to capture ACL graphs` to the first error (the
`Calculated` / `Adjusted` batch-size lines live there), the plog of the `dst_rank`
devices, `npu-smi info`, and the full `vllm serve` command including its env.
