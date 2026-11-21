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
before quoting the string on a newer image. It was replaced by a *signature gate*
that names one error code; read the v0.21–v0.23 section below before answering "the
log gave no hint".

## Step 2 — classify the appended inner error

- `EI0006 ... Getting socket times out. Reason: Remote Rank did not send the data
  in time` + `ERR00100 PTA call acl api failed` + `The current working operator
  name is HcclAllreduce`, raised from `capture_end()` → a collective never
  completed during capture: lane C. Continue in
  `../hccl-and-hardware/collective-error-chain.md` and
  `../hccl-and-hardware/device-health-checks.md`.
- A Python-shaped exception (`AttributeError`, `TypeError`, allocation failure)
  that merely surfaced here → classify on the exception, not on the phase.

- A **stream synchronize refusal** — `c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream),
  error code is 107027` + `EE9999 … Not allow to synchronize captured-stream,
  stream_id=N [FUNC:StreamSynchronize][FILE:api_error.cc]`, under the generic
  `ERR00100 PTA call acl api failed` — is a host readback issued while capture was
  active, not a device fault: `captured-stream-host-sync.md`.

  The gate is `copy_stream` **plus** the CANN sentence: a bare
  `AclrtSynchronizeStreamWithTimeout(stream), error code is 107027` out of
  `RunOpApiV2` with no `Not allow to synchronize captured-stream` line is a
  different lane, where the reporter-confirmed lever at v0.9.2rc1 was to **set**
  `ASCEND_LAUNCH_BLOCKING=1` - `captured-stream-host-sync.md` section 6.

Capture-mode copy family: `MemCopySync:Memory copy sync failed` +
`rtMemcpy:ErrCode=107030, desc=[the current capture mode does not support this
operation]` with `kind = 1` and a small `cnt` (72/456 bytes at #9183) is the
signature of a blocking host→device copy issued while a stream is being captured
or replayed. **The `kind` field is the cheapest split**: `kind = 1` is the load
side (host→device), `kind = 2` is a device→host **readback** — `.to("cpu")`,
`.item()` — and on 310P that is the fallback splitfuse mask under MTP +
`FULL_DECODE_ONLY` (`../soc-platform-paths/310p-mtp-graph-splitfuse-mask.md`). Recorded family: #6002 (v0.11.0, DS-V3.1-w8a8, `deepseek_mtp` +
`FULL_DECODE_ONLY`), #4208 (also `107033 AclmdlRICaptureTaskUpdateEnd`), #8502
(reporter-confirmed: `multistream_overlap_shared_expert: true` and serving fails,
"关闭多流后正常" — and at v0.19.1rc1 the *tutorial* is the side that sets it
`true` while `tests/e2e/nightly/.../GLM-5.yaml:34` sets `false`, so "I used the
documented command" refutes nothing: `../hccl-and-hardware/aclgraph-full-decode-runtime-crash.md`
§7, recorded #9263), #10700 (v0.21.0rc1 and v0.22.1rc1, GLM-5.1 on 2×A2, DP2×TP8,
`deepseek_mtp` + `FULL_DECODE_ONLY` + `npugraph_ex`: the record lands in engine
init, the report surfaces 27 min (v0.21.0rc1) or 101 min (v0.22.1rc1) later at
`execute_dummy_batch` ->
`acl_graph.__call__ -> synchronize()` — `../spec-decode-mtp/draft-graph-capture-device-fault.md`).
#6002 attributes it to a framework-side blocking copy and
proposes a capture-mode switch; that is the reporter's reading with no confirmed
fix in the record, so the cheap test stays `--enforce-eager` on the same traffic.
Two guards before you use it: it must belong to **this process** (`../hccl-and-hardware/error-report-timestamps.md` —
at #9183 the identical block was the previous day's crash, 36 h older than the
failure; at #10700 an equally old-looking block was this process's own init
window, because the report was late, so compare it with the log's startup lines
rather than discarding it), and the same block prints `threadCaptureMode=0, exchangeCaptureMode=0,
contextCaptureMode=0`, so its own fields do not prove a capture was in progress.

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

Host-side counterpart (from #3902): the same ticket shape — "the 910C machine
crashes", TP=16 of a 235B model on one host — also arrives with a `top` paste and
no log at all, and then the number that decides it is host RAM: 16 python3 ranks
at 88-100 GiB RES each on a 2 TB host with swap 0, before any capture work. Rule
out host memory before reading capture logs and keep the two artefacts separate;
`../runtime-memory/host-ram-and-per-rank-rss.md` + `scripts/host-ram-check.sh`.

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

## At v0.21–v0.23: a bare device code, a one-signature hint gate, no reserved pool

Verified at vllm-ascend **v0.23.0rc1** (`f4a08bd`) with vllm **v0.23.0** (`0fc695f`)
while triaging #16581 (ModelSlim W8A8 Qwen3-32B, one 910, default config):
`RuntimeError: npuSynchronizeDevice … AclrtSynchronizeDeviceWithTimeout, error code
is 507015` raised inside `torch.npu.graph.__enter__`
(`torch_npu/npu/graphs.py` → `torch.npu.synchronize()`, downstream frame
`compilation/acl_graph.py:192`), then wrapped as `NPUModelRunner failed, error is …`,
with a device dump `fftsplus aivector error … error code = 0x800000` / `The DDR
address of the MTE instruction is out of range` / `retCode=0x26, [aicore exception]`.
The reporter's frame numbers match that commit, so the pin is citable even though the
image is not byte-verified.

- **The advice gate now matches exactly one signature.** `_STREAM_RESOURCE_ERROR_CODE
  = "207008"` plus the markers `insufficient_stream_resources` /
  `stream resources are insufficient` (`acl_graph.py:28-39`); only a match gets
  `_STREAM_RESOURCE_GUIDANCE` prepended (`:50-51`, called from `:209-211`). Every
  other code — `507015`, `507033`, an aicore dump — is re-raised raw. So "the log
  gave no hint" is this predicate, not a gap in the code's knowledge, and the
  guidance text it would have printed (fewer `cudagraph_capture_sizes`, lower
  `max_cudagraph_capture_size`, prefer FULL/FULL_DECODE_ONLY, disable graph mode)
  must never be read as a diagnosis of a code that did not match.
- **The outer wrapper erases the code.** `worker/model_runner_v1.py:5177`
  `raise RuntimeError(f"NPUModelRunner failed, error is {e}")`. Quote the inner block;
  the wrapper sentence is the same for every capture failure.
- **At this pin nothing reserves the graph pool before capture — but make the log
  say it.** Upstream `vllm/v1/worker/gpu_worker.py:421-464` calls
  `profile_cudagraph_memory()` and subtracts `cudagraph_memory_estimate_applied` from
  the KV budget under `current_platform.is_cuda()` **and**
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` (default `1`, `vllm/envs.py:1978`), so an
  NPU does not enter it; vllm-ascend overrides `determine_available_memory`
  (`worker/worker.py:527-597`) and `rg -c profile_cudagraph_memory vllm_ascend/`
  returns 0, with `available_kv_cache_memory_bytes = requested_memory -
  non_kv_cache_memory` (`:590`). The pool is measured only *after* capture (`:719`)
  and feeds a `--kv-cache-memory=…` suggestion for the next launch (`:728-757`) —
  never a re-size. So at `--gpu-memory-utilization 0.9` the KV cache fills the
  requested budget and capture must fit in the slack above it.
  **The recorded logs disagree, and the ref decides it:** startup logs at
  v0.21.0rc1 / v0.22.1rc1 do carry
  `ACL graph memory profiling is enabled (default since v0.21.0) …` /
  `ACL graph pool memory: … (actual), … (estimated)` and `… GiB for NPU graph memory`
  (#10700, quoted in `../runtime-memory/index.md`, *Reading a v0.21+ startup memory
  block*), while neither pinned v0.23.0rc1 tree contains those strings (upstream still
  prints `CUDA graph …`). Therefore the presence of that line in the reporter's log is
  the check: present ⇒ the pool was accounted for and the reserving gap is *not* the
  cause; absent on a v0.21+-era image ⇒ the gap is live. #16581 never opened its two
  linked repro logs, so this stayed open there.
- **The default capture set is large at this ref.** `platform.py:207-261` injects
  `max_cudagraph_capture_size = min(max_num_seqs * decode_query_len, 512)` (upstream's
  trailing `* 2` deliberately removed) and upstream's expansion
  (`vllm/config/vllm.py:1662-1730` at this pin) turns that into **51 sizes**
  `[1,2,4,8,…,512]`. Only A5 prunes (`prune_capture_sizes_for_950`,
  `MAX_CAPTURE_SIZES_FOR_950 = 4`, `platform.py:76,109-124`, called `:621`), so a
  910/310P paste with a long capture list is the default, not a misconfiguration —
  and Step 3's rewrite is gone: `rg -n "update_aclgraph_sizes|MAX_CAPTURE_SIZE\b"
  vllm_ascend/` returns nothing at this ref, so do not compute a v0.11-style
  `67 → N sizes` budget for a v0.23 paste.

**The attribution run is one variable at a time.** (a) default util +
`--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'`; (b)
`--gpu-memory-utilization` lowered ~0.05 + default sizes. (a) alone passes ⇒ the
captured size set is the trigger; (b) alone passes ⇒ the capture-phase slack is. A
"working config" that moved util, capture sizes *and* `--max-model-len` together
(#16581: 0.9 / default sizes / 16 K failing vs 0.85 / `[1,2,4,8]` / 40 K working) has
identified none of them — say so instead of crediting one, and note that its 16 K→40 K
change *raises* KV demand, which argues against the reporter's own memory story.

**`0x800000` MTE DDR out-of-range is not the allocator's `NPU out of memory`.** It is
an execution-time addressing fault, so the artefact that separates a memory-pressure
cause from a shape/op cause is the plog/rank line carrying `fault kernel_name=`
(precedent #3263: same block, `fault kernel_name = paged_attention_mask_1_mix_aic`;
#2424: Qwen3-32B startup, `vector core exception`). `torch.npu.graph.__enter__`
synchronizes the device before capturing, so the faulting kernel was launched earlier
— the warmup `_dummy_run` or the previous captured subgraph — and the innermost
Python frame is a sync point, not the owner
(`../hccl-and-hardware/error-report-timestamps.md`).

## What the pasted log usually lacks

Reporters paste the traceback and the warnings, not the lines in between. Ask for
the log from `Starting to capture ACL graphs` to the first error (the
`Calculated` / `Adjusted` batch-size lines live there), the plog of the `dst_rank`
devices, `npu-smi info`, and the full `vllm serve` command including its env.
