# `207005` / `Alloc sq cq fail` inside `AclmdlRICaptureBegin`: the device stream table ran out during graph capture

Scope: startup reached graph capture and a worker raises, with two wrappers in front of
the real line:

```text
RuntimeError: Worker failed with error 'NPUModelRunner init failed, error is NPUModelRunner
  failed, error is capture_begin:…/torch_npu.dir/compiler_depend.ts:214 NPU function error:
  c10_npu::acl::AclmdlRICaptureBegin(capture_stream_, capture_mode), error code is 207005
[ERROR] … (PID:…, Device:0, RankID:-1) ERR00100 PTA call acl api failed
[Error]: Failed to apply for memory.
        Check the remaining storage space in the hardware environment.
EL0006: … The resources are insufficient.
        Solution: Close applications not in use.
       TraceBack (most recent call last):
       [SqCqManage]Alloc sq cq fail, stream_id=1984, retCode=0x7020023.[FUNC:AllocStreamSqCq][FILE:stream_sqcq_manage.cc][LINE:78]
       [SqCqManage]Alloc sq cq fail, stream_id=1984, retCode=0x7020023.[FUNC:Setup][FILE:stream.cc][LINE:655]
       rtStreamBeginCapture execute failed, reason=[driver error:resource alloc fail]
       begin capture stream failed, runtime result = 207005[FUNC:ReportCallError][FILE:log_inner.cpp:161]
```

Recorded: **#2629** (v0.9.1, Qwen3 MoE with tp/dp/ep — `stream_id=1984` and `1985`),
**#8393** (image `v0.18.0rc1-a3`, A3, TP=4, Qwen3.5-9B, `--max-model-len 262144`,
`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`) and
**#5557** (v0.12.0rc1 `42ceaf0`, 910B4 32G cut into vNPUs with
`npu-smi set -t create-vnpu … vir10_3c_32g`, `--task embed` on bge-m3 =
`XLMRobertaModel`, 24 layers, TP=1, no MTP, `dtype float32`, 51 default sizes). Same
three lines every time, five releases apart. #5557 is the plainest instance and the
first where the budget *passed*: `Calculated maximum supported batch sizes for ACL
graph: 72` then `No adjustment needed for ACL graph batch sizes: … with 51 sizes`, and
the device still refused at graph **38 of 51** with `stream_id=2011` — ~53 streams per
captured graph against the formula's `layers + 1 = 25` (§2).

## 1. What owns the failure, and the two lines that mislead

- `[Error]: Failed to apply for memory. / Check the remaining storage space in the
  hardware environment` + `EL0006 … resources are insufficient / Close applications not
  in use` is ACL's **canned** text for the code, not a host-disk, container-size or HBM
  measurement — nothing about this report needs a `df` or a migration.
- `207005` is `ACL_ERROR_RT_RESOURCE_ALLOC_FAIL`
  (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:66`;
  `cann/runtime/docs/api_docs/aclError.md:140`: 资源申请失败，请检查 Stream 等资源的使用情况).
  The line under it names *which* table refused: `Alloc sq cq fail` in
  `AllocStreamSqCq`/`stream_sqcq_manage.cc` then `stream.cc` — the stream's
  submit/completion queue.
- The downstream budget is written against the same number: "we can only capture 1800
  graphs at most … bounded by the number of streams, which is 2048, we save 248 streams
  as a buffer" (`vllm_ascend/utils.py:459-464` at v0.18.0rc1 `99e1ea0`). `stream_id=1984`
  sitting next to that 2048 is consistent with the table being at its wall; it is an
  inference from the code's own comment, not a device readout.
- This is **not** an `NPU out of memory` and not the KV pool: `--gpu-memory-utilization`
  and `--max-model-len` do not move it, while `--enforce-eager` (no capture at all) does.
- It is also not a defect owed a fix at the reported ref: the project documents exactly
  this failure as FAQ #18 at v0.18.0rc1 (`docs/source/faqs.md:209-223`, "size capture
  failures resulting from stream resource exhaustion") together with its own root cause —
  the requirement formula counts only measurable factors (dp/tp/ep, piece graph count,
  multistream-overlap, HCCL mode) while "numerous unquantifiable elements, such as
  operator characteristics and specific hardware features, consume additional streams
  outside of this calculation framework".

## 2. A3 is the setting where the formula under-counts most

Quote the tree's own comment instead of theorising
(`vllm_ascend/utils.py:536-547` at `99e1ea0`): under FFTS+ ("applicable to the A2
hardware") one collective adds a single stream, whereas "on A3 hardware, HCCL defaults to
the AICPU method. This approach may additionally allocate up to rank_size (max 16) - 1
streams per collective communication domain on the device (worst case) … will lead to a
significant reduction in the maximum supported sizes". That is why #2629 and #8393 are
both A3-flavoured, and why the same paste on A2 is unremarkable. The consequence for the
answer: recommend a reduced capture set for A3, and treat `HCCL_OP_EXPANSION_MODE=AIV`
as the code's lever **only after pricing both branches** (§3) — on a dense model the AIV
branch *subtracts* `parallel_factor × resources_per_graph` from the budget.

**The A2 branch under-counts too, and the failing `stream_id` prices it.** #5557 runs
the FFTS+/A2 arithmetic (`floor((1800 - 0*40) / 25 / (1 + 0*2)) = 72`) on a 24-layer
dense encoder with no collectives at all, and the table still ran out two-thirds of the
way through a 51-size list — so "A3-flavoured" is a tendency, not a predicate, and a
vNPU-partitioned card is not required for the symptom. The cheap discriminator between
"this box's table is smaller" and "the formula's per-graph cost is wrong" is the
refusal's own `stream_id`: a shrunk table would have stopped allocating near its size,
while #5557 reached `2011` of the comment's ~2048, i.e. the table is full-size and the
per-graph cost is what the formula under-states (`2048 / 53 ≈ 38` sizes fit). Both
numbers — `stream_id` at refusal and `Capturing ACL graphs … N/M` — are readouts the
reply needs; the 2048 itself stays an inference from the code's comment, not a device
probe.

## 3. Reproduce the budget at the reporter's ref

`vllm_ascend/utils.py:456-560` at `99e1ea0` (`update_aclgraph_sizes`, called from
`platform.py:370` in the PIECEWISE branch **only**):

- `MAX_CAPTURE_SIZE = 1800` (`:464`), `CP_ADDITIONAL_STREAM_NUM = 100` (`:467`) subtracted
  once per `pcp`/`dcp` > 1 (`:529-533`);
- `resources_per_graph = num_hidden_layers + 1` (`:489`) **plus** `draft.get_total_num_hidden_layers() + 1`
  when a draft model is configured (`:492-495`) — MTP doubles this term, which is the
  #8393 shape;
- `num_comm_groups = sum(dp > 1, tp > 1)` (`:498-504`);
- FFTS+ (default): `floor((1800 - ncg*40) / resources_per_graph / (1 + 2*ncg))`
  (`:548-555`, and the warning there tells the user AIV would raise the ceiling);
- AIV: dense models get `MAX_CAPTURE_SIZE -= parallel_factor*resources_per_graph`, then
  `floor(budget / resources_per_graph / parallel_factor)` (`:506-524`).

One call, no source reading:

```bash
bash <skill_root>/scripts/aclgraph-size-budget.sh --layers <num_hidden_layers> \
  --tp 4 --draft-layers <MTP layers> --era 0.18
# then add --aiv to price the alternative, and compare both numbers with the log's
# "Calculated maximum supported batch sizes for ACL graph: N" / "Adjusted ACL graph
# batch sizes … 67 → N sizes" lines
```

How many sizes get captured in the first place at this ref: the platform injects a
default ceiling `min(max_num_seqs * decode_query_len, 512)` with
`decode_query_len = 1 + num_speculative_tokens` (`platform.py:157-192`; upstream's own
`* 2` is the term that injected default deliberately drops), so TP=4 + MTP
`num_speculative_tokens=2` at the default `max_num_seqs` gives 512 → a 51-entry list
(`[1,2,4]` + step 8 to 248 + step 16 to 512, `vllm/config/vllm.py:1101-1110` at vllm
v0.13.0), and the MTP drafter captures **that same list again**
(`model_runner_v1.py:3318` → `set_draft_graph_params`, `compilation/acl_graph.py:275-297`).
Two capture runs of 51 graphs each is what the stream table has to absorb. Note the
asymmetry: `FULL_DECODE_ONLY`/`FULL` takes `platform.py:374-…`, which does **not** call
`update_aclgraph_sizes` — a full-decode launch gets no budget rewrite at all, so ask which
mode the log banner names (`PIECEWISE compilation enabled on NPU` vs
`FULL_DECODE_ONLY compilation enabled on NPU`) before quoting a computed maximum.
**Whether the log carries the stream advice is ref-specific, and the interval is not
monotone.** The block is present at v0.11.0rc0 (`model_runner_v1.py:3553-3565`) and at
v0.12.0rc1 — `model_runner_v1.py:3143-3152` at `42ceaf0`, gated on a
`retCode=([^,\s\.]+)` match for `0x7020023` and followed by `raise` — **absent** at
v0.18.0rc1 (`99e1ea0`: `rg -n 'insufficient available streams|_STREAM_RESOURCE|207008'
vllm_ascend/` returns nothing), and back `207008`-gated by v0.21+
(`acl_graph.py:28-51`). So "the log gave no advice" is a v0.13-v0.20 statement only: at
an earlier rc the advice block *is* in the paste, its presence says nothing about a
fix, and at any ref it is printed for this retCode rather than for a measured stream
budget.

## 4. Levers in order, and the one-variable rule

1. `--enforce-eager` on the same command: separates capture-phase exhaustion from every
   request-path and fabric explanation, and is the proof the reporter can run in minutes.
2. Explicit smaller set — `--cudagraph-capture-sizes 1 2 4 8 16` (or
   `--max-cudagraph-capture-size 32`) — FAQ #18's own first recommendation. The flag
   exists at this ref (`vllm/engine/arg_utils.py:1106`).
3. `HCCL_OP_EXPANSION_MODE=AIV`, only with the §3 arithmetic showing a higher ceiling for
   *this* model's layer count and `num_comm_groups`.
4. Fewer captured graphs by construction: drop `--speculative-config` (removes the second
   51-graph run and the `+1` draft term) or lower `--max-num-seqs` — a sizing decision, and
   the answer should say which of the four it is testing.

**On a pooling serve, the advice block's own solution 2 does not exist.** The
`ACLgraph has insufficient available streams … Recommended solutions: 2. Utilize
ACLgraph's full graph mode as an alternative to the piece-wise approach` text (and FAQ
#18's second recommendation, which repeats it) is unreachable for `--task embed` /
`score`: upstream overrides the mode — `WARNING … Pooling models do not support full
cudagraphs. Overriding cudagraph_mode to PIECEWISE.` (`vllm/config/vllm.py:702` at vllm
v0.13.0; the #5557 paste prints it as `[vllm.py:693]`, its vllm being v0.12.0, one line
above `PIECEWISE compilation enabled on NPU`). So an embedding server's ladder is
`--enforce-eager` or a reduced size set, and quoting the log's solution 2 back at the
reporter buys a restart that changes nothing.

Each lever must be tested one at a time. A "working command" that changed the capture set,
the parallelism and `--max-model-len` together identifies none of them (the #16581 trap in
`acl-graph-capture.md`).

## 5. Precedent status and what the paste lacks

#2629 carries a maintainer note that the bug "has been resolved in `main` … PR 2511", yet
the identical `AclmdlRICaptureBegin` / `207005` / `Alloc sq cq fail, stream_id=1984`
block reappears at a 2026 `v0.18.0rc1-a3` image. So do not answer "fixed in main, upgrade"
from the old thread: the sentence recurs because the budget formula under-counts (FAQ #18
says so in the tree), and the ref-specific arithmetic is what decides whether this
reporter's set is over the ceiling.

Ask for, rather than derive: the full log from the capture banner to the raise
(`Calculated maximum supported batch sizes for ACL graph: N`, `Adjusted ACL graph batch
sizes …`, the `Starting to capture ACL graphs for cases: [...]` list, and which of
`PIECEWISE`/`FULL_DECODE_ONLY` banner appeared), the model's `num_hidden_layers` and MTP
layer count from `config.json`, `--max-num-seqs`, and `npu-smi info` from one rank. A
paste that starts at the traceback has none of them.
