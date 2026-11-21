# A full/partial ACL graph that captured fine, then dies under load

Scope: `compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"}` (or `FULL`),
the mode printed its own banner, every rank finished capture, requests are being
served — and then, typically once concurrency rises rather than on the first
request, the worker dies with `507011`, `fftsplus aivector error`,
`Illegal instruction, which is usually caused by unaligned UUB addresses`,
`mte error`, and/or `sqe_type=7(notify wait) … sq sw status error`.

Sibling lanes: the **hang** variant of the same mode is
`aclgraph-full-decode-runtime-hang.md`; a failure during capture is
`../startup-failures/acl-graph-capture.md`; a Python frame naming a graph API is
`../spec-decode-mtp/draft-graph-capture-device-fault.md`.

## 1. Parse the code before the mechanism

`507011` is `ACL_ERROR_RT_MODEL_EXECUTE` "model execute failed"
(`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:93`; also
`cann/runtime/docs/api_docs/aclError.md:164`). It is emitted for any failure of
the model stream and names no operator — the same wrapper precedes a MoE
dispatch fault (#4256) and an MTP capture-mode copy (#10700), so it localizes
nothing by itself. The lines that name the owner:

- `Op execute failed. origin_op_name [<Op>], op_name [<Op>], error_info:
  task_id …, stream_id …, device_id …, retcode 0x…` — the error-tracking
  callback; this is what named `MoeDistributeDispatchV2` in #4256.
- `[AIC_INFO] … fault kernel_name=<Op>_<hash>_high_performance_<n>` in the
  plog.

Ask for those two first. Without them the ranked list below is a family, not a
cause.

## 2. The victim lines are victims

`sqe_type=7(notify wait) … errType=0x20(sq sw status error)` on the model stream
and `mte error … streamId=<other>` are **downstream**: once a task in a stream
faults, the next notify-wait in that stream reports a bad software status, and a
second stream's copy reports the latched state. Do not read them as an HCCL,
event or multi-stream bug — same attribution rule as
`collective-error-chain.md`. The one structural fact worth keeping is that two
streams were live, which §3 predicts.

## 3. What FULL mode actually does per replay

At vllm-ascend v0.13.0rc1 (`1b5d5ab`) the decode graph is not frozen. Before
every replay, `update_attn_params()` (`vllm_ascend/compilation/acl_graph.py:299-304`)
writes the step's attention arguments into the captured graph on a separate
`update_stream` (`torch.npu.graph_task_update_begin`, `:240`, `:276`), then
`torch.npu.synchronize()` (`:193`) is called before `entry.aclgraph.replay()`
(`:194`) because the CPU-side record for iteration *i* can complete before the
replay of *i-1* — the comment at `:189-192` states it.

Two consequences for triage:

- A per-replay argument that disagrees with what was captured — workspace size,
  `seq_lens`, `block_table` width, stream/event ordering — reaches the device as
  an **address** fault, i.e. exactly this aivector / MTE / illegal-instruction
  shape, with no Python frame.
- So attribution starts at *which replay* and *which argument*, and both are in
  the serve command (next section). It does not start at a source walk.

**Version flip: on the v0.11 line the ordering guard is absent.** At v0.11.0rc1
(`c5fe179`) the same file records update events (`event.record(update_stream)` at
`:249` and `:307`) but has **no** `torch.npu.synchronize()` before
`entry.aclgraph.replay()` (`:190`), and nothing consumes `graph_params.events`
(`rg -n 'event\.wait' vllm_ascend/compilation/acl_graph.py
vllm_ascend/worker/model_runner_v1.py` is empty). So there `--async-scheduling` is
a live hazard and `--async-scheduling False` is a real ablation; at v0.13+ the
`:189-196` synchronize is the repair and the same ablation tests something else.
Re-grep the two symbols before reusing this either way.

## 4. Two discriminators, both read off the config

- **Bucket.** `cudagraph_capture_sizes` against the concurrency at first
  failure: `[1, 8, 16]` dying at 8 means bucket 8 is the failing graph; `[1]`
  dying at 1 means it is not bucket-specific. The log also prints
  `Starting to capture ACL graphs for cases: […]`. One datum, and it splits
  "one bad bucket" from "every replay".
- **Which attention patch path.** `using_paged_attention()` is True only when
  `cudagraph_mode == FULL_DECODE_ONLY`, the SoC is not A5, there is no spec
  config, **and** `runtime_shape in pa_shape_list`
  (`vllm_ascend/attention/utils.py:18-28`); `pa_shape_list` defaults to `[]`
  (`vllm_ascend/ascend_config.py:119`). The legacy `_npu_paged_attention` path —
  and the tree's own known-bug note about a workspace sized from `max_model_len`
  at capture being too small for some `seq_lens`
  (`acl_graph.py:198-254`, `_update_attn_pa_params`) — therefore applies **only
  to a deployment that opted in** through `--additional-config`. A default
  launch takes `_update_attn_fia_params` (`:256`); ranking the paged-attention
  workspace first for such a run spends the session on a path the config
  excludes. Read `pa_shape_list` before ranking that mechanism.
- Corollary: when the serve command says `async_scheduling: false` and there is
  no PP, the ordering clause in the `:189-196` comment does not describe that
  run — downgrade the ordering story and keep the argument-mismatch story.
- The opt-in condition itself is v0.13+ behaviour. At v0.11.0rc1 neither
  `using_paged_attention` nor `pa_shape_list` exists anywhere in `vllm_ascend/`,
  so there a non-MLA model takes the `_npu_paged_attention` update path with the
  tree's own GQA workspace note (`compilation/acl_graph.py:216-224`) by default
  and ranking it is legitimate — check which symbols the reporter's ref has
  before applying the paragraph above.

## 5. Version before mechanism (rule 16 unchanged)

Check what the pinned tree validates for the mode before reading code paths:
at v0.13.0rc1 the FULL-mode e2e coverage is Qwen3-0.6B (single-card accuracy),
`Qwen3-30B-A3B` at TP2 with `max_model_len=1024`
(`tests/e2e/multicard/test_full_graph_mode.py`, wired at
`.github/workflows/_e2e_test.yaml:208`), plus Qwen3-Next, shared-expert DP, MTP
and long-sequence files — no 32B dense, no TP8, no concurrency stress. A release
candidate outside that envelope, plus a device fault that only appears under
load, is a version answer at least as much as a mechanism one: state both, and
give the ablation that separates them (rung 1).

## 6. Ladder

1. Drop `cudagraph_mode` (default `PIECEWISE`) and repeat the same stress test.
   Survives → the mode's replay path is the lever; still dies → the fault is not
   graph-specific, go back to the device-fault lanes.
2. `--enforce-eager` (required before `ASCEND_LAUNCH_BLOCKING=1`, which the
   platform refuses while a graph mode is set — grep the gate message in
   `vllm_ascend/platform.py`).
3. Single bucket: `cudagraph_capture_sizes: [<the failing concurrency>]` against
   `[1]`, and lower `--max-num-seqs` / `--max-num-batched-tokens` to the failing
   concurrency.
4. `ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
   ASCEND_LAUNCH_BLOCKING=1` with **one** request, to get the first-fault op with
   attribution intact.
5. `asys health -d=$i` on every device plus the `chipId` in the `EZ9999` dump —
   only if the fault follows one chip (`device-health-checks.md`).
6. If it only survives with the mode off, file with the capture-size list, the
   concurrency at first fault, the `origin_op_name` / `fault kernel_name`
   lines and both serve commands.

## 7. Recorded case

#6018 — title says vllm-ascend `0.12.0rc1`, the environment block says
`0.12.0rc2` (`SOURCES.json` marks the reported ref `conflicting_versions`; the
snapshot pins v0.13.0rc1 `1b5d5ab`), 910B4 ×8 single node, "Qwen2.5 30B" (the
name is ambiguous between dense Qwen2.5-32B and MoE Qwen3-30B-A3B — ask), CANN
8.3.RC2 + torch_npu 2.8.0, which **is** the documented pair for the 0.13 line
(`docs/source/installation.md:15`, `requirements.txt:31`), so the driver/CANN
pairing is not the first cause here. The reporter's later comment supplies the
config: `tensor_parallel_size 8`, `data_parallel_size 1`, ray executor,
`max_model_len 32768`, `async_scheduling false`, `cudagraph_capture_sizes
[1, 8, 16]`, `cudagraph_mode FULL_DECODE_ONLY`, `max_num_seqs 16`,
`max_num_batched_tokens 4096`, crash at concurrency 8. By §4 that is bucket 8 on
the FIA patch path with no `pa_shape_list`, and no async scheduling. The
recorded run reached the per-replay patch family and the coverage verdict and
delivered a ranked answer — but emitted **zero message text on its first 13
turns** and wrote the whole answer on the 14th, its last.

#4326 — title `Qwen3-moe FULL_DECODE_ONLY inference crash`, vllm-ascend
v0.11.0rc1 (`c5fe179`) + vllm v0.11.0 (`b8b302c`), Qwen3-30B-A3B-W8A8, DP8×TP1 +
`--enable-expert-parallel`, 32768 ctx, `--async-scheduling`,
`VLLM_ASCEND_ENABLE_FLASHCOMM`. Two paste facts decide the shape before any
mechanism: the body is only the OpenAI frontend re-raising `EngineDeadError … See
stack trace (above)` once per queued streaming output (~15 times), so the death is
at **concurrency**, not on the first request; and the attached plog is 44 minutes
earlier, at interpreter teardown (`cann_kb_finalize`, `ConnectionRefused`), so no
root-cause stack and no in-window plog exist in the report. Coverage at that pin:
the only FULL-mode e2e for this model is `tests/e2e/multicard/test_full_graph_mode.py`
(`Qwen3-30B-A3B`, TP2, `max_model_len=1024`, bf16) and **no** `.github/workflows`
file references it — the wired graph runs are `singlecard/test_aclgraph.py` and
`test_aclgraph_mem.py` (`_e2e_test.yaml:77,91-92`) — so DP+EP+W8A8+32K+async is
outside everything that release ran. The recorded run reached this section's rungs
and the coverage verdict and delivered a ranked answer on its 11th and last turn,
after seven of its eleven turns carried no message text at all (rule 1: a
truncation at that point would have kept nothing).
