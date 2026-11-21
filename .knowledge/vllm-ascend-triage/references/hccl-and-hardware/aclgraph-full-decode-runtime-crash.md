# A full/partial ACL graph that captured fine, then dies under load

Scope: `compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"}` (or `FULL`),
the mode printed its own banner, every rank finished capture, requests are being
served — and then, typically once concurrency rises rather than on the first
request, the worker dies with `507011`, `fftsplus aivector error`,
`Illegal instruction, which is usually caused by unaligned UUB addresses`,
`mte error`, and/or `sqe_type=7(notify wait) … sq sw status error` - or it dies at
the **first replay** with a Python-level `ValueError: not enough values to unpack
(expected 13, got 9)` from `attention_v1.update_graph_params` (section 9).

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

## 7. A host-scalar readback frame is the owner, not the victim

A variant that looks like a device fault and is not: the deepest Python frame is a
`torch.repeat_interleave` / `bincount` / `.item()` on a **device** tensor, and the raise is

```text
RuntimeError: operator():../torch_npu/csrc/aten/common/LocalScalarDenseNpu.cpp:23
NPU function error: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream),
error code is 507011
```

`LocalScalarDenseNpu` is torch_npu materializing **one device value on the host** — it blocks
`copy_stream`, so it reports whatever the model stream already did (here the same
`507011`/`fftsplus aicore error`/MTE-OOB dump). The owner is the statement that needed a
Python scalar from a device tensor. Two readings, and the timestamps decide between them:
(a) the readback ran while a stream was in capture, so the refusal is the fault
(`rtMemcpy … the current capture mode does not support this operation`, `MemCopySync …
kind = 1, runtime result = 107030`); (b) that refusal is a **capture-window** record and the
`507011` is the stream's death — order the timestamps first
(`error-report-timestamps.md`, rule 13: recorded #11680 carried the capture block at
`09:20:38` and the traceback at `09:27:19`, six minutes apart).

Where the readback lives matters for the ladder. At v0.22.1rc1 (`5f6faa0`) the EP all-to-all
dispatch preprocess mixes host and device on purpose —
`vllm_ascend/ops/fused_moe/token_dispatcher.py:581-620`: `torch.histc` → two
`.to("cpu").numpy()` splits (`:591`, `:604`), then
`:612 torch.repeat_interleave(self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())`
only when `num_local_experts > 1`, else `torch.npu.synchronize()`. And capture wraps the whole
model forward (`compilation/acl_graph.py:206 with torch.npu.graph(aclgraph, pool=self.graph_pool)`
→ `:209 self.runnable(...)`; the `:157-164` early return is the eager/mismatched-mode path that
can still be nested inside an enclosing capture). So a MoE forward captured with any graph mode
executes that readback inside the capture region. Ladder: `cudagraph_mode` → default
`PIECEWISE`, then `additional_config` `enable_npugraph_ex: false`, then `multistream_overlap_shared_expert`
off, then `--enforce-eager` with `ASCEND_LAUNCH_BLOCKING=1` on one request.

**Recipe diff before any of that.** The pinned tree ships its own validated launch configs per
model × SoC under `tests/e2e/<tier>/**/configs/`; find the reporter's model there and diff it
flag by flag — the deltas are the discriminators, and a match is a version verdict:

```bash
rg -ln -i 'glm-5|glm5' tests/e2e/*/*/configs/ tests/e2e/*/*/models/ 2>/dev/null
sed -n '1,60p' tests/e2e/weekly/single_node/configs/GLM-5_1-W8A8_A3_weekly.yaml
```

Recorded #11680: that file matches the reporter's serve command on every env var
(`HCCL_OP_EXPANSION_MODE=AIV`, `HCCL_BUFFSIZE=200`, `VLLM_ASCEND_BALANCE_SCHEDULING=1`,
`VLLM_ASCEND_ENABLE_MLAPO=1`), on `additional_config` (`enable_npugraph_ex`, `fuse_muls_add`,
`multistream_overlap_shared_expert`), on `--compilation-config '{"cudagraph_mode":
"FULL_DECODE_ONLY"}'`, on `--speculative-config '{"num_speculative_tokens": 3, "method":
"deepseek_mtp"}'`, `--enable-expert-parallel`, `--max-num-batched-tokens 4096`,
`--quantization ascend`, `--seed 1024` — **except** `TP16/DP1` where the reporter ran
`--data-parallel-size 2 --tensor-parallel-size 8`, plus the reporter's added
`--async-scheduling`. That is the cheapest single ask (run the shipped shape), and it is owed
in the first answer, not after a source walk.

## 8. Recorded case

#9944 — vllm-ascend **V0.18.0** image (the snapshot pins only v0.19.1rc1 `da421af`, so
"upstream already fixed it" is unverifiable there), two A2 nodes, Qwen3.5-397B-A17B-w4a8,
PD disaggregation, `MooncakeLayerwiseConnector`, driver 25.5.1, and a *reproduction* rather
than a deviation: the D node's line is the published recipe's own
(`--compilation-config '{"cudagraph_capture_sizes":[1,4,8,12,…,128,160,172,196,200,212,232,256],
"cudagraph_mode":"FULL_DECODE_ONLY"}'`, `--max-num-seqs 128`, `--max-num-batched-tokens 8192`,
`--async-scheduling`, `--speculative_config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`,
`--no-disable-hybrid-kv-cache-manager`), while the P node is the same model with
`--enforce-eager` and no graph. The death is the D worker on `rejection_sampler` →
`repeat_indices_cpu.to(...)` / `synchronize_input_prep` → `SUSPECT REMOTE ERROR 507057`, plog
`aivec error … The DDR address of the MTE instruction is out of range`, at ~concurrency 60
under a performance test. Two consequences for this page: (a) rung 1 is a **deviation from
the documented recipe**, so say that in those words and quote its capture list, or the
reporter will answer "I used the official command"; (b) §4's bucket read: 60 pads to the
**64** entry, which the list contains, and 128 covers `--max-num-seqs` — so "the concurrency
fell outside the capture list" is excluded before the run starts, and the per-replay argument
mismatch (§3) or the device-fault lanes are what remain. The run delivered this ranking plus
the artefact list, and the reporter's next comment supplied exactly those two serve commands
(the P/D scripts above), so that ask was the right one.

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

#11680 — `v0.22.1rc1 A3单机 GLM-5.1-w8a8 推理过程中 LocalScalarDenseNpu 算子报错`, image
`quay.io/ascend/vllm-ascend:v0.22.1rc1-a3` (`5f6faa0`), DP2×TP8 + EP, `--async-scheduling`,
`FULL_DECODE_ONLY`, `enable_npugraph_ex`/`fuse_muls_add`/`multistream_overlap_shared_expert`,
`deepseek_mtp` ×3, 20 480 ctx. The paste's decisive artefacts: (a) the `LocalScalarDenseNpu` /
`copy_stream` / `507011` frame under `acl_graph.py:164` → `fused_experts` →
`token_dispatcher.py:561 → :612` (§7), (b) an `EE9999` block stamped six minutes earlier
carrying `rtMemcpy … current capture mode does not support this operation` — a capture-window
record, not this fault, and (c) the shipped GLM-5.1 A3 weekly recipe differing only in
TP/DP + `--async-scheduling`. Precedent #9816 (A3, DP+EP, w8a8+MTP, `FULL_DECODE_ONLY`,
`multistream_overlap_shared_expert`, 507011/MTE-OOB on the idle dummy batch) is the nearest
body in the corpus. The recorded session had all three by its 8th assistant turn and then took
"one last guard read", ending at the 16 384-token cap with **no answer message** — 51.2 K
characters of reasoning against 2.8 K of text.

## 9. A replay-time unpack error is a capture/replay dispatch mismatch, not a model defect

Signature (recorded #8370, Gemma4 hybrid `sliding`+full attention, TP=4, 910B1,
vllm-ascend v0.18.0rc1 `99e1ea0f` / vllm v0.18.0 `bcf2be96`): capture with
`cudagraph_capture_sizes=[1]` completes (`Graph capturing finished in 4 secs`), the
first request logs `Replaying aclgraph`, then `ValueError: not enough values to
unpack (expected 13, got 9)` from `compilation/acl_graph.py update_full_graph_params`
-> `attention_v1.py update_graph_params`, killing the engine.

Source-verified at that pin: the two sides pick the attention primitive by
**different predicates**, and the captured tuple's arity follows the choice.
- Capture: `attention_v1.py:920-926` takes paged attention only when
  `attn_state == DecodeOnly` **and** `using_paged_attention(num_tokens, vllm_config)`
  **and** `self.sliding_window is None`; `full_graph_pa` (`:626`) appends a
  **9**-field tuple (`:657-670`) and closes its task group at `:672-685`.
- Replay: `update_graph_params:397` takes paged attention on
  `using_paged_attention(num_tokens, vllm_config)` **alone** - the
  `sliding_window is None` term is missing - and the FIA branch unpacks **13**
  fields (`:470-493`).
- `using_paged_attention` (`attention/utils.py:44-55`) needs
  `cudagraph_mode == FULL_DECODE_ONLY`, no `speculative_config`, a non-A5 SoC and
  `num_tokens in pa_shape_list`, whose default is `[]` (`ascend_config.py:127`).

So `expected 13, got 9` means replay entered the FIA branch over a key capture filled
from the PA path. Discriminating checks: the reporter's `cudagraph_capture_sizes`,
whether `--additional-config '{"pa_shape_list": [...]}'` covers those same sizes
(docs describe it as a temporary PA-vs-FIA tuning knob,
`docs/source/tutorials/models/Qwen3-Dense.md:196`; `faqs.md:270` warns that setting it
swaps the operator), and whether the layers carry per-layer `sliding_window` - a
hybrid sliding/full stack is exactly where the two predicates diverge. Ladder: set
`pa_shape_list` to the captured sizes; else move `FULL_DECODE_ONLY` to `PIECEWISE`;
else `--enforce-eager`. Name the predicate you verified instead of a model defect.

Capture-side sibling in the same report: `[1,2,4,8]` failed at `AclmdlRICaptureEnd ...
error code is 107033` / `EE9999 ... reason=task group status error` while `[1]`
captured - a task-group close failure at multi-size capture on the same PA path
(`graph_task_group_begin/end` at `:672-685`); worth the `acl-graph-capture.md` Step 3
size-set check, not a confirmed mechanism. Coverage note: neither tree in that
snapshot registers `Gemma4ForConditionalGeneration`
(`vllm/model_executor/models/` stops at `gemma3n`), so that pair carries no validated
FULL-graph path for the model at all - state it as the coverage gap, not the cause.
