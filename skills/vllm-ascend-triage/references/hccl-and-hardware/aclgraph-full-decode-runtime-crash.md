# A full/partial ACL graph that captured fine, then dies under load

Scope: `compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"}` (or `FULL`),
the mode printed its own banner, every rank finished capture, requests are being
served — and then, typically once concurrency rises rather than on the first
request, the worker dies with `507011`, `fftsplus aivector error`,
`Illegal instruction, which is usually caused by unaligned UUB addresses`,
`mte error`, and/or `sqe_type=7(notify wait) … sq sw status error` - or it dies at
the **first replay** with a Python-level `ValueError: not enough values to unpack
(expected 13, got 9)` from `attention_v1.update_graph_params` (section 9) - or the
**first decode** raises a host-side `RuntimeError` whose two numbers are the
batch-padding arithmetic, `Target sizes: [N]. Tensor sizes: [M]` at
`model_runner_v1.py _prepare_inputs`, with no device code at all (section 10).

Sibling lanes: the **hang** variant of the same mode is
`aclgraph-full-decode-runtime-hang.md`; a failure during capture is
`../startup-failures/acl-graph-capture.md`; a Python frame naming a graph API is
`../spec-decode-mtp/draft-graph-capture-device-fault.md`.

## 0. Before any mechanism: the reported release's own Known Issues row

One grep on the pin of the **reported** release, before a single code path:

```bash
rg -n -i 'known issue' -A 12 <pin>/docs/source/user_guide/release_notes.md
```

At v0.12.0rc1 (`42ceaf0`) `:42` reads **"DeepSeek 3/3.1 and Qwen3 doesn't work with
FULL_DECODE_ONLY graph mode. We'll fix it in next release. #4990"** while the same
section's Core bullet (`:13`) invites the reports — "[Experimental] Full decode only
graph mode is supported now … Let us know if you hit any error. We'll keep improve it
and enable it by default in next few release". So a crash whose model is in the row's
family and whose command sets the row's mode is *answered by the row*: quote it, give
the tracked PR, name the release to try, and keep the mechanism below as the
explanation rather than the verdict. That is what #5076 was (Qwen3-Coder-480B-A35B,
`{"cudagraph_mode":"FULL_DECODE_ONLY"}`, DP4×TP8, `--max-num-seqs 64`); the reporter
later confirmed **0.13.0rc1 fixed it**, which is "next release" as the row promised.
The row itself is already recorded in this library for the **accuracy** context
(`../accuracy-benchmarks/qwen3-next-gdn-numeric-path.md` §v0.12.0rc1, "quote the release's
own statements"); what is new here is reading it as the *first* check on a **crash** of
that mode, before any code path. Two limits: it is scoped to the families it names (a
DeepSeek-V3.2, Qwen3.5 or GLM report does not inherit it — read the same release's other
bullets and `../release-known-issues/index.md` rule 6), and it names no mechanism, so it
does not tell you *which* line — sections 1-10 still decide that. The recorded #5076
session reached this row on its last turn and shipped no message (rule 1).

`Replaying aclgraph` proves only that *some* captured graph replayed. On a
spec-decode launch read the gate before attributing the fault to that replay: with
`--async-scheduling` on, `method: mtp` forces the drafter eager
(`vllm_ascend/spec_decode/eagle_proposer.py:136-142` at v0.18.0rc1), so the lines are
the **target** model's `FULL_DECODE_ONLY` buckets and the drafter's host-side input
preparation (padded-batch indices, `_get_positions`) is an equally ranked candidate —
`../spec-decode-mtp/empty-first-response.md` Step 6b.

On a **DeepSeek-V3.2 (`deepseek_v32`) MTP** launch read one more gate before ranking
anything: the combination is eager-only **upstream**, and the plugin deliberately overrides
that. Upstream vLLM v0.13.0 sets `self.enforce_eager = True` when
`target_model_config.hf_text_config.model_type == "deepseek_v32"`, commented
`# FIXME(luccafong): cudgraph with v32 MTP is not supported, remove this when the issue is
fixed` (`vllm/config/speculative.py:254-257`); vllm-ascend force-clears it —
`vllm_ascend/platform.py:359-369` at v0.13.0 (`6281c120`) and `:504-517` at v0.18.0rc1
(`99e1ea0`), under a comment that admits the fix "will also implicitly ignore user setting
of `speculative_config.enforce_eager`". Four consequences: (a) the spec-scoped
`"enforce_eager": true` inside `--speculative-config` is a **dead lever** for this family —
never offer it as an ablation or a fix (global `--enforce-eager` still works); (b) there is
no startup gate left to ask for, because the plugin removes upstream's; (c) the block
survives v0.13.0 → v0.18.0rc1, so a reporter who reports "upgrading fixed it" is not
pointing at this gate — re-grep both symbols in the new tree before crediting an upgrade;
(d) the support status is the upstream FIXME, not the plugin's comment.

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

`107030` is `ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT`, "stream is captured, not support
current oper" (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:51`). That
definition is stronger than the ordering heuristic: the code can only be minted **while a
stream is in capture**, so an `rtMemcpy … the current capture mode does not support this
operation` / `synchronized memcpy failed, kind = 1, runtime result = 107030` record is
never the *replay* mechanism, however the reporter words it ("an rtMemcpy inside graph
replay is illegal"). Treat it as a capture-window call that did a blocking copy
(`.item()`, `.cpu()`, a host-visible scalar) — a defect of its own when it is in this
process (#6002, #8502) — and take the replay-time death from the other records in the same
block (`507011`, `fftplus aivector error`, `The DDR address of the MTE instruction is out of
range`).

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

**On that same ref the MLA update function is asymmetric between its two branches
— read the pairing before any stream or blocking-copy theory.** `update_mla_attn_params`
(`compilation/acl_graph.py:252-307`) calls `torch.npu.graph_task_update_begin(
update_stream, handle)` only inside the **non-MTP** `else` (`:280-283`), while
`torch.npu.graph_task_update_end(update_stream)` is unconditional (`:305`) — so a
`deepseek_mtp` launch (`:270`) reaches end-without-begin, and the reporter's last
frame is that very line: `RuntimeError: graph_task_update_end … AclmdlRICaptureTaskUpdateEnd(stream),
error code is 107033` / `EE1001 … Unable to end update tasks because the stream is
busy` / `rtsStreamEndTaskUpdate … [task group status error]`. Recorded #4208
(v0.11.0rc1 `c5fe179`, 910C single node, DS-V3.1-w8a8 + MTP + 整图/`FULL_DECODE_ONLY`,
dying at startup in `execute_dummy_batch` → `_generate_dummy_run_hidden_states`,
`worker/model_runner_v1.py:2325`; the serving path calls the same function at
`:1558-1564`, so skipping the dummy batch only moves the crash to the first decode).
The handle is not the missing piece — the capture side runs `graph_task_group_begin/end`
and appends to `graph_params.handles` for spec decoding too
(`attention/mla_v1.py:1062-1071`), so the MTP branch *can* begin. Do not discard the
sibling reading: the same paste carries `rtMemcpy … kind = 1 … 107030` *inside*
`rtsStreamEndTaskUpdate`, and a Python list where the op wants a tensor
(`actual_seq_lengths = [spec_multiple * (i + 1) …]`, `:276-279`) is exactly what
mints a blocking host→device copy in that window. So rank the pairing first (one
read, and the paste's line number says which branch ran), the capture-window copy
second, and split them with two ablations on the same traffic: drop the full-graph
flag (MTP kept, PIECEWISE), then drop `--speculative-config` (full graph, no MTP).
Neither is a fix — each removes one term of the intersection.

**That intersection has no CI coverage, and the test names say otherwise.** At the
same ref `tests/e2e/singlecard/spec_decode_v1/test_v1_mtp_correctness.py:99-110`
defines `test_mtp{1,2}_correctness_full_graph` passing `CUDAGraphMode.FULL`, and the
file *is* wired (`.github/workflows/_e2e_test.yaml:107`) — but the plugin's selector
branches on `NONE` / `PIECEWISE` / `FULL_DECODE_ONLY` (`FULL_AND_PIECEWISE` is
rewritten to `PIECEWISE` first) and sends everything else to its `else`:
`… cudagraph_mode is not support on NPU. falling back to NONE` (`platform.py:205-242`),
so those two cases run eager. A green run there says nothing about
`FULL_DECODE_ONLY` × MTP. The rest of that ref's graph coverage is the same shape:
the wired ACL-graph runs were `PIECEWISE` at `max_model_len=1024` on
Qwen3-0.6B / DeepSeek-V2-Lite / Qwen3-30B-A3B, so no VL, 72B or 16 K case ran
under any full mode. That `else` also dates the config: a paste which reached this
update ran `FULL_DECODE_ONLY`, not `FULL`, because `cudagraph_runtime_mode ==
CUDAGraphMode.FULL` comes only from that mode's decode half.

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
flag by flag — the deltas are the discriminators, and a match is *a* verdict, not the end:
**diff the tutorial the reporter copied against that CI config as well**, because the two can
disagree and the tutorial side is the one nothing runs (next paragraph).

```bash
rg -ln -i 'glm-5|glm5' tests/e2e/*/*/configs/ tests/e2e/*/*/models/ 2>/dev/null
sed -n '1,60p' tests/e2e/weekly/single_node/configs/GLM-5_1-W8A8_A3_weekly.yaml
```

**When the reporter's command matches the tutorial and the tutorial is the deviating side.**
Verified at vllm-ascend v0.19.1rc1 (`da421af`): the A3 `glm-5-w4a8` block of
`docs/source/tutorials/models/GLM5.md` ships
`--additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true,
"ascend_compilation_config": {"enable_npugraph_ex": true}}'` together with
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` and
`--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'`, while
`tests/e2e/nightly/single_node/models/configs/GLM-5.yaml:34` runs the same model on the same
TP16/EP/MTP-n3/FULL+`npugraph_ex` shape with **`multistream_overlap_shared_expert: false`** (and
tightens the graph buckets at `:76`, `VLLM_ASCEND_BALANCE_SCHEDULING: "0"` at `:12`,
`--max-model-len 16384`), wired at `.github/workflows/schedule_nightly_test_a3.yaml:283`. So
"your command is the documented one" is not a refutation: the documented flag is the path the
nightly never exercises, and the first rung is to run the CI's value (recorded #9263, #8502 where
the reporter turned the same flag off and the crash stopped). Check the pairing this way
whenever the symptom is a graph-mode / shared-expert-stream copy fault — one `rg -n` of the flag
in `docs/source/tutorials/models/<model>.md` and in `tests/e2e/**/configs/`, and quote both
line numbers.

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

#8587 — vllm-ascend **v0.13.0** (`6281c120`, the reporter's exact pair with vllm
`72506c98`), 2×A2 × 8 cards, DeepSeek-V3.2 w4a8 QuaRot, `--data-parallel-size 2
--data-parallel-size-local 1 --tensor-parallel-size 8 --enable-expert-parallel`,
`--max-num-seqs 16 --max-model-len 65536 --max-num-batched-tokens 4096`,
`FULL_DECODE_ONLY`, `--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'`,
`deepseek_mtp` n=3, `VLLM_ASCEND_ENABLE_MLAPO=1`; all eight node-1 workers die together at
**concurrency 4** in `vllm bench serve` (c=1 and c=2 clean), and removing the
`--compilation-config` removes it. The paste carries the `EE9999` 107030 block plus
`The DDR address of the MTE instruction is out of range` and a **reconstructed** call stack
(`sample_tokens → propose_draft_token_ids → mtp_proposer._propose → deepseek_mtp forward →
acl_graph.__call__ → aclnnKvRmsNormRopeCache`) — no plog, no `fault kernel_name=`, no first
`[E…]` — that ask was the right one, and the reporter's later comment supplied complete
node0/node1 logs. Read it as §1/§7: the 107030 record is the capture window, the MTE-DDR line is the
replay death, and the reconstructed op is `torch_npu.npu_kv_rmsnorm_rope_cache`
(`sfa_v1.py:527-549` at that pin), the DSA KV scatter. The header gate above *is* the answer
to the reporter's ask ("raise a clear error at startup if this is unsupported"): upstream
already does, and `platform.py` removes it. Outcome, attribution still open: the reporter's
later comment confirms **v0.18.0 *and* dropping `layer_sharding`** fixed it — two variables
at once — and the `enforce_eager` override is still present at v0.18.0rc1, so the upgrade
cannot be credited to that gate. `layer_sharding` gets no cross-check against graph mode at
either pin: its only validation is the FLASHCOMM2-conditional one that demands `["o_proj"]`
alone (`utils.py:1020-1029` at v0.13.0, `:1030-1037` at v0.18.0rc1 ⇒ `Found invalid
layer_sharding`), so a plain launch with `["q_b_proj", "o_proj"]` is accepted while
`mla_v1.py:751-759` registers both layers into the shard-weight series. Cheapest
single-axis test, therefore, is dropping `--additional-config layer_sharding` on the release
the reporter already has; say that instead of a source walk.

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

#9263 — image `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3` (`da421af`), GLM-5-w4a8 on one A3
(16 cards), TP16 + EP, `FULL_DECODE_ONLY` + `deepseek_mtp` n=3 + `enable_npugraph_ex` +
`multistream_overlap_shared_expert: true` + `--async-scheduling`, 200 K ctx, and **the first
curl request** kills it. The attached screenshot's OCR is the whole evidence (there is no
`collect_env` in the report): `torch_npu/npu/streams.py:187 in synchronize` →
`aclrtSynchronizeEvent(event_)`, `[PID: 755, Device: 8, RankID: -1] … Model execution failed`,
`EE9999 … Inner error code is 507011`, and ~20 s earlier by the OCR's two stamps
`EE9999 … rtMemcpy execution failed, reason=the current capture mode does not support this
operation[LINE:61]` /
`synchronized memcpy failed, kind = 1` / `rtEventSynchronize … reason=the model stream execute
failed` / `wait event to be complete failed`. Read it as §2/§7: the Event sync names the model
stream's death, so the owner is whatever killed that stream, and the `kind = 1` record is the
first-request replay window. Two things settle it without a source walk — the recipe diff of
§7 (the reporter's command is the *tutorial's*, and the tutorial is the side with
`multistream_overlap_shared_expert: true`; the nightly runs `false`, reporter-confirmed fix on
#8502) and, second, the `--async-scheduling` + spec-decode `Event` readback
(`worker/model_runner_v1.py:957-968`, `:2387` at that pin). Do not oversell that second row:
`rg -n 'torch\.npu\.Event\(' vllm_ascend/` returns 13 sites at that pin, among them the
sampler's `async_exponential_event` (`sample/sampler.py:68`) and
`model_runner_v1.py:1862 sampling_done_event`, so an `Event.synchronize` frame in the paste
identifies **no** particular Event — the `--async-scheduling` / `--speculative-config` ablations
pick the owner, and the frame only says a host wait surfaced the stream's death.

Delivery note on the same run: 21 assistant turns carried a tool call and **no message text**,
so the ranked answer it finally wrote was cut at the 16 384-token cap inside the "please send
back" list — the first three ranks and the evidence table survived, the artefacts and scope did
not (rule 1).

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

### The same signature at v0.23.0: arity 13 → 22, and a third argument creates a new divergence

Verified at vllm-ascend **v0.23.0** (`5cb98caa`) — the #8370 shape recurred with a different
number and a different trigger, so count both numbers at the reported pin instead of
reusing the v0.18.0rc1 ones:

| Anchor (v0.23.0) | Fact |
|---|---|
| `attention/utils.py:104-120` | `using_paged_attention(runtime_shape, vllm_config, head_size=None)` gained a third argument; `:109-113` is a **Gemma4 fallback** — `if head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE: return True` (`= 512`, `device/utils.py:23`), returned **before** the `cudagraph_mode`/`pa_shape_list` test |
| `attention_v1.py:1471` | forward/capture dispatch passes `self.head_size`, so a 512-dim global head layer takes PA in **every** mode |
| `attention_v1.py:464`, `worker/model_runner_v1.py:3502` | `update_graph_params` (and the runner's own call) evaluate the predicate **without** `head_size` |
| `attention_v1.py:470-489` | PA branch unpacks **9** fields (`query, key_cache, value_cache, num_kv_heads, num_heads, scale, block_table, seq_lens, output`) |
| `attention_v1.py:715-738` | FIA branch unpacks **22** fields (ends `c8_*_scale/offset`, `layer_name`) — `:715` is the line the ValueError reports |
| `ascend_config.py:246` | `pa_shape_list` default `[]` |

So a hybrid stack with a 512-dim global head (Gemma4-family `gemma4_unified`) captures the
9-tuple for those layers and, under `FULL_AND_PIECEWISE`/`FULL`, is unpacked as 22 on the
first decode — deterministic, mode-independent, and reported as `expected 22, got 9`. Under
`FULL_DECODE_ONLY` the ladder above works verbatim (`pa_shape_list` = the captured sizes makes
the update side take PA too); under `FULL_AND_PIECEWISE` no config reaches that branch
(`:117-118` returns False before the list is read), so the honest answer is `--enforce-eager`
plus the two-line source fix (pass `self.head_size` at update, or put the branch in the
captured tuple). Coverage half, verified at the same pin: `release_notes.md:170` "Added
experimental Gemma4 support on A2 and Ascend 950", while `support_matrix/supported_models.md`
lists only `Gemma-2` (`:84`) and `Gemma-3` (`:85`, `:185`) — no Gemma-4 row, and no ACL-graph
e2e for the family. Recheck by listing every `using_paged_attention(` call site and counting
the unpack targets in each branch: both moved between v0.18.0rc1 and v0.23.0.

## 10. A host-side `RuntimeError` at the first decode: the DP-wide token count meets `max_num_reqs`

Signature (recorded #5076, vllm-ascend v0.12.0rc1 `42ceaf0` + vllm v0.12.0 `4fd9d6a`;
`FULL_DECODE_ONLY`, `--data-parallel-size 4 --data-parallel-size-local 2`,
`--tensor-parallel-size 8`, `--enable-expert-parallel`, `--max-num-seqs 64`,
`--max-num-batched-tokens 8192`, `--async-scheduling`, chunked prefill, 7168-token
prompts at concurrency 200): every rank of one DP group raises, in the same second,

```text
RuntimeError: The expanded size of the tensor (63) must match the existing size (6151) at
non-singleton dimension 0.  Target sizes: [63].  Tensor sizes: [6151]
```

at `vllm_ascend/worker/model_runner_v1.py:1035`, stack
`multiproc_executor.worker_busy_loop → worker_v1.execute_model → model_runner_v1.execute_model (1484) → _prepare_inputs (1035)`,
and the API server then reports `EngineDeadError`. There is no `ERR`/`EZ` code and no
plog: this is a Python shape error, so no device, tiling or card lane applies.

**The two numbers decode.** `N` is the *clamped* slice length and `M` the length of
`fill_values`:

- `:202` `self.max_num_reqs = scheduler_config.max_num_seqs` = 64 in that launch;
- `:392` `self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, …)`, so the
  buffer holds 65 slots and `gpu[num_reqs + 1 : num_reqs_padded + 1]` truncates at 65;
- with `num_reqs = 1`: `65 - 2 = 63 = N` ✓ and `M = pad_size = num_reqs_padded - 1 =
  6151`, i.e. `num_reqs_padded = num_input_tokens // uniform_decode_query_len = 6152`
  with `uniform_decode_query_len = 1`.

**Where 6152 comes from** is the mechanism, one line above:

| Anchor (v0.12.0rc1) | Fact |
|---|---|
| `:617-632` | local `num_input_tokens` = `pad_for_cudagraph(total_num_scheduled_tokens)` while the local total fits `cudagraph_batch_sizes[-1]`, else the raw total (eager / above the largest capture), else TP-rounded under SP |
| `:650-655` | `num_input_tokens = maybe_padded_num_tokens`, i.e. the **maximum over DP ranks** returned by `_sync_metadata_across_dp` (`:448-505`, `max_tokens_across_dp`); the skip path (`_skip_all_reduce_acorss_dp_group`, MoE + `recompute_scheduler_enable` + MC2) is off by default |
| `:1016-1039` | `if cudagraph_mode.decode_mode() == FULL and uniform_decode:` → `num_reqs_padded = num_input_tokens // uniform_decode_query_len`, then the slice write at `:1035` and `num_reqs = num_reqs_padded` |
| upstream `vllm/v1/worker/gpu_model_runner.py:2761-2768`, `:2939-2941` | upstream computes `uniform_decode` from the *padded-by-SP* token count and takes `num_reqs_padded = batch_desc.num_reqs` from the graph descriptor **after** `coordinate_batch_across_dp` — a capture-time constant ≤ `max_num_reqs`, never a division on the DP-padded count |

So the crash needs **both** a locally uniform decode batch (all requests at one token,
else the block is skipped) **and** a sibling DP rank whose step carries more tokens than
`max_num_seqs` — the first prefill chunk in the group does it. That is why `PIECEWISE`
never shows the line, why the failure arrives "when concurrency is large" rather than on
the first request, and why no single rank's log explains it: the DP-wide max is the
input. Nothing about the model, the quantization or the cards is involved.

Ladder: drop the `--compilation-config` (rung 1; the block is `decode_mode() == FULL`
only) → `--enforce-eager` → if it must be reported, the two numbers plus
`--max-num-seqs`, `--max-num-batched-tokens` and the DP/TP layout are the artefacts,
and §0's release row says whether the reported release already lists the combination as
broken. A per-rank `num_reqs` / `num_input_tokens` print is the confirming log, not a
prerequisite for the verdict.
