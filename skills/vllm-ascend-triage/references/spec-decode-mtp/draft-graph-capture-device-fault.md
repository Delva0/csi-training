# Device fault under MTP + a captured decode graph (#10700)

## Signature

The server is up and has answered requests, then one step dies with a device dump;
there is no bad flag, no OOM and no collective error in the paste:

- traceback frames, in order: `vllm/v1/executor/multiproc_executor.py
  worker_busy_loop` → `vllm_ascend/worker/worker.py execute_dummy_batch` →
  `vllm_ascend/worker/model_runner_v1.py _dummy_run` → `_model_forward` →
  `vllm_ascend/compilation/acl_graph.py __call__` →
  `torch.npu.current_stream().synchronize()` → `507011`, then
  `EngineCore encountered a fatal error`, then `EngineDeadError` in both API
  servers;
- the `EE9999` block underneath carries `fftsplus aicore/aivector error …
  error code = 0x800000 … The DDR address of the MTE instruction is out of range`,
  `aic error mask: 0x6500020bd00028c`, `synchronized memcpy failed, kind = 1,
  runtime result = 107030`, and an `rtMemcpy execution failed, reason=the current
  capture mode does not support this operation` record;
- the launch is MTP (`--speculative-config … "method": "mtp"|"deepseek_mtp"`) with
  `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` and, since
  v0.21.0, `npugraph_ex` on by default;
- every rank reports the same `pc start` / `aic error mask` within one or two
  seconds, and the ranks' inner `EE9999` records are microseconds apart → one
  software address/shape problem, not several cards
  (`../hccl-and-hardware/error-report-timestamps.md` step 5).

Recorded #10700: v0.21.0rc1 `80610e4` (+ vllm v0.21.0 `ad7125a`), 2× Atlas 800I-A2,
GLM-5.1, DP2 × TP8 + EP, `--max-num-seqs 8 --max-model-len 200000`,
`--async-scheduling --enable-prefix-caching`, `num_speculative_tokens: 3`,
`FULL_DECODE_ONLY`, `enable_npugraph_ex` (passed flat, i.e. inert — the default is
already `True`). It recurred unchanged on v0.22.1rc1, so do not answer "upgrade".

## What the frame proves, and what it does not

- **Proves:** the engine was executing a **DP dummy pass**, not a user request.
  Upstream `vllm/v1/engine/core.py:1815-1821` calls `execute_dummy_batch()` when the
  local rank had no ready request while the DP group was running, so the crash
  arrives at an arbitrary moment, unrelated to the last `POST`. Do not read it as
  "a request killed it".
- **Proves:** a captured (or capture-adjacent) graph step was involved, because
  `acl_graph.__call__` synchronizes on every `FULL`-mode replay unless
  `enable_enpu` or the EAGLE-merge draft path (`acl_graph.py:267-270` at this pin).
  That same line bounds the claim: the sync runs on **every** such replay, so a
  record from init reported much later means either the first replay of the
  affected bucket/size happened only then, or the runtime re-printed the process's
  earliest device record. Say which one your evidence supports; the plog's
  `task_id` / `serial number` around the record decides it.
- **Does not prove which graph.** `_dummy_run` runs the target forward and then
  `self.drafter.dummy_run(...)` in the same pass
  (`model_runner_v1.py:3549` then `:3553`), while the traceback stops inside the
  target's `_model_forward`. So the paste cannot separate target graph from drafter
  graph; the reporter's A/B can (below).

## The two timestamps decide what "ran fine for a while" means

In #10700 the inner record is `2026-06-24-10:01:23` on every chip, inside the same
log's `init engine (profile, create kv cache, warmup model) took 144.14 s
(compilation: 52.37 s)` (printed `10:02:16`, capture finished `10:02:13`), while the
traceback's own prefix is `10:28:19`. On v0.22.1rc1 the same shape repeats
(`01:19:08` → `03:00:17`, ~101 min). So the device recorded the fault during engine
init and the sync point reported it much later; the service answering `200 OK` in
between is not evidence that the trigger was a request. Read
`../hccl-and-hardware/error-report-timestamps.md` ("A record from this process's own
init is not a stale replay") before calling the block a replay.

## Ranked candidates, one cheap check each

1. **MTP draft forward captured into the graph, replaying a capture-mode-hostile
   copy.** A sibling report on the same family (GLM-5.2-W8A8, 2× A2, DP2 × TP8 +
   EP16, `deepseek_mtp` 5, `FULL_DECODE_ONLY`, `async-scheduling`,
   `enable_npugraph_ex`) captured the full stack: `sample_tokens` →
   `propose_draft_token_ids` → `llm_base_proposer._propose` → `acl_graph.__call__`
   → `_run_merged_draft` → `deepseek_mtp.forward` → `npugraph_ex` →
   `aclnnAddRmsNormBias` fails with the same `0x800000` / MTE out-of-range. Check:
   startup line `[spec_decode/base] Wrapping draft model with ACLGraphWrapper:
   runtime_mode=FULL, …` plus the plog's `[AIC_INFO] fault kernel_name=`; if the
   kernel is `aclnnAddRmsNormBias`/`AddRmsNormBias`, this row owns it
   (`../custom-ops-runtime-faults/add-rms-norm-bias-op-map.md`, #8944 precedent).
2. **SFA / lightning-indexer reading past `block_table` at the context boundary**
   (MTP advances `seq_lens` by the draft length): precedent #9098, same GLM-5.1-w4a8
   family, same `507015`/MTE DDR wording. Check: does the crash need prompts near
   `--max-model-len` (200000 here)? Re-run at `--max-model-len 32768` with the same
   traffic; a clean run demotes this row.
3. **A graph bucket first replayed later than capture** (a captured size/descriptor
   whose constants were never validated during capture). Check: the plog's capture
   list against the `batch_size` in the failing step; and
   `--compilation-config '{"cudagraph_capture_sizes": [...]}'` restricted to the
   sizes the traffic actually uses.
4. **MC2 MoE dispatch/combine under `HCCL_OP_EXPANSION_MODE=AIV`** — only if the
   failing kernel is a MoE all-to-all: DP2 × TP8 = 16 is exactly the A2 MC2
   eligibility boundary (`../feature-enable-recipes/moe-expert-parallel-with-dp.md`).
   Check: unset `HCCL_OP_EXPANSION_MODE` for one run
   (`../hccl-and-hardware/aiv-mc2-env-ablation.md`).

## Ablations, cheapest first — with the cost the reporter measured

1. `--speculative-config '{"num_speculative_tokens":3,"method":"mtp","enforce_eager":true}'`
   — turns **only the draft** eager: `use_cuda_graph = runner._use_aclgraph() and
   not speculative_config.enforce_eager` (`llm_base_proposer.py:175`), so the target
   keeps `FULL_DECODE_ONLY`. Reporter-confirmed stable, **but several times slower**:
   the drafter graph carries much of MTP's decode speedup, so do not sell this rung
   as a small loss. (Global `--enforce-eager` is a different knob — it disables the
   whole compilation, `platform.py:483-489` logs `Compilation disabled, using eager
   mode by default` — and is the attribution run, not a workaround.)
2. `MTP off` (`--speculative-config` removed): confirms the whole leg; the sibling
   report's stable baseline.
3. One `num_speculative_tokens` step down, then `--async-scheduling` off — separate
   the draft-depth and scheduler-ordering axes from the graph axis.
4. `ASCEND_LAUNCH_BLOCKING=1` is **not available** while any graph mode is on
   (`ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1`,
   `platform.py:595`), so attribution here needs the plog, not blocking mode.

## Evidence table (vllm-ascend `80610e4` / vllm `ad7125a`)

| Fact | Location |
|---|---|
| dummy pass on a rank with no ready request | `vllm/v1/engine/core.py:1815-1821`, `:735-736` |
| the dummy pass runs target **and** drafter | `vllm_ascend/worker/model_runner_v1.py:3300`, `:3549`, `:3553-3558` |
| dummy batch is one decode-token-per-req uniform step | `vllm_ascend/worker/worker.py:832-834` |
| `FULL` replay synchronizes on every call (unless `enable_enpu`/EAGLE draft) | `vllm_ascend/compilation/acl_graph.py:267-270` |
| spec-scoped `enforce_eager` gates **only** the drafter graph | `vllm_ascend/spec_decode/llm_base_proposer.py:175`, `:425-441` |
| global `enforce_eager` disables compilation | `vllm_ascend/platform.py:469`, `:483-489` |
| `FULL_DECODE_ONLY` keeps `npugraph_ex` on; PIECEWISE/NONE turn it off | `vllm_ascend/platform.py:550`, `:575`, `:586` |
| flat `enable_npugraph_ex` / `fuse_*` under `additional_config` is ignored (only the nested `ascend_compilation_config` dict is read) | `vllm_ascend/ascend_config.py:40`, `:500`, `:536`; `vllm_ascend/compilation/graph_fusion_pass_manager.py:59` |

## Scope and open items

- Traced at v0.21.0rc1 / vllm v0.21.0 and reproduced by the reporter on v0.22.1rc1;
  re-grep the graph gates after an upgrade, they move between releases.
- Unresolved in the record: the failing kernel name on the reporter's own run (no
  plog), whether the draft or the target graph was replaying, the CANN/torch_npu
  version, and whether the 200 K context boundary participates.
- "You deviated from the validated command" is **not** available as an answer for
  #10700: the tutorial's A2 multi-node GLM-5 block
  (`docs/source/tutorials/models/GLM5.md`, A2 tab) and the nightly
  `tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-A2-dual-nodes.yaml`
  disagree on four axes (`VLLM_ASCEND_BALANCE_SCHEDULING` 1 vs 0,
  `VLLM_ASCEND_ENABLE_FLASHCOMM1` on vs absent, prefix caching on vs
  `--no-enable-prefix-caching`, `--async-scheduling` present vs absent) and on
  `max-num-seqs` / `max-model-len`. List them as ablation rows; nothing there is a
  cause of a device address fault.
- Ask for, in this order: the plog covering **both** the init window and the report
  minute for one chip, with the `[AIC_INFO] fault kernel_name=` / `task_id` lines;
  the per-rank first `[E…]` line; the request bodies live in the report minute; and
  the result of the spec-scoped `enforce_eager` A/B already run.
