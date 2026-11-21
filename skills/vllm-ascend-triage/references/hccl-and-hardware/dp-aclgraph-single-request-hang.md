# One request at a time hangs under `--data-parallel-size` > 1 + ACL graph, concurrency is fine

Symptom family (recorded #3128, no answer shipped): Qwen3-235B-A22B on one node,
`-dp 8 -tp 2 --max-num-seqs 8 --no-enable-expert-parallel --block-size 128
--max-model-len 66560 --compilation-config '{"cudagraph_capture_sizes": [8,1]}'`,
`HCCL_OP_EXPANSION_MODE=AIV`, `VLLM_USE_V1=1`. The engine is up; **one** in-flight
request stalls at graph replay (在途重放时卡住) and the client gets the bare
`TimeoutError: RPC call to execute_model timed out` 300 s later, while several
concurrent requests are served normally. No per-rank log, no device text.

This is the branch of `rpc-watchdog-timeouts.md` where the trigger is visible in
the *shape of the traffic*, not in the log. Read that page first for the watchdog
classification and the artefacts; read this one before deriving any collective
shape arithmetic from source, because most of that arithmetic is already settled
and is the part that consumes sessions.

Not this page: **nothing** being served after one oversized request — tiny prompts
time out too and only a restart recovers, with no watchdog frame anywhere — is the
KV admission wedge, `../scheduler-admission/waiting-queue-wedge.md`.

## What "concurrency is fine" actually selects

With DP > 1 every engine has its own scheduler, so a single request leaves the
other DP ranks with **zero** scheduled tokens for that step. At vllm-ascend
**v0.10.2rc1 (`048bfd5`)** + vllm **v0.9.1 (`b6553be`)** — the pins this page was
verified at — those two populations take different code:

| Rank state | Entry | Mode |
|---|---|---|
| ≥1 scheduled token | `execute_model` (`worker/model_runner_v1.py:1710`), pads to a capture bucket (`:1103-1109`) | ACL graph replay of that bucket |
| 0 scheduled tokens (`execute_model` early-returns `EMPTY_MODEL_RUNNER_OUTPUT`, `:1721-1725`), so the DP engine calls `execute_dummy_batch` (`worker/worker_v1.py:315-316`) → `_dummy_run(1)` | `_dummy_run` (`:2081`) | **eager** — its `aclgraph_runtime_mode` default is `CUDAGraphMode.NONE` (`:2086`) |

The upstream caller is `vllm/v1/engine/core.py:857-869`: a DP engine core whose
step executed nothing but whose group is still `engines_running` calls
`execute_dummy_batch()` — exactly the partial-load case. So in the failing case
exactly one rank replays a captured graph while every peer runs an eager 1-token
forward. That asymmetry is real and is the reason the
symptom needs both DP and graph; it is *not* by itself a defect, and the two
attractive follow-ups below are already refuted at this pin.

## Settled: the DP-wide pad agrees, and the idle rank does join the sync

Both paths call the same sync exactly **once per step** — `_dummy_run`'s first
statement (`:2100-2101`), and on the busy rank inside `_prepare_inputs`
(`:1129-1134`):

- `_sync_metadata_across_dp` (`model_runner_v1.py:696-738`) is one
  `dist.all_reduce` over `get_dp_group().device_group` of an int32 tensor holding
  the per-rank token counts plus two flags, returning the group **max** and
  `num_tokens_after_padding = [max] * dp_size`. `dp_size == 1` returns early
  (`:705-706`), and the comment at `:699-704` is explicit that with DP > 1 it is
  **not** skipped even in eager mode.
- the busy rank passes `pad_for_cudagraph(total_num_scheduled_tokens)`
  (`:1103-1109`); the idle rank passes the literal `1` (`worker_v1.py:316`).
  Both then read the same group max, so the per-rank contribution to every DP
  collective in that step is identical.
- the MoE-side DP collectives (`ops/fused_moe.py:454-470`): with `dp_size > 1`
  and state `AllGather`, `hidden_states` (and `router_logits`) are padded to
  `forward_context.max_tokens_across_dp` and then
  `get_dp_group().all_gather(..., 0)`; state `NaiveMulticast` (prefill) broadcasts
  slices from `cu_tokens_across_dp_cpu` instead.
- **`--no-enable-expert-parallel` does not remove those collectives.**
  `_get_fused_moe_state` (`ascend_forward_context.py:26-42`) maps `ep_size == 1`
  to `NaiveMulticast` (prefill) / `AllGather` (decode) — both cross-DP; EP only
  changes which branch runs. So "EP off, therefore no DP collective" is wrong at
  this pin, and an EP-off ablation cannot exonerate DP either
  (`../feature-enable-recipes/dp-moe-padding-and-owner-slice.md` has the full
  selector table and the per-phase pad vector).

Consequence: *"the idle rank skipped the DP all_reduce"* and *"the pad target
differs by construction"* are refuted. Do not spend turns rebuilding them; the
pad-target derivation, the phase (`with_prefill`) trap and the owner-slice
convention live in `../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`.

## What is left, and the one that the traffic shape points at first

`cudagraph_capture_sizes: [8, 1]` gives this pad map (vllm `config.py:4146-4159`
builds it; `pad_for_cudagraph` is `:4335-4340`): **1 → 1**, 2-7 → 8, 8 → 8. So a
single decode request replays the **size-1 bucket** — the bucket that the
concurrency case never uses — while eight tokens replay the size-8 bucket that is
known good on the same deployment. `scripts/dp-single-request-bucket.sh` prints
the map and the per-step branch for the reported command.

Ranked candidates:

1. **The size-1 bucket / the eager-vs-graph step asymmetry** (hypothesis, not
   source-verified): the 1-token capture, or a captured op whose peers issue it
   eagerly, does not return. Cheapest discriminator: drop `1` from the list
   (`'{"cudagraph_capture_sizes": [8]}'`) so a single request pads to 8 and
   replays the bucket that works.
2. **Graph mode in general** (rung before any DP theory): re-run with
   `--enforce-eager` on the same DP/TP. Pass ⇒ the graph path owns it, and the
   lane becomes `../startup-failures/acl-graph-capture.md`'s replay appends, not
   a collective mismatch.
3. **DP coordination** (only if 1 and 2 both still hang): `--data-parallel-size 1`
   with the same world size and graph config; pass ⇒ the DP/bucket pairing owns
   it, and the per-rank `dp_rank`/pad-target print from the owner-slice page is
   the next artefact.
4. **Env knobs exercised only on this path** (cheap, one variable per launch):
   `HCCL_OP_EXPANSION_MODE` (AIV), `CPU_AFFINITY_CONF` + `VLLM_CPU_BINDING`
   (CPU-affinity pinning is the only thing here that can starve a rank),
   `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`. The AIV *provenance* family
   is `aiv-mc2-env-ablation.md`: note that with `ep_size == 1` the MC2 dispatch
   is not the selected path (`ascend_forward_context.py:26-42`), so AIV here
   would act on the plain DP/Tensor all-gather, not on `MoeDistribute*`.
5. **A device-side stall on one rank** — only if the plog has an in-window
   timeout/`EZ9999` for the failing minute (`error-report-timestamps.md`).

`ASCEND_LAUNCH_BLOCKING=1` needs `cudagraph_mode == NONE` (it raises
`ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1`,
`platform.py` `check_and_update_config`), so it is a *second* run after
`--enforce-eager`, never a replacement.

## The deliverable for this shape

The paste cannot name the owner: there is no rank prefix, no op, no per-rank log.
Ship the classification (watchdog, not HCCL; trigger = DP > 1 + one-request
traffic) plus the ordered ablation above, each row one launch, and name what to
send back: the full per-rank log, which rank's `execute_model` timed out
(EngineCore's fatal line carries it), the plog for that minute, `npu-smi info`
taken while it is still hung, and the resolved capture-size echo (at this pin the
capture banner is a `tqdm` description, `Capturing ACL graphs (<mode>, <uniform>)`,
`model_runner_v1.py:2873`, so take the bucket list from the startup config echo or
a `../startup-failures/graph-mode-not-applied.md` grep rather than from the banner). That is a complete answer; a
derived-but-unproven collective mismatch is not worth a session
(`rpc-watchdog-timeouts.md`'s own warning).

## Scope

Verified at vllm-ascend v0.10.2rc1 / vllm v0.9.1 (case #3128); the upstream tree
is a dated reference for the reporter's vllm v0.10.2rc1, so re-grep
`pad_for_cudagraph` / `bs_to_padded_graph_size` before quoting the pad map at
another ref. `execute_dummy_batch`, `_dummy_run`'s default runtime mode,
`_sync_metadata_across_dp` and the `fused_moe.py` DP branch have kept their shape
into v0.11.0rc0 (`00ba071`), the pin of case #3314.
