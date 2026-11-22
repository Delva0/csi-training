# PD decode node: every token costs ~0.35 s, nothing errors

Symptom family: a P/D deployment on **v0.14.x** serves every request and returns
correct text, but the decode side is an order of magnitude slow — headline
`Average time per output token (s) 0.3519`, `Average inter-token latency 0.3518`,
`Output token throughput 54 tok/s` at 20 concurrency, "only 3 tok/s" per request,
with no traceback, no `WorkerProc` death and no timeout. Reporters phrase it as
"How do I use PD correctly in this version?" because nothing looks broken.
This page is the *serving-but-slow* sibling of the transport pages: if the peers
fail to connect, the output is garbled, or the prefiller logs a done-signal
expiry, use `transport-topology.md` / `llmdatadist-done-signal-and-ttft.md`
instead.

## Step 0 — two arithmetics that pick the shape, before any source read

- **Per-token cost.** `2000 out × 0.3519 s = 703.8 s`, and `latency 717.5 −
  TTFT 14.0 = 703.5 s`, so the whole latency is the decode leg;
  `40 000 tokens / 739.9 s wall = 54.06 tok/s` reproduces the headline. It is the
  step, not a slow queue and not a slow prefill.
- **Batch axis — this is the one that ranks the causes.** Those 20 requests show
  `Average latency 717.5 s` inside `739.9 s` of wall time, so **all 20 decoded at
  once** (sequential would be ~14 400 s): the ~350 ms buys one token for twenty
  streams, i.e. it is one step of the 20-stream batch.
- The decode node's `VLLM_LOGGING_LEVEL=DEBUG` window is a **different, one
  request** run (ask for it yourself — one env var) and prints `Running: 1 reqs,
  Waiting: 0 reqs, … Avg generation throughput: 2.8 tokens/s`: a batch of **one**
  pays the same ~350 ms.
- **So per-step wall time is flat from batch 1 to batch 20.** Nothing that scales
  with tokens — batch padding, `--max-num-batched-tokens`, the token axis of a MoE
  ALLGATHER/MC2 path — is shaped like that. Two shapes are left: a *fixed
  per-step wait*, or a decode path running one-to-two orders below the cards'
  weight-bandwidth floor (8×910B2 at 22B-active w8a8 should be tens of ms per
  step, not 350). Both are priced by the same two launches below, so rank them and
  move; and in the answer say which paste each number came from, because they are
  two runs.
- Two entries in that metrics block are **not** faults: `External prefix cache
  hit rate: 100.0%` is the connector's own counter
  (`vllm/v1/metrics/loggers.py:250-252`, printed only when a connector reports
  stats — on a decode node every prompt token arrives externally) and `GPU KV
  cache usage: 0.0%` is one or two 7 000-token requests in a `0.9 × 64 GB` pool.
- Ask for the decode node's **startup** log as well, not only the DEBUG window:
  it carries the single `Asynchronous scheduling is enabled|disabled.` line and
  the config dump, and the DEBUG window cannot decide Cause A.

## The DEBUG paste: what it proves and what it cannot be

Per request the decode engine prints `MooncakeConnector get_num_new_matched_tokens`,
`update_state_after_alloc`, per rank `start_load_kv … Num local_block_ids: 1`,
`KV cache transfer for request … took 132.15 ms (1 groups, 1 blocks)`,
`Received response … ACK`, `Finished recving KV transfer`, and in between ~30
repeats of `chatcmpl-… is still in WAITING_FOR_REMOTE_KVS state` with
`Number of completed KV cache send requests: 0`.

Read it in the right lane: that is ~1 s of **handoff** cost per request, once,
before its first token — TTFT, not TPOT. A 2 000-token stream at 0.352 s/token
cannot be explained by a 0.13-1 s connect/transfer per request, so do not spend
the session deriving `mooncake_connector.py`'s socket pool. Recorded #6485 did
exactly that for 24 tool turns and 57 K characters of reasoning and shipped no
message text; when a handoff number is finite and per-request, name it as the
TTFT component and move to the per-step causes.

## Cause A — async scheduling is on unless something turns it off (first check)

`vllm/config/scheduler.py:133` defaults `async_scheduling` to **None**, and
`vllm/config/vllm.py:588-628` then sets it **True** unless `pipeline_parallel_size
> 1`, the spec method is not an EAGLE/MTP one, `disable_padded_drafter_batch`, or
the executor backend is outside `mp`/`uni`/`external_launcher` (each disable
branch logs its own warning; `:628` is the `= True`, `:630-633` the one
`Asynchronous scheduling is …` line). **KV transfer is in neither list** — not
the explicit-enable hard-fail block (`:562-587`) and not the auto-disable chain —
so a PD launch whose command shows no `--async-scheduling` still runs it.

The reporter of #6485 reported on v0.14.0rc1 that this default is what makes PD
disaggregation slow and that the pair is not compatible. Note the scope: their
scripts set no `--speculative-config`, so the branch that leaves it on is the
plain `else: True` at `:628`, not the eagle3 case they quoted — **every** 0.14.x
launch that does not opt out, not only MTP/EAGLE serves.

Checks, one variable each, cheapest first:

1. Read that startup line (or `rg` the launcher for `--async-scheduling` /
   `--no-async-scheduling`: "not in the script" does not mean "not in play").
2. Relaunch with it off. The negation exists at this pin because the field is
   `bool | None`, so the CLI builder uses `BooleanOptionalAction`
   (`vllm/engine/arg_utils.py:288-290`; `--async-scheduling` is registered from
   that field at `:1127`): hand over **`--no-async-scheduling`** — do not send the
   reporter off to grep the parser, and do not write `--async-scheduling False`.
3. Same request body, same load, same client. If TPOT moves, that was it; if not,
   Cause B and the launch-diff rows are still open.

**Do not read the pinned CI PD yaml as exoneration.**
`tests/e2e/nightly/multi_node/config/Qwen3-235B-disagg-pd.yaml:80` runs
`--async-scheduling` on the decode node and the yaml has a perf case
(`:109-110` `baseline: 1, threshold: 0.97`), but `tools/aisbench.py:207-213`
asserts `output_throughput >= threshold * baseline` — for that row ≥ **0.97
aggregate tok/s**, an absolute floor that a deployment 50× slower than healthy
passes. Recorded #6485 read "the same pin's CI PD yaml passes it, so the
combination is exercised somewhere" as evidence against this cause and ranked the
MoE layout first; the reporter's own follow-up then named `async_scheduling`.
Presence in a CI recipe proves coverage, never cost (also
`../evidence-workspace/index.md`).

What is **not** established: the mechanism. Nothing at these pins names the
per-step wait async scheduling is supposed to add on a KV-consumer node, so keep
it a hypothesis in the wording and let the relaunch (not a source read) decide.

## Cause B — the MoE communication path: a real threshold, the wrong size for 350 ms

`vllm_ascend/ascend_forward_context.py:204-272` (`select_moe_comm_method`): with
expert parallel on, `soc_version in {AscendDeviceType.A2}` takes MC2 **only if**
`num_tokens <= mc2_tokens_capacity` **and**
`world_size_across_dp / pipeline_parallel_size >= 16`; otherwise every MoE layer
runs `MoECommType.ALLGATHER` (`:239-247`). FUSED_MC2 is the **A3** branch
(`:249-261`, `VLLM_ASCEND_ENABLE_FUSED_MC2` + `quant_type == "w8a8_dynamic"`), so
on a 910B2/A2 node that env var is a dead lever — do not sell it as the fix.

So `--tensor-parallel-size 8 --data-parallel-size 1 --enable-expert-parallel` on
one 8-card node gives world 8 → ALLGATHER everywhere, and the pinned project's own
recipe for the same model does the opposite on the decode node —
`tests/e2e/nightly/multi_node/config/Qwen3-235B-disagg-pd.yaml:65-80` runs
`--data-parallel-size 4 --tensor-parallel-size 4 --enable-expert-parallel`
(world 16 → MC2) with `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
and `--async-scheduling`.

Rank it as a **separate defect to fix**, not as the explanation of 0.35 s: a
batch-1 ALLGATHER MoE step of a 22B-active model is single-digit milliseconds, and
Step 0's batch-flat 350 ms is not the shape a comm-path change produces. The row
still matters for throughput, and the check is unchanged — quote the reporter's two
flags against the yaml row; the cheapest measurement is a **non-PD TP8 baseline**
on the same node (one serve command, no proxy): if that also runs at ~3 tok/s the
deployment shape and PD are exonerated and the frame is model/quant/card/launch,
if it is fast the decode node's launch is the lever.

## Also on the launch-diff list (pinned recipe, both sides)

`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, `VLLM_ASCEND_ENABLE_FUSED_MC2=2` (A3 only, see
Cause B), `OMP_NUM_THREADS=1`, `HCCL_BUFFSIZE=1024`, `TASK_QUEUE_ENABLE=1`,
`HCCL_OP_EXPANSION_MODE=AIV`, a `--max-num-seqs` bound, and the reporter's
`OMP_NUM_THREADS=10` / `HCCL_BUFFSIZE=512` / `--max-num-batched-tokens 7168`
against the yaml's `8192`. None of these is established as the cause; they are
the ablation rows, one per launch.

## Deliverable

Verdict line with the arithmetic — "20 streams × 2.7 tok/s inside one ~350 ms
step, and the same ~350 ms for a single request, so the cost is a fixed per-step
wait: not queueing, not prefill, not batch padding" — then ranked:

1. the decode node's `Asynchronous scheduling is …` line, then one relaunch with
   `--no-async-scheduling` (Cause A);
2. the non-PD TP8 baseline on the same node (one serve command, no proxy), which
   separates PD from a 50×-slow decode path on the same cards;
3. the decode node's parallel layout against the pinned `dp4/tp4` row;
4. the launch-diff rows.

State plainly that the pasted KV-transfer DEBUG numbers are the TTFT component,
and ask for the decode node's startup log plus config dump and one non-PD
baseline. Verified at vllm-ascend v0.14.0rc1 `52d4acf` / vllm v0.14.1 `d7de043`;
#6485 left open, reporter's later comments cited above.
