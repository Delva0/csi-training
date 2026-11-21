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

## Step 0 — turn the headline into a per-step cost, before any source read

One line of arithmetic separates the lanes and costs no tool call:

- `output_tokens_total / wall_time` and `ITL` are enough. `54 tok/s` aggregate at
  20 concurrent streams is per-request `1/0.352 = 2.8 tok/s`: the engine is
  emitting about one token per request per 0.35 s, i.e. **the decode step itself
  costs ~350 ms**, not a slow queue and not a slow prefill.
- The reporter's own metrics line fixes the batch axis:
  `Running: 1 reqs, Waiting: 0 reqs, … Avg generation throughput: 2.8 tokens/s`.
  A single running request whose generation throughput equals `1/ITL` rules out
  queueing, admission and proxy fan-out, and largely rules out
  `--max-num-batched-tokens` padding (a padded batch would show up as an
  aggregate rate, not as `2.8 tok/s` on one sequence). Say so in the answer and
  drop those lanes.
- Ask for `VLLM_LOGGING_LEVEL=DEBUG` on the **decode** node, one env var. The
  reporter of #6485 supplied exactly that and it settles the phase split (below).

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

## Cause A — the MoE communication path (source-verified at v0.14.0rc1)

`vllm_ascend/ascend_forward_context.py:204-272` (`select_moe_comm_method`): with
expert parallel on, `soc_version in {AscendDeviceType.A2}` takes MC2 **only if**
`num_tokens <= mc2_tokens_capacity` **and**
`world_size_across_dp / pipeline_parallel_size >= 16`; otherwise every MoE layer
runs `MoECommType.ALLGATHER` (`:239-247`). FUSED_MC2 is the **A3** branch
(`:249-261`, `VLLM_ASCEND_ENABLE_FUSED_MC2` + `quant_type == "w8a8_dynamic"`), so
on a 910B2/A2 node that env var is a dead lever — do not sell it as the fix.

So: `--tensor-parallel-size 8 --data-parallel-size 1 --enable-expert-parallel`
on one 8-card node gives world 8 → ALLGATHER everywhere, and the pinned project's
own recipe for the same model does the opposite on the decode node —
`tests/e2e/nightly/multi_node/config/Qwen3-235B-disagg-pd.yaml:65-80` runs
`--data-parallel-size 4 --tensor-parallel-size 4 --enable-expert-parallel`
(world 16 → MC2) with `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`
and `--async-scheduling`. Check the reporter's two flags against that row; the
cheapest action is a **non-PD TP8 baseline** on the same node (one serve command,
no proxy) — if that also runs at ~3 tok/s the deployment shape is exonerated and
the frame is the model/quant/card, if it is fast the decode node's parallel
layout is the lever.

## Cause B — async scheduling is on by default at 0.14.x (reporter-reported)

`vllm/config/vllm.py:588-628` (vllm v0.14.1, the pin paired with vllm-ascend
v0.14.0rc1): when `async_scheduling is None` (the field default,
`vllm/config/scheduler.py:133`) it is set **True** unless PP>1, a non-eagle
speculative method, `disable_padded_drafter_batch`, or an unsupported executor
backend. **KV transfer is not in that list**, and `:630-633` logs one
`Asynchronous scheduling is enabled.` line once. The #6485 reporter reported on
0.14.0rc1 that async scheduling is not compatible with PD disaggregation, and
that with `method: eagle3` the disable chain above leaves it on. Treat that as
the reporter's statement plus a source-verified reason it can be on at all, not
as a verified defect: the same pin's CI PD yaml passes `--async-scheduling` on
the decode node, so the combination is exercised somewhere.

Deciding check, one variable, same request body: read the decode node's
`Asynchronous scheduling is enabled|disabled` line out of its startup log; if it
is enabled, re-run with async scheduling off. Whether that ref exposes a CLI
negation at all is version-specific — grep the parser for `async_scheduling` at
the reported ref before telling the reporter which flag to pass.

## Also on the launch-diff list (pinned recipe, both sides)

`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`, `VLLM_ASCEND_ENABLE_FUSED_MC2=2` (A3 only, see
Cause A), `OMP_NUM_THREADS=1`, `HCCL_BUFFSIZE=1024`, `TASK_QUEUE_ENABLE=1`,
`HCCL_OP_EXPANSION_MODE=AIV`, a `--max-num-seqs` bound, and the reporter's
`OMP_NUM_THREADS=10` / `HCCL_BUFFSIZE=512` / `--max-num-batched-tokens 7168`
against the yaml's `8192`. None of these is established as the cause; they are
the ablation rows, one per launch.

## Deliverable

Verdict line with the arithmetic ("one running request, 2.8 tok/s, so each
decode step costs 350 ms — not queueing"), then ranked: (1) parallel layout vs
the pinned yaml, with the non-PD baseline as its check; (2) async-scheduling
state from the startup log; (3) the launch-diff rows. State plainly that the
pasted KV-transfer DEBUG numbers are the TTFT component and ask for the decode
node's `Asynchronous scheduling is` line, the startup config dump and one
non-PD baseline run. Verified at vllm-ascend v0.14.0rc1 `52d4acf` / vllm
v0.14.1 `d7de043`; #6485 left open, reporter's later comments cited above.
