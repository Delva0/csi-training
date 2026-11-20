# Qwen3-Next-80B-A3B at ~5 tokens/s: single-stream decode on the documented command

## When this page applies

A hybrid GDN MoE (Qwen3-Next-80B-A3B and siblings) at TP=4 with `Running: 1`
and `Avg generation throughput` in the low single digits — recorded 3.6-4.7 tok/s
across the sampled windows at v0.11.0, 5.3 tok/s at v0.11.0rc2. The report is
either a bare "太慢 / too slow" with one metrics paste (#4357) or the performance
half of a two-part issue whose failing half is a Triton kernel compile (#5750 →
`../image-dependency-faults/triton-ascend-pair.md`).

Read `index.md` first for the metric rules; this page is the *version* lever for
this model, and it exists because the usual "diff the command against the
tutorial" answer does not apply here.

## Arithmetic that decides the shape (do this before any source)

From the #4357 paste: `Running: 1`, `Waiting: 0`, `GPU KV cache usage: 0.5%`,
no `Preemptions` line, `Avg prompt throughput: 0.0` in the sampled windows.

- 5.3 tok/s ⇒ **~190 ms per decode step**, single request, no prefill competing.
  Queueing, capacity, `--gpu-memory-utilization`, `--max-num-seqs`, KV tokens and
  `--max-model-len` are therefore all out — say so and move on.
- `Loading model weights took 37.4525 GB` per card at TP=4 (bf16, ~150 GB total);
  a 3B-active MoE reads ~1.5 GB/card/token ⇒ **~1 ms at ~1.6 TB/s**.
- Two orders of magnitude of headroom means the step is **host-dispatch/latency
  bound**, not compute- or bandwidth-bound: that is the class of cost a captured
  graph removes, and nothing else on the launch line does.

## What the pinned version says (vllm-ascend v0.11.0rc2 `a2e4c3f`, vllm v0.11.0)

| Site | Fact |
|---|---|
| `docs/source/tutorials/multi_npu_qwen3_next.md:89` | the project's **own** serve command at this ref is `--tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.7 --enforce-eager`; the offline example repeats `enforce_eager=True` (`:134`) |
| `vllm_ascend/platform.py:151-153` | `if enforce_eager:` → `Compilation disabled, using eager mode by default` (the paste's `[platform.py:152]`) and `level = NO_COMPILATION` |
| `vllm_ascend/platform.py:207-208` | `FULL_AND_PIECEWISE` (which the upstream hybrid config auto-sets, "in order to optimize performance") is **downgraded to `PIECEWISE`** — "Full graph is fully supported later" |
| `vllm_ascend/platform.py:213-215` | the success line to look for once eager is dropped: `PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode` |
| `vllm_ascend/worker/model_runner_v1.py:591-592` | `_use_aclgraph()` needs `cudagraph_mode != NONE and level == PIECEWISE and not enforce_eager` |
| `vllm_ascend/patch/worker/patch_triton.py:16` → `vllm_ascend/ops/fla.py:205` | prefill `chunk_gated_delta_rule` is rebound to `torch_chunk_gated_delta_rule`, a pure-torch fp32 chunk loop (`chunk_size=64`) — prefill/TTFT is dispatch-bound too |
| `vllm_ascend/ops/sigmoid_gating.py:45-46`, launched `:214` | decode keeps a Triton `fused_recurrent_gated_delta_rule_fwd_kernel` |
| `tests/` | **no** `qwen3_next` reference at this ref — there is no CI baseline the reporter could be compared against |

Consequence: the reporter's command is not a misconfiguration, and a config diff
against this release's tutorial finds nothing. Nor is a published number being
missed — the tutorial carries no throughput table. State that, and put the version
ladder in the answer instead of a flag hunt.

## Ranked answer for this shape

1. **Move the release, and use the newer tutorial command.** At v0.14.0rc1 the
   tutorial's Qwen3-Next command (quoted in `index.md`) has no `--enforce-eager`
   and carries `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` plus
   `--max-num-batched-tokens 4096` — i.e. the graph path the v0.11 line cannot
   reach (`platform.py:207-208`) plus a FLA path that is not the torch loop. This
   is also the move #5750's maintainer answer prescribes for the same model.
2. **Dropping `--enforce-eager` on the old release is a hypothesis, not a fix.**
   It is the only in-version lever that touches the dominant term, so keep it as
   rung 1 with a named confirmation (the `PIECEWISE compilation enabled on NPU`
   line at `platform.py:213-215`) and a re-measure. Do not promise a number:
   #4357's reporter answered "并没有" two days after opening, and nothing in the
   recorded thread confirmed a speedup from that flag at v0.11.0rc2.
3. **Then the generic levers**, in the order of the arithmetic: a multi-prompt
   benchmark (`--num-prompts 8+`) for aggregate throughput, since single-stream
   decode is the worst case for a MoE at TP=4; a quantized W8A8 checkpoint if one
   exists for the model; expert-parallel or TP layout only after a measurement.
4. **Explicitly refuted by the paste**, so they do not belong in the answer as
   candidates: KV capacity/`--gpu-memory-utilization`, `--max-num-seqs`,
   `--max-model-len`, and `Prefix cache hit rate: 0.0%` (hybrid models are
   force-disabled in the config layer at this line —
   `zero-prefix-cache-hit-hybrid.md`).

## Paste hygiene for this track

- The reporter's own extras are not the cause but are worth one A/B line:
  `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32` appears in #4357 and in no
  tutorial; `--max-model-len 32768` vs the tutorial's 4096 only moves KV size.
- **Confirm the metrics belong to the reported model.** #4357's log carries
  `served_model_name: Qwen3-235B-A22B-Instruct-2507` and a request for
  `Qwen3-Next-80B` that 404s, while the issue's command says
  `--served-model-name Qwen3-Next-80B-A3B-Instruct`: the throughput samples came
  from an instance whose served name differs from the reported command. Say it in
  one line and ask which invocation produced the numbers.
- A hand-built image that adds a bisheng toolkit and a triton wheel over the stock
  image (both recorded reports did) is a second variable on the same run — the
  unmodified image is the isolation step, and the compile-failure page is where
  that thread belongs.
