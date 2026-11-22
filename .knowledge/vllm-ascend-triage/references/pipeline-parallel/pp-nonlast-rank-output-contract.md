# PP>1, clean start, first request dies on the non-last rank on `IntermediateTensors`

## Signature (recorded #2974, vllm-ascend v0.10.2rc1 `048bfd5`, vllm 0.10.2)

`vllm serve <ckpt> --quantization ascend --pipeline-parallel-size 2 …` starts
normally, the first `/v1/completions` returns `200`, and the worker log shows, on
**`Worker_PP0` only**, repeated blocks of:

```
(Worker_PP0 …) INFO [acl_graph.py:184] Replaying aclgraph
(Worker_PP0 …) ERROR [multiproc_executor.py:654]   File ".../vllm_ascend/worker/worker_v1.py", line 218, in execute_model
(Worker_PP0 …) ERROR [multiproc_executor.py:654]     assert isinstance(output, IntermediateTensors)
(Worker_PP0 …) ERROR [multiproc_executor.py:654] AssertionError
```

Bare `AssertionError` (no message), no device error code, no HCCL line, one rank
of the pair. `Worker_PP1` logs `Replaying aclgraph` too and reports nothing.

## What the paste already proves

- **The PP contract is the thing that broke, not the model.** `execute_model`
  builds `IntermediateTensors` from `recv_tensor_dict` on a non-first rank and
  sends `output.tensors` on a non-last rank (`vllm_ascend/worker/worker_v1.py:212-222`
  at `048bfd5`), so the runner returned a differently-typed object. Upstream model
  bodies return `IntermediateTensors` on a non-last rank
  (`vllm/model_executor/models/qwen2.py:359-365`), and the same `models/` code
  serves TP-only deployments, so the model file and the weights are exonerated.
- **The runner's replay path is the owner.** `ACLGraphWrapper.__call__` returns
  `entry.output` on replay (`vllm_ascend/compilation/acl_graph.py:186`), which was
  stored as `weak_ref_tensors(output)` at capture (`:164`; the capture-time
  `weak_ref_output` branch is `:160`). The downstream `weak_ref_tensors`
  (`vllm_ascend/utils.py:606-619`, a verbatim copy of vllm `utils.py:1913-1926`)
  handles `Tensor`, `list`, `tuple` and raises `ValueError` for anything else —
  so a container's **class** does not round-trip through the graph pool the way
  the eager return object does. Eager mode never takes this path
  (`acl_graph.py:103-111` returns the runnable's output when the runtime mode is
  `NONE`), which is exactly why one flag separates the two explanations.
- **So the classification is "graph-mode × PP output contract", and the ranked
  cause list is short.** Anything about W8A8, the 8B checkpoint, the OpenAI
  server or the scheduler predicts a failure that also breaks TP-only serving,
  which the reporter's own evidence contradicts.

## Ranked causes and the one check each

1. **The ACLGraphWrapper output path loses the `IntermediateTensors` identity on
   a non-last PP rank** (source-verified line set above; mechanism hypothesis).
   Documented precedent for this axis: `docs/source/user_guide/release_notes.md:359`
   lists "Pipeline parallel does not work with ray and graph mode: #1751 #1754"
   under v0.9.2rc1 Known Issues — the same PP × graph-mode pairing, filed a
   release before this report and never closed in the pinned tree.
   Check: re-run the identical command with `--enforce-eager` and require the
   pin's own banner in the paste (`platform.py:175` `Compilation disabled, using
   eager mode by default`) — the assert gone confirms this cause, while the same
   assert at `:218` on a paste *carrying that banner* confirms cause 2 instead.
   `--compilation-config '{"cudagraph_mode": "NONE"}'` is the same ablation in
   config form; prefer the one-word flag first.
2. **PP itself is broken on that model/hardware with or without graphs.** Check:
   the same `--enforce-eager` run — if it still fails at `:218`, the owner moves to
   the worker/runner PP plumbing and capture is irrelevant. Two data points
   separate this from cause 1 and cost one launch.
3. **The image/wheel pair for the graph path.** Only worth ranking if (1) and (2)
   disagree in an unexpected direction; cite the version correspondence
   (`../startup-failures/environment-correspondence.md`) rather than reading
   `acl_graph.py` further.

Note the reporter's executor: `mp` here, so the `ray` half of #1751/#1754 does
not apply — say that explicitly rather than quoting the known issue as if it
covered both backends.

Corroboration the reporter supplied on 2025-09-24 (same thread): PP for
`qwen3-32b` fails identically and an **offline** `LLM()` PP run fails too — i.e.
not W8A8, not the 8B checkpoint, not `vllm serve`. Quote it as reporter-confirmed
scope, and note the follow-up says "还没有解决" (no workaround found yet), so the
`--enforce-eager` rung is a *diagnostic*, not a reported fix.

**That offline run is not a graph ablation, so it does not retire cause 1**, and
saying otherwise is the one correction this thread's follow-up needs.
`vllm_ascend/platform.py` reads `enforce_eager` from the model config (`:170`), prints
the eager banner only when it is true (`:175`), and maps the *unchanged* `PIECEWISE`
level to `cudagraph_mode = PIECEWISE` (`:182-184`) at `048bfd5`; upstream sets that
level for V1 whenever `enforce_eager` is false (`vllm/config.py:4439-4446` in this
snapshot's v0.9.1 export — the working side, so cite the shape, not the line). Both
entry points pass through the same `check_and_update_config`, so an offline `LLM(…)`
without `enforce_eager=True` lands in the same `Replaying aclgraph` lane and excludes
only the serve/OpenAI front end. The discriminating run is the *offline* one-command
shape — `LLM(model=…, tensor_parallel_size=1, pipeline_parallel_size=2,
enforce_eager=True)` plus one short prompt — cheaper than relaunching the server, and
its own `Compilation disabled`/eager line is the proof the graph really was off.

## Delivery shape for this class

Verdict + rung 1 in the first message ("this is the PP × aclgraph output path;
run it once with `--enforce-eager`"), then the two `path:line` anchors, then the
CI-coverage sentence (`../pipeline-parallel/index.md` rule 2) and the documented
status quote (`supported_features.md:22`). Three turns is enough; the recorded run
spent its whole output budget inside `acl_graph.py`/`worker_v1.py` cross-reads and
shipped no message at all, and a second session on the same issue repeated it. The
**follow-up turn is a delta, never a re-issue**: 「离线也报错」 narrows the model,
quantisation and entry point and leaves the ranking unchanged, so the whole message is
the paragraph above (2-4 lines) plus the ask.

## Limits

- Which object the wrapper actually returns (a `tuple` of weak-ref tensors, the
  inner `.tensors` dict value, or something re-wrapped) was **not** observed on
  hardware here; only the return-site asymmetry (eager vs replay) is verified at
  `048bfd5`. Say "hypothesis, one-flag check" in the answer.
- The graph wrapper at that ref is a *fork of an older upstream design*; vllm
  0.10.2's own equivalent is `vllm/compilation/cuda_piecewise_backend.py:186-193`
  (`entry.output = weak_ref_tensors(output)` too), so do not present the upstream
  file as the fixed implementation without checking its ref.
