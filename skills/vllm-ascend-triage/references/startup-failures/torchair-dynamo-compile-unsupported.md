# Torchair graph capture dies on a dynamo graph break (`fullgraph=True`)

Symptom family: the engine gets past weight loading and profiling, then every rank
prints `Capturing torchair graph, this usually takes 1.0~3.0 mins.` (`torchair_model_runner.py:284`
at v0.11.0rc3) and the capture fails with a *Python-level* compile error, not a device
error:

```
ERROR … [worker_base.py:275] Error executing method 'compile_or_warm_up_model'.
      This might cause deadlock in distributed execution.
  … worker_v1.py … capture_model → model_runner_v1.py … _capture_model
  … torchair_model_runner.py … _compile_torchair_graph
torch._dynamo.exc.Unsupported: call_function DelayGraphBreakVariable() [TensorVariable()] {}
```

The `worker_base.py` line is the executor's canned advice for *any* exception raised
inside a worker RPC; the owner is the innermost frame. Recorded twice in this family
(#1615 at v0.9.1rc1, w8a8, TP4 PP4; #4948 at v0.11.0rc3, BF16, TP8 PP4).

## Step 1 — the innermost `vllm_ascend` frame names the owner

At v0.11.0rc3 (`b6d63bb`) the refused callee is `Tensor.set_` inside a downstream helper:

- `vllm_ascend/utils.py:504-505` — `def dispose_tensor(x): x.set_(torch.empty((0,), …))`.
- `vllm_ascend/torchair/models/torchair_deepseek_v2.py:985-986` — the MLA decoder
  disposes `previous_hidden_states`/`previous_residual` on every layer with a residual,
  i.e. unconditionally for DeepSeek-R1/V3 (`VLLM_ASCEND_ENABLE_MLAPO` defaults to `'0'`
  at `envs.py:165-166`, so the alternative branch is off).
- Also called from `torchair/ops/shared_weight_layer.py:10` (its own copy),
  `torchair/ops/torchair_fused_moe.py`, `torchair/quantization/torchair_w8a8_dynamic.py`.
- Upstream vLLM at v0.11.0 has **no** `dispose_tensor` — this is a downstream-only
  helper, so the owner is vllm-ascend (plus its torch pairing), and the model weights,
  HCCL and the card count are not implicated by this traceback.

Why it is fatal here and only a split graph elsewhere: torchair mode compiles the whole
forward with `fullgraph=True` (`torchair_model_runner.py:423,446`, in
`_get_torchair_lazy_compiled_model`), so **any** dynamo graph break aborts engine start.
The piecewise ACL-graph path would only split the graph — so do not answer this with
`max_cudagraph_capture_size` or the ACL-graph capture pages.

## Step 2 — check the environment pairing before believing "the code is broken"

The dispose site is on every DeepSeek torchair compile path, so the literal reading
("this release cannot compile DeepSeek at all") predicts a failure for every such
deployment, while `tests/e2e/multicard/test_torchair_graph_mode.py` compiles
`vllm-ascend/DeepSeek-V3-Pruning` (2 hidden layers, so layer 1 reaches the dispose) at the
same ref with `tensor_parallel_size=2`, `distributed_executor_backend="mp"` and **no PP** —
i.e. the CI shape differs from the reported shape on exactly two axes. Name the split and
ask for one command:

- `python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"`
  and `pip show vllm vllm-ascend` **inside the image**, against the pinned pair
  `torch==2.7.1` / `torch-npu==2.7.1` (`pyproject.toml:16-17`, `requirements.txt:13,28`).
  A third-party/mirrored registry tag (`quay.nju.edu.cn/ascend/…` in #4948) is exactly
  where a repackaged torch build hides.
- Do not try to settle dynamo internals locally: torch's source is in no export of these
  case views, so "which torch build made `Tensor.set_` a delayed graph break" is an open
  question, not something a grep closes. Two calls of PyTorch archaeology is the whole
  budget in this lane.

## Step 3 — the layout axis: PP > 1 with ray and graph is a documented non-working pair

`docs/source/user_guide/release_notes.md:383` at v0.11.0rc3 still carries the v0.9.2rc1
known issue *"Pipeline parallel does not work with ray and graph mode: #1751 #1754"*,
and the CI torchair test above is TP-only. Both recorded reports of this signature ran
`--pipeline-parallel-size 4` with `--distributed-executor-backend ray` and
`torchair_graph_config.enabled`. The traceback itself names the callee, not PP, so carry
PP as the *layout risk* with its one ablation (graph off, or `--pipeline-parallel-size 1`
with TP/DP absorbing the cards) rather than as the proven trigger.

Alternative 32-die layout with no PP at all: #2343 (same model, 910B1 64G × 32) runs
`--data-parallel-size 8 --tensor-parallel-size 4` with torchair enabled.

## Step 4 — the workaround must not trip the config checker

`check_ascend_config` (`ascend_config.py:277-285`) raises
`Can't enable graph mode and eager mode at the same time. Please set 'enforce_eager=False' …`
when `torchair_graph_config.enabled` and `--enforce-eager` are both present. So "run
eager to get past the compile" means **removing** the `torchair_graph_config` key from
`--additional-config`, not adding a flag beside it; leaving both produces a second,
unrelated startup failure that looks like the fix did not work.

## Step 5 — on a full-precision giant MoE, the envelope is a co-equal cause

When the checkpoint is the unquantized one (e.g. DeepSeek-R1-0528-BF16, 671 B), price the
per-rank bytes before promising that a compile fix will get the deployment up:
`671e9 × 2 B ÷ (TP × PP)` = ~42 GB of a 64 GiB card at TP8×PP4, and the paste's own
`worker_v1.py:257` line (`Available memory: <N>, total memory: <M>`, the KV budget left
after weights plus the activation peak) said **5.08e9 of 6.545e10** — 7.8 % headroom at a
`--max-num-batched-tokens 65536` profile. Graph capture, workspace and prefill activations
all grow inside that remainder.

Reproduce the weight term with `scripts/card-budget.sh --params 671 --dtype bf16 --tp 32 …`
— only the **product** TP×PP shards the weights, so a `--tp 8` run reports 168 GB/card and a
false infeasibility. The script prints `41.94 GB per card … 18.9 GB left` here, i.e. a PASS
("not yet excluded"), while the reporter's own line says 5.08 GB was actually left: the gap
is the 65 K-token activation peak, which is why the paste's arithmetic outranks the script's
remainder whenever the profile is large.

Reporter-confirmed outcome of #4948 (later comments on the same issue, not in this
snapshot): full-weight BF16 R1-0528 needs **at least 4 Atlas 800I A2 servers
(32 × 910B4)** — the pasted layout was already at that minimum — and the DeepSeek-R1
tutorial page the maintainers point at is written for the **W8A8** weights, not BF16. So
the ranked answer pairs the trace fix with the checkpoint question (quantized weights =
the documented path, or more servers), one `scripts/card-budget.sh --params 671 --dtype bf16 …`
number, and the open question of which of the two the reporter actually needs.

## Delivery

This family is a two- or three-call answer: name the helper and line, name the
`fullgraph=True` consequence, then hand the reporter the ladder (environment pairing →
graph off with the config key removed → PP=1 / DP layout → quantized checkpoint). The
recorded #4948 run spent its whole 16 384-token output budget (blocks of 8.5 K and 10.7 K
characters) re-deriving what `DelayGraphBreakVariable` is and shipped **no answer**; the
G1 sentence already existed on its second read.
