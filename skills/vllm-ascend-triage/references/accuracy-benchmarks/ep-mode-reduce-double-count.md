# 开启 EP 就胡言乱语、关掉 EP 正常: the model's own all-reduce re-sums what the MoE comm already aggregated

## Scope

An Ascend MoE serve (`--enable-expert-parallel`, A2 or A3, one or more nodes) whose
answers are **fluent but wrong / 乱码 / repeated** while the same command without
`--enable-expert-parallel` is correct — nothing crashes, throughput is normal. Also the
code-change form of the same finding: the reporter has already named the mechanism and
asks *which* layer should carry the fix, which is where the sweep in Step 3 is the
deliverable. The ownership question decides the track: the reduce contract, not a kernel.

Not this topic: a report that *drops* EP on purpose and asks about throughput (that is
`../performance-throughput/moe-comm-path-when-ep-off.md`, which carries the selector from
the perf side), an EP+DP engine that fails to **start** or a dispatch tiling refusal
(`../feature-enable-recipes/moe-expert-parallel-with-dp.md`), or corruption that survives
with EP off (start at `index.md`).

## Step 1 — the pair of lines that settles it, two greps

```bash
rg -n 'reduce_results=False|tensor_model_parallel_all_reduce\(final_hidden_states\)' \
   <vllm>/vllm/model_executor/models/<model>.py
rg -n 'def maybe_all_reduce_tensor_model_parallel' -A8 \
   <vllm>/vllm/model_executor/layers/fused_moe/layer.py \
   <ascend>/vllm_ascend/ops/fused_moe/fused_moe.py
```

Verified at vllm-ascend **v0.14.0rc1** (`52d4acfa51fb`) + vllm **v0.14.1**
(`d7de043d55d1`) — the pair a `v0.14.0rc1` image actually runs:

- downstream `AscendFusedMoE.maybe_all_reduce_tensor_model_parallel`
  (`vllm_ascend/ops/fused_moe/fused_moe.py:271-280`) overrides the upstream method and
  forwards to `torch.ops.vllm.maybe_all_reduce_tensor_model_parallel`; its impl returns
  the tensor **unchanged** for `moe_comm_type in {ALLTOALL, MC2, FUSED_MC2}` **or
  `forward_context.sp_enabled`**, and only otherwise calls
  `tensor_model_parallel_all_reduce`
  (`vllm_ascend/ops/register_custom_ops.py:250-262`). The `sp_enabled` clause is the part
  reporters never mention: **sequence parallelism breaks the same models the same way**,
  so an EP-only motivation understates the bug. `sp_enabled` is set per batch in
  `set_ascend_forward_context` (`ascend_forward_context.py:88-99`) from
  `enable_sp(vllm_config)` = `compilation_config.pass_config.enable_sp` **or**
  `VLLM_ASCEND_ENABLE_FLASHCOMM1` **or** `VLLM_ASCEND_ENABLE_FLASHCOMM=1`
  (`vllm_ascend/utils.py:812-825`) — and for a MoE model it needs no token threshold, while
  a dense model's path also requires `num_tokens > 1000`. So a serve whose command carries
  neither EP nor a graph knob can still be in this lane via FlashComm.
- upstream `FusedMoE` (`vllm/model_executor/layers/fused_moe/layer.py`): the constructor
  default is `reduce_results: bool = False` (`:342`), dispatch/combine is gated on
  `self.dp_size > 1` (`:1922`, `:1943`) — which is why GPU EP with `TP>1, DP=1` genuinely
  needs the model's own all-reduce — and
  `maybe_all_reduce_tensor_model_parallel` (`:1696-1703`) returns unchanged when
  `must_reduce_shared_expert_outputs()` (`:1677-1690`).

So on Ascend the model-level bare `tensor_model_parallel_all_reduce` sums `tp_size`
copies of an already-complete tensor: the MoE output arrives ~`tp_size`× too large, which
destroys the logit distribution without raising anything.

## Step 2 — do **not** quote "EP always dispatches": read the per-SoC gate

`select_moe_comm_method` (`vllm_ascend/ascend_forward_context.py:225-272` at this pin,
assigned to the forward context at `:63-65`):

| condition | method | bare all-reduce there |
|---|---|---|
| not a MoE model / EP off / `ep_group.world_size == 1` | `ALLGATHER` | **correct** |
| A2 + EP on + `num_tokens <= mc2_tokens_capacity` + `world_size_across_dp / pp >= 16` | `MC2` | double counts |
| A2 + EP on, otherwise | `ALLGATHER` | **correct** |
| A3 + EP on, decode (`num_tokens <= capacity`) | `MC2`, or `FUSED_MC2` when `VLLM_ASCEND_ENABLE_FUSED_MC2` and `w8a8_dynamic` | double counts |
| A3 + EP on, prefill | `FUSED_MC2`/`ALLTOALL` | double counts |

Three consequences that must be in the answer:

- **The method is chosen per batch**, so one server can be corrupt in decode and correct in
  prefill (a prefill chunk above the capacity lands on `ALLGATHER`, where the bare
  all-reduce is exactly what `maybe_all_reduce` would have done). That is why the first
  token can look plausible while the answer collapses.
- **On A2 the bug needs `world_size_across_dp / pp >= 16`.** A 16-card TP16+EP serve is
  affected; the *same* model at TP8 (world 8 → `ALLGATHER`) should be clean. That is the
  free discriminating run, and it also retires an fp8/quant defect without a rebuild:
  run the reported body at `--tensor-parallel-size 8` with EP still on.
- On A3 the comm choice also depends on the checkpoint's quant type, so a 910B result does
  not transfer; state the SoC in the verdict.

## Step 3 — the affected surface is a sweep, not two models

`rg -n 'tensor_model_parallel_all_reduce\(final_hidden_states\)' <vllm>/vllm/model_executor/models/`
returns **eight** files at vllm v0.14.1: `arctic.py:257`, `minimax_m2.py:140`,
`dots1.py:194`, `kimi_linear.py:173`, `minicpm.py:182`, `bert_with_rope.py:339`,
`hunyuan_v1.py:464`, `step3_text.py:98`. Read each one's `reduce_results` before claiming
it: `minimax_m2` (`:109`) and `step3_text` (`:74`) pass `False` explicitly; `dots1`,
`kimi_linear` and `hunyuan_v1` have `reduce_results=False` MoE call paths;
`minicpm`/`bert_with_rope` never spell it out and so ride the constructor default (`False`);
`arctic.py:256` guards its own call with `self.reduce_results and self.tp_size > 1`, so it
needs its own reading of `:367/:383` and must not be folded into the same PR sweep
unexamined. For a code-change request this table *is* the deliverable — fixing only the
reported two leaves the same bug to resurface model by model.

## Step 4 — where the fix belongs

Upstream model files, replacing the bare call with
`self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)`: the Ascend
override is the comm-type-aware place, and on a GPU `DP=1` serve the method is the plain
all-reduce. Two caveats to keep honest rather than silent:

- "backward-compatible: on GPU it is just the all-reduce" is *nearly* true — the upstream
  method returns unchanged when `must_reduce_shared_expert_outputs()` is true, which is
  inert only for models that pass no `shared_experts` into `FusedMoE` (both reported ones
  do not). Say which it is per model.
- downstream has only `vllm_ascend/patch/{platform,worker}/` at this pin — **no
  `patch/model/`** — so a downstream override would be new machinery patching two upstream
  forward bodies, i.e. the drift this library keeps paying for. Recommend upstream, and
  list `sp_enabled` in the motivation.

## Ladder (cheapest first)

1. **`--tensor-parallel-size 8` with EP still on** (A2): predicts clean output and costs one
   restart. Confirm with the `logprobs` of one fixed request rather than the word "garbage".
2. Apply the one-line edit to the *installed* vllm model files and restart — they are pure
   Python, no rebuild: `python -c "import vllm.model_executor.models.minimax_m2 as m;
   print(m.__file__)"`, then `sed -i` the two call sites. Acceptance is the restored logit
   scale, not "no exception".
3. EP off (the reporter's own working control) — keeps serving while the fix lands, at the
   price of EP weight sharding.
4. Turn sequence parallelism off — `--compilation-config '{"pass_config":
   {"enable_sp": false}}'` and unset `VLLM_ASCEND_ENABLE_FLASHCOMM1` /
   `VLLM_ASCEND_ENABLE_FLASHCOMM` — because the same skip list applies: a report that says
   "EP is off and it is still corrupt" but whose command carries FlashComm is an SP report,
   not an EP one.

## What refutes this page

- The reported config resolves to `ALLGATHER` (EP off, `ep_group.world_size == 1`, A2 world
  below 16, or a prefill chunk above `mc2_tokens_capacity`): the bare all-reduce is then
  *correct*, and the corruption belongs to `garbled-long-context-output.md`,
  `constant-token-degeneracy.md` or the sampler pages.
- Anything that raises: a traceback routes to the startup/dispatch lanes
  (`../feature-enable-recipes/moe-expert-parallel-with-dp.md`).
- The same checkpoint served correctly on GPU with identical flags is *not* a control here
  (GPU `DP=1` never dispatches); the control that matters is EP-on vs EP-off and TP8 vs
  TP16 on the same box.

## Scope and recheck keys

Pins: vllm-ascend v0.14.0rc1 `52d4acfa`, vllm v0.14.1 `d7de043`. Re-grep
`select_moe_comm_method`, `maybe_all_reduce_tensor_model_parallel`, `MoECommType` and the
model files' `reduce_results` at another ref: the selector moved from the runner into
`ascend_forward_context.py`, the A2 branch lost its `ALLTOALL` row at v0.13.0rc1, and a
later release may add a model-side helper or a different skip list. Not verified here:
whether a fix has landed upstream after these pins, and the runtime value of
`mc2_tokens_capacity` (a profiling-time global, `ascend_forward_context.py:186-187`).
