# `Unsupported: Attempted to call function marked as skipped` / `posix.putenv`: an `os.environ` write inside the compiled region

## Scope

The engine never becomes ready and the worker raises a **dynamo graph break**
while the model forward is being compiled, not at load and not at capture:

```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `posix.putenv.`
  Developer debug context: module: posix, qualname: putenv
from user code:
   File "…/vllm_ascend/ops/rotary_embedding.py", line 500, in forward_triton
     os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
   File "<frozen os>", line 685, in __setitem__
```

recorded on Qwen3-VL-8B-Instruct (#8224, image `v0.18.0rc1`, A2, `--dtype
bfloat16 --max_model_len 81920 --max-num-batched-tokens 81920`), reached from
`profile_run` → `_dummy_run` → the `support_torch_compile`d
`language_model.model(...)`, i.e. during the memory-profiling forward.

Owner: **the downstream forward that writes `os.environ` inside the region vLLM
compiles**, because vLLM compiles with `fullgraph=True`, where a Python builtin
that dynamo cannot trace is a hard error rather than a silent graph break.
Sibling lanes: the torchair form of the same `fullgraph` fatalism is
`torchair-dynamo-compile-unsupported.md`; when the untraceable call is an
accelerator wheel's pybind capsule it is
`../image-dependency-faults/mindie-turbo-graph-break.md`. This page is the
**downstream-code** form, and its repair needs no change of image or wheel.

## Pinned facts (vllm-ascend v0.18.0rc1 `99e1ea0`, vllm v0.18.0 `bcf2be96`)

| Fact | Anchor |
|---|---|
| The env write sits inside the traced forward, under a shape-conditioned branch | `vllm_ascend/ops/rotary_embedding.py:474` `def forward_triton`, `:499` `if query_shape[0] > self._ASCEND_TRITON_GRID_LIMIT and os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") != "1":`, `:500` the assignment; `:472` `_ASCEND_TRITON_GRID_LIMIT = 65535` |
| Why this model takes that path | `:515` `forward_oot` → `:521` `if HAS_TRITON and positions.ndim == 2 and self.mrope_interleaved:` → `:523` `return self.forward_triton(...)`; the reporter's traceback already proves all three conditions held (reporter-confirmed) |
| Which models are affected | `vllm_ascend/utils.py:623,638` registers `"MRotaryEmbedding": AscendMRotaryEmbedding` as the global override, so every checkpoint whose `rope_parameters` carry `mrope_interleaved: true` (`vllm/model_executor/layers/rotary_embedding/__init__.py:104,247,311`, default `False`) inherits it — Qwen3-VL family, not one model |
| Why a graph break is fatal | `vllm/compilation/wrapper.py:148-152` `torch.compile(compiled_ptr, fullgraph=True, dynamic=False, …)`; the paste's `TorchCompileWithNoGuardsWrapper.__call__` / `_call_with_optional_nvtx_range` / `compile_wrapper` frames are that call |
| Not repaired in the newer pin | at v0.23.0rc1 (`f4a08bd`) the same statement is still in `forward_triton` (`vllm_ascend/ops/rotary_embedding.py:501-502`), so "upgrade" is not the rung |
| Why the size knob is the trigger | the profiling dummy run is `max_num_batched_tokens` tokens (the reporter's log: `Encoder cache will be initialized with a budget of 81920 tokens`), so `81920 > 65535` selects the branch on the very first compiled forward |
| Why nobody's CI caught it | `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mrope.py:6,16` imports `triton_mrope` and calls the op directly with `NUM_TOKENS = [1, 4, 8, 16]`, bypassing `forward_oot`; no file under `tests/` mentions `65535` or `TRITON_ALL_BLOCKS_PARALLEL`; the tutorial recipe is `--max_model_len 16384 --max-num-batched-tokens 16384` (`docs/source/tutorials/models/Qwen-VL-Dense.md:324-327`) |

The paste's own frame numbers differ from the tag by 10-30 lines (`profile_run`
is `vllm_ascend/worker/model_runner_v1.py:2538` in the tree, `:2550` in the
paste), so take the **symbol** from the paste and the `path:line` from the pin;
do not read the delta as a different release.

## Do not re-derive the reachability

The version-independent question — "would dynamo have traced the branch anyway?"
— is not worth a call. Two recorded readings are already enough: the traceback
shows the assignment was traced, and the condition is what decides whether it is
in the compiled code. Branch on the reporter's own numbers, not on a dynamo
semantics argument.

## Answer shape (report language follows the report; #8224 was Chinese)

1. First sentence: **not "模型不支持图模式"** — the model is fine (the pinned
   tutorial runs Qwen3-VL-8B in graph mode and its sample log shows
   `Replaying aclgraph`); the failure is vllm-ascend writing an environment
   variable inside the `torch.compile` region, and `fullgraph=True` turns that
   into a startup error.
2. Cheapest action first: `export TRITON_ALL_BLOCKS_PARALLEL=1` before
   `vllm serve` — the `os.environ.get(...) != "1"` half of the condition becomes
   false, no `putenv` is traced, and the value the code wanted to set is already
   set. The knob is standard practice in this ecosystem's launch recipes
   (`export TRITON_ALL_BLOCKS_PARALLEL=1` appears in the run blocks of #9111,
   #9092, #8670, #8784, #8037; #3865/#3868 carry it commented out).
3. Then the alternatives, each with its own cost: `--max-num-batched-tokens`
   ≤ 65535 (keeps graphs, caps the batch), or `--enforce-eager` (loses
   torch.compile/ACL graph — demand its own lines,
   `Compilation disabled, using eager mode by default` / `Cudagraph is disabled
   under eager mode`, before accepting "tried it, same").
4. Name the control the reporter already ran: Qwen2.5-VL on the same image works
   because its rope config has no `mrope_interleaved` (`get_rope` default
   `False`), so it never enters `forward_triton` — that is the dispatch, not a
   per-model support verdict.
5. Scope: the branch also reaches Qwen3-VL-30B-A3B and any other
   `mrope_interleaved` checkpoint; it needs `HAS_TRITON` true (so not the
   `-310p` image), a `positions.ndim == 2` prefill and a batch over 65535.
6. Queue the missing artefact only if the paste lacks it: the full `vllm serve`
   line and the log's own `budget of N tokens` line (they fix the batch size).

## The generalisation worth carrying

Any `os.environ[...] = ...`, `os.putenv`, `.item()`-style host readback or
`sys.modules` write executed inside a `support_torch_compile`d forward is a
startup crash on the **vLLM torch.compile** lane. When the `from user code`
block ends in `<frozen os>`/`posix.putenv` or a Python builtin, grep the named
`vllm_ascend` file for `os.environ` and read the branch condition — the fix is
always to move the write out of the traced region or to pre-set the variable on
the launch line, never to touch the image.
