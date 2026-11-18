# torch.compile graph break inside a traced forward: a side effect the tracer cannot follow

## Scope

The run dies during the *dummy/profile* forward, after the weights load and after the
`… compilation enabled on NPU` banner, with a `torch._dynamo` error rather than an operator error:

```text
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `posix.putenv.` …
  Developer debug context: module: posix, qualname: putenv, skip reason: <missing reason>
```

and the whole traceback is vLLM/vllm-ascend wrapper frames plus a `from user code:` tail that ends at
a *framework* call. The cause is CPython side effect code executed inside the region `torch.compile`
traces — most often an `os.environ[...] = ...` assignment inside a model-method override.

Not this page: `attempted to run this operator with Meta` (a missing meta kernel, also a tracing
failure — `graph-capture-and-custom-ops.md` §A), a capture-window refusal (`107041`/`EE1016`, §C), a
device kernel fault (`device-kernel-faults.md`), or a Triton launch-geometry ceiling
(`kernel-launch-limits.md`, whose `coreDim … UINT16_MAX` is host-side but a different check).

## Common rules

- **The traceback names no owner; the skipped callable does.** Nothing ran on the device and no
  `vllm_ascend/` line raises: the frame above the error in `vllm_ascend/ops/<x>.py` is the whole clue.
  Keep the `Explanation:` / `Developer debug context:` lines — they name the CPython entry point
  (`posix.putenv`, `posix.getenv`, `builtins.open`, …). The `from user code:` block stops at the *last
  framework line before* the side effect (e.g. `vllm/model_executor/models/qwen3.py:159 q, k =
  self.rotary_emb(...)`), which points at the module, never at the line. Grep the downstream repo for the
  side effect:

  ```bash
  git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'os\.environ\[' <ref> -- vllm_ascend
  git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'os\.environ\.get\|os\.putenv\|setenv' <ref> -- vllm_ascend
  ```

- **Every such write is guarded by a value predicate, and the predicate is the workaround.** Read the
  three conditions together — the dispatch gate that reaches the function, the value gate on a tensor
  extent, and the env guard — because each is one lever.
- **A graph break is not a device fault and not a memory bug.** Nothing ran on the NPU: the failure is
  in the tracing of the Python, which is why `--enforce-eager` (no `torch.compile`) always avoids it.
- **The `UserWarning` above the error is the same fact, not a second one.** `Dynamo does not know how
  to trace the builtin …` repeats the `Explanation:` text; do not treat the warning and the exception as
  two findings.

## Worked case: #8224 (Qwen3-VL-8B graph mode, `v0.18.0rc1`)

Reported: `quay.io/ascend/vllm-ascend:v0.18.0rc1`, Atlas 800I A2, `Qwen3-VL-8B-Instruct`,
`--dtype bfloat16 --max_model_len 81920 --max-num-batched-tokens 81920`, no `--enforce-eager`; the
reporter notes Qwen2.5-VL-7B starts fine on the same image. Thread: one comment, from another user —
「Please try to delete `--max-num-batched-tokens 81920`. This method works for me.」

Chain at `v0.18.0rc1` (tag `99e1ea0`, `vllm_ascend/ops/rotary_embedding.py`):

- `:470 class AscendMRotaryEmbedding`, `:472 _ASCEND_TRITON_GRID_LIMIT = 65535`.
- `:515 forward_oot` dispatches to `forward_triton` when
  `HAS_TRITON and positions.ndim == 2 and self.mrope_interleaved` (`:521`).
- `:497-500` inside `forward_triton`, i.e. inside the compiled language model:

  ```python
  # When the grid becomes large, enable TRITON_ALL_BLOCKS_PARALLEL
  # to avoid scheduler/runtime failures.
  if query_shape[0] > self._ASCEND_TRITON_GRID_LIMIT and os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") != "1":
      os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
  ```

- The profile dummy run schedules `max_num_batched_tokens` tokens, and the reporter set `81920`, so
  `query_shape[0] = 81920 > 65535` and the first trace executes `putenv` → `Unsupported`, inside
  `core.py` → `determine_available_memory` → `worker.py:343` → `profile_run` → `_dummy_run` →
  `_model_forward` (the pasted frames). `[verified]` for the frames, the lines and the gate;
  `[reported]` for the token count (their flag), corroborated by the one comment: dropping the flag,
  i.e. the default budget below 65535, starts the server.
- **Why the sibling model starts:** `mrope_interleaved` comes from the checkpoint's rope parameters
  (`vllm/model_executor/layers/rotary_embedding/__init__.py:311-321` at vLLM `v0.18.0`), not from a
  flag. Read `<model>/config.json` → `text_config.rope_parameters` (or `rope_scaling`); the interleaved
  MRoPE family takes `forward_triton`, the non-interleaved one (`mrope_section [16, 24, 24]`,
  `npu_mrope` path) never enters the function. `[verified]` for the gate, `[reported]` for the two
  models' configs — quote the config before asserting it.

Workaround ladder, strongest first:

1. **Pre-set the variable before launch** — `export TRITON_ALL_BLOCKS_PARALLEL=1` — so the guard's
   `os.environ.get(...) != "1"` is False and the assignment is never traced. Keeps the 81920-token budget
   and graph mode. `[verified]` that the guard short-circuits: the reporter's own error names only
   `posix.putenv` after the `.get` guard was evaluated. `[log-consistent, not run]` that a pre-set value
   is equivalent for triton-ascend's own kernel-compile read — if the server still dies with a
   `posix.getenv`/`os.environ` skip, this lever is wrong and nothing is lost by falling to 2.
2. **Keep the per-forward token count at or below 65535** (`--max-num-batched-tokens ≤ 65535`, default
   8192; the reporter's one comment removed the flag entirely — `[reported]`, commenter-confirmed).
   Cost: throughput on long prompts, and `--max-model-len` must stay consistent with the budget.
3. **`--enforce-eager` / `-O0`** — no `torch.compile`, no tracing. Cost: the ACL-graph gains.

Root cause to file: `vllm_ascend/ops/rotary_embedding.py:497-500` writes `os.environ` inside a function
that runs under `torch.compile`; move the write to import/first-use time, or guard the function with
`torch._dynamo.disable` / `torch.compiler.allow_in_graph`. The variable is also the launch-geometry lever
of `kernel-launch-limits.md` — there the *value* is wrong; here the line that sets it is the defect, so
do not confuse the two reports.

**Falsifier.** Pre-set the variable and re-run: a successful capture (or a different error) confirms the
graph break. The same traceback means another host write is in the traced body — `git grep -n
'os\.environ\[' <ref> -- vllm_ascend` and read the one on the reporter's model path.

## Status and ancestry

`TRITON_ALL_BLOCKS_PARALLEL` was introduced by `26b83f8bd` ([Bugfix] Improve Triton stability on
Ascend for large grids, #6301, 2026-02-03), first tag `v0.15.0rc1`; `git log -S'TRITON_ALL_BLOCKS_PARALLEL'
origin/main -- vllm_ascend` returns only that commit, and the same three lines are still at `main`
(2026-11, `:670-673`). So this is not a 0.18-only regression:
any tag from `v0.15.0rc1` on, any MRoPE-interleaved model, and any step scheduling more than 65535
tokens per forward reproduces it. `[verified]` by grep + `tag --contains`; no hardware run.

## Recheck conditions

Line numbers move (`:497-500` → `:670-673` between `v0.18.0rc1` and `HEAD`) and the mrope classes are
being split (`AscendMRotaryEmbedding` vs `AscendMRotaryEmbeddingInterleaved` at HEAD). Grep the
literal `TRITON_ALL_BLOCKS_PARALLEL` and the `os.environ[` writes at the reported ref rather than
quoting the lines above; re-read the dispatch gate in `forward_oot` too, because a new branch can
route an interleaved model away from `forward_triton`.
