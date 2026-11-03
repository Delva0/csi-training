# Dynamo graph breaks inside the compiled forward

Scope: start-up or first-request failures whose raising exception is
`torch._dynamo.exc.Unsupported` (often `Attempted to call function marked as skipped`) rather than a
device error. Verified by reading tag-pinned `vllm-project/vllm-ascend` and `vllm-project/vllm`
sources plus the recorded report thread; nothing was run on an NPU.

Distinguish this class from the two neighbours before reading kernels:

| signature | class |
| --- | --- |
| `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped`, explanation names a builtin (`posix.putenv.`), `Developer debug context: module: posix, qualname: putenv` | a host side effect executed *while Dynamo traces* the model — this file |
| `107027 ... Not allow to synchronize captured-stream` / `reason=[stream is captured]` | a host round-trip inside ACL-graph *capture* (a real device/runtime prohibition); see `log-signatures.md` |
| any CANN/HDK error code (`507015`, `561000`, `EZ1001`) with an operator name | device operator fault; see the model/op references |
| `[ASSERT] .../opp/built-in/op_impl/.../ascendc/<op>_base.h:NNN` + `dex <i> out of range[0 N)` in the plog, no Python traceback | device-side bound check inside a CANN built-in kernel — the caller allocated the `N`-sized axis; see `log-signatures.md`. A report whose only trigger is ">65535 tokens" can be this class rather than the Dynamo one, so check which of the two signatures the run actually shows |

## Why a graph break is fatal here

vLLM wraps the model with `torch.compile(..., fullgraph=True)` (`vllm/compilation/wrapper.py:151` at
`v0.18.0`), so a construct Dynamo can only graph-break on raises instead of falling back to eager.
The engine therefore dies during the compile/profiling dummy run, before any request is served.

## Verified instance: `os.environ` written in `AscendMRotaryEmbedding.forward_triton`

`#8224` (Qwen3-VL-8B-Instruct, `v0.18.0rc1`, `--max-num-batched-tokens 81920`, graph mode default):

- The traceback's `from user code:` tail is the locator — `qwen3_vl.py:2288 forward` ->
  `language_model.model` -> `decorators.py:597 TorchCompileWithNoGuardsWrapper.__call__` ->
  `qwen3.py:159 q, k = self.rotary_emb(...)` -> `custom_op.py:135` ->
  `vllm_ascend/ops/rotary_embedding.py:523 forward_oot` -> `:500`. The raising statement is
  `os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"`, i.e. `posix.putenv`; everything above it is the
  dispatch chain, and no model, checkpoint or CANN frame is at fault.
- Tag source (`v0.18.0rc1`): `AscendMRotaryEmbedding._ASCEND_TRITON_GRID_LIMIT = 65535` (`:472`);
  `forward_triton` (`:474-513`) writes the env var when
  `query_shape[0] > 65535 and os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") != "1"` (`:499-500`);
  `forward_oot` (`:515-523`) reaches it only when
  `HAS_TRITON and positions.ndim == 2 and self.mrope_interleaved`. The class is installed as the
  `MRotaryEmbedding` implementation at `vllm_ascend/utils.py:623`/`:638`.
- Introduced by `26b83f8bd` ("[Bugfix] Improve Triton stability on Ascend for large grids", #6301,
  2026-02-03, "set `TRITON_ALL_BLOCKS_PARALLEL=1` when grids > 65535"), first tag `v0.15.0rc1`.
  The write is still on `origin/main` at `:672-673` (2026-11-01), so **no release fixes it** and an
  upgrade is not the answer.

## Budget the trigger from the step's token count, not from the prompt length

- `query_shape[0]` is the number of tokens in the forward being compiled. With graph mode the dummy
  run is sized from the scheduler: `profile_run` calls `_dummy_run(self.max_num_tokens, ...)` and
  `self.max_num_tokens = scheduler_config.max_num_batched_tokens`
  (`vllm/v1/worker/gpu_model_runner.py:434`, `:5516-5517` at `v0.18.0`). `--max-num-batched-tokens
  81920` therefore crosses 65535 during start-up; the same threshold is reached at decode only with
  at least 65536 tokens in one step.
- Only interleaved-MRoPE models take the Triton branch: a non-interleaved mrope model (Qwen2.5-VL)
  falls through `forward_oot` to `torch.ops.vllm.npu_rotary_embedding` and never touches
  `os.environ`. That is the code-implied reason the same reporter's Qwen2.5-VL run worked — their
  observation, our inference; the `:523` frame in the traceback itself proves the interleaved
  condition held for the failing run.

## Workarounds, in the order to offer them

1. `--max-num-batched-tokens` ≤ 65535. This is the thread's own reply ("delete
   `--max-num-batched-tokens 81920`"), and it is consistent with the threshold, so it is a
   mechanism-backed workaround rather than a guess.
2. Preset `TRITON_ALL_BLOCKS_PARALLEL=1` in the process environment: the guard tests
   `os.environ.get(...) != "1"`, so the write is skipped and nothing host-side runs inside the traced
   region. Candidate only — code-implied, untested, and it applies that Triton setting to every
   kernel in the process.
3. `--enforce-eager`: no compile, so nothing is traced (throughput cost). Use it as the
   applicability test, not as the recommendation, when the tag's graph mode is the point of the
   deployment.

The durable fix shape is to hoist the env write out of the forward (module import, worker init, or a
one-time check the runtime can trace), which is what to suggest in an issue.

## Generalisation

Treat any host-side side effect in a compiled region as this class: `os.environ[...] = ...`,
`os.putenv`, `.item()`, `print`, file or socket I/O inside a `forward`. When the traceback tail names
a plugin `ops/` file, grep that tag's file for `os.environ` and read only the statements inside the
forward methods; the compile wrapper, decorator and vLLM model frames above are the path, not the
cause. Evidence boundary: the raise site is quoted from the reporter's traceback and verified in tag
source; the workarounds were not executed here.
