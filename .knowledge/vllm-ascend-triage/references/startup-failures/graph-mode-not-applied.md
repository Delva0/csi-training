# A graph / compilation setting that never takes effect

Scope: the report is "开了 `--compilation-config '{"cudagraph_mode":
"FULL_DECODE_ONLY"}'`（或 yaml 里的 `compilation_config`）但 FULL_DECODE_ONLY 未生效",
with no crash, often no log pasted at all, and the only symptom is performance or the
absence of the expected startup banner. One `mode` value is mutated by a chain of
gates between argument parsing and the worker's first capture; **every gate prints
its own line**, so the deliverable is the gate table below plus one grep of the
worker log. Verified at vllm-ascend v0.14.0rc1 (`52d4acfa`) + vllm v0.14.1
(`d7de043`) — the pins of the recorded #6972 case; re-grep the line numbers at
another ref, the gate order has been stable but the messages are not frozen.

## Step 0 — `level:` is not an optimisation level

`CompilationConfig` maps the deprecated `level` onto `mode` only when `mode` is unset
(`vllm/config/compilation.py:748-758`, with the "Level is deprecated" warning), and
`mode` is the `CompilationMode` **enum value** (`compilation.py:355-369`):
`0 NONE`, `1 STOCK_TORCH_COMPILE`, `2 DYNAMO_TRACE_ONCE`, `3 VLLM_COMPILE`.
A stage yaml carrying `compilation_config: {level: 1, cudagraph_mode: FULL_DECODE_ONLY}`
therefore asks for `STOCK_TORCH_COMPILE`, and vllm-ascend drops any mode outside
`{NONE, VLLM_COMPILE}` and sets `cudagraph_mode = NONE`
(`vllm_ascend/platform.py:241-245`). The whole request dies on one WARNING line the
reporter usually pastes as "no effect". Fix: delete `level`, keep only
`cudagraph_mode` (mode then defaults to `VLLM_COMPILE` at
`vllm/config/vllm.py:695-699`), or write `mode: 3` / `VLLM_COMPILE`.

`--omni` (multi-stage vllm-omni) is not an upstream flag: `grep -rn '"--omni"'
vllm/` is empty at v0.14.1 and vllm-omni itself is absent from `SOURCES.json`. So the
top-level CLI → per-stage `engine_args` merge cannot be read here — rank it as an
open cause (stage `engine_args` win over the CLI; a stage running eager) and decide it
per stage from the line below, never from the launcher source.

## Gate table — one log line each

| Gate | Site (v0.14.0rc1 / v0.14.1) | Line it prints |
|---|---|---|
| `enforce_eager`, from **any** source (CLI, stage yaml, `--speculative-config`'s own `enforce_eager`) | upstream `vllm/config/vllm.py:763-769` empties `cudagraph_capture_sizes`; vllm-ascend `platform.py:234-238` sets `mode = NONE` | `Cudagraph is disabled under eager mode` and/or `Compilation disabled, using eager mode by default` |
| encoder-decoder checkpoint (`is_encoder_decoder`) | upstream `vllm/config/vllm.py:752-763` (forces `FULL_DECODE_ONLY`), then vllm-ascend `platform.py:275-280` forces `PIECEWISE` | `Encoder-decoder models do not support %s …` / `encoder-decoder model doesn't support FULL_DECODE_ONLY, fallback to PIECEWISE` |
| `mode` outside `{NONE, VLLM_COMPILE}` (the `level: 1` case) | `platform.py:241-245` | `NPU does not support %s compilation mode. Setting CUDAGraphMode to NONE` |
| `FULL_AND_PIECEWISE` | `platform.py:273-274` — silent downgrade | *(nothing)*; only `PIECEWISE compilation enabled on NPU` appears |
| `xlite_graph_config.enabled` full mode | `platform.py:227-232` | `ACLGraph is disabled under xlite full mode` |
| `cudagraph_mode` incompatible with `mode` | upstream `vllm/config/vllm.py:712-723` | `Cudagraph mode %s is not compatible with compilation mode %s. Overriding to NONE.` |
| attention backend cannot do this graph (`AttentionCGSupport`) | upstream `vllm/model_executor/…/gpu_model_runner.py:5036-5140` via `vllm_ascend/worker/model_runner_v1.py:2803-2815`; a `NEVER` backend raises `ValueError` instead | `CUDAGraphMode.X is not supported with <Backend> backend (support: …); setting cudagraph_mode=…` |
| any graph mode + the debugging env | `platform.py:339-348` | `ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1` |
| **success** | `platform.py:289` / `platform.py:313-330` | `PIECEWISE compilation enabled on NPU …` / `FULL_DECODE_ONLY compilation enabled on NPU …` plus the red "You have enabled the *full graph* feature" block |

Also gated by mode: `PIECEWISE` asserts `mode == VLLM_COMPILE`
(`platform.py:293-295`), and `cudagraph_mode == NONE` forces `mode = NONE` and turns
`npugraph_ex` off (`platform.py:285-287`). Dispatch itself is in
`vllm_ascend/attention/utils.py:51-52`, which special-cases `FULL_DECODE_ONLY` and
otherwise runs the piecewise path — so a worker whose log lacks the success line at
`:313` never had the mode, whatever the launcher said.

## Procedure

1. Grep the **worker** log (per rank) for the strings above, not the frontend's
   `Initializing a V1 LLM engine … with config:` dump: that dump does print
   `compilation_config=CompilationConfig(...)` (upstream `vllm/v1/engine/core.py:98` +
   `vllm/config/vllm.py:1361`) but it is the pre-worker state.
2. If the first gate that fired reads "does not support %s compilation mode", the
   reporter wrote `level:` — that is the answer, one JSON edit.
3. If it reads "disabled under eager mode", find the flag: `--enforce-eager` in the
   serve line, in a stage yaml, or upstream's `speculative_config.enforce_eager`
   (vllm-ascend clears that one at `platform.py:424-437`).
4. If no gate line exists at all, the setting never reached the process that
   matters: the stage yaml's `engine_args` and the top-level CLI are different
   inputs, and vllm-omni's merge is unreadable here — ask for each stage's log.
5. Sizes are a second, independent question: the user list is de-duplicated and
   filtered to `<= max_num_tokens` (`vllm/config/vllm.py:1160-1200`), sizes outside
   it are never captured (`../performance-throughput/sequence-count-cliff-capture-size.md`),
   and with sequence parallelism the list must keep multiples of `tp_size` or
   `platform.py:250-267` asserts.
6. State the version scope: the `level→mode` mapping, the platform's
   `{NONE, VLLM_COMPILE}` gate and the success message are all ref-specific.

The complement of this page: when the success line **is** there and capture
finished on every rank, the mode took effect and the request path is the question —
`../hccl-and-hardware/aclgraph-full-decode-runtime-hang.md` (recorded #4178).

Recorded case: #6972 (Qwen3-Omni-30B-A3B via `vllm serve --omni`, vllm-ascend
v0.14.0rc1, 8×910) — the yaml route is explained by `level: 1` at this ref; the CLI
route needs the per-stage line. The recorded session consumed the full output budget
with no answer emitted (see `../evidence-workspace/recorded-cases-startup-and-environment.md`).
