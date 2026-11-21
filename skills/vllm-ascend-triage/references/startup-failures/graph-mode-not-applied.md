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
| `cudagraph_mode` outside every branch above — in practice `FULL`, **including the legacy `full_cuda_graph: true` that upstream maps onto it** | `platform.py:256-259` at v0.11.0rc0; re-grep the `else` at another ref | `%s cudagraph_mode is not support on NPU. falling back to NONE`, then `cudagraph_mode = NONE` + `level = NO_COMPILATION`; **intended**, pinned by `tests/ut/test_platform.py:342-364` — see the `FULL` section below |
| `xlite_graph_config.enabled` full mode | `platform.py:227-232` | `ACLGraph is disabled under xlite full mode` |
| `cudagraph_mode` incompatible with `mode` | upstream `vllm/config/vllm.py:712-723` | `Cudagraph mode %s is not compatible with compilation mode %s. Overriding to NONE.` |
| attention backend cannot do this graph (`AttentionCGSupport`) | upstream `vllm/model_executor/…/gpu_model_runner.py:5036-5140` via `vllm_ascend/worker/model_runner_v1.py:2803-2815`; a `NEVER` backend raises `ValueError` instead | `CUDAGraphMode.X is not supported with <Backend> backend (support: …); setting cudagraph_mode=…` |
| any graph mode + the debugging env | `platform.py:339-348`; re-grepped at v0.26.0rc1 (`f2f74a16`) as `platform.py:601-613`, TODO comment intact | `ValueError: ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1` — refuse to start, so the report is "set the env, service will not come up" with no other symptom |
| **success** | `platform.py:289` / `platform.py:313-330` | `PIECEWISE compilation enabled on NPU …` / `FULL_DECODE_ONLY compilation enabled on NPU …` plus the red "You have enabled the *full graph* feature" block |

Also gated by mode: `PIECEWISE` asserts `mode == VLLM_COMPILE`
(`platform.py:293-295`), and `cudagraph_mode == NONE` forces `mode = NONE` and turns
`npugraph_ex` off (`platform.py:285-287`). Dispatch itself is in
`vllm_ascend/attention/utils.py:51-52`, which special-cases `FULL_DECODE_ONLY` and
otherwise runs the piecewise path — so a worker whose log lacks the success line at
`:313` never had the mode, whatever the launcher said.

## `FULL` is not a supported mode, and the legacy flag lands on it

Scope: the paste's own line is `CUDAGraphMode.FULL cudagraph_mode is not support
on NPU. falling back to NONE`, usually one line under `full_cuda_graph is
deprecated, use cudagraph_mode=FULL instead.` Both facts that decide it are in the
tree (verified at vllm-ascend v0.11.0rc0 `00ba0710` + vllm v0.11.0rc3 `8ce5d319`):

- **The mapping is upstream's, not the launcher's.** `vllm/config/compilation.py:473-481`
  turns `full_cuda_graph: true` into `cudagraph_mode = FULL` (`FULL = 2`, `:41`) and
  refuses both keys together with `ValueError`. So when the reporter's `non-default
  args` dump carries `"cudagraph_mode":2` beside `"full_cuda_graph":true`, the flag
  *did* reach the config and there is nothing to find in the serve script.
- **The fallback is design at that ref.** `tests/ut/test_platform.py:342-364`
  (`test_check_and_update_config_unsupported_cudagraph_mode`) asserts exactly this
  line with `cudagraph_mode == NONE` and `level == NO_COMPILATION`. Quote the test
  instead of ranking causes: "why did my full-graph setting not take effect" is
  answered by the project's own unit test, and nothing on the reporter's box is
  broken.
- **It is worse than sending nothing.** With no flag, upstream defaults
  `cudagraph_mode` to `FULL_AND_PIECEWISE` (`vllm/config/__init__.py:366`), which this
  plugin silently downgrades to `PIECEWISE` (row above); requesting `FULL` ends at
  `NONE` + `NO_COMPILATION`, i.e. the flag switches graph mode *off*.
- **The NPU long-graph lane is `FULL_DECODE_ONLY`** (its own banner plus the red
  capture-size warning), so the fix is one JSON edit —
  `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`. No flag yields
  `FULL`, and `--enforce-eager` is the retreat, not the answer.

**When the request came from a doc page, the page is the defect.** Recorded #4696:
a `[Doc]` report on the Qwen2.5-Omni tutorial, whose "Graph Mode / full graph"
section told readers to launch with `--compilation-config {'full_cuda_graph': 1}`;
the pasted line is this fallback. At that pin the project already says so itself —
`docs/source/user_guide/release_notes.md:192` (`test_aclgraph.py` failed with
`"full_cuda_graph": True` on A2 (910B1), #2182) and `:350` ("Full graph mode support
are not yet available for specific hardware types with full_cuda_graph enable") — so
quote the release note rather than deriving the hardware matrix. Grade the
**command**, not the page body: at v0.11.0rc0 `full_cuda_graph` occurs in no doc page
and `docs/source/tutorials/` holds no Omni page, so a `latest`/`zh-cn` URL cannot be
quoted — deliver the corrected command plus the three-mode table and say the page
text is not in the snapshot (`../tutorial-verification/index.md` rule 3). The
snippet's quote style is cosmetic at most: `{'full_cuda_graph': 1}` parsed on the
reporter's own box (their dump shows the resolved field), so never grade a
Python-literal dict for a dataclass-valued flag as "breaks the command"
(`../tutorial-verification/tutorial-command-audit.md` check 1).

**Re-grep the key at the reported ref before quoting this section.** The mapping
above is v0.11.0's; at a later ref the same tutorial key can be *undefined* in
`CompilationConfig` (`extra="forbid"`), so the command raises a validation error at
startup instead of printing the fallback — recorded #6972 at v0.14.0rc1
(`../evidence-workspace/recorded-cases-startup-and-environment.md`). Which of the
two shapes the paste carries is the ref, not a second defect.

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

**A reporter's own `export` is not a flag.** `export ENFORCE_EAGER=1` sitting above an
unchanged refusal or an unchanged banner (recorded #1456, vllm-ascend v0.9.1rc1
`c30ddb8` + vllm v0.9.1 `b6553be`) buys nothing: `rg -n ENFORCE_EAGER` returns **zero**
bindings in both trees, so nothing reads it — and because the name carries no
`VLLM_`/`VLLM_ASCEND_` prefix it does not even trip the unknown-variable warning
(`../observability-and-logs/env-var-validation-warning.md`), which is why the paste
looks like a setting that was ignored rather than a name that was never a knob. The
value the gate reads is `model_config.enforce_eager` (`vllm_ascend/platform.py:151` at
that pair), written only by `--enforce-eager` / `LLM(enforce_eager=True)`
(`vllm/engine/arg_utils.py:339`, `:929`), and the proof it took is one line —
`Compilation disabled, using eager mode by default` (`platform.py:163-165`). So 0
bindings ⇒ answer with the CLI flag plus the doc gap that invited the guess, never with
a gate defect. When the refusal is the `ACL Graph does not support deepseek` form, that
same flag is lever 1 of `../feature-enable-recipes/torchair-mla-head-count-tp-gate.md`.
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

**Report shape without a traceback.** `加入环境变量 export ASCEND_LAUNCH_BLOCKING=1，
无法拉起服务` / `acl 流同步失败` on a recipe that ships a graph mode (the Kimi-K3 A3
recipe carries `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`) is this
gate, not a broken ACL stream: the env var is refused while `cudagraph_mode != NONE`,
so answer "unset it (or `0`), or debug with `--enforce-eager` **and** the env together"
and classify it as a config-normalisation raise before weight load — one grep of
`platform.py` settles it (verified at v0.26.0rc1 `f2f74a16`, recorded #16268, answered
in 19 turns from the gate alone).

**Ref-scope that gate before using it.** At vllm-ascend **v0.9.2rc1** (`b5b7e0ec`)
there is no such refusal: `rg -n ASCEND_LAUNCH_BLOCKING vllm_ascend/` finds only
`examples/run_dp_attention_etp16.sh`, and `check_ascend_config`
(`ascend_config.py:126`) has no blocking term - so a 0.9.x report with the env set
starts graph mode normally and fails *later* with a bare `RunOpApiV2 ...
AclrtSynchronizeStreamWithTimeout(stream), error code is 107027`
(`captured-stream-host-sync.md` section 6, recorded #2138). Same paste on a
0.14+/0.26 ref is this gate; on 0.9.x it is not.

The complement of this page: when the success line **is** there and capture
finished on every rank, the mode took effect and the request path is the question —
`../hccl-and-hardware/aclgraph-full-decode-runtime-hang.md` (recorded #4178).

Recorded case: #6972 (Qwen3-Omni-30B-A3B via `vllm serve --omni`, vllm-ascend
v0.14.0rc1, 8×910) — the yaml route is explained by `level: 1` at this ref; the CLI
route needs the per-stage line. The recorded session consumed the full output budget
with no answer emitted (see `../evidence-workspace/recorded-cases-startup-and-environment.md`).
