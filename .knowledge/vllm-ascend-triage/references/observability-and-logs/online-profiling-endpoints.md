# Online profiling: `POST /start_profile` answers 404

Scope: the reporter enabled profiling and the endpoint is not there — `curl -X
POST http://host:port/start_profile` returns `404 Not Found` while the server is
otherwise healthy, or the request returns 500, or it returns 200 and no trace
appears under `torch_profiler_dir`. The offline form (`LLM(...)`,
`llm.start_profile()`, `profiler_config=` in Python) shares the config gates but
has no HTTP layer: skip the route half of this page for it.

**Not this page: the endpoint answers and then the process dies** —
`!!!!!!! Segfault encountered !!!!!!!`, `Worker proc … died unexpectedly (exit code:
None)`, `Invocation of profile method failed` — that is a `torch_npu` patch-release
lane, not a route gate: `online-profiler-stop-segfault.md`.

Verified against vllm **v0.18.0** (`bcf2be9612`) and vllm-ascend **v0.18.0rc1**
(`99e1ea0fe6`), the pair in #9583. Line numbers move between releases; the gates
do not.

## Answer rule

A 404 here is a **route-registration fact**, not a model, device, quantization or
NPU problem. Upstream vLLM registers `/start_profile` and `/stop_profile` only
when the API-server process's own `args.profiler_config.profiler` is not `None`.
So the deliverable is two lines: name that gate with its `path:line`, then the one
request that lists the live paths. Three or four calls, then answer.

## The gate at v0.18.0

| # | Site | Fact |
|---|---|---|
| 1 | `vllm/entrypoints/serve/profile/api_router.py:14,21-30` | `router = APIRouter()` carrying `POST /start_profile` / `POST /stop_profile`; both call `request.app.state.engine_client.start_profile()` |
| 2 | `.../profile/api_router.py:37-46` | `attach_router(app)`: `profiler_config = getattr(app.state.args, "profiler_config", None)`; `assert profiler_config is None or isinstance(profiler_config, ProfilerConfig)`; `include_router(router)` **only if** `profiler_config is not None and profiler_config.profiler is not None`, next to a `warning_once` ("Profiler with mode '%s' is enabled in the API server…") |
| 3 | `vllm/entrypoints/serve/__init__.py:12-29` | `register_vllm_serve_api_routers(app)` calls `attach_profile_router(app)` unconditionally. `VLLM_SERVER_DEV_MODE` (`:13-17`) only prints a security warning — it is **not** a gate for this route |
| 4 | `vllm/entrypoints/openai/api_server.py:165-190`, `:558-577` | `build_app(args, supported_tasks)` sets `app.state.args = args` and immediately calls `register_vllm_serve_api_routers(app)`; `build_and_serve` calls `app = build_app(...)`; `run_server` (`:649-656`) reaches it from the CLI |
| 5 | `vllm/entrypoints/cli/serve.py:118`, `:234+` | single API server: `uvloop.run(run_server(args))` with the **raw argparse Namespace**. `--api-server-count > 1` / DP>1 go through `run_multi_api_server` instead, so "which process answers the port" is worth one question |
| 6 | `vllm/engine/arg_utils.py:1289`, `:247-295` | `--profiler-config` is generated from `get_kwargs(VllmConfig)`; `ProfilerConfig` is a `@config` pydantic dataclass, so `_compute_kwargs` assigns it `type=parse_dataclass` = `TypeAdapter(ProfilerConfig).validate_json`. The CLI JSON is a **ProfilerConfig instance during parsing** |
| 7 | `vllm/engine/arg_utils.py:620-635` | `EngineArgs.__post_init__` converts only `compilation_config`, `attention_config`, `kernel_config`, `eplb_config`, `weight_transfer_config`. `profiler_config` is not among them, so "it was still a dict when the app was built" is dead: a dict would trip the `assert isinstance` in row 2 — an AssertionError at startup, not a 404 |
| 8 | `grep -rn "TORCH_PROFILER_DIR"` at v0.18.0 → no match | the env-var path is gone. An env-only `VLLM_TORCH_PROFILER_DIR=…` cannot register the route and produces exactly this 404 (vllm-ascend records the deprecation in `docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:50`) |
| 9 | vllm-ascend v0.18.0rc1: `grep -rn "include_router\|start_profile" vllm_ascend/` → nothing; `vllm_ascend/worker/worker.py:528-535` | downstream adds no route. Its only surface is the worker-side `RuntimeError("Profiling is not enabled. Please set --profiler-config …")`, raised before the profiler is created |

Row 9 is the sharper discriminator: **a missing config on the worker side is a
500 carrying that sentence, never a 404.** A 404 therefore means the route was
not in the app at all — the process that answered was built without
`profiler_config`.

The documented command the reporter is copying is
`docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:35-59`
(`python3 -m vllm.entrypoints.openai.api_server … --profiler-config '{"profiler":
"torch", "torch_profiler_dir": …}'` then `curl -X POST …/start_profile`), so a
404 is not a doc/flag mismatch on its face — the question is which process
received the flag.

## Cheapest checks, in order

1. Ask the live app which paths it has — one request, no restart:

   ```bash
   curl -s http://<host>:<port>/openapi.json \
    | python -c "import json,sys; print([p for p in json.load(sys.stdin)['paths'] if 'profile' in p])"
   ```

   `[]` ⇒ route absent, continue below. A list containing `/start_profile` ⇒ the
   registration is fine and the request went elsewhere (proxy, another port,
   `root_path`, a stale client URL).
2. Ask the startup log for the gate's own line: `Profiler with mode 'torch' is
   enabled in the API server. This should ONLY be used for local development!`
   (row 2, `warning_once`). Present ⇒ the flag reached the serving process.
   Absent ⇒ it did not: wrapper script, a different container, `docker run`
   quoting, a `--config` file, or an env-only command from an older guide.
3. Remove shell-quoting doubt with the dotted form, which
   `FlexibleArgumentParser` flattens back into one JSON value
   (`vllm/utils/argparse_utils.py:300-360`):
   `--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/abs/dir`.
   Route appears ⇒ the original JSON was mangled by the launcher, not by vLLM. The
   absolute form is here only so the two launches differ in one visible token — a
   *relative* dir is valid and gets rewritten (next section), so never present it as
   the repair.
4. Confirm the installed vllm is the reported one, and that the gate file you read
   is the live one: `python -c "import vllm; print(vllm.__version__)"` and
   `python -c "import vllm.entrypoints.serve.profile.api_router as m; print(m.__file__)"`.
   Older vLLM releases gated this route on `VLLM_TORCH_PROFILER_DIR` instead
   (unverified here — no pre-0.18 tree in the snapshot); a mixed install is how a
   reporter ends up with that behaviour while quoting 0.18.0.
5. Only then suspect a proxy in front: two vllm-ascend example proxies define
   their own `POST /start_profile` and fan it out
   (`examples/disaggregated_encoder/disagg_epd_proxy.py:676`,
   `examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py:797`).

## When the route answers: `torch_profiler_dir` and where the files land

A 200 from `/start_profile` moves the question from the route to the directory. Three
facts decide it at this pin:

- **A relative `torch_profiler_dir` is legal and is rewritten to an absolute one.**
  `ProfilerConfig._validate_profiler_config` (`@model_validator(mode="after")`,
  `vllm/config/profiler.py:126-147`) raises `ValueError("torch_profiler_dir must be set
  when profiler is 'torch'")` for an empty value (`:139-140`), then replaces every
  non-URI value with `os.path.abspath(os.path.expanduser(...))` (`:144-145`;
  `_is_uri_path` at `:19` exempts `gs://`-style paths from the rewrite). So
  `./vllm_profile` is resolved against the cwd of the process that parsed
  `--profiler-config`, and the workers are handed that absolute string through the
  engine config — "the worker's cwd is different, so the trace landed elsewhere" is not
  a failure mode here, and the field docstring's *"must be an absolute path"*
  (`:43-46`) is satisfied by the rewrite, not by the reporter. Both shipped examples use
  the relative form (vllm-ascend
  `docs/source/developer_guide/performance_and_debug/service_profiling_guide.md:45`,
  upstream `docs/contributing/profiling.md:46` and
  `examples/offline_inference/simple_profiling.py:26`), so a reporter who copied the
  guide has nothing to fix at that line. **Do not hand back "switch to an absolute
  path"** — that was a delivered answer's mistake (recorded run below).
- **One named subdirectory per writer, not one file at the top.** vllm-ascend names the
  NPU trace by rank: `vllm_ascend/worker/worker.py:538-545` builds `trace_name` from
  `get_worker_rank_suffix(global_rank=self.rank)` and passes it as
  `torch_npu.profiler.tensorboard_trace_handler(dir, worker_name=trace_name)`
  (`:626`), so TP/PP ranks write sibling directories instead of overwriting each
  other. In front of them the API server's own CPU trace goes to the *same* directory
  as `worker_name = f"{socket.gethostname()}_{pid}.async_llm"`
  (`vllm/v1/engine/async_llm.py:186-206`), and is skipped entirely when
  `"ignore_frontend": true` (`:186-188`). Ask for `ls -R <torch_profiler_dir>` (or
  `find <dir> -maxdepth 2`), never for "the trace file".
- **Nothing is written before `/stop_profile`.** The worker's `profile(is_start=False)`
  branch only calls `self.profiler.stop()` (`worker.py:551-555`); the export happens
  when torch/torch_npu's `tensorboard_trace_handler` reaches trace-ready, i.e. at stop,
  and the guide says the same in prose (`…service_profiling_guide.md:70`, "stop the
  profiling service, and the data will be automatically saved"). A listing taken
  between start and stop is *expected* to be empty: `curl -X POST …/stop_profile`, then
  list again. If that stop logs
  `Profiler was not started, nothing to stop.` (`worker.py:552-554`), the start never
  reached that worker.

The two config errors in this lane are loud, not silent — both are `RuntimeError`s off
the worker's `_create_profiler` and surface as a 500 at `/start_profile`:
`torch_profiler_dir cannot be empty.` (`worker.py:596-597`) and
`MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.`
(`:598-599`, i.e. `MSMONITOR_USE_DAEMON=1` in the launch env). "Collected, but
`kernel_details.csv` absent" is a different question — that directory has to be
*analysed* before the CSV exists: `../performance-throughput/profiler-no-kernel-details-graph-mtp.md`.

Model class, quantization, TP and NPU health cannot cause this symptom:
`build_app` sets `state.args` and registers the serve routers before any
`supported_tasks` branch (`api_server.py:186-190` compares with the
`"generate" in supported_tasks` blocks that follow). Say so explicitly — the
recorded issue title blames "qwen3.5系列模型", and the route has no model input at
all.

## What to ask back

Verbatim serve command (quoting included), `pip show vllm vllm-ascend`, the first
~100 startup lines (the `Profiler with mode` line lives there), and the
`openapi.json` path list. Once the route answers, add `ls -R <torch_profiler_dir>`
taken *after* a `/stop_profile`, plus the absolute path the server logged for it.

## Recorded run: #9583 (vllm 0.18.0, vllm-ascend v0.18.0rc1, Qwen3.5, 910B2)

Reporter: start with `--profiler-config '{"profiler": "torch",
"torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}'`,
`curl -X POST http://localhost:8000/start_profile` → `404 Not Found`, no
profiling data. No startup log, no response body, no full serve command.

The run had the gate (rows 2-4), the downstream 500-vs-404 discriminator (row 9)
and the `openapi.json` check by its 5th assistant turn, then spent the rest on the
CLI-parsing question in row 6-7 ("is `args.profiler_config` still a dict?") and
ended the session on a long *plan* block: 20 turns, 30 tool calls, a one-line
message, **no answer**. Rows 6-7 are
one `sed -n '247,300p' vllm/engine/arg_utils.py` away and kill that whole thread;
`../evidence-workspace/delivery-discipline.md` has the pattern.

A later rerun on this same knowledge (5 assistant turns, 13 messages) finally delivered —
full ranked answer on its 5th assistant turn, no source archaeology — but its action
list closed with *"`torch_profiler_dir` 文档要求绝对路径 … 你用的 `./vllm_profile` 是相对路径"*
and told the reporter to change it. That is wrong, and it is the one thing the run got
wrong: the value is rewritten at `config/profiler.py:144-145` and the vllm-ascend guide
it was quoting uses `./vllm_profile` itself. The dir section above exists so the next run
spends that line on `ls -R` / `/stop_profile` instead.

Answer this shape as: (a) the symptom is route registration only, (b) the gate,
(c) the two commands of check 1-2, (d) the ranked ways the flag can miss the
process (wrapper/env-only/multi-API-server/proxy/version mismatch), each with its
check, (e) the gap list. That is a complete deliverable and it fits in ~2K
characters.
