# Observability and logs: "the flag is set but nothing is printed"

Scope: the report is about what the server prints rather than about a crash — a
logging/verbosity flag that appears to do nothing (`--enable-log-requests`,
`--enable-log-outputs`, `VLLM_LOGGING_LEVEL=DEBUG`, `--max-log-len`), a stage of
the startup log the reporter expected and cannot find, or a proxy/frontend in
front of the engine that swallows output. Also the inverse question: which log
line proves which phase actually ran, and the same question one level down for a
*route*: `/start_profile` / `/stop_profile` (or any router-attached endpoint)
answering 404/500 although the reporter configured it.

Leave this topic when the server printed an error and died
(`../startup-failures/index.md`, `../hccl-and-hardware/index.md`) or when the
complaint is about numbers rather than text (`../accuracy-benchmarks/index.md`).

Inherit the rules in SKILL.md. Rules that decide success here:

1. **Establish ownership in one answer, not in one source walk.** The request/output
   logging flags are upstream vLLM OpenAI-server features; vllm-ascend adds no
   logic to them and documents none of them (`grep -rn "enable-log" docs/` at
   vllm-ascend v0.18.0 returns nothing). A vllm-ascend-specific defect is the
   *last* hypothesis here, not the first: the only downstream surface is the two
   name-keyed patches that copy `OpenAIServingChat` methods, and both keep the
   logging calls (`request-output-logging-flags.md`, "Which implementation is
   actually live").
2. **This class is a checklist plus one grep, not archaeology.** Both recorded
   runs of this topic ended with no answer at all. #10985 answered a two-line
   "flags do not print prompts in 0.18.0" report by reading
   `vllm/entrypoints/openai/**` for 15 turns / 28 tool calls, spent the full
   16384-token completion budget, and emitted 70 characters of text (an "I'll
   start by…" line). #9583 answered a `404` on `/start_profile` with 20 turns /
   30 tool calls and 47.6K reasoning characters, emitted 69 characters, and hit
   the cap on a plan block. The four facts that decide the flags case are
   `logger.py:21-30` (the two warnings), `logger.py:44` (prompt text is
   DEBUG-only), `generate/api_router.py:144-152` (completions never gets
   `enable_log_outputs`) and `cli_args.py:368` (the flag pairing gate); the
   endpoint case is one `attach_router` gate plus one `openapi.json` request
   (`online-profiling-endpoints.md`). Cap this class at ~4 tool calls, then
   answer.
3. **Ask which endpoint and which level before theorising.** "Nothing printed"
   almost always means one of: chat vs `/v1/completions`, INFO vs DEBUG, a
   `--log-config-file` / `VLLM_CONFIGURE_LOGGING=0` override, `--max-log-len`, or
   the reporter reading a different process's stdout (API server vs
   `EngineCore`) than the one they piped. Those five are the ranked answer.
4. **A log line the report quotes can be fixed wrapper text** (SKILL.md rule 11):
   before accepting a sentence as the cause, grep its emitting site.

Conditional routes:

- Request/output logging flags, `VLLM_LOGGING_LEVEL`, `--max-log-len`, whether a
  given endpoint logs outputs at all, and how to prove which
  `chat_completion_*_generator` is live in the installed tree →
  `request-output-logging-flags.md`.
- An endpoint that is simply *not there* — `POST /start_profile` /
  `/stop_profile` → `404 Not Found` after `--profiler-config '{…}'`, or a 500
  naming `Profiling is not enabled` — a route-registration question in the
  serving process (upstream `serve/profile/api_router.py` `attach_router`), not a
  model or device question: `online-profiling-endpoints.md`.
- The missing text is a *stage* of startup rather than a flag (silence between
  "Starting to load model" and the first error) → `../startup-failures/index.md`
  and, for the graph-capture stage, `../startup-failures/acl-graph-capture.md`.
- The missing text is engine-side and the reporter is on a 310P image, where the
  platform rewrites parts of the run → `../soc-platform-paths/atlas-300i-310p.md`.
