# Observability and logs: "the flag is set but nothing is printed"

Scope: the report is about what the server prints rather than about a crash — a
logging/verbosity flag that appears to do nothing (`--enable-log-requests`,
`--enable-log-outputs`, `VLLM_LOGGING_LEVEL=DEBUG`, `--max-log-len`), a `Received
request … prompt:` block the reporter *does* see on their NVIDIA/GPU box and cannot
find on the Ascend one, a stage of
the startup log the reporter expected and cannot find, a proxy/frontend in
front of the engine that swallows output, or a startup `WARNING … [envs.py:1717]
Unknown vLLM environment variable detected: VLLM_ASCEND_<NAME>` for a variable the
reporter says they never exported. Also the inverse question: which log
line proves which phase actually ran, and the same question one level down for a
*route*: `/start_profile` / `/stop_profile` (or any router-attached endpoint)
answering 404/500 although the reporter configured it. The pure capability form of the
same question — **"where do I read KV-cache size / utilization"**, `curl metrics` 有没有
相关的统计, a dashboard gauge stuck at `0` — is the same class: the numbers are two
startup lines plus `/metrics` gauges, and nothing has to be grepped
(`kv-cache-capacity-and-usage-metrics.md`). So is the **warning-only** report: a paste
whose entire content is a line the reporter quotes with no error in it, e.g.
`[WARNING] Please DO NOT tune args ['num_warps']!` on a hybrid GDN serve
(`triton-autotune-tune-args-warning.md`) — the deliverable is a verdict plus one
provenance check, and its trap is that the same block can sit directly above a real
crash.

Leave this topic when the server printed an error and died
(`../startup-failures/index.md`, `../hccl-and-hardware/index.md`) — **except the two
profiler endpoints' own crash variant**: `POST /start_profile` / `/stop_profile`
taking the process down with `!!!!!!! Segfault encountered !!!!!!!` while engine
startup, load and the request path were all healthy. That is
`online-profiler-stop-segfault.md`, a `torch_npu` patch-release lane. Leave it also
when the complaint is about the *correctness* of numbers rather than about where they
are reported (`../accuracy-benchmarks/index.md`).

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
   `vllm/entrypoints/openai/**` for 15 turns / 28 tool calls and ended the session
   having emitted one "I'll start by…" line. #9583 answered a `404` on
   `/start_profile` with 20 turns / 30 tool calls, emitted one line, and ran out
   on a plan block. The four facts that decide the flags case are
   `logger.py:21-30` (the two warnings), `logger.py:44` (prompt text is
   DEBUG-only — **at the v0.18.0 pin only**; at v0.11.0 and earlier the INFO line
   carries the prompt and `--enable-log-requests` is enough),
   `generate/api_router.py:144-152` (completions never gets
   `enable_log_outputs`) and `cli_args.py:368` (the flag pairing gate); the
   endpoint case is one `attach_router` gate plus one `openapi.json` request
   (`online-profiling-endpoints.md`). Cap this class at ~4 tool calls, then
   answer.
3. **Ask which endpoint and which level before theorising.** "Nothing printed"
   almost always means one of: chat vs `/v1/completions`, INFO vs DEBUG, a
   `--log-config-file` / `VLLM_CONFIGURE_LOGGING=0` override, `--max-log-len`, or
   the reporter reading a different process's stdout (API server vs
   `EngineCore`) than the one they piped. Those five are the ranked answer. A sixth
   is a flag the *platform* force-clears with one warning and no output:
   `--enable-layerwise-nvtx-tracing` prints `relies on NVTX (NVIDIA Tools) and is not
   supported on Ascend. Resetting to False.` (`vllm_ascend/platform.py:774-783` at
   v0.18.0rc1), so the answer is "unsupported on this backend", not a logging config.
4. **A log line the report quotes can be fixed wrapper text** (SKILL.md rule 11):
   before accepting a sentence as the cause, grep its emitting site. The inventory —
   the `ACLgraph … Recommended solutions` block and which appended code owns which
   page, the `vllm_ascend_C` tail, the CANN resource-alloc and aclop-capture wrappers,
   `WorkerProc failed to start.`, the flag/env-default checks and the `/health` route's
   own code — is `wrapper-text-vs-owner.md`; read it whenever the decisive line is
   canned text. One recorded
   instance to recognise by its wording: `skip this step for we receive the data from
   remote disaggregate prefill node` sits under `if not
   scheduler_output.total_num_scheduled_tokens:` **and** `if not
   has_kv_transfer_group():` (`vllm_ascend/worker/model_runner_v1.py:1929-1936` at
   vllm-ascend v0.11.0 `2f1aed98ccdb`), so it prints precisely when there is **no** KV
   connector and means "this step scheduled no tokens, engine idle". A paste ending in
   it is neither a PD/disaggregation report nor a lost response — recorded #5760
   answered a "curl returns empty on a 4096-token completion though the same body
   redirects to file with content" report with the client-side `-w
   '%{http_code} %{size_download}'` check and had to correct this line in a second
   message; grep the string at the claimed ref before letting a paste's own wording
   choose the track. Recheck: the guard, not the sentence, is the evidence — read the
   `if` above the log call.
5. **A warning that names its own variable is a registry fact, not a mechanism
   hunt.** `Unknown vLLM environment variable detected: VLLM_ASCEND_…` is upstream
   `envs.validate_environ` (warn-only: `fail_on_environ_validation` defaults
   `False`) firing because vllm-ascend keeps a private registry it never merges
   into `vllm.envs.environment_variables`. Three or four calls: name the emitter
   and the process prefix, give `env | grep -i <flag substring>` (or
   `/proc/<pid>/environ` — substring, because the legacy spelling that is not in
   the registry warns just the same), and split the answer into "benign for you" plus "the merge belongs in
   `AscendPlatform.pre_register_and_update`". When the reporter says their export
   is commented out, one grep of the model tutorial the block was copied from names
   the likely source (tutorial blocks export the flag uncommented; #9986). Then
   branch on the name before advising anything: if it appears in the pin's own
   *Migration Guide* table (`docs/source/user_guide/configuration/additional_config.md:6-24`
   at v0.22.1rc1 — the table is **absent** at v0.18.0rc1 and v0.19.1rc1, where
   `unset` is the right answer) it is a knob being retired, so the deliverable is
   its `--additional-config` key and the reporter keeps their value — `unset` is
   the wrong answer (`VLLM_ASCEND_ENABLE_NZ` → `{"weight_nz_mode": 0}`, #11624
   reporter-confirmed). Do not accept "it is
   not in my `collect_env.py`" as proof it is unset: `collect_env.py:304-316`
   prints only upstream-registered names plus the `TORCH`/`PYTORCH`/`ASCEND_`/`ATB_`
   prefixes.

6. **A "where is this number reported" question is answered by the inventory, not by a
   source walk.** Capacity (bytes / tokens), utilization (gauge + periodic line) and
   block residency (`--kv-cache-metrics`, off by default) are all named in the pinned
   tree; the recorded #8285 run spent 16 tool turns re-deriving them from
   `vllm/v1/core` and `vllm_ascend/worker/` while its own answer held nothing a table
   would not. Deliver the table, the one `curl`, the conversion formula and the ask for
   the startup block, and cap the class at ~3 calls
   (`kv-cache-capacity-and-usage-metrics.md`).

7. **A warning the reporter quotes as the whole report is a verdict, not a mechanism
   hunt.** `[WARNING] Please DO NOT tune args ['num_warps']!` / `['num_warps',
   'num_stages']!` on a Qwen3-Next/Coder-Next serve is the triton autotune layer
   refusing to tune those keys: the bracketed lists are upstream vLLM's two
   `@triton.autotune` config shapes (`fla/ops/l2norm.py:23` warp-only vs
   `chunk_o.py:33` warp+stages at vllm v0.14.1), vllm-ascend v0.14.0rc1 has **zero**
   `autotune` in its own tree, and the string is in **no** exported tree — it prints
   from the installed wheel, which is why it carries no timestamp and no
   `[file.py:line]` that every vLLM `logger` line has. Answer "harmless, nothing to
   change", hand over the one grep, and read the paste for an error *below* the block
   before ranking anything: #3548's identical block sits above a real
   `coreDim … UINT16_MAX` failure. Cap at ~3 calls
   (`triton-autotune-tune-args-warning.md`).

Conditional routes:

- A quoted log line or a raise site *is* the decisive evidence, and the report expects you
  to accept its wording: grep the emitting site and classify the appended error first —
  `wrapper-text-vs-owner.md` (the `ACLgraph … Recommended solutions` wrapper, the
  `vllm_ascend_C` tail, CANN's resource-alloc block, the aclop-capture guard,
  `WorkerProc failed to start.`, and why a `/health` 200 is not liveness).
- "怎么查看当前HBM上kvcache的使用情况 / 总量、已使用量", `curl metrics` 有没有 kvcache
  统计, a `/metrics` gauge stuck at `0.0` or carrying the old name
   `vllm:gpu_cache_usage_perc` → `kv-cache-capacity-and-usage-metrics.md` (the four
   surfaces with their pins, the block-pool-vs-HBM caveat, the `--disable-log-stats`
   freeze, and why `npu-smi` cannot answer it).
- Request/output logging flags, `VLLM_LOGGING_LEVEL`, `--max-log-len`, whether a
  given endpoint logs outputs at all, and how to prove which
  `chat_completion_*_generator` is live in the installed tree →
  `request-output-logging-flags.md`.
- The line prints on the reporter's NVIDIA/GPU box and not on Ascend, or the
  report asks why the default output "lost" a block it used to have: settle the
  **vllm version on both hosts** first — the switch is opt-out
  (`--disable-log-requests`) at ≤ v0.9.x and opt-in (`--enable-log-requests`,
  default False) from v0.10.x, so one line of `pip show vllm` and one
  `vllm serve --help | grep -n 'log-request'` replace the whole device theory
  (`request-output-logging-flags.md`, the polarity table).
- `Unknown vLLM environment variable detected: VLLM_ASCEND_*` — why it fires, which
  process printed it, why `collect_env.py` never shows the variable, where the
  downstream registry fix belongs, and whether the name is instead a deprecated
  knob whose `--additional-config` replacement is the deliverable:
  `env-var-validation-warning.md`.
- The number came out of an *external* profiler rather than out of a vLLM flag —
  `msprof`/MindStudio operator counts, "NPU 空闲时间 95%", an NPU-vs-GPU operator
  compare: that is not this topic's route and not vLLM's `/start_profile`, it is
  collection validity plus per-step normalization —
  `../performance-throughput/profiler-opcount-evidence.md`.
- The collection produced **no table at all** (`kernel_details.csv` absent) on a
  graph-mode + MTP launch: a capability/regression question, decided by the analysed
  directory's file listing and then by a version bisect, not by an op-count argument —
  `../performance-throughput/profiler-no-kernel-details-graph-mtp.md`.
- An endpoint that is simply *not there* — `POST /start_profile` /
  `/stop_profile` → `404 Not Found` after `--profiler-config '{…}'`, or a 500
  naming `Profiling is not enabled` — a route-registration question in the
  serving process (upstream `serve/profile/api_router.py` `attach_router`), not a
  model or device question: `online-profiling-endpoints.md`.
- `POST /start_profile` answers 200 but `<torch_profiler_dir>` is empty, holds an
  unexpected layout, or the reporter wants to know which path it even is: the relative
  dir from the guide is rewritten to an absolute one (`vllm/config/profiler.py:144-145`),
  one named subdirectory per rank/API-server writer appears only after `/stop_profile`,
  and the loud variants are `torch_profiler_dir cannot be empty.` /
  `MSMONITOR_USE_DAEMON` — `online-profiling-endpoints.md`, the `torch_profiler_dir`
  section. "Collected but `kernel_details.csv` absent" stays on
  `../performance-throughput/profiler-no-kernel-details-graph-mtp.md`.
- The endpoint answers and then the process **dies**: `/start_profile` or
  `/stop_profile` → `!!!!!!! Segfault encountered !!!!!!!` /
  `Worker proc … died unexpectedly (exit code: None)` /
  `Invocation of profile method failed`, and the body says another `torch_npu`
  patch level works — one cross-patch control plus the release's own
  `torch-npu==` pin decide it; do not walk the profiler source:
  `online-profiler-stop-segfault.md`.
- Warnings only, no error anywhere — `[WARNING] Please DO NOT tune args ['num_warps']!`
  / `['num_warps', 'num_stages']!` per worker on a hybrid GDN model whose serve is
  returning 200s → `triton-autotune-tune-args-warning.md` (benign; the provenance
  grep; and #3548, the case where the same block precedes a real grid-limit crash).
- The missing text is a *stage* of startup rather than a flag (silence between
  "Starting to load model" and the first error) → `../startup-failures/index.md`
  and, for the graph-capture stage, `../startup-failures/acl-graph-capture.md`.
- The missing text is engine-side and the reporter is on a 310P image, where the
  platform rewrites parts of the run → `../soc-platform-paths/atlas-300i-310p.md`.
