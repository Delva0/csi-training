# `--enable-log-requests` / `--enable-log-outputs` print nothing

Verified against vllm **v0.18.0** (`bcf2be9612`) and vllm-ascend **v0.18.0**
(`e18643f8a4d5`), the pair in #10985. Line numbers move between releases; the
gates, not the numbers, are the reusable part.

## The gates, in the order they fail

| Gate | Code at v0.18.0 | What the reporter sees |
|---|---|---|
| `--enable-log-outputs` without `--enable-log-requests` | `vllm/entrypoints/openai/cli_args.py:368-369` raises `TypeError: Error: --enable-log-outputs requires --enable-log-requests` | the server never starts; if they claim it runs, this is not the gate |
| Neither flag present on the **serving** command | `vllm/engine/arg_utils.py:2208` — `enable_log_requests` defaults **False**; `vllm/entrypoints/openai/api_server.py:339-342,435-438` build `RequestLogger` only when it is truthy | zero request lines; flags exported in a wrapper script or set on a proxy process do nothing |
| Effective level above INFO | `vllm/entrypoints/logger.py:21-26` prints `warning_once`: `` `--enable-log-requests` is set but the minimum log level is higher than INFO. No request information will be logged.`` | that warning is the single decisive grep. **Search the log for it** before any source work |
| Level INFO but not DEBUG | `logger.py:27-33` `info_once`: "…minimum log level is higher than DEBUG. Only limited information will be logged…" | request IDs/params and generation text appear; the **prompt text does not** |
| Prompt text expected at INFO | `logger.py:44,53` — the `prompt:`/`prompt_token_ids:` detail line is inside `if logger.isEnabledFor(logging.DEBUG)`; `logger.py:63` "Received request …" is INFO | "no prompt printed" at INFO is **correct behaviour**, not a bug: ask for `VLLM_LOGGING_LEVEL=DEBUG` |
| `--log-config-file` / `VLLM_LOGGING_CONFIG_PATH`, or `VLLM_CONFIGURE_LOGGING=0` | `vllm/logger.py:187-203` replaces the whole dict config; `vllm/envs.py:39-43` (defaults `True` / `INFO`) | vllm loggers lose their handler and INFO is dropped silently; a custom JSON config must define a `vllm` logger at the wanted level |
| `--max-log-len N` | `vllm/entrypoints/openai/cli_args.py:135` (`int \| None`, None = no truncation), applied at `logger.py:45-51,80-87` | prompt/output cut to N characters; `--max-log-len 0` prints empty values while every other line is normal |
| Endpoint does not log outputs at all | `vllm/entrypoints/openai/generate/api_router.py:144-152` constructs `OpenAIServingCompletion(...)` **without** `enable_log_outputs`; `completion/serving.py` contains no `log_outputs` call. Chat (`chat_completion/serving.py:122,291,1045,1245,1626`), `/v1/responses` (`responses/serving.py:881-889`) and `ServingTokens` (`generate/api_router.py:177-183`) do take the flag | `/v1/completions` **can never** print generations at v0.18.0 — say so; the reporter must re-test on `/v1/chat/completions`. `enable_log_deltas` (default True, `cli_args.py:149-152`) additionally decides whether streaming logs per-delta or only the complete text |

Two environmental traps worth one line each: vllm-ascend's own CI exports
`VLLM_LOGGING_LEVEL: ERROR` (`.github/workflows/_e2e_test.yaml:41` and five more
job blocks at v0.18.0), so a launch script copied from CI or from
`examples/external_online_dp/run_dp_template.sh:5` (`info`) starts one level too
quiet; and vllm-ascend documents no `--enable-log*` flag anywhere in `docs/`, so
the upstream flag docs are the only authority for the reporter's version.

## Which implementation is actually live (the only downstream surface)

Two vllm-ascend patches touch this path, both imported unconditionally in
`vllm_ascend/patch/platform/__init__.py` (v0.18.0, `:30-31`):

- `patch_glm_tool_call_parser.py:1083` **replaces** `OpenAIServingChat.chat_completion_stream_generator`
  with a frozen copy for every model. The copy keeps the logging blocks
  (`:612,630` per-delta, `:794,801` final), so it should still log — but it is a
  fork of an upstream method, which is exactly the failure mode of
  `../upstream-downstream-drift/index.md`.
- `patch_minimax_usage_accounting.py:249` **wraps**
  `OpenAIServingChat.chat_completion_full_generator`; the wrapper delegates to the
  saved original (`:203`), so the upstream `enable_log_outputs` block survives.

One command tells the reporter which copy is running, and it is worth more than
any amount of reading:

```bash
python -c "import inspect; from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat as C; print(inspect.getsourcefile(C.chat_completion_stream_generator), inspect.getsourcefile(C.chat_completion_full_generator))"
```

A path under `vllm_ascend/patch/` for either method means the frozen copy is
live; a path under `vllm/entrypoints/openai/` means upstream's. If the live copy
is vllm-ascend's **and** its logging block looks different from upstream's at the
installed vllm version, that is a real drift finding — otherwise this family is
configuration and the answer is the checklist above.

## Deliverable shape for this class (keep it under ~2.5K characters)

1. One sentence: not an Ascend defect at v0.18.0; these are upstream frontend
   flags, and N of the gates below explain it.
2. The two grep commands that settle it on the reporter's host:
   `grep -n "minimum log level is higher" serve.log` and
   `grep -n "Received request" serve.log | head`. Their answers pick the gate.
3. The endpoint caveat (`/v1/completions` cannot log outputs) and the DEBUG
   caveat (prompt text is DEBUG-only) as corrections the reporter can act on now.
4. What to send back: the exact `vllm serve` line, the client call (path +
   `stream=`), `pip show vllm vllm-ascend`, and the first 200 lines of the API
   server's stdout. #10985 had **no** environment block at all, so the vllm
   version the flags were parsed with is unknown — say that.

## Limits

- Everything except the `inspect.getsourcefile` check is upstream-owned code; a
  maintainer's "works here" reply does not refute a gate failure on the
  reporter's host, and a working chat endpoint does not imply completions logs.
- `--enable-log-outputs` is spelled with a trailing `s`; report the exact flag
  the reporter typed. Do not claim FlexibleArgumentParser accepts abbreviations
  unless the installed version's parser source says so.
