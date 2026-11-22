# `--enable-log-requests` / `--enable-log-outputs` print nothing

Verified against vllm **v0.18.0** (`bcf2be9612`) and vllm-ascend **v0.18.0**
(`e18643f8a4d5`), the pair in #10985; the default-flip below against the two vllm
pins the snapshot does export, **v0.9.1** (`b6553be1bc75`) and **v0.11.0**
(`b8b302cde434`). Line numbers move between releases; the gates, not the numbers,
are the reusable part.

## The default flipped mid-stack: one serve line prints on a 0.9.x box and not on 0.10.x+

**Two boxes with two vllm versions is not a device contrast.** `RequestLogger` is
pure Python in the OpenAI frontend (`vllm/entrypoints/logger.py`) with no device
code, so when the reporter shows an NVIDIA log carrying `Received request … prompt:`
and an Ascend log that has none, the deciding variable is the *vllm* version on each
host — ask `pip show vllm | head -2` for both before any CANN/SoC/vllm-ascend theory,
and read the polarity off the flag list with
`vllm serve --help | grep -n 'log-request'`.

| vllm | Switch | Default | Prompt text at INFO? |
|---|---|---|---|
| ≤ v0.9.x | `--disable-log-requests` (`engine/arg_utils.py:1614`, `disable_log_requests: bool = False`; CLI at `:1625`) | request logging **ON**: `openai/api_server.py:1133-1136` builds `RequestLogger` *unless* the flag is set | **yes** — `entrypoints/logger.py:43-46` is `logger.info("Received request %s: prompt: %r, params: %s, …")` |
| v0.10.x – v0.11.x | `--enable-log-requests` (`arg_utils.py:1688` at v0.11.0, `enable_log_requests: bool = False`); v0.11.0 keeps `disable_log_requests` only as a **deprecated** property/setter that negates it (`:1691-1704`, "will be removed in v0.12.0") | **OFF**: `api_server.py:1617-1620` builds the logger only when truthy | **yes** — same INFO line, `logger.py:40-45`, and `isEnabledFor(logging.DEBUG)` occurs nowhere in that file |
| v0.18.0 (the pin above) | `--enable-log-requests` | OFF | **no** — the `prompt:`/`prompt_token_ids:` detail moved inside `if logger.isEnabledFor(logging.DEBUG)` (`logger.py:44,53`); INFO keeps the short line |

The flip is therefore bounded to **v0.10.x**, and vllm-ascend `v0.10.2rc1`
(`048bfd5`) pairs with **vllm v0.10.2** by its own `docs/source/conf.py:68,76`
(`vllm_version` / `pip_vllm_version`) — a 0.10.x reporter comparing against a GPU
box on an older vllm sees exactly this difference and nothing else. So do **not**
answer the 0.9.x/0.10.x generation with "the prompt text is DEBUG-gated": that gate
appears only at the newer pins, and `VLLM_LOGGING_LEVEL=DEBUG` there buys nothing.
The `--enable-log-outputs`-requires-`--enable-log-requests` gate is already present
at v0.11.0 (`cli_args.py:280-281`), so it is not diagnostic of the version.

## The gates, in the order they fail

| Gate | Code at v0.18.0 | What the reporter sees |
|---|---|---|
| `--enable-log-outputs` without `--enable-log-requests` | `vllm/entrypoints/openai/cli_args.py:368-369` raises `TypeError: Error: --enable-log-outputs requires --enable-log-requests` | the server never starts; if they claim it runs, this is not the gate |
| Neither flag present on the **serving** command | `vllm/engine/arg_utils.py:2208` — `enable_log_requests` defaults **False**; `vllm/entrypoints/openai/api_server.py:339-342,435-438` build `RequestLogger` only when it is truthy | zero request lines; flags exported in a wrapper script or set on a proxy process do nothing |
| Effective level above INFO | `vllm/entrypoints/logger.py:21-26` prints `warning_once`: `` `--enable-log-requests` is set but the minimum log level is higher than INFO. No request information will be logged.`` | that warning is the single decisive grep. **Search the log for it** before any source work |
| Level INFO but not DEBUG (**v0.18.0-era only** — at v0.11.0 and earlier the INFO line already carries the prompt, see the polarity table) | `logger.py:27-33` `info_once`: "…minimum log level is higher than DEBUG. Only limited information will be logged…" | request IDs/params and generation text appear; the **prompt text does not** |
| Prompt text expected at INFO (**v0.18.0**; the same line prints the prompt at INFO on ≤ v0.11.0, so apply this row only once the version is known) | `logger.py:44,53` — the `prompt:`/`prompt_token_ids:` detail line is inside `if logger.isEnabledFor(logging.DEBUG)`; `logger.py:63` "Received request …" is INFO | "no prompt printed" at INFO is **correct behaviour at that pin**, not a bug: ask for `VLLM_LOGGING_LEVEL=DEBUG` |
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
3. The version answer first when the contrast is another machine: which vllm each
   box runs and which switch its generation takes (the polarity table), then the
   endpoint caveat (`/v1/completions` cannot log outputs) and — only on the newer
   pins — the DEBUG caveat for prompt text.
4. What to send back: the exact `vllm serve` line, the client call (path +
   `stream=`), `pip show vllm vllm-ascend`, and the first 200 lines of the API
   server's stdout. #10985 had **no** environment block at all, so the vllm
   version the flags were parsed with is unknown — say that.

## Limits

- Everything except the `inspect.getsourcefile` check is upstream-owned code; a
  maintainer's "works here" reply does not refute a gate failure on the
  reporter's host, and a working chat endpoint does not imply completions logs.
- The polarity table's v0.10.x row is bounded, not read: v0.9.1 and v0.11.0 are
  exported and v0.10.2 is not, so the flag name for the reporter's exact `0.10.2`
  is inferred from the deprecation shim's wording — have them run the
  `--help | grep log-request` probe before quoting it back as verified.
- `--enable-log-outputs` is spelled with a trailing `s`; report the exact flag
  the reporter typed. Do not claim FlexibleArgumentParser accepts abbreviations
  unless the installed version's parser source says so.
