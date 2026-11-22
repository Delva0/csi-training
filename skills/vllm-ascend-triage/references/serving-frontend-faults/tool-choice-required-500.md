# `tool_choice: "required"` (or a named function) 500s while `"auto"` answers

## Signature

One body, two outcomes, told apart by a single field: `POST /v1/chat/completions`
carrying `tools` **plus** `tool_choice: "required"` (or
`{"type":"function","function":{"name":<one of tools>}}`) returns `500 Internal
Server Error`, while the same body with `"auto"` answers normally. On a P/D
deployment the paste is usually only the **proxy's** view — `httpx.HTTPStatusError:
Server error '500 Internal Server Error' for url 'http://<node>:9000/v1/chat/…'`,
`Attempt 2/3 failed …`, `All 3 attempts failed`, `Error occurred in disagg prefill
proxy server - /chat/completions endpoint`, then an `ERROR: Exception in ASGI
application` traceback that stops inside starlette — and the engine log shows
`Delaying free of N blocks` lines and no traceback at all. The reporter's question
is the right one: *what changes between `required` and `auto` in the framework?*

## Step 0 — the proxy already discarded the cause

`load_balance_proxy_server_example.py`'s `send_request_to_service` does
`response.raise_for_status()`, so the upstream 500 **body** — the only place the
EngineCore traceback lives — is thrown away and only `str(exc)` survives. The ASGI
traceback in the paste is the proxy's own, not the engine's. At v0.17.0rc1 the file
is `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
(`send_request_to_service` `:576`, `raise_for_status` `:608`/`:632`,
`_handle_select_instance` `:659`, `_handle_completions` `:710`), and #7719's paste
carries `:608` unchanged — the drift on the later frames (`:667`/`:716`) is a few
lines of the reporter's own copy, so do not spend a turn on it. The single
cheapest step, and the one to request/perform first, is to POST the identical body
straight at the node port the proxy was dialling (`:9000` in that traceback) and
read the response body plus that instance's APIServer log. Everything below is what
you can already say **before** that body arrives.

## Step 1 — what `required` adds that `auto` never does

Source-verified at vllm v0.17.0 (`b31e9326`) — the pin of the recorded case — on
the chat-completions path:

1. `vllm/entrypoints/openai/engine/serving.py:996-1008`: **only** when a tool parser
   is configured and `tool_choice != "none"`, the renderer hands the request to
   `tool_parser(tokenizer).adjust_request(request)`. No parser flag ⇒ no change at
   all, and a `required` request would answer like a plain one — so a
   `required`-only 500 also tells you the serve line *has* `--tool-call-parser` /
   `--enable-auto-tool-choice`, which the pinned GLM-5 tutorial's serve blocks do
   not pass).
2. `vllm/tool_parsers/abstract_tool_parser.py:56-81` (`adjust_request`): if a schema
   comes back, the request silently acquires
   `structured_outputs = StructuredOutputsParams(json=<schema>)` and
   `response_format = None` (Responses API: `text.format` = `json_schema`,
   `strict=True`).
3. `vllm/tool_parsers/utils.py:181-215` decides that schema: `"required"` → an
   **array** schema `{"type":"array","minItems":1,"items":{"type":"object",
   "anyOf":[<per-tool>…]}}` (`:164-178`); a named function → that tool's own
   `parameters`; `"auto"` / `"none"` / `None` → `None`. `_get_tool_schema_from_tool`
   (`:133-142`) wraps each tool as `{properties:{name:{type:string,enum:[name]},
   parameters:<params>}, required:[name,parameters]}`, normalizes an empty
   `parameters: {}` to `{"type":"object","properties":{}}`, and merges `$defs` from
   the parameters — so an empty-schema tool is **not** the defect.
4. With the field set the engine compiles the grammar and computes a bitmask per
   step: `vllm/v1/structured_output/request.py:89-90` (`STRUCTURAL_TAG` /
   `JSON_SCHEMA` selection) and `vllm/v1/structured_output/backend_xgrammar.py:104-106,338-354`
   (`compile_structural_tag` / `Grammar.from_structural_tag`).
5. After generation the frontend parses with a **hard-fail** `required` branch,
   present twice at this pin: `engine/serving.py:1211-1223`
   (`EngineBase._parse_tool_calls_from_content`, the chat-completions call site
   `chat_completion/serving.py:1515`) and `parser/abstract_parser.py:423-435`
   (`Parser._parse_tool_calls`, the Responses / `DelegatingParser` path). Both do
   `assert content is not None` then
   `TypeAdapter(list[FunctionDefinition]).validate_json(content)`. The `auto` branch
   below them hands the text to the model's own tool parser and keeps prose as
   content when nothing parses — it cannot raise this way.

So the contrast is reproduced by (2)+(3) alone: *the same body acquires a
structured-output config, and the reply acquires a validating parser.* The tool
parser, the PD transport and the device are not implicated by construction — say
that in the first sentence, then rank.

## Step 2 — the lanes, one check each

1. **The grammar did not constrain the reply** (fallback backend, bitmask never
   applied, or `response_format` clobbered later): the model emits its native tool
   markup/prose and step 5 raises. Check: the node's 500 body; then the
   `maybe_backend_fallback` warning sequence in that process's log — read the
   **last** assignment (`guided-decoding-backend-aarch64.md`), not the first.
2. **A reasoning split left `content` empty**: a GLM-family serve with
   `--reasoning-parser` + `"stream": true`, where everything lands in
   `reasoning_content` and `assert content is not None` fires. Check: same body with
   thinking off (`chat_template_kwargs`) or `"stream": false`; a 200 with a
   tool-free reply moves the case to lane 1.
3. **Schema shape**: the one unusual feature of the generated schema is the `anyOf`
   items having no `"type": "object"` of their own (the parent supplies it) plus
   `$defs` merged from parameters. Check: a body whose only tool declares a plain
   `{"type":"object","properties":{…}}` and whose `tool_choice` is *named* (simpler
   schema = `parameters` only). Still 500 ⇒ not the schema.
4. **PD-only, spec decode / recompute scheduler on the D node**: the grammar matcher
   is handed a `-1` placeholder token and asserts in `get_grammar_bitmask`
   (`../pd-disaggregation/structured-output-draft-placeholder-assert.md`, #6421).
   Check the P/D command diff (spec config, `recompute_scheduler_enable`) before
   anything else on a 2P+1D.

## Step 3 — pins, dependency drift, coverage

- `xgrammar` is pinned **exactly** at vllm v0.17.0 for aarch64 too
  (`requirements/common.txt:27`, `xgrammar == 0.1.29; platform_machine == "x86_64"
  or "aarch64" or …`), while vllm-ascend v0.17.0rc1 requires `xgrammar>=0.1.30`
  (`requirements.txt:19`) — a floor *above* upstream's pin. `pip show xgrammar`
  inside the serving image is therefore a real check, and the old "#1592: aarch64 has
  no xgrammar wheel" story no longer applies on these pins.
- Coverage: vllm-ascend ships `tests/e2e/singlecard/test_guided_decoding.py` only,
  so the `required`/named `tool_choice` path under PD or multi-card is unvalidated
  at v0.17.0rc1 — a valid "coverage gap" sentence for the answer, not a cause.
- The scheduler-fork route (`RuntimeError: Invalid request status: WAITING_FOR_FSM`
  from `vllm_ascend/core/scheduler.py`) needs the **older** pins: at v0.17.0rc1
  `vllm_ascend/core/` holds only `recompute_scheduler.py` and
  `scheduler_dynamic_batch.py`, so a `required`-only 500 at this pin is not the fork
  (`../upstream-downstream-drift/scheduler-fork-missing-status.md` keeps that shape).

## Recorded case

#7719 — vllm-ascend v0.17.0rc1 (`e20f0b1a`) + vllm v0.17.0 (`b31e9326`), GLM-5
A3 "大EP" **2P+1D** behind the docs' `load_balance_proxy_server_example.py`, `auto`
fine / `required` 500, one tool declaring `"parameters": {}`, `"stream": true`. The
run routed through `parallel-state-and-kv.md` first (an older-pin mechanism),
re-read the pin's `vllm_ascend/core/` to retire it, then verified steps 1-3 above and
delivered the ranked answer with "bypass the proxy for the 500 body" as the one
required step. Citation slip worth not repeating: it named
`parser/abstract_parser.py:423-435` as *the* `required` branch, while the
chat-completions path at this pin is `engine/serving.py:1211-1223`; grep both copies
before quoting.

## Scope and recheck

Source-verified at vllm v0.17.0 `b31e9326a7d9` / vllm-ascend v0.17.0rc1
`e20f0b1a0d2f`; nothing here was executed (no NPU in the case view). Recheck on a
newer pin: `rg -n 'def adjust_request' vllm/tool_parsers/abstract_tool_parser.py
vllm/tool_parsers/*_tool_parser.py` — a parser that stops calling
`super().adjust_request()` (or an upstream that moved the injection into the
protocol model) removes step 1; and `rg -n 'tool_choice == "required"'
vllm/entrypoints vllm/parser` — if only one copy remains, cite that one.
