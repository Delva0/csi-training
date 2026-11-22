# Faults owned by the serving frontend or the image's Python stack

## Scope

The report carries a Python traceback or error line that surfaces in the OpenAI
server process but whose frames are **only** `vllm/**` and the installed Python
packages: no `vllm_ascend/**` frame, no `npu_*` / `aclnn` / `EZ…` / `HCCL` text,
no plog. Recorded twice: `RuntimeError: Already borrowed` from a transformers fast
tokenizer while the chat path builds a tool parser — #7197 at vllm v0.16.0 through the
non-streaming preprocessing wrapper, #7609 at vllm v0.17.0 through
`chat_completion_stream_generator` (`Error in tool parser creation.`).

The deliverable is an ownership verdict plus the cheapest client-side A/B. The
NPU stack cannot own a raise that never enters it — say that in the first
sentence, then give the mechanism at the pinned upstream files.

Second shape in the same topic: **the cause is a request field, and no frame from
the failing process is in the paste at all** — the same body 500s with
`tool_choice: "required"` (or a named function) and answers with `"auto"`, and a P/D
proxy has already thrown the upstream body away (`raise_for_status()`), leaving an
`httpx.HTTPStatusError`, `All 3 attempts failed` and a starlette-truncated ASGI
traceback. The frontend is still the owner: `required`/named make it inject
`structured_outputs` and then hard-validate the reply, which `auto` never does →
`tool-choice-required-500.md`. Ask the node for its own 500 body, and answer the
mechanism meanwhile.

Not this topic, leave immediately:

- A Python `AttributeError`/`TypeError` from an Ascend replacement of an upstream
  class, or raised inside an upstream file *on a class vllm-ascend replaces* →
  `../upstream-downstream-drift/index.md`.
- A pydantic `ValidationError` whose text names a **vllm protocol model**
  (`Input should be a valid dictionary or instance of UsageInfo [type=model_type]`,
  or the same on `ChatCompletionResponse.usage`) although the value already *is* an
  instance of that class — the frames are then all pydantic/vllm and the frontend has
  no defect: a `patch/platform/` module replaced the class with a subclass, so a
  consumer that bound the old name is rejected →
  `../upstream-downstream-drift/protocol-model-class-swap.md` (#8201, vllm-omni over
  v0.18.0rc1).
- The symptom is log text (a flag that prints nothing, a missing stage, a `404`
  route) rather than an exception → `../observability-and-logs/index.md`.
- The process aborts without a Python frame, or dies in a bundled native module
  (`corrupted size vs. prev_size`, `std::logic_error`) →
  `../image-dependency-faults/index.md`.
- The engine never becomes ready, or a worker raised under a `WorkerProc` prefix
  → `../startup-failures/index.md`.
- The raise is real, but the request came through a **P/D** deployment and the
  failing process is the decoder (`kv_consumer`) — prompt-derived outputs cannot be
  computed on a node that skips the prompt forward (`prompt_logprobs` →
  `OverflowError: out of range integral type conversion attempted`):
  `../pd-disaggregation/prompt-side-outputs-on-decode-node.md`.
- The request carries `guided_*` / `response_format` and vllm-ascend's own
  scheduler raises `RuntimeError: Invalid request status: …` → that is downstream
  fork drift, `../upstream-downstream-drift/scheduler-fork-missing-status.md`, not
  the frontend.

## Rules

- **Count the frames per package before any Ascend reasoning.** Zero
  `vllm_ascend` frames plus a `vllm/**` or `site-packages/**` raise is an upstream
  or dependency fault; the model, the SoC and the dataset then enter only through
  the code path that was reached (a chat template, a parser flag, a request
  field), never as suspects. One `grep -rn "<symbol>" vllm_ascend/` that returns
  nothing is the positive form of that claim and belongs in the answer.
- **Ask the pinned vLLM tree whether it already knows the message**:
  `grep -rn "<shortest clause of the message>" -g '*.py' vllm/`. A single hit that
  catches and retries the exception is the cheapest ownership proof there is — it shows
  the condition is expected upstream and names the one path that was protected. Do not
  search the `vllm_ascend` tree first. Grep the **shortest distinctive clause**, never
  the whole sentence: vLLM emits messages as adjacent literals split across source
  lines, so a full-sentence grep finds nothing and reads as "not in this tree", and vLLM
  also **re-raises its dependency's text verbatim**, so a `site-packages`/transformers
  traceback is not proof the raise lives there (#6345 lost three calls to both traps on
  `As of transformers v4.44, default chat template is no longer allowed …`).
  Read the **file** that emits each hit, not the count: at vllm v0.17.0
  `Error in preprocessing prompt inputs` lives only in the completion/tokenize/
  speech-to-text servers, while the chat endpoint emits `Error in tool parser
  creation.` — so the message identifies the upstream path (and the version) that ran.
- **When the raise is about a chat template, the tokenizer mode is the first
  question, not the checkpoint.** `As of transformers v4.44, default chat template is no
  longer allowed …` is vLLM's own raise when template resolution comes back empty
  (`chat_utils.py:1800,1816-1822` at vllm v0.13.0); the fix may be a serve flag
  (`--tokenizer-mode deepseek_v32` for a family whose chat encoding is Python), and
  quantization, SoC and model support are irrelevant to it. Follow the ladder in
  `chat-template-missing-and-tokenizer-mode.md` before proposing `--chat-template`.
- **Do not request plog, `npu-smi`, tiling or per-rank logs for this class.** The
  artefacts that decide it are the full serve command and its parser flags, the
  request body, the client concurrency, and whether *every* request fails or only
  some (sporadic = a race, every-request = read the message again).
- **An error response is not a crash.** Upstream chat preprocessing catches
  `(ValueError, TypeError, RuntimeError, jinja2.TemplateError)` and returns an
  error to the client (`vllm/entrypoints/openai/chat_completion/serving.py:313-316`
  at vllm v0.16.0), so the engine stays healthy while requests fail. Separate the
  two in the answer; a reporter who says "the service failed" may only mean the
  requests did.
- **Backend auto-selection prints a series of warnings that are one loop, not
  several causes.** vLLM's guided-decoding `maybe_backend_fallback` is a sequence of
  independent `if` blocks, so a later one undoes an earlier one: a host with no
  xgrammar wheel sends `xgrammar`→`outlines` by *both* the arch gate and the import
  gate, then the `json_object` gate sends it back to `xgrammar` and the dispatch
  imports the missing backend anyway. Read the warnings in order, name the **last**
  assignment, and check the request field that reached that block (a prompt asking
  for JSON in words never does) — `guided-decoding-backend-aarch64.md`.
- **One `history` grep, then stop.** These races are usually absent from the
  prior-month corpus (no `Already borrowed` hit at all across the recorded
  history); say "no precedent" and cite the upstream issue/comment instead of
  hunting for a matching title.

## Pages

- `chat-template-missing-and-tokenizer-mode.md` — the worked #6345 case
  (vllm-ascend v0.13.0rc2 ↔ vllm v0.13.0, DeepSeek-V3.2-w8a8; no SoC stated): the exact
  `ValueError: As of transformers v4.44, default chat template is no longer allowed …`
  that is **vLLM's own** raise, the four-rung `resolve_hf_chat_template` ladder and the
  fallback table with no DeepSeek-text entry, why `--tokenizer-mode auto` never selects
  `deepseek_v32` (`tokenizers/registry.py:33,144-158`, `config/model.py:76`,
  `serving_engine.py:1151`), the in-tree corroboration (the e2e W8A8 config passes the
  flag at `:40,64`, the tutorial's four serve blocks do not), the `thinking` default, the
  vLLM<0.13 error that separates a version problem, and the thin-checkpoint and DEBUG
  checks.
- `tokenizer-already-borrowed.md` — the worked #7197 (vllm v0.16.0) and #7609
  (vllm v0.17.0, Qwen3-Omni under concurrency) cases: `RuntimeError: Already borrowed`
  at `Hermes2ProToolParser.__init__`, first inside `_preprocess_chat` and then inside
  `chat_completion_stream_generator` (`Error in tool parser creation.`, one parser built
  per streaming request), both chains side by side with their own `path:line`, the two
  threads that share one HF fast tokenizer in the API-server process, the single retry
  upstream already ships, the three-command recipe that re-pins the sites at any
  release (including the ast check that stops a mis-counted pre-encoding parser list —
  `rg 'encode('` also counts `str.encode("utf-8")` calls that never touch the
  tokenizer), the serial-request / retry ladder, and the transformers-version evidence
  gap.
- `guided-decoding-backend-aarch64.md` — the worked #1592 case: a
  `response_format: {"type": "json_object"}` request 500s on aarch64/Kylin while
  plain chat answers; the three warnings as one loop with the pinned `path:line`s
  (`requirements-common.txt:24` `xgrammar == 0.1.11; platform_machine == "x86_64"`,
  `guided_decoding/__init__.py:47-50, 81-86, 89-93`, the unbound `xgr` at
  `xgrammar_decoding.py:89`), why `--guided-decoding-backend outlines` cannot escape
  and `lm-format-enforcer` can (including the `json_schema`-only per-request
  default), and the two checks that place a new report.
- `tool-choice-required-500.md` — the worked #7719 case (vllm-ascend v0.17.0rc1 ↔
  vllm v0.17.0, GLM-5 A3 2P+1D behind the docs' load-balance proxy): `auto` fine /
  `required` 500 with the cause already discarded by the proxy (`:608`), the
  frontend chain that only `required`/named traverse
  (`engine/serving.py:996-1008` → `tool_parsers/abstract_tool_parser.py:56-81` →
  `tool_parsers/utils.py:181-215` schema selection → `structured_output/`
  grammar+bitmask → the hard-fail `required` branch at
  `engine/serving.py:1211-1223` / `parser/abstract_parser.py:423-435`), the lanes
  with their one check (unconstrained grammar, reasoning split emptying
  `content`, schema shape) plus the PD `-1` placeholder lane, and the two
  pin facts that move the answer (`xgrammar == 0.1.29` at the vllm pin vs
  `xgrammar>=0.1.30` at the vllm-ascend pin; guided-decoding e2e coverage is
  single-card only). Also records the scope boundary: the
  `WAITING_FOR_FSM` scheduler-fork route needs the pre-0.17 pins, because
  `vllm_ascend/core/scheduler.py` no longer exists there.
