# A string argument streamed unquoted: the schema-type gate in the GLM parser

## Signature (recognise it from the paste, before reading any code)

The client's assembled `arguments` has **balanced braces but an odd quote count**,
and the damage sits *inside* a value: the value has no opening `"` while a closing
`"` and the `}` are present.

```
assembled : {"hvd": Pura900"}          <- reporter's value
            {"hvd": Pura90"}           <- same fault, other sample
client JS : JSON.parse -> Unexpected identifier " pura" ...     (screenshot OCR)
```

Two things make this class easy to misroute, so state both in the answer:

- It is **not** a lost tail. `finish_reason: "tool_calls"` and the closing `}`
  both arrived; the missing quote is at the *start* of a value.
- The **parser's own final JSON is correct**. In the recorded case the parser's
  last `args_so_far` was the valid `{"hvd": "Pura90"}`; only the *deltas* it
  emitted were wrong. "The assembled string is invalid" therefore does not put the
  fault in the model, the sampler, or the parser's JSON builder — it puts it in
  the incremental-delta bookkeeping (`_compute_args_diff`) driven by a type gate.

This is why the "unbalanced in the middle → the text itself is wrong" outcome
of `index.md` rule 1 must not be applied blindly: check whether the *deltas* are a
length prefix of a valid final string (outcome C vs D in
`streamed-arguments-audit.md`) before calling it a generation fault.

## Verified mechanism (vllm v0.20.2 `bc150f5`, GLM family)

Reporter environment: `vllm==0.20.2`, `vllm-ascend==0.20.2rc1`, GLM-5.1 w4a8/w8a8,
`stream: true` only, `tool_choice: auto`. The model emits GLM's native XML
(`render.json` / `v1_no_stream.json` show it verbatim):

```
</think><tool_call>start_app<arg_key>hvd</arg_key><arg_value>Pura90</arg_value></tool_call><|observation|>
```

`vllm/tool_parsers/glm4_moe_tool_parser.py` (504 lines) owns the conversion;
`glm47_moe_tool_parser.py` is a 40-line subclass that only overrides the region
regexes, so **both `--tool-call-parser glm4_moe` and `glm47_moe` are affected**.

1. **The gate** — `_is_string_type` (`:127-148`):
   ```python
   arg_type = (tool.function.parameters.get("properties", {})
               .get(arg_name, {}).get("type", None))
   return arg_type == "string"
   ```
   Only a **top-level** `"type": "string"` counts. A nullable string declared as
   `anyOf: [{"type": "string"}, {"type": "null"}]` — how most SDKs emit an
   optional string, and exactly what the reporter's `start_app.hvd` schema used —
   has no top-level `type`, so the argument is classified **non-string**.
2. **The raw branch** — `_build_args_json_so_far` (`:331-416`). For a value
   that has opened but not closed, a string-typed argument gets
   `f'{kj}: "{escaped}'` (`:405`); the non-string branch appends the content with
   **no quotes at all**: `f"{kj}: {partial_content}"` (`:408`).
3. **The delta** — `_compute_args_diff` (`:418-427`) emits
   `args_so_far[len(streamed_args_for_tool[index]):]`, i.e. a pure length prefix.
   When `</arg_value>` finally arrives the completed pair is re-rendered through
   `json.dumps` (`:354-360`), so the new string **gains** the opening quote
   (`{"hvd": Pura90` → `{"hvd": "Pura90"}`). The insert is one character *before*
   the already-streamed length, so the length diff drops it for ever.
4. **`stream: false` cannot reproduce it** — `extract_tool_calls` (`:190-226`)
   builds the whole `arg_dct` and emits `json.dumps(arg_dct, ensure_ascii=False)`
   (`:223`), type-agnostically. That is the reporter's own "非流式不复现".

### Reproduction (arithmetic, not opinion)

Transcribing `_is_string_type` + `_build_args_json_so_far` + `_compute_args_diff`
and feeding the observed token pieces (`<arg_value>`, `P`, `ura`, `90`,
`</arg_value>`, `</tool_call>`) reproduces the paste exactly:

| schema | deltas | assembled |
|---|---|---|
| `anyOf: [string, null]` | `{"hvd": `, `P`, `ura`, `90`, `0"`, `}` | `{"hvd": Pura90"}` → `json.loads` fails |
| `{"type": "string"}` | `{"hvd": "`, `P`, `ura`, `90`, `"`, `}` | `{"hvd": "Pura90"}` → `json.loads` OK |

The recorded SSE delivered them as three argument chunks (`{"hvd": P`, `ura90`,
`0"}`) because one chunk can carry several token steps. The counterfactual row is
the proof that the gate, not the model, is the trigger.

## Procedure (four checks, cheapest first)

1. **Reassemble the deltas** — `scripts/sse-toolcall-audit.sh <raw-sse>`. Look for
   the signature above, not just brace balance.
2. **Audit the request's declared types** —
   `uv run --no-project python <skill_root>/scripts/tool-schema-type-audit.py
   <request-body.json>`. In the recorded request (28 tools, 73 arguments) **ten**
   arguments were nullable strings of that shape — `start_app.hvd` among them —
   plus sixteen non-string ones that take the raw branch by design; the failing
   argument must be one of the ten. This is the single check that decides the whole
   class, and it reads the *client's own body*, not the engine.
3. **Confirm the engine's text is fine** — request the same prompt as a raw
   `/v1/completions` body with `stream: true` and `stream: false`. At v0.20.2 both
   returned byte-identical text **and identical `token_ids`** (the issue attached
   both files); the engine, the sampler and the Ascend stack are then out, and the
   defect is post-sampling. This is the same instrument as rule 4 of `index.md`,
   applied to the completions endpoint instead of the chat stream.
4. **Read the two functions at the reporter's ref** — `_is_string_type` and
   `_compute_args_diff`. Do not read the request path or the sampler.
   `import vllm` will fail in a review sandbox (`ModuleNotFoundError: torch`), and
   `import regex` is usually absent too: transcribe the two functions with stdlib
   `re` instead of writing an integration test (the recorded run spent three turns
   discovering both missing modules).

## Ownership: what the Ascend tree does and does not own

`vllm_ascend/patch/platform/` carries **two** GLM tool-call patches at
v0.20.2rc1, and reading both is what makes "upstream owns it" defensible:

- `patch_glm47_tool_call_parser.py:38-42` rewrites `_extract_tool_call_regions`
  only for **complete regions with no `<arg_key>` and no newline**, i.e. inline
  zero-argument calls such as `<tool_call>get_current_time</tool_call>`
  (upstream vllm #44326 / vllm-ascend #8327). An argument-bearing call never
  enters that condition, so this patch is excluded by inspection.
- `patch_glm_tool_call_streaming.py` replaces
  `OpenAIServingChat._create_remaining_args_delta` and wraps
  `chat_completion_stream_generator`, splitting one SSE chunk into an argument
  chunk plus an empty finish chunk — but only when
  `finish_reason == "tool_calls"` **and** the same chunk carries non-empty
  `arguments` (`_terminal_tool_arg_choice`). The recorded stream's terminal chunk
  had `delta: {}`, so the wrapper is a no-op there; and it only re-frames finished
  chunks, it never recomputes a delta. Excluded, with the condition quoted.

Grep `vllm_ascend/patch/` by **model family** (`glm`, `deepseek`, `minimax`) and
read `patch/__init__.py`'s numbered comment blocks — they state each patch's
`Why`/`How`/`Future Plan` and the upstream PR it backports. Grepping the symptom's
own word (`tool_parser`, `tool-call-parser`) misses `patch_glm_tool_call_streaming.py`
even though it patches the same SSE generator.

Adjacent code to read before proposing a frontend fix: `serving.py:1141-1163`
(v0.20.2) computes a `remaining_call` suffix from
`tool_parser.prev_tool_call_arr[index]["arguments"]` versus
`streamed_args_for_tool[index]` and emits it as one more delta. Both fields are
written by `_compute_args_diff`, so this reconciliation repeats the same
bookkeeping rather than repairing it — but it is *why* the answer should say
"fix the diff, not the frontend re-send", and the terminal chunk in the reporter's
paste (`delta: {}`, `finish_reason: tool_calls`, no extra argument chunk before
it) is the evidence that it contributed nothing here.

## What the answer owes

- The assembled string, its brace/quote balance and the client's `JSON.parse`
  message — the same sentence as the reporter's error.
- The named branch: `_is_string_type` at `glm4_moe_tool_parser.py:127-148`
  returning `False` for `anyOf: [string, null]`, the raw branch at `:408`, and the
  length-prefix diff at `:424`. Label it source-verified at the reporter's ref and
  note both parser names plus the non-streaming path that hides it.
- **The reporter's workaround** (this is the actionable half): declare the
  argument as a plain `"type": "string"` — drop the `null` variant, or move
  optionality into the description — which restores the quoted branch; or send
  `stream: false`. Neither changes the model or the Ascend runtime.
- The owner: upstream vLLM's tool parser. Candidate fixes: treat an `anyOf`/
  `oneOf` containing `string` (and a nullable string) as a string type, and/or make
  the delta comparison detect prefix divergence instead of trusting a length.
- Scope limit: verified on vllm v0.20.2 for the GLM XML parsers. Other families
  stream their arguments differently (JSON-native parsers, `hermes`, `qwen3xml`),
  so do not generalise the mechanism without reading their parser. Number/boolean
  arguments streamed raw are **by design** — only a *string* value that takes the
  raw branch is a defect.
