# `content` null **and** `tool_calls: []`: the DeepSeek-V3 parser's silent drop

## Signature

The serve has `--enable-auto-tool-choice --tool-call-parser deepseek_v3` (often with
`--chat-template tool_chat_template_deepseekv3.jinja`) and the failing response comes back
with **both** `message.content` empty/null **and** `message.tool_calls` an empty list, often
with `finish_reason: "tool_calls"`. The reporter usually adds those flags *because* a
previous report showed the model's raw markers in `content`:

````text
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather
```json
{"location":"北京","unit":"celsius"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
````

So read the sequence, not the last message: enabling the parser did not "fail to work", it
moved a parse failure from visible text to a silent empty message. Answer both halves.

## Mechanism — one branch produces this shape (verified, vllm v0.9.2 `a5dd03c`)

`vllm/entrypoints/openai/tool_parsers/deepseekv3_tool_parser.py`:

- `:78-81` — no `<｜tool▁calls▁begin｜>` in the output ⇒ `tools_called=False`, `content=model_output`.
  That is the *first* report's shape, not this one.
- `:41-43` — the extractor regex, and it is strict: `.*` does not cross newlines (no
  `re.DOTALL`), and the body must be
  `<｜tool▁call▁begin｜><type><｜tool▁sep｜><name>\n` + a fenced ```` ```json ```` block + `\n```<｜tool▁call▁end｜>`.
- `:84-107` — with the start marker present but **no** regex match, `findall` returns `[]`,
  and it still returns `tools_called=True, tool_calls=[], content=model_output[:find(start)]`.
  When the call starts at offset 0 (the normal case), `content` is `""` → `None`.

Then `serving_chat.py:1085-1113` copies that through: `auto_tools_called = tool_call_info.tools_called`,
the message keeps `content=None, tool_calls=[]`, and `finish_reason` is still set to
`"tool_calls"`. That path raises nothing and logs nothing above debug level; the
`except Exception` at `:110-114` is a *different* shape — it logs and returns the whole text
as `content`, so a paste that still shows the markers is not the branch above.

Checked at that ref by running the regex over three texts: the reporter's own "expected"
format **matches** (`function`, `get_current_weather`, `{"location":…}`); a repeated
`<｜tool▁calls▁begin｜>` marker run and the same call **without** the ```` ```json ```` fence
both match nothing — i.e. both produce empty/empty through that branch.

## The one comparison that decides it

Do not answer with more enable-flag advice and do not read the parser again. Get the
model's own text for the failing request and compare it with the three required pieces:

1. Raw text with the parser out of the way — `POST /v1/completions` with the rendered
   prompt (or the same chat body on a serve without `--enable-auto-tool-choice`, which is
   what the first report already pasted).
2. Against the regex: is there a `<｜tool▁call▁begin｜>` … `<｜tool▁call▁end｜>` pair at all, does
   `function<｜tool▁sep｜><name>` sit on **one** line followed by a newline, is the argument
   block fenced with ```` ```json ````, and is the name a tool the request declared?
   The missing piece is the answer: a marker run is the degeneration lane (sampling/graph
   ablations, `index.md` scope), a fence-less or differently-spelled call is a
   template/model-format mismatch — the template's tools block and the parser's expected
   fence disagree — and a complete, fence-correct call that still comes back empty/empty is
   an engine defect worth its own issue.
3. `stream: true` on the same body is the independent witness: the streaming path matches
   the **name** alone (`stream_tool_call_name_regex`, `:49-50`, needs only
   `function<｜tool▁sep｜><name>\n`) and emits `delta.tool_calls` with that name. A stream that
   names the call while the non-streaming response drops it localises the fault to
   `extract_tool_calls`, not to the model.

## The reporter's own workaround, and its scope (#1798)

The reporter's confirmed alternative is to stop using the parser: render the tool schemas
into the **system prompt** yourself (what the Jinja tools block does), send no `tools`
field, and read `message.content`. Then the call comes back in the marker format above,
consistently, with no parser in the path — reporter-verified on vLLM DeepSeek-V3-0324
**on H200**, not on Ascend. Quote it as a model-format statement plus a client-side
workaround: it does not repair the parser and it says nothing about the w8a8 Ascend path
beyond "the same prompt must produce the same text there".

## What to send back

The raw completion text of a failing request (not just `message`), `finish_reason`,
`stream: false` and `stream: true` for the same body, the checkpoint's
`tokenizer_config.json` `chat_template` (does its tools block teach the fence?), and
whether the failing case is a **marker repetition loop** (then the count of such requests
out of N, and the `temperature`/`top_k` of the body, decides the lane).

## Limits

Source-verified only at vllm v0.9.2; later refs move this parser (`deepseek_v3` may be
subclassed or patched downstream) — re-grep `extract_tool_calls` at the reporter's ref
before quoting the lines. Which regex piece *their* text lacked is not in the recorded
evidence (the follow-up is a screenshot), so name it as the thing the comparison decides,
not as a conclusion.
