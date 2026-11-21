#!/usr/bin/env bash
# sse-toolcall-audit.sh -- decide WHO lost the tail of a streamed tool call.
#
# usage: bash sse-toolcall-audit.sh <sse-capture|paste> [more files ...]
#        ... - < capture.txt
#
# "JSON Parse error: Unterminated string" / "Expecting ',' delimiter" raised by an
# OpenAI client on a tool call means the *concatenation* of
# choices[0].delta.tool_calls[0].function.arguments is not valid JSON. That text is
# assembled client-side, so the paste must be the raw SSE body (`curl -N`) or the
# reporter's own echo of every `data:` line -- not the client's rendered error and
# not a chat transcript. This script concatenates the argument fragments per
# tool-call index and reports the balance of `{ } [ ] "`, so the question
#   "did the server stop early / drop tokens, or did the client mis-assemble?"
# is answered from the paste instead of from a mechanism story.
#
# Tolerates HTML/Excel clipboard wrappers, `[...redacted...]` placeholders and
# console prefixes such as "decoded ok: data: ...".
#
# Per-tool-call verdicts: balanced | quote-gap | head-lost | tail-lost |
# extra-tail-brace | malformed. `extra-tail-brace` (head present, more closing braces
# than opening ones -- `{"a": 1}}`) is the synthesizing parser's double close or a
# stray character in the last numeric parameter, NOT "corrupted text in the middle";
# see references/tool-call-streaming/extra-tail-brace-synthesized-args.md.
# `head-lost` (more closing braces than opening ones: the assembly starts *inside*
# the object) is the shape where nothing but the last characters of the arguments
# ever reached the client -- the delta bookkeeping advanced past characters it never
# sent. It is NOT the schema-type gate and NOT a client bug; see
# references/tool-call-streaming/suffix-only-arguments.md.
set -u
LC_ALL=C
if [ "$#" -eq 0 ]; then
  echo "usage: bash $0 <sse-capture> [more files ...]  |  -" >&2
  exit 2
fi
PYSRC=$(cat <<'PY'

import json, re, sys, html

def load(path):
    if path == '-':
        return sys.stdin.read()
    with open(path, 'r', errors='replace') as f:
        return f.read()

text = "\n".join(load(p) for p in sys.argv[1:])
text = html.unescape(text)
text = re.sub(r'(?is)<(script|style)\b.*?</\1>', ' ', text)
text = re.sub(r'(?s)<[^>]{1,200}>', ' ', text)          # tags incl. the mso block

# --- split into candidate SSE payloads -------------------------------------
payloads = []
for line in text.splitlines():
    for m in re.finditer(r'data:\s*(\{.*?)(?=\s*data:|$)', line):
        payloads.append(m.group(1))
if not payloads:
    # a JSON-lines capture (one chunk per line) or a python list of dicts
    for m in re.finditer(r'\{"id":.*?"object":"chat\.completion\.chunk".*?\}(?=\s|$)', text):
        payloads.append(m.group(0))
if not payloads:
    print("no chat.completion.chunk payload found -- paste the raw SSE body "
          "(curl -N ...) or the reporter's per-line echo of `data: {...}`")
    sys.exit(1)

def try_json(s):
    for cand in (s, s.replace('\\"', '"')):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None

def heuristic(chunk):
    """Recover choices[0].delta.tool_calls[*] from a payload whose quotes were
    mangled by an HTML/console copy: keep the substring after "arguments": up to
    the next known key, and read stop_reason / finish_reason separately."""
    out = []
    for idx, m in enumerate(re.finditer(r'"arguments"\s*:\s*', chunk)):
        rest = chunk[m.end():]
        if rest.startswith('"'):
            rest = rest[1:]
        cut = len(rest)
        for marker in ('","type"', '","name"', '"}', '"]', '},'):
            p = rest.find(marker)
            if p != -1:
                cut = min(cut, p)
        arg = rest[:cut]
        arg = arg.replace('\\"', '"').replace('\\n', '\n')
        out.append((idx, arg))
    sr = re.search(r'"stop_reason"\s*:\s*("?[0-9A-Za-z_.\-]+"?)', chunk)
    fr = re.search(r'"finish_reason"\s*:\s*("?[A-Za-z_]+"?)', chunk)
    name = re.search(r'"name"\s*:\s*"([^"]*)"', chunk)
    return out, (sr.group(1) if sr else None), (fr.group(1) if fr else None), \
           (name.group(1) if name else None)

parsed = 0
acc = {}          # tool-call index -> list of fragments
meta = []         # (chunk#, stop_reason, finish_reason, name, nfrag, emptyfrag)
for n, p in enumerate(payloads, 1):
    obj = try_json(p)
    if obj is not None:
        parsed += 1
        raw_choices = obj.get("choices")
        if not raw_choices:
            continue          # usage-only payload: no choice, nothing to audit
        ch = raw_choices[0]
        delta = ch.get("delta") or {}
        tcs = delta.get("tool_calls") or []
        frags, name = [], None
        for tc in tcs:
            i = tc.get("index", 0)
            fn = tc.get("function") or {}
            name = name or fn.get("name")
            a = fn.get("arguments")
            if a is not None:
                acc.setdefault(i, []).append(a)
                frags.append(a)
        silent = (not tcs) and not (delta.get("content") or "")
        meta.append((n, ch.get("stop_reason"), ch.get("finish_reason"), name,
                     len(frags), sum(1 for f in frags if f == ""), silent))
    else:
        frags, sr, fr, name = heuristic(p)
        for i, a in frags:
            acc.setdefault(i, []).append(a)
        meta.append((n, sr, fr, name, len(frags), sum(1 for f in frags if f == ""),
                     not frags))

print(f"payloads: {len(payloads)}  (clean JSON: {parsed}, recovered from mangled text: "
      f"{len(payloads)-parsed})")
print(f"chunks carrying tool_calls arguments: {sum(1 for m in meta if m[4])}")
if not acc:
    print("no tool_calls delta in the paste -> this is not the streaming tool-call shape")
    sys.exit(0)

def balance(s):
    d = {"{": 0, "}": 0, "[": 0, "]": 0, '"': 0}
    esc = False
    for c in s:
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c in d:
            d[c] += 1
    return d

verdicts = []
for i in sorted(acc):
    frags = acc[i]
    s = "".join(frags)
    b = balance(s)
    ok = False
    try:
        json.loads(s); ok = True
    except Exception as e:
        err = str(e)
    print(f"\n-- tool_calls[{i}] --")
    print(f"  fragments: {len(frags)}  empty: {sum(1 for f in frags if f=='')}  "
          f"chars: {len(s)}")
    print(f"  braces {{ {b['{']} / }} {b['}']}   brackets [ {b['[']} / ] {b[']']}   "
          f"quotes: {b['\"']} ({'even' if b['\"'] % 2 == 0 else 'ODD'})")
    print(f"  shallow slice: {frags[:3]} ... {frags[-3:]}")
    print(f"  assembled (first 200 / last 60): {s[:200]!r} ... {s[-60:]!r}")
    if ok:
        print("  json.loads(assembled): OK")
        verdicts.append("balanced")
    else:
        print(f"  json.loads(assembled): FAIL -> {err}")
        # Five outcomes, not two (tool-call-streaming/index.md rule 1). An open
        # brace/bracket means the tail never arrived. MORE closing braces than
        # opening ones means the assembly starts inside the object: the head never
        # arrived at all (nothing but a suffix of the arguments was ever sent) --
        # that is the delta bookkeeping's length, not the type gate and not the
        # client. BALANCED braces with an odd quote count is the third family: the
        # closing `}` and a quote are there, the gap is the *opening* quote inside a
        # value -- the parser's type-gated delta, not a lost tail, and it must not be
        # routed to the engine. Anything else is unbalanced text in the middle.
        open_tail = (b['{'] > b['}']) or (b['['] > b[']'])
        lost_head = (((b['}'] > b['{']) or (b[']'] > b['[']))
                     and not s.lstrip().startswith(("{", "[")))
        # The sixth outcome: the head IS there (the assembly starts with `{`), the
        # only imbalance is MORE closing braces than opening ones -- one or more
        # extra `}` appended after the last value, e.g. `{"similarity": 0.5}}`. A
        # parser that SYNTHESIZES the JSON from XML tags writes that closing brace
        # itself, so this is the close-brace bookkeeping lane (or a stray character
        # inside the last numeric parameter, passed through unquoted), not the
        # head-lost delta arithmetic and not "corrupted text in the middle".
        extra_tail = (((b['}'] > b['{']) or (b[']'] > b['[']))
                      and s.lstrip().startswith(("{", "[")))
        if lost_head:
            verdicts.append("head-lost")
            print("  -> MORE closing braces than opening ones: the assembly starts")
            print("     INSIDE the object, so the head (the `{`, the keys) never")
            print("     arrived -- only a suffix of the arguments was ever sent. Do")
            print("     NOT run tool-schema-type-audit.py: this is not the type gate.")
            print("     diff it against `stream: false`'s message.tool_calls[0].")
            print("     function.arguments: a strict suffix means the delta")
            print("     bookkeeping advanced past characters it never emitted")
            print("     (references/tool-call-streaming/suffix-only-arguments.md).")
        elif open_tail:
            verdicts.append("tail-lost")
        elif extra_tail:
            verdicts.append("extra-tail-brace")
            print("  -> MORE closing braces than opening ones while the head IS")
            print("     present: one or more extra `}` sit after the last value. Do")
            print("     NOT call this corrupt text and do NOT run")
            print("     tool-schema-type-audit.py. A parser that synthesizes the")
            print("     arguments JSON from XML tags (`qwen3_xml`, `qwen3_coder`,")
            print("     `hermes`) emits the closing `}` itself, so rank (a) a")
            print("     close brace emitted twice by the parser's fallback/guard")
            print("     paths and (b) a stray character inside the last numeric")
            print("     parameter passed through unquoted. Split them with")
            print("     `stream: false` on the same body, a run with the parser")
            print("     disabled, and a `--tool-call-parser qwen3_coder` A/B")
            print("     (references/tool-call-streaming/")
            print("     extra-tail-brace-synthesized-args.md).")
        elif b['"'] % 2:
            verdicts.append("quote-gap")
            print("  -> braces/brackets balance but a quote is missing: the gap is")
            print("     INSIDE a value (the opening quote), not a lost tail. Run")
            print("     scripts/tool-schema-type-audit.py <request-body.json>; an")
            print("     argument declared anyOf:[string,null] has no top-level type")
            print("     (references/tool-call-streaming/arg-type-gate-unquoted-value.md).")
        else:
            verdicts.append("malformed")

stop = [(m[0], m[1]) for m in meta if m[1] not in (None, "None")]
fin = [(m[0], m[2]) for m in meta if m[2] not in (None, "None")]
last_frag = max((m[0] for m in meta if m[4] and not m[5]), default=0)
print(f"\nstop_reason chunks: {stop or 'none'}   (int = an upstream vLLM-only field "
      f"carrying the stop token id, ChatCompletionResponseStreamChoice.stop_reason; "
      f"not corruption)")
print(f"finish_reason chunks: {fin or 'none'}   last non-empty argument chunk: #{last_frag}")
empty_tail = [m[0] for m in meta if m[4] and m[5] == m[4]]
if empty_tail:
    print(f"chunks whose arguments fragment is the EMPTY string: {empty_tail}")
silent = [m[0] for m in meta if m[6]]
if silent:
    print(f"chunks carrying NEITHER content nor tool_calls: {len(silent)} of "
          f"{len(meta)} (first #{silent[0]}, last #{silent[-1]}) -> the model's "
          f"output text for those steps reached no client field; a buffering tool "
          f"parser emits exactly this while it walks the model's XML")

if all(v == "balanced" for v in verdicts):
    print("\nVERDICT: every assembled tool call is valid JSON. The failure is in the "
          "client's own accumulation/framing (or it parsed a single chunk instead of "
          "the concatenation). Ask for the client version and its parser; the server "
          "stream is not the owner.")
elif "head-lost" in verdicts:
    print("\nVERDICT: at least one assembled tool call starts INSIDE the object (more")
    print("closing braces than opening ones): the client received only a suffix of the")
    print("arguments, so the head was never sent. That is the same delta-bookkeeping")
    print("layer as quote-gap, in its extreme form -- the parser's")
    print("`_compute_args_diff` / `streamed_args_for_tool` length, or the frontend's")
    print("`remaining_call` suffix, was already ahead of what it had emitted. Not the")
    print("model, not the tokenizer, not the client. One arithmetic decides it: take")
    print("`message.tool_calls[0].function.arguments` from the SAME body with")
    print("`stream: false` and test `assembled == args[-len(assembled):]`; a strict")
    print("suffix means only the tail was ever emitted. Ask for `return_token_ids:")
    print("true` on the streaming body to show the engine's own text is complete")
    print("(references/tool-call-streaming/suffix-only-arguments.md).")
elif "extra-tail-brace" in verdicts:
    print("\nVERDICT: at least one assembled tool call carries MORE closing braces than")
    print("opening ones while its head (the `{` and every key) is present -> a closing")
    print("`}` was written one time too many. That owner is the tool parser that")
    print("SYNTHESIZES the arguments JSON from the model's XML tags, or the model's own")
    print("character inside the last numeric parameter passing through unquoted -- not")
    print("the client, not a lost tail. Order the checks by cost: `stream: false` on")
    print("the same body (one `}` there = the model's text / decoding, two = the")
    print("parser), the same body with `--tool-call-parser` removed (is the extra `}`")
    print("in the model's raw `<parameter=...>` value?), then")
    print("`--tool-call-parser qwen3_coder` as an alternate implementation of the same")
    print("markers. On an MTP serve, read the block-verify gate before blaming")
    print("sampling (references/tool-call-streaming/")
    print("extra-tail-brace-synthesized-args.md).")
elif "quote-gap" in verdicts and "tail-lost" not in verdicts:
    print("\nVERDICT: at least one assembled tool call has BALANCED braces/brackets with")
    print("an odd quote count -> the missing quote sits inside a value, so the tail did")
    print("arrive and the server did not truncate. That shape is a streaming parser's")
    print("incremental delta keyed on the argument's declared JSON-Schema type: check the")
    print("request body with scripts/tool-schema-type-audit.py, read")
    print("references/tool-call-streaming/arg-type-gate-unquoted-value.md, and confirm the")
    print("engine text is clean with /v1/completions `stream: true` vs `false`")
    print("(identical text and token_ids = post-sampling fault).")
elif "tail-lost" in verdicts:
    print("\nVERDICT: the assembled arguments end with unbalanced braces/quotes while a "
          "finish_reason chunk follows -> the server stream is missing the CLOSING tail "
          "of the tool-call JSON. Compare `return_token_ids: true` (or --enable-log-outputs "
          "text) for the same request: if the token ids also stop short, the engine "
          "emitted a short sequence (stop token / dropped accepted tokens); if the token "
          "ids are complete, the delta assembly in the serving path lost them. On a "
          "num_speculative_tokens>=3 request the block-verify acceptance path is the first "
          "candidate (see the skill's mtp-block-verify-acceptance.md); its slots are "
          "filtered by RejectionSampler.parse_output without a log line.")
else:
    print("\nVERDICT: the arguments are unbalanced in the middle (not only the tail) -> "
          "the text itself is corrupted/duplicated, so treat it as a sampling/decoding "
          "correctness question rather than a delta-assembly question.")
PY
)
exec uv run --no-project python -c "$PYSRC" "$@"
