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
        choices = obj.get("choices") or [{}]
        ch = choices[0]
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
        meta.append((n, ch.get("stop_reason"), ch.get("finish_reason"), name,
                     len(frags), sum(1 for f in frags if f == "")))
    else:
        frags, sr, fr, name = heuristic(p)
        for i, a in frags:
            acc.setdefault(i, []).append(a)
        meta.append((n, sr, fr, name, len(frags), sum(1 for f in frags if f == "")))

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
        tail_bad = (b['{'] - b['}'] > 0) or (b['['] - b[']'] > 0) or b['"'] % 2
        verdicts.append("tail-lost" if tail_bad else "malformed")

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

if all(v == "balanced" for v in verdicts):
    print("\nVERDICT: every assembled tool call is valid JSON. The failure is in the "
          "client's own accumulation/framing (or it parsed a single chunk instead of "
          "the concatenation). Ask for the client version and its parser; the server "
          "stream is not the owner.")
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
