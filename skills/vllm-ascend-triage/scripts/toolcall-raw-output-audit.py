#!/usr/bin/env python3
"""toolcall-raw-output-audit.py — read a case's request / rendered-prompt /
tokenized-request / raw-completion dumps and decide whether the model emitted a
tool call at all, before any parser, delta-assembly or kernel is read.

Why: the recorded #12658 report ("tool call empty", non-thinking mode) shipped
exactly `bad_case.json`, `rendered.json`, `rendered_prompt.json` and
`raw_completion.json`, and three of its four facts settled the case: the request
was sampled at temperature 1.0, the rendered prompt carried the model's own tool
grammar, and the engine's completion for that same prompt (5944 tokens on both
sides) contained no tool-call marker — so no parser could have produced a call.
See references/tool-call-streaming/no-tool-call-in-raw-output.md.

Usage:
  uv run --no-project python toolcall-raw-output-audit.py <dir-or-file> [...]
  uv run --no-project python toolcall-raw-output-audit.py <dir> --markers '<a>,<b>'

Each JSON is classified by shape (chat request body / rendered prompt /
tokenized request / completion) and every fact is read from the dump, never
guessed. Exit status: 0 = a completion dump was judged, 1 = no completion dump
in hand, 2 = nothing readable. Read the report, not just the status.
"""

from __future__ import annotations

import json
import os
import sys

#: one marker per tool-parser family; printed so the reader can see what the
#: dumps were probed for instead of trusting a silent "not found".
DEFAULT_MARKERS = [
    "<｜DSML｜tool_calls>",
    "<｜DSML｜function_calls>",
    "<｜DSML｜invoke",
    "<tool_call>",
    "<|tool_call>",
    "<arg_key>",
    "[TOOL_CALLS]",
    "<function=",
]


def norm(s: str) -> str:
    """Fold the fullwidth bar used by DeepSeek/GLM special tokens, so a dump
    written with ``｜DSML｜`` matches the marker spelled with ASCII pipes."""
    return (s or "").replace("\uff5c", "|")


def load(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  ! {os.path.basename(path)}: unreadable ({exc})")
        return None


def first_choice(doc):
    if not isinstance(doc, dict):
        return None
    choices = doc.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return None


def classify(doc):
    """Return (kind, payload) for one JSON document."""
    if isinstance(doc, str):
        return "rendered prompt", doc
    if not isinstance(doc, dict):
        return "other", doc
    if isinstance(doc.get("messages"), list):
        return "request body", doc
    ch = first_choice(doc)
    if ch is not None:
        text = ch.get("text")
        if text is None and isinstance(ch.get("message"), dict):
            msg = ch["message"]
            text = (msg.get("content") or "") + (msg.get("reasoning") or "")
        if text is None and isinstance(ch.get("delta"), dict):
            text = ch["delta"].get("content") or ""
        if text is not None:
            return "completion", {"choice": ch, "text": text, "usage": doc.get("usage") or {}}
    prompt = doc.get("prompt")
    if isinstance(prompt, str):
        return "rendered prompt", prompt
    if isinstance(prompt, list):
        return "tokenized request", doc
    if "sampling_params" in doc or "token_ids" in doc:
        return "tokenized request", doc
    return "other", doc


def read_dumps(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += [os.path.join(a, f) for f in sorted(os.listdir(a)) if f.endswith(".json")]
        elif os.path.isfile(a):
            paths.append(a)
    out = []
    for p in paths:
        doc = load(p)
        if doc is None:
            continue
        kind, payload = classify(doc)
        out.append((p, kind, payload))
    return out


def main(argv):
    args = list(argv[1:])
    markers = list(DEFAULT_MARKERS)
    if "--markers" in args:
        i = args.index("--markers")
        markers = [m for m in args[i + 1].split(",") if m]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        return 2
    dumps = read_dumps(args)
    if not dumps:
        print("nothing readable — pass the attachments directory or the dump files")
        return 2

    reqs = [(p, v) for p, k, v in dumps if k == "request body"]
    prompts = [(p, v) for p, k, v in dumps if k == "rendered prompt"]
    tokens = [(p, v) for p, k, v in dumps if k == "tokenized request"]
    comps = [(p, v) for p, k, v in dumps if k == "completion"]
    print(f"dumps read: {len(dumps)}  (request bodies {len(reqs)}, rendered prompts "
          f"{len(prompts)}, tokenized requests {len(tokens)}, completions {len(comps)})")

    for p, r in reqs:
        tools = r.get("tools") or []
        names = []
        for t in tools:
            fn = t.get("function") if isinstance(t, dict) else None
            names.append((fn or {}).get("name") or (t.get("name") if isinstance(t, dict) else "?"))
        last_user = ""
        for m in reversed(r.get("messages") or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content"))[:60]
                break
        print(f"\nrequest body  {os.path.basename(p)}")
        print(f"  model={r.get('model')}  stream={r.get('stream')}  tools={len(tools)} {names}")
        print(f"  chat_template_kwargs={json.dumps(r.get('chat_template_kwargs') or {}, ensure_ascii=False)}"
              f"  reasoning_effort={r.get('reasoning_effort')}")
        print(f"  last user message={last_user!r}")

    for p, pr in prompts:
        tail = pr[-44:].replace("\n", "\\n")
        hit = [m for m in markers if norm(m) in norm(pr)]
        print(f"\nrendered prompt  {os.path.basename(p)}")
        print(f"  chars={len(pr)}  '## Tools' section={'## Tools' in pr}")
        print(f"  tool grammar in prompt: {hit or 'NONE'}")
        print(f"  think tokens: '<think>' x{pr.count('<think>')}  '</think>' x{pr.count('</think>')}")
        print(f"  tail: ...{tail}")
        if not hit:
            print("  ! the prompt teaches no tool-call grammar: a missing call can be a prompt\n"
                  "    rendering problem (tools not rendered in this thinking mode), not the model")

    for p, d in tokens:
        ids = d.get("token_ids") or []
        sp = d.get("sampling_params") or {}
        print(f"\ntokenized request  {os.path.basename(p)}")
        print(f"  prompt token ids={len(ids)}  model={d.get('model')}  stream={d.get('stream')}")
        if sp:
            print(f"  sampling: temperature={sp.get('temperature')} top_p={sp.get('top_p')} "
                  f"max_tokens={sp.get('max_tokens')} seed={sp.get('seed')}")
            t = sp.get("temperature")
            if isinstance(t, (int, float)) and t > 0:
                print("  temperature > 0: this body is sampled, so a rate of missing calls is\n"
                      "    expected from sampling alone — the greedy (temperature 0) repeat is the\n"
                      "    deterministic row, not the fix")

    if not comps:
        print("\nVERDICT: no completion dump in hand. The engine-side raw output is the first\n"
              "  artefact: POST the rendered prompt to /v1/completions (or add\n"
              "  `return_token_ids: true` to the same chat body) before reading any parser.")
        return 1

    rc = 0
    for p, c in comps:
        ch, text = c["choice"], c["text"]
        found = [m for m in markers if norm(m) in norm(text)]
        ntok = ch.get("token_ids")
        print(f"\ncompletion  {os.path.basename(p)}")
        print(f"  finish_reason={ch.get('finish_reason')}  stop_reason={ch.get('stop_reason')}  "
              f"seq tokens={len(ntok) if isinstance(ntok, list) else 'n/a'}  chars={len(text)}")
        usage = c["usage"] or {}
        if usage:
            print(f"  usage: prompt_tokens={usage.get('prompt_tokens')} "
                  f"completion_tokens={usage.get('completion_tokens')}")
        print(f"  tool-call markers: {found or 'NONE'}")
        print(f"  first 160 chars: {text[:160]!r}")
        if ch.get("message", {}).get("tool_calls") if isinstance(ch.get("message"), dict) else False:
            print("  the response body already carries tool_calls ⇒ parser/assembly lane, not this one")
        elif not found:
            print("\nVERDICT: no tool-call marker in the model's own output ⇒ no parser, delta\n"
                  "  assembly or kernel can produce a call for this request; the sampled sequence\n"
                  "  contains none. Model-choice lane — repeat the identical body N times, repeat\n"
                  "  with thinking on, then the vendor's own API:\n"
                  "  references/tool-call-streaming/no-tool-call-in-raw-output.md")
        else:
            print("\nVERDICT: a tool-call marker IS in the raw output while the client saw no\n"
                  "  tool_calls ⇒ parser lane: marker spelling vs the parser's start token,\n"
                  "  reasoning-phase gating, tool_choice, or the delta assembly\n"
                  "  (`scripts/sse-toolcall-audit.sh <raw-sse>`).")

    # cross-dump consistency: the dumps must describe the same request
    if prompts and reqs:
        pr = prompts[0][1]
        last_user = ""
        for m in reversed(reqs[0][1].get("messages") or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content") or "")
                break
        if last_user and last_user not in pr:
            print(f"\n! consistency: request body's last user message {last_user[:30]!r} does not\n"
                  "  appear in the rendered prompt — the dumps are not all from the same request;\n"
                  "  say which pair you compared instead of blending them")
    if tokens and comps:
        ids = tokens[0][1].get("token_ids") or []
        usage = comps[0][1]["usage"] or {}
        pt = usage.get("prompt_tokens")
        if isinstance(pt, int) and ids:
            same = pt == len(ids)
            print(f"\n[consistency] tokenized-request ids={len(ids)} vs completion "
                  f"prompt_tokens={pt} → {'SAME prompt' if same else 'DIFFERENT prompt'}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
