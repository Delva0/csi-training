#!/usr/bin/env python3
"""tool-schema-type-audit.py — read a chat request body and report, per tool
argument, the JSON-Schema keyword that decides which branch a streaming tool
parser takes.

Why: the GLM tool parsers build the streamed ``arguments`` string from the
``<arg_key>/<arg_value>`` XML and choose between a *quoted* and a *raw* branch
with ``properties.<arg>.type == "string"``. A nullable argument declared as
``anyOf: [{type: string}, {type: null}]`` (how most SDKs emit an optional
string) has no top-level ``type``, so it is classified as non-string and its
partial value is streamed unquoted — see
references/tool-call-streaming/arg-type-gate-unquoted-value.md.

Usage:
  uv run --no-project python tool-schema-type-audit.py request.json
  uv run --no-project python tool-schema-type-audit.py - < request.json

Exit status: 0 = no argument takes the raw branch, 1 = at least one does,
2 = input could not be read/parsed. Read the report, not just the status.
"""

import json
import sys


def declared(schema):
    """Return (label, is_string) for one argument schema."""
    if not isinstance(schema, dict):
        return ("<not an object>", False)
    t = schema.get("type")
    if isinstance(t, list):
        return (f"type={t}", "string" in t)
    if t == "string":
        return ("type=string", True)
    if t is not None:
        return (f"type={t}", False)
    for kw in ("anyOf", "oneOf", "allOf"):
        subs = schema.get(kw)
        if isinstance(subs, list):
            kinds = [s.get("type") for s in subs if isinstance(s, dict)]
            has_null = any(k == "null" for k in kinds)
            has_str = any(k == "string" for k in kinds)
            label = f"{kw}={kinds}"
            return (label + (" (nullable string, NO top-level type)" if has_str and has_null else ""),
                    has_str and not has_null)
    if "enum" in schema or "$ref" in schema or "const" in schema:
        return ("no type keyword (enum/$ref/const)", False)
    return ("no type keyword", False)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        if argv[1] == "-":
            body = json.load(sys.stdin)
        else:
            with open(argv[1], encoding="utf-8") as fh:
                body = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot read request body: {exc}")
        return 2

    tools = body.get("tools") or body.get("functions") or []
    print(f"stream={body.get('stream')}  tool_choice={body.get('tool_choice')!r}  tools={len(tools)}")
    if not tools:
        print("no tools[] in this body — nothing to audit (is this the request that failed?)")

    raw = []
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name", "<unnamed>")
        props = ((fn.get("parameters") or {}).get("properties") or {})
        if not props:
            print(f"  {name}: no properties")
            continue
        for arg, schema in props.items():
            label, is_str = declared(schema)
            mark = "  " if is_str else "->"
            print(f"  {mark} {name}.{arg}: {label}")
            if not is_str:
                required = (fn.get("parameters") or {}).get("required") or []
                raw.append((name, arg, label, arg in required))

    if raw:
        print("\narguments that take the parser's RAW (unquoted) branch while streaming:")
        for name, arg, label, req in raw:
            print(f"  {name}.{arg}  ({label}; required={req})")
        print("Only a declared top-level type \"string\" (not anyOf/oneOf with null) keeps the\n"
              "quoted branch. A raw-branch argument whose value is a string therefore streams\n"
              "unquoted and the assembled JSON is unbalanced — the bug, not a model fault.")
        return 1
    print("\nall arguments declare type=string or a non-string type without a string branch.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
