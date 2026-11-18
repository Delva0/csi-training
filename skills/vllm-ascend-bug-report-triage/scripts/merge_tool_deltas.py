#!/usr/bin/env python3
"""Merge an OpenAI chat-completions SSE dump and print the tool-call shape.

Why: a `tool_calls` entry with no `id` and an empty `function.arguments` is a
*stream delta*, not a broken whole call, and which of the three known shapes it
is decides the owner (client merge / serving finish chunk / parser drop). Read
the raw stream, not the client's rendering.

usage:
    curl -sN http://host:port/v1/chat/completions -H 'Content-Type: application/json' \
      -d @req.json > stream.txt
    uv run --no-project python scripts/merge_tool_deltas.py stream.txt

Input: a raw SSE capture, any log holding `data: {...}` lines, or a JSON file
with a list of chunks (or one non-streaming response).

Dependencies: Python standard library only (json, sys); no clone, no network,
  no package install — run it as `uv run --no-project python`.
Validation: exercised on synthetic captures of the three known shapes (terminal
  empty-arguments delta, terminal argument chunk, id-never-streamed) and on a
  non-streaming body; it parses and reports, it does not prove a server bug.
Source: the shapes come from vLLM `v0.20.2`
  `vllm/entrypoints/openai/chat_completion/serving.py` `_should_check_for_unstreamed_tool_arg_tokens`
  / `_create_remaining_args_delta` and the vllm-ascend patches
  `vllm_ascend/patch/platform/patch_glm_tool_call_streaming.py`,
  `patch_glm47_tool_call_parser.py` (see
  `references/symptoms/tool-call-streaming.md`).
"""

from __future__ import annotations

import json
import sys


def load_chunks(path: str) -> tuple[list[dict], str]:
    raw = open(path, encoding="utf-8", errors="replace").read()
    stripped = raw.strip()
    if stripped[:1] in "[{":
        try:
            doc = json.loads(stripped)
        except json.JSONDecodeError:
            doc = None
        if isinstance(doc, dict):
            return [doc], "json"
        if isinstance(doc, list):
            return [c for c in doc if isinstance(c, dict)], "json"
    chunks: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            chunks.append(obj)
    return chunks, "sse"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: merge_tool_deltas.py <sse-or-json-file>", file=sys.stderr)
        return 2
    chunks, kind = load_chunks(sys.argv[1])
    if not chunks:
        print("no chunks parsed: not an SSE capture and not a JSON chunk list")
        return 1

    ids: dict[int, tuple[str, int]] = {}
    names: dict[int, tuple[str, int]] = {}
    args: dict[int, list[str]] = {}
    findings: list[str] = []
    finish_seen = False
    saw_tool_delta = False
    has_choices = False

    for n, chunk in enumerate(chunks, 1):
        choices = chunk.get("choices")
        if choices is None:  # non-streaming body
            for tc in (chunk.get("message") or {}).get("tool_calls") or []:
                idx = tc.get("index", 0) if isinstance(tc.get("index"), int) else 0
                if not tc.get("id"):
                    findings.append(
                        f"non-streaming body: tool_call {idx} has no id "
                        "(the server cannot emit that; client/proxy artifact)"
                    )
                if (tc.get("function") or {}).get("arguments") == "":
                    findings.append(
                        f"non-streaming body: tool_call {idx} has arguments='' "
                        "(server emits json.dumps of the parsed args, '{}' for none)"
                    )
            continue
        has_choices = True
        for choice in choices or []:
            if "delta" not in choice:  # non-streaming choice: complete tool calls
                for tc in (choice.get("message") or {}).get("tool_calls") or []:
                    idx = tc.get("index") if isinstance(tc.get("index"), int) else 0
                    fn = tc.get("function") or {}
                    ids.setdefault(idx, (tc.get("id") or "<missing>", n))
                    if not tc.get("id"):
                        findings.append(
                            f"non-streaming body: tool_call {idx} has no id "
                            "(the server always sets one; client/proxy artifact)"
                        )
                    names.setdefault(idx, (fn.get("name") or "", n))
                    args.setdefault(idx, []).append(str(fn.get("arguments") or ""))
                    if fn.get("arguments") == "":
                        findings.append(
                            f"non-streaming body: tool_call {idx} has arguments='' "
                            "(the server emits json.dumps of the parsed args, '{}' for none)"
                        )
                if choice.get("finish_reason"):
                    finish_seen = True
                continue
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason")
            tcs = delta.get("tool_calls") or []
            if tcs:
                saw_tool_delta = True
            for tc in tcs:
                idx = tc.get("index")
                if not isinstance(idx, int):
                    idx = 0
                fn = tc.get("function") or {}
                if tc.get("id"):
                    ids.setdefault(idx, (tc["id"], n))
                if fn.get("name"):
                    names.setdefault(idx, (fn["name"], n))
                if isinstance(fn.get("arguments"), str):
                    args.setdefault(idx, []).append(fn["arguments"])
                if finish:
                    frag = fn.get("arguments")
                    if frag == "":
                        findings.append(
                            f"chunk {n}: index {idx} arguments='' in the same chunk as "
                            f"finish_reason={finish!r} (terminal remaining-args delta, nothing left to stream)"
                        )
                    elif frag:
                        findings.append(
                            f"chunk {n}: index {idx} carries argument bytes {frag!r} in the same chunk as "
                            f"finish_reason={finish!r} (terminal arg chunk not split off)"
                        )
            if finish:
                finish_seen = True
                if finish == "tool_calls" and not saw_tool_delta:
                    findings.append(
                        f"chunk {n}: finish_reason='tool_calls' but no delta.tool_calls was ever streamed "
                        "(parser dropped the call: zero-argument inline call, or an incomplete region)"
                    )

    print(f"input: {kind}, {len(chunks)} chunk(s)")
    if not has_choices:
        print("no 'choices' array: this is not an SSE chat-completions capture")
    for idx in sorted(set(ids) | set(names) | set(args)):
        merged = "".join(args.get(idx, []))
        first_id = ids.get(idx)
        name = names.get(idx)
        state = "valid JSON" if _is_json(merged) else "not valid JSON"
        print(
            f"index {idx}: id="
            + (f"{first_id[0]!r} (chunk {first_id[1]})" if first_id else "NEVER STREAMED")
            + f", name={(name[0] if name else None)!r}"
            + f", {len(merged)} argument chars, merged args {state}"
        )
        if not first_id:
            findings.append(
                f"index {idx}: no chunk ever carried an id — the tool-call header never streamed "
                "(see the parser-drop mode, not the finish-chunk mode)"
            )
    if not finish_seen:
        print("note: no finish_reason in the capture (truncated stream?)")
    print()
    if findings:
        print("findings:")
        for f in dict.fromkeys(findings):
            print(f"- {f}")
        print()
        print(
            "route: header never streamed -> glm47 inline zero-arg parser drop; "
            "arguments='' with finish_reason -> terminal remaining-args delta; "
            "argument bytes with finish_reason -> finish chunk not split; "
            "an id on an earlier chunk -> client delta merge."
        )
    else:
        print("findings: none — the stream is OpenAI-shaped (id first, fragments after, empty finish chunk)")
    return 0


def _is_json(text: str) -> str:
    if not text:
        return ""
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return ""
    return "valid JSON"


if __name__ == "__main__":
    raise SystemExit(main())
