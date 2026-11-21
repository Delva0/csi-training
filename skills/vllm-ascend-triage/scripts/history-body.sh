#!/usr/bin/env bash
# Read a precedent issue body without paying for its collect_env dump.
#
#   scripts/history-body.sh [-n LINES] [-g REGEX] [--env] [--all] [--root DIR] <id> [<id> ...]
#
# Why this exists: an issue body is one GitHub form flattened into text, and the
# `### Your current environment` field usually carries a 100-250 line
# `python collect_env.py` paste (CPU flags, pip freeze, npu-smi) before the
# `### 🐛 Describe the bug` section that decides triage. `sed -n '1,60p'` — the
# command this skill's case-layout page used to recommend — therefore returns the
# title and the environment and *no symptom at all* (recorded #3963: its bug
# heading is at line 148, and one run paid two calls for it — first lines for the env,
# then an inline `awk '/Describe the bug/{f=1} f'` for the bug).
#
# So this prints the body with the collect_env block replaced by a one-line marker,
# and keeps it one flag away (`--env`) because the dump is sometimes the artefact
# (wrong torch_npu / CANN pin, card count, host RAM). Note that reporters also type
# free prose *into* the environment field ("环境：… 问题：…"), which is why the
# marker is a deletion of the dump only, never of the whole field.
#
#   -n LINES     max body lines to print (default 80; `-n 0` = no cap)
#   -g REGEX     print only matching lines, with line numbers (case-insensitive):
#                on a 400-line body pull the traceback or the serve command
#   --env        print the collect_env dump instead of the body
#   --all        print the whole body, dump included
#   --root DIR   history root (default /workspace/history); both layouts are
#                understood: flat `<root>/<id>.md` and per-case `<root>/<id>/<id>.md`.
#                When several case dirs share the id, pass `--root <root>/<case>`.
#
# One id per argument, several allowed, so 2-3 precedents cost one call. Output is
# prefixed `=== #<id> :: <title>`; each line is capped at 1000 chars. Exit 1 = an id
# was not found, 2 = usage/regex error. Read-only: it never writes to the tree.
#
# A missing id prints its line on **stdout and** stderr: a run that pipes the script
# `2>/dev/null | head -n` (recorded #6533) otherwise sees an empty result and reads it
# as an empty body. Many ids this skill's pages cite as precedents are newer than a
# case's own corpus, so `not found` is expected there -- the page is the evidence.
set -u

lines=80
grep_pat=""
mode="body"          # body | env | all
root="/workspace/history"
ids=()

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--lines)  lines=${2:-}; shift 2 ;;
    -g|--grep)   grep_pat=${2:-}; shift 2 ;;
    --env)       mode="env"; shift ;;
    --all)       mode="all"; shift ;;
    --root)      root=${2:-}; shift 2 ;;
    --root=*)    root=${1#--root=}; shift ;;
    -h|--help)   sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)          echo "history-body.sh: unknown option '$1'" >&2
                 echo "usage: history-body.sh [-n LINES] [-g REGEX] [--env] [--all] [--root DIR] <id> [<id> ...]" >&2
                 exit 2 ;;
    *)           ids+=("$1"); shift ;;
  esac
done

if [ ${#ids[@]} -eq 0 ]; then
  echo "usage: history-body.sh [-n LINES] [-g REGEX] [--env] [--all] [--root DIR] <id> [<id> ...]" >&2
  exit 2
fi
if [ ! -d "$root" ]; then
  echo "history-body.sh: no such history root: $root" >&2
  exit 2
fi

exec uv run --no-project python - "$root" "$mode" "$lines" "$grep_pat" "${ids[@]}" <<'PY'
import os
import re
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` must not traceback

root, mode, cap, grep_pat = sys.argv[1], sys.argv[2], int(sys.argv[3] or 80), sys.argv[4]
ids = sys.argv[5:]

COLLECT_ENV = re.compile(r"collect_env", re.I)
STRUCT = re.compile(r"^\s*</?(details|summary)([\s>]|$)", re.I)


def dump_span(raw):
    """Line range of the pasted collect_env.py output: the fenced block that follows
    the `collect_env.py` <summary> line. None when the reporter pasted none."""
    for i, line in enumerate(raw):
        if COLLECT_ENV.search(line):
            for j in range(i + 1, min(i + 6, len(raw))):
                if raw[j].lstrip().startswith("```"):
                    for k in range(j + 1, len(raw)):
                        if raw[k].lstrip().startswith("```"):
                            return j, k
                    return j, len(raw) - 1
            return None
    return None


wanted = set(ids)
paths = {}
for dirpath, _dirnames, names in os.walk(root):
    for name in names:
        if not name.endswith(".md"):
            continue
        issue = name[:-3]
        if issue in wanted:                 # one pass, only the asked-for ids
            paths.setdefault(issue, []).append(os.path.join(dirpath, name))

try:
    rx = re.compile(grep_pat, re.I) if grep_pat else None
except re.error as exc:
    print(f"history-body.sh: bad regex '{grep_pat}': {exc}", file=sys.stderr)
    raise SystemExit(2)


def emit(raw, span):
    """Print raw[span] with the collect_env dump replaced by a marker.

    HTML scaffolding (`<details>`, `<summary>…`) and runs of blank lines are dropped
    so -n counts report text, not template furniture: the recorded #3963 spends 12
    lines on `<details>`/`</details>`/blanks before its `### Describe the bug`.
    """
    lo, hi = span
    dump = dump_span(raw)
    # (raw line number, text) of everything worth printing
    rows = []
    i = lo
    while i < hi:
        if dump and dump[0] <= i <= dump[1]:
            rows.append((i, "[... %d lines of collect_env dump omitted (--env prints it) ...]"
                         % (dump[1] - dump[0] + 1)))
            i = dump[1] + 1
            continue
        line = raw[i].rstrip()
        if STRUCT.search(line):
            i += 1
            continue
        if not line.strip() and rows and rows[-1][1] == "":
            i += 1
            continue
        rows.append((i, line))
        i += 1

    shown = 0
    for k, (idx, line) in enumerate(rows):
        if rx is not None:
            if not rx.search(line):
                continue
            print(f"{idx + 1}: {line[:1000]}")
        else:
            print(line[:1000])
        shown += 1
        if cap and shown >= cap and k + 1 < len(rows):
            print(f"... ({len(rows) - k - 1} more lines; raise -n or use -g REGEX)")
            return


rc = 0
for issue in ids:
    found = sorted(paths.get(issue, []), key=lambda p: (len(p), p))
    if not found:
        msg = (f"=== #{issue} :: not in this corpus (no {issue}.md under {root});"
               " ids that this skill's reference pages and recorded-cases index cite"
               " are often newer than a case's corpus -- use the page's own pin and"
               " facts as the evidence, do not re-open the id")
        print(msg)                  # stdout: `2>/dev/null | head` must not hide it
        print(msg, file=sys.stderr)
        rc = 1
        continue
    try:
        raw = open(found[0], encoding="utf-8", errors="replace").read().splitlines()
    except OSError as exc:
        print(f"=== #{issue} :: unreadable ({exc})")
        print(f"=== #{issue} :: unreadable ({exc})", file=sys.stderr)
        rc = 1
        continue

    nonempty = [i for i, line in enumerate(raw) if line.strip()]
    if not nonempty:
        print(f"=== #{issue} :: (empty body)")
        continue
    head = nonempty[0]
    copies = (" (x%d; several case dirs share the id"
              " -- pass --root /workspace/history/<case>)" % len(found)) if len(found) > 1 else ""
    print(f"=== #{issue} :: {raw[head].strip()[:200]}{copies}")

    dump = dump_span(raw)
    if mode == "all":
        emit(raw, (head + 1, len(raw))) if not dump else print(
            "\n".join(l.rstrip()[:1000] for l in raw[head + 1:head + 1 + (cap or len(raw))]))
    elif mode == "env":
        if dump:
            print("\n".join(l.rstrip()[:1000] for l in raw[dump[0]:dump[1] + 1][: cap or None]))
        else:
            print("(no collect_env.py dump in this body; --all shows the whole body)")
    else:
        emit(raw, (head + 1, len(raw)))

raise SystemExit(rc)
PY
