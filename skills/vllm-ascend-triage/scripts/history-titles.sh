#!/usr/bin/env bash
# Cheap precedent index over the read-only issue history.
#
#   scripts/history-titles.sh [-i] [-b] '<extended-regex>' [history-root] [limit]
#
# Prints one line per matching issue file:  <file> :: <first non-empty line>
# so a whole family of issues can be triaged without reading any body.
#
# Title-only is the DEFAULT and that is deliberate. A body match is almost always
# a lexical accident: every issue that pasted collect_env.py contains
# `ATB_SHARE_MEMORY_NAME_SUFFIX=`, `Speculative Store Bypass disabled via prctl`,
# `torch_npu`, `vllm-ascend`, so a body search for `suffix` / `mtp` / `ngram` /
# `spec` returns essentially the whole corpus (recorded #9278: `-i
# 'suffix|507035|aivec|MPU address'` came back as "207 hits / 207 distinct issue
# ids", every one of them matching on the env-var name, and the run read the whole
# result as noise and stopped — while 488.md in that same output carries the
# reporter's exact `ACL stream synchronize failed, error code:507035`). Pass `-b`
# when you *want* a body search, and prefer one distinctive token (an op name, an
# error string, a code) over an alternation that mixes one in with a plain word.
#
# `-i` makes the match case-insensitive (issue titles capitalise tokens:
# `reranker` vs `Reranker`).
# Never `ls` the history directory: it holds thousands of files.
set -u

icase=""   # case folding for the regex
body=""    # 1 = search whole files instead of titles
while [ $# -gt 0 ]; do
  case "$1" in
    -i|--ignore-case) icase=1; shift ;;
    -b|--body) body=1; shift ;;
    --) shift; break ;;
    -h|--help)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    -*)
      # A grep-style flag as the first argument used to be taken for the regex and
      # the pattern for the history root, burning a call on "no such history root".
      echo "history-titles.sh: unknown option '$1'" >&2
      echo "usage: history-titles.sh [-i] [-b] <extended-regex> [history-root] [limit]" >&2
      exit 2 ;;
    *) break ;;
  esac
done

pat=${1:-}
if [ -z "${pat:-}" ]; then
  echo "usage: history-titles.sh [-i] [-b] <extended-regex> [history-root] [limit]" >&2
  exit 2
fi
root=${2:-/workspace/history}
limit=${3:-40}

if [ ! -d "$root" ]; then
  echo "no such history root: $root" >&2
  exit 2
fi

# Layouts seen: flat  history/<issue>.md  and per-case  history/<issue>/<id>.md;
# keep one line per id and report how many copies exist so `limit` is not wasted.
exec uv run --no-project python - "$pat" "$root" "$limit" "${icase:-0}" "${body:-0}" <<'PY'
import os
import re
import sys

pat, root, limit, icase, body = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5] == "1"
flags = re.IGNORECASE if icase else 0
try:
    rx = re.compile(pat, flags)
except re.error as exc:                       # report, do not traceback
    print(f"history-titles.sh: bad regex '{pat}': {exc}", file=sys.stderr)
    raise SystemExit(2)

files = []
for dirpath, _dirnames, names in os.walk(root):
    for name in names:
        if name.endswith(".md"):
            files.append(os.path.join(dirpath, name))

total_files = len(files)
by_id = {}
for path in files:
    issue = os.path.basename(path)[:-3]
    by_id.setdefault(issue, []).append(path)

hits = []
for issue, paths in by_id.items():
    best = None
    for path in sorted(paths):
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        title = lines[0].strip()[:160] if lines else ""
        if rx.search(title):
            best = (path, title, None)
            break
        if body and best is None:
            for ln in lines[1:]:
                if rx.search(ln):
                    best = (path, title, ln.strip()[:160])
                    break
    if best:
        hits.append((issue, best[0], best[1], best[2], len(paths)))

if not hits:
    print(f"0 hits for '{pat}' under {root}")
    print("(the prior-month corpus is not evidence of absence for a current issue)")
    raise SystemExit(0)

hits.sort(key=lambda h: (0, int(h[0])) if h[0].isdigit() else (1, h[0]))
print(f"{sum(1 for _ in hits)} hits / {len(hits)} distinct issue ids for '{pat}' under {root}"
      f" (titles{'+body' if body else ''}; showing up to {limit})")
share = len(hits) / total_files if total_files else 0.0
if share > 0.3 and not body:
    print(f"!! {share:.0%} of the {total_files} issue titles match '{pat}' - the pattern, not the")
    print("   corpus, is the problem: split the alternation and keep the distinctive")
    print("   token (an op name, an error string, an error code).")
for issue, path, title, hit, copies in hits[:limit]:
    where = os.path.relpath(path, root)
    extra = f" (x{copies})" if copies > 1 else ""
    if hit:
        print(f"{where} :: {title} [body] {hit}{extra}")
    else:
        print(f"{where} :: {title}{extra}")
if len(hits) > limit:
    print(f"... ({len(hits) - limit} more; raise limit or narrow the pattern)")
PY
