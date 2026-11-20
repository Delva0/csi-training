#!/usr/bin/env bash
# Cheap precedent index over the read-only issue history.
#
#   scripts/history-titles.sh '<extended-regex>' [history-root] [limit]
#
# Prints one line per matching issue file:  <file> :: <first non-empty line>
# so a whole family of issues can be triaged without reading any body.
# Never `ls` the history directory: it holds thousands of files.
set -u

pat=${1:-}
if [ -z "${pat:-}" ]; then
  echo "usage: history-titles.sh <extended-regex> [history-root] [limit]" >&2
  exit 2
fi
root=${2:-/workspace/history}
limit=${3:-40}

if [ ! -d "$root" ]; then
  echo "no such history root: $root" >&2
  exit 2
fi

# Layouts seen: flat  history/<issue>.md  and per-case  history/<issue>/<id>.md.
matches=$(grep -rlE -e "$pat" --include='*.md' "$root" 2>/dev/null)
if [ -z "$matches" ]; then
  echo "0 hits for '$pat' under $root"
  echo "(the prior-month corpus is not evidence of absence for a current issue)"
  exit 0
fi

sortkey() { b=${1##*/}; echo "${b%.md}"; }
total=$(printf '%s\n' "$matches" | wc -l | tr -d ' ')
shown=0
echo "$total hits for '$pat' under $root (showing up to $limit)"
# numeric where possible, then lexical, so issue numbers order naturally
for f in $(printf '%s\n' "$matches" | while read -r p; do printf '%s\t%s\n' "$(sortkey "$p")" "$p"; done \
           | sort -t"$(printf '\t')" -k1,1n -k1,1 | cut -f2); do
  shown=$((shown + 1))
  [ "$shown" -gt "$limit" ] && { echo "... ($((total - limit)) more; raise limit or narrow the pattern)"; break; }
  title=$(grep -m1 -v '^[[:space:]]*$' "$f" 2>/dev/null | cut -c1-160)
  printf '%s :: %s\n' "${f#"$root"/}" "$title"
done
