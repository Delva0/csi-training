#!/usr/bin/env bash
# Cheap precedent index over the read-only issue history.
#
#   scripts/history-titles.sh [-i] '<extended-regex>' [history-root] [limit]
#
# Prints one line per matching issue file:  <file> :: <first non-empty line>
# so a whole family of issues can be triaged without reading any body.
# `-i` makes the file match case-insensitive (issue titles capitalise tokens:
# `reranker` vs `Reranker`); the body-line fallback below is always
# case-insensitive. Never `ls` the history directory: it holds thousands of files.
set -u

icase=""
case "${1:-}" in
  -i|--ignore-case) icase="-i"; shift ;;
  --) shift ;;
  -h|--help)
    echo "usage: history-titles.sh [-i] <extended-regex> [history-root] [limit]"
    exit 0 ;;
  -*)
    # A grep-style flag as the first argument used to be taken for the regex and
    # the pattern for the history root, burning a call on "no such history root".
    echo "history-titles.sh: unknown option '$1'" >&2
    echo "usage: history-titles.sh [-i] <extended-regex> [history-root] [limit]" >&2
    exit 2 ;;
esac

pat=${1:-}
if [ -z "${pat:-}" ]; then
  echo "usage: history-titles.sh [-i] <extended-regex> [history-root] [limit]" >&2
  exit 2
fi
root=${2:-/workspace/history}
limit=${3:-40}

if [ ! -d "$root" ]; then
  echo "no such history root: $root" >&2
  exit 2
fi

# Layouts seen: flat  history/<issue>.md  and per-case  history/<issue>/<id>.md.
# Note: keep the flag separate from -rlE — `grep -rl${icase}E` becomes the
# invalid `-rl-iE` and, with stderr dropped, silently reports 0 hits.
matches=$(grep -rlE ${icase:+"$icase"} -e "$pat" --include='*.md' "$root" 2>/dev/null)
if [ -z "$matches" ]; then
  echo "0 hits for '$pat' under $root"
  echo "(the prior-month corpus is not evidence of absence for a current issue)"
  exit 0
fi

sortkey() { b=${1##*/}; echo "${b%.md}"; }
# A case-relative root can hold the same issue id once per case directory
# (history/<issue>/<id>.md); keep one line per id and say how many copies exist,
# so `limit` is not spent on duplicates.
total=$(printf '%s\n' "$matches" | wc -l | tr -d ' ')
uniq=$(printf '%s\n' "$matches" | while read -r p; do sortkey "$p"; done | sort -u | wc -l | tr -d ' ')
echo "$total hits / $uniq distinct issue ids for '$pat' under $root (showing up to $limit)"
shown=0
# numeric where possible, then lexical, so issue numbers order naturally
for id in $(printf '%s\n' "$matches" | while read -r p; do sortkey "$p"; done | sort -t"$(printf '\t')" -k1,1n -k1,1 | awk '!seen[$0]++'); do
  shown=$((shown + 1))
  [ "$shown" -gt "$limit" ] && { echo "... ($((uniq - limit)) more; raise limit or narrow the pattern)"; break; }
  f=$(printf '%s\n' "$matches" | grep -m1 -E "/($id|$id\.md)$")
  [ -z "$f" ] && continue
  title=$(grep -m1 -v '^[[:space:]]*$' "$f" 2>/dev/null | cut -c1-160)
  # A body-only match yields a title that does not contain the pattern and looks
  # like noise. Show the matching body line so one call separates a real issue
  # family from a lexical accident.
  if printf '%s' "$title" | grep -qiE -e "$pat"; then
    printf '%s :: %s\n' "${f#"$root"/}" "$title"
  else
    hit=$(grep -m1 -iE -e "$pat" "$f" 2>/dev/null | cut -c1-160)
    printf '%s :: %s [body] %s\n' "${f#"$root"/}" "$title" "$hit"
  fi
done
