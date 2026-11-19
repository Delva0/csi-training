#!/usr/bin/env bash
# Fingerprint the ref a traceback was produced from: for a file and a statement
# pattern, print the 1-based line number of that statement at each tag/branch.
# A traceback gives path + line; prose version claims are often wrong or mixed.
#
# usage: bash ref_line_of.sh <repo-dir|repo-name> <path-in-repo> <grep-pattern> [ref ...]
#   no refs given: all tags plus origin/releases/* branches, newest first
set -u
[ "$#" -ge 3 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
repo=$1; path=$2; pat=$3; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=$(bash "$HERE/repo_root.sh" "$repo") || exit 3
cd "$repo" || { echo "no such repo path: $repo" >&2; exit 2; }

if [ "$#" -gt 0 ]; then
  refs=("$@")
else
  refs=()
  while IFS= read -r t; do refs+=("$t"); done < <(git tag --sort=-creatordate)
  while IFS= read -r b; do refs+=("$b"); done \
    < <(git for-each-ref --format='%(refname:short)' refs/remotes/origin/releases)
fi

for r in "${refs[@]}"; do
  line=$(git show "$r:$path" 2>/dev/null | grep -n -E -- "$pat" | head -1 | cut -d: -f1)
  [ -n "$line" ] && printf '%-22s %s\n' "$r" "$line"
done
