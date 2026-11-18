#!/usr/bin/env bash
# Grep a pattern at git refs of a (read-only) clone.
# Use when the reported traceback path is gone at HEAD, or when `git log -S` returns nothing.
#
# usage: bash ref_grep.sh <repo-dir|repo-name> <pattern> [ref ...]
#   no refs given: newest 6 tags plus origin/releases/* branches are used
# exit: 0 if any ref matched, 1 otherwise
set -u
usage() { sed -n '2,8p' "$0" >&2; exit 2; }
[ "$#" -ge 2 ] || usage
repo=$1; pat=$2; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=$(bash "$HERE/repo_root.sh" "$repo") || exit 3
cd "$repo" || { echo "no such repo path: $repo" >&2; exit 2; }

if [ "$#" -gt 0 ]; then
  refs=("$@")
else
  refs=()
  while IFS= read -r t; do refs+=("$t"); done < <(git tag --sort=-creatordate | head -6)
  while IFS= read -r b; do refs+=("$b"); done \
    < <(git for-each-ref --format='%(refname:short)' refs/remotes/origin/releases | tail -6)
fi
[ "${#refs[@]}" -gt 0 ] || { echo "no candidate refs; pass them explicitly" >&2; exit 2; }

found=1
for r in "${refs[@]}"; do
  out=$(git grep -n -I -- "$pat" "$r" -- '*.py' '*.md' '*.json' '*.yaml' '*.yml' 2>/dev/null | head -20)
  if [ -n "$out" ]; then
    printf '== %s\n%s\n' "$r" "$out"
    found=0
  else
    printf '== %s (no match)\n' "$r"
  fi
done
exit "$found"
