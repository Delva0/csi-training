#!/usr/bin/env bash
# Is a commit contained in a release ref? Distinguishes "fixed after the reported tag"
# from "fix already present, so not the explanation".
#
# usage: bash ref_ancestry.sh <repo-dir|repo-name> <ref> <commit> [commit ...]
set -u
[ "$#" -ge 3 ] || { sed -n '2,6p' "$0" >&2; exit 2; }
repo=$1; ref=$2; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=$(bash "$HERE/repo_root.sh" "$repo") || exit 3
cd "$repo" || { echo "no such repo path: $repo" >&2; exit 2; }
git rev-parse --verify --quiet "$ref^{commit}" >/dev/null \
  || { echo "unknown ref: $ref" >&2; exit 2; }

for c in "$@"; do
  if ! git rev-parse --verify --quiet "$c^{commit}" >/dev/null; then
    printf '%-12s %-8s %s\n' "$c" "MISSING" "not in this clone"
    continue
  fi
  if git merge-base --is-ancestor "$c" "$ref" 2>/dev/null; then st=IN; else st=OUT; fi
  printf '%-12s %-8s %s\n' "$c" "$st" "$(git log -1 --format='%ci %s' "$c")"
done
