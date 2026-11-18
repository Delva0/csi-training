#!/usr/bin/env bash
# Materialise a git ref into a scratch directory for whole-tree grep/read.
# The source clone is read-only, so no worktrees or checkouts.
#
# usage: bash export_ref.sh <repo-dir|repo-name> <ref> [dest-dir]
#   default dest: /workspace/scratch/refs/<owner>__<repo>__<ref>
set -eu
[ "$#" -ge 2 ] || { sed -n '2,7p' "$0" >&2; exit 2; }
repo=$1; ref=$2; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=$(bash "$HERE/repo_root.sh" "$repo") || exit 3
cd "$repo" || { echo "no such repo path: $repo" >&2; exit 1; }
git rev-parse --verify --quiet "$ref^{commit}" >/dev/null \
  || { echo "unknown ref: $ref" >&2; exit 1; }

top=$(git rev-parse --show-toplevel)
slug=$(printf '%s__%s' "$(basename "$(dirname "$top")")" "$(basename "$top")")
safe=$(printf '%s' "$ref" | tr '/ ' '__')
dest=${1:-/workspace/scratch/refs/${slug}__${safe}}
mkdir -p "$dest"
git archive "$ref" | tar -x -C "$dest"
echo "exported $ref -> $dest ($(find "$dest" -type f | wc -l) files)"
