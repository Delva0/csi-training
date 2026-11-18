#!/usr/bin/env bash
# Map a repo argument to an existing clone root. Accepts a path, "<owner>/<repo>", or a bare
# clone name ("vllm-ascend", "vllm", "Mooncake", "torchair").
#
#   bash repo_root.sh [<repo>]
#
# stdout: the clone root, one line, exit 0. An empty argument means
#         /workspace/repos/vllm-project/vllm-ascend (the usual default).
# stderr + exit 3: no such clone — prints the clones that do exist.
#
# Why this exists: callers pass the repo as a NAME (`open_answer.sh "…#N — symptom" 0.20.2rc1
# vllm-ascend`), and a bare `git -C vllm-ascend` fails every git call, which is indistinguishable
# from "the reported tag is absent from the clone". Recorded runs then spent a message listing
# `git tag --list` and reported the tag as missing while `git tag --list 'v0.20.2*'` showed it.
set -uo pipefail
ROOT=${CLONE_ROOT:-/workspace/repos}

arg=${1:-}
if [ -z "$arg" ]; then
  printf '%s\n' "$ROOT/vllm-project/vllm-ascend"
  exit 0
fi

is_clone() { git -C "$1" rev-parse --git-dir >/dev/null 2>&1; }

# 1. A path as written, absolute or relative to the caller's directory.
for c in "$arg" "$PWD/$arg"; do
  if is_clone "$c"; then (cd "$c" && pwd); exit 0; fi
done

# 2. "<owner>/<repo>" resolved under the clone root.
if [ -d "$ROOT/$arg" ] && is_clone "$ROOT/$arg"; then (cd "$ROOT/$arg" && pwd); exit 0; fi

# 3. A bare repo name: unique match one level below the clone root.
name=$(basename "$arg")
matches=()
while IFS= read -r d; do
  [ -n "$d" ] && is_clone "$d" && matches+=("$d")
done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -type d -name "$name" 2>/dev/null | sort)

if [ "${#matches[@]}" -eq 1 ]; then
  (cd "${matches[0]}" && pwd)
  exit 0
fi
if [ "${#matches[@]}" -gt 1 ]; then
  echo "ambiguous repo '$arg': ${matches[*]}" >&2
  echo "pass the full path, e.g. ${matches[0]}" >&2
  exit 3
fi

echo "no clone named '$arg' under $ROOT ($0)" >&2
echo "available (owner/repo):" >&2
while IFS= read -r d; do
  is_clone "$d" && printf '  %s\n' "${d#"$ROOT"/}"
done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort) | head -20 >&2
echo "fix the repo argument; this is not evidence that a tag or path is absent" >&2
exit 3
