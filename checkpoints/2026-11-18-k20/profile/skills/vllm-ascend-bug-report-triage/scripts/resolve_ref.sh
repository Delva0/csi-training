#!/usr/bin/env bash
# Map the version string a reporter wrote ("0.18.0", "0.23.0rc1", "0.9.2RC1", "0.26.rc1", "0.7.3.post1",
# "0.21.0rc1-310p", "0.19.1rc2.dev849+g05e095a20") — or a ref already in git spelling — to an
# existing git ref in a clone. The suffix case is normalised (`0.9.2RC1` -> tag `v0.9.2rc1`), because a
# reporter copying a `collect_env` line writes it upper case.
#
#   bash resolve_ref.sh <version-or-ref> [repo-root|repo-name]
#
# stdout: the resolved ref (tag name, branch name or sha), one line, exit 0.
# stderr: the candidate tags that do exist near it, exit 1 — so a caller can print something useful
#         instead of a git fatal when the tag is absent from the clone. Exit 3 means the *repo*
#         argument names no clone, which is a different problem with a different fix.
# Read-only; never touches the index or the working tree.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
raw=${1:?usage: resolve_ref.sh <version-or-ref> [repo-root|repo-name]}
repo=$(bash "$HERE/repo_root.sh" "${2:-}") || exit 3

# Reporters and docs write the pre-release suffix in upper case ("0.9.2RC1", "v0.23.0RC1") while the tags
# are lower case; every lookup below therefore also tries the lower-cased spelling. Both are tried, raw
# first, so a genuinely mixed-case tag still wins.
raw_lc=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')

# 1. Already a ref in git spelling (tag with the v, branch, or full/short sha)?
for cand in "$raw" "$raw_lc"; do
  if git -C "$repo" rev-parse -q --verify "${cand}^{commit}" >/dev/null 2>&1; then
    printf '%s\n' "$cand"; exit 0
  fi
done

# 2. A development version string carries the built commit: ...dev849+g05e095a20 -> 05e095a20.
sha=$(printf '%s' "$raw_lc" | grep -oE 'g[0-9a-f]{7,40}' | head -1 | cut -c2- || true)
if [ -n "$sha" ] && git -C "$repo" rev-parse -q --verify "${sha}^{commit}" >/dev/null 2>&1; then
  printf '%s\n' "$sha"; exit 0
fi

# 3. Normalise the way reporters write versions: 0.26.rc1 / 0.26.0rc1 / 0.7.3.post1 / 0.21.0rc1-310p.
ver=$(printf '%s' "$raw_lc" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?((\.|-)?rc[0-9]+|\.post[0-9]+)?' | head -1 || true)
if [ -n "$ver" ]; then
  kind=""
  if [[ $ver =~ ^([0-9]+\.[0-9]+(\.[0-9]+)?)[.-]?(rc|post)([0-9]+)$ ]]; then
    ver=${BASH_REMATCH[1]}; kind="${BASH_REMATCH[3]}${BASH_REMATCH[4]}"
  fi
  [[ $ver =~ ^[0-9]+\.[0-9]+$ ]] && ver="$ver.0"
  case "$kind" in
    post*) cands=("v${ver}.${kind}") ;;
    rc*)   cands=("v${ver}${kind}" "v${ver}.${kind}") ;;
    *)     cands=("v${ver}") ;;
  esac
  for c in "${cands[@]}"; do
    if git -C "$repo" rev-parse -q --verify "refs/tags/$c" >/dev/null 2>&1; then
      printf '%s\n' "$c"; exit 0
    fi
    if git -C "$repo" rev-parse -q --verify "refs/heads/$c" >/dev/null 2>&1; then
      printf '%s\n' "$c"; exit 0
    fi
  done
  base=$(printf '%s' "$ver" | sed -E 's/\.(rc|post)[0-9]+$//')
  echo "unresolved: no tag or branch for '$raw' (tried: ${cands[*]}) in $repo" >&2
  echo "near tags:" >&2
  git -C "$repo" tag --list "v${base}*" "v${base%.*}*" | sort -V -u | tail -6 >&2
  echo "release branches:" >&2
  git -C "$repo" branch -r --format='%(refname:short)' 2>/dev/null \
    | grep -E "origin/releases/v${base%.*}([.rc]|$)" | sort -V | tail -3 >&2
  exit 1
fi

echo "unresolved: no version number and no such ref in '$raw' ($repo)" >&2
exit 1
