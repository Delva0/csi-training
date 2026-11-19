#!/usr/bin/env bash
# Map a version string printed by a bug reporter ("0.23.0rc1", "vLLM Ascend
# Version: 0.18.0rc1", "0.26.rc1", "0.21.0rc1-310p", "0.7.3.post1") to git refs
# in a local clone, and print the default-branch HEAD for contrast. Missing patch
# levels and a dotted ".rc1" are normalised to the tag spelling (vX.Y.ZrcN).
#
# Use this before reading any source for a report: the clone's checkout is
# usually far newer than the reporter's release, and paths/flags move.
#
# usage: version_ref.sh <version-string> [repo-root|repo-name]
# default repo-root: /workspace/repos/vllm-project/vllm-ascend
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
raw=${1:?usage: version_ref.sh <version-string> [repo-root|repo-name]}
repo=$(bash "$HERE/repo_root.sh" "${2:-}") || exit 3
# normalise the way reporters write versions: 0.26.rc1, 0.26.0rc1, v0.26.0rc1,
# 0.26.0-rc1, 0.9.2RC1 (collect_env prints the suffix upper case), 0.7.3.post1,
# 0.21.0rc1-310p -> vX.Y.Z[rcN|postN]
raw_lc=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
ver=$(printf '%s' "$raw_lc" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?((\.|-)?rc[0-9]+|\.post[0-9]+)?' | head -1)
[ -n "$ver" ] || { echo "no version number found in: $raw" >&2; exit 2; }
kind=""; alt=""
if [[ $ver =~ ^([0-9]+\.[0-9]+(\.[0-9]+)?)[.-]?(rc|post)([0-9]+)$ ]]; then
  ver=${BASH_REMATCH[1]}; kind="${BASH_REMATCH[3]}${BASH_REMATCH[4]}"
fi
[[ $ver =~ ^[0-9]+\.[0-9]+$ ]] && ver="$ver.0"
# tag spellings: post tags keep the dot (v0.7.3.post1); rc tags normally do not
# (v0.10.2rc1) but v0.28.0.rc1/v0.29.0.rc1 do, so keep that variant as a fallback.
case "$kind" in
  post*) ver="v${ver}.${kind}" ;;
  rc*)   alt="v${ver}.${kind}"; ver="v${ver}${kind}" ;;
  *)     ver="v${ver}" ;;
esac
echo "repo: $repo"
echo "parsed version: $ver"
echo
echo "--- exact tag ---"
if git -C "$repo" rev-parse -q --verify "refs/tags/$ver" >/dev/null; then
  git -C "$repo" log -1 --date=short --format='%h %ad %s' "$ver"
elif [ -n "$alt" ] && git -C "$repo" rev-parse -q --verify "refs/tags/$alt" >/dev/null; then
  echo "(tag has the dotted spelling $alt)"
  git -C "$repo" log -1 --date=short --format='%h %ad %s' "$alt"
else
  echo "(no tag named $ver${alt:+ or $alt})"
fi
echo
echo "--- near tags ---"
git -C "$repo" tag --list "${ver}*" | sort -V | head -8
echo
echo "--- release branches ---"
base=$(printf '%s' "$ver" | sed -E 's/(rc[0-9]+|\.post[0-9]+)$//')
{
  git -C "$repo" branch -r --list "origin/releases/${base}*"
  git -C "$repo" branch -r --list 'origin/releases/*' | grep -F "$base"
} | sort -u | head -8
echo
echo "--- default branch HEAD (never assume it is the reported version) ---"
git -C "$repo" log -1 --date=short --format='%h %ad %s' HEAD
exit 0
