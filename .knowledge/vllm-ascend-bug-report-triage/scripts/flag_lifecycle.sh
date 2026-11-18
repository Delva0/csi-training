#!/usr/bin/env bash
# Show where an --additional-config / env flag exists, changed or disappeared
# across release tags of the local vllm-ascend clone.
#
# usage: flag_lifecycle.sh <flag-name> [repo-root|repo-name]
set -uo pipefail
flag=${1:?usage: flag_lifecycle.sh <flag-name> [repo-root|repo-name]}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo=$(bash "$HERE/repo_root.sh" "${2:-}") || exit 3
py='*.py'
echo "flag: $flag"
echo
printf '%-16s %s\n' TAG 'files(*.py)'
for tag in $(git -C "$repo" tag --list 'v0.1*' 'v0.2*' 'v0.3*' | sort -V); do
  n=$(git -C "$repo" grep -l -F -- "$flag" "$tag" -- "$py" 2>/dev/null | wc -l)
  printf '%-16s %s\n' "$tag" "$n"
done
echo
echo "--- release_notes mentions (docs/source/user_guide/release_notes.md @ HEAD) ---"
git -C "$repo" grep -n -F -- "$flag" HEAD -- docs/source/user_guide/release_notes.md 2>/dev/null | cut -c1-240 | head -10
echo
echo "--- commits that added/removed it (pickaxe, all refs) ---"
git -C "$repo" log --oneline --all -S"$flag" 2>/dev/null | head -12
echo
echo "--- current definition sites at HEAD ---"
git -C "$repo" grep -n -F -- "$flag" HEAD -- "$py" 2>/dev/null | cut -c1-200 | head -12
exit 0
