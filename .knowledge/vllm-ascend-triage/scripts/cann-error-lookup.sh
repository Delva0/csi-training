#!/usr/bin/env bash
# cann-error-lookup.sh <CODE|NUMBER> [<CODE|NUMBER> ...] [--root DIR]
#
# Looks a CANN/HCCL error code up in the pinned exports instead of guessing what it
# means. Two kinds of argument:
#
#   E-code  EJ0001, EI0006, EL0004, EI0017 ...  -> the CANN error table entry:
#               errClass, errTitle, ErrMessage, arguably the vendor's own
#               "Possible Cause" and "Solution" text.
#             Searched in (whichever the snapshot exports):
#               <root>/*/cann/metadef/error_manager/error_code.json   (one entry per line)
#               <root>/*/cann/hcomm/src/common/error_code/error_code.cc
#   number  0..21                               -> the HcclResult enum name, e.g.
#               7 = HCCL_E_UNAVAIL, from <root>/*/Ascend/pytorch/.../hccl_types.h
#
# --root defaults to /workspace/repos; pass /workspace/repos/<issue> to scope it.
# Read the printed Solution as the vendor's statement (citable), not as a diagnosis:
# the code still has to be matched to the line that produced it (SKILL.md rule 6).
# Exit 0 when every argument found something, 1 otherwise; 2 on bad usage.
#
# Originating case: #3172 `hcclCommInitRootInfoConfig ... error code is 7` beside
# `EJ0001 Failed to initialize the HCCP process. Reason: Maybe the last training
# process is running.` (references/runtime-memory/sleep-mode-multi-instance.md).
set -uo pipefail

root="${CANN_ERROR_LOOKUP_ROOT:-/workspace/repos}"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --root) root="${2:?--root needs a directory}"; shift 2 ;;
    -h|--help) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [ ${#args[@]} -eq 0 ]; then
  echo "usage: bash $0 <CODE|NUMBER> [...] [--root DIR]" >&2
  exit 2
fi
if [ ! -d "$root" ]; then
  echo "root not found: $root (pass --root /workspace/repos/<issue>)" >&2
  exit 2
fi

err_tables=()
enum_headers=()
while IFS= read -r f; do err_tables+=("$f"); done < <(
  find "$root" -maxdepth 7 -type f \
    \( -path '*error_manager/error_code.json' -o -path '*common/error_code/error_code.cc' \) \
    2>/dev/null | sort)
while IFS= read -r f; do enum_headers+=("$f"); done < <(
  find "$root" -maxdepth 9 -type f -name 'hccl_types.h' 2>/dev/null | sort)

echo "root: $root"
echo "tables: ${#err_tables[@]}   enum headers: ${#enum_headers[@]}"
if [ ${#err_tables[@]} -eq 0 ]; then
  echo "  (no CANN error table exported in this snapshot — quote the code from the paste"
  echo "   and look the enum up separately; e.g. the v0.9.1-era cann/ export holds only catlass)"
fi
echo

status=0
for code in "${args[@]}"; do
  echo "=== $code ==="
  found=0
  if [[ "$code" =~ ^[0-9]+$ ]]; then
    for h in "${enum_headers[@]}"; do
      hit=$(rg -n "HCCL_E_[A-Z_]+\s*=\s*${code}\s*," "$h" 2>/dev/null | head -3)
      if [ -n "$hit" ]; then
        while IFS= read -r line; do echo "  $h:$line"; done <<< "$hit"
        found=1
      fi
    done
    [ "$found" -eq 0 ] && echo "  no HCCL_E_* = $code in any exported hccl_types.h"
    [ "$found" -eq 1 ] && echo "  (hccl_types.h gives the enum name; the printed message names the precondition)"
  else
    for t in "${err_tables[@]}"; do
      if [[ "$t" == *.json ]]; then
        hits=$(rg -n --no-heading -B2 -A5 "\"ErrCode\": \"${code}\"" "$t" 2>/dev/null)
      else
        hits=$(rg -n --no-heading -A6 "\"ErrCode\": \"${code}\"" "$t" 2>/dev/null)
      fi
      [ -z "$hits" ] && continue
      found=1
      echo "  file: $t"
      printf '%s\n' "$hits" | sed \
        -e 's/,[[:space:]]*"errTitle"/\n    errTitle  /' \
        -e 's/,[[:space:]]*"ErrMessage"/\n    message   /' \
        -e 's/,[[:space:]]*"Arglist"/\n    arglist   /' \
        -e 's/,[[:space:]]*"suggestion"/\n    suggestion/' \
        -e 's/,[[:space:]]*"Possible Cause"/\n      cause   /' \
        -e 's/,[[:space:]]*"Solution"/\n      solution/' \
        -e 's/^\([0-9]*\)[-:]/\1: /' | head -20
    done
    if [ "$found" -eq 0 ]; then
      echo "  not in the exported tables — one wider grep before concluding:"
      echo "  rg -ln '\"ErrCode\": \"${code}\"' $root"
      status=1
    fi
  fi
  echo
done
exit $status
