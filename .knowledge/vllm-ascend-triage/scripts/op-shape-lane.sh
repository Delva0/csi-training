#!/usr/bin/env bash
# op-shape-lane.sh -- classify a vllm-ascend op by launch lane, which is what
# decides whether msprof's kernel_details.csv can carry its Input/Output Shapes.
#
# usage: bash op-shape-lane.sh <vllm-ascend-root> [op-name ...]
#
# No op-name: print the custom-op build list (csrc/build_aclnn.sh CUSTOM_OPS*)
# and the csrc/adapter binding counts, i.e. the two sets that can be compared
# with a kernel_details.csv that has shapes for some ops and N/A for others.
# With op-name(s): for each name print
#   aclnn-built      -> the name is in a CUSTOM_OPS* list, so an aclnn<Op>
#                       entry point is built and bound through EXEC_NPU_CMD;
#                       the aclnn op executor registers tensor metadata, so
#                       shapes are expected in the profiler table
#   raw-kernel-lane  -> the name has a SetCustomHandler site and no aclnn
#                       binding: the lambda launches the AscendC kernel
#                       directly, nothing registers an op/tensor desc with the
#                       profiler, and N/A shapes are the expected outcome
#   unknown          -> not in the build lists and no adapter site at this pin
#
# The script extracts sites; it does not prove what the installed CANN reports.
# Quote its output as the check and keep the claim labelled (SKILL.md rule 6).
# Exit: 0 classified, 2 usage/root error. Requires rg (falls back to grep -rn).
set -u

ROOT="${1:-}"
if [ -z "$ROOT" ] || [ ! -d "$ROOT/csrc" ]; then
  echo "usage: bash op-shape-lane.sh <vllm-ascend-root> [op-name ...]" >&2
  echo "error: '$ROOT' has no csrc/ directory" >&2
  exit 2
fi
shift

SEARCH="rg -n --no-heading"
command -v rg >/dev/null 2>&1 || SEARCH="grep -rnH"

echo "root: $ROOT"
if [ -f "$ROOT/requirements.txt" ]; then
  echo "pin:  $(grep -h -m1 'triton-ascend' "$ROOT/requirements.txt" 2>/dev/null | tr -d '\r')"
fi

BUILD="$ROOT/csrc/build_aclnn.sh"
if [ -f "$BUILD" ]; then
  echo
  echo "== CUSTOM_OPS lists in csrc/build_aclnn.sh (ops with a built aclnn interface) =="
  grep -n -E 'CUSTOM_OPS(_ARRAY)?(=|\()' "$BUILD" | tr -d '\r'
  grep -n -E '^\s+"[a-z0-9_]+,?"\s*$' "$BUILD" | tr -d '\r' | sed 's/^/  entry /'
else
  echo
  echo "warn: $BUILD absent -- no custom-op build list at this pin"
fi

echo
echo "== binding style counts in csrc (adapter headers are what a torch op runs) =="
echo "EXEC_NPU_CMD sites : $($SEARCH 'EXEC_NPU_CMD\(' "$ROOT/csrc" 2>/dev/null | wc -l | tr -d ' ')"
echo "SetCustomHandler   : $($SEARCH 'SetCustomHandler' "$ROOT/csrc" 2>/dev/null | wc -l | tr -d ' ')"
$SEARCH 'SetCustomHandler' "$ROOT/csrc" 2>/dev/null | sed 's/^/  custom-handler /'

if [ "$#" -eq 0 ]; then
  echo
  echo "next: re-run with the op names from the profiler table (e.g. mla_preprocess)"
  exit 0
fi

rc=0
for op in "$@"; do
  echo
  echo "== $op =="
  inlist=""
  if [ -f "$BUILD" ]; then
    inlist=$(grep -c -E "(^|[^A-Za-z0-9_])$op([^A-Za-z0-9_]|$)" "$BUILD" 2>/dev/null | tr -d ' ')
  fi
  files=$($SEARCH "$op" "$ROOT/csrc" 2>/dev/null | cut -d: -f1 | sort -u)
  echo "  build-list hits: ${inlist:-0}"
  if [ -z "$files" ]; then
    echo "  files in csrc   : none"
  else
    echo "  csrc files mentioning it: $(echo "$files" | wc -w | tr -d ' ') (lanes of the adapter files follow)"
    for f in $files; do
      lane=""
      if grep -q "EXEC_NPU_CMD(" "$f" 2>/dev/null; then lane="EXEC_NPU_CMD"; fi
      if grep -q "SetCustomHandler" "$f" 2>/dev/null; then lane="${lane:+$lane + }SetCustomHandler"; fi
      [ -z "$lane" ] && continue
      echo "    ${f#$ROOT/}  [$lane]"
      grep -n -E "EXEC_NPU_CMD\(|SetCustomHandler" "$f" 2>/dev/null | grep -E "aclnn|SetCustomHandler" | head -3 | sed 's/^/      /'
    done
  fi
  op_aclnn=0; op_handler=0
  for f in $files; do
    case "$f" in
      */"$op"/*|*/"$op"_*|*/"$op".*) ;;
      *) continue ;;
    esac
    a=$(grep -c "EXEC_NPU_CMD(" "$f" 2>/dev/null); a=${a:-0}
    h=$(grep -c "SetCustomHandler" "$f" 2>/dev/null); h=${h:-0}
    op_aclnn=$((op_aclnn + a))
    op_handler=$((op_handler + h))
  done
  if [ "${inlist:-0}" != "0" ]; then
    echo "  verdict: aclnn-built -- this name is in a CUSTOM_OPS list, so an aclnn entry point is"
    echo "           built and reached through EXEC_NPU_CMD; the aclnn op executor registers the"
    echo "           tensors, so shapes in kernel_details.csv are expected"
  elif [ "$op_handler" -gt 0 ] && [ "$op_aclnn" -eq 0 ]; then
    echo "  verdict: raw-kernel-lane -- the op's own adapter has a SetCustomHandler site and never"
    echo "           calls EXEC_NPU_CMD, so the lambda launches the AscendC kernel directly and no"
    echo "           aclnn op executor / tensor desc is registered: N/A Input/Output Shapes are this"
    echo "           lane's expected outcome, not a dispatch bug. The discriminator is what the"
    echo "           lambda invokes, NOT SetCustomHandler itself (EXEC_NPU_CMD sets it too, line above)"
  elif [ "$op_aclnn" -gt 0 ]; then
    echo "  verdict: aclnn lane by binding -- the op's own adapter calls EXEC_NPU_CMD(aclnn<Op>, ...);"
    echo "           shapes are expected even though the macro itself goes through SetCustomHandler"
  elif [ -n "$files" ]; then
    echo "  verdict: check the per-file lanes above -- only files outside the op directory mention it"
  else
    echo "  verdict: unknown at this pin -- not in a CUSTOM_OPS list and no csrc mention; it may be"
    echo "           a built-in aclnn op from the CANN inventory, or newer than this tree"
  fi
done
exit $rc
