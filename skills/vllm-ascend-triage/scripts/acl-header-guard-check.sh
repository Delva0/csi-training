#!/usr/bin/env bash
# acl-header-guard-check.sh — compare the two `acl/acl_base.h` copies that a
# host-side C++ JIT translation unit can end up including, and say whether they
# collide on one include guard.
#
# Background: torch_npu ships a *vendored* copy of the CANN ACL headers under
# `third_party/acl/inc/acl/` (it also builds against that copy), and a newer CANN
# install ships its own `acl/acl_base.h` under a different tree.  Both copies use
# the guard `INC_EXTERNAL_ACL_ACL_BASE_H_`.  Only the torch_npu copy typedefs
# `aclOpExecutor`; newer CANN copies do not declare it at all and no longer ship
# `acl/acl_op_compiler.h`.  In one translation unit the guard makes the *first*
# copy win, so a TU that mixes the two include roots can compile
# `AclOpCompileInterface.h:146` with no `aclOpExecutor` in scope:
#   error: use of undeclared identifier 'aclOpExecutor'; did you mean 'aclopExecute'?
# The compile fails in torch_npu's own header, but the owner is the wheel/CANN
# header pair that put two copies in one TU — see
# references/image-dependency-faults/host-cxx-compile-header-conflict.md.
#
# It reports facts, not the cause: it prints each copy's path, include guard,
# whether it typedefs `aclOpExecutor` and whether `acl_op_compiler.h` exists
# beside it, then the collision verdict.
#
#   scripts/acl-header-guard-check.sh --torch-npu-root /path/to/torch_npu [--torch-npu-root ...]
#                                     --cann-root /usr/local/Ascend/ascend-toolkit/latest
#   scripts/acl-header-guard-check.sh --torch-npu-root /root/site-packages/torch_npu \
#                                     --cann-root /usr/local/Ascend/cann-8.5.0
#   scripts/acl-header-guard-check.sh --sources /workspace/repos   # pinned exports
#
# Notes for reading the output: a guard collision needs BOTH copies present under
# the same value; a missing copy is reported as absent, and a TU that cannot find
# `third_party/acl/inc/acl/acl_base.h` fails differently (`fatal error: ...
# file not found`), so an "undeclared identifier" at AclOpCompileInterface.h is
# never the missing-file case.
set -u

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

np_roots=(); cann_roots=(); sources=""
while [ $# -gt 0 ]; do
  case "$1" in
    --torch-npu-root) np_roots+=("${2:-}"); shift 2 ;;
    --cann-root) cann_roots+=("${2:-}"); shift 2 ;;
    --sources) sources="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -n "$sources" ]; then
  [ -d "$sources/Ascend/pytorch" ] && np_roots+=("$sources/Ascend/pytorch")
  for c in "$sources/cann/runtime" "$sources/cann/asc-devkit"; do
    [ -d "$c" ] && cann_roots+=("$c")
  done
fi

if [ "${#np_roots[@]}" -eq 0 ] && [ "${#cann_roots[@]}" -eq 0 ]; then
  tp="$(uv run --no-project python -c 'import os,torch_npu;print(os.path.dirname(torch_npu.__file__)+"/include")' 2>/dev/null || true)"
  [ -n "$tp" ] && np_roots+=("$tp")
  for v in ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH ASCEND_OPP_PATH; do
    eval "r=\${$v:-}" 2>/dev/null || r=""
    [ -n "${r:-}" ] && [ -d "$r" ] && { cann_roots+=("$r"); break; }
  done
  if [ "${#np_roots[@]}" -eq 0 ] && [ "${#cann_roots[@]}" -eq 0 ]; then
    echo "no roots given and none discovered; pass --torch-npu-root/--cann-root" >&2
    exit 2
  fi
fi

# find_base ROOT — first existing acl_base.h under the layouts seen in wheels,
# source trees and CANN install roots.
find_base() {
  local root="$1" cand
  for cand in \
    "$root/third_party/acl/inc/acl/acl_base.h" \
    "$root/include/third_party/acl/inc/acl/acl_base.h" \
    "$root/include/acl/acl_base.h" \
    "$root/include/external/acl/acl_base.h" \
    "$root/acl/acl_base.h"; do
    [ -f "$cand" ] && { echo "$cand"; return 0; }
  done
  return 1
}

declare -a guards=() typedefs=() files=()
report() {
  local label="$1" root="$2" f guard typ compiler
  echo "== $label: $root"
  if ! f="$(find_base "$root")"; then
    echo "   acl_base.h: ABSENT under this root (not a candidate copy)"
    return
  fi
  guard="$(grep -m1 -o '^#ifndef[[:space:]]\+[A-Za-z0-9_]*' "$f" | awk '{print $2}')"
  typ="no"; grep -q 'typedef struct aclOpExecutor aclOpExecutor' "$f" && typ="yes"
  compiler="absent"; [ -f "$(dirname "$f")/acl_op_compiler.h" ] && compiler="present"
  echo "   acl_base.h:        $f"
  echo "   include guard:     ${guard:-<none found>}"
  echo "   aclOpExecutor:     typedef ${typ}"
  echo "   acl_op_compiler.h: ${compiler} in the same directory"
  files+=("$f"); guards+=("${guard:-}"); typedefs+=("$typ")
}

for r in "${np_roots[@]:-}"; do [ -n "$r" ] && report "torch_npu copy" "$r"; done
for r in "${cann_roots[@]:-}"; do [ -n "$r" ] && report "CANN copy" "$r"; done

echo
if [ "${#files[@]}" -lt 2 ]; then
  echo "verdict: only one acl_base.h copy found — a guard collision cannot be shown from these roots."
  echo "         Re-run with the torch_npu wheel's include root AND the CANN install root."
  exit 0
fi

same=1; for g in "${guards[@]}"; do [ "$g" = "${guards[0]}" ] || same=0; done
typ_yes=0; typ_no=0
for t in "${typedefs[@]}"; do [ "$t" = yes ] && typ_yes=$((typ_yes+1)); [ "$t" = no ] && typ_no=$((typ_no+1)); done

if [ "$same" -eq 1 ] && [ "$typ_yes" -ge 1 ] && [ "$typ_no" -ge 1 ]; then
  echo "verdict: GUARD COLLISION. Both copies use ${guards[0]} and they disagree on"
  echo "         aclOpExecutor, so the first copy included in a translation unit wins"
  echo "         and the other body is skipped. This reproduces"
  echo "         'use of undeclared identifier aclOpExecutor' at"
  echo "         AclOpCompileInterface.h:<n> whenever the CANN copy is reached first"
  echo "         in a TU that also includes torch_npu's OpParamMaker.h."
  exit 1
fi
if [ "$same" -eq 1 ]; then
  echo "verdict: same guard, same aclOpExecutor status — no collision from these copies."
elif [ "$typ_yes" -ge 1 ] && [ "$typ_no" -ge 1 ]; then
  echo "verdict: guards differ, so the copies are independent; a mixed TU is legal here"
  echo "         and this error must be explained another way (report the two guards)."
else
  echo "verdict: no copy without the typedef — nothing here predicts the error."
fi
exit 0
