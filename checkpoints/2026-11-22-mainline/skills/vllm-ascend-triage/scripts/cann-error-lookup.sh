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
#   runtime 6-digit 507xxx / 107xxx / 207xxx     -> the ACL_ERROR_RT_* define with the
#               header's own comment, plus the vendor table row (meaning + action):
#                 ACL_ERROR_RT_LOST_HEARTBEAT    507010 // ts lost heartbeat
#                 ... from <root>/*/cann/runtime/include/external/acl/error_codes/
#                     rt_error_codes.h  and  <root>/*/cann/runtime/docs/api_docs/aclError.md
#               These are runtime/device codes, not HcclResult enums: 5070xx is the TS,
#               aicore or device lane (507010 lost heartbeat, 507014-507016 aicore
#               timeout/exception/trap), so the name decides the lane, not a source read.
#   hex TS code 0x2A, 0x90B                      -> the task-scheduler (TS) error enum:
#               `TS_ERROR_AICPU_EXCEPTION = 0x2A   /**< aicpu exception */` from
#               <root>/*/cann/driver/src/sdk_driver/ts_agent/inc/tsch/task_scheduler_error.h
#               (`typedef enum tag_ts_error`, mirrored under cann/hcomm). This is the family
#               of a plog's `retCode=0x2a, [aicpu exception]`: the bracketed name IS that
#               enum's comment, and no ACL_ERROR_RT_* carries the value.
#   errcode N   the `errcode:N, msg:<text>` field of an AICPU exception record -> an
#               `enum rtAICPUErrorType` name from
#               <root>/*/cann/runtime/src/runtime/feature/inc/device/device_error_info.hpp
#               (1 = KERNEL_EXECUTE_PARAM_INVLID "kernel execute param invalid",
#                3 = KERNEL_EXECUTE_TIMEOUT, 7 = AE_STATUS_SILENT_FAULT,
#                21001 = AICPU_PARAMETER_NOT_VALID, ...). Every decimal argument also gets
#               this lookup, as its own heading: these small numbers collide with the
#               HcclResult enum, and only the surrounding line says which family applies.
#   dcmi -8xxx  e.g. -8005, -8020                 -> the DCMI return code the host's
#               `npu-smi` prints (`dcmi module initialize failed. ret is -8005`): the
#               enum name from <root>/*/cann/driver/src/custom/include/dcmi_interface_api.h
#               (`DCMI_ERROR_CODE_BASE (-8000)` + `DCMI_ERR_CODE_<NAME> (BASE - N)`), the
#               vendor row from <root>/*/cann/driver/docs/**/return_codes.md, and — under
#               a separate heading — any `LQ_DCMI_ERR_CODE_*` collision from
#               `src/custom/lqdrv/kernel/ioctl_comm_def.h`, whose numbering is a
#               DIFFERENT table (-8005 is IOCTL_FAIL there, INNER_ERR in DCMI).
#               -8020 = DCMI_ERR_CODE_RESOURCE_OCCUPIED, vendor action 「请排查设备是否
#               被其他容器、进程等占用」 — the code that names the another-holder lane.
#               Keep the default --root /workspace/repos for this branch, not only the
#               reported issue's directory: a report whose snapshot exports no driver tree
#               (the v0.8.4/v0.9-era ones) still gets the name from any other pin, because
#               a return-code table is an ABI — cite the pin that answered.
#
# --root defaults to /workspace/repos; pass /workspace/repos/<issue> to scope it.
# Read the printed Solution as the vendor's statement (citable), not as a diagnosis:
# the code still has to be matched to the line that produced it (SKILL.md rule 6).
# Exit 0 when every argument found something, 1 otherwise; 2 on bad usage.
#
# Originating cases: #3172 `hcclCommInitRootInfoConfig ... error code is 7` beside
# `EJ0001 Failed to initialize the HCCP process. Reason: Maybe the last training
# process is running.` (references/runtime-memory/sleep-mode-multi-instance.md); the
# runtime branch from #11171, where `runtime result = 507010` answered as
# "no HCCL_E_* = 507010" and the reporter's own case ended as a hardware fault
# (references/hccl-and-hardware/device-health-checks.md); the dcmi branch from #8809
# (references/startup-failures/no-npu-in-container.md); the TS and rtAICPUErrorType
# branches from #12776 (`retCode=0x2a, [aicpu exception]` + `errcode:1, msg:execute kernel
# param invalid` on a v0.23.0rc1 DeepSeek-V4 MTP graph launch), where the ACL tables said
# "not in the exported tables" and the two answers were one find away
# (references/hccl-and-hardware/aicpu-task-exception-empty-kernel.md).
set -uo pipefail

root="${CANN_ERROR_LOOKUP_ROOT:-/workspace/repos}"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --root) root="${2:?--root needs a directory}"; shift 2 ;;
    -h|--help) sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d' \
                 | sed 's/^# \{0,1\}//'; exit 0 ;;
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
rt_headers=()
while IFS= read -r f; do rt_headers+=("$f"); done < <(
  find "$root" -maxdepth 10 -type f -name 'rt_error_codes.h' 2>/dev/null | sort)
rt_docs=()
while IFS= read -r f; do rt_docs+=("$f"); done < <(
  find "$root" -maxdepth 10 -type f -name 'aclError.md' 2>/dev/null | sort)
dcmi_headers=()
while IFS= read -r f; do dcmi_headers+=("$f"); done < <(
  find "$root" -maxdepth 12 -type f -name 'dcmi_interface_api.h' 2>/dev/null | sort)
dcmi_docs=()
while IFS= read -r f; do dcmi_docs+=("$f"); done < <(
  find "$root" -maxdepth 12 -type f -name 'return_codes.md' 2>/dev/null | sort)
ts_headers=()
while IFS= read -r f; do ts_headers+=("$f"); done < <(
  find "$root" -maxdepth 12 -type f -name 'task_scheduler_error.h' 2>/dev/null | sort)
aicpu_headers=()
while IFS= read -r f; do aicpu_headers+=("$f"); done < <(
  find "$root" -maxdepth 12 -type f -name 'device_error_info.hpp' 2>/dev/null | sort)
lq_headers=()
while IFS= read -r f; do lq_headers+=("$f"); done < <(
  find "$root" -maxdepth 14 -type f -name 'ioctl_comm_def.h' 2>/dev/null | sort)

echo "root: $root"
echo "tables: ${#err_tables[@]}   enum headers: ${#enum_headers[@]}   rt headers: ${#rt_headers[@]}   rt doc tables: ${#rt_docs[@]}   dcmi headers: ${#dcmi_headers[@]}   ts tables: ${#ts_headers[@]}   aicpu type tables: ${#aicpu_headers[@]}"
if [ ${#err_tables[@]} -eq 0 ]; then
  echo "  (no CANN error table exported in this snapshot — quote the code from the paste"
  echo "   and look the enum up separately; e.g. the v0.9.1-era cann/ export holds only catlass)"
fi
echo

status=0
for code in "${args[@]}"; do
  echo "=== $code ==="
  found=0
  if [[ "$code" =~ ^-[0-9]+$ && "$code" -le -8000 && "$code" -ge -8099 ]]; then
    for h in "${dcmi_headers[@]}"; do
      base=$(rg -o --no-heading 'define[[:space:]]+DCMI_ERROR_CODE_BASE[[:space:]]+\((-?[0-9]+)\)' \
               -r '$1' "$h" 2>/dev/null | head -1)
      [ -z "$base" ] && continue
      n=$(( base - code ))
      hit=$(rg -n --no-heading "DCMI_ERR_CODE_[A-Z0-9_]+[[:space:]]+\((DCMI_ERROR_CODE_BASE|BASE)[[:space:]]*-[[:space:]]*${n}\)" \
              "$h" 2>/dev/null | head -3)
      [ -z "$hit" ] && continue
      found=1
      echo "  $h  (base $base, offset -$n)"
      while IFS= read -r line; do echo "    $line"; done <<< "$hit"
    done
    for d in "${dcmi_docs[@]}"; do
      line=$(rg -n --no-heading "(^|[^0-9.])${code}([^0-9.]|$)" "$d" 2>/dev/null | head -1)
      [ -z "$line" ] && continue
      found=1
      start=${line%%:*}
      echo "  $d:$start (vendor row, tags stripped)"
      sed -n "$(( start )),$(( start + 24 ))p" "$d" | sed 's/<[^>]*>//g' | tr -s ' \n' ' \n' \
        | grep -v '^ *$' | head -8 | sed 's/^/    /'
    done
    if [ ${#lq_headers[@]} -gt 0 ]; then
      for f in "${lq_headers[@]}"; do
        lq=$(rg -n --no-heading "#define LQ_DCMI_ERR_CODE_[A-Z0-9_]+[[:space:]]+\(${code}\)" "$f" 2>/dev/null | head -2)
        [ -z "$lq" ] && continue
        echo "  SEPARATE TABLE, do not cross-apply — $f"
        while IFS= read -r line; do echo "    $line"; done <<< "$lq"
      done
    fi
    if [ "$found" -eq 0 ]; then
      echo "  not in any exported dcmi_interface_api.h or DCMI return_codes.md"
      echo "  (widen the root before hedging the name: --root /workspace/repos — another"
      echo "   pin's driver export answers, the table is an ABI. Only the holder is open)"
      status=1
    fi
  elif [[ "$code" =~ ^0[xX][0-9a-fA-F]+$ ]]; then
    for h in "${ts_headers[@]}"; do
      hit=$(rg -n --no-heading -i "TS_[A-Z0-9_]+[[:space:]]*=[[:space:]]*${code}([^0-9a-fA-F]|$)" \
              "$h" 2>/dev/null | head -3)
      [ -z "$hit" ] && continue
      found=1
      echo "  task-scheduler (TS) error code — $h"
      while IFS= read -r line; do echo "    $line"; done <<< "$hit"
      echo "  (a TS code belongs to the driver/task-scheduler lane; the bracketed text a"
      echo "   plog prints beside retCode= is this enum's comment, so it adds no mechanism"
      echo "   beyond the name — references/hccl-and-hardware/device-health-checks.md)"
      break   # three copies of this header ride along in the exports; one is enough
    done
    if [ "$found" -eq 0 ]; then
      echo "  no TS_* = $code in any exported task_scheduler_error.h"
      echo "  (that header ships with cann/driver; a snapshot without the driver tree"
      echo "   cannot resolve a TS code — quote it as text)"
      status=1
    fi
  elif [[ "$code" =~ ^[0-9]+$ ]]; then
    for h in "${enum_headers[@]}"; do
      hit=$(rg -n "HCCL_E_[A-Z_]+\s*=\s*${code}\s*," "$h" 2>/dev/null | head -3)
      if [ -n "$hit" ]; then
        while IFS= read -r line; do echo "  $h:$line"; done <<< "$hit"
        found=1
      fi
    done
    # runtime codes: many copies of rt_error_codes.h ride along in the exports, so print
    # the canonical cann/runtime copy once (else the first copy) instead of all of them.
    rthit=""
    for h in "${rt_headers[@]}"; do
      hit=$(rg -n --no-heading "^#define[[:space:]]+ACL_ERROR_RT_[A-Z0-9_]+[[:space:]]+${code}\b" "$h" 2>/dev/null | head -1)
      [ -z "$hit" ] && continue
      rthit="$h:$hit"
      case "$h" in */cann/runtime/*) break ;; esac
    done
    if [ -n "$rthit" ]; then
      found=1
      echo "  $rthit"
      echo "  (runtime/device code: the header comment names the owner — 5070xx is TS/aicore/"
      echo "   device, i.e. below vllm-ascend; match the number to the op that printed it,"
      echo "   SKILL.md rule 13)"
    fi
    for d in "${rt_docs[@]}"; do
      hit=$(rg -n "= ${code}; \|" "$d" 2>/dev/null | head -2)
      [ -z "$hit" ] && continue
      found=1
      while IFS= read -r line; do echo "  $d:$line"; done <<< "$hit"
    done
    for h in "${aicpu_headers[@]}"; do
      body=$(awk '/enum rtAICPUErrorType/{f=1} f{print NR": "$0} f&&/};/{exit}' "$h" 2>/dev/null)
      [ -z "$body" ] && continue
      hit=$(printf '%s\n' "$body" | rg "=[[:space:]]*${code}([^0-9]|$)" | head -3)
      [ -z "$hit" ] && continue
      found=1
      echo "  AICPU error type (the errcode: of an \`AICPU exception\` record) — $h"
      while IFS= read -r line; do echo "    $line"; done <<< "$hit"
      echo "  (separate namespace from HcclResult: 1 = KERNEL_EXECUTE_PARAM_INVLID, i.e. an"
      echo "   AICPU-executed kernel rejected its own execution params; 21001+ are the aicpu"
      echo "   scheduler/model/tool layers — references/hccl-and-hardware/"
      echo "   aicpu-task-exception-empty-kernel.md)"
    done
    if [ -z "$rthit" ] && [ "$found" -eq 0 ]; then
      echo "  no HCCL_E_* = $code in any exported hccl_types.h, no"
      echo "  ACL_ERROR_RT_* = $code in any exported rt_error_codes.h, and no"
      echo "  rtAICPUErrorType member = $code in any exported device_error_info.hpp"
    fi
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
