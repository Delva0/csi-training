#!/usr/bin/env bash
# Decode a vllm-ascend device-type failure without reading the tree.
#
#   bash device-type-check.sh <paste|serve-log|->            # assertion / soc map text
#   bash device-type-check.sh --soc-version 260             # classify one numeric code
#   bash device-type-check.sh --soc-version 260 --stamp A2  # also compare a build stamp
#
# Verified at vllm-ascend v0.21.0rc1 (80610e4): the live map and the assertion text
# are utils.py:794-809. Line numbers move; the two-sided message does not.
set -u

MAP_NOTE="map (v0.21.0rc1 utils.py:794-804): 220-225=A2  250-255=A3  200-205=_310P  260=A5  else RuntimeError"

classify() { # $1 = numeric soc_version
  local n="$1"
  case "$n" in
    ''|*[!0-9]*) echo "unknown: '$n' is not a numeric soc_version"; return 1 ;;
  esac
  if [ "$n" -ge 220 ] && [ "$n" -le 225 ]; then echo A2
  elif [ "$n" -ge 250 ] && [ "$n" -le 255 ]; then echo A3
  elif [ "$n" -ge 200 ] && [ "$n" -le 205 ]; then echo _310P
  elif [ "$n" -eq 260 ]; then echo A5
  else echo "UNMAPPED (RuntimeError: Can not support soc_version: $n.)"; fi
}

probe_hint() {
  cat <<'HINT'
next two commands on the reporter's host (they decide it, no rebuild):
  python -c "import torch_npu; print(torch_npu.npu.get_soc_version())"
  python -c "import vllm_ascend._build_info as b; print(b.__device_type__)"
HINT
}

soc=""; stamp=""; src=""
while [ $# -gt 0 ]; do
  case "$1" in
    --soc-version) soc="${2:-}"; shift 2 ;;
    --stamp)       stamp="${2:-}"; shift 2 ;;
    -h|--help)     sed -n '2,9p' "$0"; exit 0 ;;
    *)             src="$1"; shift ;;
  esac
done

if [ -n "$soc" ]; then
  live="$(classify "$soc")"
  echo "soc_version $soc => live device type: $live"
  echo "$MAP_NOTE"
  if [ -n "$stamp" ]; then
    echo "build stamp (_build_info.__device_type__) = $stamp"
    if [ "$live" = "$stamp" ]; then
      echo "verdict: MATCH - a device-type failure is not explainable by this pair; look elsewhere"
    else
      echo "verdict: MISMATCH - the installed package is a $stamp build running on $live silicon;"
      echo "         rebuild for $live (source build on the node, or SOC_VERSION starting ascend950 for A5)"
    fi
  fi
  probe_hint
  exit 0
fi

if [ -z "$src" ]; then echo "usage: $0 <paste|serve-log|-> | --soc-version N [--stamp X]" >&2; exit 2; fi
if [ "$src" = "-" ]; then txt="$(cat)"; else txt="$(cat "$src" 2>/dev/null)"; fi
if [ -z "$txt" ]; then echo "no input text" >&2; exit 2; fi

found=0
# 1) the two-sided assertion: first value = live, second = installed build
while IFS= read -r line; do
  [ -n "$line" ] || continue
  found=1
  live="$(printf '%s' "$line" | sed -n 's/.*Current device type: \([A-Za-z0-9._]*\).*/\1/p')"
  inst="$(printf '%s' "$line" | sed -n "s/.*installed version's device type: \([A-Za-z0-9._]*\).*/\1/p")"
  echo "ASSERTION: live=$live  installed-build=${inst:-<unparsed>}"
  echo "  first value is the CARD (torch_npu.npu.get_soc_version classified); the second is what the WHEEL was built as."
  if [ -n "$live" ] && [ -n "$inst" ] && [ "$live" != "$inst" ]; then
    echo "  verdict: SoC/package mismatch - an $inst build on $live silicon; get an $live build, do not patch the guard"
    case "$inst" in *A2|*A3|*_310P) echo "  the release pipeline ships a2/a3/310p wheels only; a 950/A5 host must build from source on the node (SOC_VERSION starts ascend950) and re-check _build_info.__device_type__ == 'A5'";; esac
  fi
done < <(printf '%s\n' "$txt" | grep -F "does not match the installed version's device type")

# 2) the unmapped-code RuntimeError
while IFS= read -r n; do
  [ -n "$n" ] || continue
  found=1
  echo "UNMAPPED soc_version in log: $n  => $(classify "$n")"
  echo "  this is the driver/firmware/CANN pairing for that SoC reporting, not the wheel; a bare"
  echo "  torch.nn.functional.linear repro on the host settles whether it is below vllm-ascend (#10206, #6796)."
done < <(printf '%s\n' "$txt" | sed -n 's/.*Can not support soc_version: \([0-9][0-9]*\).*/\1/p' | sort -u)

# 3) neighbouring evidence that is NOT this fault
if printf '%s\n' "$txt" | grep -q "liboptiling.so"; then
  echo "note: liboptiling.so dlopen failure in the paste is expected and non-fatal (auto_exclude.py excludes it); it never causes EngineCore failed to start"
fi
if [ "$found" -eq 0 ]; then echo "no device-type assertion or unmapped soc_version found in the paste"; fi
echo "$MAP_NOTE"
probe_hint
