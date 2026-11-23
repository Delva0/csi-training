#!/usr/bin/env bash
# abi-symbol-check.sh -- classify a pasted native "undefined symbol" against the
# two ATB/NNAL builds (cxx_abi_0 = pre-C++11 std::string, cxx_abi_1 = __cxx11).
#
# usage: bash abi-symbol-check.sh <log-file> [more files ...]
#        ... - < paste.txt
#
# The class: `python -c "import <pkg>"` (or the first real use) raises
#   ImportError: <site-packages>/<pkg>/_ops.cpython-310-aarch64-linux-gnu.so:
#   undefined symbol: _ZNK6AsdOps3Ops18GetOperationByNameERKSs
# The mangled name decides it. `Ss` is the old (pre-C++11, _GLIBCXX_USE_CXX11_ABI=0)
# substitution for std::string; the cxx11 ABI spells it
# `NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE`. Two different mangled
# names => the loader found no definition at all, so this is a build/install
# pairing fault (ATB cxx_abi_0 vs cxx_abi_1), not a wrong call signature.
#
# Reported as reporter-confirmed on vllm-project/vllm-ascend#1144: pointing
# ATB_HOME_PATH and LD_LIBRARY_PATH at atb/cxx_abi_0 made `import mindie_turbo`
# work. This script only reads a paste; the nm commands it prints are the
# reporter's own one-liners.
set -u
LC_ALL=C

if [ "$#" -eq 0 ]; then
  echo "usage: bash $0 <log-file> [more files ...]  |  -" >&2
  exit 2
fi

STDIN_TMP=""
cleanup() { [ -n "$STDIN_TMP" ] && rm -f "$STDIN_TMP"; }
trap cleanup EXIT

demangle() {
  if command -v c++filt >/dev/null 2>&1; then
    printf '%s' "$1" | c++filt 2>/dev/null || printf '%s' "$1"
  else
    printf '%s' "$1"
  fi
}

classify() {
  case "$1" in
    *NSt7__cxx11*) echo "new-ABI (std::__cxx11::basic_string, _GLIBCXX_USE_CXX11_ABI=1)" ;;
    *Ss|*RKSs|*St3__1*) echo "old-ABI (std::basic_string, _GLIBCXX_USE_CXX11_ABI=0)" ;;
    *) echo "not a std::string signature (other ABI/version mismatch)" ;;
  esac
}

for f in "$@"; do
  if [ "$f" = "-" ]; then
    STDIN_TMP=$(mktemp "${TMPDIR:-/tmp}/abi-sym.XXXXXX"); cat > "$STDIN_TMP"
    src="$STDIN_TMP"; label="<stdin>"
  else
    src="$f"; label="$f"
  fi
  if [ ! -r "$src" ]; then echo "cannot read $f" >&2; continue; fi
  echo "== $label  ($(wc -l < "$src") lines, $(wc -c < "$src") bytes)"

  syms=$(grep -oE 'undefined symbol: [_A-Za-z][_A-Za-z0-9.$]*' "$src" 2>/dev/null \
         | sed 's/^undefined symbol: //' | sort -u)
  missing=$(grep -oE "[A-Za-z0-9_.+-]+\.so(\.[0-9]+)*: cannot open shared object file" "$src" 2>/dev/null | sort -u)
  glibcxx=$(grep -oE "GLIBCXX_[0-9.]+' not found" "$src" 2>/dev/null | sort -u)
  atb0=$(grep -c 'cxx_abi_0' "$src" 2>/dev/null || true)
  atb1=$(grep -c 'cxx_abi_1' "$src" 2>/dev/null || true)
  home=$(grep -oE 'ATB_HOME_PATH=[^ "]*' "$src" 2>/dev/null | sort -u)
  whl=$(grep -oE '/[A-Za-z0-9_.+-]+/(_ops|[A-Za-z0-9_]+_C)\.[A-Za-z0-9_.+-]*\.so' "$src" 2>/dev/null | sort -u | head -3)
  npu=$(grep -oE 'torch_npu/[A-Za-z0-9_./+-]*\.so' "$src" 2>/dev/null | sort -u | head -3)

  if [ -n "$missing" ]; then
    echo "  other class present: missing library (cannot open shared object file):"
    printf '    %s\n' "$missing"
    echo "    -> that is the NNAL/ATB-unsourced or absent-package case, not this check"
    echo "       (references/image-dependency-faults/nnal-atb-missing-library.md)."
  fi
  if [ -n "$glibcxx" ]; then
    echo "  other class present: host libstdc++ older than the .so expects:"
    printf '    %s\n' "$glibcxx"
    echo "    -> versioned GLIBCXX_* symbols mean an old toolchain image, not a"
    echo "       cxx_abi_0/1 choice; fix the image's GCC/libstdc++."
  fi

  if [ -z "$syms" ]; then
    echo "  no 'undefined symbol: <mangled>' text in this paste"
    continue
  fi

  echo "  undefined symbol(s):"
  printf '%s\n' "$syms" | while read -r s; do
    [ -z "$s" ] && continue
    printf '    - %s\n' "$s"
    if [ "$(printf '%s\n' "$syms" | grep -c "^$s.")" -gt 0 ]; then
      printf '        line-wrapped fragment of a longer symbol below; ignored\n'
      continue
    fi
    printf '        demangled: %s\n' "$(demangle "$s")"
    printf '        class:     %s\n' "$(classify "$s")"
  done
  [ -n "$whl" ] && { echo "  failing extension module:"; printf '    %s\n' "$whl"; }
  [ -n "$npu" ] && { echo "  torch_npu object in the traceback:"; printf '    %s\n' "$npu"; }
  [ -n "$home" ] && echo "  environment selects: $home"
  echo "  ATB ABI markers in the paste: cxx_abi_0 x$atb0, cxx_abi_1 x$atb1"

  echo "  verdict:"
  if printf '%s' "$syms" | grep -qE 'Ss$|RKSs$|NSt7__cxx11'; then
    echo "    std::string C++ ABI (cxx_abi_0 vs cxx_abi_1) mismatch between a"
    echo "    prebuilt wheel's extension .so and the NNAL/ATB build on the loader path."
    echo "    Owner: the wheel <-> NNAL/ATB pairing, before any vllm-ascend code."
    if [ "${atb1:-0}" -gt 0 ] && [ "${atb0:-0}" -eq 0 ]; then
      echo "    The paste shows only cxx_abi_1 selected: if the failing module needs"
      echo "    the old-ABI symbol, that is the whole cause."
    fi
  else
    echo "    a symbol is missing, but the signature is not std::string: check the"
    echo "    library version/VNNI-ABI pairing before the cxx_abi_0/1 story."
  fi

  echo "  next, on the reporter's host (cheap -> decisive):"
  echo "    1) the failing .so wants it (U = undefined):"
  echo "       nm -D --undefined-only <failing.so> | grep -F '<mangled>'"
  echo "    2) which ATB root defines it (T = defined):"
  echo "       for f in <ASCEND_HOME>/nnal/atb/latest/atb/cxx_abi_{0,1}/lib/*.so*; do"
  echo "         echo \"== \$f\"; nm -D --defined-only \"\$f\" 2>/dev/null | grep GetOperationByName; done"
  echo "    3) does the torch stack need the other side?"
  echo "       nm -D --undefined-only <venv>/site-packages/torch_npu/lib/libop_plugin_atb.so | grep -c 'NSt7__cxx11'"
  echo "    Fix when step 1 and 2 agree on one root and step 3 does not contradict it:"
  echo "       export ATB_HOME_PATH=<ASCEND_HOME>/nnal/atb/latest/atb/cxx_abi_0"
  echo "       export LD_LIBRARY_PATH=<ASCEND_HOME>/nnal/atb/latest/atb/cxx_abi_0/lib:\$LD_LIBRARY_PATH"
  echo "       python -c 'import <pkg>'   # both libatb.so variants share a SONAME: one per process"
  echo "    In a launcher-managed deployment put the exports in the launch command"
  echo "    (references/image-dependency-faults/launch-env-and-library-path.md)."
done
