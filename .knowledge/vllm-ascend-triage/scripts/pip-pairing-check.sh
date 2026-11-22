#!/usr/bin/env bash
# pip-pairing-check.sh -- the torch / torch-npu / torchvision pin arithmetic of a
# vllm + vllm-ascend pair, read straight from the two trees, so an install-time
# "could not find a version that satisfies the requirement <pkg>==<ver>" report
# (or a version-mismatch complaint) is decided by the files instead of by memory.
#
# Usage:
#   bash pip-pairing-check.sh <vllm-ascend-root> [vllm-root]
#   bash pip-pairing-check.sh --ascend <root> --vllm <root>
#
# Reads, when present:
#   <ascend>/requirements.txt            (torch / torchvision / torch-npu pins and the
#                                         --pre / --extra-index-url / -f hint lines)
#   <ascend>/pyproject.toml              ([build-system] requires, the same set)
#   <ascend>/setup.py                    whether the '--' hint lines survive into
#                                         install_requires (they do not: grep the
#                                         startswith("--") branch)
#   <vllm>/requirements/cuda.txt|common.txt|build.txt   torch / torchaudio / torchvision
#   <vllm>/pyproject.toml                "torch == X" under [build-system]
#   <ascend>/docs/source/community/versioning_policy.md   the release compatibility
#                                         matrix row (vLLM Ascend | vLLM | python | CANN |
#                                         torch/torch-npu) -- the pairing to QUOTE for a
#                                         "can you bump torch-npu to X" report
#   <ascend>/docs/source/conf.py          the pip pair substitutions; note that
#                                         installation.md prints those as |tokens|, so a
#                                         grep there for a version number finds nothing
#
# Exit: 0 always (a report tool). Every line is either a pin or a labelled finding.
set -u

asc=""; vl=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ascend) asc="${2:-}"; shift 2 ;;
    --vllm)   vl="${2:-}";   shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) if [ -z "$asc" ]; then asc="$1"; else vl="$1"; fi; shift ;;
  esac
done
[ -n "$asc" ] || { echo "usage: bash $0 <vllm-ascend-root> [vllm-root]" >&2; exit 2; }

req="$asc/requirements.txt"
py="$asc/pyproject.toml"

# first "<name> <spec>" pair on a line, ignoring comments
pin_of() { # $1=file  $2=package regex (extended)
  [ -f "$1" ] || return 0
  grep -nE "^[[:space:]]*$2" "$1" 2>/dev/null | head -1 | sed -E 's/[[:space:]]*#.*$//'
}

echo "== vllm-ascend: $asc"
for p in 'torch-npu' 'torch([=<>![:space:]]|$)' 'torchvision'; do
  disp="$p"; case "$p" in 'torch([=<>![:space:]]|$)') disp='torch' ;; esac
  line="$(pin_of "$req" "$p")"
  if [ -n "$line" ]; then printf '  requirements.txt:%-4s %s\n' "${line%%:*}" "${line#*:}"
  else printf '  requirements.txt     no %s pin\n' "$disp"; fi
done

hints="$(grep -nE '^[[:space:]]*--(pre|extra-index-url|index-url|find-links)' "$req" 2>/dev/null | tr '\n' ' | ')"
if [ -n "$hints" ]; then
  echo "  hint lines in requirements.txt: $hints"
else
  echo "  hint lines in requirements.txt: none"
fi

if [ -n "$hints" ] && [ -f "$asc/setup.py" ] && grep -q 'startswith("--")' "$asc/setup.py"; then
  echo "  FINDING: setup.py skips every line starting with '--' when it builds"
  echo "           install_requires, so --pre / --extra-index-url / -f are NOT in the"
  echo "           published wheel's metadata. pip install <name>==<ver> sees only the"
  echo "           index it is given on the command line."
fi

if [ -f "$py" ]; then
  build_torch="$(sed -n '/\[build-system\]/,/^\[/p' "$py" | grep -nE '"(torch|torch-npu|torchvision)' | head -4)"
  [ -n "$build_torch" ] && { echo "  pyproject [build-system] pins:"; echo "$build_torch" | sed 's/^/    /'; }
fi

# ---- the release's own contract ------------------------------------------
vc="$asc/docs/source/community/versioning_policy.md"
cf="$asc/docs/source/conf.py"
inst="$asc/docs/source/installation.md"
if [ -f "$vc" ]; then
  echo "== release compatibility matrix: $vc"
  echo "   (quote this row for a pairing/pin-bump question; installation.md cannot answer it)"
  grep -nE '^\|[[:space:]]*vLLM Ascend' "$vc" 2>/dev/null | head -1 | sed 's/^/  /'
  grep -nE '^\|[[:space:]]*v?[0-9]+\.[0-9]+\.[0-9]+' "$vc" 2>/dev/null | head -6 | sed 's/^/  /'
else
  echo "== release compatibility matrix: absent at this ref"
  echo "   (v0.7.x-era trees have no docs/source/community/, which is itself an answer)"
fi
if [ -f "$cf" ]; then
  echo "  conf.py substitutions (pip pair + what CI builds):"
  grep -nE "'(vllm_version|vllm_ascend_version|pip_vllm_version|pip_vllm_ascend_version|ci_vllm_version)'" "$cf" 2>/dev/null | sed 's/^/    /'
fi
if [ -f "$inst" ] && grep -q '|pip_vllm_version|' "$inst" 2>/dev/null; then
  echo "  installation.md prints |pip_vllm_version| substitutions: a grep there for a"
  echo "  version number returns nothing by design -- do not read that as 'no contract'."
fi

if [ -n "$vl" ]; then
  echo "== vllm: $vl  (this is what the vllm wheel requires)"
  found=0
  for f in requirements/cuda.txt requirements/common.txt requirements/build.txt requirements/cpu.txt; do
    [ -f "$vl/$f" ] || continue
    hit="$(grep -nE '^[[:space:]]*(torch|torchaudio|torchvision)[=<>! ]' "$vl/$f" 2>/dev/null)"
    [ -n "$hit" ] && { echo "  $f:"; echo "$hit" | sed 's/^/    /'; found=1; }
  done
  [ -f "$vl/pyproject.toml" ] && { echo "  pyproject.toml:"; grep -nE '"(torch|torchaudio|torchvision) *[=<>]' "$vl/pyproject.toml" | sed 's/^/    /'; }
  [ "$found" = 0 ] && echo "  (no torch pin found under requirements/ -- check the tree layout)"
fi

# ---- the arithmetic -------------------------------------------------------
get_ver() { # $1=file  $2=regex -> the first X.Y[.Z] after the package name
  [ -f "$1" ] || return 0
  grep -nE "$2" "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1
}
a_torch="$(get_ver "$req" '^[[:space:]]*torch[=<>! ]')"
a_tnp="$(get_ver "$req" '^[[:space:]]*torch-npu[=<>! ]')"
a_tv="$(get_ver "$req" '^[[:space:]]*torchvision')"
a_tv_op="$(grep -nE '^[[:space:]]*torchvision' "$req" 2>/dev/null | head -1 | sed -E 's/^[0-9]+://; s/[[:space:]]*#.*$//')"
a_tnp_raw="$(grep -nE '^[[:space:]]*torch-npu' "$req" 2>/dev/null | head -1 | sed -E 's/^[0-9]+://; s/[[:space:]]*#.*$//')"
v_torch=""; v_tv=""
if [ -n "$vl" ]; then
  for f in requirements/cuda.txt requirements/common.txt requirements/build.txt; do
    [ -n "$v_torch" ] || v_torch="$(get_ver "$vl/$f" '^[[:space:]]*torch[=<>! ]')"
    [ -n "$v_tv" ]    || v_tv="$(get_ver "$vl/$f" '^[[:space:]]*torchvision')"
  done
  [ -n "$v_torch" ] || v_torch="$(get_ver "$vl/pyproject.toml" 'torch *==')"
fi

mm() { echo "${1:-}" | cut -d. -f1-2; }

# the exact-pin-beside-floor shape of the build requirements: this pair is the
# predictor of the build-deps "conflicting dependencies" error (§D), independent of
# which tree the reporter has installed
bs_tnp="$(sed -n '/\[build-system\]/,/^\[/p' "$py" 2>/dev/null \
          | grep -oE 'torch-npu[=<>! ]+[Vv0-9][^",[:space:]]*' | head -1)"
bs_torch="$(sed -n '/\[build-system\]/,/^\[/p' "$py" 2>/dev/null \
          | grep -oE '"torch[=<>! ]+[Vv0-9][^",[:space:]]*' | sed 's/^"//' | head -1)"
tnp_exact=""; case "$bs_tnp" in *"=="*) tnp_exact="${bs_tnp#*==}" ;; esac
torch_floor=""; case "$bs_torch" in *">="*) torch_floor="${bs_torch#*>=}" ;; esac
cpu_idx_line=""
[ -f "$inst" ] && cpu_idx_line="$(grep -nE 'download\.pytorch\.org/whl/cpu' "$inst" 2>/dev/null | head -1 | cut -d: -f1)"
req_has_tnp="$(grep -cE '^[[:space:]]*torch-npu[=<>! ]' "$req" 2>/dev/null || true)"
setup_feeds=""
[ -f "$asc/setup.py" ] && grep -q 'get_requirements()' "$asc/setup.py" && setup_feeds=1

echo "== findings"
n=0
if [ -n "$tnp_exact" ] && [ -n "$torch_floor" ] && [ "$tnp_exact" = "$torch_floor" ]; then
  n=$((n+1)); echo "$n. [build-system] pins \"$bs_tnp\" beside \"$bs_torch\": the two specs are only"
  echo "   jointly satisfiable through a torch carrying the LOCAL label '+cpu' (torch_npu wheels"
  echo "   pin 'torch==$tnp_exact+cpu' on x86_64; '>= $torch_floor' does accept '2.5.1+cpu' per PEP 440),"
  echo "   and only download.pytorch.org/whl/cpu serves that label."
  if [ -n "$cpu_idx_line" ]; then
    echo "   The tree's own pip tab carries the x86 extra-index (installation.md:$cpu_idx_line), i.e. an"
    echo "   x86 pip install is a documented path -- so do NOT read this report as 'no wheel for my arch'."
  fi
  echo "   Expected pip text when that index is not named: 'ERROR: Cannot install $bs_tnp and $bs_torch"
  echo "   because these package versions have conflicting dependencies' + 'torch-npu $tnp_exact depends"
  echo "   on torch==$tnp_exact+cpu' -- under 'x pip subprocess to install build dependencies did not run"
  echo "   successfully' (§D) or from the main install (§C). Rung 1: --no-build-isolation (the host"
  echo "   already carries the pair) or --extra-index-url https://download.pytorch.org/whl/cpu/."
  if [ "${req_has_tnp:-0}" -gt 0 ] && [ -n "$setup_feeds" ]; then
    echo "   requirements.txt carries the torch-npu pin and setup.py reads it into install_requires, so the"
    echo "   WHEEL's metadata reaches torch through torch-npu's own ==torch pin (both legs) -- unlike the"
    echo "   sdist-only v0.7.3rc2 shape, where nothing in the wheel's metadata names torch-npu."
  fi
fi
# the torch line vllm-ascend actually lands on: its own torch pin, else what the
# torch-npu pin implies (a torch_npu wheel hard-pins the matching torch)
a_line="$a_torch"
[ -n "$a_tnp" ] && a_line="$a_tnp"
if [ -n "$a_tnp" ] && [ -n "$a_torch" ] && [ "$(mm "$a_tnp")" != "$(mm "$a_torch")" ]; then
  n=$((n+1)); echo "$n. torch-npu $a_tnp is not on the torch $(mm "$a_torch").x line this tree also pins."
fi
tnp_spec="${a_tnp_raw#*torch-npu}"   # strip the package name: 'torch' itself contains 'rc'
if [ -n "$tnp_spec" ] && echo "$tnp_spec" | grep -qE '(dev|post|rc)[0-9.]'; then
  n=$((n+1)); echo "$n. the torch-npu pin $a_tnp_raw is a pre/dev build: it is served only by the Ascend"
  echo "   mirror named in the hint lines above, and the wheel metadata cannot carry those hints."
  echo "   An exact '==' pin already allows the prerelease (pip: Specifier.prereleases is True),"
  echo "   so 'could not find a version that satisfies' means a missing INDEX, not a missing --pre."
fi
if [ -n "$v_torch" ] && [ -n "$a_line" ] && [ "$(mm "$v_torch")" != "$(mm "$a_line")" ]; then
  n=$((n+1)); echo "$n. cross-repo: vllm requires torch $v_torch while this vllm-ascend tree lands on"
  echo "   torch $(mm "$a_line").x (torch-npu $a_tnp). torch_npu wheels hard-pin the exactly"
  echo "   matching torch (Ascend/pytorch setup.py: requirements = ['torch==X+cpu' if x86_64"
  echo "   else 'torch==X'], install_requires=requirements), so installing vllm first and then"
  echo "   vllm-ascend downgrades torch and only warns: 'vllm <v> requires torch==..., but you"
  echo "   have ... which is incompatible'. The pin set, not the index, is then the blocker."
fi
if [ -n "$a_tv" ] && [ -n "$v_tv" ] && [ "$(mm "$a_tv")" != "$(mm "$v_tv")" ]; then
  n=$((n+1)); echo "$n. torchvision: vllm wheels require torchvision $v_tv, vllm-ascend carries"
  echo "   '$a_tv_op'. Each torchvision 0.2x belongs to one torch line, so this is the same split."
fi
[ "$n" = 0 ] && echo "no contradiction found among the pins that are present (check the loose '>=' pins by hand)."
echo
echo "state the verdict with these labels: source-verified at <ref>, or hypothesis."
exit 0
