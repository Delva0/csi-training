#!/usr/bin/env bash
# Which CPU architecture and which SoC each published image/wheel tag was built
# for, read from the tree's own CI, without opening the workflow files by hand.
#
#   bash image-arch-matrix.sh <vllm-ascend-root> [--soc 310p|a2|a3|a5|none]
#
# Prints, per section: tag suffixes (which tags belong to which SoC variant),
# the `platforms:` / `arch:` matrix as written, the runner each leg uses, every
# `FROM` base image (the arm64 leg can only exist if the base does), the
# wheel-build bases (`SOC_VERSION` is baked into `vllm_ascend/_build_info.py`, so
# a wheel is one SoC), and the variant list of a newer-layout pipeline.
# Verified against vllm-ascend v0.9.2rc1 (b5b7e0ec, #2178: old
# `.github/workflows/image_*.yml` layout) and v0.23.0rc1 (f4a08bdd, new
# `_schedule_image_build.yaml` + `configs/` layout). It decides nothing about a
# registry: the manifest list is only readable with `docker buildx imagetools`.
set -u

soc=""
root=""
while [ $# -gt 0 ]; do
  case "$1" in
    --soc) soc="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) root="$1"; shift ;;
  esac
done
[ -n "$root" ] || { echo "usage: bash image-arch-matrix.sh <vllm-ascend-root> [--soc 310p]"; exit 2; }
[ -d "$root/.github" ] || { echo "no $root/.github - is this a vllm-ascend checkout?"; exit 2; }

hits() { # $1 = regex, $2.. = paths (missing paths are skipped)
  local pat="$1"; shift
  grep -nH -E "$pat" "$@" 2>/dev/null
}

echo "=== tree: $root${soc:+   soc filter: $soc}"

echo
echo "--- tag suffixes (tag -> SoC variant; a bare tag with no suffix is the generic image)"
wf=$(ls "$root"/.github/workflows/*.yml "$root"/.github/workflows/*.yaml 2>/dev/null)
[ -n "$wf" ] && hits 'suffix=|suffix:|type=pep440|type=ref' $wf | sed "s|$root/||"

echo
echo "--- platforms matrix (the push branch is the only one that publishes a multi-arch list)"
[ -n "$wf" ] && grep -nH -A7 -E '^\s*platforms:|^\s*arch:\s*linux/' $wf 2>/dev/null | sed "s|$root/||"

echo
echo "--- runner legs of the image pipelines (an arm runner is how an arm64 leg gets built)"
imgwf=$(ls "$root"/.github/workflows/*image*.yml "$root"/.github/workflows/*image*.yaml \
          "$root"/.github/workflows/*image*.y* 2>/dev/null | sort -u)
[ -n "$imgwf" ] && grep -nH -A4 -E '^\s*runs-on:' $imgwf 2>/dev/null | sed "s|$root/||" | head -40

echo
echo "--- wheel release matrix (arm64 wheels exist or not, independently of the images)"
whlwf=$(ls "$root"/.github/workflows/*whl*.y*ml "$root"/.github/workflows/*wheel*.y*ml \
          "$root"/.github/workflows/*release*.y*ml 2>/dev/null | sort -u)
[ -n "$whlwf" ] && grep -nH -E '^\s*(os|runs-on):\s*\[?[-\$\. ]*(ubuntu|matrix|\{)|uname' $whlwf 2>/dev/null | sed "s|$root/||" | head -20

echo
echo "--- base images: every Dockerfile's FROM (the arm64 leg dies here if the base is amd64-only)"
dfs=$(find "$root" -maxdepth 3 \( -name 'Dockerfile*' \) -not -path '*/docs/*' 2>/dev/null | sort)
[ -n "$dfs" ] && hits '^FROM |SOC_VERSION=|ascend-toolkit/latest' $dfs | sed "s|$root/||"

echo
echo "--- build-wheel bases (SOC_VERSION baked at build time => one wheel per SoC)"
bw=$(find "$root" -maxdepth 4 -name '*buildwheel*' 2>/dev/null | sort)
[ -n "$bw" ] && hits '^FROM |SOC_VERSION|soc_version|chip_type' $bw | sed "s|$root/||"

if [ -n "$soc" ] && [ "$soc" != none ]; then
  echo
  echo "--- '$soc' variant wiring (newer layout: one file per SoC)"
  hits "$soc" $wf $bw 2>/dev/null | sed "s|$root/||" | head -30
fi

cat <<'HINT'

=== what this does not answer
The registry is not in the snapshot. Two commands on any host with the network
decide the report:
  docker buildx imagetools inspect quay.io/ascend/vllm-ascend:<ver>-<soc>     # Platform: list
  docker buildx imagetools inspect quay.io/ascend/cann:<tag from the FROM>    # the base's arch
At the old layout `platforms:` and `push:` carry the *same* condition
(event_name == 'push' && repository_owner == 'vllm-project'), so every build that
may publish is requested multi-arch: a published tag that still resolves to one
arch points at the failed arm64 leg (check the `FROM` base) or at the client /
mirror, not at the fallback.
HINT
