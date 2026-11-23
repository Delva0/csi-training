#!/usr/bin/env bash
# What the official image at a given vllm-ascend checkout actually installs, in
# one call -- the answer to "你们的镜像里装了什么依赖 / where can I see the image's
# dependency list, so I can compare with my own env".
#
#   bash image-recipe-inventory.sh <vllm-ascend-root>
#        [--soc 910b|a3|310p|openeuler|a3-openeuler|310p-openeuler]
#                                   [--image-tag vX.Y.ZrcN]
#
# Prints, per section:
#   1. the Dockerfile for that SoC/OS and the lines that decide the dependency set:
#      the `FROM` CANN base image (where CANN, NNAL/ATB and the kernel packages come
#      from), `ARG VLLM_TAG`, the two `--extra-index` URLs (torch cpu + the
#      huaweicloud ascend index that serves `torch-npu`), the `pip uninstall -y
#      triton` line, both `set_env.sh` sources, and the `devlib` LD_LIBRARY_PATH
#      append that no doc reproduces;
#   1b. whether the recipe's `torch-npu` install line agrees with `requirements.txt`'s
#      pin or overrides it (at v0.13.0rc2 every recipe installs 2.8.0.post1 over the
#      2.8.0 pin) -- the check that separates "the image ships this pairing" from "the
#      reporter modified their env", before any version-mismatch advice;
#   2. the `requirements.txt` pins (torch / torch-npu / the `--pre` +
#      `--extra-index-url` hint lines) as the same list in file form;
#   3. the docs' own component table and manual CANN/NNAL install block, i.e. the
#      hand-built counterpart of section 1, plus the host mounts the container
#      needs (the driver is host-side: `-v /usr/local/Ascend/driver/lib64/ …`);
#   4. a ready-to-paste block the reporter runs *inside their own container* to
#      enumerate the same set and diff it against the image's.
# Exit 0 always (a report tool); 2 on a missing checkout.
set -u

root=""; soc="910b"; tag=""
while [ $# -gt 0 ]; do
  case "$1" in
    --soc)       soc="${2:-910b}"; shift 2 ;;
    --image-tag) tag="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) root="$1"; shift ;;
  esac
done
if [ -z "$root" ] || [ ! -d "$root" ]; then
  echo "usage: bash image-recipe-inventory.sh <vllm-ascend-root> [--soc 910b|a3|310p|openeuler|a3-openeuler]" >&2
  exit 2
fi

case "$soc" in
  910b|a2|"")      df="Dockerfile" ;;
  a3)              df="Dockerfile.a3" ;;
  310p)            df="Dockerfile.310p" ;;
  openeuler)       df="Dockerfile.openEuler" ;;
  a3-openeuler)    df="Dockerfile.a3.openEuler" ;;
  310p-openeuler)  df="Dockerfile.310p.openEuler" ;;
  *) echo "unknown --soc $soc (910b|a3|310p|openeuler|a3-openeuler|310p-openeuler)" >&2; exit 2 ;;
esac

d="$root/$df"
echo "=== checkout: $root   soc=$soc   recipe=$df"

echo
echo "--- 1. image recipe lines ($([ -f "$d" ] && echo present || echo "MISSING: $d"))"
if [ -f "$d" ]; then
  grep -nE '^FROM |VLLM_TAG|extra-index|extra_index|uninstall -y triton|set_env\.sh|devlib|pip install .*(modelscope|ray)' "$d" \
    | sed "s|^|$df:|"
  echo "   (a base tag like 'cann:<ver>-<soc>-<os>-py<ver>' is what supplies CANN + NNAL/ATB; the vllm-ascend"
  echo "    steps only add python packages, so a hand-built env that skipped NNAL is short one component.)"
fi

echo
echo "--- 1b. recipe vs requirements pin: does the image override torch-npu?"
if [ -f "$d" ] && [ -f "$root/requirements.txt" ]; then
  rec=$(grep -oE "torch[-_]npu==[^'\"[:space:]]+" "$d" | sed -E 's/.*==//' | sort -u | tr '\n' ' ')
  req=$(grep -oE '^torch-npu==[^[:space:]]+' "$root/requirements.txt" | sed -E 's/.*==//' | sort -u | tr '\n' ' ')
  rec=${rec% }; req=${req% }
  echo "   $df installs torch-npu : ${rec:-<no explicit torch-npu install line>}"
  echo "   requirements.txt pins   : ${req:-<no torch-npu pin>}"
  if [ -z "$rec" ]; then
    echo "   (the recipe adds no torch-npu of its own: a reported version that differs from the pin is not the image's.)"
  elif [ -z "$req" ]; then
    echo "   (nothing to compare against in requirements.txt; cite the recipe line.)"
  else
    case " $rec " in
      *" $req "*)
        echo "   (recipe and requirements agree; a reported version that differs from them is not the image's.)" ;;
      *)
        echo "   (the recipe installs a DIFFERENT torch-npu than requirements.txt pins: a paste from this tag showing that"
        echo "    version is the SHIPPED pairing -- do not call the env hand-modified, and 'use the release image' is not a"
        echo "    rung. Recorded #6271: the v0.13.0rc2 recipes install 2.8.0.post1 over the 2.8.0 pin.)" ;;
    esac
  fi
else
  echo "   (need both $df and requirements.txt)"
fi

echo
echo "--- 2. requirements.txt pins (the same list in file form)"
if [ -f "$root/requirements.txt" ]; then
  grep -nE '^(torch|torch-npu|torchvision|--pre|--extra-index-url|--index-url)|numpy|numba|quart|msgpack' \
    "$root/requirements.txt" | sed 's|^|requirements.txt:|'
else
  echo "   requirements.txt missing"
fi

echo
echo "--- 3. docs: component table + manual CANN/NNAL install + the container's host mounts"
if [ -f "$root/docs/source/installation.md" ]; then
  grep -nE '^\s*\|\s*(CANN|torch-npu|torch)\s*\||Ascend-cann-toolkit|Ascend-cann-kernels|Ascend-cann-nnal|set_env\.sh|No more extra step|docker run|-v /usr/local/Ascend/driver|-v /etc/ascend_install.info' \
    "$root/docs/source/installation.md" | sed 's|^|installation.md:|'
  echo "   (the driver/firmware is host-side and enters the container only through those -v mounts, so a"
  echo "    reporter with container-only access cannot reinstall it -- compare pip + CANN + NNAL, not the driver.)"
fi
if [ -f "$root/docs/source/community/versioning_policy.md" ]; then
  echo
  echo "--- 3b. release compatibility rows (the *pair*, not the single package)"
  grep -nE '^\|\s*v?0\.[0-9]+\.[0-9]+' "$root/docs/source/community/versioning_policy.md" | head -6 \
    | sed 's|^|versioning_policy.md:|'
  echo "   (swapping only vllm-ascend between two rows leaves torch/torch_npu at the old row's values:"
  echo "    that is not a control, it is an out-of-matrix pair -- say so before ranking a mechanism.)"
fi

echo
echo "--- 4. hand this to the reporter: run it INSIDE their container, no host access needed"
tw=${tag:-<the-tag-you-run>}
cat <<EOF
   # (a) what my env has
   pip list --format=freeze | sort > /tmp/mine.txt
   ls -l /usr/local/Ascend/nnal/atb/set_env.sh 2>&1; echo "ATB_HOME_PATH=\$ATB_HOME_PATH"
   env | grep -Ei 'ascend|atb|nnal' | sort
   cat /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info 2>/dev/null | head -5

   # (b) what the official image of the SAME release has (a SoC suffix may be needed on
   #     the tag: vX.Y.ZrcN-a3 / -310p -- see image-arch-and-tag-availability.md)
   docker run --rm --entrypoint bash quay.io/ascend/vllm-ascend:$tw -lc '\\
     pip list --format=freeze | sort; \\
     ls -l /usr/local/Ascend/nnal/atb/set_env.sh; env | grep -Ei "ascend|atb|nnal" | sort' \\
     > /tmp/image.txt 2>&1

   # (c) the diff that answers the question
   diff /tmp/mine.txt /tmp/image.txt | grep -E '^[<>]' | head -40
EOF
echo "   (a conda env's Ascend root is often \$CONDA_PREFIX/Ascend, not /usr/local/Ascend: pass that root"
echo "    to the ls/cat checks above instead of the doc's absolute path.)"
