#!/bin/bash
# Grade a "国内用户下载模型失败 / 请在 Quick Start 强调 ModelScope" report in one call.
#
#   bash scripts/modelscope-mirror-audit.sh <vllm-ascend-root> [<vllm-root>]
#
# Prints, for the pinned vllm-ascend checkout (and the pinned vllm checkout when
# given): where VLLM_USE_MODELSCOPE is documented and in what order, whether the
# two entry pages reach each other, how many tutorial pages already carry the
# flag, whether the image installs the package, and how upstream reads the flag
# (the bare import that decides cosmetic vs breaks-the-command).
#
# Verified against vllm-ascend v0.18.0 e18643f8a4d5 + vllm v0.18.0 bcf2be961200
# (recorded #14502). Re-read the printed lines at another pin before quoting.
# Exit 0 always, 2 on a missing checkout.
set -u

root="${1:-}"
vllm_root="${2:-}"
if [ -z "$root" ] || [ ! -d "$root/docs/source" ]; then
  echo "usage: bash modelscope-mirror-audit.sh <vllm-ascend-root> [<vllm-root>]" >&2
  exit 2
fi
cd "$root" || exit 2

echo "== 1. where the flag is documented (locale excluded) =="
rg -n --glob '!**/locale/**' -e 'VLLM_USE_MODELSCOPE' -e 'ModelScope mirror' docs/source \
  | head -20

echo
echo "== 2. China framing on the image mirror vs the weights mirror =="
rg -n -e 'in China' -e 'China region' -e '国内' --glob '!**/locale/**' docs/source/faqs.md \
  docs/source/quick_start.md 2>/dev/null | head -10

echo
echo "== 3. reachability: do the entry pages link each other? =="
echo "installation.md -> quick_start: $(rg -c -e 'quick_start' -e 'Quickstart' docs/source/installation.md 2>/dev/null || echo 0) line(s)"
echo "quick_start.md -> installation: $(rg -c 'installation\.md' docs/source/quick_start.md 2>/dev/null || echo 0) line(s)"

echo
echo "== 4. parity: tutorial pages already exporting the flag =="
total=$(rg --files docs/source/tutorials 2>/dev/null | wc -l)
carry=$(rg -l 'VLLM_USE_MODELSCOPE' docs/source/tutorials 2>/dev/null | wc -l)
echo "tutorials carrying VLLM_USE_MODELSCOPE: ${carry} of ${total}"
echo "with the '# Load model from ModelScope to speed up download' comment: $(rg -l 'Load model from ModelScope' docs/source/tutorials 2>/dev/null | wc -l)"
rg -n -B1 'VLLM_USE_MODELSCOPE' docs/source/tutorials/models 2>/dev/null | head -6

echo
echo "== 5. does the image install the package? (tiers the snippet) =="
rg -n 'pip install modelscope' Dockerfile* 2>/dev/null | head -8

echo
echo "== 6. upstream: default value and the bare import under the flag =="
if [ -n "$vllm_root" ] && [ -d "$vllm_root/vllm" ]; then
  rg -n 'VLLM_USE_MODELSCOPE' "$vllm_root/vllm/envs.py" | head -4
  rg -n -e 'def maybe_download_from_modelscope' -e 'if envs.VLLM_USE_MODELSCOPE' \
     -e 'from modelscope.hub.snapshot_download import snapshot_download' \
     "$vllm_root/vllm/model_executor/model_loader/weight_utils.py" | head -6
  echo "(bare import, no try/except: an export without the package raises ImportError at load)"
else
  echo "SKIPPED: no vllm checkout given; the tier decider is unread."
fi

echo
echo "== 7. example model ids on the entry page =="
rg -n -e 'LLM\(model=' -e 'vllm serve ' -e '"model":' docs/source/quick_start.md 2>/dev/null | head -6

echo
echo "== 8. the page's own maintenance comments (do the named paths exist?) =="
rg -n -o 'tests/e2e/doctest[s]*/[A-Za-z0-9._-]+' docs/source/quick_start.md 2>/dev/null | sort -u \
  | while IFS=: read -r line path; do
      if [ -e "$path" ]; then echo "line ${line}: ${path} EXISTS"; else echo "line ${line}: ${path} MISSING"; fi
    done

echo
echo "Verdict template: documented but demoted (usability) + missing 'pip install modelscope'"
echo "(correctness on the pip path only) + examples stay on HF ids; grade at the pin, never a run."
