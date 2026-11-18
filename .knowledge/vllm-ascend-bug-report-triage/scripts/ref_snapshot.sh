#!/usr/bin/env bash
# Print the facts needed to pin a ref: for vllm-ascend, the paired vLLM tag, runtime
# pins, package layout and existence of the paths that keep moving; for any other clone
# (vllm, Mooncake, transformers, …) the repo's own top-level layout instead.
#
#   bash ref_snapshot.sh [ref] [repo-path-or-name]
#
# Defaults: ref=HEAD, repo=/workspace/repos/vllm-project/vllm-ascend. The repo argument may be a
# clone name ("vllm-ascend", "vllm"); a name that matches no clone is exit 3, not "unknown ref".
# The vllm-ascend-only sections are gated on `vllm_ascend/` existing at the ref, so a snapshot of
# an upstream ref prints upstream facts instead of a column of `no` for paths that repo never had
# (a recorded run asked for vllm v0.23.0 and got eleven `no` lines plus empty sections).
# Read-only: uses git show/ls-tree/grep only, never touches the index.
set -eu

REF="${1:-HEAD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=$(bash "$HERE/repo_root.sh" "${2:-}") || exit 3
G="git --no-pager -C $REPO"
ROOT=${CLONE_ROOT:-/workspace/repos}
REPO_LABEL=${REPO#"$ROOT"/}

# Accept a reported version string ("0.18.0", "0.23.0rc1", a dev string ending in "+g<sha>") as well
# as a git ref. An unresolved ref turns every later git call into `fatal: ambiguous argument` and
# silently costs the ref facts; report it as a version-string problem with the near tags instead.
if ! $G rev-parse -q --verify "${REF}^{commit}" >/dev/null 2>&1; then
  if resolved=$(bash "$HERE/resolve_ref.sh" "$REF" "$REPO" 2>/dev/null); then
    echo "# reported version '$REF' resolved to $resolved"
    REF="$resolved"
  else
    echo "== ref =="
    echo "UNRESOLVED: '$REF' is not a tag, branch or commit in $REPO."
    echo "This is a version-string problem, not an absent feature: resolve it before quoting a line."
    bash "$HERE/resolve_ref.sh" "$REF" "$REPO" 2>&1 | sed -n '1,14p'
    exit 2
  fi
fi

# Which clone is this? `vllm_ascend/` at the ref decides the plugin sections, not the clone name.
PLUGIN=0
$G cat-file -e "$REF:vllm_ascend" 2>/dev/null && PLUGIN=1

echo "== ref =="
echo "repo: $REPO_LABEL"
$G log -1 --format='%H %ad %s' --date=short "$REF"
echo "describe (MISLEADING, do not use as the version): $($G describe --tags --always "$REF" 2>/dev/null || echo n/a)"

if [ "$PLUGIN" = 1 ]; then
  echo
  echo "== paired vLLM tag (.github/vllm-release-tag.commit) =="
  $G show "$REF:.github/vllm-release-tag.commit" 2>/dev/null || echo "(absent at this ref: use the same-version rule and the issue's collect_env)"

  echo
  echo "== runtime pins (requirements.txt) =="
  $G show "$REF:requirements.txt" 2>/dev/null | grep -iE '^(torch|torch-npu|torch_npu|triton-ascend)' || echo "(no torch lines)"

  echo
  echo "== vllm_ascend top-level packages =="
  $G ls-tree --name-only "$REF:vllm_ascend" 2>/dev/null | tr '\n' ' '; echo

  echo
  echo "== path existence (moved between releases) =="
  for p in \
    vllm_ascend/models \
    vllm_ascend/model_executor \
    vllm_ascend/attention/context_parallel \
    vllm_ascend/attention/sfa_v1.py \
    vllm_ascend/attention/dsa_v1.py \
    vllm_ascend/ops/fused_moe/prepare_finalize.py \
    vllm_ascend/worker/pcp_utils.py \
    vllm_ascend/_310p/attention/attention_mask.py \
    vllm_ascend/distributed/kv_transfer \
    vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h \
    vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_context.py
  do
    if $G cat-file -e "$REF:$p" 2>/dev/null; then echo "  yes  $p"; else echo "  no   $p"; fi
  done

  echo
  echo "== model runner and 310P layout policy =="
  if $G cat-file -e "$REF:vllm_ascend/_310p/worker/v2/model_runner.py" 2>/dev/null; then
    echo "  310P MRv2 runner (NPUModelRunner310V2): yes (_310p/worker/v2 exists)"
  else
    echo "  310P MRv2 runner (NPUModelRunner310V2): no (_310p/worker/v2 absent: VLLM_USE_V2_MODEL_RUNNER cannot select a 310P V2 runner at this ref)"
  fi
  $G show "$REF:vllm_ascend/_310p/worker_310p.py" 2>/dev/null | grep -nE 'NPUModelRunner310|Using NPUWorker310' | sed 's/^/  runner: /' || true
  $G grep -n -E 'def _should_trans_nz|is_310p\(\)' "$REF" -- vllm_ascend/utils.py 2>/dev/null | sed 's/^/  nz-policy: /' | head -6 || true

  echo
  echo "== vllm_ascend/core and vllm_ascend/patch/platform files (names and packages move) =="
  echo "  core:   $($G ls-tree --name-only "$REF:vllm_ascend/core" 2>/dev/null | tr '\n' ' ')"
  echo "  patch:  $($G ls-tree --name-only "$REF:vllm_ascend/patch/platform" 2>/dev/null | tr '\n' ' ')"
  echo "  (a downstream patch named after the feature is a first-class suspect: read its Why/How block in vllm_ascend/patch/__init__.py)"

  echo
  echo "== attention package =="
  $G ls-tree --name-only "$REF:vllm_ascend/attention" 2>/dev/null | tr '\n' ' '; echo
else
  echo
  echo "== not the plugin repo: vllm_ascend/ is absent at this ref =="
  echo "  Paths under /vllm-workspace/vllm/ in a report or traceback live here, at the tag the"
  echo "  image's Dockerfile passes as VLLM_TAG (vllm-ascend 'ARG VLLM_TAG='); read them in this clone"
  echo "  and run the plugin half with: bash $0 <vllm-ascend-ref> vllm-ascend"
  echo "  A grep for such a file under vllm-ascend is empty by construction, not a negative result."
  echo
  echo "== top-level layout at this ref =="
  $G ls-tree --name-only "$REF" 2>/dev/null | tr '\n' ' '; echo
fi

echo
echo "== commits since ref (grep-ready) =="
echo "  total: $($G rev-list --count "$REF..HEAD" 2>/dev/null || echo n/a)"
$G log --oneline -8 "$REF..HEAD" 2>/dev/null || true
