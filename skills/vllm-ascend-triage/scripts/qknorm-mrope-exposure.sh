#!/usr/bin/env bash
# Is a fused linearnorm kernel (qkv split + RMSNorm + RoPE) exposed to the
# triton-ascend silent miscompute, and does the tree's own test cover the shape?
#
#   bash <skill_root>/scripts/qknorm-mrope-exposure.sh <vllm-ascend-root> [--kv-heads N]
#
# One call prints, from one checkout: the commit and the triton-ascend pin; every
# kernel file that carries a per-head reduction (`tl.sum(..., axis=1)`) feeding a
# `reshape(<heads>, 1)` broadcast; which of them hand a slice of the *normalized*
# tile straight to a cos/sin multiply (the consumer shape the miscompile needs);
# every caller of the op and the one gate the FX pass applies but a direct
# `torch.ops` call skips; whether any env switch could turn the fused op off; and
# the parameter constants of the nightly kernel tests, i.e. the head counts CI runs
# (`--kv-heads 4` says whether the reported model's count is one of them).
#
# A printed line is a site to quote, not a cause (SKILL.md rule 6). Re-read on another
# release: vllm-ascend v0.23.0rc1 f4a08bd / v0.23.0 5cb98caa, triton-ascend 3.2.1
# (requirements.txt:20, pyproject.toml:32).
set -u

ROOT=""; KV=""
while [ $# -gt 0 ]; do
  case "$1" in
    --kv-heads) KV="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) ROOT="$1"; shift ;;
  esac
done
if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
  echo "usage: bash <skill_root>/scripts/qknorm-mrope-exposure.sh <vllm-ascend-root> [--kv-heads N]" >&2
  exit 2
fi

K=""
for d in "$ROOT/vllm_ascend/ops/triton/linearnorm" "$ROOT/vllm_ascend/ops/linear_norm"; do
  [ -d "$d" ] && K="$d" && break
done

echo "== checkout"
if [ -d "$ROOT/.git" ]; then
  ( cd "$ROOT" && git log -1 --format='  commit %H  %ci  %s' ) 2>/dev/null
else
  echo "  (not a git checkout; commit unknown)"
fi
echo "  path: $ROOT"
echo

echo "== triton-ascend pin (the wheel that owns a silent miscompute)"
found=0
for f in requirements.txt pyproject.toml; do
  if [ -f "$ROOT/$f" ] && rg -q 'triton-ascend' "$ROOT/$f"; then
    rg -n 'triton-ascend' "$ROOT/$f" | sed "s|^|  $f:|"; found=1
  fi
done
[ "$found" = 0 ] && echo "  no pin in requirements.txt/pyproject.toml -> <= v0.13.0rc1 era: read the tutorial's install block"
echo "  the installed build is not in collect_env: ask for \`pip show triton-ascend\`"
echo

echo "== per-head reductions (squares/sum -> reshape(<heads>, 1) broadcast)"
if [ -n "$K" ]; then
  rg -n 'tl\.sum\(.*axis=1\) */|1 / tl\.sqrt\(.*\)\.reshape' "$K" | sed "s|$ROOT/||" | sed 's|^|  |'
else
  echo "  no linearnorm kernel directory at this ref (pre-0.9 trees have none)"
fi
echo

echo "== consumer shape: what the slice is taken from, and the cos/sin multiply"
if [ -n "$K" ]; then
  for f in "$K"/*.py; do
    [ -f "$f" ] || continue
    sl=$(rg -n '(orig_qk|normalized_values_tmp1?|q1|k1) = extract_slice\(' "$f" | tr '\n' ' ')
    mul=$(rg -n '(\* *cos|\* *sin)' "$f" | head -4 | tr '\n' ' ')
    [ -z "$sl" ] && continue
    echo "  $(basename "$f")"
    echo "    slice origin: $sl"
    echo "    cos/sin multiply: ${mul:-none}"
  done
else
  echo "  (no kernel dir)"
fi
echo "  reading: a weight multiply or a dtype cast between the reduction and the slice,"
echo "  or a fresh tl.zeros + insert_slice assembly (the reporter's cat_y), is the safe"
echo "  side; a direct extract_slice(<float32 normalized tile>) * cos is the exposed one."
echo

echo "== callers of the op, and the gate each route bypasses"
if rg -q 'triton_split_qkv_rmsnorm_mrope' "$ROOT/vllm_ascend" 2>/dev/null; then
  rg -n 'torch\.ops\.vllm\.triton_split_qkv_rmsnorm_mrope|op_name="triton_split_qkv_rmsnorm_mrope"' \
    "$ROOT/vllm_ascend" | sed "s|$ROOT/||" | sed 's|^|  |'
else
  echo "  op name not found under vllm_ascend/ (older ref: kernels imported directly)"
fi
P="$ROOT/vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py"
if [ -f "$P" ]; then
  echo "  -- FX fusion pass gates (a direct torch.ops call from patch/worker/ skips both):"
  rg -n 'head_size != 128|dtype not in \(' "$P" | sed 's|^|     |'
else
  echo "  -- no qknorm_rope_fusion_pass.py at this ref (eager kernels only)"
fi
echo "  -- env switch that turns the fused op off:"
if [ -f "$ROOT/vllm_ascend/envs.py" ] && rg -qi 'qknorm|qk_norm|rope_fusion' "$ROOT/vllm_ascend/envs.py"; then
  rg -ni 'qknorm|qk_norm|rope_fusion' "$ROOT/vllm_ascend/envs.py" | sed 's|^|     |'
else
  echo "     none in vllm_ascend/envs.py -> the mitigation is a source patch or the release, not a flag"
fi
echo

echo "== nightly kernel tests: which head counts CI actually runs"
T=$(rg --files "$ROOT/tests" 2>/dev/null | rg 'test_split_qkv_(rmsnorm|tp_rmsnorm)_(mrope|rope)\.py$' | sort)
if [ -z "$T" ]; then
  echo "  no nightly kernel test found -> say \"untested shape\", not \"no test\""
else
  for f in $T; do
    echo "  ${f#$ROOT/}"
    rg -n '^(NUM_QKV_HEADS|HEAD_SIZES|ROPE_DIMS|MROPE_SECTION|DTYPES)' "$f" | sed 's/\r$//' | sed 's|^|    |'
    kv=$(rg -o '^NUM_QKV_HEADS = .*' "$f" | rg -o '[0-9]+ *, *[0-9]+' \
         | awk -F, '{gsub(/ /,"",$2); print $2}' | sort -un | tr '\n' ' ')
    echo "    kv-head counts covered: ${kv:-unknown}"
    if [ -n "$KV" ]; then
      case " $kv " in
        *" $KV "*) echo "    -> kv_heads=$KV IS covered by this file (check the rest of the tuple: head size, rope_dim, interleaved)" ;;
        *)         echo "    -> kv_heads=$KV is NOT covered by this file" ;;
      esac
    fi
  done
  echo "  reading: a green nightly says only that the tuples above pass. At v0.23.0rc1 the"
  echo "  mrope test runs (kv=1, hs=128/256, rope_dim 64/128) and the non-mrope test runs"
  echo "  (kv=1,4 at hs=128) — so kv=4 is exercised only on the kernel whose chain casts"
  echo "  and weights before slicing, not on the direct float32 slice x cos chain."
fi
echo

echo "== the K variance chain to quote (line numbers move; re-grep)"
if [ -n "$K" ] && [ -f "$K/split_qkv_rmsnorm_mrope.py" ]; then
  rg -n 'in_k_tensor = tl\.load|squares = in_k_tensor|variances = tl\.sum|reciprocal_std = \(1|k_normalized = in_k_tensor|k_normalized = k_normalized \* k_rmsnorm_weight|orig_qk = extract_slice|roped_k = cat_y|k_normalized = insert_slice|IS_PARTIAL_ROPE = rope_dim' \
    "$K/split_qkv_rmsnorm_mrope.py" | sed "s|$K/||" | sed 's|^|  |'
else
  echo "  split_qkv_rmsnorm_mrope.py absent at this ref"
fi
echo

echo "== the reporter's ladder (each row is one kernel edit, not a runtime flag)"
echo "   squares and tl.sum stored back:  correct   -> corruption is downstream of the sum"
echo "   square_sums * 1.0:               correct   -> identity scaling is folded away"
echo "   square_sums * 2.0 / 1.5 / 0.5, /2, /256: all 0 -> the scalar scale is miscompiled"
echo "   delete the cos multiply consumer: correct   -> the consumer chooses the lowering"
echo "   only num_kv_heads == 4 wrong (2 right)      -> shape-keyed: ask for the sweep, not one datapoint"
