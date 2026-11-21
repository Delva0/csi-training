#!/usr/bin/env bash
# EZ1001 family + the NZ (FRACTAL_NZ) weight-format gate at a given ref.
#
# Usage:
#   bash nz-format-gate-check.sh <paste|-> [vllm-ascend-root]
#   echo "$LOG" | bash nz-format-gate-check.sh - /workspace/repos/<issue>/vllm-project/vllm-ascend
#
# One call answers the three questions that decide a weight-layout report:
#   1. which EZ1001 family the paste belongs to — the aclnn *registry* family
#      (build/install owner), the dtype x layout family (the Python that
#      produced the layout owns it), or the *shape* family (`mat2 n = 1 or
#      k = 1` cannot be FRACTAL_NZ at all, so any one-dimension weight that the
#      gate converts is the owner);
#   2. what the NZ gate does with a float32 weight at the checked-out ref, which
#      is the difference between the fp32 pooling head crashing and surviving;
#   3. at a three-valued ref, how `weight_nz_mode` resolves and whether the gate
#      tests the weight's shape at all (it does not at v0.20.2rc1, so a bf16
#      MoE shared-expert gate `[1, hidden]` crashes at mode 2).
#
# The root argument is optional: without it the paste is still classified, and
# the gate is described by signature only.
set -u
PASTE=${1:-}
ROOT=${2:-}
if [ -z "$PASTE" ]; then
  echo "usage: bash nz-format-gate-check.sh <paste|-> [vllm-ascend-root]" >&2
  exit 2
fi
if [ "$PASTE" = "-" ]; then
  BODY=$(cat)
else
  if [ ! -f "$PASTE" ]; then
    echo "no such file: $PASTE" >&2
    exit 2
  fi
  BODY=$(cat "$PASTE")
fi

echo "== EZ1001 / 161002 family"
REG=$(printf '%s\n' "$BODY" | grep -cE 'Get regInfo failed|binary_info_config\.json of socVersion|does not support opType')
SHAPE=$(printf '%s\n' "$BODY" | grep -cE 'n = 1 or k = 1|n==1 or k==1|format is FRACTAL_NZ, mat2:')
LAY=$(printf '%s\n' "$BODY" | grep -cE 'weight NZ is unsupported|is unsupported by the current SOC|now self is DT_|mat2 is DT_')
if [ "$REG" -gt 0 ]; then
  echo "  registry family present ($REG line(s)): aclnn has no binary for this SoC."
  echo "  owner: build/install — references/custom-ops-runtime-faults/op-not-registered-for-soc.md"
  printf '%s\n' "$BODY" | grep -E 'socVersion|opType' | head -2 | sed 's/^/    /'
fi
if [ "$SHAPE" -gt 0 ]; then
  echo "  SHAPE family present ($SHAPE line(s)): the op is registered and the dtype is"
  echo "  fine, but this weight's shape cannot be stored in FRACTAL_NZ (mat2 is the"
  echo "  transposed weight: n == 1 is out_features == 1, k == 1 is in_features == 1)."
  echo "  owner: the NZ gate, which tests dtype/SoC/mode and not shape —"
  echo "         references/startup-failures/nz-weight-format-refusal.md"
  printf '%s\n' "$BODY" | grep -oE 'mat2: \[[0-9]+, [0-9]+\]' | head -3 | sed 's/^/    /'
  echo "  lever: weight_nz_mode / VLLM_ASCEND_ENABLE_NZ  0 = off, 1 = quantised only"
  echo "         (default), 2 = bf16/fp16 too. 2 is the WRONG direction; 0 is the fix."
fi
if [ "$LAY" -gt 0 ]; then
  echo "  dtype x layout family present ($LAY line(s)): the op is registered, the launch"
  echo "  was rejected for an input dtype/layout pair on this SoC."
  echo "  owner: the Python that produced the layout — references/pooling-embedding/fp32-head-nz-format-crash.md"
  printf '%s\n' "$BODY" | grep -oE 'now self is DT_[A-Z0-9_]+, mat2 is DT_[A-Z0-9_]+' | head -2 | sed 's/^/    /'
  printf '%s\n' "$BODY" | grep -oE 'call aclnn[A-Za-z0-9_]+ failed' | sort -u | head -3 | sed 's/^/    /'
fi
if [ "$REG" -eq 0 ] && [ "$SHAPE" -eq 0 ] && [ "$LAY" -eq 0 ]; then
  echo "  no EZ1001 text in the paste — this is not the EZ1001 lane"
fi

echo
echo "== phase of the first hit"
printf '%s\n' "$BODY" | grep -nE 'determine_available_memory|profile_run|_dummy_pooler_run|_dummy_run|process_weights_after_loading|_validate_shared_expert_consistency|Starting to capture|Application startup complete' | head -6 | sed 's/^/  /'
if printf '%s\n' "$BODY" | grep -qE '_validate_shared_expert_consistency|process_weights_after_loading'; then
  echo "  load-phase raise: --enforce-eager, --compilation-config and --max-model-len"
  echo "  cannot move it; the self-check only exists under"
  echo "  additional_config.multistream_overlap_shared_expert (default false)."
fi

echo
echo "== NZ gate at the checked-out ref"
if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
  cat <<'TXT'
  (no root given) the gate moved across releases:
    up to ~v0.11: vllm_ascend/utils.py is_enable_nz(dtype) — False ONLY for
      float16/bfloat16, so fp32 passes; cast in
      AscendUnquantizedLinearMethod.process_weights_after_loading (ops/linear.py)
    v0.20.2rc1:   utils.py _should_trans_nz / maybe_trans_nz — fp32 -> ND, 310P ->
      always NZ, mode 0 -> ND, bf16/fp16 -> NZ only at mode 2, other dtypes -> NZ;
      no shape test, cast in ops/linear.py process_weights_after_loading
    ~v0.23:       same names, "FP32 cannot use NZ" first rule; cast sites are the
      quantised methods
  Re-run with the reporter's vllm-ascend root, or grep:
    rg -n "def is_enable_nz|_should_trans_nz|maybe_trans_nz" vllm_ascend/utils.py
TXT
  exit 0
fi
UT="$ROOT/vllm_ascend/utils.py"
MODE_AT_REF=0
HAVE_GATE=0
if [ -f "$UT" ]; then
  grep -nE 'def is_enable_nz|def _should_trans_nz|def maybe_trans_nz|torch\.float32|torch\.float16|torch\.bfloat16' "$UT" \
    | grep -iE 'nz|float' | head -12 | sed 's/^/  utils.py:/'
  if grep -qE 'def is_enable_nz|def _should_trans_nz|def maybe_trans_nz' "$UT"; then
    HAVE_GATE=1
  fi
  if grep -q 'FP32 cannot use NZ' "$UT"; then
    echo "  VERDICT: this ref never converts fp32 -> NZ; an fp32-head weight-layout"
    echo "           crash cannot come from this gate."
  elif grep -q 'def is_enable_nz' "$UT"; then
    echo "  VERDICT: is_enable_nz excludes only fp16/bf16 => an fp32 weight IS cast to"
    echo "           NZ; the fp32 pooling head is a candidate owner."
  fi
  if [ "$HAVE_GATE" = "1" ]; then
    if sed -n '/def _should_trans_nz/,/^def /p' "$UT" | grep -qE 'shape\[[01]\]|numel\(\)|\.dim\(\)'; then
      echo "  SHAPE GUARD: present in _should_trans_nz at this ref — a one-dimension"
      echo "               weight is left ND, so the shape family is already fixed here."
    else
      echo "  SHAPE GUARD: absent in _should_trans_nz at this ref — every converted"
      echo "               weight with a 1-length dimension reaches aclnn and fails."
    fi
  else
    echo "  NO NZ GATE FUNCTION at this ref (no is_enable_nz / _should_trans_nz /"
    echo "  maybe_trans_nz in utils.py): the layout is decided inline, so there is no"
    echo "  gate and no shape guard to read. Grep the cast sites instead:"
    echo "    rg -n 'npu_format_cast' $ROOT/vllm_ascend/ | head -20"
    echo "    rg -n 'is_310p\(\)' $ROOT/vllm_ascend/       # polarity per site:"
    echo "      'if not is_310p():' => cast runs on 910/A2/A3, skipped on 310P"
    echo "      'if is_310p():'     => 310P-only"
    echo "  Weight-layout capability answers at such a ref:"
    echo "    references/feature-enable-recipes/weight-nz-layout-by-soc.md"
  fi
fi
EN="$ROOT/vllm_ascend/envs.py"
if [ -f "$EN" ]; then
  grep -nA4 'VLLM_ASCEND_ENABLE_NZ' "$EN" | head -12 | sed 's/^/  envs.py:/'
  grep -q 'VLLM_ASCEND_ENABLE_NZ' "$EN" && MODE_AT_REF=1
fi
AC="$ROOT/vllm_ascend/ascend_config.py"
if [ -f "$AC" ]; then
  grep -nB2 -A6 'weight_nz_mode' "$AC" | head -18 | sed 's/^/  ascend_config.py:/'
  grep -q 'weight_nz_mode' "$AC" && MODE_AT_REF=1
fi
if [ "$MODE_AT_REF" = "1" ]; then
  echo "  escape for the reporter: weight_nz_mode 0 (or VLLM_ASCEND_ENABLE_NZ=0);"
  echo "  1 also works when the failing weight is bf16/fp16; 2 makes it certain."
else
  echo "  NO weight_nz_mode / VLLM_ASCEND_ENABLE_NZ at this ref — the layout is not a"
  echo "  flag here, so 'set weight_nz_mode' is advice that does nothing. The answer is"
  echo "  the per-path inventory (which weights are already NZ) plus a code change,"
  echo "  not a switch: references/feature-enable-recipes/weight-nz-layout-by-soc.md"
fi
LIN="$ROOT/vllm_ascend/ops/linear.py"
[ -f "$LIN" ] && grep -nE 'npu_format_cast|FRACTAL_NZ|is_enable_nz|maybe_trans_nz' "$LIN" | head -6 | sed 's/^/  ops\/linear.py:/'
