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
#   3. at a three-valued ref, how `weight_nz_mode` resolves and whether anything
#      tests the weight's shape: `_should_trans_nz` never does (so a bf16 MoE
#      shared-expert gate `[1, hidden]` crashes at mode 2 from v0.20.2rc1 on),
#      while from v0.23.0 the *caller* in `ops/linear.py` keeps `n == 1` / `k == 1`
#      weights ND on 310P only — so the SoC decides whether the case is fixed.
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
echo "== resolved weight_nz_mode in the paste (the discriminator; read it first)"
MODELINE=$(printf '%s\n' "$BODY" | grep -E 'AscendConfig\.weight_nz_mode (is set from additional_config|falls back to environment variable)' | head -2)
if [ -n "$MODELINE" ]; then
  printf '%s\n' "$MODELINE" | sed 's/^/  /'
  if printf '%s\n' "$MODELINE" | grep -q 'with value 2'; then
    echo "  mode 2: bf16/fp16 weights are cast too, so a 1-length bf16 weight crashes here."
  elif printf '%s\n' "$MODELINE" | grep -q 'with value 1'; then
    echo "  mode 1: bf16/fp16 stay ND — a bf16 gate is NOT the owner; read the casting dtype."
  elif printf '%s\n' "$MODELINE" | grep -q 'with value 0'; then
    echo "  mode 0: NZ is off, so this gate is not the owner at all."
  fi
else
  echo "  no resolution line: mode is the envs.py default 1 unless additional_config"
  echo "  carries weight_nz_mode (the line is logged only when one of the two is set)."
fi
if printf '%s\n' "$BODY" | grep -q 'Unknown vLLM environment variable detected: VLLM_ASCEND_ENABLE_NZ'; then
  echo "  ('Unknown vLLM environment variable detected: VLLM_ASCEND_ENABLE_NZ' is upstream"
  echo "   envs.py vocabulary noise — it flags VLLM_USE_V1 too; the switch was read.)"
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
    v0.22.1rc1+:  same names, "FP32 cannot use NZ" is the first rule
      (utils.py:209-236; mode 0 -> ND, bf16/fp16 -> NZ only at mode 2); the cast
      sites are the quantised methods and ops/linear.py:78-89
    v0.23.0:      same gate (utils.py:261-287, mode read from ascend_config) with
      still no shape test inside it, but ops/linear.py:79-80,88,97-98 keeps
      n == 1 / k == 1 weights ND on 310P ONLY (is_310p()), so at that ref the
      shape family is fixed on 310P and open on 910B/A2/A3
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
      echo "  SHAPE GUARD: none inside _should_trans_nz at this ref."
      # The caller may keep narrow weights ND; its SoC scope, not its presence,
      # decides whether the reporter's crash is already fixed at this ref.
      GUARD=$(grep -nE 'def _should_keep_nd_for_310p_weight|keep_nd_weight|shape\[-[12]\] == 1' \
        "$ROOT/vllm_ascend/ops/linear.py" 2>/dev/null | head -4)
      if [ -n "$GUARD" ]; then
        printf '%s\n' "$GUARD" | sed 's/^/    ops\/linear.py:/'
        if printf '%s\n' "$GUARD" | grep -q 'def _should_keep_nd_for_310p_weight'; then
          echo "               caller-side guard PRESENT, 310P-only (is_310p()): a 310P"
          echo "               [1, N] / [N, 1] weight stays ND, on 910B/A2/A3 it is still"
          echo "               converted — read the reporter's SoC before answering."
        else
          echo "               caller-side guard present with another scope: read it at"
          echo "               ops/linear.py before saying where the shape family stands."
        fi
      else
        echo "               no caller-side keep-ND guard either — every converted weight"
        echo "               with a 1-length dimension reaches aclnn and fails."
      fi
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
HAVE_MODE_KEY=0
if [ -f "$AC" ]; then
  grep -nB2 -A6 'weight_nz_mode' "$AC" | head -18 | sed 's/^/  ascend_config.py:/'
  if grep -q 'weight_nz_mode' "$AC"; then
    MODE_AT_REF=1
    HAVE_MODE_KEY=1
  fi
fi
if [ "$MODE_AT_REF" = "1" ] && [ "$HAVE_MODE_KEY" = "1" ]; then
  echo "  escape for the reporter: weight_nz_mode 0 via --additional-config (or"
  echo "  VLLM_ASCEND_ENABLE_NZ=0); 1 also works when the failing weight is bf16/fp16;"
  echo "  2 makes it certain."
elif [ "$MODE_AT_REF" = "1" ]; then
  echo "  escape for the reporter: VLLM_ASCEND_ENABLE_NZ=0 ONLY — this ref has no"
  echo "  weight_nz_mode key, so --additional-config {\"weight_nz_mode\": 0} is inert"
  echo "  here. The flag is boolean at this ref and the dtype gate skips fp16/bf16, so"
  echo "  0 turns off the only conversions that exist (fp32 and every non-half dtype)."
  if sed -n '/def _should_trans_nz/,/^def /p' "$UT" 2>/dev/null | grep -q 'is_310p()'; then
    echo "  caveat: the gate returns True on 310P BEFORE the mode is read, so on a 310P"
    echo "  host this switch is inert — do not offer mode 0 there (the 310P path keeps"
    echo "  n == 1 / k == 1 weights ND instead; ../soc-platform-paths/atlas-300i-310p.md)."
  fi
else
  echo "  NO weight_nz_mode / VLLM_ASCEND_ENABLE_NZ at this ref — the layout is not a"
  echo "  flag here, so 'set weight_nz_mode' is advice that does nothing. The answer is"
  echo "  the per-path inventory (which weights are already NZ) plus a code change,"
  echo "  not a switch: references/feature-enable-recipes/weight-nz-layout-by-soc.md"
fi
LIN="$ROOT/vllm_ascend/ops/linear.py"
[ -f "$LIN" ] && grep -nE 'npu_format_cast|FRACTAL_NZ|is_enable_nz|maybe_trans_nz|keep_nd_weight|_should_keep_nd_for_310p_weight' "$LIN" | head -8 | sed 's/^/  ops\/linear.py:/'
