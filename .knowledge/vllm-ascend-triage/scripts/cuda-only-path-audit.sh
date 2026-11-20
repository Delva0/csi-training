#!/usr/bin/env bash
# cuda-only-path-audit.sh <vllm-root> [subpath] [vllm-ascend-root]
#
# Decides the upstream half of "Torch not compiled with CUDA enabled" at engine
# start: which classes in a vllm subpackage hardcode a CUDA device with no
# platform guard, whether each still has a portable forward_* implementation,
# and - when a vllm-ascend root is given - whether downstream mentions the class
# or its CustomOp-registered op name at all (i.e. replaces it with an OOT impl).
#
# Verified on #4309 (vllm v0.11.0 b8b302cde434, vllm-ascend v0.11.0rc1
# c5fe179cef7f): one hit in the whole rotary_embedding package,
# dual_chunk_rope.py:37, 0 current_platform refs, forward_cuda+forward_native,
# and the downstream tree mentions only the torchair model wrapper.
set -uo pipefail

ROOT=${1:-}
SUB=${2:-vllm/model_executor/layers/rotary_embedding}
DOWN=${3:-}

if [ -z "$ROOT" ] || [ ! -d "$ROOT/vllm" ]; then
  echo "usage: $0 <vllm-root containing vllm/> [subpath] [vllm-ascend-root]" >&2
  exit 2
fi
DIR="$ROOT/$SUB"
if [ ! -d "$DIR" ]; then
  echo "no such directory: $DIR" >&2
  exit 2
fi

echo "== unguarded torch.cuda uses under $SUB =="
echo "   (a file that also references current_platform is guardable; one that does not"
echo "    is CUDA-only wherever the checkpoint config selects it)"
echo

found=0
for f in "$DIR"/*.py; do
  hits=$(grep -nE 'torch\.cuda\.|torch\.device\(["'"'"']cuda' "$f" || true)
  [ -z "$hits" ] && continue
  found=1
  base=${f#"$ROOT"/}
  guard=$(grep -c 'current_platform' "$f" || true)
  fmethods=$(grep -oE 'def forward_(native|cuda|hip|oot)' "$f" | sed 's/def //' | sort -u | paste -sd, -)
  cls=$(grep -oE 'class [A-Za-z_][A-Za-z0-9_]*' "$f" | sed 's/class //' | paste -sd, -)
  ops=$(grep -oE 'CustomOp\.register\("[^"]+"' "$f" | sed -E 's/.*"(.*)"/\1/' | paste -sd, -)
  echo "$base"
  printf '%s\n' "$hits" | sed 's/^/    /'
  echo "    current_platform refs: $guard   forward impls: ${fmethods:-none}"
  echo "    classes: ${cls:-none}   registered ops: ${ops:-none}"

  if [ -n "$DOWN" ] && [ -d "$DOWN" ]; then
    patterns=""
    [ -n "$ops" ] && patterns=$(printf '%s' "$ops" | tr ',' '\n' | paste -sd'|' -)
    [ -n "$cls" ] && patterns="$patterns${patterns:+|}${cls//,/\|}"
    if [ -z "$patterns" ]; then
      # an empty alternation matches every line: say "nothing to match" instead
      echo "    downstream: no class or registered op name in this file -> nothing to match"
    else
      dl=$(grep -rnE "$patterns" "$DOWN" --include='*.py' 2>/dev/null | head -6)
      if [ -z "$dl" ]; then
        echo "    downstream: no mention -> upstream class runs unreplaced"
      else
        echo "    downstream mentions:"
        printf '%s\n' "$dl" | sed 's/^/        /'
        echo "        (an import/pass-through is not a replacement: look for register_oot /"
        echo "         a subclass with its own __init__ that builds the device portably)"
      fi
    fi
  fi
  echo
done

if [ "$found" -eq 0 ]; then
  echo "no torch.cuda reference under $SUB at this ref -> the device hardcode is gone"
  echo "(or lives in another subpackage: rerun with that path as \$2)"
fi

cat <<'NOTE'
== how to read this ==
1. A flagged file with 0 current_platform refs and a single forward_cuda is
   CUDA-only: a checkpoint config key that selects it fails in __init__ before any
   device op, so do not propose CANN/driver/flag changes.
2. forward_native present means the forward path is portable and only the device
   (or the cache tensors) hardcode blocks startup - the *shape* it emits still has
   to have a downstream consumer. Grep the attention backend for the feature's
   fields (chunk_size / local_size / q_inter and friends) before proposing a
   one-line device patch.
3. No downstream mention of the class or op name means vllm-ascend does not
   replace it; ask for an OOT implementation as a feature request, and offer the
   checkpoint-config fallback (strip the key from a copy of config.json, quoting
   the accuracy caveat for the feature being disabled).
NOTE
