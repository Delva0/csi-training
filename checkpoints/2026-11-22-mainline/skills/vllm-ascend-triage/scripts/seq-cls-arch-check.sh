#!/usr/bin/env bash
# Which model class will the `--hf_overrides architectures` string resolve to?
#
# Usage:
#   bash seq-cls-arch-check.sh <vllm-root> <Arch>[,<Arch> ...]
#   bash seq-cls-arch-check.sh <vllm-root> --list 'Qwen3'
#
# Reproduces ModelRegistry._normalize_arch at the checked-out ref: if the name is
# registered it is used directly, otherwise the architecture's suffix is rewritten
# to each default suffix in priority order and the first registered rewrite wins.
# That rewrite is what decides a text model versus a multimodal one, so a
# `<Family>VL...` prefix on a text checkpoint routes to the VL class.
#
# Prints, per architecture: the registered base it resolves to, the family, and
# "NOT FOUND" when no rewrite is registered (then the report is a model-support
# question, not a routing one).
set -u
ROOT=${1:-}
if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
  echo "usage: bash seq-cls-arch-check.sh <vllm-root> <Arch>[,<Arch>...]" >&2
  exit 2
fi
shift
REG="$ROOT/vllm/model_executor/models/registry.py"
if [ ! -f "$REG" ]; then
  echo "no registry.py under $ROOT — pass the vllm checkout root" >&2
  exit 2
fi

# Registered architecture keys: "<Arch>": ("module", "Class")  or  "<Arch>":
KEYS=$(grep -oE '^[[:space:]]*"[A-Za-z0-9_]+"[[:space:]]*:' "$REG" \
       | tr -d ' ":' | sort -u)

# iter_architecture_defaults() priority order (vllm/config/model.py _SUFFIX_TO_DEFAULTS)
DEFAULTS="ForCausalLM ForConditionalGeneration ChatModel LMHeadModel LMHead ForTextEncoding EmbeddingModel ForSequenceClassification ForTokenClassification ForProcessRewardModel ForRewardModel Model"

is_reg() { printf '%s\n' "$KEYS" | grep -qx "$1"; }

family_of() {
  case "$1" in
    *VL*|*Vision*|*MultiModal*) echo multimodal ;;
    *) echo text ;;
  esac
}

mode=resolve
[ "${1:-}" = "--list" ] && { mode=list; shift; }

for arch in $(printf '%s' "$*" | tr ',' ' '); do
  [ -z "$arch" ] && continue
  if [ "$mode" = list ]; then
    echo "registered architectures matching '$arch':"
    printf '%s\n' "$KEYS" | grep -F "$arch" | sed 's/^/  /'
    continue
  fi
  echo "== $arch"
  if is_reg "$arch"; then
    echo "  registered directly -> $arch ($(family_of "$arch"))"
    continue
  fi
  suffix=""
  for s in $DEFAULTS; do
    case "$arch" in *"$s") suffix="$s"; break ;; esac
  done
  if [ -z "$suffix" ]; then
    echo "  no known default suffix on this name -> unresolved; treat as a model-support question"
    continue
  fi
  echo "  matched suffix: $suffix"
  hit=""
  for repl in $DEFAULTS; do
    [ "$repl" = "$suffix" ] && continue
    base=${arch%"$suffix"}$repl
    if is_reg "$base"; then
      echo "  rewrite -> $base  ($(family_of "$base"))   <-- model class used"
      hit=$base
      break
    fi
  done
  [ -z "$hit" ] && echo "  no registered rewrite (tried $DEFAULTS) -> model-support question"
done
echo
echo "Check the checkpoint's own model_type against the family above: a multimodal"
echo "class on a text checkpoint raises 'TypeError: Invalid type of HuggingFace"
echo "config. Expected type: ...VLConfig, but found type: ...Config' in the APIServer."
