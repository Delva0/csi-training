#!/usr/bin/env bash
# PP layer partition + stage-start fingerprint.
#
# Mirrors vllm/distributed/utils.py::get_pp_indices (v0.12.0 `:95-141`): base =
# layers // pp_size, and the `layers % pp_size` remainder is added one layer at a
# time to partitions [-2], [-3], ... (the LAST partition is never enlarged). The
# naive `layers//pp` boundary is therefore wrong whenever layers % pp != 0.
#
# usage: pp-layer-partition.sh --layers N --pp P [--key-layer K]
#   --layers N     num_hidden_layers of the checkpoint (e.g. 61 for Kimi-K2 family)
#   --pp P         --pipeline-parallel-size
#   --key-layer K  optional layer index from a KeyError such as
#                  "KeyError: 'model.layers.30.self_attn.attn'"
#
# Prints one line per stage (start-end inclusive) and, with --key-layer, whether
# K is a stage start (the only indices a *foreign* per-worker kv_cache_config can
# name when `initialize_from_config` reads the wrong list element).
#
# Exit codes: 0 ok, 2 usage. No network, no python.
set -u
layers=''; pp=''; key=''
while [ $# -gt 0 ]; do
  case "$1" in
    --layers) layers="${2:-}"; shift 2 ;;
    --pp) pp="${2:-}"; shift 2 ;;
    --key-layer) key="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "pp-layer-partition.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done
case "$layers" in ''|*[!0-9]*) echo "pp-layer-partition.sh: --layers N required (positive integer)" >&2; exit 2 ;; esac
case "$pp" in ''|*[!0-9]*) echo "pp-layer-partition.sh: --pp P required (positive integer)" >&2; exit 2 ;; esac
if [ "$pp" -lt 1 ] || [ "$layers" -lt "$pp" ]; then
  echo "pp-layer-partition.sh: need layers >= pp >= 1 (got layers=$layers pp=$pp)" >&2; exit 2
fi

base=$((layers / pp)); rem=$((layers % pp))
declare -a part=()
i=0
while [ "$i" -lt "$pp" ]; do part[$i]=$base; i=$((i+1)); done
if [ "$rem" -gt 0 ]; then
  i=2
  while [ "$i" -le $((rem + 1)) ]; do
    idx=$((pp - i))
    part[$idx]=$(( ${part[$idx]} + 1 ))
    i=$((i+1))
  done
fi

starts=''; start=0
i=0
while [ "$i" -lt "$pp" ]; do
  last=$((start + part[$i] - 1))
  printf 'pp_rank %d: layers %d-%d (%d layers)%s\n' "$i" "$start" "$last" "${part[$i]}" \
    "$( [ "$i" -eq 0 ] && echo '  <- stage start' || echo '')"
  if [ -z "$starts" ]; then starts="$start"; else starts="$starts,$start"; fi
  start=$((last + 1))
  i=$((i+1))
done
printf 'stage starts: %s\n' "$starts"

if [ -n "$key" ]; then
  case "$key" in *[!0-9]*) echo "pp-layer-partition.sh: --key-layer K must be an integer" >&2; exit 2 ;; esac
  [ "$key" -ge "$layers" ] && echo "pp-layer-partition.sh: --key-layer $key >= layers $layers" >&2 && exit 2
  start=0; i=0; owner=''
  while [ "$i" -lt "$pp" ]; do
    last=$((start + part[$i] - 1))
    if [ "$key" -ge "$start" ] && [ "$key" -le "$last" ]; then owner="$i"; fi
    start=$((last + 1)); i=$((i+1))
  done
  printf 'key layer %d: belongs to pp_rank %s\n' "$key" "$owner"
  hit=no
  s=0
  for p in $(echo "$starts" | tr ',' ' '); do
    [ "$p" = "$key" ] && hit="yes ($key is the first layer of that stage)"
  done
  if [ "$hit" = "no" ]; then
    echo "foreign-config fingerprint: NO - a worker reading another stage's list would name that"
    echo "  stage's FIRST layer, so the printed index contradicts the rank/ordering theory;"
    echo "  look for a per-layer registration gap instead."
  else
    echo "foreign-config fingerprint: $hit - consistent with a wrong per-worker kv_cache_config"
    echo "  (sibling of vllm/v1/worker/worker_base.py initialize_from_config, see"
    echo "  references/pipeline-parallel/kv-config-rank-mismatch.md)."
  fi
fi
exit 0
