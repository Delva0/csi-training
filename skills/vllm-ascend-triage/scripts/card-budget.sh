#!/usr/bin/env bash
# card-budget.sh — reproduce the per-card HBM arithmetic behind a deployment
# layout question ("can this model run on N cards?", "why does my 3-stage /
# multi-process layout fail on 2 NPUs?").
#
# It answers two things, and only two:
#   1. one engine's weights per card vs the HBM that engine is allowed to use
#      (gpu_memory_utilization), i.e. is there room left for KV and activations;
#   2. the sum of gpu_memory_utilization over the processes/stages that share a
#      device, i.e. do two stages on one card over-subscribe it by construction.
#
# It does NOT model activations, KV growth, MTP/draft weights, EP/DP overlap or
# fragmentation: a layout that passes here can still OOM on KV at load time.
# Treat a FAIL as decisive and a PASS as "not yet excluded".
set -euo pipefail

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

usage: card-budget.sh --params P [--dtype bf16] [--hbm 64] [--gmu 0.9] [--tp 1]
                      [--cards N] [--stage-gmu 0.9,0.6,0.3] [--stage-devices '0,1;2;2']

  --params P         total parameters IN BILLIONS, not a raw count: 30.5 for
                     Qwen3-Omni-30B-A3B, 671 for DeepSeek-R1 (671e9 params).
                     Take it from the checkpoint, not from the model name:
                     sum the tensor sizes in the safetensors index, or
                     `du -sb <dir>/*.safetensors | tail -1` / bytes-per-weight
  --dtype D          bf16|fp16|fp32|w8a8|int8|w4a8|w4a4   (default bf16)
  --hbm GB           HBM per card in GB                     (default 64)
  --gmu F            gpu_memory_utilization of the engine   (default 0.9)
  --tp N             tensor-parallel size sharding it       (default 1)
  --cards N          cards on the host, used to validate device indices
  --stage-gmu L      comma-separated per-stage gpu_memory_utilization
  --stage-devices L  per-stage device lists, ';'-separated ('0,1;2;2')

examples:
  bash card-budget.sh --params 30.5 --tp 2 --gmu 0.9 --cards 2
  bash card-budget.sh --params 671 --dtype w8a8 --tp 16 --gmu 0.8   # DeepSeek-R1
  bash card-budget.sh --params 30.5 --cards 2 --stage-gmu 0.9,0.6,0.3 \
       --stage-devices '0,1;2;2'          # the bundled 3-NPU layout
  bash card-budget.sh --params 30.5 --cards 2 --stage-gmu 0.9,0.6,0.3 \
       --stage-devices '0,1;0;0'          # the same pipeline squeezed onto 2
EOF
}

params=""; dtype="bf16"; hbm="64"; gmu="0.9"; tp="1"; cards=""; stage_gmu=""; stage_dev=""
while [ $# -gt 0 ]; do
  case "$1" in
    --params) params="${2:-}"; shift 2 ;;
    --dtype) dtype="${2:-}"; shift 2 ;;
    --hbm) hbm="${2:-}"; shift 2 ;;
    --gmu) gmu="${2:-}"; shift 2 ;;
    --tp) tp="${2:-}"; shift 2 ;;
    --cards) cards="${2:-}"; shift 2 ;;
    --stage-gmu) stage_gmu="${2:-}"; shift 2 ;;
    --stage-devices) stage_dev="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$params" ] || { echo "error: --params is required" >&2; usage >&2; exit 2; }

case "$dtype" in
  bf16|fp16) bytes=2 ;;
  fp32) bytes=4 ;;
  w8a8|int8) bytes=1 ;;
  w4a8|w4a4) bytes=0.5 ;;
  *) echo "error: unsupported --dtype '$dtype'" >&2; exit 2 ;;
esac

awk -v P="$params" -v B="$bytes" -v HBM="$hbm" -v GMU="$gmu" -v TP="$tp" \
    -v CARDS="$cards" -v SG="$stage_gmu" -v SD="$stage_dev" '
function gb(x) { return sprintf("%.1f", x) }
BEGIN {
  # Unit guard. The single most common misuse is passing the parameter COUNT
  # (671e9) or a byte total instead of billions (671): every number below is then
  # off by 1e9 and the verdict inverts to a confident, wrong FAIL. Refuse it
  # instead of printing arithmetic a reader may quote.
  if (P >= 1e6) {
    printf("error: --params takes PARAMETERS IN BILLIONS, got %.0f.\n", P)
    printf("       Divide by 1e9: DeepSeek-R1 is 671, Qwen3-32B is 32.8.\n")
    exit 2
  }
  if (TP <= 0) {
    printf("error: --tp must be a positive integer, got %s\n", TP)
    exit 2
  }
  total = P * B                       # 1e9 params * B bytes = B GB per 1e9 params
  per_card = total / TP
  budget = HBM * GMU
  printf("weights: %.2f GB for one full copy (%s), %.2f GB per card at TP=%d\n", total, B " B/param", per_card, TP)
  printf("budget : %.1f GB per card (hbm %s GB x gmu %s)\n", budget, HBM, GMU)
  room = budget - per_card
  if (per_card > budget) {
    printf("VERDICT: FAIL — one engine copy does not fit at this TP/gmu " \
           "(%.1f GB short). Raise TP, quantise, or lower the model footprint.\n", per_card - budget)
  } else if (room < 0.10 * budget) {
    printf("VERDICT: MARGINAL — %.1f GB left for KV/activations (<10%% of budget).\n", room)
  } else {
    printf("VERDICT: weights fit, %.1f GB left per card for KV/activations " \
           "(this does not predict the KV pool).\n", room)
  }

  if (SG == "") exit 0

  n = split(SG, g, ",")
  m = split(SD, d, ";")
  printf("\nper-device over-subscription (sum of gpu_memory_utilization):\n")
  if (m != n)
    printf("  note: %d stage-gmu values vs %d stage-device lists — pairing by index\n", n, m)
  worst = 0
  for (i = 1; i <= n; i++) {
    k = split(d[i], dev, ",")
    for (j = 1; j <= k; j++) {
      name = dev[j]; gsub(/[ \t]/, "", name)
      if (name == "") continue
      sum[name] += g[i] + 0
      w[name] += (total / k) + 0        # this stage shards its weights over its devices
      seen[name] = 1
      if (CARDS != "" && (name + 0) >= (CARDS + 0))
        printf("  stage %d names device %s but --cards %s has indices 0..%d — " \
               "that card does not exist on this host\n", i - 1, name, CARDS, CARDS - 1)
    }
  }
  for (name in seen) {
    verdict = "ok"
    if (sum[name] > 1.0) verdict = "FAIL — over-subscribed: weights+KV cannot both fit"
    else if (sum[name] > 0.95) verdict = "MARGINAL — no room for activations/KV"
    if (sum[name] > worst) worst = sum[name]
    printf("  device %-3s gmu sum %.2f  weights %.1f GB  -> %s\n", name, sum[name], w[name], verdict)
  }
  if (worst > 1.0)
    printf("VERDICT: the layout is memory-infeasible as written; it is not a " \
           "scheduler/connector bug. Give each stage its own card, cut the " \
           "stage set, or quantise.\n")
}' 
