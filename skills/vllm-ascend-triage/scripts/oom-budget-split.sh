#!/usr/bin/env bash
# oom-budget-split.sh — read a device-side OOM line as a budget, not as a number.
#
# Given `NPU out of memory. Tried to allocate X (NPU n; T total capacity;
# A already allocated; ...; F free; ...)`, this prints
#   1. the parsed total / already-allocated / free / requested values;
#   2. the budget the engine granted itself, gmu x T, and whether the failing
#      request sits INSIDE it (the accounting is wrong) or OUTSIDE it (the
#      reserve above the budget, i.e. the phase, is what ran out);
#   3. with --params/--tp, the weights-per-card estimate and what the remaining
#      "already allocated" bytes then belong to.
#
# It does not decide the cause: it splits capacity vs post-budget phase, which is
# the first branch of the memory pages. Worked example: vllm-ascend #6572.
#
#   scripts/oom-budget-split.sh --line 'NPU out of memory. Tried to allocate 2.36 GiB (NPU 0; 60.94 GiB total capacity; 58.82 GiB already allocated; 58.82 GiB current active; 823.69 MiB free; 58.93 GiB reserved in total by PyTorch)' --gmu 0.8 --params 80 --tp 8
#   scripts/oom-budget-split.sh --line '<line>' --gmu 0.8 --params 80 --tp 4 --dtype w8a8 --hbm 32
#   scripts/oom-budget-split.sh --file oom.txt --gmu 0.85 --params 30 --tp 4
set -u

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

line=""; file=""; gmu=""; params=""; tp=""; dtype="bf16"; hbm=""
while [ $# -gt 0 ]; do
  case "$1" in
    --line) line="${2:-}"; shift 2 ;;
    --file) file="${2:-}"; shift 2 ;;
    --gmu) gmu="${2:-}"; shift 2 ;;
    --params) params="${2:-}"; shift 2 ;;
    --tp) tp="${2:-}"; shift 2 ;;
    --dtype) dtype="${2:-}"; shift 2 ;;
    --hbm) hbm="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [ -z "$line" ] && [ -n "$file" ]; then line="$(cat "$file")"; fi
if [ -z "$line" ] && [ ! -t 0 ]; then line="$(cat)"; fi
[ -n "$line" ] || { echo "error: pass --line '<OOM line>' or --file <path> (or pipe it on stdin)" >&2; exit 2; }
[ -n "$gmu" ] || { echo "error: --gmu is required (the deployment's --gpu-memory-utilization)" >&2; exit 2; }

OOM_LINE="$line" GMU="$gmu" PARAMS="$params" TP="${tp:-1}" DTYPE="$dtype" HBM="$hbm" \
uv run --no-project python - <<'PY'
import os, re, sys

line = os.environ["OOM_LINE"]
gmu = float(os.environ["GMU"])
params = os.environ["PARAMS"].strip()
tp = max(1, int(os.environ["TP"]))
dtype = os.environ["DTYPE"].strip()
hbm_override = os.environ["HBM"].strip()

UNIT = {"B": 1 / 2**30, "KiB": 1 / 2**20, "MiB": 1 / 2**10, "GiB": 1.0, "Gi": 1.0}

def grab(label):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB)\s+" + label, line)
    return float(m.group(1)) * UNIT[m.group(2)] if m else None

total = grab("total capacity")
allocated = grab("already allocated")
free = grab(r"free")
reserved = grab(r"reserved in total")
req = None
m = re.search(r"[Tt]ried to allocate\s+([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB)", line)
if m:
    req = float(m.group(1)) * UNIT[m.group(2)]

if total is None:
    total = None
if allocated is None:
    print("could not find '<N> GiB already allocated' in the line — paste the full OOM text")
    print("line:", line[:400])
    sys.exit(2)
if total is None and hbm_override:
    total = float(hbm_override)
if total is None:
    print("could not find '<N> GiB total capacity' — pass --hbm <GB> to supply the card size")
    sys.exit(2)

print(f"parsed      : total={total:.2f} GiB  already_allocated={allocated:.2f} GiB  "
      + (f"requested={req:.2f} GiB  " if req is not None else "")
      + (f"free={free*1024:.0f} MiB  " if free is not None else "")
      + (f"reserved={reserved:.2f} GiB" if reserved is not None else ""))

budget = gmu * total
delta = allocated - budget
print(f"gmu budget  : {gmu:g} x {total:.2f} GiB = {budget:.2f} GiB")
print(f"delta       : already_allocated - budget = {delta:+.2f} GiB")
if delta > 0.5:
    verdict = "OUTSIDE the profiled budget"
    print(f"VERDICT     : {verdict} — the failing request ran past the memory the engine "
          f"grants itself, i.e. after the KV pool was sized (graph capture / workspace / "
          f"collective buffers) or with a wrong total-memory figure. The reserve "
          f"(1-gmu) x total = {(1-gmu)*total:.2f} GiB is the quantity that is too small.")
elif delta < -0.5:
    verdict = "INSIDE the profiled budget"
    print(f"VERDICT     : {verdict} — the allocation that failed is smaller than the "
          f"budget, so the capacity arithmetic itself is wrong: per-card weights + "
          f"non-KV peak already exceed gmu x total at this TP, or the total-memory "
          f"figure the engine used is not the {total:.2f} GiB reported here.")
else:
    verdict = "AT the budget edge"
    print(f"VERDICT     : {verdict} — the KV pool was filled to gmu x total and the next "
          f"allocation (a block/page or a capture buffer) did not fit. On a hybrid "
          f"(attention+mamba) model the page is tens of MiB, so lower gmu in ~0.05 "
          f"steps and quote the resulting 'GPU KV cache size' line.")

bytes_per_param = {"bf16": 2, "fp16": 2, "fp32": 4, "w8a8": 1, "int8": 1,
                   "w4a8": 0.5, "w4a4": 0.5}
if params:
    try:
        p = float(params)
    except ValueError:
        print(f"note        : --params '{params}' is not a number of billions; skipping the weights split")
        p = None
    if p is not None:
        if dtype not in bytes_per_param:
            print(f"note        : unknown --dtype '{dtype}' (bf16|fp16|fp32|w8a8|int8|w4a8|w4a4); using bf16")
            dtype = "bf16"
        w = p * bytes_per_param[dtype]
        per_card = w / tp
        other = allocated - per_card
        print(f"weights     : {w:.1f} GB for one copy ({dtype}); {per_card:.1f} GB per card at TP={tp}")
        print(f"remainder   : {other:+.2f} GiB of already_allocated is NOT weights "
              f"({'KV pool + activation peak + capture/workspace' if other > 0 else 'negative — the checkpoint is smaller than the name suggests: read the safetensors index'})")
        if other > 0:
            kvh_note = ("for a model with num_key_value_heads <= TP the attention KV per "
                        "token does NOT shrink between TP4 and TP8 (get_num_kv_heads = "
                        "max(1, kv_heads//TP)), so a TP8 run cannot buy KV headroom there")
            print(f"              {kvh_note}")

print()
print("next checks (cheapest first):")
print("  1. the startup log's own numbers: 'Available KV cache memory', "
      "'GPU KV cache size: N tokens', 'Setting attention block size to N tokens', "
      "'Padding mamba page size by X%', 'Using max model len N' — they either match the "
      "arithmetic or the checkpoint differs from its defaults")
print("  2. diff the command against the pinned tutorial/CI serve line "
      "(docs/source/tutorials/<model>.md, tests/e2e/**/test_<model>*.py)")
print("  3. one capture ablation (--enforce-eager, or cudagraph_capture_sizes=[1]) "
      "only if the verdict above is OUTSIDE the budget")
PY
