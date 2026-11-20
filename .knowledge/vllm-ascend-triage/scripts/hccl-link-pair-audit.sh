#!/usr/bin/env bash
# Audit an HCCL link-creation failure: which (src,dst) edges failed, and are they
# cross-card or card-local?
#
#   scripts/hccl-link-pair-audit.sh <log|->            # log or paste on stdin
#   scripts/hccl-link-pair-audit.sh <log> --visible 0-3 # rank -> device order
#
# Reads the `Transport_Init_Error(EI0009)` / `[Create][DestLink]Create Dest error!`
# / `createLink para:rank[i]…dst_rank[j]` block, the `[drv api] halShmemOpenHandle
# … drvRetCode=N, device_id=N` and `rtsIpcMemImportByKey … runtime result = N`
# lines, and the reporter's `npu-smi` rows (chip / device / Bus-Id) when the same
# text carries them. Rank r maps to the r-th entry of --visible (default: the
# device ids sorted ascending, i.e. 0..N-1).
#
# Output: the distinct failed pairs, the driver/runtime codes, and one verdict —
#   * every failed edge crosses physical cards  -> host transport / driver pairing
#   * a card-local edge failed as well          -> transport or driver under that
#                                                  card; not a rank-table/topology
#                                                  problem either way
#   * no npu-smi rows in the paste              -> ask for `npu-smi info`
# Equal `localIpAddr` / `remote_ip_addr` text is NOT reported as a finding: the
# field is printed from the local rank on both sides (see the reference page).
set -u

if [ "$#" -eq 0 ]; then
  echo "usage: hccl-link-pair-audit.sh <log|-> [--visible 0-3|0,1,2,3]" >&2
  exit 2
fi

log="$1"; shift || true
visible=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --visible) visible="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$log" = "-" ]; then
  text="$(cat)"
else
  if [ ! -f "$log" ]; then echo "not a file: $log" >&2; exit 2; fi
  text="$(cat "$log")"
fi

PAGE="references/hccl-and-hardware/transport-destlink-ipc-import.md"

tmp="$(mktemp "${TMPDIR:-/tmp}/hccl-link-audit.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
printf '%s\n' "$text" > "$tmp"

uv run --no-project python - "$tmp" "$visible" "$PAGE" <<'PY'
import re
import sys

path = sys.argv[1]
visible_arg = sys.argv[2] if len(sys.argv) > 2 else ""
page = sys.argv[3] if len(sys.argv) > 3 else ""
text = open(path, encoding="utf-8", errors="replace").read()
lines = text.splitlines()

link_re = re.compile(r"createLink para\(s\)", re.I)
pair_re = re.compile(
    r"createLink para:rank\[(\d+)\][^\n]*?dst_rank\[(\d+)\]", re.I)
edge_re = re.compile(r"\[Create\]\[DestLink\]|Transport_Init_Error\(EI\d+\)", re.I)
ipc_re = re.compile(
    r"halShmemOpenHandle\w* failed: name=([0-9A-Fa-f]+), drvRetCode=(\d+), device_id=(\d+)")
rt_re = re.compile(r"rtsIpcMemImportByKey[^\n]*?result = (\d+)")
addr_re = re.compile(r"localIpAddr\[([^\]]*)\][^\n]*?remote_ip_addr\[([^\]]*)\]")
devcount_re = re.compile(r"rtGetDeviceCount:ErrCode=(\d+)")
# npu-smi rows observed as:  | <chip> <device> | 0000:03:00.0 | ...
smi_re = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])")

pairs = []
addrs = set()
drv = set()
rts = set()
smi = {}
for ln in lines:
    for m in pair_re.finditer(ln):
        pairs.append((int(m.group(1)), int(m.group(2))))
    for m in addr_re.finditer(ln):
        addrs.add((m.group(1).strip("[] "), m.group(2).strip("[] ")))
    for m in ipc_re.finditer(ln):
        drv.add((m.group(1)[-8:], int(m.group(2)), int(m.group(3))))
    for m in rt_re.finditer(ln):
        rts.add(int(m.group(1)))
    m = smi_re.match(ln)
    if m:
        smi[int(m.group(2))] = m.group(3)

has_edge = any(edge_re.search(ln) for ln in lines)
devcount = set(m.group(1) for m in (devcount_re.search(ln) for ln in lines) if m)

print("== HCCL link-creation audit ==")
if not has_edge:
    print("no `Transport_Init_Error(EI…)` / `[Create][DestLink]` line in this text")
    if devcount:
        print(f"  BUT `rtGetDeviceCount:ErrCode={', '.join(sorted(devcount))}` is present: the device")
        print("  stack failed before any collective — driver/device layer, not HCCL.")
    print(f"  page: {page}")
    sys.exit(0)

uniq = sorted(set(pairs))
if uniq:
    print(f"failed createLink pairs ({len(uniq)} distinct of {len(pairs)} occurrences):")
    for src, dst in uniq:
        print(f"  rank{src} -> rank{dst}")
else:
    print("failed createLink pairs: none parsed (the paste may be truncated)")

if drv:
    print("driver API:")
    for key, code, dev in sorted(drv):
        print(f"  shmem key …{key}  drvRetCode={code}  local device_id={dev}")
if rts:
    print(f"runtime result(s): {sorted(rts)}"
          + ("  (= [driver error:internal error], generic; the operation verb is the"
             " meaning)" if 507899 in rts else ""))
if addrs:
    same = all(a == b for a, b in addrs)
    note = "identical on both sides — normal text for this record, NOT a duplicate device_ip"
    print(f"address fields: {len(addrs)} distinct (local,remote) pattern(s)"
          + (f"; {note}" if same else ""))
    print("  rank-table ids are decided by scripts/ranktable-check.sh, not this line")

if visible_arg:
    if re.fullmatch(r"\d+-\d+", visible_arg):
        lo, hi = (int(x) for x in visible_arg.split("-"))
        order = list(range(lo, hi + 1))
    else:
        order = [int(x) for x in re.split(r"[,\s]+", visible_arg) if x.strip()]
else:
    order = sorted(smi) if smi else []

if not uniq:
    sys.exit(0)

if not smi:
    print("verdict: cannot place the edges without the rank->device map — ask for `npu-smi info`")
    print(f"  page: {page}")
    sys.exit(0)

missing = [d for d in order if d not in smi]
if missing:
    print(f"verdict: npu-smi rows do not cover device id(s) {missing}; map unknown")
    print(f"  page: {page}")
    sys.exit(0)

def card_of(rank):
    return smi.get(order[rank]) if rank < len(order) else None

cross, local, unknown = [], [], []
for src, dst in uniq:
    a, b = card_of(src), card_of(dst)
    if a is None or b is None:
        unknown.append((src, dst))
    elif a == b:
        local.append((src, dst))
    else:
        cross.append((src, dst))

print("edge classification (physical card = Bus-Id):")
for src, dst in uniq:
    a, b = card_of(src), card_of(dst)
    kind = "unknown" if a is None or b is None else ("card-local" if a == b else "cross-card")
    print(f"  rank{src} (dev {order[src]}, {a}) -> rank{dst} (dev {order[dst]}, {b}): {kind}")

print()
if unknown:
    print("verdict: incomplete map; ask for the full `npu-smi info` block")
elif cross and not local:
    print("verdict: every failing edge is CROSS-CARD and the card-local edges built.")
    print("  host transport / driver: driver-vs-image-CANN pairing, container device")
    print("  passthrough, then a 2-rank (one card) vs 4-rank (both cards) allreduce.")
    print("  Not the model, not the flags, not a duplicate device_ip.")
elif local:
    print("verdict: a CARD-LOCAL edge also failed, so the fault is under one card")
    print("  (driver transport or hardware), not the inter-card path. Check per-device")
    print("  health and the plog before any version theory.")
else:
    print("verdict: no cross-card edge failed; re-read the paste — the symptom is elsewhere")
print(f"page: {page}")
PY
