#!/usr/bin/env bash
# Probe the CHECKPOINT SOURCE of a start-up that is frozen inside weight loading
# ("Loading safetensors checkpoint shards: 0/N" with no error and no
# `Loading model weights took` line). Prints the size arithmetic, the mount that
# backs the path, a timed raw read, and the shard-file completeness checks; with
# a worker pid it also samples /proc/<pid>/io so "reading" and "blocked" are told
# apart (see references/startup-failures/weight-load-stall-in-shard-loop.md).
#
#   scripts/shard-source-probe.sh <model-dir> [pid] [--tp N] [--bytes MiB]
#
# Run it INSIDE the container (so the mount view is the one the loader sees) and,
# when the two differ, once on the host as well. Origin: #11095 (vllm-ascend
# v0.21.0rc1 / vllm v0.21.0), where a fresh copy of the directory plus a fresh
# bind mount fixed the stall.
set -u

if [ "$#" -eq 0 ]; then
  echo "usage: shard-source-probe.sh <model-dir> [pid] [--tp N] [--bytes MiB]" >&2
  exit 2
fi

DIR=""
PID=""
TP=1
BYTES=1024
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tp) TP="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    -*) echo "shard-source-probe.sh: unknown flag $1" >&2; exit 2 ;;
    *) if [ -z "$DIR" ]; then DIR="$1"; elif [ -z "$PID" ]; then PID="$1"; fi; shift ;;
  esac
done

if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
  echo "shard-source-probe.sh: not a directory: '${DIR}'" >&2
  exit 2
fi

echo "== checkpoint source: $DIR"
echo "-- mount backing the path (findmnt / /proc/mounts longest prefix)"
if command -v findmnt >/dev/null 2>&1; then
  findmnt -T "$DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null || true
else
  uv run --no-project python - "$DIR" <<'PY'
import os, sys
p = os.path.realpath(sys.argv[1])
best = ("", "", "")
for line in open("/proc/mounts"):
    parts = line.split()
    if len(parts) < 4:
        continue
    mp, src, fstype = parts[1], parts[0], parts[2]
    if (p == mp or p.startswith(os.path.join(mp, ""))) and len(mp) > len(best[0]):
        best = (mp, src, fstype)
print(" ".join(best) if best[0] else "no mount entry matched")
PY
fi
echo

echo "-- size arithmetic (per-rank = total / TP)"
uv run --no-project python - "$DIR" "$TP" <<'PY'
import glob, json, os, sys
d, tp = sys.argv[1], max(1, int(sys.argv[2]))
files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
total = sum(os.path.getsize(f) for f in files)
gib = total / 1024 ** 3
print(f"shard files : {len(files)}  ({gib:.2f} GiB total)")
if files:
    print(f"per rank @TP={tp}: {gib / tp:.2f} GiB = {gib / tp * 1024:.0f} MiB")
    big = max(files, key=os.path.getsize)
    print(f"largest     : {os.path.basename(big)}  {os.path.getsize(big) / 1024 ** 3:.2f} GiB")
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    try:
        wm = json.load(open(idx)).get("weight_map", {})
        want = len(set(wm.values()))
        print(f"index.json  : {want} shards expected by weight_map"
              + ("" if want == len(files) else "  <-- MISMATCH vs files on disk"))
    except Exception as e:  # noqa: BLE001
        print(f"index.json  : unreadable ({e})")
else:
    print("index.json  : absent (single-file or non-standard checkpoint)")
print("note: resident HBM equal to per-rank size does NOT prove the shards were")
print("      read - vLLM allocates the params on the device before reading")
print("      (vllm/model_executor/model_loader/base_loader.py:53-64).")
PY
echo

echo "-- newest / oldest file mtimes (a copy in flight shows seconds apart)"
ls -l --time-style=full-iso "$DIR" 2>/dev/null | awk 'NR>1{print $6, $7, $9}' | sort | sed -n '1p;$p'
echo

echo "-- timed raw read: ${BYTES} MiB from the largest shard"
BIG=$(ls -S "$DIR"/*.safetensors 2>/dev/null | head -1)
if [ -z "$BIG" ]; then
  echo "   no *.safetensors in this directory"
else
  start=$(date +%s.%N)
  if timeout 20 dd if="$BIG" of=/dev/null bs=1M count="$BYTES" status=none; then
    end=$(date +%s.%N)
    uv run --no-project python - "$start" "$end" "$BYTES" <<'PY'
import sys
s, e, mb = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
dt = max(e - s, 1e-6)
rate = mb / dt
print(f"   read {mb:.0f} MiB in {dt:.2f} s = {rate:.0f} MiB/s")
if rate < 20:
    print("   -> slow read path: the checkpoint source can own a stall this long")
else:
    print("   -> read path is healthy; a frozen bar is then a blocked/re-queued read")
    print("      or work after the copy, so take the /proc sample below")
PY
  else
    echo "   dd did not finish within 20 s -> the READ PATH IS BLOCKED"
    echo "   next: ps -o pid,stat,wchan:32,cmd -p <worker pid>; dmesg -T | tail -60"
  fi
fi
echo

if [ -n "$PID" ]; then
  echo "== is pid $PID reading or blocked? (/proc/$PID/io sampled 5 s apart)"
  if [ ! -r "/proc/$PID/io" ]; then
    echo "   cannot read /proc/$PID/io - run in the pid namespace that owns it"
  else
    A=$(awk '/^read_bytes/{print $2}' "/proc/$PID/io")
    sleep 5
    B=$(awk '/^read_bytes/{print $2}' "/proc/$PID/io")
    uv run --no-project python - "$A" "$B" <<'PY'
import sys
a, b = int(sys.argv[1] or 0), int(sys.argv[2] or 0)
d = b - a
print(f"   read_bytes delta: {d / 1024 ** 2:.1f} MiB in 5 s = {d / 1024 ** 2 / 5:.1f} MiB/s")
print("   -> progressing" if d > 0 else "   -> ZERO progress: blocked in the loader, not slow")
PY
    ps -o pid,stat,wchan:32,cmd -p "$PID" 2>/dev/null || true
  fi
  echo
  echo "== dmesg (if permitted)"
  dmesg -T 2>/dev/null | grep -iE 'hung_task|blocked for more than|nfs|cifs|I/O error|EXT4-fs error' | tail -10 \
    || echo "   (no dmesg access or no matching line)"
else
  echo "== pass the worker pid (from the reporter's npu-smi process table) to also"
  echo "   sample /proc/<pid>/io and get the reading-vs-blocked verdict"
fi
