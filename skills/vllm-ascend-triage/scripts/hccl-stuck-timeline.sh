#!/usr/bin/env bash
# hccl-stuck-timeline.sh -- read a pasted Ascend plog / HCCL log for a *silent*
# whole-service hang (no error text, AICore 0 %, the port still listening) and say
# which collective the ranks are stuck in and whether they are still stepping.
#
# usage: bash hccl-stuck-timeline.sh <log-file> [more files ...]
#        ... - < log.txt
#
# Per file it prints the comm group config (`[Load] comm config of [group_name_N]`
# -> bufferSize in bytes and MiB, execTimeOut), the `HCCL_BUFFSIZE set by` line the
# process actually read, the collective tag from `[BuildRemoteResByTag] newtag[...]`
# (that string carries the op family: `mc2RunAlltoAllDirectFullmesh_device` is an MC2
# MoE all-to-all), the `[heartbeat.cc] ... status[STUCK]` span, and the `count of
# currently executed operators` series per rank marked FROZEN (the count did not move
# across the window), CLIMBING (still stepping), or ONE SAMPLE (a single line cannot
# tell frozen from slow - ask for the series). Then one cross-file verdict: files
# whose ranks all froze on the same operator count in the same second are inside one
# collective instance waiting on the ranks that never arrived. Timestamps are compared
# to the second, because the ranks' millisecond suffixes differ.
#
# Recorded #8975: the D0/D1 pair froze on 2706 at 03:12:39 in a 32-rank 2-server MC2
# AlltoAll group, while the P node printed STUCK at 03:34 and still climbed
# 110822 -> 4191386. `execTimeOut[1836]` is the same 1836 s as the
# `fftsplus task timeout ... timeout=1836s` block, so no error text is expected in
# the first half hour (error-report-timestamps.md), and AICore 0 % at that moment is
# not evidence of a host-side stall (multinode-mc2-alltoall-silent-deadlock.md).
set -u
LC_ALL=C

if [ "$#" -eq 0 ]; then
  echo "usage: bash $0 <log-file> [more files ...]  |  -" >&2
  exit 2
fi

AWK_PROG=$(mktemp "${TMPDIR:-/tmp}/hccl-stuck.XXXXXX.awk")
trap 'rm -f "$AWK_PROG"' EXIT

cat > "$AWK_PROG" <<'AWK'
function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
function longtime(line) {
  if (match(line, /20[0-9][0-9]-[01][0-9]-[0-3][0-9]-[0-2][0-9]:[0-5][0-9]:[0-5][0-9]/))
    return substr(line, RSTART, RLENGTH)
  return ""
}
function stamp(line) { t = longtime(line); if (t != "") return t; return hhmmss(line) }
function hhmmss(line) {
  if (match(line, /[0-2][0-9]:[0-5][0-9]:[0-5][0-9](\.[0-9]+)?/))
    return substr(line, RSTART, RLENGTH)
  return ""
}
function emit_file(   k, r, nfroz, onecount, onecountseen) {
  if (FILE == "") return
  printf "=== %s ===\n", FILE
  if (CCLINE != "") printf "  comm config    : %s\n", CCLINE
  if (KEYLINE != "") printf "  communicator   : %s\n", KEYLINE
  if (BUFLINE != "") printf "  HCCL_BUFFSIZE  : %s\n", BUFLINE
  else printf "  HCCL_BUFFSIZE  : (no \"HCCL_BUFFSIZE set by\" line in this window)\n"
  if (NTAG == 0) printf "  collective tag : (none - ask for a wider plog window)\n"
  for (k = 1; k <= NTAG; k++) printf "  collective tag : %s  (%d line(s))\n", TAG[k], NTAGS[TAG[k]]
  if (NSTUCK == 0) printf "  STUCK lines    : none\n"
  else printf "  STUCK lines    : %d, %s .. %s, %d distinct crimer rank(s)\n", NSTUCK, STUCK_FIRST, STUCK_LAST, NPEER
  if (NRANK == 0) printf "  op count       : (no \"count of currently executed operators\" line)\n"
  nfroz = 0; onecount = ""; onecountseen = 0; nsingle = 0
  for (k = 1; k <= NRANK; k++) {
    r = RID[k]
    printf "  op count       : %s  %d line(s)  first %s @ %s  last %s @ %s", r, RCNT[r], RFIRST[r], RFIRSTT[r], RLAST[r], RLASTT[r]
    if (RCNT[r] == 1) {
      printf "  [ONE SAMPLE - the series is the evidence]\n"
      nsingle++
    } else if (RFIRST[r] == RLAST[r]) {
      printf "  [FROZEN]\n"
      nfroz++
      if (onecount == "") { onecount = RLAST[r]; onecountseen = 1 }
      else if (onecount != RLAST[r]) onecountseen = 0
    } else printf "  [CLIMBING]\n"
  }
  FROZENID = (NRANK > 0 && nfroz == NRANK && onecountseen) ? onecount : ""
  FROZENT = (FROZENID != "") ? RLASTT[RID[1]] : ""
  if (NRANK == 0) STATUS = "NONE"
  else if (nsingle > 0) STATUS = "SINGLE"
  else if (FROZENID == "" && nfroz < NRANK) STATUS = "STEPPING"
  else STATUS = "FROZEN"
  fin[++NFIN] = FILE "|" FROZENID "|" FROZENT "|" GRP "|" STATUS
}

FNR == 1 {
  emit_file()
  FILE = FILENAME; CCLINE = ""; KEYLINE = ""; BUFLINE = ""; GRP = ""
  NTAG = 0; NSTUCK = 0; NPEER = 0; NRANK = 0; STUCK_FIRST = ""; STUCK_LAST = ""
  delete PEER; delete TAG; delete NTAGS; delete RID; delete RCNT
  delete RFIRST; delete RFIRSTT; delete RLAST; delete RLASTT
  FROZENID = ""; FROZENT = ""
}
/comm config of \[/ {
  CCLINE = trim($0)
  if (match($0, /of \[[^]]+\]/)) GRP = substr($0, RSTART + 4, RLENGTH - 5)
  if (match($0, /bufferSize\[[0-9]+\]/)) {
    B = substr($0, RSTART + 11, RLENGTH - 12) + 0
    mib = B / 1048576
    CCLINE = sprintf("%s  [bufferSize=%d B = %s MiB]", CCLINE, B, (mib == int(mib)) ? mib : sprintf("%.1f", mib))
  }
  if (match($0, /execTimeOut\[[0-9]+\]/))
    CCLINE = sprintf("%s  [execTimeOut=%ss]", CCLINE, substr($0, RSTART + 12, RLENGTH - 13))
  next
}
/\[Communicator Key Info\]/ { KEYLINE = trim($0); next }
/HCCL_BUFFSIZE set by/ { BUFLINE = trim($0); next }
/\[BuildRemoteResByTag\]/ {
  if (match($0, /newtag\[[^]]+\]/)) {
    t = substr($0, RSTART + 7, RLENGTH - 8)
    if (!(t in NTAGS)) { TAG[++NTAG] = t; NTAGS[t] = 0 }
    NTAGS[t]++
  }
  next
}
/status\[STUCK\]/ {
  NSTUCK++
  ts = stamp($0)
  if (STUCK_FIRST == "") STUCK_FIRST = ts
  if (ts != "") STUCK_LAST = ts
  if (match($0, /crimer rank \[[^]]+\]\/[0-9]+/)) {
    p = substr($0, RSTART, RLENGTH); sub(/^crimer rank /, "", p)
    if (!(p in PEER)) { PEER[p] = 1; NPEER++ }
  }
  next
}
/count of currently executed operators:/ {
  c = $0; sub(/.*count of currently executed operators:/, "", c)
  sub(/[^0-9].*$/, "", c)
  r = $0
  if (match(r, /rank:\[[^]]+\]\/[0-9]+/)) { r = substr(r, RSTART, RLENGTH); sub(/^rank:/, "", r) }
  else r = "?"
  if (!(r in RCNT)) { RID[++NRANK] = r; RCNT[r] = 0 }
  RCNT[r]++
  ts = stamp($0)
  if (!(r in RFIRSTT)) { RFIRST[r] = c; RFIRSTT[r] = ts }
  RLAST[r] = c; RLASTT[r] = ts
  next
}
END {
  emit_file()
  if (NFIN == 0) exit 0
  print ""
  print "--- cross-file read ---"
  NFROZ = 0
  for (k = 1; k <= NFIN; k++) {
    split(fin[k], a, "|")
    if (a[5] == "NONE") { printf "  %s: no operator-count line in this window\n", a[1]; continue }
    if (a[5] == "SINGLE") { printf "  %s: one operator-count sample only - ask for the series before calling it frozen\n", a[1]; continue }
    if (a[5] == "STEPPING") { printf "  %s: still stepping (at least one rank's count moved)\n", a[1]; continue }
    sec = a[3]; sub(/\.[0-9]+$/, "", sec)
    printf "  %s: frozen on %s (%s) in group [%s]\n", a[1], a[2], sec, a[4]
    key = a[4] "|" a[2] "|" sec
    if (!(key in GCNT)) { GK[++NFROZ] = key; GCNT[key] = 0 }
    GCNT[key]++
  }
  HIT = 0
  for (k = 1; k <= NFROZ; k++) {
    if (GCNT[GK[k]] < 2) continue
    split(GK[k], b, "|")
    HIT++
    printf "  VERDICT: %d files froze on the same operator count %s at %s in group [%s] - one collective instance, every rank waiting on the ones that never arrived. Read newtag[...] for the op family and HCCL_BUFFSIZE/bufferSize for the comm space.\n", GCNT[GK[k]], b[2], b[3], b[1]
  }
  if (NFROZ > 0 && HIT == 0)
    print "  VERDICT: no two files agree on one frozen count - either the peers\047 plogs are missing or they froze on a different number; get every rank plog before calling it a collective deadlock."
  print "  Next: the group rank count and op family decide which vllm-ascend group it is; MC2 all-to-all policies are references/hccl-and-hardware/multinode-mc2-alltoall-silent-deadlock.md."
}
AWK

awk -f "$AWK_PROG" "$@"
