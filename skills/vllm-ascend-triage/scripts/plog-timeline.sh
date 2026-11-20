#!/usr/bin/env bash
# plog-timeline.sh -- decide which error block in a pasted Ascend plog / serve log
# belongs to the reported incident and which carry an older date: Ascend re-prints
# the device/runtime error record inside a new traceback and keeps that record's
# original date, so an older record is either an earlier run's replay or this
# process's own init/capture window reported late. Process identity and position in
# the log decide which (SKILL.md rule 13), never the size of the gap.
#
# usage: bash plog-timeline.sh [--since YYYY-MM-DD] <log-file> [more files ...]
#        ... - < log.txt
#
#   --since <date>  the date the reporter's failure happened. Records dated before
#                   it are older: a previous run's replay, or this process's own
#                   `init engine (...)` / graph-capture window when the report is
#                   late -- check the startup lines and the pid before discarding
#                   one. If no record reaches it, the pasted window does not contain
#                   the fault at all.
#
# For each error family it prints the span of the embedded device-record timestamps
# (long form `YYYY-MM-DD-HH:MM:SS`) and of the log-line timestamps (`MM-DD HH:MM:SS`
# prefixes), how many lines carry a record older than their own log line, and a
# verdict naming the replayed families and the first live error.
set -u
LC_ALL=C

SINCE=""
case "${1:-}" in
  --since) SINCE="${2:-}"; shift 2 || true ;;
  --since=*) SINCE="${1#--since=}"; shift ;;
esac
SINCE_KEY=$(printf '%s' "$SINCE" | tr -d -- '-')
case "$SINCE_KEY" in
  "") ;;
  [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) ;;
  *) echo "--since needs YYYY-MM-DD (got '$SINCE')" >&2; exit 2 ;;
esac

if [ "$#" -eq 0 ]; then
  echo "usage: bash $0 [--since YYYY-MM-DD] <log-file> [more files ...]  |  -" >&2
  exit 2
fi

AWK_PROG=$(mktemp "${TMPDIR:-/tmp}/plog-timeline.XXXXXX.awk")
DETECT_PROG=$(mktemp "${TMPDIR:-/tmp}/plog-detect.XXXXXX.awk")
HELPERS=$(mktemp "${TMPDIR:-/tmp}/plog-helpers.XXXXXX.awk")
trap 'rm -f "$AWK_PROG" "$DETECT_PROG" "$HELPERS"' EXIT

cat > "$HELPERS" <<'AWKHELPERS'
# --- time-base helpers -------------------------------------------------------
# Ascend dumps put the collector's prefix ([YYYY-MM-DD HH:MM:SS.mmm][sv-...]) in
# LOCAL time while the app logger (`ERROR MM-DD HH:MM:SS`) and every embedded
# device record (`YYYY-MM-DD-HH:MM:SS.mmm.uuu`) keep the container's UTC clock.
# A constant whole-hour lag between the two is therefore a timezone offset, not a
# replayed record (#16652: prefix 21:43:12 vs record 13:43:12 on a UTC+8 host).
function days_from_civil(y, m, d,   era, yoe, doy, doe) {
  y -= (m <= 2)
  era = int((y >= 0 ? y : y - 399) / 400)
  yoe = y - era * 400
  doy = int((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1
  doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
  return era * 146097 + doe - 719468
}
function civil_from_days(z,   era, doe, yoe, y, doy, mp, d, m) {
  z += 719468
  era = int((z >= 0 ? z : z - 146096) / 146097)
  doe = z - era * 146097
  yoe = int((doe - int(doe / 1460) + int(doe / 36524) - int(doe / 146096)) / 365)
  y = yoe + era * 400
  doy = doe - (365 * yoe + int(yoe / 4) - int(yoe / 100))
  mp = int((5 * doy + 2) / 153)
  d = doy - int((153 * mp + 2) / 5) + 1
  m = mp + (mp < 10 ? 3 : -9)
  y += (m <= 2)
  return sprintf("%04d%02d%02d", y, m, d)
}
function key_seconds(k,   d) {
  d = days_from_civil(substr(k, 1, 4) + 0, substr(k, 5, 2) + 0, substr(k, 7, 2) + 0)
  return d * 86400 + (substr(k, 9, 2) + 0) * 3600 + (substr(k, 11, 2) + 0) * 60 + (substr(k, 13, 2) + 0)
}
function shiftkey(k, h,   t, day, sod) {
  t = key_seconds(k) + h * 3600
  day = int(t / 86400); sod = t - day * 86400
  if (sod < 0) { sod += 86400; day -= 1 }
  return civil_from_days(day) sprintf("%02d%02d%02d", int(sod / 3600), int((sod % 3600) / 60), sod % 60)
}
function keyshow(k) {
  return substr(k,1,4) "-" substr(k,5,2) "-" substr(k,7,2) "-" substr(k,9,2) ":" substr(k,11,2) ":" substr(k,13,2)
}
function roundhour(delta,   h) {
  h = (delta >= 0) ? int((delta + 1800) / 3600) : -int((-delta + 1800) / 3600)
  return h
}
AWKHELPERS

# Pass 1: the modal whole-hour offset between an embedded device record and the
# log-line prefix on the same line (timezone of the collector vs UTC container).
# Sub-minute differences are ignored and a genuinely older re-printed record never
# counts, because its difference is not a whole number of hours.
cat > "$DETECT_PROG" <<'AWKDETECT'
{
  line = $0
  lkey = ""; rkey = ""
  if (match(line, /[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]/)) {
    lkey = defyear substr(line, RSTART, RLENGTH); gsub(/[-: ]/, "", lkey)
  }
  if (match(line, /[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/)) {
    rkey = substr(line, RSTART, RLENGTH); gsub(/[-:]/, "", rkey)
  } else rkey = ""
  if (lkey == "" || rkey == "") next
  delta = key_seconds(rkey) - key_seconds(lkey)
  h = roundhour(delta)
  if (h != 0 && (delta - h * 3600 <= 3) && (h * 3600 - delta <= 3)) hist[h]++
}
END {
  best = 0; besth = 0
  for (h in hist) if (hist[h] > best) { best = hist[h]; besth = h + 0 }
  # negate: the shift applied to a record must move it INTO the prefix's base
  print -besth
}
AWKDETECT
cat "$HELPERS" "$DETECT_PROG" > "$DETECT_PROG.tmp" && mv "$DETECT_PROG.tmp" "$DETECT_PROG"

cat > "$AWK_PROG" <<'AWK'

function add(l, r) { nf++; lab[nf] = l; re[nf] = r }
BEGIN {
  add("EZ9999 device exception (chipId/dieId)", "EZ9999")
  add("device core error (aivector / MTE-DDR out of range)", "fftsplus|[Aa]ivector|MTE instruction|DDR address")
  add("EE9999 acl/runtime error record", "EE9999")
  add("ERR00100/ERR01100 PTA/OPS acl api failed", "ERR00100|ERR01100")
  add("ERR02005 DIST internal error", "ERR02005")
  add("107030 rtMemcpy capture-mode", "107030")
  add("507011 aclrtSynchronizeEvent", "507011")
  add("507057 / SUSPECT REMOTE ERROR", "507057|SUSPECT REMOTE ERROR")
  add("EL0004 / 207001 allocation failure", "EL0004|207001")
  add("HCCL / hcclRet", "HCCL|hcclRet")
}

{
  line = $0
  lshow = ""; lkey = ""; rshow = ""; rkey = ""
  if (match(line, /[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]/)) {
    lshow = defyear "-" substr(line, RSTART, RLENGTH)
    lkey = lshow; gsub(/[-: ]/, "", lkey)
  }
  if (match(line, /[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/)) {
    rshow = substr(line, RSTART, RLENGTH)
    rkey = rshow; gsub(/[-:]/, "", rkey)
    if (ofs != 0) { rkey = shiftkey(rkey, ofs); rshow = keyshow(rkey) }
  } else if (lkey != "") { rshow = lshow; rkey = lkey }
  if (rkey == "") next
  if (lkey == "") { lshow = rshow; lkey = rkey }
  ld = substr(lkey, 1, 8); rd = substr(rkey, 1, 8)
  lday[ld]++; rday[rd]++
  if (lfirst == "" || lkey < lfirst) lfirst = lkey
  if (lkey > llast) llast = lkey
  if (rfirst == "" || rkey < rfirst) { rfirst = rkey; rfirstshow = rshow }
  if (rkey > rlast) { rlast = rkey; rlastshow = rshow }
  cur = line; gsub(/^[ \t]+|[ \t]+$/, "", cur)
  for (k = 1; k <= nf; k++) {
    if (line ~ re[k]) {
      cnt[k]++
      if (!(k in rmin) || rkey < rmin[k]) { rmin[k] = rkey; rminshow[k] = rshow; rsamp[k] = cur }
      if (!(k in rmax) || rkey > rmax[k]) { rmax[k] = rkey; rmaxshow[k] = rshow }
      if (!(k in lmin) || lkey < lmin[k]) { lmin[k] = lkey; lminshow[k] = lshow }
      if (!(k in lmax) || lkey > lmax[k]) { lmax[k] = lkey; lmaxshow[k] = lshow }
      if (!((k, rd) in rmindk) || rkey < rmindk[k, rd]) { rmindk[k, rd] = rkey; rminshowd[k, rd] = rshow }
      if (rd != ld) carry[k]++
      ch = substr(rkey, 1, 10)
      clus[ch]++
      if (!(ch in clusfirstkey) || rkey < clusfirstkey[ch]) { clusfirstkey[ch] = rkey; clusfirst[ch] = lab[k] }
      seen[k, rd] = 1
    }
  }
}
END {
  if (lfirst == "") { print "VERDICT: no timestamped error line found in the paste (no MM-DD HH:MM:SS prefix and no YYYY-MM-DD-HH:MM:SS record) - ask for the log itself."; exit }
  printf "log-line dates:"
  for (d in lday) { printf " %s-%s-%s(%d)", substr(d,1,4), substr(d,5,2), substr(d,7,2), lday[d]; if (d > lognewest) lognewest = d }
  newest = (forcenewest != "" ? forcenewest : lognewest)
  if (ofs != 0) printf "\nNOTE: embedded device records use a different clock than the log-line prefix; shifted by %+d h before comparing (a constant whole-hour lag is the host timezone offset, not a replayed record).\n", ofs
  printf "\nrecord dates:  "
  for (d in rday) printf " %s-%s-%s(%d)", substr(d,1,4), substr(d,5,2), substr(d,7,2), rday[d]
  printf "\n"
  printf "log-line span:  %s .. %s\nrecord span:    %s .. %s\n", lfirst, llast, rfirstshow, rlastshow
  print "families (table order; record span = embedded device/runtime error records):"
  for (k = 1; k <= nf; k++) {
    if (!(k in cnt)) continue
    printf "  %-52s n=%-3d record %s .. %s | line %s .. %s | older-record lines=%d\n", \
      lab[k], cnt[k], rminshow[k], rmaxshow[k], lminshow[k], lmaxshow[k], carry[k] + 0
    printf "     earliest: %s\n", substr(rsamp[k], 1, 190)
  }
  stale = ""; live = ""; livekey = ""
  for (k = 1; k <= nf; k++) {
    if (!(k in cnt)) continue
    inwin = 0
    for (d in rday) if (seen[k, d] && d >= newest) inwin = 1
    if (!inwin) { stale = stale (stale == "" ? "" : ", ") lab[k]; continue }
    if (livekey == "" || rmindk[k, newest] < livekey) { livekey = rmindk[k, newest]; live = lab[k]; liveshow = rminshowd[k, newest] }
  }
  printf "VERDICT: incident window %s-%s-%s (%s); device records span %s..%s\n", \
    substr(newest,1,4), substr(newest,5,2), substr(newest,7,2), \
    (forcenewest != "" ? "given by --since" : "newest log line in the paste"), rfirstshow, rlastshow
  if (stale != "") printf "  older records (dated before the incident window): %s\n", stale
  if (live != "") printf "  first error inside the incident window: %s at %s\n", live, liveshow
  else print "  no device-record block reaches the newest log-line day."
  print "  record clusters (day-hour, n=lines, first family):"
  for (c in clus) printf "    %s-%s-%s %s:00  n=%-3d  %s\n", substr(c,1,4), substr(c,5,2), substr(c,7,2), substr(c,9,2), clus[c], clusfirst[c]
  print "  --> the window above is anchored on the NEWEST log line, which is right only for a paste that covers the failure minute. A dump spanning hours or days holds several incidents: pick the cluster at the time the reporter named and re-run with `--since <YYYY-MM-DD>`. A block earlier than that window is not automatically stale: check whether it sits inside this same log's `init engine (...)` / graph-capture window (this incident recorded at init, reported late, rule 13); otherwise quote it as an earlier run's and ask for the plog window around the reported failure."
}
AWK
cat "$HELPERS" "$AWK_PROG" > "$AWK_PROG.tmp" && mv "$AWK_PROG.tmp" "$AWK_PROG"

STDIN_TMP=""
cleanup() { [ -n "$STDIN_TMP" ] && rm -f "$STDIN_TMP"; rm -f "$AWK_PROG"; }
trap cleanup EXIT

for f in "$@"; do
  if [ "$f" = "-" ]; then
    STDIN_TMP=$(mktemp "${TMPDIR:-/tmp}/plog-stdin.XXXXXX"); cat > "$STDIN_TMP"
    src="$STDIN_TMP"; label="<stdin>"
  else
    src="$f"; label="$f"
  fi
  if [ ! -r "$src" ]; then echo "cannot read $f" >&2; continue; fi
  years=$(grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}:[0-9]{2}:[0-9]{2}' "$src" 2>/dev/null | cut -c1-4 | sort -u)
  if [ -n "$years" ]; then
    defyear=$(printf '%s\n' "$years" | head -1)
    yearnote="long-form years: $(printf '%s ' $years)"
  else
    defyear=$(date -r "$src" +%Y 2>/dev/null || echo 0000)
    yearnote="no device-record timestamps; year for the MM-DD prefixes taken from the file date ($defyear)"
  fi
  echo "== $label  ($(wc -l < "$src") lines, $(wc -c < "$src") bytes, $yearnote)"
  ofs=$(awk -v defyear="$defyear" -f "$DETECT_PROG" "$src")
  [ -n "$ofs" ] || ofs=0
  awk -v defyear="$defyear" -v forcenewest="$SINCE_KEY" -v ofs="$ofs" -f "$AWK_PROG" "$src"
  echo
done
