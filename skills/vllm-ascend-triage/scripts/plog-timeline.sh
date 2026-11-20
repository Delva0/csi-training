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
trap 'rm -f "$AWK_PROG"' EXIT
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
      seen[k, rd] = 1
    }
  }
}
END {
  if (lfirst == "") { print "VERDICT: no timestamped error line found in the paste (no MM-DD HH:MM:SS prefix and no YYYY-MM-DD-HH:MM:SS record) - ask for the log itself."; exit }
  printf "log-line dates:"
  for (d in lday) { printf " %s-%s-%s(%d)", substr(d,1,4), substr(d,5,2), substr(d,7,2), lday[d]; if (d > lognewest) lognewest = d }
  newest = (forcenewest != "" ? forcenewest : lognewest)
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
  else print "  no device-record block falls inside the incident window: the pasted window does not contain the fault."
  if (stale != "" || live == "") print "  --> before quoting an older block as history, check whether it sits inside this same log's `init engine (...)` / graph-capture window: then it is this incident recorded at init and reported late (rule 13). Otherwise quote it as the previous run's and ask for the plog window around the reported failure."
}
AWK

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
  awk -v defyear="$defyear" -v forcenewest="$SINCE_KEY" -f "$AWK_PROG" "$src"
  echo
done
