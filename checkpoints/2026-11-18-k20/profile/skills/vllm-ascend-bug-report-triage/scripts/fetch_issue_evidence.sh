#!/usr/bin/env bash
# Download bug-report attachments (startup logs, docker run scripts, collect_env
# dumps) into a scratch directory and print a cheap digest instead of the file.
#
# Why: GitHub issue *attachments* (https://github.com/user-attachments/files/<id>/<name>)
# are not readable through the GitHub REST/MCP tools, but plain `curl -sSL` can
# fetch them. The crash trace or the exact hang point usually lives in them, and
# the tail of a log is the cheapest decisive evidence for a stall.
#
# usage: fetch_issue_evidence.sh <outdir> <url> [url ...]
set -uo pipefail
outdir=${1:?usage: fetch_issue_evidence.sh <outdir> <url> [url ...]}
shift
mkdir -p "$outdir"
for url in "$@"; do
  name=$(basename "${url%%\?*}")
  case "$name" in
    *.txt|*.log|*.json|*.yaml|*.yml|*.sh|*.md) ;;
    *) printf 'SKIP %-36s (not a text attachment; read the URL as an image)\n' "$name"; continue ;;
  esac
  dest="$outdir/$name"
  code=$(curl -sSL --max-time 180 -o "$dest" -w '%{http_code}' "$url" || echo 000)
  if [ "$code" != "200" ]; then
    printf 'FAIL %s http=%s\n' "$url" "$code"
    rm -f "$dest"
    continue
  fi
  printf 'OK   %-36s %8s bytes %7s lines\n' "$name" "$(wc -c <"$dest")" "$(wc -l <"$dest")"
done
echo
echo '--- exception / stall markers (file:line, INFO/WARNING noise filtered) ---'
for f in "$outdir"/*; do
  [ -f "$f" ] || continue
  grep -nE 'Traceback|Error \(|Error:|Exception|AssertionError|RuntimeError|timed out|TimeoutError|OOM|Killed|core dump|segmentation' "$f" 2>/dev/null \
    | grep -vE '^[0-9]+:.*(INFO|WARNING) ' \
    | head -5 | sed "s|^|$(basename "$f"):|"
done
echo
echo '--- stall signatures ---'
for f in "$outdir"/*; do
  [ -f "$f" ] || continue
  grep -nE 'No available shared memory|Watchdog|stuck|hang|deadlock|HCCL.*(timeout|error)|NPU out of memory' "$f" 2>/dev/null \
    | head -5 | sed "s|^|$(basename "$f"):|" | cut -c1-240
done
echo
echo '--- last 8 lines of each file (where a stall stops producing output) ---'
for f in "$outdir"/*; do
  [ -f "$f" ] || continue
  echo "### $(basename "$f")"
  tail -n 8 "$f" | cut -c1-200
done
exit 0
