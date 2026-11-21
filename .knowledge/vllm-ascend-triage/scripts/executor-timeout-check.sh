#!/usr/bin/env bash
# executor-timeout-check.sh -- decide whether a pasted "communication timeout" is
# vLLM's own RPC watchdog (a hung rank) or a real collective/transport timeout.
#
# usage: bash executor-timeout-check.sh <log-file> [more files ...]
#        ... - < log.txt
#
# vLLM v1's MultiprocExecutor runs every worker call through a shared-memory
# message queue with a deadline; `execute_model` uses
# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (300 s at vllm v0.9.2, envs.py:134). When the
# reply does not arrive, the shm_broadcast acquire_read/recv spin raises a bare
# TimeoutError and multiproc_executor re-raises it with the text
# "RPC call to <method> timed out." -- no rank, no op, no error code. So the
# presence of that text is evidence of a stalled worker, not of a bad link:
# the RCA needs the per-rank log / plog of the rank that stopped replying.
#
# One lane inverts that: a launch carrying a dump/profiler feature (msprobe
# precision dump via --additional-config dump_config_path/dump_config,
# --profiler-config) makes every step slow on purpose, so the same text means the
# deadline is the finding and raising VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS is the
# fix (vllm-ascend docs/tests raise it to 30000/3600000). This script reports that
# lane when it sees such a feature instead of blaming a dead worker.
set -u
LC_ALL=C

if [ "$#" -eq 0 ]; then
  echo "usage: bash $0 <log-file> [more files ...]  |  -" >&2
  exit 2
fi

STDIN_TMP=""
cleanup() { [ -n "$STDIN_TMP" ] && rm -f "$STDIN_TMP"; }
trap cleanup EXIT

count() { grep -cE "$1" "$2" 2>/dev/null || true; }

for f in "$@"; do
  if [ "$f" = "-" ]; then
    STDIN_TMP=$(mktemp "${TMPDIR:-/tmp}/exec-timeout.XXXXXX"); cat > "$STDIN_TMP"
    src="$STDIN_TMP"; label="<stdin>"
  else
    src="$f"; label="$f"
  fi
  if [ ! -r "$src" ]; then echo "cannot read $f" >&2; continue; fi
  echo "== $label  ($(wc -l < "$src") lines, $(wc -c < "$src") bytes)"

  rpc=$(grep -oE 'RPC call to [A-Za-z_]+ timed out' "$src" 2>/dev/null | sort -u)
  rd=$(count 'acquire_read' "$src"); wr=$(count 'acquire_write' "$src")
  sock=$(count 'socket.*poll|MessageQueue\.recv' "$src")
  ring=$(grep -c 'No available shared memory broadcast block found' "$src" 2>/dev/null || true)
  bound=$(grep -oE 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS[^0-9]*[0-9]+' "$src" 2>/dev/null | sort -u)
  chain=$(count 'direct cause of the following exception' "$src")
  hccl=$(grep -oE 'hccl[A-Za-z]*[^ ]* ?(timeout|Timeout)|HCCL[^ ]* (timeout|Timeout)|EI[0-9]{4}|Execute timeout' "$src" 2>/dev/null | sort -u | head -5)
  # Deliberately-slow lane: a dump/profiler feature in the launch makes the step slow
  # on purpose, so the watchdog is the finding and not a stalled worker.
  slow=$(grep -oE 'dump_config_path|dump_config|PrecisionDebugger|msprobe|profiler-config|collect_detailed_traces' "$src" 2>/dev/null | sort -u | head -4)

  # Graph-mode evidence: a mode that printed its banner and finished capture is not
  # the capture lane nor a dropped setting, whatever the watchdog says after it.
  gmode=$(grep -oE '(FULL_DECODE_ONLY|FULL_AND_PIECEWISE|PIECEWISE) compilation enabled on NPU' "$src" 2>/dev/null | sort -u | head -1)
  gcap=$(grep -oE 'Starting to capture ACL graphs for cases: [^,]*, mode: [A-Z]+' "$src" 2>/dev/null | sort -u | head -1)
  gdone=$(count 'Graph capturing finished in [0-9]+ secs' "$src")
  gdump=$(grep -oE '"cudagraph_mode":[ ]?\[[0-9]+,[ ]?[0-9]+\]|"cudagraph_mode":[ ]?"?[A-Z_]+"?' "$src" 2>/dev/null | sort -u | head -2)
  gnew=$(count 'scheduled_new_reqs=\[NewRequestData' "$src")
  gtok=$(grep -oE 'total_num_scheduled_tokens=[0-9]+' "$src" 2>/dev/null | sort -u | head -2)
  geager=$(grep -oE 'enforce_eager=(True|False)' "$src" 2>/dev/null | sort -u | head -1)
  gname=""
  case "$gdump" in
    *'[2, 0]'*|*'[2,0]'*) gname='FULL_DECODE_ONLY = (FULL, NONE)' ;;
    *'[2, 1]'*|*'[2,1]'*) gname='FULL_AND_PIECEWISE = (FULL, PIECEWISE)' ;;
    *FULL_DECODE_ONLY*) gname='FULL_DECODE_ONLY' ;;
  esac

  echo "  evidence:"
  [ -n "$rpc" ] && echo "    - watchdog raise: $rpc"
  [ "$rd" -gt 0 ] && echo "    - shm_broadcast acquire_read frame(s): $rd  (EngineCore waiting for a worker reply)"
  [ "$wr" -gt 0 ] && echo "    - shm_broadcast acquire_write frame(s): $wr  (reply could not be written: readers not consuming)"
  [ "$sock" -gt 0 ] && echo "    - socket recv/poll frame(s): $sock  (remote reader / overflow path)"
  [ "${ring:-0}" -gt 0 ] && echo "    - ringbuffer starvation line(s): $ring  (same mailbox, DEBUG level)"
  [ "$chain" -gt 0 ] && echo "    - 'direct cause of the following exception': $chain  (RuntimeError/TimeoutError wrap)"
  [ -n "$bound" ] && echo "    - configured bound: $bound"
  [ -n "$slow" ] && printf '    - debug/profiling feature in the launch: %s  (the step may be slow on purpose)\n' "$(echo $slow | tr '\n' ' ')"
  [ -n "$hccl" ] && printf '    - collective-side timeout text: %s\n' "$hccl"
  [ -n "$gmode" ] && echo "    - graph mode banner: $gmode"
  [ -n "$gcap" ] && echo "    - capture start line: $gcap"
  [ "${gdone:-0}" -gt 0 ] && echo "    - 'Graph capturing finished' rank line(s): $gdone  (capture completed)"
  [ -n "$gdump" ] && echo "    - dumped cudagraph_mode: $gdump${gname:+  ($gname, vllm/config/compilation.py:39-43)}"
  [ -n "$geager" ] && echo "    - dumped config: $geager"
  [ "${gnew:-0}" -gt 0 ] && echo "    - dump carries scheduled_new_reqs: $gnew  (the stalled step is a FIRST/new-request step)"
  [ -n "$gtok" ] && printf '    - hung-step token count: %s\n' "$gtok"

  echo "  verdict:"
  if [ -n "$rpc" ] || [ "$rd" -gt 0 ] || [ "$wr" -gt 0 ] || [ "${ring:-0}" -gt 0 ]; then
    echo "    vLLM RPC/ringbuffer watchdog, NOT an HCCL or NIC timeout."
    if [ -n "$slow" ]; then
      : # a dump/profiler launch owns the direction: slow-step verdict below
    elif [ "$rd" -gt 0 ] || [ -n "$rpc" ]; then
      echo "    The message names no rank, op or error code: the fault is in the worker"
      echo "    that stopped replying, and this paste does not contain it."
    elif [ "$wr" -gt 0 ]; then
      echo "    Direction is the write side: a reply could not be placed because the"
      echo "    readers are not consuming. The stalled side is the reader set, not the"
      echo "    writer in the frame."
    else
      echo "    The message names no rank, op or error code: the fault is in the worker"
      echo "    that stopped replying, and this paste does not contain it."
    fi
    if [ -n "$gmode" ] || [ "${gdone:-0}" -gt 0 ]; then
      echo "    graph mode engaged AND its capture finished, so neither the capture lane"
      echo "    nor a dropped setting explains this: the stall is on the mode's request path."
      echo "    Settle coverage BEFORE any source read - which models the pinned tests/e2e"
      echo "    validate for that mode, and whether .github/workflows actually runs that test"
      echo "    file. An rc-only hang is a version verdict plus a fallback mode, because the"
      echo "    fix is not in the pinned tree."
      echo "    see references/hccl-and-hardware/aclgraph-full-decode-runtime-hang.md"
    fi
    if [ -n "$slow" ]; then
      echo "    A debug/profiling feature is configured, so read this as a DELIBERATELY"
      echo "    SLOW step, not a stalled worker: the ranks were answering and the bound"
      echo "    is the finding. Raising VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS before the"
      echo "    launch IS the fix for this lane (in-tree precedent: Kimi-K2.5.md:356"
      echo "    =30000, nightly MTPX-DeepSeek-R1-0528-W8A8.yaml:10 =3600000). Three"
      echo "    caveats: it only applies with TP>1 (vllm/envs.py:1424); the 10 s"
      echo "    check_health RPC (multiproc_executor.py:470-471) is NOT covered by it;"
      echo "    and 0 or empty means instant timeout, not disabled. Shrink the dump"
      echo "    (level L0, a step range, async_dump, task=statistics) as well."
      echo "    see references/hccl-and-hardware/rpc-watchdog-timeouts.md"
    else
      echo "    next: per-rank worker log around the last successful step, plog for the"
      echo "    same window, npu-smi info taken while the process is still hung, then a"
      echo "    re-run with --enforce-eager + ASCEND_LAUNCH_BLOCKING=1 to move attribution"
      echo "    onto the failing op. Raising the bound only buys observation time."
    fi
  elif [ -n "$hccl" ]; then
    echo "    collective/transport-side timeout text present -- this is a real HCCL"
    echo "    lane-C case; use references/hccl-and-hardware/collective-error-chain.md"
    echo "    and run plog-timeline.sh on the plog before naming a device."
  else
    echo "    no watchdog and no collective-timeout text found: the paste does not"
    echo "    contain the timeout. Ask for the window around the reported failure."
  fi
  echo
done
