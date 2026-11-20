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

  echo "  evidence:"
  [ -n "$rpc" ] && echo "    - watchdog raise: $rpc"
  [ "$rd" -gt 0 ] && echo "    - shm_broadcast acquire_read frame(s): $rd  (EngineCore waiting for a worker reply)"
  [ "$wr" -gt 0 ] && echo "    - shm_broadcast acquire_write frame(s): $wr  (reply could not be written: readers not consuming)"
  [ "$sock" -gt 0 ] && echo "    - socket recv/poll frame(s): $sock  (remote reader / overflow path)"
  [ "${ring:-0}" -gt 0 ] && echo "    - ringbuffer starvation line(s): $ring  (same mailbox, DEBUG level)"
  [ "$chain" -gt 0 ] && echo "    - 'direct cause of the following exception': $chain  (RuntimeError/TimeoutError wrap)"
  [ -n "$bound" ] && echo "    - configured bound: $bound"
  [ -n "$hccl" ] && printf '    - collective-side timeout text: %s\n' "$hccl"

  echo "  verdict:"
  if [ -n "$rpc" ] || [ "$rd" -gt 0 ] || [ "$wr" -gt 0 ] || [ "${ring:-0}" -gt 0 ]; then
    echo "    vLLM RPC/ringbuffer watchdog, NOT an HCCL or NIC timeout."
    if [ "$rd" -gt 0 ] || [ -n "$rpc" ]; then
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
    echo "    next: per-rank worker log around the last successful step, plog for the"
    echo "    same window, npu-smi info taken while the process is still hung, then a"
    echo "    re-run with --enforce-eager + ASCEND_LAUNCH_BLOCKING=1 to move attribution"
    echo "    onto the failing op. Raising the bound only buys observation time."
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
