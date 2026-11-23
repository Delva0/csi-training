#!/usr/bin/env bash
# Sum the per-process resident memory in a pasted `top`, `ps` or /proc/meminfo
# excerpt and say whether the HOST (not the NPU) is the resource that ran out.
#
#   scripts/host-ram-check.sh <paste.txt> [...]     # or  -  for stdin
#   scripts/host-ram-check.sh --total-gib 2014 -     # force the host RAM size
#
# Reads the shapes a reporter actually pastes:
#   * `top` header  `MiB Mem : 2062270.+total, 459119.2 free, 1600621.+used`
#   * `free`   row  `Mem:  2014  1560  452 ...`  (unit inferred, or forced)
#   * `MemTotal: N kB` from /proc/meminfo
#   * `top` process rows  `6482 root 20 0 8326.4g 92.5g 703164 R 107.3 4.6 0:54.78 python3`
#   * `ps`  rows  `6482 96864896 python3 ...`   (RSS in KiB)
#
# Verdict bands come from the #3902 reporter paste (v0.11.0rc0-a3, Qwen3-VL-235B,
# `vllm serve ... -tp 16` on one 910C): 16 python3 ranks at ~88-100 GiB RES each
# on a 2 TB host with swap 0 = ~73 % of RAM held by the ranks before any request.
# That paste is the worked case: run this script on it and it prints 1470 GiB /
# 2014 GiB (73 %).
set -u

if [ "$#" -eq 0 ]; then
  echo "usage: host-ram-check.sh [--total-gib N] <paste.txt> [...] | -" >&2
  exit 2
fi

read -r -d '' PY <<'PYEOF'
import re, sys

FORCED = None
paths = []
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == '--total-gib':
        i += 1
        FORCED = float(args[i])
    else:
        paths.append(args[i])
    i += 1


def load(p):
    if p == '-':
        return sys.stdin.read()
    with open(p, encoding='utf-8', errors='replace') as fh:
        return fh.read()


UNIT = {'k': 1 / 1024 ** 2, 'm': 1 / 1024, 'g': 1.0, 't': 1024.0, '': None}


def to_gib(tok):
    m = re.fullmatch(r'([\d.]+)([kmgtKMGT]?)', tok)
    if not m:
        return None
    f = UNIT[m.group(2).lower()]
    return float(m.group(1)) * f if f is not None else None


TOP_HDR = re.compile(r'(MiB|GiB|KiB|TiB)\s+Mem\s*:\s*([\d.]+)\+?\s*total', re.I)
MEMINFO = re.compile(r'MemTotal:\s+(\d+)\s*kB')
FREE_HDR = re.compile(r'^\s*Mem:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
TOP_ROW = re.compile(
    r'^\s*(\d+)\s+\S+\s+-?\d+\s+-?\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+([RDSZT])\s+'
    r'([\d.]+)\s+([\d.]+)\s+\S+\s+(.*)$')
PS_ROW = re.compile(r'^\s*(\d+)\s+(\d+)\s+([A-Za-z/.].*)$')

total = None
total_src = ''
free_row = None
procs = []          # (pid, res_gib, virt_tok, pct_mem, cmd)

for p in paths:
    for line in load(p).splitlines():
        if total is None:
            m = TOP_HDR.search(line)
            if m:
                total = to_gib(m.group(2) + m.group(1)[0])
                total_src = 'top header'
                continue
            m = MEMINFO.search(line)
            if m:
                total = int(m.group(1)) / 1024 ** 2
                total_src = '/proc/meminfo'
                continue
            m = FREE_HDR.match(line)
            if m and free_row is None:
                free_row = m
                continue
        m = TOP_ROW.match(line)
        if m:
            res = to_gib(m.group(3))
            if res is not None:
                procs.append((m.group(1), res, m.group(2), m.group(7), m.group(8).strip()))
            continue
        m = PS_ROW.match(line)
        if m:
            procs.append((m.group(1), int(m.group(2)) / 1024 ** 2, '-', '-', m.group(3).strip()))

if total is None and free_row is not None:
    vals = [float(free_row.group(k)) for k in (1, 2, 3)]
    total = vals[0]
    total_src = 'free row (unit inferred: >64000 => MiB, else GiB)'
    if total > 64000:
        total /= 1024
if FORCED is not None:
    total, total_src = FORCED, '--total-gib'

ranks = [p for p in procs if re.search(r'python|vllm|EngineCore|Worker', p[4])]
rank_sum = sum(p[1] for p in ranks)
all_sum = sum(p[1] for p in procs)

print(f'host RAM total      : {total:.0f} GiB' if total else 'host RAM total      : not in this paste')
if total:
    print(f'  source            : {total_src}')
print(f'process rows parsed : {len(procs)}  (python/vllm-looking: {len(ranks)})')
if ranks:
    print(f'worker RSS sum      : {rank_sum:.0f} GiB  '
          f'(mean {rank_sum / len(ranks):.0f} GiB over {len(ranks)} ranks)')
print(f'all parsed RSS sum  : {all_sum:.0f} GiB')
big = sorted(ranks, key=lambda r: -r[1])[:3]
for pid, res, virt, pct, cmd in big:
    print(f'  biggest           : pid {pid} RES {res:.0f} GiB  VIRT {virt}  %MEM {pct}  {cmd[:60]}')
print()
if not total:
    print('verdict: no host RAM total in the paste — ask for `free -g` and the `MiB Mem` line of `top`')
elif not ranks:
    print('verdict: no worker processes in the paste — ask for `ps -eo pid,rss,cmd --sort=-rss | head -25`')
else:
    ratio = rank_sum / total
    print(f'verdict: worker processes hold {ratio * 100:.0f} % of host RAM '
          f'({rank_sum:.0f} of {total:.0f} GiB)')
    if ratio >= 0.60:
        print('  -> host memory is a first-class cause of "the machine crashed": all ranks + the')
        print('     OpenAI frontend must fit in this RAM. Fewer ranks per host or a smaller')
        print('     --max-model-len is the cheap test; check `dmesg -T | grep -iE \'oom|kill\'`')
        print('     and `free -g` for swap (0 here means the kernel OOM-kills or thrashes).')
    elif ratio >= 0.35:
        print('  -> large but not conclusive on its own; compare against a second sample and')
        print('     against the same model at a smaller TP before ranking it.')
    else:
        print('  -> host RAM does not explain this paste; look at the NPU-side lanes.')
    print('  VIRT (e.g. 8326.4g) is torch_npu\'s per-process virtual reservation, not usage:')
    print('  quote RES, never VIRT. %CPU on a first `top` sample is a since-boot average;')
    print('  quote the load average and the run queue instead.')
PYEOF

uv run --no-project python -c "$PY" "$@"
