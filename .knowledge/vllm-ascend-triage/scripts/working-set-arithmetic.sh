#!/usr/bin/env bash
# Turn a container-memory complaint into a mechanism class before any source read.
#
#   scripts/working-set-arithmetic.sh --growth 80GiB --days 7 --rps 20-40 \
#       [--tokens 500]
#   scripts/working-set-arithmetic.sh --samples <paste|->     # cAdvisor / prometheus
#
# Mode 1 (growth): bytes/request and bytes/decode-step from the reported slope,
#   and which mechanism class that slope is compatible with. Rates accept a range
#   ("20-40") or a single number; --tokens is the assumed output length per
#   request (default 500) and only affects the per-step bound.
# Mode 2 (samples): first-vs-last value per metric in a pasted prometheus /
#   cAdvisor block, so `container_memory_working_set_bytes` can be split into
#   `container_memory_rss` (anonymous, process retention) and
#   `container_memory_cache` (page cache, not a leak). Two samples are the
#   minimum; more is fine. Each label set (container) is split separately, so a
#   multi-container paste reports one block per container instead of pooling.
#
# Worked case (#16563, vllm-ascend 0.23.0RC1-era, DeepSeek-V4-Flash on one 800I
# A2, TP8+EP+MTP+--async-scheduling, 20-40 req/s): +80 GiB over 7 days =
# 8.1 MiB/min = ~3.5-7 KB per request, which is a per-request retained object,
# not a per-step host buffer.
set -u

if [ "$#" -eq 0 ]; then
  echo "usage: working-set-arithmetic.sh --growth 80GiB --days 7 --rps 20-40 [--tokens 500]" >&2
  echo "       working-set-arithmetic.sh --samples <paste.txt|->" >&2
  exit 2
fi

read -r -d '' PY <<'PYEOF'
import re, sys

args = sys.argv[1:]
growth = days = rps = tokens = samples = None
i = 0
while i < len(args):
    a = args[i]
    if a == '--growth':
        i += 1; growth = args[i]
    elif a == '--days':
        i += 1; days = float(args[i])
    elif a == '--rps':
        i += 1; rps = args[i]
    elif a == '--tokens':
        i += 1; tokens = float(args[i])
    elif a == '--samples':
        i += 1; samples = args[i]
    else:
        sys.stderr.write(f'unknown argument: {a}\n')
        sys.exit(2)
    i += 1

UNITS = {'b': 1.0, 'kb': 1024.0, 'kib': 1024.0, 'mb': 1024.0 ** 2,
         'mib': 1024.0 ** 2, 'gb': 1024.0 ** 3, 'gib': 1024.0 ** 3,
         'tb': 1024.0 ** 4, 'tib': 1024.0 ** 4}


def to_bytes(text):
    m = re.fullmatch(r'\s*([\d.]+)\s*([a-zA-Z]*)\s*', str(text))
    if not m:
        raise SystemExit(f'cannot read a size from {text!r}')
    unit = (m.group(2) or 'b').lower()
    if unit not in UNITS:
        raise SystemExit(f'unknown unit in {text!r} (use B/KiB/MiB/GiB/TiB)')
    return float(m.group(1)) * UNITS[unit]


def parse_rate(text):
    if text is None:
        return None, None
    m = re.fullmatch(r'\s*([\d.]+)\s*(?:-\s*([\d.]+))?\s*(?:rps|/s)?\s*', str(text))
    if not m:
        raise SystemExit(f'cannot read a request rate from {text!r} (use 20-40)')
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return lo, hi


def fmt_bytes(n):
    for unit, scale in (('GiB', 1024.0 ** 3), ('MiB', 1024.0 ** 2), ('KiB', 1024.0)):
        if abs(n) >= scale:
            return f'{n / scale:.2f} {unit}'
    return f'{n:.0f} B'


# ---------------------------------------------------------------- samples mode
METRICS = ('container_memory_working_set_bytes', 'container_memory_rss',
           'container_memory_cache')


def parse_samples(text):
    found = {}   # metric name -> label set -> [values in paste order]
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'([A-Za-z_:][A-Za-z0-9_:]*)\s*(?:\{([^}]*)\})?\s+(.*)$', line)
        if not m:
            continue
        name, labels, rest = m.group(1), (m.group(2) or '').strip(), m.group(3)
        if name not in METRICS:
            continue
        vm = re.search(r'([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)', rest)
        if vm:
            found.setdefault(name, {}).setdefault(labels, []).append(float(vm.group(1)))
    return found


def verdict(deltas):
    ws = deltas.get('container_memory_working_set_bytes')
    rss = deltas.get('container_memory_rss')
    cache = deltas.get('container_memory_cache')
    if ws is None or (rss is None and cache is None):
        print('  -> need working_set plus at least one of rss/cache at two timestamps.')
        return
    if rss is not None and cache is not None:
        if rss > 0 and cache <= max(rss * 0.25, 64 * 1024 * 1024):
            print('  -> anonymous RSS dominates the rise: process retention (leak or allocator')
            print('     growth). Continue with the per-PID RSS sample and the bytes/request')
            print('     arithmetic (mode 1); page cache is not the answer.')
        elif cache > rss:
            print('  -> page cache dominates the rise: not a leak. Look at where `vllm serve`')
            print('     stdout goes (a file inside the container is charged here), whether the')
            print('     checkpoint sits on the container writable layer, and any file the')
            print('     service rewrites; re-run with the log outside the container.')
        else:
            print('  -> rss and cache both move: split them per process with')
            print('     `ps -eo pid,rss,nlwp,cmd --sort=-rss | head -15` at the same two times,')
            print('     then run mode 1 on the RSS slope only.')
    elif rss is not None:
        print('  -> cache not in the paste: ask for container_memory_cache; on RSS alone the')
        print('     reading is process retention until page cache is excluded.')
    else:
        print('  -> RSS not in the paste: ask for container_memory_rss; on cache alone the')
        print('     reading is page cache until anonymous memory is excluded.')


def samples_mode(path):
    text = sys.stdin.read() if path == '-' else open(path, encoding='utf-8', errors='replace').read()
    found = parse_samples(text)
    ws_series = found.get('container_memory_working_set_bytes', {})
    if not ws_series:
        print('verdict: no container_memory_working_set_bytes sample found in the paste.')
        print('  ask for the cAdvisor/prometheus block for container_memory_working_set_bytes,')
        print('  container_memory_rss and container_memory_cache at two timestamps.')
        return
    keys = list(ws_series)
    if len(keys) > 1:
        print(f'{len(keys)} container series in the paste: splitting each separately.')
        print('  Pick the container the reporter watches; do not pool different containers.')
    for key in keys:
        if len(keys) > 1 or key:
            print()
            print(f'series: {key or "(no labels)"}')
        print('sample split (first vs last value in the paste):')
        deltas = {}
        for name in METRICS:
            series = found.get(name, {}).get(key)
            if not series:
                continue
            first, last = series[0], series[-1]
            deltas[name] = last - first
            note = '' if len(series) > 1 else '   (one sample only: needs two timestamps)'
            print(f'  {name:<40} {fmt_bytes(first):>12} -> {fmt_bytes(last):>12}'
                  f'   delta {fmt_bytes(last - first):>12}{note}')
        print('verdict:')
        verdict(deltas)

if samples:
    samples_mode(samples)
    sys.exit(0)

if not growth or days is None or not rps:
    sys.stderr.write('mode 1 needs --growth, --days and --rps (or use --samples)\n')
    sys.exit(2)

if days <= 0:
    sys.stderr.write('--days must be positive\n')
    sys.exit(2)

gb = to_bytes(growth)
minutes = days * 24 * 60
per_min = gb / minutes
per_s = gb / (minutes * 60)
lo, hi = parse_rate(rps)
if lo <= 0:
    sys.stderr.write('--rps must be positive\n')
    sys.exit(2)
tokens = tokens or 500.0

print(f'growth {fmt_bytes(gb)} over {days:g} d ({minutes:g} min) '
      f'= {fmt_bytes(per_min)}/min = {per_s / 1024:.1f} KiB/s')
print(f'assumed request rate {lo:g}-{hi:g} req/s, {tokens:g} output tokens/request')

b_req_hi = per_s / lo          # slower rate -> more bytes per request
b_req_lo = per_s / hi
print(f'  per request:            {fmt_bytes(b_req_lo)} - {fmt_bytes(b_req_hi)}')
steps_per_s_lo, steps_per_s_hi = lo * tokens, hi * tokens
print(f'  per decode step:        {per_s / steps_per_s_hi:.2f} B - {per_s / steps_per_s_lo:.2f} B'
      f'   (at {steps_per_s_lo:.0f}-{steps_per_s_hi:.0f} steps/s)')

print('verdict:')
if per_s / steps_per_s_hi < 16:
    print('  -> NOT a per-decode-step allocation: the whole growth is under 16 B per step at')
    print('     this rate, so a per-step host buffer (pinned staging, per-step metadata)')
    print('     prices as a plateau, not as this slope. Rank per-request retained objects')
    print('     (frontend/engine request lifecycle) first, then allocator retention and')
    print('     page cache; the single-axis ablations are the request-path features.')
elif b_req_hi >= 512:
    print('  -> both classes fit the slope: ~KB per request or tens of bytes per decode step.')
    print('     Separate them with a per-PID RSS sample during load, then the MTP-off and')
    print('     scheduler-mode ablations at constant request rate; the request-path')
    print('     features are rows 1-2, not conclusions.')
else:
    print('  -> the slope is too small per request and too large per step for a clean class:')
    print('     re-check the window, the request rate and the units (GB vs GiB) with the')
    print('     reporter before ranking anything.')

print('  Next asks: three cAdvisor series at two timestamps (working_set / rss / cache),')
print('  per-PID RSS at those times plus one idle sample, and whether the serve log is')
print('  written inside the container. Idle-time growth is not on the request path.')
PYEOF

uv run --no-project python -c "$PY" "$@"
