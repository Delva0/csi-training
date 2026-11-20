#!/usr/bin/env bash
# Read a Mooncake store config and/or a serve log and say whether the KV-pool
# store segment size is the cause of an ADXL channel failure.
#
#   scripts/kvpool-segment-check.sh <mooncake.json|serve-log.txt> [...]
#   scripts/kvpool-segment-check.sh -            # read stdin
#
# Two input shapes are recognised per file:
#   * a Mooncake store config (json with global_segment_size / metadata_server /
#     protocol) -> the per-rank segment size the workers register;
#   * a serve/worker log -> the "TransferExecutor register mem addr:..., length:N,
#     mem type:host" lines, which carry the same number when the json is absent,
#     plus the npu-smi (HDK) version line and the pool markers.
#
# Verdicts follow the worked case #13439: HDK 25.2.3 + CANN 9.0.1 on 910B2
# requires global_segment_size < 20 GB per rank (40 GB -> every ADXL channel
# fails at HcclCommPrepare with ret 0x13 / status 503900 and the decode node
# emits garbled output; 2 GB works). The pinned default is 1.0 GiB per rank
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/
# mooncake_backend.py:23-24, doc "needs to be aligned to 1GB").
set -u

if [ "$#" -eq 0 ]; then
  echo "usage: kvpool-segment-check.sh <mooncake.json|serve-log.txt> [...] | -" >&2
  exit 2
fi

read -r -d '' PYSETUP <<'PY'
import json, re, sys

GIB = 1024 ** 3
CEILING = 20 * GIB          # #13439 reporter-confirmed working ceiling, HDK 25.2.3
DEFAULT_SEG = 1 * GIB       # mooncake_backend.py:23 DEFAULT_GLOBAL_SEGMENT_SIZE

def load(path):
    if path == '-':
        return sys.stdin.read()
    with open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()

def blobs(text):
    start = depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]

CONFIG_KEYS = ('global_segment_size', 'metadata_server', 'master_server_address',
               'local_buffer_size', 'prefer_alloc_in_same_node')
UNIT = {'gb': GIB, 'mb': 1024 ** 2, 'kb': 1024, 'b': 1}

def parse_size(value):
    """Mirror vllm-ascend's _parse_global_segment_size; None when it would raise."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    m = re.match(r'^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$', value.strip().lower())
    if not m:
        return None
    return int(float(m.group(1)) * UNIT[m.group(2) or 'b'])

def fmt(n):
    if n is None:
        return 'unparsed'
    return '{} B  ({:.2f} GiB)'.format(n, n / GIB)

def verdict(seg, where):
    print('  VERDICT ({}):'.format(where))
    if seg is None:
        print('    no store segment size found. Ask for the file behind'
              ' $MOONCAKE_CONFIG_PATH (or `echo $MOONCAKE_GLOBAL_SEGMENT_SIZE`).')
        return
    if seg == 0:
        print('    0 bytes = this client contributes no store memory'
              ' (contribute_memory=False / scheduler client). Zeros here are normal and')
        print('    are NOT the fault; look at a Worker_* process instead.')
        return
    if seg >= CEILING:
        print('    {:.2f} GiB per rank is AT or ABOVE the #13439 ceiling'
              ' (HDK 25.2.3 requires global_segment_size < 20 GB).'.format(seg / GIB))
        print('    FIX: set global_segment_size to "2GB" (or export'
              ' MOONCAKE_GLOBAL_SEGMENT_SIZE=2147483648), keep every other flag, retry.')
        print('    Reporter-confirmed resolution of #13439: a segment this size is registered')
        print('    into the process ADXL engine, so the p2p connector fails too with')
        print('    503900 / HcclCommPrepare ret 0x13 and the decode node emits garbled output.')
    elif seg >= 8 * GIB:
        print('    {:.2f} GiB per rank: inside the reporter-confirmed working band'
              ' (< 20 GB) but far above the {:.0f} GiB pinned default.'.format(seg / GIB, DEFAULT_SEG / GIB))
        print('    Check the documented budget: 1 GB alignment, and on A3 fabric mem')
        print('    fabric_memory.max_capacity >= segment + MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES')
        print('    (kv_pool.md:1010-1042). If the channel still fails, take the rank-table route.')
    else:
        print('    {:.2f} GiB per rank: inside the documented range'
              ' (pinned default 1 GiB). The store segment size is unlikely to be the'.format(seg / GIB))
        print('    cause - use the rank-table / plog route'
              ' (references/pd-disaggregation/transport-topology.md).')
    if seg % GIB:
        print('    ALIGNMENT: not a 1 GiB multiple. The docs require 1 GB alignment')
        print('    (kv_pool.md:113, :1010-1021); misaligned fabric-mem allocations fail with')
        print('    adxl MallocMem / aclrtMapMem Invalid_Argument.')

def report_config(source, cfg):
    print('=== {}  (Mooncake store config)'.format(source))
    seg = parse_size(cfg.get('global_segment_size', DEFAULT_SEG))
    buf = parse_size(cfg.get('local_buffer_size', DEFAULT_SEG))
    print('  global_segment_size: {!r} -> {}'.format(
        cfg.get('global_segment_size', '<absent -> default 1 GiB>'), fmt(seg)))
    print('  local_buffer_size  : {!r} -> {}'.format(
        cfg.get('local_buffer_size', '<absent -> default 1 GiB>'), fmt(buf)))
    for key in ('metadata_server', 'protocol', 'device_name', 'master_server_address',
                'prefer_alloc_in_same_node', 'enable_ssd_offload', 'ssd_offload_path'):
        if key in cfg:
            print('  {}: {!r}'.format(key, cfg[key]))
    verdict(seg, 'config')

REG_MEM = re.compile(r'register mem addr:(\S+?),?\s*length:(\d+),\s*mem type:(\w+)'
                     r'(?:,\s*engine index:(\d+))?')
NPU_SMI = re.compile(r'npu-smi\s+([\d.]+)\s+Version:\s*([\d.]+)')
ENV_SEG = re.compile(r'MOONCAKE_GLOBAL_SEGMENT_SIZE\s*=\s*(\S+)')

def report_log(source, text):
    print('=== {}  (log)'.format(source))
    versions = sorted({m.group(2) for m in NPU_SMI.finditer(text)})
    if versions:
        print('  npu-smi / HDK version in the paste: {}  (the 20 GB ceiling above is'
              ' reported for 25.2.3)'.format(', '.join(versions)))
    envs = sorted({m.group(1) for m in ENV_SEG.finditer(text)})
    if envs:
        print('  MOONCAKE_GLOBAL_SEGMENT_SIZE seen: {}  (takes precedence over the'
              ' json)'.format(', '.join(envs)))
    host, device = [], []
    for m in REG_MEM.finditer(text):
        size = int(m.group(2))
        (host if m.group(3).lower() == 'host' else device).append((size, m.group(1), m.group(4)))
    markers = [k for k in ('pool_worker.py', 'pool_scheduler.py', 'mooncake_hybrid_connector.py',
                           'AscendStoreConnector', 'Global segment size', 'HcclCommPrepare',
                           '503900', 'MOONCAKE_CONFIG_PATH') if k in text]
    if not host and not device:
        print('  no "TransferExecutor register mem" lines')
        print('  pool markers present: ' + (', '.join(markers) if markers else 'none'))
        print('  Ask for the D-node worker log covering the failure minute and for the store'
              ' config behind $MOONCAKE_CONFIG_PATH.')
        return
    print('  registered memory regions: {} host, {} device'.format(len(host), len(device)))
    for size, addr, idx in sorted(set(host), reverse=True)[:5]:
        print('    host   {}  addr={} engine={}'.format(fmt(size), addr, idx or '?'))
    for size, addr, idx in sorted(set(device), reverse=True)[:3]:
        print('    device {}  addr={} engine={}'.format(fmt(size), addr, idx or '?'))
    if markers:
        print('  pool / channel markers: ' + ', '.join(markers))
    biggest = max((s for s, _, _ in host), default=None)
    if biggest is None:
        print('  no host registration: this process registered only device memory, so the')
        print('  store segment is elsewhere (or the pool client is scheduler-only).')
        return
    print('  largest host registration: {}  <- the store segment is aclrtMallocHost-ed'
          ' and then registered like this'.format(fmt(biggest)))
    verdict(biggest, 'largest host registration in the log')

for path in sys.argv[1:]:
    text = load(path)
    cfg = None
    for chunk in blobs(text):
        try:
            doc = json.loads(chunk)
        except Exception:
            continue
        if isinstance(doc, dict) and any(k in doc for k in CONFIG_KEYS):
            cfg = doc
    if cfg is not None:
        report_config(path, cfg)
    else:
        report_log(path, text)
PY

uv run --no-project python -c "$PYSETUP" "$@"
