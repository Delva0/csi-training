#!/usr/bin/env bash
# Read a pasted HCCL rank table and say what HCCL will conclude from it.
#
#   scripts/ranktable-check.sh <rank_table.json|log.txt> [...]
#   scripts/ranktable-check.sh -            # read stdin
#
# Accepts a bare rank table or a log/paste containing one (the first {...}
# blob that mentions server_list/server_count is used). Reports the effective
# server_count, per-server server_id / device_id / device_ip, the duplicate and
# contiguity problems, and the decision the rank table forces:
#
#   >1 distinct server_id  -> HCCL treats the peers as different servers, so
#                             every server entry must carry device_ip and the
#                             data path is the NIC (RoCE/RDMA), not HCCS.
#   ==1 distinct server_id -> single-server view: HCCL may select HCCS between
#                             the local cards; device_ip is not required.
#
# This decides most "HIXL/ADCCL Connect failed" and "P/D KV transfer will not
# start without hostNetwork" reports before any source reading.
set -u

# `-h` / `--help` is not a filename: treating it as one reached the python reader and
# died with `FileNotFoundError: '-'` on a recorded call (issue #2379's replay spent a
# turn on that traceback).
case "${1-}" in
  -h|--help)
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
esac

if [ "$#" -eq 0 ]; then
  echo "usage: ranktable-check.sh <rank_table.json|log.txt> [...] | -" >&2
  exit 2
fi

uv run --no-project python - "$@" <<'PY'
import json, sys

def blobs(text):
    """Yield candidate JSON objects found in free text."""
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]

def load(path):
    if path == '-':
        return sys.stdin.read()
    with open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()

def as_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None

def report(source, text):
    table = None
    for chunk in blobs(text):
        try:
            doc = json.loads(chunk)
        except Exception:
            continue
        if isinstance(doc, dict) and ('server_list' in doc or 'server_count' in doc):
            table = doc
    print(f'=== {source}')
    if table is None:
        print('  no rank table found (looked for an object with server_list/server_count)')
        print('  paste the table json, or grep the plog: grep -nE '
              '"server_count|server_id|ranktable|RanktableCheck|device_ip" ~/ascend/log/debug/plog/plog-*.log')
        return

    servers = table.get('server_list') or []
    declared = as_int(table.get('server_count'))
    ids, all_ranks, problems = [], [], []
    no_ip_servers = []
    for idx, srv in enumerate(servers):
        sid = str(srv.get('server_id', srv.get('serverId', '<missing>')))
        ids.append(sid)
        devs = srv.get('device') or srv.get('devices') or []
        dev_ids, rank_ids, with_ip = [], [], 0
        for dev in devs:
            dev_ids.append(str(dev.get('device_id', dev.get('deviceId', '?'))))
            rid = as_int(dev.get('rank_id', dev.get('rankId')))
            if rid is not None:
                rank_ids.append(rid)
                all_ranks.append(rid)
            ip = dev.get('device_ip', dev.get('deviceIp'))
            if ip:
                with_ip += 1
        if devs and with_ip == 0:
            no_ip_servers.append(sid)
        print(f'  server[{idx}] server_id={sid} devices={len(devs)} '
              f'device_id={dev_ids} rank_id={rank_ids} device_ip={"yes" if with_ip else "NO"}')
        dup = {d for d in dev_ids if dev_ids.count(d) > 1}
        if dup:
            problems.append(f'server[{idx}] repeats device_id {sorted(dup)}; within one server '
                            'ranks are distinguished by device_id, so two processes on the same '
                            'card collide')
        if rank_ids and len(set(rank_ids)) != len(rank_ids):
            problems.append(f'server[{idx}] repeats rank_id {rank_ids}')

    distinct = len(dict.fromkeys(ids))
    print(f'  declared server_count={declared}  parsed server_list={len(servers)}  distinct server_id={distinct}')
    if declared is not None and declared != len(servers):
        problems.append(f'declared server_count={declared} but {len(servers)} entries are listed')
    if all_ranks and sorted(all_ranks) != list(range(len(all_ranks))):
        problems.append(f'rank_id set {sorted(all_ranks)} is not 0..N-1')

    # The declared count decides HCCL's view even when the entries happen to share
    # one server_id: issue #2399 failed with server_count="2" and two identical
    # server_id values, while the reference table declared "1".
    multi = (declared is not None and declared > 1) or distinct > 1
    if multi:
        print('  VERDICT: multi-server in HCCL\'s view -> device_ip is mandatory and the peer data')
        print('           path is the NIC (RoCE/RDMA). With DOWN/absent RoCE links the connect')
        print('           fails regardless of the Python layer. Make the peers report one')
        print('           server_id and server_count=1 (host network / shared hostname) before')
        print('           chasing connectors.')
        if distinct == 1 and declared and declared > 1:
            problems.append(f'declared server_count={declared} with {len(servers)} entries sharing '
                            'one server_id: the declaration alone keeps HCCL in the multi-server '
                            'view (issue #2399)')
        if no_ip_servers:
            problems.append('server entries without device_ip: ' + ', '.join(no_ip_servers))
    elif distinct == 1:
        print('  VERDICT: single-server in HCCL\'s view -> HCCS between local cards is selectable')
        print('           and device_ip is not required. If it still fails, the remaining')
        print("           levers are HCCL_INTRA_ROCE_ENABLE=1 (forces RoCE intra-node),")
        print('           device_id collisions above, and the transport port bands.')
    else:
        print('  VERDICT: empty server_list; nothing to conclude')
    for p in problems:
        print(f'  PROBLEM: {p}')

for path in sys.argv[1:]:
    report(path, load(path))
PY
