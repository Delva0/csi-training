#!/usr/bin/env bash
# Decode the codes around an AscendStore / Mooncake "store put failed" report and
# say which side owns it.
#
#   scripts/store-put-code-check.sh <paste|serve-log.txt> [...]   # or '-' for stdin
#
# The input is whatever the reporter pasted: the prefill worker log with
#   "Failed to invoke HcclBatchPut, hccl_result = 4"
#   "Failed to put 8 keys out of 8. error_codes=[-1]. Check memory and store capacity."
# plus, often, the mooncake_master metrics block (Batch Requests: PutStart /
# PutEnd / PutRevoke, "Mem Storage: 0 B / 32.00 GB").
#
# Maps, all pinned (vllm-ascend v0.22.1rc1 = 5f6faa0, Mooncake v0.3.11.post1 =
# e9c6107, cann/hccl ce20062d, cann/hcomm 818068c8):
#   * HCCL codes      cann/hccl/docs/api/context/HcclResult.md:10-35
#                     (0 SUCCESS, 1 PARA, 2 PTR, 3 MEMORY, 4 INTERNAL, 5 NOT_SUPPORT,
#                      6 NOT_FOUND, 7 UNAVAIL, 8 SYSCALL, 9 TIMEOUT, ... 19 NETWORK)
#   * store codes     Mooncake mooncake-store/include/types.h:254-310
#                     (-1 INTERNAL_ERROR, -10 BUFFER_OVERFLOW, -100..-103 segment,
#                      -200 NO_AVAILABLE_HANDLE = out of space, -500 WRITE_FAIL,
#                      -800 TRANSFER_FAIL, -900 RPC_FAIL)
#   * who revokes     mooncake-store/src/client_service.cpp:1692-1761
#                     (PutEnd on success, BatchPutRevoke on a failed transfer)
#   * the canned text vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/
#                     backend/mooncake_backend.py:188-200 ("Check memory and store
#                     capacity" is printed for ANY negative code)
#
# Verdicts follow the worked case #11454. See
# references/kv-pool-store/hccl-onesided-store-put.md.
set -u

if [ "$#" -eq 0 ]; then
  echo "usage: store-put-code-check.sh <paste|serve-log.txt> [...] | -" >&2
  exit 2
fi

read -r -d '' PYSETUP <<'PY'
import re, sys

HCCL = {
    0: 'HCCL_SUCCESS', 1: 'HCCL_E_PARA (parameter)', 2: 'HCCL_E_PTR (null pointer)',
    3: 'HCCL_E_MEMORY', 4: 'HCCL_E_INTERNAL (generic)', 5: 'HCCL_E_NOT_SUPPORT',
    6: 'HCCL_E_NOT_FOUND', 7: 'HCCL_E_UNAVAIL', 8: 'HCCL_E_SYSCALL',
    9: 'HCCL_E_TIMEOUT', 10: 'HCCL_E_OPEN_FILE_FAILURE', 11: 'HCCL_E_TCP_CONNECT',
    12: 'HCCL_E_ROCE_CONNECT', 13: 'HCCL_E_TCP_TRANSFER', 14: 'HCCL_E_ROCE_TRANSFER',
    15: 'HCCL_E_RUNTIME', 16: 'HCCL_E_DRV', 17: 'HCCL_E_PROFILING', 18: 'HCCL_E_CCE',
    19: 'HCCL_E_NETWORK', 20: 'HCCL_E_AGAIN', 21: 'HCCL_E_REMOTE',
    22: 'HCCL_E_SUSPENDING', 23: 'HCCL_E_RESERVED',
}

STORE = {
    0: 'OK', -1: 'INTERNAL_ERROR (generic)', -10: 'BUFFER_OVERFLOW',
    -100: 'SHARD_INDEX_OUT_OF_RANGE', -101: 'SEGMENT_NOT_FOUND',
    -102: 'SEGMENT_ALREADY_EXISTS', -103: 'CLIENT_NOT_FOUND',
    -200: 'NO_AVAILABLE_HANDLE (out of space)', -300: 'INVALID_VERSION',
    -400: 'INVALID_KEY', -500: 'WRITE_FAIL', -600: 'INVALID_PARAMS',
    -601: 'ILLEGAL_CLIENT', -700: 'INVALID_WRITE', -703: 'REPLICA_IS_NOT_READY',
    -707: 'LEASE_EXPIRED', -800: 'TRANSFER_FAIL', -900: 'RPC_FAIL',
}

ONE_SIDED = ('HcclBatchPut', 'HcclBatchGet', 'HcclRegisterMem',
             'HcclExchangeMemDesc', 'HcclEnableMemAccess')


def load(path):
    if path == '-':
        return sys.stdin.read()
    with open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()


def num(token):
    try:
        return int(token)
    except ValueError:
        return None


def report(path, text):
    print('== ' + path)
    hccl = []
    for m in re.finditer(r'hccl_result\s*=\s*(-?\d+)|hcclRet\s*->\s*(-?\d+)', text):
        hccl.append(num(m.group(1) or m.group(2)))
    hccl = sorted({h for h in hccl if h is not None})

    store_codes = []
    for m in re.finditer(r'error_codes\s*=\s*\[([^\]]*)\]', text):
        for tok in re.findall(r'-?\d+', m.group(1)):
            store_codes.append(int(tok))
    store_codes = sorted(set(store_codes))

    api = sorted({a for a in ONE_SIDED if a in text})
    transfer_lines = [l.strip() for l in text.splitlines() if 'Transfer failed for key' in l]

    def counter(name):
        # Master metric lines look like "Batch Requests: PutStart 8" or
        # "Batch Requests: PutEnd:(Req=0.00/0.00/0.00, Item=0.00/0.00)", and some
        # reporters paste "PutStart ... non-zero" instead of a number. Lines that
        # state an expectation ("PutStart > 0", "PutRevoke = 0 or very small") are
        # skipped: they describe the wanted outcome, not the observed counters.
        vals = []
        for line in text.splitlines():
            if name not in line or '>' in line or '<' in line:
                continue
            tail = line.split(name, 1)[1]
            if 'non-zero' in tail:
                vals.append(1)
                continue
            m = re.search(r'(\d+)', tail)
            if m:
                vals.append(int(m.group(1)))
        return max(vals) if vals else None

    put_start = counter('PutStart')
    put_end = counter('PutEnd')
    put_revoke = counter('PutRevoke')

    mem = re.search(r'Mem Storage:\s*([\d.]+)\s*(\w+)\s*/\s*([\d.]+)\s*(\w+)', text)
    master_full = None
    if mem:
        used, total = float(mem.group(1)), float(mem.group(3))
        # percentages and byte counts both appear; only judge when total > 0
        master_full = (used, mem.group(2), total, mem.group(4))

    print('  HCCL codes seen: ' + (', '.join(
        '{} = {}'.format(h, HCCL.get(h, 'unknown HCCL code')) for h in hccl) or 'none'))
    print('  store codes seen: ' + (', '.join(
        '{} = {}'.format(c, STORE.get(c, 'unmapped ErrorCode')) for c in store_codes)
        or 'none'))
    if api:
        print('  one-sided HCCL API named in the log: ' + ', '.join(api)
              + '  (cann/hcomm/pkg_inc/hccl/hccl_one_sided_services.h:63-139)')
    print('  master counters: PutStart={} PutEnd={} PutRevoke={}'.format(
        put_start if put_start is not None else '?',
        put_end if put_end is not None else '?',
        put_revoke if put_revoke is not None else '?'))

    if transfer_lines:
        print('  authoritative client line(s) present (quote these, they carry the real code):')
        for line in transfer_lines[:3]:
            print('    ' + line[:180])

    # ---- verdicts -------------------------------------------------------
    if put_start is not None and put_end is not None and put_start > 0 and put_end == 0:
        print('  -> control plane allocated replicas (PutStart) and never finalised them:')
        print('     the object-store key path worked, the data write did not. That is the')
        print('     store data plane, not AscendStoreConnector key/registry logic.')
    if put_revoke:
        print('  -> PutRevoke non-zero: the client itself revoked the replicas after the')
        print('     transfer failed (client_service.cpp:1719-1723, :1760-1761).')
    for c in store_codes:
        if c == -1:
            print('  -> store code -1 = INTERNAL_ERROR, i.e. the store did not classify it.')
            if master_full and master_full[2] > 0 and master_full[0] < master_full[2] * 0.05:
                print('     A space failure would be -200 NO_AVAILABLE_HANDLE, and your master')
                print('     reports {} {} / {} {} used, so capacity is excluded twice over.'
                      .format(master_full[0], master_full[1], master_full[2], master_full[3]))
            else:
                print('     A space failure would be -200 NO_AVAILABLE_HANDLE, not -1.')
        elif c == -200:
            print('  -> store code -200 NO_AVAILABLE_HANDLE: this IS a space/handle failure;')
            print('     go to ../pd-disaggregation/kv-pool-store-segments.md.')
        elif c in (-10, -500, -800, -900):
            print('     store code {} is a classified failure - quote it as the owner.'
                  .format(c))
    for h in hccl:
        if h == 4:
            print('  -> HCCL 4 = HCCL_E_INTERNAL: a generic rejection inside HCCL. It is not')
            print('     PARA(1)/PTR(2)/MEMORY(3)/TIMEOUT(9), so do not answer "bad parameter"')
            print('     or "out of memory" from this number; name the one-sided precondition')
            print('     chain (RegisterMem -> ExchangeMemDesc -> EnableMemAccess) instead.')
        elif h in (1, 2, 3, 6, 7):
            print('  -> HCCL {} = {}: a classified failure, prefer it over the generic text.'
                  .format(h, HCCL[h]))
    if 'Check memory and store capacity' in text:
        print('  -> "Check memory and store capacity" is canned advice printed for any')
        print('     negative code (mooncake_backend.py:194); it did not check capacity.')
    for marker in ('install AscendDirectTransport', 'reset AscendDirectTransport',
                   'ascend_transport.so', 'HcclTransport'):
        if marker in text:
            print('  -> transport marker in this paste: ' + marker)
            break
    else:
        print('  -> no transport marker: ask for the prefill log lines above the failure')
        print('     ("install <X>Transport for: ...", "register mem addr:...") to learn which')
        print('     engine was built in - protocol "ascend" maps to three transport classes.')
    if not transfer_lines and ('Failed to invoke HcclBatchPut' in text
                              or 'HcclBatchPut' in text):
        print('  -> "Failed to invoke HcclBatchPut" is in no pinned export (Mooncake')
        print('     v0.3.11.post1, vllm-ascend, cann/*): it comes from the')
        print('     mooncake-transfer-engine-npu wheel. State that gap and ask for the')
        print('     store client\'s own "Transfer failed for key ..." line instead of')
        print('     grepping more trees.')


for arg in sys.argv[1:]:
    report(arg, load(arg))
PY

uv run --no-project python -c "$PYSETUP" "$@"
