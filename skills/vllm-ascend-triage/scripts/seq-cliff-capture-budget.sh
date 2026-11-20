#!/usr/bin/env bash
# Locate the running-sequence threshold where FULL-graph decode is silently
# abandoned, for a report that says throughput drops in a step at a concurrency
# number (recorded #14440: 256 -> 258 on tp=8, DeepSeek-V4-Flash-w8a8-mtp).
#
#   scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1
#   scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1 --running 256,258,264
#   scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1 \
#       --max-cudagraph-capture-size 528
#
# Implementation sources (re-read them before trusting this on another release):
#   vllm-ascend v0.23.0rc1 f4a08bd  platform.py:207-241 (default max capture =
#     min(max_num_seqs * decode_query_len, 512)), :259-261 (injected early),
#     utils.py:1173-1204 (the warning_once that names both numbers),
#     utils.py:647-669 (truncation of a user-set value), platform.py:565 (SP)
#   vllm v0.23.0 0fc695f  config/vllm.py:1618-1790 (_set_cudagraph_sizes),
#     v1/cudagraph_dispatcher.py:276-285 (num_tokens > max_size -> NONE, no log)
set -u

uv run --no-project python - "$@" <<'PY'
import argparse, math

ap = argparse.ArgumentParser(description='running-sequence threshold for graph dispatch')
ap.add_argument('--max-num-seqs', type=int, required=True,
                help='the serve command --max-num-seqs')
ap.add_argument('--num-spec', type=int, default=0,
                help='num_speculative_tokens (MTP/EAGLE); decode_query_len = 1 + this')
ap.add_argument('--max-num-batched-tokens', type=int, default=8192,
                help='clips the capture size (vllm config/vllm.py:1683-1684)')
ap.add_argument('--running', default=None,
                help="comma list of running-sequence counts to evaluate, e.g. '256,258,264'")
ap.add_argument('--max-cudagraph-capture-size', type=int, default=None,
                help='value the reporter put in --compilation-config, if any')
ap.add_argument('--tp', type=int, default=1, help='informational only')
ap.add_argument('--dp', type=int, default=1,
                help='informational only: dp engines each get their own ceiling')
ap.add_argument('--enforce-eager', action='store_true',
                help='the launch has --enforce-eager: there is no graph at all')
a = ap.parse_args()

dql = 1 + a.num_spec
ascend_default = min(a.max_num_seqs * dql, 512)
upstream_default = min(a.max_num_seqs * dql * 2, 512)

print(f'max_num_seqs={a.max_num_seqs} num_spec={a.num_spec} '
      f'(decode_query_len={dql}) max_num_batched_tokens={a.max_num_batched_tokens} '
      f'tp={a.tp} dp={a.dp}')
print(f'ascend default max_cudagraph_capture_size = min({a.max_num_seqs}*{dql}, 512)'
      f' = {ascend_default}   <- platform.py:241, injected at :259-261')
print(f'upstream default (only when the platform injects nothing) = '
      f'min({a.max_num_seqs}*{dql}*2, 512) = {upstream_default}')
if a.enforce_eager:
    print('launch has --enforce-eager -> _set_cudagraph_sizes() zeroes the list;'
          ' no graph, no cliff from this mechanism. An --enforce-eager run cannot'
          ' refute the mechanism either: it moves the baseline.')

requested = a.max_cudagraph_capture_size if a.max_cudagraph_capture_size else ascend_default
requested = min(a.max_num_batched_tokens, requested)
print(f'requested max capture size (after the max_num_batched_tokens clip) = {requested}')

# capture-size list, vllm config/vllm.py:1710-1730
sizes = [i for i in [1, 2, 4] if i <= requested]
if requested >= 8:
    sizes += list(range(8, min(requested + 1, 256), 8))
if requested >= 256:
    sizes += list(range(256, requested + 1, 16))
if a.max_num_batched_tokens <= requested and a.max_num_batched_tokens not in sizes:
    sizes.append(a.max_num_batched_tokens)
sizes = sorted(set(sizes))
tail = sizes[-6:]
print(f'{len(sizes)} capture sizes; ... {tail}   (step 8 below 256, step 16 above)')

# the last element of the list wins: vllm config/vllm.py:1755-1775 truncates, and
# vllm-ascend utils.py:647-669 truncates again (it raises only when both knobs are
# user-set)
max_cap = sizes[-1]
if requested != max_cap:
    print(f'  TRUNCATED: the ceiling becomes {max_cap}, not {requested}, because the'
          ' 16-step grid above 256 cannot land on the requested value'
          ' (vllm config/vllm.py:1755-1759, vllm-ascend utils.py:647-669). grep the'
          ' serve log for "Truncating max_cudagraph_capture_size"')
if a.max_cudagraph_capture_size is not None:
    print('  note: platform.py:563-565 overwrites the value again when sequence'
          ' parallelism rewrites the size list')
print(f'effective max capture size = {max_cap}')

boundary = max_cap // dql
print(f'\nuniform-decode tokens per step = running_seqs * {dql}')
label = 'INAPPLICABLE (eager launch)' if a.enforce_eager else 'CLIFF'
print(f'graph dispatch holds while num_tokens <= {max_cap}'
      f'  ->  running_seqs <= {boundary}')
print(f'{label}: {boundary} sequences healthy, {boundary + 1} silently eager'
      f' (vllm v1/cudagraph_dispatcher.py:276-285 returns NONE with no log)')

if boundary == a.max_num_seqs:
    print(f'  note: --max-num-seqs {a.max_num_seqs} is itself the boundary, so the'
          ' reported cliff sits exactly at the value of the flag')

rows = [boundary - 1, boundary, boundary + 1]
if a.running:
    rows += [int(x) for x in a.running.replace(' ', '').split(',') if x]
print(f'\n{"running":>8} {"num_tokens":>11} {"padded_to":>10}  dispatch')
for r in sorted(set(rows)):
    if r < 1:
        continue
    n = r * dql
    if n > max_cap:
        print(f'{r:>8} {n:>11} {"-":>10}  NONE (eager)')
    else:
        padded = next((s for s in sizes if s >= n), max_cap)
        print(f'{r:>8} {n:>11} {padded:>10}  FULL/PIECEWISE graph')

print(f'\nsuggested A/B: --compilation-config '
      f'\'{{"cudagraph_mode": "FULL_DECODE_ONLY", "max_cudagraph_capture_size": '
      f'{a.max_num_seqs * dql}}}\'  (= max_num_seqs * (1 + num_spec); same formula as'
      ' docs/source/user_guide/release_notes.md:844)')
print('  confirmed if the same N above the cliff returns to its below-cliff rate')
print('  and the startup line "The max_cudagraph_capture_size (...) is smaller than'
      ' the potential max tokens required for decode (...)" stops appearing')
PY
