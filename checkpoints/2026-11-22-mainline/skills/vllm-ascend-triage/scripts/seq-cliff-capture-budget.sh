#!/usr/bin/env bash
# Locate the running-sequence threshold where FULL-graph decode is silently
# abandoned, for a report that says throughput drops in a step at a concurrency
# number (recorded #14440: 256 -> 258 on tp=8, DeepSeek-V4-Flash-w8a8-mtp).
#
#   scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1
#   scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1 --running 256,258,264
#   scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1 \
#       --observed-cliff 256 --max-cudagraph-capture-size 528
#   scripts/seq-cliff-capture-budget.sh --max-num-seqs 224 --num-spec 1 --dp 2 \
#       --observed-cliff 224        # the dp=2/tp=4 half of #14440
#
# Implementation sources (re-read them before trusting this on another release):
#   vllm-ascend v0.23.0rc1 f4a08bd  platform.py:207-241 (default max capture =
#     min(max_num_seqs * decode_query_len, 512)), :259-261 (injected early),
#     utils.py:1173-1204 (the warning_once that names both numbers),
#     utils.py:647-669 (truncation of a user-set value), platform.py:555-567 (SP),
#     :100-119 (950-only prune to 4 sizes)
#   vllm v0.23.0 0fc695f  config/vllm.py:1618-1790 (_set_cudagraph_sizes: the
#     8/16-step grid, the max_num_batched_tokens clip, and the "Truncating
#     max_cudagraph_capture_size" warning), v1/cudagraph_dispatcher.py:276-285
#     (num_tokens > max_size -> NONE, no log)
set -u

uv run --no-project python - "$@" <<'PY'
import argparse

ap = argparse.ArgumentParser(description='running-sequence threshold for graph dispatch')
ap.add_argument('--max-num-seqs', type=int, required=True,
                help='the serve command --max-num-seqs (per engine when --dp > 1)')
ap.add_argument('--num-spec', type=int, default=0,
                help='num_speculative_tokens (MTP/EAGLE); decode_query_len = 1 + this')
ap.add_argument('--max-num-batched-tokens', type=int, default=8192,
                help='clips the capture size (vllm config/vllm.py:1683-1684)')
ap.add_argument('--running', default=None,
                help="comma list of running-sequence counts to evaluate, e.g. '256,258,264'")
ap.add_argument('--observed-cliff', default=None,
                help='comma list of the running-sequence counts where the reporter saw the'
                     ' step (per engine). Each value is reconciled against the arithmetic,'
                     ' and on a mismatch the per-engine --max-num-seqs values that would'
                     ' produce it are listed')
ap.add_argument('--max-cudagraph-capture-size', type=int, default=None,
                help='value the reporter put in --compilation-config, if any')
ap.add_argument('--tp', type=int, default=1, help='informational only')
ap.add_argument('--dp', type=int, default=1,
                help='dp engines: each gets its own identical ceiling; the aggregate'
                     ' boundary is per-engine x dp')
ap.add_argument('--enforce-eager', action='store_true',
                help='the launch has --enforce-eager: there is no graph at all')
a = ap.parse_args()

dql = 1 + a.num_spec
HARD_CAP = 512
ascend_default = min(a.max_num_seqs * dql, HARD_CAP)
upstream_default = min(a.max_num_seqs * dql * 2, HARD_CAP)
mns = a.max_num_seqs


def ceiling(mns_, dql_, mnbt, explicit=None):
    """Effective max capture size: platform default -> token clip -> grid floor."""
    req = min(mns_ * dql_, HARD_CAP) if explicit is None else explicit
    req = min(mnbt, req)
    sizes = [i for i in [1, 2, 4] if i <= req]
    if req >= 8:
        sizes += list(range(8, min(req + 1, 256), 8))
    if req >= 256:
        sizes += list(range(256, req + 1, 16))
    if mnbt <= req and mnbt not in sizes:
        sizes.append(mnbt)
    sizes = sorted(set(sizes))
    return (sizes[-1] if sizes else 0), sizes, req


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
          ' refute the mechanism either: it moves the baseline. What it CAN do is'
          ' settle it: measure the same N below and above the cliff -- a flat pair'
          ' means the step was the graph, a step that survives means it was not.')

max_cap, sizes, requested = ceiling(mns, dql, a.max_num_batched_tokens,
                                    a.max_cudagraph_capture_size)
print(f'requested max capture size (after the max_num_batched_tokens clip) = {requested}')

tail = sizes[-6:]
print(f'{len(sizes)} capture sizes; ... {tail}   (step 8 below 256, step 16 at/above)')

if requested != max_cap:
    print(f'  TRUNCATED: the ceiling becomes {max_cap}, not {requested}, because the'
          ' grid cannot land on the requested value. Grep the serve log for'
          ' "Truncating max_cudagraph_capture_size" (vllm config/vllm.py:1755-1759'
          ' prints "Truncating max_cudagraph_capture_size to %d"; vllm-ascend'
          ' utils.py:661 prints "original_size=..., truncated_size=..."). This is the'
          ' line that proves a run never raised its own ceiling -- e.g. MTP off on'
          ' the #14440 serve line injects 264 and lands on 256, so the cliff does'
          ' NOT move when speculation is removed.')
if a.max_cudagraph_capture_size is not None:
    print('  note: platform.py:555-567 overwrites the value again when sequence'
          ' parallelism rewrites the size list (enable_sp + tp>1)')
print(f'effective max capture size = {max_cap}')

boundary = max_cap // dql
print(f'\nuniform-decode tokens per step = running_seqs * {dql}')
label = 'INAPPLICABLE (eager launch)' if a.enforce_eager else 'CLIFF'
print(f'graph dispatch holds while num_tokens <= {max_cap}'
      f'  ->  running_seqs <= {boundary}')
print(f'{label}: {boundary} sequences healthy, {boundary + 1} silently eager'
      f' (vllm v1/cudagraph_dispatcher.py:276-285 returns NONE with no log)')

if mns * dql > HARD_CAP:
    print(f'  why it is a power-of-two-looking number: max_num_seqs*dq = '
          f'{mns * dql} > the hard 512 cap, so the boundary is 512//{dql} = '
          f'{HARD_CAP // dql}, independent of the flag above 264-ish')
elif boundary == mns:
    print(f'  why it sits on a round number: max_num_seqs*dq = {mns * dql} <= 512 and'
          f' lands on the grid, so the boundary IS --max-num-seqs ({mns}) -- the'
          ' reported cliff sits at the value of the flag, not at a power of two')
else:
    print(f'  max_num_seqs*dq = {mns * dql} <= 512 but off the capture grid: the grid'
          f' floors it to {max_cap}, so the boundary is {boundary}, one grid step'
          f' below the flag {mns} (8-step below 256, 16-step at/above)')
if a.dp > 1:
    print(f'  dp={a.dp}: per-engine {boundary} -> aggregate {boundary * a.dp}'
          f' running requests at the step ({a.max_num_seqs} per engine is the flag)')

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

if a.observed_cliff:
    print('\nreconciling the reporter\'s own measurement (per engine):')
    for tok in a.observed_cliff.replace(' ', '').split(','):
        if not tok:
            continue
        obs = int(tok)
        if obs == boundary:
            print(f'  observed {obs}: MATCH -- the capture ceiling is the owner, and no'
                  ' per-rank memory budget is needed to explain it')
            continue
        cands = [m for m in range(1, max(2048, 4 * mns) + 1)
                 if ceiling(m, dql, a.max_num_batched_tokens,
                            a.max_cudagraph_capture_size)[0] // dql == obs]
        head = ', '.join(str(c) for c in cands[:6]) or 'none'
        more = '' if len(cands) <= 6 else f' (+{len(cands) - 6} more)'
        print(f'  observed {obs}: MISMATCH with {boundary} from --max-num-seqs {mns}.'
              f' Per-engine --max-num-seqs values that would produce {obs}: {head}{more}.'
              ' Re-read the run\'s own serve line before concluding a second ceiling:'
              ' the flag is per engine under dp, and one grid step is 8 (below 256) or'
              ' 16 (at/above)')
        if not cands:
            print('    no --max-num-seqs reproduces it -> the step is NOT the capture'
                  ' grid at these settings; ask for the startup capture list'
                  ' ("Starting to capture ACL graphs for cases: [...]") and the'
                  ' per-step "Running batch with cudagraph_mode: ..." DEBUG line')

print(f'\nsuggested A/B: --compilation-config '
      f'\'{{"cudagraph_mode": "FULL_DECODE_ONLY", "max_cudagraph_capture_size": '
      f'{a.max_num_seqs * dql}}}\'  (= max_num_seqs * (1 + num_spec); same formula as'
      ' docs/source/user_guide/release_notes.md:844)')
print('  confirmed if the same N above the cliff returns to its below-cliff rate')
print('  and the startup line "The max_cudagraph_capture_size (...) is smaller than'
      ' the potential max tokens required for decode (...)" stops appearing')
PY
