#!/usr/bin/env bash
# Price the graph bucket a *single* request replays under DP > 1, for the
# symptom "one request at a time hangs at replay / RPC timeout, concurrency is
# fine" (recorded #3128).
#
#   bash <skill_root>/scripts/dp-single-request-bucket.sh \
#       --capture-sizes 8,1 --dp 8 --tp 2
#   bash <skill_root>/scripts/dp-single-request-bucket.sh \
#       --capture-sizes '{"cudagraph_capture_sizes": [8,1]}' --dp 8 --tp 2 \
#       --max-num-seqs 8 --report 1,2,8
#
# Run it with `bash`, never `uv run --no-project python <this file>`: the file is
# a bash wrapper around an embedded Python program, so Python reads line 1 as
# source and dies with a SyntaxError.
#
# What it does NOT do: it does not choose the owner. It prints the pad map and
# the per-step branch so the ablation ladder (drop the 1-token bucket,
# --enforce-eager, --data-parallel-size 1) is stated with numbers instead of
# re-derived from source.
#
# Implementation sources (re-read before trusting another release):
#   vllm-ascend v0.10.2rc1 048bfd5  worker/model_runner_v1.py:1103-1109
#     (use_aclgraph and total <= aclgraph_batch_sizes[-1] -> pad_for_cudagraph,
#      else eager), :394-395 (aclgraph_batch_sizes = reversed(capture sizes)),
#     :2081-2101 (_dummy_run, aclgraph_runtime_mode default NONE = eager),
#     :696-738 (_sync_metadata_across_dp: one all_reduce, cannot be skipped
#      when dp_size > 1)
#   vllm-ascend v0.10.2rc1 048bfd5  worker/worker_v1.py:315-316
#     (execute_dummy_batch -> _dummy_run(1) on a rank with 0 scheduled tokens)
#   vllm-ascend v0.10.2rc1 048bfd5  ops/fused_moe.py:454-470
#     (dp_size > 1: pad to forward_context.max_tokens_across_dp, then
#      get_dp_group().all_gather) and ascend_forward_context.py:26-42
#     (ep_size == 1 -> NaiveMulticast | AllGather: EP off keeps the DP
#      collectives)
#   vllm v0.9.1 b6553be  config.py:4146-4159 (bs_to_padded_graph_size),
#     :4335-4340 (pad_for_cudagraph)
set -u

uv run --no-project python - "$@" <<'PY'
import argparse
import re

ap = argparse.ArgumentParser(
    prog='dp-single-request-bucket.sh',
    description='graph bucket chosen for N running sequences per DP rank')
ap.add_argument('--capture-sizes', default=None,
                help='the --compilation-config cudagraph_capture_sizes list, e.g. 8,1; a '
                     'pasted --compilation-config value such as \'{"cudagraph_capture_sizes": '
                     '[8,1]}\' is accepted as-is (defaults are release-specific: print them '
                     'with scripts/fia-tnd-seqlen-check.sh --capture-sizes)')
ap.add_argument('--dp', type=int, default=None, help='--data-parallel-size')
ap.add_argument('--tp', type=int, default=None, help='--tensor-parallel-size')
ap.add_argument('--max-num-seqs', type=int, default=None,
                help='--max-num-seqs (per DP rank); informational, the cap on one step')
ap.add_argument('--report', default='1,2,8',
                help="comma list of sequences per DP rank to tabulate (default '1,2,8')")
ap.add_argument('--num-spec', type=int, default=0,
                help='num_speculative_tokens: decode_query_len = 1 + this')
ap.add_argument('positional', nargs='*',
                help='bare fallback, same order as the flags: the capture-size list (or the '
                     'pasted --compilation-config value), then dp, tp, max-num-seqs')
a = ap.parse_args()


def _as_sizes(text: str):
    """Take the bracketed integer list out of a pasted config value, else the
    bare comma list. Recorded failure this absorbs: the #3128 launch line's
    '{"cudagraph_capture_sizes": [8,1]}' was passed verbatim, first as the
    --capture-sizes value ('takes integers'), then as a bare positional
    (argparse 'required: --capture-sizes'), and the map was then re-derived
    by hand from vllm/config.py — what this script exists to avoid."""
    if not any(ch.isdigit() for ch in text):
        raise SystemExit('error: no capture sizes in that value: a launch without '
                         '--compilation-config has a release-specific default list, so '
                         'print it with scripts/fia-tnd-seqlen-check.sh and pass it here '
                         '(no default is assumed)')
    m = re.search(r'\[\s*([0-9][0-9,\s]*)\]', text)
    body = m.group(1) if m else text
    try:
        return sorted({int(x) for x in body.replace(' ', '').split(',') if x},
                      reverse=True)
    except ValueError:
        raise SystemExit('error: --capture-sizes takes integers, e.g. 8,1 '
                         '(a pasted --compilation-config dict is fine too)')


_raw = a.capture_sizes if a.capture_sizes is not None else (
    a.positional[0] if a.positional else None)
if _raw is None:
    raise SystemExit('error: --capture-sizes is required: --capture-sizes 8,1 --dp 8 --tp 2 '
                     '(or paste the launch\'s --compilation-config value as the first '
                     'positional); -h lists the flags')
sizes = _as_sizes(_raw)
if not sizes or sizes[-1] < 1:
    raise SystemExit('error: --capture-sizes must be >= 1')

# Positional fill-in for the flags, printed so a reader sees what was assumed.
# The first bare positional is the size list only when --capture-sizes is absent.
_rest = a.positional[1:] if a.capture_sizes is None else a.positional
for _name, _val in zip(('dp', 'tp', 'max-num-seqs'), _rest):
    _attr = _name.replace('-', '_')
    if getattr(a, _attr) is None:
        try:
            setattr(a, _attr, int(_val))
        except ValueError:
            raise SystemExit(f'error: positional {_val!r} is not an integer '
                             f'(positional order is sizes, dp, tp, max-num-seqs)')
        print(f'note: positional {_val!r} read as --{_name}')

max_cap = sizes[0]
pad = [0] * (max_cap + 1)
for end, start in zip(sizes, sizes[1:] + [0]):
    for bs in range(start, end):
        pad[bs] = start if bs == start else end
pad[max_cap] = max_cap

print(f'capture sizes (descending): {sizes}   max_capture_size={max_cap}')
print(f'sequences per rank at one step, inferred: '
      f'dp={a.dp if a.dp else "?"} tp={a.tp if a.tp else "?"} '
      f'max_num_seqs={a.max_num_seqs if a.max_num_seqs else "?"} '
      f'num_spec={a.num_spec} (decode_query_len={1 + a.num_spec})')
print()
print('pad map (vllm config.py:4146-4159): num_tokens -> replay bucket')
runs = sorted({1, max_cap} | {int(x) for x in a.report.split(',') if x.strip()})
for b in runs:
    if b < 1:
        continue
    if b > max_cap:
        print(f'  {b:>6} -> eager (above max_capture_size)')
        continue
    n = b * (1 + a.num_spec)
    if n > max_cap:
        print(f'  {b:>6} seqs x {1 + a.num_spec} tok = {n:>6} -> eager (above max_capture_size)')
    else:
        print(f'  {b:>6} seqs x {1 + a.num_spec} tok = {n:>6} -> bucket {pad[n]:>5}'
              f'{"   <- the single-request case" if b == 1 else ""}')
print()
single = pad[1 * (1 + a.num_spec)] if (1 + a.num_spec) <= max_cap else None
busy = pad[max_cap]
print('what differs between one request and concurrency (verified at the pins above):')
if single is not None and single != busy:
    print(f'  - the busy rank replays bucket {single} for a single request but bucket {busy}'
          f' at size, so the two cases never share a captured graph')
else:
    print(f'  - a single request and a full step resolve to the same bucket ({single});'
          f' the bucket is therefore not the discriminator, go to --enforce-eager')
if a.dp and a.dp > 1:
    print(f'  - the other {a.dp - 1} DP rank(s) have 0 scheduled tokens: each calls'
          f' execute_dummy_batch() -> _dummy_run(1), which is EAGER (runtime mode NONE)')
    print('    and still joins the one DP all_reduce in _sync_metadata_across_dp first,')
    print('    so the DP pad target agrees; that is not the mechanism.')
else:
    print('  - dp is 1 (or unknown): there is no dummy-rank population, so this page\'s')
    print('    DP branch does not apply')
print(f'  - --no-enable-expert-parallel keeps the cross-DP MoE collective'
      f' (ep_size=1 -> AllGather in decode / NaiveMulticast in prefill)')
print()
print('ladder, one launch per row, cheapest first:')
if single is not None and single != busy:
    print(f'  1. drop the 1-token bucket: --compilation-config '
          f'\'{{"cudagraph_capture_sizes": [{busy}]}}\''
          f'  (a single request then pads to {busy}, the bucket concurrency already uses)')
else:
    print('  1. no bucket A/B available (a single request and a full step share a bucket):'
          ' start at row 2')
print('  2. --enforce-eager with the same DP/TP: pass => the graph path owns it')
print('  3. --data-parallel-size 1 at the same world size and graph config: pass => DP pairing')
print('  4. unset HCCL_OP_EXPANSION_MODE (AIV) / CPU_AFFINITY_CONF+VLLM_CPU_BINDING, one per launch')
print('  5. --enforce-eager + ASCEND_LAUNCH_BLOCKING=1, then plog-timeline.sh on the failing minute')
print('ask back: the full per-rank log, EngineCore\'s fatal line (which rank timed out),')
print('the plog for that minute, npu-smi during the hang, and the resolved')
print('cudagraph_capture_sizes echo (the capture banner itself is only a tqdm')
print('description at v0.10.2rc1, so it does not list the buckets).')
PY
