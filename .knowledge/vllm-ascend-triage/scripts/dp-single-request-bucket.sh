#!/usr/bin/env bash
# Price the graph bucket a *single* request replays under DP > 1, for the
# symptom "one request at a time hangs at replay / RPC timeout, concurrency is
# fine" (recorded #3128).
#
#   scripts/dp-single-request-bucket.sh --capture-sizes 8,1 --dp 8 --tp 2
#   scripts/dp-single-request-bucket.sh --capture-sizes 8,1 --dp 8 --tp 2 \
#       --max-num-seqs 8 --report 1,2,8
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

ap = argparse.ArgumentParser(description='graph bucket chosen for N running sequences per DP rank')
ap.add_argument('--capture-sizes', required=True,
                help='the --compilation-config cudagraph_capture_sizes list, e.g. 8,1 '
                     '(defaults are release-specific: print them with '
                     'scripts/fia-tnd-seqlen-check.sh --capture-sizes)')
ap.add_argument('--dp', type=int, default=None, help='--data-parallel-size')
ap.add_argument('--tp', type=int, default=None, help='--tensor-parallel-size')
ap.add_argument('--max-num-seqs', type=int, default=None,
                help='--max-num-seqs (per DP rank); informational, the cap on one step')
ap.add_argument('--report', default='1,2,8',
                help="comma list of sequences per DP rank to tabulate (default '1,2,8')")
ap.add_argument('--num-spec', type=int, default=0,
                help='num_speculative_tokens: decode_query_len = 1 + this')
a = ap.parse_args()

try:
    sizes = sorted({int(x) for x in a.capture_sizes.replace('[', ' ').replace(']', ' ')
                    .replace('"', ' ').split(',') if x.strip()}, reverse=True)
except ValueError:
    raise SystemExit('error: --capture-sizes takes integers, e.g. 8,1')
if not sizes or sizes[-1] < 1:
    raise SystemExit('error: --capture-sizes must be >= 1')

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
