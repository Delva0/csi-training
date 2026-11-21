#!/usr/bin/env bash
# Reproduce vllm-ascend's ACL-graph capture-size budget: how many sizes the
# deployment can capture and which ones survive, given the model and parallel
# configuration. Prints the arithmetic, so a log line like
# "Adjusted ACL graph batch sizes ... 67 -> 6 sizes" can be checked instead of
# trusting the formula by eye.
#
#   scripts/aclgraph-size-budget.sh --layers 94 --tp 16 --dp 1 --era 0.11
#   scripts/aclgraph-size-budget.sh --layers 61 --tp 8 --dp 2 --ep --max-num-seqs 16
#   scripts/aclgraph-size-budget.sh --layers 61 --sizes 1,2,4,8,16,32,64,96,128
#
# Implementation sources (verified, re-read before trusting on another release):
#   v0.11.0rc0 00ba071 vllm_ascend/utils.py:286-390, platform.py:237
#   v0.20.2rc1 367b8e6 vllm_ascend/utils.py:567-690, platform.py:425
#   v0.18.0rc1 99e1ea0 vllm_ascend/utils.py:456-560, platform.py:370 (same era as 0.20:
#     CP subtraction + MTP draft layers already present; --era 0.18 == --era 0.20)
set -u

uv run --no-project python - "$@" <<'PY'
import argparse, math, sys

ap = argparse.ArgumentParser(description='vllm-ascend ACL graph capture-size budget')
ap.add_argument('--layers', type=int, required=True,
                help='num_hidden_layers of the model under test')
ap.add_argument('--tp', type=int, default=1)
ap.add_argument('--dp', type=int, default=1)
ap.add_argument('--pcp', type=int, default=1, help='prefill_context_parallel_size (>=0.20 only)')
ap.add_argument('--dcp', type=int, default=1, help='decode_context_parallel_size (>=0.20 only)')
ap.add_argument('--draft-layers', type=int, default=0,
                help='speculative draft/MTP layers counted in resources_per_graph (>=0.20 only)')
ap.add_argument('--ep', action='store_true', help='enable_expert_parallel')
ap.add_argument('--moe', action='store_true', help='model is MoE (is_moe_model)')
ap.add_argument('--aiv', action='store_true', help='HCCL_OP_EXPANSION_MODE=AIV')
ap.add_argument('--multistream-shared-expert', action='store_true')
ap.add_argument('--max-num-seqs', type=int, default=256,
                help='used to derive the upstream default capture sizes')
ap.add_argument('--max-num-batched-tokens', type=int, default=None,
                help='upstream filters sizes above this value')
ap.add_argument('--sizes', default=None,
                help="explicit cudagraph_capture_sizes, e.g. '1,2,4,8,16,32'")
ap.add_argument('--era', choices=['0.11', '0.18', '0.20'], default='0.20',
                help='formula era: 0.11 has no CP subtraction and no MTP draft term; 0.18 and 0.20 are'
                     ' identical (both present at v0.18.0rc1 99e1ea0)')
a = ap.parse_args()

MAX_CAPTURE_SIZE = 1800          # 2048 streams, 248 kept as buffer
resources_per_graph = a.layers + 1
if a.era != '0.11' and a.draft_layers:
    resources_per_graph += a.draft_layers + 1
num_comm_groups = sum(size > 1 for size in (a.dp, a.tp))

# upstream list derivation (vllm v0.11.0rc3 config/scheduler.py:222-227,
# config/__init__.py:551-612)
if a.sizes:
    original = sorted({int(x) for x in a.sizes.replace(' ', '').split(',') if x},
                      reverse=True)
    origin = f'explicit --cudagraph-capture-sizes ({len(original)} sizes)'
else:
    cap = min(a.max_num_seqs * 2, 512)
    if a.era == '0.11':
        # vllm v0.11.0rc3 config/__init__.py:551-612
        original = [1, 2, 4] + list(range(8, cap + 1, 8))
    else:
        # vllm v0.13.0 config/vllm.py:1101-1110 (also current at 0.18/0.20):
        # step 8 below 256, step 16 above it -> 51 sizes at cap 512, not 67
        original = [1, 2, 4] + list(range(8, min(cap + 1, 256), 8))
        if cap >= 256:
            original += list(range(256, cap + 1, 16))
    if a.max_num_batched_tokens is not None:
        original = [s for s in original if s <= a.max_num_batched_tokens]
    original = sorted(set(original), reverse=True)
    filt = ('no max_num_batched_tokens filter'
            if a.max_num_batched_tokens is None
            else f'filtered by max_num_batched_tokens={a.max_num_batched_tokens}')
    steps = ('step 8' if a.era == '0.11' else 'step 8 <256, then step 16')
    origin = (f'upstream default: cuda_graph_sizes=[min({a.max_num_seqs}*2,512)]='
              f'[{cap}] -> [1,2,4] + ({steps}), {filt}')

print(f'layers={a.layers} tp={a.tp} dp={a.dp} pcp={a.pcp} dcp={a.dcp} '
      f'ep={a.ep} moe={a.moe} aiv={a.aiv} era={a.era}')
print(f'resources_per_graph = layers + 1 = {resources_per_graph}')
print(f'num_comm_groups     = {num_comm_groups}')
print(origin + f' -> {len(original)} sizes')

budget = MAX_CAPTURE_SIZE
if a.aiv:
    factor = 1 + num_comm_groups + int(a.ep) + int(a.multistream_shared_expert)
    if a.moe:
        factor += int(a.dp > 1)
    else:
        budget -= factor * resources_per_graph
    print(f'AIV branch: parallel_factor = 1 + ncg + ep + shared_expert'
          f'{"" if a.moe else " (+ MAX -= factor*resources)"} = {factor}')
    print(f'  floor({budget} / {resources_per_graph} / {factor})')
    max_sizes = math.floor(budget / resources_per_graph / factor)
else:
    if a.era != '0.11':
        for cp in ('pcp', 'dcp'):
            if getattr(a, cp) > 1:
                budget -= 100
                print(f'FFTS+ branch: -100 streams for {cp}={getattr(a, cp)} -> budget {budget}')
    print(f'FFTS+ branch: floor(({budget} - {num_comm_groups}*40) / '
          f'{resources_per_graph} / (1 + {num_comm_groups}*2))')
    max_sizes = math.floor((budget - num_comm_groups * 40) / resources_per_graph
                           / (1 + num_comm_groups * 2))

print(f'max_num_batch_sizes = {max_sizes}   <- compare with'
      ' "Calculated maximum supported batch sizes for ACL graph: N"')

if max_sizes <= 1:
    print('degenerate: the formula allows at most one graph; a PIECEWISE deployment'
          ' cannot honour this model/parallel configuration')
elif max_sizes < len(original):
    step = (len(original) - 1) / (max_sizes - 1)
    idx = [round(i * step) for i in range(max_sizes)]
    idx[0], idx[-1] = 0, len(original) - 1
    sampled = [original[i] for i in idx]
    print(f'sampled {max_sizes} of {len(original)}: {sampled}')
    print('  <- compare with "Adjusted ACL graph batch sizes ... N -> M sizes"'
          ' and with the "Starting to capture ACL graphs for cases: [...]" list')
else:
    print('no adjustment: upstream list kept:' , original)
PY
