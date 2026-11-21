#!/usr/bin/env bash
# Print the per-rank `x.shape[0]` ("bs") that the MC2 MoE dispatch would receive
# for a reported serve line, for the symptom `call aclnnMoeDistributeDispatchV2
# failed, error code is 561002` + `CheckShapeAndSetTiling` + `batchsize is
# invalid` (the token-dim tiling refusal, feature-enable-recipes/
# moe-expert-parallel-with-dp.md §Step 2c).
#
#   scripts/mc2-dispatch-bs-check.sh --tp 8 --dp 2 \
#       --max-num-batched-tokens 1024 --max-num-seqs 10 --experts 128
#   scripts/mc2-dispatch-bs-check.sh --tp 8 --dp 2 --max-num-batched-tokens 4096 \
#       --max-num-seqs 10 --enforce-eager --pin 0.10.1rc1
#
# It answers three questions with numbers, so the answer to the reporter is a
# statement about bs, not a theory: (1) is MC2 selectable at all (EP world size,
# and the token gate), (2) what bs does a prefill chunk / a decode step hand the
# op, (3) can the ranks disagree -- the DP all-reduce is skipped under
# `--enforce-eager`, and at v0.10.1rc1 the op is always called in uniform mode
# (`global_bs=0`, no `x_active_mask` on A2), so unequal per-rank counts are not
# representable.
#
# What it does NOT do: it cannot print the op's own acceptance limit. The CANN
# op source is not in the snapshot (`cann/ops-nn` / `cann/ops-transformer` are
# `unavailable` in the 0.10.1rc1 or the 0.22.1rc1 SOURCES.json), so 512 is the
# plugin's own per-rank constant, not a quoted op contract.
#
# Implementation sources (re-read before trusting on another release):
#   vllm-ascend v0.10.1rc1 7e16b4a7
#     worker/model_runner_v1.py:369            mc2_tokens_capacity = 512 * tp
#     worker/model_runner_v1.py:1623-1631      selector: num_tokens <= capacity
#                                              (no with_prefill term)
#     worker/model_runner_v1.py:596-599        _sync_metadata_across_dp early
#                                              return when dp_size==1 or
#                                              enforce_eager
#     worker/model_runner_v1.py:1101           num_input_tokens = maybe_padded
#     ascend_forward_context.py:125-126        padded = ceil(max_across_dp/tp)*tp
#     distributed/moe_comm_method.py:334-350   pad to padded, tensor_split by tp
#     distributed/moe_comm_method.py:315,396   need_extra_args = is_a3; global_bs 0
#     ops/common_fused_moe.py:487-488          ep_size < 16 -> allgathercommimpl
#   vllm-ascend v0.22.1rc1 5f6faa0c
#     ops/fused_moe/token_dispatcher.py:120-141,183-184  global_bs derivation,
#                                              x_active_mask only when global_bs==0
#     utils.py:1083-1095                       why: skipped DP all-reduce => ranks
#                                              differ => non-zero global_bs
set -u

uv run --no-project python - "$@" <<'PY'
import argparse
import math

ap = argparse.ArgumentParser(description='MC2 dispatch per-rank bs / uniformity arithmetic')
ap.add_argument('--tp', type=int, required=True, help='--tensor-parallel-size per node')
ap.add_argument('--dp', type=int, default=1, help='--data-parallel-size (total)')
ap.add_argument('--pp', type=int, default=1, help='--pipeline-parallel-size')
ap.add_argument('--max-num-batched-tokens', type=int, default=None,
                help='the largest extend chunk one rank can be handed')
ap.add_argument('--max-num-seqs', type=int, default=16)
ap.add_argument('--num-spec', type=int, default=0, help='num_speculative_tokens')
ap.add_argument('--experts', type=int, default=None, help='checkpoint num_experts')
ap.add_argument('--enforce-eager', action='store_true',
                help='the serve line carries --enforce-eager (skips the DP all-reduce)')
ap.add_argument('--pin', default='0.10.1rc1', choices=['0.10.1rc1', '0.22.1rc1'],
                help='which derivation to print (both are printed as a note)')
a = ap.parse_args()

tp, dp, pp = a.tp, a.dp, a.pp
ep = tp * dp * pp
print(f'layout: tp={tp} dp={dp} pp={pp} -> ep_size if EP enabled = {ep}')
if ep < 16:
    print('  EP world < 16: the MoE layer is forced to `allgathercommimpl` '
          '(common_fused_moe.py:487-488 at v0.10.1rc1) -> this op is never called.')
    print('  So a paste naming npu_moe_distribute_dispatch* proves EP is on and '
          'ep_size >= 16 (TP8xDP2 or TP16 on a 2-node A2 box).')
    raise SystemExit(0)

decode_tokens = a.max_num_seqs * (1 + a.num_spec)

if a.pin == '0.10.1rc1':
    cap = 512 * tp
    print(f'capacity (v0.10.1rc1 model_runner_v1.py:369): 512 * tp = {cap} tokens, '
          f'i.e. per-rank bs cap 512')
else:
    base = decode_tokens
    cap = math.ceil(min(base, 512) / tp) * tp
    print(f'capacity (later derivation, set_mc2_tokens_capacity): '
          f'ceil(min(max_num_reqs*(1+num_spec), 512)/tp)*tp = {cap} tokens')

print(f'DP all-reduce across ranks: {"SKIPPED (--enforce-eager)" if a.enforce_eager else "runs"}'
      '  [model_runner_v1.py:596-599]')
print()

def bs_for(tokens, label, uniform):
    if tokens > cap:
        print(f'  {label}: {tokens} tokens > capacity {cap} -> ALLGATHER '
              f'(not this op) -- a mixed-method step if only some ranks exceed it')
        return None
    padded = math.ceil(tokens / tp) * tp
    bs = padded // tp
    uniform_txt = 'uniform across ranks' if uniform else 'PER-RANK (peers may differ)'
    print(f'  {label}: {tokens} tokens -> padded {padded} -> bs = {bs} ({uniform_txt})')
    return bs

uniform = not a.enforce_eager
print('bs handed to npu_moe_distribute_dispatch* (= padded_num_tokens / tp):')
if a.max_num_batched_tokens:
    bs_prefill = bs_for(a.max_num_batched_tokens, 'extend/prefill chunk', uniform)
else:
    bs_prefill = None
    print('  extend/prefill chunk: pass --max-num-batched-tokens to compute it')
bs_decode = bs_for(decode_tokens, 'decode step (max_num_seqs*(1+num_spec))', uniform)
print()

if a.experts is not None:
    per_card = a.experts // ep
    print(f'experts per card: {a.experts} / {ep} = {per_card}'
          f'{"  > 24 -> the OTHER A2 tiling refusal (Step 2b)" if per_card > 24 else "  (<= 24, the expert cap is not this one)"}')
    print()

print('how to read it:')
print('  bs cannot exceed 512 here by construction: the gate and the pad target use the')
print('  same DP-max token count, so a bs refusal is not a "batch too large" statement')
print('  unless the two counts came from different places.')
if not uniform:
    print('  --enforce-eager splits them: each rank keeps its OWN count, selects on it and')
    print('  dispatches it, while at v0.10.1rc1 the op is called in uniform mode on A2')
    print('  (global_bs=0, no x_active_mask -- moe_comm_method.py:315,396). Unequal bs across')
    print('  the EP group is the first thing to close: ask for both launch lines, re-run')
    print('  without --enforce-eager, and separate "the MC2 path" from "the weights" by')
    print('  dropping to --data-parallel-size 1 (ep world < 16 -> ALLGATHER).')
else:
    print('  Counts are uniform and <= 512, so the plugin-side arithmetic does not explain')
    print('  it. The op\'s own tiling is then the remaining owner: hand the CANN/torch_npu')
    print('  versions plus the full EZ9999 block (FUNC/FILE/LINE of the check, any number it')
    print('  prints) to the op owner, and say which cheap check above was already run. No')
    print('  env knob is recorded as a lever for this signature.')
print('  ask for (cheapest first): the two `vllm serve` lines, the whole EZ9999/')
print('  CheckShapeAndSetTiling block, and the DEBUG `num_tokens: N, moe_comm_method: mc2`')
print('  line (global first rank only, model_runner_v1.py:1635-1636).')
PY
