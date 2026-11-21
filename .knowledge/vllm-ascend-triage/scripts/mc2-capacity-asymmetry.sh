#!/usr/bin/env bash
# Compare the MC2 capacity / `global_bs` that each node of a multi-node EP start
# derives from *its own* --compilation-config, for the symptom "2 nodes, EP on,
# engine dies at torch.compile / initial profiling with 507014 aicore timeout in
# profile_run's extra dummy run". One asymmetric flag makes the two halves of the
# EP group declare different dispatch buffer sizes to the same all-to-all.
#
#   scripts/mc2-capacity-asymmetry.sh \
#       --capture-a 1,2,4,8,16,32,64 --capture-b 1,2,4,8,16,32,64,128,256,512 \
#       --tp 8 --dp 2 --experts 128 --max-num-batched-tokens 8192
#
#   scripts/mc2-capacity-asymmetry.sh --capture-a 'default' --capture-b 'default' \
#       --tp 8 --dp 2 --max-num-seqs 16            # both nodes omit --compilation-config
#
# What it does NOT do: it does not prove the hang. It prints the per-rank
# arithmetic so the symmetry check is a numbers statement, and it says which lane
# to take when the numbers agree.
#
# Implementation sources (re-read before trusting on another release):
#   vllm-ascend v0.18.0rc1 99e1ea0  ascend_forward_context.py:170-182
#     (set_mc2_tokens_capacity: max_num_tokens = max_cudagraph_capture_size when
#      cudagraph_capture_sizes else min(max_num_reqs * uniform_decode_query_len,
#      512); capacity = ceil(max_num_tokens / tp) * tp)
#   vllm-ascend v0.18.0rc1 99e1ea0  ops/fused_moe/token_dispatcher.py:100-118
#     (same derivation, then global_bs = ceil(max_num_tokens / tp) * ep_world_size)
#     and :156 (global_bs is passed to npu_moe_distribute_dispatch as an attribute)
#   vllm-ascend v0.18.0rc1 99e1ea0  ascend_forward_context.py:236-247
#     (selector: EP off or ep world 1 -> ALLGATHER; A2 -> MC2 iff
#      num_experts_per_device <= 24 and ep_world_size >= 16 and
#      num_tokens <= capacity; A3 branch is where VLLM_ASCEND_ENABLE_FUSED_MC2 is read)
#   vllm-ascend v0.18.0rc1 99e1ea0  worker/model_runner_v1.py:2538-2545
#     (profile_run runs _dummy_run(mc2_tokens_capacity, with_prefill=True) first)
set -u

uv run --no-project python - "$@" <<'PY'
import argparse
import math

ap = argparse.ArgumentParser(description='MC2 capacity/global_bs symmetry across nodes')
ap.add_argument('--capture-a', required=True,
                help="node 0 cudagraph_capture_sizes, e.g. '1,2,4,8,16,32,64' (or 'default' when the flag is absent)")
ap.add_argument('--capture-b', required=True, help='node 1 list, or \'default\'')
ap.add_argument('--tp', type=int, required=True, help='--tensor-parallel-size (per node)')
ap.add_argument('--dp', type=int, default=1, help='--data-parallel-size (total)')
ap.add_argument('--pp', type=int, default=1, help='--pipeline-parallel-size')
ap.add_argument('--experts', type=int, default=None,
                help='num_experts of the checkpoint (config.json) - decides the A2 <=24 gate')
ap.add_argument('--max-num-seqs', type=int, default=16, help='used only for the default derivation')
ap.add_argument('--num-spec', type=int, default=0, help='num_speculative_tokens (decode query len = 1 + this)')
ap.add_argument('--max-num-batched-tokens', type=int, default=None,
                help='informational: profile_run only takes the extra dummy run when this exceeds capacity')


def sizes(text):
    t = text.strip().strip('\'"')
    if t.lower() in ('default', 'none', ''):
        return None
    try:
        out = sorted({int(x) for x in t.replace('[', ' ').replace(']', ' ').split(',') if x.strip()})
    except ValueError:
        raise SystemExit(f'error: --capture-a/-b take integers, got {text!r}')
    if not out or out[0] < 1:
        raise SystemExit(f'error: capture sizes must be >= 1, got {text!r}')
    return out


def derive(label, cap, a):
    if cap:
        max_num_tokens = max(cap)
        origin = f'max(cudagraph_capture_sizes) = {max_num_tokens}'
    else:
        max_num_tokens = min(a.max_num_seqs * (1 + a.num_spec), 512)
        origin = (f'no --compilation-config: min(max_num_seqs * query_len, 512) = '
                  f'min({a.max_num_seqs} * {1 + a.num_spec}, 512) = {max_num_tokens}')
    per_rank = math.ceil(max_num_tokens / a.tp)
    capacity = per_rank * a.tp
    ep = a.tp * a.pp * a.dp
    return dict(label=label, cap=cap, max_num_tokens=max_num_tokens, origin=origin,
                per_rank=per_rank, capacity=capacity, ep=ep, global_bs=per_rank * ep)


a = ap.parse_args()
nodes = [derive('node 0', sizes(a.capture_a), a), derive('node 1', sizes(a.capture_b), a)]
ep_world = nodes[0]['ep']

print(f'world: tp={a.tp} pp={a.pp} dp={a.dp} -> ep_world_size={ep_world}'
      f'   (MC2 on A2 needs ep_world_size >= 16)')
if a.experts:
    per_dev = a.experts / ep_world
    gate = 'passes' if per_dev <= 24 else 'FAILS (A2 MC2 needs num_experts_per_device <= 24)'
    print(f'experts: {a.experts} -> {a.experts}/{ep_world} = {per_dev:g} per device, {gate}')
else:
    print('experts: not given (--experts); the A2 <=24 gate is unverified - ask for config.json')
if a.max_num_batched_tokens is not None:
    for n in nodes:
        extra = (f'profile_run takes the extra _dummy_run({n["capacity"]})'
                 if a.max_num_batched_tokens > n['capacity']
                 else 'capacity >= max_num_batched_tokens: the extra dummy run is skipped')
        print(f'  {n["label"]}: max_num_batched_tokens={a.max_num_batched_tokens} -> {extra}')
print()

for n in nodes:
    listed = 'absent (default derivation)' if not n['cap'] else str(n['cap'])
    print(f'{n["label"]}: cudagraph_capture_sizes {listed}')
    print(f'  {n["origin"]}')
    print(f'  tokens/TP rank = ceil({n["max_num_tokens"]}/{a.tp}) = {n["per_rank"]}'
          f'  -> mc2_tokens_capacity = {n["capacity"]}')
    print(f'  global_bs = {n["per_rank"]} * ep_world_size {ep_world} = {n["global_bs"]}')
print()

cap_a, cap_b = nodes[0]['capacity'], nodes[1]['capacity']
gb_a, gb_b = nodes[0]['global_bs'], nodes[1]['global_bs']
print('verdict')
mc2_eligible = ep_world >= 16 and (a.experts is None or a.experts / ep_world <= 24)
if not mc2_eligible:
    print('  NOTE: this configuration does not reach the A2 MC2 row (see the gates above), so the')
    print('  MC2-dispatch lane is not it - the first collective in profile_run is ALLGATHER and the')
    print('  hang belongs to whatever op the log names. The numbers below are printed only to show')
    print('  why the row was not taken.')
if cap_a != cap_b or gb_a != gb_b:
    ratio = max(cap_a, cap_b) / max(1, min(cap_a, cap_b))
    print(f'  ASYMMETRIC: capacity {cap_a} vs {cap_b} ({ratio:g}x), global_bs {gb_a} vs {gb_b} ({ratio:g}x).')
    print('  Each rank declares the dispatch buffer from its own launch command, so the two halves of')
    print(f'  the ep_world_size={ep_world} group hand different sizes to the same all-to-all: rank(s) on node 0')
    print(f'  issue a {cap_a}-token profiling batch, node 1 issues {cap_b}, and the dispatch that must')
    print('  complete on every rank cannot pair up.')
    print('  Status: mechanism is source-verified (derivation + kwargs above); that it hangs rather than')
    print('  errors is a hypothesis - close it with the identical-config rerun, not with env ablation.')
    print('  Rung 1: make both nodes identical (the tutorial pair for GLM-4.x A2 multi-node uses')
    print('          [1,2,4,8,16,32,64,128,256,512] on BOTH nodes, docs/source/tutorials/models/GLM4.x.md'
          ' at v0.18.0rc1 :226 and :279), or drop --compilation-config on both.')
else:
    print(f'  SYMMETRIC: both nodes declare capacity {cap_a} / global_bs {gb_a}.')
    print('  This check is closed - do not blame the capture sizes. Next lanes, in order:')
    print('  the headless node\'s own log (a rank that never entered the dispatch), a 16-rank hccl_test')
    print('  plus hccn_tool -i <id> -link -g and a device-IP ping, then the image-CANN vs host-driver')
    print('  pairing (cat /usr/local/Ascend/driver/version.info on both nodes).')
print()
print('reminders for the answer:')
print('  - VLLM_ASCEND_ENABLE_FUSED_MC2 is read only in the A3 branch: on A2 it is a no-op, not a fix.')
print('  - there is no env that forces ALLGATHER while EP stays on; dropping --enable-expert-parallel')
print('    is the ALLGATHER route (A2 MC2 also needs ep_world_size >= 16).')
print('  - a single-node success at world 8 never selected MC2, so it does not clear this path.')
print('  - ask for the headless node\'s full log and the plog [AIC_INFO] fault kernel_name for the')
print('    failure minute: the op name is the authoritative owner.')
PY
