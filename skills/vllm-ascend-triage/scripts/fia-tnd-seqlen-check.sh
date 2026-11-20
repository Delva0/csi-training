#!/usr/bin/env bash
# Classify a npu_fused_infer_attention_score TND refusal by arithmetic:
#
#   EZ9999 ... When layout is TND, queryT(16) must be equal to the last element
#   of actualSequenceLengthQ(4) [FUNC: CheckFASeqlenDataInTND]
#
# The op compares two numbers that different padding sites produce. This prints
# the capture-size list the serve command actually resolves to, every
# decomposition of the pair that fits the launch, and the ordered ablations, so
# the owner is named from the numbers instead of from a source walk.
#
#   scripts/fia-tnd-seqlen-check.sh --query-t 16 --seqlen-last 4 --max-num-seqs 16 --num-spec 3 --dp 2
#   scripts/fia-tnd-seqlen-check.sh --paste serve.log --max-num-seqs 16 --num-spec 1
#   scripts/fia-tnd-seqlen-check.sh --query-t 512 --seqlen-last 4 --max-num-seqs 256 --num-spec 0 --mode piecewise
#
# Implementation sources (re-read before trusting on another release):
#   vllm v0.14.1 d7de043 config/vllm.py:1068-1170 (default capture sizes:
#     max = min(max_num_seqs * decode_query_len * 2, 512), then min with
#     max_num_batched_tokens; [1,2,4] + 8..min(max,255) step 8 + 256..max step 16)
#   vllm-ascend v0.14.0rc1 52d4acf worker/model_runner_v1.py:588-591
#     (pad_for_cudagraph), :439-487/:606 (DP-wide max), :856-881 (FULL +
#     uniform_decode padding of query_start_loc), attention/mla_v1.py:325-368
#     (FIA_SEQ_LEN_LIMIT = 16), :565-605 (graph_pad_size gates),
#     spec_decode/mtp_proposer.py:284, :434-448
#   Constraint text: RFC #3003 (history/<case>/3003.md:11-18), RFC #4300
set -u

read -r -d '' PY <<'PYEOF'
import argparse, re, sys

ap = argparse.ArgumentParser(description='FIA TND queryT vs actualSequenceLengthQ arithmetic')
ap.add_argument('--query-t', type=int, default=None, help='the queryT(N) number from the error')
ap.add_argument('--seqlen-last', type=int, default=None,
                help='the last element of actualSequenceLengthQ(M) from the error')
ap.add_argument('--paste', default=None, help='serve log / pasted error text (reads both numbers)')
ap.add_argument('--max-num-seqs', type=int, default=16, help='serve command --max-num-seqs')
ap.add_argument('--num-spec', type=int, default=0, help='num_speculative_tokens (0 = no MTP)')
ap.add_argument('--max-num-batched-tokens', type=int, default=16384)
ap.add_argument('--capture-sizes', default=None, help="explicit cudagraph_capture_sizes, e.g. '1,2,4,8,16,32'")
ap.add_argument('--dp', type=int, default=1, help='--data-parallel-size (engines that share the forwarding sync)')
ap.add_argument('--requests', type=int, default=None, help='requests in flight in the failing step, if known')
ap.add_argument('--mode', choices=['full', 'piecewise', 'eager'], default='full',
                help='cudagraph_mode.decode_mode(); eager/--enforce-eager removes the bucket')
a = ap.parse_args()

N, M = a.query_t, a.seqlen_last
if a.paste:
    try:
        text = sys.stdin.read() if a.paste == '-' else open(a.paste, errors='ignore').read()
    except OSError as e:
        sys.exit(f'cannot read {a.paste}: {e}')
    if N is None:
        hits = re.findall(r'queryT\((\d+)\)', text)
        if hits:
            N = int(hits[0])
    if M is None:
        hits = re.findall(r'actualSequenceLengthQ\((\d+)\)', text)
        if hits:
            M = int(hits[0])
if N is None or M is None:
    sys.exit('need both numbers: --query-t/--seqlen-last, or --paste with the '
             'CheckFASeqlenDataInTND line')

K = a.num_spec
dql = 1 + K
print(f'queryT(N)                 : {N}')
print(f'last(actualSequenceLengthQ): {M}   invariant: the two must be equal')
print(f'decode_query_len (1+K)    : {dql}   (num_speculative_tokens = {K})')

if a.mode != 'full':
    print('\nmode is not FULL decode: there is no padded graph bucket, so the two')
    print('numbers can only come from the metadata builder itself. Check the MTP')
    print('drafter path (1 token per request) before the target model.')

if a.capture_sizes:
    sizes = sorted({int(x) for x in a.capture_sizes.replace(' ', '').split(',') if x})
    origin = 'explicit --compilation-config cudagraph_capture_sizes'
else:
    cap = min(a.max_num_seqs * dql * 2, 512)
    cap = min(a.max_num_batched_tokens, cap)
    sizes = [i for i in (1, 2, 4) if i <= cap]
    if cap >= 8:
        sizes += list(range(8, min(cap + 1, 256), 8))
    if cap >= 256:
        sizes += list(range(256, cap + 1, 16))
    origin = (f'default list at max_num_seqs={a.max_num_seqs}, 1+K={dql}, '
              f'max_cudagraph_capture_size={cap}')
print(f'\ncapture sizes ({origin}):')
print('  ' + ','.join(str(s) for s in sizes))
if N == M:
    print('  -> N == M: no padding disagreement in these two numbers.')
elif N in sizes and a.mode == 'full':
    print(f'  -> {N} IS a capture bucket: the query rows were padded to the graph bucket.')
elif N in sizes:
    print(f'  -> {N} is a capture size, but mode={a.mode} has no FULL bucket that pads rows:')
    print('     read the rows as scheduler/drafter bookkeeping, not as a padded graph step.')
elif N > sizes[-1]:
    print(f'  -> {N} is above the largest bucket {sizes[-1]}: no graph for this step, so the')
    print('     padding came from another source (DP-wide max, or the drafter).')
else:
    nxt = min([s for s in sizes if s > N], default=None)
    print(f'  -> {N} is not a bucket (next one up: {nxt}).')
    if a.dp > 1:
        print('     With DP>1 this is the expected shape: a value raised to the DP-wide')
        print("     maximum, i.e. a peer rank's token count.")

print('\nwhat the pair can be:')
cands = []
def note(what, why):
    cands.append((what, why))

if a.requests and M % a.requests != 0:
    note(f'{a.requests} request(s) in flight but M={M} is not a multiple: M is a token count, '
         f'not the drafter arange', 'target model path, not the drafter')
if M == N:
    note(f'equal ({M} == {N}) - the op would not have refused; re-read the paste', 'not this error')
# M as a token count over r requests, for every r that divides it
rs = sorted({r for r in (a.requests, M, M // dql if K >= 0 and M % dql == 0 else None,
                         N // dql if M and N % dql == 0 else None)
             if r and r > 0 and M % r == 0})
for r in rs:
    q = M // r
    tag = []
    if q == 1:
        tag.append('1 token/request: the drafter (arange) shape, or a plain decode step')
    elif q == dql:
        tag.append(f'1+K tokens/request: uniform decode for the target model')
    else:
        tag.append('per-request query length')
    note(f'M={M} = {r} request(s) x {q} token(s) - ' + '; '.join(tag),
         'drafter' if q == 1 else ('target, uniform decode' if q == dql else 'target, partial schedule'))
if a.mode == 'full' and N in sizes and M != N:
    note(f'N={N} is a capture bucket, M={M} is not: the rows were padded and the per-request '
         f'cumsum was not',
         'the :856 FULL+uniform_decode gate did not fire (mixed step, non-uniform query '
         'lengths, or num_input_tokens above max_decode_tokens)')
if a.dp > 1 and N not in sizes:
    note(f'DP={a.dp} and N={N} is not a local bucket: num_input_tokens is the max over DP ranks',
         'DP-wide metadata sync (model_runner_v1.py:606) - the peer rank had the bigger batch')
if a.dp > 1:
    note(f'DP={a.dp}: even when N is a bucket, the DP max can lift a small local batch to it',
         'DP-wide metadata sync (model_runner_v1.py:606)')
if K > 1:
    note(f'K={K} > 1 with FULL decode graph is the path RFC #4300 built; its padding branch is '
         f'gated on graph_pad_size > num_reqs (mla_v1.py:565) and on '
         f'static[num_reqs] - actual[-1] > 16',
         'padded drafter batch (or the target) with MTP>1')
for what, why in cands:
    print(f'  - {what}')
    print(f'      => {why}')

print('\nhow to name the owner in one more call:')
print('  1. the Running/Waiting line of the failing step - the request count decides')
print('     "M is requests" (drafter) vs "M is tokens" (target).')
print('  2. --enable-log-requests / the per-step log: did the step hold prefills too?')
print('     A mixed step fails the uniform_decode gate at :856 while :590 already padded.')

print('\nladder (cheapest first):')
print('  1. cudagraph_mode PIECEWISE (or --enforce-eager): no bucket, N == M again.')
print('  2. num_speculative_tokens -> 1 (the family tutorial value).')
print('  3. pin cudagraph_capture_sizes to the tutorial list so bucket == captured metadata.')
print('  4. --data-parallel-size 1: removes the DP-wide max from the pair.')
print('  5. --enable-expert-parallel off (also changes the MoE/MC2 path).')
print('  6. --async-scheduling off last.')
PYEOF

uv run --no-project python -c "$PY" "$@"
