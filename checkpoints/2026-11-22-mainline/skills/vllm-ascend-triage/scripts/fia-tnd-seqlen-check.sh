#!/usr/bin/env bash
# Classify a npu_fused_infer_attention_score TND refusal by arithmetic:
#
#   EZ9999 ... When layout is TND, queryT(16) must be equal to the last element
#   of actualSequenceLengthQ(4) [FUNC: CheckFASeqlenDataInTND]
#
# The op compares two numbers that different padding bookkeepings produce. This
# prints the capture-size list the serve command actually resolves to, the
# **request count implied by the pair** (the v0.19+ helper decides its branch from
# exactly these numbers), the older drawings of the pair, and the ordered
# ablations — so the owner is named from the numbers instead of a source walk.
#
#   scripts/fia-tnd-seqlen-check.sh --query-t 16 --seqlen-last 13 --max-num-seqs 200 --num-spec 3 --dp 2
#   scripts/fia-tnd-seqlen-check.sh --query-t 16 --seqlen-last 4 --max-num-seqs 16 --num-spec 3 --dp 2
#   scripts/fia-tnd-seqlen-check.sh --paste serve.log --max-num-seqs 200 --num-spec 3
#
# Implementation sources (re-read before trusting on another release):
#   vllm v0.19.1 b1388b1 config/vllm.py:1343-1435 (default capture sizes),
#     config/compilation.py:1237-1283 (explicit sizes rounded to multiples of 1+K)
#   vllm-ascend v0.19.1rc1 da421af worker/model_runner_v1.py:566-604
#     (_pad_query_start_loc_for_fia: FULL forces num_reqs_padded = num_reqs at
#     :580-582, guard at :586, mixed branch writes last = num_tokens_padded at
#     :594-600), call site :1628-1630, query_start_loc_cpu at :2473;
#     worker/v2/model_runner.py:395-431 (twin, self.decode_query_len);
#     _310p/model_runner_310p.py:139-160 (ngram override, uniform_decode_query_len=1);
#     attention/attention_v1.py:322 (actual_seq_lengths_q), :591 (workspace key only)
#   vllm-ascend v0.14.0rc1 52d4acf worker/model_runner_v1.py:588-591 (pad_for_cudagraph),
#     :606 (DP-wide max), :856-881 (FULL + uniform_decode gate), attention/mla_v1.py:565-605,
#     spec_decode/mtp_proposer.py:284 (Lane A: the pre-refactor sites)
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
ap.add_argument('--num-spec', type=int, default=0, help='num_speculative_tokens (0 = no spec decode)')
ap.add_argument('--method', default=None, help='speculative method (mtp / ngram / eagle / draft_model)')
ap.add_argument('--max-num-batched-tokens', type=int, default=16384)
ap.add_argument('--capture-sizes', default=None, help="explicit cudagraph_capture_sizes, e.g. '1,2,4,8,32'")
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
    print('\nmode is not FULL decode: no padded graph bucket exists, so read the rows and')
    print('the cumsum as metadata-builder output (drafter arange, or a mixed batch).')

if a.capture_sizes:
    sizes = sorted({int(x) for x in a.capture_sizes.replace(' ', '').split(',') if x})
    origin = 'explicit cudagraph_capture_sizes (upstream rounds these to multiples of 1+K)'
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
    print('  -> N == M: these two numbers do not disagree; re-read the paste.')
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

# --- the branch the v0.19+ helper took, from the pair alone -------------------
print('\nv0.19+ helper `_pad_query_start_loc_for_fia` (worker/model_runner_v1.py:566-604,')
print('worker/v2/model_runner.py:395-431): under FULL it sets num_reqs_padded = num_reqs')
print('(:580-582), then:\n'
      '  :586  if num_tokens_padded == num_reqs_padded * (1+K):   # == the bucket vs num_reqs*(1+K)\n'
      '  :587-593    uniform branch: fills only indices num_reqs+1..num_reqs_padded,\n'
      '              so with num_reqs_padded == num_reqs the last entry stays the REAL cumsum\n'
      '  :594-600    else: append a dummy request with last = num_tokens_padded\n')
print('=> the else-branch ALWAYS makes last == queryT, so a refusal on a FULL step means the')
print('   uniform branch ran:  N == r*(1+K),  M < N,  r = N/(1+K),  M - r = accepted drafts.\n')

laneB = False
if a.mode == 'full' and N != M and N % dql == 0 and N // dql >= 1:
    r = N // dql
    drafts = M - r
    fit_r = r <= a.max_num_seqs
    fit_d = 0 <= drafts <= K * r if K > 0 else drafts == 0
    if fit_r and fit_d:
        laneB = True
        print(f'  LANE B FITS: r = N/(1+K) = {N}/{dql} = {r} request(s) in flight,')
        print(f'               accepted drafts M - r = {M} - {r} = {drafts} '
              f'(range 0..{K * r})')
        if drafts > 0:
            print(f'               => the step ran {r} decode request(s) whose drafts did not all')
            print(f'                  reach K={K}, i.e. the bucket {N} coincided with {r}*(1+K).')
            print('                  This is a Lane B fingerprint: `M - r > 0` rules out the')
            print('                  drafter/1-token reading below.')
        else:
            print(f'               => M == r, so the pair reads EITHER as {r} request(s) whose')
            print('                  drafts all came out empty (Lane B) OR as the drafter/1-token')
            print('                  arange of a Lane A ref. The pair alone cannot separate them:')
            print('                  decide by (a) the ref (`rg -n _pad_query_start_loc_for_fia`')
            print('                  in vllm_ascend/ finds the helper only from v0.19), (b) whether')
            print('                  the failing op belongs to the drafter or the target forward,')
            print('                  and (c) `Running:` = r plus zero accepted drafts in the')
            print('                  `SpecDecoding metrics` line of the same step.')
        if a.requests is not None and a.requests != r:
            print(f'               WARNING: --requests {a.requests} disagrees with r={r}; the pair')
            print('               then belongs to another lane, not to this branch.')
    else:
        why = []
        if not fit_r:
            why.append(f'r={r} exceeds max_num_seqs={a.max_num_seqs}')
        if not fit_d:
            why.append(f'drafts={drafts} outside 0..{K * r}')
        print(f'  Lane B does NOT fit: N/(1+K) = {r}, ' + '; '.join(why))
        print('      => this pair cannot come from the v0.19 uniform branch; use the Lane A rows.')
elif a.mode == 'full' and N != M:
    print(f'  Lane B impossible: N={N} is not a multiple of 1+K={dql} (or N < 1+K), so the')
    print('      guard at :586 could not have been true. Use the Lane A rows below.')
elif a.mode == 'full' and dql == 1:
    print('  Lane B cannot fire at K = 0: with 1+K = 1 the guard reduces to bucket == num_reqs')
    print('      and the last entry is that same count, so nothing disagrees.')

print('\nother drawings of the pair (Lane A, v0.14-era sites; or a different metadata builder):')
cands = []
def note(what, why):
    cands.append((what, why))

if a.requests and M % a.requests != 0:
    note(f'{a.requests} request(s) in flight but M={M} is not a multiple: M is a token count, '
         f'not the drafter arange', 'target model path, not the drafter')
if M == N:
    note(f'equal ({M} == {N}) - the op would not have refused; re-read the paste', 'not this error')
rs = sorted({r for r in (a.requests, M, M // dql if M % dql == 0 else None,
                         N // dql if N % dql == 0 else None)
             if r and r > 0 and M % r == 0})
for r in rs:
    q = M // r
    if laneB and (M - N // dql) > 0:
        # Lane B has a positive accepted-draft count, so its request count is pinned;
        # a reading that needs a different request count is refuted, not an alternative.
        rr = N // dql
        note(f'M={M} = {r} request(s) x {q} token(s) would need {r} requests, but Lane B '
             f'derives {rr} from N/(1+K) with {M - rr} accepted draft(s) - refuted for this pair',
             'do not offer the drafter/1-token reading on a Lane B step')
        continue
    tag = []
    if q == 1:
        tag.append('1 token/request: the drafter (arange) shape, or a plain decode step')
    elif q == dql:
        tag.append('1+K tokens/request: uniform decode for the target model')
    else:
        tag.append('per-request query length')
    note(f'M={M} = {r} request(s) x {q} token(s) - ' + '; '.join(tag),
         'drafter' if q == 1 else ('target, uniform decode' if q == dql else 'target, partial schedule'))
if not laneB:
    if a.mode == 'full' and N in sizes and M != N:
        note(f'N={N} is a capture bucket, M={M} is not: rows padded, per-request cumsum not',
             'Lane A classic: the :856 FULL+uniform_decode gate did not fire at a v0.13-v0.14 ref')
    if a.dp > 1 and N not in sizes:
        note(f'DP={a.dp} and N={N} is not a local bucket: num_input_tokens is the max over DP ranks',
             'DP-wide metadata sync (model_runner_v1.py:606 at v0.14, :520-558 today)')
    if a.dp > 1:
        note(f'DP={a.dp}: even when N is a bucket, the DP max can lift a small local batch to it',
             'DP-wide metadata sync (model_runner_v1.py:520-558)')
    if K > 1:
        note(f'K={K} > 1 under a FULL decode graph is the path RFC #4300 built; its padded branch '
             f'is gated on graph_pad_size > num_reqs and static[num_reqs] - actual[-1] > 16',
             'padded drafter batch (mla_v1.py:565-605 at v0.14) or the target')
for what, why in cands:
    print(f'  - {what}')
    print(f'      => {why}')

print('\nhow to name the owner in one more call:')
print(f'  1. the Running/Waiting line of the failing step. On Lane B it must read '
      f'Running: {N // dql if N % dql == 0 else "N/(1+K)"}')
print('     if the helper is the owner; a different count refutes that lane outright.')
print('  2. --enable-log-requests / the per-step log: did the step hold prefills too?')
print('     A mixed step fails the uniform-decode gate while the rows were already padded.')
print('  3. the reporter\'s own ablation is evidence: spec off + FULL fine, spec + eager fine,')
print('     both together crash => the padding BRANCH, not the kernel (Lane B fingerprint).')

print('\nladder (cheapest first):')
print('  1. cudagraph_mode PIECEWISE (or --enforce-eager): no bucket, N == M again.')
print('  2. num_speculative_tokens -> 1; on Lane B the sharper A/B is method ngram -> mtp')
print('     (fixed K drafts) with FULL kept on: survives => the variable draft length owns it.')
print('  3. pin cudagraph_capture_sizes: on Lane B this only moves which buckets exist.')
print('  4. --data-parallel-size 1: removes the DP-wide max from the pair.')
print('  5. --enable-expert-parallel off (also changes the MoE/MC2 path).')
print('  6. --async-scheduling off last.')
PYEOF

uv run --no-project python -c "$PY" "$@"
