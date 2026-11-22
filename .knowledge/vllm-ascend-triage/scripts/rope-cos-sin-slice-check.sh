#!/usr/bin/env bash
# Classify a host-side rope reshape by arithmetic:
#
#   File ".../vllm_ascend/ops/triton/rope.py", line 155, in rope_forward_triton
#       cos = cos.view(num_tokens, -1)
#   RuntimeError: shape '[6, -1]' is invalid for input of size 512
#
# `cos` is the process-global slice from get_cos_and_sin_slice(), so the message is a
# disagreement between this forward's token count and the count of whichever forward
# last called update_cos_sin. This prints the two counts, the width the allocation
# implies, the pad map the launch resolves to (pad_for_cudagraph), and which request
# counts raise vs silently produce wrong rope — so the owner is named from the numbers
# instead of a model-constant hunt.
#
#   scripts/rope-cos-sin-slice-check.sh --num-tokens 6 --size 512 \
#       --capture-sizes 1,4,8,16 --num-spec 1 --max-num-seqs 16
#   scripts/rope-cos-sin-slice-check.sh --paste serve.log --capture-sizes 1,4,8,16 --num-spec 1
#   scripts/rope-cos-sin-slice-check.sh --num-tokens 2 --size 256 --head-size 128 --partial 0.5
#
# Implementation sources (re-read before trusting on another release):
#   vllm-ascend v0.14.0rc1 52d4acf ops/rotary_embedding.py:87-97 (_cos width),
#     :147-165 (update_cos_sin, _cos_slice at :163), :167-168 (getter), :189 read,
#     :218-232 triton branch (:221 cos.view(-1, rotary_dim), :226 the op call);
#     ops/triton/rope.py:142-155 (num_tokens = q.shape[0], the reshape at :155);
#     worker/model_runner_v1.py:392-393 (_use_aclgraph), :587-591 (target pad),
#     :1375 / :2067 (update_cos_sin, target padded positions);
#     spec_decode/eagle_proposer.py:138-141 (mtp + async -> no drafter aclgraph),
#     :326 / :506 / :670 (EAGLE refreshes the global), spec_decode/mtp_proposer.py:131,
#     :245-249 (drafter token count), :319-321 (model call, no refresh)
#   vllm v0.14.1 d7de043 config/vllm.py:339-344 (pad_for_cudagraph),
#     config/compilation.py:1126-1138 (bs_to_padded_graph_size),
#     v1/worker/gpu_model_runner.py:621 (uniform_decode_query_len = 1 + num_spec),
#     :2697 (positions sliced to the padded num_input_tokens);
#     model_executor/models/glm4_moe.py:287 (rope_parameters partial 0.5 -> rotary_dim 64)
set -u

read -r -d '' PY <<'PYEOF'
import argparse, re, sys

ap = argparse.ArgumentParser(description='rope cos/sin slice vs token-count arithmetic')
ap.add_argument('--num-tokens', type=int, default=None, help="num_tokens from shape '[N, -1]'")
ap.add_argument('--size', type=int, default=None, help='input size from the message (cos.numel())')
ap.add_argument('--paste', default=None, help='serve log / pasted traceback (reads both numbers)')
ap.add_argument('--width', type=int, default=None,
                help='_cos last dim, if known (overrides --head-size/--partial)')
ap.add_argument('--head-size', type=int, default=128, help='model head_size (config head_dim)')
ap.add_argument('--partial', type=float, default=0.5,
                help='hf_text_config.partial_rotary_factor, if the checkpoint carries it')
ap.add_argument('--capture-sizes', default='1,4,8,16',
                help="the launch's cudagraph_capture_sizes, e.g. '1,4,8,16'")
ap.add_argument('--num-spec', type=int, default=1, help='num_speculative_tokens')
ap.add_argument('--max-num-seqs', type=int, default=16, help='serve command --max-num-seqs')
a = ap.parse_args()

N, M = a.num_tokens, a.size
if a.paste:
    try:
        text = sys.stdin.read() if a.paste == '-' else open(a.paste, errors='ignore').read()
    except OSError as e:
        sys.exit(f'cannot read {a.paste}: {e}')
    if N is None:
        hits = re.findall(r"shape '\[(\d+), -1\]'", text)
        if hits:
            N = int(hits[0])
    if M is None:
        hits = re.findall(r'invalid for input of size (\d+)', text)
        if hits:
            M = int(hits[0])
if N is None or M is None:
    sys.exit('need --num-tokens and --size (or a --paste carrying both)')

sizes = [int(x) for x in str(a.capture_sizes).replace(' ', '').split(',') if x]
sizes = sorted(set(sizes))
cap = sizes[-1]

part = f'head_size x partial {a.partial}' if a.partial and a.partial != 1.0 else 'head_size'
cands = [('rotary_dim (partial visible on hf_text_config)', int(a.head_size * (a.partial or 1.0))),
         ('head_size (attribute absent -> allocation at head size)', a.head_size)]
if a.width:
    cands = [(f'explicit --width {a.width}', a.width)]

print(f'message            num_tokens(N)={N}  cos.numel()(M)={M}')
print(f'scheduled tokens   a decode step with K={a.num_spec} schedules {1 + a.num_spec} per request')
print()
print('width candidates (set_cos_and_sin reads hf_text_config.partial_rotary_factor):')
for name, w in cands:
    if w <= 0 or M % w:
        print(f'  {name:56s} w={w:4d}  M % w != 0 -> not this width')
        continue
    rows = M // w
    if rows == N:
        verdict = 'rows == num_tokens -> the slice matches; not this page'
    elif M % N:
        verdict = 'ROWS != num_tokens and M % N != 0 -> the pasted RuntimeError'
    else:
        verdict = 'ROWS != num_tokens but M % N == 0 -> no exception, WRONG rope (silent)'
    print(f'  {name:56s} w={w:4d}  rows={rows:4d}  {verdict}')
print('  (the width that matters is the one whose rows equal a *padded* token count below)')
print()

tbl = {}
for end, start in zip(sizes + [cap + 1], [0] + sizes):
    for bs in range(start, end):
        tbl[bs] = start if bs == start else end


def padded(total: int) -> int:
    return tbl[total] if total <= cap else total


print(f'pad map for cudagraph_capture_sizes={sizes} (pad_for_cudagraph up to {cap}):')
gaps = sorted({v for v in tbl.values() if v})
print('  buckets ' + ', '.join(str(g) for g in gaps))
print()
w = cands[0][1] if not a.width else a.width
print(f'per request count (K={a.num_spec}, capture sizes {sizes}, width {w} for numel):')
print('  reqs  tokens  padded  cos.numel  outcome')
for r in range(1, max(1, a.max_num_seqs) + 1):
    tok = r * (1 + a.num_spec)
    pad = padded(tok)
    numel = pad * w
    if pad == tok:
        out = 'correct'
    elif numel % tok:
        out = 'RAISES cos.view(num_tokens, -1)'
    else:
        out = 'no exception, wrong rope'
    mark = '  <-- this step' if tok == N else ''
    print(f'  {r:4d}  {tok:6d}  {pad:6d}  {numel:9d}  {out}{mark}')
print()
print('next calls:')
print("  rg -n 'update_cos_sin' vllm_ascend/    # who refreshes the global; mtp_proposer has no hit at v0.14.0rc1")
print('  then the ladder: 1/2/3 concurrent requests -> --async-scheduling off -> dense capture sizes')
PYEOF

exec uv run --no-project python -c "$PY" "$@"
