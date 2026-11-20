#!/usr/bin/env bash
# Price the eager `max_model_len x max_model_len` attention mask that
# vllm-ascend builds on the HOST in every worker at engine start, and decide
# whether it alone can take the box down (#2873, #3002).
#
#   scripts/attn-mask-budget.sh --max-model-len 262144 --ranks 8 --host-gib 1024
#   scripts/attn-mask-budget.sh --log <serve.log>          # reads the two lines
#   scripts/attn-mask-budget.sh --max-model-len 131072    # ranks default 1
#
# Implementation source (verified at the pin, re-read before trusting another
# release):
#   v0.10.1rc1 7e16b4a  vllm_ascend/worker/model_runner_v1.py:234
#                       (AttentionMaskBuilder(self.model_config.max_model_len,
#                        self.dtype) inside NPUModelRunner.__init__)
#   v0.10.1rc1 7e16b4a  vllm_ascend/attention/attention_mask.py:19-48
#                       (_generate_attn_mask: bool ones -> tril -> ~ ->
#                        float32 zeros -> masked_fill -> .to(dtype))
# The dtype of the mask is the model dtype (bf16 for a bf16 checkpoint), and
# the whole construction is CPU-side: no device memory is touched, so `npu-smi`
# never sees it.
#
# Peak = 9 * N**2 bytes per rank (2 N^2 bool copies + 2 float32 4 N^2 buffers),
# retained = 2 * N**2 bytes per rank for bf16/fp16. Both are linear in the rank
# count because every worker builds its own copy.
set -u

uv run --no-project python - "$@" <<'PY'
import argparse
import math
import re
import sys

GIB = 1024 ** 3
DTYPE_BYTES = {'bf16': 2, 'fp16': 2, 'fp32': 4}
PEAK_FACTOR = 9          # bytes per element at the peak of _generate_attn_mask

ap = argparse.ArgumentParser(description='host attention-mask budget')
ap.add_argument('--max-model-len', type=int, default=None,
                help='effective max model len (the log line "Using max model len N")')
ap.add_argument('--ranks', type=int, default=None,
                help='worker processes that build a mask on this host (TP x DP x PP)')
ap.add_argument('--dtype', choices=sorted(DTYPE_BYTES), default='bf16')
ap.add_argument('--host-gib', type=float, default=None,
                help='host RAM total, for the verdict and the fitting N')
ap.add_argument('--headroom', type=float, default=0.5,
                help='fraction of host RAM the masks may use (default 0.5)')
ap.add_argument('--log', default=None,
                help='parse "Using max model len N" and the rank count (world size / '
                     'tensor_parallel_size / -tp) from a serve log')
a = ap.parse_args()

log_text = ''
if a.log:
    with open(a.log, encoding='utf-8', errors='replace') as fh:
        log_text = fh.read()

n = a.max_model_len
if n is None and log_text:
    m = re.search(r'Using max model len (\d+)', log_text)
    if m:
        n = int(m.group(1))
if n is None:
    sys.exit('error: need --max-model-len N or a --log containing "Using max model len N"')

ranks = a.ranks
if ranks is None and log_text:
    # worker count on the node, in order of reliability: the rank-assignment
    # lines every worker prints, then the parsed engine args, then the CLI form
    m = re.search(r'in world size (\d+)', log_text) or \
        re.search(r"'tensor_parallel_size':\s*(\d+)", log_text) or \
        re.search(r'tensor_parallel_size[=:\s]+(\d+)', log_text) or \
        re.search(r'--tensor-parallel-size[= ](\d+)', log_text) or \
        re.search(r'-tp\s+(\d+)', log_text)
    if m:
        ranks = int(m.group(1))
if ranks is None:
    ranks = 1

if n < 1 or ranks < 1:
    sys.exit('error: --max-model-len and --ranks must be >= 1')

elem = DTYPE_BYTES[a.dtype]
peak_rank = PEAK_FACTOR * n * n
keep_rank = elem * n * n
peak_all = peak_rank * ranks
keep_all = keep_rank * ranks

print(f'max model len        : {n}  (dtype {a.dtype}, {ranks} rank(s) on this host)')
print(f'peak per rank        : {peak_rank / GIB:8.1f} GiB   <- bool pair + two float32 buffers live together')
print(f'retained per rank    : {keep_rank / GIB:8.1f} GiB   <- the mask that stays resident')
print(f'peak, all ranks      : {peak_all / GIB:8.1f} GiB')
print(f'retained, all ranks  : {keep_all / GIB:8.1f} GiB')
print()

if a.host_gib:
    budget = a.host_gib * a.headroom
    ratio = peak_all / GIB / a.host_gib
    print(f'host RAM             : {a.host_gib:.0f} GiB  (budget {budget:.0f} GiB = '
          f'{a.headroom * 100:.0f} % for the masks)')
    print(f'peak as share of RAM : {ratio * 100:.0f} %')
    if ratio >= 0.5:
        print('verdict: the mask construction alone cannot fit — the box thrashes or the')
        print('         kernel OOM-kills the ranks before any weight is loaded. This is an')
        print('         allocation-size bug at this max-model-len, not a leak and not the card.')
    elif ratio >= 0.2:
        print('verdict: large. It is a first-class candidate only if the log also stops before')
        print('         "Starting to load model" and the host went down; else keep it as headroom.')
    else:
        print('verdict: the mask is not the host-memory cause at this max-model-len.')
    fit = int(math.sqrt(budget * GIB / (PEAK_FACTOR * ranks)))
    print(f'largest fitting len  : {fit}  (~{fit // 1024}K) at {ranks} rank(s), '
          f'{a.headroom * 100:.0f} % headroom')
    print()

print('reference table (bf16/fp16, peak per rank):')
for size in (8192, 16384, 32768, 65536, 131072, 262144):
    print(f'  {size:>7}  peak {PEAK_FACTOR * size * size / GIB:7.1f} GiB   '
          f'retained {2 * size * size / GIB:6.1f} GiB')
print()
print('next: `free -g` + `dmesg -T | grep -iE \'oom|kill\'` on the host that died, then one')
print('re-run with --max-model-len 32768 (or 8192) on the same command; compare with a')
print('release before the mask refactor (#2442, v0.10.1rc1 release note).')
PY
