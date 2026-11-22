# `[WARNING] Please DO NOT tune args ['num_warps']!` — the autotune notice, not a fault

## Scope

The paste is warnings only — a few lines per worker, no traceback, on a hybrid
linear-attention model (Qwen3-Next / Qwen3-Coder-Next, the GDN families), with the
serving log above or below showing a normal response:

```text
(Worker_TP0 pid=109) [WARNING] Please DO NOT tune args ['num_warps']!
(Worker_TP1 pid=110) [WARNING] Please DO NOT tune args ['num_warps', 'num_stages']!
```

The title says 告警 / warning rather than 报错, and the body asks some form of
「这个告警是问题吗」. The deliverable is a verdict plus one check, not a ranked cause
list. Leave this page when the same block sits above an error (section below), and
when the report instead carries a `MLIRCompilationError` traceback running through
`triton/runtime/autotuner.py` — that is `../image-dependency-faults/triton-ascend-pair.md`,
a request-killing compiler fault that merely shares the autotuner's file name.

## Answer rule

**The line is the triton autotune layer reporting that `num_warps`/`num_stages` are
not tunable on this backend; nothing in vllm-ascend raises it and nothing is fixed by
it.** Say that in the first sentence, add that the reporter's own log already shows
the serve answering (`POST /v1/chat/completions … 200 OK`), and offer no lever:
`--enforce-eager`, `--dtype`, an HCCL env variable, a triton-ascend upgrade and the
`FULL_DECODE_ONLY` capture flags are all spent here, and `../accuracy-benchmarks/`
is the wrong track. Then ask which symptom they actually saw — the warning is not one.

## Why it is benign (verified at vllm-ascend v0.14.0rc1 / vllm v0.14.1)

The two message variants are upstream vLLM's two autotune config shapes, and the
bracketed lists in the paste are exactly their tuned keys:

| Variant | Upstream site at vllm `d7de043` (v0.14.1) |
|---|---|
| `['num_warps']` | `vllm/model_executor/layers/fla/ops/l2norm.py:23`, `:52`; `cumsum.py:24`, `:78`; `kda.py:595` |
| `['num_warps', 'num_stages']` | `fla/ops/chunk_o.py:33`, `chunk_delta_h.py:34`, `chunk_scaled_dot_kkt.py:27`, `solve_tril.py:31,106,231`, `wy_fast.py:22`, `kda.py:485,782,982,1237` |

- **Downstream does not tune, so it is not the emitter.** vllm-ascend v0.14.0rc1
  (`52d4acf`) has **zero** `autotune` occurrences in its own tree
  (`grep -rl autotune vllm_ascend/` → 0 files), and its GDN copies are plain
  `@triton.jit` + `@triton.heuristics` with `num_warps=`/`num_stages=` passed at
  launch (`vllm_ascend/ops/triton/fla/chunk_o.py:19-25`, `chunk_delta_h.py:213-233`).
  That is one call, and it is the call that ends a repo-wide hunt in the downstream
  tree.
- **The string is in no exported tree**, so do not spend a second call on it: not
  `triton-lang/triton` v3.5.1, not the `triton-lang/triton-ascend` export (v3.2.1,
  itself truncated), not `vllm/`, not `vllm_ascend/`. It prints from the **installed
  wheel** — which is also why the line carries no timestamp and no `[file.py:line]`,
  while every vLLM `logger` line carries both. Keep the attribution hedged ("the
  wheel that ships the tuner") and hand the reporter the grep. Recorded #6562 spent
  four calls naming the exact upstream/downstream kernel that fired; that changes no
  rung of the answer.
- **The count is not diagnostic.** One line per distinct autotuned kernel entered in
  that process, per worker: 1×warp-only + 2×warp+stages is three sites, not three
  faults, and all TP ranks print the same set. Never quote it as severity.

## The trap: the same block sits directly above a real crash

`scripts/history-titles.sh -i 'num_warps|DO NOT tune|autotune'` returns **0 hits** —
the token lives in bodies, so body-search it once: `-b 'DO NOT tune'` returns **six**
issues, and five of them are about something else entirely (#2976 an aicore
`kernel execute failed`, #3219 a self-compiled triton, #3556 a
`PagedAttentionOperation` error, #8990 a `FLASHCOMM1` refusal, #11083
`--enable-log-outputs`) — that is the ambient-line evidence, and it is why a paste's
title, not the block, decides the subject. The sixth is **#3548** (Qwen3-Next,
v0.10.2rc1), where the identical block immediately precedes
`WorkerProc hit an exception` and `EE1001 … coreDim=66976 can't be greater than
UINT16_MAX` in `layer_norm_fwd_kernel`. There the warning is not the cause:
`66976/8 = 8372` tokens is the grid arithmetic and the owner is
`../custom-ops-runtime-faults/triton-grid-uint16-limit.md`. So:

- a paste with any error under the block makes this page **one line** of the answer
  ("the block above is autotune noise"), and that error is classified normally;
- a paste with no error is this page, whole.

## Deliverable

Verdict, one provenance command, then the ask:

```bash
python -c "import triton,os;print(os.path.dirname(triton.__file__))"
rg -n 'DO NOT tune' "$(python -c 'import triton,os;print(os.path.dirname(triton.__file__))')"
```

A hit quotes the emitting site and ends the case; no hit means the wheel keeps it in
a packaged module, and the verdict rests on the rules above plus the reporter's own
`200 OK`. Scope: vllm-ascend v0.14.0rc1 × vllm v0.14.1, Qwen3-Next/Qwen3-Coder-Next
on 910B3 (recorded #6562, `v0.14.0rc1-openeuler` image). Re-grep the fla config
shapes before reusing the table at another pin — the two-variant split, not the line
numbers, is what has to hold.
