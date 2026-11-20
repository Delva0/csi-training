# Startup failures (engine/worker never becomes ready)

Scope: the process tree dies or stalls before the server accepts requests —
`EngineCore failed to start`, `WorkerProc initialization failed`, model load or
memory-profiling windows, OOM, wrong/unsupported config, version mismatch.

Inherit the rules in SKILL.md. Conditional routes:

- No per-rank worker traceback in the pasted log → `silent-worker-exit.md`
  (how the vLLM v1 multiproc executor reports each lane, what to request, and
  why `Parent process exited` / `TBE Subprocess` lines are consequences).
- Failure at load/profile/graph-capture time on a machine with 64 GB cards, or a
  deployment command that omits `--max-model-len` → `max-model-len-budget.md`
  first. This is the reporter-confirmed answer for the recorded Qwen3-VL-30B-A3B
  case.
- Suspicion of "model X is unsupported", wrong image/vllm pairing, or an
  obvious maintainer "works on my machine" reply → `environment-correspondence.md`.
- If the log names a collective (`HcclAllreduce`, `hcclRet`, `ERR00100`) instead
  of a Python-level cause, leave this topic and use
  `../hccl-and-hardware/index.md`.

Common rules for this topic:

1. Establish which lane the failure is in before reading implementation source.
2. Ask for the complete per-rank log plus `npu-smi info`, the exact `vllm serve`
   command, the image tag, and the `Using max model len N` line. Trimmed logs are
   the norm; name the specific lines you need.
3. A start-up flag difference (explicit `--max-model-len`, `--enforce-eager`,
   `--compilation-config`) is cheaper to test than a source-level theory. Ask for
   the re-run before proposing a bug.
