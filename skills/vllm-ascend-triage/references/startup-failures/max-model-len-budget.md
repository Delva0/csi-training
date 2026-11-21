# Check `max_model_len` before anything else at load time

Reports that read "the model could not load / the process crashed during startup"
with an unclear or missing worker error are frequently a context-length budget
problem, and the budget is invisible because the flag was simply omitted.

## How the default is produced

vllm derives `max_model_len` from the HF config when `--max-model-len` is absent:
it takes the smallest of `max_position_embeddings`, `n_positions`,
`max_seq_len`, `seq_length`, `model_max_length`, `max_target_positions`,
`max_sequence_length`, `max_seq_length`, `seq_len`, then the tokenizer's
`model_max_length` (`_get_and_verify_max_len` in `vllm/config/model.py`,
v0.11.0rc3, lines 1766+). It logs `Using max model len N`
(`vllm/config/model.py:1510`).

For long-context VL/MoE checkpoints that default is the model's advertised
context — hundreds of thousands of tokens — which is far beyond what a 64 GB
card can hold together with weights for a given `--gpu-memory-utilization`.

## Reporter-confirmed case (vllm-ascend issue #3543)

- Environment: image `quay.io/ascend/vllm-ascend:v0.11.0rc0` (vllm-ascend
  v0.11.0rc0 + vllm v0.11.0rc3), 910B3, `--tensor-parallel-size 4`,
  `--max-num-batched-tokens 4096`, `--gpu-memory-utilization 0.85`, **no
  `--max-model-len`**, model explicitly named `Qwen3-VL-30B-A3B-Instruct`.
- Symptom: rank assignment is logged, then a long silence, then the parent-only
  failure lane (`WorkerProc initialization failed ...`, no worker traceback) plus
  `TBE Subprocess ... main process disappeared`. See `silent-worker-exit.md`.
- Reporter's own finding: on the 910B3 machine `max_model_len` took the model's
  default maximum, which prevented the model from loading and crashed the
  process; after lowering `max_model_len` the model loaded and the service
  started normally. (Reporter-confirmed, not derived by us.)
- Corroboration available in the same release's docs (pinned vllm-ascend
  source): `docs/source/tutorials/multi_node_qwen3vl.md` passes `--max-model-len 32768`
  in both node commands for `Qwen3-VL-235B-A22B-Instruct`. The failing command
  passed none. Diffing a reported command against the shipped tutorial/CI command
  is a cheap, high-yield step.

## What to do

1. Ask for / read the `Using max model len N` line, or compute the default from
   `config.json` (`max_position_embeddings`) — one file read, no guessing.
2. Have the reporter re-run with an explicit modest budget, e.g.
   `--max-model-len 32768` (keep `--max-num-batched-tokens` at or below it). If
   the service starts, the default was the problem; no code defect needs filing.
   **Ask first whether the task needs the advertised context.** When the report
   says it does ("上下文长度得256K才行", a 256K/1M-context workload), this rung is a
   non-answer and the context-preserving levers below have to lead instead.
3. Keep the same explicitness in any suggested command you publish, and say why
   (the tutorials already do it).

## The explicit error: `Available memory: 0` is the profiling peak, not the card (recorded #3843)

The same omitted flag also produces a *named* error instead of a silent exit:
```
ValueError: No available memory for the cache blocks. Try increasing
`gpu_memory_utilization` when initializing the engine.
```
(upstream `check_enough_kv_cache_memory`, `vllm/v1/core/kv_cache_utils.py:698-701`
at vllm v0.11.0). The worker's line right above it is
`Available memory: 0, total memory: <bytes>` (`vllm_ascend/worker/worker_v1.py:235`),
which is `max(total × gpu_memory_utilization − peak, 0)` (`:226-233`) — a
*computed* number, not "the card is full". The `peak` term is weights **plus the
profiling dummy forward**, and on a non-MLA model at these versions the dummy
batch is sized by `max_model_len`:

| term | anchor at vllm-ascend v0.11.0rc0 `00ba0710` + vllm v0.11.0 `b8b302cd` |
|---|---|
| budget = `total × utilization − peak`, clamped at 0 | `vllm_ascend/worker/worker_v1.py:226-235` |
| chunked prefill is force-disabled for non-MLA | `vllm_ascend/platform.py:139-144` (the log line is `Non-MLA LLMs forcibly disable the chunked prefill feature`) |
| then `max_num_batched_tokens = max_model_len` | `vllm_ascend/platform.py:160-163`, skipped only when `ascend_scheduler_config.enable_chunked_prefill` is true |
| the profile batch *is* that number | `model_runner_v1.py:254` (`self.max_num_tokens`) and `:2506-2517` (`_dummy_run(self.max_num_tokens, with_prefill=True)`) |
| with chunked prefill off the default is `max(max_model_len, 2048)` | `vllm/config/scheduler.py:171-183` (at v0.11.0) |

So **raising `--gpu-memory-utilization` is not a discriminator here.** #3843's
reporter did 0.5 → 0.9 and reported the identical error ("没用，都试了"): at
262144 a dummy prefill leaves the available term at 0 at either ratio, because
the peak scales with the batch, not with the ratio. Recorded arithmetic: TP8 on
8×64 GB, weights 18.7767 GB/rank, `total 65 452 113 920` ⇒ 0.5 gives 32.7 GB, so
the 262144-token dummy run transiently needed ≥ ~14 GB *above* the weights — and
0.9 still has to fit that same batch inside 58.9 GB.

**Two gates, two messages, and the text decides the lever** — ask for the verbatim
line, it is cheaper than any source walk:

- `available_memory <= 0` → "No available memory for the cache blocks"
  (`kv_cache_utils.py:698-701`): **peak-dominated** (weights + profiling batch).
- `needed_memory > available_memory` → "To serve at least one request with the
  models's max seq len (N), (X GiB KV cache is needed, which is larger than the
  available KV cache memory (Y GiB)) … the estimated maximum model length is E"
  (`:704-723`): **KV-volume-dominated**, and `E` is the number to negotiate
  against. A report can move from gate 1 to gate 2 as the ratio goes up without
  the server ever starting.

`--max-num-batched-tokens` alone cannot cap the profile batch while chunked
prefill is off: `platform.py:160-163` raises it back to `max_model_len`. That is
why "just add `--max-num-batched-tokens 8192`" is wrong advice at these versions.

**When the reporter needs the advertised context** (256K here), rank the
context-preserving levers, each one restart, before touching `--max-model-len`:

1. `--additional-config '{"ascend_scheduler_config": {"enabled": true,
   "enable_chunked_prefill": true}}'` + `--max-num-batched-tokens 8192`, leaving
   `--max-model-len` at the checkpoint value: this skips the rewrite, so the dummy
   prefill is 8192 tokens while the pool still has to cover 262144. Applicability:
   text-only models with `structured_outputs_config.backend == "auto"` and no
   `scheduler_delay_factor` (`platform.py:139-150` skips multimodal models).
   `AscendSchedulerConfig` copies unknown keys from `additional_config`
   (`vllm_ascend/ascend_config.py:168-174`), so the key is accepted as written;
   the cost is `platform.py:155-157`'s operator-path warning, i.e. throughput.
2. `--enforce-eager` — removes the graph-capture part of the peak; the project's
   own tutorial for the same model ships it
   (`docs/source/tutorials/multi_npu_qwen3_next.md:89` at the same pin).
3. more TP (fewer weight bytes per rank, since the peak includes the weights).
4. only then `--max-model-len`, and say so explicitly as a capability limit.
   `--max-num-seqs` is **not** a lever for either shortfall: it bounds concurrent
   sequences and the per-slot mamba/GDN state, not the per-token KV or the
   prefill peak (advice to "试试 max_num_seqs" was tried and did nothing on #3843).

Recheck conditions: `rg -n 'forcibly disable the chunked prefill|max_num_batched_tokens = scheduler_config.max_model_len' vllm_ascend/`
and `rg -n 'def determine_available_memory' -A30 vllm_ascend/worker/`. If either
moved, re-derive before quoting these line numbers. The recorded #3843 answer's
ranked cause "utilization 0.5 too low" is *falsified* by the later observation
that 0.9 changed nothing; keep the failure mode separate from the fix.

## Raising the budget past the checkpoint has a runtime signature (recorded #2349)

`--max-model-len` larger than the derived maximum needs
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, and that env var turns the guard into a
warning ("This may lead to incorrect model outputs or CUDA errors",
`vllm/config.py:3507-3523` at v0.9.2). One of those errors is concrete: the
RoPE cache is built from the **checkpoint's** `max_position_embeddings`
(`vllm/model_executor/models/qwen3.py:178` →
`vllm/model_executor/layers/rotary_embedding.py:132`), is not re-sized by the flag, and on
Ascend is indexed by `positions`
(`torch_npu._npu_rotary_embedding(positions, …, cos_sin_cache, …)`,
`vllm_ascend/ops/rotary_embedding.py:78-85`, installed at `:286`). A request
whose position passes that value trips a device-side gather assert —
`gather_v3_base.h:137 … Index N out of range[0 M)` — which reads like a card
fault and is not one. Two commands separate it from every other out-of-range
report: `jq .max_position_embeddings <model>/config.json` against the serve
line's `--max-model-len`. Page:
`../custom-ops-runtime-faults/gather-index-out-of-range.md`.

## The same default can kill the **host**, not the card (recorded #3314)

On a long-context checkpoint the omitted flag has a second failure signature that
is not an HBM OOM and not a card problem: every worker builds an eager
`max_model_len²` attention mask on the **host** in `NPUModelRunner.__init__`, so
the box thrashes or the kernel OOM-kills the ranks before `Starting to load model`
(structured prompt, then 1.5 TiB of RAM at 100 %, `WorkerProc initialization
failed` with no worker traceback). #3314: Qwen3-30B-A3B-2507 (ceiling 262144),
no `--max-model-len`, TP=4/8 → worked at 65536, and the reporter's follow-up
confirmed the derived 256K and that lowering it is the circumvention.

So when the reporter asks *"how large a `max_model_len` can I use?"*, the answer is
never a number seen in another report: it is `sqrt(headroom · host_RAM / (9 ·
ranks))` per `scripts/attn-mask-budget.sh`, minus what the weights and KV setup
need. `../runtime-memory/attention-mask-host-allocation.md` carries the arithmetic,
the version/CANN gate that changes it, and the two rungs that do not help
(`--enforce-eager`, `--max-num-batched-tokens`).

## Limits of this check

- It is a configuration/budget effect, not a proven defect in vllm-ascend: the
  same maintainer with the same vllm/vllm-ascend versions did not reproduce on
  their own hardware. Per-card HBM and model variant decide whether the default
  fits.
- It does not explain collective failures: the recorded CANN/HCCL case
  (vllm-ascend issue #7477) already passed `--max-model-len 32768` and still
  failed inside `HcclAllreduce` — that route is
  `../hccl-and-hardware/index.md`.
